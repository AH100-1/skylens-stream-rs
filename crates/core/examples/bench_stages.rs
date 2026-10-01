//! 구간별 시간 측정: 합성 장면(기본 80 위치 × 3 카메라 = 240장)에서
//! 특징 검출, 매칭(짝 생성 포함), 두 시점 기하 검증, 회전 평균, 번들 조정의 벽시계 시간을 잰다.
//!
//! 실행: `cargo run --release -p skylens-core --example bench_stages -- [옵션]`
//!   --width W --height H   렌더 해상도(기본 320×180, 1920×1080 은 선택)
//!   --positions N          촬영 위치 수(기본 80 → 240장)
//!   --threads T            rayon 스레드 수(기본: rayon 기본값)
//!   --json PATH            결과를 JSON 으로도 쓴다
//!
//! 시간은 기계 부하에 따라 흔들리므로 단독·직렬로 잰다. 회귀 판정은 `tests/perf_structure.rs` 의
//! 연산량 단언이 맡는다.

use nalgebra::{Point3, Rotation3, Vector2, Vector3};
use skylens_core::ba::{bundle_adjust, BaOptions, BaProblem, Observation};
use skylens_core::distortion::Distortion;
use skylens_core::features::{detect_and_describe, DetectorConfig, Feature, GrayImage};
use skylens_core::matching::{
    candidate_pairs, ratio_match, RansacConfig, PAIR_CROSS, PAIR_POW2_MAX, PAIR_TEMPORAL,
};
use skylens_core::rotation_averaging::{
    aligned_errors, average_rotations, AveragingConfig, RelativeRotation,
};
use skylens_core::synth::{CamId, Scene, SceneConfig};
use skylens_core::two_view::{ransac_essential_candidates, recover_pose};
use std::time::Instant;

struct Args {
    width: u32,
    height: u32,
    positions: usize,
    threads: Option<usize>,
    json: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        width: 320,
        height: 180,
        positions: 80,
        threads: None,
        json: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = |k: usize| -> String {
            argv.get(k + 1)
                .cloned()
                .unwrap_or_else(|| panic!("{} 뒤에 값이 필요하다", argv[k]))
        };
        match argv[i].as_str() {
            "--width" => a.width = val(i).parse().expect("--width"),
            "--height" => a.height = val(i).parse().expect("--height"),
            "--positions" => a.positions = val(i).parse().expect("--positions"),
            "--threads" => a.threads = Some(val(i).parse().expect("--threads")),
            "--json" => a.json = Some(val(i)),
            other => panic!("알 수 없는 인자: {other}"),
        }
        i += 2;
    }
    a
}

/// 검증된 짝: (i, j, 상대 회전 R_j R_iᵀ, 정상 대응 수).
type Verified = (usize, usize, Rotation3<f64>, usize);

struct Stage {
    name: &'static str,
    seconds: f64,
    detail: String,
}

fn main() {
    let args = parse_args();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("rayon 스레드 풀");
    }
    let threads = rayon::current_num_threads();
    let cfg = SceneConfig {
        positions: args.positions,
        width: args.width,
        height: args.height,
        ..SceneConfig::default()
    };
    let mut stages: Vec<Stage> = Vec::new();

    // 0. 렌더(측정 대상 아님, 참고로만 기록).
    let t = Instant::now();
    let scene = Scene::new(cfg);
    let grays: Vec<GrayImage> = {
        use rayon::prelude::*;
        scene
            .views
            .par_iter()
            .map(|v| {
                let (rgb, _) = scene.render(v);
                GrayImage::from_rgb(rgb.width as usize, rgb.height as usize, &rgb.data)
            })
            .collect()
    };
    stages.push(Stage {
        name: "render (참고)",
        seconds: t.elapsed().as_secs_f64(),
        detail: format!("{} 장", grays.len()),
    });

    // 1. 특징 검출 + 기술자. 영상 사이 병렬은 검출 내부 병렬에 맡긴다(직렬 순회).
    let det = DetectorConfig::default();
    let t = Instant::now();
    let feats: Vec<Vec<Feature>> = grays.iter().map(|g| detect_and_describe(g, &det)).collect();
    let nfeat: usize = feats.iter().map(Vec::len).sum();
    stages.push(Stage {
        name: "detect",
        seconds: t.elapsed().as_secs_f64(),
        detail: format!(
            "특징 {nfeat} (장당 평균 {:.0})",
            nfeat as f64 / feats.len() as f64
        ),
    });

    // 2. 매칭: 짝 생성 + 비율 검사 상호 매칭.
    let t = Instant::now();
    let ids: Vec<(usize, usize)> = scene
        .views
        .iter()
        .map(|v| {
            let c = CamId::ALL.iter().position(|&c| c == v.cam).unwrap();
            (c, v.position)
        })
        .collect();
    let pairs = candidate_pairs(&ids, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);
    let matches: Vec<Vec<(usize, usize)>> = pairs
        .iter()
        .map(|&(i, j)| ratio_match(&feats[i], &feats[j], 0.8, true))
        .collect();
    let comparisons: u64 = pairs
        .iter()
        .map(|&(i, j)| feats[i].len() as u64 * feats[j].len() as u64)
        .sum();
    let nmatch: usize = matches.iter().map(Vec::len).sum();
    stages.push(Stage {
        name: "match",
        seconds: t.elapsed().as_secs_f64(),
        detail: format!(
            "짝 {} / 대응 {nmatch} / 기술자 비교 {comparisons}",
            pairs.len()
        ),
    });

    // 3. 두 시점 기하 검증: 정규 좌표 RANSAC(5점) + 자세 복원. 짝 사이 병렬.
    let t = Instant::now();
    let rcfg = RansacConfig::default();
    let verified: Vec<Option<Verified>> = {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .zip(matches.par_iter())
            .map(|(&(i, j), m)| {
                if m.len() < 15 {
                    return None;
                }
                let ki = &scene.views[i].camera.intrinsics;
                let kj = &scene.views[j].camera.intrinsics;
                let to_n = |k: &skylens_core::camera::Intrinsics, f: &Feature| {
                    k.index_to_normalized(&Vector2::new(f.kp.x as f64, f.kp.y as f64))
                };
                let n1: Vec<_> = m.iter().map(|&(a, _)| to_n(ki, &feats[i][a])).collect();
                let n2: Vec<_> = m.iter().map(|&(_, b)| to_n(kj, &feats[j][b])).collect();
                let cands = ransac_essential_candidates(&n1, &n2, ki.fx, &rcfg);
                let (e, inl) = cands.first()?;
                let (s1, s2): (Vec<_>, Vec<_>) = inl
                    .iter()
                    .enumerate()
                    .filter(|(_, &b)| b)
                    .map(|(k, _)| (n1[k], n2[k]))
                    .unzip();
                let pose = recover_pose(e, &s1, &s2)?;
                Some((i, j, pose.rotation, s1.len()))
            })
            .collect()
    };
    let edges: Vec<RelativeRotation> = verified
        .iter()
        .flatten()
        .map(|&(i, j, r, w)| RelativeRotation {
            i,
            j,
            rotation: r,
            weight: w as f64,
        })
        .collect();
    // 정답 대비 상대 회전 오차 중앙값(검증 단계가 제대로 돌았는지 확인용).
    let mut rel_err: Vec<f64> = edges
        .iter()
        .map(|e| {
            let ri = scene.views[e.i].camera.pose.rotation;
            let rj = scene.views[e.j].camera.pose.rotation;
            (rj * ri.inverse() * e.rotation.inverse())
                .angle()
                .to_degrees()
        })
        .collect();
    rel_err.sort_by(f64::total_cmp);
    let med = rel_err.get(rel_err.len() / 2).copied().unwrap_or(f64::NAN);
    stages.push(Stage {
        name: "two_view",
        seconds: t.elapsed().as_secs_f64(),
        detail: format!(
            "검증 통과 {}/{} 짝, 상대 회전 오차 중앙 {med:.3}°",
            edges.len(),
            pairs.len()
        ),
    });

    // 4. 회전 평균.
    let t = Instant::now();
    let ra = average_rotations(scene.views.len(), &edges, &AveragingConfig::default());
    let ra_detail = match &ra {
        Some(r) => {
            let truth: Vec<_> = scene.views.iter().map(|v| v.camera.pose.rotation).collect();
            let mut err: Vec<f64> = aligned_errors(&r.rotations, &truth)
                .into_iter()
                .filter(|x| x.is_finite())
                .collect();
            err.sort_by(f64::total_cmp);
            let m = err.get(err.len() / 2).copied().unwrap_or(f64::NAN);
            format!(
                "반복 {}, 정답 대비 오차 중앙 {:.3}°",
                r.iterations,
                m.to_degrees()
            )
        }
        None => "실패".to_string(),
    };
    stages.push(Stage {
        name: "rotation_averaging",
        seconds: t.elapsed().as_secs_f64(),
        detail: ra_detail,
    });

    // 5. 번들 조정: 정답 장면 점(영상 격자 광선과 장면 교차)과 흔든 포즈로 문제를 만든다.
    // 트랙 생성 단계가 아직 없으므로 정답 관측을 쓴다. 측정 대상은 bundle_adjust 호출 시간뿐이다.
    let mut problem = ba_problem(&scene);
    let nobs = problem.observations.len();
    let npts = problem.points.len();
    let t = Instant::now();
    let rep = bundle_adjust(&mut problem, &BaOptions::default());
    stages.push(Stage {
        name: "bundle_adjust",
        seconds: t.elapsed().as_secs_f64(),
        detail: format!(
            "점 {npts} / 관측 {nobs} / 반복 {} / RMS {:.3}→{:.3}px",
            rep.iterations, rep.initial_rms, rep.final_rms
        ),
    });

    let total: f64 = stages.iter().skip(1).map(|s| s.seconds).sum();
    println!(
        "합성 {}장 {}×{}, rayon 스레드 {threads}",
        scene.views.len(),
        args.width,
        args.height
    );
    println!("{:<22} {:>10}  내용", "구간", "시간(s)");
    for s in &stages {
        println!("{:<22} {:>10.3}  {}", s.name, s.seconds, s.detail);
    }
    println!("{:<22} {:>10.3}", "합계(렌더 제외)", total);

    if let Some(path) = args.json {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let mut j = format!(
            "{{\"images\":{},\"width\":{},\"height\":{},\"threads\":{threads},\"stages\":[",
            scene.views.len(),
            args.width,
            args.height
        );
        for (k, s) in stages.iter().enumerate() {
            if k > 0 {
                j.push(',');
            }
            j.push_str(&format!(
                "{{\"name\":\"{}\",\"seconds\":{:.6},\"detail\":\"{}\"}}",
                esc(s.name),
                s.seconds,
                esc(&s.detail)
            ));
        }
        j.push_str(&format!("],\"total_seconds\":{total:.6}}}\n"));
        std::fs::write(&path, j).expect("JSON 쓰기");
    }
}

/// 장면 위 점을 정답 카메라 몇 장의 화소 격자 광선으로 만들고, 모든 카메라에 투영해 관측을 만든다.
/// 포즈는 결정적으로 흔든다(회전 ~0.3°, 위치 ~0.2 m). 카메라 0 은 게이지로 고정되므로 흔들지 않는다.
fn ba_problem(scene: &Scene) -> BaProblem {
    let mut points: Vec<Point3<f64>> = Vec::new();
    for v in scene.views.iter().step_by(6) {
        let k = &v.camera.intrinsics;
        for gy in 1..6 {
            for gx in 1..8 {
                let p = Vector2::new(
                    k.width as f64 * gx as f64 / 8.0,
                    k.height as f64 * gy as f64 / 6.0,
                );
                let x = v.camera.unproject(&p, 1.0);
                let o = v.camera.pose.center();
                if let Some(h) = scene.intersect(&o, &(x - o).normalize()) {
                    points.push(h.point);
                }
            }
        }
    }
    let mut observations = Vec::new();
    for (c, v) in scene.views.iter().enumerate() {
        for (p, x) in points.iter().enumerate() {
            if let Some(px) = v
                .camera
                .project(x)
                .filter(|px| v.camera.intrinsics.contains(px))
            {
                observations.push(Observation {
                    camera: c,
                    point: p,
                    pixel: px,
                });
            }
        }
    }
    let k0 = &scene.views[0].camera.intrinsics;
    let groups = vec![k0.with_distortion(Distortion::default())];
    let poses = scene
        .views
        .iter()
        .enumerate()
        .map(|(c, v)| {
            if c == 0 {
                return v.camera.pose;
            }
            let s = |k: u64| (((c as u64 * 7919 + k * 104_729) % 1000) as f64 / 1000.0) - 0.5;
            let dr = Rotation3::from_scaled_axis(Vector3::new(s(1), s(2), s(3)) * 0.01);
            let dc = Vector3::new(s(4), s(5), s(6)) * 0.4;
            skylens_core::camera::Pose::from_center(
                dr * v.camera.pose.rotation,
                &(v.camera.pose.center() + dc),
            )
        })
        .collect();
    BaProblem {
        groups,
        poses,
        camera_group: vec![0; scene.views.len()],
        points,
        observations,
    }
}
