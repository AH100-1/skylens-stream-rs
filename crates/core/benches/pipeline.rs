//! 구간별 시간 측정(합성 장면, SPEC §6 기준 규모 80곳 × 3대 = 240장).
//!
//! 사용법: `cargo bench --bench pipeline -- [인자]`
//!
//! | 인자 | 기본값 | 뜻 |
//! |---|---|---|
//! | `--positions N` | 8 | 촬영 위치 수(영상 수 = 3N). `ba-scale` 에서는 지정하지 않으면 80 |
//! | `--width W` `--height H` | 480 270 | 렌더 해상도 |
//! | `--repeat R` | 3 | 같은 입력으로 구간마다 반복하는 횟수(중앙·최소 보고) |
//! | `--threads T` | 0 | rayon 스레드 수(0 = rayon 기본 = 논리 코어 수) |
//! | `--max-pairs P` | 0 | 매칭·검증·자세 구간에서 잴 영상 짝 수 상한(0 = 전부). 앞에서부터 고르게 뽑는다. 회전 평균(검증 결과)을 확인할 때는 짝 전부(`--max-pairs 0`)를 명시한다 |
//! | `--ba-points M` | 3000 | 번들 조정 문제의 점 수 |
//! | `--full` | | SPEC 기준 규모: `--positions 80 --width 960 --height 540 --repeat 3 --ba-points 20000` 과 같다 |
//! | `--quick` | | 기본값과 같다(예전 이름, 그대로 받는다) |
//!
//! 순서 규칙: `--full`·`--quick` 은 어디에 두든 먼저 적용하고, 개별 인자(`--positions` 등)가 그 위에 덮어쓴다.
//! `--positions 20 --full` 과 `--full --positions 20` 은 모두 위치 20·960×540 이다. 둘 다 주면 뒤에 준 묶음이 이긴다.
//! 해석은 `benches/support/args.rs`, 순서 시험은 `tests/perf_structure.rs`.
//! | `--json PATH` | | 표를 JSON 으로도 쓴다(`{cores, threads, mode, rows[{name, items, unit, median_s, min_s, note}]}`) |
//! | `--mode M` | `pipeline` | `pipeline`(구간 전체), `ba-scale`(번들 조정 실제 규모), `detect`(1920×1080 한 장 검출) |
//! | `--ba-tracks N` | 100000 | `ba-scale` 의 트랙(점) 수 |
//! | `--ba-iters K` | 3 | `ba-scale` 의 LM 반복 수(조기 종료 없이 K 회) |
//!
//! 예상 시간(4 코어 측정 기계, 부하 없음 기준 어림): 인자 없음(24장, 480×270) 약 1 분 안,
//! `--full`(240장, 960×540, 짝 3663 전부, 반복 3) 수십 분 — 짝을 줄이려면 `--max-pairs` 를 함께 준다,
//! `--full --width 320 --height 180` 약 5 분, `--mode ba-scale --ba-iters 2 --repeat 1` 약 1~2 분(최대 메모리 약 2.5 GB),
//! `--mode detect` 수 초.
//!
//! `ba-scale`: 위치 `--positions`(기본 80 → 카메라 240)의 정답 포즈에서 영상마다 고르게 화소를 골라 깊이
//! 10~60 m 로 역투영한 점을 모든 카메라에 투영(깊이 1~80 m·영상 안, 가림 무시)한다. 렌더는 하지 않는다.
//! `bundle_adjust` 를 반복 0(평가만)·1·K 회로 따로 돌려 (K회 − 0회)/K 를 반복당 시간으로 적는다.
//!
//! `detect`: 합성 장면 첫 영상 1920×1080 한 장을 `detect_and_describe` 로 R 회 검출해 F-014 기준(0.4 s)과 함께
//! 중앙·최소를 적는다. 회귀 판정은 `tests/perf_structure.rs` 의 검출 해시 시험이 맡고 여기서는 기록만 한다.
//!
//! 회전 평균은 두 가지를 잰다: 정답 그래프(전체 짝, 정답 + 0.2° 잡음)와, 이 실행의 두 시점 자세 결과
//! (정상 대응 수 가중)를 입력으로 한 것. 뒤의 것은 정답 대비 정렬 오차 중앙값을 함께 적는다.
//!
//! 회전 평균과 번들 조정 입력은 매칭 결과가 아니라 정답 장면에서 만든다(정답 상대 회전 + 0.2° 잡음,
//! 정답 점 투영 + 0.5 px 잡음, 자세 흔들기). 짝 수 상한과 무관하게 전체 그래프 크기로 재기 위해서다.
//! 시간 단언은 하지 않는다. 같은 기계에서 다른 작업이 돌면 숫자가 흔들린다.

use std::hint::black_box;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use skylens_core::ba::{bundle_adjust, BaOptions, BaProblem, Observation};
use skylens_core::camera::Pose;
use skylens_core::distortion::Distortion;
use skylens_core::features::{detect_and_describe, DetectorConfig, Feature, GrayImage};
use skylens_core::matching::{
    candidate_pairs, ransac_fundamental, ratio_match, RansacConfig, PAIR_CROSS, PAIR_POW2_MAX,
    PAIR_TEMPORAL,
};
use skylens_core::math::{Point3, Rotation3, Vector2, Vector3};
use skylens_core::rotation_averaging::{
    aligned_errors, average_rotations, AveragingConfig, RelativeRotation,
};
use skylens_core::synth::{CamId, Scene, SceneConfig};
use skylens_core::two_view::{ransac_essential, recover_pose};

#[path = "support/args.rs"]
mod args;
use args::Args;

/// 짝 하나의 대응 좌표: (픽셀 a, 픽셀 b, 정규 a, 정규 b).
type PairCoords = (
    Vec<Vector2<f64>>,
    Vec<Vector2<f64>>,
    Vec<Vector2<f64>>,
    Vec<Vector2<f64>>,
);

/// 비율 검사 문턱(SPEC 기본 0.8)과 상호 최근접.
const RATIO: f32 = 0.8;

/// 한 구간의 측정 결과.
struct Row {
    name: &'static str,
    items: usize,
    unit: &'static str,
    times: Vec<Duration>,
    note: String,
}

/// `f` 를 `repeat` 번 돌려 시간을 모은다. 마지막 결과를 돌려준다.
fn measure<T>(repeat: usize, mut f: impl FnMut() -> T) -> (Vec<Duration>, T) {
    let mut times = Vec::with_capacity(repeat);
    let mut last = None;
    for _ in 0..repeat {
        let t0 = Instant::now();
        let out = black_box(f());
        times.push(t0.elapsed());
        last = Some(out);
    }
    (times, last.expect("repeat >= 1"))
}

fn median(t: &[Duration]) -> Duration {
    let mut v = t.to_vec();
    v.sort();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// 결정적 표준 정규 난수(선형 합동 + Box–Muller).
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn small_rotation(&mut self, sigma_rad: f64) -> Rotation3<f64> {
        Rotation3::new(Vector3::new(self.gauss(), self.gauss(), self.gauss()) * sigma_rad)
    }
}

/// `n` 개 중 `k` 개를 고르게 뽑은 번호(k = 0 이거나 k ≥ n 이면 전부).
fn spread(n: usize, k: usize) -> Vec<usize> {
    if k == 0 || k >= n {
        return (0..n).collect();
    }
    (0..k).map(|i| i * n / k).collect()
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = args::parse(&argv);
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("rayon 전역 풀은 한 번만 만든다");
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "코어 {cores}, rayon 스레드 {}, 위치 {}, 영상 {}, 해상도 {}×{}, 반복 {}",
        rayon::current_num_threads(),
        args.positions,
        3 * args.positions,
        args.width,
        args.height,
        args.repeat
    );
    let rows = match args.mode.as_str() {
        "pipeline" => pipeline(&args),
        "ba-scale" => ba_scale(&args),
        "detect" => detect(&args),
        other => panic!("알 수 없는 --mode: {other} (pipeline | ba-scale | detect)"),
    };
    report(&rows, &args, cores);
}

fn pipeline(args: &Args) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();

    // 1. 장면 생성 + 렌더.
    let config = SceneConfig {
        positions: args.positions,
        width: args.width,
        height: args.height,
        ..SceneConfig::default()
    };
    let scene = Scene::new(config);
    let nv = scene.views.len();
    let (t, renders) = measure(args.repeat, || {
        scene
            .views
            .iter()
            .map(|v| scene.render(v))
            .collect::<Vec<_>>()
    });
    rows.push(Row {
        name: "합성 렌더",
        items: nv,
        unit: "장",
        times: t,
        note: String::new(),
    });
    let grays: Vec<GrayImage> = renders
        .iter()
        .map(|(img, _)| GrayImage::from_rgb(img.width as usize, img.height as usize, &img.data))
        .collect();

    // 2. 특징 검출 + 기술자.
    let det = DetectorConfig::default();
    let (t, feats) = measure(args.repeat, || {
        grays
            .iter()
            .map(|g| detect_and_describe(g, &det))
            .collect::<Vec<Vec<Feature>>>()
    });
    let nf: usize = feats.iter().map(Vec::len).sum();
    rows.push(Row {
        name: "특징 검출",
        items: nv,
        unit: "장",
        times: t,
        note: format!("특징 평균 {:.0}/장", nf as f64 / nv.max(1) as f64),
    });

    // 3. 짝 생성.
    let cam_index = |c: CamId| CamId::ALL.iter().position(|&x| x == c).unwrap_or(0);
    let keys: Vec<(usize, usize)> = scene
        .views
        .iter()
        .map(|v| (cam_index(v.cam), v.position))
        .collect();
    let (t, pairs) = measure(args.repeat, || {
        candidate_pairs(&keys, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX)
    });
    rows.push(Row {
        name: "짝 생성",
        items: pairs.len(),
        unit: "짝",
        times: t,
        note: String::new(),
    });
    let chosen: Vec<(usize, usize)> = spread(pairs.len(), args.max_pairs)
        .into_iter()
        .map(|k| pairs[k])
        .collect();
    let np = chosen.len();
    let pair_note = if np < pairs.len() {
        format!("짝 {np}/{} 만 잼", pairs.len())
    } else {
        String::new()
    };

    // 4. 비율 매칭(짝 안에서 병렬, 짝은 차례로 — 제품 구현과 같은 병렬 단위).
    let (t, matches) = measure(args.repeat, || {
        chosen
            .iter()
            .map(|&(i, j)| ratio_match(&feats[i], &feats[j], RATIO, true))
            .collect::<Vec<_>>()
    });
    let nm: usize = matches.iter().map(Vec::len).sum();
    rows.push(Row {
        name: "비율 매칭",
        items: np,
        unit: "짝",
        times: t,
        note: format!(
            "대응 평균 {:.0}/짝 {pair_note}",
            nm as f64 / np.max(1) as f64
        ),
    });

    // 대응 좌표(픽셀·정규).
    let coords: Vec<PairCoords> = chosen
        .iter()
        .zip(&matches)
        .map(|(&(i, j), m)| {
            let (ki, kj) = (
                &scene.views[i].camera.intrinsics,
                &scene.views[j].camera.intrinsics,
            );
            let mut out = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for &(a, b) in m {
                let pa = Vector2::new(feats[i][a].kp.x as f64, feats[i][a].kp.y as f64);
                let pb = Vector2::new(feats[j][b].kp.x as f64, feats[j][b].kp.y as f64);
                out.0.push(pa);
                out.1.push(pb);
                out.2.push(ki.index_to_normalized(&pa));
                out.3.push(kj.index_to_normalized(&pb));
            }
            out
        })
        .collect();

    // 5a. 기하 검증: 8점 F RANSAC(짝 사이 병렬).
    let rcfg = RansacConfig::default();
    let (t, fres) = measure(args.repeat, || {
        coords
            .par_iter()
            .map(|c| ransac_fundamental(&c.0, &c.1, &rcfg).is_some())
            .collect::<Vec<bool>>()
    });
    rows.push(Row {
        name: "RANSAC F(8점)",
        items: np,
        unit: "짝",
        times: t,
        note: format!(
            "성공 {}/{np} {pair_note}",
            fres.iter().filter(|&&b| b).count()
        ),
    });

    // 5b. 기하 검증: 5점 E RANSAC(짝 사이 병렬).
    let focal = scene.views.first().map_or(1.0, |v| v.camera.intrinsics.fx);
    let (t, eres) = measure(args.repeat, || {
        coords
            .par_iter()
            .map(|c| ransac_essential(&c.2, &c.3, focal, &rcfg))
            .collect::<Vec<_>>()
    });
    // 짝 종류별(같은 카메라 / 다른 카메라) 성공 수. 짝 상한이 걸리면 뽑힌 짝 안에서만 센다.
    let kind_count = |same: bool| -> (usize, usize) {
        let idx: Vec<usize> = (0..np)
            .filter(|&k| (keys[chosen[k].0].0 == keys[chosen[k].1].0) == same)
            .collect();
        (
            idx.iter().filter(|&&k| eres[k].is_some()).count(),
            idx.len(),
        )
    };
    let (same_ok, same_n) = kind_count(true);
    let (cross_ok, cross_n) = kind_count(false);
    rows.push(Row {
        name: "RANSAC E(5점)",
        items: np,
        unit: "짝",
        times: t,
        note: format!(
            "성공 {}/{np}, 같은 카메라 {same_ok}/{same_n}, 다른 카메라 {cross_ok}/{cross_n} {pair_note}",
            eres.iter().filter(|e| e.is_some()).count()
        ),
    });

    // 6. 두 시점 자세(정상 대응으로 E 분해·삼각측량 검사).
    let (t, poses) = measure(args.repeat, || {
        coords
            .par_iter()
            .zip(&eres)
            .map(|(c, e)| {
                let (e, inl) = e.as_ref()?;
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for (k, _) in inl.iter().enumerate().filter(|(_, &ok)| ok) {
                    a.push(c.2[k]);
                    b.push(c.3[k]);
                }
                recover_pose(e, &a, &b)
            })
            .collect::<Vec<_>>()
    });
    rows.push(Row {
        name: "두 시점 자세",
        items: np,
        unit: "짝",
        times: t,
        note: format!(
            "성공 {}/{np} {pair_note}",
            poses.iter().filter(|p| p.is_some()).count()
        ),
    });

    // 7. 회전 평균: 전체 짝 그래프, 정답 상대 회전 + 0.2° 잡음.
    let mut rng = Rng(7);
    let edges: Vec<RelativeRotation> = pairs
        .iter()
        .map(|&(i, j)| {
            let (ri, rj) = (
                scene.views[i].camera.pose.rotation,
                scene.views[j].camera.pose.rotation,
            );
            RelativeRotation {
                i,
                j,
                rotation: rng.small_rotation(0.2f64.to_radians()) * rj * ri.inverse(),
                weight: 100.0,
            }
        })
        .collect();
    let acfg = AveragingConfig::default();
    let (t, avg) = measure(args.repeat, || average_rotations(nv, &edges, &acfg));
    rows.push(Row {
        name: "회전 평균(정답 그래프)",
        items: edges.len(),
        unit: "간선",
        times: t,
        note: format!("반복 {}", avg.as_ref().map_or(0, |r| r.iterations)),
    });

    // 7b. 회전 평균: 이 실행의 두 시점 자세 결과(정상 대응 수 가중)를 입력으로.
    let measured: Vec<RelativeRotation> = chosen
        .iter()
        .zip(&poses)
        .zip(&eres)
        .filter_map(|((&(i, j), p), e)| {
            let p = p.as_ref()?;
            let w = e.as_ref()?.1.iter().filter(|&&ok| ok).count();
            Some(RelativeRotation {
                i,
                j,
                rotation: p.rotation,
                weight: w as f64,
            })
        })
        .collect();
    let (t, avg_m) = measure(args.repeat, || average_rotations(nv, &measured, &acfg));
    let truth: Vec<Rotation3<f64>> = scene.views.iter().map(|v| v.camera.pose.rotation).collect();
    // 입력 간선 가운데 정답 상대 회전(R_j R_iᵀ)과 2° 넘게 다른 것의 비율. 회전 평균이 무너질 때 원인이
    // 입력(틀린 간선)인지 평균 쪽인지 가르는 값이다.
    let bad_edges = measured
        .iter()
        .filter(|e| {
            let t = truth[e.j] * truth[e.i].inverse();
            (e.rotation * t.inverse()).angle().to_degrees() > 2.0
        })
        .count();
    // 통과 간선 그래프의 연결 성분 수(시점 24개가 한 덩어리로 이어졌는지).
    let mut comp: Vec<usize> = (0..nv).collect();
    for e in &measured {
        let (a, b) = (comp[e.i], comp[e.j]);
        for c in comp.iter_mut() {
            if *c == b {
                *c = a;
            }
        }
    }
    let mut labels = comp.clone();
    labels.sort_unstable();
    labels.dedup();
    let bad_note = format!(
        "연결 성분 {}, 간선 오차>2° {}/{} ({:.1}%)",
        labels.len(),
        bad_edges,
        measured.len(),
        100.0 * bad_edges as f64 / measured.len().max(1) as f64
    );
    let note = match &avg_m {
        Some(r) => {
            let returned = r.rotations.iter().filter(|x| x.is_some()).count();
            let mut err: Vec<f64> = aligned_errors(&r.rotations, &truth)
                .into_iter()
                .filter(|x| x.is_finite())
                .collect();
            err.sort_by(f64::total_cmp);
            let m = err.get(err.len() / 2).copied().unwrap_or(f64::NAN);
            format!(
                "반복 {}, 반환 시점 {returned}/{nv}, {bad_note}, 정답 대비 정렬 오차 중앙 {:.3}° {pair_note}",
                r.iterations,
                m.to_degrees()
            )
        }
        None => format!("실패, {bad_note} {pair_note}"),
    };
    rows.push(Row {
        name: "회전 평균(검증 결과)",
        items: measured.len(),
        unit: "간선",
        times: t,
        note,
    });

    // 8. 번들 조정: 깊이 지도에서 정답 점을 뽑아 모든 카메라에 투영(가림 무시), 자세를 흔든다.
    let problem = ba_problem(&scene, &renders, args.ba_points, &mut rng);
    let nobs = problem.observations.len();
    let opts = BaOptions {
        fixed_cameras: vec![0, 1],
        ..BaOptions::default()
    };
    let (t, rep) = measure(args.repeat, || {
        let mut p = problem.clone();
        bundle_adjust(&mut p, &opts)
    });
    rows.push(Row {
        name: "번들 조정",
        items: nv,
        unit: "대",
        times: t,
        note: format!(
            "점 {} 관측 {nobs} 반복 {} RMS {:.2}→{:.2} px",
            problem.points.len(),
            rep.iterations,
            rep.initial_rms,
            rep.final_rms
        ),
    });

    rows
}

/// 표(마크다운)와 JSON 을 쓴다.
fn report(rows: &[Row], args: &Args, cores: usize) {
    // 표.
    println!();
    println!("| 구간 | 단위 수 | 중앙 (ms) | 최소 (ms) | 단위당 중앙 (ms) | 비고 |");
    println!("|---|---:|---:|---:|---:|---|");
    let mut total = Duration::ZERO;
    for r in rows {
        let m = median(&r.times);
        let lo = r.times.iter().min().copied().unwrap_or_default();
        total += m;
        println!(
            "| {} | {} {} | {:.1} | {:.1} | {:.3} | {} |",
            r.name,
            r.items,
            r.unit,
            ms(m),
            ms(lo),
            ms(m) / r.items.max(1) as f64,
            r.note.trim()
        );
    }
    if args.mode == "pipeline" {
        println!("| 합계(중앙) | | {:.1} | | | |", ms(total));
    }
    if let Some(path) = &args.json {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let mut j = format!(
            "{{\"cores\":{cores},\"threads\":{},\"mode\":\"{}\",\"rows\":[",
            rayon::current_num_threads(),
            esc(&args.mode)
        );
        for (k, r) in rows.iter().enumerate() {
            if k > 0 {
                j.push(',');
            }
            let lo = r.times.iter().min().copied().unwrap_or_default();
            j.push_str(&format!(
                "{{\"name\":\"{}\",\"items\":{},\"unit\":\"{}\",\"median_s\":{:.6},\"min_s\":{:.6},\"note\":\"{}\"}}",
                esc(r.name),
                r.items,
                esc(r.unit),
                median(&r.times).as_secs_f64(),
                lo.as_secs_f64(),
                esc(r.note.trim())
            ));
        }
        j.push_str("]}\n");
        std::fs::write(path, j).expect("JSON 쓰기");
    }
}

/// 정답 장면에서 번들 조정 문제를 만든다.
fn ba_problem(
    scene: &Scene,
    renders: &[(skylens_core::synth::RgbImage, Vec<f32>)],
    num_points: usize,
    rng: &mut Rng,
) -> BaProblem {
    let nv = scene.views.len();
    // 점: 영상마다 고르게 깊이 지도 화소를 골라 역투영.
    let per_view = num_points.div_ceil(nv.max(1)).max(1);
    let mut points = Vec::new();
    for (vi, (v, (_, depth))) in scene.views.iter().zip(renders).enumerate() {
        let k = &v.camera.intrinsics;
        let (w, h) = (k.width as usize, k.height as usize);
        let side = ((per_view as f64).sqrt().ceil() as usize).max(1);
        'grid: for gy in 0..side {
            for gx in 0..side {
                if points.len() >= num_points || points.len() >= per_view * (1 + vi) {
                    break 'grid;
                }
                let x = (gx * w + w / 2) / side;
                let y = (gy * h + h / 2) / side;
                let d = depth[y * w + x];
                if d.is_finite() && d > 0.0 && d < 200.0 {
                    let p = Vector2::new(x as f64 + 0.5, y as f64 + 0.5);
                    points.push(v.camera.unproject(&p, d as f64));
                }
            }
        }
    }
    let mut observations = Vec::new();
    for (pi, x) in points.iter().enumerate() {
        for (ci, v) in scene.views.iter().enumerate() {
            let xc = v.camera.pose.transform(x);
            // 영상 밖·너무 가깝거나 먼 관측은 뺀다(실제 트랙 길이에 가깝게).
            if xc.z <= 1.0 || xc.z > 80.0 {
                continue;
            }
            if let Some(px) = v
                .camera
                .project(x)
                .filter(|p| v.camera.intrinsics.contains(p))
            {
                let noisy = px + Vector2::new(rng.gauss(), rng.gauss()) * 0.5;
                observations.push(Observation {
                    camera: ci,
                    point: pi,
                    pixel: noisy,
                });
            }
        }
    }
    perturbed_problem(scene, points, observations, rng)
}

/// F-040: 번들 조정 실제 규모(카메라 3·위치, 기본 240 대·트랙 10만). 렌더 없이 정답 포즈만 쓴다.
fn ba_scale(args: &Args) -> Vec<Row> {
    let scene = Scene::new(SceneConfig {
        positions: args.positions,
        width: args.width,
        height: args.height,
        ..SceneConfig::default()
    });
    let nv = scene.views.len();
    let mut rng = Rng(40);
    let t_build = Instant::now();
    // 점: 영상을 돌아가며 화소를 고르게 뽑아 깊이 10~60 m 로 역투영.
    let mut points = Vec::with_capacity(args.ba_tracks);
    for k in 0..args.ba_tracks {
        let v = &scene.views[k % nv];
        let kk = &v.camera.intrinsics;
        let p = Vector2::new(rng.unit() * kk.width as f64, rng.unit() * kk.height as f64);
        let d = 10.0 + 50.0 * rng.unit();
        points.push(v.camera.unproject(&p, d));
    }
    let mut observations = Vec::new();
    for (pi, x) in points.iter().enumerate() {
        for (ci, v) in scene.views.iter().enumerate() {
            let z = v.camera.pose.transform(x).z;
            if z <= 1.0 || z > 80.0 {
                continue;
            }
            if let Some(px) = v
                .camera
                .project(x)
                .filter(|p| v.camera.intrinsics.contains(p))
            {
                observations.push(Observation {
                    camera: ci,
                    point: pi,
                    pixel: px + Vector2::new(rng.gauss(), rng.gauss()) * 0.5,
                });
            }
        }
    }
    let problem = perturbed_problem(&scene, points, observations, &mut rng);
    let build = t_build.elapsed();
    let nobs = problem.observations.len();
    let mut per_track = vec![0usize; problem.points.len()];
    for o in &problem.observations {
        per_track[o.point] += 1;
    }
    let used = per_track.iter().filter(|&&n| n >= 2).count();
    println!(
        "번들 조정 규모: 카메라 {nv}, 트랙 {} (관측 2 이상 {used}), 관측 {nobs}, 평균 트랙 길이 {:.2}, 문제 생성 {:.1} s",
        problem.points.len(),
        nobs as f64 / problem.points.len().max(1) as f64,
        build.as_secs_f64()
    );
    let mut rows = Vec::new();
    let mut run = |name: &'static str, iters: usize| {
        let opts = BaOptions {
            max_iterations: iters,
            fixed_cameras: vec![0, 1],
            function_tolerance: 0.0,
            ..BaOptions::default()
        };
        let reps = if iters == 0 { args.repeat } else { 1 };
        let (t, rep) = measure(reps, || {
            let mut p = problem.clone();
            bundle_adjust(&mut p, &opts)
        });
        let note = format!(
            "반복 {} 트랙 {} 관측 {} RMS {:.3}→{:.3} px 수렴 {}",
            rep.iterations,
            rep.num_tracks_used,
            rep.num_observations_used,
            rep.initial_rms,
            rep.final_rms,
            rep.converged
        );
        rows.push(Row {
            name,
            items: rep.iterations.max(1),
            unit: "반복",
            times: t,
            note,
        });
    };
    run("번들 조정 반복 0(준비·평가)", 0);
    run("번들 조정 반복 1", 1);
    run("번들 조정 반복 K", args.ba_iters);
    let t0 = median(&rows[0].times);
    let tk = median(&rows[2].times);
    let k = rows[2].items.max(1);
    println!(
        "반복당 (K회 − 0회)/K = {:.3} s (K = {k}), 반복 1회 − 0회 = {:.3} s, 전체 K회 {:.3} s",
        (tk.saturating_sub(t0)).as_secs_f64() / k as f64,
        median(&rows[1].times).saturating_sub(t0).as_secs_f64(),
        tk.as_secs_f64()
    );
    rows
}

/// F-031/F-014: 합성 1920×1080 한 장 검출 시간 기록(기준 0.4 s, 단언하지 않는다).
fn detect(args: &Args) -> Vec<Row> {
    let scene = Scene::new(SceneConfig {
        positions: 1,
        width: 1920,
        height: 1080,
        ..SceneConfig::default()
    });
    let (img, _) = scene.render(&scene.views[0]);
    let gray = GrayImage::from_rgb(img.width as usize, img.height as usize, &img.data);
    let cfg = DetectorConfig::default();
    let _ = black_box(detect_and_describe(&gray, &cfg)); // 예열
    let (t, f) = measure(args.repeat, || detect_and_describe(&gray, &cfg));
    let m = median(&t).as_secs_f64();
    vec![Row {
        name: "검출 1920×1080",
        items: 1,
        unit: "장",
        times: t,
        note: format!(
            "특징 {}, F-014 기준 0.4 s 대비 중앙 {:.0}%",
            f.len(),
            100.0 * m / 0.4
        ),
    }]
}

/// 정답 점·관측에 포즈·점을 흔들어 번들 조정 문제를 만든다(카메라 종류별 내부 파라미터 그룹 3개).
fn perturbed_problem(
    scene: &Scene,
    points: Vec<Point3<f64>>,
    observations: Vec<Observation>,
    rng: &mut Rng,
) -> BaProblem {
    let poses: Vec<Pose> = scene
        .views
        .iter()
        .map(|v| {
            let c = v.camera.pose.center();
            let jitter = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.05;
            Pose::from_center(
                rng.small_rotation(0.2f64.to_radians()) * v.camera.pose.rotation,
                &Point3::from(c.coords + jitter),
            )
        })
        .collect();
    let groups = CamId::ALL
        .iter()
        .map(|&c| {
            let k = scene
                .views
                .iter()
                .find(|v| v.cam == c)
                .map(|v| v.camera.intrinsics)
                .unwrap_or(scene.views[0].camera.intrinsics);
            k.with_distortion(Distortion::default())
        })
        .collect();
    let camera_group = scene
        .views
        .iter()
        .map(|v| CamId::ALL.iter().position(|&c| c == v.cam).unwrap_or(0))
        .collect();
    let points = points
        .into_iter()
        .map(|p| {
            Point3::from(p.coords + Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.05)
        })
        .collect();
    BaProblem {
        groups,
        poses,
        camera_group,
        points,
        observations,
    }
}
