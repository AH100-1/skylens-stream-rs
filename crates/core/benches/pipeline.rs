//! 구간별 시간 측정(합성 장면, SPEC §6 기준 규모 80곳 × 3대 = 240장).
//!
//! 사용법: `cargo bench --bench pipeline -- [인자]`
//!
//! | 인자 | 기본값 | 뜻 |
//! |---|---|---|
//! | `--positions N` | 80 | 촬영 위치 수(영상 수 = 3N) |
//! | `--width W` `--height H` | 960 540 | 렌더 해상도 |
//! | `--repeat R` | 3 | 같은 입력으로 구간마다 반복하는 횟수(중앙·최소 보고) |
//! | `--threads T` | 0 | rayon 스레드 수(0 = rayon 기본 = 논리 코어 수) |
//! | `--max-pairs P` | 0 | 매칭·검증·자세 구간에서 잴 영상 짝 수 상한(0 = 전부). 앞에서부터 고르게 뽑는다 |
//! | `--ba-points M` | 20000 | 번들 조정 문제의 점 수 |
//! | `--quick` | | `--positions 8 --width 480 --height 270 --repeat 3 --ba-points 3000` 과 같다 |
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
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};
use skylens_core::synth::{CamId, Scene, SceneConfig};
use skylens_core::two_view::{ransac_essential, recover_pose};

/// 짝 하나의 대응 좌표: (픽셀 a, 픽셀 b, 정규 a, 정규 b).
type PairCoords = (
    Vec<Vector2<f64>>,
    Vec<Vector2<f64>>,
    Vec<Vector2<f64>>,
    Vec<Vector2<f64>>,
);

/// 비율 검사 문턱(SPEC 기본 0.8)과 상호 최근접.
const RATIO: f32 = 0.8;

struct Args {
    positions: usize,
    width: u32,
    height: u32,
    repeat: usize,
    threads: usize,
    max_pairs: usize,
    ba_points: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        positions: 80,
        width: 960,
        height: 540,
        repeat: 3,
        threads: 0,
        max_pairs: 0,
        ba_points: 20_000,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let num = |v: Option<&String>, name: &str| -> usize {
        v.and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("{name} 뒤에 0 이상의 정수가 필요하다"))
    };
    while i < argv.len() {
        let next = argv.get(i + 1);
        match argv[i].as_str() {
            "--positions" => a.positions = num(next, "--positions"),
            "--width" => a.width = num(next, "--width") as u32,
            "--height" => a.height = num(next, "--height") as u32,
            "--repeat" => a.repeat = num(next, "--repeat").max(1),
            "--threads" => a.threads = num(next, "--threads"),
            "--max-pairs" => a.max_pairs = num(next, "--max-pairs"),
            "--ba-points" => a.ba_points = num(next, "--ba-points"),
            "--quick" => {
                a.positions = 8;
                a.width = 480;
                a.height = 270;
                a.repeat = 3;
                a.ba_points = 3000;
                i += 1;
                continue;
            }
            // cargo bench 가 붙이는 인자는 무시한다.
            "--bench" => {
                i += 1;
                continue;
            }
            other => panic!("알 수 없는 인자: {other}"),
        }
        i += 2;
    }
    a
}

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
    let args = parse_args();
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
    rows.push(Row {
        name: "RANSAC E(5점)",
        items: np,
        unit: "짝",
        times: t,
        note: format!(
            "성공 {}/{np} {pair_note}",
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
        name: "회전 평균",
        items: edges.len(),
        unit: "간선",
        times: t,
        note: format!("반복 {}", avg.as_ref().map_or(0, |r| r.iterations)),
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

    // 표.
    println!();
    println!("| 구간 | 단위 수 | 중앙 (ms) | 최소 (ms) | 단위당 중앙 (ms) | 비고 |");
    println!("|---|---:|---:|---:|---:|---|");
    let mut total = Duration::ZERO;
    for r in &rows {
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
    println!("| 합계(중앙) | | {:.1} | | | |", ms(total));
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
