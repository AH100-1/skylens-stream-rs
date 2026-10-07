//! 장착 상대 회전 공유 정규화: 카메라 하나에 기울기 편향을 준 시작점에서 켬/끔 비교.

use skylens_core::ba::{bundle_adjust, BaOptions, BaProblem, Observation, RigShare};
use skylens_core::camera::Pose;
use skylens_core::distortion::{DistortedIntrinsics, Distortion};
use skylens_core::math::{Point3, Rotation3, Vector2, Vector3};
use skylens_core::synth::{Scene as SynthScene, SceneConfig};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uni(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
    fn gauss(&mut self) -> f64 {
        let u = self.next().max(1e-300);
        let v = self.next();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn intr() -> DistortedIntrinsics {
    DistortedIntrinsics {
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
        dist: Distortion {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        },
    }
}

/// 카메라 번호 = 위치 * 3 + 슬롯(0 F, 1 R, 2 L).
struct Scene {
    gt: BaProblem,
}

fn scene(seed: u64, stations: usize, n_pts: usize, noise: f64) -> Scene {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let down = Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI);
    let slot_rot = [
        Rotation3::identity(),
        Rotation3::from_axis_angle(&Vector3::y_axis(), 35f64.to_radians()),
        Rotation3::from_axis_angle(&Vector3::y_axis(), -35f64.to_radians()),
    ];
    let mut poses = Vec::new();
    for s in 0..stations {
        let yaw = Rotation3::from_axis_angle(&Vector3::z_axis(), rng.uni(-0.08, 0.08));
        let c = Point3::new(
            s as f64 * 8.0,
            rng.uni(-3.0, 3.0),
            60.0 + rng.uni(-2.0, 2.0),
        );
        for sr in &slot_rot {
            poses.push(Pose::from_center(*sr * down * yaw, &c));
        }
    }
    let points: Vec<Point3<f64>> = (0..n_pts)
        .map(|_| {
            Point3::new(
                rng.uni(-20.0, stations as f64 * 8.0 + 20.0),
                rng.uni(-70.0, 70.0),
                rng.uni(0.0, 8.0),
            )
        })
        .collect();
    let k = intr();
    let mut observations = Vec::new();
    for (c, pose) in poses.iter().enumerate() {
        for (p, x) in points.iter().enumerate() {
            let xc = pose.transform(x);
            if xc.z <= 1.0 {
                continue;
            }
            let px = k.project_camera(&xc);
            if px.x < 0.0 || px.x > 640.0 || px.y < 0.0 || px.y > 480.0 {
                continue;
            }
            let mut o = Observation {
                camera: c,
                point: p,
                pixel: px,
            };
            o.pixel.x += noise * rng.gauss();
            o.pixel.y += noise * rng.gauss();
            observations.push(o);
        }
    }
    let n = poses.len();
    Scene {
        gt: BaProblem {
            groups: vec![k],
            poses,
            camera_group: vec![0; n],
            points,
            observations,
        },
    }
}

/// SPEC §1·§6 실측 편대(드론 3대 약 10 m 간격, 드론마다 카메라 1대, F −3°·R +125°·L −116°,
/// 내려다보는 각 60°, 화각 65°, 위치 간 1 m) 합성 장면. 정답 포즈에 기체별 독립 흔들림
/// (요·구름·키 각 σ = `jitter_deg`)을 위치마다 따로 더한다. 정답 상대 회전은 위치마다 다르다.
fn fleet_scene(seed: u64, stations: usize, n_pts: usize, noise: f64, jitter_deg: f64) -> Scene {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let syn = SynthScene::new(SceneConfig {
        positions: stations,
        seed,
        ..SceneConfig::default()
    });
    let ik = syn.views[0].camera.intrinsics;
    let k = DistortedIntrinsics {
        fx: ik.fx,
        fy: ik.fy,
        cx: ik.cx,
        cy: ik.cy,
        dist: Distortion {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        },
    };
    let (w, h) = (ik.width as f64, ik.height as f64);
    let poses: Vec<Pose> = syn
        .views
        .iter()
        .map(|v| {
            let j = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * jitter_deg.to_radians();
            Pose::from_center(
                Rotation3::new(j) * v.camera.pose.rotation,
                &v.camera.pose.center(),
            )
        })
        .collect();
    let length = (stations - 1) as f64;
    let points: Vec<Point3<f64>> = (0..n_pts)
        .map(|_| {
            let (x, y) = (rng.uni(-15.0, length + 25.0), rng.uni(-45.0, 45.0));
            Point3::new(x, y, syn.surface_height(x, y))
        })
        .collect();
    let mut observations = Vec::new();
    for (c, pose) in poses.iter().enumerate() {
        for (p, x) in points.iter().enumerate() {
            let xc = pose.transform(x);
            if xc.z <= 1.0 {
                continue;
            }
            let px = k.project_camera(&xc);
            if px.x < 0.0 || px.x > w || px.y < 0.0 || px.y > h {
                continue;
            }
            observations.push(Observation {
                camera: c,
                point: p,
                pixel: Vector2::new(px.x + noise * rng.gauss(), px.y + noise * rng.gauss()),
            });
        }
    }
    let n = poses.len();
    Scene {
        gt: BaProblem {
            groups: vec![k],
            poses,
            camera_group: vec![0; n],
            points,
            observations,
        },
    }
}

/// 시작점: 모든 포즈에 작은 잡음, 슬롯 `biased` 의 모든 카메라에 카메라 x 축 `bias_deg` 기울기,
/// 위치마다 다른 성분 `bias_jitter_deg`.
fn start(sc: &Scene, seed: u64, biased: usize, bias_deg: f64, jitter_deg: f64) -> BaProblem {
    let mut rng = Rng(seed ^ 0xABCD_1234_5678_9F01);
    let mut p = sc.gt.clone();
    for (c, pose) in p.poses.iter_mut().enumerate() {
        let w = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.1f64.to_radians();
        let mut r = Rotation3::new(w) * pose.rotation;
        if c % 3 == biased {
            let a = (bias_deg + jitter_deg * rng.gauss()).to_radians();
            r = Rotation3::from_axis_angle(&Vector3::x_axis(), a) * r;
        }
        let ctr = pose.center() + Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.2;
        *pose = Pose::from_center(r, &Point3::from(ctr));
    }
    for x in p.points.iter_mut() {
        *x += Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.3;
    }
    p
}

fn angle_deg(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    skylens_core::math::rotation_angle_between(a, b).to_degrees()
}

/// 슬롯별 (회전 오차 중앙, 카메라 x 축 기울기 부호 평균) 도.
fn per_slot(p: &BaProblem, gt: &BaProblem) -> [(f64, f64); 3] {
    let mut out = [(0.0, 0.0); 3];
    for (sl, o) in out.iter_mut().enumerate() {
        let mut errs = Vec::new();
        let mut tilt = 0.0;
        for c in (sl..p.poses.len()).step_by(3) {
            errs.push(angle_deg(&p.poses[c].rotation, &gt.poses[c].rotation));
            let d = p.poses[c].rotation * gt.poses[c].rotation.inverse();
            // d = R_est R_gt^T 는 이미 카메라 좌표 축이다(왼쪽 곱).
            let w = d.scaled_axis();
            tilt += w.x.to_degrees();
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *o = (errs[errs.len() / 2], tilt / errs.len() as f64);
    }
    out
}

fn run(sc: &Scene, mut p: BaProblem, weight_sigma_deg: Option<f64>) -> BaProblem {
    let n = p.poses.len();
    let opts = BaOptions {
        max_iterations: 30,
        default_free_intrinsics: [false; 8],
        fixed_cameras: vec![0, 3],
        rig_share: weight_sigma_deg.map(|s| {
            RigShare::new(
                (0..n).map(|c| c / 3).collect(),
                (0..n).map(|c| c % 3).collect(),
                s,
            )
        }),
        ..BaOptions::default()
    };
    let _ = sc;
    bundle_adjust(&mut p, &opts);
    p
}

fn table(label: &str, rows: &[(&str, [(f64, f64); 3])]) {
    eprintln!("[{label}]");
    for (name, r) in rows {
        eprintln!(
            "  {name:<14} F {:.3}/{:+.3}  R {:.3}/{:+.3}  L {:.3}/{:+.3}",
            r[0].0, r[0].1, r[1].0, r[1].1, r[2].0, r[2].1
        );
    }
}

/// `fleet_jitter` 가 Some 이면 편대 장면(기체별 흔들림 σ 도), None 이면 한 기체 장면.
fn measure(
    stations: usize,
    n_pts: usize,
    noise: f64,
    jitter: f64,
    fleet_jitter: Option<f64>,
    seeds: &[u64],
) -> Vec<[(f64, f64); 3]> {
    // 행: 시작, 끔, 켬(σ 1.0 도), 켬(σ 0.3 도), 켬(σ 0.1 도). 시드 평균.
    let sig = [None, Some(1.0), Some(0.3), Some(0.1)];
    let mut acc = vec![[(0.0, 0.0); 3]; 5];
    for &seed in seeds {
        let sc = match fleet_jitter {
            Some(j) => fleet_scene(seed, stations, n_pts, noise, j),
            None => scene(seed, stations, n_pts, noise),
        };
        let st = start(&sc, seed, 1, 0.7, jitter);
        let mut rows = vec![per_slot(&st, &sc.gt)];
        for s in sig {
            rows.push(per_slot(&run(&sc, st.clone(), s), &sc.gt));
        }
        for (a, r) in acc.iter_mut().zip(&rows) {
            for (x, y) in a.iter_mut().zip(r) {
                x.0 += y.0 / seeds.len() as f64;
                x.1 += y.1 / seeds.len() as f64;
            }
        }
    }
    acc
}

#[test]
fn tilt_bias_with_and_without_rig_share() {
    let seeds = [1u64, 2, 3];
    let names = ["시작", "끔", "켬 σ1.0", "켬 σ0.3", "켬 σ0.1"];
    // 위치마다 다른 편향 성분이 있는 경우와 약한 관측(점 적음, 잡음 큼) 경우.
    for (label, st, np, noise, jit) in [
        (
            "편향 0.7 도 + 위치별 0.3 도, 점 600",
            12usize,
            600usize,
            0.5,
            0.3,
        ),
        ("편향 0.7 도 상수, 점 600", 12, 600, 0.5, 0.0),
        (
            "편향 0.7 도 + 위치별 0.3 도, 점 150, 잡음 1.5px",
            12,
            150,
            1.5,
            0.3,
        ),
    ] {
        let acc = measure(st, np, noise, jit, None, &seeds);
        let rows: Vec<(&str, [(f64, f64); 3])> =
            names.iter().cloned().zip(acc.iter().cloned()).collect();
        table(label, &rows);
        // F-443: 시작 행 R 열의 기울기 부호 평균이 넣은 편향 0.7 도 근처여야 한다.
        assert!(
            (acc[0][1].1 - 0.7).abs() < 0.1,
            "{label} 시작 기울기 {}",
            acc[0][1].1
        );
        // 끔(1) 대비 켬 σ0.1(4): 어느 카메라도 나빠지지 않고, 약한 관측에서는 크게 줄어든다.
        #[allow(clippy::needless_range_loop)]
        for k in 0..3 {
            assert!(acc[4][k].0 <= acc[1][k].0 * 1.02, "{label} 슬롯 {k}");
            if np == 150 {
                assert!(acc[4][k].0 < acc[1][k].0 * 0.8, "{label} 슬롯 {k}");
            }
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn fleet_independent_wobble_rig_share() {
    let seeds = [1u64, 2, 3, 4];
    let names = ["시작", "끔", "켬 σ1.0", "켬 σ0.3", "켬 σ0.1"];
    let mut worse_at_1deg = false;
    for jit in [0.5, 1.0, 2.0] {
        let label = format!("편대, 기체별 흔들림 {jit} 도, 편향 0.7 + 위치별 0.3 도, 점 1500");
        let acc = measure(12, 1500, 0.5, 0.3, Some(jit), &seeds);
        let rows: Vec<(&str, [(f64, f64); 3])> =
            names.iter().cloned().zip(acc.iter().cloned()).collect();
        table(&label, &rows);
        for k in 0..3 {
            let (off, on) = (acc[1][k].0, acc[4][k].0);
            eprintln!(
                "  슬롯 {k}: 켬 σ0.1 / 끔 = {:.3} (중앙 오차 {on:.3} / {off:.3})",
                on / off
            );
            if jit == 1.0 && on > off * 1.02 {
                worse_at_1deg = true;
            }
        }
    }
    if worse_at_1deg {
        // 나빠지면 기본 끔 유지가 근거다.
        eprintln!("  1 도 흔들림에서 켬이 끔보다 2% 넘게 나쁨: 기본 끔 유지");
        assert!(BaOptions::default().rig_share.is_none());
    }
}
