//! 회전 평균: 간선 종류별 이상치 문턱(덩어리 내부 / 덩어리 사이)과 Geman–McClure 반복.
//! 합성 정답: 두 묶음(서로 84° 회전), 묶음 내부 간선 잡음 0.03°, 묶음 사이 간선은 몇 개뿐이며
//! 맞는 것은 3~4° 잡음, 틀린 것은 서로 일관되게 83° 틀렸다.

use skylens_core::math::{rotation_angle_between, Rotation3, Vector3};
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};

fn rot(ax: [f64; 3], deg: f64) -> Rotation3<f64> {
    Rotation3::new(Vector3::new(ax[0], ax[1], ax[2]).normalize() * deg.to_radians())
}

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn noise(&mut self, deg: f64) -> Rotation3<f64> {
        let v = Vector3::new(self.unit() - 0.5, self.unit() - 0.5, self.unit() - 0.5);
        Rotation3::new(v.normalize() * deg.to_radians())
    }
}

/// 묶음 사이 간선 명세: (끝점 a(묶음 A), 끝점 b(묶음 B 안 번호), 잡음 축, 잡음 각, 가중치).
type Cross = (usize, usize, [f64; 3], f64, f64);

fn scene(
    half: usize,
    cross: &[Cross],
    inner_deg: f64,
) -> (Vec<Rotation3<f64>>, Vec<RelativeRotation>) {
    let n = 2 * half;
    let truth: Vec<Rotation3<f64>> = (0..n)
        .map(|v| {
            let base = rot([0.3, 1.0, 0.2], 3.0 * (v % half) as f64);
            if v < half {
                base
            } else {
                base * rot([0.1, 0.2, 1.0], 84.0)
            }
        })
        .collect();
    let mut rng = Lcg(7);
    let mut e = Vec::new();
    let mk = |i: usize, j: usize, noise: Rotation3<f64>, w: f64| RelativeRotation {
        i,
        j,
        rotation: noise * truth[j] * truth[i].inverse(),
        weight: w,
    };
    for base in [0usize, half] {
        for v in 0..half - 1 {
            for d in 1..=3 {
                if v + d < half {
                    e.push(mk(base + v, base + v + d, rng.noise(inner_deg), 600.0));
                }
            }
        }
    }
    for &(a, b, ax, deg, w) in cross {
        e.push(mk(a, half + b, rot(ax, deg), w));
    }
    (truth, e)
}

/// 묶음 B 중앙 오차(도): A 에서 구한 전역 회전 하나를 뺀 뒤.
fn block_b_error(half: usize, truth: &[Rotation3<f64>], got: &[Option<Rotation3<f64>>]) -> f64 {
    let g = truth[0].inverse() * got[0].unwrap();
    let mut errs: Vec<f64> = (half..2 * half)
        .map(|v| (truth[v] * g).inverse() * got[v].unwrap())
        .map(|r| rotation_angle_between(&Rotation3::identity(), &r).to_degrees())
        .collect();
    errs.sort_by(f64::total_cmp);
    errs[errs.len() / 2]
}

fn cfgs() -> [(&'static str, AveragingConfig); 6] {
    let d = AveragingConfig::default();
    [
        ("default", d.clone()),
        (
            "floor 5",
            AveragingConfig {
                outlier_threshold_rad: 5f64.to_radians(),
                ..d.clone()
            },
        ),
        (
            "floor 10",
            AveragingConfig {
                outlier_threshold_rad: 10f64.to_radians(),
                ..d.clone()
            },
        ),
        (
            "bridge_components",
            AveragingConfig {
                bridge_components: true,
                ..d.clone()
            },
        ),
        (
            "class_thresholds",
            AveragingConfig {
                class_thresholds: true,
                ..d.clone()
            },
        ),
        ("gm_irls", AveragingConfig { gm_irls: true, ..d }),
    ]
}

fn run(name: &str, half: usize, cross: &[Cross]) -> [f64; 6] {
    let (truth, edges) = scene(half, cross, 0.03);
    let mut out = [0.0; 6];
    for (m, (_, cfg)) in cfgs().iter().enumerate() {
        let r = average_rotations(2 * half, &edges, cfg).unwrap();
        out[m] = block_b_error(half, &truth, &r.rotations);
        if m == 0 {
            eprintln!(
                "  [{name}] cross edges {} positions {} weak {}",
                r.bridge_edge_count, r.bridge_position_count, r.weak_bridge
            );
        }
    }
    out
}

const X: [f64; 3] = [1.0, 0.0, 0.0];
const Y: [f64; 3] = [0.0, 1.0, 0.0];
const Z: [f64; 3] = [0.0, 0.0, 1.0];
const W: [f64; 3] = [1.0, 1.0, 0.0];

fn base_cross(h: usize) -> Vec<Cross> {
    vec![
        (1, 1, X, 3.5, 30.0),
        (4, 3, Y, 4.0, 30.0),
        (h - 4, h - 4, Z, 3.2, 30.0),
        (2, 6, W, 83.0, 20.0),
        (h - 6, h - 6, W, 83.5, 20.0),
    ]
}

#[test]
fn threshold_table() {
    let mut rows: Vec<(String, [f64; 6])> = Vec::new();
    for h in [12usize, 30] {
        rows.push((
            format!("base {}v 3 right/2 wrong", 2 * h),
            run("base", h, &base_cross(h)),
        ));
    }
    let h = 12;
    rows.push((
        "2 right/2 wrong".into(),
        run(
            "2v2",
            h,
            &[
                (1, 1, X, 3.5, 30.0),
                (8, 8, Z, 3.2, 30.0),
                (2, 6, W, 83.0, 20.0),
                (6, 2, W, 83.5, 20.0),
            ],
        ),
    ));
    rows.push((
        "3 right(low w)/3 wrong".into(),
        run(
            "3v3",
            h,
            &[
                (1, 1, X, 3.5, 10.0),
                (4, 3, Y, 4.0, 10.0),
                (8, 8, Z, 3.2, 10.0),
                (2, 6, W, 83.0, 20.0),
                (6, 2, W, 83.5, 20.0),
                (9, 11, W, 82.5, 20.0),
            ],
        ),
    ));
    rows.push(("45 vs 70 weight".into(), {
        let mut c = base_cross(h);
        for x in c.iter_mut().take(3) {
            x.4 = 70.0 / 3.0;
        }
        for x in c.iter_mut().skip(3) {
            x.4 = 22.5;
        }
        run("45v70", h, &c)
    }));
    rows.push(("wrong w60 (init wrong)".into(), {
        let mut c = base_cross(h);
        for x in c.iter_mut().skip(3) {
            x.4 = 60.0;
        }
        run("wrongheavy", h, &c)
    }));
    rows.push((
        "2 right w30/3 wrong w15".into(),
        run(
            "initwrong",
            h,
            &[
                (1, 1, X, 3.5, 30.0),
                (8, 8, Z, 3.2, 30.0),
                (2, 6, W, 83.0, 15.0),
                (6, 2, W, 83.5, 15.0),
                (9, 11, W, 82.5, 15.0),
            ],
        ),
    ));
    eprintln!(
        "{:<26} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "scene (B median deg)", "default", "floor5", "floor10", "bridge", "class", "gm_irls"
    );
    for (n, r) in &rows {
        eprintln!(
            "{:<26} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            n, r[0], r[1], r[2], r[3], r[4], r[5]
        );
    }
    // 기본 장면(24·60 정점): 새 방법 둘 다 묶음 오차 중앙 5° 이하.
    for (n, r) in rows.iter().filter(|(n, _)| n.starts_with("base")) {
        assert!(r[4] <= 5.0, "{n}: class_thresholds {}", r[4]);
        assert!(r[5] <= 5.0, "{n}: gm_irls {}", r[5]);
    }
}

#[test]
fn clean_scene_not_worse() {
    // 이상치 없음, 묶음 사이 간선 5개 잡음 1°.
    let h = 12;
    let cross: Vec<Cross> = vec![
        (1, 1, X, 1.0, 30.0),
        (4, 3, Y, 1.0, 30.0),
        (8, 8, Z, 1.0, 30.0),
        (2, 6, W, 1.0, 30.0),
        (6, 2, X, 1.0, 30.0),
    ];
    let r = run("clean", h, &cross);
    eprintln!(
        "clean: default {:.3} class {:.3} gm {:.3}",
        r[0], r[4], r[5]
    );
    assert!(r[4] <= r[0] + 0.1, "class {} vs default {}", r[4], r[0]);
    assert!(r[5] <= r[0] + 0.1, "gm {} vs default {}", r[5], r[0]);
}

#[test]
fn weak_bridge_warning() {
    let h = 12;
    let (_, edges) = scene(h, &base_cross(h)[..2], 0.03);
    let r = average_rotations(2 * h, &edges, &AveragingConfig::default()).unwrap();
    assert_eq!(r.bridge_edge_count, 2);
    assert!(r.weak_bridge);
    let (_, edges) = scene(h, &base_cross(h), 0.03);
    let r = average_rotations(2 * h, &edges, &AveragingConfig::default()).unwrap();
    assert_eq!((r.bridge_edge_count, r.bridge_position_count), (5, 4));
    assert!(!r.weak_bridge);
}
