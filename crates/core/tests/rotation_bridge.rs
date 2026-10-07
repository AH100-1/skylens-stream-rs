//! 회전 평균: 덩어리 사이 간선이 모두 좁은 이상치 문턱에 걸려 빠질 때 덩어리 상대 회전이 맞는지.
//! 합성 정답: 같은 카메라 묶음 내부 간선은 정확하고, 묶음 사이 간선은 몇 개뿐이며 3~4° 잡음이 있고 일부는 83° 틀렸다.

use skylens_core::math::{Rotation3, Vector3};
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};

fn rot(ax: [f64; 3], deg: f64) -> Rotation3<f64> {
    Rotation3::new(Vector3::new(ax[0], ax[1], ax[2]).normalize() * deg.to_radians())
}

fn scene() -> (Vec<Rotation3<f64>>, Vec<RelativeRotation>) {
    // 정점 0..12 는 묶음 A, 12..24 는 묶음 B. 묶음 B 는 A 에 대해 약 84° 돌아가 있다.
    let n = 24;
    let truth: Vec<Rotation3<f64>> = (0..n)
        .map(|v| {
            let base = rot([0.3, 1.0, 0.2], 3.0 * (v % 12) as f64);
            if v < 12 {
                base
            } else {
                base * rot([0.1, 0.2, 1.0], 84.0)
            }
        })
        .collect();
    let rel = |i: usize, j: usize, noise: Rotation3<f64>, w: f64| RelativeRotation {
        i,
        j,
        rotation: noise * truth[j] * truth[i].inverse(),
        weight: w,
    };
    let mut e = Vec::new();
    for base in [0usize, 12] {
        for v in 0..11 {
            for d in 1..=3 {
                if v + d < 12 {
                    e.push(rel(base + v, base + v + d, Rotation3::identity(), 600.0));
                }
            }
        }
    }
    // 묶음 사이: 올바른 3개(잡음 3~4°), 틀린 2개(서로 다른 83° 쪽).
    e.push(rel(1, 13, rot([1.0, 0.0, 0.0], 3.5), 30.0));
    e.push(rel(4, 15, rot([0.0, 1.0, 0.0], 4.0), 30.0));
    e.push(rel(8, 20, rot([0.0, 0.0, 1.0], 3.2), 30.0));
    e.push(rel(2, 18, rot([1.0, 1.0, 0.0], 83.0), 20.0));
    e.push(rel(6, 22, rot([1.0, 1.0, 0.0], 83.5), 20.0));
    (truth, e)
}

/// 묶음 A·B 각각 정답에서 같은 전역 회전 하나를 뺀 뒤의 B 중앙 오차(도).
fn block_b_error(truth: &[Rotation3<f64>], got: &[Option<Rotation3<f64>>]) -> f64 {
    // R_got = R_truth G 꼴로 A 에서 G 를 구하고, B 에서 같은 G 로 맞췄을 때 오차.
    let g = truth[0].inverse() * got[0].unwrap();
    let mut errs: Vec<f64> = (12..24)
        .map(|v| (truth[v] * g).inverse() * got[v].unwrap())
        .map(|r| {
            skylens_core::math::rotation_angle_between(&Rotation3::identity(), &r).to_degrees()
        })
        .collect();
    errs.sort_by(f64::total_cmp);
    errs[errs.len() / 2]
}

#[test]
fn bridge_components_recovers_weakly_linked_block() {
    let (truth, edges) = scene();
    let cfg = AveragingConfig {
        bridge_components: true,
        ..AveragingConfig::default()
    };
    let r = average_rotations(24, &edges, &cfg).unwrap();
    let err = block_b_error(&truth, &r.rotations);
    eprintln!("bridge on: block B median error {err:.3} deg");
    assert!(err < 6.0, "묶음 B 오차 {err}");
}

#[test]
fn bridge_components_keeps_connected_case() {
    // 덩어리가 하나뿐이면 켜도 끄나 같은 해.
    let (truth, edges) = scene();
    let inner: Vec<RelativeRotation> = edges.into_iter().filter(|e| e.i < 12 && e.j < 12).collect();
    let on = AveragingConfig {
        bridge_components: true,
        ..AveragingConfig::default()
    };
    let a = average_rotations(24, &inner, &AveragingConfig::default()).unwrap();
    let b = average_rotations(24, &inner, &on).unwrap();
    // 묶음 A 안의 회전은 같아야 한다(B 는 연결되지 않은 성분이라 제외).
    for v in 0..12 {
        let (x, y) = (a.rotations[v].unwrap(), b.rotations[v].unwrap());
        let d = skylens_core::math::rotation_angle_between(&x, &y);
        assert!(d < 1e-9, "v {v} diff {d}");
        let _ = &truth;
    }
}
