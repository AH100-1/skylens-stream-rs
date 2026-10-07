//! 회전 평균: 묶음 사이 간선만 별도 문턱으로 거르면 3~4° 잡음 간선이 살아 83° 틀린 간선을 이기는지.
//! 합성 정답: 묶음 3 개(각 12 정점), 묶음 내부 간선은 잡음 0.05°, 묶음 사이 간선은 3° 잡음이고
//! 묶음 B 는 A 와 올바른 3 개와 서로 일관되게 83° 틀린 2 개로만 이어진다.

use skylens_core::math::{rotation_angle_between, Rotation3, Vector3};
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};

fn rot(ax: [f64; 3], deg: f64) -> Rotation3<f64> {
    Rotation3::new(Vector3::new(ax[0], ax[1], ax[2]).normalize() * deg.to_radians())
}

fn scene() -> (Vec<Rotation3<f64>>, Vec<RelativeRotation>, Vec<usize>) {
    let n = 36;
    let offs = [
        rot([0.0, 0.0, 1.0], 0.0),
        rot([0.1, 0.2, 1.0], 84.0),
        rot([1.0, 0.3, 0.1], -40.0),
    ];
    let truth: Vec<Rotation3<f64>> = (0..n)
        .map(|v| rot([0.3, 1.0, 0.2], 3.0 * (v % 12) as f64) * offs[v / 12])
        .collect();
    let groups: Vec<usize> = (0..n).map(|v| v / 12).collect();
    let rel = |i: usize, j: usize, noise: Rotation3<f64>, w: f64| RelativeRotation {
        i,
        j,
        rotation: noise * truth[j] * truth[i].inverse(),
        weight: w,
    };
    let mut e = Vec::new();
    for base in [0usize, 12, 24] {
        for v in 0..11 {
            for d in 1..=3 {
                if v + d < 12 {
                    // 묶음 내부 잡음 0.05°: 축을 바꿔 가며.
                    let ax = [(v + d) as f64 + 1.0, 1.0, (v % 3) as f64 - 1.0];
                    e.push(rel(base + v, base + v + d, rot(ax, 0.05), 600.0));
                }
            }
        }
    }
    // A–C: 잡음 3° 의 올바른 간선 4 개.
    for (k, (i, j)) in [(1, 25), (4, 28), (7, 31), (10, 34)].iter().enumerate() {
        e.push(rel(*i, *j, rot([1.0, k as f64, 0.5], 3.0), 30.0));
    }
    // A–B: 올바른 3 개(3~4°), 서로 일관되게 틀린 2 개(83°).
    e.push(rel(1, 13, rot([1.0, 0.0, 0.0], 3.5), 30.0));
    e.push(rel(4, 15, rot([0.0, 1.0, 0.0], 4.0), 30.0));
    e.push(rel(8, 20, rot([0.0, 0.0, 1.0], 3.2), 30.0));
    e.push(rel(2, 18, rot([1.0, 1.0, 0.0], 83.0), 60.0));
    e.push(rel(6, 22, rot([1.0, 1.0, 0.0], 83.5), 60.0));
    (truth, e, groups)
}

/// 묶음 `b` 의 중앙 오차(도): 묶음 A 에서 정해지는 전역 회전 하나를 맞춘 뒤.
fn block_error(truth: &[Rotation3<f64>], got: &[Option<Rotation3<f64>>], b: usize) -> f64 {
    let g = truth[0].inverse() * got[0].unwrap();
    let mut errs: Vec<f64> = (12 * b..12 * b + 12)
        .map(|v| rotation_angle_between(&(truth[v] * g), &got[v].unwrap()).to_degrees())
        .collect();
    errs.sort_by(f64::total_cmp);
    errs[errs.len() / 2]
}

fn cfg_on(groups: Vec<usize>) -> AveragingConfig {
    AveragingConfig {
        cross_thresh: true,
        vertex_groups: groups,
        ..AveragingConfig::default()
    }
}

#[test]
fn cross_threshold_keeps_good_cross_edges() {
    let (truth, edges, groups) = scene();
    let r = average_rotations(36, &edges, &cfg_on(groups)).unwrap();
    let (eb, ec) = (
        block_error(&truth, &r.rotations, 1),
        block_error(&truth, &r.rotations, 2),
    );
    let ct = r.cross_threshold_rad.unwrap().to_degrees();
    eprintln!(
        "on: B {eb:.3} C {ec:.3} deg, cross thr {ct:.2} deg, warn {:?}",
        r.sparse_group_links
    );
    assert!(eb <= 5.0 && ec <= 5.0, "B {eb} C {ec}");
    assert!(ct >= 5.0);
    // 올바른 묶음 사이 간선 3 개 유지, 틀린 2 개 제외(마지막 5 개가 A–B).
    let m = edges.len();
    assert!(r.inliers[m - 5..m - 2].iter().all(|&x| x));
    assert!(r.inliers[m - 2..].iter().all(|&x| !x));
    // A–B 유지 간선 3 개 > 2 이므로 경고 없음(A–C 는 4 개).
    assert!(r
        .sparse_group_links
        .iter()
        .all(|&(a, b, _)| (a, b) != (0, 2)));
}

#[test]
fn cross_threshold_warns_on_sparse_links() {
    let (_, edges, groups) = scene();
    // 올바른 A–B 간선을 하나만 남기면 유지 간선이 2 개 이하여서 경고 목록에 든다.
    let m = edges.len();
    let few: Vec<RelativeRotation> = edges
        .iter()
        .enumerate()
        .filter(|(k, _)| *k <= m - 5)
        .map(|(_, e)| e.clone())
        .collect();
    let r = average_rotations(36, &few, &cfg_on(groups)).unwrap();
    assert!(
        r.sparse_group_links
            .iter()
            .any(|&(a, b, c)| (a, b) == (0, 1) && c <= 2),
        "{:?}",
        r.sparse_group_links
    );
}

#[test]
fn off_reproduces_default_behavior() {
    let (truth, edges, groups) = scene();
    let base = average_rotations(36, &edges, &AveragingConfig::default()).unwrap();
    // 묶음 정보만 주고 옵션을 끄면 기본과 완전히 같다.
    let off = AveragingConfig {
        vertex_groups: groups,
        ..AveragingConfig::default()
    };
    let r = average_rotations(36, &edges, &off).unwrap();
    assert_eq!(base.inliers, r.inliers);
    assert!(r.cross_threshold_rad.is_none() && r.sparse_group_links.is_empty());
    for v in 0..36 {
        let d = rotation_angle_between(&base.rotations[v].unwrap(), &r.rotations[v].unwrap());
        assert!(d < 1e-12);
    }
    let eb = block_error(&truth, &base.rotations, 1);
    eprintln!(
        "off: B error {eb:.3} deg, thr {:.3} deg, cross inliers {}",
        base.outlier_threshold_rad.to_degrees(),
        base.inliers[edges.len() - 9..]
            .iter()
            .filter(|&&x| x)
            .count()
    );
    // 기본 문턱은 하한 1° 에 걸리고, 그 때문에 3~4° 잡음의 올바른 묶음 사이 간선 7 개를 다 지키지 못한다.
    assert!((base.outlier_threshold_rad.to_degrees() - 1.0).abs() < 1e-9);
    let m = edges.len();
    let kept_good = base.inliers[m - 9..m - 2].iter().filter(|&&x| x).count();
    assert!(kept_good < 7, "{kept_good}");
}
