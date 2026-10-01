//! 밀집 깊이 준비: 사진마다 이웃 사진 고르기와 깊이 탐색 범위.
//!
//! 이웃 점수(사진 i 에 대한 후보 j): 두 사진이 함께 본 희소 점 X 마다
//! `w_angle(θ) · w_scale(ρ)` 를 더한다.
//! - θ: X 에서 두 카메라 중심으로 가는 광선 사이 각. 목표 θ₀ = 10° 근처에서 1,
//!   θ < θ₀ 이면 폭 σ = 5°, θ > θ₀ 이면 σ = 15° 인 가우스 꼴로 줄어든다
//!   (작은 각은 깊이 분해능이 나쁘고 큰 각은 겉모습이 달라 정합이 어렵다).
//! - ρ: 두 사진에서 X 한 점이 차지하는 화소 크기(깊이 / 초점 거리)의 비.
//!   `w_scale = (min/max)²` 이라 축척이 같을 때 1.
//!
//! 점수 상위 k 장(기본 [`DEFAULT_NEIGHBORS`] = 8)을 고른다. 점수 0 인 후보는 빼므로
//! 공유 점이 적으면 k 장보다 적을 수 있다.
use std::collections::HashMap;

use crate::camera::Camera;
use crate::math::Point3;

/// 사진마다 고를 이웃 수 기본값.
pub const DEFAULT_NEIGHBORS: usize = 8;
/// 목표 광선 사이 각(도).
pub const TARGET_ANGLE_DEG: f64 = 10.0;
/// 목표보다 작은 각 쪽 감점 폭(도).
pub const SIGMA_BELOW_DEG: f64 = 5.0;
/// 목표보다 큰 각 쪽 감점 폭(도).
pub const SIGMA_ABOVE_DEG: f64 = 15.0;
/// 깊이 범위 분위(아래, 위).
pub const DEPTH_QUANTILES: (f64, f64) = (0.05, 0.95);
/// 깊이 범위 여유: 가까운 끝 × (1 − m), 먼 끝 × (1 + m).
pub const DEPTH_MARGIN: f64 = 0.1;

/// 등록된 사진 한 장.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// 왜곡 없는 핀홀 내부 파라미터와 세계→카메라 자세.
    pub cam: Camera,
    /// 사진 식별자. [`SparsePoint::observers`] 가 이 값을 가리킨다.
    pub id: usize,
}

/// 희소 복원 점 하나.
#[derive(Clone, Debug, PartialEq)]
pub struct SparsePoint {
    /// 세계 좌표.
    pub xyz: Point3<f64>,
    /// 이 점을 관측한 사진들의 [`View::id`].
    pub observers: Vec<usize>,
}

/// 광선 사이 각(라디안)에 대한 가중치.
pub fn angle_weight(theta: f64) -> f64 {
    let t0 = TARGET_ANGLE_DEG.to_radians();
    let s = if theta < t0 {
        SIGMA_BELOW_DEG
    } else {
        SIGMA_ABOVE_DEG
    }
    .to_radians();
    let d = (theta - t0) / s;
    (-0.5 * d * d).exp()
}

/// 축척 비 가중치: 두 화소 크기 a, b(> 0)에 대해 (min/max)².
pub fn scale_weight(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    let r = a.min(b) / a.max(b);
    r * r
}

/// 점 X 의 사진 안 화소 크기(카메라 z / 평균 초점 거리). 카메라 뒤면 None.
fn footprint(view: &View, x: &Point3<f64>) -> Option<f64> {
    let z = view.cam.pose.transform(x).z;
    let k = &view.cam.intrinsics;
    (z > 0.0).then(|| z / (0.5 * (k.fx + k.fy)))
}

/// 두 사진이 함께 본 점 하나의 쌍 점수.
pub fn pair_score(a: &View, b: &View, x: &Point3<f64>) -> f64 {
    let (Some(fa), Some(fb)) = (footprint(a, x), footprint(b, x)) else {
        return 0.0;
    };
    let ra = a.cam.pose.center() - x;
    let rb = b.cam.pose.center() - x;
    let c = (ra.dot(&rb) / (ra.norm() * rb.norm())).clamp(-1.0, 1.0);
    angle_weight(c.acos()) * scale_weight(fa, fb)
}

/// 사진마다 이웃 k 장을 고른다.
///
/// 반환값 `out[i]` 는 `views[i]` 의 이웃들을 점수 내림차순으로 담은 `views` 안 위치(색인)다.
/// 관측자 id 가 `views` 에 없으면 무시한다.
pub fn select_neighbors(views: &[View], points: &[SparsePoint], k: usize) -> Vec<Vec<usize>> {
    let pos: HashMap<usize, usize> = views.iter().enumerate().map(|(i, v)| (v.id, i)).collect();
    let n = views.len();
    let mut score = vec![0.0f64; n * n];
    let mut obs = Vec::new();
    for p in points {
        obs.clear();
        obs.extend(p.observers.iter().filter_map(|id| pos.get(id).copied()));
        obs.sort_unstable();
        obs.dedup();
        for (a, &i) in obs.iter().enumerate() {
            for &j in &obs[a + 1..] {
                let s = pair_score(&views[i], &views[j], &p.xyz);
                score[i * n + j] += s;
                score[j * n + i] += s;
            }
        }
    }
    (0..n)
        .map(|i| {
            let mut cand: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i && score[i * n + j] > 0.0)
                .map(|j| (j, score[i * n + j]))
                .collect();
            cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            cand.truncate(k);
            cand.into_iter().map(|(j, _)| j).collect()
        })
        .collect()
}

/// 정렬된 표본의 선형 보간 분위.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let t = q * (sorted.len() - 1) as f64;
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (t - lo as f64)
}

/// 사진의 깊이 탐색 범위 (가까운 끝, 먼 끝). 깊이는 카메라 z.
///
/// 그 사진(`view.id`)이 관측한 점 중 카메라 앞에 있는 점들의 깊이에서
/// [`DEPTH_QUANTILES`] 분위를 잡고 [`DEPTH_MARGIN`] 만큼 넓힌다.
/// 그런 점이 없으면 (0, 0).
pub fn depth_range(view: &View, points: &[SparsePoint]) -> (f64, f64) {
    let mut d: Vec<f64> = points
        .iter()
        .filter(|p| p.observers.contains(&view.id))
        .map(|p| view.cam.pose.transform(&p.xyz).z)
        .filter(|z| *z > 0.0)
        .collect();
    if d.is_empty() {
        return (0.0, 0.0);
    }
    d.sort_by(f64::total_cmp);
    let near = quantile(&d, DEPTH_QUANTILES.0) * (1.0 - DEPTH_MARGIN);
    let far = quantile(&d, DEPTH_QUANTILES.1) * (1.0 + DEPTH_MARGIN);
    (near, far)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Intrinsics, Pose};
    use crate::math::{Matrix3, Rotation3};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// 고도 30 m, x 방향 1 m 간격 직선 비행, 연직 하방 촬영. 지면 점은 고도 0 ± 1 m.
    fn line_flight(n: usize) -> (Vec<View>, Vec<SparsePoint>) {
        let k = Intrinsics::from_hfov(1600, 1200, 70f64.to_radians());
        // 카메라 x = 세계 x, 카메라 y = −세계 y, 카메라 z = −세계 z(아래).
        let r = Rotation3::from_matrix_unchecked(Matrix3::new(
            1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0,
        ));
        let views: Vec<View> = (0..n)
            .map(|i| View {
                cam: Camera {
                    intrinsics: k,
                    pose: Pose::from_center(r, &Point3::new(i as f64, 0.0, 30.0)),
                },
                // id 를 위치와 다르게 두어 대응을 확인한다.
                id: 100 + i,
            })
            .collect();
        let mut rng = Lcg(7);
        let mut pts = Vec::new();
        for _ in 0..3000 {
            let x = Point3::new(
                -25.0 + (n as f64 + 50.0) * rng.next(),
                -25.0 + 50.0 * rng.next(),
                -1.0 + 2.0 * rng.next(),
            );
            let observers = views
                .iter()
                .filter(|v| v.cam.project(&x).is_some_and(|p| k.contains(&p)))
                .map(|v| v.id)
                .collect::<Vec<_>>();
            if observers.len() >= 2 {
                pts.push(SparsePoint { xyz: x, observers });
            }
        }
        (views, pts)
    }

    #[test]
    fn weights_shape() {
        assert!((angle_weight(10f64.to_radians()) - 1.0).abs() < 1e-12);
        assert!(angle_weight(1f64.to_radians()) < 0.25);
        assert!(angle_weight(40f64.to_radians()) < 0.2);
        assert!((scale_weight(2.0, 2.0) - 1.0).abs() < 1e-12);
        assert!((scale_weight(1.0, 2.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn far_baseline_beats_adjacent() {
        let (views, pts) = line_flight(21);
        let nb = select_neighbors(&views, &pts, DEFAULT_NEIGHBORS);
        let c = 10;
        let ranks = |j: usize| nb[c].iter().position(|&x| x == j);
        eprintln!(
            "center neighbors (offsets): {:?}",
            nb[c]
                .iter()
                .map(|&j| j as i64 - c as i64)
                .collect::<Vec<_>>()
        );
        assert_eq!(nb.len(), 21);
        assert_eq!(nb[c].len(), DEFAULT_NEIGHBORS);
        assert!(!nb[c].contains(&c));
        // 8칸(8 m) 이웃은 뽑히고, 1칸(1 m) 이웃(광선 각 약 2° 이하)은 그보다 아래거나 빠진다.
        for (far, near) in [(c - 8, c - 1), (c + 8, c + 1)] {
            let rf = ranks(far).expect("8칸 이웃이 뽑혀야 한다");
            assert!(
                ranks(near).is_none_or(|rn| rn > rf),
                "far {rf} near {:?}",
                ranks(near)
            );
        }
        // 1위는 5~9칸 떨어진 사진(광선 각 약 9~17°).
        let d0 = (nb[c][0] as i64 - c as i64).abs();
        assert!((5..=9).contains(&d0), "1위 간격 {d0}");
    }

    #[test]
    fn depth_range_covers_truth_quantiles() {
        let (mut views, mut pts) = line_flight(21);
        let v = views.remove(10);
        let mut truth: Vec<f64> = pts
            .iter()
            .filter(|p| p.observers.contains(&v.id))
            .map(|p| 30.0 - p.xyz.z)
            .collect();
        truth.sort_by(f64::total_cmp);
        let (q05, q95) = (quantile(&truth, 0.05), quantile(&truth, 0.95));
        // 이상점 1 %: 깊이 300 m 짜리 점이 섞여도 범위는 흔들리지 않아야 한다.
        let n_out = truth.len() / 100;
        for i in 0..n_out {
            pts.push(SparsePoint {
                xyz: Point3::new(10.0 + 0.01 * i as f64, 0.0, -270.0),
                observers: vec![v.id],
            });
        }
        let (near, far) = depth_range(&v, &pts);
        eprintln!("truth q05 {q05:.3} q95 {q95:.3} range {near:.3}..{far:.3}");
        assert!(near <= q05 && far >= q95);
        assert!(near >= 0.85 * q05 && far <= 1.15 * q95);
        assert_eq!(depth_range(&View { id: 999, ..v }, &pts), (0.0, 0.0));
    }
}
