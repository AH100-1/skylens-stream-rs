//! 다시점 삼각측량: DLT 초기값 + 가우스–뉴턴 정밀화, 광선 각·양의 깊이 검사.
//!
//! 관측은 정규화 좌표(x/z, y/z)와 세계→카메라 자세(`camera::Pose`)의 쌍이다.
//! DLT 는 관측마다 x·P₃ − P₁, y·P₃ − P₂ 두 줄을 쌓고 AᵀA 의 최소 고유벡터를 쓴다.
//! 정밀화는 정규화 좌표 재투영 오차 제곱합을 점 X 에 대해 가우스–뉴턴으로 줄인다.

use crate::camera::Pose;
use crate::math::{Matrix3, Point3, SMatrix, Vector2, Vector3};

/// 삼각측량 설정.
#[derive(Clone, Debug)]
pub struct TriangulationConfig {
    /// 관측 광선 사이 최대 각이 이보다 작으면 버린다(rad).
    pub min_ray_angle_rad: f64,
    /// 정밀화 후 관측별 재투영 오차(정규화 좌표) 상한.
    pub max_reprojection: f64,
    /// 가우스–뉴턴 반복 수(0 이면 정밀화하지 않음).
    pub refine_iterations: usize,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_ray_angle_rad: 1.5f64.to_radians(),
            max_reprojection: 4e-3,
            refine_iterations: 10,
        }
    }
}

/// 삼각측량된 점.
#[derive(Clone, Debug)]
pub struct TriangulatedPoint {
    pub point: Point3<f64>,
    /// 관측 광선 쌍 사이의 최대 각(rad).
    pub max_ray_angle_rad: f64,
    /// 정규화 좌표 재투영 오차의 RMS.
    pub rms_reprojection: f64,
}

/// 거절 사유.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangulationFailure {
    TooFewViews,
    Degenerate,
    SmallRayAngle,
    BehindCamera,
    LargeReprojection,
}

/// DLT 만으로 구한 점(검사 없음).
pub fn triangulate_dlt(obs: &[(Pose, Vector2<f64>)]) -> Option<Point3<f64>> {
    if obs.len() < 2 {
        return None;
    }
    let mut ata = SMatrix::<f64, 4, 4>::zeros();
    for (pose, n) in obs {
        let r = pose.rotation.matrix();
        let t = pose.translation;
        let row = |k: usize| SMatrix::<f64, 1, 4>::new(r[(k, 0)], r[(k, 1)], r[(k, 2)], t[k]);
        let p3 = row(2);
        for a in [p3 * n.x - row(0), p3 * n.y - row(1)] {
            let a = a / a.norm().max(1e-300);
            ata += a.transpose() * a;
        }
    }
    let eig = ata.symmetric_eigen();
    let h = eig.eigenvectors.column(eig.eigenvalues.imin());
    (h[3].abs() > 1e-12).then(|| Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
}

fn reprojection(obs: &[(Pose, Vector2<f64>)], x: &Point3<f64>) -> Option<f64> {
    let mut sum = 0.0;
    for (pose, n) in obs {
        let xc = pose.transform(x);
        if xc.z <= 0.0 {
            return None;
        }
        sum += (Vector2::new(xc.x / xc.z, xc.y / xc.z) - n).norm_squared();
    }
    Some((sum / obs.len() as f64).sqrt())
}

/// 가우스–뉴턴 정밀화. 비용이 줄지 않으면 멈춘다.
pub fn refine_point(
    obs: &[(Pose, Vector2<f64>)],
    x0: &Point3<f64>,
    iterations: usize,
) -> Point3<f64> {
    let mut x = *x0;
    let Some(mut cost) = reprojection(obs, &x) else {
        return x;
    };
    for _ in 0..iterations {
        let mut h = Matrix3::zeros();
        let mut g = Vector3::zeros();
        for (pose, n) in obs {
            let xc = pose.transform(&x);
            let iz = 1.0 / xc.z;
            let r = Vector2::new(xc.x * iz - n.x, xc.y * iz - n.y);
            let dproj =
                nalgebra::Matrix2x3::new(iz, 0.0, -xc.x * iz * iz, 0.0, iz, -xc.y * iz * iz);
            let j = dproj * pose.rotation.matrix();
            h += j.transpose() * j;
            g += j.transpose() * r;
        }
        let Some(step) = h.try_inverse().map(|hi| -(hi * g)) else {
            break;
        };
        let cand = x + step;
        match reprojection(obs, &cand) {
            Some(c) if c < cost => {
                let done = cost - c < 1e-15;
                x = cand;
                cost = c;
                if done {
                    break;
                }
            }
            _ => break,
        }
    }
    x
}

/// 다시점 삼각측량 + 검사.
pub fn triangulate_multiview(
    obs: &[(Pose, Vector2<f64>)],
    cfg: &TriangulationConfig,
) -> Result<TriangulatedPoint, TriangulationFailure> {
    if obs.len() < 2 {
        return Err(TriangulationFailure::TooFewViews);
    }
    let x0 = triangulate_dlt(obs).ok_or(TriangulationFailure::Degenerate)?;
    let x = if cfg.refine_iterations > 0 && reprojection(obs, &x0).is_some() {
        refine_point(obs, &x0, cfg.refine_iterations)
    } else {
        x0
    };
    if !x.coords.iter().all(|v| v.is_finite()) {
        return Err(TriangulationFailure::Degenerate);
    }
    let rays: Vec<Vector3<f64>> = obs
        .iter()
        .map(|(pose, _)| (x - pose.center()).normalize())
        .collect();
    let mut max_angle: f64 = 0.0;
    for (a, ra) in rays.iter().enumerate() {
        for rb in &rays[a + 1..] {
            max_angle = max_angle.max(ra.cross(rb).norm().atan2(ra.dot(rb)));
        }
    }
    if max_angle < cfg.min_ray_angle_rad {
        return Err(TriangulationFailure::SmallRayAngle);
    }
    let rms = reprojection(obs, &x).ok_or(TriangulationFailure::BehindCamera)?;
    let worst = obs
        .iter()
        .map(|(pose, n)| {
            let xc = pose.transform(&x);
            (Vector2::new(xc.x / xc.z, xc.y / xc.z) - n).norm()
        })
        .fold(0.0, f64::max);
    if worst > cfg.max_reprojection {
        return Err(TriangulationFailure::LargeReprojection);
    }
    Ok(TriangulatedPoint {
        point: x,
        max_ray_angle_rad: max_angle,
        rms_reprojection: rms,
    })
}

/// 자국(트랙) 묶음을 삼각측량해 초벌 모델을 만든다. 트랙은 (카메라 번호, 정규화 좌표) 목록.
/// 자세가 없는 카메라의 관측은 뺀다.
pub fn triangulate_tracks(
    poses: &[Option<Pose>],
    tracks: &[Vec<(usize, Vector2<f64>)>],
    cfg: &TriangulationConfig,
) -> Vec<Result<TriangulatedPoint, TriangulationFailure>> {
    tracks
        .iter()
        .map(|track| {
            let obs: Vec<(Pose, Vector2<f64>)> = track
                .iter()
                .filter_map(|&(cam, n)| poses.get(cam).copied().flatten().map(|p| (p, n)))
                .collect();
            triangulate_multiview(&obs, cfg)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rotation3;

    fn look(center: Point3<f64>, target: Point3<f64>) -> Pose {
        let f = (target - center).normalize();
        let right = f.cross(&Vector3::z()).normalize();
        let down = f.cross(&right);
        let r = Matrix3::from_rows(&[right.transpose(), down.transpose(), f.transpose()]);
        Pose::from_center(Rotation3::from_matrix_unchecked(r), &center)
    }

    fn project(p: &Pose, x: &Point3<f64>) -> Vector2<f64> {
        let xc = p.transform(x);
        Vector2::new(xc.x / xc.z, xc.y / xc.z)
    }

    #[test]
    fn multiview_recovers_point_with_noise() {
        let target = Point3::new(0.0, 0.0, 0.0);
        let poses: Vec<Pose> = (0..5)
            .map(|k| look(Point3::new(-20.0 + 2.0 * k as f64, -10.0, 30.0), target))
            .collect();
        let x = Point3::new(1.0, -2.0, 0.5);
        // 결정적 잡음 ±1e-3(정규화 좌표, 약 1.5 px).
        let obs: Vec<(Pose, Vector2<f64>)> = poses
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let s = if k % 2 == 0 { 1.0 } else { -1.0 };
                (*p, project(p, &x) + Vector2::new(1e-3 * s, -7e-4 * s))
            })
            .collect();
        let exact: Vec<_> = poses.iter().map(|p| (*p, project(p, &x))).collect();
        let e = triangulate_multiview(&exact, &TriangulationConfig::default()).unwrap();
        assert!((e.point - x).norm() < 1e-8);
        let r = triangulate_multiview(&obs, &TriangulationConfig::default()).unwrap();
        // 기선 8 m, 거리 ~35 m: 1e-3 잡음에서 깊이 오차는 수십 cm 이하.
        assert!((r.point - x).norm() < 0.3, "{}", (r.point - x).norm());
        assert!(r.rms_reprojection < 1.5e-3);
    }

    #[test]
    fn rejects_small_angle_and_behind() {
        let target = Point3::origin();
        let a = look(Point3::new(0.0, 0.0, 30.0), target);
        let b = look(Point3::new(0.3, 0.0, 30.0), target);
        let x = Point3::new(0.5, 0.5, 0.0);
        let obs = vec![(a, project(&a, &x)), (b, project(&b, &x))];
        assert!(triangulate_multiview(&obs, &TriangulationConfig::default()).is_err());
        // 두 카메라가 서로 반대 방향을 볼 때 한 쪽 뒤에 놓이는 점.
        let c = look(Point3::new(10.0, 0.0, 30.0), Point3::new(40.0, 0.0, 0.0));
        let d = look(Point3::new(0.0, 0.0, 30.0), target);
        let behind = Point3::new(5.0, 0.0, 0.0);
        let xc = c.transform(&behind);
        let obs = vec![
            (c, Vector2::new(xc.x / xc.z, xc.y / xc.z)),
            (d, project(&d, &behind)),
        ];
        assert!(triangulate_multiview(&obs, &TriangulationConfig::default()).is_err());
        assert_eq!(
            triangulate_multiview(&obs[..1], &TriangulationConfig::default()).unwrap_err(),
            TriangulationFailure::TooFewViews
        );
    }
}
