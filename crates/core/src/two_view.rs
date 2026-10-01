//! 두 시점 기하: 본질 행렬, 상대 자세 복원, 삼각측량.
//!
//! 규약: 정규화 좌표 n = K⁻¹ [u v 1]ᵀ, 카메라 좌표 x_c = R X + t.
//! 첫 카메라를 [I | 0], 둘째를 [R | t] 로 둘 때 n2ᵀ E n1 = 0, E = [t]× R.

use crate::camera::Intrinsics;
use crate::matching::fundamental_8pt;
use crate::math::{skew, Matrix3, Point3, Rotation3, SMatrix, Vector2, Vector3};

/// 정규화 좌표 대응으로 본질 행렬을 구한다(정규화 8점 + 특이값 (1,1,0) 투영).
pub fn essential_8pt(n1: &[Vector2<f64>], n2: &[Vector2<f64>]) -> Option<Matrix3<f64>> {
    fundamental_8pt(n1, n2).map(|e| project_to_essential(&e))
}

/// 픽셀 기본 행렬 F 를 본질 행렬 E = K2ᵀ F K1 로 바꾼다(본질 행렬 다양체로 투영).
pub fn essential_from_fundamental(
    f: &Matrix3<f64>,
    k1: &Intrinsics,
    k2: &Intrinsics,
) -> Matrix3<f64> {
    project_to_essential(&(kmat(k2).transpose() * f * kmat(k1)))
}

fn kmat(k: &Intrinsics) -> Matrix3<f64> {
    Matrix3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

/// 가장 가까운 본질 행렬(특이값 1,1,0)로 투영한다.
pub fn project_to_essential(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    // nalgebra 특이값은 정렬되지 않을 수 있어 가장 작은 것을 0 으로 둔다.
    let mut s = Vector3::new(1.0, 1.0, 1.0);
    s[svd.singular_values.imin()] = 0.0;
    u * Matrix3::from_diagonal(&s) * vt
}

/// E 의 네 가지 분해 (R, t) 후보. t 는 단위 벡터.
pub fn decompose_essential(e: &Matrix3<f64>) -> [(Rotation3<f64>, Vector3<f64>); 4] {
    let svd = e.svd(true, true);
    let (mut u, mut vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    // 영 특이값 축을 셋째 열로 옮긴다.
    let k = svd.singular_values.imin();
    if k != 2 {
        u.swap_columns(k, 2);
        vt.swap_rows(k, 2);
    }
    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = Rotation3::from_matrix(&(u * w * vt));
    let r2 = Rotation3::from_matrix(&(u * w.transpose() * vt));
    let t: Vector3<f64> = u.column(2).into();
    [(r1, t), (r1, -t), (r2, t), (r2, -t)]
}

/// 선형(DLT) 삼각측량. 카메라 1 = [I|0], 카메라 2 = [R|t], 입력은 정규화 좌표.
pub fn triangulate(
    r: &Rotation3<f64>,
    t: &Vector3<f64>,
    n1: &Vector2<f64>,
    n2: &Vector2<f64>,
) -> Option<Point3<f64>> {
    let p2 = r.matrix();
    let mut a = SMatrix::<f64, 4, 4>::zeros();
    a.set_row(0, &SMatrix::<f64, 1, 4>::new(-1.0, 0.0, n1.x, 0.0));
    a.set_row(1, &SMatrix::<f64, 1, 4>::new(0.0, -1.0, n1.y, 0.0));
    for (i, (c, row)) in [(n2.x, 0usize), (n2.y, 1)].into_iter().enumerate() {
        let v = |j: usize| c * p2[(2, j)] - p2[(row, j)];
        a.set_row(
            2 + i,
            &SMatrix::<f64, 1, 4>::new(v(0), v(1), v(2), c * t[2] - t[row]),
        );
    }
    let eig = (a.transpose() * a).symmetric_eigen();
    let h = eig.eigenvectors.column(eig.eigenvalues.imin());
    (h[3].abs() > 1e-12).then(|| Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
}

/// 상대 자세 복원 결과.
#[derive(Clone, Debug)]
pub struct RelativePose {
    pub rotation: Rotation3<f64>,
    /// 단위 길이 이동(스케일 미정).
    pub translation: Vector3<f64>,
    /// 두 카메라 앞에 삼각측량된 대응의 표시.
    pub in_front: Vec<bool>,
}

/// E 의 네 후보 중 두 카메라 앞(양의 깊이)에 놓이는 점이 가장 많은 것을 고른다.
pub fn recover_pose(
    e: &Matrix3<f64>,
    n1: &[Vector2<f64>],
    n2: &[Vector2<f64>],
) -> Option<RelativePose> {
    let mut best: Option<RelativePose> = None;
    let mut best_n = 0;
    for (r, t) in decompose_essential(e) {
        let in_front: Vec<bool> = n1
            .iter()
            .zip(n2)
            .map(|(a, b)| {
                triangulate(&r, &t, a, b)
                    .map(|x| x.z > 0.0 && (r * x.coords + t).z > 0.0)
                    .unwrap_or(false)
            })
            .collect();
        let n = in_front.iter().filter(|&&b| b).count();
        if n > best_n {
            best_n = n;
            best = Some(RelativePose {
                rotation: r,
                translation: t,
                in_front,
            });
        }
    }
    best
}

/// 상대 자세가 주어졌을 때 E = [t]× R.
pub fn essential_from_pose(r: &Rotation3<f64>, t: &Vector3<f64>) -> Matrix3<f64> {
    skew(t) * r.matrix()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Camera, Pose};
    use crate::math::rotation_angle_between;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let (u1, u2) = (self.next().max(1e-300), self.next());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// 드론처럼 아래를 보는 두 카메라(기선 3 m, 고도 ~40 m)와 지면 근처 점.
    fn scene(n: usize, sigma_px: f64, seed: u64) -> Scene {
        let k = Intrinsics::from_hfov(960, 540, 70f64.to_radians());
        let r1 = Rotation3::from_euler_angles(0.04, -0.03, 0.2);
        let r2 = Rotation3::from_euler_angles(-0.05, 0.06, 0.28);
        let c1 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r1, &Point3::new(0.0, 0.0, -40.0)),
        };
        let c2 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r2, &Point3::new(3.0, 0.8, -40.5)),
        };
        let mut rng = Lcg(seed);
        let (mut x1, mut x2, mut pts) = (vec![], vec![], vec![]);
        while x1.len() < n {
            let x = Point3::new(
                (rng.next() - 0.5) * 40.0,
                (rng.next() - 0.5) * 24.0,
                (rng.next() - 0.5) * 10.0,
            );
            if let (Some(p), Some(q)) = (c1.project(&x), c2.project(&x)) {
                let noise = |r: &mut Lcg| Vector2::new(r.gauss(), r.gauss()) * sigma_px;
                x1.push(k.to_normalized(&(p + noise(&mut rng))));
                x2.push(k.to_normalized(&(q + noise(&mut rng))));
                pts.push(x);
            }
        }
        Scene {
            c1,
            c2,
            x1,
            x2,
            pts,
        }
    }

    struct Scene {
        c1: Camera,
        c2: Camera,
        x1: Vec<Vector2<f64>>,
        x2: Vec<Vector2<f64>>,
        pts: Vec<Point3<f64>>,
    }

    impl Scene {
        fn rel(&self) -> (Rotation3<f64>, Vector3<f64>) {
            let r = self.c2.pose.rotation * self.c1.pose.rotation.inverse();
            let t = self.c2.pose.translation - r * self.c1.pose.translation;
            (r, t)
        }
    }

    #[test]
    fn decomposition_contains_true_pose() {
        let s = scene(10, 0.0, 1);
        let (r, t) = s.rel();
        let e = essential_from_pose(&r, &t);
        let tu = t.normalize();
        let hit = decompose_essential(&e)
            .iter()
            .any(|(rc, tc)| rotation_angle_between(rc, &r) < 1e-9 && (tc - tu).norm() < 1e-9);
        assert!(hit, "네 후보 중 정답 없음");
    }

    #[test]
    fn triangulation_exact_without_noise() {
        let s = scene(50, 0.0, 2);
        let (r, t) = s.rel();
        let mut worst: f64 = 0.0;
        for i in 0..s.x1.len() {
            let x = triangulate(&r, &t, &s.x1[i], &s.x2[i]).unwrap();
            // 카메라 1 좌표계의 정답 점.
            let g = s.c1.pose.transform(&s.pts[i]);
            worst = worst.max((x.coords - g).norm());
        }
        eprintln!("triangulation worst err={worst:.2e} m");
        assert!(worst < 1e-6, "삼각측량 오차 {worst} m");
    }

    #[test]
    fn relative_pose_under_noise() {
        // (σ px, 회전 오차 상한 도, 이동 방향 오차 상한 도, 삼각측량 중앙값 오차 상한 m)
        // 선형 8점만(비선형 정밀화 없음) 기준. 기선 3 m / 깊이 ~40 m 라 이동 방향이 잡음에 민감하다.
        // 상한은 시드 7·11·23 측정 최댓값의 약 1.5 배.
        for (sigma, max_rot, max_dir, max_pt) in [
            (0.0, 1e-6, 1e-6, 1e-6),
            (0.5, 0.2, 4.5, 1.7),
            (1.0, 0.4, 12.0, 4.0),
        ] {
            let s = scene(200, sigma, 7);
            let e = essential_8pt(&s.x1, &s.x2).unwrap();
            let rp = recover_pose(&e, &s.x1, &s.x2).unwrap();
            let (r, t) = s.rel();
            let rot_err = rotation_angle_between(&rp.rotation, &r).to_degrees();
            let dir_err = rp.translation.angle(&t.normalize()).to_degrees();
            let front = rp.in_front.iter().filter(|&&b| b).count() as f64 / s.x1.len() as f64;
            // 정답 기선 길이로 스케일을 맞춰 삼각측량 오차를 잰다.
            let tt = rp.translation * t.norm();
            let mut errs: Vec<f64> = (0..s.x1.len())
                .filter_map(|i| {
                    let x = triangulate(&rp.rotation, &tt, &s.x1[i], &s.x2[i])?;
                    Some((x.coords - s.c1.pose.transform(&s.pts[i])).norm())
                })
                .collect();
            errs.sort_by(|a, b| a.total_cmp(b));
            let med = errs[errs.len() / 2];
            eprintln!(
                "sigma={sigma} rot_err={rot_err:.4} deg dir_err={dir_err:.3} deg \
                 front={front:.3} median_pt_err={med:.3} m"
            );
            assert!(rot_err < max_rot, "회전 오차 {rot_err} 도");
            assert!(dir_err < max_dir, "이동 방향 오차 {dir_err} 도");
            assert!(front > 0.97, "앞쪽 비율 {front}");
            assert!(med < max_pt, "삼각측량 중앙값 {med} m");
        }
    }

    #[test]
    fn essential_from_true_fundamental_matches_pose() {
        let s = scene(10, 0.0, 3);
        let f = crate::matching::fundamental_from_cameras(&s.c1, &s.c2);
        let e = essential_from_fundamental(&f, &s.c1.intrinsics, &s.c2.intrinsics);
        let (r, t) = s.rel();
        let g = project_to_essential(&essential_from_pose(&r, &t));
        let d = (e - g).norm().min((e + g).norm());
        assert!(d < 1e-9, "E 차이 {d}");
    }
}
