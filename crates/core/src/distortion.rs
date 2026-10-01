//! 방사·접선 왜곡(k1,k2,p1,p2)과 투영 야코비안. 모델: r²=x²+y², x_d = x(1+k1 r²+k2 r⁴) + 2p1 xy + p2(r²+2x²), y_d = y(1+k1 r²+k2 r⁴) + p1(r²+2y²) + 2p2 xy.

use crate::camera::Pose;
use crate::math::{skew, Matrix2, Point3, SMatrix, Vector2, Vector3};

/// 정규 좌표 왜곡 계수.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Distortion {
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
}

impl Distortion {
    /// 왜곡 전 정규 좌표 → 왜곡 후 정규 좌표.
    pub fn distort(&self, n: &Vector2<f64>) -> Vector2<f64> {
        let (x, y) = (n.x, n.y);
        let r2 = x * x + y * y;
        let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        Vector2::new(
            x * rad + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x),
            y * rad + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y,
        )
    }

    /// ∂distort/∂n.
    pub fn jacobian_point(&self, n: &Vector2<f64>) -> Matrix2<f64> {
        let (x, y) = (n.x, n.y);
        let r2 = x * x + y * y;
        let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        let drad = self.k1 + 2.0 * self.k2 * r2; // ∂rad/∂r2
        Matrix2::new(
            rad + 2.0 * x * x * drad + 2.0 * self.p1 * y + 6.0 * self.p2 * x,
            2.0 * x * y * drad + 2.0 * self.p1 * x + 2.0 * self.p2 * y,
            2.0 * x * y * drad + 2.0 * self.p1 * x + 2.0 * self.p2 * y,
            rad + 2.0 * y * y * drad + 6.0 * self.p1 * y + 2.0 * self.p2 * x,
        )
    }

    /// ∂distort/∂(k1,k2,p1,p2).
    pub fn jacobian_params(&self, n: &Vector2<f64>) -> SMatrix<f64, 2, 4> {
        let (x, y) = (n.x, n.y);
        let r2 = x * x + y * y;
        SMatrix::<f64, 2, 4>::new(
            x * r2,
            x * r2 * r2,
            2.0 * x * y,
            r2 + 2.0 * x * x,
            y * r2,
            y * r2 * r2,
            r2 + 2.0 * y * y,
            2.0 * x * y,
        )
    }

    /// 계수가 모두 0 인가(핀홀).
    pub fn is_zero(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0
    }

    /// 왜곡 역변환(뉴턴 반복). 수렴하지 않으면 None.
    pub fn undistort(&self, nd: &Vector2<f64>) -> Option<Vector2<f64>> {
        let (n, ok) = self.undistort_best(nd);
        ok.then_some(n)
    }

    /// 왜곡 역변환의 마지막 반복값과 수렴 여부(잔차 < 1e-10).
    /// 야코비안이 특이해지면 그 직전 값을 돌려준다. 계수가 0 이면 입력 그대로(수렴).
    pub fn undistort_best(&self, nd: &Vector2<f64>) -> (Vector2<f64>, bool) {
        if self.is_zero() {
            return (*nd, nd.iter().all(|v| v.is_finite()));
        }
        let mut n = *nd;
        for _ in 0..30 {
            let r = self.distort(&n) - nd;
            if r.norm() < 1e-14 {
                return (n, true);
            }
            match self.jacobian_point(&n).try_inverse() {
                Some(ji) => n -= ji * r,
                None => break,
            }
        }
        let ok = (self.distort(&n) - nd).norm() < 1e-10;
        (n, ok)
    }
}

/// 왜곡 전 정규 좌표 → 픽셀. 핀홀·왜곡 카메라가 함께 쓰는 유일한 투영 경로.
/// 계수가 0 이면 왜곡 계산을 건너뛰어 핀홀 식 fx·x + cx 와 비트 단위로 같다.
pub fn normalized_to_pixel(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    dist: &Distortion,
    n: &Vector2<f64>,
) -> Vector2<f64> {
    let d = if dist.is_zero() { *n } else { dist.distort(n) };
    Vector2::new(fx * d.x + cx, fy * d.y + cy)
}

/// 픽셀 → 왜곡 전 정규 좌표(광선 방향 (x, y, 1))의 마지막 반복값과 수렴 여부.
/// 핀홀·왜곡 카메라가 함께 쓰는 유일한 정규화 경로. 계수가 0 이면 (p − c)/f 와 비트 단위로 같다.
pub fn pixel_to_normalized(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    dist: &Distortion,
    p: &Vector2<f64>,
) -> (Vector2<f64>, bool) {
    dist.undistort_best(&Vector2::new((p.x - cx) / fx, (p.y - cy) / fy))
}

/// 왜곡 포함 카메라 내부 파라미터: [fx, fy, cx, cy, k1, k2, p1, p2].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistortedIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub dist: Distortion,
}

/// 투영 결과와 야코비안.
#[derive(Clone, Copy, Debug)]
pub struct ProjectionJacobian {
    pub pixel: Vector2<f64>,
    /// ∂픽셀/∂세계점.
    pub d_point: SMatrix<f64, 2, 3>,
    /// ∂픽셀/∂(ω, t): 회전은 왼쪽 섭동 R ← exp([ω]×)·R.
    pub d_pose: SMatrix<f64, 2, 6>,
    /// ∂픽셀/∂[fx, fy, cx, cy, k1, k2, p1, p2].
    pub d_intrinsics: SMatrix<f64, 2, 8>,
}

impl DistortedIntrinsics {
    pub fn project_camera(&self, xc: &Vector3<f64>) -> Vector2<f64> {
        normalized_to_pixel(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            &self.dist,
            &Vector2::new(xc.x / xc.z, xc.y / xc.z),
        )
    }

    /// 픽셀 → 왜곡 없는 정규 좌표 (광선 방향 (x, y, 1)). 역왜곡이 수렴하지 않으면 None.
    /// [`crate::camera::Intrinsics::to_normalized`] 와 같은 경로([`pixel_to_normalized`])다.
    pub fn unproject(&self, p: &Vector2<f64>) -> Option<Vector2<f64>> {
        let (n, ok) = pixel_to_normalized(self.fx, self.fy, self.cx, self.cy, &self.dist, p);
        ok.then_some(n)
    }

    /// 영상 크기를 붙여 [`crate::camera::Intrinsics`] 로 바꾼다(왜곡 계수 유지).
    pub fn with_size(&self, width: u32, height: u32) -> crate::camera::Intrinsics {
        crate::camera::Intrinsics {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            width,
            height,
            dist: self.dist,
        }
    }

    /// 세계 점 투영과 야코비안. 카메라 뒤면 None.
    pub fn project_with_jacobian(
        &self,
        pose: &Pose,
        x: &Point3<f64>,
    ) -> Option<ProjectionJacobian> {
        let xc = pose.transform(x);
        if xc.z <= 0.0 {
            return None;
        }
        let iz = 1.0 / xc.z;
        let n = Vector2::new(xc.x * iz, xc.y * iz);
        let d = self.dist.distort(&n);
        let pixel = Vector2::new(self.fx * d.x + self.cx, self.fy * d.y + self.cy);
        let f = Matrix2::new(self.fx, 0.0, 0.0, self.fy);
        // ∂n/∂xc
        let dn = SMatrix::<f64, 2, 3>::new(iz, 0.0, -xc.x * iz * iz, 0.0, iz, -xc.y * iz * iz);
        let d_xc = f * self.dist.jacobian_point(&n) * dn;
        let r = pose.rotation.matrix();
        let d_point = d_xc * r;
        // xc = exp([ω]×)R X + t → ∂xc/∂ω = -[R X]×, ∂xc/∂t = I.
        let rx = r * x.coords;
        let mut d_pose = SMatrix::<f64, 2, 6>::zeros();
        d_pose
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&(d_xc * -skew(&rx)));
        d_pose.fixed_view_mut::<2, 3>(0, 3).copy_from(&d_xc);
        let mut d_intrinsics = SMatrix::<f64, 2, 8>::zeros();
        d_intrinsics[(0, 0)] = d.x;
        d_intrinsics[(1, 1)] = d.y;
        d_intrinsics[(0, 2)] = 1.0;
        d_intrinsics[(1, 3)] = 1.0;
        d_intrinsics
            .fixed_view_mut::<2, 4>(0, 4)
            .copy_from(&(f * self.dist.jacobian_params(&n)));
        Some(ProjectionJacobian {
            pixel,
            d_point,
            d_pose,
            d_intrinsics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rotation3;

    fn intr() -> DistortedIntrinsics {
        DistortedIntrinsics {
            fx: 820.0,
            fy: 815.0,
            cx: 481.0,
            cy: 268.0,
            dist: Distortion {
                k1: -0.12,
                k2: 0.03,
                p1: 0.001,
                p2: -0.0015,
            },
        }
    }

    fn pose() -> Pose {
        Pose::from_center(
            Rotation3::from_euler_angles(2.4, 0.1, -0.3),
            &Point3::new(1.0, 2.0, 30.0),
        )
    }

    fn rel_err(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1.0)
    }

    #[test]
    fn undistort_roundtrip() {
        let d = intr().dist;
        let mut worst: f64 = 0.0;
        for i in 0..200 {
            let n = Vector2::new(
                (i as f64 * 0.137) % 1.2 - 0.6,
                (i as f64 * 0.071) % 0.7 - 0.35,
            );
            let back = d.undistort(&d.distort(&n)).unwrap();
            worst = worst.max((back - n).norm());
        }
        assert!(worst < 1e-12, "{worst}");
    }

    #[test]
    fn jacobians_match_numeric() {
        let k = intr();
        let p = pose();
        let h = 1e-6;
        let mut worst: f64 = 0.0;
        let pts = [
            Point3::new(3.0, -4.0, 0.5),
            Point3::new(-6.0, 1.0, 2.0),
            Point3::new(0.0, 8.0, -1.0),
        ];
        for x in pts {
            let j = k.project_with_jacobian(&p, &x).unwrap();
            for a in 0..3 {
                let (mut xp, mut xm) = (x, x);
                xp[a] += h;
                xm[a] -= h;
                let num = (k.project_with_jacobian(&p, &xp).unwrap().pixel
                    - k.project_with_jacobian(&p, &xm).unwrap().pixel)
                    / (2.0 * h);
                for r in 0..2 {
                    worst = worst.max(rel_err(j.d_point[(r, a)], num[r]));
                }
            }
            for a in 0..6 {
                let perturb = |s: f64| {
                    let mut q = p;
                    if a < 3 {
                        let mut w = Vector3::zeros();
                        w[a] = s;
                        q.rotation = Rotation3::new(w) * q.rotation;
                    } else {
                        q.translation[a - 3] += s;
                    }
                    k.project_with_jacobian(&q, &x).unwrap().pixel
                };
                let num = (perturb(h) - perturb(-h)) / (2.0 * h);
                for r in 0..2 {
                    worst = worst.max(rel_err(j.d_pose[(r, a)], num[r]));
                }
            }
            for a in 0..8 {
                let perturb = |s: f64| {
                    let mut q = k;
                    match a {
                        0 => q.fx += s,
                        1 => q.fy += s,
                        2 => q.cx += s,
                        3 => q.cy += s,
                        4 => q.dist.k1 += s,
                        5 => q.dist.k2 += s,
                        6 => q.dist.p1 += s,
                        _ => q.dist.p2 += s,
                    }
                    q.project_with_jacobian(&p, &x).unwrap().pixel
                };
                let num = (perturb(h) - perturb(-h)) / (2.0 * h);
                for r in 0..2 {
                    worst = worst.max(rel_err(j.d_intrinsics[(r, a)], num[r]));
                }
            }
        }
        eprintln!("jacobian_worst_rel {worst:.3e}");
        assert!(worst < 1e-6, "야코비안 최대 상대 오차 {worst}");
    }

    #[test]
    fn unproject_inverts_project() {
        let k = intr();
        let xc = Vector3::new(4.0, -2.5, 20.0);
        let px = k.project_camera(&xc);
        let n = k.unproject(&px).unwrap();
        assert!((n - Vector2::new(0.2, -0.125)).norm() < 1e-12);
    }

    #[test]
    fn zero_distortion_is_pinhole() {
        let mut k = intr();
        k.dist = Distortion::default();
        let px = k.project_camera(&Vector3::new(1.0, 2.0, 10.0));
        assert!((px - Vector2::new(481.0 + 82.0, 268.0 + 163.0)).norm() < 1e-12);
    }
}
