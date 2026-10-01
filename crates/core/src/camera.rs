//! 카메라 내부 파라미터 [`Intrinsics`](초점·주점·렌즈 왜곡), 자세 [`Pose`], 둘을 묶은 [`Camera`].
//!
//! 규약: 세계 점 X 의 카메라 좌표 = R·X + t, 카메라 중심 C = -Rᵀt.
//! 카메라 좌표계는 x 오른쪽, y 아래, z 앞(광축).
//!
//! 픽셀 좌표는 연속 좌표다: 화소 (i, j) 의 중심이 (i + 0.5, j + 0.5) 이고,
//! [`Intrinsics::from_hfov`] 의 주점 (w/2, h/2) 은 영상의 기하 중심이다.
//! 특징점 검출기(`features::Keypoint`)도 같은 연속 좌표를 내보내므로 보정 없이
//! [`Intrinsics::to_normalized`] 에 넣는다. 정수 화소 번호로 적은 위치만
//! [`Intrinsics::index_to_normalized`] 를 쓴다.
//!
//! 카메라 형은 하나다. [`Intrinsics`] 는 방사·접선 왜곡 계수(k1,k2,p1,p2, 모델은
//! [`crate::distortion`])를 가지며 계수가 모두 0 이면 핀홀이다.
//! - 투영 [`Intrinsics::to_pixel`]·[`Camera::project`] 는 왜곡을 적용한다.
//! - 정규화 [`Intrinsics::to_normalized`]·[`Intrinsics::unproject`] 는 왜곡을 되돌린
//!   광선 방향 (x, y, 1) 을 준다. 매칭·두 시점·번들 조정 초기값이 모두 이 경로를 쓰면
//!   같은 관측이 한 규약으로만 정규화된다.
//! - 왜곡을 무시한 선형 정규화는 [`Intrinsics::pinhole_normalized`] 로 따로 둔다.
//!
//! [`crate::distortion::DistortedIntrinsics`] 는 번들 조정이 고치는 8개 값
//! [fx, fy, cx, cy, k1, k2, p1, p2] 묶음(영상 크기 없음)으로 남아 있으며
//! [`Intrinsics::params`]·[`Intrinsics::from_params`] 로 오간다.

use crate::distortion::{DistortedIntrinsics, Distortion};
use crate::math::{Point3, Rotation3, Vector2, Vector3};

/// 카메라 내부 파라미터(픽셀 단위) + 렌즈 왜곡. 왜곡 계수가 0 이면 핀홀.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
    /// 정규 좌표 왜곡 계수. 기본값(0)은 왜곡 없음.
    pub dist: Distortion,
}

impl Intrinsics {
    /// 수평 화각(라디안)으로 만든다. 주점은 영상 중심, 정사각 화소, 왜곡 없음.
    pub fn from_hfov(width: u32, height: u32, hfov: f64) -> Self {
        let f = 0.5 * width as f64 / (0.5 * hfov).tan();
        Self {
            fx: f,
            fy: f,
            cx: 0.5 * width as f64,
            cy: 0.5 * height as f64,
            width,
            height,
            dist: Distortion::default(),
        }
    }

    /// 번들 조정 파라미터 묶음과 영상 크기로 만든다.
    pub fn from_params(p: &DistortedIntrinsics, width: u32, height: u32) -> Self {
        Self {
            fx: p.fx,
            fy: p.fy,
            cx: p.cx,
            cy: p.cy,
            width,
            height,
            dist: p.dist,
        }
    }

    /// 번들 조정 파라미터 묶음 [fx, fy, cx, cy, k1, k2, p1, p2].
    pub fn params(&self) -> DistortedIntrinsics {
        DistortedIntrinsics {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            dist: self.dist,
        }
    }

    /// 같은 초점·주점·크기에 왜곡 계수를 바꾼 내부 파라미터.
    pub fn with_distortion(&self, dist: Distortion) -> Self {
        Self { dist, ..*self }
    }

    /// 왜곡 계수가 모두 0 인가.
    pub fn is_pinhole(&self) -> bool {
        self.dist == Distortion::default()
    }

    /// 정규 좌표 (x/z, y/z) → 픽셀. 왜곡을 적용한다.
    pub fn to_pixel(&self, n: &Vector2<f64>) -> Vector2<f64> {
        let d = self.dist.distort(n);
        Vector2::new(self.fx * d.x + self.cx, self.fy * d.y + self.cy)
    }

    /// 픽셀 → 왜곡을 되돌린 정규 좌표. 역변환이 수렴하지 않으면(시야 밖 강한 왜곡) None.
    pub fn unproject(&self, p: &Vector2<f64>) -> Option<Vector2<f64>> {
        self.dist.undistort(&self.pinhole_normalized(p))
    }

    /// 픽셀 → 왜곡을 되돌린 정규 좌표. 왜곡이 0 이면 선형 (p − c)/f 와 같다.
    /// 역변환이 수렴하지 않는 드문 경우에는 선형 정규화 값을 돌려준다
    /// (실패를 구별해야 하면 [`Self::unproject`]).
    pub fn to_normalized(&self, p: &Vector2<f64>) -> Vector2<f64> {
        let n = self.pinhole_normalized(p);
        if self.is_pinhole() {
            return n;
        }
        self.dist.undistort(&n).unwrap_or(n)
    }

    /// 왜곡을 무시한 선형 정규화 (p − c)/f.
    pub fn pinhole_normalized(&self, p: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new((p.x - self.cx) / self.fx, (p.y - self.cy) / self.fy)
    }

    /// 화소 번호 규약 위치(화소 (i, j) 의 중심을 (i, j) 로 적음) → 정규 좌표.
    /// 화소 중심 규약으로 +0.5 를 더한 뒤 [`Self::to_normalized`] 와 같다.
    /// 특징점 좌표는 이미 연속 좌표이므로 여기에 넣지 않는다.
    pub fn index_to_normalized(&self, p: &Vector2<f64>) -> Vector2<f64> {
        self.to_normalized(&Vector2::new(p.x + 0.5, p.y + 0.5))
    }

    pub fn contains(&self, p: &Vector2<f64>) -> bool {
        p.x >= 0.0 && p.y >= 0.0 && p.x < self.width as f64 && p.y < self.height as f64
    }
}

impl From<Intrinsics> for DistortedIntrinsics {
    fn from(k: Intrinsics) -> Self {
        k.params()
    }
}

/// 세계→카메라 강체 변환.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub rotation: Rotation3<f64>,
    pub translation: Vector3<f64>,
}

impl Pose {
    pub fn new(rotation: Rotation3<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// 중심 C 와 회전 R 로 만든다 (t = -R·C).
    pub fn from_center(rotation: Rotation3<f64>, center: &Point3<f64>) -> Self {
        let translation = -(rotation * center.coords);
        Self {
            rotation,
            translation,
        }
    }

    pub fn center(&self) -> Point3<f64> {
        Point3::from(-(self.rotation.inverse() * self.translation))
    }

    pub fn transform(&self, x: &Point3<f64>) -> Vector3<f64> {
        self.rotation * x.coords + self.translation
    }
}

/// 내부 + 외부 파라미터를 가진 카메라.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    pub pose: Pose,
}

impl Camera {
    /// 세계 점을 픽셀로 투영(왜곡 적용). 카메라 뒤(z ≤ 0)면 None.
    pub fn project(&self, x: &Point3<f64>) -> Option<Vector2<f64>> {
        let xc = self.pose.transform(x);
        if xc.z <= 0.0 {
            return None;
        }
        Some(
            self.intrinsics
                .to_pixel(&Vector2::new(xc.x / xc.z, xc.y / xc.z)),
        )
    }

    /// 픽셀과 깊이(카메라 z)로 세계 점을 복원(왜곡 되돌림).
    pub fn unproject(&self, p: &Vector2<f64>, depth: f64) -> Point3<f64> {
        let n = self.intrinsics.to_normalized(p);
        let xc = Vector3::new(n.x * depth, n.y * depth, depth);
        Point3::from(self.pose.rotation.inverse() * (xc - self.pose.translation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> Camera {
        let k = Intrinsics::from_hfov(1920, 1080, 70f64.to_radians());
        let r = Rotation3::from_euler_angles(0.2, -0.1, 0.4);
        let c = Point3::new(3.0, -2.0, 30.0);
        Camera {
            intrinsics: k,
            pose: Pose::from_center(r, &c),
        }
    }

    #[test]
    fn center_roundtrip() {
        let cam = test_camera();
        let c = cam.pose.center();
        assert!((c - Point3::new(3.0, -2.0, 30.0)).norm() < 1e-12);
        // 중심은 카메라 좌표 원점으로 간다.
        assert!(cam.pose.transform(&c).norm() < 1e-12);
    }

    #[test]
    fn project_unproject_roundtrip() {
        let cam = test_camera();
        let mut worst: f64 = 0.0;
        for i in 0..50 {
            let px = Vector2::new(37.0 * i as f64 % 1920.0, 21.0 * i as f64 % 1080.0);
            let depth = 5.0 + i as f64;
            let x = cam.unproject(&px, depth);
            let back = cam.project(&x).unwrap();
            worst = worst.max((back - px).norm());
            assert!((cam.pose.transform(&x).z - depth).abs() < 1e-9);
        }
        assert!(worst < 1e-9, "최대 왕복 오차 {worst} px");
    }

    #[test]
    fn behind_camera_is_none() {
        let cam = test_camera();
        let x = cam.unproject(&Vector2::new(960.0, 540.0), 10.0);
        // 중심 기준 반대쪽 점은 카메라 뒤.
        let behind = Point3::from(2.0 * cam.pose.center().coords - x.coords);
        assert!(cam.project(&behind).is_none());
    }

    #[test]
    fn index_convention_is_half_pixel_shift() {
        let k = Intrinsics::from_hfov(640, 480, 60f64.to_radians());
        // 화소 (319, 239) 의 중심 = (319.5, 239.5) 는 주점 (320, 240) 에서 반 화소 왼쪽 위.
        let n = k.index_to_normalized(&Vector2::new(319.0, 239.0));
        assert!((n.x * k.fx + 0.5).abs() < 1e-12);
        assert!((n.y * k.fy + 0.5).abs() < 1e-12);
        let c = k.index_to_normalized(&Vector2::new(319.5, 239.5));
        assert!(c.norm() < 1e-15);
    }

    #[test]
    fn zero_distortion_matches_pinhole() {
        let cam = test_camera();
        let d = cam.intrinsics.with_distortion(Distortion::default());
        for i in 0..20 {
            let px = Vector2::new(13.0 + 91.0 * i as f64, 7.0 + 49.0 * i as f64);
            let a = cam.intrinsics.to_normalized(&px);
            let b = d.unproject(&px).unwrap();
            assert!((a - b).norm() < 1e-12);
            let xc = Vector3::new(a.x * 4.0, a.y * 4.0, 4.0);
            assert!((d.params().project_camera(&xc) - px).norm() < 1e-9);
        }
    }

    /// 실제 렌즈 수준 왜곡(k1 = −0.12 등)에서 투영 → 정규화가 정답 광선을 되돌리고,
    /// 선형 정규화는 가장자리에서 화소 단위로 어긋남을 확인한다.
    #[test]
    fn distorted_normalization_recovers_true_ray() {
        let k = Intrinsics::from_hfov(1920, 1080, 70f64.to_radians()).with_distortion(Distortion {
            k1: -0.12,
            k2: 0.03,
            p1: 0.001,
            p2: -0.0015,
        });
        assert!(!k.is_pinhole());
        let mut worst: f64 = 0.0;
        let mut edge_lin: f64 = 0.0;
        for i in 0..40 {
            // 영상 전체에 퍼진 정답 광선(정규 좌표 |x| ≤ 0.65, |y| ≤ 0.37 ≈ 화각 안).
            let n = Vector2::new(
                -0.65 + 1.3 * (i as f64 / 39.0),
                0.37 * ((i * 7 % 40) as f64 / 20.0 - 1.0),
            );
            let px = k.to_pixel(&n);
            let back = k.to_normalized(&px);
            assert_eq!(Some(back), k.unproject(&px));
            worst = worst.max((back - n).norm() * k.fx);
            edge_lin = edge_lin.max((k.pinhole_normalized(&px) - n).norm() * k.fx);
        }
        // 뉴턴 역변환은 1e-14 정규 단위까지 수렴(distortion::undistort) → 화소로 1e-9 미만.
        assert!(worst < 1e-9, "왕복 오차 {worst} px");
        // k1 = −0.12, r ≈ 0.75 에서 r³k1·f ≈ 0.05·1370 ≈ 69 px: 왜곡 무시 정규화는 수십 px 틀린다.
        assert!(edge_lin > 20.0, "선형 정규화 가장자리 오차 {edge_lin} px");
        // 카메라 왕복도 왜곡을 포함해 맞는다.
        let cam = Camera {
            intrinsics: k,
            pose: test_camera().pose,
        };
        let px = Vector2::new(37.25, 1003.5);
        let x = cam.unproject(&px, 12.0);
        assert!((cam.project(&x).unwrap() - px).norm() < 1e-9);
    }

    #[test]
    fn params_roundtrip() {
        let k = Intrinsics::from_hfov(640, 480, 60f64.to_radians()).with_distortion(Distortion {
            k1: 0.1,
            ..Distortion::default()
        });
        let p: DistortedIntrinsics = k.into();
        assert_eq!(Intrinsics::from_params(&p, 640, 480), k);
        let xc = Vector3::new(0.3, -0.2, 2.0);
        let n = Vector2::new(0.15, -0.1);
        assert!((p.project_camera(&xc) - k.to_pixel(&n)).norm() < 1e-12);
    }

    #[test]
    fn hfov_focal_known() {
        // 화각 90° 이면 f = w/2.
        let k = Intrinsics::from_hfov(1000, 800, std::f64::consts::FRAC_PI_2);
        assert!((k.fx - 500.0).abs() < 1e-9);
    }
}
