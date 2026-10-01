//! 핀홀 카메라(왜곡 없음)와 자세. 왜곡 모델은 T02 에서 추가한다.
//!
//! 규약: 세계 점 X 의 카메라 좌표 = R·X + t, 카메라 중심 C = -Rᵀt.
//! 카메라 좌표계는 x 오른쪽, y 아래, z 앞(광축).

use crate::math::{Point3, Rotation3, Vector2, Vector3};

/// 핀홀 내부 파라미터(픽셀 단위).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    /// 수평 화각(라디안)으로 만든다. 주점은 영상 중심, 정사각 화소.
    pub fn from_hfov(width: u32, height: u32, hfov: f64) -> Self {
        let f = 0.5 * width as f64 / (0.5 * hfov).tan();
        Self {
            fx: f,
            fy: f,
            cx: 0.5 * width as f64,
            cy: 0.5 * height as f64,
            width,
            height,
        }
    }

    /// 정규 좌표 (x/z, y/z) → 픽셀.
    pub fn to_pixel(&self, n: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new(self.fx * n.x + self.cx, self.fy * n.y + self.cy)
    }

    /// 픽셀 → 정규 좌표.
    pub fn to_normalized(&self, p: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new((p.x - self.cx) / self.fx, (p.y - self.cy) / self.fy)
    }

    pub fn contains(&self, p: &Vector2<f64>) -> bool {
        p.x >= 0.0 && p.y >= 0.0 && p.x < self.width as f64 && p.y < self.height as f64
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
    /// 세계 점을 픽셀로 투영. 카메라 뒤(z ≤ 0)면 None.
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

    /// 픽셀과 깊이(카메라 z)로 세계 점을 복원.
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
    fn hfov_focal_known() {
        // 화각 90° 이면 f = w/2.
        let k = Intrinsics::from_hfov(1000, 800, std::f64::consts::FRAC_PI_2);
        assert!((k.fx - 500.0).abs() < 1e-9);
    }
}
