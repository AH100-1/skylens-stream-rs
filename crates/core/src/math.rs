//! 공용 수학 타입과 보조 함수.

pub use nalgebra::{
    Matrix2, Matrix3, Point3, Rotation3, SMatrix, UnitQuaternion, Vector2, Vector3,
};

/// 벡터 a 에 대한 반대칭 행렬 [a]_x (a × b = [a]_x b).
pub fn skew(a: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -a.z, a.y, a.z, 0.0, -a.x, -a.y, a.x, 0.0)
}

/// 두 회전 사이의 각도(라디안).
pub fn rotation_angle_between(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    (a.inverse() * b).angle()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_matches_cross_product() {
        let a = Vector3::new(0.3, -1.2, 2.5);
        let b = Vector3::new(-0.7, 0.4, 1.1);
        assert!((skew(&a) * b - a.cross(&b)).norm() < 1e-12);
    }

    #[test]
    fn rotation_angle_known() {
        let a = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.1);
        let b = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.35);
        assert!((rotation_angle_between(&a, &b) - 0.25).abs() < 1e-12);
    }
}
