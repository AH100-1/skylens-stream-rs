//! 공용 수학 타입과 보조 함수.

pub use nalgebra::{
    Matrix2, Matrix3, Point3, Rotation3, SMatrix, UnitQuaternion, Vector2, Vector3,
};

/// 벡터 a 에 대한 반대칭 행렬 [a]_x (a × b = [a]_x b).
pub fn skew(a: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -a.z, a.y, a.z, 0.0, -a.x, -a.y, a.x, 0.0)
}

/// 두 회전 사이의 각도(라디안). 거의 같은 회전에서도 NaN 이 나지 않도록 사원수로 잰다.
pub fn rotation_angle_between(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    UnitQuaternion::from_rotation_matrix(&(a.inverse() * b)).angle()
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

    #[test]
    fn rotation_angle_of_nearly_equal_is_finite() {
        let a = Rotation3::from_euler_angles(0.04, -0.03, 0.2);
        let b = Rotation3::from_matrix_unchecked(a.matrix() * (1.0 + 1e-15));
        let d = rotation_angle_between(&a, &b);
        assert!(d.is_finite() && d < 1e-6, "각도 {d}");
    }
}
