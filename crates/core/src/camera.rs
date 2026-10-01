//! 핀홀 + 렌즈 왜곡 내부 파라미터 [`Intrinsics`], 자세 [`Pose`], 둘을 묶은 [`Camera`].
//!
//! 규약: 세계 점 X 의 카메라 좌표 = R·X + t, 카메라 중심 C = -Rᵀt.
//! 카메라 좌표계는 x 오른쪽, y 아래, z 앞(광축).
//!
//! 픽셀 좌표는 연속 좌표다: 화소 (i, j) 의 중심이 (i + 0.5, j + 0.5) 이고,
//! [`Intrinsics::from_hfov`] 의 주점 (w/2, h/2) 은 영상의 기하 중심이다.
//! 정수 화소 번호로 주어진 위치는 [`Intrinsics::index_to_normalized`] 로 정규화한다.
//!
//! 렌즈 왜곡(k1,k2,p1,p2, SPEC §3.3)은 [`Intrinsics::dist`] 에 들어 있고 카메라마다 공유한다.
//! 픽셀 ↔ 정규 좌표 변환은 한 경로뿐이다: 투영은
//! [`crate::distortion::normalized_to_pixel`], 정규화(역왜곡 포함)는
//! [`crate::distortion::pixel_to_normalized`]. [`Intrinsics::to_pixel`]·[`Intrinsics::to_normalized`]·
//! [`Intrinsics::unproject`]·[`Camera::project`]·[`Camera::unproject`] 와
//! [`crate::distortion::DistortedIntrinsics`](번들 조정의 매개변수 벡터 형태) 가 모두 이 둘을 부른다.
//! 따라서 매칭·두 시점·번들 조정이 같은 관측을 같은 규약으로 정규화한다.
//! 왜곡 계수가 0 이면(기본값, [`Intrinsics::from_hfov`]) 계산이 핀홀 식과 비트 단위로 같다.
//! [`Intrinsics::with_distortion`] 은 번들 조정용 [`DistortedIntrinsics`] 를,
//! [`Intrinsics::distorted`] 는 계수를 바꾼 [`Intrinsics`] 를 만든다.

use crate::distortion::{
    normalized_to_pixel, pixel_to_normalized, DistortedIntrinsics, Distortion, ProjectionJacobian,
};
use crate::math::{Point3, Rotation3, Vector2, Vector3};

/// 핀홀 + 왜곡 내부 파라미터(초점·주점은 픽셀 단위).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
    /// 정규 좌표 왜곡 계수. 0 이면 핀홀.
    pub dist: Distortion,
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
            dist: Distortion::default(),
        }
    }

    /// 왜곡 계수를 `dist` 로 바꾼 내부 파라미터.
    pub fn distorted(&self, dist: Distortion) -> Self {
        Self { dist, ..*self }
    }

    /// 왜곡 전 정규 좌표 (x/z, y/z) → 픽셀(왜곡 적용).
    pub fn to_pixel(&self, n: &Vector2<f64>) -> Vector2<f64> {
        normalized_to_pixel(self.fx, self.fy, self.cx, self.cy, &self.dist, n)
    }

    /// 픽셀 → 왜곡 전 정규 좌표(역왜곡 포함). 왜곡이 0 이면 (p − c)/f 와 비트 단위로 같다.
    /// 역왜곡 뉴턴 반복이 수렴하지 않는 점(영상 밖 먼 곳 등)은 마지막 반복값을 돌려준다.
    /// 수렴 여부가 필요하면 [`Self::unproject`] 를 쓴다.
    pub fn to_normalized(&self, p: &Vector2<f64>) -> Vector2<f64> {
        self.unproject_best(p).0
    }

    /// 픽셀 → 왜곡 전 정규 좌표. 역왜곡이 수렴하지 않으면 None.
    pub fn unproject(&self, p: &Vector2<f64>) -> Option<Vector2<f64>> {
        let (n, ok) = self.unproject_best(p);
        ok.then_some(n)
    }

    fn unproject_best(&self, p: &Vector2<f64>) -> (Vector2<f64>, bool) {
        pixel_to_normalized(self.fx, self.fy, self.cx, self.cy, &self.dist, p)
    }

    /// 번들 조정 매개변수 형태([fx, fy, cx, cy, k1, k2, p1, p2]). 자기 왜곡 계수를 쓴다.
    pub fn to_distorted(&self) -> DistortedIntrinsics {
        self.with_distortion(self.dist)
    }

    /// 화소 번호 규약 위치(화소 (i, j) 의 중심을 (i, j) 로 적음, 특징점 `Keypoint::x, y` 가 이 규약)
    /// → 정규 좌표. 화소 중심 규약으로 +0.5 를 더한 뒤 [`Self::to_normalized`] 와 같다.
    pub fn index_to_normalized(&self, p: &Vector2<f64>) -> Vector2<f64> {
        self.to_normalized(&Vector2::new(p.x + 0.5, p.y + 0.5))
    }

    /// 같은 초점·주점에 왜곡 계수 `dist` 를 붙인 번들 조정용 내부 파라미터(자기 계수는 무시).
    pub fn with_distortion(&self, dist: Distortion) -> DistortedIntrinsics {
        DistortedIntrinsics {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            dist,
        }
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

    /// 픽셀과 깊이(카메라 z)로 세계 점을 복원(역왜곡 포함, [`Intrinsics::to_normalized`]).
    pub fn unproject(&self, p: &Vector2<f64>, depth: f64) -> Point3<f64> {
        let n = self.intrinsics.to_normalized(p);
        let xc = Vector3::new(n.x * depth, n.y * depth, depth);
        Point3::from(self.pose.rotation.inverse() * (xc - self.pose.translation))
    }

    /// 투영과 야코비안(∂픽셀/∂세계점, ∂픽셀/∂자세, ∂픽셀/∂[fx, fy, cx, cy, k1, k2, p1, p2]).
    /// [`DistortedIntrinsics::project_with_jacobian`] 과 같다. 카메라 뒤면 None.
    pub fn project_with_jacobian(&self, x: &Point3<f64>) -> Option<ProjectionJacobian> {
        self.intrinsics
            .to_distorted()
            .project_with_jacobian(&self.pose, x)
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
            assert!((d.project_camera(&xc) - px).norm() < 1e-9);
        }
    }

    /// 확인용 렌즈: k1≈−0.1, k2≈0.01, p1·p2 ~1e-3.
    fn lens() -> Distortion {
        Distortion {
            k1: -0.1,
            k2: 0.01,
            p1: 1.2e-3,
            p2: -0.9e-3,
        }
    }

    #[test]
    fn zero_distortion_is_bitwise_pinhole() {
        let cam = test_camera();
        let k = cam.intrinsics;
        assert!(k.dist.is_zero());
        for i in 0..200 {
            let px = Vector2::new(-50.0 + 10.3 * i as f64, 1100.0 - 5.7 * i as f64);
            let n = k.to_normalized(&px);
            assert_eq!(n.x, (px.x - k.cx) / k.fx);
            assert_eq!(n.y, (px.y - k.cy) / k.fy);
            assert_eq!(k.unproject(&px), Some(n));
            let back = k.to_pixel(&n);
            assert_eq!(back.x, k.fx * n.x + k.cx);
            assert_eq!(back.y, k.fy * n.y + k.cy);
            let d = k.to_distorted();
            assert_eq!(d.unproject(&px), Some(n));
            assert_eq!(d.project_camera(&Vector3::new(n.x, n.y, 1.0)), back);
        }
    }

    #[test]
    fn distorted_project_unproject_roundtrip() {
        let mut cam = test_camera();
        cam.intrinsics = cam.intrinsics.distorted(lens());
        let k = cam.intrinsics;
        // 영상 전체 격자(모서리 포함): 픽셀 → 세계 → 픽셀.
        let mut worst_px: f64 = 0.0;
        for i in 0..=32 {
            for j in 0..=18 {
                let px = Vector2::new(60.0 * i as f64, 60.0 * j as f64);
                let depth = 5.0 + (i + j) as f64;
                let x = cam.unproject(&px, depth);
                assert!((cam.pose.transform(&x).z - depth).abs() < 1e-9);
                worst_px = worst_px.max((cam.project(&x).unwrap() - px).norm());
            }
        }
        // 세계 → 픽셀 → 정규 좌표: 광선 방향 (x/z, y/z) 복원.
        let mut worst_n: f64 = 0.0;
        for i in 0..200 {
            let n = Vector2::new(
                (i as f64 * 0.137) % 1.3 - 0.65,
                (i as f64 * 0.071) % 0.74 - 0.37,
            );
            let xc = Vector3::new(n.x, n.y, 1.0) * (3.0 + i as f64 * 0.2);
            let x = Point3::from(cam.pose.rotation.inverse() * (xc - cam.pose.translation));
            let px = cam.project(&x).unwrap();
            let back = k.unproject(&px).unwrap();
            worst_n = worst_n.max((back - n).norm());
            // 같은 관측을 번들 조정 형태로 정규화해도 같은 값.
            assert_eq!(k.to_distorted().unproject(&px), Some(back));
        }
        eprintln!("distorted roundtrip worst {worst_px:.2e} px, normalized {worst_n:.2e}");
        assert!(worst_px < 1e-9, "왕복 오차 {worst_px} px");
        assert!(worst_n * k.fx < 1e-9, "정규 좌표 왕복 오차 {worst_n}");
        // 왜곡이 실제로 걸렸는지: 모서리에서 핀홀 투영과 수 px 이상 다르다.
        let corner = Vector2::new(0.0, 0.0);
        let shift = (k.to_normalized(&corner)
            - k.distorted(Distortion::default()).to_normalized(&corner))
        .norm()
            * k.fx;
        assert!(shift > 5.0, "모서리 왜곡 {shift} px");
    }

    #[test]
    fn camera_jacobian_matches_numeric() {
        let mut cam = test_camera();
        cam.intrinsics = cam.intrinsics.distorted(lens());
        let h = 1e-6;
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
        let mut worst: f64 = 0.0;
        // 영상 모서리 가까이(왜곡이 큰 곳)를 포함한 세 점.
        for (u, v, depth) in [
            (60.0, 40.0, 20.0),
            (1850.0, 1000.0, 35.0),
            (900.0, 600.0, 12.0),
        ] {
            let x = cam.unproject(&Vector2::new(u, v), depth);
            let j = cam.project_with_jacobian(&x).unwrap();
            assert_eq!(j.pixel, cam.project(&x).unwrap());
            for a in 0..3 {
                let (mut xp, mut xm) = (x, x);
                xp[a] += h;
                xm[a] -= h;
                let num = (cam.project(&xp).unwrap() - cam.project(&xm).unwrap()) / (2.0 * h);
                for r in 0..2 {
                    worst = worst.max(rel(j.d_point[(r, a)], num[r]));
                }
            }
            for a in 0..6 {
                let f = |s: f64| {
                    let mut q = cam;
                    if a < 3 {
                        let mut w = Vector3::zeros();
                        w[a] = s;
                        q.pose.rotation = Rotation3::new(w) * q.pose.rotation;
                    } else {
                        q.pose.translation[a - 3] += s;
                    }
                    q.project(&x).unwrap()
                };
                let num = (f(h) - f(-h)) / (2.0 * h);
                for r in 0..2 {
                    worst = worst.max(rel(j.d_pose[(r, a)], num[r]));
                }
            }
            for a in 0..8 {
                let f = |s: f64| {
                    let mut q = cam;
                    let k = &mut q.intrinsics;
                    match a {
                        0 => k.fx += s,
                        1 => k.fy += s,
                        2 => k.cx += s,
                        3 => k.cy += s,
                        4 => k.dist.k1 += s,
                        5 => k.dist.k2 += s,
                        6 => k.dist.p1 += s,
                        _ => k.dist.p2 += s,
                    }
                    q.project(&x).unwrap()
                };
                let num = (f(h) - f(-h)) / (2.0 * h);
                for r in 0..2 {
                    worst = worst.max(rel(j.d_intrinsics[(r, a)], num[r]));
                }
            }
        }
        eprintln!("camera jacobian worst rel {worst:.2e}");
        assert!(worst < 1e-6, "야코비안 최대 상대 오차 {worst}");
    }

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

    fn quantile(v: &[f64], q: f64) -> f64 {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    /// 아래를 보는 두 카메라(고도 ~40 m, 기선 3 m, 960×540, 화각 70°)로 `lens_true` 렌즈를 통해
    /// 투영하고 σ px 잡음을 더한 뒤, 내부 파라미터 `k_used` 로 정규화해 8점 + 정밀화 자세를 구한다.
    /// 시드별 (회전 오차, 이동 방향 오차)(도)와 이동 관측 불가 수.
    fn distorted_pose_errors(
        lens_true: Distortion,
        lens_used: Distortion,
        sigma: f64,
        seeds: std::ops::RangeInclusive<u64>,
    ) -> (Vec<f64>, Vec<f64>, usize) {
        use crate::math::rotation_angle_between;
        use crate::two_view::{essential_8pt, refine_relative_pose};
        let base = Intrinsics::from_hfov(960, 540, 70f64.to_radians());
        let k = base.distorted(lens_true);
        let k_used = base.distorted(lens_used);
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
        let r = c2.pose.rotation * c1.pose.rotation.inverse();
        let t = c2.pose.translation - r * c1.pose.translation;
        let (mut rot, mut dir, mut unobs) = (vec![], vec![], 0);
        for seed in seeds {
            let mut rng = Lcg(seed);
            let (mut x1, mut x2) = (vec![], vec![]);
            while x1.len() < 200 {
                let x = Point3::new(
                    (rng.next() - 0.5) * 40.0,
                    (rng.next() - 0.5) * 24.0,
                    (rng.next() - 0.5) * 10.0,
                );
                if let (Some(p), Some(q)) = (c1.project(&x), c2.project(&x)) {
                    if !(k.contains(&p) && k.contains(&q)) {
                        continue;
                    }
                    let mut noise = || Vector2::new(rng.gauss(), rng.gauss()) * sigma;
                    let (pn, qn) = (p + noise(), q + noise());
                    x1.push(k_used.to_normalized(&pn));
                    x2.push(k_used.to_normalized(&qn));
                }
            }
            let e = essential_8pt(&x1, &x2).unwrap();
            let rp = refine_relative_pose(&e, &x1, &x2, 50).unwrap();
            if !rp.translation_observable {
                unobs += 1;
                continue;
            }
            rot.push(rotation_angle_between(&rp.rotation, &r).to_degrees());
            dir.push(rp.translation.angle(&t.normalize()).to_degrees());
        }
        (rot, dir, unobs)
    }

    /// F-023 확인: 왜곡 렌즈로 투영한 두 장의 대응을 단일 정규화 경로로 되돌리면
    /// 자세 오차가 왜곡 없는 경우와 같은 기준(two_view 의 σ0.5 정밀화 기준:
    /// 회전 중앙값·90%·최대 < 0.14·0.24·0.4°, 방향 < 1.05·2.4·3.0°)을 통과한다.
    /// 대조: 같은 영상을 왜곡을 무시하고 정규화하면 기준을 넘는다.
    #[test]
    fn distorted_two_view_pose_meets_pinhole_bounds() {
        let rb = [0.14, 0.24, 0.4];
        let db = [1.05, 2.4, 3.0];
        let q = |v: &[f64]| [0.5, 0.9, 1.0].map(|p| quantile(v, p));
        let passes = |rot: &[f64], dir: &[f64]| {
            let (r, d) = (q(rot), q(dir));
            (0..3).all(|i| r[i] < rb[i]) && (0..3).all(|i| d[i] < db[i])
        };
        for (what, lens_true, lens_used) in [
            ("핀홀", Distortion::default(), Distortion::default()),
            ("왜곡", lens(), lens()),
        ] {
            let (rot, dir, unobs) = distorted_pose_errors(lens_true, lens_used, 0.5, 1..=100);
            eprintln!(
                "{what}: n={} unobservable={unobs} rot {:.3?} dir {:.2?}",
                rot.len(),
                q(&rot),
                q(&dir)
            );
            assert_eq!(unobs, 0, "{what}: 관측 불가로 오판");
            assert!(
                passes(&rot, &dir),
                "{what}: 기준 미달 rot {:?} dir {:?}",
                q(&rot),
                q(&dir)
            );
        }
        // 대조: 왜곡을 무시한 정규화.
        let (rot, dir, unobs) = distorted_pose_errors(lens(), Distortion::default(), 0.5, 1..=100);
        eprintln!(
            "왜곡 무시: n={} unobservable={unobs} rot {:.3?} dir {:.2?}",
            rot.len(),
            q(&rot),
            q(&dir)
        );
        assert!(
            unobs > 0 || !passes(&rot, &dir),
            "왜곡을 무시해도 통과: 시험이 둔감"
        );
    }

    #[test]
    fn hfov_focal_known() {
        // 화각 90° 이면 f = w/2.
        let k = Intrinsics::from_hfov(1000, 800, std::f64::consts::FRAC_PI_2);
        assert!((k.fx - 500.0).abs() < 1e-9);
    }
}
