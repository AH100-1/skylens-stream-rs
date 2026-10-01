//! 닮음 변환(배율·회전·평행이동) 추정과 GPS 동-북-위 정렬.
//!
//! - [`umeyama`]: 대응점 최소제곱 닮음 변환 (Umeyama 1991, 반사 보정 포함).
//! - [`robust_similarity`]: 반복 트리밍. 임계 = max(3 × 잔차 중앙값, 바닥값).
//! - [`gps_align`]: 카메라 중심을 GPS(동-북-위)에 1회 정렬, 잔차 3 m 초과 대응 제외.

use crate::geo::{geodetic_to_enu, Geodetic};
use crate::math::{Matrix3, Vector3};
use nalgebra::Rotation3;

/// 닮음 변환 `x ↦ s·R·x + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub s: f64,
    pub r: Rotation3<f64>,
    pub t: Vector3<f64>,
}

impl Similarity {
    pub fn identity() -> Self {
        Self {
            s: 1.0,
            r: Rotation3::identity(),
            t: Vector3::zeros(),
        }
    }

    /// 점에 적용: `s·R·p + t`.
    pub fn apply_point(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.s * (self.r * p) + self.t
    }

    /// 법선(방향)에 적용: 회전만.
    pub fn apply_normal(&self, n: &Vector3<f64>) -> Vector3<f64> {
        self.r * n
    }

    /// 역변환 `y ↦ (1/s)·Rᵀ·(y − t)`.
    pub fn inverse(&self) -> Self {
        let ri = self.r.inverse();
        let si = 1.0 / self.s;
        Self {
            s: si,
            r: ri,
            t: -(si * (ri * self.t)),
        }
    }

    /// 합성: `self ∘ other` (먼저 `other`, 다음 `self`).
    pub fn compose(&self, other: &Similarity) -> Self {
        Self {
            s: self.s * other.s,
            r: self.r * other.r,
            t: self.s * (self.r * other.t) + self.t,
        }
    }

    /// 4×4 동차 행렬.
    pub fn to_matrix4(&self) -> nalgebra::Matrix4<f64> {
        let mut m = nalgebra::Matrix4::identity();
        let sr = self.r.matrix() * self.s;
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&sr);
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.t);
        m
    }
}

fn finite(v: &Vector3<f64>) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Umeyama(1991) 최소제곱 닮음 변환: `dst ≈ s·R·src + t`.
///
/// 길이 불일치, 3점 미만, NaN/무한, 원본 분산 0, 공분산 계수 < 2(일직선 등 퇴화)이면 `None`.
/// det(Σ) < 0 이면 최소 특이값 부호를 뒤집어 반사 대신 고유 회전을 돌려준다.
pub fn umeyama(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Option<Similarity> {
    let n = src.len();
    if n != dst.len() || n < 3 {
        return None;
    }
    if !src.iter().chain(dst.iter()).all(finite) {
        return None;
    }
    let nf = n as f64;
    let mu_s = src.iter().sum::<Vector3<f64>>() / nf;
    let mu_d = dst.iter().sum::<Vector3<f64>>() / nf;
    let mut var_s = 0.0;
    let mut cov = Matrix3::zeros();
    for (a, b) in src.iter().zip(dst) {
        let da = a - mu_s;
        let db = b - mu_d;
        var_s += da.norm_squared();
        cov += db * da.transpose();
    }
    var_s /= nf;
    cov /= nf;
    let scale_ref = var_s.max(f64::MIN_POSITIVE);
    if var_s <= 1e-24 {
        return None;
    }
    let svd = cov.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let d = svd.singular_values;
    // 특이값 정렬(내림차순) 색인.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| d[j].total_cmp(&d[i]));
    let dmax = d[idx[0]];
    // 계수 < 2 이면 회전이 정해지지 않는다(일직선·한 점 등).
    if dmax <= 0.0 || d[idx[1]] <= 1e-9 * dmax {
        return None;
    }
    let mut sgn = Vector3::new(1.0, 1.0, 1.0);
    if (u.determinant() * vt.determinant()) < 0.0 {
        sgn[idx[2]] = -1.0;
    }
    let r = u * Matrix3::from_diagonal(&sgn) * vt;
    let trace: f64 = (0..3).map(|i| d[i] * sgn[i]).sum();
    let s = trace / scale_ref;
    if !(s.is_finite() && s > 0.0) {
        return None;
    }
    let r = Rotation3::from_matrix_unchecked(r);
    let t = mu_d - s * (r * mu_s);
    let out = Similarity { s, r, t };
    if !finite(&out.t) || !out.r.matrix().iter().all(|x| x.is_finite()) {
        return None;
    }
    Some(out)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn residuals(sim: &Similarity, src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Vec<f64> {
    src.iter()
        .zip(dst)
        .map(|(a, b)| (sim.apply_point(a) - b).norm())
        .collect()
}

/// 반복 트리밍 닮음 변환.
///
/// 전체 대응으로 시작해 `iters` 번: 잔차 중앙값 m 을 구하고 임계 `max(3m, floor_m)` 이하만
/// 정상으로 두어 다시 추정한다. 정상 대응이 3개 미만이 되면 직전 추정에서 멈춘다.
/// 반환: (변환, 정상 표시, 정상 대응 잔차 중앙값).
pub fn robust_similarity(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    iters: usize,
    floor_m: f64,
) -> Option<(Similarity, Vec<bool>, f64)> {
    let mut sim = umeyama(src, dst)?;
    let n = src.len();
    let mut inl = vec![true; n];
    for _ in 0..iters {
        let res = residuals(&sim, src, dst);
        let mut cur: Vec<f64> = res
            .iter()
            .zip(&inl)
            .filter(|(_, &k)| k)
            .map(|(r, _)| *r)
            .collect();
        let med = median(&mut cur);
        let thr = (3.0 * med).max(floor_m);
        let next: Vec<bool> = res.iter().map(|&r| r <= thr).collect();
        let (s2, d2): (Vec<_>, Vec<_>) = src
            .iter()
            .zip(dst)
            .zip(&next)
            .filter(|(_, &k)| k)
            .map(|((a, b), _)| (*a, *b))
            .unzip();
        let Some(new_sim) = umeyama(&s2, &d2) else {
            break;
        };
        let same = next == inl;
        sim = new_sim;
        inl = next;
        if same {
            break;
        }
    }
    // 최종 변환 기준으로 정상 표시를 다시 매긴다(같은 임계 규칙).
    let res = residuals(&sim, src, dst);
    let mut cur: Vec<f64> = res
        .iter()
        .zip(&inl)
        .filter(|(_, &k)| k)
        .map(|(r, _)| *r)
        .collect();
    let thr = (3.0 * median(&mut cur)).max(floor_m);
    let inl: Vec<bool> = res.iter().map(|&r| r <= thr).collect();
    let mut cur: Vec<f64> = res
        .iter()
        .zip(&inl)
        .filter(|(_, &k)| k)
        .map(|(r, _)| *r)
        .collect();
    let med = median(&mut cur);
    Some((sim, inl, med))
}

/// GPS 정렬 결과.
#[derive(Clone, Debug)]
pub struct GpsAlignment {
    /// 복원 좌표 → 동-북-위 변환.
    pub sim: Similarity,
    /// 대응별 정상 표시(잔차 ≤ 최대 잔차).
    pub inliers: Vec<bool>,
    /// 대응별 잔차(m, 최종 변환 기준).
    pub residuals: Vec<f64>,
    /// 정상 대응 잔차 중앙값(m).
    pub median_residual: f64,
}

/// SPEC §3.4 기본 잔차 상한(m).
pub const GPS_MAX_RESIDUAL_M: f64 = 3.0;

/// 카메라 중심(복원 좌표)을 동-북-위 GPS 위치에 1회 정렬한다.
///
/// 전체 대응으로 Umeyama 추정 → 잔차 `max_residual_m` 초과 대응 제외 → 남은 대응으로 한 번
/// 다시 추정. 남은 대응이 3개 미만이면 첫 추정을 쓴다.
pub fn align_to_enu(
    centers: &[Vector3<f64>],
    enu: &[Vector3<f64>],
    max_residual_m: f64,
) -> Option<GpsAlignment> {
    let first = umeyama(centers, enu)?;
    let res0 = residuals(&first, centers, enu);
    let keep: Vec<bool> = res0.iter().map(|&r| r <= max_residual_m).collect();
    let (s2, d2): (Vec<_>, Vec<_>) = centers
        .iter()
        .zip(enu)
        .zip(&keep)
        .filter(|(_, &k)| k)
        .map(|((a, b), _)| (*a, *b))
        .unzip();
    let sim = umeyama(&s2, &d2).unwrap_or(first);
    let res = residuals(&sim, centers, enu);
    let inliers: Vec<bool> = res.iter().map(|&r| r <= max_residual_m).collect();
    let mut cur: Vec<f64> = res
        .iter()
        .zip(&inliers)
        .filter(|(_, &k)| k)
        .map(|(r, _)| *r)
        .collect();
    let median_residual = median(&mut cur);
    Some(GpsAlignment {
        sim,
        inliers,
        residuals: res,
        median_residual,
    })
}

/// 위경도 GPS 를 `origin` 기준 동-북-위로 바꾼 뒤 [`align_to_enu`] (상한 3 m).
pub fn gps_align(
    centers: &[Vector3<f64>],
    gps: &[Geodetic],
    origin: &Geodetic,
) -> Option<GpsAlignment> {
    if centers.len() != gps.len() {
        return None;
    }
    let enu: Vec<Vector3<f64>> = gps.iter().map(|g| geodetic_to_enu(g, origin)).collect();
    align_to_enu(centers, &enu, GPS_MAX_RESIDUAL_M)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::enu_to_geodetic;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uni(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.uni().max(1e-300);
            let u2 = self.uni();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
        fn gvec(&mut self, sigma: f64) -> Vector3<f64> {
            Vector3::new(self.gauss(), self.gauss(), self.gauss()) * sigma
        }
        fn uvec(&mut self, h: f64) -> Vector3<f64> {
            Vector3::new(
                (2.0 * self.uni() - 1.0) * h,
                (2.0 * self.uni() - 1.0) * h,
                (2.0 * self.uni() - 1.0) * h,
            )
        }
    }

    fn random_sim(rng: &mut Rng) -> Similarity {
        let axis = rng.gvec(1.0);
        let ang = rng.uni() * std::f64::consts::PI;
        Similarity {
            s: 0.2 + 9.8 * rng.uni(),
            r: Rotation3::from_scaled_axis(axis.normalize() * ang),
            t: rng.uvec(500.0),
        }
    }

    fn rot_err_deg(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
        let m = (a.inverse() * b).into_inner();
        ((m.trace() - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    /// 정답과 비교: (배율 상대오차, 회전 오차 도, 평행이동 오차 m(정답 좌표 원점 기준)).
    fn errs(est: &Similarity, gt: &Similarity) -> (f64, f64, f64) {
        (
            (est.s / gt.s - 1.0).abs(),
            rot_err_deg(&est.r, &gt.r),
            (est.t - gt.t).norm(),
        )
    }

    #[test]
    fn exact_recovery_noise_free() {
        let mut rng = Rng(1);
        for _ in 0..50 {
            let gt = random_sim(&mut rng);
            let src: Vec<_> = (0..10).map(|_| rng.uvec(50.0)).collect();
            let dst: Vec<_> = src.iter().map(|p| gt.apply_point(p)).collect();
            let est = umeyama(&src, &dst).unwrap();
            let (es, er, et) = errs(&est, &gt);
            assert!(es < 1e-10 && er < 1e-5 && et < 1e-7, "{es} {er} {et}");
            assert!((est.r.matrix().determinant() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn helpers_inverse_compose_normal() {
        let mut rng = Rng(2);
        let a = random_sim(&mut rng);
        let b = random_sim(&mut rng);
        let p = rng.uvec(10.0);
        let q = a.inverse().apply_point(&a.apply_point(&p));
        assert!((q - p).norm() < 1e-9);
        let c = a.compose(&b);
        assert!((c.apply_point(&p) - a.apply_point(&b.apply_point(&p))).norm() < 1e-8);
        let n = Vector3::new(0.0, 0.0, 1.0);
        assert!((a.apply_normal(&n).norm() - 1.0).abs() < 1e-12);
        let m = a.to_matrix4();
        let ph = m * nalgebra::Vector4::new(p.x, p.y, p.z, 1.0);
        assert!((ph.xyz() - a.apply_point(&p)).norm() < 1e-9);
    }

    /// 거울상 대응: 고유 회전만 돌려주고(det = +1), 반사 행렬을 내지 않는다.
    #[test]
    fn reflection_yields_proper_rotation() {
        let mut rng = Rng(3);
        for _ in 0..20 {
            let src: Vec<_> = (0..20).map(|_| rng.uvec(10.0)).collect();
            let dst: Vec<_> = src.iter().map(|p| Vector3::new(-p.x, p.y, p.z)).collect();
            let est = umeyama(&src, &dst).unwrap();
            assert!((est.r.matrix().determinant() - 1.0).abs() < 1e-9);
            // 반사로는 잔차 0 이 될 수 없다.
            let rms = (residuals(&est, &src, &dst)
                .iter()
                .map(|r| r * r)
                .sum::<f64>()
                / 20.0)
                .sqrt();
            assert!(rms > 0.5, "{rms}");
        }
        // 평면 점군의 반사: 평면 안에서 x 뒤집기는 법선 축 180° 회전과 구분 가능해야 하고,
        // 평면 밖 축(z) 뒤집기는 평면에선 회전으로 정확히 맞는다.
        let src: Vec<_> = (0..12)
            .map(|_| {
                let v = rng.uvec(10.0);
                Vector3::new(v.x, v.y, 0.0)
            })
            .collect();
        let dst: Vec<_> = src.iter().map(|p| Vector3::new(p.x, p.y, -p.z)).collect();
        let est = umeyama(&src, &dst).unwrap();
        assert!((est.r.matrix().determinant() - 1.0).abs() < 1e-9);
        assert!(residuals(&est, &src, &dst).iter().all(|&r| r < 1e-9));
    }

    #[test]
    fn degenerate_inputs_are_none() {
        let p = |x: f64, y: f64, z: f64| Vector3::new(x, y, z);
        let a = vec![p(0., 0., 0.), p(1., 0., 0.), p(0., 1., 0.)];
        // 길이 불일치
        assert!(umeyama(&a, &a[..2]).is_none());
        // 3점 미만
        assert!(umeyama(&a[..2], &a[..2]).is_none());
        // 빈 입력
        assert!(umeyama(&[], &[]).is_none());
        // 모두 같은 점
        let same = vec![p(1., 2., 3.); 5];
        assert!(umeyama(&same, &same).is_none());
        // 일직선
        let line: Vec<_> = (0..6).map(|i| p(i as f64, 2.0 * i as f64, 0.5)).collect();
        assert!(umeyama(&line, &line).is_none());
        // 목표가 한 점으로 무너짐
        assert!(umeyama(&a, &[p(1., 1., 1.); 3]).is_none());
        // NaN / 무한
        let mut b = a.clone();
        b[1].y = f64::NAN;
        assert!(umeyama(&a, &b).is_none());
        b[1].y = f64::INFINITY;
        assert!(umeyama(&b, &a).is_none());
        assert!(robust_similarity(&line, &line, 5, 0.3).is_none());
        assert!(align_to_enu(&line, &line, 3.0).is_none());
    }

    /// 잡음 σ(축당) 1~2 m, 이상치 20~30% (수십~수백 m 오프셋), 여러 시드.
    ///
    /// 기준값 근거: 정상 N≈140, 범위 ±200 m 일 때 회전 오차 표준편차 ≈ σ/(s·범위·√N) 라디안
    /// 수준 → 2 m 에서도 0.1° 이하가 기대치. 여유 두 배 남짓으로 0.25°, 배율 0.5%,
    /// 평행이동 1.5 m(중심 추정 σ/√N ≈ 0.17 m 에 원점까지 회전 오차 지렛대 포함).
    #[test]
    fn robust_recovers_with_noise_and_outliers() {
        let mut worst = (0.0f64, 0.0f64, 0.0f64);
        for seed in 0..20u64 {
            let mut rng = Rng(100 + seed);
            let gt = random_sim(&mut rng);
            let n = 200;
            let sigma = 1.0 + rng.uni();
            let frac = 0.2 + 0.1 * rng.uni();
            let mut src = Vec::new();
            let mut dst = Vec::new();
            let mut truth = Vec::new();
            for i in 0..n {
                let p = rng.uvec(200.0) / gt.s;
                let mut q = gt.apply_point(&p) + rng.gvec(sigma);
                let out = (i as f64) < frac * n as f64;
                if out {
                    let dir = rng.gvec(1.0).normalize();
                    q += dir * (30.0 + 300.0 * rng.uni());
                }
                src.push(p);
                dst.push(q);
                truth.push(!out);
            }
            let (est, inl, med) = robust_similarity(&src, &dst, 5, 0.3).unwrap();
            let (es, er, et) = errs(&est, &gt);
            worst = (worst.0.max(es), worst.1.max(er), worst.2.max(et));
            assert!(es < 5e-3, "seed {seed}: scale {es}");
            assert!(er < 0.25, "seed {seed}: rot {er} deg");
            assert!(et < 1.5, "seed {seed}: trans {et} m");
            // 이상치는 전부 걸러진다(최소 오프셋 30 m ≫ 3·중앙 잔차).
            for (k, (&a, &b)) in inl.iter().zip(&truth).enumerate() {
                if !b {
                    assert!(!a, "seed {seed}: outlier {k} kept");
                }
            }
            let kept = inl.iter().zip(&truth).filter(|(&a, &b)| a && b).count();
            let good = truth.iter().filter(|&&b| b).count();
            assert!(
                kept as f64 >= 0.95 * good as f64,
                "seed {seed}: {kept}/{good}"
            );
            // 3D 가우스 잔차 크기 중앙 ≈ 1.54σ.
            assert!(
                med > 1.2 * sigma && med < 1.9 * sigma,
                "seed {seed}: med {med}"
            );
        }
        eprintln!(
            "worst scale {:.2e} rot {:.4} deg trans {:.3} m",
            worst.0, worst.1, worst.2
        );
    }

    /// 이상치가 있으면 단순 Umeyama 는 크게 틀리고, 트리밍은 바로잡는다(음성 대조).
    #[test]
    fn plain_umeyama_fails_with_outliers() {
        let mut rng = Rng(7);
        let gt = random_sim(&mut rng);
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..100 {
            let p = rng.uvec(100.0);
            let mut q = gt.apply_point(&p) + rng.gvec(1.0);
            if i < 25 {
                q += Vector3::new(0.0, 0.0, 200.0 * gt.s);
            }
            src.push(p);
            dst.push(q);
        }
        let plain = umeyama(&src, &dst).unwrap();
        let (_, _, et) = errs(&plain, &gt);
        assert!(et > 20.0, "{et}");
        let (rob, _, _) = robust_similarity(&src, &dst, 5, 0.3).unwrap();
        let (_, _, et) = errs(&rob, &gt);
        assert!(et < 1.0, "{et}");
    }

    /// 바닥값: 잡음 0 이면 중앙 잔차 0 이라도 임계는 floor_m 이라 정상 대응을 잃지 않는다.
    #[test]
    fn floor_keeps_inliers_when_noise_free() {
        let mut rng = Rng(9);
        let gt = random_sim(&mut rng);
        let src: Vec<_> = (0..30).map(|_| rng.uvec(20.0)).collect();
        let mut dst: Vec<_> = src.iter().map(|p| gt.apply_point(p)).collect();
        dst[0].x += 0.2; // 바닥 0.3 m 이내
        dst[1].x += 50.0;
        let (_, inl, _) = robust_similarity(&src, &dst, 5, 0.3).unwrap();
        assert!(inl[0]);
        assert!(!inl[1]);
        assert_eq!(inl.iter().filter(|&&b| b).count(), 29);
    }

    /// 카메라 경로(격자 비행, 고도 100 m)를 임의 닮음 변환으로 흐트러뜨린 복원 좌표와
    /// σ 1.5 m GPS(위경도), 20~30% 는 10~50 m 튄 GPS. 3 m 넘는 대응은 제외되어야 한다.
    /// 비행 경로가 거의 평면(고도 흔들림 1 m)이라 기울기 방향 회전은 축당 σ/(반경·√N)
    /// ≈ 0.87/(80·√90) ≈ 0.07° 표준편차 → 10 시드 최악 여유 포함 0.5°, 영역 내 최대 위치
    /// 오차 2 m(0.5° × 반경 170 m ≈ 1.5 m + 잡음 평균).
    #[test]
    fn gps_alignment_recovers_enu_frame() {
        let origin = Geodetic {
            lat_deg: 37.5,
            lon_deg: 127.0,
            alt: 50.0,
        };
        for seed in 0..10u64 {
            let mut rng = Rng(500 + seed);
            let mut enu_true = Vec::new();
            for i in 0..12 {
                for j in 0..10 {
                    enu_true.push(Vector3::new(
                        i as f64 * 25.0,
                        j as f64 * 20.0,
                        100.0 + rng.gauss(),
                    ));
                }
            }
            // gt: 복원 → ENU
            let gt = random_sim(&mut rng);
            let gi = gt.inverse();
            let centers: Vec<_> = enu_true.iter().map(|e| gi.apply_point(e)).collect();
            let n = enu_true.len();
            let frac = 0.2 + 0.1 * rng.uni();
            let mut bad = vec![false; n];
            let gps: Vec<Geodetic> = enu_true
                .iter()
                .enumerate()
                .map(|(k, e)| {
                    let mut m = e + rng.gvec(1.5 / 3f64.sqrt() * 1.0);
                    if (k as f64) < frac * n as f64 {
                        let d = rng.gvec(1.0).normalize();
                        m += d * (10.0 + 40.0 * rng.uni());
                        bad[k] = true;
                    }
                    enu_to_geodetic(&m, &origin)
                })
                .collect();
            let al = gps_align(&centers, &gps, &origin).unwrap();
            let (es, er, _) = errs(&al.sim, &gt);
            // 평행이동은 복원 원점이 아니라 비행 영역에서 본다: 참 위치 대비 최대 오차.
            let ep = centers
                .iter()
                .zip(&enu_true)
                .map(|(c, e)| (al.sim.apply_point(c) - e).norm())
                .fold(0.0, f64::max);
            eprintln!("gps seed {seed}: scale {es:.2e} rot {er:.4} deg max pos {ep:.3} m");
            assert!(es < 5e-3, "seed {seed}: scale {es}");
            assert!(er < 0.5, "seed {seed}: rot {er}");
            assert!(ep < 2.0, "seed {seed}: pos {ep}");
            for (&inl, &r) in al.inliers.iter().zip(&al.residuals) {
                assert_eq!(inl, r <= 3.0);
            }
            let bad_kept = (0..n).filter(|&k| bad[k] && al.inliers[k]).count();
            assert!(bad_kept <= 2, "seed {seed}: {bad_kept} bad kept");
            assert!(
                al.median_residual < 2.0,
                "seed {seed}: {}",
                al.median_residual
            );
        }
        // 길이 불일치
        assert!(gps_align(&[Vector3::zeros()], &[], &origin).is_none());
    }
}
