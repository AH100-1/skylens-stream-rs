//! 닮음 변환(배율·회전·평행이동) 추정과 GPS 동-북-위 정렬.
//!
//! - [`umeyama`]: 대응점 최소제곱 닮음 변환 (Umeyama 1991, 반사 보정 포함).
//! - [`robust_similarity`]: 최소 표본 무작위 합의(LMedS) 첫 추정 + 반복 트리밍.
//!   임계 = max(3 × 잔차 중앙값, 바닥값).
//! - [`gps_align`]: 카메라 중심을 GPS(동-북-위)에 1회 정렬, 잔차 3 m(또는 잡음 비례) 초과 대응
//!   제외. 연직축 고정([`similarity_fixed_up`])·경로 폭 판정 포함.
//!
//! 호출 규약: 제품 경로(`sparse.rs` 의 GPS 정렬 두 곳)는 정밀 포즈 회전에서
//! [`up_from_rotations`] 로 위 방향을 구해 `GpsAlignConfig::up` 에 넣고 [`align_to_enu_with`]
//! 를 부른다(`gps_align_poses` 와 같은 구성이며 그 함수 자체는 파이프라인에서 쓰이지 않는다).
//! `GpsAlignment::spread_m` 은 정렬된 정상 카메라 중심(복원 중심을 닮음 변환한 점)의 주축
//! 표준편차라 GPS 잡음이 들어가지 않는다. SPEC §1 실측 편대(약 10 m 삼각형)의 둘째 주축
//! 표준편차는 실제 배치(`SceneConfig::default()`)에서 약 4.3 m, 시험 띠(AB 가 진행 방향과
//! 나란한 배치)에서 약 4.4 m 로 경로 폭 문턱(5 m)보다 0.6~0.7 m 좁다. 이 배치에서는 위 방향
//! 없는 [`gps_align`]·[`align_to_enu`] 가 `None` 이다. GPS 점 자체의 표준편차는 잡음이 더해져
//! 이보다 커서(σ 1.5 m·오프셋 +10% 에서 약 5.0 m) 문턱에 닿지만 판정은 정렬된 카메라 중심의
//! 값(같은 조건에서 약 4.7 m)을 보므로, 그래도 판정에 기대지 말고 편대는 위 방향을 주는
//! 경로로 정렬한다.

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

/// 결정적 난수(최소 표본 추출용). 같은 입력이면 같은 결과.
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// 최소 표본(3점) 가설 수. 정상 비율 55% 에서 3점이 모두 정상일 확률 0.166 →
/// 256회 모두 실패할 확률 ≈ 1e-20.
const MIN_SAMPLE_TRIALS: usize = 256;
/// 가설 평가에 쓰는 최대 대응 수(균등 간격 추출).
const MIN_SAMPLE_EVAL: usize = 1024;

/// 최소 제곱 중앙값(LMedS, Rousseeuw 1984) 첫 추정: 3점 무작위 표본의 Umeyama 해 중
/// 잔차 제곱 중앙값이 가장 작은 것. 문턱이 필요 없고 이상치 50% 미만까지 버틴다.
fn lmeds_similarity(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Option<Similarity> {
    let n = src.len();
    if n <= 3 {
        return umeyama(src, dst);
    }
    let step = n.div_ceil(MIN_SAMPLE_EVAL);
    let eval: Vec<usize> = (0..n).step_by(step).collect();
    let mut rng = SplitMix(0x5EED ^ n as u64);
    let mut best: Option<(f64, Similarity)> = None;
    let mut buf = Vec::with_capacity(eval.len());
    for _ in 0..MIN_SAMPLE_TRIALS {
        let i = rng.below(n);
        let j = rng.below(n);
        let k = rng.below(n);
        if i == j || j == k || i == k {
            continue;
        }
        let Some(h) = umeyama(&[src[i], src[j], src[k]], &[dst[i], dst[j], dst[k]]) else {
            continue;
        };
        buf.clear();
        buf.extend(
            eval.iter()
                .map(|&e| (h.apply_point(&src[e]) - dst[e]).norm_squared()),
        );
        let m = median(&mut buf);
        if m.is_finite() && best.as_ref().is_none_or(|(bm, _)| m < *bm) {
            best = Some((m, h));
        }
    }
    match best {
        Some((_, h)) => Some(h),
        None => umeyama(src, dst),
    }
}

/// 반복 트리밍 닮음 변환.
///
/// 첫 추정은 최소 표본 무작위 합의(LMedS, [`lmeds_similarity`])라 한쪽으로 몰린 이상치나
/// 수백 m 이상치에 끌리지 않는다. 그 뒤 `iters` 번: 정상 대응 잔차 중앙값 m 으로 임계
/// `max(3m, floor_m)` 이하만 정상으로 두어 다시 추정한다.
/// 유한하지 않은 대응은 처음부터 제외(정상 표시 false). 최종 정상 대응이 3개 미만이면 `None`.
/// 반환: (변환, 정상 표시, 정상 대응 잔차 중앙값).
pub fn robust_similarity(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    iters: usize,
    floor_m: f64,
) -> Option<(Similarity, Vec<bool>, f64)> {
    if src.len() != dst.len() {
        return None;
    }
    let ok: Vec<bool> = src
        .iter()
        .zip(dst)
        .map(|(a, b)| finite(a) && finite(b))
        .collect();
    let (fs, fd): (Vec<_>, Vec<_>) = src
        .iter()
        .zip(dst)
        .zip(&ok)
        .filter(|(_, &k)| k)
        .map(|((a, b), _)| (*a, *b))
        .unzip();
    let mut sim = lmeds_similarity(&fs, &fd)?;
    let mut inl = ok.clone();
    for _ in 0..iters {
        let res = residuals(&sim, src, dst);
        let thr = (3.0 * masked_median(&res, &inl)).max(floor_m);
        let next: Vec<bool> = res.iter().zip(&ok).map(|(&r, &k)| k && r <= thr).collect();
        let Some(new_sim) = fit_masked(src, dst, &next, None) else {
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
    let thr = (3.0 * masked_median(&res, &inl)).max(floor_m);
    let inl: Vec<bool> = res.iter().zip(&ok).map(|(&r, &k)| k && r <= thr).collect();
    if inl.iter().filter(|&&b| b).count() < 3 {
        return None;
    }
    let med = masked_median(&res, &inl);
    Some((sim, inl, med))
}

/// 공유 이미지 양방향 검사 닮음 변환 결과.
#[derive(Clone, Debug)]
pub struct ImageSim3 {
    pub sim: Similarity,
    /// 대응마다 정상 여부(정상 이미지에서 양방향 문턱 안인 대응).
    pub inlier: Vec<bool>,
    /// 정상 이미지 수 / 전체 이미지 수.
    pub good_images: usize,
    pub total_images: usize,
    /// 정상 대응의 정방향 잔차 중앙값.
    pub median: f64,
}

/// 같은 이미지·특징 관측으로 이어진 3D 점 대응 `src[i] ↔ dst[i]` 의 닮음 변환을 공유 이미지 단위
/// 합의로 구한다. `images[i]` 는 대응 i 를 본 이미지 번호들.
///
/// 가설은 3점 Umeyama 해. 대응의 정방향 잔차 `|S(x)-y|` 가 `tol_m` 이하이고 역방향 잔차
/// `|S⁻¹(y)-x|` 가 `tol_m / s` 이하일 때 양방향 통과. 이미지는 그 이미지가 본 대응의
/// `image_frac`(0.3) 이상이 통과하면 정상 이미지. 정상 이미지가 전체의 `min_good_ratio`(0.2)
/// 이상인 가설 중 정상 이미지 수(동률이면 통과 대응 수)가 가장 큰 것을 고른 뒤,
/// 정상 이미지의 통과 대응으로 Umeyama 를 다시 풀고 문턱 `max(3·중앙값, tol_m)` 로 재가중 3회.
pub fn image_consensus_similarity(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    images: &[Vec<u32>],
    tol_m: f64,
    image_frac: f64,
    min_good_ratio: f64,
) -> Option<ImageSim3> {
    let n = src.len();
    if n < 3 || dst.len() != n || images.len() != n {
        return None;
    }
    let mut ids: Vec<u32> = images.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let total = ids.len();
    if total == 0 {
        return None;
    }
    let slot = |im: u32| ids.binary_search(&im).unwrap();
    let mut seen = vec![0usize; total];
    for o in images {
        for &im in o {
            seen[slot(im)] += 1;
        }
    }
    let passes = |sim: &Similarity, tol: f64| -> Vec<bool> {
        let inv = sim.inverse();
        (0..n)
            .map(|i| {
                (sim.apply_point(&src[i]) - dst[i]).norm() <= tol
                    && (inv.apply_point(&dst[i]) - src[i]).norm() <= tol / sim.s
            })
            .collect()
    };
    // 정상 이미지 표시와 정상 이미지 안의 통과 대응 표시.
    let classify = |pass: &[bool]| -> (Vec<bool>, usize, usize) {
        let mut ok = vec![0usize; total];
        for i in 0..n {
            if pass[i] {
                for &im in &images[i] {
                    ok[slot(im)] += 1;
                }
            }
        }
        let good: Vec<bool> = (0..total)
            .map(|a| seen[a] > 0 && ok[a] as f64 >= image_frac * seen[a] as f64)
            .collect();
        let ng = good.iter().filter(|&&g| g).count();
        let np = (0..n)
            .filter(|&i| pass[i] && images[i].iter().any(|&im| good[slot(im)]))
            .count();
        (good, ng, np)
    };
    let mut rng = SplitMix(0xA11C_E5ED ^ n as u64);
    let mut best: Option<((usize, usize), Similarity)> = None;
    for _ in 0..MIN_SAMPLE_TRIALS * 2 {
        let (i, j, k) = (rng.below(n), rng.below(n), rng.below(n));
        if i == j || j == k || i == k {
            continue;
        }
        let Some(h) = umeyama(&[src[i], src[j], src[k]], &[dst[i], dst[j], dst[k]]) else {
            continue;
        };
        if !(h.s.is_finite() && h.s > 0.0) {
            continue;
        }
        let (_, ng, np) = classify(&passes(&h, tol_m));
        if (ng as f64) < min_good_ratio * total as f64 {
            continue;
        }
        if best.as_ref().is_none_or(|(b, _)| (ng, np) > *b) {
            best = Some(((ng, np), h));
        }
    }
    let mut sim = best?.1;
    let mut tol = tol_m;
    for _ in 0..3 {
        let pass = passes(&sim, tol);
        let (good, _, _) = classify(&pass);
        let use_: Vec<bool> = (0..n)
            .map(|i| pass[i] && images[i].iter().any(|&im| good[slot(im)]))
            .collect();
        let new = fit_masked(src, dst, &use_, None)?;
        let res = residuals(&new, src, dst);
        tol = (3.0 * masked_median(&res, &use_)).max(tol_m);
        sim = new;
    }
    let pass = passes(&sim, tol);
    let (good, ng, _) = classify(&pass);
    let inlier: Vec<bool> = (0..n)
        .map(|i| pass[i] && images[i].iter().any(|&im| good[slot(im)]))
        .collect();
    if inlier.iter().filter(|&&b| b).count() < 3 || (ng as f64) < min_good_ratio * total as f64 {
        return None;
    }
    let res = residuals(&sim, src, dst);
    let median = masked_median(&res, &inlier);
    Some(ImageSim3 {
        sim,
        inlier,
        good_images: ng,
        total_images: total,
        median,
    })
}

fn masked_median(res: &[f64], mask: &[bool]) -> f64 {
    let mut cur: Vec<f64> = res
        .iter()
        .zip(mask)
        .filter(|(_, &k)| k)
        .map(|(r, _)| *r)
        .collect();
    median(&mut cur)
}

/// 표시된 대응만으로 추정. `up` 이 있으면 연직축 고정 추정([`similarity_fixed_up`]).
fn fit_masked(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    mask: &[bool],
    up: Option<&Vector3<f64>>,
) -> Option<Similarity> {
    let (s2, d2): (Vec<_>, Vec<_>) = src
        .iter()
        .zip(dst)
        .zip(mask)
        .filter(|(_, &k)| k)
        .map(|((a, b), _)| (*a, *b))
        .unzip();
    match up {
        Some(u) => similarity_fixed_up(&s2, &d2, u),
        None => umeyama(&s2, &d2),
    }
}

/// 연직축을 고정한 닮음 변환: 복원 좌표의 위 방향 `up_src` 를 목표의 +z(위)로 보내고
/// z 축 둘레 회전(방위)·배율·평행이동만 최소제곱으로 푼다.
///
/// 편대 띠처럼 폭이 좁은 경로에서 긴 축 둘레 기울기가 GPS 잡음에 흔들리는 것을 막는다.
/// 위 방향은 중력에 해당하는 독립 정보(카메라 평균 아래 방향의 반대, 지면 평면 법선 등)여야 한다.
/// 회전 R = R_z(θ)·R₀ (R₀: up_src → e_z). 중심화한 a = R₀(x − μ_x), b = y − μ_y 에 대해
/// θ = arg Σ[(a_x b_x + a_y b_y) + i(a_x b_y − a_y b_x)],
/// s = (|Σ…| + Σ a_z b_z) / Σ|a|².
pub fn similarity_fixed_up(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    up_src: &Vector3<f64>,
) -> Option<Similarity> {
    let n = src.len();
    if n != dst.len() || n < 2 || !finite(up_src) || up_src.norm() <= 0.0 {
        return None;
    }
    if !src.iter().chain(dst.iter()).all(finite) {
        return None;
    }
    let ez = Vector3::new(0.0, 0.0, 1.0);
    let r0 = Rotation3::rotation_between(up_src, &ez).unwrap_or_else(|| {
        // 정반대: 임의의 수평축 둘레 180°.
        Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI)
    });
    let nf = n as f64;
    let mu_s = src.iter().sum::<Vector3<f64>>() / nf;
    let mu_d = dst.iter().sum::<Vector3<f64>>() / nf;
    let (mut re, mut im, mut zz, mut var) = (0.0, 0.0, 0.0, 0.0);
    for (x, y) in src.iter().zip(dst) {
        let a = r0 * (x - mu_s);
        let b = y - mu_d;
        re += a.x * b.x + a.y * b.y;
        im += a.x * b.y - a.y * b.x;
        zz += a.z * b.z;
        var += a.norm_squared();
    }
    let c = re.hypot(im);
    if var <= 1e-24 || c <= 1e-12 * var {
        return None;
    }
    let r = Rotation3::from_axis_angle(&Vector3::z_axis(), im.atan2(re)) * r0;
    let s = (c + zz) / var;
    if !(s.is_finite() && s > 0.0) {
        return None;
    }
    let t = mu_d - s * (r * mu_s);
    Some(Similarity { s, r, t })
}

/// GPS 정렬 결과.
#[derive(Clone, Debug)]
pub struct GpsAlignment {
    /// 복원 좌표 → 동-북-위 변환.
    pub sim: Similarity,
    /// 대응별 정상 표시(잔차 ≤ 사용한 임계, 유한하지 않은 대응은 false).
    pub inliers: Vec<bool>,
    /// 대응별 잔차(m, 최종 변환 기준, 유한하지 않은 대응은 무한대).
    pub residuals: Vec<f64>,
    /// 정상 대응 잔차 중앙값(m).
    pub median_residual: f64,
    /// 실제로 쓴 제외 임계(m) = max(상한, 잡음 비례 임계).
    pub threshold_m: f64,
    /// 정상 카메라 중심의 주축별 표준편차(동-북-위 m, 내림차순). 둘째 값이 경로 폭.
    pub spread_m: [f64; 3],
    /// 첫째 주축 둘레 기울기의 추정 표준편차(도) ≈ σ_축 / (폭 · √N).
    /// 연직축을 고정했으면 0.
    pub tilt_sigma_deg: f64,
}

/// SPEC §3.4 기본 잔차 상한(m).
pub const GPS_MAX_RESIDUAL_M: f64 = 3.0;

/// 기울기 판정 문턱: 경로 둘째 주축 표준편차 < max(이 값, 첫째 축의 [`TILT_MIN_REL`]) 이면
/// 연직축 없이 정렬하지 않는다. 축당 GPS 잡음 1~2 m 에서 폭 5 m 표준편차, N≈100 이면
/// 기울기 σ ≈ 2/(5·10) rad ≈ 2.3° 로 이미 지면 기울기를 쓸 수 없는 수준이다.
pub const TILT_MIN_SPREAD_M: f64 = 5.0;
pub const TILT_MIN_REL: f64 = 0.01;

/// 잡음 비례 임계 배수: 3차원 가우스 잔차 크기 중앙 ≈ 1.538σ, 99% 분위 ≈ 3.37σ
/// → 중앙의 2.2배면 참 대응의 약 99% 를 남긴다.
pub const GPS_NOISE_THRESHOLD_MUL: f64 = 2.2;
/// 잡음 비례 임계의 상한 = 상한(m)의 이 배수. 3 m 기준 9 m 는 축당 σ ≈ 4 m 까지 받는다
/// (SPEC §6 의 1~2 m 의 두 배). 무관한 GPS 가 들어와 중앙값이 커져도 임계가 따라 커지지 않게 한다.
pub const GPS_NOISE_THRESHOLD_CAP: f64 = 3.0;

/// GPS 정렬 설정.
#[derive(Clone, Copy, Debug)]
pub struct GpsAlignConfig {
    /// 잔차 상한(m, SPEC §3.4 의 3 m).
    pub max_residual_m: f64,
    /// 참이면 임계 = max(상한, min(2.2 × 정상 잔차 중앙값, 3 × 상한)). GPS 잡음이 축당 1.4 m 를 넘을 때
    /// 고정 3 m 가 참 대응을 절반 넘게 버리는 것을 막는다.
    pub noise_adaptive: bool,
    /// 복원 좌표의 위(중력 반대) 방향. 있으면 연직축을 고정해 방위·배율·평행이동만 푼다.
    pub up: Option<Vector3<f64>>,
}

impl Default for GpsAlignConfig {
    fn default() -> Self {
        Self {
            max_residual_m: GPS_MAX_RESIDUAL_M,
            noise_adaptive: true,
            up: None,
        }
    }
}

/// 카메라 중심(복원 좌표)을 동-북-위 GPS 위치에 1회 정렬한다(기본 설정, 상한 `max_residual_m`).
pub fn align_to_enu(
    centers: &[Vector3<f64>],
    enu: &[Vector3<f64>],
    max_residual_m: f64,
) -> Option<GpsAlignment> {
    align_to_enu_with(
        centers,
        enu,
        &GpsAlignConfig {
            max_residual_m,
            ..Default::default()
        },
    )
}

/// 카메라 중심을 동-북-위 GPS 위치에 정렬한다.
///
/// 1. 유한하지 않은 대응 제외 → 강건 첫 추정([`robust_similarity`], 5회, 바닥 = 상한).
/// 2. 임계(상한 또는 잡음 비례) 초과 대응 제외 → 남은 대응으로 한 번 다시 추정
///    (`up` 이 있으면 연직축 고정 추정).
/// 3. 정상 대응이 max(3, 유한 대응의 50%) 미만이면 `None`.
/// 4. `up` 이 없고 경로 폭(둘째 주축 표준편차)이 [`TILT_MIN_SPREAD_M`]·첫째 축 1% 보다
///    작으면(거의 일직선) 경로 축 둘레 회전이 잡음으로 정해지므로 `None`.
pub fn align_to_enu_with(
    centers: &[Vector3<f64>],
    enu: &[Vector3<f64>],
    cfg: &GpsAlignConfig,
) -> Option<GpsAlignment> {
    if centers.len() != enu.len() {
        return None;
    }
    let ok: Vec<bool> = centers
        .iter()
        .zip(enu)
        .map(|(a, b)| finite(a) && finite(b))
        .collect();
    let n_ok = ok.iter().filter(|&&b| b).count();
    let (first, inl0, _) = robust_similarity(centers, enu, 5, cfg.max_residual_m)?;
    let mut sim = first;
    if let Some(u) = cfg.up.as_ref() {
        sim = fit_masked(centers, enu, &inl0, Some(u))?;
    }
    let res0 = residuals(&sim, centers, enu);
    let mut thr = cfg.max_residual_m;
    if cfg.noise_adaptive {
        let m = masked_median(&res0, &inl0);
        if m.is_finite() {
            thr = thr.max((GPS_NOISE_THRESHOLD_MUL * m).min(GPS_NOISE_THRESHOLD_CAP * thr));
        }
    }
    let keep: Vec<bool> = res0.iter().zip(&ok).map(|(&r, &k)| k && r <= thr).collect();
    if let Some(s) = fit_masked(centers, enu, &keep, cfg.up.as_ref()) {
        sim = s;
    }
    let res: Vec<f64> = residuals(&sim, centers, enu)
        .into_iter()
        .zip(&ok)
        .map(|(r, &k)| if k { r } else { f64::INFINITY })
        .collect();
    let inliers: Vec<bool> = res.iter().map(|&r| r <= thr).collect();
    let n_in = inliers.iter().filter(|&&b| b).count();
    if n_in < 3 || 2 * n_in < n_ok {
        return None;
    }
    let median_residual = masked_median(&res, &inliers);
    // 정상 중심의 주축 분산(동-북-위 m 단위).
    let pts: Vec<Vector3<f64>> = centers
        .iter()
        .zip(&inliers)
        .filter(|(_, &k)| k)
        .map(|(c, _)| sim.apply_point(c))
        .collect();
    let mu = pts.iter().sum::<Vector3<f64>>() / n_in as f64;
    let mut cov = Matrix3::zeros();
    for p in &pts {
        let d = p - mu;
        cov += d * d.transpose();
    }
    cov /= n_in as f64;
    let mut ev: Vec<f64> = cov
        .symmetric_eigenvalues()
        .iter()
        .map(|&e| e.max(0.0).sqrt())
        .collect();
    ev.sort_by(|a, b| b.total_cmp(a));
    let spread_m = [ev[0], ev[1], ev[2]];
    // 축당 잡음 σ ≈ 중앙 잔차 / 1.538.
    let sigma_axis = median_residual / 1.538;
    let tilt_sigma_deg = if cfg.up.is_some() {
        0.0
    } else {
        (sigma_axis / (spread_m[1].max(1e-12) * (n_in as f64).sqrt())).to_degrees()
    };
    if cfg.up.is_none() && spread_m[1] < TILT_MIN_SPREAD_M.max(TILT_MIN_REL * spread_m[0]) {
        return None;
    }
    Some(GpsAlignment {
        sim,
        inliers,
        residuals: res,
        median_residual,
        threshold_m: thr,
        spread_m,
        tilt_sigma_deg,
    })
}

/// 정밀 포즈(세계→카메라 회전)에서 복원 좌표의 위 방향을 구한다.
///
/// 짐벌 카메라는 옆으로 구르지 않으므로 카메라 x 축(세계 좌표 Rᵀe_x)은 수평이다.
/// 위 방향 = Σ x_i x_iᵀ 의 최소 고유벡터, 부호는 평균 광축(Rᵀe_z)이 아래를 보게 정한다.
/// 보는 방위가 한 방향뿐이면(둘째 고유값 < 최대의 [`UP_MIN_HEADING_SPREAD`]) 위 방향이
/// x 축 둘레로 정해지지 않으므로 `None`. 기울기 잡음 σ_p(카메라별)에서 오차 ≈ σ_p / √N.
pub fn up_from_rotations(rotations: &[Rotation3<f64>]) -> Option<Vector3<f64>> {
    let mut m = Matrix3::zeros();
    let mut zsum = Vector3::zeros();
    let mut n = 0usize;
    for r in rotations {
        let mt = r.matrix().transpose();
        let x: Vector3<f64> = mt.column(0).into();
        let z: Vector3<f64> = mt.column(2).into();
        if !(finite(&x) && finite(&z)) {
            continue;
        }
        m += x * x.transpose();
        zsum += z;
        n += 1;
    }
    if n < 2 {
        return None;
    }
    let eig = m.symmetric_eigen();
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let (l_mid, l_max) = (eig.eigenvalues[idx[1]], eig.eigenvalues[idx[2]]);
    if !(l_max > 0.0 && l_mid >= UP_MIN_HEADING_SPREAD * l_max) {
        return None;
    }
    let mut up: Vector3<f64> = eig.eigenvectors.column(idx[0]).into();
    let d = up.dot(&zsum);
    if !d.is_finite() || d.abs() < 1e-9 * n as f64 {
        return None;
    }
    if d > 0.0 {
        up = -up;
    }
    Some(up.normalize())
}

/// 위 방향 교차 검사의 경고 문턱(도). 한 묶음(기체·카메라 폴더)의 구름 치우침이 이 값을
/// 넘으면 [`UpCrossCheck::exceeds`] 가 참이다.
pub const UP_CROSS_WARN_DEG: f64 = 0.3;

/// 묶음별 위 방향 교차 검사 결과([`up_cross_check`]).
#[derive(Debug, Clone)]
pub struct UpCrossCheck {
    /// 묶음 라벨(오름차순).
    pub labels: Vec<usize>,
    /// 묶음 자기 회전만으로 구한 위 방향. 방위가 한 방향뿐이면 `None`.
    pub up_own: Vec<Option<Vector3<f64>>>,
    /// 그 묶음을 뺀 나머지 전체로 구한 위 방향. 나머지의 방위가 모자라면 `None`.
    pub up_rest: Vec<Option<Vector3<f64>>>,
    /// 묶음별 어긋남(도). 자기 위 방향이 있으면 `up_own` 과 `up_rest` 의 각 차, 없으면
    /// (방위가 한 방향뿐인 기체) 그 묶음 카메라 x 축이 `up_rest` 에 수직인 면에서 벗어난
    /// 부호 있는 각의 평균 절댓값(카메라 구름 r 이면 약 r·cos(아래 기울기)). 못 구하면 `None`.
    pub diff_deg: Vec<Option<f64>>,
    /// `diff_deg` 의 최댓값(도). 구한 묶음이 없으면 `None`.
    pub max_diff_deg: Option<f64>,
}

impl UpCrossCheck {
    /// 어긋남이 `threshold_deg` (보통 [`UP_CROSS_WARN_DEG`]) 를 넘는 묶음이 있는가.
    pub fn exceeds(&self, threshold_deg: f64) -> bool {
        self.max_diff_deg.is_some_and(|d| d > threshold_deg)
    }
}

/// 카메라 묶음(`labels[i]` 는 `rotations[i]` 의 묶음 라벨)별로 위 방향을 따로 구해
/// 묶음 사이의 어긋남을 돌려주는 순수 함수. [`up_from_rotations`] 의 동작은 바꾸지 않는다.
/// 세 기체가 따로 날아 짐벌 구름 치우침이 다를 때 한 기체의 치우침이 전체 위 방향에
/// 섞여 들어오는 것을 알아채기 위한 검사다. 길이가 다르거나 묶음이 둘 미만이면 `None`.
pub fn up_cross_check(rotations: &[Rotation3<f64>], labels: &[usize]) -> Option<UpCrossCheck> {
    if rotations.len() != labels.len() {
        return None;
    }
    let mut ids: Vec<usize> = labels.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() < 2 {
        return None;
    }
    let (mut up_own, mut up_rest, mut diff_deg) = (Vec::new(), Vec::new(), Vec::new());
    for &g in &ids {
        let pick = |inside: bool| -> Vec<Rotation3<f64>> {
            rotations
                .iter()
                .zip(labels)
                .filter(|(_, &l)| (l == g) == inside)
                .map(|(r, _)| *r)
                .collect()
        };
        let own_rots = pick(true);
        let own = up_from_rotations(&own_rots);
        let rest = up_from_rotations(&pick(false));
        let d = match (own, rest) {
            (Some(a), Some(b)) => Some(a.angle(&b).to_degrees()),
            (None, Some(b)) => {
                let t: Vec<f64> = own_rots
                    .iter()
                    .map(|r| Vector3::from(r.matrix().transpose().column(0)))
                    .filter(finite)
                    .map(|x| x.dot(&b).clamp(-1.0, 1.0).asin())
                    .collect();
                (!t.is_empty()).then(|| (t.iter().sum::<f64>() / t.len() as f64).abs().to_degrees())
            }
            _ => None,
        };
        up_own.push(own);
        up_rest.push(rest);
        diff_deg.push(d);
    }
    let max_diff_deg = diff_deg.iter().flatten().copied().reduce(f64::max);
    Some(UpCrossCheck {
        labels: ids,
        up_own,
        up_rest,
        diff_deg,
        max_diff_deg,
    })
}

/// [`up_from_rotations`] 의 방위 퍼짐 문턱(둘째/최대 고유값). 0.05 는 방위가 약 ±13° 이상
/// 퍼진 것에 해당한다.
pub const UP_MIN_HEADING_SPREAD: f64 = 0.05;

/// 정밀 포즈(세계→카메라 회전·카메라 중심)를 GPS 에 정렬하는 기본 진입점.
/// 위 방향을 [`up_from_rotations`] 로 데이터에서 구해 연직축 고정 추정을 쓰고,
/// 위 방향이 정해지지 않을 때만 자유 추정으로 돌아간다(이때 `tilt_sigma_deg` > 0).
pub fn gps_align_poses(
    rotations: &[Rotation3<f64>],
    centers: &[Vector3<f64>],
    gps: &[Geodetic],
    origin: &Geodetic,
) -> Option<GpsAlignment> {
    let cfg = GpsAlignConfig {
        up: up_from_rotations(rotations),
        ..Default::default()
    };
    gps_align_with(centers, gps, origin, &cfg)
}

/// 위경도 GPS 를 `origin` 기준 동-북-위로 바꾼 뒤 [`align_to_enu`] (상한 3 m).
pub fn gps_align(
    centers: &[Vector3<f64>],
    gps: &[Geodetic],
    origin: &Geodetic,
) -> Option<GpsAlignment> {
    gps_align_with(centers, gps, origin, &GpsAlignConfig::default())
}

/// [`gps_align`] 에 설정을 준다(연직축 고정 등).
pub fn gps_align_with(
    centers: &[Vector3<f64>],
    gps: &[Geodetic],
    origin: &Geodetic,
    cfg: &GpsAlignConfig,
) -> Option<GpsAlignment> {
    if centers.len() != gps.len() {
        return None;
    }
    let enu: Vec<Vector3<f64>> = gps.iter().map(|g| geodetic_to_enu(g, origin)).collect();
    align_to_enu_with(centers, &enu, cfg)
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
    fn image_consensus_survives_bad_images() {
        let mut rng = Rng(77);
        let gt = Similarity {
            s: 1.1,
            r: Rotation3::from_scaled_axis(Vector3::new(0.1, -0.2, 0.7)),
            t: Vector3::new(30.0, -10.0, 4.0),
        };
        let (mut src, mut dst, mut imgs) = (vec![], vec![], vec![]);
        for i in 0..400usize {
            let x = rng.uvec(60.0);
            // 이미지 0..8 은 정상, 8..12 는 관측이 모두 어긋난 이미지.
            let im = (i % 12) as u32;
            let mut y = gt.apply_point(&x) + rng.gvec(0.2);
            if im >= 8 {
                y += rng.uvec(40.0);
            }
            src.push(x);
            dst.push(y);
            imgs.push(vec![im, (im + 1) % 12]);
        }
        let r = image_consensus_similarity(&src, &dst, &imgs, 1.0, 0.3, 0.2).unwrap();
        let (se, re, te) = errs(&r.sim, &gt);
        assert!(se < 0.01 && re < 0.5 && te < 1.0, "{se} {re} {te}");
        assert!(r.good_images >= 6 && r.good_images <= 12);
        assert!(r.median < 0.5);
    }

    #[test]
    fn image_consensus_none_when_all_images_bad() {
        let mut rng = Rng(5);
        let src: Vec<_> = (0..60).map(|_| rng.uvec(50.0)).collect();
        let dst: Vec<_> = (0..60).map(|_| rng.uvec(50.0)).collect();
        let imgs: Vec<Vec<u32>> = (0..60).map(|i| vec![(i % 10) as u32]).collect();
        assert!(image_consensus_similarity(&src, &dst, &imgs, 0.5, 0.3, 0.2).is_none());
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
            // 정답 이상치는 하나도 남지 않고, 정답 정상은 95% 이상 유지.
            let bad_kept = (0..n).filter(|&k| bad[k] && al.inliers[k]).count();
            assert_eq!(bad_kept, 0, "seed {seed}: {bad_kept} bad kept");
            let good = (0..n).filter(|&k| !bad[k]).count();
            let good_kept = (0..n).filter(|&k| !bad[k] && al.inliers[k]).count();
            assert!(
                good_kept as f64 >= 0.95 * good as f64,
                "seed {seed}: kept {good_kept}/{good}"
            );
            assert!(
                al.median_residual < 2.0,
                "seed {seed}: {}",
                al.median_residual
            );
        }
        // 길이 불일치
        assert!(gps_align(&[Vector3::zeros()], &[], &origin).is_none());
    }

    /// 격자 비행(12×10, 25 m × 20 m 간격, 고도 100 m ± 1 m) 정답 위치.
    fn grid_enu(rng: &mut Rng) -> Vec<Vector3<f64>> {
        let mut v = Vec::new();
        for i in 0..12 {
            for j in 0..10 {
                v.push(Vector3::new(
                    i as f64 * 25.0,
                    j as f64 * 20.0,
                    100.0 + rng.gauss(),
                ));
            }
        }
        v
    }

    /// SPEC §1 실측 편대 띠: 위치 80곳 × 0.97 m 진행, 드론 3대가 쌍 거리 9.8·10.5·10.6 m 의
    /// 삼각형(진행 방향 x 에 변 AB 가 나란함), 고도 30 m ± 0.5 m.
    fn strip_enu(rng: &mut Rng) -> Vec<Vector3<f64>> {
        // AB = 9.8, AC = 10.5, BC = 10.6 m 인 평면 삼각형.
        let cx = (9.8f64 * 9.8 + 10.5 * 10.5 - 10.6 * 10.6) / (2.0 * 9.8);
        let cy = (10.5f64 * 10.5 - cx * cx).sqrt();
        let tri = [(0.0, 0.0), (9.8, 0.0), (cx, cy)];
        let mx = tri.iter().map(|p| p.0).sum::<f64>() / 3.0;
        let my = tri.iter().map(|p| p.1).sum::<f64>() / 3.0;
        let mut v = Vec::new();
        for i in 0..80 {
            for p in &tri {
                v.push(Vector3::new(
                    i as f64 * 0.97 + p.0 - mx,
                    p.1 - my,
                    30.0 + 0.5 * rng.gauss(),
                ));
            }
        }
        v
    }

    /// 편대 오프셋(x 앞, y 왼쪽)을 위치 `n` 곳 × `spacing` m 진행으로 늘어놓은 동-북-위 좌표.
    fn layout_enu(off: &[[f64; 3]; 3], n: usize, spacing: f64) -> Vec<Vector3<f64>> {
        (0..n)
            .flat_map(|i| {
                off.iter()
                    .map(move |o| Vector3::new(i as f64 * spacing + o[0], o[1], 30.0 + o[2]))
            })
            .collect()
    }

    /// 점 집합의 둘째 주축 표준편차(`gps_align` 의 `spread_m[1]` 과 같은 정의).
    fn second_axis_sd(pts: &[Vector3<f64>]) -> f64 {
        let mu = pts.iter().sum::<Vector3<f64>>() / pts.len() as f64;
        let mut cov = Matrix3::zeros();
        for p in pts {
            let d = p - mu;
            cov += d * d.transpose();
        }
        cov /= pts.len() as f64;
        let mut ev: Vec<f64> = cov
            .symmetric_eigenvalues()
            .iter()
            .map(|&e| e.max(0.0).sqrt())
            .collect();
        ev.sort_by(|a, b| b.total_cmp(a));
        ev[1]
    }

    /// F-361: 문서가 적은 편대 둘째 주축 표준편차를 실제 배치에서 계산해 맞춘다.
    /// 실제 배치(`SceneConfig::default()`, R–L 변이 진행 방향에 수직)는 횡 좌표 −0.52·−4.99·+5.51 m
    /// 라 약 4.3 m, 시험 띠(`strip_enu`, AB 가 진행 방향과 나란함)는 횡 0·0·9.34 m 라 약 4.4 m 이다.
    /// 둘 다 문턱 5 m 까지 여유가 0.6~0.7 m 로 작으므로 그 값을 단언하고 출력한다.
    #[test]
    fn formation_second_axis_spread_matches_doc() {
        let cfg = crate::synth::SceneConfig::default();
        let real = second_axis_sd(&layout_enu(&cfg.offsets, cfg.positions, cfg.spacing));
        let mut rng = Rng(1);
        let strip = second_axis_sd(&strip_enu(&mut rng));
        println!(
            "second axis sd: real {real:.3} m (margin {:.3} m), strip {strip:.3} m (margin {:.3} m), threshold {TILT_MIN_SPREAD_M} m",
            TILT_MIN_SPREAD_M - real,
            TILT_MIN_SPREAD_M - strip
        );
        // 모듈 문서의 값.
        assert!((real - 4.3).abs() < 0.1, "real {real}");
        assert!((strip - 4.4).abs() < 0.1, "strip {strip}");
        assert!(TILT_MIN_SPREAD_M - real > 0.5 && TILT_MIN_SPREAD_M - strip > 0.5);
        // `spread_m` 은 정렬된 카메라 중심 기준이라 GPS 잡음(σ 2 m)과 무관하게 잡음 없는 띠 값과 0.1 m 안.
        let c = gps_case(&mut rng, strip_enu(&mut Rng(2)), 2.0, 0.0, 0.0, 0.0);
        let cfgu = GpsAlignConfig {
            up: Some(c.gt.r.inverse() * Vector3::new(0.0, 0.0, 1.0)),
            ..Default::default()
        };
        let al = align_to_enu_with(&c.centers, &c.enu, &cfgu).unwrap();
        println!("align spread_m {:?}, strip {strip:.3}", al.spread_m);
        assert!((al.spread_m[1] - strip).abs() < 0.1, "{:?}", al.spread_m);
    }

    /// F-361: 오프셋 흔들림에서 위 방향 없는 판정(`None` ↔ 정렬)이 바뀌는지 잰다.
    /// 전체 배율 0.9~1.1 과 오프셋별 독립 ±10%(x·y 각각, 200회)를 잡음 0·1·1.5 m 에서 본다.
    /// 판정이 보는 값은 정렬된 카메라 중심의 둘째 주축 표준편차(`spread_m[1]`, 위 방향을 줘서
    /// 읽음)이며 GPS 점의 표준편차(잡음 포함)와 다르다. 둘 다 출력하고, 판정이 `spread_m[1]` 과
    /// 문턱의 비교와 일치하는지 단언한다. 잡음 0 에서는 ±10% 어디서도 `None` 이어야 한다.
    #[test]
    fn formation_threshold_sensitivity() {
        let base = crate::synth::SceneConfig::default();
        // (GPS 점 표준편차, 판정 값 `spread_m[1]`, 위 방향 없는 정렬 성공 여부)
        let decide = |off: &[[f64; 3]; 3], sigma: f64, seed: u64| -> (f64, f64, bool) {
            let mut rng = Rng(seed);
            let c = gps_case(
                &mut rng,
                layout_enu(off, base.positions, base.spacing),
                sigma,
                0.0,
                0.0,
                0.0,
            );
            let gps_sd = second_axis_sd(&c.enu);
            let clean_sd = second_axis_sd(&c.enu_true);
            let cfgu = GpsAlignConfig {
                up: Some(c.gt.r.inverse() * Vector3::new(0.0, 0.0, 1.0)),
                ..Default::default()
            };
            let al = align_to_enu_with(&c.centers, &c.enu, &cfgu).unwrap();
            let ok = align_to_enu(&c.centers, &c.enu, 3.0).is_some();
            // `spread_m` 은 정렬된 카메라 중심 기준이라 잡음 없는 배치 값과 0.1 m 안(GPS 점 값과는 다름).
            assert!(
                (al.spread_m[1] - clean_sd).abs() < 0.1,
                "spread_m {:?} clean {clean_sd} sigma {sigma}",
                al.spread_m
            );
            let gate = TILT_MIN_SPREAD_M.max(TILT_MIN_REL * al.spread_m[0]);
            // 판정은 `spread_m[1]` 만 본다(문턱과 0.05 m 안이면 두 적합의 차이로 갈릴 수 있어 제외).
            if (al.spread_m[1] - gate).abs() > 0.05 {
                assert_eq!(ok, al.spread_m[1] >= gate, "{:?} gate {gate}", al.spread_m);
            }
            (gps_sd, al.spread_m[1], ok)
        };
        for &sigma in &[0.0, 1.0, 1.5] {
            for &k in &[0.9, 0.95, 1.0, 1.05, 1.1] {
                let mut off = base.offsets;
                for o in off.iter_mut() {
                    o[0] *= k;
                    o[1] *= k;
                }
                let (gps_sd, sd, ok) = decide(&off, sigma, 7);
                println!(
                    "scale sigma {sigma} k {k:.2}: gps sd {gps_sd:.3} spread_m[1] {sd:.3} aligned {ok}"
                );
                if sigma == 0.0 {
                    assert!(!ok && sd < TILT_MIN_SPREAD_M, "k {k}: sd {sd}");
                }
            }
            let mut rng = Rng(99);
            let (mut flips, mut max_gps, mut max_sd) = (0, 0.0f64, 0.0f64);
            for t in 0..200u64 {
                let mut off = base.offsets;
                for o in off.iter_mut() {
                    o[0] *= 1.0 + 0.2 * (rng.uni() - 0.5);
                    o[1] *= 1.0 + 0.2 * (rng.uni() - 0.5);
                }
                let (gps_sd, sd, ok) = decide(&off, sigma, 1000 + t);
                max_gps = max_gps.max(gps_sd);
                max_sd = max_sd.max(sd);
                flips += ok as usize;
            }
            println!(
                "random ±10% sigma {sigma}: aligned {flips}/200, max gps sd {max_gps:.3}, max spread_m[1] {max_sd:.3}"
            );
            if sigma == 0.0 {
                assert_eq!(flips, 0);
                assert!(max_sd < TILT_MIN_SPREAD_M);
            }
        }
    }

    struct GpsCase {
        gt: Similarity,
        centers: Vec<Vector3<f64>>,
        enu_true: Vec<Vector3<f64>>,
        enu: Vec<Vector3<f64>>,
        bad: Vec<bool>,
    }

    /// 정답 위치 → 복원 좌표(임의 닮음 변환의 역) + 축당 σ 잡음 GPS, 앞쪽 `frac` 는 이상치.
    fn gps_case(
        rng: &mut Rng,
        enu_true: Vec<Vector3<f64>>,
        sigma_axis: f64,
        frac: f64,
        out_lo: f64,
        out_hi: f64,
    ) -> GpsCase {
        let gt = random_sim(rng);
        let gi = gt.inverse();
        let centers: Vec<_> = enu_true.iter().map(|e| gi.apply_point(e)).collect();
        let n = enu_true.len();
        // 이상치 위치를 고르게 섞는다(앞쪽 몰림 방지).
        let mut bad = vec![false; n];
        let n_bad = (frac * n as f64).round() as usize;
        let mut picked = 0;
        while picked < n_bad {
            let k = (rng.next_u64() % n as u64) as usize;
            if !bad[k] {
                bad[k] = true;
                picked += 1;
            }
        }
        let enu = enu_true
            .iter()
            .zip(&bad)
            .map(|(e, &b)| {
                let mut m = e + rng.gvec(sigma_axis);
                if b {
                    let d = rng.gvec(1.0).normalize();
                    m += d * (out_lo + (out_hi - out_lo) * rng.uni());
                }
                m
            })
            .collect();
        GpsCase {
            gt,
            centers,
            enu_true,
            enu,
            bad,
        }
    }

    /// (회전 오차 도, 영역 내 최대 위치 오차 m, 남은 이상치 수, 참 대응 유지율).
    fn gps_errs(c: &GpsCase, al: &GpsAlignment) -> (f64, f64, usize, f64) {
        let er = rot_err_deg(&al.sim.r, &c.gt.r);
        let ep = c
            .centers
            .iter()
            .zip(&c.enu_true)
            .filter(|(p, _)| finite(p))
            .map(|(p, e)| (al.sim.apply_point(p) - e).norm())
            .fold(0.0, f64::max);
        let n = c.bad.len();
        let bad_kept = (0..n).filter(|&k| c.bad[k] && al.inliers[k]).count();
        let good = (0..n).filter(|&k| !c.bad[k]).count();
        let good_kept = (0..n).filter(|&k| !c.bad[k] && al.inliers[k]).count();
        (er, ep, bad_kept, good_kept as f64 / good as f64)
    }

    /// F-094: 100~300 m 이상치 10·20·40%, 시드 10개씩. 첫 추정이 끌리면 수 도 틀린다.
    /// 기준: 회전 < 0.5°, 영역 내 위치 < 2 m, 남은 이상치 0 (격자 시험과 같은 근거).
    #[test]
    fn gps_alignment_survives_large_outliers() {
        for &frac in &[0.1, 0.2, 0.4] {
            let mut worst = (0.0f64, 0.0f64, 1.0f64);
            for seed in 0..10u64 {
                let mut rng = Rng(900 + seed);
                let g = grid_enu(&mut rng);
                let c = gps_case(&mut rng, g, 1.5 / 3f64.sqrt(), frac, 100.0, 300.0);
                let al = align_to_enu(&c.centers, &c.enu, 3.0).unwrap();
                let (er, ep, bk, keep) = gps_errs(&c, &al);
                worst = (worst.0.max(er), worst.1.max(ep), worst.2.min(keep));
                assert!(er < 0.5, "frac {frac} seed {seed}: rot {er}");
                assert!(ep < 2.0, "frac {frac} seed {seed}: pos {ep}");
                assert_eq!(bk, 0, "frac {frac} seed {seed}");
                assert!(al.median_residual.is_finite());
            }
            eprintln!(
                "outlier {frac}: rot max {:.4} deg pos max {:.3} m keep min {:.3}",
                worst.0, worst.1, worst.2
            );
        }
    }

    /// F-094·F-119: 정상 대응이 3개 미만이거나 절반 미만이면 `None`(NaN 중앙값 금지).
    #[test]
    fn gps_alignment_none_without_enough_inliers() {
        let mut rng = Rng(77);
        let g = grid_enu(&mut rng);
        // 전부 이상치: GPS 가 경로와 무관한 임의 위치.
        let c = gps_case(&mut rng, g.clone(), 0.5, 0.0, 0.0, 0.0);
        let junk: Vec<_> = (0..g.len()).map(|_| rng.uvec(300.0)).collect();
        assert!(align_to_enu(&c.centers, &junk, 3.0).is_none());
        // 정상 2개뿐.
        let mut few = junk.clone();
        few[0] = c.enu[0];
        few[1] = c.enu[1];
        assert!(align_to_enu(&c.centers, &few, 3.0).is_none());
        // 이상치 60%: 정상이 절반 미만.
        let c6 = gps_case(&mut rng, g, 0.5, 0.6, 100.0, 300.0);
        assert!(align_to_enu(&c6.centers, &c6.enu, 3.0).is_none());
        // 유한 대응 2개뿐인 강건 추정.
        let p = |x: f64, y: f64| Vector3::new(x, y, 0.0);
        let src = vec![p(0., 0.), p(1., 0.), p(0., 1.), p(f64::NAN, 0.)];
        let mut dst = src.clone();
        dst[2].x = f64::INFINITY;
        assert!(robust_similarity(&src, &dst, 5, 0.3).is_none());
    }

    /// 편대 띠 측정: 설정별 (회전 최악, 기울기 최악, 위치 최악, 유지 최소, 기울기σ 보고 최대, None 수).
    /// 설정: 0 = 연직 고정(위 방향은 정밀 포즈에서 추정), 1 = 자유(잡음 비례 임계), 2 = 자유 + 고정 3 m.
    /// 둘째 값: 연직 고정 설정의 시드별 (방위 오차 도(부호 있음), 방위 이론 σ 도).
    type StripWorst = [(f64, f64, f64, f64, f64, usize); 3];
    fn strip_stats(sigma: f64) -> (StripWorst, Vec<(f64, f64)>) {
        let ez = Vector3::new(0.0, 0.0, 1.0);
        let ex = Vector3::new(1.0, 0.0, 0.0);
        let mut w = [(0.0f64, 0.0f64, 0.0f64, 1.0f64, 0.0f64, 0usize); 3];
        let mut yaw = Vec::new();
        let mut up_err = Vec::new();
        for seed in 0..20u64 {
            let mut rng = Rng(1300 + seed);
            let s = strip_enu(&mut rng);
            let c = gps_case(&mut rng, s, sigma, 0.1, 10.0, 50.0);
            // 위 방향은 정밀 포즈(카메라당 0.1° 잡음)에서 데이터로 구한다.
            let rots = strip_rotations(&mut rng, &c, 0.1);
            let up = up_from_rotations(&rots).expect("up");
            let up_true = c.gt.r.inverse() * ez;
            up_err.push(up.angle(&up_true).to_degrees());
            let cfgs = [
                GpsAlignConfig {
                    up: Some(up),
                    ..Default::default()
                },
                GpsAlignConfig::default(),
                GpsAlignConfig {
                    noise_adaptive: false,
                    ..Default::default()
                },
            ];
            for (k, cfg) in cfgs.iter().enumerate() {
                let Some(al) = align_to_enu_with(&c.centers, &c.enu, cfg) else {
                    w[k].5 += 1;
                    continue;
                };
                let (er, ep, _, keep) = gps_errs(&c, &al);
                // 기울기 = 추정 변환이 참 위 방향을 +z 에서 얼마나 기울이는가.
                let up_true = c.gt.r.inverse() * ez;
                let et = (al.sim.r * up_true).angle(&ez).to_degrees();
                w[k].0 = w[k].0.max(er);
                w[k].1 = w[k].1.max(et);
                w[k].2 = w[k].2.max(ep);
                w[k].3 = w[k].3.min(keep);
                w[k].4 = w[k].4.max(al.tilt_sigma_deg);
                if k == 0 {
                    // 방위 = 오차 회전이 동쪽 축을 수평면에서 돌린 각.
                    let v = al.sim.r * c.gt.r.inverse() * ex;
                    let e_yaw = v.y.atan2(v.x).to_degrees();
                    yaw.push((e_yaw, yaw_sigma_deg(&c, &al, sigma)));
                }
            }
        }
        eprintln!(
            "strip sigma {sigma} up from poses: err max {:.4} deg",
            up_err.iter().fold(0.0f64, |a, &b| a.max(b))
        );
        for (k, name) in ["up-fixed", "free", "free-3m"].iter().enumerate() {
            if w[k].5 == 20 {
                eprintln!("strip sigma {sigma} {name}: no result 20/20");
                continue;
            }
            eprintln!(
                "strip sigma {sigma} {name}: rot max {:.3} deg tilt max {:.3} deg pos max {:.3} m keep min {:.3} tilt-sigma {:.3} deg none {}/20",
                w[k].0, w[k].1, w[k].2, w[k].3, w[k].4, w[k].5
            );
        }
        (w, yaw)
    }

    /// 편대 카메라 회전(세계→카메라, 복원 좌표): 드론마다 SPEC §1 실측 방위 −3°·+125°·−116°
    /// 를 쓰고 아래로 60° 기울인다(SPEC §1 의 58~62°). 카메라마다 σ `noise_deg` 의
    /// 임의 축 회전 잡음(구름 포함)을 넣는다.
    fn strip_rotations(rng: &mut Rng, c: &GpsCase, noise_deg: f64) -> Vec<Rotation3<f64>> {
        (0..c.centers.len())
            .map(|k| {
                let a = [-3.0f64, 125.0, -116.0][k % 3].to_radians();
                let t = 60f64.to_radians();
                let d = Vector3::new(t.cos() * a.cos(), t.cos() * a.sin(), -t.sin());
                let z = d;
                let x = z.cross(&Vector3::z()).normalize();
                let y = z.cross(&x);
                let r_enu = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
                    x.transpose(),
                    y.transpose(),
                    z.transpose(),
                ]));
                let e = rng.gvec(noise_deg.to_radians());
                let noise = Rotation3::from_scaled_axis(e);
                noise * r_enu * c.gt.r
            })
            .collect()
    }

    /// [`strip_rotations`] 와 같되 드론마다 지그재그 왕복을 한다: 사진 k 의 드론은 k % 3,
    /// 구간은 (k / 3) % 2 이고 구간마다 방위를 ±35° 번갈아 돌린다(정확히 180° 반대면 카메라
    /// x 축이 한 직선이라 위 방향이 정해지지 않으므로 옆바람 보정 각을 둔다).
    fn shuttle_rotations(rng: &mut Rng, c: &GpsCase, noise_deg: f64) -> Vec<Rotation3<f64>> {
        (0..c.centers.len())
            .map(|k| {
                let leg = if (k / 3) % 2 == 0 { 35.0 } else { -35.0 };
                let a = ([-3.0f64, 125.0, -116.0][k % 3] + leg).to_radians();
                let t = 60f64.to_radians();
                let z = Vector3::new(t.cos() * a.cos(), t.cos() * a.sin(), -t.sin());
                let x = z.cross(&Vector3::z()).normalize();
                let y = z.cross(&x);
                let r_enu = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
                    x.transpose(),
                    y.transpose(),
                    z.transpose(),
                ]));
                let noise = Rotation3::from_scaled_axis(rng.gvec(noise_deg.to_radians()));
                noise * r_enu * c.gt.r
            })
            .collect()
    }

    /// 연직축을 고정한 최소제곱 방위의 이론 표준편차(도): 정상 대응의 참 위치가
    /// 수평 중심에서 떨어진 거리 r_i 로 σ_ψ = σ_축 / √Σ r_i².
    /// (수평 회전 δψ 는 점 i 를 r_i·δψ 만큼 접선 방향으로 옮기고, 접선 방향 잡음은 축당 σ.)
    fn yaw_sigma_deg(c: &GpsCase, al: &GpsAlignment, sigma_axis: f64) -> f64 {
        let pts: Vec<Vector3<f64>> = c
            .enu_true
            .iter()
            .zip(&al.inliers)
            .filter(|(_, &k)| k)
            .map(|(e, _)| *e)
            .collect();
        let mu = pts.iter().sum::<Vector3<f64>>() / pts.len() as f64;
        let s2: f64 = pts
            .iter()
            .map(|p| (p.x - mu.x).powi(2) + (p.y - mu.y).powi(2))
            .sum();
        (sigma_axis / s2.sqrt()).to_degrees()
    }

    /// F-099·F-152 기울기·위치: SPEC §1 실측 편대 띠(77 m × 폭 약 10 m 삼각형), 이상치 10%(10~50 m), 시드 20, 축당 σ 1·2 m.
    /// 위 방향은 주입하지 않고 정밀 포즈(카메라당 축별 0.1° 잡음)에서 [`up_from_rotations`] 로 구한다.
    /// 연직 고정이면 기울기 = 위 방향 추정 오차 ≈ 0.1°·√2/√240 ≈ 0.01° 수준이라
    /// 기준 < 0.1°(F-152 의 0.5° 보다 엄격, 이론의 약 10배 여유). 위치 < 2 m, 유지 ≥ 90%, 실패 0.
    #[test]
    fn gps_alignment_formation_strip_tilt() {
        for &sigma in &[1.0, 2.0] {
            let (w, _) = strip_stats(sigma);
            let u = w[0];
            assert_eq!(u.5, 0, "sigma {sigma}: None");
            assert!(u.1 < 0.1, "sigma {sigma}: tilt {}", u.1);
            assert!(u.2 < 2.0, "sigma {sigma}: pos {}", u.2);
            assert!(u.3 >= 0.9, "sigma {sigma}: keep {}", u.3);
            // F-095·F-198·F-361: 이 시험 띠의 둘째 주축 표준편차는 약 4.4 m(잡음 포함 약 4.5 m) < 5 m 라
            // 위 방향 없는 자유 추정(잡음 비례 임계·고정 3 m 모두)은 20/20 `None` 이 기대 동작이다.
            // 이 배치의 정렬은 `up_from_rotations` 로 구한 위 방향이 있어야 한다.
            assert_eq!(w[1].5, 20, "sigma {sigma}: free None");
            assert_eq!(w[2].5, 20, "sigma {sigma}: fixed 3 m None");
        }
    }

    /// F-154: 위 방향 추정 단계를 임의 방향으로 바꾸면 같은 기준이 깨져야 한다
    /// (기울기 시험이 구성 성질만 보는 것이 아님을 확인). 또 방위가 한 방향뿐인 카메라는 `None`.
    #[test]
    fn up_from_rotations_is_load_bearing() {
        let ez = Vector3::new(0.0, 0.0, 1.0);
        let mut worst_rand = 0.0f64;
        for seed in 0..20u64 {
            let mut rng = Rng(1300 + seed);
            let s = strip_enu(&mut rng);
            let c = gps_case(&mut rng, s, 2.0, 0.1, 10.0, 50.0);
            let up_true = c.gt.r.inverse() * ez;
            let rots = strip_rotations(&mut rng, &c, 0.1);
            let up = up_from_rotations(&rots).unwrap();
            assert!(up.angle(&up_true).to_degrees() < 0.1, "seed {seed}");
            let rand_up = rng.gvec(1.0).normalize();
            let cfg = GpsAlignConfig {
                up: Some(rand_up),
                ..Default::default()
            };
            if let Some(al) = align_to_enu_with(&c.centers, &c.enu, &cfg) {
                worst_rand = worst_rand.max((al.sim.r * up_true).angle(&ez).to_degrees());
            } else {
                worst_rand = f64::INFINITY;
            }
        }
        assert!(worst_rand > 0.5, "random up tilt {worst_rand}");
        // 한 방위만 보는 카메라: 위 방향이 정해지지 않는다.
        let mut rng = Rng(5);
        let s = strip_enu(&mut rng);
        let c = gps_case(&mut rng, s, 1.0, 0.0, 0.0, 0.0);
        let one: Vec<_> = strip_rotations(&mut rng, &c, 0.0)
            .into_iter()
            .step_by(3)
            .collect();
        assert!(up_from_rotations(&one).is_none());
        // 기본 진입점이 포즈에서 위 방향을 써서 연직 고정 추정을 한다.
        let origin = Geodetic {
            lat_deg: 37.5,
            lon_deg: 127.0,
            alt: 50.0,
        };
        let gps: Vec<_> = c.enu.iter().map(|e| enu_to_geodetic(e, &origin)).collect();
        let rots = strip_rotations(&mut rng, &c, 0.1);
        let al = gps_align_poses(&rots, &c.centers, &gps, &origin).unwrap();
        assert_eq!(al.tilt_sigma_deg, 0.0);
        let up_true = c.gt.r.inverse() * ez;
        assert!((al.sim.r * up_true).angle(&ez).to_degrees() < 0.1);
    }

    /// F-199: 한 기체 카메라에만 일정한 구름(광축 둘레 회전)이 있을 때 묶음별 교차 검사.
    /// SPEC 삼각 편대 띠, σ 2 m 와 같은 시드 20. 구름 0.5/1/2° 의 최악 어긋남과 전체
    /// 위 방향 기울기를 출력한다(구름 민감도 표). 구름이 없으면 어긋남·기울기 < 0.1°.
    #[test]
    fn up_cross_check_flags_single_drone_roll() {
        let ez = Vector3::new(0.0, 0.0, 1.0);
        let mut worst = [(0.0f64, 0.0f64, f64::INFINITY); 4];
        for (i, &roll) in [0.0f64, 0.5, 1.0, 2.0].iter().enumerate() {
            for seed in 0..20u64 {
                let mut rng = Rng(1300 + seed);
                let s = strip_enu(&mut rng);
                let c = gps_case(&mut rng, s, 2.0, 0.1, 10.0, 50.0);
                let mut rots = strip_rotations(&mut rng, &c, 0.1);
                let labels: Vec<usize> = (0..rots.len()).map(|k| k % 3).collect();
                let rz = Rotation3::from_axis_angle(&Vector3::z_axis(), roll.to_radians());
                for (r, &l) in rots.iter_mut().zip(&labels) {
                    if l == 0 {
                        *r = rz * *r;
                    }
                }
                let up_true = c.gt.r.inverse() * ez;
                let up = up_from_rotations(&rots).unwrap();
                let tilt = up.angle(&up_true).to_degrees();
                let chk = up_cross_check(&rots, &labels).unwrap();
                assert_eq!(chk.labels, vec![0, 1, 2]);
                // 이 배치는 묶음마다 방위가 한 방향뿐이라 대체(x 축 기울기 평균) 갈래만 돈다.
                // 자기 위 방향 갈래는 `up_cross_check_own_branch_on_shuttle_layout` 에서 본다.
                assert!(chk.up_own.iter().all(|u| u.is_none()));
                assert!(chk.diff_deg.iter().all(|d| d.is_some()));
                let d = chk.max_diff_deg.unwrap();
                worst[i].0 = worst[i].0.max(d);
                worst[i].1 = worst[i].1.max(tilt);
                // 가장 작은 어긋남(구름 있을 때 모든 시드에서 경고가 나는지 확인).
                worst[i].2 = worst[i].2.min(d);
            }
            eprintln!(
                "roll {roll} deg: cross diff max {:.3} min {:.3} deg, overall up tilt max {:.3} deg",
                worst[i].0, worst[i].2, worst[i].1
            );
        }
        // 구름 없음: 어긋남·기울기 < 0.1°.
        assert!(worst[0].0 < 0.1, "no roll diff {}", worst[0].0);
        assert!(worst[0].1 < 0.1, "no roll tilt {}", worst[0].1);
        // 구름이 있으면 어긋남이 구름과 함께 커지고 2° 에서는 모든 시드가 문턱을 넘는다.
        assert!(worst[1].2 < worst[2].2 && worst[2].2 < worst[3].2);
        // 문턱 0.3° 는 1° 이상 구름을 모든 시드에서 잡고, 0.5° 는 시드에 따라 갈린다(0.22~0.31°).
        assert!(worst[2].2 > UP_CROSS_WARN_DEG, "1 deg min {}", worst[2].2);
        assert!(worst[3].2 > UP_CROSS_WARN_DEG, "2 deg min {}", worst[3].2);
        assert!(worst[3].0 < 1.2 * 1.116, "2 deg max {}", worst[3].0);
        // 라벨 길이 불일치·묶음 하나뿐이면 검사하지 않는다.
        let mut rng = Rng(7);
        let s = strip_enu(&mut rng);
        let c = gps_case(&mut rng, s, 1.0, 0.0, 0.0, 0.0);
        let rots = strip_rotations(&mut rng, &c, 0.1);
        assert!(up_cross_check(&rots, &[0]).is_none());
        assert!(up_cross_check(&rots, &vec![0; rots.len()]).is_none());
    }

    /// F-440: 묶음마다 방위가 둘 이상인 지그재그 왕복 배치에서는 `up_own` 이 `Some` 이라
    /// 자기 위 방향 갈래(`up_own` 과 `up_rest` 의 각)가 돈다. 한 기체(라벨 0)에만 구름 ±2°.
    /// 단언: 모든 묶음의 `up_own`·`up_rest` 가 `Some`, 어긋남이 두 위 방향의 각과 정확히 같음
    /// (x 축 평균 갈래와 구별), 구름 없음 < 0.1°, 구름 2° 에서 0.3° 초과·구름 낀 묶음이 가장 큼,
    /// 구름 부호를 바꿔도 어긋남 크기가 같음.
    #[test]
    fn up_cross_check_own_branch_on_shuttle_layout() {
        let ez = Vector3::new(0.0, 0.0, 1.0);
        let mut none_max = 0.0f64;
        let (mut roll_min, mut roll_max) = (f64::INFINITY, 0.0f64);
        let mut other_max = 0.0f64;
        let mut sign_gap = 0.0f64;
        for seed in 0..20u64 {
            let mut diffs = [[0.0f64; 3]; 3];
            for (i, &roll) in [0.0f64, 2.0, -2.0].iter().enumerate() {
                let mut rng = Rng(1300 + seed);
                let s = strip_enu(&mut rng);
                let c = gps_case(&mut rng, s, 2.0, 0.1, 10.0, 50.0);
                let mut rots = shuttle_rotations(&mut rng, &c, 0.1);
                let labels: Vec<usize> = (0..rots.len()).map(|k| k % 3).collect();
                let rz = Rotation3::from_axis_angle(&Vector3::z_axis(), roll.to_radians());
                for (r, &l) in rots.iter_mut().zip(&labels) {
                    if l == 0 {
                        *r = rz * *r;
                    }
                }
                let chk = up_cross_check(&rots, &labels).unwrap();
                assert_eq!(chk.labels, vec![0, 1, 2]);
                assert!(
                    chk.up_own.iter().all(|u| u.is_some()),
                    "seed {seed}: up_own"
                );
                assert!(
                    chk.up_rest.iter().all(|u| u.is_some()),
                    "seed {seed}: up_rest"
                );
                for (g, slot) in diffs[i].iter_mut().enumerate() {
                    let want = chk.up_own[g]
                        .unwrap()
                        .angle(&chk.up_rest[g].unwrap())
                        .to_degrees();
                    let got = chk.diff_deg[g].unwrap();
                    assert!((got - want).abs() < 1e-9, "seed {seed} g {g}: {got} {want}");
                    *slot = got;
                }
                // 구름 없는 묶음의 자기 위 방향은 참값에 가깝다(0.1°·√2/√80 ≈ 0.016° 수준).
                let up_true = c.gt.r.inverse() * ez;
                if roll == 0.0 {
                    for u in chk.up_own.iter().flatten() {
                        assert!(u.angle(&up_true).to_degrees() < 0.1, "seed {seed}");
                    }
                    none_max = none_max.max(chk.max_diff_deg.unwrap());
                } else {
                    let d0 = chk.diff_deg[0].unwrap();
                    roll_min = roll_min.min(d0);
                    roll_max = roll_max.max(d0);
                    // 구름 있는 묶음의 자기 위 방향은 어긋나 있다.
                    assert!(
                        chk.up_own[0].unwrap().angle(&up_true).to_degrees() > 0.3,
                        "seed {seed}"
                    );
                }
            }
            sign_gap = sign_gap.max((diffs[1][0] - diffs[2][0]).abs());
            // 나머지 묶음의 `up_rest` 에는 구름 낀 묶음이 섞여 있어 그쪽도 어긋나 보이지만 더 작다.
            assert!(
                diffs[1][0] > diffs[1][1] && diffs[1][0] > diffs[1][2],
                "seed {seed}"
            );
            other_max = other_max.max(diffs[1][1]).max(diffs[1][2]);
        }
        eprintln!(
            "shuttle: no roll max {none_max:.3} deg, roll 2 deg own diff min {roll_min:.3} max {roll_max:.3} deg, other groups max {other_max:.3}, sign gap {sign_gap:.3}"
        );
        assert!(none_max < 0.1, "no roll {none_max}");
        assert!(roll_min > UP_CROSS_WARN_DEG, "2 deg min {roll_min}");
        assert!(sign_gap < 0.15, "sign gap {sign_gap}");
    }

    /// F-099 방위(연직축 둘레): 최소제곱 방위 오차는 잡음 한계 σ_ψ = σ_축 / √Σ r_i²
    /// (SPEC §1 편대 띠, 정상 약 216개, 이론값은 시험이 시드별로 계산해 출력)를 따라야 한다.
    /// 시드별 정규화 오차 z = 오차 / σ_ψ 에 대해
    /// - 최대 |z| < 3.5 (가우스면 시드 하나가 넘을 확률 4.7e-4, 20개 중 하나라도 ≈ 0.9%),
    /// - 제곱평균 z ∈ [0.61, 1.41] (χ²₂₀ 의 0.5%·99.5% 분위 7.43·40.0 을 20 으로 나눈 제곱근),
    /// - 절대 최악 < 3.5 × 이론 σ 최대.
    ///
    /// 고정 0.5° 는 σ 2 m 에서 1.52 σ_ψ 라 20 시드 모두 들어올 확률이 0.871²⁰ ≈ 6% 뿐이어서
    /// 기준으로 쓰지 않는다.
    #[test]
    fn gps_alignment_formation_strip_yaw() {
        for &sigma in &[1.0, 2.0] {
            let (_, yaw) = strip_stats(sigma);
            assert_eq!(yaw.len(), 20);
            let zmax = yaw.iter().map(|(e, s)| (e / s).abs()).fold(0.0, f64::max);
            let zrms = (yaw.iter().map(|(e, s)| (e / s).powi(2)).sum::<f64>() / 20.0).sqrt();
            let emax = yaw.iter().map(|(e, _)| e.abs()).fold(0.0, f64::max);
            let smax = yaw.iter().map(|(_, s)| *s).fold(0.0, f64::max);
            let smin = yaw.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
            eprintln!(
                "strip sigma {sigma} yaw: max {emax:.3} deg theory sigma {smin:.3}..{smax:.3} deg z max {zmax:.2} z rms {zrms:.2}"
            );
            assert!(zmax < 3.5, "sigma {sigma}: z max {zmax}");
            assert!((0.61..=1.41).contains(&zrms), "sigma {sigma}: z rms {zrms}");
            assert!(emax < 3.5 * smax, "sigma {sigma}: yaw {emax}");
        }
    }

    /// F-095: 거의 일직선(600 m, 옆·위 흔들림 σ 0.3 m)이면 연직축 없이 `None`,
    /// 연직축을 주면 정렬된다. SPEC §1 삼각 편대(폭 약 10 m)는 연직축 없이는 `None`(폭 20 m 일렬만 통과).
    #[test]
    fn gps_alignment_straight_path_tilt_undetermined() {
        let ez = Vector3::new(0.0, 0.0, 1.0);
        for seed in 0..5u64 {
            let mut rng = Rng(1700 + seed);
            let line: Vec<_> = (0..60)
                .map(|i| {
                    Vector3::new(
                        i as f64 * 10.0,
                        0.3 * rng.gauss(),
                        100.0 + 0.3 * rng.gauss(),
                    )
                })
                .collect();
            let c = gps_case(&mut rng, line, 1.5 / 3f64.sqrt(), 0.0, 0.0, 0.0);
            assert!(
                align_to_enu(&c.centers, &c.enu, 3.0).is_none(),
                "seed {seed}"
            );
            let cfg = GpsAlignConfig {
                up: Some(c.gt.r.inverse() * ez),
                ..Default::default()
            };
            let al = align_to_enu_with(&c.centers, &c.enu, &cfg).unwrap();
            let (er, ep, _, _) = gps_errs(&c, &al);
            assert!(er < 0.5 && ep < 2.0, "seed {seed}: {er} {ep}");
        }
        let mut rng = Rng(1800);
        let s = strip_enu(&mut rng);
        let c = gps_case(&mut rng, s, 1.0, 0.0, 0.0, 0.0);
        // 시험 띠(SPEC §1 삼각 편대): 둘째 주축 표준편차 약 4.4 m < 5 m 라 위 방향 없이는 `None`.
        assert!(align_to_enu(&c.centers, &c.enu, 3.0).is_none());
        // 폭 20 m 일렬(횡 −10/0/+10 m)은 문턱을 넘어 위 방향 없이도 정렬된다.
        let wide: Vec<_> = (0..80)
            .flat_map(|i| {
                (0..3).map(move |d| Vector3::new(i as f64 * 0.97, (d as f64 - 1.0) * 10.0, 100.0))
            })
            .collect();
        let c = gps_case(&mut rng, wide, 1.0, 0.0, 0.0, 0.0);
        let al = align_to_enu(&c.centers, &c.enu, 3.0).unwrap();
        assert!(al.spread_m[1] > TILT_MIN_SPREAD_M, "{:?}", al.spread_m);
        assert!(al.tilt_sigma_deg > 0.0 && al.tilt_sigma_deg.is_finite());
    }

    /// F-100: 같은 방향 오프셋 이상치(한 구역이 통째로 어긋남) 30·40·45%, 시드 20.
    /// 기준(40%): 배율 < 0.5%, 회전 < 0.25°, 이상치 전부 제외. 30·45% 도 같은 기준.
    #[test]
    fn robust_one_sided_outliers() {
        for &frac in &[0.3, 0.4, 0.45] {
            let mut worst = (0.0f64, 0.0f64);
            for seed in 0..20u64 {
                let mut rng = Rng(2100 + seed);
                let gt = random_sim(&mut rng);
                let n = 200;
                let sigma = 1.0 + rng.uni();
                let off = rng.gvec(1.0).normalize() * (20.0 + 30.0 * rng.uni());
                let mut src = Vec::new();
                let mut dst = Vec::new();
                let mut truth = Vec::new();
                for i in 0..n {
                    let p = rng.uvec(200.0) / gt.s;
                    let mut q = gt.apply_point(&p) + rng.gvec(sigma);
                    let out = (i as f64) < frac * n as f64;
                    if out {
                        q += off;
                    }
                    src.push(p);
                    dst.push(q);
                    truth.push(!out);
                }
                let (est, inl, _) = robust_similarity(&src, &dst, 5, 0.3).unwrap();
                let (es, er, _) = errs(&est, &gt);
                worst = (worst.0.max(es), worst.1.max(er));
                assert!(es < 5e-3, "frac {frac} seed {seed}: scale {es}");
                assert!(er < 0.25, "frac {frac} seed {seed}: rot {er}");
                let kept_bad = inl.iter().zip(&truth).filter(|(&a, &b)| a && !b).count();
                assert_eq!(kept_bad, 0, "frac {frac} seed {seed}");
            }
            eprintln!(
                "one-sided {frac}: scale max {:.2e} rot max {:.4} deg",
                worst.0, worst.1
            );
        }
    }

    /// F-079: 완전 평면(z = 0) 점은 det(U·Vᵀ) < 0 분기를 자주 탄다. 50 시드 정확 복원,
    /// det R = +1. 반사 보정을 빼면 이 시험이 실패한다.
    #[test]
    fn planar_points_exact_with_reflection_branch() {
        let mut rng = Rng(2500);
        let mut branch = 0;
        for seed in 0..50 {
            let gt = random_sim(&mut rng);
            let src: Vec<_> = (0..15)
                .map(|_| {
                    let v = rng.uvec(30.0);
                    Vector3::new(v.x, v.y, 0.0)
                })
                .collect();
            let dst: Vec<_> = src.iter().map(|p| gt.apply_point(p)).collect();
            // 분기 사용 여부를 같은 공분산으로 센다.
            let mu_s = src.iter().sum::<Vector3<f64>>() / 15.0;
            let mu_d = dst.iter().sum::<Vector3<f64>>() / 15.0;
            let mut cov = Matrix3::zeros();
            for (a, b) in src.iter().zip(&dst) {
                cov += (b - mu_d) * (a - mu_s).transpose();
            }
            let svd = cov.svd(true, true);
            if svd.u.unwrap().determinant() * svd.v_t.unwrap().determinant() < 0.0 {
                branch += 1;
            }
            let est = umeyama(&src, &dst).unwrap();
            let (es, _, _) = errs(&est, &gt);
            // acos 기반 각도는 1e-8 rad 근처에서 정밀도가 모자라 행렬 차 노름(≈ √2·각, rad)으로 본다.
            let er = (est.r.matrix() - gt.r.matrix()).norm();
            assert!(er < 1e-8, "seed {seed}: rot {er}");
            assert!(es < 1e-10, "seed {seed}: scale {es}");
            assert!((est.r.matrix().determinant() - 1.0).abs() < 1e-10);
            let (rob, _, _) = robust_similarity(&src, &dst, 5, 0.3).unwrap();
            assert!(
                (rob.r.matrix() - gt.r.matrix()).norm() < 1e-8,
                "seed {seed}"
            );
        }
        eprintln!("planar reflection branch taken {branch}/50");
        assert!(branch > 0);
    }

    /// F-096: 대응 하나가 NaN 이어도 정렬은 살아 있고, 그 대응만 제외된다.
    #[test]
    fn gps_alignment_ignores_nan_pair() {
        for seed in 0..10u64 {
            let mut rng = Rng(2700 + seed);
            let g = grid_enu(&mut rng);
            let mut c = gps_case(&mut rng, g, 1.5 / 3f64.sqrt(), 0.2, 10.0, 50.0);
            let k = 5 + seed as usize;
            if seed % 2 == 0 {
                c.centers[k].x = f64::NAN;
            } else {
                c.enu[k].y = f64::NAN;
            }
            c.bad[k] = true;
            let al = align_to_enu(&c.centers, &c.enu, 3.0).unwrap();
            assert!(!al.inliers[k] && al.residuals[k].is_infinite());
            let (er, ep, bk, keep) = gps_errs(&c, &al);
            assert!(er < 0.5 && ep < 2.0, "seed {seed}: {er} {ep}");
            assert_eq!(bk, 0, "seed {seed}");
            assert!(keep >= 0.95, "seed {seed}: keep {keep}");
            assert!(al.median_residual.is_finite());
        }
    }
}
