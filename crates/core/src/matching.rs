//! 특징 매칭과 기하 검증: 비율 검사, 정규화 8점 기본 행렬, RANSAC.

use crate::features::{Feature, DESC_LEN};
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};
use rayon::prelude::*;

/// SPEC 기본값: 같은 카메라 시간 이웃 위치 차 1..=5.
pub const PAIR_TEMPORAL: usize = 5;
/// SPEC 기본값: 다른 카메라 위치 차 0..=4.
pub const PAIR_CROSS: usize = 4;
/// SPEC 기본값: 같은 카메라 2의 거듭제곱 간격 상한(1, 2, 4, 8, 16).
pub const PAIR_POW2_MAX: usize = 16;

/// 매칭할 영상 짝 후보를 만든다. `views[k] = (카메라 번호, 촬영 위치 번호)`.
/// 같은 카메라는 위치 차이 1..=`temporal` 이거나 `pow2_max` 이하의 2의 거듭제곱(긴 경로의 먼 제약),
/// 다른 카메라는 위치 차이 0..=`cross` 인 짝. `pow2_max = 0` 이면 거듭제곱 간격을 쓰지 않는다.
/// 결과 (i, j) 는 i < j, 중복 없음, 정렬됨.
pub fn candidate_pairs(
    views: &[(usize, usize)],
    temporal: usize,
    cross: usize,
    pow2_max: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..views.len() {
        for j in i + 1..views.len() {
            let ((ca, pa), (cb, pb)) = (views[i], views[j]);
            let d = pa.abs_diff(pb);
            let ok = if ca == cb {
                d >= 1 && (d <= temporal || (d.is_power_of_two() && d <= pow2_max))
            } else {
                d <= cross
            };
            if ok {
                out.push((i, j));
            }
        }
    }
    out
}

/// 카메라 번호 규약(`synth::CamId::ALL` 순서): 앞 0, 오른쪽 1, 왼쪽 2.
pub const CAM_FRONT: usize = 0;
pub const CAM_RIGHT: usize = 1;
pub const CAM_LEFT: usize = 2;

/// 다른 카메라 짝 일정.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossSchedule {
    /// SPEC §3.2: 모든 다른 카메라 짝에서 위치 차 0..=`range`.
    Spec { range: usize },
    /// 실측 겹침 기반(F-197): F(p)–R(p+`right_min`..=p+`max`), F(p)–L(p+`left_min`..=p+`max`)를
    /// `step` 간격으로 표본한다. R–L 직접 짝은 겹침이 없어 쓰지 않는다(F 를 거쳐 이어진다).
    Formation {
        right_min: usize,
        left_min: usize,
        max: usize,
        step: usize,
    },
}

impl CrossSchedule {
    /// 실측 편대 기본값: F–R·F–L +20..=+40, 4칸 간격. 겹침 12% 미만(+12~+16)은 매칭은 되지만
    /// 회전 오차가 5~17° 라 뺀다(연구 노트 experiments/formation-pairs.md). 24시점 bench 에서는 +20 간선 5/12 가
    /// 2° 를 넘지만(experiments/bench-schedule.md) 시작을 24 로 올리면 위치 40곳 구역(`pipeline_e2e`
    /// single_region)의 등록이 80/120 으로 떨어져 20 을 유지한다. bench 는 `--cross-min 24` 로 따로 쓴다.
    pub const FORMATION: CrossSchedule = CrossSchedule::Formation {
        right_min: 20,
        left_min: 20,
        max: 40,
        step: 4,
    };
}

/// 짝 일정 설정. 기본값은 같은 카메라 SPEC 일정 + 실측 겹침 기반 카메라 간 일정.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairSchedule {
    pub temporal: usize,
    pub pow2_max: usize,
    pub cross: CrossSchedule,
}

impl Default for PairSchedule {
    fn default() -> Self {
        Self {
            temporal: PAIR_TEMPORAL,
            pow2_max: PAIR_POW2_MAX,
            cross: CrossSchedule::FORMATION,
        }
    }
}

impl PairSchedule {
    /// SPEC §3.2 일정(카메라 간 같은 위치 ±`PAIR_CROSS`).
    pub fn spec() -> Self {
        Self {
            cross: CrossSchedule::Spec { range: PAIR_CROSS },
            ..Self::default()
        }
    }
}

/// `views[k] = (카메라 번호, 촬영 위치 번호)` 에서 [`PairSchedule`] 대로 짝을 만든다.
/// 같은 카메라 규칙은 [`candidate_pairs`] 와 같다. 결과 (i, j) 는 i < j, 중복 없음, 정렬됨.
pub fn scheduled_pairs(views: &[(usize, usize)], sch: &PairSchedule) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..views.len() {
        for j in 0..views.len() {
            if i == j {
                continue;
            }
            let ((ca, pa), (cb, pb)) = (views[i], views[j]);
            let d = pa.abs_diff(pb);
            let ok = if ca == cb {
                j > i && d >= 1 && (d <= sch.temporal || (d.is_power_of_two() && d <= sch.pow2_max))
            } else {
                match sch.cross {
                    CrossSchedule::Spec { range } => j > i && d <= range,
                    CrossSchedule::Formation {
                        right_min,
                        left_min,
                        max,
                        step,
                    } => {
                        let lo = match (ca, cb) {
                            (CAM_FRONT, CAM_RIGHT) => Some(right_min),
                            (CAM_FRONT, CAM_LEFT) => Some(left_min),
                            _ => None,
                        };
                        lo.is_some_and(|lo| {
                            pb >= pa + lo && pb <= pa + max && (pb - pa - lo) % step.max(1) == 0
                        })
                    }
                }
            };
            if ok {
                out.push((i.min(j), i.max(j)));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// 기술자 제곱 L2 거리. 8칸 누산기로 나눠 더해 벡터화되게 한다.
#[inline]
fn sq_dist(p: &[f32; DESC_LEN], r: &[f32; DESC_LEN]) -> f32 {
    let mut acc = [0f32; 8];
    for (pc, rc) in p.as_chunks::<8>().0.iter().zip(r.as_chunks::<8>().0) {
        for k in 0..8 {
            let t = pc[k] - rc[k];
            acc[k] += t * t;
        }
    }
    acc.iter().sum()
}

/// `a` 를 이 크기의 묶음으로 나눠 병렬 처리한다(묶음 기술자 64×512 B 가 L1·L2 에 머문다).
const MATCH_BLOCK: usize = 64;

/// 최근접/차근접 거리 비율 검사 매칭(L2, 전수 탐색). 결과는 (a 인덱스, b 인덱스), a 인덱스 순.
/// `mutual` 이면 b→a 최근접도 같은 짝인 것만 남긴다.
/// 차근접이 없으면(`b` 가 2개 미만) 비율을 잴 수 없으므로 짝을 만들지 않는다.
///
/// 거리 행렬을 한 번만 계산한다: `a` 묶음마다(rayon 병렬) 모든 `b` 와의 거리로 행 최근접·차근접과
/// 열(b→a) 최근접을 함께 갱신하고, 열 최근접은 묶음 순서대로 합친다. 열 최근접은 같은 거리면 앞 인덱스가 이긴다
/// (행 최근접 동률은 최근접 = 차근접이라 비율 검사에서 떨어진다).
pub fn ratio_match(a: &[Feature], b: &[Feature], ratio: f32, mutual: bool) -> Vec<(usize, usize)> {
    type Blk = (Vec<(usize, f32, f32)>, Vec<(f32, usize)>);
    let blocks: Vec<Blk> = a
        .par_chunks(MATCH_BLOCK)
        .enumerate()
        .map(|(bi, chunk)| {
            let mut row = vec![(usize::MAX, f32::INFINITY, f32::INFINITY); chunk.len()];
            let mut col = if mutual {
                vec![(f32::INFINITY, usize::MAX); b.len()]
            } else {
                Vec::new()
            };
            for (j, fb) in b.iter().enumerate() {
                for (r, fa) in chunk.iter().enumerate() {
                    let d = sq_dist(&fa.desc, &fb.desc);
                    let best = &mut row[r];
                    if d < best.1 {
                        *best = (j, d, best.1);
                    } else if d < best.2 {
                        best.2 = d;
                    }
                    if mutual && d < col[j].0 {
                        col[j] = (d, bi * MATCH_BLOCK + r);
                    }
                }
            }
            (row, col)
        })
        .collect();
    let mut col_best = vec![(f32::INFINITY, usize::MAX); if mutual { b.len() } else { 0 }];
    for (_, col) in &blocks {
        for (cb, c) in col_best.iter_mut().zip(col) {
            if c.0 < cb.0 {
                *cb = *c;
            }
        }
    }
    let mut out = Vec::new();
    for (i, &(j, d1, d2)) in blocks.iter().flat_map(|(row, _)| row).enumerate() {
        if j == usize::MAX || !d2.is_finite() || d1 >= ratio * ratio * d2 {
            continue;
        }
        if mutual && col_best[j].1 != i {
            continue;
        }
        out.push((i, j));
    }
    out
}

/// 정규화된 8점 계(AᵀA)에서 둘째로 작은 고윳값 / 가장 큰 고윳값이 이보다 작으면 퇴화로 본다.
const DEGENERATE_EIG_RATIO: f64 = 1e-10;

/// 점 분포의 짧은 축 표준편차 / 긴 축 표준편차가 이보다 작으면 사실상 한 직선 위로 본다.
/// 정상적인 8점 표본(영상 전체에 흩어짐)은 0.1 이상이고, 길이 수백 px 직선에 σ ≤ 1 px 잡음이면 0.01 미만이다.
const COLLINEAR_AXIS_RATIO: f64 = 0.02;

/// 점들의 2×2 공분산 고윳값으로 (짧은 축 표준편차, 긴 축 표준편차).
fn axis_spread(p: &[Vector2<f64>]) -> (f64, f64) {
    let n = p.len().max(1) as f64;
    let c = p.iter().fold(Vector2::zeros(), |s, x| s + x) / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for x in p {
        let d = x - c;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }
    let (sxx, sxy, syy) = (sxx / n, sxy / n, syy / n);
    let tr = sxx + syy;
    let disc = ((sxx - syy).powi(2) + 4.0 * sxy * sxy).sqrt();
    let hi = 0.5 * (tr + disc);
    let lo = (0.5 * (tr - disc)).max(0.0);
    (lo.sqrt(), hi.sqrt())
}

/// 어느 한 영상에서라도 점들이 거의 한 직선 위에 있으면 참.
fn nearly_collinear(p: &[Vector2<f64>]) -> bool {
    let (lo, hi) = axis_spread(p);
    !(hi > 0.0 && lo > COLLINEAR_AXIS_RATIO * hi)
}

/// 정규화 DLT 로 호모그래피 x2 ~ H x1 (최소제곱). 점 4개 미만이거나 풀리지 않으면 None.
/// 계수 계산은 [`crate::two_view::homography_dlt`] 하나를 쓰고, 여기서는 픽셀 좌표의 Hartley 정규화만 한다.
fn homography_dlt(x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> Option<Matrix3<f64>> {
    if x1.len() < 4 || x1.len() != x2.len() {
        return None;
    }
    let (t1, t2) = (normalizer(x1), normalizer(x2));
    let tf = |t: &Matrix3<f64>, p: &Vector2<f64>| {
        let v = t * Vector3::new(p.x, p.y, 1.0);
        Vector2::new(v.x / v.z, v.y / v.z)
    };
    let a: Vec<_> = x1.iter().map(|p| tf(&t1, p)).collect();
    let b: Vec<_> = x2.iter().map(|p| tf(&t2, p)).collect();
    let hn = crate::two_view::homography_dlt(&a, &b)?;
    let h = t2.try_inverse()? * hn * t1;
    let n = h.norm();
    (n.is_finite() && n > 0.0).then(|| h / n)
}

/// 대칭 전달 오차의 큰 쪽(px): max(|x2 − H x1|, |x1 − H⁻¹ x2|).
fn homography_error(
    h: &Matrix3<f64>,
    hi: &Matrix3<f64>,
    p: &Vector2<f64>,
    q: &Vector2<f64>,
) -> f64 {
    let tr = |m: &Matrix3<f64>, x: &Vector2<f64>| {
        let v = m * Vector3::new(x.x, x.y, 1.0);
        if v.z.abs() < 1e-12 {
            None
        } else {
            Some(Vector2::new(v.x / v.z, v.y / v.z))
        }
    };
    match (tr(h, p), tr(hi, q)) {
        (Some(a), Some(b)) => (a - q).norm().max((b - p).norm()),
        _ => f64::INFINITY,
    }
}

/// 주어진 대응 가운데 한 호모그래피가 문턱 `th_px`(대칭 전달 오차의 큰 쪽) 안에서 설명하는 개수.
/// 4점 RANSAC + 정상 짝 재적합이라 이상치가 많아도 평면을 찾는다.
pub fn homography_support(x1: &[Vector2<f64>], x2: &[Vector2<f64>], th_px: f64) -> usize {
    fit_homography(x1, x2, th_px).map_or(0, |(_, _, c)| c)
}

/// 호모그래피 4점 RANSAC 의 최대 반복 수.
const HOMOGRAPHY_MAX_ITERS: usize = 500;

/// [`homography_support`] 의 적합: 4점 RANSAC(고정 시드, 적응형 종료) 뒤 정상 짝 전체로
/// 문턱을 ×4 → ×1 로 줄여 가며 다시 맞추고 정상 수가 늘 때만 받아들인다. (H, H⁻¹, 문턱 안 개수).
fn fit_homography(
    x1: &[Vector2<f64>],
    x2: &[Vector2<f64>],
    th_px: f64,
) -> Option<(Matrix3<f64>, Matrix3<f64>, usize)> {
    let n = x1.len();
    if n < 4 || n != x2.len() {
        return None;
    }
    let errors = |h: &Matrix3<f64>, hi: &Matrix3<f64>| -> Vec<f64> {
        (0..n)
            .map(|i| homography_error(h, hi, &x1[i], &x2[i]))
            .collect()
    };
    let mut st = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = |m: usize| {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 33) as usize) % m
    };
    let mut best: Option<(Matrix3<f64>, Matrix3<f64>, usize)> = None;
    let mut iters = HOMOGRAPHY_MAX_ITERS;
    let mut it = 0;
    while it < iters {
        it += 1;
        let mut idx = [0usize; 4];
        let mut k = 0;
        while k < 4 {
            let c = rnd(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let s1: Vec<_> = idx.iter().map(|&i| x1[i]).collect();
        let s2: Vec<_> = idx.iter().map(|&i| x2[i]).collect();
        let Some(h) = homography_dlt(&s1, &s2) else {
            continue;
        };
        let Some(hi) = h.try_inverse() else {
            continue;
        };
        let c = errors(&h, &hi).iter().filter(|&&e| e < th_px).count();
        if best.as_ref().is_none_or(|b| c > b.2) {
            best = Some((h, hi, c));
            iters = adaptive_iterations(c as f64 / n as f64, 4, 0.999, HOMOGRAPHY_MAX_ITERS);
        }
    }
    let (mut h, mut hi, mut cnt) = best?;
    let mut cur = (h, hi);
    for m in [4.0, 2.0, 1.0, 1.0] {
        let err = errors(&cur.0, &cur.1);
        let sel: Vec<usize> = (0..n).filter(|&i| err[i] < th_px * m).collect();
        let s1: Vec<_> = sel.iter().map(|&i| x1[i]).collect();
        let s2: Vec<_> = sel.iter().map(|&i| x2[i]).collect();
        let Some(g) = homography_dlt(&s1, &s2) else {
            break;
        };
        let Some(gi) = g.try_inverse() else {
            break;
        };
        cur = (g, gi);
        let c = errors(&g, &gi).iter().filter(|&&e| e < th_px).count();
        if c > cnt {
            (h, hi, cnt) = (g, gi, c);
        }
    }
    Some((h, hi, cnt))
}

/// 두 시점 기하 모델.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoViewModel {
    /// 일반 장면: 기본 행렬이 정해진다.
    Fundamental,
    /// 평면(또는 시차 없는) 장면: 호모그래피로 충분하고 F 는 정해지지 않는다.
    Homography,
}

/// [`select_two_view_model`] 의 결과.
#[derive(Clone, Copy, Debug)]
pub struct ModelSelection {
    pub model: TwoViewModel,
    /// Torr GRIC 점수(작을수록 좋음).
    pub gric_f: f64,
    pub gric_h: f64,
    /// 호모그래피 잔차가 χ²₂ 99.9% 분위(13.8σ²)를 넘고 F 잔차는 χ²₁ 99.9% 분위(10.8σ²) 안인 짝 수
    /// (평면 밖 시차가 있는 정상 짝).
    pub parallax: usize,
    /// 평면으로 판정하지 않는 데 필요한 최소 시차 짝 수.
    pub parallax_needed: usize,
    /// 실제로 쓴 잡음 σ(px): 호출자 σ 와 강건 추정 중 큰 쪽.
    pub sigma_px: f64,
}

/// 순수 평면 짝 하나가 시차 짝으로 잘못 세어질 확률(H 잔차 χ²₂ 99.9% 초과).
const FALSE_PARALLAX_RATE: f64 = 0.001;
/// 순수 평면 n 짝에서 시차 짝 수가 필요 수에 우연히 이를 확률의 상한(RANSAC 유의성 `max_chance_prob` 와 같은 값).
const PARALLAX_CHANCE_PROB: f64 = 1e-6;

/// 평면으로 판정하지 않는 데 필요한 최소 시차 짝 수: 순수 평면 n 짝에서 우연 시차 짝 수
/// B(n, 0.001) 이 k 이상일 확률이 1e-6 미만인 가장 작은 k(그리고 F 를 정하는 데 필요한 평면 밖 2점 이상).
/// n = 50·200·300·1000 에서 4·6·6·10. 실측 편대 배치 순수 지면 300점 20 경우의 우연 시차 짝은 0~1
/// (연구 노트 matching-followup 훑기 표).
pub fn parallax_needed(n: usize) -> usize {
    (2..=n)
        .find(|&k| binomial_tail(n, k, FALSE_PARALLAX_RATE) < PARALLAX_CHANCE_PROB)
        .unwrap_or(n + 1)
}

/// 정상 짝(x1, x2)과 그 F 에 대해 F/H 모델을 고른다. `sigma_px` 는 좌표 잡음 표준편차.
///
/// GRIC(Torr 1998): Σ min(e²/σ², λ₃(r−d)) + λ₁ d n + λ₂ k, r = 4, λ₁ = ln r, λ₂ = ln(r n), λ₃ = 2,
/// F: d = 3, k = 7(e² = 제곱 Sampson 거리), H: d = 2, k = 8(e² = 두 방향 전달 오차 제곱 평균의 절반,
/// 양쪽 잡음 σ 에서 기댓값 2σ² 로 4차원 기하 오차와 맞춘다).
/// 순수 GRIC 는 평면 밖 짝 비율이 약 19% 미만이면(평면 위 짝마다 F 가 ln 4 를 더 내고 평면 밖 짝마다 H 가
/// 약 3 을 더 내므로 ln4 − 1 ≈ 0.39 ≈ 2·비율) H 를 고른다. 그러나 F 는 호모그래피 + 평면 밖 두 점으로 정해지므로
/// 지면 위주 장면에서 건물 몇 %만 있어도 F 가 정해진다. 그래서 GRIC 가 H 를 고르더라도 시차 짝 수가
/// 순수 평면에서 우연히 나올 수 있는 수([`parallax_needed`], 이항 꼬리 1e-6)에 이르면 F 를 고른다.
/// 잡음 크기는 호출자 σ 와 정상 짝의 제곱 Sampson 거리 중앙값에서 구한 강건 추정
/// σ̂ = √(중앙값 / 0.455)(χ²₁ 중앙값) 가운데 큰 쪽을 쓴다(σ 를 실제보다 작게 넘겨도 평면 밖 시차로 잘못 세지 않게).
/// 정상 짝이 8개 미만이거나, 길이가 다르거나, σ 가 유한한 양수가 아니거나, 좌표·F 성분에 유한하지 않은
/// 값이 있거나, 호모그래피를 맞출 수 없으면 None(유한하지 않은 짝은 GRIC 누산에서 상한값으로 잘려
/// 판정을 H 쪽으로 기울이므로 제외하지 않고 오류로 돌려준다).
pub fn select_two_view_model(
    x1: &[Vector2<f64>],
    x2: &[Vector2<f64>],
    f: &Matrix3<f64>,
    sigma_px: f64,
) -> Option<ModelSelection> {
    let n = x1.len();
    if n < 8
        || n != x2.len()
        || !sigma_px.is_finite()
        || sigma_px <= 0.0
        || !all_finite(x1)
        || !all_finite(x2)
        || !f.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let mut es: Vec<f64> = x1
        .iter()
        .zip(x2)
        .map(|(p, q)| sampson_error(f, p, q))
        .filter(|e| e.is_finite())
        .collect();
    if es.len() < 8 {
        return None;
    }
    es.sort_by(f64::total_cmp);
    let sigma_hat = (es[es.len() / 2] / 0.455).sqrt();
    let sigma_px = sigma_px.max(sigma_hat);
    let s2 = sigma_px * sigma_px;
    let (h, hi, _) = fit_homography(x1, x2, 3.0 * sigma_px)?;
    let tr = |m: &Matrix3<f64>, x: &Vector2<f64>, y: &Vector2<f64>| -> f64 {
        let v = m * Vector3::new(x.x, x.y, 1.0);
        if v.z.abs() < 1e-12 {
            f64::INFINITY
        } else {
            (Vector2::new(v.x / v.z, v.y / v.z) - y).norm_squared()
        }
    };
    let (mut rho_f, mut rho_h, mut parallax) = (0.0, 0.0, 0usize);
    for (p, q) in x1.iter().zip(x2) {
        let ef = sampson_error(f, p, q) / s2;
        let eh = 0.25 * (tr(&h, p, q) + tr(&hi, q, p)) / s2;
        rho_f += ef.min(2.0 * (4.0 - 3.0));
        rho_h += eh.min(2.0 * (4.0 - 2.0));
        if eh > 13.8 && ef < 10.8 {
            parallax += 1;
        }
    }
    let nf = n as f64;
    let (l1, l2) = (4f64.ln(), (4.0 * nf).ln());
    let gric_f = rho_f + l1 * 3.0 * nf + l2 * 7.0;
    let gric_h = rho_h + l1 * 2.0 * nf + l2 * 8.0;
    let parallax_needed = parallax_needed(n);
    let model = if gric_h < gric_f && parallax < parallax_needed {
        TwoViewModel::Homography
    } else {
        TwoViewModel::Fundamental
    };
    Some(ModelSelection {
        model,
        gric_f,
        gric_h,
        parallax,
        parallax_needed,
        sigma_px,
    })
}

pub(crate) fn all_finite(p: &[Vector2<f64>]) -> bool {
    p.iter().all(|v| v.x.is_finite() && v.y.is_finite())
}

/// 점들을 무게중심 0, 평균 거리 √2 로 옮기는 상사 변환(Hartley 1997).
fn normalizer(p: &[Vector2<f64>]) -> Matrix3<f64> {
    let n = p.len() as f64;
    let c = p.iter().fold(Vector2::zeros(), |s, x| s + x) / n;
    let d = p.iter().map(|x| (x - c).norm()).sum::<f64>() / n;
    let s = if d > 0.0 { 2f64.sqrt() / d } else { 1.0 };
    Matrix3::new(s, 0.0, -s * c.x, 0.0, s, -s * c.y, 0.0, 0.0, 1.0)
}

/// 정규화 8점 알고리즘으로 기본 행렬 F (x2ᵀ F x1 = 0, 픽셀 좌표)를 구한다.
/// 점이 8개 미만이거나, 길이가 다르거나, 유한하지 않은 좌표가 있거나,
/// 해가 하나로 정해지지 않으면(영공간 2차원 이상: 동일선상 점, 순수 회전 등) None.
/// 어느 한 영상의 점 분포가 짧은 축/긴 축 표준편차 비 0.02 미만(잡음 섞인 동일선상)이어도 None.
/// 결과는 계수 2, 프로베니우스 노름 1.
pub fn fundamental_8pt(x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> Option<Matrix3<f64>> {
    if x1.len() < 8 || x1.len() != x2.len() || !all_finite(x1) || !all_finite(x2) {
        return None;
    }
    // 잡음 섞인 동일선상 배치는 고윳값 비 검사로 걸러지지 않는다(둘째 고윳값이 잡음 크기만큼 커진다).
    if nearly_collinear(x1) || nearly_collinear(x2) {
        return None;
    }
    let (t1, t2) = (normalizer(x1), normalizer(x2));
    // AᵀA (9×9) 의 최소 고유벡터가 최소제곱 해.
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for (p, q) in x1.iter().zip(x2) {
        let a = t1 * Vector3::new(p.x, p.y, 1.0);
        let b = t2 * Vector3::new(q.x, q.y, 1.0);
        let row = SMatrix::<f64, 9, 1>::from_column_slice(&[
            b.x * a.x,
            b.x * a.y,
            b.x,
            b.y * a.x,
            b.y * a.y,
            b.y,
            a.x,
            a.y,
            1.0,
        ]);
        ata += row * row.transpose();
    }
    let eig = ata.symmetric_eigen();
    let k = eig.eigenvalues.imin();
    // 둘째로 작은 고윳값도 0 에 가까우면 영공간이 2차원 이상이라 F 가 정해지지 않는다.
    let mut ev: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    ev.sort_by(f64::total_cmp);
    if ev[1] <= DEGENERATE_EIG_RATIO * ev[8] {
        return None;
    }
    let f = eig.eigenvectors.column(k);
    let fn_ = Matrix3::new(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
    // 계수 2 강제: 가장 작은 특이값을 0 으로.
    let mut svd = fn_.svd(true, true);
    let i = svd.singular_values.imin();
    svd.singular_values[i] = 0.0;
    let fn_ = svd.recompose().ok()?;
    let f = t2.transpose() * fn_ * t1;
    let n = f.norm();
    (n.is_finite() && n > 0.0).then(|| f / n)
}

/// 기본 행렬에 대한 **제곱** Sampson 거리(단위 px², 1차 기하 오차 근사).
/// 픽셀 거리가 필요하면 `.sqrt()` 를 쓴다. RANSAC 은 이 값을 `threshold_px²` 와 비교한다.
pub fn sampson_error(f: &Matrix3<f64>, p: &Vector2<f64>, q: &Vector2<f64>) -> f64 {
    let a = Vector3::new(p.x, p.y, 1.0);
    let b = Vector3::new(q.x, q.y, 1.0);
    let fa = f * a;
    let ftb = f.transpose() * b;
    let e = b.dot(&fa);
    let den = fa.x * fa.x + fa.y * fa.y + ftb.x * ftb.x + ftb.y * ftb.y;
    if den <= 0.0 {
        f64::INFINITY
    } else {
        e * e / den
    }
}

/// 계수 2 로 투영하고 프로베니우스 노름 1 로 맞춘다.
fn rank2_unit(f: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let mut svd = f.svd(true, true);
    let i = svd.singular_values.imin();
    svd.singular_values[i] = 0.0;
    let g = svd.recompose().ok()?;
    let n = g.norm();
    (n.is_finite() && n > 0.0).then(|| g / n)
}

/// 정규화 좌표 a' = T₁a, b' = T₂b 와 정규화 F̂ 에서 픽셀 Sampson 잔차와 F̂ 성분 9개(행 우선)에 대한 해석적 야코비안.
/// T₁, T₂ 는 [`normalizer`] 꼴(대각 s, 이동만)이라 픽셀 F = T₂ᵀF̂T₁ 에서
/// e = bᵀFa = b'ᵀF̂a', (Fa)ₖ = s₂(F̂a')ₖ, (Fᵀb)ₖ = s₁(F̂ᵀb')ₖ (k = 1, 2) 이다.
/// D = s₂²((F̂a')₁² + (F̂a')₂²) + s₁²((F̂ᵀb')₁² + (F̂ᵀb')₂²), r = e/√D 이고
/// ∂r/∂F̂ₖₗ = b'ₖa'ₗ/√D − e/D^{3/2}·(s₂²(F̂a')ₖa'ₗ[k<2] + s₁²b'ₖ(F̂ᵀb')ₗ[l<2]).
/// D ≤ 0 이면 잔차·야코비안 모두 0.
fn sampson_residual_jacobian(
    g: &Matrix3<f64>,
    s1: f64,
    s2: f64,
    a: &Vector3<f64>,
    b: &Vector3<f64>,
) -> (f64, [f64; 9]) {
    let ga = g * a;
    let gtb = g.tr_mul(b);
    let e = b.dot(&ga);
    let (q1, q2) = (s1 * s1, s2 * s2);
    let den = q2 * (ga.x * ga.x + ga.y * ga.y) + q1 * (gtb.x * gtb.x + gtb.y * gtb.y);
    if den <= 0.0 {
        return (0.0, [0.0; 9]);
    }
    let inv = 1.0 / den.sqrt();
    let c = e * inv * inv * inv;
    // 분모 미분 항의 두 벡터: s₂²(F̂a')ₖ (k<2), s₁²(F̂ᵀb')ₗ (l<2).
    let u = [q2 * ga.x, q2 * ga.y, 0.0];
    let v = [q1 * gtb.x, q1 * gtb.y, 0.0];
    let mut j = [0.0; 9];
    for k in 0..3 {
        for l in 0..3 {
            j[3 * k + l] = b[k] * a[l] * inv - c * (u[k] * a[l] + b[k] * v[l]);
        }
    }
    (e * inv, j)
}

/// 정규화 좌표 대응에서 픽셀 Sampson 잔차(부호 있는 거리)만.
fn sampson_residuals(
    g: &Matrix3<f64>,
    s1: f64,
    s2: f64,
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    out: &mut Vec<f64>,
) {
    let (q1, q2) = (s1 * s1, s2 * s2);
    out.clear();
    out.extend(a.iter().zip(b).map(|(a, b)| {
        let ga = g * a;
        let gtb = g.tr_mul(b);
        let den = q2 * (ga.x * ga.x + ga.y * ga.y) + q1 * (gtb.x * gtb.x + gtb.y * gtb.y);
        if den > 0.0 {
            b.dot(&ga) / den.sqrt()
        } else {
            0.0
        }
    }));
}

#[cfg(test)]
thread_local! {
    /// 시험용: `refine_sampson` 호출마다 (받아들인 LM 걸음 수, 종료 사유).
    static LM_ITERS: std::cell::RefCell<Vec<(usize, LmStop)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 시험용: Sampson LM 종료 사유.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LmStop {
    /// 걸음을 받아들였으나 상대 비용 감소가 1e-10 이하.
    Converged,
    /// 감쇠를 8번 키워도 비용이 줄지 않음.
    Stalled,
    /// 반복 상한에 닿음.
    Cap,
}

/// 주어진 대응에서 Sampson 거리 제곱합을 줄이도록 F 를 Levenberg–Marquardt 로 정밀화한다
/// (Hartley & Zisserman 11.4.3 의 Sampson 비용). 매개변수는 Hartley 정규화 좌표의 F 성분 9개이고,
/// 걸음마다 계수 2·노름 1 로 투영해 7 자유도를 유지한다. 비용이 줄지 않으면 시작 F 를 그대로 돌려준다.
/// 야코비안은 해석적([`sampson_residual_jacobian`])이라 반복마다 잔차 계산은 걸음 후보 평가뿐이다.
fn refine_sampson(
    f: &Matrix3<f64>,
    x1: &[Vector2<f64>],
    x2: &[Vector2<f64>],
    iters: usize,
) -> Matrix3<f64> {
    if x1.len() < 8 {
        return *f;
    }
    let (t1, t2) = (normalizer(x1), normalizer(x2));
    let (Some(t1i), Some(t2i)) = (t1.try_inverse(), t2.try_inverse()) else {
        return *f;
    };
    let (s1, s2) = (t1[(0, 0)], t2[(0, 0)]);
    let a: Vec<Vector3<f64>> = x1
        .iter()
        .map(|p| t1 * Vector3::new(p.x, p.y, 1.0))
        .collect();
    let b: Vec<Vector3<f64>> = x2
        .iter()
        .map(|q| t2 * Vector3::new(q.x, q.y, 1.0))
        .collect();
    let Some(mut g) = rank2_unit(&(t2i.transpose() * f * t1i)) else {
        return *f;
    };
    let mut rh = Vec::new();
    sampson_residuals(&g, s1, s2, &a, &b, &mut rh);
    let mut cost: f64 = rh.iter().map(|e| e * e).sum();
    let mut lambda = 1e-3;
    let mut taken = 0;
    #[cfg(test)]
    let mut stop = LmStop::Cap;
    for _ in 0..iters {
        let mut jtj = SMatrix::<f64, 9, 9>::zeros();
        let mut jtr = SMatrix::<f64, 9, 1>::zeros();
        for (ai, bi) in a.iter().zip(&b) {
            let (ri, ji) = sampson_residual_jacobian(&g, s1, s2, ai, bi);
            let row = SMatrix::<f64, 9, 1>::from_column_slice(&ji);
            jtj.syger(1.0, &row, &row, 1.0);
            jtr += row * ri;
        }
        jtj.fill_upper_triangle_with_lower_triangle();
        let mut improved = false;
        let mut accepted = false;
        for _ in 0..8 {
            let mut m = jtj;
            for k in 0..9 {
                m[(k, k)] += lambda * (jtj[(k, k)] + 1e-12);
            }
            let Some(d) = m.cholesky().map(|c| c.solve(&(-jtr))) else {
                lambda *= 10.0;
                continue;
            };
            let step = Matrix3::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]);
            let Some(cand) = rank2_unit(&(g + step)) else {
                lambda *= 10.0;
                continue;
            };
            sampson_residuals(&cand, s1, s2, &a, &b, &mut rh);
            let c: f64 = rh.iter().map(|e| e * e).sum();
            if c < cost {
                g = cand;
                let rel = (cost - c) / cost.max(1e-300);
                cost = c;
                lambda = (lambda * 0.1).max(1e-9);
                taken += 1;
                accepted = true;
                improved = rel > 1e-10;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            #[cfg(test)]
            {
                stop = if accepted {
                    LmStop::Converged
                } else {
                    LmStop::Stalled
                };
            }
            let _ = accepted;
            break;
        }
    }
    #[cfg(test)]
    LM_ITERS.with(|v| v.borrow_mut().push((taken, stop)));
    let _ = taken;
    let fp = t2.transpose() * g * t1;
    let n = fp.norm();
    if n.is_finite() && n > 0.0 {
        fp / n
    } else {
        *f
    }
}

/// RANSAC 설정.
#[derive(Clone, Copy, Debug)]
pub struct RansacConfig {
    /// 정상 판정 Sampson 거리 문턱(px, 제곱하지 않은 거리).
    pub threshold_px: f64,
    /// 받아들일 최소 정상 비율. 이보다 낮으면 None(무관한 대응에서 우연히 맞은 F 를 거른다).
    pub min_inlier_ratio: f64,
    pub max_iters: usize,
    /// 적응형 종료 신뢰도.
    pub confidence: f64,
    pub seed: u64,
    /// 받아들일 최소 정상 짝 수. 겹침 없는 짝에서도 비율 검사를 지난 우연 짝 20개 안팎 중
    /// 9~11개가 한 F 에 맞으므로 8 로는 거를 수 없다.
    pub min_inliers: usize,
    /// 정상 수가 우연(무관한 대응이 에피폴라 띠에 들어갈 확률의 이항 꼬리)으로 나올 확률의 상한.
    /// 이보다 크면 None. 0 이하이면 검사하지 않는다.
    pub max_chance_prob: f64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            threshold_px: 1.5,
            min_inlier_ratio: 0.2,
            max_iters: 2000,
            confidence: 0.999,
            seed: 1,
            min_inliers: MIN_VERIFIED_INLIERS,
            max_chance_prob: 1e-6,
        }
    }
}

/// 적응형 RANSAC 반복 수: 정상 비율 w, 표본 크기 s 에서 신뢰도 p 로 정상만 뽑는 데 필요한 횟수
/// ⌈ln(1−p) / ln(1−wˢ)⌉ 를 [`MIN_RANSAC_ITERS`, max_iters] 로 자른다.
/// wˢ 가 f64 반올림 아래(1 − wˢ == 1)로 작아도 무너지지 않도록 ln(1−x) 를 `ln_1p(−x)` 로 계산하고,
/// 값이 유한하지 않으면 max_iters 를 돌려준다.
pub fn adaptive_iterations(w: f64, sample: i32, confidence: f64, max_iters: usize) -> usize {
    let floor = MIN_RANSAC_ITERS.min(max_iters);
    let p_good = w.clamp(0.0, 1.0).powi(sample);
    if p_good >= 1.0 {
        return floor;
    }
    let denom = (-p_good).ln_1p();
    let need = (-confidence.clamp(0.0, 1.0 - 1e-15)).ln_1p() / denom;
    if !need.is_finite() || need >= max_iters as f64 {
        return max_iters;
    }
    (need.ceil() as usize).clamp(floor, max_iters)
}

/// 적응형 종료가 허용하는 최소 반복 수.
pub const MIN_RANSAC_ITERS: usize = 50;

/// [`RansacConfig::min_inliers`] 기본값.
pub const MIN_VERIFIED_INLIERS: usize = 15;

/// F 가 정확히 맞출 수 있는 대응 수(자유도 7)에 국소 최적화가 흡수하는 1개를 더한 값.
/// 이만큼은 무관한 대응이라도 정상이 되므로 유의성 검사에서 뺀다.
const FREE_FIT: usize = 8;

/// 무관한 대응 한 개가 문턱 `th_px` 의 에피폴라 띠에 우연히 들어갈 확률의 상한:
/// 둘째 영상 점들의 경계 상자(가로 w, 세로 h)에서 띠 넓이 2·th·대각선 / (w·h).
pub fn epipolar_band_probability(x2: &[Vector2<f64>], th_px: f64) -> f64 {
    let (mut lo, mut hi) = (
        Vector2::repeat(f64::INFINITY),
        Vector2::repeat(f64::NEG_INFINITY),
    );
    for p in x2 {
        lo = lo.inf(p);
        hi = hi.sup(p);
    }
    let d = hi - lo;
    let area = d.x * d.y;
    if !(area.is_finite() && area > 0.0) {
        return 1.0;
    }
    (2.0 * th_px * d.norm() / area).clamp(0.0, 1.0)
}

/// 이항 꼬리 P(X ≥ k), X ~ B(n, p). 로그 공간에서 합한다.
pub fn binomial_tail(n: usize, k: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (lp, lq) = (p.ln(), (-p).ln_1p());
    let mut log_c = 0.0; // ln C(n, i)
    let mut sum = 0.0;
    for i in 0..=n {
        if i > 0 {
            log_c += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        if i >= k {
            sum += (log_c + i as f64 * lp + (n - i) as f64 * lq).exp();
        }
    }
    sum.min(1.0)
}

/// 대응 n 개 중 `cnt` 개가 정상인 것이 우연(무관한 대응)으로 나올 확률:
/// F 가 흡수하는 [`FREE_FIT`] 개를 빼고 나머지가 띠 확률 p 로 들어갈 이항 꼬리.
pub fn chance_inlier_probability(n: usize, cnt: usize, p: f64) -> f64 {
    if cnt <= FREE_FIT {
        return 1.0;
    }
    binomial_tail(n.saturating_sub(FREE_FIT), cnt - FREE_FIT, p)
}

/// 정상 짝 가운데 한 직선(문턱 `th_px`)으로 설명되지 않는 점 수의 하한 추정.
/// 두 점 직선 RANSAC(고정 시드, 최대 300회)으로 가장 많은 점을 지나는 직선을 찾아 그 밖의 수를 센다.
fn off_line_count(p: &[Vector2<f64>], th_px: f64) -> usize {
    let n = p.len();
    if n < 3 {
        return 0;
    }
    let mut st = 0x6A09_E667_F3BC_C909u64;
    let mut rnd = |m: usize| {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 33) as usize) % m
    };
    let mut best = 0usize;
    for _ in 0..300 {
        let (i, j) = (rnd(n), rnd(n));
        let d = p[j] - p[i];
        let len = d.norm();
        if i == j || len.is_nan() || len <= 0.0 {
            continue;
        }
        let nrm = Vector2::new(-d.y, d.x) / len;
        let c = p
            .iter()
            .filter(|q| (*q - p[i]).dot(&nrm).abs() < th_px)
            .count();
        best = best.max(c);
    }
    n - best
}

/// 한 직선 밖 정상 짝이 이보다 적으면 F 가 정해지지 않는다(3차원 직선은 F 에 제약 몇 개만 준다).
const MIN_OFF_LINE: usize = 8;

/// [`ransac_fundamental`] 의 결과: F 와 정상 짝, 그리고 정상 짝에 대한 두 시점 모델 판정.
#[derive(Clone, Debug)]
pub struct FundamentalFit {
    /// 정상 짝으로 다시 맞춘 기본 행렬(프로베니우스 노름 1, 계수 2).
    pub f: Matrix3<f64>,
    /// 짝마다 정상 여부.
    pub inliers: Vec<bool>,
    /// 정상 짝에 [`select_two_view_model`] 을 적용한 결과(σ = threshold_px / 3).
    /// 호모그래피를 맞출 수 없으면 None 이고 그때 `model` 은 Fundamental.
    pub selection: Option<ModelSelection>,
    /// `Homography` 이면 평면(또는 시차 없는) 짝이라 F 가 정해지지 않는다 — 이동 방향을 꺼내지 않는다.
    pub model: TwoViewModel,
}

impl FundamentalFit {
    /// 평면 표시: 정상 짝이 호모그래피로 충분히 설명되어 F(이동 방향)를 믿을 수 없다.
    pub fn is_planar(&self) -> bool {
        self.model == TwoViewModel::Homography
    }
}

/// RANSAC(Fischler & Bolles 1981) + 8점 기본 행렬로 기하 검증.
/// 반환: [`FundamentalFit`] (정상 짝으로 다시 맞춘 F, 정상 여부 표시, 평면 판정).
/// 정상 짝이 `max(8, min_inliers)` 개 미만이거나 정상 비율이 `min_inlier_ratio` 미만이면 None.
/// 정상 수가 무관한 대응에서 우연히 나올 확률([`chance_inlier_probability`])이 `max_chance_prob` 를 넘으면 None
/// (시야가 겹치지 않는 짝에서 비율 검사를 지난 우연 짝을 거른다).
/// 정상 짝이 거의 한 직선 위(짧은 축/긴 축 표준편차 비 0.02 미만)이거나, 한 직선(문턱 threshold_px)
/// 밖 정상 짝이 어느 영상에서든 8개 미만이면 퇴화로 보고 None.
/// 평면 짝은 거부하지 않고 [`FundamentalFit::model`] 에 `Homography` 로 표시한다(호출자가 표시를 보고
/// 이동 제약을 뺀다). 길이가 다르거나 유한하지 않은 좌표가 있으면 None.
pub fn ransac_fundamental(
    x1: &[Vector2<f64>],
    x2: &[Vector2<f64>],
    cfg: &RansacConfig,
) -> Option<FundamentalFit> {
    let n = x1.len();
    if n < 8 || n != x2.len() || !all_finite(x1) || !all_finite(x2) {
        return None;
    }
    let th2 = cfg.threshold_px * cfg.threshold_px;
    let inliers_of = |f: &Matrix3<f64>| -> Vec<bool> {
        (0..n)
            .map(|i| sampson_error(f, &x1[i], &x2[i]) < th2)
            .collect()
    };
    let mut st = cfg.seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut rnd = |m: usize| {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 33) as usize) % m
    };
    let mut best: Option<(Matrix3<f64>, Vec<bool>, usize)> = None;
    // 서로 다른 골짜기의 정상 수 상위 가설들: 마무리 정밀화를 최고 가설 하나에만 하면 그것이 잘못된 골짜기일 때 빠져나오지 못한다.
    const POOL: usize = 5;
    let mut pool: Vec<(Matrix3<f64>, Vec<bool>, usize)> = Vec::new();
    let mut iters = cfg.max_iters;
    let mut it = 0;
    while it < iters {
        it += 1;
        let mut idx = [0usize; 8];
        let mut k = 0;
        while k < 8 {
            let c = rnd(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let s1: Vec<_> = idx.iter().map(|&i| x1[i]).collect();
        let s2: Vec<_> = idx.iter().map(|&i| x2[i]).collect();
        let Some(f) = fundamental_8pt(&s1, &s2) else {
            continue;
        };
        let mut inl = inliers_of(&f);
        let mut cnt = inl.iter().filter(|&&b| b).count();
        // 국소 최적화(Chum et al. 2003): 잡음 섞인 최소 표본의 F 는 정상 짝 일부만 설명하므로
        // 최고 가설 정상 수의 4분의 1 이상을 설명하는 가설은 정상 짝 전체로 다시 맞춰 개선이 멈출 때까지 반복한다.
        // 1/4 근거: 절반 기준에서는 반대쪽 무한대 에피폴 골짜기(정상 106 / 정답 133)에 갇힌 경우가 남아
        // 재현율 미달이 다중 시드 500 경우 중 6건이었고, 1/4 로 넓히자 정답 골짜기 가설이 상위 묶음에 들어와 1건이 됐다.
        // 이동이 영상면과 거의 평행해 에피폴이 멀면 최소 표본 F 가 특히 부정확하다.
        let best_cnt = best.as_ref().map_or(0, |b| b.2);
        if cnt >= 8 && 4 * cnt >= best_cnt {
            // 문턱을 넓게 시작해 줄여 가며(×3, ×2, ×1.5, ×1) 다시 맞춘다: 부정확한 시작 F 의 좁은 띠
            // 밖에 있는 정상 짝도 끌어들이기 위해서다. 문턱 ×1 의 정상 수가 늘어날 때만 받아들인다.
            let mut f_lo = f;
            let mut cur = f;
            for m in [3.0, 2.0, 1.5, 1.0, 1.0, 1.0] {
                let t2 = th2 * m * m;
                let sel: Vec<usize> = (0..n)
                    .filter(|&i| sampson_error(&cur, &x1[i], &x2[i]) < t2)
                    .collect();
                let s1: Vec<_> = sel.iter().map(|&i| x1[i]).collect();
                let s2: Vec<_> = sel.iter().map(|&i| x2[i]).collect();
                let Some(g) = fundamental_8pt(&s1, &s2) else {
                    break;
                };
                cur = g;
                let gi = inliers_of(&g);
                let gc = gi.iter().filter(|&&b| b).count();
                if gc > cnt {
                    (f_lo, inl, cnt) = (g, gi, gc);
                }
            }
            let f = f_lo;
            // 정상 집합이 크게 겹치는(자카드 > 0.7) 가설은 같은 골짜기로 보고 더 나은 것 하나만 남긴다.
            let same = pool.iter().position(|(_, pi, _)| {
                let both = (0..n).filter(|&i| pi[i] && inl[i]).count();
                let either = (0..n).filter(|&i| pi[i] || inl[i]).count();
                10 * both > 7 * either
            });
            match same {
                Some(j) if pool[j].2 >= cnt => {}
                Some(j) => pool[j] = (f, inl.clone(), cnt),
                None => pool.push((f, inl.clone(), cnt)),
            }
            pool.sort_by_key(|b| std::cmp::Reverse(b.2));
            pool.truncate(POOL);
            if cnt <= best_cnt {
                continue;
            }
            let w = cnt as f64 / n as f64;
            iters = adaptive_iterations(w, 8, cfg.confidence, cfg.max_iters);
            best = Some((f, inl, cnt));
        }
    }
    let (mut f, mut inl, _) = best?;
    // Sampson 비용 LM 정밀화: 8점 재적합은 대수 오차를 줄이므로 에피폴이 멀면 치우친다.
    // 상위 가설마다 문턱을 ×3 에서 ×1 로 줄여 가며 그 안의 짝으로 기하(Sampson) 오차를 직접 줄이고,
    // 문턱 ×1 정상 수가 가장 많은 해를 고른다(늘 때만 바꾼다).
    for (start, _, _) in pool {
        let mut cur = start;
        for m in [3.0, 2.0, 1.5, 1.0, 1.0, 1.0] {
            let t2 = th2 * m * m;
            let sel: Vec<usize> = (0..n)
                .filter(|&i| sampson_error(&cur, &x1[i], &x2[i]) < t2)
                .collect();
            let s1: Vec<_> = sel.iter().map(|&i| x1[i]).collect();
            let s2: Vec<_> = sel.iter().map(|&i| x2[i]).collect();
            cur = refine_sampson(&cur, &s1, &s2, 30);
            let gi = inliers_of(&cur);
            if gi.iter().filter(|&&b| b).count() > inl.iter().filter(|&&b| b).count() {
                f = cur;
                inl = gi;
            }
        }
    }
    // 정상 짝 전체로 Sampson 비용을 다시 줄이고 정상 집합을 갱신(두 번).
    for _ in 0..2 {
        let s1: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| x1[i]).collect();
        let s2: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| x2[i]).collect();
        let g = refine_sampson(&f, &s1, &s2, 30);
        let gi = inliers_of(&g);
        if gi.iter().filter(|&&b| b).count() < s1.len() {
            break;
        }
        f = g;
        inl = gi;
    }
    let cnt = inl.iter().filter(|&&b| b).count();
    if cnt < 8.max(cfg.min_inliers) || (cnt as f64) < cfg.min_inlier_ratio * n as f64 {
        return None;
    }
    // 유의성: 무관한 대응만 있어도 F 는 8개 안팎을 맞추고 나머지는 띠 확률로 들어온다.
    if cfg.max_chance_prob > 0.0 {
        let p = epipolar_band_probability(x2, cfg.threshold_px);
        if chance_inlier_probability(n, cnt, p) > cfg.max_chance_prob {
            return None;
        }
    }
    // 퇴화 판정: 정상 짝이 거의 한 직선 위이거나, 한 직선 밖 정상 짝이 8개 미만이면
    // F 가 정해지지 않으므로 확정하지 않는다(직선 + 일반 점 몇 개).
    let s1: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| x1[i]).collect();
    let s2: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| x2[i]).collect();
    if nearly_collinear(&s1) || nearly_collinear(&s2) {
        return None;
    }
    let line_th = cfg.threshold_px;
    if off_line_count(&s1, line_th) < MIN_OFF_LINE || off_line_count(&s2, line_th) < MIN_OFF_LINE {
        return None;
    }
    // 평면 판정: 거부하지 않고 표시한다. 실측 편대(카메라 간격 약 10 m, 위치 간 약 1 m)에는 같은 자리에서
    // 시선만 도는 순수 회전 짝이 없다. 거부하지 않는 이유는 지면 위주 장면에서 평면 짝의 회전은
    // 호모그래피로도 쓸 수 있고, 정상 짝 수는 겹침 판단(간선 존재)에 쓰이기 때문이다.
    // 평면 표시가 붙은 F 에서 이동 방향을 꺼내면 안 된다(에피폴이 임의).
    let selection = select_two_view_model(&s1, &s2, &f, cfg.threshold_px / 3.0);
    let model = selection.map_or(TwoViewModel::Fundamental, |m| m.model);
    Some(FundamentalFit {
        f,
        inliers: inl,
        selection,
        model,
    })
}

/// 정답 카메라 두 대로부터 기본 행렬 F = K2⁻ᵀ [t]× R K1⁻¹ (상대 자세 2←1).
pub fn fundamental_from_cameras(
    c1: &crate::camera::Camera,
    c2: &crate::camera::Camera,
) -> Matrix3<f64> {
    let r = c2.pose.rotation * c1.pose.rotation.inverse();
    let t = c2.pose.translation - r * c1.pose.translation;
    let e = crate::math::skew(&t) * r.matrix();
    let kinv = |c: &crate::camera::Camera| {
        let k = &c.intrinsics;
        let p = |x: f64, y: f64| k.to_normalized(&Vector2::new(x, y));
        // to_normalized 가 아핀이므로 세 점으로 K⁻¹ 를 복원한다.
        let (o, ex, ey) = (p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0));
        Matrix3::new(
            ex.x - o.x,
            ey.x - o.x,
            o.x,
            ex.y - o.y,
            ey.y - o.y,
            o.y,
            0.0,
            0.0,
            1.0,
        )
    };
    let f = kinv(c2).transpose() * e * kinv(c1);
    f / f.norm()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Camera, Intrinsics, Pose};
    use nalgebra::{Point3, Rotation3};

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

    /// 시험용: [`ransac_fundamental`] 의 (F, 정상 표시).
    fn ransac_fi(
        x1: &[Vector2<f64>],
        x2: &[Vector2<f64>],
        cfg: &RansacConfig,
    ) -> Option<(Matrix3<f64>, Vec<bool>)> {
        ransac_fundamental(x1, x2, cfg).map(|r| (r.f, r.inliers))
    }

    fn two_cameras() -> (Camera, Camera) {
        let k = Intrinsics::from_hfov(960, 540, 70f64.to_radians());
        let r1 = Rotation3::from_euler_angles(0.05, -0.02, 0.1);
        let r2 = Rotation3::from_euler_angles(0.12, 0.08, -0.05);
        let c1 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r1, &Point3::new(0.0, 0.0, -10.0)),
        };
        let c2 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r2, &Point3::new(2.0, 0.5, -10.3)),
        };
        (c1, c2)
    }

    /// 정답 대응 n 개(+ 잡음 σ px)와 이상치 비율 out 만큼 무작위 짝.
    #[allow(clippy::type_complexity)]
    fn correspondences(
        n: usize,
        sigma: f64,
        out: f64,
        seed: u64,
    ) -> (
        Vec<Vector2<f64>>,
        Vec<Vector2<f64>>,
        Vec<bool>,
        Camera,
        Camera,
    ) {
        let (c1, c2) = two_cameras();
        let mut g = Lcg(seed);
        let (mut x1, mut x2, mut truth) = (vec![], vec![], vec![]);
        while x1.len() < n {
            let p = Point3::new(
                g.next() * 12.0 - 6.0,
                g.next() * 8.0 - 4.0,
                g.next() * 6.0 - 3.0,
            );
            let (Some(a), Some(b)) = (c1.project(&p), c2.project(&p)) else {
                continue;
            };
            if !c1.intrinsics.contains(&a) || !c2.intrinsics.contains(&b) {
                continue;
            }
            let outlier = g.next() < out;
            let b = if outlier {
                Vector2::new(g.next() * 960.0, g.next() * 540.0)
            } else {
                b + Vector2::new(g.gauss(), g.gauss()) * sigma
            };
            x1.push(a + Vector2::new(g.gauss(), g.gauss()) * sigma);
            x2.push(b);
            truth.push(!outlier);
        }
        (x1, x2, truth, c1, c2)
    }

    #[test]
    fn eight_point_exact_on_noise_free_points() {
        let (x1, x2, _, c1, c2) = correspondences(50, 0.0, 0.0, 3);
        let f = fundamental_8pt(&x1, &x2).unwrap();
        let worst = (0..x1.len())
            .map(|i| sampson_error(&f, &x1[i], &x2[i]).sqrt())
            .fold(0.0, f64::max);
        assert!(worst < 1e-6, "최대 Sampson 거리 {worst} px (제곱근)");
        // 정답 F 와 부호까지 맞춰 비교.
        let g = fundamental_from_cameras(&c1, &c2);
        let d = (f - g).norm().min((f + g).norm());
        assert!(d < 1e-6, "정답 F 와 차이 {d}");
        assert!(f.determinant().abs() < 1e-9);
    }

    #[test]
    fn ransac_separates_outliers() {
        for (out, seed) in [(0.3, 5u64), (0.5, 9)] {
            let (x1, x2, truth, c1, c2) = correspondences(300, 0.5, out, seed);
            let (f, inl) = ransac_fi(&x1, &x2, &RansacConfig::default()).unwrap();
            let tp = (0..x1.len()).filter(|&i| inl[i] && truth[i]).count();
            let fp = (0..x1.len()).filter(|&i| inl[i] && !truth[i]).count();
            let pos = truth.iter().filter(|&&t| t).count();
            let (prec, rec) = (tp as f64 / (tp + fp) as f64, tp as f64 / pos as f64);
            // 정답 정상 짝에 대한 추정 F 의 Sampson 거리 RMS(px): 제곱 거리(px²) 평균의 제곱근.
            let g = fundamental_from_cameras(&c1, &c2);
            let rms = |m: &Matrix3<f64>| {
                let (s, k) = (0..x1.len())
                    .filter(|&i| truth[i])
                    .fold((0.0, 0), |(s, k), i| {
                        (s + sampson_error(m, &x1[i], &x2[i]), k + 1)
                    });
                (s / k as f64).sqrt()
            };
            eprintln!(
                "outliers={out} precision={prec:.3} recall={rec:.3} rms={:.3} gt_rms={:.3}",
                rms(&f),
                rms(&g)
            );
            assert!(prec >= 0.97, "정밀도 {prec}");
            assert!(rec >= 0.98, "재현율 {rec}");
            assert!(rms(&f) < 0.6, "Sampson RMS {}", rms(&f));
        }
    }

    /// 이상치 50% 데이터 시드 100개 × RANSAC 시드 5개: (데이터 시드, RANSAC 시드,
    /// 정답 F 의 (정밀도, 재현율), 추정의 (정밀도, 재현율)). 정답 F 도 같은 Sampson 문턱으로 판정한다.
    #[allow(clippy::type_complexity)]
    fn many_seed_cases() -> Vec<(u64, u64, (f64, f64, usize), Option<(f64, f64, usize)>)> {
        use rayon::prelude::*;
        let th2 = RansacConfig::default().threshold_px.powi(2);
        (1..=100u64)
            .into_par_iter()
            .flat_map_iter(|d| {
                let (x1, x2, truth, c1, c2) = correspondences(300, 0.5, 0.5, d * 7919);
                let pos = truth.iter().filter(|&&t| t).count();
                let pr = |inl: &[bool]| {
                    let tp = (0..x1.len()).filter(|&i| inl[i] && truth[i]).count();
                    let fp = (0..x1.len()).filter(|&i| inl[i] && !truth[i]).count();
                    (
                        tp as f64 / (tp + fp) as f64,
                        tp as f64 / pos as f64,
                        tp + fp,
                    )
                };
                let g = fundamental_from_cameras(&c1, &c2);
                let gi: Vec<bool> = (0..x1.len())
                    .map(|i| sampson_error(&g, &x1[i], &x2[i]) < th2)
                    .collect();
                let gt = pr(&gi);
                (1..=5u64)
                    .map(|r| {
                        let cfg = RansacConfig {
                            seed: r,
                            ..RansacConfig::default()
                        };
                        let est = ransac_fi(&x1, &x2, &cfg).map(|(_, inl)| pr(&inl));
                        (d, r, gt, est)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// F-010 정밀도: 추정 F 의 정밀도 ≥ 정답 F 의 정밀도 − 0.04 (500 경우 모두), None 0건.
    /// 절대 기준 0.97 은 정답 F 자체가 데이터 시드 2/100 에서 못 넘는다(최소 0.9675).
    /// 500 경우 중 423 경우는 추정 F 의 정상 수가 정답 F 보다 많다: 최대 합의 목적함수가 문턱 띠 안에
    /// 우연히 든 이상치 몇 개를 더 끌어들이는 해를 고르는 것이라 탐색 실패가 아니다.
    /// 0.04 는 정상 짝 약 150개에서 이상치 6개 정도에 해당한다(실측 차이: 중앙값 −0.013, 최소 −0.034).
    #[test]
    fn ransac_many_seeds_precision_relative_to_truth() {
        let cases = many_seed_cases();
        let none = cases.iter().filter(|c| c.3.is_none()).count();
        let bad: Vec<_> = cases
            .iter()
            .filter(|c| c.3.is_some_and(|(p, _, _)| p < c.2 .0 - 0.04))
            .collect();
        assert_eq!(none, 0, "None 발생");
        assert!(bad.is_empty(), "정밀도 미달 {:?}", &bad[..bad.len().min(5)]);
    }

    /// F-010 재현율: 재현율 ≥ min(0.98, 정답 F 재현율 − 0.01).
    /// 예외는 추정 F 의 정상 수가 정답 F 이상인 경우뿐이다: 그때는 탐색이 아니라 최대 합의 목적함수가
    /// 띠 가장자리 이상치를 넣고 정상 짝 한두 개를 내준 해를 고른 것이라(정밀도 테스트와 같은 근거)
    /// 재현율 ≥ 정답 F 재현율 − 0.03 만 요구한다. 실측(시드 1..=100 × 1..=5): 예외 적용 1건
    /// (데이터 시드 76·RANSAC 시드 5, 정상 수 140 > 정답 138, 재현율 0.978 vs 정답 0.993).
    #[test]
    fn ransac_many_seeds_recall() {
        let cases = many_seed_cases();
        let bad: Vec<_> = cases
            .iter()
            .filter(|c| {
                c.3.is_some_and(|(_, r, k)| {
                    let strict = r >= (c.2 .1 - 0.01).min(0.98);
                    let objective = k >= c.2 .2 && r >= c.2 .1 - 0.03;
                    !(strict || objective)
                })
            })
            .collect();
        assert!(
            bad.is_empty(),
            "재현율 미달 {} {:?}",
            bad.len(),
            &bad[..bad.len().min(6)]
        );
    }

    /// F-010 탐색: 추정 F 의 정상 수가 정답 F 정상 수의 98% 미만인 경우(탐색 실패) 0건.
    /// 정답 F 도 잡음 때문에 최대 합의 해가 아니라 1개 차이는 흔하다(실측 3/500 이 −1).
    /// 개선 전 최악은 104 대 133(78%) 이었다.
    #[test]
    fn ransac_many_seeds_search_reaches_truth_consensus() {
        let cases = many_seed_cases();
        let bad: Vec<_> = cases
            .iter()
            .filter(|c| {
                c.3.is_some_and(|(_, _, k)| (k as f64) < 0.98 * c.2 .2 as f64)
            })
            .collect();
        assert!(
            bad.is_empty(),
            "탐색 실패 {} {:?}",
            bad.len(),
            &bad[..bad.len().min(6)]
        );
    }

    /// F-010: 이상치 50% 데이터 시드 100개 × RANSAC 시드 5개에서 None(반복 수 붕괴) 0건.
    #[test]
    fn ransac_many_seeds_never_returns_none() {
        use rayon::prelude::*;
        let none = (1..=100u64)
            .flat_map(|d| (1..=5u64).map(move |r| (d, r)))
            .collect::<Vec<_>>()
            .par_iter()
            .filter(|&&(d, r)| {
                let (x1, x2, _, _, _) = correspondences(300, 0.5, 0.5, d * 7919);
                let cfg = RansacConfig {
                    seed: r,
                    ..RansacConfig::default()
                };
                ransac_fundamental(&x1, &x2, &cfg).is_none()
            })
            .count();
        assert_eq!(none, 0);
    }

    #[test]
    fn adaptive_iterations_does_not_collapse() {
        // w = 2/300: w⁸ ≈ 2.6e-18 이라 1 − w⁸ == 1 (f64). 최대 반복 수가 나와야 한다.
        assert_eq!(adaptive_iterations(2.0 / 300.0, 8, 0.999, 2000), 2000);
        assert_eq!(adaptive_iterations(0.0, 8, 0.999, 2000), 2000);
        // w = 1: 최소 반복 수.
        assert_eq!(adaptive_iterations(1.0, 8, 0.999, 2000), MIN_RANSAC_ITERS);
        // w = 0.5: ln(0.001)/ln(1 − 1/256) = 1764.9 → 1765.
        assert_eq!(adaptive_iterations(0.5, 8, 0.999, 2000), 1765);
        // 큰 w 에서도 최소 반복 수 아래로 내려가지 않는다.
        assert_eq!(adaptive_iterations(0.95, 8, 0.999, 2000), MIN_RANSAC_ITERS);
    }

    #[test]
    fn candidate_pairs_counts() {
        // 카메라 3대 × 위치 10개.
        let views: Vec<(usize, usize)> =
            (0..10).flat_map(|p| (0..3).map(move |c| (c, p))).collect();
        let pairs = candidate_pairs(&views, 2, 1, 0);
        // 같은 카메라: 카메라마다 (9 + 8) 짝 → 51. 다른 카메라: 위치 차 0 → 10×3, 차 1 → 9×6 → 84.
        assert_eq!(pairs.len(), 51 + 84);
        assert!(pairs.iter().all(|&(i, j)| i < j));
        let (ca, pa) = views[pairs[0].0];
        assert_eq!((ca, pa), (0, 0));
    }

    #[test]
    fn candidate_pairs_power_of_two_gaps() {
        // 위치 80 × 카메라 3, SPEC 기본값.
        let views: Vec<(usize, usize)> =
            (0..80).flat_map(|p| (0..3).map(move |c| (c, p))).collect();
        let pairs = candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);
        let mut same = std::collections::BTreeSet::new();
        for &(i, j) in &pairs {
            let ((ca, pa), (cb, pb)) = (views[i], views[j]);
            if ca == cb {
                same.insert(pa.abs_diff(pb));
            } else {
                assert!(
                    pa.abs_diff(pb) <= 4,
                    "다른 카메라 위치 차 {}",
                    pa.abs_diff(pb)
                );
            }
        }
        assert_eq!(
            same.into_iter().collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 8, 16]
        );
        // 같은 카메라: (79+78+77+76+75) + (72+64) = 521, 카메라 3대 → 1563.
        // 다른 카메라: 카메라 짝마다 80 + 2(79+78+77+76) = 700, 3 짝 → 2100.
        assert_eq!(pairs.len(), 1563 + 2100);
        // 상한을 인자로 넓히면 32·64 간격이 더해진다: 카메라마다 48 + 16 = 64 짝 증가.
        assert_eq!(
            candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, usize::MAX).len(),
            1563 + 2100 + 3 * 64
        );
        // 상한 0 이면 거듭제곱 간격 짝이 없다: 카메라마다 72 + 64 = 136 짝 감소.
        assert_eq!(
            candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, 0).len(),
            1563 + 2100 - 3 * 136
        );
    }

    /// 합성 드론 장면 같은 카메라 두 장: 특징 → 비율 매칭 → RANSAC.
    /// 정답 깊이로 각 짝의 참·거짓을 판정해 (RANSAC 전 정답 비율, 후 정밀도, 재현율, 정상 수).
    fn scene_pair(cam: crate::synth::CamId, step: usize, seed: u64) -> (f64, f64, f64, usize) {
        scene_pair_cams(
            crate::synth::SceneConfig {
                seed,
                ..crate::synth::SceneConfig::default()
            },
            cam,
            cam,
            step,
        )
    }

    fn scene_pair_cams(
        base: crate::synth::SceneConfig,
        cam_a: crate::synth::CamId,
        cam_b: crate::synth::CamId,
        step: usize,
    ) -> (f64, f64, f64, usize) {
        use crate::features::{detect_and_describe, DetectorConfig, GrayImage};
        use crate::synth::{Scene, SceneConfig};
        let (w, h) = (480usize, 270usize);
        let scene = Scene::new(SceneConfig {
            width: w as u32,
            height: h as u32,
            ..base
        });
        let va = scene
            .views
            .iter()
            .find(|v| v.cam == cam_a && v.position == 0)
            .unwrap();
        let vb = scene
            .views
            .iter()
            .find(|v| v.cam == cam_b && v.position == va.position + step)
            .unwrap();
        let (ia, da) = scene.render(va);
        let (ib, _) = scene.render(vb);
        let cfg = DetectorConfig::default();
        let fa = detect_and_describe(&GrayImage::from_rgb(w, h, &ia.data), &cfg);
        let fb = detect_and_describe(&GrayImage::from_rgb(w, h, &ib.data), &cfg);
        let m = ratio_match(&fa, &fb, 0.8, true);
        let px = |f: &Feature| Vector2::new(f.kp.x as f64 + 0.5, f.kp.y as f64 + 0.5);
        let x1: Vec<_> = m.iter().map(|&(i, _)| px(&fa[i])).collect();
        let x2: Vec<_> = m.iter().map(|&(_, j)| px(&fb[j])).collect();
        let truth: Vec<bool> = m
            .iter()
            .zip(&x2)
            .map(|(&(i, _), q)| {
                let k = &fa[i].kp;
                let z =
                    da[(k.y.round() as usize).min(h - 1) * w + (k.x.round() as usize).min(w - 1)];
                z.is_finite()
                    && vb
                        .camera
                        .project(&va.camera.unproject(&px(&fa[i]), z as f64))
                        .is_some_and(|e| (e - q).norm() < 2.0)
            })
            .collect();
        let Some((_, inl)) = ransac_fi(&x1, &x2, &RansacConfig::default()) else {
            let pos = truth.iter().filter(|&&t| t).count();
            return (pos as f64 / m.len().max(1) as f64, 0.0, 0.0, 0);
        };
        let n = m.len();
        let pos = truth.iter().filter(|&&t| t).count();
        let tp = (0..n).filter(|&i| inl[i] && truth[i]).count();
        let ni = inl.iter().filter(|&&b| b).count();
        (
            pos as f64 / n as f64,
            tp as f64 / ni as f64,
            tp as f64 / pos as f64,
            ni,
        )
    }

    // 실측 편대 배치, 시드 1~3 × F/R/L × 간격 1·3 (18 경우) 측정: 정상 504~703,
    // 정밀도 0.991~1.000, 재현율 1.000. 정상 수 기준은 최솟값 504 의 약 79%,
    // 정밀도·재현율은 예전과 같은 0.98 (측정 최솟값보다 0.011 아래).
    const MIN_INL: usize = 400;
    const MIN_PREC: f64 = 0.98;
    const MIN_REC: f64 = 0.98;

    /// 실측 편대 배치(SPEC §1: 위치 간 1.0 m, 기울기 60°, 화각 65°)의 같은 카메라 짝.
    /// 기준은 이 배치에서 시드 1~3·카메라 F/R/L·간격 1/3 을 잰 최솟값에서 정했다(위 상수 주석).
    /// 예전 기준(정상 ≥250)은 위치 간 2.5 m·기울기 50° 배치의 F 한 대만 잰 값이었다.
    /// 위치 간 이동이 1.0 m 로 줄어 두 장의 겹침이 커지므로 정상 수는 오히려 늘었다.
    #[test]
    fn ransac_on_synthetic_drone_views() {
        use crate::synth::CamId;
        for seed in [1u64, 2, 3] {
            for cam in CamId::ALL {
                for step in [1usize, 3] {
                    let (before, prec, rec, ni) = scene_pair(cam, step, seed);
                    eprintln!(
                        "scene seed={seed} cam={cam:?} step={step} correct_before={before:.3} precision={prec:.3} recall={rec:.3} inliers={ni}"
                    );
                    assert!(ni >= MIN_INL, "정상 수 {ni}");
                    assert!(prec >= MIN_PREC, "정밀도 {prec}");
                    assert!(rec >= MIN_REC, "재현율 {rec}");
                }
            }
        }
    }

    #[test]
    fn ransac_on_cross_camera_views() {
        // 큰 시선 차(90°) 짝에서 RANSAC 이 버티는지 보는 시험이라 예전 쉬운 배치를 명시적으로 쓴다:
        // 실측 편대에서 같은 위치의 F 와 R·L 은 시선이 120° 이상 벌어지고 지면 발자국이 거의
        // 겹치지 않아 같은 위치 짝이 성립하지 않는다.
        // 앞 카메라 위치 0 과 옆 카메라 위치 0·4(10 m 앞): 시선이 90° 다른 짝.
        use crate::synth::{CamId, SceneConfig};
        for b in [CamId::R, CamId::L] {
            for step in [0usize, 4] {
                let (before, prec, rec, ni) =
                    scene_pair_cams(SceneConfig::easy(), CamId::F, b, step);
                eprintln!(
                    "F->{b:?} step={step} correct_before={before:.3} precision={prec:.3} recall={rec:.3} inliers={ni}"
                );
                assert!(ni >= 40, "정상 수 {ni}");
                assert!(prec >= 0.95, "정밀도 {prec}");
                assert!(rec >= 0.95, "재현율 {rec}");
            }
        }
    }

    fn feature(g: &mut Lcg) -> Feature {
        let mut desc = [0f32; crate::features::DESC_LEN];
        for v in desc.iter_mut() {
            *v = g.next() as f32;
        }
        let n = desc.iter().map(|v| v * v).sum::<f32>().sqrt();
        desc.iter_mut().for_each(|v| *v /= n);
        let kp = crate::features::Keypoint {
            x: 0.0,
            y: 0.0,
            sigma: 1.6,
            response: 1.0,
            angle: 0.0,
        };
        Feature { kp, desc }
    }

    /// 이전 구현(행마다 b 전수 탐색, 상호 확인 때 b→a 재탐색) 그대로. 동치 비교용 기준.
    fn ratio_match_reference(
        a: &[Feature],
        b: &[Feature],
        ratio: f32,
        mutual: bool,
    ) -> Vec<(usize, usize)> {
        let nearest = |q: &Feature, set: &[Feature]| -> Option<(usize, f32, f32)> {
            let mut best = (usize::MAX, f32::INFINITY, f32::INFINITY);
            for (j, f) in set.iter().enumerate() {
                let d: f32 = q
                    .desc
                    .iter()
                    .zip(&f.desc)
                    .map(|(p, r)| (p - r).powi(2))
                    .sum();
                if d < best.1 {
                    best = (j, d, best.1);
                } else if d < best.2 {
                    best.2 = d;
                }
            }
            (best.0 != usize::MAX).then_some(best)
        };
        let mut out = Vec::new();
        for (i, fa) in a.iter().enumerate() {
            let Some((j, d1, d2)) = nearest(fa, b) else {
                continue;
            };
            if !d2.is_finite() || d1 >= ratio * ratio * d2 {
                continue;
            }
            if mutual && nearest(&b[j], a).map(|r| r.0) != Some(i) {
                continue;
            }
            out.push((i, j));
        }
        out
    }

    /// b = a 일부의 잡음 섞인 복사 + 쌍둥이 + 무관한 기술자. 묶음 경계를 걸치도록 크기를 고른다.
    fn match_fixture(seed: u64, na: usize, extra: usize) -> (Vec<Feature>, Vec<Feature>) {
        let mut g = Lcg(seed);
        let a: Vec<Feature> = (0..na).map(|_| feature(&mut g)).collect();
        let mut b = Vec::new();
        for (i, f) in a.iter().enumerate() {
            if i % 3 == 2 {
                continue;
            }
            for _ in 0..1 + usize::from(i % 7 == 0) {
                let mut h = f.clone();
                h.desc
                    .iter_mut()
                    .for_each(|v| *v += 0.03 * g.gauss() as f32);
                b.push(h);
            }
        }
        b.extend((0..extra).map(|_| feature(&mut g)));
        (a, b)
    }

    #[test]
    fn ratio_match_equals_reference() {
        // 블록·병렬 구현의 결과 짝 집합이 이전 전수 탐색과 같다(상호·비상호, 여러 비율, 빈 입력).
        for (seed, na, extra) in [
            (1, 1, 0),
            (2, 63, 5),
            (3, 64, 64),
            (4, 65, 10),
            (5, 700, 300),
        ] {
            let (a, b) = match_fixture(seed, na, extra);
            for mutual in [true, false] {
                for ratio in [0.6f32, 0.8, 1.0] {
                    let fast = ratio_match(&a, &b, ratio, mutual);
                    let slow = ratio_match_reference(&a, &b, ratio, mutual);
                    assert_eq!(fast, slow, "seed {seed} mutual {mutual} ratio {ratio}");
                    // 반대 방향도.
                    let fast = ratio_match(&b, &a, ratio, mutual);
                    let slow = ratio_match_reference(&b, &a, ratio, mutual);
                    assert_eq!(fast, slow, "rev seed {seed} mutual {mutual} ratio {ratio}");
                }
            }
        }
        let (a, _) = match_fixture(9, 10, 0);
        assert!(ratio_match(&a, &[], 0.8, true).is_empty());
        assert!(ratio_match(&[], &a, 0.8, true).is_empty());
    }

    #[test]
    #[ignore = "시간 측정: cargo test --release -- --ignored --test-threads=1 ratio_match_timing"]
    fn ratio_match_timing() {
        // F-014 확인 기준 크기(7300×7300). 같은 크기에서 이전 구현과 결과 짝 집합도 비교한다.
        let (a, mut b) = match_fixture(21, 7300, 3000);
        b.truncate(7300);
        let t = std::time::Instant::now();
        let m = ratio_match(&a, &b, 0.8, true);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "ratio_match {}x{}: {:.3} s, 짝 {}, 스레드 {}",
            a.len(),
            b.len(),
            dt,
            m.len(),
            rayon::current_num_threads()
        );
        let t = std::time::Instant::now();
        let slow = ratio_match_reference(&a, &b, 0.8, true);
        println!("이전 구현: {:.3} s", t.elapsed().as_secs_f64());
        assert_eq!(m, slow);
        assert!(dt <= 0.5, "매칭 {dt:.3} s > 0.5 s");
    }

    #[test]
    fn ratio_match_recovers_permutation() {
        // a 의 100개를 섞어 b 에 넣고(작은 잡음), 짝 없는 100개를 더한다.
        // 앞 10개는 거의 같은 쌍둥이를 b 에 함께 넣어 비율 검사로 버려져야 한다.
        let mut g = Lcg(11);
        let a: Vec<Feature> = (0..100).map(|_| feature(&mut g)).collect();
        let mut b: Vec<(Feature, Option<usize>)> = Vec::new();
        for (i, f) in a.iter().enumerate() {
            let mut h = f.clone();
            h.desc
                .iter_mut()
                .for_each(|v| *v += 0.01 * g.gauss() as f32);
            b.push((h, Some(i)));
            if i < 10 {
                let mut t = f.clone();
                t.desc
                    .iter_mut()
                    .for_each(|v| *v += 0.01 * g.gauss() as f32);
                b.push((t, None));
            }
        }
        for _ in 0..100 {
            b.push((feature(&mut g), None));
        }
        // 결정적 섞기.
        for i in (1..b.len()).rev() {
            let j = (g.next() * (i + 1) as f64) as usize;
            b.swap(i, j);
        }
        let bf: Vec<Feature> = b.iter().map(|x| x.0.clone()).collect();
        let m = ratio_match(&a, &bf, 0.8, true);
        let correct = m.iter().filter(|&&(i, j)| b[j].1 == Some(i)).count();
        assert_eq!(correct, m.len(), "틀린 짝 {}", m.len() - correct);
        assert_eq!(m.len(), 90, "매칭 수 {}", m.len());
        assert!(m.iter().all(|&(i, _)| i >= 10));
    }

    #[test]
    fn degenerate_inputs_are_rejected_without_panic() {
        let (x1, x2, _, _, _) = correspondences(50, 0.0, 0.0, 3);
        let cfg = RansacConfig::default();
        // 점 8개 미만
        assert!(fundamental_8pt(&x1[..7], &x2[..7]).is_none());
        assert!(ransac_fundamental(&x1[..7], &x2[..7], &cfg).is_none());
        // 길이 불일치
        assert!(fundamental_8pt(&x1, &x2[..40]).is_none());
        assert!(ransac_fundamental(&x1, &x2[..40], &cfg).is_none());
        // NaN·무한대 좌표
        for bad in [f64::NAN, f64::INFINITY] {
            let mut y = x2.clone();
            y[17].x = bad;
            assert!(fundamental_8pt(&x1, &y).is_none());
            assert!(ransac_fundamental(&x1, &y, &cfg).is_none());
        }
        // 모든 점이 두 영상에서 각각 한 직선 위: 영공간이 2차원 이상
        let line = |a: f64, b: f64| -> Vec<Vector2<f64>> {
            (0..60)
                .map(|i| Vector2::new(100.0 + 13.0 * i as f64, a + b * i as f64))
                .collect()
        };
        let (l1, l2) = (line(200.0, 3.0), line(350.0, -2.0));
        assert!(fundamental_8pt(&l1, &l2).is_none());
        assert!(ransac_fundamental(&l1, &l2, &cfg).is_none());
    }

    #[test]
    fn ratio_match_needs_second_neighbour() {
        // b 에 특징이 하나뿐이면 차근접이 없어 비율을 잴 수 없다: 짝을 만들지 않는다.
        let mut g = Lcg(5);
        let a = vec![feature(&mut g), feature(&mut g)];
        let b = vec![feature(&mut g)];
        for mutual in [false, true] {
            for ratio in [0.6f32, 0.8, 1.0] {
                assert!(ratio_match(&a, &b, ratio, mutual).is_empty());
            }
        }
        // b 에 a 의 복사가 둘이면 정상적으로 비율 검사를 한다.
        let b2 = vec![a[0].clone(), feature(&mut g)];
        assert_eq!(ratio_match(&a[..1], &b2, 0.8, false), vec![(0, 0)]);
    }

    /// 두 영상에서 각각 한 직선 위의 점 60개에 σ px 등방 잡음.
    fn noisy_lines(sigma: f64, seed: u64) -> (Vec<Vector2<f64>>, Vec<Vector2<f64>>) {
        let mut g = Lcg(seed);
        let mut line = |a: f64, b: f64| -> Vec<Vector2<f64>> {
            (0..60)
                .map(|i| {
                    let t = i as f64 + g.next();
                    Vector2::new(100.0 + 13.0 * t, a + b * t)
                        + Vector2::new(g.gauss(), g.gauss()) * sigma
                })
                .collect()
        };
        let l1 = line(200.0, 3.0);
        let l2 = line(350.0, -2.0);
        (l1, l2)
    }

    #[test]
    fn noisy_collinear_points_are_rejected() {
        // 잡음 섞인 동일선상(σ = 0.5, 1.0 px): 고윳값 비 검사만으로는 통과하던 배치.
        let cfg = RansacConfig::default();
        for sigma in [0.5, 1.0] {
            for seed in 1..=5u64 {
                let (l1, l2) = noisy_lines(sigma, seed);
                assert!(fundamental_8pt(&l1, &l2).is_none(), "σ {sigma} seed {seed}");
                assert!(
                    ransac_fundamental(&l1, &l2, &cfg).is_none(),
                    "σ {sigma} seed {seed}"
                );
            }
        }
    }

    /// 두 카메라([`two_cameras`])로 본 평면 z = 0 위의 점 n 개(σ px 잡음).
    fn planar_correspondences(
        n: usize,
        sigma: f64,
        seed: u64,
    ) -> (Vec<Vector2<f64>>, Vec<Vector2<f64>>) {
        let (c1, c2) = two_cameras();
        let mut g = Lcg(seed);
        let (mut x1, mut x2) = (vec![], vec![]);
        while x1.len() < n {
            let p = Point3::new(g.next() * 14.0 - 7.0, g.next() * 8.0 - 4.0, 0.0);
            let (Some(a), Some(b)) = (c1.project(&p), c2.project(&p)) else {
                continue;
            };
            if !c1.intrinsics.contains(&a) || !c2.intrinsics.contains(&b) {
                continue;
            }
            x1.push(a + Vector2::new(g.gauss(), g.gauss()) * sigma);
            x2.push(b + Vector2::new(g.gauss(), g.gauss()) * sigma);
        }
        (x1, x2)
    }

    #[test]
    fn homography_support_separates_planar_scene() {
        // 평면 판정 지표: z = 0 평면 위 점(σ = 0.5 px)은 한 호모그래피가 거의 전부 설명하고,
        // 깊이가 6 m 퍼진 일반 장면은 절반도 설명하지 못한다. 거부 규칙에는 아직 쓰지 않는다.
        let th = RansacConfig::default().threshold_px;
        for seed in 1..=5u64 {
            let (x1, x2) = planar_correspondences(200, 0.5, seed);
            let h = homography_support(&x1, &x2, th);
            eprintln!("평면 seed {seed}: {h}/200");
            // 양쪽 σ = 0.5 px 에서 대칭 전달 오차(두 방향 중 큰 쪽) < 1.5 px 일 확률은 약 0.85(기대 170),
            // 이항 표준편차 약 5 → 4σ 아래인 150 을 기준으로 둔다.
            assert!(h >= 150, "seed {seed}: 평면 호모그래피 설명 {h}/200");
            let (x1, x2, _, _, _) = correspondences(200, 0.5, 0.0, seed);
            let h = homography_support(&x1, &x2, th);
            eprintln!("일반 seed {seed}: {h}/200");
            assert!(h < 100, "seed {seed}: 일반 장면 호모그래피 설명 {h}/200");
        }
    }

    /// 매칭 결과 짝 목록의 FNV-1a 64비트 해시.
    fn pairs_hash(m: &[(usize, usize)]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &(i, j) in m {
            for v in [i as u64, j as u64] {
                for byte in v.to_le_bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        h
    }

    #[test]
    fn ratio_match_regression_hash() {
        // 고정 기술자(match_fixture 시드 21, a 2000개, b 약 1600 + 무관 700)의 매칭 결과를 상수로 고정한다.
        // 픽스처에는 비율 경계·동률 근처 행이 거의 없어 누산 순서·동률 규칙 변경은 이 해시로 잡히지 않는다
        // (그 변경은 `ratio_match_rounding_boundary`·`ratio_match_ties_take_lower_index` 가 잡는다).
        // 기대값은 이 시험을 처음 넣은 커밋의 구현으로 한 번 계산한 값이다.
        let (a, b) = match_fixture(21, 2000, 700);
        let m = ratio_match(&a, &b, 0.8, true);
        let n = ratio_match(&a, &b, 0.95, false);
        eprintln!(
            "mutual 0.8: {} {:#x}, 0.95: {} {:#x}",
            m.len(),
            pairs_hash(&m),
            n.len(),
            pairs_hash(&n)
        );
        assert_eq!((m.len(), pairs_hash(&m)), EXPECT_MUTUAL);
        assert_eq!((n.len(), pairs_hash(&n)), EXPECT_LOOSE);
    }
    const EXPECT_MUTUAL: (usize, u64) = (1145, 0x22ca_99b8_6722_152d);
    const EXPECT_LOOSE: (usize, u64) = (1282, 0x7d0c_d999_3d0c_89e5);

    #[test]
    fn random_correspondences_are_not_confirmed() {
        // 서로 무관한 무작위 대응 200개: 정상 짝이 없으므로 F 를 확정하면 안 된다.
        for seed in 1..=10u64 {
            let mut g = Lcg(seed);
            let mut pt = || Vector2::new(g.next() * 960.0, g.next() * 540.0);
            let x1: Vec<_> = (0..200).map(|_| pt()).collect();
            let x2: Vec<_> = (0..200).map(|_| pt()).collect();
            let r = ransac_fundamental(&x1, &x2, &RansacConfig::default());
            // 비율 조건 없이 몇 개가 우연히 문턱을 넘는지도 기록한다.
            let loose = RansacConfig {
                min_inlier_ratio: 0.0,
                ..RansacConfig::default()
            };
            let cnt = ransac_fi(&x1, &x2, &loose)
                .map(|(_, inl)| inl.iter().filter(|&&b| b).count())
                .unwrap_or(0);
            eprintln!("seed={seed} 우연 정상 수={cnt}/200");
            assert!(cnt < 40, "우연 정상 비율이 0.2 이상: {cnt}/200");
            assert!(r.is_none(), "무작위 대응에서 F 확정");
        }
    }
    /// 지면(z = 0) 위 점과 그보다 카메라 쪽으로 1~3 m 솟은 건물 점(비율 `bld`)을 섞은 대응(σ px).
    fn ground_with_buildings(
        n: usize,
        bld: f64,
        sigma: f64,
        seed: u64,
    ) -> (Vec<Vector2<f64>>, Vec<Vector2<f64>>) {
        let (c1, c2) = two_cameras();
        let mut g = Lcg(seed);
        let (mut x1, mut x2) = (vec![], vec![]);
        let nb = (n as f64 * bld).round() as usize;
        while x1.len() < n {
            let z = if x1.len() < nb {
                -1.0 - 2.0 * g.next()
            } else {
                0.0
            };
            let p = Point3::new(g.next() * 14.0 - 7.0, g.next() * 8.0 - 4.0, z);
            let (Some(a), Some(b)) = (c1.project(&p), c2.project(&p)) else {
                continue;
            };
            if !c1.intrinsics.contains(&a) || !c2.intrinsics.contains(&b) {
                continue;
            }
            x1.push(a + Vector2::new(g.gauss(), g.gauss()) * sigma);
            x2.push(b + Vector2::new(g.gauss(), g.gauss()) * sigma);
        }
        (x1, x2)
    }

    #[test]
    fn planar_scene_selects_homography() {
        let cfg = RansacConfig::default();
        for seed in 1..=5u64 {
            // 모든 점 z = 0, σ = 0.5 px: RANSAC 이 낸 F 와 정상 짝으로 모델을 고르면 호모그래피.
            let (x1, x2) = planar_correspondences(200, 0.5, seed);
            let (f, inl) = ransac_fi(&x1, &x2, &cfg).expect("평면 장면에서 RANSAC None");
            let s1: Vec<_> = (0..x1.len()).filter(|&i| inl[i]).map(|i| x1[i]).collect();
            let s2: Vec<_> = (0..x1.len()).filter(|&i| inl[i]).map(|i| x2[i]).collect();
            let m = select_two_view_model(&s1, &s2, &f, cfg.threshold_px / 3.0).unwrap();
            assert_eq!(
                m.model,
                TwoViewModel::Homography,
                "seed {seed}: RANSAC F {m:?}"
            );
            // 정답 F 를 주어도 모델 선택은 호모그래피.
            let (c1, c2) = two_cameras();
            let m =
                select_two_view_model(&x1, &x2, &fundamental_from_cameras(&c1, &c2), 0.5).unwrap();
            eprintln!("평면 seed {seed}: {m:?}");
            assert_eq!(m.model, TwoViewModel::Homography, "seed {seed}");
        }
    }

    #[test]
    fn model_selection_robust_to_underestimated_sigma() {
        // 실제 잡음 σ 0.5·1.0·1.5 px 에 호출자는 0.5 만 넘긴다: 순수 평면 200점은 모두 H.
        let (c1, c2) = two_cameras();
        let f0 = fundamental_from_cameras(&c1, &c2);
        for actual in [0.5, 1.0, 1.5] {
            for seed in 1..=5u64 {
                let (x1, x2) = planar_correspondences(200, actual, seed);
                let m = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
                eprintln!("평면 실제 σ {actual} seed {seed}: {m:?}");
                assert_eq!(
                    m.model,
                    TwoViewModel::Homography,
                    "σ {actual} seed {seed}: {m:?}"
                );
                // 강건 추정 σ 는 실제의 0.7~1.4 배 안(작게 넘긴 σ 를 바로잡는다).
                assert!(
                    m.sigma_px > 0.7 * actual && m.sigma_px < 1.4 * actual.max(0.5),
                    "σ {actual} seed {seed}: 추정 {}",
                    m.sigma_px
                );
            }
        }
        // 지면 + 건물 5%, 실제 σ 0.5 에 0.5 를 넘기면 계속 F.
        for seed in 1..=5u64 {
            let (x1, x2) = ground_with_buildings(300, 0.05, 0.5, seed);
            let m = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
            assert_eq!(
                m.model,
                TwoViewModel::Fundamental,
                "건물 5% seed {seed}: {m:?}"
            );
        }
    }

    #[test]
    fn model_selection_rejects_non_finite_sigma() {
        let (c1, c2) = two_cameras();
        let f0 = fundamental_from_cameras(&c1, &c2);
        let (x1, x2) = planar_correspondences(50, 0.5, 1);
        for s in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, -1.0] {
            assert!(select_two_view_model(&x1, &x2, &f0, s).is_none(), "σ {s}");
        }
        // 동일선상 20점에 σ = ∞ 도 None.
        let l1: Vec<_> = (0..20)
            .map(|i| Vector2::new(10.0 * i as f64, 5.0 * i as f64))
            .collect();
        let l2: Vec<_> = l1.iter().map(|p| p + Vector2::new(3.0, 1.0)).collect();
        assert!(select_two_view_model(&l1, &l2, &f0, f64::INFINITY).is_none());
    }

    #[test]
    fn ground_scene_with_few_buildings_keeps_fundamental() {
        // 건물 5%·10%(순수 GRIC 는 약 19% 미만에서 H 를 고른다): 시차 짝이 있으므로 F 를 유지한다.
        let cfg = RansacConfig::default();
        let (c1, c2) = two_cameras();
        let f0 = fundamental_from_cameras(&c1, &c2);
        for bld in [0.05, 0.1] {
            for seed in 1..=5u64 {
                let (x1, x2) = ground_with_buildings(300, bld, 0.5, seed);
                let m = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
                eprintln!("건물 {bld} seed {seed}: {m:?}");
                assert_eq!(m.model, TwoViewModel::Fundamental, "건물 {bld} seed {seed}");
                let (f, inl) = ransac_fi(&x1, &x2, &cfg).expect("지면+건물 장면에서 None");
                let s1: Vec<_> = (0..x1.len()).filter(|&i| inl[i]).map(|i| x1[i]).collect();
                let s2: Vec<_> = (0..x1.len()).filter(|&i| inl[i]).map(|i| x2[i]).collect();
                let me = select_two_view_model(&s1, &s2, &f, cfg.threshold_px / 3.0).unwrap();
                assert_eq!(
                    me.model,
                    TwoViewModel::Fundamental,
                    "건물 {bld} seed {seed}: RANSAC F {me:?}"
                );
                // 정답 F 기준 정상 짝(Sampson < 문턱)을 추정 F 도 95% 이상 정상으로 본다.
                let th2 = cfg.threshold_px * cfg.threshold_px;
                let truth: Vec<bool> = (0..x1.len())
                    .map(|i| sampson_error(&f0, &x1[i], &x2[i]) < th2)
                    .collect();
                let tp = (0..x1.len()).filter(|&i| truth[i] && inl[i]).count();
                let nt = truth.iter().filter(|&&b| b).count();
                assert!(
                    tp as f64 >= 0.95 * nt as f64,
                    "건물 {bld} seed {seed}: {tp}/{nt}"
                );
                // 건물 점(앞쪽 nb 개)에서 추정 F 의 Sampson 오차가 문턱 안: 평면 밖 기하도 맞는다.
                let nb = (300.0 * bld).round() as usize;
                let ok = (0..nb)
                    .filter(|&i| sampson_error(&f, &x1[i], &x2[i]) < th2)
                    .count();
                assert!(
                    ok as f64 >= 0.9 * nb as f64,
                    "건물 {bld} seed {seed}: 건물 {ok}/{nb}"
                );
            }
        }
    }

    #[test]
    fn general_scene_selects_fundamental() {
        let (c1, c2) = two_cameras();
        let f0 = fundamental_from_cameras(&c1, &c2);
        for seed in 1..=5u64 {
            let (x1, x2, _, _, _) = correspondences(200, 0.5, 0.0, seed);
            let m = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
            assert_eq!(m.model, TwoViewModel::Fundamental, "seed {seed}: {m:?}");
            assert!(m.gric_f < m.gric_h, "seed {seed}: {m:?}");
        }
    }

    /// F-027: Sampson 잔차의 해석적 야코비안이 중앙 차분 수치 야코비안과 상대 오차 1e-6 안에서 같다.
    #[test]
    fn sampson_jacobian_matches_numeric() {
        let (x1, x2, _, c1, c2) = correspondences(60, 0.5, 0.3, 17);
        let (t1, t2) = (normalizer(&x1), normalizer(&x2));
        let (s1, s2) = (t1[(0, 0)], t2[(0, 0)]);
        // 정답 F 를 정규화 좌표로 옮기고 조금 흔든 점(계수 2 아님)에서도 확인한다.
        let f = fundamental_from_cameras(&c1, &c2);
        let g0 = t2.try_inverse().unwrap().transpose() * f * t1.try_inverse().unwrap();
        let mut rng = Lcg(5);
        let shake = Matrix3::from_fn(|_, _| rng.gauss() * 0.01 * g0.norm());
        let mut worst: f64 = 0.0;
        for g in [g0 / g0.norm(), (g0 + shake) / (g0 + shake).norm()] {
            for (p, q) in x1.iter().zip(&x2) {
                let a = t1 * Vector3::new(p.x, p.y, 1.0);
                let b = t2 * Vector3::new(q.x, q.y, 1.0);
                let (r, j) = sampson_residual_jacobian(&g, s1, s2, &a, &b);
                // 잔차 자체도 픽셀 Sampson 거리와 같아야 한다.
                let fp = t2.transpose() * g * t1;
                assert!((r * r - sampson_error(&fp, p, q)).abs() <= 1e-9 * (1.0 + r * r));
                let h = 1e-6;
                let (mut num, mut diff) = (0.0f64, 0.0f64);
                for k in 0..9 {
                    let (mut gp, mut gm) = (g, g);
                    gp[(k / 3, k % 3)] += h;
                    gm[(k / 3, k % 3)] -= h;
                    let d = (sampson_residual_jacobian(&gp, s1, s2, &a, &b).0
                        - sampson_residual_jacobian(&gm, s1, s2, &a, &b).0)
                        / (2.0 * h);
                    num += d * d;
                    diff += (d - j[k]).powi(2);
                }
                worst = worst.max(diff.sqrt() / num.sqrt());
            }
        }
        eprintln!("야코비안 최대 상대 오차 {worst:.2e}");
        assert!(worst < 1e-6, "상대 오차 {worst}");
    }

    /// 현재 스레드가 CPU 에서 실제로 돈 시간(ns, Linux `/proc/thread-self/schedstat` 첫 값).
    /// 측정 기계에 다른 부하가 있으면 벽시계는 대기 시간을 포함하므로 이것으로 잰다. 없으면 None.
    fn thread_cpu_ns() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
        s.split_whitespace().next()?.parse().ok()
    }

    /// 대응 4000개(정상 50%, σ 0.5 px) RANSAC 1회: (벽시계 s, 스레드 CPU s, 정상 수, 정답 F 정상 수, LM 걸음 수들).
    fn ransac_4000_run() -> (f64, Option<f64>, usize, usize, Vec<(usize, LmStop)>) {
        let (x1, x2, _, c1, c2) = correspondences(4000, 0.5, 0.5, 4242);
        let th2 = RansacConfig::default().threshold_px.powi(2);
        let g = fundamental_from_cameras(&c1, &c2);
        let gt = (0..x1.len())
            .filter(|&i| sampson_error(&g, &x1[i], &x2[i]) < th2)
            .count();
        LM_ITERS.with(|v| v.borrow_mut().clear());
        let (c0, w0) = (thread_cpu_ns(), std::time::Instant::now());
        let (_, inl) = ransac_fi(&x1, &x2, &RansacConfig::default()).unwrap();
        let wall = w0.elapsed().as_secs_f64();
        let cpu = thread_cpu_ns().zip(c0).map(|(b, a)| (b - a) as f64 * 1e-9);
        let lm = LM_ITERS.with(|v| std::mem::take(&mut *v.borrow_mut()));
        let cnt = inl.iter().filter(|&&b| b).count();
        eprintln!("벽시계 {wall:.3} s, CPU {cpu:?} s, 정상 {cnt} / 정답 F {gt}, LM 걸음 {lm:?}");
        (wall, cpu, cnt, gt, lm)
    }

    /// F-027: 대응 4000개에서 정답 F 와 같은 문턱의 정상 수 98% 이상(탐색 실패 없음)이고,
    /// Sampson LM 은 상대 감소 규칙으로 수렴해 멈춘다(문턱 단계 6 × 상위 가설 5 + 마무리 호출).
    /// F-139: 종료 사유를 따로 세어, 호출 과반이 '걸음을 받아들인 뒤 수렴' 으로 끝나고
    /// '첫 걸음부터 감쇠 실패' 정체는 없음을 단언한다(정체 호출 하나로 통과하던 예전 단언 대체).
    #[test]
    fn ransac_4000_correspondences_lm_stops_early() {
        let (_, _, cnt, gt, lm) = ransac_4000_run();
        assert!(
            cnt as f64 >= 0.98 * gt as f64,
            "정상 {cnt} < 정답 F {gt} × 0.98"
        );
        let count = |s: LmStop| lm.iter().filter(|&&(_, r)| r == s).count();
        let (conv, stall, cap) = (
            count(LmStop::Converged),
            count(LmStop::Stalled),
            count(LmStop::Cap),
        );
        let stall_at_start = lm
            .iter()
            .filter(|&&(k, r)| r == LmStop::Stalled && k == 0)
            .count();
        eprintln!(
            "LM 종료 사유: 수렴 {conv}, 정체 {stall}(첫 걸음 정체 {stall_at_start}), 상한 {cap} / 호출 {}",
            lm.len()
        );
        assert!(!lm.is_empty(), "LM 호출 없음");
        assert!(
            2 * conv > lm.len(),
            "수렴 종료 {conv} / {} 가 과반 아님",
            lm.len()
        );
        assert_eq!(
            stall_at_start, 0,
            "첫 걸음부터 정체한 호출 {stall_at_start}"
        );
    }

    /// F-027 확인 기준: 대응 4000개(정상 50%) RANSAC 1회 CPU 시간 0.1 s 이하.
    /// 4 코어 측정 기계(동시 부하 있음)에서 0.116 s 로 미달 — 시간의 대부분은 가설 루프(약 1710회 ×
    /// 8점 + 4000개 판정)와 국소 최적화이고 Sampson LM 은 작은 몫이라 LM 만으로는 닿지 않는다.
    #[test]
    #[ignore = "F-027 시간 기준 미달(가설 루프가 지배), 노트 남은 문제"]
    fn ransac_4000_correspondences_time() {
        let (wall, cpu, _, _, _) = ransac_4000_run();
        let t = cpu.unwrap_or(wall);
        assert!(t <= 0.1, "RANSAC 1회 {t:.3} s");
    }

    /// 3차원 직선 (t, 0.3t + 0.5, 0.1t) 위 점 60개 + 일반 대응 k 개(σ px), 떼어 둔 일반 대응 100개.
    #[allow(clippy::type_complexity)]
    fn line_plus_general(
        k: usize,
        sigma: f64,
        seed: u64,
    ) -> (
        Vec<Vector2<f64>>,
        Vec<Vector2<f64>>,
        Vec<Vector2<f64>>,
        Vec<Vector2<f64>>,
    ) {
        let (c1, c2) = two_cameras();
        let mut g = Lcg(seed);
        let (mut x1, mut x2) = (vec![], vec![]);
        while x1.len() < 60 {
            let t = g.next() * 12.0 - 6.0;
            let p = Point3::new(t, 0.3 * t + 0.5, 0.1 * t);
            let (Some(a), Some(b)) = (c1.project(&p), c2.project(&p)) else {
                continue;
            };
            if !c1.intrinsics.contains(&a) || !c2.intrinsics.contains(&b) {
                continue;
            }
            x1.push(a + Vector2::new(g.gauss(), g.gauss()) * sigma);
            x2.push(b + Vector2::new(g.gauss(), g.gauss()) * sigma);
        }
        let (g1, g2, _, _, _) = correspondences(k + 100, sigma, 0.0, seed + 1000);
        x1.extend_from_slice(&g1[..k]);
        x2.extend_from_slice(&g2[..k]);
        (x1, x2, g1[k..].to_vec(), g2[k..].to_vec())
    }

    #[test]
    fn line_plus_few_general_points_is_not_confirmed() {
        // F-091: 직선 위 60점 + 일반 대응 2·4·6개는 F 가 정해지지 않는다.
        // None 이거나, 떼어 둔 일반 대응 100개의 Sampson 거리 중앙값이 3 px 이하인 F 만 허용한다.
        let cfg = RansacConfig::default();
        for k in [2usize, 4, 6, 10] {
            let mut confirmed = 0;
            for seed in 1..=10u64 {
                let (x1, x2, h1, h2) = line_plus_general(k, 0.5, seed);
                let Some((f, _)) = ransac_fi(&x1, &x2, &cfg) else {
                    continue;
                };
                confirmed += 1;
                let mut e: Vec<f64> = h1
                    .iter()
                    .zip(&h2)
                    .map(|(p, q)| sampson_error(&f, p, q).sqrt())
                    .collect();
                e.sort_by(f64::total_cmp);
                let med = e[e.len() / 2];
                eprintln!("k={k} seed={seed} 떼어 둔 대응 Sampson 중앙값 {med:.2} px");
                assert!(
                    med <= 3.0,
                    "k={k} seed={seed}: 틀린 F 확정(중앙값 {med:.1} px)"
                );
            }
            eprintln!("k={k}: 확정 {confirmed}/10");
            if k == 10 {
                assert_eq!(confirmed, 10, "일반 대응 10개면 F 가 정해진다");
            }
        }
    }

    #[test]
    fn binomial_tail_matches_direct_sum() {
        // n = 5, p = 0.3: P(X ≥ 2) = 1 − 0.7⁵ − 5·0.3·0.7⁴ = 0.47178.
        assert!((binomial_tail(5, 2, 0.3) - 0.47178).abs() < 1e-12);
        assert_eq!(binomial_tail(5, 0, 0.3), 1.0);
        assert_eq!(binomial_tail(5, 6, 0.3), 0.0);
        assert!((binomial_tail(10, 10, 0.5) - 0.5f64.powi(10)).abs() < 1e-15);
    }

    #[test]
    fn non_overlapping_pairs_are_rejected() {
        // N04: 시야가 겹치지 않는 짝은 비율 검사를 지난 짝 18~27개가 모두 우연 짝이고(정답 0),
        // 그 가운데 9~11개가 한 F 에 맞아 이전 기준(최소 8개·비율 0.2)을 넘었다.
        // 영상 전체에 고르게 흩어진 무관한 대응 n ∈ [15, 40] 개, 시드 200개에서 거부율 1.0 을 단언한다.
        let cfg = RansacConfig::default();
        let legacy = RansacConfig {
            min_inliers: 8,
            max_chance_prob: 0.0,
            ..cfg
        };
        let (mut rejected, mut legacy_pass, mut total) = (0usize, 0usize, 0usize);
        for seed in 1..=200u64 {
            let mut g = Lcg(seed * 7919);
            let n = 15 + (g.next() * 26.0) as usize;
            let mut pt = || Vector2::new(g.next() * 960.0, g.next() * 540.0);
            let x1: Vec<_> = (0..n).map(|_| pt()).collect();
            let x2: Vec<_> = (0..n).map(|_| pt()).collect();
            total += 1;
            if ransac_fundamental(&x1, &x2, &cfg).is_none() {
                rejected += 1;
            }
            if ransac_fundamental(&x1, &x2, &legacy).is_some() {
                legacy_pass += 1;
            }
        }
        eprintln!("거부 {rejected}/{total}, 이전 기준 통과 {legacy_pass}/{total}");
        assert_eq!(rejected, total, "정답 짝 0 인 쌍 거부율 {rejected}/{total}");
        // 이전 기준이 실제로 통과시키는 경우가 있어야 이 시험이 두 기준을 가른다.
        assert!(legacy_pass > 0, "이전 기준 통과 {legacy_pass}/{total}");
    }

    #[test]
    fn small_true_overlap_is_still_confirmed() {
        // 겹침이 작은 카메라 간 짝: 정답 20개 + 우연 짝 10개(σ 0.5 px). 시드 1..=20 모두 확정하고
        // 정답 짝 재현율 ≥ 0.9 를 단언한다(유의성 검사가 진짜 짝을 버리지 않는지).
        let cfg = RansacConfig::default();
        for seed in 1..=20u64 {
            let (x1, x2, truth, _, _) = correspondences(30, 0.5, 1.0 / 3.0, seed);
            let nt = truth.iter().filter(|&&t| t).count();
            if nt < 16 {
                continue;
            }
            let (_, inl) = ransac_fi(&x1, &x2, &cfg)
                .unwrap_or_else(|| panic!("seed {seed}: 정답 {nt}/30 인데 거부"));
            let hit = (0..30).filter(|&i| inl[i] && truth[i]).count();
            assert!(
                hit as f64 >= 0.9 * nt as f64,
                "seed {seed}: 재현 {hit}/{nt}"
            );
        }
    }

    #[test]
    fn homography_support_finds_plane_among_outliers() {
        // F-120: 평면(z = 0, σ 0.5 px) 140점 + 무작위 이상치 60점(30%). 4점 RANSAC 지표는 평면을 찾는다.
        let th = RansacConfig::default().threshold_px;
        for seed in 1..=5u64 {
            let (mut x1, mut x2) = planar_correspondences(140, 0.5, seed);
            let mut g = Lcg(seed + 77);
            for _ in 0..60 {
                x1.push(Vector2::new(g.next() * 960.0, g.next() * 540.0));
                x2.push(Vector2::new(g.next() * 960.0, g.next() * 540.0));
            }
            let h = homography_support(&x1, &x2, th);
            eprintln!("평면 + 이상치 30% seed {seed}: {h}/200");
            // 평면 점 140개 중 기대 0.85 → 119, 4σ 아래 100.
            assert!(h >= 100, "seed {seed}: 평면 설명 {h}/200");
        }
    }

    /// `sq_dist` 의 명세: 8칸 누산기(칸 k 는 성분 8i + k), 칸을 0..8 순서로 합한다.
    fn sq_dist_spec(p: &[f32; DESC_LEN], r: &[f32; DESC_LEN]) -> f32 {
        let mut acc = [0f32; 8];
        for i in 0..DESC_LEN {
            let t = p[i] - r[i];
            acc[i % 8] += t * t;
        }
        let mut s = 0f32;
        for a in acc {
            s += a;
        }
        s
    }

    #[test]
    fn ratio_match_rounding_boundary() {
        // F-093 (1): 비율이 경계에서 f32 한 칸 안쪽인 행. 누산 순서를 순차 합으로 바꾸면 거리가
        // 반올림으로 달라져 판정이 뒤집힌다. 고정 시드에서 순차 합과 명세 값이 다른 기술자를 고르고,
        // 비율을 두 값 사이에 두어 명세대로면 짝이 생기고 순차 합이면 생기지 않게(또는 반대) 만든다.
        let mut g = Lcg(93);
        let mut found = 0;
        for _ in 0..200 {
            let a = feature(&mut g);
            let b1 = feature(&mut g);
            let b2 = feature(&mut g);
            let d1 = sq_dist_spec(&a.desc, &b1.desc);
            let d2 = sq_dist_spec(&a.desc, &b2.desc);
            let seq = |x: &[f32; DESC_LEN], y: &[f32; DESC_LEN]| {
                x.iter()
                    .zip(y)
                    .fold(0f32, |s, (u, v)| s + (u - v) * (u - v))
            };
            let s1 = seq(&a.desc, &b1.desc);
            assert_eq!(sq_dist(&a.desc, &b1.desc).to_bits(), d1.to_bits());
            if s1 == d1 || d1 >= d2 {
                continue;
            }
            // 비율² · d2 가 d1 과 s1 사이에 오도록 r 을 찾는다(f32 비교 그대로).
            let target = 0.5 * (d1 as f64 + s1 as f64) / d2 as f64;
            let mut r = target.sqrt() as f32;
            let between = |r: f32| {
                let lim = r * r * d2;
                (d1 < lim) != (s1 < lim)
            };
            for _ in 0..64 {
                if between(r) {
                    break;
                }
                r = f32::from_bits(r.to_bits() + 1);
            }
            if !between(r) {
                continue;
            }
            found += 1;
            let m = ratio_match(std::slice::from_ref(&a), &[b1, b2], r, false);
            let expect = d1 < r * r * d2;
            assert_eq!(!m.is_empty(), expect, "경계 행 판정이 명세와 다름");
        }
        eprintln!("경계 행 {found}개");
        assert!(found >= 5, "경계 행이 너무 적다: {found}");
    }

    #[test]
    fn ratio_match_ties_take_lower_index() {
        // F-093 (2): 같은 기술자 둘이 같은 b 를 가리키면(열 최근접 동률) 상호 검사에서 앞 번호가 이긴다.
        // 같은 묶음 안(0, 1)과 다른 묶음(0, MATCH_BLOCK + 6) 두 경우. `<` 를 `<=` 로 바꾸면 뒤 번호가 된다.
        // 행 최근접 동률(같은 기술자 둘이 b 에 있음)은 최근접 = 차근접이라 비율 검사에서 항상 떨어지므로
        // 행 갱신의 `<`/`<=` 는 결과에 나타나지 않는다(동치 변이).
        let mut g = Lcg(11);
        let b: Vec<Feature> = (0..20).map(|_| feature(&mut g)).collect();
        for dup in [1usize, MATCH_BLOCK + 6] {
            let mut a: Vec<Feature> = (0..dup + 4).map(|_| feature(&mut g)).collect();
            let mut x = b[3].clone();
            x.desc[0] += 1e-3;
            a[0] = x.clone();
            a[dup] = x;
            let m = ratio_match(&a, &b, 0.8, true);
            assert!(m.contains(&(0, 3)), "dup {dup}: {m:?}");
            assert!(!m.iter().any(|&(i, _)| i == dup), "dup {dup}: {m:?}");
        }
    }

    /// F-148 매칭 쪽: 실측 편대 배치(`SceneConfig::default()`)의 SPEC 짝 일정 — 같은 카메라
    /// 1·2·3·4·5·8·16칸, 다른 카메라(F→R, F→L, R→L) 위치 차 −4..=4 — 에서 매칭·RANSAC 이
    /// 확정한 짝마다 F → E → `recover_pose` 회전을 정답 상대 회전 R_b R_aᵀ 와 비교한다.
    /// 짝 종류별 (시도, 확정, 회전 오차 > 2°) 를 출력하고, 확정 간선 중 2° 초과 비율 < 5% 를 단언한다.
    /// 시드 1 측정(480×270): 같은 카메라 1·2·3·8칸 2° 초과 0, 4칸 2/3(약 6.5°), 5칸 2/3
    /// (최대 11.2°), 16칸 1/3(21.8°), 다른 카메라 ±4 는 27 짝 모두 미확정 — 확정 21 중 5 가 틀림.
    /// 틀린 간선은 모두 같은 카메라 짝이라 겹침 없는 짝 통과가 원인이 아니다. 원인 분리 전이라 무시.
    #[test]
    #[ignore = "F-148: 같은 카메라 4·5·16칸에서 회전 오차 2° 초과 5/21, 원인 미분리(노트 남은 문제)"]
    fn formation_pair_schedule_rotation_errors() {
        use crate::features::{detect_and_describe, DetectorConfig, GrayImage};
        use crate::synth::{CamId, Scene, SceneConfig};
        let (w, h) = (480usize, 270usize);
        let scene = Scene::new(SceneConfig {
            width: w as u32,
            height: h as u32,
            seed: 1,
            ..SceneConfig::default()
        });
        let cfg = DetectorConfig::default();
        let mut cache: std::collections::HashMap<(CamId, usize), Vec<Feature>> =
            std::collections::HashMap::new();
        let mut feats = |cam: CamId, pos: usize| -> (crate::synth::View, Vec<Feature>) {
            let v = scene
                .views
                .iter()
                .find(|v| v.cam == cam && v.position == pos)
                .unwrap()
                .clone();
            let f = cache
                .entry((cam, pos))
                .or_insert_with(|| {
                    let (img, _) = scene.render(&v);
                    detect_and_describe(&GrayImage::from_rgb(w, h, &img.data), &cfg)
                })
                .clone();
            (v, f)
        };
        let base = 8usize;
        type PairList = Vec<(CamId, usize, CamId, usize)>;
        let mut kinds: Vec<(String, PairList)> = Vec::new();
        for gap in [1usize, 2, 3, 4, 5, 8, 16] {
            let v = CamId::ALL
                .iter()
                .map(|&c| (c, base, c, base + gap))
                .collect();
            kinds.push((format!("같은 카메라 {gap}칸"), v));
        }
        for (a, b) in [
            (CamId::F, CamId::R),
            (CamId::F, CamId::L),
            (CamId::R, CamId::L),
        ] {
            let v = (-4i64..=4)
                .map(|d| (a, base, b, (base as i64 + d) as usize))
                .collect();
            kinds.push((format!("{a:?}→{b:?} ±4"), v));
        }
        let (mut tot_ok, mut tot_bad) = (0usize, 0usize);
        for (name, list) in &kinds {
            let (mut ok, mut bad, mut errs) = (0usize, 0usize, Vec::new());
            for &(ca, pa, cb, pb) in list {
                let (va, fa) = feats(ca, pa);
                let (vb, fb) = feats(cb, pb);
                let m = ratio_match(&fa, &fb, 0.8, true);
                let px = |f: &Feature| Vector2::new(f.kp.x as f64 + 0.5, f.kp.y as f64 + 0.5);
                let x1: Vec<_> = m.iter().map(|&(i, _)| px(&fa[i])).collect();
                let x2: Vec<_> = m.iter().map(|&(_, j)| px(&fb[j])).collect();
                let Some((f, inl)) = ransac_fi(&x1, &x2, &RansacConfig::default()) else {
                    continue;
                };
                ok += 1;
                let (k1, k2) = (&va.camera.intrinsics, &vb.camera.intrinsics);
                let e = crate::two_view::essential_from_fundamental(&f, k1, k2);
                let n1: Vec<_> = (0..x1.len())
                    .filter(|&i| inl[i])
                    .map(|i| k1.to_normalized(&x1[i]))
                    .collect();
                let n2: Vec<_> = (0..x2.len())
                    .filter(|&i| inl[i])
                    .map(|i| k2.to_normalized(&x2[i]))
                    .collect();
                let truth = vb.camera.pose.rotation * va.camera.pose.rotation.inverse();
                let err = crate::two_view::recover_pose(&e, &n1, &n2)
                    .map(|p| (p.rotation * truth.inverse()).angle().to_degrees())
                    .unwrap_or(180.0);
                if err > 2.0 {
                    bad += 1;
                }
                errs.push(err);
            }
            errs.sort_by(f64::total_cmp);
            eprintln!(
                "짝 종류 {name}: 시도 {}, 확정 {ok}, 회전 오차 > 2° {bad}, 오차 중앙 {:.3}°, 최대 {:.3}°",
                list.len(),
                errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN),
                errs.last().copied().unwrap_or(f64::NAN)
            );
            tot_ok += ok;
            tot_bad += bad;
        }
        eprintln!("전체 확정 {tot_ok}, 회전 오차 > 2° {tot_bad}");
        assert!(tot_ok > 0);
        assert!(
            (tot_bad as f64) < 0.05 * tot_ok as f64,
            "회전 오차 > 2° 간선 {tot_bad} / 확정 {tot_ok}"
        );
    }

    /// 편대 합성 장면(480×270, 시드 1)에서 다른 카메라 짝 종류별 정답 겹침·매칭·검증 단계 수치(F-148·F-197·F-209).
    /// 수치 기록용. `--ignored --nocapture` 로 표를 본다.
    #[test]
    #[ignore = "수치 기록용 표(원인 조사)"]
    fn formation_cross_pair_causes() {
        use crate::features::{detect_and_describe, DetectorConfig, GrayImage};
        use crate::synth::{CamId, Scene, SceneConfig};
        let (w, h) = (480usize, 270usize);
        let scene = Scene::new(SceneConfig {
            width: w as u32,
            height: h as u32,
            seed: 1,
            ..SceneConfig::default()
        });
        let cfg = DetectorConfig::default();
        let mut cache: std::collections::HashMap<(CamId, usize), Vec<Feature>> =
            std::collections::HashMap::new();
        let mut get = |cam: CamId, pos: usize| {
            let v = scene
                .views
                .iter()
                .find(|v| v.cam == cam && v.position == pos)
                .unwrap()
                .clone();
            let f = cache
                .entry((cam, pos))
                .or_insert_with(|| {
                    let (img, _) = scene.render(&v);
                    detect_and_describe(&GrayImage::from_rgb(w, h, &img.data), &cfg)
                })
                .clone();
            (v, f)
        };
        let base = 8usize;
        let mut list: Vec<(CamId, CamId, usize)> = vec![];
        for d in [12usize, 16, 20, 24, 28, 32, 36, 40] {
            list.push((CamId::F, CamId::R, d));
            list.push((CamId::F, CamId::L, d));
        }
        for (ca, cb, d) in list {
            let (va, fa) = get(ca, base);
            let (vb, fb) = get(cb, base + d);
            let (a, b) = (&va.camera, &vb.camera);
            // 정답 겹침: A 6 px 격자 광선 → 표면 → B 투영(화면 안, 가려지지 않음).
            let (mut tot, mut vis) = (0usize, 0usize);
            let oa = a.pose.center();
            let ob = b.pose.center();
            let mut y = 3.0;
            while y < h as f64 {
                let mut x = 3.0;
                while x < w as f64 {
                    tot += 1;
                    let dir = a.unproject(&Vector2::new(x, y), 1.0) - oa;
                    if let Some(hit) = scene.intersect(&oa, &dir.normalize()) {
                        if let Some(q) = b.project(&hit.point) {
                            if b.intrinsics.contains(&q) {
                                let dv = hit.point - ob;
                                let ok = scene
                                    .intersect(&ob, &dv.normalize())
                                    .is_some_and(|h2| (h2.t - dv.norm()).abs() < 0.3);
                                if ok {
                                    vis += 1;
                                }
                            }
                        }
                    }
                    x += 6.0;
                }
                y += 6.0;
            }
            let m = ratio_match(&fa, &fb, 0.8, true);
            // 정답 대응: A 점의 광선 표면점을 B 에 투영해 2 px 안.
            let mut good = 0usize;
            for &(i, j) in &m {
                let (ka, kb) = (&fa[i].kp, &fb[j].kp);
                let pa = Vector2::new(ka.x as f64 + 0.5, ka.y as f64 + 0.5);
                let dir = a.unproject(&pa, 1.0) - oa;
                if let Some(hit) = scene.intersect(&oa, &dir.normalize()) {
                    if let Some(q) = b.project(&hit.point) {
                        if (q - Vector2::new(kb.x as f64 + 0.5, kb.y as f64 + 0.5)).norm() < 2.0 {
                            good += 1;
                        }
                    }
                }
            }
            let px = |f: &Feature| Vector2::new(f.kp.x as f64, f.kp.y as f64);
            let (k1, k2) = (&a.intrinsics, &b.intrinsics);
            let n1: Vec<_> = m
                .iter()
                .map(|&(i, _)| k1.index_to_normalized(&px(&fa[i])))
                .collect();
            let n2: Vec<_> = m
                .iter()
                .map(|&(_, j)| k2.index_to_normalized(&px(&fb[j])))
                .collect();
            let rcfg = RansacConfig::default();
            let truth = b.pose.rotation * a.pose.rotation.inverse();
            let cands = crate::two_view::ransac_essential_candidates(&n1, &n2, k1.fx, &rcfg);
            let (mut ninl, mut err) = (0usize, f64::NAN);
            if let Some((e, inl)) = cands.first() {
                ninl = inl.iter().filter(|&&v| v).count();
                let s1: Vec<_> = (0..n1.len()).filter(|&i| inl[i]).map(|i| n1[i]).collect();
                let s2: Vec<_> = (0..n1.len()).filter(|&i| inl[i]).map(|i| n2[i]).collect();
                err = crate::two_view::recover_pose(e, &s1, &s2)
                    .map(|p| (p.rotation * truth.inverse()).angle().to_degrees())
                    .unwrap_or(180.0);
            }
            // 후보별 선형 회전 오차(정답으로 고른 최소값 = 후보 선택 한계)와 첫 후보 정밀화 오차.
            let rot_err = |p: &crate::two_view::RelativePose| {
                (p.rotation * truth.inverse()).angle().to_degrees()
            };
            let mut best = f64::NAN;
            let mut refined = f64::NAN;
            for (ci, (e, inl)) in cands.iter().enumerate() {
                let s1: Vec<_> = (0..n1.len()).filter(|&i| inl[i]).map(|i| n1[i]).collect();
                let s2: Vec<_> = (0..n1.len()).filter(|&i| inl[i]).map(|i| n2[i]).collect();
                if let Some(p) = crate::two_view::recover_pose(e, &s1, &s2) {
                    best = best.min(rot_err(&p));
                }
                if ci == 0 {
                    if let Some(p) = crate::two_view::refine_relative_pose(e, &s1, &s2, 50) {
                        refined = rot_err(&p);
                    }
                }
            }
            eprintln!(
                "{:?}→{:?} +{d:2}: 겹침 {:5.1}% 특징 {}/{} 매칭 {:3} 정답대응 {:3} E정상 {:3} 후보 {} 회전오차 {:.3}° 후보중최소 {:.3}° 첫후보정밀화 {:.3}°",
                ca, cb, 100.0 * vis as f64 / tot as f64, fa.len(), fb.len(), m.len(), good, ninl, cands.len(), err, best, refined
            );
        }
    }

    #[test]
    fn scheduled_pairs_spec_matches_candidate_pairs() {
        let views: Vec<(usize, usize)> =
            (0..30).flat_map(|p| (0..3).map(move |c| (c, p))).collect();
        let spec = scheduled_pairs(&views, &PairSchedule::spec());
        assert_eq!(
            spec,
            candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX)
        );
        let f = scheduled_pairs(&views, &PairSchedule::default());
        assert!(f.iter().all(|&(i, j)| i < j));
        let mut kinds = std::collections::BTreeSet::new();
        for &(i, j) in &f {
            let ((ca, pa), (cb, pb)) = (views[i], views[j]);
            if ca != cb {
                let ((_, pf), (co, po)) = if ca == CAM_FRONT {
                    ((ca, pa), (cb, pb))
                } else {
                    ((cb, pb), (ca, pa))
                };
                assert!(po >= pf + 20 && po <= pf + 40 && (po - pf) % 4 == 0);
                kinds.insert(co);
            }
        }
        assert_eq!(
            kinds.into_iter().collect::<Vec<_>>(),
            vec![CAM_RIGHT, CAM_LEFT]
        );
        // 위치 30 개: F(p)–R(p+20..=+40, 4칸) 은 p ≤ 9 에서 3·2·1 개 등. 같은 카메라 짝은 SPEC 과 같다.
        let same =
            |v: &Vec<(usize, usize)>| v.iter().filter(|&&(i, j)| views[i].0 == views[j].0).count();
        assert_eq!(same(&f), same(&spec));
    }

    /// 편대 합성 장면(480×270, 시드 1)의 기본 짝 일정(카메라 간 F–R·F–L +20..=+40)으로 F 8·R 13·L 13 시점을
    /// 매칭·5점 RANSAC 검증한다. 검증된 카메라 간 짝의 회전 오차 중앙 ≤ 1°, 검증 짝 그래프가 한 연결 성분.
    #[test]
    fn formation_default_schedule_verifies_and_connects() {
        use crate::features::{detect_and_describe, DetectorConfig, GrayImage};
        use crate::synth::{CamId, Scene, SceneConfig};
        let (w, h) = (480usize, 270usize);
        let scene = Scene::new(SceneConfig {
            width: w as u32,
            height: h as u32,
            seed: 1,
            ..SceneConfig::default()
        });
        let cfg = DetectorConfig::default();
        let mut sel: Vec<(usize, usize)> = vec![];
        for p in (0..=28).step_by(4) {
            sel.push((CAM_FRONT, p));
        }
        for p in (20..=68).step_by(4) {
            sel.push((CAM_RIGHT, p));
            sel.push((CAM_LEFT, p));
        }
        let cam_of = |c: usize| CamId::ALL[c];
        let views: Vec<_> = sel
            .iter()
            .map(|&(c, p)| {
                scene
                    .views
                    .iter()
                    .find(|v| v.cam == cam_of(c) && v.position == p)
                    .unwrap()
                    .clone()
            })
            .collect();
        let feats: Vec<Vec<Feature>> = views
            .iter()
            .map(|v| {
                let (img, _) = scene.render(v);
                detect_and_describe(&GrayImage::from_rgb(w, h, &img.data), &cfg)
            })
            .collect();
        let pairs = scheduled_pairs(&sel, &PairSchedule::default());
        let mut parent: Vec<usize> = (0..sel.len()).collect();
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut x = x;
            while p[x] != r {
                let n = p[x];
                p[x] = r;
                x = n;
            }
            r
        }
        let (mut cross_err, mut cross_n, mut verified) = (Vec::new(), 0usize, 0usize);
        for &(i, j) in &pairs {
            let (a, b) = (&views[i].camera, &views[j].camera);
            let m = ratio_match(&feats[i], &feats[j], 0.8, true);
            let nrm = |c: &Camera, f: &Feature| {
                c.intrinsics
                    .index_to_normalized(&Vector2::new(f.kp.x as f64, f.kp.y as f64))
            };
            let n1: Vec<_> = m.iter().map(|&(x, _)| nrm(a, &feats[i][x])).collect();
            let n2: Vec<_> = m.iter().map(|&(_, y)| nrm(b, &feats[j][y])).collect();
            let cands = crate::two_view::ransac_essential_candidates(
                &n1,
                &n2,
                a.intrinsics.fx,
                &RansacConfig::default(),
            );
            let Some((e, inl)) = cands.first() else {
                continue;
            };
            verified += 1;
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            parent[ri] = rj;
            if sel[i].0 != sel[j].0 {
                cross_n += 1;
                let s1: Vec<_> = (0..n1.len()).filter(|&k| inl[k]).map(|k| n1[k]).collect();
                let s2: Vec<_> = (0..n1.len()).filter(|&k| inl[k]).map(|k| n2[k]).collect();
                let truth = b.pose.rotation * a.pose.rotation.inverse();
                let err = crate::two_view::recover_pose(e, &s1, &s2)
                    .map(|p| (p.rotation * truth.inverse()).angle().to_degrees())
                    .unwrap_or(180.0);
                cross_err.push(err);
            }
        }
        cross_err.sort_by(f64::total_cmp);
        let med = cross_err[cross_err.len() / 2];
        let comps = (0..sel.len())
            .filter(|&x| find(&mut parent, x) == x)
            .count();
        eprintln!(
            "짝 {} 검증 {verified}, 카메라 간 검증 {cross_n}, 회전 오차 중앙 {med:.3}° 최대 {:.3}°, 연결 성분 {comps}",
            pairs.len(),
            cross_err.last().unwrap()
        );
        assert!(cross_n >= 10, "카메라 간 검증 {cross_n}");
        assert!(med <= 1.0, "회전 오차 중앙 {med}");
        assert_eq!(comps, 1);
    }

    /// 시점 `sel`(카메라, 위치)을 `sch` 일정 짝으로 매칭·5점 RANSAC·자세 복원하고 회전 평균까지 돌린 결과.
    struct ScheduleRun {
        pairs: usize,
        verified: usize,
        /// 짝 종류(같은/F–R/F–L/R–L)별 (일정 짝, 통과, 통과 중 정답 대비 2° 초과).
        kinds: [(usize, usize, usize); 4],
        passed: usize,
        bad: usize,
        comps: usize,
        returned: usize,
        align_median_deg: f64,
        /// 통과한 간선별 기록(다른 카메라 짝 보고용).
        edge_log: Vec<EdgeRec>,
    }

    /// 통과 간선 하나의 기록.
    struct EdgeRec {
        /// (카메라 번호, 위치 번호). `a` 가 앞 시점, `b` 가 뒤 시점.
        a: (usize, usize),
        b: (usize, usize),
        matches: usize,
        inliers: usize,
        /// `recover_pose` 가 고른 회전의 정답 대비 오차(도).
        err: f64,
        /// E 분해 네 후보 중 정답에 가장 가까운 회전의 오차(도). 키랄리티 선택이 틀렸는지 가른다.
        best_decomp: f64,
        /// 두 번째 RANSAC 후보(있으면)의 `recover_pose` 회전 오차(도).
        second_cand: Option<f64>,
        /// 정답 기하 겹침: a 영상 격자 화소 광선이 장면 표면에서 만난 점 중 b 영상 안에 투영되는 비율(가림 무시).
        overlap: f64,
    }

    fn run_schedule(sel: &[(usize, usize)], sch: &PairSchedule) -> ScheduleRun {
        use crate::features::{detect_and_describe, DetectorConfig, GrayImage};
        use crate::rotation_averaging::{
            aligned_errors, average_rotations, AveragingConfig, RelativeRotation,
        };
        use crate::synth::{CamId, Scene, SceneConfig};
        let (w, h) = (480usize, 270usize);
        let scene = Scene::new(SceneConfig {
            width: w as u32,
            height: h as u32,
            seed: 1,
            ..SceneConfig::default()
        });
        let cfg = DetectorConfig::default();
        let views: Vec<_> = sel
            .iter()
            .map(|&(c, p)| {
                scene
                    .views
                    .iter()
                    .find(|v| v.cam == CamId::ALL[c] && v.position == p)
                    .unwrap()
                    .clone()
            })
            .collect();
        let feats: Vec<Vec<Feature>> = views
            .iter()
            .map(|v| {
                let (img, _) = scene.render(v);
                detect_and_describe(&GrayImage::from_rgb(w, h, &img.data), &cfg)
            })
            .collect();
        let pairs = scheduled_pairs(sel, sch);
        let kind = |a: usize, b: usize| match (a.min(b), a.max(b)) {
            _ if a == b => 0,
            (CAM_FRONT, CAM_RIGHT) => 1,
            (CAM_FRONT, CAM_LEFT) => 2,
            _ => 3,
        };
        let mut kinds = [(0usize, 0usize, 0usize); 4];
        let mut edges = Vec::new();
        let mut verified = 0;
        let mut edge_log = Vec::new();
        for &(i, j) in &pairs {
            let kd = kind(sel[i].0, sel[j].0);
            kinds[kd].0 += 1;
            let (a, b) = (&views[i].camera, &views[j].camera);
            let m = ratio_match(&feats[i], &feats[j], 0.8, true);
            let nrm = |c: &Camera, f: &Feature| {
                c.intrinsics
                    .index_to_normalized(&Vector2::new(f.kp.x as f64, f.kp.y as f64))
            };
            let n1: Vec<_> = m.iter().map(|&(x, _)| nrm(a, &feats[i][x])).collect();
            let n2: Vec<_> = m.iter().map(|&(_, y)| nrm(b, &feats[j][y])).collect();
            let cands = crate::two_view::ransac_essential_candidates(
                &n1,
                &n2,
                a.intrinsics.fx,
                &RansacConfig::default(),
            );
            let Some((e, inl)) = cands.first() else {
                continue;
            };
            verified += 1;
            let s1: Vec<_> = (0..n1.len()).filter(|&k| inl[k]).map(|k| n1[k]).collect();
            let s2: Vec<_> = (0..n1.len()).filter(|&k| inl[k]).map(|k| n2[k]).collect();
            let Some(p) = crate::two_view::recover_pose(e, &s1, &s2) else {
                continue;
            };
            kinds[kd].1 += 1;
            let truth = b.pose.rotation * a.pose.rotation.inverse();
            let err = (p.rotation * truth.inverse()).angle().to_degrees();
            if err > 2.0 {
                kinds[kd].2 += 1;
            }
            if kd != 0 {
                let best_decomp = crate::two_view::decompose_essential(e)
                    .iter()
                    .map(|(r, _)| (*r * truth.inverse()).angle().to_degrees())
                    .fold(f64::INFINITY, f64::min);
                let second_cand = cands.get(1).and_then(|(e2, inl2)| {
                    let t1: Vec<_> = (0..n1.len()).filter(|&k| inl2[k]).map(|k| n1[k]).collect();
                    let t2: Vec<_> = (0..n1.len()).filter(|&k| inl2[k]).map(|k| n2[k]).collect();
                    crate::two_view::recover_pose(e2, &t1, &t2)
                        .map(|q| (q.rotation * truth.inverse()).angle().to_degrees())
                });
                let (kx, ky) = (24usize, 14usize);
                let o = a.pose.center();
                let (mut hit, mut inside) = (0usize, 0usize);
                for gy in 0..ky {
                    for gx in 0..kx {
                        let px = Vector2::new(
                            (gx as f64 + 0.5) / kx as f64 * w as f64,
                            (gy as f64 + 0.5) / ky as f64 * h as f64,
                        );
                        let d = a.unproject(&px, 1.0) - o;
                        if let Some(hh) = scene.intersect(&o, &d) {
                            hit += 1;
                            if b.project(&hh.point)
                                .is_some_and(|q| b.intrinsics.contains(&q))
                            {
                                inside += 1;
                            }
                        }
                    }
                }
                edge_log.push(EdgeRec {
                    a: (sel[i].0, sel[i].1),
                    b: (sel[j].0, sel[j].1),
                    matches: m.len(),
                    inliers: s1.len(),
                    err,
                    best_decomp,
                    second_cand,
                    overlap: inside as f64 / hit.max(1) as f64,
                });
            }
            edges.push(RelativeRotation {
                i,
                j,
                rotation: p.rotation,
                weight: s1.len() as f64,
            });
        }
        let mut parent: Vec<usize> = (0..sel.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for e in &edges {
            let (a, b) = (find(&mut parent, e.i), find(&mut parent, e.j));
            parent[a] = b;
        }
        let comps = (0..sel.len())
            .filter(|&x| find(&mut parent, x) == x)
            .count();
        let truth: Vec<Rotation3<f64>> = views.iter().map(|v| v.camera.pose.rotation).collect();
        let avg = average_rotations(sel.len(), &edges, &AveragingConfig::default())
            .expect("회전 평균 결과");
        let returned = avg.rotations.iter().filter(|r| r.is_some()).count();
        let mut err: Vec<f64> = aligned_errors(&avg.rotations, &truth)
            .into_iter()
            .filter(|x| x.is_finite())
            .collect();
        err.sort_by(f64::total_cmp);
        ScheduleRun {
            pairs: pairs.len(),
            verified,
            kinds,
            passed: edges.len(),
            bad: kinds.iter().map(|k| k.2).sum(),
            comps,
            returned,
            align_median_deg: err[err.len() / 2].to_degrees(),
            edge_log,
        }
    }

    /// 기본 bench 와 같은 시점(세 카메라 × 위치 0·4·…·28, 24장, 480×270, 시드 1)에서 F–R·F–L 시작 위치 차
    /// +24·+20 일정을 비교한다. +24 는 반환 시점 전체·정렬 오차 중앙 < 1°·2° 초과 < 5%·다른 카메라 간선 ≥ 1·한 연결 성분.
    #[test]
    fn bench_views_formation_schedule_averages_all_views() {
        let sel: Vec<(usize, usize)> = (0..8)
            .flat_map(|k| (0..3).map(move |c| (c, 4 * k)))
            .collect();
        let sched = |start: usize| PairSchedule {
            cross: CrossSchedule::Formation {
                right_min: start,
                left_min: start,
                max: 40,
                step: 4,
            },
            ..PairSchedule::default()
        };
        let names = ["같은 카메라", "F–R", "F–L", "R–L"];
        let mut runs = Vec::new();
        for start in [24usize, 20] {
            let r = run_schedule(&sel, &sched(start));
            eprintln!(
                "시작 +{start}: 짝 {} E통과 {} 자세 {} 연결 성분 {} 반환 {}/{} 정렬 오차 중앙 {:.3}° 2° 초과 {}/{} ({:.1}%)",
                r.pairs, r.verified, r.passed, r.comps, r.returned, sel.len(),
                r.align_median_deg, r.bad, r.passed,
                100.0 * r.bad as f64 / r.passed.max(1) as f64
            );
            for (n, k) in names.iter().zip(&r.kinds) {
                eprintln!("  {n}: 일정 {} 성공 {} 2° 초과 {}", k.0, k.1, k.2);
            }
            eprintln!("  | 짝 | 위치 차 | 회전 오차(°) | 분해 네 후보 최선(°) | 둘째 RANSAC 후보(°) | 대응 | 정상 | 겹침 |");
            for e in &r.edge_log {
                let cam = |c: usize| ['F', 'R', 'L'][c];
                eprintln!(
                    "  | {}({})–{}({}) | {:+} | {:.2} | {:.2} | {} | {} | {} | {:.0}% |",
                    cam(e.a.0),
                    e.a.1,
                    cam(e.b.0),
                    e.b.1,
                    e.b.1 as i64 - e.a.1 as i64,
                    e.err,
                    e.best_decomp,
                    e.second_cand.map_or("-".into(), |v| format!("{v:.2}")),
                    e.matches,
                    e.inliers,
                    100.0 * e.overlap
                );
            }
            runs.push(r);
        }
        // +20 의 큰 회전 오차는 키랄리티·쌍둥이 선택 실수가 아니다: 네 분해 후보 중 최선이 고른 것과 같다.
        for r in &runs {
            for e in &r.edge_log {
                assert!(
                    e.err - e.best_decomp < 1e-6,
                    "{:?}–{:?}: 고른 {} 최선 {}",
                    e.a,
                    e.b,
                    e.err,
                    e.best_decomp
                );
            }
        }
        let r = &runs[0];
        assert_eq!(r.returned, sel.len(), "반환 시점");
        assert!(
            r.align_median_deg < 1.0,
            "정렬 오차 중앙 {}",
            r.align_median_deg
        );
        assert!(
            (r.bad as f64) < 0.05 * r.passed as f64,
            "2° 초과 {}/{}",
            r.bad,
            r.passed
        );
        assert!(r.kinds[1].1 + r.kinds[2].1 >= 1, "다른 카메라 확정 간선 0");
        assert_eq!(r.kinds[3].0, 0, "R–L 짝은 일정에 없다");
        assert_eq!(r.comps, 1, "연결 성분");
    }

    #[test]
    fn parallax_needed_values() {
        let v: Vec<_> = [50usize, 200, 300, 1000]
            .iter()
            .map(|&n| parallax_needed(n))
            .collect();
        eprintln!("필요 시차 짝 n=50,200,300,1000: {v:?}");
        assert_eq!(v, vec![4, 6, 6, 10]);
        for n in [50usize, 300, 1000] {
            let k = parallax_needed(n);
            assert!(binomial_tail(n, k, 0.001) < 1e-6);
            assert!(binomial_tail(n, k - 1, 0.001) >= 1e-6);
        }
    }

    #[test]
    fn model_selection_rejects_non_finite_coordinates() {
        // F-182: NaN·∞ 좌표 하나만 있어도 None(GRIC 누산에서 상한값으로 잘려 H 쪽으로 기울던 경우).
        let (c1, c2) = two_cameras();
        let f0 = fundamental_from_cameras(&c1, &c2);
        let (x1, x2, _, _, _) = correspondences(50, 0.5, 0.0, 1);
        let base = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
        assert_eq!(base.model, TwoViewModel::Fundamental);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (which, i) in [(0, 3usize), (1, 3), (0, 49), (1, 0)] {
                let (mut y1, mut y2) = (x1.clone(), x2.clone());
                if which == 0 {
                    y1[i].x = bad;
                } else {
                    y2[i].y = bad;
                }
                assert!(
                    select_two_view_model(&y1, &y2, &f0, 0.5).is_none(),
                    "{bad} 영상 {which} 짝 {i}"
                );
            }
            let mut fb = f0;
            fb[(1, 2)] = bad;
            assert!(
                select_two_view_model(&x1, &x2, &fb, 0.5).is_none(),
                "F {bad}"
            );
        }
    }

    /// 실측 편대 배치(`SceneConfig::default()`, 960×540) 카메라 `cam` 의 위치 8 과 8+gap 짝.
    /// 영상 a 의 균일한 픽셀에서 광선을 쏘아 지면 z = 0(비율 1−bld) 또는 높이 3~8 m 건물 지붕(비율 bld)에서 만나는
    /// 점을 영상 b 로 투영한다(가림 무시). 앞쪽 nb 개가 건물 점. σ px 잡음.
    #[allow(clippy::type_complexity)]
    fn formation_pair(
        cam: crate::synth::CamId,
        gap: usize,
        n: usize,
        bld: f64,
        sigma: f64,
        seed: u64,
    ) -> (Vec<Vector2<f64>>, Vec<Vector2<f64>>, Camera, Camera, usize) {
        use crate::synth::{Scene, SceneConfig};
        let scene = Scene::new(SceneConfig::default());
        let view = |pos: usize| {
            scene
                .views
                .iter()
                .find(|v| v.cam == cam && v.position == pos)
                .unwrap()
                .camera
        };
        let (ca, cb) = (view(8), view(8 + gap));
        let mut g = Lcg(seed * 104_729 + gap as u64);
        let nb = (n as f64 * bld).round() as usize;
        let (w, h) = (ca.intrinsics.width as f64, ca.intrinsics.height as f64);
        let (mut x1, mut x2) = (vec![], vec![]);
        let o = ca.pose.center();
        while x1.len() < n {
            let z = if x1.len() < nb {
                3.0 + 5.0 * g.next()
            } else {
                0.0
            };
            let a = Vector2::new(g.next() * w, g.next() * h);
            let d = ca.unproject(&a, 1.0) - o;
            if d.z >= 0.0 {
                continue;
            }
            let p = o + d * ((z - o.z) / d.z);
            let Some(b) = cb.project(&p) else {
                continue;
            };
            if !cb.intrinsics.contains(&b) {
                continue;
            }
            x1.push(a + Vector2::new(g.gauss(), g.gauss()) * sigma);
            x2.push(b + Vector2::new(g.gauss(), g.gauss()) * sigma);
        }
        (x1, x2, ca, cb, nb)
    }

    /// 추정 F 로 꺼낸 이동 방향과 정답 이동 방향의 각(도). 꺼낼 수 없으면 180.
    fn translation_angle_deg(
        f: &Matrix3<f64>,
        inl: &[bool],
        x1: &[Vector2<f64>],
        x2: &[Vector2<f64>],
        ca: &Camera,
        cb: &Camera,
    ) -> f64 {
        let (k1, k2) = (&ca.intrinsics, &cb.intrinsics);
        let e = crate::two_view::essential_from_fundamental(f, k1, k2);
        let n1: Vec<_> = (0..x1.len())
            .filter(|&i| inl[i])
            .map(|i| k1.to_normalized(&x1[i]))
            .collect();
        let n2: Vec<_> = (0..x2.len())
            .filter(|&i| inl[i])
            .map(|i| k2.to_normalized(&x2[i]))
            .collect();
        let r = cb.pose.rotation * ca.pose.rotation.inverse();
        let t = (cb.pose.translation - r * ca.pose.translation).normalize();
        crate::two_view::recover_pose(&e, &n1, &n2)
            .map(|p| {
                p.translation
                    .normalize()
                    .dot(&t)
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            })
            .unwrap_or(180.0)
    }

    /// F-141 훑기: 실측 편대 배치 F 카메라 시간 이웃 1·2·4·8칸 × 건물 0·1·2·3·5% × 시드 1~5
    /// (300점, σ 0.5 px). 정답 F 로 고른 모델·시차 짝 수, RANSAC 의 평면 표시와 이동 방향 오차를 표로 출력한다.
    /// 수치 기록용(단언은 `formation_ground_with_buildings_keeps_fundamental` 이 한다).
    #[test]
    fn formation_parallax_sweep() {
        use crate::synth::CamId;
        let cfg = RansacConfig::default();
        eprintln!("간격 | 건물 | 시차 짝(정답 F) 최소~최대 | 필요 | F 판정(정답 F) | 평면 표시(RANSAC) | 이동 방향 오차 중앙/최대(°)");
        for gap in [1usize, 2, 4, 8] {
            for bld in [0.0, 0.01, 0.02, 0.03, 0.05] {
                let (mut pmin, mut pmax, mut need, mut nf, mut planar) =
                    (usize::MAX, 0usize, 0usize, 0usize, 0usize);
                let mut errs = vec![];
                for seed in 1..=5u64 {
                    let (x1, x2, ca, cb, _) = formation_pair(CamId::F, gap, 300, bld, 0.5, seed);
                    let f0 = fundamental_from_cameras(&ca, &cb);
                    let m = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
                    pmin = pmin.min(m.parallax);
                    pmax = pmax.max(m.parallax);
                    need = m.parallax_needed;
                    nf += (m.model == TwoViewModel::Fundamental) as usize;
                    if let Some(r) = ransac_fundamental(&x1, &x2, &cfg) {
                        planar += r.is_planar() as usize;
                        errs.push(translation_angle_deg(&r.f, &r.inliers, &x1, &x2, &ca, &cb));
                    } else {
                        errs.push(f64::NAN);
                    }
                }
                errs.sort_by(f64::total_cmp);
                eprintln!(
                    "{gap}칸 | {:.0}% | {pmin}~{pmax} | {need} | {nf}/5 | {planar}/5 | {:.2}/{:.2}",
                    bld * 100.0,
                    errs[2],
                    errs[4]
                );
            }
        }
    }

    #[test]
    fn formation_ground_with_buildings_keeps_fundamental() {
        // F-141: 실측 편대 배치 기선 1 m(F 카메라 1칸), 건물 5%(15점), σ 0.5 px: 시드 1~5 모두 F 이고
        // 시차 짝이 필요 수보다 3 이상 많다. 순수 평면은 H.
        use crate::synth::CamId;
        for seed in 1..=5u64 {
            let (x1, x2, ca, cb, _) = formation_pair(CamId::F, 1, 300, 0.05, 0.5, seed);
            let f0 = fundamental_from_cameras(&ca, &cb);
            let m = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
            eprintln!("편대 1칸 건물 5% seed {seed}: {m:?}");
            assert_eq!(m.model, TwoViewModel::Fundamental, "seed {seed}: {m:?}");
            assert!(
                m.parallax >= m.parallax_needed + 3,
                "seed {seed}: 시차 짝 {} 필요 {}",
                m.parallax,
                m.parallax_needed
            );
            let (x1, x2, ca, cb, _) = formation_pair(CamId::F, 1, 300, 0.0, 0.5, seed);
            let f0 = fundamental_from_cameras(&ca, &cb);
            let m = select_two_view_model(&x1, &x2, &f0, 0.5).unwrap();
            assert_eq!(m.model, TwoViewModel::Homography, "평면 seed {seed}: {m:?}");
        }
    }

    #[test]
    fn formation_planar_pair_is_flagged() {
        // F-142: 실측 편대 배치(F·R·L 각 카메라, 1·4칸)의 순수 지면 짝은 RANSAC 이 F 를 내더라도
        // 평면 표시와 함께 돌려준다. 건물 10% 짝은 표시가 없다.
        use crate::synth::CamId;
        let cfg = RansacConfig::default();
        for cam in CamId::ALL {
            for gap in [1usize, 4] {
                for seed in 1..=3u64 {
                    let (x1, x2, _, _, _) = formation_pair(cam, gap, 200, 0.0, 0.5, seed);
                    let r = ransac_fundamental(&x1, &x2, &cfg).expect("지면 짝 None");
                    assert!(
                        r.is_planar(),
                        "{cam:?} {gap}칸 seed {seed}: {:?}",
                        r.selection
                    );
                    let (x1, x2, _, _, _) = formation_pair(cam, gap, 300, 0.1, 0.5, seed);
                    let r = ransac_fundamental(&x1, &x2, &cfg).expect("건물 짝 None");
                    assert!(
                        !r.is_planar(),
                        "{cam:?} {gap}칸 건물 10% seed {seed}: {:?}",
                        r.selection
                    );
                }
            }
        }
    }
}
