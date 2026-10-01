//! 특징 매칭과 기하 검증: 비율 검사, 정규화 8점 기본 행렬, RANSAC.

use crate::features::{Feature, DESC_LEN};
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};
use rayon::prelude::*;

/// SPEC 기본값: 같은 카메라 시간 이웃 위치 차 1..=5.
pub const PAIR_TEMPORAL: usize = 5;
/// SPEC 기본값: 다른 카메라 위치 차 0..=4.
pub const PAIR_CROSS: usize = 4;
/// 같은 카메라 2의 거듭제곱 간격 상한 기본값(제한 없음).
pub const PAIR_POW2_MAX: usize = usize::MAX;

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
///
/// 거리 행렬을 한 번만 계산한다: `a` 묶음마다(rayon 병렬) 모든 `b` 와의 거리로 행 최근접·차근접과
/// 열(b→a) 최근접을 함께 갱신하고, 열 최근접은 묶음 순서대로 합친다. 같은 거리면 앞 인덱스가 이긴다.
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
        if j == usize::MAX || d1 >= ratio * ratio * d2 {
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
/// 결과는 계수 2, 프로베니우스 노름 1.
pub fn fundamental_8pt(x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> Option<Matrix3<f64>> {
    if x1.len() < 8 || x1.len() != x2.len() || !all_finite(x1) || !all_finite(x2) {
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

/// 주어진 대응에서 Sampson 거리 제곱합을 줄이도록 F 를 Levenberg–Marquardt 로 정밀화한다
/// (Hartley & Zisserman 11.4.3 의 Sampson 비용). 매개변수는 Hartley 정규화 좌표의 F 성분 9개이고,
/// 걸음마다 계수 2·노름 1 로 투영해 7 자유도를 유지한다. 비용이 줄지 않으면 시작 F 를 그대로 돌려준다.
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
    let to_px = |g: &Matrix3<f64>| t2.transpose() * g * t1;
    // 픽셀 Sampson 잔차(부호 있는 거리).
    let resid = |g: &Matrix3<f64>, out: &mut Vec<f64>| {
        let fp = to_px(g);
        out.clear();
        for (p, q) in x1.iter().zip(x2) {
            let a = Vector3::new(p.x, p.y, 1.0);
            let b = Vector3::new(q.x, q.y, 1.0);
            let fa = fp * a;
            let ftb = fp.transpose() * b;
            let den = (fa.x * fa.x + fa.y * fa.y + ftb.x * ftb.x + ftb.y * ftb.y).sqrt();
            out.push(if den > 0.0 { b.dot(&fa) / den } else { 0.0 });
        }
    };
    let Some(mut g) = rank2_unit(&(t2i.transpose() * f * t1i)) else {
        return *f;
    };
    let (mut r, mut rh) = (Vec::new(), Vec::new());
    resid(&g, &mut r);
    let mut cost: f64 = r.iter().map(|e| e * e).sum();
    let mut lambda = 1e-3;
    let m = r.len();
    let mut jac = vec![[0.0f64; 9]; m];
    for _ in 0..iters {
        let h = 1e-7;
        for k in 0..9 {
            let mut gh = g;
            gh[(k / 3, k % 3)] += h;
            resid(&gh, &mut rh);
            for i in 0..m {
                jac[i][k] = (rh[i] - r[i]) / h;
            }
        }
        let mut jtj = SMatrix::<f64, 9, 9>::zeros();
        let mut jtr = SMatrix::<f64, 9, 1>::zeros();
        for i in 0..m {
            let row = SMatrix::<f64, 9, 1>::from_column_slice(&jac[i]);
            jtj += row * row.transpose();
            jtr += row * r[i];
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut a = jtj;
            for k in 0..9 {
                a[(k, k)] += lambda * (jtj[(k, k)] + 1e-12);
            }
            let Some(d) = a.cholesky().map(|c| c.solve(&(-jtr))) else {
                lambda *= 10.0;
                continue;
            };
            let step = Matrix3::new(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]);
            let Some(cand) = rank2_unit(&(g + step)) else {
                lambda *= 10.0;
                continue;
            };
            resid(&cand, &mut rh);
            let c: f64 = rh.iter().map(|e| e * e).sum();
            if c < cost {
                g = cand;
                std::mem::swap(&mut r, &mut rh);
                let rel = (cost - c) / cost.max(1e-300);
                cost = c;
                lambda = (lambda * 0.1).max(1e-9);
                improved = rel > 1e-10;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    let fp = to_px(&g);
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
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            threshold_px: 1.5,
            min_inlier_ratio: 0.2,
            max_iters: 2000,
            confidence: 0.999,
            seed: 1,
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

/// RANSAC(Fischler & Bolles 1981) + 8점 기본 행렬로 기하 검증.
/// 반환: (정상 짝으로 다시 맞춘 F, 정상 여부 표시).
/// 정상 짝이 8개 미만이거나 정상 비율이 `min_inlier_ratio` 미만이면 None.
/// 길이가 다르거나 유한하지 않은 좌표가 있으면 None.
pub fn ransac_fundamental(
    x1: &[Vector2<f64>],
    x2: &[Vector2<f64>],
    cfg: &RansacConfig,
) -> Option<(Matrix3<f64>, Vec<bool>)> {
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
        // 최고 가설의 절반 이상을 설명하는 가설은 정상 짝 전체로 다시 맞춰 개선이 멈출 때까지 반복한다.
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
    (cnt >= 8 && cnt as f64 >= cfg.min_inlier_ratio * n as f64).then_some((f, inl))
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
            let (f, inl) = ransac_fundamental(&x1, &x2, &RansacConfig::default()).unwrap();
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
                        let est = ransac_fundamental(&x1, &x2, &cfg).map(|(_, inl)| pr(&inl));
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
            vec![1, 2, 3, 4, 5, 8, 16, 32, 64]
        );
        // 같은 카메라: (79+78+77+76+75) + (72+64+48+16) = 585, 카메라 3대 → 1755.
        // 다른 카메라: 카메라 짝마다 80 + 2(79+78+77+76) = 700, 3 짝 → 2100.
        assert_eq!(pairs.len(), 1755 + 2100);
        // 상한 16 이면 32·64 간격이 빠진다: 카메라마다 48 + 16 = 64 짝 감소.
        assert_eq!(
            candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, 16).len(),
            1755 + 2100 - 3 * 64
        );
    }

    /// 합성 드론 장면 같은 카메라 두 장: 특징 → 비율 매칭 → RANSAC.
    /// 정답 깊이로 각 짝의 참·거짓을 판정해 (RANSAC 전 정답 비율, 후 정밀도, 재현율, 정상 수).
    fn scene_pair(step: usize) -> (f64, f64, f64, usize) {
        scene_pair_cams(crate::synth::CamId::F, crate::synth::CamId::F, step)
    }

    fn scene_pair_cams(
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
            ..SceneConfig::default()
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
        let Some((_, inl)) = ransac_fundamental(&x1, &x2, &RansacConfig::default()) else {
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

    #[test]
    fn ransac_on_synthetic_drone_views() {
        for step in [1usize, 3] {
            let (before, prec, rec, ni) = scene_pair(step);
            eprintln!(
                "scene step={step} correct_before={before:.3} precision={prec:.3} recall={rec:.3} inliers={ni}"
            );
            assert!(ni >= 250, "정상 수 {ni}");
            assert!(prec >= 0.98, "정밀도 {prec}");
            assert!(rec >= 0.98, "재현율 {rec}");
        }
    }

    #[test]
    fn ransac_on_cross_camera_views() {
        // 앞 카메라 위치 0 과 옆 카메라 위치 0·4(10 m 앞): 시선이 90° 다른 짝.
        use crate::synth::CamId;
        for b in [CamId::R, CamId::L] {
            for step in [0usize, 4] {
                let (before, prec, rec, ni) = scene_pair_cams(CamId::F, b, step);
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
            if d1 >= ratio * ratio * d2 {
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
            let cnt = ransac_fundamental(&x1, &x2, &loose)
                .map(|(_, inl)| inl.iter().filter(|&&b| b).count())
                .unwrap_or(0);
            eprintln!("seed={seed} 우연 정상 수={cnt}/200");
            assert!(cnt < 40, "우연 정상 비율이 0.2 이상: {cnt}/200");
            assert!(r.is_none(), "무작위 대응에서 F 확정");
        }
    }
}
