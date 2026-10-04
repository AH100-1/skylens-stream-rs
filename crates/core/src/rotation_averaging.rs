//! 회전 평균: 상대 회전 그래프 → 전역 회전.
//!
//! 회전 R_i 는 세계→카메라 i. 간선 (i, j) 의 상대 회전은 R_ij = R_j R_iᵀ
//! (`two_view::RelativePose::rotation` 과 같은 규약: x_j = R_ij x_i + t).
//!
//! 0. 쓸 수 있는 간선으로 연결 성분을 나누고 가장 큰 성분(정점 수, 같으면 간선 가중합, 그래도 같으면
//!    가장 작은 번호 정점이 든 성분)만 푼다. 나머지 정점은 None.
//! 1. 다수결 탐욕 초기화: 이미 놓인 정점과 간선이 가장 많은 정점부터 놓는다. 그 정점의 회전은
//!    놓인 이웃들이 예측한 회전 가운데 서로 `init_agree_rad` 안에서 가장 많이 일치하는 무리의 평균이다.
//!    일치 문턱은 이상치 문턱과 따로 둔다: σ 2° 잡음에서 이웃 예측끼리 이미 4~5° 벌어지므로
//!    이상치 하한만큼 좁으면 정상 무리가 쪼개져 이상치 무리가 이긴다.
//!    시작 정점을 여러 개 바꿔 보고 문턱 안 간선이 가장 많은 초기값을 고른다.
//! 2. 정점마다 이웃 예측의 가중 현(chordal) 평균을 SO(3) 로 사영하는 가우스–자이델 반복.
//!    가중치는 코시형 1/(1 + (r/σ)²) 로 이상치 간선을 누른다.
//!    강건 단계는 이상치 판별용이므로 변화량이 잡음 수준(`tolerance_rad`)으로 내려가면 멈춘다.
//! 3. 이상치 문턱을 잔차 분포에서 정한다: 문턱 = clamp(k·σ̂, `outlier_threshold_rad`, `outlier_cap_rad`),
//!    σ̂ = 중앙값(잔차)/1.5382(축당 σ 정규 잡음의 각 크기는 자유도 3 카이 분포, 중앙값 1.5382σ).
//!    상한에 걸리거나 정상 비율이 `min_inlier_ratio` 아래면 결과의 `reliable` 을 거짓으로 둔다.
//!    문턱을 넘는 간선을 이상치로 빼고, 남은 간선으로 리 대수 선형화 가중 최소제곱을 몇 번 반복한다.
//!    정규 방정식은 간선마다 3×3 블록인 희소 블록 행렬로 모으고 블록 야코비 전처리 켤레 기울기로 푼다.
//!    긴 사슬에서 느린 저주파 오차를 한 번에 없앤다.
//!
//! 전역 회전은 세계 좌표 회전 하나만큼 정해지지 않는다. 기준 정점(고른 성분 안 가장 작은 번호)을
//! 단위 회전으로 둔다.

use crate::math::{Matrix3, Rotation3, Vector3};
#[cfg(test)]
use nalgebra::{DMatrix, DVector};

/// 상대 회전 관측 하나.
#[derive(Clone, Debug)]
pub struct RelativeRotation {
    pub i: usize,
    pub j: usize,
    /// R_ij = R_j R_iᵀ.
    pub rotation: Rotation3<f64>,
    /// 신뢰도(예: 정상 대응 수). 0 이하·NaN 인 간선은 쓰지 않는다.
    pub weight: f64,
}

/// 회전 평균 설정.
#[derive(Clone, Debug)]
pub struct AveragingConfig {
    /// 가우스–자이델 최대 반복 수.
    pub max_iterations: usize,
    /// 한 번 훑는 동안 가장 큰 회전 변화(rad)가 이보다 작으면 멈춘다.
    pub tolerance_rad: f64,
    /// 코시 가중치 축척 σ(rad).
    pub robust_scale_rad: f64,
    /// 이상치 문턱의 하한(rad). 실제 문턱은 clamp(`outlier_sigma_factor`·σ̂, 이 값, `outlier_cap_rad`).
    pub outlier_threshold_rad: f64,
    /// 이상치 문턱의 상한(rad). 추정 문턱이 이를 넘으면 상한을 쓰고 결과를 믿을 수 없다고 표시한다.
    pub outlier_cap_rad: f64,
    /// 다수결 초기화에서 이웃 예측끼리 같은 무리로 보는 각(rad). 이상치 문턱과 별개.
    pub init_agree_rad: f64,
    /// 정상 간선 비율이 이보다 낮으면 결과를 믿을 수 없다고 표시한다.
    pub min_inlier_ratio: f64,
    /// 잔차에서 추정한 잡음 σ̂ 의 몇 배를 이상치 문턱으로 할지. 0 이면 하한 고정 문턱.
    pub outlier_sigma_factor: f64,
    /// 초기화 시작 정점 후보 수.
    pub init_starts: usize,
    /// 마지막 선형 최소제곱 반복 수.
    pub global_iterations: usize,
    /// 최소제곱·정상 집합 재선정을 되풀이하는 최대 횟수(1 이상으로 다룬다).
    pub active_set_rounds: usize,
}

impl Default for AveragingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance_rad: 1e-4,
            robust_scale_rad: 2f64.to_radians(),
            outlier_threshold_rad: 1f64.to_radians(),
            outlier_cap_rad: 15f64.to_radians(),
            init_agree_rad: 15f64.to_radians(),
            min_inlier_ratio: 0.5,
            outlier_sigma_factor: 6.0,
            init_starts: 8,
            global_iterations: 5,
            active_set_rounds: 8,
        }
    }
}

/// 회전 평균 결과.
#[derive(Clone, Debug)]
pub struct AveragingResult {
    /// 정점별 전역 회전. 기준 정점과 이어지지 않은 정점은 None.
    pub rotations: Vec<Option<Rotation3<f64>>>,
    /// 입력 간선별 최종 잔차 각(rad). 쓰이지 않은 간선은 NaN.
    pub residuals_rad: Vec<f64>,
    /// 입력 간선별 정상 표시 = 마지막 최소제곱에 실제로 쓴 간선 집합.
    pub inliers: Vec<bool>,
    /// 가우스–자이델 반복 수.
    pub iterations: usize,
    /// 정상 집합을 다시 정해 다시 푼 횟수(첫 풀이 제외).
    pub reselections: usize,
    /// 참이면 정상 집합이 두 번 연속 같아 멈췄다. 거짓이면 재선정 상한 도달 또는 집합 순환
    /// (이때도 `inliers` 는 마지막으로 푼 집합이고 해는 그 집합의 최소제곱 해다).
    pub active_set_converged: bool,
    /// 실제로 쓴 이상치 문턱(rad).
    pub outlier_threshold_rad: f64,
    /// 거짓이면 이상치가 너무 많아 잡음 추정이 무너졌다(문턱이 상한에 걸림 또는 정상 비율 부족).
    pub reliable: bool,
}

/// 잔차 각(rad) 목록에서 이상치 문턱을 정한다: clamp(k·중앙값/1.5382, 하한, 상한).
/// 둘째 값은 상한에 걸렸는지.
fn adaptive_threshold(residuals: &[f64], cfg: &AveragingConfig) -> (f64, bool) {
    let floor = cfg.outlier_threshold_rad;
    let mut r: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if cfg.outlier_sigma_factor <= 0.0 || r.is_empty() {
        return (floor, false);
    }
    let mid = r.len() / 2;
    let (_, med, _) = r.select_nth_unstable_by(mid, f64::total_cmp);
    // 자유도 3 카이 분포의 중앙값 = 1.5382σ.
    let sigma = *med / 1.5382;
    let raw = floor.max(cfg.outlier_sigma_factor * sigma);
    let cap = cfg.outlier_cap_rad.max(floor);
    (raw.min(cap), raw > cap)
}

/// 3×3 행렬을 프로베니우스 거리로 가장 가까운 회전에 사영한다.
pub fn project_to_rotation(m: &Matrix3<f64>) -> Option<Rotation3<f64>> {
    if !m.iter().all(|v| v.is_finite()) || m.norm() == 0.0 {
        return None;
    }
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut d = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    Some(Rotation3::from_matrix_unchecked(u * d * vt))
}

fn angle(a: &Rotation3<f64>) -> f64 {
    // trace 로 재면 작은 각에서 정밀도가 떨어지므로 사원수 각을 쓴다.
    crate::math::UnitQuaternion::from_rotation_matrix(a).angle()
}

/// 간선 잔차 각: R_ij 와 R_j R_iᵀ 의 차이.
fn edge_residual(e: &RelativeRotation, r: &[Rotation3<f64>]) -> f64 {
    angle(&(e.rotation.inverse() * r[e.j] * r[e.i].inverse()))
}

#[cfg(test)]
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// 간선 k 가 정점 v 의 회전으로 예측하는 값(다른 끝 회전 `other` 에서).
fn predict(e: &RelativeRotation, v: usize, other: &Rotation3<f64>) -> Rotation3<f64> {
    if e.j == v {
        e.rotation * other
    } else {
        e.rotation.inverse() * other
    }
}

fn other_end(e: &RelativeRotation, v: usize) -> usize {
    if e.i == v {
        e.j
    } else {
        e.i
    }
}

/// `start` 에서 다수결 탐욕 초기화. 놓인 정점만 Some.
fn greedy_init(
    n: usize,
    edges: &[RelativeRotation],
    incident: &[Vec<usize>],
    start: usize,
    thr: f64,
) -> Vec<Option<Rotation3<f64>>> {
    let mut rot: Vec<Option<Rotation3<f64>>> = vec![None; n];
    let mut placed_links = vec![0usize; n];
    rot[start] = Some(Rotation3::identity());
    for &k in &incident[start] {
        placed_links[other_end(&edges[k], start)] += 1;
    }
    while let Some(v) = (0..n)
        .filter(|&v| rot[v].is_none() && placed_links[v] > 0)
        .max_by_key(|&v| (placed_links[v], std::cmp::Reverse(v)))
    {
        let preds: Vec<(Rotation3<f64>, f64)> = incident[v]
            .iter()
            .filter_map(|&k| {
                let o = other_end(&edges[k], v);
                rot[o].map(|r| (predict(&edges[k], v, &r), edges[k].weight))
            })
            .collect();
        let mut best = (f64::MIN, Matrix3::zeros());
        for (p, _) in &preds {
            let mut support = 0.0;
            let mut m = Matrix3::zeros();
            for (q, w) in &preds {
                if angle(&(p * q.inverse())) < thr {
                    support += w;
                    m += q.matrix() * *w;
                }
            }
            if support > best.0 {
                best = (support, m);
            }
        }
        rot[v] = project_to_rotation(&best.1).or(Some(preds[0].0));
        for &k in &incident[v] {
            placed_links[other_end(&edges[k], v)] += 1;
        }
    }
    rot
}

/// 강건 가중 가우스–자이델 반복.
fn refine_robust(
    rot: &mut [Rotation3<f64>],
    edges: &[RelativeRotation],
    incident: &[Vec<usize>],
    fixed: usize,
    cfg: &AveragingConfig,
) -> usize {
    let mut prev_inliers: Option<Vec<bool>> = None;
    let mut stable = 0;
    for it in 0..cfg.max_iterations {
        let mut max_change: f64 = 0.0;
        for v in 0..rot.len() {
            if v == fixed || incident[v].is_empty() {
                continue;
            }
            let mut m = Matrix3::zeros();
            for &k in &incident[v] {
                let e = &edges[k];
                let pred = predict(e, v, &rot[other_end(e, v)]);
                let r = angle(&(rot[v] * pred.inverse())) / cfg.robust_scale_rad;
                m += pred.matrix() * (e.weight / (1.0 + r * r));
            }
            if let Some(new) = project_to_rotation(&m) {
                max_change = max_change.max(angle(&(new * rot[v].inverse())));
                rot[v] = new;
            }
        }
        if max_change < cfg.tolerance_rad {
            return it + 1;
        }
        // 정상 집합(잔차 분포 문턱 안)이 두 번 연속 그대로면 멈춘다. 뒤의 최소제곱이 정밀화한다.
        let res: Vec<f64> = edges
            .iter()
            .map(|e| {
                if incident[e.i].is_empty() || e.i == e.j {
                    f64::NAN
                } else {
                    edge_residual(e, rot)
                }
            })
            .collect();
        let (thr, _) = adaptive_threshold(&res, cfg);
        let set: Vec<bool> = res.iter().map(|&r| r < thr).collect();
        if prev_inliers.as_ref() == Some(&set) {
            stable += 1;
            if stable >= 2 {
                return it + 1;
            }
        } else {
            stable = 0;
        }
        prev_inliers = Some(set);
    }
    cfg.max_iterations
}

/// 기준 정점을 뺀 희소 블록 정규 방정식 H δ = g.
struct BlockSystem {
    /// 정점별 대각 블록.
    diag: Vec<Matrix3<f64>>,
    /// 비대각 블록 (a, b, H_ab), a ≠ b. H_ba = H_abᵀ.
    off: Vec<(usize, usize, Matrix3<f64>)>,
    g: Vec<Vector3<f64>>,
}

impl BlockSystem {
    fn mul(&self, x: &[Vector3<f64>], y: &mut [Vector3<f64>]) {
        for (k, d) in self.diag.iter().enumerate() {
            y[k] = d * x[k];
        }
        for (a, b, h) in &self.off {
            y[*a] += h * x[*b];
            y[*b] += h.transpose() * x[*a];
        }
    }

    /// 블록 야코비 전처리 켤레 기울기.
    fn solve_pcg(&self) -> Option<Vec<Vector3<f64>>> {
        let m = self.diag.len();
        let pre: Vec<Matrix3<f64>> = self
            .diag
            .iter()
            .map(|d| d.try_inverse().unwrap_or_else(Matrix3::identity))
            .collect();
        let dot = |a: &[Vector3<f64>], b: &[Vector3<f64>]| -> f64 {
            a.iter().zip(b).map(|(x, y)| x.dot(y)).sum()
        };
        let mut x = vec![Vector3::zeros(); m];
        let mut r = self.g.clone();
        let g_norm = dot(&r, &r).sqrt();
        if g_norm == 0.0 {
            return Some(x);
        }
        let mut z: Vec<Vector3<f64>> = (0..m).map(|k| pre[k] * r[k]).collect();
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut hp = vec![Vector3::zeros(); m];
        for _ in 0..(30 * m).max(100) {
            self.mul(&p, &mut hp);
            let php = dot(&p, &hp);
            if php.is_nan() || php <= 0.0 {
                break;
            }
            let alpha = rz / php;
            for k in 0..m {
                x[k] += p[k] * alpha;
                r[k] -= hp[k] * alpha;
            }
            if dot(&r, &r).sqrt() <= 1e-14 * g_norm {
                break;
            }
            for k in 0..m {
                z[k] = pre[k] * r[k];
            }
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            for k in 0..m {
                p[k] = z[k] + p[k] * beta;
            }
        }
        x.iter()
            .all(|v| v.iter().all(|c| c.is_finite()))
            .then_some(x)
    }

    #[cfg(test)]
    fn solve_dense(&self) -> Option<Vec<Vector3<f64>>> {
        let m = self.diag.len();
        let mut h = DMatrix::<f64>::zeros(3 * m, 3 * m);
        let mut put = |a: usize, b: usize, blk: &Matrix3<f64>| {
            for x in 0..3 {
                for y in 0..3 {
                    h[(3 * a + x, 3 * b + y)] += blk[(x, y)];
                }
            }
        };
        for (k, d) in self.diag.iter().enumerate() {
            put(k, k, d);
        }
        for (a, b, blk) in &self.off {
            put(*a, *b, blk);
            put(*b, *a, &blk.transpose());
        }
        let g = DVector::from_iterator(3 * m, self.g.iter().flat_map(|v| v.iter().copied()));
        let d = h.cholesky()?.solve(&g);
        Some(
            (0..m)
                .map(|k| Vector3::new(d[3 * k], d[3 * k + 1], d[3 * k + 2]))
                .collect(),
        )
    }
}

#[cfg(test)]
impl BlockSystem {
    /// 띠 촐레스키 직접 풀이(밀집 촐레스키와 같은 분해, 띠 밖이 0 인 것만 이용).
    /// 블록 번호 차가 `bw` 이하인 비대각 블록만 있어야 한다(아니면 None).
    fn solve_banded(&self, bw: usize) -> Option<Vec<Vector3<f64>>> {
        let m = self.diag.len();
        let n = 3 * m;
        let b = 3 * bw + 2; // 스칼라 반띠폭
                            // 아래 삼각 띠 저장: l[i][b + j - i], j ∈ [i-b, i].
        let mut l = vec![vec![0.0f64; b + 1]; n];
        let mut put = |r: usize, c: usize, v: f64| {
            if c <= r {
                l[r][b + c - r] += v;
            }
        };
        for (k, d) in self.diag.iter().enumerate() {
            for x in 0..3 {
                for y in 0..3 {
                    put(3 * k + x, 3 * k + y, d[(x, y)]);
                }
            }
        }
        for (a, c, blk) in &self.off {
            if a.abs_diff(*c) > bw {
                return None;
            }
            for x in 0..3 {
                for y in 0..3 {
                    put(3 * a + x, 3 * c + y, blk[(x, y)]);
                    put(3 * c + y, 3 * a + x, blk[(x, y)]);
                }
            }
        }
        for i in 0..n {
            let lo = i.saturating_sub(b);
            for j in lo..=i {
                let mut sum = l[i][b + j - i];
                for k in lo.max(j.saturating_sub(b))..j {
                    sum -= l[i][b + k - i] * l[j][b + k - j];
                }
                if i == j {
                    if sum <= 0.0 {
                        return None;
                    }
                    l[i][b] = sum.sqrt();
                } else {
                    l[i][b + j - i] = sum / l[j][b];
                }
            }
        }
        let g: Vec<f64> = self.g.iter().flat_map(|v| v.iter().copied()).collect();
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = g[i];
            for k in i.saturating_sub(b)..i {
                sum -= l[i][b + k - i] * y[k];
            }
            y[i] = sum / l[i][b];
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in i + 1..(i + b + 1).min(n) {
                sum -= l[k][b + i - k] * y[k];
            }
            y[i] = sum / l[i][b];
        }
        Some(
            (0..m)
                .map(|k| Vector3::new(y[3 * k], y[3 * k + 1], y[3 * k + 2]))
                .collect(),
        )
    }
}

/// 정규 방정식 풀이 방법.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Solver {
    Pcg,
    #[cfg(test)]
    Dense,
    /// 블록 반띠폭을 준 띠 촐레스키.
    #[cfg(test)]
    Banded(usize),
}

/// 리 대수 선형화 가중 최소제곱.
///
/// 왼쪽 섭동 R_v ← exp(ω_v) R_v 를 두면 Q = R_j R_iᵀ 에 대해
/// exp(ω_j) Q exp(−ω_i) = exp(ω_j − Q ω_i) Q 이므로, 잔차 r = log(Q R_ijᵀ) 의 1차 근사는
/// r(ω) ≈ r + ω_j − Q ω_i. Σ w ‖ω_j − Q ω_i + r‖² 를 기준 정점 ω = 0 으로 풀고 갱신을 되풀이한다.
fn refine_global(
    rot: &mut [Rotation3<f64>],
    edges: &[RelativeRotation],
    active: &[usize],
    nodes: &[usize],
    fixed: usize,
    iterations: usize,
    solver: Solver,
) {
    let mut index = vec![usize::MAX; rot.len()];
    let mut m = 0;
    for &v in nodes {
        if v != fixed {
            index[v] = m;
            m += 1;
        }
    }
    if m == 0 {
        return;
    }
    for _ in 0..iterations {
        // 이상치 제거로 떨어져 나간 정점이 있어도 풀리도록 아주 작은 감쇠.
        let mut sys = BlockSystem {
            diag: vec![Matrix3::identity() * 1e-12; m],
            off: Vec::with_capacity(active.len()),
            g: vec![Vector3::zeros(); m],
        };
        for &k in active {
            let e = &edges[k];
            let q = (rot[e.j] * rot[e.i].inverse()).into_inner();
            let r = (Rotation3::from_matrix_unchecked(q) * e.rotation.inverse()).scaled_axis();
            // 잔차 야코비안: ∂/∂ω_j = I, ∂/∂ω_i = −Q.
            let (ij, ii) = (index[e.j], index[e.i]);
            let w = e.weight;
            if ij != usize::MAX {
                sys.diag[ij] += Matrix3::identity() * w;
                sys.g[ij] -= r * w;
            }
            if ii != usize::MAX {
                let qt = q.transpose();
                sys.diag[ii] += qt * q * w;
                sys.g[ii] += qt * r * w;
            }
            if ij != usize::MAX && ii != usize::MAX {
                // H_{j,i} = Iᵀ (−Q) w.
                sys.off.push((ij, ii, -q * w));
            }
        }
        let delta = match solver {
            Solver::Pcg => sys.solve_pcg(),
            #[cfg(test)]
            Solver::Dense => sys.solve_dense(),
            #[cfg(test)]
            Solver::Banded(bw) => sys.solve_banded(bw),
        };
        let Some(delta) = delta else {
            return;
        };
        let mut max_step: f64 = 0.0;
        for &v in nodes {
            if index[v] == usize::MAX {
                continue;
            }
            let w = delta[index[v]];
            if !w.iter().all(|x| x.is_finite()) {
                return;
            }
            max_step = max_step.max(w.norm());
            rot[v] = Rotation3::new(w) * rot[v];
        }
        if max_step < 1e-13 {
            return;
        }
    }
}

/// 정점 `n` 개와 상대 회전 간선으로 전역 회전을 구한다.
///
/// None: n = 0, 범위 밖 정점 번호, 쓸 수 있는 간선 없음.
/// NaN·무한대가 든 회전, 0 이하 가중치, 자기 자신 간선은 무시한다(결과 잔차 NaN, 이상치 표시).
pub fn average_rotations(
    n: usize,
    edges: &[RelativeRotation],
    cfg: &AveragingConfig,
) -> Option<AveragingResult> {
    if n == 0 || edges.iter().any(|e| e.i >= n || e.j >= n) {
        return None;
    }
    let usable: Vec<bool> = edges
        .iter()
        .map(|e| {
            e.i != e.j
                && e.weight.is_finite()
                && e.weight > 0.0
                && e.rotation.matrix().iter().all(|v| v.is_finite())
        })
        .collect();
    let ids: Vec<usize> = (0..edges.len()).filter(|&k| usable[k]).collect();
    let comp = largest_component(n, edges, &ids)?;
    let root = comp[0];
    let mut reached = vec![false; n];
    for &v in &comp {
        reached[v] = true;
    }
    // 고른 성분 밖 정점은 간선 목록을 비워 초기화·반복에서 건드리지 않는다.
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &k in &ids {
        if reached[edges[k].i] {
            incident[edges[k].i].push(k);
            incident[edges[k].j].push(k);
        }
    }
    let comp_ids: Vec<usize> = ids
        .iter()
        .copied()
        .filter(|&k| reached[edges[k].i])
        .collect();

    // 1. 시작 정점을 바꿔 가며 다수결 초기화, 일치 문턱 안 간선이 가장 많은 것.
    let agree = cfg.init_agree_rad;
    let starts = cfg.init_starts.max(1).min(comp.len());
    let mut best: Option<(usize, Vec<Rotation3<f64>>)> = None;
    for s in 0..starts {
        let start = comp[s * comp.len() / starts];
        let init = greedy_init(n, edges, &incident, start, agree);
        // 기준 정점이 단위 회전이 되도록 세계 회전을 맞춘다: R_v ← R_v R_rootᵀ.
        let g = init[root]?.inverse();
        let r: Vec<Rotation3<f64>> = init
            .iter()
            .map(|x| x.map_or_else(Rotation3::identity, |x| x * g))
            .collect();
        let support = comp_ids
            .iter()
            .filter(|&&k| edge_residual(&edges[k], &r) < agree)
            .count();
        if best.as_ref().is_none_or(|(b, _)| support > *b) {
            best = Some((support, r));
        }
    }
    let (_, mut rot) = best?;

    // 2. 강건 반복.
    let iterations = refine_robust(&mut rot, edges, &incident, root, cfg);

    // 3. 잔차 분포로 문턱을 정하고, 이상치를 빼고 선형 최소제곱.
    let robust_res: Vec<f64> = comp_ids
        .iter()
        .map(|&k| edge_residual(&edges[k], &rot))
        .collect();
    let (thr, capped) = adaptive_threshold(&robust_res, cfg);
    let mut active: Vec<usize> = comp_ids
        .iter()
        .zip(&robust_res)
        .filter(|(_, &r)| r < thr)
        .map(|(&k, _)| k)
        .collect();
    // 최소제곱 뒤 잔차로 정상 집합을 다시 정하고, 바뀌었으면 다시 푼다(두 번 연속 같을 때까지, 최대
    // `active_set_rounds` 번 풀이). 상한에 닿거나 전에 푼 집합으로 되돌아가면(순환) 다시 정한 집합을
    // 버리고 멈춘다. 어느 경우든 돌려주는 정상 표시는 마지막으로 푼 집합 그대로다.
    let rounds = cfg.active_set_rounds.max(1);
    let mut seen: Vec<Vec<usize>> = Vec::new();
    let mut reselections = 0;
    let mut converged = false;
    loop {
        refine_global(
            &mut rot,
            edges,
            &active,
            &comp,
            root,
            cfg.global_iterations,
            Solver::Pcg,
        );
        let next: Vec<usize> = comp_ids
            .iter()
            .copied()
            .filter(|&k| edge_residual(&edges[k], &rot) < thr)
            .collect();
        if next == active {
            converged = true;
            break;
        }
        if reselections + 1 >= rounds || seen.contains(&next) {
            break;
        }
        seen.push(std::mem::replace(&mut active, next));
        reselections += 1;
    }

    let residuals_rad: Vec<f64> = (0..edges.len())
        .map(|k| {
            if usable[k] && reached[edges[k].i] {
                edge_residual(&edges[k], &rot)
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut inliers = vec![false; edges.len()];
    for &k in &active {
        inliers[k] = true;
    }
    let ratio = active.len() as f64 / comp_ids.len().max(1) as f64;
    let rotations = (0..n).map(|v| reached[v].then_some(rot[v])).collect();
    Some(AveragingResult {
        rotations,
        residuals_rad,
        inliers,
        iterations,
        reselections,
        active_set_converged: converged,
        outlier_threshold_rad: thr,
        reliable: !capped && ratio >= cfg.min_inlier_ratio,
    })
}

/// 쓸 수 있는 간선 `ids` 로 나눈 연결 성분 가운데 가장 큰 것의 정점(오름차순).
/// 정점 수가 같으면 간선 가중합이 큰 것, 그것도 같으면 가장 작은 번호 정점이 든 것.
/// 간선이 없으면 None.
fn largest_component(n: usize, edges: &[RelativeRotation], ids: &[usize]) -> Option<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for &k in ids {
        let (a, b) = (find(&mut parent, edges[k].i), find(&mut parent, edges[k].j));
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut size = vec![0usize; n];
    let mut weight = vec![0f64; n];
    let mut touched = vec![false; n];
    for &k in ids {
        touched[edges[k].i] = true;
        touched[edges[k].j] = true;
        let r = find(&mut parent, edges[k].i);
        weight[r] += edges[k].weight;
    }
    for v in (0..n).filter(|&v| touched[v]) {
        let r = find(&mut parent, v);
        size[r] += 1;
    }
    // 뿌리는 성분 안 가장 작은 번호(합칠 때 작은 쪽을 뿌리로). 앞선 뿌리가 같은 크기에서 이긴다.
    let mut best: Option<usize> = None;
    for r in 0..n {
        if size[r] == 0 {
            continue;
        }
        let better =
            best.is_none_or(|b| size[r] > size[b] || (size[r] == size[b] && weight[r] > weight[b]));
        if better {
            best = Some(r);
        }
    }
    let b = best?;
    Some(
        (0..n)
            .filter(|&v| touched[v] && find(&mut parent, v) == b)
            .collect(),
    )
}

/// 시점 그래프 정리 설정.
#[derive(Clone, Debug)]
pub struct ViewGraphConfig {
    /// 평균 뒤 전역 회전과 상대 회전의 차가 이를 넘는 간선을 뺀다(rad). 0 이하면 이 단계를 건너뛴다.
    pub mismatch_rad: f64,
    /// 평균 → 불일치 간선 제거 → 가장 큰 성분 되풀이 최대 횟수(제거가 없으면 일찍 멈춘다).
    pub max_rounds: usize,
    /// 간선 가중치(정상 대응 수)의 하한. 이보다 작은 간선은 처음부터 쓰지 않는다.
    pub min_weight: f64,
    /// 카메라 간 간선의 정상 대응 삼각측량각 중앙 하한(rad). 0 이하면 건너뛴다.
    pub min_triangulation_rad: f64,
}

impl Default for ViewGraphConfig {
    fn default() -> Self {
        Self {
            mismatch_rad: 4f64.to_radians(),
            max_rounds: 4,
            min_weight: 30.0,
            min_triangulation_rad: 0.0,
        }
    }
}

/// 시점 그래프 정리 + 회전 평균 결과.
#[derive(Clone, Debug)]
pub struct PrunedAveraging {
    /// 마지막 평균 결과(입력 간선 번호 기준).
    pub result: AveragingResult,
    /// 입력 간선별 마지막까지 남은 표시.
    pub kept: Vec<bool>,
    /// 단계별로 빠진 간선 수: 가중치·삼각측량각, 불일치(라운드 합), 가장 큰 성분 밖.
    pub removed_prefilter: usize,
    pub removed_mismatch: usize,
    pub removed_component: usize,
    /// 평균을 푼 횟수.
    pub rounds: usize,
}

/// 간선의 정상 대응으로 계산한 삼각측량각(rad)의 중앙값. 카메라 1 = [I|0], 카메라 2 = [R|t].
/// 이동 방향을 알 수 없거나 삼각측량되는 점이 없으면 0.
pub fn median_triangulation_angle(
    pose: &crate::two_view::RelativePose,
    n1: &[crate::math::Vector2<f64>],
    n2: &[crate::math::Vector2<f64>],
) -> f64 {
    if !pose.translation_observable {
        return 0.0;
    }
    let c2 = -(pose.rotation.inverse() * pose.translation);
    let mut v: Vec<f64> = n1
        .iter()
        .zip(n2)
        .filter_map(|(a, b)| crate::two_view::triangulate(&pose.rotation, &pose.translation, a, b))
        .filter_map(|x| {
            let (u, w) = (-x.coords, c2 - x.coords);
            let a = u.cross(&w).norm().atan2(u.dot(&w));
            a.is_finite().then_some(a)
        })
        .collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// 시점 그래프를 정리하며 회전을 평균한다.
///
/// 1. 가중치가 `min_weight` 미만이거나, `cross[k]` 인 간선의 삼각측량각 `tri_rad[k]` 가
///    `min_triangulation_rad` 미만이면 뺀다(`cross`·`tri_rad` 가 비면 그 검사는 건너뛴다).
/// 2. 평균 → 전역 회전과 상대 회전 차가 `mismatch_rad` 넘는 간선 제거 → 가장 큰 연결 성분만 유지 →
///    다시 평균. 제거가 없을 때까지 최대 `max_rounds` 번 푼다.
///
/// None: 쓸 수 있는 간선이 없거나 입력이 틀렸을 때.
pub fn average_rotations_pruned(
    n: usize,
    edges: &[RelativeRotation],
    cross: &[bool],
    tri_rad: &[f64],
    gcfg: &ViewGraphConfig,
    cfg: &AveragingConfig,
) -> Option<PrunedAveraging> {
    let mut alive: Vec<bool> = (0..edges.len())
        .map(|k| {
            let tri_ok = gcfg.min_triangulation_rad <= 0.0
                || !cross.get(k).copied().unwrap_or(false)
                || tri_rad
                    .get(k)
                    .is_some_and(|&a| a >= gcfg.min_triangulation_rad);
            edges[k].weight >= gcfg.min_weight && tri_ok
        })
        .collect();
    let removed_prefilter = alive.iter().filter(|&&a| !a).count();
    let (mut removed_mismatch, mut removed_component, mut rounds) = (0, 0, 0);
    loop {
        let sub: Vec<RelativeRotation> = (0..edges.len())
            .map(|k| {
                let mut e = edges[k].clone();
                if !alive[k] {
                    e.weight = 0.0;
                }
                e
            })
            .collect();
        let result = average_rotations(n, &sub, cfg)?;
        rounds += 1;
        let mut changed = false;
        for k in 0..edges.len().min(alive.len()) {
            if !alive[k] {
                continue;
            }
            let e = &edges[k];
            let (Some(ri), Some(rj)) = (result.rotations[e.i], result.rotations[e.j]) else {
                alive[k] = false;
                removed_component += 1;
                changed = true;
                continue;
            };
            if gcfg.mismatch_rad > 0.0
                && angle(&(e.rotation.inverse() * rj * ri.inverse())) > gcfg.mismatch_rad
                && rounds <= gcfg.max_rounds
            {
                alive[k] = false;
                removed_mismatch += 1;
                changed = true;
            }
        }
        if !changed || rounds > gcfg.max_rounds {
            return Some(PrunedAveraging {
                result,
                kept: alive,
                removed_prefilter,
                removed_mismatch,
                removed_component,
                rounds,
            });
        }
    }
}

/// 추정 회전을 정답 회전에 맞추는 세계 회전 G(R_est G ≈ R_gt)를 구하고, 정점별 각 오차(rad)를 돌려준다.
/// 둘 중 하나라도 None 인 정점은 건너뛴다.
pub fn aligned_errors(estimated: &[Option<Rotation3<f64>>], truth: &[Rotation3<f64>]) -> Vec<f64> {
    let pairs: Vec<(Rotation3<f64>, Rotation3<f64>)> = estimated
        .iter()
        .zip(truth)
        .filter_map(|(e, t)| e.map(|e| (e, *t)))
        .collect();
    let mut m = Matrix3::zeros();
    for (e, t) in &pairs {
        m += (e.inverse() * t).matrix();
    }
    let Some(g) = project_to_rotation(&m) else {
        return Vec::new();
    };
    pairs
        .iter()
        .map(|(e, t)| angle(&((e * g).inverse() * t)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3;
    use crate::synth::{Scene, SceneConfig};

    struct Rng(u64);
    impl Rng {
        fn unit(&mut self) -> f64 {
            (splitmix(&mut self.0) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.unit().max(1e-300);
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
        fn rotation(&mut self, sigma: f64) -> Rotation3<f64> {
            Rotation3::new(Vector3::new(self.gauss(), self.gauss(), self.gauss()) * sigma)
        }
    }

    /// 합성 장면(SPEC §6 편대 배치) 정답 회전(240대)과 SPEC 짝 일정 간선
    /// (`matching::candidate_pairs(.., 5, 4, 16)`: 같은 카메라 1..5·2의 거듭제곱 ≤ 16칸, 카메라 간 ±4칸).
    fn scene_graph() -> (Vec<Rotation3<f64>>, Vec<(usize, usize)>) {
        let scene = Scene::new(SceneConfig::default());
        let truth: Vec<_> = scene.views.iter().map(|v| v.camera.pose.rotation).collect();
        let views: Vec<(usize, usize)> = scene
            .views
            .iter()
            .map(|v| (v.cam as usize, v.position))
            .collect();
        let pairs = crate::matching::candidate_pairs(&views, 5, 4, 16);
        (truth, pairs)
    }

    /// 간선 생성. `outlier_every` 개마다 하나를 오염: `outlier_angle` 이 Some 이면 그 크기 각(임의 축),
    /// None 이면 임의 회전(축당 σ = 1 rad).
    fn make_edges_with(
        truth: &[Rotation3<f64>],
        pairs: &[(usize, usize)],
        sigma: f64,
        outlier_every: usize,
        outlier_angle: Option<f64>,
        rng: &mut Rng,
    ) -> (Vec<RelativeRotation>, Vec<bool>) {
        let mut is_outlier = Vec::new();
        let edges = pairs
            .iter()
            .enumerate()
            .map(|(k, &(i, j))| {
                let outlier = outlier_every > 0 && k % outlier_every == outlier_every / 2;
                is_outlier.push(outlier);
                let rel = truth[j] * truth[i].inverse();
                let rotation = if outlier {
                    match outlier_angle {
                        Some(a) => {
                            let axis = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss());
                            Rotation3::new(axis.normalize() * a) * rng.rotation(sigma) * rel
                        }
                        None => rng.rotation(1.0) * rel,
                    }
                } else {
                    rng.rotation(sigma) * rel
                };
                RelativeRotation {
                    i,
                    j,
                    rotation,
                    weight: 1.0,
                }
            })
            .collect();
        (edges, is_outlier)
    }

    fn make_edges(
        truth: &[Rotation3<f64>],
        pairs: &[(usize, usize)],
        sigma: f64,
        outlier_every: usize,
        rng: &mut Rng,
    ) -> (Vec<RelativeRotation>, Vec<bool>) {
        make_edges_with(truth, pairs, sigma, outlier_every, None, rng)
    }

    fn stats(err: &[f64]) -> (f64, f64) {
        let mean = err.iter().sum::<f64>() / err.len() as f64;
        let max = err.iter().cloned().fold(0.0, f64::max);
        (mean.to_degrees(), max.to_degrees())
    }

    fn rms_deg(err: &[f64]) -> f64 {
        (err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64)
            .sqrt()
            .to_degrees()
    }

    /// 한 경우의 결과 요약.
    #[derive(Debug)]
    struct Case {
        mean: f64,
        max: f64,
        rms: f64,
        outliers: usize,
        caught: usize,
        missed: usize,
        missed_strict: usize,
        false_rej: usize,
        thr_deg: f64,
        diff: f64,
        reliable: bool,
        all_some: bool,
    }

    fn evaluate(
        truth: &[Rotation3<f64>],
        edges: &[RelativeRotation],
        is_outlier: &[bool],
        res: &AveragingResult,
        with_oracle: bool,
    ) -> Case {
        let err = aligned_errors(&res.rotations, truth);
        let (mean, max) = stats(&err);
        let thr = res.outlier_threshold_rad;
        let count = |f: &dyn Fn(usize) -> bool| (0..edges.len()).filter(|&k| f(k)).count();
        let all_some = res.rotations.iter().all(|r| r.is_some());
        let diff = if with_oracle && all_some {
            let est: Vec<_> = res.rotations.iter().map(|r| r.unwrap()).collect();
            let orc = oracle(truth, edges, &res.inliers);
            est.iter()
                .zip(&orc)
                .map(|(a, b)| angle(&(a * b.inverse())))
                .fold(0.0, f64::max)
        } else {
            f64::NAN
        };
        Case {
            mean,
            max,
            rms: rms_deg(&err),
            outliers: count(&|k| is_outlier[k]),
            caught: count(&|k| is_outlier[k] && !res.inliers[k]),
            // 놓침: 오염 간선 중 정답 잔차가 문턱을 넘는데도 정상으로 남은 것.
            // 추정 회전으로 잰 잔차와 정답으로 잰 잔차는 두 끝 정점 오차 합까지 다를 수 있어,
            // 정답 잔차가 문턱 + 두 끝 오차를 넘는 것만 센다(문턱 바로 위 간선은 추정으로 판별 불가).
            missed: count(&|k| {
                let slack = if all_some {
                    err[edges[k].i] + err[edges[k].j]
                } else {
                    0.0
                };
                is_outlier[k] && res.inliers[k] && edge_residual(&edges[k], truth) >= thr + slack
            }),
            missed_strict: count(&|k| {
                is_outlier[k] && res.inliers[k] && edge_residual(&edges[k], truth) >= thr
            }),
            false_rej: count(&|k| !is_outlier[k] && !res.inliers[k]),
            thr_deg: thr.to_degrees(),
            diff,
            reliable: res.reliable,
            all_some,
        }
    }

    fn run_case(
        truth: &[Rotation3<f64>],
        pairs: &[(usize, usize)],
        sigma_deg: f64,
        outlier_every: usize,
        seed: u64,
        with_oracle: bool,
    ) -> Case {
        let mut rng = Rng(seed);
        let (edges, is_outlier) = make_edges(
            truth,
            pairs,
            sigma_deg.to_radians(),
            outlier_every,
            &mut rng,
        );
        let res = average_rotations(truth.len(), &edges, &AveragingConfig::default()).unwrap();
        evaluate(truth, &edges, &is_outlier, &res, with_oracle)
    }

    /// (중앙값, 최댓값).
    fn med_max(v: impl Iterator<Item = f64>) -> (f64, f64) {
        let mut v: Vec<f64> = v.collect();
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], *v.last().unwrap())
    }

    #[test]
    fn exact_graph_recovers_truth() {
        let (truth, pairs) = scene_graph();
        assert_eq!(truth.len(), 240);
        let mut rng = Rng(1);
        let (edges, _) = make_edges(&truth, &pairs, 0.0, 0, &mut rng);
        let res = average_rotations(truth.len(), &edges, &AveragingConfig::default()).unwrap();
        assert!(res.rotations.iter().all(|r| r.is_some()));
        let err = aligned_errors(&res.rotations, &truth);
        let (mean, max) = stats(&err);
        println!(
            "잡음 0: 간선 {} 평균 {mean:.2e}° 최대 {max:.2e}°",
            edges.len()
        );
        assert!(max < 1e-6, "최대 오차 {max}°");
    }

    /// 기준 해: 같은 정상 간선으로 정답(기준 정점 단위 회전으로 맞춤)에서 출발해 수렴까지 푼 최소제곱 해.
    fn oracle(
        truth: &[Rotation3<f64>],
        edges: &[RelativeRotation],
        inliers: &[bool],
    ) -> Vec<Rotation3<f64>> {
        let g = truth[0].inverse();
        let mut rot: Vec<_> = truth.iter().map(|t| t * g).collect();
        let active: Vec<usize> = (0..edges.len()).filter(|&k| inliers[k]).collect();
        let nodes: Vec<usize> = (0..truth.len()).collect();
        refine_global(&mut rot, edges, &active, &nodes, 0, 100, Solver::Dense);
        rot
    }

    /// 선형화 공분산 σ²H⁻¹(기준 정점 0 고정)의 정점 블록 대각합 평균의 제곱근 / σ.
    /// 정점 회전 오차 RMS 의 이론 예측(σ 단위).
    fn predicted_rms_factor(truth: &[Rotation3<f64>], pairs: &[(usize, usize)]) -> f64 {
        let n = truth.len();
        let m = n - 1;
        let mut h = DMatrix::<f64>::zeros(3 * m, 3 * m);
        for &(i, j) in pairs {
            let q = (truth[j] * truth[i].inverse()).into_inner();
            let blocks = [(j, Matrix3::identity()), (i, -q)];
            for (a, ja) in &blocks {
                for (b, jb) in &blocks {
                    if *a == 0 || *b == 0 {
                        continue;
                    }
                    let hab = ja.transpose() * jb;
                    for x in 0..3 {
                        for y in 0..3 {
                            h[(3 * (a - 1) + x, 3 * (b - 1) + y)] += hab[(x, y)];
                        }
                    }
                }
            }
        }
        let inv = h.cholesky().unwrap().inverse();
        (inv.trace() / n as f64).sqrt()
    }

    /// 시드 100..129 를 병렬로 돌린다.
    fn seeds_par(f: impl Fn(u64) -> Case + Sync) -> Vec<(u64, Case)> {
        use rayon::prelude::*;
        (100u64..130).into_par_iter().map(|s| (s, f(s))).collect()
    }

    #[test]
    fn noisy_graph_matches_oracle_and_noise_model() {
        let (truth, pairs) = scene_graph();
        assert_eq!(pairs.len(), 3663);
        let pred = predicted_rms_factor(&truth, &pairs);
        println!(
            "간선 {} 선형화 공분산 예측 정점 RMS = {pred:.3}σ",
            pairs.len()
        );
        for sigma_deg in [0.5, 1.0, 2.0] {
            let rows = seeds_par(|seed| run_case(&truth, &pairs, sigma_deg, 0, seed, true));
            for (seed, c) in &rows {
                println!(
                    "σ={sigma_deg}° 시드 {seed}: 평균 {:.3}° RMS {:.3}° (예측 {:.3}°, 비 {:.2}) 최대 {:.3}° 문턱 {:.2}° 오거부 {} 기준 해 차 {:.1e} rad",
                    c.mean, c.rms, pred * sigma_deg, c.rms / (pred * sigma_deg), c.max, c.thr_deg, c.false_rej, c.diff
                );
            }
            let (m_med, m_max) = med_max(rows.iter().map(|(_, c)| c.mean));
            let (r_med, r_max) = med_max(rows.iter().map(|(_, c)| c.rms / (pred * sigma_deg)));
            let (f_med, f_max) = med_max(rows.iter().map(|(_, c)| c.false_rej as f64));
            println!(
                "σ={sigma_deg}° 시드 30개: 평균 오차 중앙 {m_med:.3}° 최악 {m_max:.3}°, RMS/예측 중앙 {r_med:.2} 최악 {r_max:.2}, 오거부 중앙 {f_med} 최악 {f_max}"
            );
            for (seed, c) in &rows {
                assert!(c.all_some && c.reliable, "σ={sigma_deg}° 시드 {seed}");
                // 전역 최적 확인: 같은 정상 집합으로 정답에서 출발한 최소제곱 해와 같다.
                assert!(
                    c.diff < 1e-6,
                    "σ={sigma_deg}° 시드 {seed}: 기준 해와 차 {}",
                    c.diff
                );
                // 잡음 모델 근거: 정점 오차 RMS 는 선형화 공분산 예측(기준 정점 0 고정)의 0.8배 안.
                // 정렬 오차는 세계 회전을 최적으로 맞춰 기준 정점 고정보다 작다(측정 비 0.6 안팎).
                assert!(
                    c.rms < 0.8 * pred * sigma_deg,
                    "σ={sigma_deg}° 시드 {seed}: RMS {}°",
                    c.rms
                );
                assert!(
                    c.mean < 1.5 * sigma_deg,
                    "σ={sigma_deg}° 시드 {seed}: 평균 {}°",
                    c.mean
                );
                assert!(
                    c.false_rej <= pairs.len() / 100,
                    "σ={sigma_deg}° 시드 {seed}: 오거부 {}",
                    c.false_rej
                );
            }
        }
    }

    #[test]
    fn outlier_edges_are_rejected() {
        let (truth, pairs) = scene_graph();
        let pred = predicted_rms_factor(&truth, &pairs);
        // 간선 10개 중 1개를 임의 회전(축당 1 rad)으로 오염.
        let rows = seeds_par(|seed| run_case(&truth, &pairs, 1.0, 10, seed, true));
        for (seed, c) in &rows {
            println!(
                "σ 1°+이상치 시드 {seed}: 이상치 {} 잡음 {} 놓침 {} (문턱만 {}) 오거부 {} 문턱 {:.2}° 평균 {:.3}° RMS {:.3}° 최대 {:.3}° 기준 해 차 {:.1e} rad",
                c.outliers, c.caught, c.missed, c.missed_strict, c.false_rej, c.thr_deg, c.mean, c.rms, c.max, c.diff
            );
        }
        let (m_med, m_max) = med_max(rows.iter().map(|(_, c)| c.mean));
        println!("σ 1°+이상치 10% 시드 30개: 평균 오차 중앙 {m_med:.3}° 최악 {m_max:.3}°");
        for (seed, c) in &rows {
            assert!(c.reliable, "시드 {seed}");
            assert!(c.diff < 1e-6, "시드 {seed}: 기준 해와 차 {}", c.diff);
            assert_eq!(c.missed, 0, "시드 {seed}");
            assert!(
                c.false_rej <= pairs.len() / 100,
                "시드 {seed}: 오거부 {}",
                c.false_rej
            );
            // 정상 간선이 90% 라 정보가 줄어든 만큼(1/√0.9) 예측을 키운 뒤 0.8배.
            assert!(
                c.rms < 0.8 * pred / 0.9f64.sqrt(),
                "시드 {seed}: RMS {}°",
                c.rms
            );
            assert!(c.mean < 1.5, "시드 {seed}: 평균 {}°", c.mean);
        }
        // σ 2° + 이상치 10%: 최대 오차 < 10°.
        let rows = seeds_par(|seed| run_case(&truth, &pairs, 2.0, 10, seed, false));
        let (x_med, x_max) = med_max(rows.iter().map(|(_, c)| c.max));
        let (m_med, m_max) = med_max(rows.iter().map(|(_, c)| c.mean));
        println!(
            "σ 2°+이상치 10% 시드 30개: 평균 오차 중앙 {m_med:.3}° 최악 {m_max:.3}°, 최대 오차 중앙 {x_med:.3}° 최악 {x_max:.3}°"
        );
        for (seed, c) in &rows {
            assert!(c.max < 10.0, "시드 {seed}: 최대 {}°", c.max);
            assert_eq!(c.missed, 0, "시드 {seed}");
        }
    }

    /// 기본 설정, σ {0.5,1,2}° × 이상치 {0,10,20}% × 시드 5/6/7.
    #[test]
    fn noise_and_outlier_grid() {
        use rayon::prelude::*;
        let (truth, pairs) = scene_graph();
        let mut cases = Vec::new();
        for sigma in [0.5, 1.0, 2.0] {
            for (pct, every) in [(0, 0), (10, 10), (20, 5)] {
                for seed in [5u64, 6, 7] {
                    cases.push((sigma, pct, every, seed));
                }
            }
        }
        let rows: Vec<_> = cases
            .par_iter()
            .map(|&(sigma, pct, every, seed)| {
                (
                    sigma,
                    pct,
                    seed,
                    run_case(&truth, &pairs, sigma, every, seed, false),
                )
            })
            .collect();
        println!(
            "| σ | 이상치 | 시드 | 이상치 수 | 잡음 | 놓침(문턱만) | 오거부 | 문턱 | 평균 | 최대 |"
        );
        for (sigma, pct, seed, c) in &rows {
            println!(
                "| {sigma}° | {pct}% | {seed} | {} | {} | {}({}) | {} | {:.2}° | {:.3}° | {:.3}° |",
                c.outliers,
                c.caught,
                c.missed,
                c.missed_strict,
                c.false_rej,
                c.thr_deg,
                c.mean,
                c.max
            );
        }
        for (sigma, pct, seed, c) in &rows {
            let tag = format!("σ {sigma}° 이상치 {pct}% 시드 {seed}");
            assert!(c.reliable, "{tag}");
            assert_eq!(c.missed, 0, "{tag}");
            assert!(
                c.false_rej <= pairs.len() / 100,
                "{tag}: 오거부 {}",
                c.false_rej
            );
            assert!(c.mean < 1.5 * sigma, "{tag}: 평균 {}°", c.mean);
        }
    }

    /// 간선 절반·전부가 임의 회전이면 문턱이 상한에 머물고 믿을 수 없다고 표시한다.
    #[test]
    fn heavy_contamination_is_flagged() {
        let (truth, pairs) = scene_graph();
        let cap = AveragingConfig::default().outlier_cap_rad.to_degrees();
        for (label, every) in [("절반", 2usize), ("전부", 1)] {
            let c = run_case(&truth, &pairs, 0.6, every, 41, false);
            println!(
                "{label} 오염: 이상치 {} 문턱 {:.2}° 놓침 {} 평균 {:.2}° 최대 {:.2}° 믿음 {}",
                c.outliers, c.thr_deg, c.missed, c.mean, c.max, c.reliable
            );
            assert!(c.thr_deg <= cap + 1e-9, "{label}: 문턱 {}°", c.thr_deg);
            assert!(!c.reliable, "{label}");
        }
    }

    /// 잡음이 작을 때(σ 0.3°) 약 3° 오차 간선 10% 를 거른다.
    #[test]
    fn small_outliers_at_low_noise() {
        let (truth, pairs) = scene_graph();
        let sigma = 0.3f64.to_radians();
        let cfg = AveragingConfig::default();
        for seed in [51u64, 52, 53] {
            let mut rng = Rng(seed);
            let (clean, _) = make_edges(&truth, &pairs, sigma, 0, &mut rng);
            let clean_res = average_rotations(truth.len(), &clean, &cfg).unwrap();
            let clean_rms = rms_deg(&aligned_errors(&clean_res.rotations, &truth));
            let mut rng = Rng(seed);
            let (edges, is_out) =
                make_edges_with(&truth, &pairs, sigma, 10, Some(3f64.to_radians()), &mut rng);
            let res = average_rotations(truth.len(), &edges, &cfg).unwrap();
            let c = evaluate(&truth, &edges, &is_out, &res, false);
            println!(
                "σ 0.3° + 3° 이상치 시드 {seed}: {} 중 {} 검출, 오거부 {} 문턱 {:.2}° RMS {:.4}° (무오염 {clean_rms:.4}°)",
                c.outliers, c.caught, c.false_rej, c.thr_deg, c.rms
            );
            assert!(
                c.caught * 100 >= c.outliers * 95,
                "검출 {}/{}",
                c.caught,
                c.outliers
            );
            assert!(c.rms <= clean_rms * 1.1, "RMS {} 무오염 {clean_rms}", c.rms);
            assert!(c.false_rej <= pairs.len() / 100);
        }
    }

    /// 기준 0번 정점이 작은 성분에 있어도 가장 큰 성분을 돌려준다.
    #[test]
    fn largest_component_is_returned() {
        let (truth, pairs) = scene_graph();
        let mut rng = Rng(61);
        let (mut edges, _) = make_edges(&truth, &pairs, 0.6f64.to_radians(), 0, &mut rng);
        // 0 은 3 과의 간선만, 3 은 0 과의 간선만 남긴다(이륙 직후 매칭 실패).
        edges.retain(|e| {
            let touches = |v| e.i == v || e.j == v;
            let pair03 = touches(0) && touches(3);
            pair03 || !(touches(0) || touches(3))
        });
        let res = average_rotations(truth.len(), &edges, &AveragingConfig::default()).unwrap();
        let some = res.rotations.iter().filter(|r| r.is_some()).count();
        assert!(res.rotations[0].is_none() && res.rotations[3].is_none());
        let err = aligned_errors(&res.rotations, &truth);
        let (mean, max) = stats(&err);
        println!("큰 성분: 정점 {some} 평균 {mean:.3}° 최대 {max:.3}°");
        assert_eq!(some, 238);
        assert!(mean < 0.3, "평균 {mean}°");
        // 정상 표시는 고른 성분의 간선만.
        for (k, e) in edges.iter().enumerate() {
            if e.i == 0 || e.j == 0 {
                assert!(!res.inliers[k] && res.residuals_rad[k].is_nan());
            }
        }
    }

    #[test]
    fn component_tie_rule() {
        let r = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let e = |i, j, weight| RelativeRotation {
            i,
            j,
            rotation: r,
            weight,
        };
        let cfg = AveragingConfig::default();
        // 크기 같음, 가중합이 큰 {2,3}.
        let res = average_rotations(4, &[e(0, 1, 1.0), e(2, 3, 2.0)], &cfg).unwrap();
        assert!(res.rotations[0].is_none() && res.rotations[2].is_some());
        assert_eq!(res.rotations[2].unwrap(), Rotation3::identity());
        // 크기·가중합 같음: 작은 번호 정점이 든 {0,1}.
        let res = average_rotations(4, &[e(2, 3, 1.0), e(0, 1, 1.0)], &cfg).unwrap();
        assert!(res.rotations[0].is_some() && res.rotations[2].is_none());
        // 정점 수가 우선: {1,2,3} 이 가중치 큰 {0,4} 를 이긴다.
        let res = average_rotations(5, &[e(0, 4, 10.0), e(1, 2, 1.0), e(2, 3, 1.0)], &cfg).unwrap();
        assert!(res.rotations[0].is_none());
        assert!((1..4).all(|v| res.rotations[v].is_some()));
        assert_eq!(res.rotations[1].unwrap(), Rotation3::identity());
    }

    /// 정점 n 개 사슬, 각 정점이 앞 1~4칸과 이어진 그래프(간선 ≈ 4n).
    fn chain_graph(
        n: usize,
        sigma: f64,
        seed: u64,
    ) -> (Vec<Rotation3<f64>>, Vec<RelativeRotation>) {
        let mut rng = Rng(seed);
        let truth: Vec<_> = (0..n).map(|_| rng.rotation(1.0)).collect();
        let mut edges = Vec::new();
        for j in 0..n {
            for d in 1..=4 {
                if j >= d {
                    let i = j - d;
                    edges.push(RelativeRotation {
                        i,
                        j,
                        rotation: rng.rotation(sigma) * truth[j] * truth[i].inverse(),
                        weight: 1.0,
                    });
                }
            }
        }
        (truth, edges)
    }

    #[test]
    fn sparse_solver_matches_dense() {
        let (truth, pairs) = scene_graph();
        let mut rng = Rng(21);
        let (edges, _) = make_edges(&truth, &pairs, 1f64.to_radians(), 0, &mut rng);
        let res = average_rotations(truth.len(), &edges, &AveragingConfig::default()).unwrap();
        let active: Vec<usize> = (0..edges.len()).filter(|&k| res.inliers[k]).collect();
        let nodes: Vec<usize> = (0..truth.len()).collect();
        // 같은 출발점(잡음 정답)에서 두 풀이로 한 단계·수렴까지 비교.
        for iters in [1, 10] {
            let mut start_rng = Rng(22);
            let start: Vec<_> = truth
                .iter()
                .map(|t| start_rng.rotation(0.02) * t * truth[0].inverse())
                .collect();
            let (mut a, mut b) = (start.clone(), start);
            a[0] = Rotation3::identity();
            b[0] = Rotation3::identity();
            refine_global(&mut a, &edges, &active, &nodes, 0, iters, Solver::Pcg);
            refine_global(&mut b, &edges, &active, &nodes, 0, iters, Solver::Dense);
            let diff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| angle(&(x * y.inverse())))
                .fold(0.0, f64::max);
            println!("희소 대 밀집 ({iters}단계): 최대 차 {diff:.2e} rad");
            assert!(diff < 1e-9, "차 {diff}");
        }
    }

    #[test]
    fn large_chain_graph_is_fast() {
        for n in [240usize, 1000, 2000] {
            let (truth, edges) = chain_graph(n, 1f64.to_radians(), 31);
            let t0 = std::time::Instant::now();
            let res = average_rotations(n, &edges, &AveragingConfig::default()).unwrap();
            let total = t0.elapsed().as_secs_f64();
            let active: Vec<usize> = (0..edges.len()).filter(|&k| res.inliers[k]).collect();
            let nodes: Vec<usize> = (0..n).collect();
            // 수렴 해에서 0.02 rad 흔든 출발점(기준 정점은 단위 회전 유지): 한 단계가 0 이 아니게.
            let mut start_rng = Rng(32);
            let mut start: Vec<_> = res
                .rotations
                .iter()
                .map(|r| start_rng.rotation(0.02) * r.unwrap())
                .collect();
            start[0] = res.rotations[0].unwrap();
            let mut rot = start.clone();
            let t1 = std::time::Instant::now();
            refine_global(&mut rot, &edges, &active, &nodes, 0, 1, Solver::Pcg);
            let step = t1.elapsed().as_secs_f64();
            // 밀집 촐레스키는 2000 정점에서 288MB·80초 넘게 걸려 1000 정점까지는 밀집, 2000 정점은
            // 같은 촐레스키 분해를 띠(사슬 간선 블록 번호 차 ≤ 4) 안에서만 하는 직접 풀이와 비교한다.
            let mut dense = start;
            let t2 = std::time::Instant::now();
            let solver = if n <= 1000 {
                Solver::Dense
            } else {
                Solver::Banded(4)
            };
            refine_global(&mut dense, &edges, &active, &nodes, 0, 1, solver);
            let step_dense = t2.elapsed().as_secs_f64();
            let diff = rot
                .iter()
                .zip(&dense)
                .map(|(x, y)| angle(&(x * y.inverse())))
                .fold(0.0, f64::max);
            // 행렬 저장량: 희소 = (대각 m + 비대각 간선 수) 블록 × 9 × 8 바이트, 밀집 = (3m)² × 8 바이트.
            let m = n - 1;
            let sparse_mb = ((m + active.len()) * 72) as f64 / 1e6;
            let dense_mb = (9 * m * m * 8) as f64 / 1e6;
            let err = aligned_errors(&res.rotations, &truth);
            let (mean, max) = stats(&err);
            println!(
                "정점 {n} 간선 {}: 전체 {total:.3}s, 최소제곱 한 단계 희소 {step:.4}s 직접({solver:?}) {step_dense:.4}s, 행렬 희소 {sparse_mb:.3}MB 밀집 {dense_mb:.1}MB, 희소-직접 차 {diff:.1e} rad, 평균 {mean:.3}° 최대 {max:.3}° 반복 {}",
                edges.len(),
                res.iterations
            );
            assert!(edges.len() >= 4 * n - 10);
            assert!(diff < 1e-9, "희소-밀집 차 {diff}");
            // 시간 상한: 단독 측정 2000 정점 전체 0.46~0.70 s. 4 코어를 여러 작업이 나눠 쓰는 부하에서
            // 7배 이상 느려진 적이 없어 5 s 를 상한으로 둔다.
            assert!(total < 5.0, "정점 {n}: {total:.3}s");
            // 사슬은 오차가 길이를 따라 쌓인다(측정 평균 240/1000 정점 1.4/4.9°). 판별력 있는 정확도 단언은
            // 선형화 공분산 예측 대비로 바꿔야 하나 2000 정점 역행렬이 무거워 아직 느슨한 상한만 둔다.
            assert!(mean < 20.0, "평균 {mean}°");
            assert!(res.inliers.iter().filter(|&&b| !b).count() <= edges.len() / 100);
        }
    }

    /// 띠 촐레스키가 밀집 촐레스키와 같은 해를 내는지(2000 정점 비교의 근거).
    #[test]
    fn banded_solver_matches_dense() {
        let n = 240;
        let (_, edges) = chain_graph(n, 1f64.to_radians(), 33);
        let active: Vec<usize> = (0..edges.len()).collect();
        let nodes: Vec<usize> = (0..n).collect();
        let mut rng = Rng(34);
        let start: Vec<_> = (0..n)
            .map(|v| {
                if v == 0 {
                    Rotation3::identity()
                } else {
                    rng.rotation(0.5)
                }
            })
            .collect();
        let (mut a, mut b) = (start.clone(), start);
        refine_global(&mut a, &edges, &active, &nodes, 0, 1, Solver::Banded(4));
        refine_global(&mut b, &edges, &active, &nodes, 0, 1, Solver::Dense);
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| angle(&(x * y.inverse())))
            .fold(0.0, f64::max);
        println!("정점 {n} 띠-밀집 1단계 차 {diff:.1e} rad");
        assert!(diff < 1e-9, "차 {diff}");
    }

    /// 정상 표시 = 실제로 푼 집합(F-137): 시드 100..129 × 4경우, 재선정 상한 8(기본)·1.
    #[test]
    fn inliers_are_the_solved_set() {
        use rayon::prelude::*;
        let (truth, pairs) = scene_graph();
        let nodes: Vec<usize> = (0..truth.len()).collect();
        let cases: [(f64, usize); 4] = [(0.5, 0), (1.0, 0), (2.0, 0), (1.0, 10)];
        for rounds in [8usize, 1] {
            let cfg = AveragingConfig {
                active_set_rounds: rounds,
                ..AveragingConfig::default()
            };
            for (sigma_deg, every) in cases {
                let rows: Vec<(u64, usize, bool, f64, f64)> = (100u64..130)
                    .into_par_iter()
                    .map(|seed| {
                        let mut rng = Rng(seed);
                        let (edges, _) =
                            make_edges(&truth, &pairs, sigma_deg.to_radians(), every, &mut rng);
                        let res = average_rotations(truth.len(), &edges, &cfg).unwrap();
                        let rot: Vec<_> = res.rotations.iter().map(|r| r.unwrap()).collect();
                        let active: Vec<usize> =
                            (0..edges.len()).filter(|&k| res.inliers[k]).collect();
                        // 돌려준 정상 표시로 50단계 더 풀어도 움직이지 않아야 한다.
                        let mut more = rot.clone();
                        refine_global(&mut more, &edges, &active, &nodes, 0, 50, Solver::Pcg);
                        let moved = rot
                            .iter()
                            .zip(&more)
                            .map(|(x, y)| angle(&(x * y.inverse())))
                            .fold(0.0, f64::max);
                        // 기준 해: 같은 정상 집합으로 정답에서 출발한 밀집 최소제곱.
                        let mut oracle: Vec<_> =
                            truth.iter().map(|t| t * truth[0].inverse()).collect();
                        refine_global(&mut oracle, &edges, &active, &nodes, 0, 100, Solver::Dense);
                        let diff = rot
                            .iter()
                            .zip(&oracle)
                            .map(|(x, y)| angle(&(x * y.inverse())))
                            .fold(0.0, f64::max);
                        (
                            seed,
                            res.reselections,
                            res.active_set_converged,
                            moved,
                            diff,
                        )
                    })
                    .collect();
                let max_resel = rows.iter().map(|r| r.1).max().unwrap();
                let unconverged = rows.iter().filter(|r| !r.2).count();
                let max_moved = rows.iter().map(|r| r.3).fold(0.0, f64::max);
                let max_diff = rows.iter().map(|r| r.4).fold(0.0, f64::max);
                println!(
                    "상한 {rounds} σ={sigma_deg}° 이상치 {}%: 재선정 최대 {max_resel}, 미수렴 {unconverged}/30, 자기 일관성 최대 차 {max_moved:.1e} rad, 기준 해 차 최대 {max_diff:.1e} rad",
                    100usize.checked_div(every).unwrap_or(0)
                );
                for (seed, resel, conv, moved, diff) in &rows {
                    assert!(*resel < rounds, "시드 {seed}: 재선정 {resel}");
                    if rounds == 8 {
                        assert!(*conv, "σ={sigma_deg}° 시드 {seed}: 상한 8 에서 미수렴");
                    }
                    assert!(
                        *moved < 1e-9,
                        "σ={sigma_deg}° 시드 {seed}: 50단계 더 {moved}"
                    );
                    assert!(
                        *diff < 1e-6,
                        "σ={sigma_deg}° 시드 {seed}: 기준 해 차 {diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn disconnected_and_invalid_inputs() {
        let r = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let e = |i, j| RelativeRotation {
            i,
            j,
            rotation: r,
            weight: 1.0,
        };
        let cfg = AveragingConfig::default();
        assert!(average_rotations(0, &[], &cfg).is_none());
        assert!(average_rotations(3, &[], &cfg).is_none());
        assert!(average_rotations(3, &[e(0, 5)], &cfg).is_none());
        // 0-1 과 2-3-4 두 성분: 큰 성분(2-3-4)만 돌려준다.
        let res = average_rotations(5, &[e(0, 1), e(2, 3), e(3, 4)], &cfg).unwrap();
        assert!(res.rotations[0].is_none() && res.rotations[1].is_none());
        assert!((2..5).all(|v| res.rotations[v].is_some()));
        assert!(res.residuals_rad[0].is_nan() && !res.inliers[0]);
        // NaN 회전·0 가중치·자기 간선은 무시.
        let mut nan = e(1, 2);
        nan.rotation = Rotation3::from_matrix_unchecked(Matrix3::from_element(f64::NAN));
        let mut zero = e(1, 2);
        zero.weight = 0.0;
        let res = average_rotations(3, &[e(0, 1), nan, zero, e(2, 2)], &cfg).unwrap();
        assert!(res.rotations[2].is_none());
        assert!(rotation_angle(&(res.rotations[1].unwrap() * r.inverse())) < 1e-12);
        assert!(average_rotations(3, &[e(1, 1)], &cfg).is_none());
    }

    fn rotation_angle(r: &Rotation3<f64>) -> f64 {
        angle(r)
    }

    /// 시점 그래프 정리: 3° 넘게 틀린 간선은 빠지고 정렬 오차가 줄며, 가중치 미달 간선은 처음부터 빠진다.
    #[test]
    fn pruning_drops_mismatched_and_light_edges() {
        let n = 12;
        let truth: Vec<Rotation3<f64>> = (0..n)
            .map(|k| Rotation3::from_euler_angles(0.1 * k as f64, 0.05 * (k * k % 7) as f64, 0.2))
            .collect();
        let mut edges = Vec::new();
        for i in 0..n {
            for d in 1..=3 {
                let j = (i + d) % n;
                edges.push(RelativeRotation {
                    i,
                    j,
                    rotation: truth[j] * truth[i].inverse(),
                    weight: 100.0,
                });
            }
        }
        // 4° 틀린 간선 하나, 가중치 10 인 간선 하나.
        let bad = Rotation3::from_axis_angle(&Vector3::z_axis(), 4f64.to_radians());
        edges[5].rotation = bad * edges[5].rotation;
        edges[9].weight = 10.0;
        let g = ViewGraphConfig {
            mismatch_rad: 2f64.to_radians(),
            ..ViewGraphConfig::default()
        };
        let p =
            average_rotations_pruned(n, &edges, &[], &[], &g, &AveragingConfig::default()).unwrap();
        assert!(!p.kept[5] && !p.kept[9]);
        assert_eq!(p.removed_prefilter, 1);
        assert_eq!(p.removed_mismatch, 1);
        assert_eq!(p.kept.iter().filter(|&&k| k).count(), edges.len() - 2);
        let err = aligned_errors(&p.result.rotations, &truth);
        assert_eq!(err.len(), n);
        assert!(err.iter().all(|&e| e < 0.01f64.to_radians()));
    }
}
