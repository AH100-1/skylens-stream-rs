//! 회전 평균: 상대 회전 그래프 → 전역 회전.
//!
//! 회전 R_i 는 세계→카메라 i. 간선 (i, j) 의 상대 회전은 R_ij = R_j R_iᵀ
//! (`two_view::RelativePose::rotation` 과 같은 규약: x_j = R_ij x_i + t).
//!
//! 1. 다수결 탐욕 초기화: 이미 놓인 정점과 간선이 가장 많은 정점부터 놓는다. 그 정점의 회전은
//!    놓인 이웃들이 예측한 회전 가운데 서로 문턱 안에서 가장 많이 일치하는 무리의 평균이다.
//!    시작 정점을 여러 개 바꿔 보고 문턱 안 간선이 가장 많은 초기값을 고른다.
//! 2. 정점마다 이웃 예측의 가중 현(chordal) 평균을 SO(3) 로 사영하는 가우스–자이델 반복.
//!    가중치는 코시형 1/(1 + (r/σ)²) 로 이상치 간선을 누른다.
//! 3. 잔차가 `outlier_threshold_rad` 를 넘는 간선을 이상치로 빼고, 남은 간선으로 리 대수 선형화
//!    가중 최소제곱(정규 방정식, 촐레스키)을 몇 번 반복한다. 긴 사슬에서 느린 저주파 오차를 한 번에 없앤다.
//!
//! 전역 회전은 세계 좌표 회전 하나만큼 정해지지 않는다. 기준 정점(쓸 수 있는 간선이 닿는 가장 작은 번호)을
//! 단위 회전으로 둔다.

use crate::math::{Matrix3, Rotation3, Vector3};
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
    /// 이 잔차(rad)를 넘는 간선은 이상치.
    pub outlier_threshold_rad: f64,
    /// 초기화 시작 정점 후보 수.
    pub init_starts: usize,
    /// 마지막 선형 최소제곱 반복 수.
    pub global_iterations: usize,
}

impl Default for AveragingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance_rad: 1e-10,
            robust_scale_rad: 2f64.to_radians(),
            outlier_threshold_rad: 5f64.to_radians(),
            init_starts: 8,
            global_iterations: 5,
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
    /// 입력 간선별 정상 표시.
    pub inliers: Vec<bool>,
    /// 가우스–자이델 반복 수.
    pub iterations: usize,
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
    }
    cfg.max_iterations
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
        let mut h = DMatrix::<f64>::zeros(3 * m, 3 * m);
        let mut g = DVector::<f64>::zeros(3 * m);
        for &k in active {
            let e = &edges[k];
            let q = (rot[e.j] * rot[e.i].inverse()).into_inner();
            let r = (Rotation3::from_matrix_unchecked(q) * e.rotation.inverse()).scaled_axis();
            // 잔차 야코비안: ∂/∂ω_j = I, ∂/∂ω_i = −Q.
            let blocks = [(e.j, Matrix3::identity()), (e.i, -q)];
            for (a, ja) in &blocks {
                if index[*a] == usize::MAX {
                    continue;
                }
                let ia = 3 * index[*a];
                let jt = ja.transpose() * e.weight;
                let ga: Vector3<f64> = jt * r;
                for d in 0..3 {
                    g[ia + d] -= ga[d];
                }
                for (b, jb) in &blocks {
                    if index[*b] == usize::MAX {
                        continue;
                    }
                    let ib = 3 * index[*b];
                    let hab = jt * jb;
                    for x in 0..3 {
                        for y in 0..3 {
                            h[(ia + x, ib + y)] += hab[(x, y)];
                        }
                    }
                }
            }
        }
        // 이상치 제거로 떨어져 나간 정점이 있어도 풀리도록 아주 작은 감쇠.
        for d in 0..3 * m {
            h[(d, d)] += 1e-12;
        }
        let Some(chol) = h.cholesky() else {
            return;
        };
        let delta = chol.solve(&g);
        let mut max_step: f64 = 0.0;
        for &v in nodes {
            if index[v] == usize::MAX {
                continue;
            }
            let o = 3 * index[v];
            let w = Vector3::new(delta[o], delta[o + 1], delta[o + 2]);
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
    let root = ids.iter().map(|&k| edges[k].i.min(edges[k].j)).min()?;
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &k in &ids {
        incident[edges[k].i].push(k);
        incident[edges[k].j].push(k);
    }

    // 1. 시작 정점을 바꿔 가며 다수결 초기화, 문턱 안 간선이 가장 많은 것.
    let comp: Vec<usize> = {
        let first = greedy_init(n, edges, &incident, root, cfg.outlier_threshold_rad);
        (0..n).filter(|&v| first[v].is_some()).collect()
    };
    let starts = cfg.init_starts.max(1).min(comp.len());
    let mut best: Option<(usize, Vec<Rotation3<f64>>)> = None;
    for s in 0..starts {
        let start = comp[s * comp.len() / starts];
        let init = greedy_init(n, edges, &incident, start, cfg.outlier_threshold_rad);
        // 기준 정점이 단위 회전이 되도록 세계 회전을 맞춘다: R_v ← R_v R_rootᵀ.
        let g = init[root]?.inverse();
        let r: Vec<Rotation3<f64>> = init
            .iter()
            .map(|x| x.map_or_else(Rotation3::identity, |x| x * g))
            .collect();
        let support = ids
            .iter()
            .filter(|&&k| edge_residual(&edges[k], &r) < cfg.outlier_threshold_rad)
            .count();
        if best.as_ref().is_none_or(|(b, _)| support > *b) {
            best = Some((support, r));
        }
    }
    let (_, mut rot) = best?;
    let mut reached = vec![false; n];
    for &v in &comp {
        reached[v] = true;
    }
    let comp_incident: Vec<Vec<usize>> = (0..n)
        .map(|v| {
            if reached[v] {
                incident[v].clone()
            } else {
                Vec::new()
            }
        })
        .collect();

    // 2. 강건 반복.
    let iterations = refine_robust(&mut rot, edges, &comp_incident, root, cfg);

    // 3. 이상치를 빼고 선형 최소제곱.
    let active: Vec<usize> = ids
        .iter()
        .copied()
        .filter(|&k| {
            reached[edges[k].i] && edge_residual(&edges[k], &rot) < cfg.outlier_threshold_rad
        })
        .collect();
    refine_global(&mut rot, edges, &active, &comp, root, cfg.global_iterations);

    let residuals_rad: Vec<f64> = (0..edges.len())
        .map(|k| {
            if usable[k] && reached[edges[k].i] {
                edge_residual(&edges[k], &rot)
            } else {
                f64::NAN
            }
        })
        .collect();
    let inliers = residuals_rad
        .iter()
        .map(|&r| r < cfg.outlier_threshold_rad)
        .collect();
    let rotations = (0..n).map(|v| reached[v].then_some(rot[v])).collect();
    Some(AveragingResult {
        rotations,
        residuals_rad,
        inliers,
        iterations,
    })
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

    /// 합성 장면 정답 회전(240대)과 간선: 같은 카메라 시간 이웃(1~3칸) + 같은 위치 카메라 간.
    fn scene_graph() -> (Vec<Rotation3<f64>>, Vec<(usize, usize)>) {
        let scene = Scene::new(SceneConfig::default());
        let truth: Vec<_> = scene.views.iter().map(|v| v.camera.pose.rotation).collect();
        let mut pairs = Vec::new();
        for (a, va) in scene.views.iter().enumerate() {
            for (b, vb) in scene.views.iter().enumerate().skip(a + 1) {
                let same_cam = va.cam == vb.cam && vb.position.abs_diff(va.position) <= 3;
                let same_pos = va.position == vb.position;
                if same_cam || same_pos {
                    pairs.push((a, b));
                }
            }
        }
        (truth, pairs)
    }

    fn make_edges(
        truth: &[Rotation3<f64>],
        pairs: &[(usize, usize)],
        sigma: f64,
        outlier_every: usize,
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
                    rng.rotation(1.0) * rel
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

    /// 정상 간선(정답 잔차가 문턱 안)에서 Σ w r² (rad²).
    fn cost(
        edges: &[RelativeRotation],
        r: &[Rotation3<f64>],
        truth: &[Rotation3<f64>],
        thr: f64,
    ) -> f64 {
        edges
            .iter()
            .filter(|e| edge_residual(e, truth) < thr)
            .map(|e| e.weight * edge_residual(e, r).powi(2))
            .sum()
    }

    fn stats(err: &[f64]) -> (f64, f64) {
        let mean = err.iter().sum::<f64>() / err.len() as f64;
        let max = err.iter().cloned().fold(0.0, f64::max);
        (mean.to_degrees(), max.to_degrees())
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

    #[test]
    fn noisy_graph_error_below_edge_noise() {
        let (truth, pairs) = scene_graph();
        for sigma_deg in [0.5, 1.0, 2.0] {
            let mut rng = Rng(11);
            let sigma = f64::to_radians(sigma_deg);
            let (edges, _) = make_edges(&truth, &pairs, sigma, 0, &mut rng);
            // 이상치 문턱은 잡음의 5배 이상으로 둔다(간선 각 오차 평균 ≈ 1.6σ).
            let cfg = AveragingConfig {
                outlier_threshold_rad: (5.0 * sigma).max(5f64.to_radians()),
                ..Default::default()
            };
            let res = average_rotations(truth.len(), &edges, &cfg).unwrap();
            let err = aligned_errors(&res.rotations, &truth);
            let (mean, max) = stats(&err);
            // 간선 하나의 각 오차 기댓값은 σ·√(8/π) ≈ 1.6σ.
            println!(
                "간선 σ={sigma_deg}°: 평균 {mean:.3}° 최대 {max:.3}° 반복 {}",
                res.iterations
            );
            // 최소제곱 해이므로 정답보다 비용이 크면 안 된다(전역 최적에 도달했다는 증거).
            let est: Vec<_> = res.rotations.iter().map(|r| r.unwrap()).collect();
            let (ce, ct) = (
                cost(&edges, &est, &truth, cfg.outlier_threshold_rad),
                cost(&edges, &truth, &truth, cfg.outlier_threshold_rad),
            );
            println!("  비용 추정 {ce:.4e} 정답 {ct:.4e}");
            assert!(ce <= ct * (1.0 + 1e-9), "비용 {ce} > 정답 {ct}");
            // 축당 σ 잡음 간선 942개, 80칸 사슬: 실측 평균 ≈ 1.06σ. 여유 1.5σ.
            assert!(mean < 1.5 * sigma_deg, "σ={sigma_deg}° 평균 {mean}°");
            assert!(res.inliers.iter().filter(|&&b| !b).count() <= edges.len() / 100);
        }
    }

    #[test]
    fn outlier_edges_are_rejected() {
        let (truth, pairs) = scene_graph();
        let mut rng = Rng(5);
        // 간선 10개 중 1개를 임의 회전(σ=1rad ≈ 57°)으로 오염.
        let (edges, is_outlier) = make_edges(&truth, &pairs, 1f64.to_radians(), 10, &mut rng);
        let cfg = AveragingConfig::default();
        let res = average_rotations(truth.len(), &edges, &cfg).unwrap();
        let err = aligned_errors(&res.rotations, &truth);
        let (mean, max) = stats(&err);
        let thr = cfg.outlier_threshold_rad;
        let true_bad: Vec<bool> = edges
            .iter()
            .map(|e| edge_residual(e, &truth) >= thr)
            .collect();
        let caught = (0..edges.len())
            .filter(|&k| true_bad[k] && !res.inliers[k])
            .count();
        let missed = (0..edges.len())
            .filter(|&k| true_bad[k] && res.inliers[k])
            .count();
        let false_rej = (0..edges.len())
            .filter(|&k| !true_bad[k] && !res.inliers[k])
            .count();
        println!(
            "이상치 {}개(정답 문턱 초과 {}개): 잡음 {caught} 잡음 놓침 {missed} 오거부 {false_rej}, 평균 {mean:.3}° 최대 {max:.3}°",
            is_outlier.iter().filter(|&&b| b).count(),
            true_bad.iter().filter(|&&b| b).count()
        );
        let est: Vec<_> = res.rotations.iter().map(|r| r.unwrap()).collect();
        let (ce, ct) = (
            cost(&edges, &est, &truth, cfg.outlier_threshold_rad),
            cost(&edges, &truth, &truth, cfg.outlier_threshold_rad),
        );
        println!("  비용 추정 {ce:.4e} 정답 {ct:.4e}");
        assert!(ce <= ct * (1.0 + 1e-9), "비용 {ce} > 정답 {ct}");
        assert!(mean < 1.5 && max < 4.0, "평균 {mean}° 최대 {max}°");
        assert_eq!(missed, 0);
        assert!(false_rej <= edges.len() / 100);
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
        // 0-1 과 2-3 두 성분: 기준(0)과 이어진 것만 돌려준다.
        let res = average_rotations(4, &[e(0, 1), e(2, 3)], &cfg).unwrap();
        assert!(res.rotations[0].is_some() && res.rotations[1].is_some());
        assert!(res.rotations[2].is_none() && res.rotations[3].is_none());
        assert!(res.residuals_rad[1].is_nan() && !res.inliers[1]);
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
}
