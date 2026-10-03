//! 이동 평균(방향 제약 위치 추정): 전역 회전 + 짝별 이동 방향 → 카메라 중심.
//!
//! 규약은 `two_view::RelativePose` 와 같다: x_j = R_ij x_i + t_ij, 세계→카메라 회전 R_i.
//! 그러면 t_ij ∝ R_j (c_i − c_j) 이므로 세계 방향 d_ij = R_jᵀ t_ij / |·| 는 c_j 에서 c_i 로 향한다.
//!
//! 1. 거르기: (a) 짝의 상대 회전이 전역 회전 R_j R_iᵀ 와 문턱 이상 다르면 버린다(회전 일관성).
//!    (b) 세 간선이 모두 있는 삼각형마다 양의 길이로 닫히는지(d_ij a + d_jk b + d_ki c = 0, a,b,c > 0)
//!    본다. 퇴화(거의 한 직선) 삼각형은 판정하지 않는다. 판정된 삼각형의 절반 이상에서 어긋나는 간선을 버린다.
//! 2. 초기값: 방향 제약 최소제곱 min Σ w |(I − d dᵀ)(c_i − c_j)|², 게이지는 c_ref = 0 과
//!    Σ d_ij·(c_i − c_j) = E(평균 길이 1). 라그랑주 승수로 푼다. 가중치를 각 잔차로 몇 번 다시 매긴다.
//! 3. 정밀화: 최소 비 편차형 손실 min Σ |c_i − c_j − s_ij d_ij|, s_ij ≥ 1 (축척 게이지) 을
//!    IRLS 로 푼다. s 는 고정 c 에 대한 닫힌 해 max(1, d·(c_i − c_j)), c 는 가중 그래프 라플라스 연립.
//! 4. 남은 간선 가운데 각 잔차가 문턱 이하인 것만으로 연결 성분을 다시 구하고, 가장 큰 성분 안에서
//!    방향이 서로 평행하지 않은 간선을 두 개 이상 가진 정점만 등록으로 표시한다(한 직선 위 정점은 위치가 정해지지 않는다).

use crate::math::{Matrix3, Point3, Rotation3, Vector3};
use crate::two_view::RelativePose;
use nalgebra::{DMatrix, DVector};

/// 짝별 이동 방향 관측 하나.
#[derive(Clone, Debug)]
pub struct RelativeTranslation {
    pub i: usize,
    pub j: usize,
    /// 카메라 j 좌표계의 t_ij 방향(길이 무관).
    pub direction: Vector3<f64>,
    /// 짝의 상대 회전 R_ij(있으면 회전 일관성 검사에 쓴다).
    pub rotation: Option<Rotation3<f64>>,
    /// 신뢰도. 0 이하·NaN 이면 쓰지 않는다.
    pub weight: f64,
}

impl RelativeTranslation {
    /// 두 시점 자세에서 만든다. 이동을 관측할 수 없는 짝은 None.
    pub fn from_pose(i: usize, j: usize, pose: &RelativePose, weight: f64) -> Option<Self> {
        (pose.translation_observable && pose.translation.norm() > 0.0).then_some(Self {
            i,
            j,
            direction: pose.translation,
            rotation: Some(pose.rotation),
            weight,
        })
    }
}

/// 카메라에서 장면 점으로 향하는 관측 방향 하나(특징 트랙의 한 관측).
///
/// 점을 미지 정점으로 두고 c_p − c_cam ∝ R_camᵀ b 를 카메라 짝 방향과 같은 방향 제약으로 쓴다.
/// 한 줄로 나는 카메라 사슬은 짝 방향만으로는 간선 길이(사슬 방향 위치)가 정해지지 않지만, 줄 밖의 점이
/// 카메라–카메라–점 삼각형을 만들어 길이를 정한다(카메라–점 방향 제약 평균).
#[derive(Clone, Debug)]
pub struct PointObservation {
    pub camera: usize,
    /// 점 번호(0..점 수).
    pub point: usize,
    /// 카메라 좌표계의 점 방향(길이 무관, 정규화 좌표 (x, y, 1) 이면 된다).
    pub bearing: Vector3<f64>,
    /// 신뢰도. 0 이하·NaN 이면 쓰지 않는다.
    pub weight: f64,
}

/// 설정.
#[derive(Clone, Debug)]
pub struct TranslationConfig {
    /// 짝 회전과 전역 회전이 이보다 다르면 버린다(rad).
    pub rotation_consistency_rad: f64,
    /// 삼각형 닫힘 문턱(각, rad).
    pub triplet_threshold_rad: f64,
    /// 이 각보다 좁은 삼각형은 퇴화로 보고 판정하지 않는다(rad).
    pub triplet_min_angle_rad: f64,
    /// 초기 최소제곱의 재가중 횟수.
    pub init_iterations: usize,
    /// 강건 정밀화 IRLS 횟수.
    pub irls_iterations: usize,
    /// 최종 이상치 판정 각 문턱(rad).
    pub outlier_threshold_rad: f64,
    /// 등록 판정에서 두 간선이 평행하지 않다고 볼 최소 각(rad).
    pub rigidity_angle_rad: f64,
    /// 정점이 이보다 많으면 정밀화 연립을 밀집 LU 대신 블록 야코비 선조건 켤레 기울기로 푼다.
    pub dense_max_vertices: usize,
    /// 점 단계: 1단계 중심에서 점을 삼각측량할 때 정상 광선으로 볼 각(rad).
    pub point_gate_rad: f64,
    /// 점 단계 IRLS 의 코시 가중 각 척도(rad): 가중 × 1 / (1 + (각 잔차 / 척도)²).
    pub robust_sigma_rad: f64,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            rotation_consistency_rad: 5f64.to_radians(),
            triplet_threshold_rad: 6f64.to_radians(),
            triplet_min_angle_rad: 3f64.to_radians(),
            init_iterations: 5,
            irls_iterations: 60,
            outlier_threshold_rad: 6f64.to_radians(),
            rigidity_angle_rad: 3f64.to_radians(),
            dense_max_vertices: 400,
            point_gate_rad: 10f64.to_radians(),
            robust_sigma_rad: 2f64.to_radians(),
        }
    }
}

/// 결과.
#[derive(Clone, Debug)]
pub struct TranslationResult {
    /// 카메라 중심(등록 안 된 정점은 None). 축척·원점은 임의.
    pub centers: Vec<Option<Point3<f64>>>,
    /// 점 위치(점 관측을 줄 때만, 등록 안 된 점은 None). 카메라 중심과 같은 틀.
    pub points: Vec<Option<Point3<f64>>>,
    /// 입력 간선별 최종 각 잔차(rad). 쓰지 않은 간선은 NaN.
    pub residuals_rad: Vec<f64>,
    /// 입력 간선별 정상 여부.
    pub inliers: Vec<bool>,
    /// 거르기 단계별로 버린 간선 수: [무효·회전 불일치, 삼각형 불일치].
    pub rejected: [usize; 2],
}

impl TranslationResult {
    pub fn registered(&self) -> usize {
        self.centers.iter().filter(|c| c.is_some()).count()
    }
}

struct Edge {
    idx: usize,
    i: usize,
    j: usize,
    d: Vector3<f64>,
    w: f64,
}

fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

/// 간선 집합에서 각 정점의 연결 성분 번호(간선이 없는 정점은 None)와 가장 큰 성분 번호.
fn components(n: usize, edges: &[(usize, usize)]) -> (Vec<Option<usize>>, Option<usize>) {
    let mut adj = vec![Vec::new(); n];
    for &(a, b) in edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut comp = vec![None; n];
    let mut sizes = Vec::new();
    for s in 0..n {
        if comp[s].is_some() || adj[s].is_empty() {
            continue;
        }
        let id = sizes.len();
        let mut stack = vec![s];
        comp[s] = Some(id);
        let mut size = 0;
        while let Some(v) = stack.pop() {
            size += 1;
            for &u in &adj[v] {
                if comp[u].is_none() {
                    comp[u] = Some(id);
                    stack.push(u);
                }
            }
        }
        sizes.push(size);
    }
    let best = (0..sizes.len()).max_by_key(|&k| sizes[k]);
    (comp, best)
}

/// 삼각형 닫힘 검사로 간선을 거른다. 반환: 간선별 유지 여부.
fn triplet_filter(n: usize, edges: &[Edge], cfg: &TranslationConfig) -> Vec<bool> {
    use std::collections::HashMap;
    // 정점 짝 → (간선 번호, 방향 c_lo − c_hi)
    let mut map: HashMap<(usize, usize), (usize, Vector3<f64>)> = HashMap::new();
    let mut adj = vec![Vec::new(); n];
    for (k, e) in edges.iter().enumerate() {
        let (lo, hi, d) = if e.i < e.j {
            (e.i, e.j, e.d)
        } else {
            (e.j, e.i, -e.d)
        };
        if map.insert((lo, hi), (k, d)).is_none() {
            adj[lo].push(hi);
            adj[hi].push(lo);
        }
    }
    let dir = |a: usize, b: usize| -> Option<(usize, Vector3<f64>)> {
        if a < b {
            map.get(&(a, b)).copied()
        } else {
            map.get(&(b, a)).map(|&(k, d)| (k, -d))
        }
    };
    let mut good = vec![0usize; edges.len()];
    let mut bad = vec![0usize; edges.len()];
    let sin_min = cfg.triplet_min_angle_rad.sin();
    for a in 0..n {
        for &b in adj[a].iter().filter(|&&b| b > a) {
            for &c in adj[b].iter().filter(|&&c| c > b) {
                let (Some((kab, u)), Some((kbc, v)), Some((kca, w))) =
                    (dir(a, b), dir(b, c), dir(c, a))
                else {
                    continue;
                };
                // u = c_a − c_b, v = c_b − c_c, w = c_c − c_a. α u + β v + w = 0, α, β > 0.
                let uv = u.cross(&v);
                if uv.norm() < sin_min
                    || u.cross(&w).norm() < sin_min
                    || v.cross(&w).norm() < sin_min
                {
                    continue;
                }
                let m = nalgebra::Matrix3x2::from_columns(&[u, v]);
                let ok = (m.transpose() * m)
                    .try_inverse()
                    .map(|inv| {
                        let ab = inv * (m.transpose() * (-w));
                        let fit = m * ab;
                        ab[0] > 0.0
                            && ab[1] > 0.0
                            && angle_between(&fit, &(-w)) < cfg.triplet_threshold_rad
                    })
                    .unwrap_or(false);
                for k in [kab, kbc, kca] {
                    if ok {
                        good[k] += 1;
                    } else {
                        bad[k] += 1;
                    }
                }
            }
        }
    }
    (0..edges.len())
        .map(|k| bad[k] == 0 || good[k] > bad[k])
        .collect()
}

/// 방향 제약 최소제곱 초기값(가중치 고정). 정점 번호는 `local` 번호(0..m), ref = 0.
fn constrained_ls(m: usize, edges: &[Edge], w: &[f64]) -> Option<Vec<Vector3<f64>>> {
    // 미지수: c_1..c_{m−1} (c_0 = 0), 승수 λ.
    let dim = 3 * (m - 1) + 1;
    let mut a = DMatrix::<f64>::zeros(dim, dim);
    let mut b = DVector::<f64>::zeros(dim);
    let col = |v: usize| (v > 0).then(|| 3 * (v - 1));
    let mut total = 0.0;
    for (e, &we) in edges.iter().zip(w) {
        let p = (Matrix3::identity() - e.d * e.d.transpose()) * we;
        let (ci, cj) = (col(e.i), col(e.j));
        for (x, sx) in [(ci, 1.0), (cj, -1.0)] {
            for (y, sy) in [(ci, 1.0), (cj, -1.0)] {
                if let (Some(x), Some(y)) = (x, y) {
                    let mut blk = a.view_mut((x, y), (3, 3));
                    blk += p * (sx * sy);
                }
            }
        }
        // 제약 Σ d·(c_i − c_j) = E
        for (x, s) in [(ci, 1.0), (cj, -1.0)] {
            if let Some(x) = x {
                for r in 0..3 {
                    a[(x + r, dim - 1)] += s * e.d[r];
                    a[(dim - 1, x + r)] += s * e.d[r];
                }
            }
        }
        total += 1.0;
    }
    b[dim - 1] = total;
    // 정칙화 항을 두지 않는다: c_0 = 0 과 축척 제약으로 게이지가 모두 고정되므로 방향 강성 그래프에서
    // 행렬은 정칙이다. 예전의 대각 1e-9 는 긴 사슬(최소 고유값 ~1e-4)에서 중심을 1e-4 상대만큼 끌어당겼다.
    let sol = a.lu().solve(&b)?;
    Some(
        (0..m)
            .map(|v| match col(v) {
                Some(x) => Vector3::new(sol[x], sol[x + 1], sol[x + 2]),
                None => Vector3::zeros(),
            })
            .collect(),
    )
}

/// 고정 가중치에서 min Σ w |c_i − c_j − s d|² 를 c 와 s 에 대해 함께 푼다(c_0 = 0).
/// `clamped` 간선은 s = 1 로 고정(항 w|c_i − c_j − d|²), 나머지는 s 를 소거해 w|(I − d dᵀ)(c_i − c_j)|².
fn joint_solve(
    m: usize,
    edges: &[Edge],
    w: &[f64],
    clamped: &[bool],
    start: &[Vector3<f64>],
    dense_max: usize,
) -> Option<Vec<Vector3<f64>>> {
    if m > dense_max {
        return joint_solve_pcg(m, edges, w, clamped, start);
    }
    let dim = 3 * (m - 1);
    let mut a = DMatrix::<f64>::zeros(dim, dim);
    let mut b = DVector::<f64>::zeros(dim);
    let col = |v: usize| (v > 0).then(|| 3 * (v - 1));
    for ((e, &we), &cl) in edges.iter().zip(w).zip(clamped) {
        let p = if cl {
            Matrix3::identity() * we
        } else {
            (Matrix3::identity() - e.d * e.d.transpose()) * we
        };
        let (ci, cj) = (col(e.i), col(e.j));
        for (x, sx) in [(ci, 1.0), (cj, -1.0)] {
            if let Some(x) = x {
                for (y, sy) in [(ci, 1.0), (cj, -1.0)] {
                    if let Some(y) = y {
                        let mut blk = a.view_mut((x, y), (3, 3));
                        blk += p * (sx * sy);
                    }
                }
                if cl {
                    for r in 0..3 {
                        b[x + r] += sx * we * e.d[r];
                    }
                }
            }
        }
    }
    let sol = a.lu().solve(&b)?;
    Some(
        (0..m)
            .map(|v| match col(v) {
                Some(x) => Vector3::new(sol[x], sol[x + 1], sol[x + 2]),
                None => Vector3::zeros(),
            })
            .collect(),
    )
}

/// [`joint_solve`] 와 같은 연립을 행렬 없이 블록 야코비 선조건 켤레 기울기로 푼다(Hestenes–Stiefel 1952;
/// 선조건은 정점별 3×3 대각 블록의 역). 행렬은 c_0 = 0 으로 줄인 그래프 라플라스형 대칭 양의 정부호이다.
/// 시작값은 직전 IRLS 해라서 반복이 적다.
fn joint_solve_pcg(
    m: usize,
    edges: &[Edge],
    w: &[f64],
    clamped: &[bool],
    start: &[Vector3<f64>],
) -> Option<Vec<Vector3<f64>>> {
    let ps: Vec<Matrix3<f64>> = edges
        .iter()
        .zip(w)
        .zip(clamped)
        .map(|((e, &we), &cl)| {
            if cl {
                Matrix3::identity() * we
            } else {
                (Matrix3::identity() - e.d * e.d.transpose()) * we
            }
        })
        .collect();
    let apply = |x: &[Vector3<f64>], y: &mut Vec<Vector3<f64>>| {
        y.iter_mut().for_each(|v| *v = Vector3::zeros());
        for (e, p) in edges.iter().zip(&ps) {
            let v = p * (x[e.i] - x[e.j]);
            y[e.i] += v;
            y[e.j] -= v;
        }
        y[0] = Vector3::zeros();
    };
    let mut b = vec![Vector3::zeros(); m];
    let mut diag = vec![Matrix3::zeros(); m];
    for (((e, &we), &cl), p) in edges.iter().zip(w).zip(clamped).zip(&ps) {
        if cl {
            b[e.i] += e.d * we;
            b[e.j] -= e.d * we;
        }
        diag[e.i] += p;
        diag[e.j] += p;
    }
    b[0] = Vector3::zeros();
    let dinv: Vec<Matrix3<f64>> = diag
        .iter()
        .map(|d| {
            let reg = Matrix3::identity() * (1e-9 * d.trace()).max(1e-300);
            (d + reg).try_inverse().unwrap_or_else(Matrix3::identity)
        })
        .collect();
    let dot = |a: &[Vector3<f64>], b: &[Vector3<f64>]| {
        a.iter().zip(b).map(|(x, y)| x.dot(y)).sum::<f64>()
    };
    let mut x: Vec<Vector3<f64>> = start.iter().map(|v| v - start[0]).collect();
    let mut ax = vec![Vector3::zeros(); m];
    apply(&x, &mut ax);
    let mut r: Vec<Vector3<f64>> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let bnorm = dot(&b, &b).sqrt().max(1e-300);
    let mut z: Vec<Vector3<f64>> = dinv.iter().zip(&r).map(|(d, r)| d * r).collect();
    z[0] = Vector3::zeros();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![Vector3::zeros(); m];
    for _ in 0..20 * m {
        if dot(&r, &r).sqrt() <= 1e-13 * bnorm {
            break;
        }
        apply(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap.is_nan() || pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for k in 0..m {
            x[k] += p[k] * alpha;
            r[k] -= ap[k] * alpha;
        }
        for k in 0..m {
            z[k] = dinv[k] * r[k];
        }
        z[0] = Vector3::zeros();
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

/// 전역 회전(세계→카메라)과 짝별 이동 방향으로 카메라 중심을 구한다.
pub fn average_translations(
    rotations: &[Option<Rotation3<f64>>],
    observations: &[RelativeTranslation],
    cfg: &TranslationConfig,
) -> TranslationResult {
    average_core(rotations, observations, &[], cfg, None)
}

/// [`average_translations`] 에 카메라→점 방향 관측을 더한다. 점은 정점 n_cam.. 으로 함께 푼다.
///
/// 두 단계로 푼다. 점 방향 이상치가 1차 최소제곱 해를 끌면 6° 거르기 뒤 그래프가 쪼개지므로,
/// (1) 카메라 짝 방향만으로 중심을 구하고, (2) 그 중심에서 점을 강건하게 삼각측량하고(두 광선 중점 가설 +
/// 정상 광선 수 최대), 등록되지 않은 카메라는 점 방향으로 선형 위치 추정(회전 고정)한 뒤, (3) 이 값에서
/// 시작해 짝·점 간선을 함께 각 잔차 기반 강건 재가중(IRLS)으로 정밀화한다. 1차 최소제곱을 다시 풀지 않는다.
pub fn average_translations_with_points(
    rotations: &[Option<Rotation3<f64>>],
    observations: &[RelativeTranslation],
    point_observations: &[PointObservation],
    cfg: &TranslationConfig,
) -> TranslationResult {
    if point_observations.is_empty() {
        return average_core(rotations, observations, &[], cfg, None);
    }
    let n_cam = rotations.len();
    let n_pts = point_observations
        .iter()
        .map(|o| o.point + 1)
        .max()
        .unwrap_or(0);
    let stage = average_core(rotations, observations, &[], cfg, None);
    if stage.registered() < 2 {
        return average_core(rotations, observations, point_observations, cfg, None);
    }
    // 세계 방향 광선.
    let rotations = &finite_rotations(rotations);
    let rays: Vec<Option<Vector3<f64>>> = point_observations
        .iter()
        .map(|o| {
            let ok = o.camera < n_cam
                && o.point < n_pts
                && o.weight > 0.0
                && o.weight.is_finite()
                && o.bearing.norm() > 1e-12
                && o.bearing.iter().all(|x| x.is_finite());
            ok.then(|| rotations[o.camera])
                .flatten()
                .map(|r| (r.inverse() * o.bearing).normalize())
        })
        .collect();
    let mut by_point = vec![Vec::new(); n_pts];
    for (k, o) in point_observations.iter().enumerate() {
        if rays[k].is_some() {
            by_point[o.point].push(k);
        }
    }
    let gate = cfg.point_gate_rad;
    let mut start: Vec<Option<Vector3<f64>>> = vec![None; n_cam + n_pts];
    for (v, c) in stage.centers.iter().enumerate() {
        start[v] = c.map(|c| c.coords);
    }
    // 점 삼각측량(등록된 카메라의 광선만).
    for (p, ks) in by_point.iter().enumerate() {
        let obs: Vec<(Vector3<f64>, Vector3<f64>)> = ks
            .iter()
            .filter_map(|&k| {
                let c = start[point_observations[k].camera]?;
                Some((c, rays[k]?))
            })
            .collect();
        start[n_cam + p] = robust_ray_point(&obs, gate);
    }
    // 등록되지 않은 카메라: 회전 고정, 삼각측량된 점으로 선형 위치 추정(c 에 대해 같은 식).
    let mut by_cam = vec![Vec::new(); n_cam];
    for (k, o) in point_observations.iter().enumerate() {
        if rays[k].is_some() {
            by_cam[o.camera].push(k);
        }
    }
    for cam in 0..n_cam {
        if start[cam].is_some() {
            continue;
        }
        let obs: Vec<(Vector3<f64>, Vector3<f64>)> = by_cam[cam]
            .iter()
            .filter_map(|&k| Some((start[n_cam + point_observations[k].point]?, -rays[k]?)))
            .collect();
        start[cam] = robust_ray_point(&obs, gate);
    }
    // 카메라와 점을 번갈아 다듬는다(점 이상치가 있으면 1단계 중심 자체가 틀려 있을 수 있다). 카메라는 자기를
    // 본 점들의 광선 교차 위치 가설 중 정상 광선이 더 많은 쪽으로 바꾸고, 점은 새 중심에서 다시 삼각측량한다.
    // 문턱은 10° 에서 점차 줄인다.
    for round in 0..4 {
        let g = gate * [1.0, 0.6, 0.4, 0.3][round];
        for cam in 0..n_cam {
            let pts: Vec<(Vector3<f64>, Vector3<f64>)> = by_cam[cam]
                .iter()
                .filter_map(|&k| Some((start[n_cam + point_observations[k].point]?, rays[k]?)))
                .collect();
            let lines: Vec<(Vector3<f64>, Vector3<f64>)> =
                pts.iter().map(|(x, r)| (*x, -*r)).collect();
            let Some(cand) = robust_ray_point(&lines, g) else {
                continue;
            };
            let support = |c: &Vector3<f64>| {
                pts.iter()
                    .filter(|(x, r)| angle_between(&(x - c), r) <= g)
                    .count()
            };
            if start[cam].is_none_or(|cur| support(&cand) > support(&cur)) {
                start[cam] = Some(cand);
            }
        }
        for (p, ks) in by_point.iter().enumerate() {
            let obs: Vec<(Vector3<f64>, Vector3<f64>)> = ks
                .iter()
                .filter_map(|&k| Some((start[point_observations[k].camera]?, rays[k]?)))
                .collect();
            if let Some(x) = robust_ray_point(&obs, g) {
                start[n_cam + p] = Some(x);
            }
        }
    }
    let refined = average_core(
        rotations,
        observations,
        point_observations,
        cfg,
        Some(&start),
    );
    // 안전장치: 정밀화가 출발 해보다 나쁘면(등록 수·짝 간선 정상 수 감소, 같으면 짝 각 잔차 중앙 증가)
    // 출발 해(1단계 중심 + 그 중심에서 삼각측량한 점)로 되돌린다.
    let (a, b) = (
        pair_score(&refined, observations.len()),
        pair_score(&stage, observations.len()),
    );
    if a.0 < b.0 || a.1 < b.1 || (a.1 == b.1 && a.2 < b.2) {
        let mut back = stage;
        // 출발 해의 점으로 카메라마다 광선 교차 위치를 다시 구해, 짝 간선만으로는 등록되지 못했거나(강성 부족)
        // 점 광선 지지가 눈에 띄게 적은 카메라를 광선 교차 위치로 바꾼다.
        for (cam, cam_obs) in by_cam.iter().enumerate().take(n_cam) {
            let pts: Vec<(Vector3<f64>, Vector3<f64>, Vector3<f64>)> = cam_obs
                .iter()
                .filter_map(|&k| {
                    let x = start[n_cam + point_observations[k].point]?;
                    Some((x, rays[k]?, rays[k]?))
                })
                .collect();
            let lines: Vec<(Vector3<f64>, Vector3<f64>)> =
                pts.iter().map(|(x, r, _)| (*x, -*r)).collect();
            let Some(cand) = robust_ray_point(&lines, gate) else {
                continue;
            };
            let support = |c: &Vector3<f64>| {
                pts.iter()
                    .filter(|(x, r, _)| angle_between(&(x - c), r) <= gate)
                    .count()
            };
            let better = match back.centers[cam] {
                None => true,
                Some(cur) => support(&cand) >= support(&cur.coords) + RAY_SWAP_MARGIN,
            };
            if better {
                back.centers[cam] = Some(Point3::from(cand));
            }
        }
        // 정밀화와 같은 셈: 쓰지 못한 점 관측도 무효로 센다.
        back.rejected[0] += rays.iter().filter(|r| r.is_none()).count();
        back.points = start[n_cam..].iter().map(|x| x.map(Point3::from)).collect();
        return back;
    }
    refined
}

/// 후보 해의 짝 간선 비용: Σ min(각 잔차, 문턱)² (등록되지 않은 끝점이 있으면 문턱). 작을수록 낫다.
fn truncated_pair_cost(
    rotations: &[Option<Rotation3<f64>>],
    observations: &[RelativeTranslation],
    centers: &[Option<Point3<f64>>],
    cap: f64,
) -> f64 {
    let mut cost = 0.0;
    for o in observations {
        let r = (|| {
            let (ci, cj) = (centers.get(o.i)?.as_ref()?, centers.get(o.j)?.as_ref()?);
            let rj = rotations.get(o.j).copied().flatten()?;
            let d = rj.inverse() * o.direction;
            let v = ci - cj;
            (v.norm() > 1e-12 && d.norm() > 1e-12).then(|| angle_between(&v, &d))
        })()
        .unwrap_or(cap)
        .min(cap);
        cost += r * r;
    }
    cost
}

/// 1차원 투영 순서 일관성 거르기(Wilson & Snavely 2014). 회전 평균 결과로 세계 좌표로 돌린 상대 방향 d_ij
/// (c_i − c_j 방향)를 무작위 축 h 에 투영한 x = ⟨d_ij, h⟩ 가 양이면 "i 가 j 뒤", 음이면 "i 가 j 앞" 이라는
/// 순서 제약이다. 카메라 순서는 간선 부호의 보르다 점수로 구하고(최소 되먹임 호 집합의 근사), 순서를
/// 거스르는 간선(|x| > 허용치)을 센다. 축 `axes` 개에서 어긋난 비율이 `max_frac` 를 넘는 간선을 버린다.
/// 반환: 입력 관측별 유지 여부.
fn projection_filter(
    rotations: &[Option<Rotation3<f64>>],
    observations: &[RelativeTranslation],
    axes: usize,
    tol: f64,
    max_frac: f64,
    seed: u64,
) -> Vec<bool> {
    let n = rotations.len();
    let mut world: Vec<Option<Vector3<f64>>> = Vec::with_capacity(observations.len());
    for o in observations {
        let d = (o.i < n && o.j < n && o.i != o.j && o.direction.iter().all(|x| x.is_finite()))
            .then(|| rotations[o.j].map(|r| r.inverse() * o.direction))
            .flatten()
            .filter(|d| d.norm() > 1e-12)
            .map(|d| d.normalize());
        world.push(d);
    }
    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let gauss = move |n: &mut dyn FnMut() -> u64| {
        let u1 = ((n() >> 11) as f64 / (1u64 << 53) as f64).max(1e-300);
        let u2 = (n() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut bad = vec![0usize; observations.len()];
    let mut used = vec![0usize; observations.len()];
    for _ in 0..axes {
        let h = Vector3::new(gauss(&mut next), gauss(&mut next), gauss(&mut next));
        if h.norm() < 1e-9 {
            continue;
        }
        let h = h.normalize();
        // 순서 점수: 각 간선이 말하는 방향으로 두 끝점에 ±부호를 준다.
        let mut score = vec![0f64; n];
        let x: Vec<f64> = world.iter().map(|d| d.map_or(0.0, |d| d.dot(&h))).collect();
        for (k, o) in observations.iter().enumerate() {
            if world[k].is_some() {
                let s = x[k].signum();
                score[o.i] += s;
                score[o.j] -= s;
            }
        }
        for (k, o) in observations.iter().enumerate() {
            if world[k].is_none() || x[k].abs() < tol {
                continue; // 축에 거의 수직인 간선은 순서를 말하지 못한다.
            }
            used[k] += 1;
            // i 가 j 뒤(x > 0)여야 점수 순서도 i 가 높아야 한다.
            if (score[o.i] - score[o.j]) * x[k] < 0.0 {
                bad[k] += 1;
            }
        }
    }
    (0..observations.len())
        .map(|k| used[k] == 0 || (bad[k] as f64) <= max_frac * used[k] as f64)
        .collect()
}

/// 여러 후보 중 짝 간선 비용이 가장 작은 해를 고르는 위치 평균.
///
/// 후보: (1) 점 포함 2단계 해(`average_translations_with_points`), (2) 짝만 푼 1단계 해, (3) 1차원 투영 순서
/// 일관성으로 이상치 간선을 거른 뒤 짝만 푼 해, (4) 무작위 시작값 3개에서 시작해 코시 척도를 각 잔차
/// 중앙값(1.4826 배, 2°~10° 로 제한)으로 정하고 반복 재가중한 해. 비용은 min(각 잔차, 6°)² 의 합이다.
pub fn average_translations_robust(
    rotations: &[Option<Rotation3<f64>>],
    observations: &[RelativeTranslation],
    point_observations: &[PointObservation],
    cfg: &TranslationConfig,
    seed: u64,
) -> TranslationResult {
    let n_cam = rotations.len();
    let rots = finite_rotations(rotations);
    let cap = cfg.outlier_threshold_rad;
    let mut cands: Vec<TranslationResult> = Vec::new();
    cands.push(average_translations_with_points(
        rotations,
        observations,
        point_observations,
        cfg,
    ));
    let stage = average_core(rotations, observations, &[], cfg, None);
    // (3) 투영 순서 거르기 뒤 다시 푼다.
    let keep = projection_filter(&rots, observations, 50, 0.1, 0.25, seed);
    let sub: Vec<RelativeTranslation> = observations
        .iter()
        .zip(&keep)
        .filter(|(_, &k)| k)
        .map(|(o, _)| o.clone())
        .collect();
    let mut filtered = average_core(rotations, &sub, &[], cfg, None);
    {
        // 입력 간선 번호로 되돌린다.
        let mut res = vec![f64::NAN; observations.len()];
        let mut inl = vec![false; observations.len()];
        let kept_idx: Vec<usize> = (0..observations.len()).filter(|&k| keep[k]).collect();
        for (s, &k) in kept_idx.iter().enumerate() {
            res[k] = filtered.residuals_rad.get(s).copied().unwrap_or(f64::NAN);
            inl[k] = filtered.inliers.get(s).copied().unwrap_or(false);
        }
        filtered.residuals_rad = res;
        filtered.inliers = inl;
    }
    filtered.rejected[1] += keep.iter().filter(|k| !**k).count();
    cands.push(filtered);
    // (4) 무작위 시작 + 중앙값 척도.
    let mut rng = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
    let mut unit = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut base = stage.clone();
    for _ in 0..3 {
        let st: Vec<Option<Vector3<f64>>> = (0..n_cam)
            .map(|_| {
                Some(Vector3::new(
                    unit() * 200.0 - 100.0,
                    unit() * 200.0 - 100.0,
                    unit() * 200.0 - 100.0,
                ))
            })
            .collect();
        // 척도: 현재 최선 해의 잔차 중앙값.
        let mut r: Vec<f64> = base
            .residuals_rad
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        let sigma = if r.is_empty() {
            cfg.robust_sigma_rad
        } else {
            let k = r.len() / 2;
            (1.4826 * *r.select_nth_unstable_by(k, f64::total_cmp).1)
                .clamp(2f64.to_radians(), 10f64.to_radians())
        };
        let c2 = TranslationConfig {
            robust_sigma_rad: sigma,
            ..cfg.clone()
        };
        let cand = average_core(rotations, observations, &[], &c2, Some(&st));
        if cand.registered() > 0 {
            base = cand.clone();
        }
        cands.push(cand);
    }
    cands.push(stage);
    let mut best: Option<(f64, usize)> = None;
    let max_reg = cands.iter().map(|c| c.registered()).max().unwrap_or(0);
    for (k, c) in cands.iter().enumerate() {
        // 등록 수가 최대의 99% 보다 적은 후보는 비용이 작아도 고르지 않는다.
        if (c.registered() as f64) < 0.99 * max_reg as f64 {
            continue;
        }
        let cost = truncated_pair_cost(&rots, observations, &c.centers, cap);
        if best.is_none_or(|(b, _)| cost < b) {
            best = Some((cost, k));
        }
    }
    let k = best.map_or(0, |(_, k)| k);
    cands.swap_remove(k)
}

/// 광선 교차 위치로 카메라를 바꿀 때 요구하는 점 광선 지지 수 차이.
const RAY_SWAP_MARGIN: usize = 3;

/// 해의 짝 간선 품질: (등록 수, 짝 간선 정상 수, −짝 각 잔차 중앙). 클수록 낫다.
fn pair_score(res: &TranslationResult, n_obs: usize) -> (usize, usize, f64) {
    let n = n_obs.min(res.inliers.len());
    let inl = res.inliers[..n].iter().filter(|b| **b).count();
    let mut r: Vec<f64> = res.residuals_rad[..n]
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let med = if r.is_empty() {
        f64::INFINITY
    } else {
        let k = r.len() / 2;
        *r.select_nth_unstable_by(k, f64::total_cmp).1
    };
    (res.registered(), inl, -med)
}

/// 원소가 유한하지 않은 회전은 없는 회전(None)으로 바꾼다. NaN 회전은 각 비교가 거짓이 되어 일관성 검사를
/// 그대로 통과하고 연립 전체를 NaN 으로 만든다.
fn finite_rotations(rotations: &[Option<Rotation3<f64>>]) -> Vec<Option<Rotation3<f64>>> {
    rotations
        .iter()
        .map(|r| r.filter(|r| r.matrix().iter().all(|x| x.is_finite())))
        .collect()
}

/// [`robust_ray_point`] 가 시험하는 광선 짝 가설 수의 상한. 정상 비율 0.5 에서 정상 짝을 하나도 못 뽑을
/// 확률은 0.75^200 ≈ 1e-25 이다.
const RAY_HYPOTHESES: usize = 200;

/// 광선들(원점, 단위 방향)이 가장 잘 만나는 점. 두 광선 중점 가설마다 각 `gate` 안에 드는 앞쪽 광선 수를 세어
/// 가장 많은 가설의 정상 광선으로 최소제곱 min Σ |(I − d dᵀ)(x − o)|² 를 푼다. 정상 광선이 2개 미만이면 None.
fn robust_ray_point(rays: &[(Vector3<f64>, Vector3<f64>)], gate: f64) -> Option<Vector3<f64>> {
    let inl = |x: &Vector3<f64>| -> Vec<usize> {
        (0..rays.len())
            .filter(|&k| {
                let v = x - rays[k].0;
                v.dot(&rays[k].1) > 0.0 && angle_between(&v, &rays[k].1) < gate
            })
            .collect()
    };
    let ls = |idx: &[usize]| -> Option<Vector3<f64>> {
        let mut a = Matrix3::zeros();
        let mut b = Vector3::zeros();
        for &k in idx {
            let p = Matrix3::identity() - rays[k].1 * rays[k].1.transpose();
            a += p;
            b += p * rays[k].0;
        }
        a.try_inverse().map(|ai| ai * b)
    };
    let mut best: Vec<usize> = Vec::new();
    let sin_min = 2f64.to_radians().sin();
    let try_pair = |a: usize, b: usize, best: &mut Vec<usize>| {
        if rays[a].1.cross(&rays[b].1).norm() < sin_min {
            return;
        }
        let Some(x) = ls(&[a, b]) else { return };
        let i = inl(&x);
        if i.len() > best.len() {
            *best = i;
        }
    };
    let k = rays.len();
    if k * k.saturating_sub(1) / 2 <= RAY_HYPOTHESES {
        for a in 0..k {
            for b in a + 1..k {
                try_pair(a, b, &mut best);
            }
        }
    } else {
        // 광선 짝 가설을 결정적 의사난수로 RAY_HYPOTHESES 개만 뽑는다: 비용 O(가설 × k).
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ k as u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % k as u64) as usize
        };
        for _ in 0..RAY_HYPOTHESES {
            let (a, b) = (next(), next());
            if a != b {
                try_pair(a, b, &mut best);
            }
        }
    }
    if best.len() < 2 {
        return None;
    }
    let mut x = ls(&best)?;
    for _ in 0..3 {
        let i = inl(&x);
        if i.len() < 2 {
            break;
        }
        x = ls(&i)?;
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// 본 풀이. `start` 가 있으면(전역 번호, 카메라 다음 점) 1차 최소제곱 대신 그 값에서 시작하고,
/// IRLS 가중치에 각 잔차 기반 코시 가중을 곱하며, 시작값이 없는 정점에 닿는 간선은 쓰지 않는다.
fn average_core(
    rotations: &[Option<Rotation3<f64>>],
    observations: &[RelativeTranslation],
    point_observations: &[PointObservation],
    cfg: &TranslationConfig,
    start: Option<&[Option<Vector3<f64>>]>,
) -> TranslationResult {
    let n_cam = rotations.len();
    let n_pts = point_observations
        .iter()
        .map(|o| o.point + 1)
        .max()
        .unwrap_or(0);
    let n = n_cam + n_pts;
    let mut rejected = [0usize; 2];
    let rotations = &finite_rotations(rotations);
    // 1a. 무효·회전 불일치 제거, 세계 방향으로 바꾸기.
    let mut edges = Vec::new();
    for (idx, o) in observations.iter().enumerate() {
        let valid = o.i < n_cam
            && o.j < n_cam
            && o.i != o.j
            && o.weight > 0.0
            && o.weight.is_finite()
            && o.direction.norm() > 1e-12
            && o.direction.iter().all(|x| x.is_finite());
        let (Some(ri), Some(rj)) = (
            valid.then(|| rotations[o.i]).flatten(),
            valid.then(|| rotations[o.j]).flatten(),
        ) else {
            rejected[0] += 1;
            continue;
        };
        if let Some(rij) = o.rotation {
            if crate::math::rotation_angle_between(&rij, &(rj * ri.inverse()))
                > cfg.rotation_consistency_rad
            {
                rejected[0] += 1;
                continue;
            }
        }
        if let Some(st) = start {
            if st[o.i].is_none() || st[o.j].is_none() {
                continue;
            }
        }
        edges.push(Edge {
            idx,
            i: o.i,
            j: o.j,
            d: (rj.inverse() * o.direction).normalize(),
            w: o.weight,
        });
    }
    // 1a'. 카메라→점 방향: 간선 (i = 점, j = 카메라), 세계 방향 d = R_camᵀ b ∝ c_p − c_cam.
    for (k, o) in point_observations.iter().enumerate() {
        let rot = (o.camera < n_cam
            && o.weight > 0.0
            && o.weight.is_finite()
            && o.bearing.norm() > 1e-12
            && o.bearing.iter().all(|x| x.is_finite()))
        .then(|| rotations[o.camera])
        .flatten();
        let Some(r) = rot else {
            rejected[0] += 1;
            continue;
        };
        if let Some(st) = start {
            let (Some(xp), Some(xc)) = (st.get(n_cam + o.point).copied().flatten(), st[o.camera])
            else {
                continue;
            };
            // 시작값에서 이미 크게 어긋난 점 방향은 쓰지 않는다.
            if angle_between(&(xp - xc), &(r.inverse() * o.bearing)) > cfg.point_gate_rad {
                continue;
            }
        }
        edges.push(Edge {
            idx: observations.len() + k,
            i: n_cam + o.point,
            j: o.camera,
            d: (r.inverse() * o.bearing).normalize(),
            w: o.weight,
        });
    }
    // 1b. 삼각형 닫힘.
    let keep = triplet_filter(n, &edges, cfg);
    rejected[1] = keep.iter().filter(|k| !**k).count();
    let edges: Vec<Edge> = edges
        .into_iter()
        .zip(keep)
        .filter_map(|(e, k)| k.then_some(e))
        .collect();

    let n_obs = observations.len();
    let mut residuals_rad = vec![f64::NAN; n_obs];
    let mut inliers = vec![false; n_obs];
    let mut centers = vec![None; n];

    // 가장 큰 성분으로 좁히고 지역 번호를 매긴다.
    let restrict = |edges: Vec<Edge>| -> (Vec<Edge>, Vec<usize>) {
        let pairs: Vec<_> = edges.iter().map(|e| (e.i, e.j)).collect();
        let (comp, best) = components(n, &pairs);
        let Some(best) = best else {
            return (Vec::new(), Vec::new());
        };
        let verts: Vec<usize> = (0..n).filter(|&v| comp[v] == Some(best)).collect();
        let mut local = vec![usize::MAX; n];
        for (k, &v) in verts.iter().enumerate() {
            local[v] = k;
        }
        let edges = edges
            .into_iter()
            .filter(|e| comp[e.i] == Some(best))
            .map(|e| Edge {
                i: local[e.i],
                j: local[e.j],
                ..e
            })
            .collect();
        (edges, verts)
    };
    let (mut ledges, mut verts) = restrict(edges);
    if verts.len() < 2 {
        centers.truncate(n_cam);
        return TranslationResult {
            centers,
            points: vec![None; n_pts],
            residuals_rad,
            inliers,
            rejected,
        };
    }

    let ang_res = |c: &[Vector3<f64>], e: &Edge| {
        let v = c[e.i] - c[e.j];
        if v.norm() < 1e-12 {
            std::f64::consts::PI
        } else {
            angle_between(&v, &e.d)
        }
    };

    let mut c = Vec::new();
    for pass in 0..2 {
        let m = verts.len();
        // 2. 초기값.
        let mut w: Vec<f64> = ledges.iter().map(|e| e.w).collect();
        let mut init = None;
        if let Some(st) = start {
            let base = if pass == 0 {
                verts
                    .iter()
                    .map(|&v| st[v].unwrap_or_default())
                    .collect::<Vec<_>>()
            } else {
                c.clone()
            };
            init = Some(base.iter().map(|x| x - base[0]).collect());
        }
        for _ in 0..if start.is_some() {
            0
        } else {
            cfg.init_iterations.max(1)
        } {
            let Some(sol) = constrained_ls(m, &ledges, &w) else {
                break;
            };
            for (wk, e) in w.iter_mut().zip(&ledges) {
                let r = ang_res(&sol, e);
                *wk = e.w / (1.0 + (r / 0.05).powi(2));
            }
            init = Some(sol);
        }
        let Some(mut cc) = init else {
            break;
        };
        // 축척: 양의 길이의 5% 분위를 1 로 둔다. 최솟값을 쓰면 이상치 간선 하나(길이 ~0)가 축척을 정해
        // 하한 s ≥ 1 이 사실상 그 간선 하나에만 걸리고 나머지는 축척 불변 사영 항만 남아 해가 쏠린다.
        let mut lens: Vec<f64> = ledges
            .iter()
            .map(|e| e.d.dot(&(cc[e.i] - cc[e.j])))
            .filter(|l| *l > 0.0)
            .collect();
        lens.sort_by(|a, b| a.total_cmp(b));
        if let Some(&l) = lens.get(lens.len() / 20) {
            for v in cc.iter_mut() {
                *v /= l;
            }
        }
        // 3. 강건 정밀화(IRLS, L1 노름).
        for _ in 0..cfg.irls_iterations {
            let s: Vec<f64> = ledges
                .iter()
                .map(|e| e.d.dot(&(cc[e.i] - cc[e.j])).max(1.0))
                .collect();
            let w: Vec<f64> = ledges
                .iter()
                .zip(&s)
                .map(|(e, &se)| {
                    let r = (cc[e.i] - cc[e.j] - e.d * se).norm();
                    let robust = if start.is_some() {
                        let a = ang_res(&cc, e) / cfg.robust_sigma_rad;
                        1.0 / (1.0 + a * a)
                    } else {
                        1.0
                    };
                    e.w * robust / r.max(1e-3 * se)
                })
                .collect();
            // s 와 c 를 함께 푼다(교대 갱신은 80칸 사슬에서 수렴이 매우 느리다).
            // 하한 s ≥ 1 에 걸린 간선만 s = 1 로 고정하고, 나머지는 s 를 소거한 사영 항으로 둔다.
            let mut clamped: Vec<bool> = ledges
                .iter()
                .map(|e| e.d.dot(&(cc[e.i] - cc[e.j])) < 1.0)
                .collect();
            if !clamped.iter().any(|&b| b) {
                // 축척 게이지: 가장 짧은 간선 하나는 고정한다.
                if let Some(k) = (0..ledges.len()).min_by(|&a, &b| {
                    let la = ledges[a].d.dot(&(cc[ledges[a].i] - cc[ledges[a].j]));
                    let lb = ledges[b].d.dot(&(cc[ledges[b].i] - cc[ledges[b].j]));
                    la.total_cmp(&lb)
                }) {
                    clamped[k] = true;
                }
            }
            match joint_solve(m, &ledges, &w, &clamped, &cc, cfg.dense_max_vertices) {
                Some(next) => cc = next,
                None => break,
            }
        }
        c = cc;
        if pass == 1 {
            break;
        }
        // 4. 정상 간선만 남기고 한 번 더.
        // 시작값을 쓴 풀이는 2차에서 1차 해에서 이어 가므로 지역 번호가 그대로여야 한다.
        if start.is_some() {
            let keep: Vec<bool> = ledges
                .iter()
                .map(|e| ang_res(&c, e) <= cfg.outlier_threshold_rad)
                .collect();
            let full: Vec<Edge> = ledges
                .into_iter()
                .zip(&keep)
                .filter_map(|(e, &k)| k.then_some(e))
                .map(|e| Edge {
                    i: verts[e.i],
                    j: verts[e.j],
                    ..e
                })
                .collect();
            let mut glob = vec![None; n];
            for (k, &v) in verts.iter().enumerate() {
                glob[v] = Some(c[k]);
            }
            (ledges, verts) = restrict(full);
            if verts.len() < 2 {
                c.clear();
                break;
            }
            c = verts.iter().map(|&v| glob[v].unwrap_or_default()).collect();
            continue;
        }
        let kept: Vec<Edge> = ledges
            .into_iter()
            .filter(|e| ang_res(&c, e) <= cfg.outlier_threshold_rad)
            .map(|e| Edge {
                i: verts[e.i],
                j: verts[e.j],
                ..e
            })
            .collect();
        (ledges, verts) = restrict(kept);
        if verts.len() < 2 {
            c.clear();
            break;
        }
    }
    if c.is_empty() {
        centers.truncate(n_cam);
        return TranslationResult {
            centers,
            points: vec![None; n_pts],
            residuals_rad,
            inliers,
            rejected,
        };
    }
    // 등록 판정: 정상 간선 중 서로 평행하지 않은 것이 둘 이상.
    let mut dirs: Vec<Vec<Vector3<f64>>> = vec![Vec::new(); verts.len()];
    for e in &ledges {
        let r = ang_res(&c, e);
        if e.idx < n_obs {
            residuals_rad[e.idx] = r;
        }
        if r <= cfg.outlier_threshold_rad {
            if e.idx < n_obs {
                inliers[e.idx] = true;
            }
            dirs[e.i].push(e.d);
            dirs[e.j].push(e.d);
        }
    }
    let sin_min = cfg.rigidity_angle_rad.sin();
    for (k, &v) in verts.iter().enumerate() {
        let rigid = dirs[k]
            .iter()
            .any(|a| dirs[k].iter().any(|b| a.cross(b).norm() > sin_min));
        if rigid {
            centers[v] = Some(Point3::from(c[k]));
        }
    }
    let points = centers.split_off(n_cam);
    TranslationResult {
        centers,
        points,
        residuals_rad,
        inliers,
        rejected,
    }
}

/// 카메라 중심의 위치 사전(예: GPS). 방향 제약만으로는 정해지지 않는 축척·한 직선 방향 위치를 정한다.
#[derive(Clone, Debug)]
pub struct PositionPrior {
    pub camera: usize,
    /// 사전 위치(방향 제약과 같은 세계 틀).
    pub position: Point3<f64>,
    /// 위치 잡음 표준편차(m, 축마다). 0 이하·NaN 이면 쓰지 않는다.
    pub sigma: f64,
}

/// 사전 결합 풀이 설정.
#[derive(Clone, Debug)]
pub struct PriorConfig {
    /// 짝 이동 방향 각 잡음 표준편차(rad). 방향 제약 위치 잔차 = 길이 × 각 오차 이므로 길이에 비례해 무게를 준다.
    pub direction_sigma_rad: f64,
    /// 무게를 줄 때 쓰는 최소 간선 길이(m).
    pub min_length: f64,
    /// 첫 회 길이 하한(m). 재가중이 진행되며 `min_length` 로 줄어든다.
    pub start_min_length: f64,
    /// 강건 재가중 횟수.
    pub iterations: usize,
    /// 재가중 첫 회 코시 각 척도(rad). 마지막 회에 `TranslationConfig::robust_sigma_rad` 가 되도록 줄인다.
    pub start_sigma_rad: f64,
}

impl Default for PriorConfig {
    fn default() -> Self {
        Self {
            direction_sigma_rad: 1f64.to_radians(),
            min_length: 0.5,
            start_min_length: 20.0,
            iterations: 30,
            start_sigma_rad: 20f64.to_radians(),
        }
    }
}

/// 위치 사전 결합 선형 풀이: min Σ a_e |(I − d dᵀ)(c_i − c_j)|² + Σ b_i |c_i − g_i|².
/// 정점 수가 `dense_max` 이하면 밀집 숄레스키, 크면 블록 야코비 선조건 켤레 기울기.
fn prior_solve(
    n: usize,
    edges: &[(usize, usize, Matrix3<f64>)],
    prior: &[Option<(Vector3<f64>, f64)>],
    dense_max: usize,
) -> Option<Vec<Vector3<f64>>> {
    let mut diag = vec![Matrix3::zeros(); n];
    let mut rhs = vec![Vector3::zeros(); n];
    for (v, p) in prior.iter().enumerate() {
        if let Some((g, b)) = p {
            diag[v] += Matrix3::identity() * *b;
            rhs[v] += g * *b;
        }
    }
    for (i, j, m) in edges {
        diag[*i] += m;
        diag[*j] += m;
    }
    let sol: Vec<Vector3<f64>> = if n <= dense_max {
        let mut a = DMatrix::<f64>::zeros(3 * n, 3 * n);
        for (v, dv) in diag.iter().enumerate().take(n) {
            a.view_mut((3 * v, 3 * v), (3, 3)).copy_from(dv);
        }
        for (i, j, m) in edges {
            for (r, c, sign) in [(*i, *j, -1.0), (*j, *i, -1.0)] {
                let mut blk = a.view_mut((3 * r, 3 * c), (3, 3));
                blk += m * sign;
            }
        }
        // 어떤 정점도 사전이 없고 간선도 없으면 특이하므로 작은 릿지를 더한다.
        for k in 0..3 * n {
            a[(k, k)] += 1e-9;
        }
        let b = DVector::from_iterator(3 * n, rhs.iter().flat_map(|v| v.iter().cloned()));
        let x = a.cholesky().map(|c| c.solve(&b))?;
        (0..n)
            .map(|v| Vector3::new(x[3 * v], x[3 * v + 1], x[3 * v + 2]))
            .collect()
    } else {
        let inv: Vec<Matrix3<f64>> = diag
            .iter()
            .map(|d| (d + Matrix3::identity() * 1e-9).try_inverse())
            .collect::<Option<_>>()?;
        let mul = |x: &[Vector3<f64>]| -> Vec<Vector3<f64>> {
            let mut y: Vec<Vector3<f64>> = (0..n).map(|v| diag[v] * x[v]).collect();
            for (i, j, m) in edges {
                y[*i] -= m * x[*j];
                y[*j] -= m * x[*i];
            }
            y
        };
        let dot = |a: &[Vector3<f64>], b: &[Vector3<f64>]| -> f64 {
            a.iter().zip(b).map(|(p, q)| p.dot(q)).sum()
        };
        let mut x: Vec<Vector3<f64>> = (0..n).map(|v| inv[v] * rhs[v]).collect();
        let ax = mul(&x);
        let mut r: Vec<Vector3<f64>> = (0..n).map(|v| rhs[v] - ax[v]).collect();
        let mut z: Vec<Vector3<f64>> = (0..n).map(|v| inv[v] * r[v]).collect();
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let tol = 1e-12 * dot(&rhs, &rhs).max(1e-300);
        for _ in 0..3000 {
            if dot(&r, &r) <= tol {
                break;
            }
            let ap = mul(&p);
            let alpha = rz / dot(&p, &ap).max(1e-300);
            for v in 0..n {
                x[v] += p[v] * alpha;
                r[v] -= ap[v] * alpha;
            }
            z = (0..n).map(|v| inv[v] * r[v]).collect();
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz.max(1e-300);
            rz = rz_new;
            for v in 0..n {
                p[v] = z[v] + p[v] * beta;
            }
        }
        x
    };
    sol.iter()
        .all(|v| v.iter().all(|c| c.is_finite()))
        .then_some(sol)
}

/// 위치 사전과 함께 카메라 중심을 구한다(결과는 사전과 같은 틀·미터 단위).
///
/// 방향 제약만으로는 한 줄로 나는 짧은 기선(축척 퇴화)·카메라 간 짝 부족 때문에 위치가 정해지지 않으므로
/// 사전 위치(GPS)를 약한 앵커로 함께 푼다. 간선 무게는 1/(길이·각 잡음)² × 코시 각 가중이고, 코시 척도를
/// 넓은 값에서 `robust_sigma_rad` 까지 줄이며 다시 푼다(처음부터 좁히면 이상치에 끌린다).
/// 등록: 회전·사전이 있고 정상 간선이 하나 이상 있는 카메라.
pub fn average_translations_with_prior(
    rotations: &[Option<Rotation3<f64>>],
    observations: &[RelativeTranslation],
    priors: &[PositionPrior],
    cfg: &TranslationConfig,
    pcfg: &PriorConfig,
) -> TranslationResult {
    let n = rotations.len();
    let rotations = &finite_rotations(rotations);
    let mut rejected = [0usize; 2];
    let mut prior: Vec<Option<(Vector3<f64>, f64)>> = vec![None; n];
    for p in priors {
        let ok = p.camera < n
            && p.sigma > 0.0
            && p.sigma.is_finite()
            && p.position.coords.iter().all(|x| x.is_finite());
        if ok {
            prior[p.camera] = Some((p.position.coords, 1.0 / (p.sigma * p.sigma)));
        }
    }
    // 회전 일관성·유효성.
    let mut edges = Vec::new();
    for (idx, o) in observations.iter().enumerate() {
        let valid = o.i < n
            && o.j < n
            && o.i != o.j
            && o.weight > 0.0
            && o.weight.is_finite()
            && o.direction.norm() > 1e-12
            && o.direction.iter().all(|x| x.is_finite());
        let (Some(ri), Some(rj)) = (
            valid.then(|| rotations[o.i]).flatten(),
            valid.then(|| rotations[o.j]).flatten(),
        ) else {
            rejected[0] += 1;
            continue;
        };
        if let Some(rij) = o.rotation {
            if crate::math::rotation_angle_between(&rij, &(rj * ri.inverse()))
                > cfg.rotation_consistency_rad
            {
                rejected[0] += 1;
                continue;
            }
        }
        if prior[o.i].is_none() && prior[o.j].is_none() {
            // 사전이 없는 정점끼리만 이어진 간선도 쓸 수 있지만 축척이 정해지지 않는다. 여기서는 쓰지 않는다.
            rejected[0] += 1;
            continue;
        }
        edges.push(Edge {
            idx,
            i: o.i,
            j: o.j,
            d: (rj.inverse() * o.direction).normalize(),
            w: o.weight,
        });
    }
    let mut res = TranslationResult {
        centers: vec![None; n],
        points: Vec::new(),
        residuals_rad: vec![f64::NAN; observations.len()],
        inliers: vec![false; observations.len()],
        rejected,
    };
    if edges.is_empty() {
        return res;
    }
    let mut c: Vec<Vector3<f64>> = prior
        .iter()
        .map(|p| p.map(|p| p.0).unwrap_or_else(Vector3::zeros))
        .collect();
    let mut rob = vec![1.0; edges.len()];
    let iters = pcfg.iterations.max(1);
    let (s0, s1) = (
        pcfg.start_sigma_rad,
        cfg.robust_sigma_rad.min(pcfg.start_sigma_rad),
    );
    let proj = |d: &Vector3<f64>| Matrix3::identity() - d * d.transpose();
    let mut sol = c.clone();
    for it in 0..iters {
        let frac = if iters > 1 {
            it as f64 / (iters - 1) as f64
        } else {
            1.0
        };
        let sigma = s0 * (s1 / s0).powf(frac);
        // 길이 하한을 넓은 값에서 `min_length` 로 줄인다: 처음에는 짧은 간선의 무게가 이상치에 끌리지 않게 한다.
        let floor = pcfg.start_min_length * (pcfg.min_length / pcfg.start_min_length).powf(frac);
        let sys: Vec<(usize, usize, Matrix3<f64>)> = edges
            .iter()
            .zip(&rob)
            .map(|(e, r)| {
                let len = (c[e.i] - c[e.j]).norm().max(floor);
                let a = e.w * r / (len * pcfg.direction_sigma_rad).powi(2);
                (e.i, e.j, proj(&e.d) * a)
            })
            .collect();
        let Some(x) = prior_solve(n, &sys, &prior, cfg.dense_max_vertices) else {
            return res;
        };
        sol = x;
        c.clone_from(&sol);
        for (e, r) in edges.iter().zip(rob.iter_mut()) {
            let th = angle_between(&e.d, &(c[e.i] - c[e.j]));
            *r = 1.0 / (1.0 + (th / sigma).powi(2));
        }
    }
    let mut has_edge = vec![false; n];
    for e in &edges {
        let th = angle_between(&e.d, &(sol[e.i] - sol[e.j]));
        res.residuals_rad[e.idx] = th;
        if th <= cfg.outlier_threshold_rad {
            res.inliers[e.idx] = true;
            has_edge[e.i] = true;
            has_edge[e.j] = true;
        } else {
            res.rejected[1] += 1;
        }
    }
    for v in 0..n {
        if has_edge[v] && prior[v].is_some() && rotations[v].is_some() {
            res.centers[v] = Some(Point3::from(sol[v]));
        }
    }
    res
}

/// 추정 중심을 정답에 닮음 변환(축척·회전·이동)으로 맞춘 뒤 정점별 거리 오차를 돌려준다.
/// 둘 중 하나라도 None 인 정점은 건너뛴다.
pub fn similarity_aligned_errors(
    estimated: &[Option<Point3<f64>>],
    truth: &[Point3<f64>],
) -> Vec<f64> {
    let pairs: Vec<(Vector3<f64>, Vector3<f64>)> = estimated
        .iter()
        .zip(truth)
        .filter_map(|(e, t)| e.map(|e| (e.coords, t.coords)))
        .collect();
    if pairs.len() < 3 {
        return Vec::new();
    }
    let k = pairs.len() as f64;
    let me = pairs.iter().map(|p| p.0).sum::<Vector3<f64>>() / k;
    let mt = pairs.iter().map(|p| p.1).sum::<Vector3<f64>>() / k;
    let mut cov = Matrix3::zeros();
    let mut var_e = 0.0;
    for (e, t) in &pairs {
        cov += (t - mt) * (e - me).transpose();
        var_e += (e - me).norm_squared();
    }
    let svd = cov.svd(true, true);
    let (Some(u), Some(vt)) = (svd.u, svd.v_t) else {
        return Vec::new();
    };
    let mut d = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = u * d * vt;
    let scale = (svd.singular_values.component_mul(&d.diagonal())).sum() / var_e.max(1e-300);
    pairs
        .iter()
        .map(|(e, t)| (scale * r * (e - me) + mt - t).norm())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Pose;
    use crate::math::Vector2;
    use crate::synth::{Scene, SceneConfig};
    use crate::triangulation::{triangulate_tracks, TriangulationConfig};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.unit().max(1e-300);
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
        }
        fn vec3(&mut self) -> Vector3<f64> {
            Vector3::new(self.gauss(), self.gauss(), self.gauss())
        }
    }

    /// 실측 편대 배치: 드론 3대(가로 −10/0/+10 m), 카메라 F 0°·R +125°·L −116°, 내려다보는 각 60°,
    /// 진행 방향 +x 로 위치 간 1 m, 80곳, 고도 30 m. 경로는 σ 5 cm 로 흔든다. 번호 = 위치·3 + 카메라.
    fn formation(rng: &mut Rng) -> Vec<Pose> {
        let mut poses = Vec::new();
        for p in 0..80 {
            for (k, yaw) in [0f64, 125.0, -116.0].iter().enumerate() {
                let c = Point3::new(p as f64, -10.0 + 10.0 * k as f64, 30.0) + rng.vec3() * 0.05;
                let (yaw, dip) = (yaw.to_radians(), 60f64.to_radians());
                let f = Vector3::new(dip.cos() * yaw.cos(), dip.cos() * yaw.sin(), -dip.sin());
                let right = f.cross(&Vector3::z()).normalize();
                let down = f.cross(&right);
                let r = Matrix3::from_rows(&[right.transpose(), down.transpose(), f.transpose()]);
                poses.push(Pose::from_center(Rotation3::from_matrix_unchecked(r), &c));
            }
        }
        poses
    }

    /// 정규화 좌표 시야 반폭: 수평 화각 65°, 16:9 (tan 32.5° = 0.637, 세로 0.637·9/16).
    const HALF_FOV: (f64, f64) = (0.637, 0.358);

    /// 바닥(z = 0)에 비친 시야 사각형(볼록, 반시계 순서 xy).
    fn footprint(p: &Pose) -> Vec<Vector2<f64>> {
        let c = p.center();
        let (hx, hy) = HALF_FOV;
        let mut poly: Vec<Vector2<f64>> = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
            .iter()
            .map(|&(x, y)| {
                let r = p.rotation.inverse() * Vector3::new(x, y, 1.0);
                let g = c.coords + r * (-c.z / r.z);
                Vector2::new(g.x, g.y)
            })
            .collect();
        if signed_area(&poly) < 0.0 {
            poly.reverse();
        }
        poly
    }

    fn signed_area(poly: &[Vector2<f64>]) -> f64 {
        (0..poly.len())
            .map(|k| {
                let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
                a.x * b.y - a.y * b.x
            })
            .sum::<f64>()
            * 0.5
    }

    /// 두 볼록 다각형의 교차 넓이(Sutherland–Hodgman 자르기).
    fn overlap_area(a: &[Vector2<f64>], b: &[Vector2<f64>]) -> f64 {
        let mut out = a.to_vec();
        for k in 0..b.len() {
            let (p, q) = (b[k], b[(k + 1) % b.len()]);
            let side = |v: &Vector2<f64>| (q - p).perp(&(v - p));
            let input = std::mem::take(&mut out);
            for m in 0..input.len() {
                let (u, v) = (input[m], input[(m + 1) % input.len()]);
                let (su, sv) = (side(&u), side(&v));
                if su >= 0.0 {
                    out.push(u);
                }
                if (su >= 0.0) != (sv >= 0.0) {
                    out.push(u + (v - u) * (su / (su - sv)));
                }
            }
            if out.is_empty() {
                return 0.0;
            }
        }
        signed_area(&out).abs()
    }

    /// 다른 카메라끼리 짝을 맺는 바닥 시야 겹침 비율 문턱(작은 쪽 시야 넓이 대비).
    const CROSS_OVERLAP_FRAC: f64 = 0.1;

    /// 짝: 같은 카메라 시간 이웃(1..5 칸과 8·16 칸, SPEC §3.2) + 바닥 시야가 작은 쪽 넓이의
    /// CROSS_OVERLAP_FRAC 이상 겹치는 다른 카메라.
    fn pairs(poses: &[Pose]) -> Vec<(usize, usize)> {
        let fp: Vec<_> = poses.iter().map(footprint).collect();
        let area: Vec<f64> = fp.iter().map(|f| signed_area(f)).collect();
        let mut out = Vec::new();
        for a in 0..poses.len() {
            for b in a + 1..poses.len() {
                let gap = b / 3 - a / 3;
                let same = a % 3 == b % 3 && (gap <= 5 || gap == 8 || gap == 16);
                let cross = a % 3 != b % 3
                    && overlap_area(&fp[a], &fp[b]) >= CROSS_OVERLAP_FRAC * area[a].min(area[b]);
                if same || cross {
                    out.push((a, b));
                }
            }
        }
        out
    }

    /// F-197 실측 겹침 일정: 같은 카메라 시간 이웃(SPEC §3.2) + F(p)–R(p+12..=p+40) + F(p)–L(p+16..=p+40).
    /// 번호 = 위치·3 + 카메라(F 0, R 1, L 2).
    fn schedule_pairs(n: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                let gap = b / 3 - a / 3;
                if a % 3 == b % 3 && (gap <= 5 || gap == 8 || gap == 16) {
                    out.push((a, b));
                }
            }
        }
        for p in 0..n / 3 {
            for q in p + 12..=(p + 40).min(n / 3 - 1) {
                out.push((3 * p, 3 * q + 1));
            }
            for q in p + 16..=(p + 40).min(n / 3 - 1) {
                out.push((3 * p, 3 * q + 2));
            }
        }
        out
    }

    /// 바닥 점 `count` 개와 그 점을 시야 안에 둔 카메라의 방향 관측(정점당 최대 `per_point` 개,
    /// 정규화 좌표 잡음 σ 1e-3, `outlier_frac` 은 시야 안 아무 방향).
    fn point_observations(
        seed: u64,
        poses: &[Pose],
        count: usize,
        per_point: usize,
        outlier_frac: f64,
    ) -> (Vec<Point3<f64>>, Vec<PointObservation>) {
        let mut rng = Rng(seed ^ 0x5EED);
        let mut pts = Vec::new();
        let mut obs = Vec::new();
        while pts.len() < count {
            let x = Point3::new(
                rng.unit() * 150.0 - 30.0,
                rng.unit() * 90.0 - 45.0,
                rng.unit() * 2.0,
            );
            let seen: Vec<(usize, Vector2<f64>)> = poses
                .iter()
                .enumerate()
                .filter_map(|(k, p)| {
                    let xc = p.transform(&x);
                    let n = Vector2::new(xc.x / xc.z, xc.y / xc.z);
                    (xc.z > 0.0 && n.x.abs() < HALF_FOV.0 && n.y.abs() < HALF_FOV.1)
                        .then_some((k, n))
                })
                .collect();
            if seen.len() < 3 {
                continue;
            }
            let id = pts.len();
            pts.push(x);
            // 트랙 전체에서 고르게 per_point 개를 뽑는다.
            let step = seen.len().div_ceil(per_point);
            for &(cam, n) in seen.iter().step_by(step) {
                let n = if rng.unit() < outlier_frac {
                    Vector2::new(
                        (rng.unit() * 2.0 - 1.0) * HALF_FOV.0,
                        (rng.unit() * 2.0 - 1.0) * HALF_FOV.1,
                    )
                } else {
                    n + Vector2::new(rng.gauss(), rng.gauss()) * 1e-3
                };
                obs.push(PointObservation {
                    camera: cam,
                    point: id,
                    bearing: Vector3::new(n.x, n.y, 1.0),
                    weight: 1.0,
                });
            }
        }
        (pts, obs)
    }

    struct Case {
        noise_deg: f64,
        outlier_frac: f64,
        unobservable_frac: f64,
    }

    /// 관측을 만든다. 반환: (정답 자세, 전역 회전(0.1° 잡음), 관측).
    fn observations(
        seed: u64,
        case: &Case,
    ) -> (
        Vec<Pose>,
        Vec<Option<Rotation3<f64>>>,
        Vec<RelativeTranslation>,
    ) {
        let mut rng = Rng(seed);
        let poses = formation(&mut rng);
        let rots: Vec<_> = poses
            .iter()
            .map(|p| Some(Rotation3::new(rng.vec3() * 0.1f64.to_radians()) * p.rotation))
            .collect();
        let mut obs = Vec::new();
        for (i, j) in pairs(&poses) {
            if rng.unit() < case.unobservable_frac {
                continue; // 관측 불가 짝(`from_pose` 가 None)
            }
            let (pi, pj) = (&poses[i], &poses[j]);
            let truth = (pj.rotation * (pi.center() - pj.center())).normalize();
            let dir = if rng.unit() < case.outlier_frac {
                rng.vec3().normalize()
            } else {
                Rotation3::new(rng.vec3() * case.noise_deg.to_radians()) * truth
            };
            obs.push(RelativeTranslation {
                i,
                j,
                direction: dir,
                rotation: Some(pj.rotation * pi.rotation.inverse()),
                weight: 1.0,
            });
        }
        (poses, rots, obs)
    }

    /// 실측 배치(`SceneConfig::default()`)의 정답 자세와 GPS(σ `gps_sigma`, 시드별). 번호 = 위치·3 + 카메라.
    fn real_formation(seed: u64, gps_sigma: f64) -> (Vec<Pose>, Vec<Point3<f64>>) {
        let cfg = SceneConfig {
            seed,
            gps_sigma,
            ..SceneConfig::default()
        };
        let d = SceneConfig::default();
        // 편대 오프셋·방위·내려다보는 각이 실측 값 그대로임을 확인한다(F-213).
        assert_eq!(cfg.heading_deg, [-3.0, 125.0, -116.0]);
        assert_eq!(cfg.tilt_deg, [60.0; 3]);
        assert_eq!(cfg.offsets, d.offsets);
        assert_eq!(cfg.spacing, 1.0);
        let scene = Scene::new(cfg);
        (
            scene.views.iter().map(|v| v.camera.pose).collect(),
            scene.gps_enu,
        )
    }

    /// 실측 배치의 관측: 짝 일정은 `pairs`(같은 카메라 SPEC §3.2 + 시야 겹침 다른 카메라).
    /// (정답 자세, GPS, 전역 회전, 짝 관측).
    type RealObs = (
        Vec<Pose>,
        Vec<Point3<f64>>,
        Vec<Option<Rotation3<f64>>>,
        Vec<RelativeTranslation>,
    );

    fn real_observations(seed: u64, case: &Case, gps_sigma: f64) -> RealObs {
        let (poses, gps) = real_formation(seed, gps_sigma);
        let mut rng = Rng(seed);
        let rots: Vec<_> = poses
            .iter()
            .map(|p| Some(Rotation3::new(rng.vec3() * 0.1f64.to_radians()) * p.rotation))
            .collect();
        let mut obs = Vec::new();
        let sched = if std::env::var("DIAG_OLDPAIRS").is_ok() {
            pairs(&poses)
        } else {
            schedule_pairs(poses.len())
        };
        for (i, j) in sched {
            if rng.unit() < case.unobservable_frac {
                continue;
            }
            let (pi, pj) = (&poses[i], &poses[j]);
            let truth = (pj.rotation * (pi.center() - pj.center())).normalize();
            let dir = if rng.unit() < case.outlier_frac {
                rng.vec3().normalize()
            } else {
                Rotation3::new(rng.vec3() * case.noise_deg.to_radians()) * truth
            };
            obs.push(RelativeTranslation {
                i,
                j,
                direction: dir,
                rotation: Some(pj.rotation * pi.rotation.inverse()),
                weight: 1.0,
            });
        }
        (poses, gps, rots, obs)
    }

    /// 진단: 실측 배치에서 방법별 등록 수·RMS·최대(짝 이상치 10·20%, 시드 1~5).
    #[test]
    #[ignore = "진단 출력용"]
    fn diag_real_formation() {
        let seeds: Vec<u64> = std::env::var("DIAG_SEEDS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| (1..=5).collect());
        let gsig: f64 = std::env::var("DIAG_GPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.5);
        let fracs: Vec<f64> = std::env::var("DIAG_FRACS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![-1.0, 0.0, 0.10, 0.20]);
        for frac in fracs {
            for &seed in &seeds {
                let case = Case {
                    noise_deg: if frac < 0.0 { 0.0 } else { 1.0 },
                    outlier_frac: frac.max(0.0),
                    unobservable_frac: 0.05,
                };
                let (poses, gps, rots, obs) = real_observations(seed, &case, gsig);
                let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
                let cfg = TranslationConfig::default();
                let old = std::env::var("DIAG_OLD").is_ok();
                let (ra, ma, rb, mb, na, nb) = if old {
                    let a = average_translations(&rots, &obs, &cfg);
                    let (ra, ma) = stats(&similarity_aligned_errors(&a.centers, &truth));
                    let (_, pobs) = point_observations(seed, &poses, POINTS.0, POINTS.1, POINTS.2);
                    let b = average_translations_with_points(&rots, &obs, &pobs, &cfg);
                    let (rb, mb) = stats(&similarity_aligned_errors(&b.centers, &truth));
                    (ra, ma, rb, mb, a.registered(), b.registered())
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0, 0)
                };
                let pri: Vec<_> = gps
                    .iter()
                    .enumerate()
                    .map(|(k, g)| PositionPrior {
                        camera: k,
                        position: *g,
                        sigma: gsig,
                    })
                    .collect();
                let c = average_translations_with_prior(
                    &rots,
                    &obs,
                    &pri,
                    &cfg,
                    &PriorConfig::default(),
                );
                let (rc, mc) = stats(&similarity_aligned_errors(&c.centers, &truth));
                let raw: Vec<f64> = c
                    .centers
                    .iter()
                    .zip(&truth)
                    .filter_map(|(e, t)| e.map(|e| (e - t).norm()))
                    .collect();
                println!(
                    "RAW rms {:.3} inl {}",
                    stats(&raw).0,
                    c.inliers.iter().filter(|b| **b).count()
                );
                let gps_opt: Vec<_> = gps.iter().map(|g| Some(*g)).collect();
                let (rg, _) = stats(&similarity_aligned_errors(&gps_opt, &truth));
                println!(
                    "REAL frac {frac} seed {seed} pairs {} | pairs-only reg {} rms {ra:.2} max {ma:.2} | +points reg {} rms {rb:.2} max {mb:.2} | +gps reg {} rms {rc:.3} max {mc:.3} (gps alone {rg:.2})",
                    obs.len(), na, nb, c.registered()
                );
            }
        }
    }

    fn stats(e: &[f64]) -> (f64, f64) {
        let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
        (rms, e.iter().cloned().fold(0.0, f64::max))
    }

    /// 시험 점 수·점당 관측 수·점 관측 이상치 비율.
    const POINTS: (usize, usize, f64) = (200, 16, 0.05);

    fn run_with(seed: u64, case: &Case, point_outliers: f64) -> (usize, f64, f64) {
        // 실측 배치(`SceneConfig::default()`)·실측 짝 일정. 배치가 실측 값임은 `real_formation` 이 단언한다.
        let (poses, _gps, rots, obs) = real_observations(seed, case, 1.5);
        let (_, pobs) = point_observations(seed, &poses, POINTS.0, POINTS.1, point_outliers);
        let res = average_translations_robust(
            &rots,
            &obs,
            &pobs,
            &TranslationConfig::default(),
            seed,
        );
        let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
        let (rms, max) = stats(&similarity_aligned_errors(&res.centers, &truth));
        (res.registered(), rms, max)
    }

    /// 실측 배치에서 점 제약을 더한 풀이. 반환: (등록 수, RMS, 최대, 짝 그래프 최대 성분 크기, 1단계 등록 수·RMS).
    fn real_run_with(
        seed: u64,
        case: &Case,
        point_outliers: f64,
    ) -> (usize, f64, f64, usize, usize, f64) {
        let (poses, _gps, rots, obs) = real_observations(seed, case, 1.5);
        let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
        let (_, pobs) = point_observations(seed, &poses, POINTS.0, POINTS.1, point_outliers);
        let cfg = TranslationConfig::default();
        let edges: Vec<(usize, usize)> = obs.iter().map(|o| (o.i, o.j)).collect();
        let (comp, best) = components(poses.len(), &edges);
        let comp_size = comp.iter().filter(|c| **c == best && c.is_some()).count();
        let stage = average_core(&rots, &obs, &[], &cfg, None);
        let (rs, _) = stats(&similarity_aligned_errors(&stage.centers, &truth));
        let res = average_translations_with_points(&rots, &obs, &pobs, &cfg);
        let (rms, max) = stats(&similarity_aligned_errors(&res.centers, &truth));
        (
            res.registered(),
            rms,
            max,
            comp_size,
            stage.registered(),
            rs,
        )
    }

    /// 진단: 실측 배치 × 짝 이상치 × 점 이상치 × 시드 표. 환경 변수 DIAG_SEEDS(기본 1~20),
    /// DIAG_FRACS(기본 0.10,0.20), DIAG_PFRACS(기본 0,0.05). 기준 미달 경우는 FAIL 로 표시한다.
    #[test]
    #[ignore = "진단 출력용"]
    fn diag_real_points_table() {
        let list = |name: &str, def: Vec<f64>| -> Vec<f64> {
            std::env::var(name)
                .ok()
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or(def)
        };
        let seeds = list("DIAG_SEEDS", (1..=20).map(|s| s as f64).collect());
        let fracs = list("DIAG_FRACS", vec![0.10, 0.20]);
        let pfracs = list("DIAG_PFRACS", vec![0.0, 0.05]);
        let mut bad = 0;
        let mut total = 0;
        for &pf in &pfracs {
            for &frac in &fracs {
                for &seed in &seeds {
                    let case = Case {
                        noise_deg: 1.0,
                        outlier_frac: frac,
                        unobservable_frac: 0.05,
                    };
                    let seed = seed as u64;
                    let (reg, rms, max, comp, sreg, srms) = real_run_with(seed, &case, pf);
                    let fail = reg < 238 || rms > 0.3;
                    total += 1;
                    bad += fail as usize;
                    println!(
                        "REALPTS point {pf} pair {frac} seed {seed}: reg {reg} rms {rms:.3} max {max:.3} | comp {comp} stage reg {sreg} rms {srms:.3} {}",
                        if fail { "FAIL" } else { "" }
                    );
                }
            }
        }
        println!("REALPTS summary: {bad} fail of {total}");
    }

    /// 진단: 점 제약 유무별 등록 수·RMS (잡음 1°, 이상치 10·20%, 시드 1~5).
    #[test]
    #[ignore = "진단 출력용"]
    fn diag_point_constraints() {
        let seeds: Vec<u64> = std::env::var("DIAG_SEEDS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| (1..=5).collect());
        for frac in [0.10, 0.20] {
            for &seed in &seeds {
                let case = Case {
                    noise_deg: 1.0,
                    outlier_frac: frac,
                    unobservable_frac: 0.05,
                };
                let (poses, rots, obs) = observations(seed, &case);
                let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
                let cfg = TranslationConfig::default();
                let a = average_translations(&rots, &obs, &cfg);
                let (ra, _) = stats(&similarity_aligned_errors(&a.centers, &truth));
                let (_, pobs) = point_observations(seed, &poses, POINTS.0, POINTS.1, POINTS.2);
                let b = average_translations_with_points(&rots, &obs, &pobs, &cfg);
                let (rb, _) = stats(&similarity_aligned_errors(&b.centers, &truth));
                let (_, pobs0) = point_observations(seed, &poses, POINTS.0, POINTS.1, 0.0);
                let c = average_translations_with_points(&rots, &obs, &pobs0, &cfg);
                let (rc, _) = stats(&similarity_aligned_errors(&c.centers, &truth));
                println!(
                    "DIAG frac {frac} seed {seed} pairs {} rej {:?}/{:?}/{:?} | nopts reg {} rms {ra:.3} | pts reg {} rms {rb:.3} | pts-clean reg {} rms {rc:.3}",
                    obs.len(), a.rejected, b.rejected, c.rejected,
                    a.registered(), b.registered(), c.registered()
                );
            }
        }
    }

    /// 진단: 실패하던 세 경우만 단계별로 본다.
    #[test]
    #[ignore = "진단 출력용"]
    fn diag_three_cases() {
        for (pf, frac, seed) in [(0.0, 0.2, 10u64), (0.05, 0.2, 6), (0.05, 0.2, 9)] {
            let case = Case {
                noise_deg: 1.0,
                outlier_frac: frac,
                unobservable_frac: 0.05,
            };
            let (poses, rots, obs) = observations(seed, &case);
            let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
            let (_, pobs) = point_observations(seed, &poses, POINTS.0, POINTS.1, pf);
            let cfg = TranslationConfig::default();
            let a = average_core(&rots, &obs, &[], &cfg, None);
            let (ra, ma) = stats(&similarity_aligned_errors(&a.centers, &truth));
            let b = average_translations_with_points(&rots, &obs, &pobs, &cfg);
            let (rb, mb) = stats(&similarity_aligned_errors(&b.centers, &truth));
            println!(
                "DIAG3 pf {pf} seed {seed}: stage reg {} rms {ra:.3} max {ma:.3} score {:?} | final reg {} rms {rb:.3} max {mb:.3} score {:?}",
                a.registered(), pair_score(&a, obs.len()), b.registered(), pair_score(&b, obs.len())
            );
        }
    }

    #[test]
    fn prior_real_formation_noiseless_is_better_than_gps() {
        // 실측 배치·방향 잡음 0·이상치 0·GPS σ 1.5 m: 사전 결합 풀이는 GPS 단독(RMS ~2.5 m)보다 낫다(측정 0.74~0.76 m).
        // 잡음·이상치가 있는 경우는 아직 목표(RMS ≤ 0.5 m)에 못 미친다(노트 참조).
        for seed in [1u64, 2] {
            let case = Case {
                noise_deg: 0.0,
                outlier_frac: 0.0,
                unobservable_frac: 0.05,
            };
            let (poses, gps, rots, obs) = real_observations(seed, &case, 1.5);
            let pri: Vec<_> = gps
                .iter()
                .enumerate()
                .map(|(k, g)| PositionPrior {
                    camera: k,
                    position: *g,
                    sigma: 1.5,
                })
                .collect();
            let res = average_translations_with_prior(
                &rots,
                &obs,
                &pri,
                &TranslationConfig::default(),
                &PriorConfig::default(),
            );
            let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
            let raw: Vec<f64> = res
                .centers
                .iter()
                .zip(&truth)
                .filter_map(|(e, t)| e.map(|e| (e - t).norm()))
                .collect();
            let (rms, max) = stats(&raw);
            assert_eq!(res.registered(), 240);
            assert!(rms < 1.0 && max < 5.0, "rms {rms} max {max}");
        }
    }

    #[test]
    fn real_schedule_cross_pairs_improve_pairs_only() {
        // F-197 일정(F(p)–R(p+12..+40), F(p)–L(p+16..+40))의 짝만 쓴 평균: 옛 일정의 RMS 9~11 m 에서
        // 실측 3.2~3.9 m(최대 19~24 m)로 줄지만 목표(0.3 m)에는 못 미친다(노트 참조).
        let sp = schedule_pairs(240);
        assert!(sp.iter().all(|&(a, b)| a < b));
        assert!(sp.contains(&(0, 3 * 12 + 1)) && sp.contains(&(0, 3 * 16 + 2)));
        assert!(!sp.contains(&(0, 3 * 11 + 1)) && !sp.contains(&(0, 3 * 15 + 2)));
        let case = Case {
            noise_deg: 1.0,
            outlier_frac: 0.10,
            unobservable_frac: 0.05,
        };
        let (poses, _, rots, obs) = real_observations(1, &case, 1.5);
        let res = average_translations(&rots, &obs, &TranslationConfig::default());
        let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
        let (rms, _) = stats(&similarity_aligned_errors(&res.centers, &truth));
        assert!(res.registered() >= 225, "registered {}", res.registered());
        assert!(rms < 6.0, "rms {rms}");
    }

    #[test]
    fn robust_selection_registers_all_seeds_and_outlier_rates() {
        // 옛 배치, 방향 잡음 1°, 관측 불가 5%, 점 200개(이상치 5%). 시드 1~10 × 짝 이상치 0·10·20%.
        // 기준: 등록 ≥ 99%(238/240), 닮음 정렬 후 중심 RMS ≤ 1.0 m. 기존 경로(점 포함)와 함께 표로 낸다.
        // 기본은 F-214 실패 시드 6·9·10 × 20%. 전체 표는 ROBUST_SEEDS=1,2,..,10 ROBUST_FRACS=0,0.1,0.2 (부하 없이도 수 분).
        let list = |k: &str, d: Vec<f64>| -> Vec<f64> {
            std::env::var(k)
                .ok()
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or(d)
        };
        let seeds = list("ROBUST_SEEDS", vec![6.0, 9.0, 10.0]);
        let fracs = list("ROBUST_FRACS", vec![0.20]);
        let mut fails = Vec::new();
        for &frac in &fracs {
            for &seed in &seeds {
                let seed = seed as u64;
                let case = Case {
                    noise_deg: 1.0,
                    outlier_frac: frac,
                    unobservable_frac: 0.05,
                };
                let (poses, rots, obs) = observations(seed, &case);
                let (_, pobs) = point_observations(seed, &poses, POINTS.0, POINTS.1, POINTS.2);
                let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
                let cfg = TranslationConfig::default();
                let old = average_translations_with_points(&rots, &obs, &pobs, &cfg);
                let (ro, _) = stats(&similarity_aligned_errors(&old.centers, &truth));
                let new = average_translations_robust(&rots, &obs, &pobs, &cfg, seed);
                let (rn, mx) = stats(&similarity_aligned_errors(&new.centers, &truth));
                println!(
                    "ROBUST pair {frac} seed {seed}: old reg {} rms {ro:.3} | robust reg {} rms {rn:.3} max {mx:.3}",
                    old.registered(), new.registered()
                );
                if new.registered() < 238 || rn > 1.0 {
                    fails.push((frac, seed, new.registered(), rn));
                }
            }
        }
        assert!(fails.is_empty(), "{fails:?}");
    }

    #[test]
    fn noiseless_exact() {
        let case = Case {
            noise_deg: 0.0,
            outlier_frac: 0.0,
            unobservable_frac: 0.0,
        };
        let (poses, _, obs) = observations(1, &case);
        let rots: Vec<_> = poses.iter().map(|p| Some(p.rotation)).collect();
        let res = average_translations(&rots, &obs, &TranslationConfig::default());
        let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
        let (rms, max) = stats(&similarity_aligned_errors(&res.centers, &truth));
        assert_eq!(res.registered(), 240);
        // 무잡음·정확한 회전이면 해는 닮음 변환을 빼고 유일하다. 장면 크기 ~80 m 에 대해 1e-6 m
        // 은 상대 ~1e-8 로, 배정밀도 선형 풀이 오차에 여유를 둔 값이다.
        assert!(max < 1e-6, "rms {rms} max {max}");
    }

    #[test]
    fn noisy_outliers_register_all_seeds() {
        // 짝 방향 잡음 1°·짝 이상치 10/20% × 점 방향 이상치 0/5% × 시드 1~20(문턱 조정에 쓴 1~5 밖 15개 포함).
        // 기준(F-214): 모든 경우 등록 ≥ 238/240, 닮음 정렬 후 중심 RMS ≤ 0.3 m. 실패를 모두 모아 한 번에 보인다.
        let mut fails = Vec::new();
        for pfrac in [0.0, 0.05] {
            for frac in [0.10, 0.20] {
                for seed in 1..=20u64 {
                    let case = Case {
                        noise_deg: 1.0,
                        outlier_frac: frac,
                        unobservable_frac: 0.05,
                    };
                    let (reg, rms, max) = run_with(seed, &case, pfrac);
                    println!(
                        "point {pfrac} pair {frac} seed {seed}: reg {reg} rms {rms:.4} m max {max:.4} m"
                    );
                    if reg < 238 || rms > 0.3 || max > 1.0 {
                        fails.push((pfrac, frac, seed, reg, rms, max));
                    }
                }
            }
        }
        assert!(fails.is_empty(), "{fails:?}");
    }

    #[test]
    fn nan_rotation_and_infinite_weight_are_isolated() {
        let case = Case {
            noise_deg: 0.0,
            outlier_frac: 0.0,
            unobservable_frac: 0.0,
        };
        let (poses, _, obs) = observations(1, &case);
        let cfg = TranslationConfig::default();
        let rots: Vec<_> = poses.iter().map(|p| Some(p.rotation)).collect();
        let base = average_translations(&rots, &obs, &cfg);
        assert_eq!(base.registered(), 240);
        let mut bad = rots.clone();
        bad[5] = Some(Rotation3::from_matrix_unchecked(Matrix3::from_element(
            f64::NAN,
        )));
        let res = average_translations(&bad, &obs, &cfg);
        assert!(res.registered() >= 239, "{}", res.registered());
        assert!(res.rejected[0] > base.rejected[0]);
        let mut inf = obs.clone();
        inf[0].weight = f64::INFINITY;
        let res = average_translations(&rots, &inf, &cfg);
        assert!(res.registered() >= 239, "{}", res.registered());
        assert_eq!(res.rejected[0], base.rejected[0] + 1);
        // 점 관측의 무한 가중치도 같은 방식으로 버린다.
        let (_, mut pobs) = point_observations(1, &poses, POINTS.0, POINTS.1, 0.0);
        let pbase = average_translations_with_points(&rots, &obs, &pobs, &cfg);
        pobs[0].weight = f64::INFINITY;
        let res = average_translations_with_points(&rots, &obs, &pobs, &cfg);
        assert!(res.registered() >= 239, "{}", res.registered());
        assert!(res.rejected[0] > pbase.rejected[0]);
    }

    #[test]
    fn ray_point_sampled_hypotheses_are_fast_and_accurate() {
        // 광선 5,000개: 정상 80%(점 x 를 지나는 방향에 0.2° 잡음), 이상치 20%(무작위 방향).
        let mut rng = Rng(7);
        let x = Vector3::new(3.0, -2.0, 1.0);
        let rays: Vec<(Vector3<f64>, Vector3<f64>)> = (0..5000)
            .map(|k| {
                let o = rng.vec3() * 40.0 + Vector3::new(0.0, 0.0, 30.0);
                let d = if k % 5 == 0 {
                    rng.vec3().normalize()
                } else {
                    Rotation3::new(rng.vec3() * 0.2f64.to_radians()) * (x - o).normalize()
                };
                (o, d)
            })
            .collect();
        let t = std::time::Instant::now();
        let p = robust_ray_point(&rays, 10f64.to_radians()).unwrap();
        let dt = t.elapsed().as_secs_f64();
        // 정상 광선만으로 푼 최소제곱(전체 탐색이 찾는 정상 집합의 해)과 비교.
        let mut a = Matrix3::zeros();
        let mut b = Vector3::zeros();
        for (k, (o, d)) in rays.iter().enumerate() {
            if k % 5 != 0 {
                let m = Matrix3::identity() - d * d.transpose();
                a += m;
                b += m * o;
            }
        }
        let full = a.try_inverse().unwrap() * b;
        let (e, e_full) = ((p - x).norm(), (full - x).norm());
        println!("rays 5000: {dt:.4} s, err {e:.5} m, inlier-only LS err {e_full:.5} m");
        assert!(dt < 0.2, "{dt} s");
        assert!((p - full).norm() < 0.01 * (full - rays[1].0).norm());
    }

    #[test]
    fn rotation_inconsistent_pairs_are_dropped() {
        let case = Case {
            noise_deg: 1.0,
            outlier_frac: 0.0,
            unobservable_frac: 0.0,
        };
        let (_, rots, mut obs) = observations(3, &case);
        for o in obs.iter_mut().step_by(7) {
            o.rotation = o.rotation.map(|r| Rotation3::new(Vector3::x() * 0.3) * r);
            o.direction = -o.direction;
        }
        let res = average_translations(&rots, &obs, &TranslationConfig::default());
        assert_eq!(res.rejected[0], obs.len().div_ceil(7));
        assert_eq!(res.registered(), 240);
    }

    #[test]
    fn disconnected_keeps_largest_component() {
        let case = Case {
            noise_deg: 0.5,
            outlier_frac: 0.0,
            unobservable_frac: 0.0,
        };
        let (_, rots, obs) = observations(4, &case);
        // 위치 0..9 와 나머지를 잇는 간선을 모두 없앤다.
        let obs: Vec<_> = obs
            .into_iter()
            .filter(|o| (o.i < 30) == (o.j < 30))
            .collect();
        let res = average_translations(&rots, &obs, &TranslationConfig::default());
        assert_eq!(res.registered(), 210);
        assert!(res.centers[..30].iter().all(|c| c.is_none()));
    }

    #[test]
    fn single_line_is_not_registered() {
        // 같은 카메라 시간 이웃만: 각 드론 경로는 거의 한 직선이라 서로 이어지지 않는다.
        let case = Case {
            noise_deg: 0.0,
            outlier_frac: 0.0,
            unobservable_frac: 0.0,
        };
        let (_, rots, obs) = observations(5, &case);
        let obs: Vec<_> = obs.into_iter().filter(|o| o.i % 3 == o.j % 3).collect();
        let res = average_translations(&rots, &obs, &TranslationConfig::default());
        assert!(res.registered() <= 80, "{}", res.registered());
    }

    #[test]
    fn rough_model_from_averaged_poses() {
        let case = Case {
            noise_deg: 1.0,
            outlier_frac: 0.15,
            unobservable_frac: 0.05,
        };
        let (poses, rots, obs) = observations(7, &case);
        let (_, pobs) = point_observations(7, &poses, POINTS.0, POINTS.1, POINTS.2);
        let res =
            average_translations_with_points(&rots, &obs, &pobs, &TranslationConfig::default());
        assert_eq!(res.registered(), 240);
        // 추정 중심을 정답 틀로 닮음 정렬한 뒤, 추정 회전과 묶어 자세를 만든다.
        let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
        let est_c: Vec<Point3<f64>> = res.centers.iter().map(|c| c.unwrap()).collect();
        let aligned = align_to(&est_c, &truth);
        let est: Vec<Option<Pose>> = aligned
            .iter()
            .zip(&rots)
            .map(|(c, r)| Some(Pose::from_center(r.unwrap(), c)))
            .collect();
        // 바닥 점 400개, 시야 안의 카메라에서 정규화 좌표 + 잡음 σ 1e-3.
        let mut rng = Rng(99);
        let mut points = Vec::new();
        let mut tracks = Vec::new();
        for _ in 0..400 {
            let x = Point3::new(
                rng.unit() * 100.0 - 10.0,
                rng.unit() * 60.0 - 30.0,
                rng.unit() * 2.0,
            );
            let track: Vec<(usize, Vector2<f64>)> = poses
                .iter()
                .enumerate()
                .filter_map(|(k, p)| {
                    let xc = p.transform(&x);
                    let n = Vector2::new(xc.x / xc.z, xc.y / xc.z);
                    (xc.z > 0.0 && n.x.abs() < 0.64 && n.y.abs() < 0.48)
                        .then(|| (k, n + Vector2::new(rng.gauss(), rng.gauss()) * 1e-3))
                })
                .collect();
            if track.len() >= 3 {
                points.push(x);
                tracks.push(track);
            }
        }
        let cfg = TriangulationConfig {
            max_reprojection: 0.02,
            ..Default::default()
        };
        let out = triangulate_tracks(&est, &tracks, &cfg);
        let ok: Vec<f64> = out
            .iter()
            .zip(&points)
            .filter_map(|(r, x)| r.as_ref().ok().map(|p| (p.point - x).norm()))
            .collect();
        let (rms, _) = stats(&ok);
        println!(
            "model: {}/{} points, rms {rms:.3} m",
            ok.len(),
            points.len()
        );
        assert!(ok.len() * 10 >= points.len() * 9);
        assert!(rms < 0.3, "rms {rms}");
    }

    fn align_to(est: &[Point3<f64>], truth: &[Point3<f64>]) -> Vec<Point3<f64>> {
        // similarity_aligned_errors 와 같은 닮음 변환을 점에 적용한다.
        let k = est.len() as f64;
        let me = est.iter().map(|p| p.coords).sum::<Vector3<f64>>() / k;
        let mt = truth.iter().map(|p| p.coords).sum::<Vector3<f64>>() / k;
        let mut cov = Matrix3::zeros();
        let mut var = 0.0;
        for (e, t) in est.iter().zip(truth) {
            cov += (t.coords - mt) * (e.coords - me).transpose();
            var += (e.coords - me).norm_squared();
        }
        let svd = cov.svd(true, true);
        let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
        let mut d = Matrix3::identity();
        if (u * vt).determinant() < 0.0 {
            d[(2, 2)] = -1.0;
        }
        let r = u * d * vt;
        let s = svd.singular_values.component_mul(&d.diagonal()).sum() / var;
        est.iter()
            .map(|e| Point3::from(s * r * (e.coords - me) + mt))
            .collect()
    }
}
