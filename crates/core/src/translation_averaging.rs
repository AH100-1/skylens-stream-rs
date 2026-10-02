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
    let rays: Vec<Option<Vector3<f64>>> = point_observations
        .iter()
        .map(|o| {
            let ok = o.camera < n_cam
                && o.point < n_pts
                && o.weight > 0.0
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
    average_core(
        rotations,
        observations,
        point_observations,
        cfg,
        Some(&start),
    )
}

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
    for a in 0..rays.len() {
        for b in a + 1..rays.len() {
            if rays[a].1.cross(&rays[b].1).norm() < sin_min {
                continue;
            }
            let Some(x) = ls(&[a, b]) else { continue };
            let i = inl(&x);
            if i.len() > best.len() {
                best = i;
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
    // 1a. 무효·회전 불일치 제거, 세계 방향으로 바꾸기.
    let mut edges = Vec::new();
    for (idx, o) in observations.iter().enumerate() {
        let valid = o.i < n_cam
            && o.j < n_cam
            && o.i != o.j
            && o.weight > 0.0
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
            && o.bearing.norm() > 1e-12
            && o.bearing.iter().all(|x| x.is_finite()))
        .then(|| rotations[o.camera])
        .flatten();
        let Some(r) = rot else {
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

    fn stats(e: &[f64]) -> (f64, f64) {
        let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
        (rms, e.iter().cloned().fold(0.0, f64::max))
    }

    /// 시험 점 수·점당 관측 수·점 관측 이상치 비율.
    const POINTS: (usize, usize, f64) = (200, 16, 0.05);

    fn run(seed: u64, case: &Case) -> (usize, f64, f64) {
        let (poses, rots, obs) = observations(seed, case);
        let (_, pobs) = point_observations(seed, &poses, POINTS.0, POINTS.1, POINTS.2);
        let res =
            average_translations_with_points(&rots, &obs, &pobs, &TranslationConfig::default());
        let truth: Vec<_> = poses.iter().map(|p| p.center()).collect();
        let (rms, max) = stats(&similarity_aligned_errors(&res.centers, &truth));
        (res.registered(), rms, max)
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
        // 짝 방향 잡음 1°·이상치 10/20%, 점 방향 이상치 5%. 기준: 240/240 등록, 닮음 정렬 후 중심 RMS < 0.15 m,
        // 최대 < 0.5 m. 깨끗한 점 방향일 때 같은 시드들의 RMS 가 0.11~0.14 m 이므로 이상치를 걸러 그 수준에 들어야 한다.
        for (frac, rms_lim, max_lim) in [(0.10, 0.15, 0.5), (0.20, 0.15, 0.5)] {
            for seed in 1..=5u64 {
                let case = Case {
                    noise_deg: 1.0,
                    outlier_frac: frac,
                    unobservable_frac: 0.05,
                };
                let (reg, rms, max) = run(seed, &case);
                println!("outlier {frac} seed {seed}: reg {reg} rms {rms:.4} m max {max:.4} m");
                assert_eq!(reg, 240, "seed {seed} frac {frac}");
                assert!(
                    rms < rms_lim && max < max_lim,
                    "seed {seed}: rms {rms} max {max}"
                );
            }
        }
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
