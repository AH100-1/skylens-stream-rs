//! 희소 Levenberg–Marquardt 번들 조정.
//!
//! - 점 블록(3×3)을 슈어 보수로 소거해 카메라 쪽 축소 계통만 밀집 촐레스키로 푼다.
//! - 강건 손실(Huber/Cauchy)은 IRLS 가중으로 넣는다.
//! - 내부 파라미터 [fx, fy, cx, cy, k1, k2, p1, p2] 는 그룹(폴더)마다 공유하고
//!   항목별로 고정/자유를 고른다.
//! - 게이지: 지정한 카메라(기본: 첫 카메라)의 포즈를 고정한다(목록이 비거나 범위 밖이면 첫 카메라).
//!   고정 카메라가 하나뿐이면 남는 축척 1자유도는 두 번째 기준 카메라의 평행이동 한 성분을
//!   고정해 없앤다. 성분은 축척 방향 기울기 |R_k(C_k − C_0)|_i 가 가장 큰 것을 고른다.
//! - 입력 검증: 포즈·그룹 길이 불일치, 범위 밖 그룹 번호는 아무것도 고치지 않고
//!   `BaStop::InvalidInput` 으로 돌려준다. 범위 밖 점·카메라 번호, 유한하지 않은 픽셀의
//!   관측은 제외하고 수를 보고한다.
//! - 관측 트랙(3D 점)은 관측 2개 이상인 점 중 최대 `max_tracks` 개만 쓴다.
//!   관측 수가 많은 점부터, 같으면 번호 순.

use crate::camera::Pose;
use crate::distortion::DistortedIntrinsics;
use crate::math::{Matrix3, Point3, Rotation3, SMatrix, Vector2, Vector3};
use nalgebra::{DMatrix, DVector};

/// 관측 하나: 카메라 번호, 점 번호, 픽셀.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub camera: usize,
    pub point: usize,
    pub pixel: Vector2<f64>,
}

/// 번들 조정 입력이자 결과(제자리 갱신).
#[derive(Clone, Debug)]
pub struct BaProblem {
    /// 내부 파라미터 그룹.
    pub groups: Vec<DistortedIntrinsics>,
    /// 카메라 포즈.
    pub poses: Vec<Pose>,
    /// 카메라마다 속한 그룹 번호.
    pub camera_group: Vec<usize>,
    pub points: Vec<Point3<f64>>,
    pub observations: Vec<Observation>,
}

/// 강건 손실. 인자는 픽셀 단위 척도 δ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    Squared,
    Huber(f64),
    Cauchy(f64),
}

impl Loss {
    /// ρ(z), z = |r|².
    pub fn rho(&self, z: f64) -> f64 {
        match *self {
            Loss::Squared => z,
            Loss::Huber(d) => {
                if z <= d * d {
                    z
                } else {
                    2.0 * d * z.sqrt() - d * d
                }
            }
            Loss::Cauchy(d) => d * d * (z / (d * d)).ln_1p(),
        }
    }

    /// IRLS 가중 ρ'(z).
    pub fn weight(&self, z: f64) -> f64 {
        match *self {
            Loss::Squared => 1.0,
            Loss::Huber(d) => {
                if z <= d * d {
                    1.0
                } else {
                    d / z.sqrt()
                }
            }
            Loss::Cauchy(d) => 1.0 / (1.0 + z / (d * d)),
        }
    }
}

/// 내부 파라미터 순서: [fx, fy, cx, cy, k1, k2, p1, p2].
pub const INTRINSIC_NAMES: [&str; 8] = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"];

#[derive(Clone, Debug, PartialEq)]
pub struct BaOptions {
    /// 0 = 초벌(평가만), 1 이상 = 정밀.
    pub max_iterations: usize,
    pub loss: Loss,
    /// 그룹마다 자유 파라미터 마스크. 비어 있으면 모든 그룹에 `default_free_intrinsics`.
    pub free_intrinsics: Vec<[bool; 8]>,
    pub default_free_intrinsics: [bool; 8],
    /// 포즈를 고정할 카메라(게이지).
    pub fixed_cameras: Vec<usize>,
    /// 관측 트랙(점) 최대 수.
    pub max_tracks: usize,
    pub initial_lambda: f64,
    /// 상대 비용 감소가 이보다 작으면 멈춘다.
    pub function_tolerance: f64,
}

impl Default for BaOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            loss: Loss::Huber(2.0),
            free_intrinsics: Vec::new(),
            default_free_intrinsics: [true; 8],
            fixed_cameras: vec![0],
            max_tracks: 100_000,
            initial_lambda: 1e-4,
            function_tolerance: 1e-10,
        }
    }
}

/// 번들 조정이 멈춘 이유.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaStop {
    /// `max_iterations == 0`: 평가만 했다.
    EvaluationOnly,
    /// 상대 비용 감소가 `function_tolerance` 아래로 내려갔다.
    Converged,
    /// 최대 반복 수에 닿았다.
    MaxIterations,
    /// 감쇠를 키워도 비용을 줄이는 단계를 찾지 못했다(촐레스키 실패 연속·비유한 비용 포함).
    StepFailed,
    /// 입력 크기·그룹 번호가 맞지 않아 아무것도 하지 않았다.
    InvalidInput,
}

/// 번들 조정 보고.
#[derive(Clone, Debug, PartialEq)]
pub struct BaReport {
    /// 0 회면 초벌, 1 회 이상이면 정밀.
    pub iterations: usize,
    pub refined: bool,
    pub num_cameras: usize,
    pub num_tracks_used: usize,
    pub num_observations_used: usize,
    /// 재투영 오차 RMS(픽셀 거리, sqrt(Σ|r|²/N)), 가중 없음.
    pub initial_rms: f64,
    pub final_rms: f64,
    pub initial_cost: f64,
    pub final_cost: f64,
    /// `stop == BaStop::Converged` 일 때만 참.
    pub converged: bool,
    pub stop: BaStop,
    /// 범위 밖 번호·비유한 픽셀로 제외한 관측 수.
    pub num_observations_rejected: usize,
    /// 축소 계통 촐레스키 실패 횟수(λ 재시도 포함).
    pub cholesky_failures: usize,
}

/// 결정적 트랙 선택: 관측 2개 이상인 점만, 관측 수 내림차순, 같으면 점 번호 오름차순.
/// 선택된 점 번호(오름차순). 범위 밖 점 번호 관측은 센하지 않는다.
pub fn select_tracks(
    num_points: usize,
    observations: &[Observation],
    max_tracks: usize,
) -> Vec<usize> {
    let mut count = vec![0usize; num_points];
    for o in observations {
        if let Some(c) = count.get_mut(o.point) {
            *c += 1;
        }
    }
    let mut ids: Vec<usize> = (0..num_points).filter(|&p| count[p] >= 2).collect();
    if ids.len() > max_tracks {
        ids.sort_by(|&a, &b| count[b].cmp(&count[a]).then(a.cmp(&b)));
        ids.truncate(max_tracks);
        ids.sort_unstable();
    }
    ids
}

/// 회전 왼쪽 섭동 R ← exp([ω]×)·R.
fn apply_pose(pose: &Pose, d: &[f64]) -> Pose {
    let w = Vector3::new(d[0], d[1], d[2]);
    Pose::new(
        Rotation3::new(w) * pose.rotation,
        pose.translation + Vector3::new(d[3], d[4], d[5]),
    )
}

fn intr_get(k: &DistortedIntrinsics, i: usize) -> f64 {
    match i {
        0 => k.fx,
        1 => k.fy,
        2 => k.cx,
        3 => k.cy,
        4 => k.dist.k1,
        5 => k.dist.k2,
        6 => k.dist.p1,
        _ => k.dist.p2,
    }
}

fn intr_set(k: &mut DistortedIntrinsics, i: usize, v: f64) {
    match i {
        0 => k.fx = v,
        1 => k.fy = v,
        2 => k.cx = v,
        3 => k.cy = v,
        4 => k.dist.k1 = v,
        5 => k.dist.k2 = v,
        6 => k.dist.p1 = v,
        _ => k.dist.p2 = v,
    }
}

/// 관측 하나의 잔차(투영 − 관측)와 야코비안. 카메라 뒤면 None.
/// 열 순서: 포즈 6 (ω, t) | 내부 8 | 점 3.
pub fn residual_jacobian(
    intr: &DistortedIntrinsics,
    pose: &Pose,
    x: &Point3<f64>,
    obs: &Vector2<f64>,
) -> Option<(Vector2<f64>, SMatrix<f64, 2, 17>)> {
    let pj = intr.project_with_jacobian(pose, x)?;
    let mut j = SMatrix::<f64, 2, 17>::zeros();
    j.fixed_view_mut::<2, 6>(0, 0).copy_from(&pj.d_pose);
    j.fixed_view_mut::<2, 8>(0, 6).copy_from(&pj.d_intrinsics);
    j.fixed_view_mut::<2, 3>(0, 14).copy_from(&pj.d_point);
    Some((pj.pixel - obs, j))
}

struct Layout {
    /// 카메라별 포즈 6성분의 축소 계통 번호(고정이면 None).
    cam_idx: Vec<[Option<usize>; 6]>,
    /// 그룹별 (내부 파라미터 번호, 축소 계통 번호).
    intr_idx: Vec<Vec<(usize, usize)>>,
    n: usize,
}

/// 고정 카메라 목록(범위 밖 제거, 비면 첫 카메라)과 축척 고정 (카메라, 평행이동 성분).
fn gauge(problem: &BaProblem, opts: &BaOptions) -> (Vec<usize>, Option<(usize, usize)>) {
    let n_cam = problem.poses.len();
    let mut fixed: Vec<usize> = opts
        .fixed_cameras
        .iter()
        .copied()
        .filter(|&c| c < n_cam)
        .collect();
    fixed.sort_unstable();
    fixed.dedup();
    if fixed.is_empty() && n_cam > 0 {
        fixed.push(0);
    }
    if fixed.len() != 1 {
        return (fixed, None);
    }
    // 고정 카메라 C_0 를 중심으로 한 축척 s 에서 t_k = −R_k(C_0 + s(C_k − C_0)) 이므로
    // ∂t_k/∂s = −R_k(C_k − C_0). 이 기울기 성분이 가장 큰 (k, i) 를 고정한다.
    let c0 = problem.poses[fixed[0]].center();
    let mut best: Option<(usize, usize, f64)> = None;
    for k in 0..n_cam {
        if k == fixed[0] {
            continue;
        }
        let pose = &problem.poses[k];
        let d = pose.rotation * (pose.center() - c0);
        for i in 0..3 {
            let v = d[i].abs();
            if v.is_finite() && v > 1e-9 && best.is_none_or(|b| v > b.2) {
                best = Some((k, i, v));
            }
        }
    }
    (fixed, best.map(|b| (b.0, b.1)))
}

fn layout(problem: &BaProblem, opts: &BaOptions) -> Layout {
    let (fixed, scale_fix) = gauge(problem, opts);
    let mut n = 0;
    let mut cam_idx = Vec::with_capacity(problem.poses.len());
    for c in 0..problem.poses.len() {
        let mut idx = [None; 6];
        if !fixed.contains(&c) {
            for (k, slot) in idx.iter_mut().enumerate() {
                if k >= 3 && scale_fix == Some((c, k - 3)) {
                    continue;
                }
                *slot = Some(n);
                n += 1;
            }
        }
        cam_idx.push(idx);
    }
    let mut intr_idx = Vec::with_capacity(problem.groups.len());
    for g in 0..problem.groups.len() {
        let mask = opts
            .free_intrinsics
            .get(g)
            .copied()
            .unwrap_or(opts.default_free_intrinsics);
        let mut v = Vec::new();
        for (i, &free) in mask.iter().enumerate() {
            if free {
                v.push((i, n));
                n += 1;
            }
        }
        intr_idx.push(v);
    }
    Layout {
        cam_idx,
        intr_idx,
        n,
    }
}

/// (가중 비용, 비가중 제곱합, 투영 실패 수).
fn evaluate(problem: &BaProblem, obs: &[Observation], loss: Loss) -> (f64, f64, usize) {
    let mut cost = 0.0;
    let mut sq = 0.0;
    let mut bad = 0;
    for o in obs {
        let k = &problem.groups[problem.camera_group[o.camera]];
        let pose = &problem.poses[o.camera];
        let xc = pose.transform(&problem.points[o.point]);
        if xc.z <= 0.0 {
            bad += 1;
            continue;
        }
        let z = (k.project_camera(&xc) - o.pixel).norm_squared();
        cost += 0.5 * loss.rho(z);
        sq += z;
    }
    (cost, sq, bad)
}

struct PointBlock {
    c: Matrix3<f64>,
    g: Vector3<f64>,
    /// 관측별 (축소 계통 번호, Jcᵀ w Jp 행).
    w: Vec<Vec<(usize, Vector3<f64>)>>,
}

struct Linearization {
    a: DMatrix<f64>,
    gc: DVector<f64>,
    points: Vec<(usize, PointBlock)>,
}

fn linearize(
    problem: &BaProblem,
    lay: &Layout,
    tracks: &[usize],
    by_point: &[Vec<usize>],
    obs: &[Observation],
    loss: Loss,
) -> Linearization {
    let mut a = DMatrix::<f64>::zeros(lay.n, lay.n);
    let mut gc = DVector::<f64>::zeros(lay.n);
    let mut points = Vec::with_capacity(tracks.len());
    for &p in tracks {
        let mut blk = PointBlock {
            c: Matrix3::zeros(),
            g: Vector3::zeros(),
            w: Vec::new(),
        };
        for &oi in &by_point[p] {
            let o = &obs[oi];
            let gidx = problem.camera_group[o.camera];
            let Some((r, j)) = residual_jacobian(
                &problem.groups[gidx],
                &problem.poses[o.camera],
                &problem.points[p],
                &o.pixel,
            ) else {
                continue;
            };
            let wgt = loss.weight(r.norm_squared());
            let jp = j.fixed_view::<2, 3>(0, 14).into_owned();
            blk.c += wgt * jp.transpose() * jp;
            blk.g += wgt * jp.transpose() * r;
            let mut cols: Vec<(usize, Vector2<f64>)> = Vec::with_capacity(14);
            for (k, idx) in lay.cam_idx[o.camera].iter().enumerate() {
                if let Some(idx) = *idx {
                    cols.push((idx, j.column(k).into_owned()));
                }
            }
            for &(i, idx) in &lay.intr_idx[gidx] {
                cols.push((idx, j.column(6 + i).into_owned()));
            }
            let mut wrow = Vec::with_capacity(cols.len());
            for &(ia, ca) in &cols {
                gc[ia] += wgt * ca.dot(&r);
                for &(ib, cb) in &cols {
                    a[(ia, ib)] += wgt * ca.dot(&cb);
                }
                wrow.push((ia, wgt * jp.transpose() * ca));
            }
            blk.w.push(wrow);
        }
        points.push((p, blk));
    }
    Linearization { a, gc, points }
}

/// 감쇠 축소 계통 (S, rhs, 점별 C⁻¹).
type Reduced = (DMatrix<f64>, DVector<f64>, Vec<Matrix3<f64>>);

/// 감쇠 λ 로 점 블록을 슈어 소거한다.
fn schur(lin: &Linearization, lambda: f64) -> Reduced {
    let n = lin.a.nrows();
    let mut s = lin.a.clone();
    for i in 0..n {
        s[(i, i)] += lambda * lin.a[(i, i)].max(1e-9);
    }
    let mut rhs = -lin.gc.clone();
    let mut cinv = Vec::with_capacity(lin.points.len());
    // 축소 계통 번호 → flat 위치. 점마다 채우고 되돌려 선형 탐색을 없앤다(누적 순서는 같다).
    let mut pos = vec![usize::MAX; n];
    let mut flat: Vec<(usize, Vector3<f64>)> = Vec::new();
    for (_, blk) in &lin.points {
        let mut c = blk.c;
        for i in 0..3 {
            c[(i, i)] += lambda * c[(i, i)].max(1e-9);
        }
        let ci = c.try_inverse().unwrap_or_else(Matrix3::zeros);
        // S -= W C⁻¹ Wᵀ, rhs += W C⁻¹ g_p
        let cg = ci * blk.g;
        flat.clear();
        for row in &blk.w {
            for &(ia, wa) in row {
                if pos[ia] == usize::MAX {
                    pos[ia] = flat.len();
                    flat.push((ia, wa));
                } else {
                    flat[pos[ia]].1 += wa;
                }
            }
        }
        for &(ia, _) in &flat {
            pos[ia] = usize::MAX;
        }
        for &(ia, wa) in &flat {
            let t = ci * wa;
            rhs[ia] += wa.dot(&cg);
            for &(ib, wb) in &flat {
                s[(ia, ib)] -= t.dot(&wb);
            }
        }
        cinv.push(ci);
    }
    (s, rhs, cinv)
}

/// 감쇠 λ 로 정규방정식을 풀어 (카메라 쪽 증분, 점별 증분) 을 돌려준다. 촐레스키 실패면 None.
fn solve(lin: &Linearization, lambda: f64) -> Option<(DVector<f64>, Vec<Vector3<f64>>)> {
    let n = lin.a.nrows();
    let (s, rhs, cinv) = schur(lin, lambda);
    let dc = if n > 0 {
        s.cholesky()?.solve(&rhs)
    } else {
        DVector::zeros(0)
    };
    let mut dp = Vec::with_capacity(lin.points.len());
    for ((_, blk), ci) in lin.points.iter().zip(&cinv) {
        // δp = C⁻¹(−g_p − Σ Wᵀ δc)
        let mut v = -blk.g;
        for row in &blk.w {
            for &(ia, wa) in row {
                v -= wa * dc[ia];
            }
        }
        dp.push(ci * v);
    }
    Some((dc, dp))
}

fn apply(
    problem: &BaProblem,
    lay: &Layout,
    lin: &Linearization,
    dc: &DVector<f64>,
    dp: &[Vector3<f64>],
) -> BaProblem {
    let mut out = problem.clone();
    for (c, idx) in lay.cam_idx.iter().enumerate() {
        if idx.iter().any(Option::is_some) {
            let d: Vec<f64> = idx.iter().map(|i| i.map_or(0.0, |i| dc[i])).collect();
            out.poses[c] = apply_pose(&problem.poses[c], &d);
        }
    }
    for (g, idx) in lay.intr_idx.iter().enumerate() {
        for &(i, k) in idx {
            let v = intr_get(&problem.groups[g], i) + dc[k];
            intr_set(&mut out.groups[g], i, v);
        }
    }
    for ((p, _), d) in lin.points.iter().zip(dp) {
        out.points[*p] = problem.points[*p] + d;
    }
    out
}

/// 크기·그룹 번호가 맞는지 본다.
fn input_is_consistent(problem: &BaProblem) -> bool {
    problem.poses.len() == problem.camera_group.len()
        && problem
            .camera_group
            .iter()
            .all(|&g| g < problem.groups.len())
}

/// 관측이 쓸 수 있는지(번호 범위·픽셀 유한성).
fn observation_is_valid(problem: &BaProblem, o: &Observation) -> bool {
    o.camera < problem.poses.len()
        && o.point < problem.points.len()
        && o.pixel.x.is_finite()
        && o.pixel.y.is_finite()
}

/// 번들 조정. `problem` 을 제자리에서 고친다. 선택되지 않은 점은 그대로 둔다.
/// 입력이 맞지 않으면(`BaStop::InvalidInput`) 아무것도 고치지 않는다.
pub fn bundle_adjust(problem: &mut BaProblem, opts: &BaOptions) -> BaReport {
    if !input_is_consistent(problem) {
        return BaReport {
            iterations: 0,
            refined: false,
            num_cameras: problem.poses.len(),
            num_tracks_used: 0,
            num_observations_used: 0,
            initial_rms: f64::NAN,
            final_rms: f64::NAN,
            initial_cost: f64::NAN,
            final_cost: f64::NAN,
            converged: false,
            stop: BaStop::InvalidInput,
            num_observations_rejected: problem.observations.len(),
            cholesky_failures: 0,
        };
    }
    let valid: Vec<Observation> = problem
        .observations
        .iter()
        .filter(|o| observation_is_valid(problem, o))
        .copied()
        .collect();
    let rejected = problem.observations.len() - valid.len();
    let tracks = select_tracks(problem.points.len(), &valid, opts.max_tracks);
    let mut used = vec![false; problem.points.len()];
    for &p in &tracks {
        used[p] = true;
    }
    let obs: Vec<Observation> = valid.into_iter().filter(|o| used[o.point]).collect();
    let mut by_point = vec![Vec::new(); problem.points.len()];
    for (i, o) in obs.iter().enumerate() {
        by_point[o.point].push(i);
    }
    let lay = layout(problem, opts);
    let n_obs = obs.len().max(1) as f64;
    let (mut cost, sq0, _) = evaluate(problem, &obs, opts.loss);
    let initial_cost = cost;
    let initial_rms = (sq0 / n_obs).sqrt();
    let mut final_sq = sq0;
    let mut lambda = opts.initial_lambda;
    let mut iterations = 0;
    let mut cholesky_failures = 0;
    let mut stop = if opts.max_iterations == 0 {
        BaStop::EvaluationOnly
    } else {
        BaStop::MaxIterations
    };
    if opts.max_iterations > 0 && !cost.is_finite() {
        stop = BaStop::StepFailed;
    }
    while stop == BaStop::MaxIterations && iterations < opts.max_iterations {
        iterations += 1;
        let lin = linearize(problem, &lay, &tracks, &by_point, &obs, opts.loss);
        let (_, _, bad0) = evaluate(problem, &obs, opts.loss);
        let mut accepted = false;
        for _ in 0..12 {
            match solve(&lin, lambda) {
                Some((dc, dp)) => {
                    let cand = apply(problem, &lay, &lin, &dc, &dp);
                    let (c_new, sq_new, bad) = evaluate(&cand, &obs, opts.loss);
                    if bad <= bad0 && c_new < cost {
                        let rel = (cost - c_new) / cost.max(1e-300);
                        *problem = cand;
                        cost = c_new;
                        final_sq = sq_new;
                        lambda = (lambda * 0.3).max(1e-12);
                        accepted = true;
                        if rel < opts.function_tolerance {
                            stop = BaStop::Converged;
                        }
                        break;
                    }
                }
                None => cholesky_failures += 1,
            }
            lambda *= 10.0;
        }
        if !accepted {
            stop = BaStop::StepFailed;
        }
    }
    BaReport {
        iterations,
        refined: iterations > 0,
        num_cameras: problem.poses.len(),
        num_tracks_used: tracks.len(),
        num_observations_used: obs.len(),
        initial_rms,
        final_rms: (final_sq / n_obs).sqrt(),
        initial_cost,
        final_cost: cost,
        converged: stop == BaStop::Converged,
        stop,
        num_observations_rejected: rejected,
        cholesky_failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion::Distortion;

    /// 결정적 난수(xorshift64*) + 가우시안(Box–Muller).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn uni(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.next()
        }
        fn gauss(&mut self) -> f64 {
            let u = self.next().max(1e-300);
            let v = self.next();
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    fn true_intr() -> DistortedIntrinsics {
        DistortedIntrinsics {
            fx: 800.0,
            fy: 805.0,
            cx: 480.0,
            cy: 270.0,
            dist: Distortion {
                k1: -0.10,
                k2: 0.02,
                p1: 0.001,
                p2: -0.0008,
            },
        }
    }

    /// 지면 위 점 구름을 여러 높이·기울기의 카메라가 내려다보는 장면.
    fn scene(seed: u64, n_cam: usize, n_pts: usize) -> (BaProblem, Rng) {
        scene_groups(seed, n_cam, n_pts, vec![true_intr()], 24.0)
    }

    /// 카메라 c 는 그룹 c % groups.len() 에 속한다.
    fn scene_groups(
        seed: u64,
        n_cam: usize,
        n_pts: usize,
        groups: Vec<DistortedIntrinsics>,
        span: f64,
    ) -> (BaProblem, Rng) {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let camera_group: Vec<usize> = (0..n_cam).map(|c| c % groups.len()).collect();
        let mut poses = Vec::new();
        for i in 0..n_cam {
            let t = i as f64 / (n_cam - 1) as f64;
            let c = Point3::new(span * (t - 0.5), rng.uni(-4.0, 4.0), rng.uni(18.0, 24.0));
            // 띠가 길면 카메라 바로 아래 근처를 본다(span = 24 이면 원점 근처).
            let tx = c.x * (1.0 - 24.0 / span);
            let target = Point3::new(tx + rng.uni(-3.0, 3.0), rng.uni(-3.0, 3.0), 0.0);
            // 카메라 z 축이 target 을 향하도록 (세계→카메라 회전).
            let z = (target - c).normalize();
            let up = Vector3::new(0.0, 1.0, 0.0);
            let x = up.cross(&z).normalize();
            let y = z.cross(&x);
            let rcw = Matrix3::from_rows(&[x.transpose(), y.transpose(), z.transpose()]);
            let roll = Rotation3::from_axis_angle(&Vector3::z_axis(), rng.uni(-0.5, 0.5));
            let r = roll * Rotation3::from_matrix_unchecked(rcw);
            poses.push(Pose::from_center(r, &c));
        }
        let mut points = Vec::new();
        let mut observations = Vec::new();
        while points.len() < n_pts {
            let hx = 0.5 * span + 2.0;
            let x = Point3::new(rng.uni(-hx, hx), rng.uni(-9.0, 9.0), rng.uni(-1.0, 3.0));
            let mut seen = Vec::new();
            for (c, pose) in poses.iter().enumerate() {
                let xc = pose.transform(&x);
                if xc.z <= 0.0 {
                    continue;
                }
                let px = groups[camera_group[c]].project_camera(&xc);
                if px.x > 0.0 && px.x < 960.0 && px.y > 0.0 && px.y < 540.0 {
                    seen.push((c, px));
                }
            }
            if seen.len() < 3 {
                continue;
            }
            let p = points.len();
            points.push(x);
            for (c, px) in seen {
                observations.push(Observation {
                    camera: c,
                    point: p,
                    pixel: px,
                });
            }
        }
        (
            BaProblem {
                groups,
                camera_group,
                poses,
                points,
                observations,
            },
            rng,
        )
    }

    fn add_noise(p: &mut BaProblem, rng: &mut Rng, sigma: f64) {
        for o in &mut p.observations {
            o.pixel += Vector2::new(rng.gauss(), rng.gauss()) * sigma;
        }
    }

    /// 포즈(첫 카메라 제외)·점·내부 파라미터 섭동.
    fn perturb(p: &mut BaProblem, rng: &mut Rng) {
        perturb_geometry(p, rng);
        let k = &mut p.groups[0];
        k.fx *= 1.03;
        k.fy *= 0.98;
        k.cx += 6.0;
        k.cy -= 5.0;
        k.dist = Distortion::default();
    }

    /// 포즈(첫 카메라 제외)·점 섭동.
    fn perturb_geometry(p: &mut BaProblem, rng: &mut Rng) {
        for c in 1..p.poses.len() {
            let w = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.01;
            let dt = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.3;
            p.poses[c] = apply_pose(&p.poses[c], &[w.x, w.y, w.z, dt.x, dt.y, dt.z]);
        }
        for x in &mut p.points {
            *x += Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.2;
        }
    }

    /// 닮음 정렬(Umeyama) 후 중심 오차 RMS.
    fn aligned_center_rms(est: &[Pose], gt: &[Pose]) -> f64 {
        let a: Vec<Vector3<f64>> = est.iter().map(|p| p.center().coords).collect();
        let b: Vec<Vector3<f64>> = gt.iter().map(|p| p.center().coords).collect();
        let n = a.len() as f64;
        let ma = a.iter().sum::<Vector3<f64>>() / n;
        let mb = b.iter().sum::<Vector3<f64>>() / n;
        let mut cov = Matrix3::zeros();
        let mut va = 0.0;
        for (x, y) in a.iter().zip(&b) {
            cov += (y - mb) * (x - ma).transpose();
            va += (x - ma).norm_squared();
        }
        cov /= n;
        va /= n;
        let svd = cov.svd(true, true);
        let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
        let mut d = Matrix3::identity();
        if (u * vt).determinant() < 0.0 {
            d[(2, 2)] = -1.0;
        }
        let r = u * d * vt;
        let s = (svd.singular_values.component_mul(&d.diagonal())).sum() / va;
        let t = mb - s * r * ma;
        let se: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (s * r * x + t - y).norm_squared())
            .sum();
        (se / n).sqrt()
    }

    fn param_count(p: &BaProblem) -> usize {
        6 * (p.poses.len() - 1) + 8 + 3 * p.points.len()
    }

    #[test]
    fn analytic_jacobian_matches_numeric() {
        let (p, _) = scene(3, 6, 40);
        let mut worst: f64 = 0.0;
        for o in p.observations.iter().take(60) {
            let k = p.groups[0];
            let pose = p.poses[o.camera];
            let x = p.points[o.point];
            let (r0, j) = residual_jacobian(&k, &pose, &x, &o.pixel).unwrap();
            let _ = r0;
            for col in 0..17 {
                let h = if col < 3 {
                    1e-6
                } else if col < 6 {
                    1e-5
                } else if col < 10 {
                    1e-4
                } else if col < 14 {
                    1e-7
                } else {
                    1e-5
                };
                let eval = |s: f64| {
                    let mut kk = k;
                    let mut pp = pose;
                    let mut xx = x;
                    if col < 6 {
                        let mut d = [0.0; 6];
                        d[col] = s * h;
                        pp = apply_pose(&pose, &d);
                    } else if col < 14 {
                        intr_set(&mut kk, col - 6, intr_get(&k, col - 6) + s * h);
                    } else {
                        xx[col - 14] += s * h;
                    }
                    residual_jacobian(&kk, &pp, &xx, &o.pixel).unwrap().0
                };
                let num = (eval(1.0) - eval(-1.0)) / (2.0 * h);
                let ana = j.column(col);
                let scale = ana.norm().max(1.0);
                worst = worst.max((num - ana).norm() / scale);
            }
        }
        assert!(worst < 1e-6, "최대 상대 차 {worst}");
    }

    #[test]
    fn zero_iterations_is_evaluation_only() {
        let (mut p, mut rng) = scene(5, 6, 80);
        add_noise(&mut p, &mut rng, 0.5);
        let before = p.clone();
        let opts = BaOptions {
            max_iterations: 0,
            ..Default::default()
        };
        let rep = bundle_adjust(&mut p, &opts);
        assert_eq!(rep.iterations, 0);
        assert!(!rep.refined);
        assert_eq!(rep.num_cameras, 6);
        assert_eq!(rep.initial_rms, rep.final_rms);
        assert_eq!(p.points, before.points);
        // 잡음만 있으니 RMS ≈ 0.5·√2.
        assert!(
            (rep.initial_rms - 0.5 * 2f64.sqrt()).abs() < 0.08,
            "{}",
            rep.initial_rms
        );
    }

    #[test]
    fn track_limit_is_deterministic() {
        let (p, _) = scene(7, 8, 300);
        let a = select_tracks(p.points.len(), &p.observations, 50);
        let b = select_tracks(p.points.len(), &p.observations, 50);
        assert_eq!(a, b);
        assert_eq!(a.len(), 50);
        let mut count = vec![0; p.points.len()];
        for o in &p.observations {
            count[o.point] += 1;
        }
        let min_sel = a.iter().map(|&i| count[i]).min().unwrap();
        let max_rest = (0..p.points.len())
            .filter(|i| !a.contains(i))
            .map(|i| count[i])
            .max()
            .unwrap();
        assert!(min_sel >= max_rest);
        let mut q = p.clone();
        let rep = bundle_adjust(
            &mut q,
            &BaOptions {
                max_iterations: 3,
                max_tracks: 50,
                ..Default::default()
            },
        );
        assert_eq!(rep.num_tracks_used, 50);
        let unused = (0..p.points.len()).find(|i| !a.contains(i)).unwrap();
        assert_eq!(q.points[unused], p.points[unused]);
    }

    /// 섭동 + 0.5px 잡음 → BA 후 RMS 가 잡음 기댓값 σ√2·√(1−P/2M) 에 근접.
    #[test]
    fn recovers_synthetic_scene() {
        let sigma = 0.5;
        for seed in [11u64, 12, 13] {
            let (gt, mut rng) = scene(seed, 10, 400);
            let mut p = gt.clone();
            add_noise(&mut p, &mut rng, sigma);
            perturb(&mut p, &mut rng);
            let rep = bundle_adjust(
                &mut p,
                &BaOptions {
                    loss: Loss::Squared,
                    max_iterations: 100,
                    ..Default::default()
                },
            );
            let m = rep.num_observations_used as f64;
            let expect = sigma * 2f64.sqrt() * (1.0 - param_count(&p) as f64 / (2.0 * m)).sqrt();
            let ce = aligned_center_rms(&p.poses, &gt.poses);
            let k = p.groups[0];
            let kt = gt.groups[0];
            eprintln!(
                "seed {seed}: init {:.2} final {:.4} expect {:.4} it {} center {:.4} fx {:.3} cx {:.3} k1 {:.5}",
                rep.initial_rms,
                rep.final_rms,
                expect,
                rep.iterations,
                ce,
                (k.fx - kt.fx) / kt.fx,
                k.cx - kt.cx,
                k.dist.k1 - kt.dist.k1
            );
            assert!(rep.refined);
            assert!(rep.initial_rms > 10.0);
            assert!(
                (rep.final_rms / expect - 1.0).abs() < 0.05,
                "{} vs {expect}",
                rep.final_rms
            );
            assert!(ce < 0.05, "중심 오차 {ce}");
            assert!(((k.fx - kt.fx) / kt.fx).abs() < 0.005);
            assert!(((k.fy - kt.fy) / kt.fy).abs() < 0.005);
            assert!((k.cx - kt.cx).abs() < 4.0 && (k.cy - kt.cy).abs() < 4.0);
            assert!((k.dist.k1 - kt.dist.k1).abs() < 0.01);
            assert!((k.dist.k2 - kt.dist.k2).abs() < 0.02);
        }
    }

    /// 관측 10% 를 20~60px 이상치로 바꿨을 때 강건 손실이 정답에 더 가깝다.
    #[test]
    fn robust_loss_resists_outliers() {
        for seed in [21u64, 22] {
            let (gt, mut rng) = scene(seed, 10, 400);
            let mut base = gt.clone();
            add_noise(&mut base, &mut rng, 0.5);
            let mut inlier = vec![true; base.observations.len()];
            for (i, o) in base.observations.iter_mut().enumerate() {
                if rng.next() < 0.1 {
                    let ang = rng.uni(0.0, std::f64::consts::TAU);
                    let mag = rng.uni(20.0, 60.0);
                    o.pixel += Vector2::new(ang.cos(), ang.sin()) * mag;
                    inlier[i] = false;
                }
            }
            perturb(&mut base, &mut rng);
            let run = |loss: Loss| {
                let mut p = base.clone();
                bundle_adjust(
                    &mut p,
                    &BaOptions {
                        loss,
                        max_iterations: 100,
                        ..Default::default()
                    },
                );
                let mut sq = 0.0;
                let mut n = 0.0;
                for (o, &ok) in p.observations.iter().zip(&inlier) {
                    if ok {
                        let k = p.groups[0];
                        let r = k.project_camera(&p.poses[o.camera].transform(&p.points[o.point]))
                            - o.pixel;
                        sq += r.norm_squared();
                        n += 1.0;
                    }
                }
                ((sq / n).sqrt(), aligned_center_rms(&p.poses, &gt.poses))
            };
            let (l2_rms, l2_c) = run(Loss::Squared);
            let (hu_rms, hu_c) = run(Loss::Huber(1.0));
            let (ca_rms, ca_c) = run(Loss::Cauchy(1.0));
            eprintln!(
                "seed {seed}: L2 inlier {l2_rms:.3} c {l2_c:.4} | Huber {hu_rms:.3} c {hu_c:.4} | Cauchy {ca_rms:.3} c {ca_c:.4}"
            );
            let clean = 0.5 * 2f64.sqrt();
            // Huber 는 이상치 영향이 유계일 뿐 0 이 아니라 Cauchy 보다 덜 줄어든다.
            assert!(hu_rms < 2.0 * clean, "{hu_rms}");
            // Cauchy 는 비볼록이라 먼 초기값에서 내부 파라미터가 국소해에 남는 시드가 있다(중심만 판정).
            assert!(l2_rms > 3.0 * clean, "{l2_rms}");
            assert!(
                ca_c < 0.2 * l2_c && hu_c < 0.2 * l2_c,
                "{ca_c} {hu_c} {l2_c}"
            );
            let _ = ca_rms;
        }
    }

    fn noisy_perturbed(seed: u64) -> BaProblem {
        let (gt, mut rng) = scene(seed, 10, 400);
        let mut p = gt;
        add_noise(&mut p, &mut rng, 0.5);
        perturb(&mut p, &mut rng);
        p
    }

    /// 축척 게이지 고정: 감쇠 초기값과 무관하게 같은 해(|c1−c0| 상대 변화 < 1e-9),
    /// 축소 계통 촐레스키 실패 0회.
    #[test]
    fn scale_gauge_is_fixed() {
        let base = noisy_perturbed(11);
        let run = |lambda: f64| {
            let mut p = base.clone();
            let rep = bundle_adjust(
                &mut p,
                &BaOptions {
                    loss: Loss::Squared,
                    max_iterations: 200,
                    initial_lambda: lambda,
                    function_tolerance: 0.0,
                    ..Default::default()
                },
            );
            ((p.poses[1].center() - p.poses[0].center()).norm(), rep)
        };
        let (d_lo, rep_lo) = run(1e-12);
        let (d_hi, rep_hi) = run(1e-4);
        eprintln!(
            "|c1-c0| {d_lo:.12} vs {d_hi:.12}, it {} {}, chol fail {} {}, rms {:.6} {:.6}",
            rep_lo.iterations,
            rep_hi.iterations,
            rep_lo.cholesky_failures,
            rep_hi.cholesky_failures,
            rep_lo.final_rms,
            rep_hi.final_rms
        );
        assert!(((d_lo - d_hi) / d_hi).abs() < 1e-9, "{d_lo} vs {d_hi}");
        assert_eq!(rep_lo.cholesky_failures, 0);
        assert_eq!(rep_hi.cholesky_failures, 0);
        assert!(rep_lo.final_rms < 1.0 && rep_hi.final_rms < 1.0);
    }

    /// `fixed_cameras` 가 비면 첫 카메라를 고정하는 기본 게이지.
    #[test]
    fn empty_fixed_cameras_uses_default_gauge() {
        let mut p = noisy_perturbed(12);
        let pose0 = p.poses[0];
        let rep = bundle_adjust(
            &mut p,
            &BaOptions {
                loss: Loss::Squared,
                max_iterations: 100,
                fixed_cameras: vec![],
                ..Default::default()
            },
        );
        assert!(rep.converged, "{:?}", rep.stop);
        assert_eq!(rep.cholesky_failures, 0);
        assert!(rep.final_rms < 1.0, "{}", rep.final_rms);
        assert_eq!(p.poses[0], pose0);
        // 범위 밖 번호만 있는 목록도 같다.
        let mut q = noisy_perturbed(12);
        let rep2 = bundle_adjust(
            &mut q,
            &BaOptions {
                loss: Loss::Squared,
                max_iterations: 100,
                fixed_cameras: vec![99],
                ..Default::default()
            },
        );
        assert_eq!(rep2.final_cost, rep.final_cost);
    }

    fn short_opts() -> BaOptions {
        BaOptions {
            loss: Loss::Squared,
            max_iterations: 30,
            ..Default::default()
        }
    }

    #[test]
    fn out_of_range_observations_are_rejected() {
        for bad in [
            Observation {
                camera: 0,
                point: 400,
                pixel: Vector2::new(10.0, 10.0),
            },
            Observation {
                camera: 10,
                point: 0,
                pixel: Vector2::new(10.0, 10.0),
            },
            Observation {
                camera: 0,
                point: 0,
                pixel: Vector2::new(f64::NAN, 10.0),
            },
            Observation {
                camera: 0,
                point: 0,
                pixel: Vector2::new(10.0, f64::INFINITY),
            },
        ] {
            let mut p = noisy_perturbed(13);
            let n_valid = p.observations.len();
            p.observations.push(bad);
            let rep = bundle_adjust(&mut p, &short_opts());
            assert_eq!(rep.num_observations_rejected, 1, "{bad:?}");
            assert_eq!(rep.num_observations_used, n_valid);
            assert!(rep.final_rms.is_finite() && rep.final_rms < 1.0);
            assert!(rep.converged, "{:?}", rep.stop);
        }
        // select_tracks 도 범위 밖 점 번호에서 패닉하지 않는다.
        let o = Observation {
            camera: 0,
            point: 7,
            pixel: Vector2::zeros(),
        };
        assert!(select_tracks(3, &[o, o], 10).is_empty());
    }

    #[test]
    fn inconsistent_input_is_reported_without_changes() {
        let base = noisy_perturbed(14);
        let mut bad_group = base.clone();
        bad_group.camera_group[3] = 1;
        let mut short_groups = base.clone();
        short_groups.camera_group.pop();
        let mut extra_pose = base.clone();
        extra_pose.poses.push(base.poses[0]);
        for mut p in [bad_group, short_groups, extra_pose] {
            let before = p.clone();
            let rep = bundle_adjust(&mut p, &short_opts());
            assert_eq!(rep.stop, BaStop::InvalidInput);
            assert!(!rep.converged && !rep.refined);
            assert_eq!(p.points, before.points);
            assert_eq!(p.poses, before.poses);
        }
    }

    #[test]
    fn single_observation_tracks_are_not_used() {
        let mut p = noisy_perturbed(15);
        let before = select_tracks(p.points.len(), &p.observations, usize::MAX).len();
        let x = Point3::new(0.5, 0.5, 1.0);
        p.points.push(x);
        let px = p.groups[0].project_camera(&p.poses[0].transform(&x));
        p.observations.push(Observation {
            camera: 0,
            point: p.points.len() - 1,
            pixel: px + Vector2::new(3.0, -2.0),
        });
        let rep = bundle_adjust(&mut p, &short_opts());
        assert_eq!(rep.num_tracks_used, before);
        assert_eq!(*p.points.last().unwrap(), x);
    }

    /// 비유한 비용·촐레스키 실패는 수렴이 아니라 단계 실패로 보고한다.
    #[test]
    fn failures_are_not_reported_as_convergence() {
        let mut p = noisy_perturbed(16);
        let q = p.observations[0].point;
        p.points[q].x = f64::NAN;
        let rep = bundle_adjust(&mut p, &short_opts());
        assert_eq!(rep.stop, BaStop::StepFailed);
        assert!(!rep.converged);

        // 관측 없는 카메라 + 감쇠 0 → 축소 계통이 특이해 촐레스키가 매번 실패.
        let mut p = noisy_perturbed(16);
        p.poses.push(p.poses[5]);
        p.camera_group.push(0);
        let before = p.clone();
        let rep = bundle_adjust(
            &mut p,
            &BaOptions {
                initial_lambda: 0.0,
                ..short_opts()
            },
        );
        assert_eq!(rep.stop, BaStop::StepFailed);
        assert!(!rep.converged);
        assert_eq!(rep.cholesky_failures, 12);
        assert_eq!(p.points, before.points);
    }

    /// 내부 파라미터 3그룹(폴더별 공유). 그룹 2 는 k2·p1·p2 를 고정한다.
    #[test]
    fn recovers_three_intrinsic_groups() {
        let sigma = 0.5;
        let mk = |fx: f64, k1: f64, cx: f64| DistortedIntrinsics {
            fx,
            fy: fx * 1.006,
            cx,
            cy: 270.0 + (fx - 800.0) * 0.3,
            dist: Distortion {
                k1,
                k2: 0.02,
                p1: 0.001,
                p2: -0.0008,
            },
        };
        let truth = vec![
            mk(800.0, -0.10, 480.0),
            mk(820.0, -0.05, 470.0),
            mk(790.0, 0.0, 490.0),
        ];
        let mask2 = [true, true, true, true, true, false, false, false];
        for seed in [31u64, 32, 33] {
            let (gt, mut rng) = scene_groups(seed, 18, 900, truth.clone(), 24.0);
            let mut p = gt.clone();
            add_noise(&mut p, &mut rng, sigma);
            perturb_geometry(&mut p, &mut rng);
            for (g, k) in p.groups.iter_mut().enumerate() {
                k.fx *= 1.0 + 0.02 * (g as f64 - 1.0) + 0.01;
                k.fy *= 0.98;
                k.cx += 5.0 - 3.0 * g as f64;
                k.cy -= 4.0;
                k.dist.k1 = 0.0;
                if g != 2 {
                    k.dist = Distortion::default();
                }
            }
            let fixed_before = [
                p.groups[2].dist.k2,
                p.groups[2].dist.p1,
                p.groups[2].dist.p2,
            ];
            let rep = bundle_adjust(
                &mut p,
                &BaOptions {
                    loss: Loss::Squared,
                    max_iterations: 100,
                    free_intrinsics: vec![[true; 8], [true; 8], mask2],
                    ..Default::default()
                },
            );
            let free: usize = 8 + 8 + 5;
            // 게이지 7 = 고정 카메라 6 + 축척 1.
            let params = 6 * p.poses.len() - 7 + free + 3 * rep.num_tracks_used;
            let m = rep.num_observations_used as f64;
            let expect = sigma * 2f64.sqrt() * (1.0 - params as f64 / (2.0 * m)).sqrt();
            eprintln!(
                "seed {seed}: init {:.2} final {:.4} expect {:.4} it {} {:?}",
                rep.initial_rms, rep.final_rms, expect, rep.iterations, rep.stop
            );
            assert!(rep.converged, "{:?}", rep.stop);
            assert!(
                (rep.final_rms / expect - 1.0).abs() < 0.05,
                "{} vs {expect}",
                rep.final_rms
            );
            for (g, (k, kt)) in p.groups.iter().zip(&truth).enumerate() {
                eprintln!(
                    "  g{g}: fx {:.4}% fy {:.4}% cx {:.3} cy {:.3} k1 {:.5}",
                    100.0 * (k.fx - kt.fx) / kt.fx,
                    100.0 * (k.fy - kt.fy) / kt.fy,
                    k.cx - kt.cx,
                    k.cy - kt.cy,
                    k.dist.k1 - kt.dist.k1
                );
                assert!(((k.fx - kt.fx) / kt.fx).abs() < 0.005, "g{g} fx");
                assert!(((k.fy - kt.fy) / kt.fy).abs() < 0.005, "g{g} fy");
                assert!(
                    (k.cx - kt.cx).abs() < 4.0 && (k.cy - kt.cy).abs() < 4.0,
                    "g{g} pp"
                );
                assert!((k.dist.k1 - kt.dist.k1).abs() < 0.01, "g{g} k1");
            }
            let k2 = &p.groups[2].dist;
            assert_eq!(k2.k2.to_bits(), fixed_before[0].to_bits());
            assert_eq!(k2.p1.to_bits(), fixed_before[1].to_bits());
            assert_eq!(k2.p2.to_bits(), fixed_before[2].to_bits());
        }
    }

    /// 선형 탐색으로 누적하던 슈어 소거(이전 방식)와 결과가 비트 단위로 같다.
    #[test]
    fn schur_matches_linear_search_accumulation() {
        let p = noisy_perturbed(17);
        let opts = BaOptions::default();
        let tracks = select_tracks(p.points.len(), &p.observations, usize::MAX);
        let mut by_point = vec![Vec::new(); p.points.len()];
        for (i, o) in p.observations.iter().enumerate() {
            by_point[o.point].push(i);
        }
        let lay = layout(&p, &opts);
        let lin = linearize(&p, &lay, &tracks, &by_point, &p.observations, opts.loss);
        let lambda = 1e-3;
        let (s, rhs, _) = schur(&lin, lambda);
        let n = lin.a.nrows();
        let mut s0 = lin.a.clone();
        for i in 0..n {
            s0[(i, i)] += lambda * lin.a[(i, i)].max(1e-9);
        }
        let mut rhs0 = -lin.gc.clone();
        for (_, blk) in &lin.points {
            let mut c = blk.c;
            for i in 0..3 {
                c[(i, i)] += lambda * c[(i, i)].max(1e-9);
            }
            let ci = c.try_inverse().unwrap_or_else(Matrix3::zeros);
            let cg = ci * blk.g;
            let mut flat: Vec<(usize, Vector3<f64>)> = Vec::new();
            for row in &blk.w {
                for &(ia, wa) in row {
                    if let Some(e) = flat.iter_mut().find(|e| e.0 == ia) {
                        e.1 += wa;
                    } else {
                        flat.push((ia, wa));
                    }
                }
            }
            for &(ia, wa) in &flat {
                let t = ci * wa;
                rhs0[ia] += wa.dot(&cg);
                for &(ib, wb) in &flat {
                    s0[(ia, ib)] -= t.dot(&wb);
                }
            }
        }
        assert_eq!(s, s0);
        assert_eq!(rhs, rhs0);
    }

    /// 실제 규모(카메라 240, 점 10만) 구간 시간. `cargo test --release -- --ignored ba_scale_timing --nocapture`.
    #[test]
    #[ignore]
    fn ba_scale_timing() {
        use std::time::Instant;
        let t0 = Instant::now();
        // 카메라 간격 4 m 띠: 점마다 관측 약 6개.
        let (mut p, mut rng) = scene_groups(41, 240, 100_000, vec![true_intr()], 960.0);
        add_noise(&mut p, &mut rng, 0.5);
        perturb(&mut p, &mut rng);
        let t_gen = t0.elapsed().as_secs_f64();
        let opts = BaOptions {
            loss: Loss::Huber(2.0),
            ..Default::default()
        };
        let tracks = select_tracks(p.points.len(), &p.observations, opts.max_tracks);
        let mut by_point = vec![Vec::new(); p.points.len()];
        for (i, o) in p.observations.iter().enumerate() {
            by_point[o.point].push(i);
        }
        let lay = layout(&p, &opts);
        let t = Instant::now();
        let lin = linearize(&p, &lay, &tracks, &by_point, &p.observations, opts.loss);
        let t_lin = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (s, rhs, _) = schur(&lin, 1e-4);
        let t_schur = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ok = s.cholesky().map(|c| c.solve(&rhs)).is_some();
        let t_chol = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (cost, _, _) = evaluate(&p, &p.observations, opts.loss);
        let t_eval = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let rep = bundle_adjust(
            &mut p,
            &BaOptions {
                max_iterations: 3,
                ..opts
            },
        );
        let t_ba = t.elapsed().as_secs_f64();
        eprintln!(
            "cams {} pts {} obs {} n {} | gen {t_gen:.2}s lin {t_lin:.3}s schur {t_schur:.3}s chol {t_chol:.3}s ({ok}) eval {t_eval:.3}s cost {cost:.3e} | 3 it {t_ba:.2}s ({:.2}s/it) rms {:.3}->{:.3}",
            p.poses.len(),
            p.points.len(),
            p.observations.len(),
            lay.n,
            t_ba / rep.iterations.max(1) as f64,
            rep.initial_rms,
            rep.final_rms
        );
        assert!(ok);
    }
}
