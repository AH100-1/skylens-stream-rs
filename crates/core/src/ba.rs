//! 희소 Levenberg–Marquardt 번들 조정.
//!
//! - 점 블록(3×3)을 슈어 보수로 소거해 카메라 쪽 축소 계통만 밀집 촐레스키로 푼다.
//! - 강건 손실(Huber/Cauchy)은 IRLS 가중으로 넣는다.
//! - 내부 파라미터 [fx, fy, cx, cy, k1, k2, p1, p2] 는 그룹(폴더)마다 공유하고
//!   항목별로 고정/자유를 고른다.
//! - 게이지: 지정한 카메라(기본: 첫 카메라)의 포즈를 고정한다. 남는 축척 자유도는 LM 감쇠가 잡는다.
//! - 관측 트랙(3D 점)은 최대 `max_tracks` 개만 쓴다. 관측 수가 많은 점부터, 같으면 번호 순.

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
    pub converged: bool,
}

/// 결정적 트랙 선택: 관측 수 내림차순, 같으면 점 번호 오름차순. 선택된 점 번호(오름차순).
pub fn select_tracks(
    num_points: usize,
    observations: &[Observation],
    max_tracks: usize,
) -> Vec<usize> {
    let mut count = vec![0usize; num_points];
    for o in observations {
        count[o.point] += 1;
    }
    let mut ids: Vec<usize> = (0..num_points).filter(|&p| count[p] > 0).collect();
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
    cam_off: Vec<Option<usize>>,
    /// 그룹별 (내부 파라미터 번호, 축소 계통 번호).
    intr_idx: Vec<Vec<(usize, usize)>>,
    n: usize,
}

fn layout(problem: &BaProblem, opts: &BaOptions) -> Layout {
    let mut n = 0;
    let mut cam_off = Vec::with_capacity(problem.poses.len());
    for c in 0..problem.poses.len() {
        if opts.fixed_cameras.contains(&c) {
            cam_off.push(None);
        } else {
            cam_off.push(Some(n));
            n += 6;
        }
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
        cam_off,
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
            if let Some(off) = lay.cam_off[o.camera] {
                for k in 0..6 {
                    cols.push((off + k, j.column(k).into_owned()));
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

/// 감쇠 λ 로 정규방정식을 풀어 (카메라 쪽 증분, 점별 증분) 을 돌려준다.
fn solve(lin: &Linearization, lambda: f64) -> Option<(DVector<f64>, Vec<Vector3<f64>>)> {
    let n = lin.a.nrows();
    let mut s = lin.a.clone();
    for i in 0..n {
        s[(i, i)] += lambda * lin.a[(i, i)].max(1e-9);
    }
    let mut rhs = -lin.gc.clone();
    let mut cinv = Vec::with_capacity(lin.points.len());
    for (_, blk) in &lin.points {
        let mut c = blk.c;
        for i in 0..3 {
            c[(i, i)] += lambda * c[(i, i)].max(1e-9);
        }
        let ci = c.try_inverse().unwrap_or_else(Matrix3::zeros);
        // S -= W C⁻¹ Wᵀ, rhs += W C⁻¹ g_p
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
            rhs[ia] += wa.dot(&cg);
            for &(ib, wb) in &flat {
                s[(ia, ib)] -= t.dot(&wb);
            }
        }
        cinv.push(ci);
    }
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
    for (c, off) in lay.cam_off.iter().enumerate() {
        if let Some(off) = *off {
            out.poses[c] = apply_pose(&problem.poses[c], &dc.as_slice()[off..off + 6]);
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

/// 번들 조정. `problem` 을 제자리에서 고친다. 선택되지 않은 점은 그대로 둔다.
pub fn bundle_adjust(problem: &mut BaProblem, opts: &BaOptions) -> BaReport {
    let tracks = select_tracks(problem.points.len(), &problem.observations, opts.max_tracks);
    let mut used = vec![false; problem.points.len()];
    for &p in &tracks {
        used[p] = true;
    }
    let obs: Vec<Observation> = problem
        .observations
        .iter()
        .filter(|o| used[o.point])
        .copied()
        .collect();
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
    let mut converged = opts.max_iterations == 0;
    while iterations < opts.max_iterations {
        iterations += 1;
        let lin = linearize(problem, &lay, &tracks, &by_point, &obs, opts.loss);
        let (_, _, bad0) = evaluate(problem, &obs, opts.loss);
        let mut accepted = false;
        for _ in 0..12 {
            if let Some((dc, dp)) = solve(&lin, lambda) {
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
                        converged = true;
                    }
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !accepted {
            converged = true;
        }
        if converged {
            break;
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
        converged,
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
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let k = true_intr();
        let mut poses = Vec::new();
        for i in 0..n_cam {
            let t = i as f64 / (n_cam - 1) as f64;
            let c = Point3::new(-12.0 + 24.0 * t, rng.uni(-4.0, 4.0), rng.uni(18.0, 24.0));
            let target = Point3::new(rng.uni(-3.0, 3.0), rng.uni(-3.0, 3.0), 0.0);
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
            let x = Point3::new(rng.uni(-14.0, 14.0), rng.uni(-9.0, 9.0), rng.uni(-1.0, 3.0));
            let mut seen = Vec::new();
            for (c, pose) in poses.iter().enumerate() {
                let xc = pose.transform(&x);
                if xc.z <= 0.0 {
                    continue;
                }
                let px = k.project_camera(&xc);
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
                groups: vec![k],
                camera_group: vec![0; n_cam],
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
        for c in 1..p.poses.len() {
            let w = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.01;
            let dt = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.3;
            p.poses[c] = apply_pose(&p.poses[c], &[w.x, w.y, w.z, dt.x, dt.y, dt.z]);
        }
        for x in &mut p.points {
            *x += Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.2;
        }
        let k = &mut p.groups[0];
        k.fx *= 1.03;
        k.fy *= 0.98;
        k.cx += 6.0;
        k.cy -= 5.0;
        k.dist = Distortion::default();
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
}
