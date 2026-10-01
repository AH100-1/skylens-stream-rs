//! 두 시점 기하: 본질 행렬, 상대 자세 복원, 삼각측량.
//!
//! 규약: 정규화 좌표 n = K⁻¹ [u v 1]ᵀ, 카메라 좌표 x_c = R X + t.
//! 첫 카메라를 [I | 0], 둘째를 [R | t] 로 둘 때 n2ᵀ E n1 = 0, E = [t]× R.

use crate::camera::Intrinsics;
use crate::matching::{all_finite, fundamental_8pt, sampson_error};
use crate::math::{skew, Matrix3, Point3, Rotation3, SMatrix, Vector2, Vector3};

/// 정규화 좌표 대응으로 본질 행렬을 구한다(정규화 8점 + 특이값 (1,1,0) 투영).
pub fn essential_8pt(n1: &[Vector2<f64>], n2: &[Vector2<f64>]) -> Option<Matrix3<f64>> {
    fundamental_8pt(n1, n2).map(|e| project_to_essential(&e))
}

/// 픽셀 기본 행렬 F 를 본질 행렬 E = K2ᵀ F K1 로 바꾼다(본질 행렬 다양체로 투영).
pub fn essential_from_fundamental(
    f: &Matrix3<f64>,
    k1: &Intrinsics,
    k2: &Intrinsics,
) -> Matrix3<f64> {
    project_to_essential(&(kmat(k2).transpose() * f * kmat(k1)))
}

fn kmat(k: &Intrinsics) -> Matrix3<f64> {
    Matrix3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

/// 가장 가까운 본질 행렬(특이값 1,1,0)로 투영한다.
pub fn project_to_essential(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    // nalgebra 특이값은 정렬되지 않을 수 있어 가장 작은 것을 0 으로 둔다.
    let mut s = Vector3::new(1.0, 1.0, 1.0);
    s[svd.singular_values.imin()] = 0.0;
    u * Matrix3::from_diagonal(&s) * vt
}

/// E 의 네 가지 분해 (R, t) 후보. t 는 단위 벡터.
pub fn decompose_essential(e: &Matrix3<f64>) -> [(Rotation3<f64>, Vector3<f64>); 4] {
    let svd = e.svd(true, true);
    let (mut u, mut vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    // 영 특이값 축을 셋째 열로 옮긴다.
    let k = svd.singular_values.imin();
    if k != 2 {
        u.swap_columns(k, 2);
        vt.swap_rows(k, 2);
    }
    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = Rotation3::from_matrix(&(u * w * vt));
    let r2 = Rotation3::from_matrix(&(u * w.transpose() * vt));
    let t: Vector3<f64> = u.column(2).into();
    [(r1, t), (r1, -t), (r2, t), (r2, -t)]
}

/// 선형(DLT) 삼각측량. 카메라 1 = [I|0], 카메라 2 = [R|t], 입력은 정규화 좌표.
pub fn triangulate(
    r: &Rotation3<f64>,
    t: &Vector3<f64>,
    n1: &Vector2<f64>,
    n2: &Vector2<f64>,
) -> Option<Point3<f64>> {
    let p2 = r.matrix();
    let mut a = SMatrix::<f64, 4, 4>::zeros();
    a.set_row(0, &SMatrix::<f64, 1, 4>::new(-1.0, 0.0, n1.x, 0.0));
    a.set_row(1, &SMatrix::<f64, 1, 4>::new(0.0, -1.0, n1.y, 0.0));
    for (i, (c, row)) in [(n2.x, 0usize), (n2.y, 1)].into_iter().enumerate() {
        let v = |j: usize| c * p2[(2, j)] - p2[(row, j)];
        a.set_row(
            2 + i,
            &SMatrix::<f64, 1, 4>::new(v(0), v(1), v(2), c * t[2] - t[row]),
        );
    }
    let eig = (a.transpose() * a).symmetric_eigen();
    let h = eig.eigenvectors.column(eig.eigenvalues.imin());
    (h[3].abs() > 1e-12).then(|| Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
}

/// 상대 자세 복원 결과.
#[derive(Clone, Debug)]
pub struct RelativePose {
    pub rotation: Rotation3<f64>,
    /// 단위 길이 이동(스케일 미정).
    pub translation: Vector3<f64>,
    /// 두 카메라 앞에 삼각측량된 대응의 표시.
    pub in_front: Vec<bool>,
}

/// 순수 회전 판정 배수: 회전만으로 설명한 각 잔차 중앙값이
/// 에피폴라 잔차 중앙값의 이 배수 이하이면 이동 방향을 관측할 수 없다고 본다.
const ROTATION_ONLY_FACTOR: f64 = 3.0;

/// 대응을 회전 하나로 설명했을 때의 각 잔차(rad) 중앙값.
/// 단위 광선 u1, u2 에 대해 Σ‖u2 − R u1‖² 를 최소화하는 R(직교 프로크루스테스) 을 쓴다.
fn rotation_only_residual(n1: &[Vector2<f64>], n2: &[Vector2<f64>]) -> f64 {
    let ray = |n: &Vector2<f64>| Vector3::new(n.x, n.y, 1.0).normalize();
    let h = n1.iter().zip(n2).fold(Matrix3::zeros(), |h, (a, b)| {
        h + ray(a) * ray(b).transpose()
    });
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let d = (vt.transpose() * u.transpose()).determinant().signum();
    let r = vt.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose();
    median(
        n1.iter()
            .zip(n2)
            .map(|(a, b)| (r * ray(a)).cross(&ray(b)).norm().asin())
            .collect(),
    )
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// E 의 네 후보 중 두 카메라 앞(양의 깊이)에 놓이는 점이 가장 많은 것을 고른다.
/// 길이가 다르거나, 유한하지 않은 값이 있거나, 대응이 순수 회전으로 설명되면
/// (이동 방향 미정) None.
pub fn recover_pose(
    e: &Matrix3<f64>,
    n1: &[Vector2<f64>],
    n2: &[Vector2<f64>],
) -> Option<RelativePose> {
    if n1.is_empty()
        || n1.len() != n2.len()
        || !all_finite(n1)
        || !all_finite(n2)
        || !e.iter().all(|v| v.is_finite())
    {
        return None;
    }
    // 정규화 좌표의 Sampson 거리 ≈ 각 잔차(rad). E 의 스케일에 무관하다.
    let epi = median(
        n1.iter()
            .zip(n2)
            .map(|(a, b)| sampson_error(e, a, b).sqrt())
            .collect(),
    );
    if rotation_only_residual(n1, n2) <= ROTATION_ONLY_FACTOR * epi + 1e-12 {
        return None;
    }
    let mut best: Option<RelativePose> = None;
    let mut best_n = 0;
    for (r, t) in decompose_essential(e) {
        let in_front: Vec<bool> = n1
            .iter()
            .zip(n2)
            .map(|(a, b)| {
                triangulate(&r, &t, a, b)
                    .map(|x| x.z > 0.0 && (r * x.coords + t).z > 0.0)
                    .unwrap_or(false)
            })
            .collect();
        let n = in_front.iter().filter(|&&b| b).count();
        if n > best_n {
            best_n = n;
            best = Some(RelativePose {
                rotation: r,
                translation: t,
                in_front,
            });
        }
    }
    best
}

/// 상대 자세가 주어졌을 때 E = [t]× R.
pub fn essential_from_pose(r: &Rotation3<f64>, t: &Vector3<f64>) -> Matrix3<f64> {
    skew(t) * r.matrix()
}

/// 3변수 3차 이하 다항식: 계수 c[a][b][d] 는 xᵃ yᵇ zᵈ 의 계수(a+b+d ≤ 3).
type Poly = [[[f64; 4]; 4]; 4];

fn pmul(p: &Poly, q: &Poly) -> Poly {
    let mut r = [[[0.0; 4]; 4]; 4];
    for a in 0..4 {
        for b in 0..4 - a {
            for d in 0..4 - a - b {
                if p[a][b][d] == 0.0 {
                    continue;
                }
                for e in 0..4 - a - b - d {
                    for f in 0..4 - a - b - d - e {
                        for g in 0..4 - a - b - d - e - f {
                            r[a + e][b + f][d + g] += p[a][b][d] * q[e][f][g];
                        }
                    }
                }
            }
        }
    }
    r
}

fn padd(p: &Poly, q: &Poly, s: f64) -> Poly {
    let mut r = *p;
    for a in 0..4 {
        for b in 0..4 {
            for d in 0..4 {
                r[a][b][d] += s * q[a][b][d];
            }
        }
    }
    r
}

/// 5점 최소 해법: 정규화 좌표 대응 5개 → 본질 행렬 후보(최대 10개).
/// E = xX + yY + zZ + W (5×9 계의 영공간) 에 det E = 0, 2EEᵀE − tr(EEᵀ)E = 0 을 넣고,
/// z 를 숨은 변수로 둔 10×10 다항 행렬 M(z) 의 다항 고윳값 문제(동반 행렬)로 푼다.
#[allow(clippy::needless_range_loop)] // 행렬 첨자식이 수식과 그대로 대응한다.
pub fn essential_5pt(n1: &[Vector2<f64>], n2: &[Vector2<f64>]) -> Vec<Matrix3<f64>> {
    if n1.len() < 5 || n2.len() < 5 || !all_finite(&n1[..5]) || !all_finite(&n2[..5]) {
        return vec![];
    }
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for (a, b) in n1.iter().zip(n2).take(5) {
        let (p, q) = (Vector3::new(a.x, a.y, 1.0), Vector3::new(b.x, b.y, 1.0));
        let mut row = SMatrix::<f64, 9, 1>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                row[3 * i + j] = q[i] * p[j];
            }
        }
        ata += row * row.transpose();
    }
    let eig = ata.symmetric_eigen();
    let mut idx: Vec<usize> = (0..9).collect();
    idx.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    let basis: Vec<SMatrix<f64, 9, 1>> = idx[..4]
        .iter()
        .map(|&k| eig.eigenvectors.column(k).into())
        .collect();
    // 각 성분 E_ij 를 x, y, z, 1 의 1차 다항식으로.
    let mut e = [[[[[0.0; 4]; 4]; 4]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let k = 3 * i + j;
            e[i][j][1][0][0] = basis[0][k];
            e[i][j][0][1][0] = basis[1][k];
            e[i][j][0][0][1] = basis[2][k];
            e[i][j][0][0][0] = basis[3][k];
        }
    }
    let mut eqs: Vec<Poly> = Vec::with_capacity(10);
    // det E
    let mut det = [[[0.0; 4]; 4]; 4];
    for (j, sgn) in [(0usize, 1.0), (1, -1.0), (2, 1.0)] {
        let (u, v) = match j {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let minor = padd(&pmul(&e[1][u], &e[2][v]), &pmul(&e[1][v], &e[2][u]), -1.0);
        det = padd(&det, &pmul(&e[0][j], &minor), sgn);
    }
    eqs.push(det);
    // EEᵀ (2차)
    let zero = [[[0.0; 4]; 4]; 4];
    let mut eet = [[zero; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                eet[i][j] = padd(&eet[i][j], &pmul(&e[i][k], &e[j][k]), 1.0);
            }
        }
    }
    let tr = padd(&padd(&eet[0][0], &eet[1][1], 1.0), &eet[2][2], 1.0);
    for i in 0..3 {
        for j in 0..3 {
            let mut v = zero;
            for k in 0..3 {
                v = padd(&v, &pmul(&eet[i][k], &e[k][j]), 2.0);
            }
            eqs.push(padd(&v, &pmul(&tr, &e[i][j]), -1.0));
        }
    }
    // x,y 단항식 열: x³ y³ x²y xy² x² y² xy x y 1, 각 계수는 z 의 다항식(차수 ≤ 3).
    let cols: [(usize, usize); 10] = [
        (3, 0),
        (0, 3),
        (2, 1),
        (1, 2),
        (2, 0),
        (0, 2),
        (1, 1),
        (1, 0),
        (0, 1),
        (0, 0),
    ];
    let mut m = [SMatrix::<f64, 10, 10>::zeros(); 4];
    for (r, q) in eqs.iter().enumerate() {
        for (c, &(a, b)) in cols.iter().enumerate() {
            for d in 0..4 - a - b {
                m[d][(r, c)] = q[a][b][d];
            }
        }
    }
    // μ = 1/z: μ³M0 + μ²M1 + μM2 + M3 = 0 → 동반 행렬(30×30).
    let Some(m0inv) = m[0].try_inverse() else {
        return vec![];
    };
    let a: Vec<SMatrix<f64, 10, 10>> = (1..4).map(|k| m0inv * m[k]).collect();
    let mut comp = nalgebra::DMatrix::<f64>::zeros(30, 30);
    for k in 0..3 {
        comp.view_mut((0, 10 * k), (10, 10)).copy_from(&(-a[k]));
    }
    for k in 0..20 {
        comp[(10 + k, k)] = 1.0;
    }
    let mut out = vec![];
    for mu in comp.complex_eigenvalues().iter() {
        if mu.im.abs() > 1e-6 * mu.norm().max(1e-12) || mu.re.abs() < 1e-10 {
            continue;
        }
        let z = 1.0 / mu.re;
        let mz = m[0] + m[1] * z + m[2] * (z * z) + m[3] * (z * z * z);
        let svd = mz.svd(false, true);
        let vt = svd.v_t.unwrap();
        let v = vt.row(svd.singular_values.imin());
        if v[9].abs() < 1e-12 {
            continue;
        }
        let (x, y) = (v[7] / v[9], v[8] / v[9]);
        let ev = basis[0] * x + basis[1] * y + basis[2] * z + basis[3];
        let em = Matrix3::new(
            ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8],
        );
        let n = em.norm();
        if n.is_finite() && n > 0.0 {
            out.push(em / n);
        }
    }
    out
}

/// 정규화 좌표에서 부호 있는 Sampson 잔차 e / ‖∇e‖.
fn sampson_residual(e: &Matrix3<f64>, a: &Vector2<f64>, b: &Vector2<f64>) -> f64 {
    let (p, q) = (Vector3::new(a.x, a.y, 1.0), Vector3::new(b.x, b.y, 1.0));
    let (ep, etq) = (e * p, e.transpose() * q);
    let den = (ep.x * ep.x + ep.y * ep.y + etq.x * etq.x + etq.y * etq.y).sqrt();
    if den > 0.0 {
        q.dot(&ep) / den
    } else {
        0.0
    }
}

/// 상대 자세 (R, 단위 t) 를 Sampson 잔차 제곱합 최소화로 정밀화한다(Levenberg–Marquardt, 5 자유도).
/// R ← R·exp([ω]×), t ← normalize(t + B β) (B 는 t 에 수직인 두 축). 야코비안은 중앙 차분.
pub fn refine_pose(
    rotation: &Rotation3<f64>,
    translation: &Vector3<f64>,
    n1: &[Vector2<f64>],
    n2: &[Vector2<f64>],
    iters: usize,
) -> (Rotation3<f64>, Vector3<f64>) {
    let (mut r, mut t) = (*rotation, translation.normalize());
    let apply = |r: &Rotation3<f64>, t: &Vector3<f64>, d: &SMatrix<f64, 5, 1>| {
        let b1 = t
            .cross(&if t.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            })
            .normalize();
        let b2 = t.cross(&b1);
        let rn = r * Rotation3::new(Vector3::new(d[0], d[1], d[2]));
        (rn, (t + b1 * d[3] + b2 * d[4]).normalize())
    };
    let cost_vec = |r: &Rotation3<f64>, t: &Vector3<f64>| -> Vec<f64> {
        let e = essential_from_pose(r, t);
        n1.iter()
            .zip(n2)
            .map(|(a, b)| sampson_residual(&e, a, b))
            .collect()
    };
    let sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    let mut res = cost_vec(&r, &t);
    let mut lambda = 1e-3;
    for _ in 0..iters {
        let h = 1e-7;
        let mut jac = vec![SMatrix::<f64, 1, 5>::zeros(); res.len()];
        for k in 0..5 {
            let mut d = SMatrix::<f64, 5, 1>::zeros();
            d[k] = h;
            let (rp, tp) = apply(&r, &t, &d);
            d[k] = -h;
            let (rm, tm) = apply(&r, &t, &d);
            let (fp, fm) = (cost_vec(&rp, &tp), cost_vec(&rm, &tm));
            for i in 0..res.len() {
                jac[i][k] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        let mut jtj = SMatrix::<f64, 5, 5>::zeros();
        let mut jtr = SMatrix::<f64, 5, 1>::zeros();
        for (j, &e) in jac.iter().zip(&res) {
            jtj += j.transpose() * j;
            jtr += j.transpose() * e;
        }
        let c0 = sq(&res);
        let mut improved = false;
        for _ in 0..10 {
            let mut a = jtj;
            for k in 0..5 {
                a[(k, k)] *= 1.0 + lambda;
            }
            let Some(d) = a.cholesky().map(|c| -c.solve(&jtr)) else {
                lambda *= 10.0;
                continue;
            };
            let (rn, tn) = apply(&r, &t, &d);
            let rn_res = cost_vec(&rn, &tn);
            if sq(&rn_res) < c0 {
                (r, t, res) = (rn, tn, rn_res);
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved || (c0 - sq(&res)) < 1e-15 * c0.max(1e-300) {
            break;
        }
    }
    (r, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Camera, Pose};
    use crate::math::rotation_angle_between;

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

    /// 드론처럼 아래를 보는 두 카메라(기선 3 m, 고도 ~40 m)와 지면 근처 점.
    fn scene(n: usize, sigma_px: f64, seed: u64) -> Scene {
        let k = Intrinsics::from_hfov(960, 540, 70f64.to_radians());
        let r1 = Rotation3::from_euler_angles(0.04, -0.03, 0.2);
        let r2 = Rotation3::from_euler_angles(-0.05, 0.06, 0.28);
        let c1 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r1, &Point3::new(0.0, 0.0, -40.0)),
        };
        let c2 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r2, &Point3::new(3.0, 0.8, -40.5)),
        };
        let mut rng = Lcg(seed);
        let (mut x1, mut x2, mut pts) = (vec![], vec![], vec![]);
        while x1.len() < n {
            let x = Point3::new(
                (rng.next() - 0.5) * 40.0,
                (rng.next() - 0.5) * 24.0,
                (rng.next() - 0.5) * 10.0,
            );
            if let (Some(p), Some(q)) = (c1.project(&x), c2.project(&x)) {
                let noise = |r: &mut Lcg| Vector2::new(r.gauss(), r.gauss()) * sigma_px;
                x1.push(k.to_normalized(&(p + noise(&mut rng))));
                x2.push(k.to_normalized(&(q + noise(&mut rng))));
                pts.push(x);
            }
        }
        Scene {
            c1,
            c2,
            x1,
            x2,
            pts,
        }
    }

    struct Scene {
        c1: Camera,
        c2: Camera,
        x1: Vec<Vector2<f64>>,
        x2: Vec<Vector2<f64>>,
        pts: Vec<Point3<f64>>,
    }

    impl Scene {
        fn rel(&self) -> (Rotation3<f64>, Vector3<f64>) {
            let r = self.c2.pose.rotation * self.c1.pose.rotation.inverse();
            let t = self.c2.pose.translation - r * self.c1.pose.translation;
            (r, t)
        }
    }

    #[test]
    fn decomposition_contains_true_pose() {
        let s = scene(10, 0.0, 1);
        let (r, t) = s.rel();
        let e = essential_from_pose(&r, &t);
        let tu = t.normalize();
        let hit = decompose_essential(&e)
            .iter()
            .any(|(rc, tc)| rotation_angle_between(rc, &r) < 1e-9 && (tc - tu).norm() < 1e-9);
        assert!(hit, "네 후보 중 정답 없음");
    }

    #[test]
    fn triangulation_exact_without_noise() {
        let s = scene(50, 0.0, 2);
        let (r, t) = s.rel();
        let mut worst: f64 = 0.0;
        for i in 0..s.x1.len() {
            let x = triangulate(&r, &t, &s.x1[i], &s.x2[i]).unwrap();
            // 카메라 1 좌표계의 정답 점.
            let g = s.c1.pose.transform(&s.pts[i]);
            worst = worst.max((x.coords - g).norm());
        }
        eprintln!("triangulation worst err={worst:.2e} m");
        assert!(worst < 1e-6, "삼각측량 오차 {worst} m");
    }

    #[test]
    fn relative_pose_under_noise() {
        // (σ px, 회전 오차 상한 도, 이동 방향 오차 상한 도, 삼각측량 중앙값 오차 상한 m)
        // 선형 8점만(비선형 정밀화 없음) 기준. 기선 3 m / 깊이 ~40 m 라 이동 방향이 잡음에 민감하다.
        // 상한은 시드 7·11·23 측정 최댓값의 약 1.5 배.
        for (sigma, max_rot, max_dir, max_pt) in [
            (0.0, 1e-6, 1e-6, 1e-6),
            (0.5, 0.2, 4.5, 1.7),
            (1.0, 0.4, 12.0, 4.0),
        ] {
            let s = scene(200, sigma, 7);
            let e = essential_8pt(&s.x1, &s.x2).unwrap();
            let rp = recover_pose(&e, &s.x1, &s.x2).unwrap();
            let (r, t) = s.rel();
            let rot_err = rotation_angle_between(&rp.rotation, &r).to_degrees();
            let dir_err = rp.translation.angle(&t.normalize()).to_degrees();
            let front = rp.in_front.iter().filter(|&&b| b).count() as f64 / s.x1.len() as f64;
            // 정답 기선 길이로 스케일을 맞춰 삼각측량 오차를 잰다.
            let tt = rp.translation * t.norm();
            let mut errs: Vec<f64> = (0..s.x1.len())
                .filter_map(|i| {
                    let x = triangulate(&rp.rotation, &tt, &s.x1[i], &s.x2[i])?;
                    Some((x.coords - s.c1.pose.transform(&s.pts[i])).norm())
                })
                .collect();
            errs.sort_by(|a, b| a.total_cmp(b));
            let med = errs[errs.len() / 2];
            eprintln!(
                "sigma={sigma} rot_err={rot_err:.4} deg dir_err={dir_err:.3} deg \
                 front={front:.3} median_pt_err={med:.3} m"
            );
            assert!(rot_err < max_rot, "회전 오차 {rot_err} 도");
            assert!(dir_err < max_dir, "이동 방향 오차 {dir_err} 도");
            assert!(front > 0.97, "앞쪽 비율 {front}");
            assert!(med < max_pt, "삼각측량 중앙값 {med} m");
        }
    }

    #[test]
    fn essential_from_true_fundamental_matches_pose() {
        let s = scene(10, 0.0, 3);
        let f = crate::matching::fundamental_from_cameras(&s.c1, &s.c2);
        let e = essential_from_fundamental(&f, &s.c1.intrinsics, &s.c2.intrinsics);
        let (r, t) = s.rel();
        let g = project_to_essential(&essential_from_pose(&r, &t));
        let d = (e - g).norm().min((e + g).norm());
        assert!(d < 1e-9, "E 차이 {d}");
    }

    #[test]
    fn refinement_reduces_pose_error() {
        // (σ px, 정밀화 후 회전 상한 도, 이동 방향 상한 도). 상한은 시드 세 개 측정 최댓값의 약 1.5 배.
        for (sigma, max_rot, max_dir) in [(0.5, 0.18, 1.3), (1.0, 0.35, 2.9)] {
            let mut worst = (0.0f64, 0.0f64);
            for seed in [7u64, 11, 23] {
                let s = scene(200, sigma, seed);
                let e = essential_8pt(&s.x1, &s.x2).unwrap();
                let rp = recover_pose(&e, &s.x1, &s.x2).unwrap();
                let (r, t) = s.rel();
                let (rr, tr) = refine_pose(&rp.rotation, &rp.translation, &s.x1, &s.x2, 50);
                let errs = |rot: &Rotation3<f64>, tv: &Vector3<f64>| {
                    (
                        rotation_angle_between(rot, &r).to_degrees(),
                        tv.angle(&t.normalize()).to_degrees(),
                    )
                };
                let (b, a) = (errs(&rp.rotation, &rp.translation), errs(&rr, &tr));
                eprintln!(
                    "refine sigma={sigma} seed={seed} rot {:.4}->{:.4} deg dir {:.3}->{:.3} deg",
                    b.0, a.0, b.1, a.1
                );
                assert!(a.0 < b.0 && a.1 < b.1, "정밀화가 오차를 줄이지 못함");
                worst = (worst.0.max(a.0), worst.1.max(a.1));
            }
            assert!(worst.0 < max_rot, "정밀화 후 회전 오차 {} 도", worst.0);
            assert!(worst.1 < max_dir, "정밀화 후 이동 방향 오차 {} 도", worst.1);
        }
    }

    #[test]
    fn pose_from_rendered_drone_views() {
        // 합성 장면 렌더 → 특징점 → 비율 매칭 → RANSAC F → E → 자세 복원·정밀화 → 정답 비교.
        use crate::features::{detect_and_describe, DetectorConfig, Feature, GrayImage};
        use crate::matching::{ransac_fundamental, ratio_match, RansacConfig};
        use crate::synth::{CamId, Scene, SceneConfig};
        let (w, h) = (480usize, 270usize);
        let scene = Scene::new(SceneConfig {
            width: w as u32,
            height: h as u32,
            ..SceneConfig::default()
        });
        let view = |pos: usize| {
            scene
                .views
                .iter()
                .find(|v| v.cam == CamId::F && v.position == pos)
                .unwrap()
        };
        let cfg = DetectorConfig::default();
        // (위치 간격, 회전 상한 도, 이동 방향 상한 도). 측정값(0.025~0.028°, 0.15~0.18°)의 약 2 배.
        for (step, max_rot, max_dir) in [(1usize, 0.06, 0.4), (3, 0.06, 0.4)] {
            let (va, vb) = (view(0), view(step));
            let (ia, _) = scene.render(va);
            let (ib, _) = scene.render(vb);
            let fa = detect_and_describe(&GrayImage::from_rgb(w, h, &ia.data), &cfg);
            let fb = detect_and_describe(&GrayImage::from_rgb(w, h, &ib.data), &cfg);
            let m = ratio_match(&fa, &fb, 0.8, true);
            let px = |f: &Feature| Vector2::new(f.kp.x as f64 + 0.5, f.kp.y as f64 + 0.5);
            let x1: Vec<_> = m.iter().map(|&(i, _)| px(&fa[i])).collect();
            let x2: Vec<_> = m.iter().map(|&(_, j)| px(&fb[j])).collect();
            let (f, inl) = ransac_fundamental(&x1, &x2, &RansacConfig::default()).unwrap();
            let (ka, kb) = (&va.camera.intrinsics, &vb.camera.intrinsics);
            let sel = |x: &[Vector2<f64>], k: &Intrinsics| -> Vec<Vector2<f64>> {
                x.iter()
                    .zip(&inl)
                    .filter(|(_, &ok)| ok)
                    .map(|(p, _)| k.to_normalized(p))
                    .collect()
            };
            let (n1, n2) = (sel(&x1, ka), sel(&x2, kb));
            let e = essential_from_fundamental(&f, ka, kb);
            let rp = recover_pose(&e, &n1, &n2).unwrap();
            let (rr, tr) = refine_pose(&rp.rotation, &rp.translation, &n1, &n2, 50);
            let r = vb.camera.pose.rotation * va.camera.pose.rotation.inverse();
            let t = vb.camera.pose.translation - r * va.camera.pose.translation;
            let rot_err = rotation_angle_between(&rr, &r).to_degrees();
            let dir_err = tr.angle(&t.normalize()).to_degrees();
            let front = rp.in_front.iter().filter(|&&b| b).count() as f64 / n1.len() as f64;
            eprintln!(
                "render step={step} inliers={} rot_err={rot_err:.4} deg dir_err={dir_err:.3} deg front={front:.3}",
                n1.len()
            );
            assert!(rot_err < max_rot, "회전 오차 {rot_err} 도");
            assert!(dir_err < max_dir, "이동 방향 오차 {dir_err} 도");
            assert!(front > 0.95, "앞쪽 비율 {front}");
        }
    }

    #[test]
    fn five_point_contains_true_essential() {
        let mut worst: f64 = 0.0;
        for seed in [1u64, 2, 3, 4, 5, 6, 7, 8] {
            let s = scene(5, 0.0, seed);
            let (r, t) = s.rel();
            let g = essential_from_pose(&r, &t);
            let g = g / g.norm();
            let sols = essential_5pt(&s.x1, &s.x2);
            assert!(
                !sols.is_empty() && sols.len() <= 10,
                "해 개수 {}",
                sols.len()
            );
            let best = sols
                .iter()
                .map(|e| (e - g).norm().min((e + g).norm()))
                .fold(f64::INFINITY, f64::min);
            eprintln!("5pt seed={seed} sols={} best_diff={best:.2e}", sols.len());
            worst = worst.max(best);
        }
        assert!(worst < 1e-6, "정답 E 와 최소 차이 {worst}");
    }

    #[test]
    fn recover_pose_rejects_degenerate_inputs() {
        let s = scene(60, 0.5, 4);
        let (r, t) = s.rel();
        let e = essential_from_pose(&r, &t);
        // 정상 장면은 그대로 복원된다(기준선).
        assert!(recover_pose(&e, &s.x1, &s.x2).is_some());
        // 길이 불일치·빈 입력
        assert!(recover_pose(&e, &s.x1, &s.x2[..30]).is_none());
        assert!(recover_pose(&e, &[], &[]).is_none());
        // NaN 좌표·NaN 행렬
        let mut y = s.x2.clone();
        y[5].y = f64::NAN;
        assert!(recover_pose(&e, &s.x1, &y).is_none());
        assert!(recover_pose(&(e * f64::NAN), &s.x1, &s.x2).is_none());
        let mut z = s.x1[..5].to_vec();
        z[2].x = f64::NAN;
        assert!(essential_5pt(&z, &s.x2[..5]).is_empty());
        // 점 부족
        assert!(essential_5pt(&s.x1[..4], &s.x2[..4]).is_empty());
        assert!(essential_8pt(&s.x1[..7], &s.x2[..7]).is_none());
    }

    #[test]
    fn pure_rotation_gives_no_translation() {
        // 같은 중심에서 회전만 한 두 카메라: 이동 방향을 관측할 수 없다.
        for (sigma, seed) in [(0.0, 1u64), (0.5, 2), (1.0, 3)] {
            let k = Intrinsics::from_hfov(960, 540, 70f64.to_radians());
            let r1 = Rotation3::from_euler_angles(0.04, -0.03, 0.2);
            let r2 = Rotation3::from_euler_angles(-0.05, 0.06, 0.28);
            let center = Point3::new(0.0, 0.0, -40.0);
            let c1 = Camera {
                intrinsics: k,
                pose: Pose::from_center(r1, &center),
            };
            let c2 = Camera {
                intrinsics: k,
                pose: Pose::from_center(r2, &center),
            };
            let mut rng = Lcg(seed);
            let (mut x1, mut x2) = (vec![], vec![]);
            while x1.len() < 80 {
                let x = Point3::new(
                    (rng.next() - 0.5) * 40.0,
                    (rng.next() - 0.5) * 24.0,
                    (rng.next() - 0.5) * 10.0,
                );
                if let (Some(p), Some(q)) = (c1.project(&x), c2.project(&x)) {
                    let mut nz = || Vector2::new(rng.gauss(), rng.gauss()) * sigma;
                    let (a, b) = (p + nz(), q + nz());
                    x1.push(k.to_normalized(&a));
                    x2.push(k.to_normalized(&b));
                }
            }
            let r = r2 * r1.inverse();
            // 임의 방향 t 로 만든 E 와 대응에서 맞춘 E 모두 이동 방향을 확정하면 안 된다.
            for t in [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.3, -0.8, 0.5)] {
                let e = essential_from_pose(&r, &t.normalize());
                assert!(
                    recover_pose(&e, &x1, &x2).is_none(),
                    "σ={sigma}: 순수 회전에서 t 확정"
                );
            }
            if let Some(e) = essential_8pt(&x1, &x2) {
                assert!(
                    recover_pose(&e, &x1, &x2).is_none(),
                    "σ={sigma}: 맞춘 E 로 순수 회전에서 t 확정"
                );
            }
        }
    }
}
