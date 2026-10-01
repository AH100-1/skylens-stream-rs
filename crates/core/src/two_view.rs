//! 두 시점 기하: 본질 행렬, 상대 자세 복원, 삼각측량.
//!
//! 규약: 정규화 좌표 n = K⁻¹ [u v 1]ᵀ, 카메라 좌표 x_c = R X + t.
//! 첫 카메라를 [I | 0], 둘째를 [R | t] 로 둘 때 n2ᵀ E n1 = 0, E = [t]× R.

use crate::camera::Intrinsics;
use crate::matching::{
    adaptive_iterations, all_finite, fundamental_8pt, sampson_error, RansacConfig,
};
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
    /// 이동 방향을 관측할 수 있는지. false 이면 대응이 회전 하나로 설명되어
    /// (순수 회전 또는 기선이 잡음에 묻힘) `rotation` 만 믿을 수 있고
    /// `translation` 은 0 벡터, `in_front` 는 모두 false 다.
    pub translation_observable: bool,
}

/// 순수 회전 판정 배수: 회전만으로 설명한 각 잔차 중앙값이
/// 에피폴라 잔차 중앙값의 이 배수 이하이면 이동 방향을 관측할 수 없다고 본다.
const ROTATION_ONLY_FACTOR: f64 = 3.0;

/// 이동 방향 미정일 때 E 분해 회전을 프로크루스테스 회전 대신 쓰는 최대 차이(5°).
const ROTATION_SNAP_RAD: f64 = 5.0 * std::f64::consts::PI / 180.0;

/// 대응을 회전 하나로 설명했을 때의 회전과 각 잔차(rad) 중앙값.
/// 단위 광선 u1, u2 에 대해 Σ‖u2 − R u1‖² 를 최소화하는 R(직교 프로크루스테스) 을 쓴다.
fn rotation_only_fit(n1: &[Vector2<f64>], n2: &[Vector2<f64>]) -> (Rotation3<f64>, f64) {
    let ray = |n: &Vector2<f64>| Vector3::new(n.x, n.y, 1.0).normalize();
    let h = n1.iter().zip(n2).fold(Matrix3::zeros(), |h, (a, b)| {
        h + ray(a) * ray(b).transpose()
    });
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let d = (vt.transpose() * u.transpose()).determinant().signum();
    let r = vt.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose();
    let res = median(
        n1.iter()
            .zip(n2)
            .map(|(a, b)| angle_between(&(r * ray(a)), &ray(b)))
            .collect(),
    );
    (Rotation3::from_matrix_unchecked(r), res)
}

/// 두 벡터 사이 각(rad). atan2 를 써서 반올림으로 |sin| > 1 이 되어도 NaN 이 나지 않는다.
fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// E 의 네 후보 중 두 카메라 앞(양의 깊이)에 놓이는 점이 가장 많은 것을 고른다.
/// 길이가 다르거나 유한하지 않은 값이 있으면 None.
/// 대응이 회전 하나로 설명되면(이동 방향 미정) `translation_observable = false` 로
/// 회전만 돌려준다.
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
    let (r_only, rot_res) = rotation_only_fit(n1, n2);
    let observable = rot_res > ROTATION_ONLY_FACTOR * epi + 1e-12;
    if !observable {
        // 프로크루스테스 회전은 기선이 만든 평균 시차까지 회전으로 흡수해 기선/깊이 비만큼
        // 치우친다. E 분해의 회전 후보 중 그것과 가까운 것이 있으면 그쪽을 쓴다.
        // 선형 E 는 짧은 기선에서 회전도 흔들리므로 Sampson LM 으로 다듬은 회전을 쓴다.
        // 네 후보와 (프로크루스테스 회전, 각 축 이동) 에서 시작해 Sampson 비용이 가장 작은 해를 쓴다.
        let cost = |r: &Rotation3<f64>, t: &Vector3<f64>| -> f64 {
            let e = essential_from_pose(r, t);
            n1.iter()
                .zip(n2)
                .map(|(a, b)| sampson_residual(&e, a, b).powi(2))
                .sum()
        };
        let starts = decompose_essential(e).into_iter().chain(
            [Vector3::x(), Vector3::y(), Vector3::z()]
                .into_iter()
                .map(|t| (r_only, t)),
        );
        let rotation = starts
            .map(|(r, t)| refine_pose(&r, &t, n1, n2, 100))
            .filter(|(r, _)| r.angle_to(&r_only) < ROTATION_SNAP_RAD)
            .min_by(|a, b| cost(&a.0, &a.1).total_cmp(&cost(&b.0, &b.1)))
            .map(|(r, _)| r)
            .unwrap_or(r_only);
        return Some(RelativePose {
            rotation,
            translation: Vector3::zeros(),
            in_front: vec![false; n1.len()],
            translation_observable: false,
        });
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
                translation_observable: true,
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
    // M0 + zM1 + z²M2 + z³M3 = 0 의 실근 z. 먼저 μ = 1/z 형(μ³M0 + μ²M1 + μM2 + M3, 동반 행렬 30×30),
    // M0 가 특이하거나 Schur 반복이 상한 안에 수렴하지 않으면 z 형(M3 을 선행 계수로)으로 다시 푼다.
    // 반복 상한이 있는 Schur 분해라 수렴하지 않는 입력에서도 무한히 돌지 않는다.
    let roots = |lead: usize, rest: [usize; 3]| -> Option<Vec<f64>> {
        let inv = m[lead].try_inverse()?;
        let mut comp = nalgebra::DMatrix::<f64>::zeros(30, 30);
        for (k, &c) in rest.iter().enumerate() {
            comp.view_mut((0, 10 * k), (10, 10))
                .copy_from(&(-(inv * m[c])));
        }
        for k in 0..20 {
            comp[(10 + k, k)] = 1.0;
        }
        let schur = nalgebra::Schur::try_new(comp, 1e-14, 3000)?;
        Some(
            schur
                .complex_eigenvalues()
                .iter()
                .filter(|mu| mu.im.abs() <= 1e-6 * mu.norm().max(1e-12))
                .map(|mu| mu.re)
                .collect(),
        )
    };
    let zs: Vec<f64> = match roots(0, [1, 2, 3]) {
        Some(mus) => mus
            .into_iter()
            .filter(|mu| mu.abs() >= 1e-10)
            .map(|mu| 1.0 / mu)
            .collect(),
        None => match roots(3, [2, 1, 0]) {
            Some(zs) => zs,
            None => return vec![],
        },
    };
    let mut out = vec![];
    for z in zs {
        let mz = m[0] + m[1] * z + m[2] * (z * z) + m[3] * (z * z * z);
        let svd = mz.svd(false, true);
        let vt = svd.v_t.unwrap();
        let v = vt.row(svd.singular_values.imin());
        if v[9].abs() < 1e-12 {
            continue;
        }
        let (x, y, z) = polish_essential(&basis, v[7] / v[9], v[8] / v[9], z);
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

/// 본질 행렬 제약 잔차(det E, 2EEᵀE − tr(EEᵀ)E 의 9 성분), E = xB0 + yB1 + zB2 + B3.
fn essential_constraints(b: &[SMatrix<f64, 9, 1>], p: &Vector3<f64>) -> SMatrix<f64, 10, 1> {
    let v = b[0] * p.x + b[1] * p.y + b[2] * p.z + b[3];
    let e = Matrix3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    let eet = e * e.transpose();
    let c = 2.0 * eet * e - eet.trace() * e;
    let mut r = SMatrix::<f64, 10, 1>::zeros();
    r[0] = e.determinant();
    for k in 0..9 {
        r[k + 1] = c[(k / 3, k % 3)];
    }
    r
}

/// 동반 행렬 고윳값에서 얻은 (x, y, z) 를 제약 잔차의 가우스–뉴턴으로 다듬는다(중앙 차분 야코비안).
/// 잔차가 줄지 않으면 그 단계는 버린다. 고윳값 분해의 반올림 오차(1e-6 수준)를 없앤다.
fn polish_essential(b: &[SMatrix<f64, 9, 1>], x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let mut p = Vector3::new(x, y, z);
    let mut r = essential_constraints(b, &p);
    for _ in 0..5 {
        let mut j = SMatrix::<f64, 10, 3>::zeros();
        for k in 0..3 {
            let h = 1e-7 * p[k].abs().max(1.0);
            let mut d = Vector3::zeros();
            d[k] = h;
            let col = (essential_constraints(b, &(p + d)) - essential_constraints(b, &(p - d)))
                / (2.0 * h);
            j.set_column(k, &col);
        }
        let Some(step) = (j.transpose() * j)
            .try_inverse()
            .map(|m| m * j.transpose() * r)
        else {
            break;
        };
        let q = p - step;
        let rq = essential_constraints(b, &q);
        if rq.norm().is_nan() || rq.norm() >= r.norm() {
            break;
        }
        (p, r) = (q, rq);
    }
    (p.x, p.y, p.z)
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

/// 본질 행렬 E 에서 시작해 상대 자세를 Sampson 비용 최소화로 구한다.
///
/// 선형 E 의 분해 해 하나만 다듬으면 잡음 큰 짝에서 정답과 다른 골짜기(비용이 정답보다 10배 큰
/// 국소 최소, 회전–이동 혼동)에 갇힐 수 있다. 그래서 분해 네 후보와 그 이동 부호를 뒤집은 네 후보,
/// (선형 회전, 좌표축 이동 셋), (프로크루스테스 회전, 좌표축 이동 셋), 모두 열네 곳에서 `refine_pose` 를
/// 돌려 비용이 가장 작은 해를 고른다. 프로크루스테스 시작점이 갇힌 경우를 꺼낸다.
/// Sampson 비용은 t 부호와 꼬인 짝에 무관하므로
/// 마지막에 정밀화된 E 를 `recover_pose` 로 다시 분해해 키랄리티(앞쪽 점 수)로 하나를 고르고,
/// 이동 관측 가능 여부도 그 E 로 판정한다.
pub fn refine_relative_pose(
    e: &Matrix3<f64>,
    n1: &[Vector2<f64>],
    n2: &[Vector2<f64>],
    iters: usize,
) -> Option<RelativePose> {
    // 입력 검사와 관측 불가 시 회전 초깃값(축 시작점)에 쓴다.
    let linear = recover_pose(e, n1, n2)?;
    let cost = |r: &Rotation3<f64>, t: &Vector3<f64>| -> f64 {
        let e = essential_from_pose(r, t);
        n1.iter()
            .zip(n2)
            .map(|(a, b)| sampson_residual(&e, a, b).powi(2))
            .sum()
    };
    let axes = [Vector3::x(), Vector3::y(), Vector3::z()];
    let (r, t) = decompose_essential(e)
        .into_iter()
        .flat_map(|(r, t)| [(r, t), (r, -t)])
        .chain(axes.into_iter().map(|t| (linear.rotation, t)))
        .chain(axes.into_iter().map(|t| (rotation_only_fit(n1, n2).0, t)))
        .map(|(r, t)| refine_pose(&r, &t, n1, n2, iters))
        .min_by(|a, b| cost(&a.0, &a.1).total_cmp(&cost(&b.0, &b.1)))?;
    // 관측 가능 여부도 정밀화된 E 로 다시 판정한다. 선형 E 가 틀린 골짜기에 있으면 에피폴라 잔차가
    // 부풀어 관측 가능한 짝을 순수 회전으로 오판하기 때문이다(기선/깊이 0.1, σ1 px 시드 5·6·10).
    recover_pose(&essential_from_pose(&r, &t), n1, n2)
}

/// 보정된 카메라 짝의 기하 검증: RANSAC(Fischler & Bolles 1981) + 5점 본질 행렬 최소 해(Nistér 2004).
///
/// 8점 F 는 장면이 평면에 가까우면 퇴화하지만 5점 해는 평면에서도 유효하다.
/// `n1`·`n2` 는 정규화 좌표, `focal_px` 는 픽셀 문턱 `cfg.threshold_px` 를 정규화 단위로 바꾸는 초점 거리(px).
/// 반환: [`ransac_essential_candidates`] 의 첫 후보(정상 수가 가장 많고 같으면 Sampson 비용이 가장 작은 해).
/// 장면이 평면이면 두 번째 후보가 같은 정도로 대응을 설명할 수 있다(평면 두 겹 모호성).
pub fn ransac_essential(
    n1: &[Vector2<f64>],
    n2: &[Vector2<f64>],
    focal_px: f64,
    cfg: &RansacConfig,
) -> Option<(Matrix3<f64>, Vec<bool>)> {
    ransac_essential_candidates(n1, n2, focal_px, cfg)
        .into_iter()
        .next()
}

/// 정규화 좌표 대응 4개 이상에서 직접 선형 변환(DLT)으로 호모그래피 H(b ≃ H a)를 구한다.
/// AᵀA(9×9)의 최소 고유벡터를 쓴다. 대응이 4개 미만이거나 결과가 유한하지 않으면 None.
pub fn homography_dlt(a: &[Vector2<f64>], b: &[Vector2<f64>]) -> Option<Matrix3<f64>> {
    if a.len() < 4 || a.len() != b.len() {
        return None;
    }
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for (p, q) in a.iter().zip(b) {
        let (x, y, u, v) = (p.x, p.y, q.x, q.y);
        let r1 = SMatrix::<f64, 9, 1>::from_column_slice(&[
            -x,
            -y,
            -1.0,
            0.0,
            0.0,
            0.0,
            u * x,
            u * y,
            u,
        ]);
        let r2 = SMatrix::<f64, 9, 1>::from_column_slice(&[
            0.0,
            0.0,
            0.0,
            -x,
            -y,
            -1.0,
            v * x,
            v * y,
            v,
        ]);
        ata += r1 * r1.transpose() + r2 * r2.transpose();
    }
    let eig = ata.symmetric_eigen();
    let h = eig.eigenvectors.column(eig.eigenvalues.imin());
    let m = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    m.iter().all(|v| v.is_finite()).then_some(m)
}

/// 호모그래피 전달 오차 ‖b − H a‖(정규화 단위). 무한원으로 가면 무한대.
fn homography_transfer(h: &Matrix3<f64>, a: &Vector2<f64>, b: &Vector2<f64>) -> f64 {
    let p = h * Vector3::new(a.x, a.y, 1.0);
    if p.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    (Vector2::new(p.x / p.z, p.y / p.z) - b).norm()
}

/// 표시된 정상 짝 가운데 한 평면(호모그래피)으로 설명되는 짝을 4점 RANSAC 으로 찾는다.
/// 문턱 `th`(정규화 단위)는 한쪽 전달 오차 기준이다. 반환: 호모그래피 정상 표시(정상 짝이 4개 미만이면 None).
fn planar_inliers(
    n1: &[Vector2<f64>],
    n2: &[Vector2<f64>],
    inl: &[bool],
    th: f64,
    seed: u64,
) -> Option<Vec<bool>> {
    let idx: Vec<usize> = (0..inl.len()).filter(|&i| inl[i]).collect();
    if idx.len() < 4 {
        return None;
    }
    let mut st = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut rnd = |m: usize| {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 33) as usize) % m
    };
    let mark = |h: &Matrix3<f64>| -> Vec<bool> {
        (0..inl.len())
            .map(|i| inl[i] && homography_transfer(h, &n1[i], &n2[i]) < th)
            .collect()
    };
    let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
    let mut best: Option<Vec<bool>> = None;
    for _ in 0..PLANAR_RANSAC_ITERS {
        let mut pick = [0usize; 4];
        let mut k = 0;
        while k < 4 {
            let c = idx[rnd(idx.len())];
            if !pick[..k].contains(&c) {
                pick[k] = c;
                k += 1;
            }
        }
        let a: Vec<_> = pick.iter().map(|&i| n1[i]).collect();
        let b: Vec<_> = pick.iter().map(|&i| n2[i]).collect();
        let Some(h) = homography_dlt(&a, &b) else {
            continue;
        };
        let m = mark(&h);
        if best.as_ref().is_none_or(|bm| count(&m) > count(bm)) {
            best = Some(m);
        }
    }
    // 최소 표본 해를 정상 짝 전체로 다시 맞추고, 다시 맞춘 H 로는 모든 대응을 판정한다
    // (에피폴라 문턱 밖으로 밀려난 정상 짝도 평면 위에 있으면 되찾는다).
    let m = best?;
    let a: Vec<_> = (0..inl.len()).filter(|&i| m[i]).map(|i| n1[i]).collect();
    let b: Vec<_> = (0..inl.len()).filter(|&i| m[i]).map(|i| n2[i]).collect();
    Some(homography_dlt(&a, &b).map_or(m, |h| {
        (0..inl.len())
            .map(|i| homography_transfer(&h, &n1[i], &n2[i]) < th)
            .collect()
    }))
}

/// 5점 RANSAC 으로 서로 다른 본질 행렬 후보를 최대 `ESSENTIAL_CANDIDATES` 개 돌려준다.
/// 표본 단계에서는 최고 정상 수의 0.7 배 이상인 가설을 최대 `ESSENTIAL_CANDIDATES + 6` 개 보관한다
/// (잡음 섞인 최소 표본에서는 정답 골짜기의 가설이 정상 수로 뒤처질 수 있다).
///
/// 최소 표본의 후보 E 마다 Sampson 거리로 정상 수를 세고, 정규화한 E 끼리 거리(부호 무관 프로베니우스)가
/// 0.1 보다 먼 가설만 따로 보관한다. 평면 장면에서는 정답과 그 쌍둥이 해(이동이 평면 법선 쪽인 해)가
/// 모든 대응을 똑같이 설명하므로 하나만 남기면 절반 확률로 쌍둥이를 고른다.
/// 각 후보는 정상 짝으로 키랄리티 분해 → `refine_pose` 정밀화 → 정상 집합 갱신을 두 번 한다.
/// 정상 짝의 `PLANAR_MIN_SHARE` 이상이 한 호모그래피로 설명되면(평면 장면) 정상 집합을 그 호모그래피의
/// 정상 짝(전달 오차 < `PLANAR_TRANSFER_FACTOR` × 문턱)으로 바꾸고 한 번 더 정밀화한다. 그 뒤
/// 정상 수 내림차순(같으면 Sampson 비용 오름차순)으로 정렬한다. 최고 정상 수의 0.9 배 미만 후보는 버린다.
/// 정상 짝이 5개 미만이거나 정상 비율이 `cfg.min_inlier_ratio` 미만, 길이가 다르거나
/// 유한하지 않은 좌표가 있으면 빈 목록.
pub fn ransac_essential_candidates(
    n1: &[Vector2<f64>],
    n2: &[Vector2<f64>],
    focal_px: f64,
    cfg: &RansacConfig,
) -> Vec<(Matrix3<f64>, Vec<bool>)> {
    let n = n1.len();
    if n < 5
        || n != n2.len()
        || !all_finite(n1)
        || !all_finite(n2)
        || focal_px.is_nan()
        || focal_px <= 0.0
    {
        return vec![];
    }
    let th = cfg.threshold_px / focal_px;
    let inliers_of = |e: &Matrix3<f64>| -> Vec<bool> {
        (0..n)
            .map(|i| sampson_residual(e, &n1[i], &n2[i]).abs() < th)
            .collect()
    };
    let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
    let distinct = |a: &Matrix3<f64>, b: &Matrix3<f64>| {
        let (a, b) = (a / a.norm(), b / b.norm());
        (a - b).norm().min((a + b).norm()) > 0.1
    };
    let mut st = cfg.seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut rnd = |m: usize| {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 33) as usize) % m
    };
    let mut pool: Vec<(Matrix3<f64>, usize)> = Vec::new();
    let mut iters = cfg.max_iters;
    let mut it = 0;
    while it < iters {
        it += 1;
        let mut idx = [0usize; 5];
        let mut k = 0;
        while k < 5 {
            let c = rnd(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let s1: Vec<_> = idx.iter().map(|&i| n1[i]).collect();
        let s2: Vec<_> = idx.iter().map(|&i| n2[i]).collect();
        for e in essential_5pt(&s1, &s2) {
            let cnt = count(&inliers_of(&e));
            let best = pool.first().map_or(0, |p| p.1);
            if cnt < 5 || 10 * cnt < 7 * best {
                continue;
            }
            match pool.iter().position(|(p, _)| !distinct(p, &e)) {
                Some(j) if pool[j].1 >= cnt => {}
                Some(j) => pool[j] = (e, cnt),
                None => pool.push((e, cnt)),
            }
            pool.sort_by_key(|p| std::cmp::Reverse(p.1));
            pool.truncate(ESSENTIAL_CANDIDATES + 6);
            if cnt > best {
                // 쌍둥이 해도 표본에 나오도록 최소 반복을 넉넉히 둔다.
                iters =
                    adaptive_iterations(cnt as f64 / n as f64, 5, cfg.confidence, cfg.max_iters)
                        .max(ESSENTIAL_MIN_ITERS.min(cfg.max_iters));
            }
        }
    }
    let cost = |e: &Matrix3<f64>, inl: &[bool]| -> f64 {
        (0..n)
            .filter(|&i| inl[i])
            .map(|i| sampson_residual(e, &n1[i], &n2[i]).powi(2))
            .sum()
    };
    let mut out: Vec<(Matrix3<f64>, Vec<bool>, usize, f64)> = Vec::new();
    for (start, _) in pool {
        let (mut e, mut inl) = (start, inliers_of(&start));
        for _ in 0..2 {
            let s1: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| n1[i]).collect();
            let s2: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| n2[i]).collect();
            // 후보 자신의 골짜기 안에서만 다듬는다(다중 시작은 쌍둥이 둘을 한 해로 합쳐 버린다).
            let Some(pose) = recover_pose(&e, &s1, &s2) else {
                break;
            };
            if !pose.translation_observable {
                break;
            }
            let (r, t) = refine_pose(&pose.rotation, &pose.translation, &s1, &s2, 30);
            let g = essential_from_pose(&r, &t);
            let gi = inliers_of(&g);
            if count(&gi) < count(&inl) {
                break;
            }
            (e, inl) = (g, gi);
        }
        // 평면 장면: 정상 짝 대부분이 한 호모그래피로 설명되면 그 호모그래피에서 벗어난 짝을 뺀다.
        // 에피폴라 문턱(1차원 제약) 안에 우연히 든 이상치가 평면의 얕은 골짜기를 크게 기울이기 때문이다.
        if let Some(hm) = planar_inliers(n1, n2, &inl, PLANAR_TRANSFER_FACTOR * th, cfg.seed) {
            let (hc, ec) = (count(&hm), count(&inl));
            if hm != inl && hc as f64 >= PLANAR_MIN_SHARE * ec as f64 {
                let s1: Vec<_> = (0..n).filter(|&i| hm[i]).map(|i| n1[i]).collect();
                let s2: Vec<_> = (0..n).filter(|&i| hm[i]).map(|i| n2[i]).collect();
                if let Some(pose) = recover_pose(&e, &s1, &s2).filter(|p| p.translation_observable)
                {
                    let (r, t) = refine_pose(&pose.rotation, &pose.translation, &s1, &s2, 30);
                    (e, inl) = (essential_from_pose(&r, &t), hm);
                }
            }
        }
        // 정밀화 뒤 같은 골짜기로 모인 후보는 하나만 남긴다.
        if out.iter().all(|o| distinct(&o.0, &e)) {
            let (c, k) = (count(&inl), cost(&e, &inl));
            out.push((e, inl, c, k));
        }
    }
    out.sort_by(|a, b| b.2.cmp(&a.2).then(a.3.total_cmp(&b.3)));
    let top = out.first().map_or(0, |o| o.2);
    out.into_iter()
        .filter(|o| {
            o.2 >= 5 && 10 * o.2 >= 9 * top && o.2 as f64 >= cfg.min_inlier_ratio * n as f64
        })
        .take(ESSENTIAL_CANDIDATES)
        .map(|o| (o.0, o.1))
        .collect()
}

/// [`ransac_essential_candidates`] 가 돌려주는 최대 후보 수.
pub const ESSENTIAL_CANDIDATES: usize = 4;

/// 평면 판정용 호모그래피 4점 RANSAC 반복 수.
pub const PLANAR_RANSAC_ITERS: usize = 200;

/// 호모그래피 전달 오차 문턱 = 이 배수 × 에피폴라 문턱. 전달 오차는 두 영상 잡음이 2차원으로 더해져
/// 에피폴라 거리보다 크다. 띠를 넓혀도 무작위 이상치가 한 점 둘레 원에 들 확률은 매우 작다(1920×1080 에서
/// 반지름 3.75 px 원 ≈ 2e-5)이므로 정상 짝을 놓치지 않는 쪽으로 넉넉히 둔다.
pub const PLANAR_TRANSFER_FACTOR: f64 = 2.5;

/// 에피폴라 정상 짝 중 이 비율 이상이 한 호모그래피로 설명되면 평면 장면으로 보고 거른다.
pub const PLANAR_MIN_SHARE: f64 = 0.8;

/// 5점 RANSAC 의 최소 반복 수.
pub const ESSENTIAL_MIN_ITERS: usize = 300;

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
        scene_with_baseline(n, sigma_px, seed, 3.0)
    }

    /// 기선 길이를 b(m)로 바꾼 같은 장면(둘째 중심 = (b, 0.8b/3, −40 − b/6)).
    fn scene_with_baseline(n: usize, sigma_px: f64, seed: u64, b: f64) -> Scene {
        scene_full(n, sigma_px, seed, b, 10.0)
    }

    /// 점 높이 범위를 h(m)로 바꾼 같은 장면(h = 0 이면 평면).
    fn scene_full(n: usize, sigma_px: f64, seed: u64, b: f64, h: f64) -> Scene {
        let k = Intrinsics::from_hfov(960, 540, 70f64.to_radians());
        let r1 = Rotation3::from_euler_angles(0.04, -0.03, 0.2);
        let r2 = Rotation3::from_euler_angles(-0.05, 0.06, 0.28);
        let c1 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r1, &Point3::new(0.0, 0.0, -40.0)),
        };
        let c2 = Camera {
            intrinsics: k,
            pose: Pose::from_center(r2, &Point3::new(b, 0.8 * b / 3.0, -40.0 - b / 6.0)),
        };
        let mut rng = Lcg(seed);
        let (mut x1, mut x2, mut pts) = (vec![], vec![], vec![]);
        while x1.len() < n {
            let x = Point3::new(
                (rng.next() - 0.5) * 40.0,
                (rng.next() - 0.5) * 24.0,
                (rng.next() - 0.5) * h,
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
    fn relative_pose_without_noise_is_exact() {
        let s = scene(200, 0.0, 7);
        let e = essential_8pt(&s.x1, &s.x2).unwrap();
        let rp = recover_pose(&e, &s.x1, &s.x2).unwrap();
        let (r, t) = s.rel();
        let rot_err = rotation_angle_between(&rp.rotation, &r).to_degrees();
        let dir_err = rp.translation.angle(&t.normalize()).to_degrees();
        let front = rp.in_front.iter().filter(|&&b| b).count();
        // 정답 기선 길이로 스케일을 맞춘 삼각측량 오차.
        let tt = rp.translation * t.norm();
        let worst = (0..s.x1.len())
            .filter_map(|i| {
                let x = triangulate(&rp.rotation, &tt, &s.x1[i], &s.x2[i])?;
                Some((x.coords - s.c1.pose.transform(&s.pts[i])).norm())
            })
            .fold(0.0f64, f64::max);
        assert!(rot_err < 1e-6 && dir_err < 1e-6, "{rot_err} {dir_err}");
        assert_eq!(front, s.x1.len());
        assert!(worst < 1e-6, "삼각측량 오차 {worst} m");
    }

    /// 선형 8점 + `recover_pose` 의 오차 분포(시드 100개). 선형 해는 정밀화 전 시작점일 뿐이라 꼬리가
    /// 길다(시드 1..=100 σ=0.5 최악 회전 4.5°·방향 79°, 원인은 정밀화 테스트 주석). 그래서 중앙값과
    /// 90% 분위만 단언한다. 상한은 시드 1..=100·101..=200 두 묶음 측정값 중 큰 값의 약 1.5 배.
    #[test]
    fn linear_pose_error_distribution() {
        // (σ px, 회전 중앙값·90%, 방향 중앙값·90%) 도. 측정(σ0.5): 0.125·0.222, 2.98·7.66.
        // 측정(σ1): 0.235·0.436, 5.92·11.95.
        for (sigma, rot_med, rot_p90, dir_med, dir_p90) in
            [(0.5, 0.2, 0.35, 4.5, 11.5), (1.0, 0.35, 0.65, 9.0, 18.0)]
        {
            for seeds in [1u64..=100, 101..=200] {
                let st = pose_stats(sigma, seeds.clone());
                let rot: Vec<f64> = st.lin.iter().map(|x| x.0).collect();
                let dir: Vec<f64> = st.lin.iter().map(|x| x.1).collect();
                let q = (
                    quantile(&rot, 0.5),
                    quantile(&rot, 0.9),
                    quantile(&dir, 0.5),
                    quantile(&dir, 0.9),
                );
                eprintln!(
                    "linear sigma={sigma} seeds={seeds:?} n={} rot med {:.3} p90 {:.3} dir med {:.2} p90 {:.2}",
                    rot.len(),
                    q.0,
                    q.1,
                    q.2,
                    q.3
                );
                assert!(q.0 < rot_med && q.1 < rot_p90, "회전 분위 {q:?}");
                assert!(q.2 < dir_med && q.3 < dir_p90, "방향 분위 {q:?}");
            }
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

    /// 원인(F-012): 꼬리 시드에서 정답 자세의 Sampson 비용은 선형 해를 다듬은 해보다 약 10 배 작다
    /// (σ0.5 시드 92: 정답 9.4e-5, 선형 해 정밀화 1.1e-3). 즉 데이터는 모호하지 않고, 선형 해 하나에서
    /// 시작한 LM 이 회전–이동 혼동 골짜기(회전 ~4.4°, 방향 ~75°)에 갇힌 것이다. `refine_relative_pose`
    /// 는 시작점을 넓혀 비용 최소 해를 고르므로 다음을 시드마다 단언한다:
    /// - 비용이 선형 해 이하(LM 단조 감소), 정답 자세 비용 이하(전역 최소라면 반드시 성립),
    /// - 오차 최댓값·중앙값·90% 분위가 상한 아래(상한 = 두 시드 묶음 측정 최댓값의 약 1.5 배).
    #[test]
    fn refined_pose_error_distribution() {
        // (σ px, 회전 중앙값·90%·최대, 방향 중앙값·90%·최대) 도.
        // 측정(σ0.5): 회전 0.093·0.159·0.266, 방향 0.70·1.57·2.00.
        // 측정(σ1): 회전 0.195·0.358·0.531, 방향 1.47·3.12·4.39.
        for (sigma, rb, db) in [
            (0.5, [0.14, 0.24, 0.4], [1.05, 2.4, 3.0]),
            (1.0, [0.3, 0.55, 0.8], [2.2, 4.7, 6.6]),
        ] {
            for seeds in [1u64..=100, 101..=200] {
                let st = pose_stats(sigma, seeds.clone());
                for (i, &(lin, rf, truth)) in st.costs.iter().enumerate() {
                    assert!(
                        rf <= lin * (1.0 + 1e-9),
                        "{i}: 정밀화가 비용을 늘림 {lin} → {rf}"
                    );
                    assert!(
                        rf <= truth * (1.0 + 1e-6),
                        "{i}: 정답보다 비싼 국소 최소 {rf} > {truth}"
                    );
                }
                let rot: Vec<f64> = st.refined.iter().map(|x| x.0).collect();
                let dir: Vec<f64> = st.refined.iter().map(|x| x.1).collect();
                let r = [0.5, 0.9, 1.0].map(|p| quantile(&rot, p));
                let d = [0.5, 0.9, 1.0].map(|p| quantile(&dir, p));
                eprintln!(
                    "refined sigma={sigma} seeds={seeds:?} n={} unobservable={} rot {r:.3?} dir {d:.2?}",
                    rot.len(),
                    st.unobservable
                );
                // 장면은 모두 관측 가능. 선형 E 로 판정하면 σ1 에서 26·18 개가 순수 회전으로 오판되지만
                // 정밀화된 E 로 다시 판정하면 0 개다.
                assert_eq!(st.unobservable, 0, "관측 불가로 오판");
                assert!((0..3).all(|k| r[k] < rb[k]), "회전 {r:?} 상한 {rb:?}");
                assert!((0..3).all(|k| d[k] < db[k]), "방향 {d:?} 상한 {db:?}");
            }
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
            let px = |f: &Feature| f.kp.pixel();
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

    /// F-011: 시드 1..=1000 에서 모든 호출이 끝나고, 정답 E 포함률(차이 < 1e-6) ≥ 99%.
    /// 반복 상한 없는 고윳값 분해로는 시드 114 에서 반환하지 않았다.
    ///
    /// 종료 보장은 구조로 한다: 30×30 동반 행렬의 Schur 분해는 반복 3000회 상한이고 실패하면
    /// 선행 계수를 바꿔 한 번 더(역시 상한) 푼 뒤 빈 결과를 돌려준다. 헤센베르크 QR 한 번은
    /// 약 6·30² 부동소수 연산이라 최악도 2 × 3000 × 5400 ≈ 3.2e7 연산(수십 ms)이다.
    /// 벽시계 단언은 다른 테스트와 CPU 를 나눠 쓰는 전체 실행에서도 흔들리지 않도록
    /// 병렬 없이 직렬로 재고, 최댓값은 위 구조 상한에 여유를 둔 100 ms, 중앙값은 2 ms 로 둔다
    /// (단독 실행 실측: 중앙값 약 0.16 ms, 최댓값 약 6~8 ms).
    #[test]
    fn five_point_terminates_on_many_seeds() {
        let t0 = std::time::Instant::now();
        let mut ms = Vec::with_capacity(1000);
        let mut hits = 0;
        for seed in 1..=1000u64 {
            let s = scene(5, 0.0, seed);
            let (r, t) = s.rel();
            let g = essential_from_pose(&r, &t);
            let g = g / g.norm();
            let c = std::time::Instant::now();
            let sols = essential_5pt(&s.x1, &s.x2);
            ms.push(c.elapsed().as_secs_f64() * 1e3);
            if sols
                .iter()
                .any(|e| (e - g).norm().min((e + g).norm()) < 1e-6)
            {
                hits += 1;
            }
        }
        let total = t0.elapsed().as_secs_f64();
        ms.sort_by(f64::total_cmp);
        let (median, worst) = (ms[ms.len() / 2], ms[ms.len() - 1]);
        eprintln!(
            "5pt 1000 seeds: hits={hits} median={median:.3}ms worst={worst:.2}ms total={total:.2}s"
        );
        assert!(hits >= 990, "정답 E 포함 {hits}/1000");
        assert!(median <= 2.0, "중앙값 {median} ms");
        assert!(worst <= 100.0, "최악 호출 {worst} ms");
        assert!(total <= 10.0, "전체 {total} s");
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
            // 임의 방향 t 로 만든 E 와 대응에서 맞춘 E 모두 이동 방향을 확정하면 안 되고,
            // 회전은 정답과 맞아야 한다.
            let check = |e: &Matrix3<f64>, what: &str| {
                let rp = recover_pose(e, &x1, &x2).unwrap();
                assert!(
                    !rp.translation_observable,
                    "σ={sigma}: {what} 순수 회전에서 t 확정"
                );
                assert_eq!(rp.translation, Vector3::zeros());
                assert!(rp.in_front.iter().all(|&b| !b));
                let err = rotation_angle_between(&rp.rotation, &r).to_degrees();
                assert!(err < 0.15, "σ={sigma}: {what} 회전 오차 {err}°");
            };
            for t in [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.3, -0.8, 0.5)] {
                check(&essential_from_pose(&r, &t.normalize()), "임의 t 의 E");
            }
            if let Some(e) = essential_8pt(&x1, &x2) {
                check(&e, "맞춘 E");
            }
        }
    }

    /// 기선/깊이 비 훑기(대응 200개, 8점 E): 관측 가능 판정, r = 회전 잔차/에피폴라 잔차,
    /// 회전 오차(도). 실험 노트 표의 출처.
    fn baseline_sweep(b: f64, sigma: f64, seed: u64) -> (bool, f64, f64) {
        let s = scene_with_baseline(200, sigma, seed, b);
        let (r, _) = s.rel();
        let e = essential_8pt(&s.x1, &s.x2).unwrap();
        let rp = recover_pose(&e, &s.x1, &s.x2).unwrap();
        let (_, rot_res) = rotation_only_fit(&s.x1, &s.x2);
        let epi = median(
            s.x1.iter()
                .zip(&s.x2)
                .map(|(a, b)| sampson_error(&e, a, b).sqrt())
                .collect(),
        );
        // 사용 경로대로 refine_relative_pose 의 회전을 잰다(관측 불가면 recover_pose 회전 그대로).
        let rot = refine_relative_pose(&e, &s.x1, &s.x2, 50).unwrap().rotation;
        (
            rp.translation_observable,
            rot_res / epi,
            rotation_angle_between(&rot, &r).to_degrees(),
        )
    }

    #[test]
    fn small_baseline_keeps_rotation() {
        // 기선/깊이 0.025(1 m / 40 m), σ=1 px: 이동 방향 판정과 무관하게 회전은 0.5° 이내.
        for seed in 1..=10u64 {
            let (obs, ratio, err) = baseline_sweep(1.0, 1.0, seed);
            println!("b=1 m seed={seed}: 관측={obs} r={ratio:.2} 회전 오차 {err:.3}°");
            assert!(err < 0.5, "seed {seed}: 회전 오차 {err}°");
        }
        for b in [0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0] {
            for sigma in [0.5, 1.0] {
                let rows: Vec<_> = (1..=10u64).map(|s| baseline_sweep(b, sigma, s)).collect();
                let obs = rows.iter().filter(|r| r.0).count();
                let rmin = rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
                let rmax = rows.iter().map(|r| r.1).fold(0.0, f64::max);
                let emax = rows.iter().map(|r| r.2).fold(0.0, f64::max);
                println!(
                    "b/d={:.3} σ={sigma}: 관측 {obs}/10, r {rmin:.2}~{rmax:.2}, 최대 회전 오차 {emax:.3}°",
                    b / 40.0
                );
            }
        }
    }

    #[test]
    fn angle_between_is_finite_near_right_angle() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        for eps in [0.0, 1e-17, 1e-12] {
            let b = Vector3::new(eps, 1.0, 0.0).normalize();
            let ang = angle_between(&a, &b);
            assert!((ang - std::f64::consts::FRAC_PI_2).abs() < 1e-9, "{ang}");
        }
        assert!(angle_between(&a, &a).abs() < 1e-12);
    }

    /// 백분위(가장 가까운 순위). `v` 는 비어 있지 않아야 한다.
    fn quantile(v: &[f64], q: f64) -> f64 {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    /// 시드마다 선형(8점 + `recover_pose`)·정밀화(`refine_relative_pose`) 자세의 (회전, 이동 방향) 오차(도)와
    /// (선형, 정밀화, 정답 자세) Sampson 비용. 이동이 관측 불가로 판정된 해는 오차 목록에서 빼고,
    /// 정밀화 해가 관측 불가인 시드 수를 센다(장면은 모두 관측 가능).
    struct PoseStats {
        lin: Vec<(f64, f64)>,
        refined: Vec<(f64, f64)>,
        costs: Vec<(f64, f64, f64)>,
        unobservable: usize,
    }

    fn pose_stats(sigma: f64, seeds: std::ops::RangeInclusive<u64>) -> PoseStats {
        let mut st = PoseStats {
            lin: vec![],
            refined: vec![],
            costs: vec![],
            unobservable: 0,
        };
        for seed in seeds {
            let s = scene(200, sigma, seed);
            let (r, t) = s.rel();
            let cost = |rot: &Rotation3<f64>, tv: &Vector3<f64>| {
                let e = essential_from_pose(rot, tv);
                s.x1.iter()
                    .zip(&s.x2)
                    .map(|(a, b)| sampson_residual(&e, a, b).powi(2))
                    .sum::<f64>()
            };
            let err = |p: &RelativePose| {
                (
                    rotation_angle_between(&p.rotation, &r).to_degrees(),
                    p.translation.angle(&t.normalize()).to_degrees(),
                )
            };
            let e = essential_8pt(&s.x1, &s.x2).unwrap();
            let lin = recover_pose(&e, &s.x1, &s.x2).unwrap();
            let rf = refine_relative_pose(&e, &s.x1, &s.x2, 50).unwrap();
            if lin.translation_observable {
                st.lin.push(err(&lin));
            }
            if !rf.translation_observable {
                st.unobservable += 1;
                continue;
            }
            st.refined.push(err(&rf));
            if !lin.translation_observable {
                continue;
            }
            st.costs.push((
                cost(&lin.rotation, &lin.translation),
                cost(&rf.rotation, &rf.translation),
                cost(&r, &t),
            ));
        }
        st
    }

    /// 평면 장면(σ0.5px, 200점)에 이상치 비율 `out` 만큼 둘째 좌표를 무작위로 바꾼 자료와 이상치 표시.
    fn planar_case_scene(seed: u64, out: f64) -> (Scene, Vec<bool>) {
        let mut s = scene_full(200, 0.5, seed, 3.0, 0.0);
        let k = s.c1.intrinsics;
        let mut rng = Lcg(seed ^ 0xABCD);
        let mut bad = vec![false; s.x2.len()];
        for (x, b) in s.x2.iter_mut().zip(bad.iter_mut()) {
            if rng.next() < out {
                let u = Vector2::new(rng.next() * 960.0, rng.next() * 540.0);
                *x = k.to_normalized(&u);
                *b = true;
            }
        }
        (s, bad)
    }

    /// 자료 하한: 이상치가 아닌 대응만으로 정답 자세에서 시작한 Sampson 정밀화의 회전 오차(도).
    fn planar_floor(s: &Scene, bad: &[bool]) -> f64 {
        let (r, t) = s.rel();
        let s1: Vec<_> = (0..bad.len())
            .filter(|&i| !bad[i])
            .map(|i| s.x1[i])
            .collect();
        let s2: Vec<_> = (0..bad.len())
            .filter(|&i| !bad[i])
            .map(|i| s.x2[i])
            .collect();
        let (rr, _) = refine_pose(&r, &t, &s1, &s2, 50);
        rotation_angle_between(&rr, &r).to_degrees()
    }

    /// 평면 장면에서 5점 RANSAC 후보 중 정답에 가장 가까운 것의 (회전, 방향, 거짓 정상, 놓친 정상).
    fn planar_ransac_case(s: &Scene, bad: &[bool]) -> (f64, f64, usize, usize) {
        let k = s.c1.intrinsics;
        let cfg = RansacConfig {
            threshold_px: 1.5,
            ..RansacConfig::default()
        };
        let cands = ransac_essential_candidates(&s.x1, &s.x2, k.fx, &cfg);
        assert!(!cands.is_empty(), "RANSAC 실패");
        let (r, t) = s.rel();
        // 후보 중 정답에 가장 가까운 것(평면이면 정답과 쌍둥이 둘이 나온다).
        cands
            .iter()
            .map(|(e, inl)| {
                let s1: Vec<_> = (0..inl.len())
                    .filter(|&i| inl[i])
                    .map(|i| s.x1[i])
                    .collect();
                let s2: Vec<_> = (0..inl.len())
                    .filter(|&i| inl[i])
                    .map(|i| s.x2[i])
                    .collect();
                let pose = recover_pose(e, &s1, &s2).expect("자세 없음");
                let rot = rotation_angle_between(&pose.rotation, &r).to_degrees();
                let dir = angle_between(&pose.translation, &t).to_degrees();
                let false_in = (0..inl.len()).filter(|&i| inl[i] && bad[i]).count();
                let missed = (0..inl.len()).filter(|&i| !inl[i] && !bad[i]).count();
                (rot, dir, false_in, missed)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap()
    }

    /// 시드 1..=20 에서 시드마다 최선 후보 회전 ≤ 그 시드의 자료 하한 + 0.1° 인지 확인하고 최대 초과량을 돌려준다.
    fn planar_against_floor(out: f64) -> f64 {
        let mut fails = vec![];
        let mut worst_excess = f64::NEG_INFINITY;
        for seed in 1..=20 {
            let (s, bad) = planar_case_scene(seed, out);
            let floor = planar_floor(&s, &bad);
            let (rot, dir, false_in, missed) = planar_ransac_case(&s, &bad);
            eprintln!(
                "평면 이상치 {out} 시드 {seed}: 최선 후보 회전 {rot:.4}° 하한 {floor:.4}° 방향 {dir:.3}° 거짓정상 {false_in} 놓침 {missed}"
            );
            worst_excess = worst_excess.max(rot - floor);
            if rot > floor + 0.1 {
                fails.push(seed);
            }
        }
        eprintln!("평면 이상치 {out}: 하한 대비 최대 초과 {worst_excess:.4}°");
        assert!(
            fails.is_empty(),
            "이상치 {out}: 하한 + 0.1° 초과 시드 {fails:?}"
        );
        worst_excess
    }

    #[test]
    fn ransac_essential_planar_rotation() {
        planar_against_floor(0.0);
    }

    #[test]
    fn ransac_essential_planar_with_outliers() {
        planar_against_floor(0.3);
    }

    /// 자료 한계: 정답 자세에서 시작한 Sampson 정밀화의 회전 오차(평면 σ0.5px 에서도 0.3° 를 넘는다).
    #[test]
    fn planar_sampson_floor_from_truth() {
        let mut floor0: f64 = 0.0;
        for h in [0.0, 1.0, 10.0] {
            let mut worst: f64 = 0.0;
            for seed in 1..=20 {
                let s = scene_full(200, 0.5, seed, 3.0, h);
                let (r, t) = s.rel();
                let (rr, _) = refine_pose(&r, &t, &s.x1, &s.x2, 50);
                worst = worst.max(rotation_angle_between(&rr, &r).to_degrees());
            }
            eprintln!("높이 {h} m: 정답에서 시작한 정밀화 최악 회전 {worst:.4}°");
            if h == 0.0 {
                floor0 = worst;
            }
        }
        // 측정값 0.430°. 이 값보다 작은 회전 오차 기준은 평면 σ0.5px 에서 달성할 수 없다.
        assert!(floor0 > 0.3 && floor0 < 0.5, "평면 하한 {floor0}°");
    }

    #[test]
    fn ransac_essential_planar_noise_free_contains_truth() {
        for seed in 1..=10 {
            let s = scene_full(100, 0.0, seed, 3.0, 0.0);
            let k = s.c1.intrinsics;
            let cands = ransac_essential_candidates(&s.x1, &s.x2, k.fx, &RansacConfig::default());
            let (r, _) = s.rel();
            let best = cands
                .iter()
                .filter_map(|(e, _)| recover_pose(e, &s.x1, &s.x2))
                .map(|p| rotation_angle_between(&p.rotation, &r).to_degrees())
                .fold(f64::INFINITY, f64::min);
            assert!(
                best < 1e-3,
                "시드 {seed}: 잡음 없는 평면 최선 후보 회전 {best}°"
            );
        }
    }

    #[test]
    fn homography_dlt_recovers_plane_transfer() {
        // 평면 장면(높이 범위 0)의 잡음 없는 대응은 한 호모그래피로 정확히 옮겨진다.
        let s = scene_full(50, 0.0, 7, 3.0, 0.0);
        let h = homography_dlt(&s.x1, &s.x2).expect("H 없음");
        let worst =
            s.x1.iter()
                .zip(&s.x2)
                .map(|(a, b)| homography_transfer(&h, a, b))
                .fold(0.0, f64::max);
        assert!(worst < 1e-9, "평면 전달 오차 {worst}");
        assert!(homography_dlt(&s.x1[..3], &s.x2[..3]).is_none());
    }

    #[test]
    fn ransac_essential_rejects_bad_input() {
        let cfg = RansacConfig::default();
        let a = vec![Vector2::new(0.1, 0.2); 4];
        assert!(ransac_essential(&a, &a, 800.0, &cfg).is_none());
        let s = scene(20, 0.0, 3);
        assert!(ransac_essential(&s.x1, &s.x2[..19], 800.0, &cfg).is_none());
        assert!(ransac_essential(&s.x1, &s.x2, 0.0, &cfg).is_none());
    }
}
