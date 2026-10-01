//! 특징 매칭과 기하 검증: 비율 검사, 정규화 8점 기본 행렬, RANSAC.

use crate::features::Feature;
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};

/// 최근접/차근접 거리 비율 검사 매칭(L2, 전수 탐색). 결과는 (a 인덱스, b 인덱스).
/// `mutual` 이면 b→a 최근접도 같은 짝인 것만 남긴다.
pub fn ratio_match(a: &[Feature], b: &[Feature], ratio: f32, mutual: bool) -> Vec<(usize, usize)> {
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

/// 점들을 무게중심 0, 평균 거리 √2 로 옮기는 상사 변환(Hartley 1997).
fn normalizer(p: &[Vector2<f64>]) -> Matrix3<f64> {
    let n = p.len() as f64;
    let c = p.iter().fold(Vector2::zeros(), |s, x| s + x) / n;
    let d = p.iter().map(|x| (x - c).norm()).sum::<f64>() / n;
    let s = if d > 0.0 { 2f64.sqrt() / d } else { 1.0 };
    Matrix3::new(s, 0.0, -s * c.x, 0.0, s, -s * c.y, 0.0, 0.0, 1.0)
}

/// 정규화 8점 알고리즘으로 기본 행렬 F (x2ᵀ F x1 = 0, 픽셀 좌표)를 구한다.
/// 점이 8개 미만이거나 퇴화하면 None. 결과는 계수 2, 프로베니우스 노름 1.
pub fn fundamental_8pt(x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> Option<Matrix3<f64>> {
    if x1.len() < 8 || x1.len() != x2.len() {
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

/// 기본 행렬에 대한 Sampson 거리(픽셀², 1차 기하 오차 근사).
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

/// RANSAC 설정.
#[derive(Clone, Copy, Debug)]
pub struct RansacConfig {
    /// 정상 판정 Sampson 거리 문턱(픽셀).
    pub threshold_px: f64,
    pub max_iters: usize,
    /// 적응형 종료 신뢰도.
    pub confidence: f64,
    pub seed: u64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            threshold_px: 1.5,
            max_iters: 2000,
            confidence: 0.999,
            seed: 1,
        }
    }
}

/// RANSAC(Fischler & Bolles 1981) + 8점 기본 행렬로 기하 검증.
/// 반환: (정상 짝으로 다시 맞춘 F, 정상 여부 표시). 정상 짝이 8개 미만이면 None.
pub fn ransac_fundamental(
    x1: &[Vector2<f64>],
    x2: &[Vector2<f64>],
    cfg: &RansacConfig,
) -> Option<(Matrix3<f64>, Vec<bool>)> {
    let n = x1.len();
    if n < 8 || n != x2.len() {
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
        let inl = inliers_of(&f);
        let cnt = inl.iter().filter(|&&b| b).count();
        if best.as_ref().is_none_or(|b| cnt > b.2) {
            let w = cnt as f64 / n as f64;
            let p_good = w.powi(8);
            if p_good > 0.0 && p_good < 1.0 {
                let need = ((1.0 - cfg.confidence).ln() / (1.0 - p_good).ln()).ceil();
                iters = iters.min(need.max(1.0) as usize);
            } else if p_good >= 1.0 {
                iters = it;
            }
            best = Some((f, inl, cnt));
        }
    }
    let (mut f, mut inl, _) = best?;
    // 정상 짝 전체로 다시 맞추고 정상 집합을 갱신(두 번).
    for _ in 0..2 {
        let s1: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| x1[i]).collect();
        let s2: Vec<_> = (0..n).filter(|&i| inl[i]).map(|i| x2[i]).collect();
        let Some(g) = fundamental_8pt(&s1, &s2) else {
            break;
        };
        let gi = inliers_of(&g);
        if gi.iter().filter(|&&b| b).count() < s1.len() {
            break;
        }
        f = g;
        inl = gi;
    }
    (inl.iter().filter(|&&b| b).count() >= 8).then_some((f, inl))
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
        assert!(worst < 1e-6, "최대 Sampson 거리 {worst} px");
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
            // 정답 정상 짝에 대한 추정 F 의 Sampson 거리 RMS.
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
}
