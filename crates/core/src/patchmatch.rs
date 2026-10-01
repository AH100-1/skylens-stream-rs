//! 시점별 PatchMatch 스테레오: 경사 평면 가설로 깊이와 법선을 추정한다.
//!
//! 각 화소는 기준 카메라 좌표계의 평면(깊이 d, 법선 n)을 가진다. 비용은
//! 평면 유도 호모그래피로 이웃 사진에 옮긴 창의 양방향 가중 NCC 이고,
//! 이웃 여러 장 중 비용이 작은 k 개를 평균한다. 무작위 초기화 → 체커보드
//! 공간 전파 + 무작위 섭동 정련을 반복한다. 각 색 반쪽은 rayon 으로 병렬.

use crate::camera::Camera;
use crate::math::{Matrix3, Vector3};
use rayon::prelude::*;

/// 회색조 영상(행 우선, 값 범위는 자유).
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// 쌍선형 보간. 영상 밖이면 None (화소 중심은 정수 좌표).
    #[inline]
    fn sample(&self, x: f32, y: f32) -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0) {
            return None;
        }
        let x0 = x as usize;
        let y0 = y as usize;
        if x0 + 1 >= self.width || y0 + 1 >= self.height {
            return None;
        }
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let i = y0 * self.width + x0;
        let a = self.data[i];
        let b = self.data[i + 1];
        let c = self.data[i + self.width];
        let d = self.data[i + self.width + 1];
        Some((a + (b - a) * fx) * (1.0 - fy) + (c + (d - c) * fx) * fy)
    }
}

/// 한 시점: 왜곡 없는 핀홀 카메라(내부 + 자세)와 회색조 영상.
/// 영상 크기는 `camera.intrinsics` 의 크기와 같아야 한다.
#[derive(Clone, Debug)]
pub struct View {
    pub camera: Camera,
    pub image: GrayImage,
}

/// PatchMatch 설정.
#[derive(Clone, Debug)]
pub struct Config {
    /// 창 반지름(화소).
    pub radius: usize,
    /// 창 안 표본 간격(화소). 2 면 창 화소의 1/4 만 쓴다.
    pub step: usize,
    /// 전파·정련 반복 횟수(한 번 = 빨강·검정 두 반쪽).
    pub iterations: usize,
    /// 양방향 가중의 밝기 척도(영상 값 단위).
    pub sigma_color: f32,
    /// 양방향 가중의 거리 척도(화소).
    pub sigma_spatial: f32,
    /// 이웃 비용 중 평균할 개수.
    pub top_k: usize,
    /// 반복마다 시도하는 섭동 가설 수.
    pub perturbations: usize,
    /// 난수 씨앗.
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            radius: 5,
            step: 2,
            iterations: 6,
            sigma_color: 0.1,
            sigma_spatial: 3.0,
            top_k: 2,
            perturbations: 3,
            seed: 0x5eed,
        }
    }
}

/// 기준 시점의 깊이(카메라 z)·법선(기준 카메라 좌표계, 카메라를 향함)·비용.
#[derive(Clone, Debug)]
pub struct DepthMap {
    pub w: usize,
    pub h: usize,
    pub depth: Vec<f32>,
    pub normal: Vec<[f32; 3]>,
    pub cost: Vec<f32>,
}

/// 비용 상한(1 - NCC 의 최댓값 2).
const MAX_COST: f32 = 2.0;

/// 이웃 하나로 옮기는 상수: 호모그래피 H = K_j (R + t·mᵀ) K_r⁻¹ 의 조각.
struct NeighborGeom {
    /// K_j R K_r⁻¹
    a: Matrix3<f32>,
    /// K_j t
    kt: Vector3<f32>,
}

/// 기준 창의 미리 계산한 표본.
struct RefPatch {
    /// (dx, dy, 정규화 가중치, 기준 값)
    samples: Vec<(f32, f32, f32, f32)>,
    mean: f32,
    var: f32,
}

#[derive(Clone, Copy)]
struct Hyp {
    depth: f32,
    n: Vector3<f32>,
}

/// splitmix64 기반 화소별 난수.
struct Rng(u64);

impl Rng {
    fn new(seed: u64, a: u64, b: u64) -> Self {
        let mut r = Rng(seed
            ^ a.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ b.wrapping_mul(0xc2b2_ae3d_27d4_eb4f));
        r.next();
        r
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// [0, 1)
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Ctx<'a> {
    w: usize,
    h: usize,
    kinv: Matrix3<f32>,
    refimg: &'a GrayImage,
    neighbors: Vec<(NeighborGeom, &'a GrayImage)>,
    range: (f32, f32),
    cfg: &'a Config,
}

impl Ctx<'_> {
    #[inline]
    fn ray(&self, x: f32, y: f32) -> Vector3<f32> {
        self.kinv * Vector3::new(x, y, 1.0)
    }

    fn ref_patch(&self, x: usize, y: usize) -> Option<RefPatch> {
        let r = self.cfg.radius as isize;
        let st = self.cfg.step.max(1);
        let c = self.refimg.at(x, y);
        let mut samples = Vec::with_capacity(((2 * r as usize) / st + 1).pow(2));
        let mut wsum = 0.0f32;
        let mut dy = -r;
        while dy <= r {
            let mut dx = -r;
            while dx <= r {
                let px = x as isize + dx;
                let py = y as isize + dy;
                if px >= 0 && py >= 0 && (px as usize) < self.w && (py as usize) < self.h {
                    let v = self.refimg.at(px as usize, py as usize);
                    let ds = ((dx * dx + dy * dy) as f32).sqrt();
                    let wt =
                        (-(v - c).abs() / self.cfg.sigma_color - ds / self.cfg.sigma_spatial).exp();
                    samples.push((dx as f32, dy as f32, wt, v));
                    wsum += wt;
                }
                dx += st as isize;
            }
            dy += st as isize;
        }
        if samples.len() < 4 || wsum <= 0.0 {
            return None;
        }
        let mut mean = 0.0;
        for s in samples.iter_mut() {
            s.2 /= wsum;
            mean += s.2 * s.3;
        }
        let mut var = 0.0;
        for s in &samples {
            var += s.2 * (s.3 - mean) * (s.3 - mean);
        }
        Some(RefPatch { samples, mean, var })
    }

    /// 화소 (x, y) 에서 가설의 비용.
    fn cost(&self, x: usize, y: usize, patch: &RefPatch, hyp: &Hyp) -> f32 {
        if patch.var < 1e-8 {
            return MAX_COST;
        }
        let rp = self.ray(x as f32, y as f32);
        let ndr = hyp.n.dot(&rp);
        if ndr >= -1e-6 {
            return MAX_COST;
        }
        // 평면 nᵀX = c, c = d·nᵀr_p. 평면 위 X 에 대해 X_j = (R + t nᵀ / c) X.
        let c = hyp.depth * ndr;
        let m = hyp.n / c;
        let mut costs: Vec<f32> = Vec::with_capacity(self.neighbors.len());
        let mk = self.kinv.transpose() * m;
        for (g, img) in &self.neighbors {
            let h = g.a + g.kt * mk.transpose();
            costs.push(ncc_cost(x, y, patch, &h, img));
        }
        costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.cfg.top_k.clamp(1, costs.len());
        costs[..k].iter().sum::<f32>() / k as f32
    }

    fn random_normal(&self, rng: &mut Rng, rp: &Vector3<f32>) -> Vector3<f32> {
        // 카메라를 향하는 반구에서 고르게 (기울기가 너무 큰 쪽은 제한).
        loop {
            let v = Vector3::new(
                2.0 * rng.f() - 1.0,
                2.0 * rng.f() - 1.0,
                2.0 * rng.f() - 1.0,
            );
            let l = v.norm();
            if !(0.05..=1.0).contains(&l) {
                continue;
            }
            let mut n = v / l;
            let rn = rp.normalize();
            let d = n.dot(&rn);
            if d > 0.0 {
                n = -n;
            }
            if n.dot(&rn) < -0.1 {
                return n;
            }
        }
    }

    fn random_depth(&self, rng: &mut Rng) -> f32 {
        // 역깊이에서 고르게: 먼 쪽 화소 시차가 작으므로 표본 밀도를 맞춘다.
        let (a, b) = (1.0 / self.range.0, 1.0 / self.range.1);
        1.0 / (a + (b - a) * rng.f())
    }

    /// 평면 가설을 다른 화소로 옮긴 깊이.
    fn transfer_depth(&self, from: (usize, usize), hyp: &Hyp, to: &Vector3<f32>) -> Option<f32> {
        let rf = self.ray(from.0 as f32, from.1 as f32);
        let c = hyp.depth * hyp.n.dot(&rf);
        let den = hyp.n.dot(to);
        if den >= -1e-6 {
            return None;
        }
        let d = c / den;
        if d >= self.range.0 && d <= self.range.1 {
            Some(d)
        } else {
            None
        }
    }

    fn perturb_normal(
        &self,
        rng: &mut Rng,
        n: &Vector3<f32>,
        scale: f32,
        rp: &Vector3<f32>,
    ) -> Vector3<f32> {
        let v = Vector3::new(
            n.x + scale * (2.0 * rng.f() - 1.0),
            n.y + scale * (2.0 * rng.f() - 1.0),
            n.z + scale * (2.0 * rng.f() - 1.0),
        );
        let v = v.normalize();
        if v.dot(&rp.normalize()) < -0.1 {
            v
        } else {
            *n
        }
    }
}

/// 기준 시점 하나의 깊이·법선 지도를 추정한다. `range` 는 (최소, 최대) 깊이.
pub fn estimate(ref_view: &View, neighbors: &[View], range: (f64, f64), cfg: &Config) -> DepthMap {
    let w = ref_view.image.width;
    let h = ref_view.image.height;
    let kr = intrinsics_matrix(&ref_view.camera);
    let kinv64 = kr.try_inverse().expect("내부 행렬은 가역");
    let kinv: Matrix3<f32> = kinv64.cast();
    let rr = ref_view.camera.pose.rotation.into_inner();
    let tr = ref_view.camera.pose.translation;
    let ngeom = neighbors
        .iter()
        .map(|v| {
            let kj = intrinsics_matrix(&v.camera);
            let rj = v.camera.pose.rotation.into_inner();
            let tj = v.camera.pose.translation;
            // X_j = Rj Rrᵀ (X_r - t_r) + t_j
            let r_rel = rj * rr.transpose();
            let t_rel = tj - r_rel * tr;
            let a = kj * r_rel * kinv64;
            let kt = kj * t_rel;
            (
                NeighborGeom {
                    a: a.cast(),
                    kt: kt.cast(),
                },
                &v.image,
            )
        })
        .collect::<Vec<_>>();
    let ctx = Ctx {
        w,
        h,
        kinv,
        refimg: &ref_view.image,
        neighbors: ngeom,
        range: (range.0 as f32, range.1 as f32),
        cfg,
    };

    let n = w * h;
    let patches: Vec<Option<RefPatch>> = (0..n)
        .into_par_iter()
        .map(|i| ctx.ref_patch(i % w, i / w))
        .collect();

    // 무작위 초기화.
    let init: Vec<(Hyp, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let mut rng = Rng::new(cfg.seed, i as u64, u64::MAX);
            let rp = ctx.ray(x as f32, y as f32);
            let hyp = Hyp {
                depth: ctx.random_depth(&mut rng),
                n: ctx.random_normal(&mut rng, &rp),
            };
            let c = match &patches[i] {
                Some(p) if !ctx.neighbors.is_empty() => ctx.cost(x, y, p, &hyp),
                _ => MAX_COST,
            };
            (hyp, c)
        })
        .collect();
    let mut hyps: Vec<Hyp> = init.iter().map(|p| p.0).collect();
    let mut costs: Vec<f32> = init.iter().map(|p| p.1).collect();

    if !ctx.neighbors.is_empty() {
        const OFFS: [(isize, isize); 8] = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-3, 0),
            (3, 0),
            (0, -3),
            (0, 3),
        ];
        let log_span = (ctx.range.1 / ctx.range.0).ln();
        for it in 0..cfg.iterations {
            for color in 0..2usize {
                let updates: Vec<(usize, Hyp, f32)> = (0..h)
                    .into_par_iter()
                    .flat_map_iter(|y| {
                        let hyps = &hyps;
                        let costs = &costs;
                        let ctx = &ctx;
                        let patches = &patches;
                        let x0 = (y + color) % 2;
                        (x0..w).step_by(2).filter_map(move |x| {
                            let i = y * w + x;
                            let patch = patches[i].as_ref()?;
                            let rp = ctx.ray(x as f32, y as f32);
                            let mut best = hyps[i];
                            let mut best_c = costs[i];
                            let start = (best, best_c);
                            let mut rng = Rng::new(cfg.seed, i as u64, (it * 2 + color) as u64);
                            // 공간 전파.
                            for &(dx, dy) in &OFFS {
                                let nx = x as isize + dx;
                                let ny = y as isize + dy;
                                if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                                    continue;
                                }
                                let j = ny as usize * w + nx as usize;
                                let nh = hyps[j];
                                let Some(d) =
                                    ctx.transfer_depth((nx as usize, ny as usize), &nh, &rp)
                                else {
                                    continue;
                                };
                                let cand = Hyp { depth: d, n: nh.n };
                                let c = ctx.cost(x, y, patch, &cand);
                                if c < best_c {
                                    best = cand;
                                    best_c = c;
                                }
                            }
                            // 무작위 섭동 정련: 척도를 줄여 가며.
                            let mut ds = 0.5f32;
                            let mut ns = 1.0f32;
                            let frac = 1.0 - it as f32 / cfg.iterations.max(1) as f32;
                            ds *= frac.max(0.05);
                            ns *= frac.max(0.05);
                            // 완전 무작위 하나.
                            let cand = Hyp {
                                depth: ctx.random_depth(&mut rng),
                                n: ctx.random_normal(&mut rng, &rp),
                            };
                            let c = ctx.cost(x, y, patch, &cand);
                            if c < best_c {
                                best = cand;
                                best_c = c;
                            }
                            for _ in 0..cfg.perturbations {
                                ds *= 0.5;
                                ns *= 0.5;
                                let f = (log_span * ds * (2.0 * rng.f() - 1.0)).exp();
                                let d = (best.depth * f).clamp(ctx.range.0, ctx.range.1);
                                let nn = ctx.perturb_normal(&mut rng, &best.n, ns, &rp);
                                for cand in [
                                    Hyp {
                                        depth: d,
                                        n: best.n,
                                    },
                                    Hyp {
                                        depth: best.depth,
                                        n: nn,
                                    },
                                    Hyp { depth: d, n: nn },
                                ] {
                                    let c = ctx.cost(x, y, patch, &cand);
                                    if c < best_c {
                                        best = cand;
                                        best_c = c;
                                    }
                                }
                            }
                            if best_c < start.1 {
                                Some((i, best, best_c))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();
                for (i, hy, c) in updates {
                    hyps[i] = hy;
                    costs[i] = c;
                }
            }
        }
    }

    DepthMap {
        w,
        h,
        depth: hyps.iter().map(|h| h.depth).collect(),
        normal: hyps.iter().map(|h| [h.n.x, h.n.y, h.n.z]).collect(),
        cost: costs,
    }
}

fn intrinsics_matrix(cam: &Camera) -> Matrix3<f64> {
    let k = &cam.intrinsics;
    Matrix3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

fn ncc_cost(x: usize, y: usize, patch: &RefPatch, h: &Matrix3<f32>, img: &GrayImage) -> f32 {
    let (xf, yf) = (x as f32, y as f32);
    let mut sw = 0.0;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut cross = 0.0;
    for &(dx, dy, wt, v) in &patch.samples {
        let q = Vector3::new(xf + dx, yf + dy, 1.0);
        let p = h * q;
        if p.z <= 1e-6 {
            return MAX_COST;
        }
        let Some(s) = img.sample(p.x / p.z, p.y / p.z) else {
            return MAX_COST;
        };
        sw += wt;
        sum += wt * s;
        sum2 += wt * s * s;
        cross += wt * s * v;
    }
    let mean = sum / sw;
    let var = sum2 / sw - mean * mean;
    if var < 1e-8 {
        return MAX_COST;
    }
    let cov = cross / sw - mean * patch.mean;
    let ncc = cov / (var * patch.var).sqrt();
    (1.0 - ncc).clamp(0.0, MAX_COST)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Intrinsics, Pose};
    use crate::math::{Point3, Rotation3, Vector2};

    /// 장면: 경사 평면 하나, 또는 기준 시점 열마다 깊이가 다른 계단 + 먼 배경.
    enum Scene {
        Slanted {
            z0: f64,
            a: f64,
            b: f64,
        },
        Steps {
            steps: Vec<(f64, f64, f64)>,
            back: f64,
        },
    }

    impl Scene {
        /// 광선 o + s·d 의 첫 교점 매개 s (d 는 z 성분이 있는 방향).
        fn hit(&self, o: &Vector3<f64>, d: &Vector3<f64>) -> Option<f64> {
            match self {
                Scene::Slanted { z0, a, b } => {
                    let den = d.z - a * d.x - b * d.y;
                    let s = (z0 + a * o.x + b * o.y - o.z) / den;
                    (s > 0.0).then_some(s)
                }
                Scene::Steps { steps, back } => {
                    let mut best = (back - o.z) / d.z;
                    for &(z, x0, x1) in steps {
                        let s = (z - o.z) / d.z;
                        let x = o.x + s * d.x;
                        if s > 0.0 && s < best && x >= x0 && x < x1 {
                            best = s;
                        }
                    }
                    (best > 0.0).then_some(best)
                }
            }
        }
    }

    fn hash(ix: i64, iy: i64) -> f64 {
        let mut z = (ix as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (iy as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn value_noise(x: f64, y: f64) -> f64 {
        let (fx, fy) = (x.floor(), y.floor());
        let (ix, iy) = (fx as i64, fy as i64);
        let s = |t: f64| t * t * (3.0 - 2.0 * t);
        let (u, v) = (s(x - fx), s(y - fy));
        let a = hash(ix, iy);
        let b = hash(ix + 1, iy);
        let c = hash(ix, iy + 1);
        let d = hash(ix + 1, iy + 1);
        (a + (b - a) * u) * (1.0 - v) + (c + (d - c) * u) * v
    }

    /// 세계 (X, Y) 에 붙은 질감. 격자 0.25, 0.1 두 단계.
    fn texture(p: &Vector3<f64>) -> f64 {
        0.6 * value_noise(p.x / 0.25, p.y / 0.25) + 0.4 * value_noise(p.x / 0.1 + 17.0, p.y / 0.1)
    }

    /// 3×3 부분 표본으로 렌더하고, 화소 중심의 카메라 z 깊이를 함께 돌려준다.
    fn render(cam: &Camera, scene: &Scene) -> (GrayImage, Vec<f64>) {
        let (w, h) = (
            cam.intrinsics.width as usize,
            cam.intrinsics.height as usize,
        );
        let rinv = cam.pose.rotation.inverse();
        let o = cam.pose.center().coords;
        let mut img = vec![0f32; w * h];
        let mut depth = vec![0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for sy in -1..=1 {
                    for sx in -1..=1 {
                        let p =
                            Vector2::new(x as f64 + sx as f64 / 3.0, y as f64 + sy as f64 / 3.0);
                        let n = cam.intrinsics.to_normalized(&p);
                        let d = rinv * Vector3::new(n.x, n.y, 1.0);
                        let s = scene.hit(&o, &d).expect("광선이 장면에 닿아야 한다");
                        acc += texture(&(o + s * d));
                        if sx == 0 && sy == 0 {
                            depth[y * w + x] = s; // d 의 카메라 z 성분이 1
                        }
                    }
                }
                img[y * w + x] = (acc / 9.0) as f32;
            }
        }
        (GrayImage::new(w, h, img), depth)
    }

    /// 기준 카메라(원점, 정면)와 상하좌우로 기선 b 만큼 떨어져 장면 중심을 보는 이웃 넷.
    fn rig(w: u32, h: u32, base: f64, look: f64) -> Vec<Camera> {
        let k = Intrinsics::from_hfov(w, h, 60f64.to_radians());
        let mut cams = vec![Camera {
            intrinsics: k,
            pose: Pose::from_center(Rotation3::identity(), &Point3::origin()),
        }];
        for (cx, cy) in [(base, 0.0), (-base, 0.0), (0.0, base), (0.0, -base)] {
            // 중심 (cx, cy, 0) 에서 (0, 0, look) 을 향하도록: 세계→카메라 회전.
            let r = Rotation3::from_euler_angles(-(cy / look).atan(), (cx / look).atan(), 0.0);
            cams.push(Camera {
                intrinsics: k,
                pose: Pose::from_center(r, &Point3::new(cx, cy, 0.0)),
            });
        }
        cams
    }

    fn views(cams: &[Camera], scene: &Scene) -> (View, Vec<View>, Vec<f64>) {
        let (img, gt) = render(&cams[0], scene);
        let refv = View {
            camera: cams[0],
            image: img,
        };
        let ns = cams[1..]
            .iter()
            .map(|c| View {
                camera: *c,
                image: render(c, scene).0,
            })
            .collect();
        (refv, ns, gt)
    }

    struct Stats {
        median_rel: f64,
        within_1pct: f64,
        median_normal_deg: f64,
        count: usize,
    }

    fn stats(
        dm: &DepthMap,
        gt: &[f64],
        gt_n: impl Fn(usize) -> Vector3<f64>,
        mask: impl Fn(usize, usize) -> bool,
    ) -> Stats {
        let mut rel = Vec::new();
        let mut ang = Vec::new();
        for y in 0..dm.h {
            for x in 0..dm.w {
                if !mask(x, y) {
                    continue;
                }
                let i = y * dm.w + x;
                rel.push((dm.depth[i] as f64 - gt[i]).abs() / gt[i]);
                let n = dm.normal[i];
                let n = Vector3::new(n[0] as f64, n[1] as f64, n[2] as f64).normalize();
                ang.push(n.dot(&gt_n(i)).clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let count = rel.len();
        let within = rel.iter().filter(|&&r| r < 0.01).count() as f64 / count as f64;
        Stats {
            median_rel: med(&mut rel),
            within_1pct: within,
            median_normal_deg: med(&mut ang),
            count,
        }
    }

    // 160×120, f ≈ 138.6 px, 깊이 10, 기선 1 → 시차 약 14 px.
    // 깊이 1% 는 시차 0.14 px 에 해당하므로 부분 화소 정합이 되어야 넘는 기준이다.

    #[test]
    fn slanted_plane_depth_and_normal() {
        let (a, b) = (0.3, -0.15);
        let scene = Scene::Slanted { z0: 10.0, a, b };
        let cams = rig(160, 120, 1.0, 10.0);
        let (refv, ns, gt) = views(&cams, &scene);
        let dm = estimate(&refv, &ns, (5.0, 20.0), &Config::default());
        let n_gt = Vector3::new(a, b, -1.0).normalize();
        let m = 8;
        let s = stats(
            &dm,
            &gt,
            |_| n_gt,
            |x, y| x >= m && y >= m && x < 160 - m && y < 120 - m,
        );
        eprintln!(
            "경사 평면: 화소 {} 상대오차 중앙값 {:.4}% 1% 이내 {:.1}% 법선 중앙값 {:.2}°",
            s.count,
            100.0 * s.median_rel,
            100.0 * s.within_1pct,
            s.median_normal_deg
        );
        assert!(s.median_rel < 0.005, "상대오차 중앙값 {}", s.median_rel);
        assert!(s.within_1pct > 0.90, "1% 이내 비율 {}", s.within_1pct);
        assert!(
            s.median_normal_deg < 5.0,
            "법선 각오차 중앙값 {}°",
            s.median_normal_deg
        );
    }

    #[test]
    fn steps_depth() {
        // 기준 시점 정규 x 범위 [-0.6,-0.2), [-0.2,0.2), [0.2,0.6) 에 깊이 8, 10, 12 계단.
        let zs = [(8.0, -0.6, -0.2), (10.0, -0.2, 0.2), (12.0, 0.2, 0.6)];
        let steps = zs.iter().map(|&(z, a, b)| (z, a * z, b * z)).collect();
        let scene = Scene::Steps { steps, back: 16.0 };
        let cams = rig(160, 120, 1.0, 10.0);
        let (refv, ns, gt) = views(&cams, &scene);
        let dm = estimate(&refv, &ns, (5.0, 20.0), &Config::default());
        let n_gt = Vector3::new(0.0, 0.0, -1.0);
        // 깊이 불연속에서 6 화소 이상 떨어진 내부만 센다.
        let (w, h) = (160usize, 120usize);
        let m = 8;
        let near_edge = |x: usize, y: usize| {
            let g = gt[y * w + x];
            (x.saturating_sub(6)..(x + 7).min(w)).any(|xx| (gt[y * w + xx] - g).abs() > 0.05 * g)
        };
        let s = stats(
            &dm,
            &gt,
            |_| n_gt,
            |x, y| x >= m && y >= m && x < w - m && y < h - m && !near_edge(x, y),
        );
        eprintln!(
            "계단: 화소 {} 상대오차 중앙값 {:.4}% 1% 이내 {:.1}% 법선 중앙값 {:.2}°",
            s.count,
            100.0 * s.median_rel,
            100.0 * s.within_1pct,
            s.median_normal_deg
        );
        assert!(s.median_rel < 0.005, "상대오차 중앙값 {}", s.median_rel);
        assert!(s.within_1pct > 0.85, "1% 이내 비율 {}", s.within_1pct);
        assert!(
            s.median_normal_deg < 5.0,
            "법선 각오차 중앙값 {}°",
            s.median_normal_deg
        );
    }

    #[test]
    fn deterministic_and_shapes() {
        let scene = Scene::Slanted {
            z0: 10.0,
            a: 0.0,
            b: 0.0,
        };
        let cams = rig(48, 36, 1.0, 10.0);
        let (refv, ns, _) = views(&cams, &scene);
        let cfg = Config {
            iterations: 2,
            ..Config::default()
        };
        let a = estimate(&refv, &ns, (5.0, 20.0), &cfg);
        let b = estimate(&refv, &ns, (5.0, 20.0), &cfg);
        assert_eq!((a.w, a.h), (48, 36));
        assert_eq!(a.depth.len(), 48 * 36);
        assert_eq!(a.normal.len(), 48 * 36);
        assert_eq!(a.cost.len(), 48 * 36);
        assert_eq!(a.depth, b.depth);
        assert!(a.depth.iter().all(|&d| (5.0..=20.0).contains(&d)));
        // 이웃이 없으면 비용은 상한.
        let c = estimate(&refv, &[], (5.0, 20.0), &cfg);
        assert!(c.cost.iter().all(|&v| v == MAX_COST));
    }

    /// 960×720 한 장 시간 측정(기본 시험에서는 빠짐): cargo test --release -- --ignored
    #[test]
    #[ignore]
    fn timing_960() {
        let scene = Scene::Slanted {
            z0: 10.0,
            a: 0.3,
            b: -0.15,
        };
        let cams = rig(960, 720, 1.0, 10.0);
        let (refv, ns, gt) = views(&cams, &scene);
        let t = std::time::Instant::now();
        let dm = estimate(&refv, &ns, (5.0, 20.0), &Config::default());
        let el = t.elapsed().as_secs_f64();
        let n_gt = Vector3::new(0.3, -0.15, -1.0).normalize();
        let s = stats(
            &dm,
            &gt,
            |_| n_gt,
            |x, y| x >= 8 && y >= 8 && x < 952 && y < 712,
        );
        eprintln!(
            "960×720 이웃 4: {el:.2} s, 상대오차 중앙값 {:.4}% 1% 이내 {:.1}%",
            100.0 * s.median_rel,
            100.0 * s.within_1pct
        );
    }
}
