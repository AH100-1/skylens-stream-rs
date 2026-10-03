//! 구역 밀집 복원(SPEC §3.6): 왜곡 보정·축소 → 이웃·깊이 범위 → 사진별 깊이·법선 → 융합.
//!
//! 사진별 깊이 단계는 [`DepthEstimator`] 함수 포인터 하나로 바꿔 끼운다. 기본값은
//! 거친-세밀 평면 스윕([`sweep_depth`])이고, [`region_cloud_with`] 로 다른 추정기를 넘긴다.

use crate::camera::{Camera, Intrinsics};
use crate::fusion::{self, DepthMap, FusionConfig, FusionView};
use crate::math::{Point3, Vector2, Vector3};
use crate::ply::PointCloud;
use crate::undistort::undistort_to_long_side;
use crate::view_selection::{self, SparsePoint};
use image::RgbImage;
use rayon::prelude::*;

/// 밀집 복원 입력 사진: 내부(왜곡 포함)·자세를 가진 카메라와 원본 영상.
#[derive(Clone, Debug)]
pub struct DenseView {
    pub camera: Camera,
    pub image: RgbImage,
}

/// 밀집 복원 설정.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseConfig {
    /// 보정 뒤 긴 변 화소 수(기본 960).
    pub max_width: u32,
    /// 사진별 이웃 수(기본 8).
    pub neighbors: usize,
    /// 융합 최소 동의 사진 수(기준 포함, 기본 3).
    pub min_views: usize,
    /// 융합 재투영 오차 상한(화소, 기본 1.0).
    pub reproj_px: f64,
    /// 융합 상대 깊이 차 상한(기본 0.01).
    pub depth_rel: f64,
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            max_width: 960,
            neighbors: 8,
            min_views: 3,
            reproj_px: 1.0,
            depth_rel: 0.01,
        }
    }
}

/// 평면 스윕 깊이 가설 수.
pub const SWEEP_HYPOTHESES: usize = 64;
/// 비용 집계에 쓰는 이웃 수.
pub const SWEEP_NEIGHBORS: usize = 3;
/// NCC 창 반폭(5×5).
const HALF: usize = 2;
/// 이 비용(1 − NCC 평균)을 넘는 화소는 깊이를 비운다.
const MAX_COST: f32 = 0.35;
/// 거친 단계 비용 상한(세밀 단계가 다시 걸러 주므로 느슨하게).
const COARSE_MAX_COST: f32 = 0.5;
/// 거친-세밀은 긴 변이 이 값 이상일 때만 쓴다(더 작으면 반해상도 창이 너무 거칠어 화소를 잃는다).
const MIN_COARSE_LONG_SIDE: usize = 384;
/// NCC 분산 안정화 항(회색 0..255 단위의 분산).
const VAR_EPS: f32 = 25.0;

/// 깊이 추정기가 받는 보정된 사진: 핀홀 자세, 회색 영상, 색, 유효 화소 표시.
pub struct DepthView {
    pub camera: Camera,
    pub gray: Vec<f32>,
    pub rgb: Vec<[u8; 3]>,
    pub valid: Vec<bool>,
}

fn prepare(v: &DenseView, max_width: u32) -> Option<DepthView> {
    let (iw, ih) = (v.image.width(), v.image.height());
    if iw == 0 || ih == 0 {
        return None;
    }
    let k = v.camera.intrinsics.to_distorted();
    let long = max_width.max(1).min(iw.max(ih));
    let u = undistort_to_long_side(&v.image, &k, long).ok()?;
    let rgb: Vec<[u8; 3]> = u.image.pixels().map(|p| p.0).collect();
    let gray = rgb
        .iter()
        .map(|c| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32)
        .collect();
    Some(DepthView {
        camera: Camera {
            intrinsics: u.pinhole,
            pose: v.camera.pose,
        },
        gray,
        rgb,
        valid: u.valid,
    })
}

fn bilinear(img: &[f32], w: usize, h: usize, x: f64, y: f64) -> Option<f32> {
    let (fx, fy) = (x - 0.5, y - 0.5);
    if !(fx >= 0.0 && fy >= 0.0 && fx < (w - 1) as f64 && fy < (h - 1) as f64) {
        return None;
    }
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (ax, ay) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
    let i = y0 * w + x0;
    let top = img[i] * (1.0 - ax) + img[i + 1] * ax;
    let bot = img[i + w] * (1.0 - ax) + img[i + w + 1] * ax;
    Some(top * (1.0 - ay) + bot * ay)
}

/// 평면 스윕 설정.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepConfig {
    /// 가설 수(거친 단계, 또는 거친-세밀을 쓰지 않을 때 전체).
    pub hypotheses: usize,
    /// 거친-세밀: 반해상도로 먼저 훑고 전체 해상도에서는 그 깊이 둘레만 찾는다.
    pub coarse_to_fine: bool,
    /// 세밀 단계 가설 수(홀수).
    pub fine_hypotheses: usize,
    /// 세밀 단계 탐색 반폭(거친 역깊이 간격의 배수).
    pub fine_span: f32,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            hypotheses: SWEEP_HYPOTHESES,
            coarse_to_fine: true,
            fine_hypotheses: 9,
            fine_span: 2.0,
        }
    }
}

/// 사진별 깊이 추정기 자리: `estimate(기준, 이웃, (가까움, 멂), 설정) -> DepthMap`.
/// 패치매치 추정기도 같은 모양으로 꽂는다.
pub type DepthEstimator = fn(&DepthView, &[&DepthView], (f64, f64), &SweepConfig) -> DepthMap;

type InvDepthAt<'a> = &'a (dyn Fn(usize, usize) -> f32 + Sync);

fn empty_map(w: usize, h: usize) -> DepthMap {
    let n = w * h;
    DepthMap {
        w,
        h,
        depth: vec![f32::NAN; n],
        normal: vec![[0.0; 3]; n],
        cost: vec![f32::INFINITY; n],
    }
}

/// 2×2 상자 평균으로 반해상도 사진을 만든다(색은 비운다).
fn downsample(v: &DepthView) -> DepthView {
    let k = v.camera.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    let (w2, h2) = (w / 2, h / 2);
    let mut gray = vec![0.0f32; w2 * h2];
    let mut valid = vec![false; w2 * h2];
    gray.par_chunks_mut(w2.max(1))
        .zip(valid.par_chunks_mut(w2.max(1)))
        .enumerate()
        .for_each(|(y, (g, va))| {
            for x in 0..w2 {
                let i = 2 * y * w + 2 * x;
                g[x] = 0.25 * (v.gray[i] + v.gray[i + 1] + v.gray[i + w] + v.gray[i + w + 1]);
                va[x] = v.valid[i] && v.valid[i + 1] && v.valid[i + w] && v.valid[i + w + 1];
            }
        });
    let intr = Intrinsics {
        fx: k.fx * 0.5,
        fy: k.fy * 0.5,
        cx: k.cx * 0.5,
        cy: k.cy * 0.5,
        width: w2 as u32,
        height: h2 as u32,
        dist: k.dist,
    };
    DepthView {
        camera: Camera {
            intrinsics: intr,
            pose: v.camera.pose,
        },
        gray,
        rgb: Vec::new(),
        valid,
    }
}

/// 기준 창 평균과 (안정화한) 분산. 가장자리는 (0, 0).
fn ref_stats(r: &DepthView, w: usize, h: usize) -> Vec<(f32, f32)> {
    let mut st = vec![(0.0f32, 0.0f32); w * h];
    let cnt = ((2 * HALF + 1) * (2 * HALF + 1)) as f32;
    st.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < HALF || y + HALF >= h {
            return;
        }
        for (x, o) in row.iter_mut().enumerate() {
            if x < HALF || x + HALF >= w {
                continue;
            }
            let (mut sr, mut srr) = (0.0f32, 0.0f32);
            for dy in 0..=2 * HALF {
                let base = (y + dy - HALF) * w + x - HALF;
                for dx in 0..=2 * HALF {
                    let v = r.gray[base + dx];
                    sr += v;
                    srr += v * v;
                }
            }
            let mr = sr / cnt;
            *o = (mr, srr / cnt - mr * mr + VAR_EPS);
        }
    });
    st
}

/// 화소별 역깊이 가설 `inv_at(화소, 가설)` 에 대한 비용 부피(가설 우선 배치, `hi * n + 화소`).
fn cost_volume(r: &DepthView, nbrs: &[&DepthView], nh: usize, inv_at: InvDepthAt) -> Vec<f32> {
    let k = r.camera.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    let n = w * h;
    let rel: Vec<(crate::math::Matrix3<f64>, Vector3<f64>)> = nbrs
        .iter()
        .map(|nb| {
            let rm = nb.camera.pose.rotation.matrix() * r.camera.pose.rotation.matrix().transpose();
            let t = nb.camera.pose.translation - rm * r.camera.pose.translation;
            (rm, t)
        })
        .collect();
    let rays: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let p = Vector2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            let q = k.to_normalized(&p);
            [q.x, q.y]
        })
        .collect();
    let stats = ref_stats(r, w, h);
    let cnt = ((2 * HALF + 1) * (2 * HALF + 1)) as f32;
    let need = nbrs.len().min(2);
    let mut volume = vec![f32::INFINITY; n * nh];
    let mut warped: Vec<Vec<f32>> = vec![vec![f32::NAN; n]; nbrs.len()];
    for (hi, vol_t) in volume.chunks_mut(n).enumerate() {
        for (j, nb) in nbrs.iter().enumerate() {
            let (rm, t) = &rel[j];
            let nk = nb.camera.intrinsics;
            let (nw, nh_) = (nk.width as usize, nk.height as usize);
            warped[j]
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, o) in row.iter_mut().enumerate() {
                        let i = y * w + x;
                        *o = f32::NAN;
                        let inv = inv_at(i, hi);
                        if !(inv.is_finite() && inv > 0.0) {
                            continue;
                        }
                        let d = 1.0 / inv as f64;
                        let ray = rays[i];
                        let xc = rm * Vector3::new(ray[0] * d, ray[1] * d, d) + t;
                        if xc.z > 1e-6 {
                            let p = nk.to_pixel(&Vector2::new(xc.x / xc.z, xc.y / xc.z));
                            if let Some(v) = bilinear(&nb.gray, nw, nh_, p.x, p.y) {
                                *o = v;
                            }
                        }
                    }
                });
        }
        let warped_ref = &warped;
        let stats = &stats;
        vol_t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                if x < HALF || y < HALF || x + HALF >= w || y + HALF >= h {
                    continue;
                }
                let (mr, vr) = stats[y * w + x];
                let (mut acc, mut m) = (0.0f32, 0usize);
                let (mut m1, mut m2) = (f32::INFINITY, f32::INFINITY);
                for wj in warped_ref {
                    let (mut sw, mut sww, mut srw) = (0.0f32, 0.0f32, 0.0f32);
                    let mut ok = true;
                    'win: for dy in 0..=2 * HALF {
                        let base = (y + dy - HALF) * w + x - HALF;
                        for dx in 0..=2 * HALF {
                            let b = wj[base + dx];
                            if b.is_nan() {
                                ok = false;
                                break 'win;
                            }
                            sw += b;
                            sww += b * b;
                            srw += b * r.gray[base + dx];
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let mw = sw / cnt;
                    let vw = sww / cnt - mw * mw + VAR_EPS;
                    let cov = srw / cnt - mr * mw;
                    let ncc = (cov / (vr * vw).sqrt()).clamp(-1.0, 1.0);
                    let c = 1.0 - ncc;
                    acc += c;
                    m += 1;
                    if c < m1 {
                        m2 = m1;
                        m1 = c;
                    } else if c < m2 {
                        m2 = c;
                    }
                }
                // 이웃 하나만 보이는 화소는 신뢰하지 않는다(이웃이 둘 이상일 때).
                // 이웃이 셋 이상 보이면 가장 좋은 둘의 평균(가려진 이웃에 강건).
                if m >= need && m > 0 {
                    *o = if m >= 3 {
                        0.5 * (m1 + m2)
                    } else {
                        acc / m as f32
                    };
                }
            }
        });
    }
    volume
}

/// 비용 부피에서 최소 가설을 고르고 포물선으로 보정해 깊이·비용 맵을 만든다.
fn select_depth(
    valid: &[bool],
    volume: &[f32],
    nh: usize,
    w: usize,
    h: usize,
    inv_at: InvDepthAt,
    max_cost: f32,
) -> DepthMap {
    let n = w * h;
    let mut map = empty_map(w, h);
    map.depth
        .par_iter_mut()
        .zip(map.cost.par_iter_mut())
        .enumerate()
        .for_each(|(i, (dep, cost))| {
            if !valid[i] {
                return;
            }
            let (mut bi, mut bc) = (usize::MAX, f32::INFINITY);
            for hi in 0..nh {
                let v = volume[hi * n + i];
                if v < bc {
                    bc = v;
                    bi = hi;
                }
            }
            if bi == usize::MAX || bc > max_cost {
                return;
            }
            let mut pos = bi as f64;
            if bi > 0 && bi + 1 < nh {
                let (a, cc) = (volume[(bi - 1) * n + i], volume[(bi + 1) * n + i]);
                if a.is_finite() && cc.is_finite() {
                    let (a, b, cc) = (a as f64, bc as f64, cc as f64);
                    let den = a - 2.0 * b + cc;
                    if den > 1e-9 {
                        pos += (0.5 * (a - cc) / den).clamp(-0.5, 0.5);
                    }
                }
            }
            let (i0, i1) = (inv_at(i, 0) as f64, inv_at(i, 1) as f64);
            let inv = i0 + (i1 - i0) * pos;
            if inv > 0.0 && inv.is_finite() {
                *dep = (1.0 / inv) as f32;
                *cost = bc;
            }
        });
    map
}

/// 이웃 3×3 중앙값과 5% 이상 어긋나는 외톨이를 지운다.
fn remove_outliers(map: &mut DepthMap) {
    let (w, h) = (map.w, map.h);
    let snapshot = map.depth.clone();
    map.depth.par_iter_mut().enumerate().for_each(|(i, d)| {
        if d.is_nan() {
            return;
        }
        let (x, y) = (i % w, i / w);
        if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
            return;
        }
        let mut v: Vec<f32> = Vec::with_capacity(8);
        for dy in 0..3 {
            for dx in 0..3 {
                if dx == 1 && dy == 1 {
                    continue;
                }
                let s = snapshot[(y + dy - 1) * w + x + dx - 1];
                if s.is_finite() {
                    v.push(s);
                }
            }
        }
        if v.len() < 3 {
            *d = f32::NAN;
            return;
        }
        v.sort_by(f32::total_cmp);
        let med = v[v.len() / 2];
        if (*d - med).abs() / med > 0.05 {
            *d = f32::NAN;
        }
    });
}

/// 한 해상도에서 `[near, far]` 전체를 역깊이 균등으로 훑는다(거친-세밀 없이).
fn full_sweep(
    r: &DepthView,
    nbrs: &[&DepthView],
    near: f64,
    far: f64,
    nh: usize,
    max_cost: f32,
) -> DepthMap {
    let k = r.camera.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    let nh = nh.max(2);
    let (inv_near, inv_far) = (1.0 / near, 1.0 / far);
    let step = (inv_far - inv_near) / (nh - 1) as f64;
    let inv_at = move |_i: usize, hi: usize| (inv_near + step * hi as f64) as f32;
    let volume = cost_volume(r, nbrs, nh, &inv_at);
    select_depth(&r.valid, &volume, nh, w, h, &inv_at, max_cost)
}

/// 기준 사진 `r` 의 깊이·법선을 평면 스윕으로 추정한다(기본 추정기).
///
/// 거친 단계는 반해상도에서 `[near, far]` 를 역깊이 균등으로 `cfg.hypotheses` 개 훑고,
/// 세밀 단계는 전체 해상도에서 거친 깊이 둘레 ± `fine_span` 간격만 `fine_hypotheses` 개로 찾는다.
/// 각 가설마다 이웃 사진을 기준 사진으로 되감아 5×5 NCC 비용을 모으고 평균이 가장 낮은 가설을
/// 포물선으로 보정해 고른다. 법선은 깊이 기울기에서 얻는다.
pub fn sweep_depth(
    r: &DepthView,
    nbrs: &[&DepthView],
    range: (f64, f64),
    cfg: &SweepConfig,
) -> DepthMap {
    let (near, far) = range;
    let k = r.camera.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    if nbrs.is_empty() || !(near > 0.0 && far > near) || w < 2 * HALF + 2 || h < 2 * HALF + 2 {
        return empty_map(w, h);
    }
    let mut map = if cfg.coarse_to_fine && w.max(h) >= MIN_COARSE_LONG_SIDE {
        let rc = downsample(r);
        let nc: Vec<DepthView> = nbrs.iter().map(|nb| downsample(nb)).collect();
        let ncr: Vec<&DepthView> = nc.iter().collect();
        let nhc = cfg.hypotheses.max(2);
        let coarse = full_sweep(&rc, &ncr, near, far, nhc, COARSE_MAX_COST);
        let (wc, hc) = (coarse.w, coarse.h);
        let nf = cfg.fine_hypotheses.max(3) | 1;
        // 역깊이 간격(가까울수록 큰 역깊이라 far 쪽이 음수 방향).
        let cstep = ((1.0 / far - 1.0 / near) / (nhc - 1) as f64) as f32;
        let fstep = cstep * cfg.fine_span / ((nf - 1) / 2) as f32;
        let cd = &coarse.depth;
        let inv_at = move |i: usize, hi: usize| {
            let (x, y) = ((i % w) / 2, (i / w) / 2);
            let (x, y) = (x.min(wc - 1), y.min(hc - 1));
            let mut d = cd[y * wc + x];
            if !(d.is_finite() && d > 0.0) {
                // 거친 깊이가 비면 3×3 안에서 가장 가까운(큰 역깊이 아닌) 값을 빌린다.
                'nb: for dy in y.saturating_sub(1)..(y + 2).min(hc) {
                    for dx in x.saturating_sub(1)..(x + 2).min(wc) {
                        let c = cd[dy * wc + dx];
                        if c.is_finite() && c > 0.0 {
                            d = c;
                            break 'nb;
                        }
                    }
                }
            }
            if d.is_finite() && d > 0.0 {
                1.0 / d + fstep * (hi as f32 - (nf / 2) as f32)
            } else {
                f32::NAN
            }
        };
        let volume = cost_volume(r, nbrs, nf, &inv_at);
        select_depth(&r.valid, &volume, nf, w, h, &inv_at, MAX_COST)
    } else {
        full_sweep(r, nbrs, near, far, cfg.hypotheses, MAX_COST)
    };
    remove_outliers(&mut map);
    fill_normals(&mut map, &k);
    map
}

/// 깊이 기울기에서 카메라 좌표 법선을 구한다(카메라를 향하게). 이웃 깊이가 없으면 0 벡터.
fn fill_normals(map: &mut DepthMap, k: &Intrinsics) {
    let (w, h) = (map.w, map.h);
    let pt = |depth: &[f32], x: usize, y: usize| -> Option<Vector3<f64>> {
        let d = depth[y * w + x];
        if !(d.is_finite() && d > 0.0) {
            return None;
        }
        let q = k.to_normalized(&Vector2::new(x as f64 + 0.5, y as f64 + 0.5));
        let d = d as f64;
        Some(Vector3::new(q.x * d, q.y * d, d))
    };
    let depth = map.depth.clone();
    map.normal
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = [0.0; 3];
                if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                    continue;
                }
                let (Some(l), Some(rr), Some(u), Some(dn)) = (
                    pt(&depth, x - 1, y),
                    pt(&depth, x + 1, y),
                    pt(&depth, x, y - 1),
                    pt(&depth, x, y + 1),
                ) else {
                    continue;
                };
                let mut nrm = (rr - l).cross(&(dn - u));
                let len = nrm.norm();
                if !(len.is_finite() && len > 1e-12) {
                    continue;
                }
                nrm /= len;
                // 점에서 카메라로 향하는 쪽(−시선)으로 맞춘다.
                let c = pt(&depth, x, y).unwrap_or(l);
                if nrm.dot(&c) > 0.0 {
                    nrm = -nrm;
                }
                *o = [nrm.x as f32, nrm.y as f32, nrm.z as f32];
            }
        });
}

/// 구역 하나의 밀집 점군: 사진들과 희소 점으로 깊이를 구해 융합한다.
///
/// 희소 점의 관측 사진은 점이 화면 안·카메라 앞에 투영되는 사진으로 잡는다.
pub fn region_cloud(
    views: &[DenseView],
    sparse_points: &[[f64; 3]],
    cfg: &DenseConfig,
) -> PointCloud {
    region_cloud_with(
        views,
        sparse_points,
        cfg,
        sweep_depth,
        &SweepConfig::default(),
    )
}

/// 패치매치 깊이 추정기를 [`DepthEstimator`] 모양으로 감싼 것(설정은 기본값).
pub fn patchmatch_depth(
    r: &DepthView,
    nbrs: &[&DepthView],
    range: (f64, f64),
    _sweep: &SweepConfig,
) -> DepthMap {
    use crate::patchmatch as pm;
    let to_view = |v: &DepthView| {
        let k = v.camera.intrinsics;
        let data: Vec<f32> = v.gray.iter().map(|g| g / 255.0).collect();
        pm::View {
            camera: v.camera,
            image: pm::GrayImage::new(k.width as usize, k.height as usize, data),
        }
    };
    let rv = to_view(r);
    let nv: Vec<pm::View> = nbrs.iter().map(|n| to_view(n)).collect();
    let m = pm::estimate(&rv, &nv, range, &pm::Config::default());
    DepthMap {
        w: m.w,
        h: m.h,
        depth: m.depth,
        normal: m.normal,
        cost: m.cost,
    }
}

/// [`region_cloud`] 의 패치매치판: 사진마다 이웃 `cfg.neighbors` 장 전부로 패치매치 깊이를 구해 융합한다.
pub fn region_cloud_patchmatch(
    views: &[DenseView],
    sparse_points: &[[f64; 3]],
    cfg: &DenseConfig,
) -> PointCloud {
    region_cloud_impl(
        views,
        sparse_points,
        cfg,
        patchmatch_depth,
        &SweepConfig::default(),
        cfg.neighbors.max(1),
    )
}

/// [`region_cloud`] 와 같고, 사진별 깊이 추정기와 그 설정을 고를 수 있다.
pub fn region_cloud_with(
    views: &[DenseView],
    sparse_points: &[[f64; 3]],
    cfg: &DenseConfig,
    estimate: DepthEstimator,
    sweep: &SweepConfig,
) -> PointCloud {
    region_cloud_impl(views, sparse_points, cfg, estimate, sweep, SWEEP_NEIGHBORS)
}

fn region_cloud_impl(
    views: &[DenseView],
    sparse_points: &[[f64; 3]],
    cfg: &DenseConfig,
    estimate: DepthEstimator,
    sweep: &SweepConfig,
    take_nbrs: usize,
) -> PointCloud {
    // 1. 왜곡 보정·축소. 실패한 사진은 건너뛴다(빈 깊이 맵).
    let prepared: Vec<Option<DepthView>> = views
        .par_iter()
        .map(|v| prepare(v, cfg.max_width))
        .collect();
    let ok: Vec<usize> = (0..views.len())
        .filter(|&i| prepared[i].is_some())
        .collect();
    if ok.len() < 2 {
        return PointCloud::default();
    }
    let preps: Vec<&DepthView> = ok.iter().map(|&i| prepared[i].as_ref().unwrap()).collect();

    // 희소 점 관측자.
    let sparse: Vec<SparsePoint> = sparse_points
        .iter()
        .filter(|p| p.iter().all(|v| v.is_finite()))
        .map(|p| {
            let xyz = Point3::new(p[0], p[1], p[2]);
            let observers = preps
                .iter()
                .enumerate()
                .filter(|(_, v)| {
                    v.camera
                        .project(&xyz)
                        .is_some_and(|q| v.camera.intrinsics.contains(&q))
                })
                .map(|(i, _)| i)
                .collect();
            SparsePoint { xyz, observers }
        })
        .collect();
    let vs: Vec<view_selection::View> = preps
        .iter()
        .enumerate()
        .map(|(i, v)| view_selection::View {
            cam: v.camera,
            id: i,
        })
        .collect();

    // 2. 이웃 선택·깊이 범위.
    let neighbors = view_selection::select_neighbors(&vs, &sparse, cfg.neighbors);

    // 3. 사진별 깊이·법선.
    let maps: Vec<DepthMap> = (0..preps.len())
        .into_par_iter()
        .map(|i| {
            let (near, far) = view_selection::depth_range(&vs[i], &sparse);
            let nb: Vec<&DepthView> = neighbors[i]
                .iter()
                .take(take_nbrs)
                .map(|&j| preps[j])
                .collect();
            estimate(preps[i], &nb, (near, far), sweep)
        })
        .collect();

    // 4. 융합.
    let fviews: Vec<FusionView> = preps
        .iter()
        .enumerate()
        .map(|(i, v)| FusionView {
            camera: v.camera,
            rgb: v.rgb.clone(),
            neighbors: neighbors[i].clone(),
            group: None,
        })
        .collect();
    let fcfg = FusionConfig {
        reproj_px: cfg.reproj_px,
        depth_rel: cfg.depth_rel,
        min_views: cfg.min_views.max(1),
        ..FusionConfig::default()
    };
    fusion::try_fuse(&fviews, &maps, fcfg).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{Scene, SceneConfig};
    use std::time::Instant;

    /// 정답 표면까지 근사 거리: 수직 차와 건물 상자까지 거리 중 작은 값.
    fn surface_dist(s: &Scene, p: &[f32; 3]) -> f64 {
        let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
        let mut d = (z - s.surface_height(x, y)).abs();
        for b in &s.buildings {
            let dx = (b.min.x - x).max(x - b.max.x).max(0.0);
            let dy = (b.min.y - y).max(y - b.max.y).max(0.0);
            let dz = (z - b.top).max(0.0);
            d = d.min((dx * dx + dy * dy + dz * dz).sqrt());
        }
        d
    }

    fn scene_views(positions: usize, w: u32, h: u32) -> (Scene, Vec<DenseView>, Vec<[f64; 3]>) {
        let s = Scene::new(SceneConfig {
            positions,
            width: w,
            height: h,
            ..SceneConfig::default()
        });
        let mut views = Vec::new();
        let mut sparse = Vec::new();
        for (vi, v) in s.views.iter().enumerate() {
            let (img, depth) = s.render(v);
            let rgb = RgbImage::from_raw(img.width, img.height, img.data).unwrap();
            if vi % 2 == 0 {
                for y in (4..h as usize).step_by(9) {
                    for x in (4..w as usize).step_by(9) {
                        let d = depth[y * w as usize + x];
                        if d.is_finite() {
                            let p = v
                                .camera
                                .unproject(&Vector2::new(x as f64 + 0.5, y as f64 + 0.5), d as f64);
                            sparse.push([p.x, p.y, p.z]);
                        }
                    }
                }
            }
            views.push(DenseView {
                camera: v.camera,
                image: rgb,
            });
        }
        (s, views, sparse)
    }

    #[test]
    fn region_cloud_matches_truth_surface() {
        let (s, views, sparse) = scene_views(4, 192, 108);
        let cfg = DenseConfig {
            max_width: 192,
            ..DenseConfig::default()
        };
        let t = Instant::now();
        let cloud = region_cloud(&views, &sparse, &cfg);
        let secs = t.elapsed().as_secs_f64();
        assert!(!cloud.has_nan());
        let mut d: Vec<f64> = cloud
            .points
            .iter()
            .map(|p| surface_dist(&s, &p.xyz))
            .collect();
        d.sort_by(f64::total_cmp);
        let q = |f: f64| d.get(((d.len() as f64 * f) as usize).min(d.len().saturating_sub(1)));
        eprintln!(
            "dense: views {} points {} median {:?} p90 {:?} secs {secs:.2}",
            views.len(),
            cloud.len(),
            q(0.5),
            q(0.9)
        );
        assert!(cloud.len() > 20000, "점 수 {}", cloud.len());
        assert!(*q(0.5).unwrap() < 0.15, "중앙값 {:?}", q(0.5));
        assert!(*q(0.9).unwrap() < 0.5, "90% {:?}", q(0.9));
        assert!(secs < 60.0, "시간 {secs}");
    }

    fn quantiles(s: &Scene, cloud: &PointCloud) -> (f64, f64) {
        let mut d: Vec<f64> = cloud
            .points
            .iter()
            .map(|p| surface_dist(s, &p.xyz))
            .collect();
        d.sort_by(f64::total_cmp);
        let q = |f: f64| d[((d.len() as f64 * f) as usize).min(d.len() - 1)];
        (q(0.5), q(0.9))
    }

    /// 편대 장면 48장(16 위치 × 3 드론), 긴 변 480 화소: 점 수·표면 거리·시간.
    #[test]
    fn formation_region_48_images() {
        let (s, views, sparse) = scene_views(16, 480, 270);
        assert_eq!(views.len(), 48);
        let cfg = DenseConfig {
            max_width: 480,
            ..DenseConfig::default()
        };
        let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        let t = Instant::now();
        let cloud = region_cloud(&views, &sparse, &cfg);
        let secs = t.elapsed().as_secs_f64();
        assert!(!cloud.has_nan());
        assert!(!cloud.is_empty());
        let (med, p90) = quantiles(&s, &cloud);
        eprintln!(
            "dense48: images {} points {} median {med:.4} p90 {p90:.4} secs {secs:.2} \
             ({:.3} s/image) cores {} load {}",
            views.len(),
            cloud.len(),
            secs / views.len() as f64,
            std::thread::available_parallelism().map_or(0, |n| n.get()),
            load.trim()
        );
        assert!(cloud.len() > POINTS48, "점 수 {}", cloud.len());
        assert!(med < MEDIAN48, "중앙값 {med}");
        assert!(p90 < P90_48, "90% {p90}");
        assert!(secs < SECS48, "시간 {secs}");
    }

    #[test]
    fn coarse_to_fine_vs_full_sweep() {
        let (s, views, sparse) = scene_views(4, 192, 108);
        let cfg = DenseConfig {
            max_width: 192,
            ..DenseConfig::default()
        };
        for ctf in [false, true] {
            let sw = SweepConfig {
                coarse_to_fine: ctf,
                ..SweepConfig::default()
            };
            let t = Instant::now();
            let cloud = region_cloud_with(&views, &sparse, &cfg, sweep_depth, &sw);
            let (med, p90) = quantiles(&s, &cloud);
            eprintln!(
                "ctf {ctf}: points {} median {med:.4} p90 {p90:.4} secs {:.2}",
                cloud.len(),
                t.elapsed().as_secs_f64()
            );
        }
    }

    const POINTS48: usize = 300_000;
    const MEDIAN48: f64 = 0.05;
    const P90_48: f64 = 0.15;
    const SECS48: f64 = 120.0;

    #[test]
    fn too_few_views_gives_empty_cloud() {
        let (_, views, sparse) = scene_views(1, 64, 36);
        let cloud = region_cloud(&views[..1], &sparse, &DenseConfig::default());
        assert!(cloud.is_empty());
    }
}
