//! 구역 밀집 복원(SPEC §3.6): 왜곡 보정·축소 → 이웃·깊이 범위 → 사진별 깊이·법선 → 융합.
//!
//! 사진별 깊이 단계는 임시 평면 스윕([`sweep_depth`])이다. 깊이 추정 모듈이 합쳐지면
//! [`estimate_depth`] 안의 호출 한 줄만 바꾸면 된다.

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
/// NCC 분산 안정화 항(회색 0..255 단위의 분산).
const VAR_EPS: f32 = 25.0;

struct Prepared {
    camera: Camera,
    gray: Vec<f32>,
    rgb: Vec<[u8; 3]>,
    valid: Vec<bool>,
}

fn prepare(v: &DenseView, max_width: u32) -> Option<Prepared> {
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
    Some(Prepared {
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

/// 기준 사진 `r` 의 깊이·법선을 임시 평면 스윕으로 추정한다.
///
/// 깊이 가설은 `[near, far]` 를 역깊이 균등으로 [`SWEEP_HYPOTHESES`] 개 나누고,
/// 각 가설마다 이웃 사진(상위 [`SWEEP_NEIGHBORS`])을 기준 사진으로 되감아 5×5 NCC 를 모아
/// 평균 비용이 가장 낮은 가설을 포물선으로 보정해 고른다. 법선은 깊이 기울기에서 얻는다.
fn sweep_depth(r: &Prepared, nbrs: &[&Prepared], near: f64, far: f64) -> DepthMap {
    let k = r.camera.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    let n = w * h;
    let mut map = DepthMap {
        w,
        h,
        depth: vec![f32::NAN; n],
        normal: vec![[0.0; 3]; n],
        cost: vec![f32::INFINITY; n],
    };
    if nbrs.is_empty() || !(near > 0.0 && far > near) || w < 2 * HALF + 2 || h < 2 * HALF + 2 {
        return map;
    }
    let nh = SWEEP_HYPOTHESES;
    let (inv_near, inv_far) = (1.0 / near, 1.0 / far);
    let hyp_depth = |i: usize| 1.0 / (inv_near + (inv_far - inv_near) * i as f64 / (nh - 1) as f64);

    // 기준 → 이웃 상대 자세.
    let rel: Vec<(crate::math::Matrix3<f64>, Vector3<f64>)> = nbrs
        .iter()
        .map(|nb| {
            let rm = nb.camera.pose.rotation.matrix() * r.camera.pose.rotation.matrix().transpose();
            let t = nb.camera.pose.translation - rm * r.camera.pose.translation;
            (rm, t)
        })
        .collect();
    // 기준 화소의 정규 좌표 광선.
    let rays: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let p = Vector2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            let q = k.to_normalized(&p);
            [q.x, q.y]
        })
        .collect();

    // 비용 부피: 화소 × 가설.
    let mut volume = vec![f32::INFINITY; n * nh];
    let mut warped: Vec<Vec<f32>> = vec![vec![f32::NAN; n]; nbrs.len()];
    let mut vol_t = vec![f32::INFINITY; n];
    for hi in 0..nh {
        let d = hyp_depth(hi);
        for (j, nb) in nbrs.iter().enumerate() {
            let (rm, t) = &rel[j];
            let nk = nb.camera.intrinsics;
            let (nw, nh_) = (nk.width as usize, nk.height as usize);
            warped[j]
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, o) in row.iter_mut().enumerate() {
                        let ray = rays[y * w + x];
                        let xc = rm * Vector3::new(ray[0] * d, ray[1] * d, d) + t;
                        *o = f32::NAN;
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
        vol_t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = f32::INFINITY;
                if x < HALF || y < HALF || x + HALF >= w || y + HALF >= h {
                    continue;
                }
                // 기준 창 통계.
                let (mut sr, mut srr) = (0.0f32, 0.0f32);
                for dy in 0..=2 * HALF {
                    let base = (y + dy - HALF) * w + x - HALF;
                    for dx in 0..=2 * HALF {
                        let v = r.gray[base + dx];
                        sr += v;
                        srr += v * v;
                    }
                }
                let cnt = ((2 * HALF + 1) * (2 * HALF + 1)) as f32;
                let mr = sr / cnt;
                let vr = srr / cnt - mr * mr + VAR_EPS;
                let (mut acc, mut m) = (0.0f32, 0usize);
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
                    acc += 1.0 - ncc;
                    m += 1;
                }
                // 이웃 하나만 보이는 화소는 신뢰하지 않는다(이웃이 둘 이상일 때).
                let need = nbrs.len().min(2);
                if m >= need && m > 0 {
                    *o = acc / m as f32;
                }
            }
        });
        for i in 0..n {
            volume[i * nh + hi] = vol_t[i];
        }
    }

    // 최소 가설 선택 + 포물선 보정.
    let dm = &mut map;
    dm.depth
        .par_iter_mut()
        .zip(dm.cost.par_iter_mut())
        .enumerate()
        .for_each(|(i, (dep, cost))| {
            if !r.valid[i] {
                return;
            }
            let c = &volume[i * nh..(i + 1) * nh];
            let (mut bi, mut bc) = (usize::MAX, f32::INFINITY);
            for (hi, &v) in c.iter().enumerate() {
                if v < bc {
                    bc = v;
                    bi = hi;
                }
            }
            if bi == usize::MAX || bc > MAX_COST {
                return;
            }
            let mut pos = bi as f64;
            if bi > 0 && bi + 1 < nh && c[bi - 1].is_finite() && c[bi + 1].is_finite() {
                let (a, b, cc) = (c[bi - 1] as f64, c[bi] as f64, c[bi + 1] as f64);
                let den = a - 2.0 * b + cc;
                if den > 1e-9 {
                    pos += (0.5 * (a - cc) / den).clamp(-0.5, 0.5);
                }
            }
            let inv = inv_near + (inv_far - inv_near) * pos / (nh - 1) as f64;
            *dep = (1.0 / inv) as f32;
            *cost = bc;
        });
    // 이웃 3×3 중앙값과 1% 이상 어긋나는 외톨이 제거.
    let snapshot = dm.depth.clone();
    dm.depth.par_iter_mut().enumerate().for_each(|(i, d)| {
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

/// 사진별 깊이 추정 자리. 깊이 추정 모듈이 합쳐지면 이 함수 안의 호출만 바꾼다.
fn estimate_depth(r: &Prepared, nbrs: &[&Prepared], near: f64, far: f64) -> DepthMap {
    sweep_depth(r, nbrs, near, far)
}

/// 구역 하나의 밀집 점군: 사진들과 희소 점으로 깊이를 구해 융합한다.
///
/// 희소 점의 관측 사진은 점이 화면 안·카메라 앞에 투영되는 사진으로 잡는다.
pub fn region_cloud(
    views: &[DenseView],
    sparse_points: &[[f64; 3]],
    cfg: &DenseConfig,
) -> PointCloud {
    // 1. 왜곡 보정·축소. 실패한 사진은 건너뛴다(빈 깊이 맵).
    let prepared: Vec<Option<Prepared>> = views
        .par_iter()
        .map(|v| prepare(v, cfg.max_width))
        .collect();
    let ok: Vec<usize> = (0..views.len())
        .filter(|&i| prepared[i].is_some())
        .collect();
    if ok.len() < 2 {
        return PointCloud::default();
    }
    let preps: Vec<&Prepared> = ok.iter().map(|&i| prepared[i].as_ref().unwrap()).collect();

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
        .map(|i| {
            let (near, far) = view_selection::depth_range(&vs[i], &sparse);
            let nb: Vec<&Prepared> = neighbors[i]
                .iter()
                .take(SWEEP_NEIGHBORS)
                .map(|&j| preps[j])
                .collect();
            estimate_depth(preps[i], &nb, near, far)
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

    #[test]
    fn too_few_views_gives_empty_cloud() {
        let (_, views, sparse) = scene_views(1, 64, 36);
        let cloud = region_cloud(&views[..1], &sparse, &DenseConfig::default());
        assert!(cloud.is_empty());
    }
}
