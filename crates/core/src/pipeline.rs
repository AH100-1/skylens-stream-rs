//! 끝까지 잇는 흐름: 이미지 폴더(+GPS) → 특징 → 매칭 → 회전 평균 → 위치 → 삼각측량 → 번들 조정
//! → 구역별 초벌/정밀 → 밀집 깊이 → 융합 → 정렬·스냅샷·manifest.
//!
//! 아직 합치지 않은 부품(트랙, 위치 평균·삼각측량, PatchMatch)은 [`stand_in`] 의 단순 구현으로 잇는다.
//! 해당 브랜치가 병합되면 `run_region` 안의 호출 한 줄씩만 바꾸면 된다.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::align::Similarity;
use crate::ba::{bundle_adjust, BaOptions, BaProblem, Observation, PositionPrior};
use crate::camera::{Camera, Intrinsics, Pose};
use crate::dataset::Dataset;
use crate::features::{detect_and_describe, DetectorConfig, Feature, GrayImage};
use crate::fusion::{fuse, FusionConfig, FusionView};
use crate::matching::{ratio_match, RansacConfig, PAIR_TEMPORAL};
use crate::math::{Matrix3, Point3, Rotation3, Vector2, Vector3};
use crate::ply::{PointCloud, PointRecord};
use crate::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};
use crate::stream::{
    align_region, align_window, apply_alignments, point_pairs, split_regions, write_outputs,
    AlignRecord, Region, Track,
};
use crate::two_view::{ransac_essential, recover_pose};

/// 실행 설정.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub max_features: usize,
    /// 밀집 깊이 맵 폭(px).
    pub dense_width: usize,
    /// 수평 화각(도). 데이터셋에 내부 파라미터가 없으므로 받는다.
    pub hfov_deg: f64,
    pub ba_iters: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_features: 1500,
            dense_width: 160,
            hfov_deg: 65.0,
            ba_iters: 15,
        }
    }
}

/// 구역 하나의 수치.
#[derive(Clone, Debug, Default)]
pub struct RegionStats {
    pub region: usize,
    pub positions: usize,
    pub images: usize,
    pub registered: usize,
    pub tracks: usize,
    pub preview_rms: f64,
    pub refined_rms: f64,
    pub preview_points: usize,
    pub refined_points: usize,
    pub secs_features: f64,
    pub secs_matching: f64,
    pub secs_sparse: f64,
    pub secs_ba: f64,
    pub secs_dense: f64,
}

/// 실행 결과.
#[derive(Clone, Debug, Default)]
pub struct PipelineResult {
    pub regions: Vec<RegionStats>,
    pub issues: Vec<String>,
    pub align: Vec<AlignRecord>,
    /// (사진 이름, 정밀 카메라 중심 동-북-위 m).
    pub centers: Vec<(String, [f64; 3])>,
}

struct ImgData {
    rgb: image::RgbImage,
    feats: Vec<Feature>,
}

/// 초벌(BA 전) 또는 정밀(BA 후) 희소 모델. 사진 번호는 구역 안 번호.
#[derive(Clone)]
struct Sparse {
    poses: Vec<Option<Pose>>,
    points: Vec<Vector3<f64>>,
    /// 점마다 (구역 안 사진 번호, 특징 번호, 픽셀) 관측.
    obs: Vec<Vec<(usize, usize, Vector2<f64>)>>,
    rms: f64,
}

/// (사진 i, 사진 j, 특징 짝 목록).
type PairList = (usize, usize, Vec<(usize, usize)>);

pub mod stand_in {
    //! 병합 전 부품의 단순 대체. 모양은 TASKS 인터페이스를 따른다.
    use super::*;

    /// 트랙: 합집합-찾기, 같은 사진이 두 번 든 성분은 버린다. 반환은 성분별 (사진, 특징) 목록.
    /// `feat_counts[i]` = 사진 i 의 특징 수, `matches` = (i, j, [(특징 i, 특징 j)]).
    pub fn build_tracks(feat_counts: &[usize], matches: &[PairList]) -> Vec<Vec<(usize, usize)>> {
        let mut off = vec![0usize; feat_counts.len() + 1];
        for (i, c) in feat_counts.iter().enumerate() {
            off[i + 1] = off[i] + c;
        }
        let mut parent: Vec<usize> = (0..off[feat_counts.len()]).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for (i, j, m) in matches {
            for &(a, b) in m {
                let (ra, rb) = (
                    find(&mut parent, off[*i] + a),
                    find(&mut parent, off[*j] + b),
                );
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
        let mut comps: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (i, j, m) in matches {
            for &(a, b) in m {
                for (img, f) in [(*i, a), (*j, b)] {
                    let r = find(&mut parent, off[img] + f);
                    let e = comps.entry(r).or_default();
                    if !e.contains(&(img, f)) {
                        e.push((img, f));
                    }
                }
            }
        }
        let mut out: Vec<Vec<(usize, usize)>> = comps
            .into_values()
            .filter(|c| {
                let mut imgs: Vec<usize> = c.iter().map(|o| o.0).collect();
                imgs.sort_unstable();
                let n = imgs.len();
                imgs.dedup();
                n >= 2 && imgs.len() == n
            })
            .collect();
        for c in &mut out {
            c.sort_unstable();
        }
        out.sort();
        out
    }

    /// 위치: 방향 제약 (I − ddᵀ)(C_j − C_i) = 0 과 GPS 사전(가중 `prior`)의 선형 최소제곱.
    /// `dirs` = (i, j, 세계 좌표 단위 방향 C_i → C_j). 중심이 정해지지 않으면 None.
    pub fn solve_centers(
        gps: &[Vector3<f64>],
        dirs: &[(usize, usize, Vector3<f64>)],
        prior: f64,
    ) -> Option<Vec<Vector3<f64>>> {
        let n = gps.len();
        let mut h = DMatrix::<f64>::zeros(3 * n, 3 * n);
        let mut rhs = DMatrix::<f64>::zeros(3 * n, 1);
        for &(i, j, d) in dirs {
            let m = Matrix3::identity() - d * d.transpose();
            for r in 0..3 {
                for c in 0..3 {
                    h[(3 * i + r, 3 * i + c)] += m[(r, c)];
                    h[(3 * j + r, 3 * j + c)] += m[(r, c)];
                    h[(3 * i + r, 3 * j + c)] -= m[(r, c)];
                    h[(3 * j + r, 3 * i + c)] -= m[(r, c)];
                }
            }
        }
        for (i, g) in gps.iter().enumerate() {
            for r in 0..3 {
                h[(3 * i + r, 3 * i + r)] += prior * prior;
                rhs[(3 * i + r, 0)] += prior * prior * g[r];
            }
        }
        let x = h.lu().solve(&rhs)?;
        Some(
            (0..n)
                .map(|i| Vector3::new(x[(3 * i, 0)], x[(3 * i + 1, 0)], x[(3 * i + 2, 0)]))
                .collect(),
        )
    }

    /// 점-광선 최소제곱 삼각측량. 반환: 점(깊이가 모두 양수, 재투영 `max_px` 이하일 때만).
    pub fn triangulate_track(cams: &[(Camera, Vector2<f64>)], max_px: f64) -> Option<Vector3<f64>> {
        let mut a = Matrix3::zeros();
        let mut b = Vector3::zeros();
        for (cam, px) in cams {
            let n = cam.intrinsics.to_normalized(px);
            let u = (cam.pose.rotation.inverse() * Vector3::new(n.x, n.y, 1.0)).normalize();
            let m = Matrix3::identity() - u * u.transpose();
            a += m;
            b += m * cam.pose.center().coords;
        }
        let x = a.lu().solve(&b)?;
        for (cam, px) in cams {
            let q = cam.project(&Point3::from(x))?;
            if (q - px).norm().partial_cmp(&max_px) != Some(std::cmp::Ordering::Less) {
                return None;
            }
        }
        Some(x)
    }

    /// 강건 다시점 삼각측량: 재투영이 가장 나쁜 관측을 하나씩 버리며(2개까지) 모든 관측이
    /// `max_px` 안에 들고 광선 최대 각이 `min_deg` 이상일 때만 돌려준다. 반환: 점, 남긴 관측 표시.
    pub fn triangulate_robust(
        cams: &[(Camera, Vector2<f64>)],
        max_px: f64,
        min_deg: f64,
    ) -> Option<(Vector3<f64>, Vec<bool>)> {
        let mut keep = vec![true; cams.len()];
        loop {
            let idx: Vec<usize> = (0..cams.len()).filter(|&i| keep[i]).collect();
            if idx.len() < 2 {
                return None;
            }
            let sub: Vec<(Camera, Vector2<f64>)> = idx.iter().map(|&i| cams[i]).collect();
            let mut a = Matrix3::zeros();
            let mut b = Vector3::zeros();
            for (cam, px) in &sub {
                let n = cam.intrinsics.to_normalized(px);
                let u = (cam.pose.rotation.inverse() * Vector3::new(n.x, n.y, 1.0)).normalize();
                let m = Matrix3::identity() - u * u.transpose();
                a += m;
                b += m * cam.pose.center().coords;
            }
            let x = a.lu().solve(&b)?;
            let errs: Vec<f64> = sub
                .iter()
                .map(|(cam, px)| {
                    cam.project(&Point3::from(x))
                        .map_or(f64::INFINITY, |q| (q - px).norm())
                })
                .collect();
            let (worst, we) = errs
                .iter()
                .copied()
                .enumerate()
                .fold((0, 0.0), |m, (i, e)| if e > m.1 { (i, e) } else { m });
            if we.partial_cmp(&max_px) == Some(std::cmp::Ordering::Less) {
                let rays: Vec<Vector3<f64>> = sub
                    .iter()
                    .map(|(c, _)| (x - c.pose.center().coords).normalize())
                    .collect();
                let mut ang = 0.0f64;
                for i in 0..rays.len() {
                    for j in i + 1..rays.len() {
                        ang = ang.max(rays[i].dot(&rays[j]).clamp(-1.0, 1.0).acos());
                    }
                }
                return (ang.to_degrees() >= min_deg).then_some((x, keep));
            }
            keep[idx[worst]] = false;
        }
    }

    /// 희소 점 보간 깊이 맵(역거리 가중, 반경 밖은 빈 화소). `cam` 은 깊이 맵 해상도의 카메라.
    pub fn depth_from_sparse(cam: &Camera, points: &[Vector3<f64>]) -> crate::fusion::DepthMap {
        let (w, h) = (
            cam.intrinsics.width as usize,
            cam.intrinsics.height as usize,
        );
        let proj: Vec<(f64, f64, f64)> = points
            .iter()
            .filter_map(|p| {
                let z = cam.pose.transform(&Point3::from(*p)).z;
                let q = cam.project(&Point3::from(*p))?;
                (q.x >= 0.0 && q.y >= 0.0 && q.x < w as f64 && q.y < h as f64)
                    .then_some((q.x, q.y, z))
            })
            .collect();
        let radius = (w as f64 * 0.1).max(3.0);
        let mut depth = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let (mut sw, mut sd) = (0.0, 0.0);
                for &(qx, qy, z) in &proj {
                    let d2 = (qx - px).powi(2) + (qy - py).powi(2);
                    if d2 < radius * radius {
                        let wt = 1.0 / (d2 + 1.0);
                        sw += wt;
                        sd += wt * z;
                    }
                }
                if sw > 0.0 {
                    depth[y * w + x] = (sd / sw) as f32;
                }
            }
        }
        crate::fusion::DepthMap {
            w,
            h,
            depth,
            normal: Vec::new(),
            cost: vec![0.0; w * h],
        }
    }
}

fn load(path: &Path, max_features: usize) -> Result<ImgData, String> {
    let rgb = image::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .to_rgb8();
    let g = GrayImage::from_rgb(rgb.width() as usize, rgb.height() as usize, rgb.as_raw());
    let feats = detect_and_describe(
        &g,
        &DetectorConfig {
            max_features,
            ..DetectorConfig::default()
        },
    );
    Ok(ImgData { rgb, feats })
}

fn norm(k: &Intrinsics, f: &Feature) -> Vector2<f64> {
    k.index_to_normalized(&Vector2::new(f.kp.x as f64, f.kp.y as f64))
}

struct PairMatch {
    i: usize,
    j: usize,
    inl: Vec<(usize, usize)>,
    rot: Rotation3<f64>,
    t: Option<Vector3<f64>>,
}

/// 편대 겹침(FEEDBACK F-197)에 맞춘 짝 목록. `views[k] = (카메라 번호 F=0 R=1 L=2, 위치 번호)`.
/// 같은 카메라는 위치 차 1..=PAIR_TEMPORAL, 같은 위치 근처(차 <= 2)의 다른 카메라,
/// 카메라 사이는 F(p)–R(p+12..=p+40), F(p)–L(p+16..=p+40) 을 `CROSS_STEP` 간격으로 표본한다.
/// 결과 (i, j) 는 i < j, 중복 없음, 정렬됨.
pub fn formation_pairs(views: &[(usize, usize)]) -> Vec<(usize, usize)> {
    const CROSS_STEP: i64 = 4;
    let mut out = Vec::new();
    for i in 0..views.len() {
        for j in 0..views.len() {
            if i == j {
                continue;
            }
            let ((ca, pa), (cb, pb)) = (views[i], views[j]);
            let dist = pa.abs_diff(pb);
            let ok = if ca == cb {
                j > i && (1..=PAIR_TEMPORAL).contains(&dist)
            } else if dist <= 2 {
                j > i
            } else if ca == 0 {
                let d = pb as i64 - pa as i64;
                let lo = if cb == 1 { 12 } else { 16 };
                (lo..=40).contains(&d) && (d - lo) % CROSS_STEP == 0
            } else {
                false
            };
            if ok {
                out.push((i.min(j), i.max(j)));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn match_pairs(imgs: &[&ImgData], views: &[(usize, usize)], k: &Intrinsics) -> Vec<PairMatch> {
    let pairs = formation_pairs(views);
    pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let (fa, fb) = (&imgs[i].feats, &imgs[j].feats);
            let m = ratio_match(fa, fb, 0.8, true);
            if m.len() < 20 {
                return None;
            }
            let n1: Vec<_> = m.iter().map(|&(a, _)| norm(k, &fa[a])).collect();
            let n2: Vec<_> = m.iter().map(|&(_, b)| norm(k, &fb[b])).collect();
            let cfg = RansacConfig {
                max_iters: 500,
                ..RansacConfig::default()
            };
            let (e, inl) = ransac_essential(&n1, &n2, k.fx, &cfg)?;
            let sel = |n: &[Vector2<f64>]| -> Vec<Vector2<f64>> {
                n.iter()
                    .zip(&inl)
                    .filter(|(_, &b)| b)
                    .map(|(x, _)| *x)
                    .collect()
            };
            let (s1, s2) = (sel(&n1), sel(&n2));
            if s1.len() < 20 {
                return None;
            }
            let rp = recover_pose(&e, &s1, &s2)?;
            let inl: Vec<(usize, usize)> = m
                .iter()
                .zip(&inl)
                .filter(|(_, &b)| b)
                .map(|(x, _)| *x)
                .collect();
            Some(PairMatch {
                i,
                j,
                inl,
                rot: rp.rotation,
                t: rp.translation_observable.then_some(rp.translation),
            })
        })
        .collect()
}

/// 회전 평균 → 방향으로 좌표계 맞춤 → 위치 → 삼각측량. 초벌 희소 모델.
fn sparse_init(
    imgs: &[&ImgData],
    pm: &[PairMatch],
    gps: &[Vector3<f64>],
    k: &Intrinsics,
) -> Result<Sparse, String> {
    let n = imgs.len();
    let edges: Vec<RelativeRotation> = pm
        .iter()
        .map(|p| RelativeRotation {
            i: p.i,
            j: p.j,
            rotation: p.rot,
            weight: p.inl.len() as f64,
        })
        .collect();
    let ra = average_rotations(n, &edges, &AveragingConfig::default())
        .ok_or("회전 평균 실패: 쓸 수 있는 간선 없음")?;
    let rots = ra.rotations;
    // 좌표계 맞춤(Kabsch): 모델 방향 b = Rᵢᵀ(−R_ijᵀ t) → GPS 방향.
    let mut h = Matrix3::zeros();
    let mut dirs_model = Vec::new();
    for p in pm {
        let (Some(ri), Some(_), Some(t)) = (rots[p.i], rots[p.j], p.t) else {
            continue;
        };
        let b = ri.inverse() * (-(p.rot.inverse() * t));
        dirs_model.push((p.i, p.j, b));
        let dg = gps[p.j] - gps[p.i];
        if dg.norm() > 3.0 {
            h += b * dg.normalize().transpose() * dg.norm();
        }
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.ok_or("SVD")?, svd.v_t.ok_or("SVD")?);
    let d = (vt.transpose() * u.transpose()).determinant().signum();
    let mut g = vt.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose();
    // 편대가 거의 한 직선으로 날면 Kabsch 는 직선 둘레 회전을 못 정한다: 비행 축 둘레 회전을
    // 카메라가 아래를 보는 쪽(보는 방향의 평균 z 가 가장 작은 쪽)으로 고른다.
    let valid: Vec<Rotation3<f64>> = rots.iter().flatten().copied().collect();
    let mut axis = Vector3::zeros();
    let mut first: Option<Vector3<f64>> = None;
    for p in pm {
        let dg = gps[p.j] - gps[p.i];
        if dg.norm() > 3.0 {
            let dn = dg.normalize();
            let r = *first.get_or_insert(dn);
            axis += if dn.dot(&r) >= 0.0 { dn } else { -dn };
        }
    }
    if axis.norm() > 1e-9 && !valid.is_empty() {
        let axis = nalgebra::Unit::new_normalize(axis);
        let mut best = (f64::INFINITY, g);
        for step in 0..180 {
            let gm =
                *Rotation3::from_axis_angle(&axis, step as f64 * 2f64.to_radians()).matrix() * g;
            let down: f64 = valid
                .iter()
                .map(|r| (gm * (r.inverse() * Vector3::z())).z)
                .sum::<f64>()
                / valid.len() as f64;
            if down < best.0 {
                best = (down, gm);
            }
        }
        g = best.1;
    }
    let ids: Vec<usize> = (0..n).filter(|&i| rots[i].is_some()).collect();
    let loc: HashMap<usize, usize> = ids.iter().enumerate().map(|(a, &i)| (i, a)).collect();
    let dirs: Vec<(usize, usize, Vector3<f64>)> = dirs_model
        .iter()
        .filter(|(i, j, _)| loc.contains_key(i) && loc.contains_key(j))
        .map(|&(i, j, b)| (loc[&i], loc[&j], (g * b).normalize()))
        .collect();
    let gl: Vec<Vector3<f64>> = ids.iter().map(|&i| gps[i]).collect();
    let centers = stand_in::solve_centers(&gl, &dirs, 0.2).ok_or("위치 풀이 실패")?;
    let mut poses: Vec<Option<Pose>> = vec![None; n];
    for (a, &i) in ids.iter().enumerate() {
        let r = Rotation3::from_matrix_unchecked(rots[i].unwrap().matrix() * g.transpose());
        poses[i] = Some(Pose::from_center(r, &Point3::from(centers[a])));
    }
    if std::env::var("PIPE_DEBUG").is_ok() {
        let mut ang: Vec<f64> = pm
            .iter()
            .filter_map(|p| {
                let (ri, t) = (rots[p.i]?, p.t?);
                let b = g * (ri.inverse() * (-(p.rot.inverse() * t)));
                let dg = gps[p.j] - gps[p.i];
                (dg.norm() > 3.0).then(|| b.angle(&dg).to_degrees())
            })
            .collect();
        ang.sort_by(f64::total_cmp);
        let down: f64 = poses
            .iter()
            .flatten()
            .map(|p| (p.rotation.inverse() * Vector3::z()).z)
            .sum::<f64>()
            / poses.iter().flatten().count().max(1) as f64;
        eprintln!(
            "debug gauge pairs {} angle median {:?} view-dir z mean {down:.2} det {}",
            ang.len(),
            ang.get(ang.len() / 2),
            g.determinant()
        );
    }
    let counts: Vec<usize> = imgs.iter().map(|d| d.feats.len()).collect();
    let ms: Vec<_> = pm
        .iter()
        .filter(|p| poses[p.i].is_some() && poses[p.j].is_some())
        .map(|p| (p.i, p.j, p.inl.clone()))
        .collect();
    let tracks = stand_in::build_tracks(&counts, &ms);
    let (mut points, mut obs) = (Vec::new(), Vec::new());
    for tr in tracks {
        let o: Vec<(usize, usize, Vector2<f64>)> = tr
            .iter()
            .map(|&(i, f)| {
                let kp = imgs[i].feats[f].kp;
                (i, f, Vector2::new(kp.x as f64 + 0.5, kp.y as f64 + 0.5))
            })
            .collect();
        let o: Vec<(usize, usize, Vector2<f64>)> = o
            .into_iter()
            .filter(|&(i, _, _)| poses[i].is_some())
            .collect();
        let cams: Vec<(Camera, Vector2<f64>)> = o
            .iter()
            .map(|&(i, _, px)| {
                (
                    Camera {
                        intrinsics: *k,
                        pose: poses[i].unwrap(),
                    },
                    px,
                )
            })
            .collect();
        if cams.len() < 2 {
            continue;
        }
        if let Some((x, keep)) = stand_in::triangulate_robust(&cams, 0.7, 4.0) {
            points.push(x);
            obs.push(
                o.into_iter()
                    .zip(keep)
                    .filter_map(|(v, kp)| kp.then_some(v))
                    .collect(),
            );
        }
    }
    let mut s = Sparse {
        poses,
        points,
        obs,
        rms: 0.0,
    };
    s.rms = run_ba(&mut s, k, 0, None);
    Ok(s)
}

/// 번들 조정(`iters == 0` 이면 재투영 오차만 잰다). 반환: 재투영 RMS(px).
/// 정밀(BA) 결과를 GPS(ENU)에 닮음 정렬한다: 카메라 중심 ↔ GPS, 강건 추정(SPEC §3.4).
/// BA 는 자유 좌표계에서 움직이므로 정밀 모델을 다시 GPS 좌표계로 돌려놓는다.
fn gps_align_refined(s: &mut Sparse, gps: &[Vector3<f64>]) -> Option<Similarity> {
    let ids: Vec<usize> = (0..s.poses.len())
        .filter(|&i| s.poses[i].is_some())
        .collect();
    let src: Vec<Vector3<f64>> = ids
        .iter()
        .map(|&i| s.poses[i].unwrap().center().coords)
        .collect();
    let dst: Vec<Vector3<f64>> = ids.iter().map(|&i| gps[i]).collect();
    let (sim, _, _) = crate::align::robust_similarity(&src, &dst, 3, 3.0)?;
    for &i in &ids {
        let p = s.poses[i].unwrap();
        let c = sim.apply_point(&p.center().coords);
        s.poses[i] = Some(Pose::from_center(
            p.rotation * sim.r.inverse(),
            &Point3::from(c),
        ));
    }
    for p in s.points.iter_mut() {
        *p = sim.apply_point(p);
    }
    Some(sim)
}

fn run_ba(s: &mut Sparse, k: &Intrinsics, iters: usize, gps: Option<&[Vector3<f64>]>) -> f64 {
    let ids: Vec<usize> = (0..s.poses.len())
        .filter(|&i| s.poses[i].is_some())
        .collect();
    let loc: HashMap<usize, usize> = ids.iter().enumerate().map(|(a, &i)| (i, a)).collect();
    let mut observations = Vec::new();
    for (p, o) in s.obs.iter().enumerate() {
        for &(i, _, px) in o {
            if let Some(&c) = loc.get(&i) {
                observations.push(Observation {
                    camera: c,
                    point: p,
                    pixel: px,
                });
            }
        }
    }
    let mut prob = BaProblem {
        groups: vec![k.to_distorted()],
        poses: ids.iter().map(|&i| s.poses[i].unwrap()).collect(),
        camera_group: vec![0; ids.len()],
        points: s.points.iter().map(|p| Point3::from(*p)).collect(),
        observations,
    };
    let opts = BaOptions {
        max_iterations: iters,
        default_free_intrinsics: [false; 8],
        position_prior: gps
            .map(|g| PositionPrior::new(ids.iter().map(|&i| Some(Point3::from(g[i]))).collect())),
        ..BaOptions::default()
    };
    let rep = bundle_adjust(&mut prob, &opts);
    if iters > 0 {
        for (a, &i) in ids.iter().enumerate() {
            s.poses[i] = Some(prob.poses[a]);
        }
        s.points = prob.points.iter().map(|p| p.coords).collect();
    }
    if iters == 0 {
        rep.initial_rms
    } else {
        rep.final_rms
    }
}

/// 밀집: 깊이 맵(stand_in) → 융합. 희소 점도 함께 담는다.
fn dense_cloud(
    s: &Sparse,
    imgs: &[&ImgData],
    k: &Intrinsics,
    in_region: &[bool],
    dw: usize,
) -> PointCloud {
    let dh = ((k.height as f64 * dw as f64 / k.width as f64).round() as usize).max(8);
    let sc = dw as f64 / k.width as f64;
    let ks = Intrinsics {
        fx: k.fx * sc,
        fy: k.fy * sc,
        cx: k.cx * sc,
        cy: k.cy * sc,
        width: dw as u32,
        height: dh as u32,
        dist: k.dist,
    };
    let ids: Vec<usize> = (0..s.poses.len())
        .filter(|&i| s.poses[i].is_some() && in_region[i])
        .collect();
    let views: Vec<FusionView> = ids
        .iter()
        .map(|&i| {
            let small = image::imageops::resize(
                &imgs[i].rgb,
                dw as u32,
                dh as u32,
                image::imageops::FilterType::Triangle,
            );
            FusionView {
                camera: Camera {
                    intrinsics: ks,
                    pose: s.poses[i].unwrap(),
                },
                rgb: small.pixels().map(|p| p.0).collect(),
                neighbors: Vec::new(),
                group: Some((i % 3) as u32),
            }
        })
        .collect();
    let maps: Vec<_> = views
        .par_iter()
        .map(|v| stand_in::depth_from_sparse(&v.camera, &s.points))
        .collect();
    let cfg = FusionConfig {
        reproj_px: 2.0,
        depth_rel: 0.05,
        min_views: 2,
        normal_deg: 180.0,
        min_ratio: 0.3,
        min_groups: 1,
        same_group_views: None,
    };
    let mut cloud = fuse(&views, &maps, cfg);
    for p in &s.points {
        cloud.points.push(PointRecord {
            xyz: [p.x as f32, p.y as f32, p.z as f32],
            normal: [0.0; 3],
            rgb: [128; 3],
        });
    }
    cloud
}

fn to_tracks(s: &Sparse, gid: &[usize]) -> Vec<Track> {
    s.points
        .iter()
        .zip(&s.obs)
        .map(|(p, o)| Track {
            xyz: *p,
            obs: o
                .iter()
                .map(|&(i, f, _)| (gid[i] as u32, f as u32))
                .collect(),
        })
        .collect()
}

/// 구역 하나의 진행 기록(메인 스레드 소유).
struct RegionRec {
    region: Region,
    stats: RegionStats,
    gids: Vec<usize>,
    ta: Vec<Track>,
    coarse: PointCloud,
    /// 지금 쓰는 초벌 → 기준 정밀 좌표 변환과 그 기준 구역.
    sim: Option<Similarity>,
    target: Option<usize>,
    /// 자기 정밀 모델과의 정렬 기록(최종 manifest 용).
    own: Option<(Option<Similarity>, AlignRecord)>,
    refined: Option<(Vec<Track>, PointCloud)>,
    centers: BTreeMap<usize, [f64; 3]>,
    registered_prev: Vec<usize>,
}

/// 정밀(BA) 작업 결과.
struct RefinedMsg {
    slot: usize,
    sparse: Sparse,
    cloud: PointCloud,
    secs: f64,
}

fn write_decimated(out: &Path, name: &str, cloud: &PointCloud) -> Result<(), String> {
    crate::ply::write_ply_file(
        out.join(name),
        &crate::stream::decimate(cloud, crate::stream::DECIMATE_EVERY),
    )
    .map_err(|e| format!("출력 쓰기 실패: {e}"))
}

/// 끝까지 돌린다. 출력 폴더에 preview·refined·snapshots·manifest.json·report.json·poses.txt 를 쓴다.
///
/// 구역은 위치 순서로 하나씩 도착한다: 사진 읽기 → 짝 맞춤 → 등록(초벌 희소 모델) → 초벌 점군을 곧바로
/// 출력(가장 최근에 나온 정밀 모델 좌표계로 정렬) → 정밀(BA) 은 다른 스레드에서 돌고, 끝나면
/// 초벌을 대신해 내보내고 아직 정밀이 없는 초벌 구역을 새 정밀 모델 좌표로 다시 맞춘다.
/// 다음 구역의 등록 전에 끝난 정밀 결과를 먼저 반영한다. 등록·희소 초기화에 실패한 구역은
/// issues 에 적고 건너뛴다.
pub fn run_pipeline(
    ds: &Dataset,
    cfg: &PipelineConfig,
    out: &Path,
) -> Result<PipelineResult, String> {
    use crate::progressive::{
        check_motion, cross_align, median, overlap_window, own_ranges, ReAlign,
    };
    use crate::stream::{preview_name, refined_name};
    use std::sync::{mpsc, Arc};

    let n_pos = ds.positions.len();
    let regions = split_regions(n_pos, ds.config.span, ds.config.ovl);
    let owns = own_ranges(&regions, n_pos);
    for sub in ["preview", "refined", "snapshots"] {
        std::fs::create_dir_all(out.join(sub)).map_err(|e| e.to_string())?;
    }
    let t_start = Instant::now();
    let mut cache: HashMap<usize, Arc<ImgData>> = HashMap::new();
    let mut k_opt: Option<Intrinsics> = None;
    let mut res = PipelineResult::default();
    let mut recs: Vec<RegionRec> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut events: Vec<String> = Vec::new();
    let mut realigns: Vec<ReAlign> = Vec::new();
    let mut latest_ref: Option<usize> = None; // recs 번호
    let mut in_flight = 0usize;
    let (tx, rx) = mpsc::channel::<RefinedMsg>();
    let t_now = || t_start.elapsed().as_secs_f64();

    // 정밀 결과 하나를 반영: 파일 쓰기, 자기 정렬, 대기 중인 초벌 재정렬.
    let handle = |m: RefinedMsg,
                  recs: &mut Vec<RegionRec>,
                  events: &mut Vec<String>,
                  realigns: &mut Vec<ReAlign>,
                  latest_ref: &mut Option<usize>|
     -> Result<(), String> {
        let k = m.slot;
        let rec = &mut recs[k];
        let tb = to_tracks(&m.sparse, &rec.gids);
        for (a, g) in rec.gids.iter().enumerate() {
            if let Some(p) = m.sparse.poses[a] {
                let c = p.center();
                rec.centers.insert(*g, [c.x, c.y, c.z]);
            }
        }
        rec.stats.refined_rms = m.sparse.rms;
        rec.stats.refined_points = m.cloud.len();
        rec.stats.secs_ba = m.secs;
        write_decimated(out, &refined_name(&rec.region), &m.cloud)?;
        // 자기 구역 초벌 → 정밀 정렬(SPEC §3.7).
        let r = rec.region;
        let pos = |i: u32| (i / 3) as usize;
        let mut pairs = point_pairs(&rec.ta, &tb, pos, align_window(&r, ds.config.ovl, n_pos));
        let (mut sim, mut ar) = align_region(&r, &pairs);
        if sim.is_none() {
            pairs = point_pairs(&rec.ta, &tb, pos, (r.lo, r.hi));
            (sim, ar) = align_region(&r, &pairs);
        }
        if let Some(s) = &sim {
            realigns.push(ReAlign {
                secs: t_now(),
                region: k,
                target: k,
                pairs: ar.pairs,
                median_m: ar.fit_median_m.unwrap_or(f64::NAN),
                scale: s.s,
            });
            rec.sim = sim.clone();
            rec.target = Some(k);
            write_decimated(
                out,
                &preview_name(&r),
                &crate::stream::apply_cloud(s, &rec.coarse),
            )?;
        }
        rec.own = Some((sim, ar));
        rec.refined = Some((tb, m.cloud));
        *latest_ref = Some(k);
        events.push(format!(
            "{:.1}s refined region {} rms {:.3} done",
            t_now(),
            r.index,
            m.sparse.rms
        ));
        // 이미 내보낸, 아직 정밀이 없는 초벌 구역을 새 정밀 모델 좌표로 다시 맞춘다.
        for j in 0..recs.len() {
            if j == k || recs[j].refined.is_some() {
                continue;
            }
            let win = overlap_window(&recs[j].region, &recs[k].region);
            let tbk = &recs[k].refined.as_ref().unwrap().0;
            if let Some((s, n, med)) = cross_align(&recs[j].ta, tbk, win) {
                realigns.push(ReAlign {
                    secs: t_now(),
                    region: j,
                    target: k,
                    pairs: n,
                    median_m: med,
                    scale: s.s,
                });
                let jr = recs[j].region;
                write_decimated(
                    out,
                    &preview_name(&jr),
                    &crate::stream::apply_cloud(&s, &recs[j].coarse),
                )?;
                recs[j].sim = Some(s);
                recs[j].target = Some(k);
                events.push(format!(
                    "{:.1}s realign coarse {} to refined {} pairs {n} median {med:.3} m",
                    t_now(),
                    jr.index,
                    recs[k].region.index
                ));
            }
        }
        Ok(())
    };

    for r in &regions {
        // 끝난 정밀 결과를 먼저 반영해 이번 등록·정렬이 최신 모델을 기준으로 삼게 한다.
        while let Ok(m) = rx.try_recv() {
            in_flight -= 1;
            handle(m, &mut recs, &mut events, &mut realigns, &mut latest_ref)?;
        }
        // 정밀 작업이 둘 넘게 밀리면 하나가 끝나길 기다린다(코어 과다 경쟁 방지).
        while in_flight >= 2 {
            let m = rx.recv().map_err(|e| e.to_string())?;
            in_flight -= 1;
            handle(m, &mut recs, &mut events, &mut realigns, &mut latest_ref)?;
        }
        let mut st = RegionStats {
            region: r.index,
            positions: r.hi - r.lo,
            images: 3 * (r.hi - r.lo),
            ..Default::default()
        };
        events.push(format!("{:.1}s arrive region {}", t_now(), r.index));
        let gids: Vec<usize> = (r.lo..r.hi)
            .flat_map(|p| (0..3).map(move |c| 3 * p + c))
            .collect();
        let t0 = Instant::now();
        let need: Vec<usize> = gids
            .iter()
            .copied()
            .filter(|g| !cache.contains_key(g))
            .collect();
        let loaded: Vec<Result<(usize, ImgData), String>> = need
            .par_iter()
            .map(|&g| {
                Ok((
                    g,
                    load(&ds.positions[g / 3].images[g % 3], cfg.max_features)?,
                ))
            })
            .collect();
        let mut load_err = None;
        for l in loaded {
            match l {
                Ok((g, d)) => {
                    cache.insert(g, Arc::new(d));
                }
                Err(e) => load_err = Some(e),
            }
        }
        if let Some(e) = load_err {
            skipped.push(format!("구역 {} 건너뜀: 사진 읽기 실패: {e}", r.index));
            continue;
        }
        let k = *k_opt.get_or_insert_with(|| {
            let d = &cache[&gids[0]].rgb;
            Intrinsics::from_hfov(d.width(), d.height(), cfg.hfov_deg.to_radians())
        });
        st.secs_features = t0.elapsed().as_secs_f64();
        let arcs: Vec<Arc<ImgData>> = gids.iter().map(|g| cache[g].clone()).collect();
        let imgs: Vec<&ImgData> = arcs.iter().map(|a| a.as_ref()).collect();
        let views: Vec<(usize, usize)> = gids.iter().map(|g| (g % 3, g / 3)).collect();
        let gps: Vec<Vector3<f64>> = gids
            .iter()
            .map(|g| ds.positions[g / 3].image_enu[g % 3])
            .collect();
        let t1 = Instant::now();
        let pm = match_pairs(&imgs, &views, &k);
        st.secs_matching = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let pair_ids: Vec<(usize, usize)> = pm.iter().map(|p| (p.i, p.j)).collect();
        let init = match check_motion(&gps, &views, &pair_ids)
            .and_then(|_| sparse_init(&imgs, &pm, &gps, &k))
        {
            Ok(s) => s,
            Err(e) => {
                skipped.push(format!("구역 {} 건너뜀: {e}", r.index));
                events.push(format!("{:.1}s skip region {}", t_now(), r.index));
                continue;
            }
        };
        st.secs_sparse = t2.elapsed().as_secs_f64();
        st.registered = init.poses.iter().filter(|p| p.is_some()).count();
        st.tracks = init.points.len();
        st.preview_rms = init.rms;
        events.push(format!(
            "{:.1}s registered region {} ({}/{})",
            t_now(),
            r.index,
            st.registered,
            st.images
        ));
        let slot = recs.len();
        // 정밀(BA)은 다른 스레드에서: 끝나면 메시지로 돌아온다.
        {
            let (tx, init, arcs) = (tx.clone(), init.clone(), arcs.clone());
            let (gps, dw, iters) = (gps.clone(), cfg.dense_width, cfg.ba_iters);
            let in_region: Vec<bool> = gids.iter().map(|g| r.contains(g / 3)).collect();
            in_flight += 1;
            std::thread::spawn(move || {
                let t = Instant::now();
                let imgs: Vec<&ImgData> = arcs.iter().map(|a| a.as_ref()).collect();
                let mut rs = init;
                rs.rms = run_ba(&mut rs, &k, iters, Some(&gps));
                gps_align_refined(&mut rs, &gps);
                let cloud = dense_cloud(&rs, &imgs, &k, &in_region, dw);
                let _ = tx.send(RefinedMsg {
                    slot,
                    sparse: rs,
                    cloud,
                    secs: t.elapsed().as_secs_f64(),
                });
            });
        }
        // 초벌 점군: 곧바로 만들어 최신 정밀 좌표계로 정렬해 내보낸다.
        let t3 = Instant::now();
        let in_region: Vec<bool> = gids.iter().map(|g| r.contains(g / 3)).collect();
        let coarse = dense_cloud(&init, &imgs, &k, &in_region, cfg.dense_width);
        st.secs_dense = t3.elapsed().as_secs_f64();
        st.preview_points = coarse.len();
        let ta = to_tracks(&init, &gids);
        while let Ok(m) = rx.try_recv() {
            in_flight -= 1;
            handle(m, &mut recs, &mut events, &mut realigns, &mut latest_ref)?;
        }
        let (mut sim, mut target) = (None, None);
        if let Some(m) = latest_ref {
            let win = overlap_window(r, &recs[m].region);
            let tbm = &recs[m].refined.as_ref().unwrap().0;
            if let Some((s, n, med)) = cross_align(&ta, tbm, win) {
                realigns.push(ReAlign {
                    secs: t_now(),
                    region: slot,
                    target: m,
                    pairs: n,
                    median_m: med,
                    scale: s.s,
                });
                events.push(format!(
                    "{:.1}s coarse region {} aligned to refined {} pairs {n} median {med:.3} m",
                    t_now(),
                    r.index,
                    recs[m].region.index
                ));
                sim = Some(s);
                target = Some(m);
            }
        }
        // 기준 모델이 없거나 겹침이 모자라면 GPS 좌표 그대로(초기 좌표계는 이미 GPS)로 내보낸다.
        let shown = sim.clone().unwrap_or_else(Similarity::identity);
        write_decimated(
            out,
            &preview_name(r),
            &crate::stream::apply_cloud(&shown, &coarse),
        )?;
        events.push(format!("{:.1}s coarse output region {}", t_now(), r.index));
        let registered_prev = (0..gids.len())
            .filter(|&a| init.poses[a].is_some())
            .map(|a| gids[a])
            .collect();
        recs.push(RegionRec {
            region: *r,
            stats: st,
            gids,
            ta,
            coarse,
            sim,
            target,
            own: None,
            refined: None,
            centers: BTreeMap::new(),
            registered_prev,
        });
    }
    while in_flight > 0 {
        let m = rx.recv().map_err(|e| e.to_string())?;
        in_flight -= 1;
        handle(m, &mut recs, &mut events, &mut realigns, &mut latest_ref)?;
    }
    if recs.is_empty() {
        return Err(format!("모든 구역 실패: {}", skipped.join("; ")));
    }

    let kept: Vec<Region> = recs.iter().map(|r| r.region).collect();
    let sims: Vec<Option<Similarity>> = recs
        .iter()
        .map(|r| r.own.as_ref().and_then(|o| o.0.clone()))
        .collect();
    let records: Vec<AlignRecord> = recs
        .iter()
        .map(|r| r.own.as_ref().unwrap().1.clone())
        .collect();
    let prelim: Vec<PointCloud> = recs.iter().map(|r| r.coarse.clone()).collect();
    let refined: Vec<PointCloud> = recs
        .iter()
        .map(|r| r.refined.as_ref().unwrap().1.clone())
        .collect();
    let aligned = apply_alignments(&prelim, &sims);
    let rep = write_outputs(out, &kept, &aligned, &refined, records.clone())
        .map_err(|e| format!("출력 쓰기 실패: {e}"))?;
    res.issues = skipped;
    res.issues.extend(rep.issues);
    res.align = records;

    // 사진 중심: 자기 구역 값을 쓰고, 자기 구역에서 등록 못한 사진만 이웃 구역 값으로 채운다.
    let name = |g: usize| {
        ds.positions[g / 3].images[g % 3]
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default()
    };
    let mut centers: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
    let mut diffs: Vec<f64> = Vec::new();
    for rec in &recs {
        let (olo, ohi) = owns[rec.region.index];
        for (&g, c) in &rec.centers {
            let own = g / 3 >= olo && g / 3 < ohi;
            if own {
                centers.insert(g, *c);
            } else {
                centers.entry(g).or_insert(*c);
            }
        }
    }
    for rec in &recs {
        for (&g, c) in &rec.centers {
            if let Some(o) = centers.get(&g) {
                let d =
                    ((o[0] - c[0]).powi(2) + (o[1] - c[1]).powi(2) + (o[2] - c[2]).powi(2)).sqrt();
                if d > 0.0 {
                    diffs.push(d);
                }
            }
        }
    }
    let overlap_med = median(diffs.clone());
    if let Some(m) = overlap_med {
        res.issues.push(format!(
            "겹침 구간 사진 중심 차이(구역 간) 중앙 {m:.3} m, 최대 {:.3} m, {} 건",
            diffs.iter().copied().fold(0.0, f64::max),
            diffs.len()
        ));
    }
    let mut poses_txt = String::new();
    for (g, c) in &centers {
        poses_txt += &format!("{} {} {} {}\n", name(*g), c[0], c[1], c[2]);
        res.centers.push((name(*g), *c));
    }
    std::fs::write(out.join("poses.txt"), poses_txt).map_err(|e| e.to_string())?;
    let (mut reg_prev, mut reg_ref) = (
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
    );
    for rec in &recs {
        reg_prev.extend(rec.registered_prev.iter().copied());
        reg_ref.extend(rec.centers.keys().copied());
    }
    for rec in recs.iter() {
        let st = &rec.stats;
        println!(
            "region {} positions {} registered {}/{} tracks {} rms {:.3}->{:.3} points {}/{} secs feat {:.1} match {:.1} sparse {:.1} ba {:.1} coarse-dense {:.1}",
            st.region, st.positions, st.registered, st.images, st.tracks, st.preview_rms,
            st.refined_rms, st.preview_points, st.refined_points, st.secs_features,
            st.secs_matching, st.secs_sparse, st.secs_ba, st.secs_dense
        );
        res.regions.push(rec.stats.clone());
    }
    for e in &events {
        println!("event {e}");
    }
    let avg = |f: fn(&RegionStats) -> f64| {
        res.regions.iter().map(f).sum::<f64>() / res.regions.len().max(1) as f64
    };
    let q = |s: &str| s.replace('\\', "/").replace('"', "'");
    let report = format!(
        "{{\"registered\": {{\"total\": {}, \"preview\": {}, \"refined\": {}}}, \"reprojection_px\": {{\"preview\": {:.4}, \"refined\": {:.4}}}, \"regions\": [{}], \"realign_count\": {}, \"realigns\": [{}], \"overlap_center_diff_median_m\": {}, \"events\": [{}]}}\n",
        ds.image_count(),
        reg_prev.len(),
        reg_ref.len(),
        avg(|s| s.preview_rms),
        avg(|s| s.refined_rms),
        res.regions
            .iter()
            .map(|s| format!(
                "{{\"region\": {}, \"positions\": {}, \"images\": {}}}",
                s.region, s.positions, s.images
            ))
            .collect::<Vec<_>>()
            .join(", "),
        realigns.len(),
        realigns
            .iter()
            .map(|a| format!(
                "{{\"secs\": {:.2}, \"region\": {}, \"target\": {}, \"pairs\": {}, \"median_m\": {:.4}, \"scale\": {:.5}}}",
                a.secs, recs[a.region].region.index, recs[a.target].region.index, a.pairs, a.median_m, a.scale
            ))
            .collect::<Vec<_>>()
            .join(", "),
        overlap_med.map_or("null".to_string(), |m| format!("{m:.4}")),
        events
            .iter()
            .map(|e| format!("\"{}\"", q(e)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    std::fs::write(out.join("report.json"), report).map_err(|e| e.to_string())?;
    Ok(res)
}
