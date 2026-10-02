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
use crate::ba::{bundle_adjust, BaOptions, BaProblem, Observation};
use crate::camera::{Camera, Intrinsics, Pose};
use crate::dataset::Dataset;
use crate::features::{detect_and_describe, DetectorConfig, Feature, GrayImage};
use crate::fusion::{fuse, FusionConfig, FusionView};
use crate::math::{Matrix3, Point3, Rotation3, Vector2, Vector3};
use crate::matching::{candidate_pairs, ratio_match, RansacConfig, PAIR_CROSS, PAIR_TEMPORAL};
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

pub mod stand_in {
    //! 병합 전 부품의 단순 대체. 모양은 TASKS 인터페이스를 따른다.
    use super::*;

    /// 트랙: 합집합-찾기, 같은 사진이 두 번 든 성분은 버린다. 반환은 성분별 (사진, 특징) 목록.
    /// `feat_counts[i]` = 사진 i 의 특징 수, `matches` = (i, j, [(특징 i, 특징 j)]).
    pub fn build_tracks(
        feat_counts: &[usize],
        matches: &[(usize, usize, Vec<(usize, usize)>)],
    ) -> Vec<Vec<(usize, usize)>> {
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
                let (ra, rb) = (find(&mut parent, off[*i] + a), find(&mut parent, off[*j] + b));
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
    pub fn triangulate_track(
        cams: &[(Camera, Vector2<f64>)],
        max_px: f64,
    ) -> Option<Vector3<f64>> {
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
            if !((q - px).norm() < max_px) {
                return None;
            }
        }
        Some(x)
    }

    /// 희소 점 보간 깊이 맵(역거리 가중, 반경 밖은 빈 화소). `cam` 은 깊이 맵 해상도의 카메라.
    pub fn depth_from_sparse(cam: &Camera, points: &[Vector3<f64>]) -> crate::fusion::DepthMap {
        let (w, h) = (cam.intrinsics.width as usize, cam.intrinsics.height as usize);
        let proj: Vec<(f64, f64, f64)> = points
            .iter()
            .filter_map(|p| {
                let z = cam.pose.transform(&Point3::from(*p)).z;
                let q = cam.project(&Point3::from(*p))?;
                (q.x >= 0.0 && q.y >= 0.0 && q.x < w as f64 && q.y < h as f64).then_some((q.x, q.y, z))
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

fn match_pairs(imgs: &[&ImgData], views: &[(usize, usize)], k: &Intrinsics) -> Vec<PairMatch> {
    let pairs = candidate_pairs(views, PAIR_TEMPORAL, PAIR_CROSS.min(2), 0);
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
    let g = vt.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose();
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
        let cams: Vec<(Camera, Vector2<f64>)> = o
            .iter()
            .filter_map(|&(i, _, px)| {
                poses[i].map(|pose| {
                    (
                        Camera {
                            intrinsics: *k,
                            pose,
                        },
                        px,
                    )
                })
            })
            .collect();
        if cams.len() < 2 {
            continue;
        }
        if let Some(x) = stand_in::triangulate_track(&cams, 6.0) {
            points.push(x);
            obs.push(o);
        }
    }
    let mut s = Sparse {
        poses,
        points,
        obs,
        rms: 0.0,
    };
    s.rms = run_ba(&mut s, k, 0);
    Ok(s)
}

/// 번들 조정(`iters == 0` 이면 재투영 오차만 잰다). 반환: 재투영 RMS(px).
fn run_ba(s: &mut Sparse, k: &Intrinsics, iters: usize) -> f64 {
    let ids: Vec<usize> = (0..s.poses.len()).filter(|&i| s.poses[i].is_some()).collect();
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

/// 끝까지 돌린다. 출력 폴더에 preview·refined·snapshots·manifest.json·report.json·poses.txt 를 쓴다.
pub fn run_pipeline(
    ds: &Dataset,
    cfg: &PipelineConfig,
    out: &Path,
) -> Result<PipelineResult, String> {
    let n_pos = ds.positions.len();
    let regions = split_regions(n_pos, ds.config.span, ds.config.ovl);
    let mut cache: HashMap<usize, ImgData> = HashMap::new();
    let mut k_opt: Option<Intrinsics> = None;
    let mut res = PipelineResult::default();
    let (mut prelim, mut refined, mut tr_pairs): (Vec<PointCloud>, Vec<PointCloud>, Vec<(Region, Vec<Track>, Vec<Track>)>) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut centers: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
    let (mut reg_prev, mut reg_ref) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    for r in &regions {
        let mut st = RegionStats {
            region: r.index,
            positions: r.hi - r.lo,
            images: 3 * (r.hi - r.lo),
            ..Default::default()
        };
        let gids: Vec<usize> = (r.lo..r.hi).flat_map(|p| (0..3).map(move |c| 3 * p + c)).collect();
        let t0 = Instant::now();
        let need: Vec<usize> = gids.iter().copied().filter(|g| !cache.contains_key(g)).collect();
        let loaded: Vec<Result<(usize, ImgData), String>> = need
            .par_iter()
            .map(|&g| Ok((g, load(&ds.positions[g / 3].images[g % 3], cfg.max_features)?)))
            .collect();
        for l in loaded {
            let (g, d) = l?;
            cache.insert(g, d);
        }
        let k = *k_opt.get_or_insert_with(|| {
            let d = &cache[&gids[0]].rgb;
            Intrinsics::from_hfov(d.width(), d.height(), cfg.hfov_deg.to_radians())
        });
        st.secs_features = t0.elapsed().as_secs_f64();
        let imgs: Vec<&ImgData> = gids.iter().map(|g| &cache[g]).collect();
        let views: Vec<(usize, usize)> = gids.iter().map(|g| (g % 3, g / 3)).collect();
        let gps: Vec<Vector3<f64>> = gids
            .iter()
            .map(|g| ds.positions[g / 3].image_enu[g % 3])
            .collect();
        let t1 = Instant::now();
        let pm = match_pairs(&imgs, &views, &k);
        st.secs_matching = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let init = sparse_init(&imgs, &pm, &gps, &k)
            .map_err(|e| format!("구역 {}: {e}", r.index))?;
        st.secs_sparse = t2.elapsed().as_secs_f64();
        let in_region: Vec<bool> = gids.iter().map(|g| r.contains(g / 3)).collect();
        let t3 = Instant::now();
        let ((pre_cloud, ref_s, ref_cloud), secs_dense_pre) = {
            let ((pc, dt), (rs, rc)) = rayon::join(
                || {
                    let t = Instant::now();
                    (dense_cloud(&init, &imgs, &k, &in_region, cfg.dense_width), t.elapsed().as_secs_f64())
                },
                || {
                    let mut rs = init.clone();
                    rs.rms = run_ba(&mut rs, &k, cfg.ba_iters);
                    let rc = dense_cloud(&rs, &imgs, &k, &in_region, cfg.dense_width);
                    (rs, rc)
                },
            );
            ((pc, rs, rc), dt)
        };
        st.secs_ba = t3.elapsed().as_secs_f64();
        st.secs_dense = secs_dense_pre;
        st.registered = init.poses.iter().filter(|p| p.is_some()).count();
        st.tracks = init.points.len();
        st.preview_rms = init.rms;
        st.refined_rms = ref_s.rms;
        st.preview_points = pre_cloud.len();
        st.refined_points = ref_cloud.len();
        for (a, g) in gids.iter().enumerate() {
            if init.poses[a].is_some() {
                reg_prev.insert(*g);
            }
            if let Some(p) = ref_s.poses[a] {
                reg_ref.insert(*g);
                let c = p.center();
                centers.insert(*g, [c.x, c.y, c.z]);
            }
        }
        // 초벌 → 정밀 좌표 정렬(공유 3D 점, 같은 사진·같은 특징).
        let (ta, tb) = (to_tracks(&init, &gids), to_tracks(&ref_s, &gids));
        tr_pairs.push((*r, ta, tb));
        prelim.push(pre_cloud);
        refined.push(ref_cloud);
        println!(
            "region {} positions {} registered {}/{} tracks {} rms {:.3}->{:.3} points {}/{} secs feat {:.1} match {:.1} sparse {:.1} ba+dense {:.1}",
            r.index, st.positions, st.registered, st.images, st.tracks, st.preview_rms,
            st.refined_rms, st.preview_points, st.refined_points, st.secs_features,
            st.secs_matching, st.secs_sparse, st.secs_ba
        );
        res.regions.push(st);
    }
    let mut sims: Vec<Option<Similarity>> = Vec::new();
    let mut records = Vec::new();
    for (r, ta, tb) in &tr_pairs {
        let pos = |i: u32| (i / 3) as usize;
        let win = align_window(r, ds.config.ovl, n_pos);
        let mut pairs = point_pairs(ta, tb, pos, win);
        let (mut sim, mut rec) = align_region(r, &pairs);
        if sim.is_none() {
            pairs = point_pairs(ta, tb, pos, (r.lo, r.hi));
            (sim, rec) = align_region(r, &pairs);
        }
        sims.push(sim);
        records.push(rec);
    }
    let aligned = apply_alignments(&prelim, &sims);
    let rep = write_outputs(out, &regions, &aligned, &refined, records.clone())
        .map_err(|e| format!("출력 쓰기 실패: {e}"))?;
    res.issues = rep.issues;
    res.align = records;
    let name = |g: usize| {
        ds.positions[g / 3].images[g % 3]
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default()
    };
    let mut poses_txt = String::new();
    for (g, c) in &centers {
        poses_txt += &format!("{} {} {} {}\n", name(*g), c[0], c[1], c[2]);
        res.centers.push((name(*g), *c));
    }
    std::fs::write(out.join("poses.txt"), poses_txt).map_err(|e| e.to_string())?;
    let avg = |f: fn(&RegionStats) -> f64| {
        res.regions.iter().map(f).sum::<f64>() / res.regions.len().max(1) as f64
    };
    let report = format!(
        "{{\"registered\": {{\"total\": {}, \"preview\": {}, \"refined\": {}}}, \"reprojection_px\": {{\"preview\": {:.4}, \"refined\": {:.4}}}, \"regions\": [{}]}}\n",
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
            .join(", ")
    );
    std::fs::write(out.join("report.json"), report).map_err(|e| e.to_string())?;
    Ok(res)
}
