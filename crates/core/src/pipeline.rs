//! 끝까지 잇는 흐름: 이미지 폴더(+GPS) → 특징 → 매칭 → 회전 평균 → 위치 → 삼각측량 → 번들 조정
//! → 구역별 초벌/정밀 → 밀집 깊이 → 융합 → 정렬·스냅샷·manifest.
//!
//! 밀집 단계는 `dense::region_cloud`(구역 사진별 깊이 + 융합)를 거친다. 밀집 점이 하나도 안 나오면
//! [`stand_in::depth_from_sparse`](희소 점 역거리 보간 깊이)로 대신한다.
//! 아직 합치지 않은 부품(트랙, 위치 평균·삼각측량)은 [`stand_in`] 의 단순 구현으로 잇는다.
//! `sparse::reconstruct` 는 이 흐름에 연결되어 있지 않다(희소 단계는 이 파일의 `sparse_init` 등이 맡는다).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::align::Similarity;
use crate::ba::{bundle_adjust, BaOptions, BaProblem, Observation, PositionPrior};
use crate::camera::{Camera, Intrinsics, Pose};
use crate::dataset::Dataset;
use crate::dense::{region_cloud, DenseConfig, DenseView};
use crate::features::{detect_and_describe, DetectorConfig, Feature, GrayImage};
use crate::fusion::{fuse, FusionConfig, FusionView};
use crate::matching::{ratio_match, scheduled_pairs, PairSchedule, RansacConfig};
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
type RegionTracks = (Region, Vec<Track>, Vec<Track>);

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

    /// 버릴 수 있는 관측 수의 상한: 관측의 1/3 을 넘지 않고 2 개까지.
    pub fn max_drops(n: usize) -> usize {
        (n / 3).min(2)
    }

    /// 강건 다시점 삼각측량: 재투영이 가장 나쁜 관측을 하나씩 버리며(최대 [`max_drops`] 개) 모든
    /// 관측이 `max_px` 안에 들고 광선 최대 각이 `min_deg` 이상일 때만 돌려준다. 상한 안에 못
    /// 맞추면 `None`. 유한하지 않은 픽셀은 처음부터 제외한다(상한에 세지 않는다). 반환: 점, 남긴
    /// 관측 표시.
    pub fn triangulate_robust(
        cams: &[(Camera, Vector2<f64>)],
        max_px: f64,
        min_deg: f64,
    ) -> Option<(Vector3<f64>, Vec<bool>)> {
        let mut keep: Vec<bool> = cams
            .iter()
            .map(|(_, px)| px.x.is_finite() && px.y.is_finite())
            .collect();
        let valid = keep.iter().filter(|&&k| k).count();
        if valid < 2 || !max_px.is_finite() {
            return None;
        }
        let mut drops = 0;
        let cap = max_drops(valid);
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
            if !x.iter().all(|v| v.is_finite()) {
                return None;
            }
            // 카메라 뒤·투영 불가·비유한 오차는 무한대로 보아 가장 먼저 버린다.
            let errs: Vec<f64> = sub
                .iter()
                .map(|(cam, px)| {
                    cam.project(&Point3::from(x))
                        .map(|q| (q - px).norm())
                        .filter(|e| e.is_finite())
                        .unwrap_or(f64::INFINITY)
                })
                .collect();
            let (worst, we) = errs
                .iter()
                .copied()
                .enumerate()
                .fold((0, -1.0), |m, (i, e)| if e > m.1 { (i, e) } else { m });
            if we < max_px {
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
            if drops >= cap {
                return None;
            }
            drops += 1;
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

fn match_pairs(imgs: &[&ImgData], views: &[(usize, usize)], k: &Intrinsics) -> Vec<PairMatch> {
    let pairs = scheduled_pairs(views, &PairSchedule::default());
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

/// 삼각측량 거름 설정. 해상도에 비례하는 느슨한 문턱과, 초벌 재투영 분포(중앙값 × 배수)에서
/// 정하는 점 문턱을 함께 쓴다.
#[derive(Clone, Copy, Debug)]
struct TriConfig {
    /// 느슨한 문턱(영상 폭 비율). 점 후보 선별과 BA 관측 목록의 바깥 한계.
    loose_frac: f64,
    /// 점 문턱 = 이 배수 × 후보 관측 재투영 중앙값.
    median_k: f64,
    /// 점 문턱의 하한(px).
    min_px: f64,
    min_deg: f64,
}

const TRI: TriConfig = TriConfig {
    loose_frac: 0.01,
    median_k: 3.0,
    min_px: 0.7,
    min_deg: 1.5,
};

/// 삼각측량 집계(관측 수는 초벌 점 문턱 전·후, BA 에 넘기는 수).
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
struct TriStats {
    tracks: usize,
    points: usize,
    obs_all: usize,
    obs_coarse: usize,
    obs_ba: usize,
    loose_px: f64,
    median_px: f64,
    thr_px: f64,
}

/// 트랙 → (점, 점별 BA 관측, 집계). 두 단계: 느슨한 문턱으로 후보 점을 만들고 그 재투영 중앙값에서
/// 점 문턱을 정해 다시 거른다. 점 문턱은 초벌 점 구름만 거르고, BA 에는 점에 속한 관측 중 느슨한
/// 문턱 안의 것을 모두 넘긴다(점 문턱이 BA 관측을 깎지 않게).
#[allow(clippy::type_complexity)]
fn triangulate_tracks(
    poses: &[Option<Pose>],
    k: &Intrinsics,
    tracks: &[Vec<(usize, usize, Vector2<f64>)>],
) -> (
    Vec<Vector3<f64>>,
    Vec<Vec<(usize, usize, Vector2<f64>)>>,
    TriStats,
) {
    let loose = (TRI.loose_frac * k.width as f64).max(2.0 * TRI.min_px);
    let cams_of = |o: &[(usize, usize, Vector2<f64>)]| -> Vec<(Camera, Vector2<f64>)> {
        o.iter()
            .filter_map(|&(i, _, px)| {
                Some((
                    Camera {
                        intrinsics: *k,
                        pose: poses.get(i).copied().flatten()?,
                    },
                    px,
                ))
            })
            .collect()
    };
    // 1 단계: 느슨한 문턱.
    let cand: Vec<Option<Vec<f64>>> = tracks
        .par_iter()
        .map(|o| {
            let cams = cams_of(o);
            if cams.len() != o.len() {
                return None;
            }
            let (x, keep) = stand_in::triangulate_robust(&cams, loose, TRI.min_deg)?;
            Some(
                cams.iter()
                    .zip(keep)
                    .filter(|&(_, kp)| kp)
                    .filter_map(|((c, px), _)| Some((c.project(&Point3::from(x))? - px).norm()))
                    .collect(),
            )
        })
        .collect();
    let mut errs: Vec<f64> = cand.iter().flatten().flatten().copied().collect();
    errs.sort_by(f64::total_cmp);
    let median = errs.get(errs.len() / 2).copied().unwrap_or(0.0);
    let thr = (TRI.median_k * median).clamp(TRI.min_px, loose);
    // 2 단계: 점 문턱. BA 관측은 느슨한 문턱 안의 모든 관측.
    let res: Vec<Option<(Vector3<f64>, Vec<(usize, usize, Vector2<f64>)>, usize)>> = tracks
        .par_iter()
        .zip(cand.par_iter())
        .map(|(o, c)| {
            c.as_ref()?;
            let cams = cams_of(o);
            let (x, keep) = stand_in::triangulate_robust(&cams, thr, TRI.min_deg)?;
            let nk = keep.iter().filter(|&&kp| kp).count();
            let ba: Vec<_> = o
                .iter()
                .zip(&cams)
                .filter(|(_, (c, px))| {
                    c.project(&Point3::from(x))
                        .is_some_and(|q| (q - px).norm() < loose)
                })
                .map(|(v, _)| *v)
                .collect();
            Some((x, ba, nk))
        })
        .collect();
    let mut st = TriStats {
        tracks: tracks.len(),
        obs_all: tracks.iter().map(Vec::len).sum(),
        loose_px: loose,
        median_px: median,
        thr_px: thr,
        ..TriStats::default()
    };
    let (mut points, mut obs) = (Vec::new(), Vec::new());
    for (x, ba, nk) in res.into_iter().flatten() {
        st.obs_coarse += nk;
        st.obs_ba += ba.len();
        points.push(x);
        obs.push(ba);
    }
    st.points = points.len();
    (points, obs, st)
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
    let track_obs: Vec<Vec<(usize, usize, Vector2<f64>)>> = tracks
        .iter()
        .map(|tr| {
            tr.iter()
                .filter(|&&(i, _)| poses[i].is_some())
                .map(|&(i, f)| {
                    let kp = imgs[i].feats[f].kp;
                    (i, f, Vector2::new(kp.x as f64 + 0.5, kp.y as f64 + 0.5))
                })
                .collect()
        })
        .collect();
    let (points, obs, stats) = triangulate_tracks(&poses, k, &track_obs);
    if std::env::var("PIPE_DEBUG").is_ok() {
        eprintln!("debug triangulation {stats:?}");
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

/// 밀집: 구역 사진을 `dense::region_cloud`(보정·이웃·사진별 깊이·융합)에 넘긴다. 희소 점도 함께 담는다.
/// `dw` 는 보정 뒤 긴 변 화소 수다. 밀집 점이 하나도 안 나오면 희소 점 보간 깊이로 대신한다.
fn dense_cloud(
    s: &Sparse,
    imgs: &[&ImgData],
    k: &Intrinsics,
    in_region: &[bool],
    dw: usize,
) -> PointCloud {
    let ids: Vec<usize> = (0..s.poses.len())
        .filter(|&i| s.poses[i].is_some() && in_region[i])
        .collect();
    let views: Vec<DenseView> = ids
        .iter()
        .map(|&i| DenseView {
            camera: Camera {
                intrinsics: *k,
                pose: s.poses[i].unwrap(),
            },
            image: imgs[i].rgb.clone(),
        })
        .collect();
    let pts: Vec<[f64; 3]> = s.points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let cfg = DenseConfig {
        max_width: dw as u32,
        ..DenseConfig::default()
    };
    let mut cloud = region_cloud(&views, &pts, &cfg);
    if cloud.is_empty() {
        cloud = interpolated_cloud(s, &ids, imgs, k, dw);
    }
    for p in &s.points {
        cloud.points.push(PointRecord {
            xyz: [p.x as f32, p.y as f32, p.z as f32],
            normal: [0.0; 3],
            rgb: [128; 3],
        });
    }
    cloud
}

/// 대체 경로: 희소 점 역거리 보간 깊이 맵 → 융합.
fn interpolated_cloud(
    s: &Sparse,
    ids: &[usize],
    imgs: &[&ImgData],
    k: &Intrinsics,
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
    fuse(&views, &maps, cfg)
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
    let (mut prelim, mut refined, mut tr_pairs): (
        Vec<PointCloud>,
        Vec<PointCloud>,
        Vec<RegionTracks>,
    ) = (Vec::new(), Vec::new(), Vec::new());
    let mut centers: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
    let (mut reg_prev, mut reg_ref) = (
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
    );
    for r in &regions {
        let mut st = RegionStats {
            region: r.index,
            positions: r.hi - r.lo,
            images: 3 * (r.hi - r.lo),
            ..Default::default()
        };
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
        let init =
            sparse_init(&imgs, &pm, &gps, &k).map_err(|e| format!("구역 {}: {e}", r.index))?;
        st.secs_sparse = t2.elapsed().as_secs_f64();
        let in_region: Vec<bool> = gids.iter().map(|g| r.contains(g / 3)).collect();
        let t3 = Instant::now();
        let ((pre_cloud, ref_s, ref_cloud), secs_dense_pre) = {
            let ((pc, dt), (rs, rc)) = rayon::join(
                || {
                    let t = Instant::now();
                    (
                        dense_cloud(&init, &imgs, &k, &in_region, cfg.dense_width),
                        t.elapsed().as_secs_f64(),
                    )
                },
                || {
                    let mut rs = init.clone();
                    rs.rms = run_ba(&mut rs, &k, cfg.ba_iters, Some(&gps));
                    gps_align_refined(&mut rs, &gps);
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
        if std::env::var("SKYLENS_DIAG").is_ok() {
            let med = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
            };
            let dp: Vec<Vector3<f64>> = init
                .points
                .iter()
                .zip(&ref_s.points)
                .map(|(a, b)| b - a)
                .collect();
            let dc: Vec<Vector3<f64>> = (0..gids.len())
                .filter_map(|a| {
                    Some(ref_s.poses[a]?.center().coords - init.poses[a]?.center().coords)
                })
                .collect();
            let gi: Vec<f64> = (0..gids.len())
                .filter_map(|a| Some((init.poses[a]?.center().coords - gps[a]).norm()))
                .collect();
            let gr: Vec<f64> = (0..gids.len())
                .filter_map(|a| Some((ref_s.poses[a]?.center().coords - gps[a]).norm()))
                .collect();
            eprintln!(
                "diag point shift |dxyz| med {:.2} dz med {:.2}; center shift med {:.2} dz {:.2}; gps dist init {:.2} refined {:.2}",
                med(dp.iter().map(|d| d.norm()).collect()),
                med(dp.iter().map(|d| d.z.abs()).collect()),
                med(dc.iter().map(|d| d.norm()).collect()),
                med(dc.iter().map(|d| d.z.abs()).collect()),
                med(gi),
                med(gr)
            );
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

#[cfg(test)]
mod tri_tests {
    use super::stand_in::{max_drops, triangulate_robust};
    use super::*;

    fn k() -> Intrinsics {
        Intrinsics::from_hfov(960, 540, 65f64.to_radians())
    }

    /// 아래를 보는 카메라(동=x, 북=위 영상). 카메라 좌표 x=동, y=남, z=아래.
    fn nadir(c: Vector3<f64>) -> Pose {
        let r = Rotation3::from_matrix_unchecked(Matrix3::from_diagonal(&Vector3::new(
            1.0, -1.0, -1.0,
        )));
        Pose::from_center(r, &Point3::from(c))
    }

    fn cam(c: Vector3<f64>) -> Camera {
        Camera {
            intrinsics: k(),
            pose: nadir(c),
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let (a, b) = (self.next().max(1e-12), self.next());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    fn view(c: Vector3<f64>, x: &Vector3<f64>) -> (Camera, Vector2<f64>) {
        let cm = cam(c);
        let px = cm.project(&Point3::from(*x)).unwrap();
        (cm, px)
    }

    /// 6 시점이 한 점을 본다(기선 10 m 간격, 고도 30 m).
    fn six_views(x: &Vector3<f64>) -> Vec<(Camera, Vector2<f64>)> {
        (0..6)
            .map(|i| {
                let c = Vector3::new(-25.0 + 10.0 * i as f64, (i % 2) as f64 * 6.0, 30.0);
                view(c, x)
            })
            .collect()
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let x = Vector3::new(3.0, 2.0, 0.0);
        assert!(triangulate_robust(&[], 1.0, 1.5).is_none());
        let v = six_views(&x);
        assert!(triangulate_robust(&v[..1], 1.0, 1.5).is_none());
        // 같은 중심, 같은 광선.
        let same = vec![v[0], v[0], v[0]];
        assert!(triangulate_robust(&same, 1.0, 1.5).is_none());
        // 같은 중심, 다른 픽셀(순수 회전): 광선이 한 점에서 만나지 않는다.
        let c0 = Vector3::new(0.0, 0.0, 30.0);
        let (ca, pa) = view(c0, &x);
        let pure = vec![(ca, pa), (ca, pa + Vector2::new(80.0, 0.0))];
        assert!(triangulate_robust(&pure, 1.0, 1.5).is_none());
        // 평행 광선(무한히 먼 점이나 같은 방향): 모든 시점이 같은 광선 방향.
        let far = Vector3::new(0.0, 0.0, -1.0e9);
        let par: Vec<_> = (0..3)
            .map(|i| view(Vector3::new(10.0 * i as f64, 0.0, 30.0), &far))
            .collect();
        assert!(triangulate_robust(&par, 1.0, 1.5).is_none());
        // NaN 픽셀: 그 관측만 빠지고 나머지로 풀린다. 전부 NaN 이면 None.
        let mut nan = v.clone();
        nan[2].1 = Vector2::new(f64::NAN, 10.0);
        let (p, keep) = triangulate_robust(&nan, 1.0, 1.5).expect("nan skipped");
        assert!(!keep[2] && keep.iter().filter(|&&k| k).count() == 5);
        assert!((p - x).norm() < 1e-6);
        let all_nan: Vec<_> = v
            .iter()
            .map(|&(c, _)| (c, Vector2::new(f64::NAN, f64::NAN)))
            .collect();
        assert!(triangulate_robust(&all_nan, 1.0, 1.5).is_none());
        // 카메라 뒤: 한 시점이 점을 뒤에서 본다(위쪽에서 위를 향하는 점).
        let mut behind = v.clone();
        behind[3].0 = Camera {
            intrinsics: k(),
            pose: Pose::from_center(
                Rotation3::from_matrix_unchecked(Matrix3::from_diagonal(&Vector3::new(
                    1.0, -1.0, -1.0,
                ))),
                &Point3::new(5.0, 0.0, -40.0),
            ),
        };
        let r = triangulate_robust(&behind, 1.0, 1.5);
        if let Some((p, keep)) = r {
            assert!(!keep[3]);
            assert!((p - x).norm() < 0.05);
        }
        // 두 시점뿐이고 한쪽이 뒤: 풀 수 없다.
        assert!(triangulate_robust(&behind[2..4], 1.0, 1.5).is_none());
    }

    #[test]
    fn outliers_are_dropped_up_to_the_cap_only() {
        assert_eq!(
            (2..=9).map(max_drops).collect::<Vec<_>>(),
            [0, 1, 1, 1, 2, 2, 2, 2]
        );
        let x = Vector3::new(1.0, -2.0, 1.0);
        let mut rng = Rng(7);
        let mut clean = six_views(&x);
        for (_, px) in clean.iter_mut() {
            *px += Vector2::new(0.3 * rng.gauss(), 0.3 * rng.gauss());
        }
        // 이상치 없음.
        let (p0, keep0) = triangulate_robust(&clean, 2.0, 1.5).unwrap();
        assert!(keep0.iter().all(|&k| k));
        assert!((p0 - x).norm() < 0.1);
        // 1 개, 2 개.
        for outliers in [vec![1usize], vec![1, 4]] {
            let mut v = clean.clone();
            for &o in &outliers {
                v[o].1 += Vector2::new(40.0, -25.0);
            }
            let (p, keep) = triangulate_robust(&v, 2.0, 1.5).expect("within cap");
            for i in 0..6 {
                assert_eq!(keep[i], !outliers.contains(&i), "{outliers:?} {keep:?}");
            }
            assert!((p - x).norm() < 0.1, "{}", (p - x).norm());
        }
        // 상한(2) 초과: 3 개가 이상치면 None.
        let mut v = clean.clone();
        for o in [0usize, 2, 4] {
            v[o].1 += Vector2::new(35.0 + 10.0 * o as f64, 20.0);
        }
        assert!(triangulate_robust(&v, 2.0, 1.5).is_none());
        // 3 시점은 하나까지: 이상치 하나는 버리고 남은 둘로 푼다.
        let mut v3 = clean[..3].to_vec();
        v3[1].1 += Vector2::new(50.0, 0.0);
        let (_, keep) = triangulate_robust(&v3, 2.0, 1.5).unwrap();
        assert_eq!(keep, [true, false, true]);
    }

    struct Scene {
        truth: Vec<Vector3<f64>>,
        poses: Vec<Pose>,
        coarse: Vec<Option<Pose>>,
        tracks: Vec<Vec<(usize, usize, Vector2<f64>)>>,
        n_pts: usize,
    }

    /// 고도 30 m 6×5 격자, 점 2000 개(3 개 이상 시점), 픽셀 잡음 σ, 초벌 포즈 흔들기.
    fn scene(px_sigma: f64, rot_sigma: f64, pos_sigma: f64) -> Scene {
        let mut rng = Rng(42);
        let mut centers = Vec::new();
        for r in 0..5 {
            for c in 0..6 {
                centers.push(Vector3::new(c as f64 * 12.0, r as f64 * 8.0, 30.0));
            }
        }
        let poses: Vec<Pose> = centers.iter().map(|&c| nadir(c)).collect();
        let coarse: Vec<Option<Pose>> = poses
            .iter()
            .map(|p| {
                let w = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * rot_sigma;
                let dr = Rotation3::from_scaled_axis(w);
                let dc = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * pos_sigma;
                Some(Pose::from_center(
                    dr * p.rotation,
                    &Point3::from(p.center().coords + dc),
                ))
            })
            .collect();
        let intr = k();
        let mut truth = Vec::new();
        let mut tracks = Vec::new();
        while truth.len() < 2000 {
            let x = Vector3::new(
                10.0 + 50.0 * rng.next(),
                8.0 + 24.0 * rng.next(),
                3.0 * rng.next(),
            );
            let o: Vec<_> = poses
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    let q = Camera {
                        intrinsics: intr,
                        pose: *p,
                    }
                    .project(&Point3::from(x))?;
                    intr.contains(&q).then(|| {
                        (
                            i,
                            tracks.len() * 100 + i,
                            q + Vector2::new(px_sigma * rng.gauss(), px_sigma * rng.gauss()),
                        )
                    })
                })
                .collect();
            if o.len() >= 3 {
                truth.push(x);
                tracks.push(o);
            }
        }
        let n_pts = tracks.len();
        Scene {
            truth,
            poses,
            coarse,
            tracks,
            n_pts,
        }
    }

    fn rms_over(
        poses: &[Option<Pose>],
        pts: &[Vector3<f64>],
        obs: &[Vec<(usize, usize, Vector2<f64>)>],
    ) -> f64 {
        let (mut s, mut n) = (0.0, 0usize);
        for (x, o) in pts.iter().zip(obs) {
            for &(i, _, px) in o {
                if let Some(q) = (Camera {
                    intrinsics: k(),
                    pose: poses[i].unwrap(),
                })
                .project(&Point3::from(*x))
                {
                    s += (q - px).norm_squared();
                    n += 1;
                }
            }
        }
        (s / n as f64).sqrt()
    }

    #[test]
    fn refined_ba_keeps_observations_at_realistic_coarse_error() {
        let sc = scene(0.3, 0.0035, 0.08);
        let (points, obs, st) = triangulate_tracks(&sc.coarse, &k(), &sc.tracks);
        let coarse_rms = rms_over(&sc.coarse, &points, &obs);
        eprintln!("coarse rms {coarse_rms:.2} px {st:?}");
        assert!((3.0..=5.5).contains(&coarse_rms), "{coarse_rms}");
        // 점 문턱이 고정 0.7 px 이 아니라 분포에서 정해진다.
        assert!(st.thr_px > 2.0, "{st:?}");
        assert!(st.obs_ba as f64 >= 0.9 * st.obs_all as f64, "{st:?}");
        assert!(st.points as f64 >= 0.9 * sc.n_pts as f64, "{st:?}");
        let mut s = Sparse {
            poses: sc.coarse.clone(),
            points,
            obs,
            rms: 0.0,
        };
        let before = s.obs.iter().map(Vec::len).sum::<usize>();
        let mut rng = Rng(5);
        let gps: Vec<Vector3<f64>> = sc
            .poses
            .iter()
            .map(|p| p.center().coords + Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.5)
            .collect();
        s.rms = run_ba(&mut s, &k(), 15, Some(&gps));
        let after = s.obs.iter().map(Vec::len).sum::<usize>();
        assert_eq!(before, after);
        // 거르지 않은 전체 관측(정답 점 기준 점별 목록)에 대한 정밀 재투영.
        let all: Vec<Vec<_>> = {
            // 점 번호가 어긋나지 않게 BA 점별 관측을 원래 트랙에서 다시 찾는다.
            s.obs
                .iter()
                .map(|o| {
                    let id = o[0].1 / 100;
                    sc.tracks[id].clone()
                })
                .collect()
        };
        let all_rms = rms_over(&s.poses, &s.points, &all);
        let mut errs: Vec<f64> = s
            .poses
            .iter()
            .zip(&sc.poses)
            .map(|(a, b)| (a.unwrap().center() - b.center()).norm())
            .collect();
        errs.sort_by(f64::total_cmp);
        let med = errs[errs.len() / 2];
        eprintln!(
            "refined rms {:.3} over unfiltered {all_rms:.3}; center err median {med:.2} max {:.2}; obs {before}/{}",
            s.rms,
            errs[errs.len() - 1],
            st.obs_all
        );
        assert!(s.rms < 0.7 && all_rms < 0.7, "{} {all_rms}", s.rms);
        assert!(med < 1.5, "{med}");
        let _ = &sc.truth;
    }
}
