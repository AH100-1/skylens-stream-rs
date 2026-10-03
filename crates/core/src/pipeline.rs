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
use crate::dense::{region_cloud, region_cloud_patchmatch, DenseConfig, DenseView};
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
use crate::translation_averaging::{
    average_translations_with_points, PointObservation, RelativeTranslation, TranslationConfig,
};
use crate::two_view::{ransac_essential, recover_pose};

/// 밀집 단계 사진별 깊이 방식.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseMethod {
    /// 평면 스윕(기존).
    Sweep,
    /// 패치매치(이웃 8장).
    PatchMatch,
}

/// 실행 설정.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub max_features: usize,
    /// 밀집 깊이 맵 폭(px).
    pub dense_width: usize,
    /// 밀집 깊이 방식.
    pub dense_method: DenseMethod,
    /// 수평 화각(도). 데이터셋에 내부 파라미터가 없으므로 받는다.
    pub hfov_deg: f64,
    pub ba_iters: usize,
    /// 초벌 위치 추정 방식.
    pub position: PositionMethod,
    /// 초벌 삼각측량의 느슨한 문턱(영상 폭 비율). 점 후보 선별과 BA 관측 목록의 바깥 한계.
    pub tri_loose_frac: f64,
    /// 점 문턱 = 이 배수 × 후보 관측 재투영 중앙값.
    pub tri_median_k: f64,
    /// 점 문턱의 하한(px).
    pub tri_min_px: f64,
    /// 정밀 BA GPS 사전항 σ, 수평(m). BA 사전항이 등방(카메라별 스칼라 σ)이라
    /// 실제로는 수평 2 : 수직 1 가중 제곱평균 √((2σh²+σv²)/3) 한 값을 쓴다.
    pub gps_sigma_h: f64,
    /// 정밀 BA GPS 사전항 σ, 수직(m).
    pub gps_sigma_v: f64,
    /// 초벌 점군을 만들기 전 GPS 사전항 BA 반복 수. 0 이면 BA 없이(SPEC §초벌) 닮음 정렬 포즈 그대로.
    pub preview_ba_iters: usize,
}

impl PipelineConfig {
    /// BA 사전항에 넘기는 등방 σ.
    pub fn prior_sigma(&self) -> f64 {
        let (h, v) = (self.gps_sigma_h, self.gps_sigma_v);
        ((2.0 * h * h + v * v) / 3.0).sqrt()
    }
}

/// 초벌 위치 추정 방식.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionMethod {
    /// 짝 방향 + GPS 사전의 선형 최소제곱(`stand_in::solve_centers`).
    GpsLeastSquares,
    /// 짝 방향 + 점 방향 제약 위치 평균(`average_translations_with_points`), 축척·원점은 GPS 로.
    TranslationAveraging,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_features: 1500,
            dense_width: 160,
            dense_method: DenseMethod::Sweep,
            hfov_deg: 65.0,
            ba_iters: 15,
            position: PositionMethod::GpsLeastSquares,
            tri_loose_frac: 0.02,
            tri_median_k: 3.0,
            tri_min_px: 0.7,
            gps_sigma_h: 2.0,
            gps_sigma_v: 2.0,
            preview_ba_iters: 0,
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
    /// 점 문턱을 못 넘었지만 느슨한 문턱 안에서 삼각측량되는 점(BA 에만 넣는다. 점 구름엔 안 쓴다).
    #[allow(clippy::type_complexity)]
    ba_only: Vec<(Vector3<f64>, Vec<(usize, usize, Vector2<f64>)>)>,
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

/// 검증된 짝 대응을 `tracks::build_tracks` 로 다시점 트랙으로 묶는다. 반환은 성분별 (사진, 특징) 목록.
fn multi_view_tracks(imgs: &[&ImgData], ms: &[PairList]) -> Vec<Vec<(usize, usize)>> {
    let keypoints: Vec<Vec<Vector2<f64>>> = imgs
        .iter()
        .map(|d| {
            d.feats
                .iter()
                .map(|f| Vector2::new(f.kp.x as f64 + 0.5, f.kp.y as f64 + 0.5))
                .collect()
        })
        .collect();
    let pairs: Vec<crate::tracks::PairMatches> = ms
        .iter()
        .map(|(i, j, m)| crate::tracks::PairMatches {
            image_a: *i,
            image_b: *j,
            matches: m.clone(),
        })
        .collect();
    let cfg = crate::tracks::TrackConfig {
        min_length: 2,
        ..Default::default()
    };
    let (tracks, st) = crate::tracks::build_tracks(&pairs, &keypoints, &cfg);
    if std::env::var("PIPE_DEBUG").is_ok() {
        let mut h = [0usize; 8];
        for t in &tracks {
            h[t.len().min(7)] += 1;
        }
        eprintln!("debug tracks {} len-hist(0..7+) {h:?} {st:?}", tracks.len());
    }
    tracks
        .iter()
        .map(|t| {
            t.observations
                .iter()
                .map(|o| (o.image, o.feature))
                .collect()
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

impl TriConfig {
    fn from_config(c: &PipelineConfig) -> Self {
        Self {
            loose_frac: c.tri_loose_frac,
            median_k: c.tri_median_k,
            min_px: c.tri_min_px,
            min_deg: 4.0,
        }
    }
}

/// 삼각측량 집계(관측 수는 초벌 점 문턱 전·후, BA 에 넘기는 수).
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
struct TriStats {
    tracks: usize,
    points: usize,
    obs_all: usize,
    obs_coarse: usize,
    obs_ba: usize,
    points_ba_only: usize,
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
    tri: &TriConfig,
) -> (
    Vec<Vector3<f64>>,
    Vec<Vec<(usize, usize, Vector2<f64>)>>,
    Vec<(Vector3<f64>, Vec<(usize, usize, Vector2<f64>)>)>,
    TriStats,
) {
    let loose = (tri.loose_frac * k.width as f64).max(6.0);
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
    // 1 단계: 느슨한 문턱. 점·남긴 관측의 재투영 오차.
    type Cand = (Vector3<f64>, Vec<f64>);
    let cand: Vec<Option<Cand>> = tracks
        .par_iter()
        .map(|o| {
            let cams = cams_of(o);
            if cams.len() != o.len() {
                return None;
            }
            let (x, keep) = stand_in::triangulate_robust(&cams, loose, tri.min_deg)?;
            let e = cams
                .iter()
                .zip(keep)
                .filter(|&(_, kp)| kp)
                .filter_map(|((c, px), _)| Some((c.project(&Point3::from(x))? - px).norm()))
                .collect();
            Some((x, e))
        })
        .collect();
    let mut errs: Vec<f64> = cand
        .iter()
        .flatten()
        .flat_map(|c| c.1.iter().copied())
        .collect();
    errs.sort_by(f64::total_cmp);
    let median = errs.get(errs.len() / 2).copied().unwrap_or(0.0);
    let thr = (tri.median_k * median).clamp(tri.min_px, loose);
    // 느슨한 문턱 안의 관측 모두.
    let loose_obs = |o: &[(usize, usize, Vector2<f64>)], x: &Vector3<f64>| -> Vec<_> {
        o.iter()
            .zip(cams_of(o))
            .filter(|(_, (c, px))| {
                c.project(&Point3::from(*x))
                    .is_some_and(|q| (q - px).norm() < loose)
            })
            .map(|(v, _)| *v)
            .collect()
    };
    // 2 단계: 점 문턱. 넘으면 점 구름 점, 못 넘으면 BA 전용 점. 둘 다 BA 관측은 느슨한 문턱 안 전부.
    type Res = (bool, Vector3<f64>, Vec<(usize, usize, Vector2<f64>)>, usize);
    let res: Vec<Option<Res>> = tracks
        .par_iter()
        .zip(cand.par_iter())
        .map(|(o, c)| {
            let (x0, _) = c.as_ref()?;
            let cams = cams_of(o);
            match stand_in::triangulate_robust(&cams, thr, tri.min_deg) {
                Some((x, keep)) => {
                    let nk = keep.iter().filter(|&&kp| kp).count();
                    Some((true, x, loose_obs(o, &x), nk))
                }
                None => Some((false, *x0, loose_obs(o, x0), 0)),
            }
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
    let (mut points, mut obs, ba_only) = (Vec::new(), Vec::new(), Vec::new());
    for (tight, x, ba, nk) in res.into_iter().flatten() {
        st.obs_ba += ba.len();
        if tight {
            st.obs_coarse += nk;
            points.push(x);
            obs.push(ba);
        }
        // 점 문턱을 못 넘은 후보는 BA 에 넣지 않는다: 합성 장면 측정에서 넣었을 때 정밀 카메라
        // 중심 최대 오차가 2.5 m 에서 8.4 m 로 나빠졌다(`ba_only` 는 그 실험용 자리).
    }
    st.points = points.len();
    st.points_ba_only = ba_only.len();
    (points, obs, ba_only, st)
}

/// 위치 평균: 짝 방향 + 트랙 점 방향 제약으로 중심을 구하고, 방향은 `g`(모델 → ENU)로 돌린 뒤
/// 축척·원점만 GPS 에 맞춘다(강건 재가중). 쓸 수 없으면 None.
fn averaged_centers(
    imgs: &[&ImgData],
    pm: &[PairMatch],
    gps: &[Vector3<f64>],
    k: &Intrinsics,
    rots: &[Option<Rotation3<f64>>],
    g: &Matrix3<f64>,
) -> Option<Vec<Option<Vector3<f64>>>> {
    let n = imgs.len();
    let rel: Vec<RelativeTranslation> = pm
        .iter()
        .filter_map(|p| {
            let t = p.t?;
            (t.norm() > 0.0).then_some(RelativeTranslation {
                i: p.i,
                j: p.j,
                direction: t,
                rotation: Some(p.rot),
                weight: p.inl.len() as f64,
            })
        })
        .collect();
    let counts: Vec<usize> = imgs.iter().map(|d| d.feats.len()).collect();
    let ms: Vec<_> = pm.iter().map(|p| (p.i, p.j, p.inl.clone())).collect();
    let mut tracks: Vec<Vec<(usize, usize)>> = stand_in::build_tracks(&counts, &ms)
        .into_iter()
        .filter(|t| t.iter().filter(|&&(i, _)| rots[i].is_some()).count() >= 3)
        .collect();
    tracks.sort_by_key(|t| std::cmp::Reverse(t.len()));
    tracks.truncate(3000);
    let mut pts = Vec::new();
    for (pi, tr) in tracks.iter().enumerate() {
        for &(i, f) in tr {
            if rots[i].is_none() {
                continue;
            }
            let b = norm(k, &imgs[i].feats[f]);
            pts.push(PointObservation {
                camera: i,
                point: pi,
                bearing: Vector3::new(b.x, b.y, 1.0),
                weight: 1.0,
            });
        }
    }
    let res = average_translations_with_points(rots, &rel, &pts, &TranslationConfig::default());
    if res.registered() < 2 {
        return None;
    }
    // 방향을 ENU 로 돌린 중심에 축척 s·원점 t 만 맞춘다: gps ≈ s·(g c) + t.
    let idx: Vec<usize> = (0..n).filter(|&i| res.centers[i].is_some()).collect();
    let src: Vec<Vector3<f64>> = idx
        .iter()
        .map(|&i| g * res.centers[i].unwrap().coords)
        .collect();
    let dst: Vec<Vector3<f64>> = idx.iter().map(|&i| gps[i]).collect();
    let mut keep = vec![true; idx.len()];
    let (mut s, mut t) = (1.0, Vector3::zeros());
    for _ in 0..6 {
        let w: Vec<usize> = (0..idx.len()).filter(|&a| keep[a]).collect();
        if w.len() < 3 {
            return None;
        }
        let m = w.len() as f64;
        let ms = w.iter().map(|&a| src[a]).sum::<Vector3<f64>>() / m;
        let md = w.iter().map(|&a| dst[a]).sum::<Vector3<f64>>() / m;
        let num: f64 = w.iter().map(|&a| (src[a] - ms).dot(&(dst[a] - md))).sum();
        let den: f64 = w.iter().map(|&a| (src[a] - ms).norm_squared()).sum();
        if !(den > 1e-18 && num > 0.0) {
            return None;
        }
        s = num / den;
        t = md - s * ms;
        let mut r: Vec<f64> = (0..idx.len())
            .map(|a| (s * src[a] + t - dst[a]).norm())
            .collect();
        let mut sorted = r.clone();
        sorted.sort_by(f64::total_cmp);
        let thr = (3.0 * sorted[sorted.len() / 2]).max(3.0);
        for (a, v) in r.drain(..).enumerate() {
            keep[a] = v <= thr;
        }
    }
    let mut out = vec![None; n];
    for (a, &i) in idx.iter().enumerate() {
        out[i] = Some(s * src[a] + t);
    }
    Some(out)
}

/// 초벌 포즈 단계 선택(후보 비교용). 기본값은 비교에서 가장 좋았던 조합.
#[derive(Clone, Copy, Debug)]
pub struct PreviewOpts {
    /// 회전 평균 뒤 상대 회전과 이 각(도)보다 어긋나는 간선을 빼고 다시 평균한다.
    pub prune_deg: Option<f64>,
    /// 위치 단계의 GPS 사전 가중(1/m).
    pub prior: f64,
    /// 위치 단계 뒤 카메라 중심을 GPS 에 닮음 변환 강건 추정으로 맞춘다.
    pub snap: bool,
    /// 이보다 정상 대응이 적은 간선은 회전 평균에 넣지 않는다.
    pub min_inl: usize,
    /// 어긋난 간선 제거를 반복하는 횟수(`prune_deg` 가 있을 때).
    pub passes: usize,
}

impl Default for PreviewOpts {
    fn default() -> Self {
        Self {
            prune_deg: Some(10.0),
            prior: 1.0,
            snap: false,
            min_inl: 0,
            passes: 1,
        }
    }
}

impl PreviewOpts {
    /// 비교 실험용: `prune=10,prior=0.5,snap=1` 꼴 문자열로 기본값을 덮어쓴다.
    pub fn parse(spec: &str) -> Self {
        let mut o = Self::default();
        for kv in spec.split(',') {
            match kv.split_once('=') {
                Some(("prune", v)) => o.prune_deg = v.parse().ok().filter(|d: &f64| *d > 0.0),
                Some(("prior", v)) => o.prior = v.parse().unwrap_or(o.prior),
                Some(("snap", v)) => o.snap = v == "1",
                Some(("min_inl", v)) => o.min_inl = v.parse().unwrap_or(0),
                Some(("passes", v)) => o.passes = v.parse().unwrap_or(1),
                _ => {}
            }
        }
        o
    }
}

/// 초벌 포즈 단계별 중간 결과(진단용).
#[derive(Clone, Debug, Default)]
pub struct PreviewStages {
    /// 회전 평균 직후 회전(모델 좌표계, 좌표계 맞춤 전).
    pub rots: Vec<Option<Rotation3<f64>>>,
    /// 좌표계 맞춤 + 위치 단계 직후 포즈(GPS 닮음 변환 전).
    pub placed: Vec<Option<Pose>>,
    /// 회전 평균 뒤 뺀 간선 수 / 전체.
    pub pruned: (usize, usize),
}

type RotsAndKeep = (Vec<Option<Rotation3<f64>>>, Vec<bool>);

/// 회전 평균 + 상대 회전과 어긋나는 간선 제거 뒤 재평균. 반환: 회전, 간선 유지 표시.
fn average_pruned(n: usize, pm: &[PairMatch], opts: &PreviewOpts) -> Result<RotsAndKeep, String> {
    let mk = |keep: &[bool]| -> Vec<RelativeRotation> {
        pm.iter()
            .zip(keep)
            .filter(|(_, &k)| k)
            .map(|(p, _)| RelativeRotation {
                i: p.i,
                j: p.j,
                rotation: p.rot,
                weight: p.inl.len() as f64,
            })
            .collect()
    };
    let mut keep: Vec<bool> = pm.iter().map(|p| p.inl.len() >= opts.min_inl).collect();
    let ra = average_rotations(n, &mk(&keep), &AveragingConfig::default())
        .ok_or("회전 평균 실패: 쓸 수 있는 간선 없음")?;
    let mut rots = ra.rotations;
    if let Some(deg) = opts.prune_deg {
        for _ in 0..opts.passes.max(1) {
            for (kp, p) in keep.iter_mut().zip(pm) {
                *kp = *kp
                    && match (rots[p.i], rots[p.j]) {
                        (Some(a), Some(b)) => {
                            (b * a.inverse() * p.rot.inverse()).angle().to_degrees() <= deg
                        }
                        _ => false,
                    };
            }
            if let Some(r2) = average_rotations(n, &mk(&keep), &AveragingConfig::default()) {
                rots = r2.rotations;
            }
        }
    }
    Ok((rots, keep))
}

/// 위치 단계 뒤 카메라 중심을 GPS 에 닮음 변환 강건 추정으로 맞춘다(중심·방향 모두).
fn snap_poses_to_gps(poses: &mut [Option<Pose>], gps: &[Vector3<f64>]) {
    let ids: Vec<usize> = (0..poses.len()).filter(|&i| poses[i].is_some()).collect();
    let src: Vec<Vector3<f64>> = ids
        .iter()
        .map(|&i| poses[i].unwrap().center().coords)
        .collect();
    let dst: Vec<Vector3<f64>> = ids.iter().map(|&i| gps[i]).collect();
    let Some((sim, _, _)) = crate::align::robust_similarity(&src, &dst, 3, 3.0) else {
        return;
    };
    for &i in &ids {
        let p = poses[i].unwrap();
        let c = sim.apply_point(&p.center().coords);
        poses[i] = Some(Pose::from_center(
            p.rotation * sim.r.inverse(),
            &Point3::from(c),
        ));
    }
}

/// 회전 평균 → 방향으로 좌표계 맞춤 → 위치 → 삼각측량. 초벌 희소 모델.
fn sparse_init(
    imgs: &[&ImgData],
    pm: &[PairMatch],
    gps: &[Vector3<f64>],
    k: &Intrinsics,
    method: PositionMethod,
    tri: &TriConfig,
    pre_ba: (usize, f64),
) -> Result<Sparse, String> {
    sparse_init_with(
        imgs,
        pm,
        gps,
        k,
        method,
        tri,
        pre_ba,
        &PreviewOpts::default(),
    )
    .map(|r| r.0)
}

#[allow(clippy::too_many_arguments)]
fn sparse_init_with(
    imgs: &[&ImgData],
    pm: &[PairMatch],
    gps: &[Vector3<f64>],
    k: &Intrinsics,
    method: PositionMethod,
    tri: &TriConfig,
    pre_ba: (usize, f64),
    opts: &PreviewOpts,
) -> Result<(Sparse, PreviewStages), String> {
    let n = imgs.len();
    let (rots, keep_edge) = average_pruned(n, pm, opts)?;
    let mut stages = PreviewStages {
        rots: rots.clone(),
        pruned: (keep_edge.iter().filter(|&&k| !k).count(), pm.len()),
        ..Default::default()
    };
    // 좌표계 맞춤(Kabsch): 모델 방향 b = Rᵢᵀ(−R_ijᵀ t) → GPS 방향.
    let mut h = Matrix3::zeros();
    let mut dirs_model = Vec::new();
    for (p, _) in pm.iter().zip(&keep_edge).filter(|(_, &k)| k) {
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
    let mut placed: Option<Vec<Option<Vector3<f64>>>> = None;
    if method == PositionMethod::TranslationAveraging {
        placed = averaged_centers(imgs, pm, gps, k, &rots, &g);
        if placed.is_none() && std::env::var("PIPE_DEBUG").is_ok() {
            eprintln!("debug translation averaging failed, GPS least squares instead");
        }
    }
    let placed: Vec<Option<Vector3<f64>>> = match placed {
        Some(c) => c,
        None => {
            let c = stand_in::solve_centers(&gl, &dirs, opts.prior).ok_or("위치 풀이 실패")?;
            let mut v = vec![None; n];
            for (a, &i) in ids.iter().enumerate() {
                v[i] = Some(c[a]);
            }
            v
        }
    };
    let mut poses: Vec<Option<Pose>> = vec![None; n];
    for &i in &ids {
        let Some(c) = placed[i] else { continue };
        let r = Rotation3::from_matrix_unchecked(rots[i].unwrap().matrix() * g.transpose());
        poses[i] = Some(Pose::from_center(r, &Point3::from(c)));
    }
    stages.placed = poses.clone();
    if opts.snap {
        snap_poses_to_gps(&mut poses, gps);
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
    let ms: Vec<_> = pm
        .iter()
        .filter(|p| poses[p.i].is_some() && poses[p.j].is_some())
        .map(|p| (p.i, p.j, p.inl.clone()))
        .collect();
    let tracks = multi_view_tracks(imgs, &ms);
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
    let (points, obs, ba_only, stats) = triangulate_tracks(&poses, k, &track_obs, tri);
    if std::env::var("PIPE_DEBUG").is_ok() {
        eprintln!("debug triangulation {stats:?}");
    }
    let mut s = Sparse {
        poses,
        points,
        obs,
        ba_only,
        rms: 0.0,
    };
    s.rms = run_ba(&mut s, k, 0, None, 2.0, &[]);
    if pre_ba.0 > 0 {
        // 짧은 GPS 사전항 BA: 초벌 포즈·점의 스케일·기울기·깊이를 정밀 쪽으로 당긴다.
        let after = run_ba(&mut s, k, pre_ba.0, Some(gps), pre_ba.1, &[]);
        if std::env::var("PIPE_DEBUG").is_ok() {
            eprintln!("debug preview ba rms {:.3} -> {after:.3}", s.rms);
        }
        s.rms = after;
    }
    Ok((s, stages))
}

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
    apply_sparse_sim(s, &sim);
    Some(sim)
}

/// 희소 모델 전체(포즈·점·BA 전용 점)에 닮음 변환을 적용한다.
fn apply_sparse_sim(s: &mut Sparse, sim: &Similarity) {
    for p in s.poses.iter_mut().flatten() {
        let c = sim.apply_point(&p.center().coords);
        *p = Pose::from_center(p.rotation * sim.r.inverse(), &Point3::from(c));
    }
    for p in s.points.iter_mut() {
        *p = sim.apply_point(p);
    }
    for e in s.ba_only.iter_mut() {
        e.0 = sim.apply_point(&e.0);
    }
}

/// 다음 구역의 정밀 BA 를 직전 정밀 모델 기준으로 시작할지. 3구역 측정(재투영 RMS 0.81/0.51 px 대 0.28/0.32 px)에서
/// 초벌 기반 닮음 변환의 잔차(1.7~3.1 m)가 커서 오히려 나빠져 기본은 끈다.
const ANCHOR_NEXT_REGION: bool = false;

/// 직전 정밀 모델에 맞춘 시작: 닮음 변환으로 같은 좌표계로 옮기고 공유 사진 포즈를 직전 값으로 고정.
struct Anchor {
    sim: Similarity,
    /// (구역 안 사진 번호, 직전 정밀 포즈)
    fixed: Vec<(usize, Pose)>,
}

/// 새 구역 `rec` 의 초벌 희소 모델을 직전 정밀 구역 `prev` 에 맞출 기준을 만든다.
/// 반환: 기준(닮음 변환 + 고정 포즈), 점 쌍 수, 잔차 중앙값. 겹침이 모자라면 `None`.
fn make_anchor(rec: &RegionRec, prev: &RegionRec) -> Option<(Anchor, usize, f64)> {
    if !ANCHOR_NEXT_REGION {
        return None;
    }
    let tb = &prev.refined.as_ref()?.0;
    let win = crate::progressive::overlap_window(&rec.region, &prev.region);
    let (sim, n, med) = crate::progressive::cross_align(&rec.ta, tb, win)?;
    let fixed: Vec<(usize, Pose)> = rec
        .gids
        .iter()
        .enumerate()
        .filter(|&(a, _)| rec.reg_flags[a])
        .filter_map(|(a, g)| prev.rposes.get(g).map(|p| (a, *p)))
        .collect();
    Some((Anchor { sim, fixed }, n, med))
}

/// 번들 조정(`iters == 0` 이면 재투영 오차만 잰다). 반환: 재투영 RMS(px).
fn run_ba(
    s: &mut Sparse,
    k: &Intrinsics,
    iters: usize,
    gps: Option<&[Vector3<f64>]>,
    prior_sigma: f64,
    fixed: &[usize],
) -> f64 {
    let ids: Vec<usize> = (0..s.poses.len())
        .filter(|&i| s.poses[i].is_some())
        .collect();
    let loc: HashMap<usize, usize> = ids.iter().enumerate().map(|(a, &i)| (i, a)).collect();
    let n_main = s.points.len();
    // 점 문턱을 못 넘은 점도 BA 에는 넣는다(`iters == 0` 인 초벌 재투영 측정에는 넣지 않는다).
    let extra: &[_] = if iters > 0 { &s.ba_only } else { &[] };
    let mut observations = Vec::new();
    let all_obs = s.obs.iter().chain(extra.iter().map(|e| &e.1));
    for (p, o) in all_obs.enumerate() {
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
        points: s
            .points
            .iter()
            .chain(extra.iter().map(|e| &e.0))
            .map(|p| Point3::from(*p))
            .collect(),
        observations,
    };
    let opts = BaOptions {
        max_iterations: iters,
        default_free_intrinsics: [false; 8],
        fixed_cameras: if fixed.is_empty() {
            vec![0]
        } else {
            fixed.iter().filter_map(|i| loc.get(i).copied()).collect()
        },
        position_prior: gps.map(|g| {
            let mut pr =
                PositionPrior::new(ids.iter().map(|&i| Some(Point3::from(g[i]))).collect());
            pr.sigma = prior_sigma;
            pr
        }),
        ..BaOptions::default()
    };
    let rep = bundle_adjust(&mut prob, &opts);
    if iters > 0 {
        for (a, &i) in ids.iter().enumerate() {
            s.poses[i] = Some(prob.poses[a]);
        }
        s.points = prob.points[..n_main].iter().map(|p| p.coords).collect();
        for (e, p) in s.ba_only.iter_mut().zip(&prob.points[n_main..]) {
            e.0 = p.coords;
        }
    }
    if iters == 0 {
        rep.initial_rms
    } else {
        rep.final_rms
    }
}

/// 점마다 관측 광선 사이의 최대 각(도). 광선 각이 작은 점은 깊이가 불안정하다.
fn ray_angles(s: &Sparse) -> Vec<f64> {
    s.points
        .iter()
        .zip(&s.obs)
        .map(|(x, o)| {
            let rays: Vec<Vector3<f64>> = o
                .iter()
                .filter_map(|&(i, _, _)| {
                    let d = x - s.poses[i]?.center().coords;
                    (d.norm() > 1e-9).then(|| d.normalize())
                })
                .collect();
            let mut best = 0.0f64;
            for a in 0..rays.len() {
                for b in a + 1..rays.len() {
                    best = best.max(rays[a].angle(&rays[b]).to_degrees());
                }
            }
            best
        })
        .collect()
}

/// 광선 각이 `min_deg` 이상인 점만 남긴 모델. 남는 점이 `keep_min` 개 미만이면 그대로 둔다.
fn well_conditioned(s: &Sparse, min_deg: f64, keep_min: usize) -> Sparse {
    let ang = ray_angles(s);
    let keep: Vec<usize> = (0..s.points.len()).filter(|&p| ang[p] >= min_deg).collect();
    if keep.len() < keep_min {
        return s.clone();
    }
    Sparse {
        poses: s.poses.clone(),
        points: keep.iter().map(|&p| s.points[p]).collect(),
        obs: keep.iter().map(|&p| s.obs[p].clone()).collect(),
        ba_only: s.ba_only.clone(),
        rms: s.rms,
    }
}

/// 초벌 점의 최소 광선 각(도): 이보다 좁은 점은 초벌 점군·정렬 대응에서 뺀다.
const PREVIEW_MIN_RAY_DEG: f64 = 2.0;

/// 밀집: 구역 사진을 `dense::region_cloud`(보정·이웃·사진별 깊이·융합)에 넘긴다. 희소 점도 함께 담는다.
/// `dw` 는 보정 뒤 긴 변 화소 수다. 밀집 점이 하나도 안 나오면 희소 점 보간 깊이로 대신한다.
fn dense_cloud(
    s: &Sparse,
    imgs: &[&ImgData],
    k: &Intrinsics,
    in_region: &[bool],
    dw: usize,
    method: DenseMethod,
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
    // 융합 문턱은 자세 정밀도에 맞춘다: 재투영 오차 상한 = 1 px + 4 × (BA 재투영 RMS 를 밀집 해상도로 환산),
    // 상대 깊이 상한은 같은 배수. 느슨한 초벌 자세(RMS ≈ 2.5 px)에서도 이웃 동의가 남게 한다.
    let rms_dense = s.rms * dw as f64 / k.width as f64;
    let scale = (1.0 + 4.0 * rms_dense).clamp(1.0, 6.0);
    let cfg = DenseConfig {
        max_width: dw as u32,
        reproj_px: DenseConfig::default().reproj_px * scale,
        depth_rel: DenseConfig::default().depth_rel * scale,
        ..DenseConfig::default()
    };
    let mut cloud = match method {
        DenseMethod::Sweep => region_cloud(&views, &pts, &cfg),
        DenseMethod::PatchMatch => region_cloud_patchmatch(&views, &pts, &cfg),
    };
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
    /// 정밀 구역 → 가장 최근 정밀 모델 좌표계 닮음 변환(공유 3D 점 대응).
    rsim: Option<Similarity>,
    centers: BTreeMap<usize, [f64; 3]>,
    registered_prev: Vec<usize>,
    /// gids 앞쪽 보조 사진 수(출력·점수 제외).
    n_help: usize,
    /// 정밀 모델의 사진별 포즈(다음 구역의 고정 기준).
    rposes: HashMap<usize, Pose>,
    /// 이 구역 정밀 작업이 직전 구역 정밀 결과를 기다리는 통로.
    anchor_tx: Option<std::sync::mpsc::Sender<Option<Anchor>>>,
    /// 기준 구역과의 정렬 기록: (직전 구역 번호, 점 쌍 수, 잔차 중앙 m, 스케일).
    anchored: Option<(usize, usize, f64, f64)>,
    /// 등록된 사진 표시(gids 순서).
    reg_flags: Vec<bool>,
}

/// 지금까지 내보낸 구역 상태(스냅샷 합성용).
fn live_state(recs: &[RegionRec]) -> Vec<crate::pipeline_stream::LiveRegion<'_>> {
    recs.iter()
        .map(|r| crate::pipeline_stream::LiveRegion {
            region: r.region,
            coarse: &r.coarse,
            coarse_sim: r.sim,
            refined: r.refined.as_ref().map(|x| &x.1),
            refined_sim: r.rsim,
        })
        .collect()
}

/// 정밀(BA) 작업 결과.
struct RefinedMsg {
    slot: usize,
    sparse: Sparse,
    cloud: PointCloud,
    secs: f64,
}

/// 기준을 정밀 작업 스레드에 보내고 기록한다(`None` 이면 기준 없이 시작).
fn send_anchor(
    rec: &mut RegionRec,
    tx: &std::sync::mpsc::Sender<Option<Anchor>>,
    a: Option<(Anchor, usize, f64)>,
    prev: usize,
    secs: f64,
    events: &mut Vec<String>,
) {
    match a {
        Some((an, n, med)) => {
            events.push(format!(
                "{secs:.1}s anchor region {} on refined {prev} pairs {n} median {med:.3} m scale {:.4} fixed {}",
                rec.region.index,
                an.sim.s,
                an.fixed.len()
            ));
            rec.anchored = Some((prev, n, med, an.sim.s));
            let _ = tx.send(Some(an));
        }
        None => {
            events.push(format!(
                "{secs:.1}s anchor region {} none (overlap too small)",
                rec.region.index
            ));
            let _ = tx.send(None);
        }
    }
}

fn write_decimated(out: &Path, name: &str, cloud: &PointCloud) -> Result<(), String> {
    crate::ply::write_ply_file(
        out.join(name),
        &crate::stream::decimate(cloud, crate::stream::DECIMATE_EVERY),
    )
    .map_err(|e| format!("출력 쓰기 실패: {e}"))
}

/// 구역 앞쪽 보조 F 사진 범위: 위치 [lo-HELPER_SPAN, lo-HELPER_MIN].
/// 짝 규칙상 R(p)·L(p) 는 F(p-40..=p-12) 와 겹치므로 구역 첫 12 위치의 R·L 은 구역 바로 앞 F 까지 필요하다.
const HELPER_SPAN: usize = 40;
const HELPER_MIN: usize = 1;

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
    run_pipeline_with(
        ds,
        cfg,
        out,
        crate::pipeline_stream::StreamOptions::default(),
    )
}

/// [`run_pipeline`] 의 실행 방식 지정판. 사건 순서는 `snapshots/manifest.json` 의 `events` 와
/// `snapshots/live/` 의 시점별 스냅샷으로 남는다.
pub fn run_pipeline_with(
    ds: &Dataset,
    cfg: &PipelineConfig,
    out: &Path,
    opts: crate::pipeline_stream::StreamOptions,
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
    let mut live = crate::pipeline_stream::LiveLog::new(out).map_err(|e| e.to_string())?;

    // 정밀 결과 하나를 반영: 파일 쓰기, 자기 정렬, 대기 중인 초벌 재정렬.
    let handle = |m: RefinedMsg,
                  recs: &mut Vec<RegionRec>,
                  events: &mut Vec<String>,
                  realigns: &mut Vec<ReAlign>,
                  latest_ref: &mut Option<usize>,
                  live: &mut crate::pipeline_stream::LiveLog|
     -> Result<(), String> {
        let k = m.slot;
        let rec = &mut recs[k];
        let tb = to_tracks(&m.sparse, &rec.gids);
        for (a, g) in rec.gids.iter().enumerate().skip(rec.n_help) {
            if let Some(p) = m.sparse.poses[a] {
                let c = p.center();
                rec.centers.insert(*g, [c.x, c.y, c.z]);
            }
        }
        for (a, g) in rec.gids.iter().enumerate() {
            if let Some(p) = m.sparse.poses[a] {
                rec.rposes.insert(*g, p);
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
            rec.sim = sim;
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
        live.snapshot(t_now(), "refined_replace", r.index, &live_state(recs))?;
        // 다음 구역의 정밀 작업이 이 모델을 기다리고 있으면 기준을 보낸다.
        if k + 1 < recs.len() {
            if let Some(tx) = recs[k + 1].anchor_tx.take() {
                let a = make_anchor(&recs[k + 1], &recs[k]);
                send_anchor(&mut recs[k + 1], &tx, a, k, t_now(), events);
            }
        }
        events.push(format!(
            "{:.1}s refined region {} rms {:.3} done",
            t_now(),
            r.index,
            m.sparse.rms
        ));
        let n_realign0 = realigns.len();
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
        // 이미 내보낸 정밀 구역도 새 정밀 모델 좌표계로 다시 맞춘다(공유 3D 점 대응 닮음 변환).
        // 새 구역과 겹치지 않는 구역은 겹치는 이웃을 거쳐 변환을 연쇄 합성한다.
        let mut total: Vec<Option<Similarity>> = vec![None; recs.len()];
        total[k] = Some(Similarity::identity());
        let mut queue = std::collections::VecDeque::from([k]);
        while let Some(m) = queue.pop_front() {
            for j in 0..recs.len() {
                if total[j].is_some() || recs[j].refined.is_none() {
                    continue;
                }
                let tm = &recs[m].refined.as_ref().unwrap().0;
                let tj = &recs[j].refined.as_ref().unwrap().0;
                let Some((s, n, med)) = crate::pipeline_stream::realign_refined(
                    (&recs[j].region, tj),
                    (&recs[m].region, tm),
                ) else {
                    continue;
                };
                let acc = total[m].as_ref().unwrap().compose(&s);
                realigns.push(ReAlign {
                    secs: t_now(),
                    region: j,
                    target: k,
                    pairs: n,
                    median_m: med,
                    scale: acc.s,
                });
                let jr = recs[j].region;
                let moved = crate::stream::apply_cloud(&acc, &recs[j].refined.as_ref().unwrap().1);
                write_decimated(out, &refined_name(&jr), &moved)?;
                events.push(format!(
                    "{:.1}s realign refined {} to refined {} via {} pairs {n} median {med:.3} m",
                    t_now(),
                    jr.index,
                    recs[k].region.index,
                    recs[m].region.index
                ));
                recs[j].rsim = Some(acc);
                total[j] = Some(acc);
                queue.push_back(j);
            }
        }
        if realigns.len() > n_realign0 {
            live.snapshot(t_now(), "realign", r.index, &live_state(recs))?;
        }
        Ok(())
    };

    for r in &regions {
        // 끝난 정밀 결과를 먼저 반영해 이번 등록·정렬이 최신 모델을 기준으로 삼게 한다.
        while let Ok(m) = rx.try_recv() {
            in_flight -= 1;
            handle(
                m,
                &mut recs,
                &mut events,
                &mut realigns,
                &mut latest_ref,
                &mut live,
            )?;
        }
        // 정밀 작업이 둘 넘게 밀리면 하나가 끝나길 기다린다(코어 과다 경쟁 방지).
        while in_flight >= 2 {
            let m = rx.recv().map_err(|e| e.to_string())?;
            in_flight -= 1;
            handle(
                m,
                &mut recs,
                &mut events,
                &mut realigns,
                &mut latest_ref,
                &mut live,
            )?;
        }
        let mut st = RegionStats {
            region: r.index,
            positions: r.hi - r.lo,
            images: 3 * (r.hi - r.lo),
            ..Default::default()
        };
        events.push(format!("{:.1}s arrive region {}", t_now(), r.index));
        // 구역 시작 쪽 R·L 은 F(p-40..=p-20) 와만 겹친다(F-197). 구역 밖 앞쪽 F 사진을 보조로 넣어
        // 구역 첫 위치들의 카메라 간 짝이 끊기지 않게 한다. 보조 사진은 출력·점수에 넣지 않는다.
        let helper_lo = r.lo.saturating_sub(HELPER_SPAN);
        let helper_hi = if r.lo >= HELPER_MIN {
            r.lo - HELPER_MIN + 1
        } else {
            0
        };
        let helpers: Vec<usize> = (helper_lo..helper_hi.max(helper_lo))
            .map(|p| 3 * p)
            .collect();
        let n_help = helpers.len();
        let gids: Vec<usize> = helpers
            .into_iter()
            .chain((r.lo..r.hi).flat_map(|p| (0..3).map(move |c| 3 * p + c)))
            .collect();
        let t0 = Instant::now();
        let need: Vec<usize> = gids
            .iter()
            .copied()
            .filter(|g| !cache.contains_key(g))
            .collect();
        // 사진은 위치 단위로 차례로 도착한다(위치마다 세 카메라 사진을 함께 읽는다).
        let mut load_err = None;
        let mut by_pos: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &g in &need {
            by_pos.entry(g / 3).or_default().push(g);
        }
        for (p, gs) in &by_pos {
            let loaded: Vec<Result<(usize, ImgData), String>> = gs
                .par_iter()
                .map(|&g| {
                    Ok((
                        g,
                        load(&ds.positions[g / 3].images[g % 3], cfg.max_features)?,
                    ))
                })
                .collect();
            for l in loaded {
                match l {
                    Ok((g, d)) => {
                        cache.insert(g, Arc::new(d));
                    }
                    Err(e) => load_err = Some(e),
                }
            }
            if *p >= r.lo {
                live.note(t_now(), "arrive_position", *p);
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
        let tri = TriConfig::from_config(cfg);
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
        // 정지 검사는 구역 자기 위치끼리의 짝만 본다(앞 구역에서 온 도우미 사진의 움직임은 세지 않는다).
        let own_pairs: Vec<(usize, usize)> = pair_ids
            .iter()
            .copied()
            .filter(|&(i, j)| gids[i] / 3 >= r.lo && gids[j] / 3 >= r.lo)
            .collect();
        let own_registered = |s: &Sparse| s.poses[n_help..].iter().filter(|p| p.is_some()).count();
        let init = match check_motion(&gps, &views, &own_pairs)
            .and_then(|_| {
                sparse_init(
                    &imgs,
                    &pm,
                    &gps,
                    &k,
                    cfg.position,
                    &tri,
                    (cfg.preview_ba_iters, cfg.prior_sigma()),
                )
            })
            .and_then(|s| {
                if own_registered(&s) < 3 {
                    Err(format!(
                        "구역 안 사진 등록 {} 장 < 3: 특징이 없어 등록할 수 없음",
                        own_registered(&s)
                    ))
                } else {
                    Ok(s)
                }
            }) {
            Ok(s) => s,
            Err(e) => {
                skipped.push(format!("구역 {} 건너뜀: {e}", r.index));
                events.push(format!("{:.1}s skip region {}", t_now(), r.index));
                continue;
            }
        };
        st.secs_sparse = t2.elapsed().as_secs_f64();
        st.registered = init.poses[n_help..].iter().filter(|p| p.is_some()).count();
        st.tracks = init.points.len();
        st.preview_rms = init.rms;
        events.push(format!(
            "{:.1}s registered region {} ({}/{})",
            t_now(),
            r.index,
            st.registered,
            st.images
        ));
        {
            let flags: Vec<bool> = init.poses.iter().map(|p| p.is_some()).collect();
            let tab = crate::pipeline_stream::missing_by_camera(&gids, &flags);
            events.push(format!(
                "registration table region {}: cam0 {} missing {:?}; cam1 {} missing {:?}; cam2 {} missing {:?}",
                r.index, tab[0].0, tab[0].1, tab[1].0, tab[1].1, tab[2].0, tab[2].1
            ));
        }
        let slot = recs.len();
        // 초벌 점군: 곧바로 만들어 최신 정밀 좌표계로 정렬해 내보낸다.
        let t3 = Instant::now();
        let in_region: Vec<bool> = gids.iter().map(|g| r.contains(g / 3)).collect();
        let coarse = dense_cloud(
            &well_conditioned(&init, PREVIEW_MIN_RAY_DEG, 200),
            &imgs,
            &k,
            &in_region,
            cfg.dense_width,
            cfg.dense_method,
        );
        st.secs_dense = t3.elapsed().as_secs_f64();
        st.preview_points = coarse.len();
        let ta = to_tracks(&init, &gids);
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
        let shown = sim.unwrap_or_else(Similarity::identity);
        write_decimated(
            out,
            &preview_name(r),
            &crate::stream::apply_cloud(&shown, &coarse),
        )?;
        events.push(format!("{:.1}s coarse output region {}", t_now(), r.index));
        {
            let mut state = live_state(&recs);
            state.push(crate::pipeline_stream::LiveRegion {
                region: *r,
                coarse: &coarse,
                coarse_sim: sim,
                refined: None,
                refined_sim: None,
            });
            live.snapshot(t_now(), "coarse_output", r.index, &state)?;
        }
        let registered_prev = (n_help..gids.len())
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
            rsim: None,
            centers: BTreeMap::new(),
            registered_prev,
            n_help,
            rposes: HashMap::new(),
            anchor_tx: None,
            anchored: None,
            reg_flags: init.poses.iter().map(|p| p.is_some()).collect(),
        });
        // 정밀(BA)은 다른 스레드에서: 직전 구역의 정밀 모델이 나오면 그 좌표계·포즈를 기준으로 시작한다.
        let (anchor_tx, anchor_rx) = mpsc::channel::<Option<Anchor>>();
        {
            let (tx, init, arcs) = (tx.clone(), init.clone(), arcs.clone());
            let (gps, dw, iters, dmethod) =
                (gps.clone(), cfg.dense_width, cfg.ba_iters, cfg.dense_method);
            let psig = cfg.prior_sigma();
            let gids_t = recs[slot].gids.clone();
            let in_region: Vec<bool> = gids_t.iter().map(|g| r.contains(g / 3)).collect();
            in_flight += 1;
            std::thread::spawn(move || {
                let anchor = anchor_rx.recv().ok().flatten();
                let t = Instant::now();
                let imgs: Vec<&ImgData> = arcs.iter().map(|a| a.as_ref()).collect();
                let mut rs = init;
                let mut fixed: Vec<usize> = Vec::new();
                if let Some(an) = &anchor {
                    apply_sparse_sim(&mut rs, &an.sim);
                    for (i, p) in &an.fixed {
                        rs.poses[*i] = Some(*p);
                        fixed.push(*i);
                    }
                }
                rs.rms = run_ba(&mut rs, &k, iters, Some(&gps), psig, &fixed);
                if anchor.is_none() {
                    gps_align_refined(&mut rs, &gps);
                }
                let cloud = dense_cloud(&rs, &imgs, &k, &in_region, dw, dmethod);
                let _ = tx.send(RefinedMsg {
                    slot,
                    sparse: rs,
                    cloud,
                    secs: t.elapsed().as_secs_f64(),
                });
            });
        }
        if slot == 0 {
            let _ = anchor_tx.send(None);
        } else if recs[slot - 1].refined.is_some() {
            let a = make_anchor(&recs[slot], &recs[slot - 1]);
            send_anchor(
                &mut recs[slot],
                &anchor_tx,
                a,
                slot - 1,
                t_now(),
                &mut events,
            );
        } else {
            recs[slot].anchor_tx = Some(anchor_tx);
        }
        if opts.sequential {
            while in_flight > 0 {
                let m = rx.recv().map_err(|e| e.to_string())?;
                in_flight -= 1;
                handle(
                    m,
                    &mut recs,
                    &mut events,
                    &mut realigns,
                    &mut latest_ref,
                    &mut live,
                )?;
            }
        }
        // 이 구역을 올린 뒤에 끝난 정밀 결과를 반영한다(방금 올린 초벌도 재정렬 대상이다).
        while let Ok(m) = rx.try_recv() {
            in_flight -= 1;
            handle(
                m,
                &mut recs,
                &mut events,
                &mut realigns,
                &mut latest_ref,
                &mut live,
            )?;
        }
    }
    while in_flight > 0 {
        let m = rx.recv().map_err(|e| e.to_string())?;
        in_flight -= 1;
        handle(
            m,
            &mut recs,
            &mut events,
            &mut realigns,
            &mut latest_ref,
            &mut live,
        )?;
    }
    if recs.is_empty() {
        return Err(format!("모든 구역 실패: {}", skipped.join("; ")));
    }

    let kept: Vec<Region> = recs.iter().map(|r| r.region).collect();
    let sims: Vec<Option<Similarity>> = recs
        .iter()
        .map(|r| r.own.as_ref().and_then(|o| o.0))
        .collect();
    let records: Vec<AlignRecord> = recs
        .iter()
        .map(|r| r.own.as_ref().unwrap().1.clone())
        .collect();
    let prelim: Vec<PointCloud> = recs.iter().map(|r| r.coarse.clone()).collect();
    let refined: Vec<PointCloud> = recs
        .iter()
        .map(|r| {
            let c = &r.refined.as_ref().unwrap().1;
            match &r.rsim {
                Some(s) => crate::stream::apply_cloud(s, c),
                None => c.clone(),
            }
        })
        .collect();
    let aligned = apply_alignments(&prelim, &sims);
    let rep = write_outputs(out, &kept, &aligned, &refined, records.clone())
        .map_err(|e| format!("출력 쓰기 실패: {e}"))?;
    live.note(t_now(), "final", recs.last().map_or(0, |r| r.region.index));
    live.finish(out)?;
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

#[cfg(test)]
mod diag {
    //! 초벌 점 오차 분해 진단(오래 걸려 기본 시험에서 뺀다): `cargo test --release diagnose_preview -- --ignored --nocapture`.
    use super::*;
    use crate::dataset::{load_dataset, DatasetConfig};
    use crate::stream::{apply_cloud, point_pairs};
    use crate::synth::{Scene, SceneConfig};
    use crate::verify::{height_pair_median, median, nn_median};

    fn med(mut v: Vec<f64>) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        median(&mut v)
    }

    /// 합성 장면 정답 포즈(첫 GPS 기준 좌표) 와 비교한 초벌 포즈 단계별 오차 한 줄.
    pub struct StageRow {
        pub rot_free_med: f64,
        pub rot_free_max: f64,
        pub placed_c_med: f64,
        pub placed_rot_med: f64,
        pub final_c_med: f64,
        pub final_c_max: f64,
        pub final_dz_med: f64,
        pub final_rot_med: f64,
        pub pruned: (usize, usize),
        pub preview_align_m: Option<f64>,
        pub vs_refined_dz: Option<f64>,
        pub surf_med: f64,
    }

    pub fn stage_rows(list: &[PreviewOpts], full: bool) -> Vec<StageRow> {
        let root = std::env::temp_dir().join(format!("skylens_stage_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let scene = Scene::new(SceneConfig {
            width: 320,
            height: 180,
            ..SceneConfig::default()
        });
        scene.write_dataset(&root).unwrap();
        let ds = load_dataset(
            &root,
            DatasetConfig {
                stride: 2,
                span: 48,
                ovl: 2,
                max_skip_run: 2,
            },
        )
        .unwrap();
        let n = ds.positions.len() * 3;
        let gids: Vec<usize> = (0..n).collect();
        let data: Vec<ImgData> = gids
            .iter()
            .map(|g| load(&ds.positions[g / 3].images[g % 3], 800).unwrap())
            .collect();
        let imgs: Vec<&ImgData> = data.iter().collect();
        let k = Intrinsics::from_hfov(
            data[0].rgb.width(),
            data[0].rgb.height(),
            65f64.to_radians(),
        );
        let views: Vec<(usize, usize)> = gids.iter().map(|g| (g % 3, g / 3)).collect();
        let gps: Vec<Vector3<f64>> = gids
            .iter()
            .map(|g| ds.positions[g / 3].image_enu[g % 3])
            .collect();
        let pm = match_pairs(&imgs, &views, &k);
        let mut out = Vec::new();
        for opts in list {
            let (init, st) = sparse_init_with(
                &imgs,
                &pm,
                &gps,
                &k,
                PipelineConfig::default().position,
                &TriConfig::from_config(&PipelineConfig::default()),
                (0, 2.0),
                opts,
            )
            .unwrap();
            let tp: Vec<Pose> = gids
                .iter()
                .map(|&g| {
                    let name = ds.positions[g / 3].images[g % 3]
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    let v = scene.views.iter().find(|v| v.name == name).unwrap();
                    let c = scene.to_first_gps_frame(&v.camera.pose.center());
                    Pose::from_center(v.camera.pose.rotation, &c)
                })
                .collect();
            // 좌표계 무관 회전 오차: 전역 회전 Q = polar(Σ Rᵢᵀ Tᵢ) 로 맞춘 뒤.
            let mut m = Matrix3::zeros();
            for (r, t) in st.rots.iter().zip(&tp) {
                if let Some(r) = r {
                    m += r.matrix().transpose() * t.rotation.matrix();
                }
            }
            let sv = m.svd(true, true);
            let q = Rotation3::from_matrix_unchecked(sv.u.unwrap() * sv.v_t.unwrap());
            let free: Vec<f64> = (0..n)
                .filter_map(|i| {
                    Some(
                        (st.rots[i]? * q * tp[i].rotation.inverse())
                            .angle()
                            .to_degrees(),
                    )
                })
                .collect();
            let rot_of = |ps: &[Option<Pose>]| -> Vec<f64> {
                (0..n)
                    .filter_map(|i| {
                        Some(
                            (ps[i]?.rotation * tp[i].rotation.inverse())
                                .angle()
                                .to_degrees(),
                        )
                    })
                    .collect()
            };
            let cen_of = |ps: &[Option<Pose>]| -> Vec<f64> {
                (0..n)
                    .filter_map(|i| Some((ps[i]?.center() - tp[i].center()).norm()))
                    .collect()
            };
            let fin_c = cen_of(&init.poses);
            let dz: Vec<f64> = (0..n)
                .filter_map(|i| Some((init.poses[i]?.center().z - tp[i].center().z).abs()))
                .collect();
            let mx = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
            let origin = scene.to_first_gps_frame(&Point3::new(0.0, 0.0, 0.0)).coords;
            let surf = |p: &Vector3<f64>| {
                let q = p - origin;
                (q.z - scene.surface_height(q.x, q.y)).abs()
            };
            let good = well_conditioned(&init, PREVIEW_MIN_RAY_DEG, 200);
            let (mut align_m, mut vs_dz) = (None, None);
            if full {
                let mut rs = init.clone();
                run_ba(&mut rs, &k, 10, Some(&gps), 2.0, &[]);
                gps_align_refined(&mut rs, &gps);
                let region = split_regions(ds.positions.len(), ds.config.span, ds.config.ovl)[0];
                let win = (region.lo, region.hi);
                let (ta, tb) = (to_tracks(&good, &gids), to_tracks(&rs, &gids));
                let pairs = point_pairs(&ta, &tb, |i| (i / 3) as usize, win);
                let (sim, rec) = align_region(&region, &pairs);
                align_m = rec.fit_median_m;
                let inr = vec![true; n];
                let rc = dense_cloud(&rs, &imgs, &k, &inr, 96, DenseMethod::Sweep);
                let pc = dense_cloud(&good, &imgs, &k, &inr, 96, DenseMethod::Sweep);
                let xyz = |c: &PointCloud| -> Vec<[f64; 3]> {
                    c.points
                        .iter()
                        .map(|p| [p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64])
                        .collect()
                };
                if let Some(sm) = sim {
                    vs_dz = height_pair_median(&xyz(&apply_cloud(&sm, &pc)), &xyz(&rc), 2.0);
                }
            }
            out.push(StageRow {
                rot_free_med: med(free.clone()),
                rot_free_max: mx(&free),
                placed_c_med: med(cen_of(&st.placed)),
                placed_rot_med: med(rot_of(&st.placed)),
                final_c_med: med(fin_c.clone()),
                final_c_max: mx(&fin_c),
                final_dz_med: med(dz),
                final_rot_med: med(rot_of(&init.poses)),
                pruned: st.pruned,
                preview_align_m: align_m,
                vs_refined_dz: vs_dz,
                surf_med: med(good.points.iter().map(&surf).collect()),
            });
        }
        let _ = std::fs::remove_dir_all(&root);
        out
    }

    /// 기본 초벌 포즈 단계의 숫자 기준(합성 장면 정답 대비). 측정: 중심 중앙 1.11 m·회전 중앙 0.82°
    /// (끔: 어긋난 간선 제거·사전 1 → 3.21 m·2.54°).
    #[test]
    fn preview_default_pose_error_bounds() {
        let rows = stage_rows(
            &[
                PreviewOpts::default(),
                PreviewOpts::parse("prune=0,prior=0.2"),
            ],
            false,
        );
        let (new, old) = (&rows[0], &rows[1]);
        assert!(
            new.pruned.0 > 0 && new.pruned.0 * 10 < new.pruned.1,
            "{:?}",
            new.pruned
        );
        assert!(new.placed_rot_med < 1.2, "{}", new.placed_rot_med);
        assert!(new.placed_c_med < 1.6, "{}", new.placed_c_med);
        assert!(new.final_c_max < 5.0, "{}", new.final_c_max);
        assert!(
            new.placed_c_med < 0.6 * old.placed_c_med,
            "{} {}",
            new.placed_c_med,
            old.placed_c_med
        );
        assert!(new.placed_rot_med < 0.6 * old.placed_rot_med);
    }

    #[test]
    #[ignore]
    fn preview_candidates() {
        let specs = std::env::var("SKYLENS_CANDS").unwrap_or_else(|_| "".into());
        let full = std::env::var("SKYLENS_FULL").is_ok();
        let list: Vec<PreviewOpts> = specs.split(';').map(PreviewOpts::parse).collect();
        for (spec, r) in specs.split(';').zip(stage_rows(&list, full)) {
            eprintln!(
                "STAGE [{spec}] rotfree med {:.2} max {:.2} pruned {:?} | placed c {:.2} rot {:.2} | final c {:.2} max {:.2} dz {:.2} rot {:.2} | align {:?} vs_refined {:?} surf {:.2}",
                r.rot_free_med, r.rot_free_max, r.pruned, r.placed_c_med, r.placed_rot_med,
                r.final_c_med, r.final_c_max, r.final_dz_med, r.final_rot_med,
                r.preview_align_m, r.vs_refined_dz, r.surf_med
            );
        }
    }

    #[test]
    #[ignore]
    fn diagnose_preview() {
        let root = std::env::temp_dir().join(format!("skylens_diag_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let scene = Scene::new(SceneConfig {
            width: 320,
            height: 180,
            ..SceneConfig::default()
        });
        scene.write_dataset(&root).unwrap();
        let ds = load_dataset(
            &root,
            DatasetConfig {
                stride: 2,
                span: 48,
                ovl: 2,
                max_skip_run: 2,
            },
        )
        .unwrap();
        let n = ds.positions.len() * 3;
        let gids: Vec<usize> = (0..n).collect();
        let data: Vec<ImgData> = gids
            .iter()
            .map(|g| load(&ds.positions[g / 3].images[g % 3], 800).unwrap())
            .collect();
        let imgs: Vec<&ImgData> = data.iter().collect();
        let k = Intrinsics::from_hfov(
            data[0].rgb.width(),
            data[0].rgb.height(),
            65f64.to_radians(),
        );
        let views: Vec<(usize, usize)> = gids.iter().map(|g| (g % 3, g / 3)).collect();
        let gps: Vec<Vector3<f64>> = gids
            .iter()
            .map(|g| ds.positions[g / 3].image_enu[g % 3])
            .collect();
        let pm = match_pairs(&imgs, &views, &k);
        let init = sparse_init(
            &imgs,
            &pm,
            &gps,
            &k,
            PipelineConfig::default().position,
            &TriConfig::from_config(&PipelineConfig::default()),
            (0, 2.0),
        )
        .unwrap();
        let mut rs = init.clone();
        run_ba(&mut rs, &k, 10, Some(&gps), 2.0, &[]);
        gps_align_refined(&mut rs, &gps);
        // 정답 카메라(첫 GPS 기준 좌표).
        let truth_pose = |g: usize| -> Pose {
            let name = ds.positions[g / 3].images[g % 3]
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let v = scene.views.iter().find(|v| v.name == name).unwrap();
            let c = scene.to_first_gps_frame(&v.camera.pose.center());
            Pose::from_center(v.camera.pose.rotation, &c)
        };
        let tp: Vec<Pose> = gids.iter().map(|&g| truth_pose(g)).collect();
        let origin = scene.to_first_gps_frame(&Point3::new(0.0, 0.0, 0.0)).coords;
        let surf = |p: &Vector3<f64>| {
            let q = p - origin;
            (q.z - scene.surface_height(q.x, q.y)).abs()
        };
        // (a) 포즈 오차
        let ce = |s: &Sparse| {
            med((0..n)
                .map(|i| (s.poses[i].unwrap().center() - tp[i].center()).norm())
                .collect())
        };
        let re = |s: &Sparse| {
            med((0..n)
                .map(|i| {
                    (s.poses[i].unwrap().rotation * tp[i].rotation.inverse())
                        .angle()
                        .to_degrees()
                })
                .collect())
        };
        eprintln!(
            "DIAG a pose: init center err med {:.2} m rot err med {:.2} deg | refined center {:.2} m rot {:.2} deg",
            ce(&init), re(&init), ce(&rs), re(&rs)
        );
        // (b) 삼각측량: 같은 트랙을 정답 포즈로 삼각측량한 점과 비교, 광선 각 구간별.
        let ang = ray_angles(&init);
        let tri_true: Vec<Option<Vector3<f64>>> = init
            .obs
            .iter()
            .map(|o| {
                let cams: Vec<_> = o
                    .iter()
                    .map(|&(i, _, px)| {
                        (
                            Camera {
                                intrinsics: k,
                                pose: tp[i],
                            },
                            px,
                        )
                    })
                    .collect();
                stand_in::triangulate_track(&cams, 6.0)
            })
            .collect();
        let bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 90.0)];
        for (lo, hi) in bins {
            let ix: Vec<usize> = (0..init.points.len())
                .filter(|&p| ang[p] >= lo && ang[p] < hi)
                .collect();
            let by =
                |f: &dyn Fn(usize) -> Option<f64>| med(ix.iter().filter_map(|&p| f(p)).collect());
            eprintln!(
                "DIAG b ray angle [{lo},{hi}) n {}: surface dist init {:.2} | true-pose tri {:.2} (n ok {}) | refined {:.2} | init-vs-refined shift {:.2}",
                ix.len(),
                by(&|p| Some(surf(&init.points[p]))),
                by(&|p| tri_true[p].map(|x| surf(&x))),
                ix.iter().filter(|&&p| tri_true[p].is_some()).count(),
                by(&|p| Some(surf(&rs.points[p]))),
                by(&|p| Some((init.points[p] - rs.points[p]).norm())),
            );
        }
        // (c) 정렬 대응: 구성과 잔차.
        let good = well_conditioned(&init, PREVIEW_MIN_RAY_DEG, 200);
        let tb = to_tracks(&rs, &gids);
        let region = split_regions(ds.positions.len(), ds.config.span, ds.config.ovl)[0];
        let win = (region.lo, region.hi);
        let run_align = |s: &Sparse| {
            let ta = to_tracks(s, &gids);
            let pairs = point_pairs(&ta, &tb, |i| (i / 3) as usize, win);
            let (sim, rec) = align_region(&region, &pairs);
            (sim, rec, pairs)
        };
        let (sim_all, rec_all, _) = run_align(&init);
        let (sim_good, rec_good, _) = run_align(&good);
        eprintln!(
            "DIAG c align: all points pairs {} resid med {:?} scale {:?} | angle>=2 pairs {} resid med {:?} scale {:?} (points {} -> {})",
            rec_all.pairs, rec_all.fit_median_m, rec_all.scale,
            rec_good.pairs, rec_good.fit_median_m, rec_good.scale,
            init.points.len(), good.points.len()
        );
        // (d) 높이 차 비교: 밀집 점군(초벌 전체 / 초벌 광선 각 필터) vs 정밀, 정답 표면 대비.
        let inr = vec![true; n];
        let ref_cloud = dense_cloud(&rs, &imgs, &k, &inr, 96, DenseMethod::Sweep);
        let xyz = |c: &PointCloud| -> Vec<[f64; 3]> {
            c.points
                .iter()
                .map(|p| [p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64])
                .collect()
        };
        let rx = xyz(&ref_cloud);
        eprintln!(
            "DIAG d refined cloud {} pts, surface dist med {:.2}",
            rx.len(),
            med(rx
                .iter()
                .map(|p| surf(&Vector3::new(p[0], p[1], p[2])))
                .collect())
        );
        for (label, s, sim) in [("all", &init, &sim_all), ("angle>=2", &good, &sim_good)] {
            let c = dense_cloud(s, &imgs, &k, &inr, 96, DenseMethod::Sweep);
            let raw = xyz(&c);
            let al = sim.as_ref().map(|m| xyz(&apply_cloud(m, &c)));
            eprintln!(
                "DIAG d preview[{label}] {} pts: raw surface dist {:.2} | aligned nn {:?} dz {:?} surface dist {:?}",
                raw.len(),
                med(raw.iter().map(|p| surf(&Vector3::new(p[0], p[1], p[2]))).collect()),
                al.as_ref().and_then(|a| nn_median(a, &rx)),
                al.as_ref().and_then(|a| height_pair_median(a, &rx, 2.0)),
                al.as_ref().map(|a| med(a.iter().map(|p| surf(&Vector3::new(p[0], p[1], p[2]))).collect())),
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }
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

    fn tri() -> TriConfig {
        TriConfig::from_config(&PipelineConfig::default())
    }

    /// 기체(i % 3)별 GPS 치우침 2 m(축마다 σ) + 잡음 0.5 m. 사전항 끔/켬(σ 2·0.7)에서 정밀 중심 오차(정답 대비, 닮음 정렬 뒤)와
    /// 거르기 전 관측 대비 정밀 재투영을 잰다.
    #[test]
    fn refined_ba_with_per_vehicle_gps_bias() {
        let sc = scene(0.3, 0.0035, 0.08);
        let mut rng = Rng(9);
        let bias: Vec<Vector3<f64>> = (0..3)
            .map(|_| Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 2.0)
            .collect();
        let gps: Vec<Vector3<f64>> = sc
            .poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                p.center().coords
                    + bias[i % 3]
                    + Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.5
            })
            .collect();
        let mut res = Vec::new();
        for (tag, prior) in [
            ("off", None),
            ("sigma2", Some(2.0)),
            ("sigma0.7", Some(0.7)),
        ] {
            let (points, obs, ba_only, _) =
                triangulate_tracks(&sc.coarse, &k(), &sc.tracks, &tri());
            let mut s = Sparse {
                poses: sc.coarse.clone(),
                points,
                obs,
                ba_only,
                rms: 0.0,
            };
            s.rms = run_ba(
                &mut s,
                &k(),
                15,
                prior.map(|_| &gps[..]),
                prior.unwrap_or(2.0),
                &[],
            );
            let med = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let src: Vec<Vector3<f64>> =
                s.poses.iter().map(|p| p.unwrap().center().coords).collect();
            let tru: Vec<Vector3<f64>> = sc.poses.iter().map(|p| p.center().coords).collect();
            let mut raw: Vec<f64> = src.iter().zip(&tru).map(|(a, b)| (a - b).norm()).collect();
            let sim = crate::align::robust_similarity(&src, &tru, 3, 3.0)
                .unwrap()
                .0;
            let mut al: Vec<f64> = src
                .iter()
                .zip(&tru)
                .map(|(a, b)| (sim.apply_point(a) - b).norm())
                .collect();
            let (m_raw, m_al) = (med(&mut raw), med(&mut al));
            eprintln!(
                "bias2 {tag}: rms {:.3} center median raw {m_raw:.2} aligned {m_al:.2}",
                s.rms
            );
            res.push((m_raw, m_al, s.rms));
        }
        // 치우침 2 m 장면: 사전항을 켜면 중심이 치우침 쪽으로 끌려 끔보다 커지고(σ 가 작을수록 더), 닮음 정렬 뒤에는
        // 모두 0.1 m 안이다(편대 전체의 쏠림은 닮음 변환이 흡수). σ 2 m(기본)는 정답에서 1.5 m 안.
        assert!(
            res[0].0 < 1.0 && res[1].0 < 1.5 && res[1].0 > res[0].0,
            "{res:?}"
        );
        assert!(res[2].0 > res[1].0, "{res:?}");
        assert!(res.iter().all(|r| r.1 < 0.1), "{res:?}");
        assert!(res.iter().all(|r| r.2 < 0.7), "{res:?}");
    }

    #[test]
    fn prior_sigma_combines_horizontal_and_vertical() {
        let c = PipelineConfig {
            gps_sigma_h: 1.0,
            gps_sigma_v: 4.0,
            ..PipelineConfig::default()
        };
        assert!((c.prior_sigma() - 6f64.sqrt()).abs() < 1e-12);
        assert!((PipelineConfig::default().prior_sigma() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn refined_ba_keeps_observations_at_realistic_coarse_error() {
        let sc = scene(0.3, 0.0035, 0.08);
        let (points, obs, ba_only, st) = triangulate_tracks(&sc.coarse, &k(), &sc.tracks, &tri());
        let coarse_rms = rms_over(&sc.coarse, &points, &obs);
        eprintln!("coarse rms {coarse_rms:.2} px {st:?}");
        assert!((3.0..=6.0).contains(&coarse_rms), "{coarse_rms}");
        // 점 문턱이 고정 0.7 px 이 아니라 분포에서 정해진다.
        assert!(st.thr_px > 2.0, "{st:?}");
        assert!(st.obs_ba as f64 >= 0.9 * st.obs_all as f64, "{st:?}");
        assert!(
            (st.points + st.points_ba_only) as f64 >= 0.9 * sc.n_pts as f64,
            "{st:?}"
        );
        let mut s = Sparse {
            poses: sc.coarse.clone(),
            points,
            obs,
            ba_only,
            rms: 0.0,
        };
        let count = |s: &Sparse| {
            s.obs.iter().map(Vec::len).sum::<usize>()
                + s.ba_only.iter().map(|e| e.1.len()).sum::<usize>()
        };
        let before = count(&s);
        let mut rng = Rng(5);
        let gps: Vec<Vector3<f64>> = sc
            .poses
            .iter()
            .map(|p| p.center().coords + Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()) * 0.5)
            .collect();
        s.rms = run_ba(&mut s, &k(), 15, Some(&gps), 2.0, &[]);
        let after = count(&s);
        assert_eq!(before, after);
        // 거르지 않은 전체 관측(정답 점 기준 점별 목록)에 대한 정밀 재투영.
        let (mut pts, mut all) = (s.points.clone(), Vec::new());
        let lists = s.obs.iter().chain(s.ba_only.iter().map(|e| &e.1));
        for o in lists {
            all.push(sc.tracks[o[0].1 / 100].clone());
        }
        pts.extend(s.ba_only.iter().map(|e| e.0));
        let all_rms = rms_over(&s.poses, &pts, &all);
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
