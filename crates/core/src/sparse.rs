//! 희소 복원: 영상 묶음 → 카메라 자세 + 희소 점(SPEC §3.2, §3.4).
//!
//! 단계: 특징 → 짝 일정·매칭·기하 검증 → 트랙(합집합-찾기) → 상대 자세 → 회전 평균 →
//! 카메라 중심(방향 제약 + 광선 교대 최소제곱) → 다시점 삼각측량(DLT) → 번들 조정(선택) → GPS 정렬(선택).
//!
//! 규약: 키포인트 좌표는 연속 픽셀 좌표(화소 중심 = i + 0.5)로 돌려준다.
//! GPS 가 3개 이상 있으면 위치 단계 초입에 방향 제약으로 얻은 중심을 GPS 에 맞춰 동-북-위 좌표계·미터 축척으로
//! 옮긴다(`align_gps` 가 꺼져도). 내부 파라미터는 입력 그대로 쓰고 번들 조정에서도 고정한다.

use crate::align::{
    align_to_enu_with, robust_similarity, up_from_rotations, GpsAlignConfig, Similarity,
};
use crate::ba::{bundle_adjust, BaOptions, BaProblem, Observation};
use crate::camera::{Camera, Intrinsics, Pose};
use crate::features::{detect_and_describe, DetectorConfig, Feature, GrayImage};
use crate::matching::{
    candidate_pairs, ratio_match, RansacConfig, MIN_VERIFIED_INLIERS, PAIR_CROSS, PAIR_POW2_MAX,
    PAIR_TEMPORAL,
};
use crate::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};
use crate::two_view::{ransac_essential, refine_relative_pose};
use nalgebra::{Matrix3, Matrix4, Point3, Rotation3, Vector2, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;

/// 입력 영상 하나.
pub struct SparseInput {
    pub name: String,
    /// 카메라 번호(F=0, R=1, L=2). `intrinsics` 의 색인.
    pub group: usize,
    /// 촬영 위치 번호.
    pub position: usize,
    pub image: image::RgbImage,
    /// 동-북-위 GPS(m).
    pub gps_enu: Option<[f64; 3]>,
}

pub struct SparseConfig {
    pub max_features: usize,
    pub bundle_adjust: bool,
    /// 번들 조정에 쓰는 트랙 상한(기본 100_000).
    pub max_ba_tracks: usize,
    pub align_gps: bool,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            max_features: 8192,
            bundle_adjust: true,
            max_ba_tracks: 100_000,
            align_gps: true,
        }
    }
}

pub struct SparsePoint3 {
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    /// (영상 번호, 그 영상 키포인트 번호).
    pub obs: Vec<(usize, usize)>,
}

pub struct SparseModel {
    /// 입력 순서. 등록하지 못한 영상은 None.
    pub cameras: Vec<Option<Camera>>,
    pub points: Vec<SparsePoint3>,
    /// 영상별 키포인트(연속 픽셀 좌표 x, y). `SparsePoint3::obs` 의 특징 번호가 가리킨다.
    pub keypoints: Vec<Vec<[f64; 2]>>,
    pub reproj_rms_px: f64,
    pub registered: usize,
    /// GPS 정렬 변환과 잔차 중앙값(m). 정렬하지 않았거나 실패하면 None.
    pub gps_fit: Option<(Similarity, f64)>,
}

/// 매칭 문턱(SPEC §3.2).
const RATIO: f32 = 0.8;
/// 재투영 오차 상한(px): 삼각측량 직후·번들 조정 직후 관측 제거.
const MAX_REPROJ_PX: f64 = 4.0;
/// 삼각측량 최소 시선 각(도).
const MIN_TRI_ANGLE_DEG: f64 = 1.0;
/// GPS 대응 제외 문턱(m, SPEC §3.4).
const GPS_MAX_M: f64 = 3.0;
/// 위치 교대 최소제곱에 쓰는 트랙 수 상한.
const ALT_MAX_TRACKS: usize = 20_000;

struct PairResult {
    i: usize,
    j: usize,
    matches: Vec<(usize, usize)>,
    rot: Rotation3<f64>,
    /// 카메라 i 좌표의 단위 이동(관측 가능할 때).
    t: Option<Vector3<f64>>,
}

pub fn reconstruct(
    inputs: &[SparseInput],
    intrinsics: &[Intrinsics],
    cfg: &SparseConfig,
) -> Result<SparseModel, String> {
    let n = inputs.len();
    if n < 2 {
        return Err("영상이 2장 미만".into());
    }
    if let Some(b) = inputs.iter().find(|x| x.group >= intrinsics.len()) {
        return Err(format!(
            "{}: 그룹 {} 의 내부 파라미터 없음",
            b.name, b.group
        ));
    }
    for (k, x) in inputs.iter().enumerate() {
        let c = &intrinsics[x.group];
        if x.image.width() != c.width || x.image.height() != c.height {
            return Err(format!("영상 {k}({}) 크기가 내부 파라미터와 다름", x.name));
        }
    }

    // 1. 특징.
    let det = DetectorConfig {
        max_features: cfg.max_features,
        ..Default::default()
    };
    let feats: Vec<Vec<Feature>> = inputs
        .par_iter()
        .map(|x| {
            let g = GrayImage::from_rgb(
                x.image.width() as usize,
                x.image.height() as usize,
                x.image.as_raw(),
            );
            detect_and_describe(&g, &det)
        })
        .collect();
    let keypoints: Vec<Vec<[f64; 2]>> = feats
        .iter()
        .map(|f| {
            f.iter()
                .map(|a| [a.kp.x as f64 + 0.5, a.kp.y as f64 + 0.5])
                .collect()
        })
        .collect();
    let norm: Vec<Vec<Vector2<f64>>> = feats
        .iter()
        .zip(inputs)
        .map(|(f, x)| {
            let k = &intrinsics[x.group];
            f.iter()
                .map(|a| k.index_to_normalized(&Vector2::new(a.kp.x as f64, a.kp.y as f64)))
                .collect()
        })
        .collect();

    // 2. 짝 일정·매칭·기하 검증.
    let views: Vec<(usize, usize)> = inputs.iter().map(|x| (x.group, x.position)).collect();
    let pairs = candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);
    let results: Vec<PairResult> = pairs
        .par_iter()
        .enumerate()
        .filter_map(|(pi, &(i, j))| {
            let m = ratio_match(&feats[i], &feats[j], RATIO, true);
            if m.len() < MIN_VERIFIED_INLIERS {
                return None;
            }
            let n1: Vec<_> = m.iter().map(|&(a, _)| norm[i][a]).collect();
            let n2: Vec<_> = m.iter().map(|&(_, b)| norm[j][b]).collect();
            let focal = 0.5 * (intrinsics[inputs[i].group].fx + intrinsics[inputs[j].group].fx);
            let rc = RansacConfig {
                seed: pi as u64 + 1,
                ..Default::default()
            };
            let (e, inl) = ransac_essential(&n1, &n2, focal, &rc)?;
            let keep: Vec<usize> = (0..m.len()).filter(|&k| inl[k]).collect();
            let a: Vec<_> = keep.iter().map(|&k| n1[k]).collect();
            let b: Vec<_> = keep.iter().map(|&k| n2[k]).collect();
            let rp = refine_relative_pose(&e, &a, &b, 20)?;
            let t = rp.translation_observable.then_some(rp.translation);
            Some(PairResult {
                i,
                j,
                matches: keep.iter().map(|&k| m[k]).collect(),
                rot: rp.rotation,
                t,
            })
        })
        .collect();
    if results.is_empty() {
        return Err("검증된 영상 짝 없음".into());
    }

    // 3. 회전 평균.
    let edges: Vec<RelativeRotation> = results
        .iter()
        .map(|p| RelativeRotation {
            i: p.i,
            j: p.j,
            rotation: p.rot,
            weight: p.matches.len() as f64,
        })
        .collect();
    let avg = average_rotations(n, &edges, &AveragingConfig::default())
        .ok_or("회전 평균 실패(쓸 간선 없음)")?;
    let mut rots: Vec<Option<Rotation3<f64>>> = avg.rotations.clone();
    let good: Vec<&PairResult> = results
        .iter()
        .zip(&avg.inliers)
        .filter(|(p, &ok)| ok && rots[p.i].is_some() && rots[p.j].is_some())
        .map(|(p, _)| p)
        .collect();

    // 4. 트랙.
    let tracks = build_tracks(&feats, &good);

    // 5. 위치.
    let gps: Vec<Option<Vector3<f64>>> = inputs
        .iter()
        .map(|x| x.gps_enu.map(|g| Vector3::new(g[0], g[1], g[2])))
        .collect();
    let (mut centers, gauge) = init_centers(n, &rots, &good, &gps)?;
    if let Some(g) = gauge {
        for r in rots.iter_mut().flatten() {
            *r = *r * g.inverse();
        }
    }
    for k in 0..n {
        if centers[k].is_none() {
            rots[k] = None;
        }
    }
    let dirs: Vec<Vec<Vector3<f64>>> = (0..n)
        .map(|k| match rots[k] {
            Some(r) => norm[k]
                .iter()
                .map(|p| (r.inverse() * Vector3::new(p.x, p.y, 1.0)).normalize())
                .collect(),
            None => Vec::new(),
        })
        .collect();
    alternate(&tracks, &dirs, &mut centers, &gps);

    // 6. 다시점 삼각측량.
    let mut cams: Vec<Option<Camera>> = (0..n)
        .map(|k| match (rots[k], centers[k]) {
            (Some(r), Some(c)) => Some(Camera {
                intrinsics: intrinsics[inputs[k].group],
                pose: Pose::from_center(r, &Point3::from(c)),
            }),
            _ => None,
        })
        .collect();
    let mut pts = triangulate_tracks(&tracks, &cams, &keypoints, &norm);

    // 7. 번들 조정.
    if cfg.bundle_adjust && pts.len() >= 8 {
        run_ba(
            &mut cams,
            &mut pts,
            &keypoints,
            inputs,
            intrinsics,
            cfg.max_ba_tracks,
        );
        prune(&mut pts, &cams, &keypoints, MAX_REPROJ_PX);
    }

    // 8. GPS 정렬.
    let mut gps_fit = None;
    if cfg.align_gps {
        let idx: Vec<usize> = (0..n)
            .filter(|&k| cams[k].is_some() && gps[k].is_some())
            .collect();
        let src: Vec<Vector3<f64>> = idx
            .iter()
            .map(|&k| cams[k].unwrap().pose.center().coords)
            .collect();
        let dst: Vec<Vector3<f64>> = idx.iter().map(|&k| gps[k].unwrap()).collect();
        let rr: Vec<Rotation3<f64>> = idx
            .iter()
            .map(|&k| cams[k].unwrap().pose.rotation)
            .collect();
        let acfg = GpsAlignConfig {
            max_residual_m: GPS_MAX_M,
            up: up_from_rotations(&rr),
            ..Default::default()
        };
        if let Some(a) = align_to_enu_with(&src, &dst, &acfg) {
            apply_similarity(&a.sim, &mut cams, &mut pts);
            gps_fit = Some((a.sim, a.median_residual));
        }
    }

    let (rms, _) = reproj_stats(&pts, &cams, &keypoints);
    let registered = cams.iter().filter(|c| c.is_some()).count();
    let points = pts
        .into_iter()
        .map(|p| SparsePoint3 {
            xyz: [p.xyz.x, p.xyz.y, p.xyz.z],
            rgb: color_of(inputs, &keypoints, &p.obs),
            obs: p.obs,
        })
        .collect();
    Ok(SparseModel {
        cameras: cams,
        points,
        keypoints,
        reproj_rms_px: rms,
        registered,
        gps_fit,
    })
}

// ---------------------------------------------------------------------------------------------
// 트랙

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// 합집합-찾기로 검증된 대응을 트랙으로 묶는다. 같은 영상 특징이 둘 이상인 성분은 버린다.
/// 반환: 트랙마다 (영상, 특징) 목록(영상 번호 오름차순).
fn build_tracks(feats: &[Vec<Feature>], pairs: &[&PairResult]) -> Vec<Vec<(usize, usize)>> {
    let mut off = vec![0usize; feats.len() + 1];
    for (k, f) in feats.iter().enumerate() {
        off[k + 1] = off[k] + f.len();
    }
    let total = off[feats.len()];
    let mut parent: Vec<usize> = (0..total).collect();
    let mut touched = vec![false; total];
    for p in pairs {
        for &(a, b) in &p.matches {
            let (x, y) = (off[p.i] + a, off[p.j] + b);
            touched[x] = true;
            touched[y] = true;
            let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
            if rx != ry {
                parent[ry.max(rx)] = ry.min(rx);
            }
        }
    }
    let img_of = |node: usize| off.partition_point(|&o| o <= node) - 1;
    let mut comps: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for node in 0..total {
        if touched[node] {
            let r = find(&mut parent, node);
            let im = img_of(node);
            comps.entry(r).or_default().push((im, node - off[im]));
        }
    }
    let mut out: Vec<Vec<(usize, usize)>> = comps
        .into_values()
        .filter_map(|mut v| {
            v.sort_unstable();
            (v.len() >= 2 && v.windows(2).all(|w| w[0].0 != w[1].0)).then_some(v)
        })
        .collect();
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------------------------
// 위치

/// 방향 제약(C_j − C_i ∥ −R_jᵀt)으로 중심 초깃값을 만든다. GPS 가 3개 이상이면 그에 맞춰 회전·중심을 옮긴다.
/// 이동 방향이 이어지지 않은 영상은 None.
fn init_centers(
    n: usize,
    rots: &[Option<Rotation3<f64>>],
    pairs: &[&PairResult],
    gps: &[Option<Vector3<f64>>],
) -> Result<(Vec<Option<Vector3<f64>>>, Option<Rotation3<f64>>), String> {
    // (i, j, 세계 좌표 단위 방향 C_j − C_i)
    let mut edges: Vec<(usize, usize, Vector3<f64>)> = Vec::new();
    for p in pairs {
        if let (Some(t), Some(rj)) = (p.t, rots[p.j]) {
            let d = -(rj.inverse() * t);
            if d.norm() > 1e-9 {
                edges.push((p.i, p.j, d.normalize()));
            }
        }
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (e, &(i, j, _)) in edges.iter().enumerate() {
        adj[i].push(e);
        adj[j].push(e);
    }
    // 가장 큰 연결 성분을 폭 우선으로 훑는다.
    let mut best: Vec<usize> = Vec::new();
    let mut seen = vec![false; n];
    for s in 0..n {
        if seen[s] || rots[s].is_none() {
            continue;
        }
        let mut comp = vec![s];
        seen[s] = true;
        let mut h = 0;
        while h < comp.len() {
            let u = comp[h];
            h += 1;
            for &e in &adj[u] {
                let v = if edges[e].0 == u {
                    edges[e].1
                } else {
                    edges[e].0
                };
                if !seen[v] && rots[v].is_some() {
                    seen[v] = true;
                    comp.push(v);
                }
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    if best.len() < 2 {
        return Err("이동 방향이 이어진 영상 2장 미만".into());
    }
    let mut c: Vec<Option<Vector3<f64>>> = vec![None; n];
    c[best[0]] = Some(Vector3::zeros());
    for &u in &best {
        // 폭 우선 순서라 u 는 이미 위치가 있다.
        let cu = c[u].unwrap();
        for &e in &adj[u] {
            let (i, j, d) = edges[e];
            if i == u && c[j].is_none() && best.contains(&j) {
                c[j] = Some(cu + d);
            } else if j == u && c[i].is_none() && best.contains(&i) {
                c[i] = Some(cu - d);
            }
        }
    }
    // 모든 간선을 만족하도록 가우스–자이델(길이는 방향 쪽 투영으로 갱신).
    for _ in 0..60 {
        for &u in &best {
            if u == best[0] {
                continue;
            }
            let (mut acc, mut cnt) = (Vector3::zeros(), 0.0);
            for &e in &adj[u] {
                let (i, j, d) = edges[e];
                let (Some(ci), Some(cj)) = (c[i], c[j]) else {
                    continue;
                };
                let l = d.dot(&(cj - ci)).max(1e-3);
                if j == u {
                    acc += ci + d * l;
                } else {
                    acc += cj - d * l;
                }
                cnt += 1.0;
            }
            if cnt > 0.0 {
                c[u] = Some(acc / cnt);
            }
        }
    }
    for k in 0..n {
        if !best.contains(&k) {
            c[k] = None;
        }
    }
    // GPS 로 좌표계·축척을 잡는다.
    let idx: Vec<usize> = best.iter().copied().filter(|&k| gps[k].is_some()).collect();
    if idx.len() >= 3 {
        let src: Vec<_> = idx.iter().map(|&k| c[k].unwrap()).collect();
        let dst: Vec<_> = idx.iter().map(|&k| gps[k].unwrap()).collect();
        if let Some((sim, _, _)) = robust_similarity(&src, &dst, 5, GPS_MAX_M) {
            let moved = c
                .into_iter()
                .map(|x| x.map(|v| sim.apply_point(&v)))
                .collect();
            return Ok((moved, Some(sim.r)));
        }
    }
    Ok((c, None))
}

/// 광선 교대 최소제곱: 회전 고정, 점 ← 광선 교점(최소제곱), 중심 ← 점을 지나는 광선 위 최근접.
/// GPS 가 있는 카메라는 중심에 약한 사전항(가중 1 관측분)을 둔다. 없으면 중심 퍼짐을 유지해 축척 붕괴를 막는다.
fn alternate(
    tracks: &[Vec<(usize, usize)>],
    dirs: &[Vec<Vector3<f64>>],
    centers: &mut [Option<Vector3<f64>>],
    gps: &[Option<Vector3<f64>>],
) {
    let n = centers.len();
    let mut sel: Vec<&Vec<(usize, usize)>> = tracks
        .iter()
        .filter(|t| t.iter().filter(|o| centers[o.0].is_some()).count() >= 2)
        .collect();
    sel.sort_by_key(|t| std::cmp::Reverse(t.len()));
    sel.truncate(ALT_MAX_TRACKS);
    let use_gps = gps
        .iter()
        .zip(centers.iter())
        .filter(|(g, c)| g.is_some() && c.is_some())
        .count()
        >= 3;
    let spread = |c: &[Option<Vector3<f64>>]| -> (Vector3<f64>, f64) {
        let v: Vec<Vector3<f64>> = c.iter().flatten().copied().collect();
        let m = v.iter().sum::<Vector3<f64>>() / v.len() as f64;
        let s = (v.iter().map(|x| (x - m).norm_squared()).sum::<f64>() / v.len() as f64).sqrt();
        (m, s)
    };
    let s0 = spread(centers).1.max(1e-9);
    let proj = |d: &Vector3<f64>| Matrix3::identity() - d * d.transpose();
    for _ in 0..12 {
        let mut acc: Vec<(Matrix3<f64>, Vector3<f64>)> =
            vec![(Matrix3::zeros(), Vector3::zeros()); n];
        for t in &sel {
            let (mut a, mut b) = (Matrix3::zeros(), Vector3::zeros());
            let mut cnt = 0;
            for &(im, f) in t.iter() {
                if let Some(c) = centers[im] {
                    let p = proj(&dirs[im][f]);
                    a += p;
                    b += p * c;
                    cnt += 1;
                }
            }
            if cnt < 2 || a.determinant() < 1e-4 {
                continue;
            }
            let Some(x) = a.try_inverse().map(|ai| ai * b) else {
                continue;
            };
            for &(im, f) in t.iter() {
                if centers[im].is_some() {
                    let p = proj(&dirs[im][f]);
                    acc[im].0 += p;
                    acc[im].1 += p * x;
                }
            }
        }
        for k in 0..n {
            if centers[k].is_none() {
                continue;
            }
            let (mut a, mut b) = acc[k];
            if use_gps {
                if let Some(g) = gps[k] {
                    a += Matrix3::identity();
                    b += g;
                }
            }
            if a.determinant() > 1e-4 {
                if let Some(ai) = a.try_inverse() {
                    centers[k] = Some(ai * b);
                }
            }
        }
        if !use_gps {
            let (m, s) = spread(centers);
            let f = s0 / s.max(1e-12);
            for c in centers.iter_mut().flatten() {
                *c = m + (*c - m) * f;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 삼각측량·재투영

struct TriPoint {
    xyz: Vector3<f64>,
    obs: Vec<(usize, usize)>,
}

fn reproj_err(cam: &Camera, x: &Vector3<f64>, px: &[f64; 2]) -> Option<f64> {
    let p = cam.project(&Point3::from(*x))?;
    Some(((p.x - px[0]).powi(2) + (p.y - px[1]).powi(2)).sqrt())
}

/// 한 트랙의 DLT 삼각측량(카메라 중심 평균을 원점으로 옮겨 조건을 개선).
fn dlt(
    obs: &[(usize, usize)],
    cams: &[Option<Camera>],
    norm: &[Vec<Vector2<f64>>],
) -> Option<Vector3<f64>> {
    let c0 = obs
        .iter()
        .map(|o| cams[o.0].unwrap().pose.center().coords)
        .sum::<Vector3<f64>>()
        / obs.len() as f64;
    let mut ata = Matrix4::<f64>::zeros();
    for &(im, f) in obs {
        let pose = &cams[im].unwrap().pose;
        let (r, t) = (
            pose.rotation.matrix(),
            pose.translation + pose.rotation * c0,
        );
        let q = norm[im][f];
        let row = |row: usize, v: f64| {
            let mut a = nalgebra::RowVector4::zeros();
            for j in 0..3 {
                a[j] = v * r[(2, j)] - r[(row, j)];
            }
            a[3] = v * t[2] - t[row];
            a
        };
        for a in [row(0, q.x), row(1, q.y)] {
            ata += a.transpose() * a;
        }
    }
    let e = ata.symmetric_eigen();
    let h = e.eigenvectors.column(e.eigenvalues.imin());
    (h[3].abs() > 1e-12).then(|| Vector3::new(h[0], h[1], h[2]) / h[3] + c0)
}

fn max_angle_deg(x: &Vector3<f64>, obs: &[(usize, usize)], cams: &[Option<Camera>]) -> f64 {
    let rays: Vec<Vector3<f64>> = obs
        .iter()
        .map(|o| (x - cams[o.0].unwrap().pose.center().coords).normalize())
        .collect();
    let mut best = 0.0f64;
    for a in 0..rays.len() {
        for b in a + 1..rays.len() {
            best = best.max(rays[a].cross(&rays[b]).norm().atan2(rays[a].dot(&rays[b])));
        }
    }
    best.to_degrees()
}

fn triangulate_tracks(
    tracks: &[Vec<(usize, usize)>],
    cams: &[Option<Camera>],
    kp: &[Vec<[f64; 2]>],
    norm: &[Vec<Vector2<f64>>],
) -> Vec<TriPoint> {
    tracks
        .par_iter()
        .filter_map(|t| {
            let mut obs: Vec<(usize, usize)> =
                t.iter().copied().filter(|o| cams[o.0].is_some()).collect();
            let mut x = None;
            for _ in 0..3 {
                if obs.len() < 2 {
                    return None;
                }
                let p = dlt(&obs, cams, norm)?;
                let keep: Vec<(usize, usize)> = obs
                    .iter()
                    .copied()
                    .filter(|&(im, f)| {
                        let c = cams[im].as_ref().unwrap();
                        c.pose.transform(&Point3::from(p)).z > 0.0
                            && reproj_err(c, &p, &kp[im][f]).is_some_and(|e| e <= MAX_REPROJ_PX)
                    })
                    .collect();
                x = Some(p);
                if keep.len() == obs.len() {
                    break;
                }
                obs = keep;
                x = None;
            }
            let x = x?;
            (max_angle_deg(&x, &obs, cams) >= MIN_TRI_ANGLE_DEG).then_some(TriPoint { xyz: x, obs })
        })
        .collect()
}

/// 재투영 오차가 `th` 를 넘는 관측(과 카메라 뒤 관측)을 떼고, 관측 2개 미만 점을 버린다.
fn prune(pts: &mut Vec<TriPoint>, cams: &[Option<Camera>], kp: &[Vec<[f64; 2]>], th: f64) {
    for p in pts.iter_mut() {
        let x = p.xyz;
        p.obs.retain(|&(im, f)| {
            cams[im].as_ref().is_some_and(|c| {
                c.pose.transform(&Point3::from(x)).z > 0.0
                    && reproj_err(c, &x, &kp[im][f]).is_some_and(|e| e <= th)
            })
        });
    }
    pts.retain(|p| p.obs.len() >= 2 && p.xyz.iter().all(|v| v.is_finite()));
}

fn reproj_stats(pts: &[TriPoint], cams: &[Option<Camera>], kp: &[Vec<[f64; 2]>]) -> (f64, usize) {
    let (mut s, mut n) = (0.0, 0usize);
    for p in pts {
        for &(im, f) in &p.obs {
            if let Some(e) = cams[im]
                .as_ref()
                .and_then(|c| reproj_err(c, &p.xyz, &kp[im][f]))
            {
                s += e * e;
                n += 1;
            }
        }
    }
    ((s / n.max(1) as f64).sqrt(), n)
}

fn color_of(inputs: &[SparseInput], kp: &[Vec<[f64; 2]>], obs: &[(usize, usize)]) -> [u8; 3] {
    let mut acc = [0u32; 3];
    let mut n = 0u32;
    for &(im, f) in obs.iter().take(8) {
        let img = &inputs[im].image;
        let (x, y) = (kp[im][f][0] as i64, kp[im][f][1] as i64);
        if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                acc[c] += p[c] as u32;
            }
            n += 1;
        }
    }
    if n == 0 {
        return [128; 3];
    }
    [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]
}

// ---------------------------------------------------------------------------------------------
// 번들 조정·정렬

fn run_ba(
    cams: &mut [Option<Camera>],
    pts: &mut [TriPoint],
    kp: &[Vec<[f64; 2]>],
    inputs: &[SparseInput],
    intrinsics: &[Intrinsics],
    max_tracks: usize,
) {
    let reg: Vec<usize> = (0..cams.len()).filter(|&k| cams[k].is_some()).collect();
    let mut map = vec![usize::MAX; cams.len()];
    for (b, &k) in reg.iter().enumerate() {
        map[k] = b;
    }
    let mut prob = BaProblem {
        groups: intrinsics.iter().map(|k| k.to_distorted()).collect(),
        poses: reg.iter().map(|&k| cams[k].unwrap().pose).collect(),
        camera_group: reg.iter().map(|&k| inputs[k].group).collect(),
        points: pts.iter().map(|p| Point3::from(p.xyz)).collect(),
        observations: Vec::new(),
    };
    for (pi, p) in pts.iter().enumerate() {
        for &(im, f) in &p.obs {
            if map[im] != usize::MAX {
                prob.observations.push(Observation {
                    camera: map[im],
                    point: pi,
                    pixel: Vector2::new(kp[im][f][0], kp[im][f][1]),
                });
            }
        }
    }
    let opts = BaOptions {
        max_iterations: 30,
        max_tracks,
        default_free_intrinsics: [false; 8],
        fixed_cameras: vec![0],
        ..Default::default()
    };
    bundle_adjust(&mut prob, &opts);
    if prob
        .poses
        .iter()
        .all(|p| p.translation.iter().all(|v| v.is_finite()))
    {
        for (b, &k) in reg.iter().enumerate() {
            cams[k].as_mut().unwrap().pose = prob.poses[b];
        }
        for (p, x) in pts.iter_mut().zip(&prob.points) {
            if x.coords.iter().all(|v| v.is_finite()) {
                p.xyz = x.coords;
            }
        }
    }
}

fn apply_similarity(sim: &Similarity, cams: &mut [Option<Camera>], pts: &mut [TriPoint]) {
    for c in cams.iter_mut().flatten() {
        let ctr = sim.apply_point(&c.pose.center().coords);
        c.pose = Pose::from_center(c.pose.rotation * sim.r.inverse(), &Point3::from(ctr));
    }
    for p in pts.iter_mut() {
        p.xyz = sim.apply_point(&p.xyz);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{CamId, Scene, SceneConfig};

    struct Measured {
        registered: usize,
        total: usize,
        rms: f64,
        c_med: f64,
        c_max: f64,
        p_med: f64,
        points: usize,
        secs: f64,
    }

    fn run(positions: usize, w: u32, h: u32, maxf: usize, precise: bool) -> Measured {
        let scene = Scene::new(SceneConfig {
            positions,
            width: w,
            height: h,
            ..SceneConfig::default()
        });
        let k = scene.views[0].camera.intrinsics;
        let inputs: Vec<SparseInput> = scene
            .views
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let (img, _) = scene.render(v);
                let g = scene.gps_enu[i];
                SparseInput {
                    name: v.name.clone(),
                    group: v.cam as usize,
                    position: v.position,
                    image: image::RgbImage::from_raw(img.width, img.height, img.data).unwrap(),
                    gps_enu: Some([g.x, g.y, g.z]),
                }
            })
            .collect();
        let _ = CamId::F;
        let t0 = std::time::Instant::now();
        let m = reconstruct(
            &inputs,
            &[k, k, k],
            &SparseConfig {
                max_features: maxf,
                bundle_adjust: precise,
                max_ba_tracks: 100_000,
                align_gps: precise,
            },
        )
        .unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let mut ce: Vec<f64> = m
            .cameras
            .iter()
            .zip(&scene.views)
            .filter_map(|(c, v)| c.map(|c| (c.pose.center() - v.camera.pose.center()).norm()))
            .collect();
        ce.sort_by(f64::total_cmp);
        let mut pe: Vec<f64> = Vec::new();
        for p in &m.points {
            let (im, _) = p.obs[0];
            let o = scene.views[im].camera.pose.center();
            let x = Point3::new(p.xyz[0], p.xyz[1], p.xyz[2]);
            let d = (x - o).normalize();
            if let Some(hit) = scene.intersect(&o, &d) {
                pe.push((hit.point - x).norm().min(1e3));
            }
        }
        pe.sort_by(f64::total_cmp);
        Measured {
            registered: m.registered,
            total: inputs.len(),
            rms: m.reproj_rms_px,
            c_med: ce.get(ce.len() / 2).copied().unwrap_or(f64::NAN),
            c_max: ce.last().copied().unwrap_or(f64::NAN),
            p_med: pe.get(pe.len() / 2).copied().unwrap_or(f64::NAN),
            points: m.points.len(),
            secs,
        }
    }

    fn show(tag: &str, m: &Measured) {
        eprintln!(
            "[{tag}] 등록 {}/{} 재투영 {:.3}px 중심오차 중앙 {:.3} 최대 {:.3} m 점오차 중앙 {:.3} m 점 {} 시간 {:.1}s",
            m.registered, m.total, m.rms, m.c_med, m.c_max, m.p_med, m.points, m.secs
        );
    }

    #[test]
    fn precise_small_scene() {
        let m = run(6, 480, 270, 1500, true);
        show("정밀 6위치", &m);
        assert!(m.registered >= m.total - 1, "등록 {}", m.registered);
        assert!(m.points > 100);
        assert!(m.rms < 1.5, "rms {}", m.rms);
        assert!(m.c_med < 3.0 && m.c_max < 8.0);
        assert!(m.p_med < 3.0);
    }

    #[test]
    fn rough_small_scene() {
        let m = run(6, 480, 270, 1500, false);
        show("초벌 6위치", &m);
        assert!(m.registered >= m.total - 1);
        assert!(m.points > 100);
        assert!(m.rms < 3.0, "rms {}", m.rms);
    }

    #[test]
    #[ignore]
    fn precise_larger_scene() {
        let m = run(10, 640, 360, 3000, true);
        show("정밀 10위치", &m);
        assert!(m.registered >= m.total - 1);
        assert!(m.rms < 1.5);
        assert!(m.c_med < 3.0 && m.c_max < 8.0);
        assert!(m.p_med < 3.0);
    }
}
