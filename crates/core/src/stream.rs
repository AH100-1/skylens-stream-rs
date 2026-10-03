//! 점진적 점군 스트림: 구역 분할, 초벌 모델의 3D 점 대응 정렬, 단계별 스냅샷.
//!
//! - 구역 `i` 는 위치 `[start-OVL, start+SPAN+OVL)` (start = i·SPAN), 전체 위치 범위로 자른다.
//!   앞 구역이 이미 끝(n)까지 덮어 새 위치가 없는 꼬리 start(start+OVL ≥ n)는 앞 구역에 합친다.
//! - 초벌 정렬은 같은 이미지의 같은 특징점 번호를 관측한 초벌·정밀 3D 점 짝으로
//!   [`robust_fit`] 을 돌린다: 3점 표본 첫 추정(잔차 분위 최소) 뒤
//!   [`align::robust_similarity`](crate::align::robust_similarity) 반복 트리밍
//!   (임계 = max(3 × 잔차 중앙값, 바닥값)). 점에는 변환, 법선에는 회전만.
//!   짝이 모자라거나 퇴화하면 그 구역은 정렬 실패(`None`)로 남긴다.
//! - 스냅샷 `step_k` (k = 1..=n) = 정밀 구역 `0..=k-2` + 초벌 구역 `k-1`.
//!   초벌 점 중 정밀 구역 `0..=k-2` 의 **추출 전** 점 전부에서 반경 안에 있는 점은 뺀다(잔상 방지).
//!   초벌 정렬이 실패한 구역은 초벌 대신 그 구역의 정밀 점군으로 채운다. 최종 = 정밀 전부.
//!   모든 구성 점군은 간격 추출한다. 단계는 하나씩 만들어 바로 쓰고 버린다.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::io;
use std::path::Path;

use nalgebra::Vector3;
use rayon::prelude::*;

pub use crate::align::Similarity;
use crate::align::{robust_similarity, umeyama};
use crate::ply::{write_ply_file, PointCloud, PointRecord};

/// 기본 구역 크기(위치 수).
pub const DEFAULT_SPAN: usize = 12;
/// 기본 겹침(위치 수).
pub const DEFAULT_OVL: usize = 2;
/// 반복 트리밍 횟수.
pub const TRIM_ITERS: usize = 5;
/// 트리밍 임계 바닥값(m).
pub const TRIM_FLOOR_M: f64 = 0.3;
/// 잔상 제거 반경(m).
pub const GHOST_RADIUS_M: f64 = 1.5;
/// 간격 추출 비율(n 개 중 1 개).
pub const DECIMATE_EVERY: usize = 6;
/// SPEC §4 초벌 정렬 기준: 정상 짝 잔차 중앙 상한(m).
pub const FIT_MEDIAN_LIMIT_M: f64 = 6.0;
/// SPEC §4: 구역 정렬 점쌍 하한.
pub const ALIGN_MIN_PAIRS: usize = 1000;
/// SPEC §4: 구역 간 스케일 차 max(s)/min(s) − 1 상한.
pub const ALIGN_SCALE_TOL: f64 = 0.10;

/// 구역 하나: 번호, 기준 시작 위치, 포함 위치 범위 `[lo, hi)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub index: usize,
    pub start: usize,
    pub lo: usize,
    pub hi: usize,
}

impl Region {
    /// 위치가 이 구역에 들어가는가.
    pub fn contains(&self, pos: usize) -> bool {
        pos >= self.lo && pos < self.hi
    }
}

/// 위치 `n_positions` 곳을 구역으로 나눈다. start = 0, SPAN, 2·SPAN, …
///
/// start ≥ 1 인 구역은 start + OVL < n 일 때만 만든다. 그렇지 않은 꼬리(새 위치 수 ≤ OVL)는
/// 앞 구역의 hi 가 이미 n 이므로 앞 구역에 합쳐진다. 결과: 구역 i ≥ 1 은 앞 구역 끝 너머 위치를
/// 1개 이상 가지고, 합집합은 `0..n` 이다. 80/12/2 → 7구역, 26/12/2 → (0,14),(10,26).
pub fn split_regions(n_positions: usize, span: usize, ovl: usize) -> Vec<Region> {
    assert!(span > 0, "SPAN 은 1 이상");
    (0..n_positions)
        .step_by(span)
        .filter(|&start| start == 0 || start + ovl < n_positions)
        .enumerate()
        .map(|(index, start)| Region {
            index,
            start,
            lo: start.saturating_sub(ovl),
            hi: (start + span + ovl).min(n_positions),
        })
        .collect()
}

/// 초벌 정렬에 쓰는 이미지의 위치 범위 `[lo, hi)`.
/// 구역 0 은 자기 구역 전체, 그 외는 `[start-OVL, start+OVL)`.
pub fn align_window(region: &Region, ovl: usize, n_positions: usize) -> (usize, usize) {
    if region.index == 0 {
        (region.lo, region.hi)
    } else {
        (
            region.start.saturating_sub(ovl),
            (region.start + ovl).min(n_positions),
        )
    }
}

/// 3D 점과 그 점을 만든 관측들(이미지 번호, 특징점 번호).
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub xyz: Vector3<f64>,
    pub obs: Vec<(u32, u32)>,
}

/// 초벌·정밀 3D 점 짝 `(초벌, 정밀)` 을 만든다.
/// 같은 이미지의 같은 특징점 번호를 관측한 점끼리 짝이며, 이미지 위치가 `window` 안인 관측만 쓴다.
/// 같은 (초벌, 정밀) 점 짝은 한 번만 센다.
pub fn point_pairs(
    prelim: &[Track],
    refined: &[Track],
    image_pos: impl Fn(u32) -> usize,
    window: (usize, usize),
) -> Vec<(Vector3<f64>, Vector3<f64>)> {
    let in_win = |img: u32| {
        let p = image_pos(img);
        p >= window.0 && p < window.1
    };
    let mut by_obs: HashMap<(u32, u32), usize> = HashMap::new();
    for (i, t) in refined.iter().enumerate() {
        for &o in &t.obs {
            if in_win(o.0) {
                by_obs.entry(o).or_insert(i);
            }
        }
    }
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut out = Vec::new();
    for (i, t) in prelim.iter().enumerate() {
        for &o in &t.obs {
            if !in_win(o.0) {
                continue;
            }
            if let Some(&j) = by_obs.get(&o) {
                if seen.insert((i, j)) {
                    out.push((t.xyz, refined[j].xyz));
                }
            }
        }
    }
    out
}

/// 점군에 닮음 변환 적용: 점에는 변환 전체, 법선에는 회전만.
pub fn apply_cloud(sim: &Similarity, cloud: &PointCloud) -> PointCloud {
    let points = cloud
        .points
        .iter()
        .map(|p| {
            let x = Vector3::new(p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64);
            let n = Vector3::new(p.normal[0] as f64, p.normal[1] as f64, p.normal[2] as f64);
            let y = sim.apply_point(&x);
            let m = sim.apply_normal(&n);
            PointRecord {
                xyz: [y.x as f32, y.y as f32, y.z as f32],
                normal: [m.x as f32, m.y as f32, m.z as f32],
                rgb: p.rgb,
            }
        })
        .collect();
    PointCloud { points }
}

/// 첫 추정 표본 수(3점 최소 표본).
pub const SEED_SAMPLES: usize = 400;
/// 첫 추정 점수로 쓰는 잔차 분위(최소 분위 제곱 계열). 정상 짝이 이 비율보다 많으면 버틴다.
pub const SEED_QUANTILE: f64 = 0.4;

fn residual_quantile(
    sim: &Similarity,
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    q: f64,
    buf: &mut Vec<f64>,
) -> f64 {
    buf.clear();
    buf.extend(
        src.iter()
            .zip(dst)
            .map(|(a, b)| (sim.apply_point(a) - b).norm()),
    );
    let k = ((buf.len() as f64 * q) as usize).min(buf.len() - 1);
    let (_, v, _) = buf.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *v
}

/// 강건 첫 추정 + 반복 트리밍.
///
/// 1. 고정 씨앗 3점 표본 `SEED_SAMPLES` 개로 닮음 변환을 세우고, 잔차 `SEED_QUANTILE` 분위가 가장
///    작은 것을 첫 추정으로 고른다(Rousseeuw 1984 최소 중앙 제곱의 분위 판).
/// 2. 그 분위 값 q 에 대해 잔차 ≤ max(3q, 바닥) 인 짝만 골라 [`robust_similarity`] 로 다듬는다.
///
/// 오대응이 없으면 결과는 전체 짝 트리밍과 같은 수준이다. 퇴화(일직선·NaN·짝 < 3)는 `None`.
pub fn robust_fit(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
) -> Option<(Similarity, Vec<bool>, f64)> {
    let n = src.len();
    if n < 3 || n != dst.len() {
        return None;
    }
    // 퇴화 판정은 전체 짝 추정에 맡긴다(일직선·NaN → None).
    let full = umeyama(src, dst)?;
    let mut buf = Vec::with_capacity(n);
    let mut best = (
        residual_quantile(&full, src, dst, SEED_QUANTILE, &mut buf),
        full,
    );
    let mut state: u64 = 0x2545_f491_4f6c_dd1d ^ n as u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    for _ in 0..SEED_SAMPLES {
        let (i, j, k) = (next(), next(), next());
        if i == j || j == k || i == k {
            continue;
        }
        let Some(sim) = umeyama(&[src[i], src[j], src[k]], &[dst[i], dst[j], dst[k]]) else {
            continue;
        };
        if !(sim.s.is_finite() && sim.s > 0.0) {
            continue;
        }
        let q = residual_quantile(&sim, src, dst, SEED_QUANTILE, &mut buf);
        if q < best.0 {
            best = (q, sim);
        }
    }
    let thr = (3.0 * best.0).max(TRIM_FLOOR_M);
    let keep: Vec<usize> = (0..n)
        .filter(|&i| (best.1.apply_point(&src[i]) - dst[i]).norm() <= thr)
        .collect();
    let s2: Vec<_> = keep.iter().map(|&i| src[i]).collect();
    let d2: Vec<_> = keep.iter().map(|&i| dst[i]).collect();
    let (sim, sub_inl, med) = robust_similarity(&s2, &d2, TRIM_ITERS, TRIM_FLOOR_M)?;
    let mut inl = vec![false; n];
    for (&i, &k) in keep.iter().zip(&sub_inl) {
        inl[i] = k;
    }
    Some((sim, inl, med))
}

/// 구역 하나의 정렬 기록. 정렬 실패 구역은 `fit_median_m`·`scale` 이 `None`(JSON null).
#[derive(Clone, Debug, PartialEq)]
pub struct AlignRecord {
    pub region: usize,
    pub pairs: usize,
    pub fit_median_m: Option<f64>,
    pub scale: Option<f64>,
}

/// 초벌 구역 하나를 정밀 좌표로 정렬한다. 짝이 모자라거나 퇴화(일직선·NaN 등)하면
/// 변환은 `None` 이고 기록에는 짝 수만 남는다.
pub fn align_region(
    region: &Region,
    pairs: &[(Vector3<f64>, Vector3<f64>)],
) -> (Option<Similarity>, AlignRecord) {
    let (src, dst): (Vec<_>, Vec<_>) = pairs.iter().copied().unzip();
    let fit = robust_fit(&src, &dst);
    let rec = AlignRecord {
        region: region.index,
        pairs: pairs.len(),
        fit_median_m: fit.as_ref().map(|f| f.2),
        scale: fit.as_ref().map(|f| f.0.s),
    };
    (fit.map(|f| f.0), rec)
}

/// 구역 안 위치 구간별 보정장: 전체 닮음 변환 뒤에 남는 잔차 벡터를 수평 칸별 평균으로 모아 두고,
/// 질의 점에서 가우스 가중 평균해 더한다. 초벌 모델의 완만한 휨(SPEC §3.7)을 흡수한다.
#[derive(Clone, Debug)]
pub struct WarpField {
    sim: Similarity,
    /// (칸 중심 수평 좌표, 평균 잔차 벡터, 짝 수)
    cells: Vec<([f64; 2], Vector3<f64>, f64)>,
    bw: f64,
}

/// 보정장 칸 수 상한 방향의 기본 칸 수(긴 변 기준).
const WARP_GRID: f64 = 12.0;

impl WarpField {
    /// 전체 닮음 변환 `sim` 과 3D 점 짝에서 만든다. 잔차가 `max(3·med, TRIM_FLOOR_M)` 를 넘는 짝은 뺀다.
    /// 쓸 짝이 `min_pairs` 미만이면 `None`(전체 닮음만 쓴다).
    pub fn fit(
        sim: &Similarity,
        pairs: &[(Vector3<f64>, Vector3<f64>)],
        med: f64,
        min_pairs: usize,
    ) -> Option<WarpField> {
        let thr = (3.0 * med).max(TRIM_FLOOR_M);
        let good: Vec<(Vector3<f64>, Vector3<f64>)> = pairs
            .iter()
            .map(|(a, b)| (sim.apply_point(a), *b))
            .filter(|(x, b)| (b - x).norm() <= thr)
            .collect();
        if good.len() < min_pairs {
            return None;
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for (x, _) in &good {
            for d in 0..2 {
                lo[d] = lo[d].min(x[d]);
                hi[d] = hi[d].max(x[d]);
            }
        }
        let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        if !(ext.is_finite() && ext > 1e-6) {
            return None;
        }
        let cs = ext / WARP_GRID;
        let mut acc: HashMap<(i64, i64), (f64, f64, f64, Vector3<f64>)> = HashMap::new();
        for (x, b) in &good {
            let key = (
                ((x.x - lo[0]) / cs).floor() as i64,
                ((x.y - lo[1]) / cs).floor() as i64,
            );
            let e = acc.entry(key).or_insert((0.0, 0.0, 0.0, Vector3::zeros()));
            e.0 += x.x;
            e.1 += x.y;
            e.2 += 1.0;
            e.3 += b - x;
        }
        let mut cells: Vec<_> = acc
            .into_values()
            .map(|(sx, sy, n, e)| ([sx / n, sy / n], e / n, n))
            .collect();
        cells.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]).then(a.0[1].total_cmp(&b.0[1])));
        Some(WarpField {
            sim: *sim,
            cells,
            bw: 0.7 * cs,
        })
    }

    /// 초벌 점 하나를 정밀 좌표로: 전체 닮음 + 보정장.
    pub fn apply_point(&self, p: &Vector3<f64>) -> Vector3<f64> {
        let x = self.sim.apply_point(p);
        let (mut w, mut e) = (0.0, Vector3::zeros());
        let mut near = (f64::MAX, Vector3::zeros());
        for (c, r, n) in &self.cells {
            let d2 = (c[0] - x.x).powi(2) + (c[1] - x.y).powi(2);
            if d2 < near.0 {
                near = (d2, *r);
            }
            let k = n.sqrt() * (-d2 / (2.0 * self.bw * self.bw)).exp();
            w += k;
            e += k * r;
        }
        if w > 1e-12 {
            x + e / w
        } else {
            x + near.1
        }
    }

    /// 점군에 적용(법선은 전체 닮음의 회전만).
    pub fn apply_cloud(&self, cloud: &PointCloud) -> PointCloud {
        let points = cloud
            .points
            .par_iter()
            .map(|p| {
                let x = Vector3::new(p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64);
                let n = Vector3::new(p.normal[0] as f64, p.normal[1] as f64, p.normal[2] as f64);
                let y = self.apply_point(&x);
                let m = self.sim.apply_normal(&n);
                PointRecord {
                    xyz: [y.x as f32, y.y as f32, y.z as f32],
                    normal: [m.x as f32, m.y as f32, m.z as f32],
                    rgb: p.rgb,
                }
            })
            .collect();
        PointCloud { points }
    }
}

/// 보정장이 있는 구역은 보정장으로, 없으면 전체 닮음으로 적용한다.
pub fn apply_alignments_warped(
    prelim: &[PointCloud],
    sims: &[Option<Similarity>],
    warps: &[Option<WarpField>],
) -> Vec<Option<PointCloud>> {
    assert_eq!(prelim.len(), warps.len(), "구역 수 불일치");
    let base = apply_alignments(prelim, sims);
    base.into_iter()
        .zip(prelim)
        .zip(warps)
        .map(|((b, c), w)| match (b, w) {
            (Some(_), Some(w)) => Some(w.apply_cloud(c)),
            (b, _) => b,
        })
        .collect()
}

/// 구역별 변환(`None` = 정렬 실패)을 초벌 점군에 적용한다. 실패 구역은 `None`.
pub fn apply_alignments(
    prelim: &[PointCloud],
    sims: &[Option<Similarity>],
) -> Vec<Option<PointCloud>> {
    assert_eq!(prelim.len(), sims.len(), "구역 수 불일치");
    prelim
        .iter()
        .zip(sims)
        .map(|(c, s)| s.as_ref().map(|s| apply_cloud(s, c)))
        .collect()
}

/// 정수 격자 칸 키용 가벼운 해시(곱셈 섞기).
#[derive(Default)]
struct CellHasher(u64);

impl Hasher for CellHasher {
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^ (h >> 33)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_i64(&mut self, x: i64) {
        self.0 = (self.0.rotate_left(21) ^ x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type CellMap = HashMap<(i64, i64, i64), u32, BuildHasherDefault<CellHasher>>;

const NIL: u32 = u32::MAX;

/// 격자 해시 반경 검사기(반경 고정). 칸마다 힙 할당 없이 평평한 배열 + 칸별 연결 목록.
///
/// 칸 한 변은 반경/√3 이라 칸 대각선이 반경보다 짧다: 질의 점과 같은 칸에 점이 하나라도 있으면
/// 곧바로 참이다. 이웃 칸은 ±2 칸까지 보되 칸 상자까지의 최소 거리가 반경을 넘는 칸은 건너뛴다.
/// 넣은 점군마다 경계 상자를 따로 두어, 어느 상자에서도 반경 밖인 질의는 칸을 보지 않고 거짓이다.
/// 촘촘한 점군에서 질의 비용이 칸 하나의 점 수에 거의 묶이므로 전체 시간이 점 수에 선형에 가깝다.
pub struct RadiusIndex {
    radius: f64,
    cell: f64,
    head: CellMap,
    next: Vec<u32>,
    pts: Vec<[f32; 3]>,
    boxes: Vec<([f64; 3], [f64; 3])>,
}

/// 점에서 축 정렬 상자까지 제곱 거리.
fn box_dist2(p: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let mut d2 = 0.0;
    for i in 0..3 {
        let d = if p[i] < lo[i] {
            lo[i] - p[i]
        } else if p[i] > hi[i] {
            p[i] - hi[i]
        } else {
            0.0
        };
        d2 += d * d;
    }
    d2
}

impl RadiusIndex {
    pub fn new(radius: f64) -> Self {
        assert!(radius > 0.0);
        Self {
            radius,
            // 반경/√3 보다 조금 작게: 반올림으로 칸 대각선이 반경을 넘지 않도록.
            cell: radius / 3f64.sqrt() * (1.0 - 1e-9),
            head: CellMap::default(),
            next: Vec::new(),
            pts: Vec::new(),
            boxes: Vec::new(),
        }
    }

    /// 들어간 점 수.
    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    fn key(&self, p: [f64; 3]) -> (i64, i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
            (p[2] / self.cell).floor() as i64,
        )
    }

    /// 점군의 점을 모두 넣는다. 유한하지 않은 점은 건너뛴다.
    pub fn insert_cloud(&mut self, cloud: &PointCloud) {
        self.pts.reserve(cloud.len());
        self.next.reserve(cloud.len());
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &cloud.points {
            if !p.xyz.iter().all(|x| x.is_finite()) {
                continue;
            }
            let q = [p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64];
            for i in 0..3 {
                lo[i] = lo[i].min(q[i]);
                hi[i] = hi[i].max(q[i]);
            }
            let k = self.key(q);
            let id = u32::try_from(self.pts.len()).expect("점 수가 u32 범위를 넘음");
            let h = self.head.entry(k).or_insert(NIL);
            self.next.push(*h);
            *h = id;
            self.pts.push(p.xyz);
        }
        if lo[0] <= hi[0] {
            self.boxes.push((lo, hi));
        }
    }

    /// 반경 안(거리 ≤ radius)에 점이 있는가.
    pub fn has_within(&self, p: [f64; 3]) -> bool {
        let r2 = self.radius * self.radius;
        if !self
            .boxes
            .iter()
            .any(|(lo, hi)| box_dist2(p, *lo, *hi) <= r2)
        {
            return false;
        }
        let (kx, ky, kz) = self.key(p);
        if self.head.contains_key(&(kx, ky, kz)) {
            return true;
        }
        let c = self.cell;
        for dx in -2i64..=2 {
            for dy in -2i64..=2 {
                for dz in -2i64..=2 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let k = (kx + dx, ky + dy, kz + dz);
                    let lo = [k.0 as f64 * c, k.1 as f64 * c, k.2 as f64 * c];
                    let hi = [lo[0] + c, lo[1] + c, lo[2] + c];
                    if box_dist2(p, lo, hi) > r2 {
                        continue;
                    }
                    let mut i = match self.head.get(&k) {
                        Some(&h) => h,
                        None => continue,
                    };
                    while i != NIL {
                        let q = self.pts[i as usize];
                        let d = [q[0] as f64 - p[0], q[1] as f64 - p[1], q[2] as f64 - p[2]];
                        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r2 {
                            return true;
                        }
                        i = self.next[i as usize];
                    }
                }
            }
        }
        false
    }
}

/// 초벌 점 중 `index` 의 점에서 반경 안에 있는 점을 뺀다(순서 유지).
pub fn remove_ghosts(prelim: &PointCloud, index: &RadiusIndex) -> PointCloud {
    if index.is_empty() {
        return prelim.clone();
    }
    PointCloud {
        points: prelim
            .points
            .par_iter()
            .filter(|p| !index.has_within([p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64]))
            .copied()
            .collect(),
    }
}

/// `every` 개 중 첫 번째만 남기는 간격 추출(0, every, 2·every, …).
pub fn decimate(cloud: &PointCloud, every: usize) -> PointCloud {
    PointCloud {
        points: cloud.points.iter().step_by(every.max(1)).copied().collect(),
    }
}

/// 스냅샷 단계: 들어간 구역 수(1부터) 또는 최종.
/// JSON 에는 정수(1, 2, …) 또는 문자열 `"final"` 로 쓴다.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Index(usize),
    Final,
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Index(k) => write!(f, "{k}"),
            Step::Final => f.write_str("final"),
        }
    }
}

/// 스냅샷 하나(한 단계 분량).
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub step: Step,
    pub cloud: PointCloud,
    pub preview_new_area: usize,
}

/// 스냅샷을 만들며 모은 요약.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotSummary {
    pub entries: Vec<SnapshotEntry>,
    /// 점에 NaN·무한이 있는 단계.
    pub nan_steps: Vec<Step>,
    /// 한 번에 메모리에 들고 있던 스냅샷 점 수의 최댓값(= 가장 큰 단계 분량).
    pub peak_points: usize,
}

/// 스냅샷을 하나씩 만들어 `sink` 에 넘긴다(넘긴 뒤 버퍼를 다음 단계에 다시 쓴다).
///
/// `prelim_aligned[i]` 는 구역 i 의 정렬된 초벌 점군(정렬 실패면 `None`), `refined[i]` 는 정밀 점군
/// (둘 다 추출 전). step_k (k = 1..=n) = 정밀 `0..=k-2` + 초벌 `k-1`. 초벌 점 중 정밀 `0..=k-2` 의
/// 추출 전 점 전부에서 `ghost_radius` 안에 있는 점은 뺀다. 초벌 `k-1` 이 `None` 이면 정밀 `k-1` 로
/// 채우고 `preview_new_area` = 0. 마지막은 정밀 전부. 구성 점군은 각각 `every`:1 간격 추출한다.
/// `preview_new_area` 는 그 단계에 들어간 초벌 점 수(추출 뒤).
pub fn for_each_snapshot<F>(
    prelim_aligned: &[Option<PointCloud>],
    refined: &[PointCloud],
    ghost_radius: f64,
    every: usize,
    mut sink: F,
) -> io::Result<SnapshotSummary>
where
    F: FnMut(&Snapshot) -> io::Result<()>,
{
    assert_eq!(prelim_aligned.len(), refined.len(), "구역 수 불일치");
    let n = refined.len();
    let every = every.max(1);
    let push_dec = |dst: &mut Vec<PointRecord>, src: &PointCloud| {
        dst.extend(src.points.iter().step_by(every).copied());
    };
    let mut index = RadiusIndex::new(ghost_radius);
    let mut cur = Snapshot {
        step: Step::Final,
        cloud: PointCloud { points: Vec::new() },
        preview_new_area: 0,
    };
    // cur.cloud.points[..base_len] = 정밀 0..=k-2 (추출 뒤).
    let mut base_len = 0usize;
    let mut base_nan = false;
    let mut summary = SnapshotSummary {
        entries: Vec::with_capacity(n + 1),
        nan_steps: Vec::new(),
        peak_points: 0,
    };
    let mut emit = |cur: &Snapshot, nan: bool, summary: &mut SnapshotSummary| {
        summary.peak_points = summary.peak_points.max(cur.cloud.len());
        if nan {
            summary.nan_steps.push(cur.step);
        }
        summary.entries.push(SnapshotEntry {
            step: cur.step,
            points: cur.cloud.len(),
            preview_new_area: cur.preview_new_area,
        });
        sink(cur)
    };
    for k in 1..=n {
        cur.cloud.points.truncate(base_len);
        if k >= 2 {
            let r = &refined[k - 2];
            index.insert_cloud(r);
            push_dec(&mut cur.cloud.points, r);
            base_len = cur.cloud.points.len();
            base_nan |= r.has_nan();
        }
        let (fresh_nan, fresh_n) = match &prelim_aligned[k - 1] {
            Some(p) => {
                let fresh = remove_ghosts(p, &index);
                push_dec(&mut cur.cloud.points, &fresh);
                (fresh.has_nan(), cur.cloud.points.len() - base_len)
            }
            None => {
                let r = &refined[k - 1];
                push_dec(&mut cur.cloud.points, r);
                (r.has_nan(), 0)
            }
        };
        cur.step = Step::Index(k);
        cur.preview_new_area = fresh_n;
        emit(&cur, base_nan || fresh_nan, &mut summary)?;
    }
    cur.cloud.points.truncate(base_len);
    let mut nan = base_nan;
    if let Some(r) = refined.last() {
        push_dec(&mut cur.cloud.points, r);
        nan |= r.has_nan();
    }
    cur.step = Step::Final;
    cur.preview_new_area = 0;
    emit(&cur, nan, &mut summary)?;
    Ok(summary)
}

/// 스냅샷 전부를 모아 돌려준다(시험·작은 입력용). 큰 입력은 [`for_each_snapshot`] 을 쓴다.
pub fn build_snapshots(
    prelim_aligned: &[Option<PointCloud>],
    refined: &[PointCloud],
    ghost_radius: f64,
    every: usize,
) -> Vec<Snapshot> {
    let mut out = Vec::new();
    for_each_snapshot(prelim_aligned, refined, ghost_radius, every, |s| {
        out.push(s.clone());
        Ok(())
    })
    .expect("모으기만 하므로 실패하지 않음");
    out
}

/// 정수 step 사이 점 수 단조 판정(흐름 issue 와 `verify` 가 같이 쓴다): 같은 수는 허용, 감소만 위반.
/// final 은 정밀 점군 간격 추출 결과라 마지막 step 보다 작을 수 있어 판정에서 뺀다.
pub fn snapshot_count_decreased(prev_points: f64, next_points: f64) -> bool {
    next_points < prev_points
}

/// SPEC §4 스냅샷 기준 검사: 정수 step 점 수 단조 증가(같은 수 허용), 2단계부터 초벌 새 영역 > 0, NaN 없음.
/// 위반을 사람이 읽을 문장으로 돌려준다(없으면 빈 목록).
pub fn check_snapshots(summary: &SnapshotSummary) -> Vec<String> {
    let mut issues = Vec::new();
    let ints: Vec<&SnapshotEntry> = summary
        .entries
        .iter()
        .filter(|e| matches!(e.step, Step::Index(_)))
        .collect();
    for w in ints.windows(2) {
        if snapshot_count_decreased(w[0].points as f64, w[1].points as f64) {
            issues.push(format!(
                "스냅샷 점 수 단조 증가 아님: step {} {} → step {} {}",
                w[0].step, w[0].points, w[1].step, w[1].points
            ));
        }
    }
    for e in &summary.entries {
        if let Step::Index(k) = e.step {
            if k >= 2 && e.preview_new_area == 0 {
                issues.push(format!("step {k}: 초벌 새 영역 0"));
            }
        }
    }
    for s in &summary.nan_steps {
        issues.push(format!("step {s}: NaN·무한 점 있음"));
    }
    issues
}

/// 매니페스트의 스냅샷 줄.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotEntry {
    pub step: Step,
    pub points: usize,
    pub preview_new_area: usize,
}

/// `snapshots/manifest.json` 내용.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Manifest {
    pub snapshots: Vec<SnapshotEntry>,
    pub align: Vec<AlignRecord>,
}

fn json_f64(x: Option<f64>) -> String {
    match x {
        // `{:?}` 는 왕복 가능한 최단 표기.
        Some(x) if x.is_finite() => format!("{x:?}"),
        _ => "null".to_string(),
    }
}

impl Manifest {
    pub fn new(snapshots: Vec<SnapshotEntry>, align: Vec<AlignRecord>) -> Self {
        Self { snapshots, align }
    }

    pub fn to_json(&self) -> String {
        let snaps: Vec<String> = self
            .snapshots
            .iter()
            .map(|s| {
                let step = match s.step {
                    Step::Index(k) => k.to_string(),
                    Step::Final => "\"final\"".to_string(),
                };
                format!(
                    "    {{\"step\": {step}, \"points\": {}, \"preview_new_area\": {}}}",
                    s.points, s.preview_new_area
                )
            })
            .collect();
        let align: Vec<String> = self
            .align
            .iter()
            .map(|a| {
                format!(
                    "    {{\"region\": {}, \"pairs\": {}, \"fit_median_m\": {}, \"scale\": {}}}",
                    a.region,
                    a.pairs,
                    json_f64(a.fit_median_m),
                    json_f64(a.scale)
                )
            })
            .collect();
        format!(
            "{{\n  \"snapshots\": [\n{}\n  ],\n  \"align\": [\n{}\n  ]\n}}\n",
            snaps.join(",\n"),
            align.join(",\n")
        )
    }

    /// 읽기: `step` 은 정수 또는 `"final"`. 예전 문자열 표기("01")도 받는다.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v = Json::parse(text)?;
        let arr = |key: &str| -> Result<Vec<Json>, String> {
            match v.get(key) {
                Some(Json::Arr(a)) => Ok(a.clone()),
                _ => Err(format!("{key} 배열 없음")),
            }
        };
        let count = |o: &Json, key: &str| -> Result<usize, String> {
            match o.get(key) {
                Some(Json::Num(x)) if *x >= 0.0 && x.fract() == 0.0 => Ok(*x as usize),
                _ => Err(format!("{key} 음 아닌 정수 없음")),
            }
        };
        let opt = |o: &Json, key: &str| -> Result<Option<f64>, String> {
            match o.get(key) {
                Some(Json::Num(x)) => Ok(Some(*x)),
                Some(Json::Null) => Ok(None),
                _ => Err(format!("{key} 숫자 없음")),
            }
        };
        let mut m = Manifest::default();
        for o in arr("snapshots")? {
            let step = match o.get("step") {
                Some(Json::Str(s)) if s == "final" => Step::Final,
                Some(Json::Str(s)) => Step::Index(
                    s.parse::<usize>()
                        .map_err(|_| format!("step 은 정수 또는 \"final\": {s}"))?,
                ),
                Some(Json::Num(_)) => Step::Index(count(&o, "step")?),
                _ => return Err("step 없음".into()),
            };
            m.snapshots.push(SnapshotEntry {
                step,
                points: count(&o, "points")?,
                preview_new_area: count(&o, "preview_new_area")?,
            });
        }
        for o in arr("align")? {
            m.align.push(AlignRecord {
                region: count(&o, "region")?,
                pairs: count(&o, "pairs")?,
                fit_median_m: opt(&o, "fit_median_m")?,
                scale: opt(&o, "scale")?,
            });
        }
        Ok(m)
    }
}

/// 매니페스트 읽기에 쓰는 작은 JSON 값.
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn parse(text: &str) -> Result<Json, String> {
        let b = text.as_bytes();
        let mut i = 0;
        let v = Self::value(b, &mut i)?;
        Self::ws(b, &mut i);
        if i != b.len() {
            return Err(format!("남은 문자 @{i}"));
        }
        Ok(v)
    }

    fn ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }

    fn lit(b: &[u8], i: &mut usize, word: &str, v: Json) -> Result<Json, String> {
        if b[*i..].starts_with(word.as_bytes()) {
            *i += word.len();
            Ok(v)
        } else {
            Err(format!("알 수 없는 값 @{i}"))
        }
    }

    fn value(b: &[u8], i: &mut usize) -> Result<Json, String> {
        Self::ws(b, i);
        match b.get(*i) {
            None => Err("입력 끝".into()),
            Some(b'n') => Self::lit(b, i, "null", Json::Null),
            Some(b't') => Self::lit(b, i, "true", Json::Bool(true)),
            Some(b'f') => Self::lit(b, i, "false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(Self::string(b, i)?)),
            Some(b'[') => {
                *i += 1;
                let mut a = Vec::new();
                Self::ws(b, i);
                if b.get(*i) == Some(&b']') {
                    *i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(Self::value(b, i)?);
                    Self::ws(b, i);
                    match b.get(*i) {
                        Some(b',') => *i += 1,
                        Some(b']') => {
                            *i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(format!("배열 구분자 @{i}")),
                    }
                }
            }
            Some(b'{') => {
                *i += 1;
                let mut kv = Vec::new();
                Self::ws(b, i);
                if b.get(*i) == Some(&b'}') {
                    *i += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    Self::ws(b, i);
                    let k = Self::string(b, i)?;
                    Self::ws(b, i);
                    if b.get(*i) != Some(&b':') {
                        return Err(format!("콜론 @{i}"));
                    }
                    *i += 1;
                    kv.push((k, Self::value(b, i)?));
                    Self::ws(b, i);
                    match b.get(*i) {
                        Some(b',') => *i += 1,
                        Some(b'}') => {
                            *i += 1;
                            return Ok(Json::Obj(kv));
                        }
                        _ => return Err(format!("객체 구분자 @{i}")),
                    }
                }
            }
            Some(_) => {
                let s = *i;
                while *i < b.len()
                    && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    *i += 1;
                }
                std::str::from_utf8(&b[s..*i])
                    .ok()
                    .and_then(|t| t.parse::<f64>().ok())
                    .map(Json::Num)
                    .ok_or_else(|| format!("숫자 @{s}"))
            }
        }
    }

    fn string(b: &[u8], i: &mut usize) -> Result<String, String> {
        if b.get(*i) != Some(&b'"') {
            return Err(format!("문자열 @{i}"));
        }
        *i += 1;
        let mut out: Vec<u8> = Vec::new();
        while let Some(&c) = b.get(*i) {
            *i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
                b'\\' => {
                    let e = *b.get(*i).ok_or("이스케이프 끝")?;
                    *i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'u' => {
                            let h = std::str::from_utf8(b.get(*i..*i + 4).ok_or("\\u 끝")?)
                                .map_err(|e| e.to_string())?;
                            let cp = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                            *i += 4;
                            let ch = char::from_u32(cp).unwrap_or('\u{fffd}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                c => out.push(c),
            }
        }
        Err("문자열 끝 없음".into())
    }
}

/// 구역 파일 이름 `preview/preview_{k:02}_pos{lo}-{hi}.ply` (k = 구역 번호, hi 는 포함 안 함).
pub fn preview_name(r: &Region) -> String {
    format!("preview/preview_{:02}_pos{}-{}.ply", r.index, r.lo, r.hi)
}

/// 구역 파일 이름 `refined/refined_{k:02}_pos{lo}-{hi}.ply`.
pub fn refined_name(r: &Region) -> String {
    format!("refined/refined_{:02}_pos{}-{}.ply", r.index, r.lo, r.hi)
}

/// 스냅샷 파일 이름 `snapshots/step_{k:02}_{k}regions.ply`, 최종은 `snapshots/step_final_all_refined.ply`.
pub fn snapshot_name(step: Step) -> String {
    match step {
        Step::Index(k) => format!("snapshots/step_{k:02}_{k}regions.ply"),
        Step::Final => "snapshots/step_final_all_refined.ply".to_string(),
    }
}

/// [`write_outputs`] 결과.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamReport {
    pub manifest: Manifest,
    /// SPEC §4 위반 목록. 검사 범위: 구역 0개, 스냅샷 단조성·2단계부터 새 영역·NaN,
    /// 초벌 정렬 실패, `align` 기록의 점쌍 < [`ALIGN_MIN_PAIRS`]·잔차 중앙 ≥ [`FIT_MEDIAN_LIMIT_M`]·
    /// 구역 간 스케일 차 > [`ALIGN_SCALE_TOL`]. 비어 있어도 이 범위 밖 기준(재투영·높이 등)은 판정하지 않는다.
    pub issues: Vec<String>,
    /// 한 번에 메모리에 들고 있던 스냅샷 점 수 최댓값.
    pub peak_snapshot_points: usize,
}

/// 출력 전부를 `out_dir` 아래에 쓴다. 구역 점군도 간격 추출해 쓴다.
///
/// `prelim_aligned[i]` 가 `None`(정렬 실패)이면 그 구역 preview 파일은 점 0개로 쓰고,
/// 스냅샷은 정밀로 채운다. `align` 의 실패 기록은 `fit_median_m`·`scale` 이 null 로 남는다.
/// 위반이 있어도 파일은 모두 쓰고, 위반은 [`StreamReport::issues`] 로 돌려준다.
pub fn write_outputs(
    out_dir: impl AsRef<Path>,
    regions: &[Region],
    prelim_aligned: &[Option<PointCloud>],
    refined: &[PointCloud],
    align: Vec<AlignRecord>,
) -> io::Result<StreamReport> {
    let n = regions.len();
    if prelim_aligned.len() != n || refined.len() != n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "구역 수 불일치: 구역 {n}, 초벌 {}, 정밀 {}",
                prelim_aligned.len(),
                refined.len()
            ),
        ));
    }
    let dir = out_dir.as_ref();
    for sub in ["preview", "refined", "snapshots"] {
        std::fs::create_dir_all(dir.join(sub))?;
    }
    let empty = PointCloud { points: Vec::new() };
    for ((r, p), q) in regions.iter().zip(prelim_aligned).zip(refined) {
        let p = p.as_ref().unwrap_or(&empty);
        write_ply_file(dir.join(preview_name(r)), &decimate(p, DECIMATE_EVERY))?;
        write_ply_file(dir.join(refined_name(r)), &decimate(q, DECIMATE_EVERY))?;
    }
    let summary = for_each_snapshot(
        prelim_aligned,
        refined,
        GHOST_RADIUS_M,
        DECIMATE_EVERY,
        |s| write_ply_file(dir.join(snapshot_name(s.step)), &s.cloud),
    )?;
    let mut issues = check_snapshots(&summary);
    if n == 0 {
        issues.push("구역 0개".to_string());
    }
    for (r, p) in regions.iter().zip(prelim_aligned) {
        if p.is_none() {
            issues.push(format!("구역 {}: 초벌 정렬 실패", r.index));
        }
    }
    for a in &align {
        if let Some(m) = a.fit_median_m {
            if m.is_nan() || m >= FIT_MEDIAN_LIMIT_M {
                issues.push(format!(
                    "구역 {}: 정렬 잔차 중앙 {m:.3} m ≥ {FIT_MEDIAN_LIMIT_M} m",
                    a.region
                ));
            }
        }
    }
    for a in &align {
        if a.pairs < ALIGN_MIN_PAIRS {
            issues.push(format!(
                "구역 {}: 정렬 점쌍 {} < {ALIGN_MIN_PAIRS}",
                a.region, a.pairs
            ));
        }
    }
    let scales: Vec<f64> = align.iter().filter_map(|a| a.scale).collect();
    if !scales.is_empty() {
        let lo = scales.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = scales.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let spread = hi / lo - 1.0;
        if !(lo > 0.0 && spread <= ALIGN_SCALE_TOL) {
            issues.push(format!(
                "구역 간 스케일 차 {spread:.3} > {ALIGN_SCALE_TOL} (최소 {lo:.4}, 최대 {hi:.4})"
            ));
        }
    }
    let manifest = Manifest::new(summary.entries, align);
    std::fs::write(dir.join("snapshots/manifest.json"), manifest.to_json())?;
    Ok(StreamReport {
        manifest,
        issues,
        peak_snapshot_points: summary.peak_points,
    })
}

#[cfg(test)]
mod tests {
    /// 휜 초벌(포물선 처짐 8 m)을 전체 닮음으로만 맞추면 높이 잔차가 남고, 보정장은 0.3 m 아래로 줄인다.
    #[test]
    fn warp_field_absorbs_bending() {
        let mut pairs = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                let (x, y) = (i as f64 * 2.5 - 50.0, j as f64 * 2.5 - 50.0);
                let truth = Vector3::new(x, y, 0.0);
                let bend = 8.0 * ((x * x + y * y) / 5000.0 - 0.5);
                pairs.push((Vector3::new(x, y, bend), truth));
            }
        }
        let (src, dst): (Vec<_>, Vec<_>) = pairs.iter().copied().unzip();
        let (sim, _, med) = robust_fit(&src, &dst).unwrap();
        let global: Vec<f64> = pairs
            .iter()
            .map(|(a, b)| (sim.apply_point(a).z - b.z).abs())
            .collect();
        let w = WarpField::fit(&sim, &pairs, med, 200).expect("보정장");
        let mut warped: Vec<f64> = pairs
            .iter()
            .map(|(a, b)| (w.apply_point(a).z - b.z).abs())
            .collect();
        let mut g = global;
        g.sort_by(|a, b| a.total_cmp(b));
        warped.sort_by(|a, b| a.total_cmp(b));
        let (gm, wm) = (g[g.len() / 2], warped[warped.len() / 2]);
        assert!(gm > 1.0, "전체 닮음 중앙 {gm}");
        assert!(wm < 0.3, "보정장 중앙 {wm}");
        // 짝이 모자라면 보정장 없음.
        assert!(WarpField::fit(&sim, &pairs[..50], med, 200).is_none());
    }

    use super::*;
    use nalgebra::{Rotation3, Unit};

    /// 결정적 의사난수(xorshift), [-1, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    fn rec(p: Vector3<f64>) -> PointRecord {
        PointRecord {
            xyz: [p.x as f32, p.y as f32, p.z as f32],
            normal: [0.0, 0.0, 1.0],
            rgb: [100, 120, 140],
        }
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        let m = v.len() / 2;
        if v.len() % 2 == 1 {
            v[m]
        } else {
            0.5 * (v[m - 1] + v[m])
        }
    }

    #[test]
    fn region_bounds_by_hand() {
        // 80곳, SPAN 12, OVL 2: start = 0,12,…,72 → 7구역.
        let r = split_regions(80, 12, 2);
        let got: Vec<(usize, usize, usize)> = r.iter().map(|r| (r.start, r.lo, r.hi)).collect();
        assert_eq!(
            got,
            vec![
                (0, 0, 14),
                (12, 10, 26),
                (24, 22, 38),
                (36, 34, 50),
                (48, 46, 62),
                (60, 58, 74),
                (72, 70, 80),
            ]
        );
        // 정렬 창: 구역 0 은 자기 구역, 그 외 [start-2, start+2).
        assert_eq!(align_window(&r[0], 2, 80), (0, 14));
        assert_eq!(align_window(&r[1], 2, 80), (10, 14));
        assert_eq!(align_window(&r[6], 2, 80), (70, 74));
        // 위치 수가 SPAN 의 배수: 24곳 → 2구역.
        let r = split_regions(24, 12, 2);
        assert_eq!(r.len(), 2);
        assert_eq!((r[1].lo, r[1].hi), (10, 24));
        assert!(r[1].contains(10) && !r[1].contains(24) && !r[1].contains(9));
        assert_eq!(preview_name(&r[1]), "preview/preview_01_pos10-24.ply");
        assert_eq!(refined_name(&r[0]), "refined/refined_00_pos0-14.ply");
        // 꼬리 합치기: 26곳 → start 24 는 새 위치가 없어 앞 구역에 합쳐진다.
        let r = split_regions(26, 12, 2);
        let got: Vec<(usize, usize)> = r.iter().map(|r| (r.lo, r.hi)).collect();
        assert_eq!(got, vec![(0, 14), (10, 26)]);
        let r = split_regions(14, 12, 2);
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].lo, r[0].hi), (0, 14));
        assert!(split_regions(0, 12, 2).is_empty());
    }

    #[test]
    fn regions_cover_and_advance() {
        for n in 1..=200 {
            for span in [4, 12] {
                for ovl in [0, 2] {
                    let r = split_regions(n, span, ovl);
                    assert!(!r.is_empty());
                    assert_eq!(r[0].lo, 0);
                    assert_eq!(r.last().unwrap().hi, n, "n={n} span={span} ovl={ovl}");
                    for (i, w) in r.windows(2).enumerate() {
                        assert_eq!(w[1].index, i + 1);
                        // 끊김 없이 이어지고 앞 구역 끝 너머 새 위치가 1개 이상.
                        assert!(w[1].lo <= w[0].hi, "n={n} span={span} ovl={ovl}");
                        assert!(w[1].hi > w[0].hi, "n={n} span={span} ovl={ovl} {w:?}");
                    }
                }
            }
        }
    }

    fn known_sim() -> Similarity {
        Similarity {
            s: 0.37,
            r: Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(0.3, -0.5, 0.8)), 1.1),
            t: Vector3::new(12.0, -4.0, 30.0),
        }
    }

    #[test]
    fn align_region_degenerate_is_none() {
        let region = split_regions(14, 12, 2)[0];
        // x 축 위 20점, 정답 s = 2: 회전이 정해지지 않으므로 None.
        let line: Vec<(Vector3<f64>, Vector3<f64>)> = (0..20)
            .map(|i| {
                let p = Vector3::new(i as f64, 0.0, 0.0);
                (p, 2.0 * p)
            })
            .collect();
        let (sim, rec) = align_region(&region, &line);
        assert!(sim.is_none());
        assert_eq!(rec.pairs, 20);
        assert_eq!((rec.fit_median_m, rec.scale), (None, None));
        // 정상 50점에 NaN 한 점.
        let g = known_sim();
        let mut rng = Rng(3);
        let mut pairs: Vec<(Vector3<f64>, Vector3<f64>)> = (0..50)
            .map(|_| {
                let p = Vector3::new(rng.next() * 10.0, rng.next() * 10.0, rng.next() * 3.0);
                (p, g.apply_point(&p))
            })
            .collect();
        assert!(align_region(&region, &pairs).0.is_some());
        pairs[7].0.y = f64::NAN;
        assert!(align_region(&region, &pairs).0.is_none());
        // 짝 0개.
        let (sim, rec) = align_region(&region, &[]);
        assert!(sim.is_none() && rec.pairs == 0);
    }

    #[test]
    fn robust_similarity_rejects_outliers() {
        let mut rng = Rng(11);
        let g = known_sim();
        let mut pairs = Vec::new();
        for i in 0..400 {
            let p = Vector3::new(rng.next() * 30.0, rng.next() * 30.0, rng.next() * 3.0);
            let mut q = g.apply_point(&p) + Vector3::new(rng.next(), rng.next(), rng.next()) * 0.02;
            if i % 5 == 0 {
                // 20 % 큰 오대응(5~15 m).
                q += Vector3::new(rng.next(), rng.next(), rng.next()).normalize() * 10.0
                    + Vector3::new(5.0 * rng.next().signum(), 0.0, 0.0);
            }
            pairs.push((p, q));
        }
        let region = split_regions(14, 12, 2)[0];
        let (e, rec) = align_region(&region, &pairs);
        let e = e.unwrap();
        // 정상 짝은 모두 정답과 맞고 오대응은 정답에서 멀다.
        for (i, (p, q)) in pairs.iter().enumerate() {
            let r = (e.apply_point(p) - q).norm();
            if i % 5 == 0 {
                assert!(r > 3.0, "오대응 {i} 잔차 {r}");
            } else {
                assert!(r < 0.1, "정상 {i} 잔차 {r}");
            }
        }
        assert!((e.s / g.s - 1.0).abs() < 1e-3);
        // 잡음 성분 ±0.02 m 균일 → 3차원 잔차 중앙 ≈ 0.02 m 수준.
        let med = rec.fit_median_m.unwrap();
        assert!(med < 0.03, "med {med}");
    }

    /// 정답 지면: x(진행 방향, 위치 1곳 = 1 m) × y(±15 m), 완만한 높낮이.
    fn truth_points(n_pos: usize) -> Vec<(usize, Vector3<f64>)> {
        let mut v = Vec::new();
        for ix in 0..n_pos * 4 {
            for iy in 0..60 {
                let x = ix as f64 * 0.25;
                let y = -15.0 + iy as f64 * 0.5;
                let z = 2.0 * (x * 0.07).sin() + 1.5 * (y * 0.11).cos();
                v.push(((x as usize).min(n_pos - 1), Vector3::new(x, y, z)));
            }
        }
        v
    }

    /// 합성 초벌 설정: 휨 z += bend·(x − bend_origin)², 오대응 비율(초벌 점을 ±20 m 무작위로 옮김).
    struct Warp {
        bend: f64,
        bend_origin: f64,
        outlier_frac: f64,
    }

    /// 구역마다 초벌 = 정답에 휨·잡음(±0.1 m)·오대응을 넣고 알려진 닮음 변환의 역을 적용.
    /// 정밀 = 정답 + 작은 잡음. 관측은 (위치·3, 점 번호) 하나.
    /// 반환: (초벌, 정밀, 정답, 오대응 표시).
    #[allow(clippy::type_complexity)]
    fn synth_region(
        region: &Region,
        truth: &[(usize, Vector3<f64>)],
        g: &Similarity,
        warp: &Warp,
        rng: &mut Rng,
    ) -> (Vec<Track>, Vec<Track>, Vec<Vector3<f64>>, Vec<bool>) {
        let ginv = g.inverse();
        let mut pre = Vec::new();
        let mut fine = Vec::new();
        let mut truth_in = Vec::new();
        let mut bad = Vec::new();
        for (id, (pos, p)) in truth.iter().enumerate() {
            if !region.contains(*pos) {
                continue;
            }
            let obs = vec![(*pos as u32 * 3, id as u32)];
            let d = p.x - warp.bend_origin;
            let mut warped = p
                + Vector3::new(0.0, 0.0, warp.bend * d * d)
                + Vector3::new(rng.next(), rng.next(), rng.next()) * 0.1;
            let is_bad = 0.5 * (rng.next() + 1.0) < warp.outlier_frac;
            if is_bad {
                warped += Vector3::new(rng.next(), rng.next(), rng.next()) * 20.0;
            }
            pre.push(Track {
                xyz: ginv.apply_point(&warped),
                obs: obs.clone(),
            });
            fine.push(Track {
                xyz: p + Vector3::new(rng.next(), rng.next(), rng.next()) * 0.02,
                obs,
            });
            truth_in.push(*p);
            bad.push(is_bad);
        }
        (pre, fine, truth_in, bad)
    }

    fn region_sim(k: usize) -> Similarity {
        Similarity {
            s: 0.5 + 0.1 * k as f64,
            r: Rotation3::from_euler_angles(0.1 * k as f64, -0.2, 0.7 + 0.3 * k as f64),
            t: Vector3::new(3.0 * k as f64, -7.0, 40.0),
        }
    }

    /// 한 구역 정렬: (스케일 비, fit 중앙, 창 안 정상 점 정답 잔차 중앙, 구역 전체 정상 점 정답 잔차 중앙).
    fn run_region(n_pos: usize, k: usize, warp: &Warp, seed: u64) -> (f64, f64, f64, f64) {
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        let r = &regions[k];
        let truth = truth_points(n_pos);
        let mut rng = Rng(seed);
        let g = region_sim(k);
        let (pre, fine, truth_in, bad) = synth_region(r, &truth, &g, warp, &mut rng);
        let win = align_window(r, DEFAULT_OVL, n_pos);
        let pairs = point_pairs(&pre, &fine, |img| (img / 3) as usize, win);
        // 창 안 위치 수 × 4줄 × 60점.
        assert_eq!(pairs.len(), (win.1 - win.0) * 4 * 60);
        let (sim, rec) = align_region(r, &pairs);
        let sim = sim.expect("정렬 실패");
        assert_eq!(rec.region, k);
        let mut all = Vec::new();
        let mut inwin = Vec::new();
        for ((a, b), &bad) in pre.iter().zip(&truth_in).zip(&bad) {
            if bad {
                continue;
            }
            let e = (sim.apply_point(&a.xyz) - b).norm();
            all.push(e);
            if b.x >= win.0 as f64 && b.x < win.1 as f64 {
                inwin.push(e);
            }
        }
        (
            rec.scale.unwrap() / g.s,
            rec.fit_median_m.unwrap(),
            median(&mut inwin),
            median(&mut all),
        )
    }

    #[test]
    fn prelim_alignment_against_truth() {
        let n_pos = 40;
        let n_regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL).len();
        assert_eq!(n_regions, 4);
        for k in 0..n_regions {
            // 휨 0.002 /m², 구역 시작 기준: 구역 끝(start+14 m)에서 0.39 m.
            let warp = Warp {
                bend: 0.002,
                bend_origin: (k * DEFAULT_SPAN) as f64,
                outlier_frac: 0.0,
            };
            let (sr, fit, _win, all) = run_region(n_pos, k, &warp, 99 + k as u64);
            eprintln!("구역 {k}: fit {fit:.4} 정답 잔차 중앙 {all:.4} 스케일비 {sr:.5}");
            // 휨이 창 안에서 작으므로 스케일은 1 % 안(SPEC 허용 ±10 %).
            assert!((sr - 1.0).abs() < 0.01, "scale {sr}");
            // 기준: 휨 최대 0.39 m + 잡음 0.1 m → 정답 잔차 중앙 0.5 m 미만.
            assert!(all < 0.5, "구역 {k} 잔차 중앙 {all}");
            assert!(fit < 0.3, "fit {fit}");
        }
    }

    /// 정렬 창 안에도 휨이 있는 경우(휨 원점을 구역 시작 8 m 앞에 둠, 0.01 /m²).
    /// 창 [start-2, start+2) 에서 z 휨 0.36~1.0 m, 기울기 0.12~0.2: 닮음 변환은 기울기를 회전으로
    /// 흡수하고 창 폭 4 m 안 2차 성분(≈ c·w²/8 = 0.02 m)만 남으므로 fit 중앙 < 0.3 m.
    /// 창 밖으로는 휨이 계속 자라므로 구역 전체 정답 잔차는 창 안보다 크다(초벌 휨의 한계).
    #[test]
    fn prelim_alignment_bent_inside_window() {
        let k = 1;
        let warp = Warp {
            bend: 0.01,
            bend_origin: (k * DEFAULT_SPAN) as f64 - 8.0,
            outlier_frac: 0.0,
        };
        let (sr, fit, win, all) = run_region(40, k, &warp, 5);
        eprintln!("창 안 휨: 스케일비 {sr:.4} fit {fit:.4} 창 안 {win:.4} 전체 {all:.4}");
        assert!((sr - 1.0).abs() < 0.10, "scale {sr}");
        assert!(fit < 0.3, "fit {fit}");
        assert!(win < 0.5, "창 안 {win}");
        assert!(all > win, "창 밖 휨이 잔차를 키워야 함");
    }

    /// 오대응 30 %·50 %: 기준(측정 전) 스케일 1 %, 창 안 정상 점 정답 잔차 중앙 < 0.3 m, fit 중앙 < 0.3 m.
    /// 첫 추정을 전체 짝 최소제곱으로 하던 때는 ±20 m 오대응에 끌려 스케일이 0.41 로 무너지고
    /// 그 잔차 중앙으로 임계를 잡아 복구하지 못했다. 표본 짝 잔차 분위수로 고르는 강건 첫 추정
    /// ([`robust_fit`]) 뒤에는 복구한다: 30 % 스케일비 1.0000·fit 0.1005, 50 % 1.0002·0.0983.
    #[test]
    fn prelim_alignment_outlier_fractions() {
        for (frac, seed) in [(0.3, 21u64), (0.5, 22)] {
            let warp = Warp {
                bend: 0.002,
                bend_origin: 12.0,
                outlier_frac: frac,
            };
            let (sr, fit, win, _all) = run_region(40, 1, &warp, seed);
            eprintln!("오대응 {frac}: 스케일비 {sr:.4} fit {fit:.4} 창 안 {win:.4}");
            assert!((sr - 1.0).abs() < 0.01, "오대응 {frac} scale {sr}");
            assert!(win < 0.3, "오대응 {frac} 창 안 {win}");
            assert!(fit < 0.3, "오대응 {frac} fit {fit}");
        }
    }

    /// 정렬이 틀렸다면 조용히 지나가면 안 된다: 오대응 30·50·70 % 에서 창 안 정상 점 정답 잔차가
    /// 1 m 이상이면 fit 중앙도 1 m 를 넘어야 한다(트리밍 바닥 0.3 m 의 3배 남짓).
    #[test]
    fn prelim_alignment_failure_is_visible() {
        for (frac, seed) in [(0.3, 21u64), (0.5, 22), (0.7, 23)] {
            let warp = Warp {
                bend: 0.002,
                bend_origin: 12.0,
                outlier_frac: frac,
            };
            let (sr, fit, win, _all) = run_region(40, 1, &warp, seed);
            eprintln!("오대응 {frac}: 스케일비 {sr:.4} fit {fit:.4} 창 안 {win:.4}");
            assert!(
                win < 1.0 || fit > 1.0,
                "조용한 실패 {frac}: 창 안 {win} fit {fit}"
            );
        }
    }

    #[test]
    fn normals_rotate_only() {
        let g = known_sim();
        let c = PointCloud {
            points: vec![PointRecord {
                xyz: [1.0, 2.0, 3.0],
                normal: [0.0, 0.0, 1.0],
                rgb: [1, 2, 3],
            }],
        };
        let o = apply_cloud(&g, &c);
        let n = Vector3::new(
            o.points[0].normal[0],
            o.points[0].normal[1],
            o.points[0].normal[2],
        );
        assert!((n.norm() - 1.0).abs() < 1e-6);
        let want = g.r * Vector3::z();
        assert!((n.cast::<f64>() - want).norm() < 1e-6);
        let p = g.apply_point(&Vector3::new(1.0, 2.0, 3.0));
        assert!((o.points[0].xyz[0] as f64 - p.x).abs() < 1e-4);
        assert_eq!(o.points[0].rgb, [1, 2, 3]);
    }

    #[test]
    fn ghost_filter_exact() {
        let refined = PointCloud {
            points: vec![rec(Vector3::new(0.0, 0.0, 0.0))],
        };
        let mut idx = RadiusIndex::new(GHOST_RADIUS_M);
        idx.insert_cloud(&refined);
        let dists = [0.0, 1.0, 1.49, 1.51, 2.9, 3.1, 10.0];
        let pre = PointCloud {
            points: dists
                .iter()
                .map(|&d| rec(Vector3::new(d * 0.6, -d * 0.8, 0.0)))
                .collect(),
        };
        let kept = remove_ghosts(&pre, &idx);
        assert_eq!(kept.len(), 4); // 1.51, 2.9, 3.1, 10.0
                                   // 무작위 점들: 두 번에 나눠 넣어도 전수 비교와 같아야 한다.
        let mut rng = Rng(5);
        let mut mk = |n: usize, s: f64| PointCloud {
            points: (0..n)
                .map(|_| {
                    rec(Vector3::new(
                        rng.next() * s,
                        rng.next() * s,
                        rng.next() * 3.0,
                    ))
                })
                .collect(),
        };
        let base_a = mk(300, 20.0);
        let base_b = mk(200, 20.0);
        let probe = mk(2000, 25.0);
        let mut idx = RadiusIndex::new(GHOST_RADIUS_M);
        idx.insert_cloud(&base_a);
        idx.insert_cloud(&base_b);
        assert_eq!(idx.len(), 500);
        let fast = remove_ghosts(&probe, &idx);
        let brute: Vec<PointRecord> = probe
            .points
            .iter()
            .filter(|p| {
                !base_a.points.iter().chain(&base_b.points).any(|q| {
                    let d: f64 = (0..3)
                        .map(|i| (p.xyz[i] as f64 - q.xyz[i] as f64).powi(2))
                        .sum();
                    d <= GHOST_RADIUS_M * GHOST_RADIUS_M
                })
            })
            .copied()
            .collect();
        assert_eq!(fast.points, brute);
        assert!(!fast.is_empty() && fast.len() < probe.len());
    }

    fn region_cloud(r: &Region, truth: &[(usize, Vector3<f64>)], dz: f64) -> PointCloud {
        PointCloud {
            points: truth
                .iter()
                .filter(|(pos, _)| r.contains(*pos))
                .map(|(_, p)| rec(p + Vector3::new(0.0, 0.0, dz)))
                .collect(),
        }
    }

    #[test]
    fn snapshots_monotone_and_manifest_roundtrip() {
        let n_pos = 60; // 마지막 구역(46..60)이 앞 정밀 구역(34..50) 밖으로 나가도록.
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        assert_eq!(regions.len(), 5);
        let truth = truth_points(n_pos);
        let refined: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.0))
            .collect();
        // 초벌은 정렬 뒤 0.4 m 떠 있다고 둔다(1.5 m 이내 → 겹침 구간은 잔상으로 빠짐).
        let prelim: Vec<Option<PointCloud>> = regions
            .iter()
            .map(|r| Some(region_cloud(r, &truth, 0.4)))
            .collect();
        let mut steps = Vec::new();
        let summary = for_each_snapshot(&prelim, &refined, GHOST_RADIUS_M, DECIMATE_EVERY, |s| {
            steps.push(s.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(steps.len(), regions.len() + 1);
        let issues = check_snapshots(&summary);
        assert!(issues.is_empty(), "{issues:?}");
        for w in steps.windows(2) {
            assert!(w[1].cloud.len() > w[0].cloud.len(), "단조 증가 아님");
        }
        // 한 번에 들고 있는 점은 가장 큰 단계 하나 분량.
        let biggest = steps.iter().map(|s| s.cloud.len()).max().unwrap();
        assert_eq!(summary.peak_points, biggest);
        for s in &steps[1..regions.len()] {
            assert!(s.preview_new_area > 0);
        }
        // 손 계산(step 2 = 정밀 0..=0 + 초벌 1): 잔상 기준은 정밀 구역 0 의 추출 전 점 전부
        // (위치 0..14 → x ≤ 13.75). 초벌 1 (위치 10..26 → x ∈ [10, 26), 0.25 간격)은 0.4 m 위라
        // 수평 √(1.5²−0.4²) = 1.446 m 안이면 빠진다. 남는 조건 x − 13.75 > 1.446 → x ≥ 15.25
        // → 15.25..25.75 = 43줄 × 60.
        let kept_pre: usize = 43 * 60;
        let expect_new = kept_pre.div_ceil(DECIMATE_EVERY);
        assert_eq!(steps[1].preview_new_area, expect_new);
        let expect_ref0 = refined[0].len().div_ceil(DECIMATE_EVERY);
        assert_eq!(steps[1].cloud.len(), expect_ref0 + expect_new);
        // 최종 = 정밀 전부.
        let total: usize = refined
            .iter()
            .map(|c| c.len().div_ceil(DECIMATE_EVERY))
            .sum();
        assert_eq!(steps.last().unwrap().cloud.len(), total);
        // 모아 만드는 판과 같다.
        assert_eq!(
            build_snapshots(&prelim, &refined, GHOST_RADIUS_M, DECIMATE_EVERY),
            steps
        );

        let align = vec![
            AlignRecord {
                region: 0,
                pairs: 3360,
                fit_median_m: Some(0.123456789),
                scale: Some(1.0),
            },
            AlignRecord {
                region: 1,
                pairs: 0,
                fit_median_m: None,
                scale: None,
            },
        ];
        let m = Manifest::new(summary.entries, align);
        assert_eq!(m.snapshots[0].step, Step::Index(1));
        assert_eq!(m.snapshots.last().unwrap().step, Step::Final);
        let text = m.to_json();
        assert!(text.contains("\"step\": 1,"), "{text}");
        assert!(text.contains("\"step\": \"final\""), "{text}");
        assert!(text.contains("\"scale\": null"), "{text}");
        let back = Manifest::from_json(&text).unwrap();
        assert_eq!(back, m);
        // 예전 문자열 표기도 읽는다.
        let old = text.replace("\"step\": 1,", "\"step\": \"01\",");
        assert_eq!(Manifest::from_json(&old).unwrap(), m);
        assert!(Manifest::from_json("{\"snapshots\": 3}").is_err());
        let frac = "{\"snapshots\": [{\"step\": 1.5, \"points\": 1, \"preview_new_area\": 0}], \"align\": []}";
        assert!(Manifest::from_json(frac).is_err());
    }

    /// 정수 step 사이 감소만 위반(같은 수는 허용), final 은 판정에서 빠진다. verify 와 같은 규칙.
    #[test]
    fn snapshot_monotone_rule_integer_steps_only() {
        let e = |step, points| SnapshotEntry {
            step,
            points,
            preview_new_area: 1,
        };
        let sum = |entries| SnapshotSummary {
            entries,
            nan_steps: vec![],
            peak_points: 0,
        };
        // 13068 → 12979 가 final 에서 일어나도 위반 아님, 정수 step 사이 같은 수도 허용.
        let ok = sum(vec![
            e(Step::Index(1), 13068),
            e(Step::Index(2), 13068),
            e(Step::Final, 12979),
        ]);
        assert!(check_snapshots(&ok).is_empty());
        let bad = sum(vec![
            e(Step::Index(1), 13068),
            e(Step::Index(2), 12979),
            e(Step::Final, 20000),
        ]);
        let issues = check_snapshots(&bad);
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(issues[0].contains("13068") && issues[0].contains("12979"));
    }

    /// 초벌 6000점·정밀 600점 두 구역: 점 수 [100, 1100, 200] → 최종에서 줄어들지만 final 은 판정 밖이다.
    #[test]
    fn snapshot_final_decrease_not_reported() {
        let grid = |n: usize, x0: f64| PointCloud {
            points: (0..n)
                .map(|i| {
                    rec(Vector3::new(
                        x0 + (i % 60) as f64 * 0.5,
                        (i / 60) as f64 * 0.5,
                        0.0,
                    ))
                })
                .collect(),
        };
        let refined = vec![grid(600, 0.0), grid(600, 100.0)];
        let prelim = vec![Some(grid(600, 0.0)), Some(grid(6000, 300.0))];
        let summary =
            for_each_snapshot(
                &prelim,
                &refined,
                GHOST_RADIUS_M,
                DECIMATE_EVERY,
                |_| Ok(()),
            )
            .unwrap();
        let pts: Vec<usize> = summary.entries.iter().map(|e| e.points).collect();
        assert_eq!(pts, vec![100, 1100, 200]);
        let issues = check_snapshots(&summary);
        assert!(issues.is_empty(), "{issues:?}");
        // NaN 은 따로 보고된다.
        let mut bad = grid(600, 0.0);
        bad.points[0].xyz[2] = f32::NAN;
        let summary = for_each_snapshot(
            &[Some(grid(60, 0.0)), Some(grid(600, 300.0))],
            &[bad, grid(600, 100.0)],
            GHOST_RADIUS_M,
            DECIMATE_EVERY,
            |_| Ok(()),
        )
        .unwrap();
        assert!(check_snapshots(&summary).iter().any(|s| s.contains("NaN")));
    }

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("stream_{tag}_{}", std::process::id()))
    }

    #[test]
    fn write_outputs_files() {
        let n_pos = 26;
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        assert_eq!(regions.len(), 2);
        let truth = truth_points(n_pos);
        let refined: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.0))
            .collect();
        let prelim: Vec<Option<PointCloud>> = regions
            .iter()
            .map(|r| Some(region_cloud(r, &truth, 0.4)))
            .collect();
        let dir = unique_dir("out");
        let rep = write_outputs(&dir, &regions, &prelim, &refined, vec![]).unwrap();
        assert!(rep.issues.is_empty(), "{:?}", rep.issues);
        let m = &rep.manifest;
        for f in [
            "preview/preview_00_pos0-14.ply",
            "preview/preview_01_pos10-26.ply",
            "refined/refined_01_pos10-26.ply",
            "snapshots/step_01_1regions.ply",
            "snapshots/step_02_2regions.ply",
            "snapshots/step_final_all_refined.ply",
        ] {
            assert!(dir.join(f).exists(), "{f}");
        }
        assert!(!dir.join("snapshots/step_03_3regions.ply").exists());
        let text = std::fs::read_to_string(dir.join("snapshots/manifest.json")).unwrap();
        assert_eq!(&Manifest::from_json(&text).unwrap(), m);
        let c = crate::ply::read_ply_file(dir.join("snapshots/step_02_2regions.ply")).unwrap();
        assert_eq!(c.len(), m.snapshots[1].points);
        assert!(m.snapshots[1].preview_new_area > 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// 쓰기 쪽 출력 폴더를 출력 검증으로 읽는다: step 은 정수·"final" 이고 스냅샷 항목이 PASS.
    #[test]
    fn write_outputs_passes_verify_snapshots() {
        let n_pos = 26;
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        let truth = truth_points(n_pos);
        let refined: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.0))
            .collect();
        let prelim: Vec<Option<PointCloud>> = regions
            .iter()
            .map(|r| Some(region_cloud(r, &truth, 0.4)))
            .collect();
        let dir = unique_dir("verify");
        write_outputs(&dir, &regions, &prelim, &refined, vec![]).unwrap();
        let text = std::fs::read_to_string(dir.join("snapshots/manifest.json")).unwrap();
        assert!(
            text.contains("\"step\": 1") || text.contains("\"step\":1"),
            "{text}"
        );
        assert!(text.contains("\"final\""), "{text}");
        assert!(!text.contains("\"01\""), "{text}");
        let rep = crate::verify::verify_dir(&dir);
        let it = rep
            .item(crate::verify::ITEM_SNAPSHOTS)
            .expect("snapshots 항목");
        assert!(it.decided && it.pass, "{it:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// 정렬 기록 점쌍 하한·스케일 차, 구역 0개가 위반으로 보고된다.
    #[test]
    fn write_outputs_reports_align_limits_and_empty() {
        let n_pos = 26;
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        let truth = truth_points(n_pos);
        let refined: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.0))
            .collect();
        let prelim: Vec<Option<PointCloud>> = regions
            .iter()
            .map(|r| Some(region_cloud(r, &truth, 0.4)))
            .collect();
        let dir = unique_dir("limits");
        let align = vec![
            AlignRecord {
                region: 0,
                pairs: 200,
                fit_median_m: Some(0.1),
                scale: Some(0.5),
            },
            AlignRecord {
                region: 1,
                pairs: 200,
                fit_median_m: Some(0.1),
                scale: Some(1.0),
            },
        ];
        let rep = write_outputs(&dir, &regions, &prelim, &refined, align).unwrap();
        assert_eq!(
            rep.issues.iter().filter(|s| s.contains("점쌍")).count(),
            2,
            "{:?}",
            rep.issues
        );
        assert!(
            rep.issues.iter().any(|s| s.contains("스케일 차")),
            "{:?}",
            rep.issues
        );
        // 정상 기록(점쌍 1000, 스케일 차 5 %)은 위반 없음.
        let ok = vec![
            AlignRecord {
                region: 0,
                pairs: 1000,
                fit_median_m: Some(0.1),
                scale: Some(1.0),
            },
            AlignRecord {
                region: 1,
                pairs: 5000,
                fit_median_m: Some(0.1),
                scale: Some(1.05),
            },
        ];
        let rep = write_outputs(&dir, &regions, &prelim, &refined, ok).unwrap();
        assert!(rep.issues.is_empty(), "{:?}", rep.issues);
        std::fs::remove_dir_all(&dir).unwrap();
        let dir = unique_dir("empty");
        let rep = write_outputs(&dir, &[], &[], &[], vec![]).unwrap();
        assert!(
            rep.issues.iter().any(|s| s.contains("구역 0개")),
            "{:?}",
            rep.issues
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// 촘촘한 점군(칸마다 점 수십 개)에서도 반경 검사가 전수 비교와 같다.
    #[test]
    fn ghost_filter_exact_dense() {
        let mut rng = Rng(9);
        let mut mk = |n: usize, s: f64| PointCloud {
            points: (0..n)
                .map(|_| {
                    rec(Vector3::new(
                        rng.next() * s,
                        rng.next() * s,
                        rng.next() * 1.0,
                    ))
                })
                .collect(),
        };
        // 4 m × 4 m 판에 3000 점(반경 칸에 수십 점) + 그 둘레 7 m 상자에 질의 3000 점.
        let base = mk(3000, 2.0);
        let probe = mk(3000, 7.0);
        let mut idx = RadiusIndex::new(GHOST_RADIUS_M);
        idx.insert_cloud(&base);
        let fast = remove_ghosts(&probe, &idx);
        let brute: Vec<PointRecord> = probe
            .points
            .iter()
            .filter(|p| {
                !base.points.iter().any(|q| {
                    let d: f64 = (0..3)
                        .map(|i| (p.xyz[i] as f64 - q.xyz[i] as f64).powi(2))
                        .sum();
                    d <= GHOST_RADIUS_M * GHOST_RADIUS_M
                })
            })
            .copied()
            .collect();
        assert_eq!(fast.points, brute);
        assert!(!fast.is_empty() && fast.len() < probe.len());
    }

    /// 둘째 구역 짝 0: 패닉 없이 모든 파일, manifest 에 null, 위반 보고.
    #[test]
    fn write_outputs_with_failed_alignment() {
        let n_pos = 40;
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        assert_eq!(regions.len(), 4);
        let truth = truth_points(n_pos);
        let refined: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.0))
            .collect();
        let raw: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.4))
            .collect();
        let mut sims = Vec::new();
        let mut align = Vec::new();
        for r in &regions {
            let pairs: Vec<(Vector3<f64>, Vector3<f64>)> = if r.index == 1 {
                Vec::new()
            } else {
                let mut rng = Rng(r.index as u64 + 1);
                (0..200)
                    .map(|_| {
                        let p = Vector3::new(rng.next() * 9.0, rng.next() * 9.0, rng.next());
                        (p, p)
                    })
                    .collect()
            };
            let (s, a) = align_region(r, &pairs);
            sims.push(s);
            align.push(a);
        }
        let prelim = apply_alignments(&raw, &sims);
        assert!(prelim[1].is_none() && prelim[0].is_some());
        let dir = unique_dir("fail");
        let rep = write_outputs(&dir, &regions, &prelim, &refined, align).unwrap();
        for r in &regions {
            assert!(dir.join(preview_name(r)).exists());
            assert!(dir.join(refined_name(r)).exists());
        }
        assert_eq!(rep.manifest.snapshots.len(), regions.len() + 1);
        for e in &rep.manifest.snapshots {
            assert!(dir.join(snapshot_name(e.step)).exists());
        }
        // 실패 구역의 preview 는 비어 있고, step_02 는 정밀 0, 1 로 채워진다.
        let pv = crate::ply::read_ply_file(dir.join(preview_name(&regions[1]))).unwrap();
        assert!(pv.is_empty());
        let s2 = &rep.manifest.snapshots[1];
        assert_eq!(s2.preview_new_area, 0);
        let want =
            refined[0].len().div_ceil(DECIMATE_EVERY) + refined[1].len().div_ceil(DECIMATE_EVERY);
        assert_eq!(s2.points, want);
        let text = std::fs::read_to_string(dir.join("snapshots/manifest.json")).unwrap();
        assert!(
            text.contains("\"region\": 1, \"pairs\": 0, \"fit_median_m\": null, \"scale\": null"),
            "{text}"
        );
        assert!(rep
            .issues
            .iter()
            .any(|s| s.contains("구역 1: 초벌 정렬 실패")));
        assert!(rep
            .issues
            .iter()
            .any(|s| s.contains("step 2: 초벌 새 영역 0")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_outputs_rejects_length_mismatch() {
        let regions = split_regions(26, 12, 2);
        let refined = vec![PointCloud { points: vec![] }; 2];
        let res = write_outputs(unique_dir("len"), &regions, &[None], &refined, vec![]);
        assert!(res.is_err());
    }

    /// 점 수에 따른 스냅샷 시간: 구역 3 × 10만/20만/40만 점(측정용).
    /// `cargo test --release -p skylens-core -- --ignored snapshot_scaling --nocapture`.
    #[test]
    #[ignore = "측정용"]
    fn snapshot_scaling() {
        let mut times = Vec::new();
        for per in [100_000usize, 200_000, 400_000] {
            let mut rng = Rng(78);
            let mut mk = |k: usize, dz: f64| PointCloud {
                points: (0..per)
                    .map(|_| {
                        rec(Vector3::new(
                            k as f64 * 12.0 + 7.0 + rng.next() * 7.0,
                            rng.next() * 15.0,
                            dz + rng.next() * 2.0,
                        ))
                    })
                    .collect(),
            };
            let refined: Vec<PointCloud> = (0..3).map(|k| mk(k, 0.0)).collect();
            let prelim: Vec<Option<PointCloud>> = (0..3).map(|k| Some(mk(k, 0.4))).collect();
            let t = std::time::Instant::now();
            for_each_snapshot(
                &prelim,
                &refined,
                GHOST_RADIUS_M,
                DECIMATE_EVERY,
                |_| Ok(()),
            )
            .unwrap();
            let secs = t.elapsed().as_secs_f64();
            eprintln!("구역 3 × {per} 점: {secs:.3} s");
            times.push(secs);
        }
        // 선형이면 4배, 제곱이면 16배. 측정 흔들림을 넉넉히 보아 8배 미만.
        assert!(times[2] / times[0] < 8.0, "{times:?}");
    }

    /// 큰 점군 속도: 구역 7 × 초벌·정밀 각 200만 점.
    /// `cargo test --release -p skylens-core -- --ignored large_snapshot_speed --nocapture`.
    #[test]
    #[ignore = "측정용(메모리 수백 MB)"]
    fn large_snapshot_speed() {
        let n_reg = 7;
        let per = 2_000_000usize;
        let mut rng = Rng(77);
        let mut mk = |k: usize, dz: f64| PointCloud {
            points: (0..per)
                .map(|_| {
                    rec(Vector3::new(
                        k as f64 * 12.0 + 7.0 + rng.next() * 7.0,
                        rng.next() * 15.0,
                        dz + rng.next() * 2.0,
                    ))
                })
                .collect(),
        };
        let refined: Vec<PointCloud> = (0..n_reg).map(|k| mk(k, 0.0)).collect();
        let prelim: Vec<Option<PointCloud>> = (0..n_reg).map(|k| Some(mk(k, 0.4))).collect();
        let t = std::time::Instant::now();
        let summary =
            for_each_snapshot(
                &prelim,
                &refined,
                GHOST_RADIUS_M,
                DECIMATE_EVERY,
                |_| Ok(()),
            )
            .unwrap();
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "구역 {n_reg} × {per} 점: {secs:.2} s, 최대 보관 {} 점, 단계 {:?}",
            summary.peak_points,
            summary.entries.iter().map(|e| e.points).collect::<Vec<_>>()
        );
        assert!(secs < 10.0, "{secs} s");
    }
}
