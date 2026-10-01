//! 점진적 점군 스트림: 구역 분할, 초벌 모델의 3D 점 대응 정렬, 단계별 스냅샷.
//!
//! - 구역 `i` 는 위치 `[start-OVL, start+SPAN+OVL)` (start = i·SPAN), 전체 위치 범위로 자른다.
//! - 초벌 정렬은 같은 이미지의 같은 특징점 번호를 관측한 초벌·정밀 3D 점 짝으로 닮음 변환을
//!   추정하고 반복 트리밍(임계 = max(3 × 잔차 중앙값, 바닥값))한다. 점에는 변환, 법선에는 회전만.
//! - 스냅샷 `step_k` = 정밀 구역 `0..k-1` + 초벌 구역 `k-1` (초벌 점 중 이미 들어간 정밀 점에서
//!   반경 안에 있는 점은 뺀다). 최종 = 정밀 전부. 모든 점군은 간격 추출한다.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::Path;

use nalgebra::{Matrix3, Rotation3, Vector3};

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

/// 위치 `n_positions` 곳을 구역으로 나눈다. start = 0, SPAN, 2·SPAN, … (start < n).
pub fn split_regions(n_positions: usize, span: usize, ovl: usize) -> Vec<Region> {
    assert!(span > 0, "SPAN 은 1 이상");
    (0..n_positions)
        .step_by(span)
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

/// 닮음 변환 `x ↦ s·R·x + t`.
// 병합 예정인 align 모듈과 같은 모양의 임시 구현.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub s: f64,
    pub r: Rotation3<f64>,
    pub t: Vector3<f64>,
}

impl Similarity {
    pub fn identity() -> Self {
        Self {
            s: 1.0,
            r: Rotation3::identity(),
            t: Vector3::zeros(),
        }
    }

    pub fn apply(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.r * p * self.s + self.t
    }

    /// 점군에 적용: 점에는 변환 전체, 법선에는 회전만.
    pub fn apply_cloud(&self, cloud: &PointCloud) -> PointCloud {
        let points = cloud
            .points
            .iter()
            .map(|p| {
                let x = Vector3::new(p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64);
                let n = Vector3::new(p.normal[0] as f64, p.normal[1] as f64, p.normal[2] as f64);
                let y = self.apply(&x);
                let m = self.r * n;
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

/// 최소제곱 닮음 변환(Umeyama 1991). `dst ≈ s·R·src + t`. 점이 3개 미만이거나 퇴화하면 `None`.
pub fn umeyama(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Option<Similarity> {
    let n = src.len();
    if n < 3 || n != dst.len() {
        return None;
    }
    let nf = n as f64;
    let mu_s = src.iter().sum::<Vector3<f64>>() / nf;
    let mu_d = dst.iter().sum::<Vector3<f64>>() / nf;
    let mut cov = Matrix3::zeros();
    let mut var_s = 0.0;
    for (a, b) in src.iter().zip(dst) {
        let da = a - mu_s;
        let db = b - mu_d;
        cov += db * da.transpose();
        var_s += da.norm_squared();
    }
    cov /= nf;
    var_s /= nf;
    if var_s <= 1e-12 {
        return None;
    }
    let svd = cov.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let mut d = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = u * d * vt;
    let sv = svd.singular_values;
    let trace = sv[0] * d[(0, 0)] + sv[1] * d[(1, 1)] + sv[2] * d[(2, 2)];
    let s = trace / var_s;
    if !s.is_finite() || s <= 0.0 {
        return None;
    }
    let t = mu_d - r * mu_s * s;
    Some(Similarity {
        s,
        r: Rotation3::from_matrix_unchecked(r),
        t,
    })
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// 반복 트리밍 닮음 변환. 매 회: 남은 짝으로 추정 → 전체 짝 잔차의 중앙값 m →
/// 임계 max(3m, floor_m) 미만인 짝만 남긴다. 마지막으로 남은 짝으로 다시 추정한다.
/// 반환: (변환, 남은 짝 표시, 남은 짝 잔차 중앙값).
pub fn robust_similarity(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    iters: usize,
    floor_m: f64,
) -> Option<(Similarity, Vec<bool>, f64)> {
    if src.len() != dst.len() {
        return None;
    }
    let mut mask = vec![true; src.len()];
    let fit = |mask: &[bool]| {
        let (a, b): (Vec<_>, Vec<_>) = src
            .iter()
            .zip(dst)
            .zip(mask)
            .filter(|(_, &m)| m)
            .map(|((a, b), _)| (*a, *b))
            .unzip();
        umeyama(&a, &b)
    };
    let mut sim = fit(&mask)?;
    for _ in 0..iters {
        let res: Vec<f64> = src
            .iter()
            .zip(dst)
            .map(|(a, b)| (sim.apply(a) - b).norm())
            .collect();
        let thr = (3.0 * median(&mut res.clone())).max(floor_m);
        let next: Vec<bool> = res.iter().map(|&r| r < thr).collect();
        if next.iter().filter(|&&m| m).count() < 3 {
            break;
        }
        mask = next;
        sim = fit(&mask)?;
    }
    let mut kept: Vec<f64> = src
        .iter()
        .zip(dst)
        .zip(&mask)
        .filter(|(_, &m)| m)
        .map(|((a, b), _)| (sim.apply(a) - b).norm())
        .collect();
    let med = median(&mut kept);
    Some((sim, mask, med))
}

/// 구역 하나의 정렬 기록.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignRecord {
    pub region: usize,
    pub pairs: usize,
    pub fit_median_m: f64,
    pub scale: f64,
}

/// 초벌 구역 하나를 정밀 좌표로 정렬한다. 짝이 모자라면 `None`.
pub fn align_region(
    region: &Region,
    pairs: &[(Vector3<f64>, Vector3<f64>)],
) -> Option<(Similarity, AlignRecord)> {
    let (src, dst): (Vec<_>, Vec<_>) = pairs.iter().copied().unzip();
    let (sim, _mask, med) = robust_similarity(&src, &dst, TRIM_ITERS, TRIM_FLOOR_M)?;
    Some((
        sim,
        AlignRecord {
            region: region.index,
            pairs: pairs.len(),
            fit_median_m: med,
            scale: sim.s,
        },
    ))
}

/// 격자 해시 최근접 검사기(반경 고정).
pub struct RadiusIndex {
    cell: f64,
    grid: HashMap<(i64, i64, i64), Vec<[f64; 3]>>,
}

impl RadiusIndex {
    pub fn new(radius: f64) -> Self {
        assert!(radius > 0.0);
        Self {
            cell: radius,
            grid: HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 3]) -> (i64, i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
            (p[2] / self.cell).floor() as i64,
        )
    }

    pub fn insert_cloud(&mut self, cloud: &PointCloud) {
        for p in &cloud.points {
            let q = [p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64];
            let k = self.key(q);
            self.grid.entry(k).or_default().push(q);
        }
    }

    /// 반경 안(거리 ≤ radius)에 점이 있는가.
    pub fn has_within(&self, p: [f64; 3]) -> bool {
        let r2 = self.cell * self.cell;
        let (kx, ky, kz) = self.key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.grid.get(&(kx + dx, ky + dy, kz + dz)) {
                        if v.iter().any(|q| {
                            let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r2
                        }) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

/// 초벌 점 중 `index` 의 점에서 반경 안에 있는 점을 뺀다.
pub fn remove_ghosts(prelim: &PointCloud, index: &RadiusIndex) -> PointCloud {
    PointCloud {
        points: prelim
            .points
            .iter()
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

/// 스냅샷 하나. `step` 은 들어간 구역 수(1부터), 최종은 `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub step: Option<usize>,
    pub cloud: PointCloud,
    pub preview_new_area: usize,
}

/// 스냅샷 만들기. `prelim_aligned[i]`, `refined[i]` 는 구역 i 의 (정렬된) 초벌·정밀 점군(추출 전).
/// step_k (k = 1..=n) = 정밀 0..k-2 + 초벌 k-1 (정밀 점 반경 `ghost_radius` 안 초벌 점 제외).
/// 마지막은 정밀 전부. 각 구성 점군은 `every`:1 간격 추출한다.
/// `preview_new_area` 는 그 단계에 들어간 초벌 점 수(추출 뒤).
pub fn build_snapshots(
    prelim_aligned: &[PointCloud],
    refined: &[PointCloud],
    ghost_radius: f64,
    every: usize,
) -> Vec<Snapshot> {
    assert_eq!(prelim_aligned.len(), refined.len());
    let n = refined.len();
    let refined_dec: Vec<PointCloud> = refined.iter().map(|c| decimate(c, every)).collect();
    let mut out = Vec::with_capacity(n + 1);
    let mut index = RadiusIndex::new(ghost_radius);
    let mut base: Vec<PointRecord> = Vec::new();
    for k in 1..=n {
        if k >= 2 {
            index.insert_cloud(&refined[k - 2]);
            base.extend_from_slice(&refined_dec[k - 2].points);
        }
        let fresh = decimate(&remove_ghosts(&prelim_aligned[k - 1], &index), every);
        let mut points = base.clone();
        points.extend_from_slice(&fresh.points);
        out.push(Snapshot {
            step: Some(k),
            cloud: PointCloud { points },
            preview_new_area: fresh.len(),
        });
    }
    let mut all = Vec::new();
    for c in &refined_dec {
        all.extend_from_slice(&c.points);
    }
    out.push(Snapshot {
        step: None,
        cloud: PointCloud { points: all },
        preview_new_area: 0,
    });
    out
}

/// 매니페스트의 스냅샷 줄.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotEntry {
    /// 단계 이름: "01", "02", …, 최종은 "final".
    pub step: String,
    pub points: usize,
    pub preview_new_area: usize,
}

/// `snapshots/manifest.json` 내용.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Manifest {
    pub snapshots: Vec<SnapshotEntry>,
    pub align: Vec<AlignRecord>,
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_f64(x: f64) -> String {
    if x.is_finite() {
        // `{:?}` 는 왕복 가능한 최단 표기.
        format!("{x:?}")
    } else {
        "null".to_string()
    }
}

impl Manifest {
    pub fn from_snapshots(snaps: &[Snapshot], align: Vec<AlignRecord>) -> Self {
        Self {
            snapshots: snaps
                .iter()
                .map(|s| SnapshotEntry {
                    step: match s.step {
                        Some(k) => format!("{k:02}"),
                        None => "final".to_string(),
                    },
                    points: s.cloud.len(),
                    preview_new_area: s.preview_new_area,
                })
                .collect(),
            align,
        }
    }

    pub fn to_json(&self) -> String {
        let snaps: Vec<String> = self
            .snapshots
            .iter()
            .map(|s| {
                format!(
                    "    {{\"step\": {}, \"points\": {}, \"preview_new_area\": {}}}",
                    json_str(&s.step),
                    s.points,
                    s.preview_new_area
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

    pub fn from_json(text: &str) -> Result<Self, String> {
        let v = Json::parse(text)?;
        let arr = |key: &str| -> Result<Vec<Json>, String> {
            match v.get(key) {
                Some(Json::Arr(a)) => Ok(a.clone()),
                _ => Err(format!("{key} 배열 없음")),
            }
        };
        let num = |o: &Json, key: &str| -> Result<f64, String> {
            match o.get(key) {
                Some(Json::Num(x)) => Ok(*x),
                Some(Json::Null) => Ok(f64::NAN),
                _ => Err(format!("{key} 숫자 없음")),
            }
        };
        let mut m = Manifest::default();
        for o in arr("snapshots")? {
            let step = match o.get("step") {
                Some(Json::Str(s)) => s.clone(),
                Some(Json::Num(x)) => format!("{:02}", *x as i64),
                _ => return Err("step 없음".into()),
            };
            m.snapshots.push(SnapshotEntry {
                step,
                points: num(&o, "points")? as usize,
                preview_new_area: num(&o, "preview_new_area")? as usize,
            });
        }
        for o in arr("align")? {
            m.align.push(AlignRecord {
                region: num(&o, "region")? as usize,
                pairs: num(&o, "pairs")? as usize,
                fit_median_m: num(&o, "fit_median_m")?,
                scale: num(&o, "scale")?,
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
pub fn snapshot_name(s: &Snapshot) -> String {
    match s.step {
        Some(k) => format!("snapshots/step_{k:02}_{k}regions.ply"),
        None => "snapshots/step_final_all_refined.ply".to_string(),
    }
}

/// 출력 전부를 `out_dir` 아래에 쓴다. 구역 점군도 간격 추출해 쓴다. 매니페스트를 돌려준다.
pub fn write_outputs(
    out_dir: impl AsRef<Path>,
    regions: &[Region],
    prelim_aligned: &[PointCloud],
    refined: &[PointCloud],
    align: Vec<AlignRecord>,
) -> io::Result<Manifest> {
    let dir = out_dir.as_ref();
    for sub in ["preview", "refined", "snapshots"] {
        std::fs::create_dir_all(dir.join(sub))?;
    }
    for ((r, p), q) in regions.iter().zip(prelim_aligned).zip(refined) {
        write_ply_file(dir.join(preview_name(r)), &decimate(p, DECIMATE_EVERY))?;
        write_ply_file(dir.join(refined_name(r)), &decimate(q, DECIMATE_EVERY))?;
    }
    let snaps = build_snapshots(prelim_aligned, refined, GHOST_RADIUS_M, DECIMATE_EVERY);
    for s in &snaps {
        write_ply_file(dir.join(snapshot_name(s)), &s.cloud)?;
    }
    let m = Manifest::from_snapshots(&snaps, align);
    std::fs::write(dir.join("snapshots/manifest.json"), m.to_json())?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Unit;

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
    }

    fn known_sim() -> Similarity {
        Similarity {
            s: 0.37,
            r: Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(0.3, -0.5, 0.8)), 1.1),
            t: Vector3::new(12.0, -4.0, 30.0),
        }
    }

    #[test]
    fn umeyama_exact_recovery() {
        let mut rng = Rng(7);
        let src: Vec<Vector3<f64>> = (0..50)
            .map(|_| Vector3::new(rng.next() * 20.0, rng.next() * 20.0, rng.next() * 5.0))
            .collect();
        let g = known_sim();
        let dst: Vec<_> = src.iter().map(|p| g.apply(p)).collect();
        let e = umeyama(&src, &dst).unwrap();
        assert!((e.s - g.s).abs() < 1e-10);
        assert!((e.r.matrix() - g.r.matrix()).norm() < 1e-10);
        assert!((e.t - g.t).norm() < 1e-9);
        assert!(umeyama(&src[..2], &dst[..2]).is_none());
    }

    #[test]
    fn robust_similarity_rejects_outliers() {
        let mut rng = Rng(11);
        let g = known_sim();
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..400 {
            let p = Vector3::new(rng.next() * 30.0, rng.next() * 30.0, rng.next() * 3.0);
            let mut q = g.apply(&p) + Vector3::new(rng.next(), rng.next(), rng.next()) * 0.02;
            if i % 5 == 0 {
                // 20 % 큰 오대응(5~15 m).
                q += Vector3::new(rng.next(), rng.next(), rng.next()).normalize() * 10.0
                    + Vector3::new(5.0 * rng.next().signum(), 0.0, 0.0);
            }
            src.push(p);
            dst.push(q);
        }
        let (e, mask, med) = robust_similarity(&src, &dst, TRIM_ITERS, TRIM_FLOOR_M).unwrap();
        // 오대응은 모두 빠지고, 정상 짝은 거의 다 남아야 한다.
        for (i, &m) in mask.iter().enumerate() {
            if i % 5 == 0 {
                assert!(!m, "오대응 {i} 남음");
            }
        }
        let kept = mask.iter().filter(|&&m| m).count();
        assert_eq!(kept, 320);
        assert!((e.s / g.s - 1.0).abs() < 1e-3);
        // 잡음 성분 ±0.02 m 균일 → 3차원 잔차 중앙 ≈ 0.02 m 수준.
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

    /// 구역마다 초벌 = 정답에 휨(구역 시작에서 멀수록 z 로 c·d²) + 잡음을 넣고 알려진 닮음 변환의 역을 적용.
    /// 정밀 = 정답 + 작은 잡음. 관측은 (위치·3, 점 번호) 하나.
    #[allow(clippy::type_complexity)]
    fn synth_region(
        region: &Region,
        truth: &[(usize, Vector3<f64>)],
        g: &Similarity,
        bend: f64,
        rng: &mut Rng,
    ) -> (Vec<Track>, Vec<Track>, Vec<Vector3<f64>>) {
        let ginv_r = g.r.inverse();
        let mut pre = Vec::new();
        let mut fine = Vec::new();
        let mut truth_in = Vec::new();
        for (id, (pos, p)) in truth.iter().enumerate() {
            if !region.contains(*pos) {
                continue;
            }
            let obs = vec![(*pos as u32 * 3, id as u32)];
            let d = p.x - region.start as f64;
            let warped = p
                + Vector3::new(0.0, 0.0, bend * d * d)
                + Vector3::new(rng.next(), rng.next(), rng.next()) * 0.1;
            let pre_xyz = ginv_r * (warped - g.t) / g.s;
            pre.push(Track {
                xyz: pre_xyz,
                obs: obs.clone(),
            });
            fine.push(Track {
                xyz: p + Vector3::new(rng.next(), rng.next(), rng.next()) * 0.02,
                obs,
            });
            truth_in.push(*p);
        }
        (pre, fine, truth_in)
    }

    #[test]
    fn prelim_alignment_against_truth() {
        let n_pos = 40;
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        let truth = truth_points(n_pos);
        let mut rng = Rng(99);
        let image_pos = |img: u32| (img / 3) as usize;
        for (k, r) in regions.iter().enumerate() {
            let g = Similarity {
                s: 0.5 + 0.1 * k as f64,
                r: Rotation3::from_euler_angles(0.1 * k as f64, -0.2, 0.7 + 0.3 * k as f64),
                t: Vector3::new(3.0 * k as f64, -7.0, 40.0),
            };
            // 휨 0.002 /m: 구역 끝(start+14 m)에서 0.39 m.
            let (pre, fine, truth_in) = synth_region(r, &truth, &g, 0.002, &mut rng);
            let win = align_window(r, DEFAULT_OVL, n_pos);
            let pairs = point_pairs(&pre, &fine, image_pos, win);
            // 창 안 위치 수 × 4줄 × 60점.
            let expect_pairs = (win.1 - win.0) * 4 * 60;
            assert_eq!(pairs.len(), expect_pairs);
            let (sim, rec) = align_region(r, &pairs).unwrap();
            assert_eq!(rec.region, k);
            // 휨이 창 안에서 작으므로 스케일은 1 % 안에서 복원되어야 한다(SPEC 허용 ±10 %).
            assert!((rec.scale / g.s - 1.0).abs() < 0.01, "scale");
            // 정렬 뒤 초벌 전체를 정답과 비교.
            let mut res: Vec<f64> = pre
                .iter()
                .zip(&truth_in)
                .map(|(a, b)| (sim.apply(&a.xyz) - b).norm())
                .collect();
            let med = median(&mut res);
            // 기준: 휨 최대 0.39 m + 잡음 0.1 m. 창 안 정렬 오차를 더해도 중앙 0.5 m 미만이어야 한다.
            eprintln!(
                "구역 {k}: 짝 {} fit {:.4} 정답 잔차 중앙 {med:.4} 스케일비 {:.5}",
                rec.pairs,
                rec.fit_median_m,
                rec.scale / g.s
            );
            assert!(med < 0.5, "구역 {k} 잔차 중앙 {med}");
            assert!(rec.fit_median_m < 0.3, "fit {}", rec.fit_median_m);
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
        let o = g.apply_cloud(&c);
        let n = Vector3::new(
            o.points[0].normal[0],
            o.points[0].normal[1],
            o.points[0].normal[2],
        );
        assert!((n.norm() - 1.0).abs() < 1e-6);
        let want = g.r * Vector3::z();
        assert!((n.cast::<f64>() - want).norm() < 1e-6);
        let p = g.apply(&Vector3::new(1.0, 2.0, 3.0));
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
                                   // 무작위 점들: 전수 비교와 같아야 한다.
        let mut rng = Rng(5);
        let base = PointCloud {
            points: (0..500)
                .map(|_| {
                    rec(Vector3::new(
                        rng.next() * 20.0,
                        rng.next() * 20.0,
                        rng.next() * 2.0,
                    ))
                })
                .collect(),
        };
        let probe = PointCloud {
            points: (0..2000)
                .map(|_| {
                    rec(Vector3::new(
                        rng.next() * 25.0,
                        rng.next() * 25.0,
                        rng.next() * 3.0,
                    ))
                })
                .collect(),
        };
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
        let prelim: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.4))
            .collect();
        let snaps = build_snapshots(&prelim, &refined, GHOST_RADIUS_M, DECIMATE_EVERY);
        assert_eq!(snaps.len(), regions.len() + 1);
        for w in snaps.windows(2) {
            assert!(w[1].cloud.len() > w[0].cloud.len(), "단조 증가 아님");
        }
        for s in &snaps {
            assert!(!s.cloud.has_nan());
        }
        for s in &snaps[1..regions.len()] {
            assert!(s.preview_new_area > 0);
        }
        // 손 계산: 구역 1 초벌(위치 10..26)에서 정밀 0(위치 0..14) 의 1.5 m 이내 점은 x ≤ 13.75+1.5 에 있음.
        // 초벌 점 x 는 0.25 간격, 위치 10..26 → x ∈ [10, 26). 남는 것 x > 15.25 (0.4 m 위라 수평 √(1.5²-0.4²)=1.446 m):
        // 정밀 0 의 마지막 x = 13.75, 남는 조건 x - 13.75 > 1.446 → x ≥ 15.25 → 15.25..25.75 = 43줄 × 60.
        let kept_pre: usize = 43 * 60;
        let expect_new = kept_pre.div_ceil(DECIMATE_EVERY);
        assert_eq!(snaps[1].preview_new_area, expect_new);
        let expect_ref0 = refined[0].len().div_ceil(DECIMATE_EVERY);
        assert_eq!(snaps[1].cloud.len(), expect_ref0 + expect_new);
        // 최종 = 정밀 전부.
        let total: usize = refined
            .iter()
            .map(|c| c.len().div_ceil(DECIMATE_EVERY))
            .sum();
        assert_eq!(snaps.last().unwrap().cloud.len(), total);

        let align = vec![
            AlignRecord {
                region: 0,
                pairs: 3360,
                fit_median_m: 0.123456789,
                scale: 1.0,
            },
            AlignRecord {
                region: 1,
                pairs: 960,
                fit_median_m: 0.2,
                scale: 0.987654321,
            },
        ];
        let m = Manifest::from_snapshots(&snaps, align);
        assert_eq!(m.snapshots[0].step, "01");
        assert_eq!(m.snapshots.last().unwrap().step, "final");
        let back = Manifest::from_json(&m.to_json()).unwrap();
        assert_eq!(back, m);
        assert!(Manifest::from_json("{\"snapshots\": 3}").is_err());
    }

    #[test]
    fn write_outputs_files() {
        let n_pos = 26;
        let regions = split_regions(n_pos, DEFAULT_SPAN, DEFAULT_OVL);
        let truth = truth_points(n_pos);
        let refined: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.0))
            .collect();
        let prelim: Vec<PointCloud> = regions
            .iter()
            .map(|r| region_cloud(r, &truth, 0.4))
            .collect();
        let dir = std::env::temp_dir().join(format!("stream_out_{}", std::process::id()));
        let m = write_outputs(&dir, &regions, &prelim, &refined, vec![]).unwrap();
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
        let text = std::fs::read_to_string(dir.join("snapshots/manifest.json")).unwrap();
        assert_eq!(Manifest::from_json(&text).unwrap(), m);
        let c = crate::ply::read_ply_file(dir.join("snapshots/step_02_2regions.ply")).unwrap();
        assert_eq!(c.len(), m.snapshots[1].points);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
