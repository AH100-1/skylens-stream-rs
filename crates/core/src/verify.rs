//! 출력 폴더 검증 (SPEC §2 출력 구조, §4 검증 기준 일곱 항목).
//!
//! 읽는 파일:
//! - `report.json` (출력 폴더 바로 아래): 포즈 단계 기록.
//!   ```json
//!   {"registered": {"total": 240, "preview": 240, "refined": 240},
//!    "reprojection_px": {"preview": 4.5, "refined": 0.59},
//!    "regions": [{"region": 0, "positions": 14, "images": 42}]}
//!   ```
//!   `registered.total` 은 입력 사진 수, `preview`/`refined` 는 초벌·정밀 포즈에 등록된 사진 수.
//!   `reprojection_px` 는 재투영 오차 RMS(px). `regions[].positions` 는 구역의 위치 수,
//!   `images` 는 구역 밀집 단계에 쓴 사진 수.
//! - `snapshots/manifest.json`: SPEC §2 형식.
//!   `snapshots[].step` 은 1부터 세는 정수(최종은 문자열 `"final"` 도 허용).
//! - `preview/preview_{k:02}_*.ply`, `refined/refined_{k:02}_*.ply`, `snapshots/step_*.ply`.
//!
//! 최근접 탐색은 격자 해시(셀 크기 고정, 바깥 껍질로 넓혀 가며 찾음)로 한다.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use crate::ply::{read_ply_file, PointCloud};

// ---------------------------------------------------------------- 기준값

pub const REPROJ_MAX_PX: f64 = 0.7;
pub const ALIGN_MIN_PAIRS: f64 = 1000.0;
pub const ALIGN_SCALE_TOL: f64 = 0.10;
pub const ALIGN_FIT_MAX_M: f64 = 6.0;
pub const NN_MEDIAN_MAX_M: f64 = 3.0;
pub const HEIGHT_PAIR_RADIUS_M: f64 = 2.0;
pub const HEIGHT_MEDIAN_MAX_M: f64 = 2.0;
pub const OVERLAP_PAIR_RADIUS_M: f64 = 1.0;
pub const OVERLAP_MEDIAN_MAX_M: f64 = 0.3;
/// 최근접 거리를 이 값에서 자른다 (중앙값 판정에는 영향 없음: 기준보다 충분히 큼).
pub const NN_CAP_M: f64 = 20.0;
/// 점군당 질의 점 상한 (넘으면 일정 간격으로 추림).
pub const MAX_QUERIES: usize = 200_000;
const EPS: f64 = 1e-9;

// ---------------------------------------------------------------- 결과

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub name: &'static str,
    pub pass: bool,
    pub measured: String,
    pub criterion: &'static str,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub items: Vec<Item>,
}

impl Report {
    pub fn all_pass(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|i| i.pass)
    }

    pub fn item(&self, name: &str) -> Option<&Item> {
        self.items.iter().find(|i| i.name == name)
    }

    /// 마크다운 표.
    pub fn to_table(&self) -> String {
        let mut s = String::from("| 항목 | 결과 | 측정값 | 기준 |\n|---|---|---|---|\n");
        for i in &self.items {
            let r = if i.pass { "PASS" } else { "FAIL" };
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} |",
                i.name, r, i.measured, i.criterion
            );
        }
        let _ = writeln!(
            s,
            "결과: {}/{} 통과",
            self.items.iter().filter(|i| i.pass).count(),
            self.items.len()
        );
        s
    }
}

pub const ITEM_REGISTERED: &str = "registered";
pub const ITEM_REGION_IMAGES: &str = "region_images";
pub const ITEM_REPROJ: &str = "refined_reprojection";
pub const ITEM_ALIGN: &str = "preview_align";
pub const ITEM_PREVIEW_REFINED: &str = "preview_vs_refined";
pub const ITEM_OVERLAP: &str = "refined_overlap";
pub const ITEM_SNAPSHOTS: &str = "snapshots";

fn item(name: &'static str, criterion: &'static str, r: Result<(bool, String), String>) -> Item {
    match r {
        Ok((pass, measured)) => Item {
            name,
            pass,
            measured,
            criterion,
        },
        Err(e) => Item {
            name,
            pass: false,
            measured: format!("오류: {e}"),
            criterion,
        },
    }
}

/// 출력 폴더 전체를 검증한다.
pub fn verify_dir(dir: &Path) -> Report {
    let report = read_json(&dir.join("report.json"));
    let manifest = read_json(&dir.join("snapshots").join("manifest.json"));
    let preview = load_regions(&dir.join("preview"), "preview_");
    let refined = load_regions(&dir.join("refined"), "refined_");

    let items = vec![
        item(
            ITEM_REGISTERED,
            "초벌·정밀 모두 전체 등록 (240/240)",
            report
                .as_ref()
                .map_err(Clone::clone)
                .and_then(check_registered),
        ),
        item(
            ITEM_REGION_IMAGES,
            "구역 사진 수 = 3 × 위치 수",
            report
                .as_ref()
                .map_err(Clone::clone)
                .and_then(check_region_images),
        ),
        item(
            ITEM_REPROJ,
            "정밀 재투영 ≤ 0.7 px",
            report.as_ref().map_err(Clone::clone).and_then(check_reproj),
        ),
        item(
            ITEM_ALIGN,
            "점쌍 ≥ 1000, 스케일 중앙 대비 ±10%, 잔차 중앙 < 6 m",
            manifest
                .as_ref()
                .map_err(Clone::clone)
                .and_then(check_align),
        ),
        item(
            ITEM_PREVIEW_REFINED,
            "최근접 중앙 < 3 m, 수평 2 m 짝 높이 차 중앙 < 2 m",
            match (&preview, &refined) {
                (Ok(p), Ok(r)) => check_preview_vs_refined(p, r),
                (Err(e), _) | (_, Err(e)) => Err(e.clone()),
            },
        ),
        item(
            ITEM_OVERLAP,
            "이웃 정밀 구역 겹침(수평 1 m 짝) 높이 차 중앙 < 0.3 m",
            refined
                .as_ref()
                .map_err(Clone::clone)
                .and_then(check_overlap),
        ),
        item(
            ITEM_SNAPSHOTS,
            "점 수 단조 증가, 2단계부터 초벌 새 영역 > 0, NaN 없음",
            manifest
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|m| check_snapshots(m, &dir.join("snapshots"))),
        ),
    ];
    Report { items }
}

// ---------------------------------------------------------------- 항목별

fn num(j: &Json, path: &[&str]) -> Result<f64, String> {
    let mut cur = j;
    for k in path {
        cur = cur
            .get(k)
            .ok_or_else(|| format!("{} 없음", path.join(".")))?;
    }
    cur.as_f64()
        .ok_or_else(|| format!("{} 가 숫자가 아님", path.join(".")))
}

fn check_registered(r: &Json) -> Result<(bool, String), String> {
    let total = num(r, &["registered", "total"])?;
    let p = num(r, &["registered", "preview"])?;
    let f = num(r, &["registered", "refined"])?;
    Ok((
        total > 0.0 && p == total && f == total,
        format!("초벌 {p}/{total}, 정밀 {f}/{total}"),
    ))
}

fn check_region_images(r: &Json) -> Result<(bool, String), String> {
    let regs = r
        .get("regions")
        .and_then(Json::as_array)
        .ok_or("regions 없음")?;
    if regs.is_empty() {
        return Ok((false, "구역 0개".into()));
    }
    let mut bad = Vec::new();
    for (i, g) in regs.iter().enumerate() {
        let k = g.get("region").and_then(Json::as_f64).unwrap_or(i as f64);
        let pos = num(g, &["positions"])?;
        let img = num(g, &["images"])?;
        if img != 3.0 * pos {
            bad.push(format!("구역 {k}: {img} ≠ 3×{pos}"));
        }
    }
    if bad.is_empty() {
        Ok((true, format!("{}개 구역 모두 일치", regs.len())))
    } else {
        Ok((false, bad.join(", ")))
    }
}

fn check_reproj(r: &Json) -> Result<(bool, String), String> {
    let f = num(r, &["reprojection_px", "refined"])?;
    let mut m = format!("정밀 {f:.3} px");
    if let Some(p) = r
        .get("reprojection_px")
        .and_then(|x| x.get("preview"))
        .and_then(Json::as_f64)
    {
        let _ = write!(m, " (초벌 {p:.3} px)");
    }
    Ok((f.is_finite() && f <= REPROJ_MAX_PX, m))
}

fn check_align(m: &Json) -> Result<(bool, String), String> {
    let arr = m
        .get("align")
        .and_then(Json::as_array)
        .ok_or("align 없음")?;
    if arr.is_empty() {
        return Ok((false, "align 0개".into()));
    }
    let mut pairs = Vec::new();
    let mut fits = Vec::new();
    let mut scales = Vec::new();
    for a in arr {
        pairs.push(num(a, &["pairs"])?);
        fits.push(num(a, &["fit_median_m"])?);
        scales.push(num(a, &["scale"])?);
    }
    let min_pairs = pairs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_fit = fits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let med_scale = median(&mut scales.clone());
    let max_dev = scales
        .iter()
        .map(|s| (s / med_scale - 1.0).abs())
        .fold(0.0, f64::max);
    let pass = min_pairs >= ALIGN_MIN_PAIRS
        && max_fit < ALIGN_FIT_MAX_M
        && med_scale > 0.0
        && max_dev <= ALIGN_SCALE_TOL + EPS;
    Ok((
        pass,
        format!(
            "점쌍 최소 {min_pairs}, 스케일 최대 편차 {:.2}%, 잔차 중앙 최대 {max_fit:.3} m",
            max_dev * 100.0
        ),
    ))
}

fn check_preview_vs_refined(
    p: &BTreeMap<usize, PointCloud>,
    r: &BTreeMap<usize, PointCloud>,
) -> Result<(bool, String), String> {
    if p.is_empty() || r.is_empty() {
        return Ok((false, "구역 점군 없음".into()));
    }
    let mut worst_nn = 0.0f64;
    let mut worst_dz = 0.0f64;
    let mut missing = Vec::new();
    for (k, pc) in p {
        let Some(rc) = r.get(k) else {
            missing.push(*k);
            continue;
        };
        let pts = xyz(pc);
        let refp = xyz(rc);
        let nn = nn_median(&pts, &refp).unwrap_or(f64::INFINITY);
        let dz = height_pair_median(&pts, &refp, HEIGHT_PAIR_RADIUS_M).unwrap_or(f64::INFINITY);
        worst_nn = worst_nn.max(nn);
        worst_dz = worst_dz.max(dz);
    }
    let pass = missing.is_empty() && worst_nn < NN_MEDIAN_MAX_M && worst_dz < HEIGHT_MEDIAN_MAX_M;
    let mut m = format!(
        "{}개 구역, 최근접 중앙 최대 {worst_nn:.3} m, 높이 차 중앙 최대 {worst_dz:.3} m",
        p.len()
    );
    if !missing.is_empty() {
        let _ = write!(m, ", 정밀 없는 구역 {missing:?}");
    }
    Ok((pass, m))
}

fn check_overlap(r: &BTreeMap<usize, PointCloud>) -> Result<(bool, String), String> {
    let keys: Vec<usize> = r.keys().copied().collect();
    if keys.len() < 2 {
        return Ok((
            true,
            format!("정밀 구역 {}개: 비교할 이웃 없음", keys.len()),
        ));
    }
    let mut worst = 0.0f64;
    for w in keys.windows(2) {
        let a = xyz(&r[&w[0]]);
        let b = xyz(&r[&w[1]]);
        let d = height_pair_median(&a, &b, OVERLAP_PAIR_RADIUS_M).unwrap_or(f64::INFINITY);
        worst = worst.max(d);
    }
    Ok((
        worst < OVERLAP_MEDIAN_MAX_M,
        format!("{}쌍, 겹침 차 중앙 최대 {worst:.3} m", keys.len() - 1),
    ))
}

fn check_snapshots(m: &Json, snap_dir: &Path) -> Result<(bool, String), String> {
    let arr = m
        .get("snapshots")
        .and_then(Json::as_array)
        .ok_or("snapshots 없음")?;
    if arr.is_empty() {
        return Ok((false, "스냅샷 0개".into()));
    }
    // (정렬 키, 단계 번호(최종은 None), 점 수, 새 영역)
    let mut rows = Vec::new();
    for s in arr {
        let step = s.get("step").ok_or("step 없음")?;
        let k = match step {
            Json::Num(v) => Some(*v),
            Json::Str(t) if t == "final" => None,
            _ => return Err("step 은 정수 또는 \"final\"".into()),
        };
        let pts = num(s, &["points"])?;
        let area = s.get("preview_new_area").and_then(Json::as_f64);
        rows.push((k.unwrap_or(f64::INFINITY), k, pts, area));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut problems = Vec::new();
    for w in rows.windows(2) {
        if w[1].2 <= w[0].2 {
            problems.push(format!("점 수 감소/정체 {}→{}", w[0].2, w[1].2));
        }
    }
    let mut min_area = f64::INFINITY;
    for (_, k, _, area) in &rows {
        if let Some(k) = k {
            if *k >= 2.0 {
                let a = area.unwrap_or(f64::NAN);
                min_area = min_area.min(if a.is_nan() { f64::NEG_INFINITY } else { a });
                if a.is_nan() || a <= 0.0 {
                    problems.push(format!("단계 {k} 새 영역 {a}"));
                }
            }
        }
    }
    // 스냅샷 PLY: NaN 없음, 파일 점 수 = manifest 점 수.
    let mut files = 0usize;
    let mut nan_files = Vec::new();
    let entries =
        std::fs::read_dir(snap_dir).map_err(|e| format!("{}: {e}", snap_dir.display()))?;
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("step_") && n.ends_with(".ply"))
        .collect();
    names.sort();
    for n in &names {
        let c = read_ply_file(snap_dir.join(n)).map_err(|e| format!("{n}: {e}"))?;
        files += 1;
        if c.has_nan() {
            nan_files.push(n.clone());
        }
        let k = if n.starts_with("step_final") {
            None
        } else {
            n[5..].split('_').next().and_then(|t| t.parse::<f64>().ok())
        };
        if let Some(row) = rows
            .iter()
            .find(|r| r.1 == k && (k.is_some() || n.starts_with("step_final")))
        {
            if row.2 != c.len() as f64 {
                problems.push(format!("{n} 점 수 {} ≠ manifest {}", c.len(), row.2));
            }
        }
    }
    if files == 0 {
        problems.push("스냅샷 PLY 없음".into());
    }
    if !nan_files.is_empty() {
        problems.push(format!("NaN: {}", nan_files.join(",")));
    }
    let mut msg = format!(
        "{}단계, PLY {files}개, 점 {}→{}",
        rows.len(),
        rows[0].2,
        rows[rows.len() - 1].2
    );
    if min_area.is_finite() {
        let _ = write!(msg, ", 새 영역 최소 {min_area}");
    }
    if !problems.is_empty() {
        let _ = write!(msg, "; {}", problems.join("; "));
    }
    Ok((problems.is_empty(), msg))
}

// ---------------------------------------------------------------- 파일

fn read_json(path: &Path) -> Result<Json, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_json(&s).map_err(|e| format!("{}: {e}", path.display()))
}

/// `{prefix}{k:02}_*.ply` 를 구역 번호 k 로 묶어 읽는다.
fn load_regions(dir: &Path, prefix: &str) -> Result<BTreeMap<usize, PointCloud>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut out = BTreeMap::new();
    for e in rd.filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = n.strip_prefix(prefix) else {
            continue;
        };
        if !n.ends_with(".ply") {
            continue;
        }
        let Some(k) = rest.split('_').next().and_then(|t| t.parse::<usize>().ok()) else {
            continue;
        };
        let c = read_ply_file(e.path()).map_err(|err| format!("{n}: {err}"))?;
        out.insert(k, c);
    }
    Ok(out)
}

fn xyz(c: &PointCloud) -> Vec<[f64; 3]> {
    c.points
        .iter()
        .filter(|p| p.xyz.iter().all(|v| v.is_finite()))
        .map(|p| [p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64])
        .collect()
}

// ---------------------------------------------------------------- 격자 해시

/// 2D(수평) 또는 3D 격자 해시.
pub struct Grid {
    cell: f64,
    dims: usize,
    map: HashMap<[i64; 3], Vec<u32>>,
    pts: Vec<[f64; 3]>,
}

impl Grid {
    /// `dims` = 2 이면 z 를 무시한다.
    pub fn new(pts: &[[f64; 3]], cell: f64, dims: usize) -> Self {
        let mut map: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        let mut g = Grid {
            cell,
            dims,
            map: HashMap::new(),
            pts: pts.to_vec(),
        };
        for (i, p) in pts.iter().enumerate() {
            map.entry(g.key(p)).or_default().push(i as u32);
        }
        g.map = map;
        g
    }

    fn key(&self, p: &[f64; 3]) -> [i64; 3] {
        let f = |v: f64| (v / self.cell).floor() as i64;
        [f(p[0]), f(p[1]), if self.dims == 3 { f(p[2]) } else { 0 }]
    }

    fn dist2(&self, a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let dz = if self.dims == 3 { a[2] - b[2] } else { 0.0 };
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + dz * dz
    }

    /// `max_r` 안의 최근접 점 (번호, 거리). 껍질을 하나씩 넓히며 찾는다.
    pub fn nearest(&self, q: &[f64; 3], max_r: f64) -> Option<(usize, f64)> {
        let c = self.key(q);
        let max_ring = (max_r / self.cell).ceil() as i64 + 1;
        let mut best: Option<(usize, f64)> = None;
        let zr = |r: i64| if self.dims == 3 { r } else { 0 };
        for ring in 0..=max_ring {
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for dz in -zr(ring)..=zr(ring) {
                        let on_shell = dx.abs() == ring || dy.abs() == ring || dz.abs() == ring;
                        if !on_shell {
                            continue;
                        }
                        let Some(v) = self.map.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                            continue;
                        };
                        for &i in v {
                            let d2 = self.dist2(q, &self.pts[i as usize]);
                            if best.is_none_or(|(_, b)| d2 < b) {
                                best = Some((i as usize, d2));
                            }
                        }
                    }
                }
            }
            // 껍질 ring 바깥 점은 최소 ring*cell 만큼 떨어져 있다.
            if let Some((_, b)) = best {
                if b.sqrt() <= ring as f64 * self.cell {
                    break;
                }
            }
        }
        best.map(|(i, d2)| (i, d2.sqrt()))
            .filter(|&(_, d)| d <= max_r)
    }
}

fn stride_of(n: usize) -> usize {
    n.div_ceil(MAX_QUERIES).max(1)
}

/// 질의 점마다 기준 점군의 3D 최근접 거리(`NN_CAP_M` 에서 자름)의 중앙값.
pub fn nn_median(query: &[[f64; 3]], reference: &[[f64; 3]]) -> Option<f64> {
    if query.is_empty() || reference.is_empty() {
        return None;
    }
    let g = Grid::new(reference, 1.0, 3);
    let mut d: Vec<f64> = query
        .iter()
        .step_by(stride_of(query.len()))
        .map(|q| g.nearest(q, NN_CAP_M).map_or(NN_CAP_M, |(_, d)| d))
        .collect();
    Some(median(&mut d))
}

/// 질의 점마다 수평 `radius` 안의 수평 최근접 기준 점과 짝을 지어 |높이 차| 중앙값.
/// 짝이 하나도 없으면 None.
pub fn height_pair_median(query: &[[f64; 3]], reference: &[[f64; 3]], radius: f64) -> Option<f64> {
    if query.is_empty() || reference.is_empty() {
        return None;
    }
    let g = Grid::new(reference, radius.max(0.25), 2);
    let mut d: Vec<f64> = query
        .iter()
        .step_by(stride_of(query.len()))
        .filter_map(|q| {
            g.nearest(q, radius)
                .map(|(i, _)| (q[2] - reference[i][2]).abs())
        })
        .collect();
    if d.is_empty() {
        None
    } else {
        Some(median(&mut d))
    }
}

/// 중앙값 (짝수 개면 가운데 둘의 평균). 빈 입력은 NaN.
pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// ---------------------------------------------------------------- 작은 JSON 읽기

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, x)| x),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }
}

pub fn parse_json(s: &str) -> Result<Json, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = parse_value(b, &mut i)?;
    skip_ws(b, &mut i);
    if i != b.len() {
        return Err(format!("JSON 위치 {i}: 남는 문자"));
    }
    Ok(v)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn expect(b: &[u8], i: &mut usize, c: u8) -> Result<(), String> {
    skip_ws(b, i);
    if b.get(*i) == Some(&c) {
        *i += 1;
        Ok(())
    } else {
        Err(format!("JSON 위치 {}: '{}' 기대", *i, c as char))
    }
}

fn parse_value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    let err = |i: usize| format!("JSON 위치 {i}: 값이 잘못됨");
    match b.get(*i).copied() {
        Some(b'{') => {
            *i += 1;
            let mut v = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Json::Obj(v));
            }
            loop {
                skip_ws(b, i);
                let k = parse_str(b, i)?;
                expect(b, i, b':')?;
                v.push((k, parse_value(b, i)?));
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Json::Obj(v));
                    }
                    _ => return Err(err(*i)),
                }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut v = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Json::Arr(v));
            }
            loop {
                v.push(parse_value(b, i)?);
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Json::Arr(v));
                    }
                    _ => return Err(err(*i)),
                }
            }
        }
        Some(b'"') => parse_str(b, i).map(Json::Str),
        Some(b't') if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Json::Bool(true))
        }
        Some(b'f') if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Json::Bool(false))
        }
        Some(b'n') if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Json::Null)
        }
        Some(c) if c == b'-' || c.is_ascii_digit() => {
            let st = *i;
            while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                *i += 1;
            }
            std::str::from_utf8(&b[st..*i])
                .ok()
                .and_then(|t| t.parse::<f64>().ok())
                .map(Json::Num)
                .ok_or_else(|| err(st))
        }
        _ => Err(err(*i)),
    }
}

fn parse_str(b: &[u8], i: &mut usize) -> Result<String, String> {
    if b.get(*i) != Some(&b'"') {
        return Err(format!("JSON 위치 {}: 문자열 기대", *i));
    }
    *i += 1;
    let mut out = Vec::new();
    while let Some(&c) = b.get(*i) {
        *i += 1;
        match c {
            b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
            b'\\' => {
                let e = *b.get(*i).ok_or("JSON: 끝난 이스케이프")?;
                *i += 1;
                match e {
                    b'n' => out.push(b'\n'),
                    b't' => out.push(b'\t'),
                    b'r' => out.push(b'\r'),
                    b'b' => out.push(8),
                    b'f' => out.push(12),
                    b'u' => {
                        let h = b.get(*i..*i + 4).ok_or("JSON: \\u 짧음")?;
                        let cp = u32::from_str_radix(std::str::from_utf8(h).unwrap_or("x"), 16)
                            .map_err(|e| e.to_string())?;
                        *i += 4;
                        let ch = char::from_u32(cp).unwrap_or('\u{fffd}');
                        let mut tmp = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                    }
                    other => out.push(other),
                }
            }
            _ => out.push(c),
        }
    }
    Err("JSON: 닫히지 않은 문자열".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_basic() {
        let j = parse_json(r#"{"a":[1,2.5,-3e1],"b":{"c":"x\"y"},"d":true,"e":null}"#).unwrap();
        assert_eq!(
            j.get("a").unwrap().as_array().unwrap()[2].as_f64(),
            Some(-30.0)
        );
        assert_eq!(
            j.get("b").unwrap().get("c"),
            Some(&Json::Str("x\"y".into()))
        );
        assert!(parse_json("{\"a\":1,}").is_err());
        assert!(parse_json("[1 2]").is_err());
    }

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn grid_nearest_matches_brute_force() {
        let mut s = 7u64;
        let mut r = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) * 20.0 - 10.0
        };
        let pts: Vec<[f64; 3]> = (0..500).map(|_| [r(), r(), r()]).collect();
        let g = Grid::new(&pts, 1.0, 3);
        for _ in 0..200 {
            let q = [r(), r(), r()];
            let bf = pts
                .iter()
                .map(|p| {
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            let (_, d) = g.nearest(&q, 50.0).unwrap();
            assert!((d - bf).abs() < 1e-12, "{d} vs {bf}");
        }
        // 반경 밖이면 None.
        assert!(g.nearest(&[100.0, 100.0, 100.0], 5.0).is_none());
    }

    #[test]
    fn height_pairs_use_horizontal_neighbours() {
        // 기준: z=0 평면 격자, 질의: 같은 xy 에서 z=1.5, 그리고 멀리 떨어진 점(짝 없음).
        let reference: Vec<[f64; 3]> = (0..10)
            .flat_map(|x| (0..10).map(move |y| [x as f64, y as f64, 0.0]))
            .collect();
        let mut query: Vec<[f64; 3]> = reference.iter().map(|p| [p[0], p[1], 1.5]).collect();
        query.push([100.0, 100.0, 50.0]);
        let m = height_pair_median(&query, &reference, 2.0).unwrap();
        assert!((m - 1.5).abs() < 1e-12);
        assert!((nn_median(&query, &reference).unwrap() - 1.5).abs() < 1e-12);
    }
}
