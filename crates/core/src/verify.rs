//! 출력 폴더 검증 (SPEC §2 출력 구조, §4 검증 기준 일곱 항목 + 여덟째 `up_cross`).
//!
//! 읽는 파일:
//! - `report.json` (출력 폴더 바로 아래, 선택): 포즈 단계 기록. SPEC §2 출력 목록에 없는 파일이다.
//!   ```json
//!   {"registered": {"total": 240, "preview": 240, "refined": 240},
//!    "reprojection_px": {"preview": 4.5, "refined": 0.59},
//!    "regions": [{"region": 0, "positions": 14, "images": 42}]}
//!   ```
//!   `registered.total` 은 입력 사진 수, `preview`/`refined` 는 초벌·정밀 포즈에 등록된 사진 수.
//!   `reprojection_px` 는 재투영 오차 RMS(px). `regions[].positions` 는 구역의 위치 수,
//!   `images` 는 구역 밀집 단계에 쓴 사진 수.
//!
//! SPEC §2 출력만으로 판정할 수 있는 항목과 없는 항목:
//! - 판정 가능(4~7): `preview_align`(manifest `align`), `preview_vs_refined`·`refined_overlap`
//!   (구역 PLY), `snapshots`(manifest `snapshots` + step PLY).
//! - 판정 불가(1~3): `registered`(등록 사진 수), `region_images`(구역에 쓴 사진 수 — 파일 이름의
//!   `pos{lo}-{hi}` 로 위치 수는 알지만 사진 수는 출력에 없음), `refined_reprojection`(재투영 오차).
//!   세 값 모두 SPEC §2 의 어떤 파일에도 없다. `report.json` 이 없으면 이 셋은 FAIL 이 아니라
//!   "판정 불가" 로 표시하고, 있으면 그 값으로 판정한다. 형식이 깨진 `report.json` 은 FAIL.
//!   출력 폴더 자체가 없으면 모든 항목 FAIL.
//!
//! 종료 코드: FAIL 이 하나라도 있으면 1, FAIL 은 없고 판정 불가가 있으면 2, 모두 PASS 면 0.
//! - `snapshots/manifest.json`: SPEC §2 형식. `snapshots[].step` 은 1부터 세는 정수,
//!   최종만 문자열 `"final"`. 그 밖의 형(예: 문자열 `"01"`)은 형식 오류로 FAIL.
//! - `preview/preview_{k:02}_*.ply`, `refined/refined_{k:02}_*.ply`, `snapshots/step_*.ply`.
//!
//! 기대 파일 목록: 구역 집합 = report `regions` ∪ preview 구역 ∪ refined 구역 ∪ {0..마지막 정수 step}.
//! 이 집합의 구역마다 preview·refined PLY 가 있어야 하고, 정수 step 은 1..=구역 수 가 모두
//! 있어야 하며, manifest 단계마다 `step_{k:02}_{k}regions.ply`(최종 `step_final_all_refined.ply`)
//! 파일이 있어야 한다. 하나라도 빠지면 해당 항목 FAIL 과 빠진 이름을 표시한다.
//!
//! 스냅샷 해석(SPEC §3.8): step_k = 정밀 0..k-2 + 초벌 k-1, final = 정밀 전부(간격 추출 뒤).
//! 점 수 단조 증가는 정수 step 사이에만 적용한다(같은 수는 허용, 감소만 FAIL).
//! final 은 마지막 step 보다 작을 수 있으므로 정밀 구역 PLY 점 수 합과 같은지로 대조한다
//! (refined/ PLY 가 이미 6:1 추출된 점군이므로 다시 추출하지 않는다).
//!
//! 최근접 탐색은 상자 경계가 붙은 k-d 트리로 하고, `NN_CAP_M`(판정 기준 3 m 의 2배)
//! 너머는 찾지 않고 상한값으로 둔다(표기 "> 상한").
//! |좌표| > `COORD_LIMIT_M` 인 점은 계산에서 뺀다.

use crate::align::UP_CROSS_WARN_DEG;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;

use crate::ply::{read_ply_file, PointCloud};

// ---------------------------------------------------------------- 기준값

pub const REPROJ_MAX_PX: f64 = 0.7;
pub const ALIGN_MIN_PAIRS: f64 = 1000.0;
/// 구역 간 스케일 차: max(s)/min(s) − 1 ≤ 이 값.
pub const ALIGN_SCALE_TOL: f64 = 0.10;
pub const ALIGN_FIT_MAX_M: f64 = 6.0;
pub const NN_MEDIAN_MAX_M: f64 = 3.0;
pub const HEIGHT_PAIR_RADIUS_M: f64 = 2.0;
pub const HEIGHT_MEDIAN_MAX_M: f64 = 2.0;
pub const OVERLAP_PAIR_RADIUS_M: f64 = 1.0;
pub const OVERLAP_MEDIAN_MAX_M: f64 = 0.3;
/// 최근접 거리를 이 값에서 자른다 (판정 기준 3 m 의 2배: 중앙값 판정에는 영향 없음).
pub const NN_CAP_M: f64 = 2.0 * NN_MEDIAN_MAX_M;
/// 점군당 질의 점 상한 (넘으면 일정 간격으로 추림).
pub const MAX_QUERIES: usize = 200_000;
/// 이 값보다 큰 |좌표|(m)의 점은 계산에서 뺀다 (지역 직교 좌표에서 나올 수 없는 값).
pub const COORD_LIMIT_M: f64 = 1e7;
/// 카메라 묶음 위 방향 일치: 구역별 최대 어긋남(도)이 이 값을 넘으면 FAIL.
/// 근거: 정상 실행 시드 1/2/3 구역별 최대 1.449° 이하, 한 기체 짐벌 구름 1~2° 장면 1.1~1.3°,
/// 회전이 틀린 묶음은 47° 이상(시드 3 구역 0: 최대 69.5°). 정상 최대의 약 7배, 고장의 1/4 이하.
pub const UP_CROSS_FAIL_DEG: f64 = 10.0;
const EPS: f64 = 1e-9;

// ---------------------------------------------------------------- 결과

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub name: &'static str,
    /// 판정했고 통과. 판정 불가면 false.
    pub pass: bool,
    /// false 면 SPEC §2 출력만으로는 판정할 수 없는 항목(입력 파일 없음).
    pub decided: bool,
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

    /// 판정했는데 통과하지 못한 항목이 있음.
    pub fn any_fail(&self) -> bool {
        self.items.iter().any(|i| i.decided && !i.pass)
    }

    /// 판정 불가 항목 수.
    pub fn undecided(&self) -> usize {
        self.items.iter().filter(|i| !i.decided).count()
    }

    /// 0 = 모두 PASS, 1 = FAIL 있음(또는 항목 없음), 2 = FAIL 없고 판정 불가 있음.
    pub fn exit_code(&self) -> u8 {
        if self.items.is_empty() || self.any_fail() {
            1
        } else if self.undecided() > 0 {
            2
        } else {
            0
        }
    }

    pub fn item(&self, name: &str) -> Option<&Item> {
        self.items.iter().find(|i| i.name == name)
    }

    /// 마크다운 표.
    pub fn to_table(&self) -> String {
        let mut s = String::from("| 항목 | 결과 | 측정값 | 기준 |\n|---|---|---|---|\n");
        for i in &self.items {
            let r = if !i.decided {
                "판정 불가"
            } else if i.pass {
                "PASS"
            } else {
                "FAIL"
            };
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
        let u = self.undecided();
        if u > 0 {
            let _ = writeln!(s, "판정 불가: {u}개 (SPEC §2 출력에 없는 값)");
        }
        s
    }
}

/// `skylens-stream verify <출력 폴더>`: 표를 찍고 종료 코드는 [`Report::exit_code`].
pub fn run_cli(dir: &str) -> ExitCode {
    let report = verify_dir(Path::new(dir));
    print!("{}", report.to_table());
    ExitCode::from(report.exit_code())
}

pub const ITEM_REGISTERED: &str = "registered";
pub const ITEM_REGION_IMAGES: &str = "region_images";
pub const ITEM_REPROJ: &str = "refined_reprojection";
pub const ITEM_ALIGN: &str = "preview_align";
pub const ITEM_PREVIEW_REFINED: &str = "preview_vs_refined";
pub const ITEM_OVERLAP: &str = "refined_overlap";
pub const ITEM_SNAPSHOTS: &str = "snapshots";
pub const ITEM_UP_CROSS: &str = "up_cross";

fn item(name: &'static str, criterion: &'static str, r: Result<(bool, String), String>) -> Item {
    match r {
        Ok((pass, measured)) => Item {
            name,
            pass,
            decided: true,
            measured,
            criterion,
        },
        Err(e) => Item {
            name,
            pass: false,
            decided: true,
            measured: format!("오류: {e}"),
            criterion,
        },
    }
}

/// `report.json` 에서만 얻는 항목(1~3). 파일이 없고 출력 폴더는 있으면 판정 불가.
/// `absent_note` 는 report.json 이 없을 때 SPEC §2 출력에서 읽은 참고 값(판정에는 쓰지 않음).
fn report_item(
    name: &'static str,
    criterion: &'static str,
    report: &ReportJson,
    check: fn(&Json) -> Result<(bool, String), String>,
    absent_note: &str,
) -> Item {
    match report {
        ReportJson::Absent => Item {
            name,
            pass: false,
            decided: false,
            measured: format!("report.json 없음 (SPEC §2 출력에 없는 값){absent_note}"),
            criterion,
        },
        ReportJson::Read(r) => item(
            name,
            criterion,
            r.as_ref().map_err(Clone::clone).and_then(check),
        ),
    }
}

/// 여덟째 항목. report.json 이 없거나 `up_cross_check` 가 없으면 '건너뜀'(통과로 셈).
fn up_cross_item(report: &ReportJson) -> Item {
    const CRIT: &str = "구역별 카메라 묶음 위 방향 최대 어긋남 ≤ 10° (0.3° 초과는 경고, report.json 에 없으면 건너뜀)";
    match report {
        ReportJson::Absent => Item {
            name: ITEM_UP_CROSS,
            pass: true,
            decided: true,
            measured: "건너뜀 (report.json 없음)".into(),
            criterion: CRIT,
        },
        ReportJson::Read(r) => item(
            ITEM_UP_CROSS,
            CRIT,
            r.as_ref().map_err(Clone::clone).and_then(check_up_cross),
        ),
    }
}

/// 구역별 최대 어긋남을 `(구역 번호, 최대 도)` 로 모은다. 값이 모두 null 인 구역은 뺀다.
fn check_up_cross(r: &Json) -> Result<(bool, String), String> {
    let Some(u) = r.get("up_cross_check") else {
        return Ok((true, "건너뜀 (report.json 에 up_cross_check 없음)".into()));
    };
    let regs = u
        .get("regions")
        .and_then(Json::as_array)
        .ok_or("up_cross_check.regions 없음")?;
    let mut per: Vec<(String, f64)> = Vec::new();
    // diff_deg 가 전부 null 인 구역(등록 부족으로 카메라 묶음이 모자란 구역).
    let mut unmeasured: Vec<String> = Vec::new();
    for (i, g) in regs.iter().enumerate() {
        let k = g
            .get("region")
            .and_then(as_index)
            .map_or(i.to_string(), |k| k.to_string());
        let d = g
            .get("diff_deg")
            .and_then(Json::as_array)
            .ok_or("up_cross_check.regions[].diff_deg 없음")?;
        let mut mx: Option<f64> = None;
        for v in d {
            if let Some(x) = v.as_f64() {
                if !x.is_finite() || x < 0.0 {
                    return Err(format!("구역 {k} diff_deg 값 {x} 이 올바르지 않음"));
                }
                mx = Some(mx.map_or(x, |m| m.max(x)));
            } else if *v != Json::Null {
                return Err(format!("구역 {k} diff_deg 가 숫자 또는 null 이 아님"));
            }
        }
        match mx {
            Some(m) => per.push((k, m)),
            None => unmeasured.push(k),
        }
    }
    if per.is_empty() {
        // 검사한 구역이 하나도 없다: 문턱을 검증하지 못했으므로 '건너뜀' 통과가 아니라 경고로 표시한다.
        return Ok((
            true,
            format!(
                "경고: 측정값 없음 (구역 {} 전부 diff_deg null, 등록 부족으로 검사 못 함)",
                if unmeasured.is_empty() {
                    "없음".to_string()
                } else {
                    unmeasured.join(",")
                }
            ),
        ));
    }
    let list = per
        .iter()
        .map(|(k, m)| format!("{k}:{m:.3}°"))
        .collect::<Vec<_>>()
        .join(" ");
    let failed: Vec<&str> = per
        .iter()
        .filter(|(_, m)| *m > UP_CROSS_FAIL_DEG)
        .map(|(k, _)| k.as_str())
        .collect();
    if !failed.is_empty() {
        return Ok((
            false,
            format!(
                "구역 {} 최대 어긋남 > {UP_CROSS_FAIL_DEG}° (구역별 최대 {list})",
                failed.join(",")
            ),
        ));
    }
    let warn = per.iter().any(|(_, m)| *m > UP_CROSS_WARN_DEG);
    let mut tag = if warn {
        format!("경고: {UP_CROSS_WARN_DEG}° 초과, ")
    } else {
        String::new()
    };
    if !unmeasured.is_empty() {
        tag = format!(
            "경고: 구역 {} 미측정 (diff_deg 전부 null), {tag}",
            unmeasured.join(",")
        );
    }
    Ok((true, format!("{tag}구역별 최대 {list}")))
}

/// `report.json` 읽기 결과: 출력 폴더는 있는데 파일만 없으면 `Absent`.
enum ReportJson {
    Absent,
    Read(Result<Json, String>),
}

fn read_report(dir: &Path) -> ReportJson {
    let path = dir.join("report.json");
    if dir.is_dir() && !path.exists() {
        ReportJson::Absent
    } else {
        ReportJson::Read(read_json(&path))
    }
}

/// 출력 폴더 전체를 검증한다.
pub fn verify_dir(dir: &Path) -> Report {
    let report_src = read_report(dir);
    let report = match &report_src {
        ReportJson::Absent => Err("report.json 없음".to_string()),
        ReportJson::Read(r) => r.clone(),
    };
    let manifest = read_json(&dir.join("snapshots").join("manifest.json"));
    let steps = manifest
        .as_ref()
        .map_err(Clone::clone)
        .and_then(parse_steps);
    let preview = load_regions(&dir.join("preview"), "preview_");
    let refined = load_regions(&dir.join("refined"), "refined_");

    // 기대 구역 집합.
    let mut expected: BTreeSet<usize> = BTreeSet::new();
    if let Ok(r) = &report {
        if let Some(regs) = r.get("regions").and_then(Json::as_array) {
            for (i, g) in regs.iter().enumerate() {
                let k = g.get("region").and_then(as_index).unwrap_or(i);
                expected.insert(k);
            }
        }
    }
    for m in [&preview, &refined].into_iter().flatten() {
        expected.extend(m.keys().copied());
    }
    if let Ok(rows) = &steps {
        // 깨진 manifest 의 큰 step 하나가 기대 구역을 수천만 개로 부풀리지 않도록,
        // 다른 근거(report·PLY)의 구역 수와 정수 단계 줄 수 중 큰 값에서 자른다(정상 출력은
        // 정수 단계 줄 수 = 구역 수). 잘린 단계는 check_snapshots 가 "구역 수 초과" 로 FAIL 한다.
        let ints = rows.iter().filter(|r| r.step.is_some()).count();
        let cap = expected.len().max(ints);
        let last = rows.iter().filter_map(|r| r.step).max().unwrap_or(0);
        expected.extend(0..last.min(cap));
    }

    let items = vec![
        report_item(
            ITEM_REGISTERED,
            "초벌·정밀 모두 전체 등록 (240/240)",
            &report_src,
            check_registered,
            "",
        ),
        report_item(
            ITEM_REGION_IMAGES,
            "구역 사진 수 = 3 × 위치 수",
            &report_src,
            check_region_images,
            &positions_note(dir),
        ),
        report_item(ITEM_REPROJ, "정밀 재투영 ≤ 0.7 px", &report_src, check_reproj, ""),
        item(
            ITEM_ALIGN,
            "점쌍 ≥ 1000, 구역 간 스케일 차(최대/최소 − 1) ≤ 10%, 잔차 중앙 < 6 m",
            manifest
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|m| check_align(m, &expected)),
        ),
        item(
            ITEM_PREVIEW_REFINED,
            "구역마다 초벌·정밀 있음, 최근접 중앙 < 3 m, 수평 2 m 짝 높이 차 중앙 < 2 m",
            match (&preview, &refined) {
                (Ok(p), Ok(r)) => check_preview_vs_refined(p, r, &expected),
                (Err(e), _) | (_, Err(e)) => Err(e.clone()),
            },
        ),
        item(
            ITEM_OVERLAP,
            "이웃 정밀 구역 겹침(수평 1 m 짝) 높이 차 중앙 < 0.3 m (정밀 0개 FAIL, 구역 1개 해당 없음)",
            refined
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|r| check_overlap(r, &expected)),
        ),
        item(
            ITEM_SNAPSHOTS,
            "정수 단계 1..=구역 수 파일 있음, 점 수 단조 증가, 2단계부터 초벌 새 영역 > 0, final = 정밀 점 수 합, NaN 없음",
            steps.and_then(|rows| {
                check_snapshots(
                    &rows,
                    &dir.join("snapshots"),
                    &expected,
                    refined.as_ref().ok(),
                )
            }),
        ),
        up_cross_item(&report_src),
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

/// 음이 아닌 정수 값.
fn as_index(j: &Json) -> Option<usize> {
    match j {
        Json::Num(v) if v.fract() == 0.0 && *v >= 0.0 && *v < 1e9 => Some(*v as usize),
        _ => None,
    }
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

/// 구역 간 스케일 차 max(s)/min(s) − 1. 양수가 아닌 값이 있으면 무한대.
pub fn scale_spread(scales: &[f64]) -> f64 {
    let lo = scales.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = scales.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if lo.is_nan() || lo <= 0.0 || !hi.is_finite() {
        return f64::INFINITY;
    }
    hi / lo - 1.0
}

/// 초벌 정렬. SPEC §3.7 은 구역 0 도 자기 구역 사진으로 정렬하므로 `align[].region` 집합은
/// 기대 구역 집합과 같아야 한다. 빠진 구역·남는 구역·중복 기록은 FAIL.
fn check_align(m: &Json, expected: &BTreeSet<usize>) -> Result<(bool, String), String> {
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
    let mut seen = BTreeSet::new();
    let mut dup = Vec::new();
    for a in arr {
        let k = a
            .get("region")
            .and_then(as_index)
            .ok_or("align[].region 이 음이 아닌 정수가 아님")?;
        if !seen.insert(k) {
            dup.push(k);
        }
        pairs.push(num(a, &["pairs"])?);
        fits.push(num(a, &["fit_median_m"])?);
        scales.push(num(a, &["scale"])?);
    }
    let min_pairs = pairs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_fit = fits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spread = scale_spread(&scales);
    let missing: Vec<usize> = expected.difference(&seen).copied().collect();
    let extra: Vec<usize> = seen.difference(expected).copied().collect();
    let pass = min_pairs >= ALIGN_MIN_PAIRS
        && max_fit < ALIGN_FIT_MAX_M
        && spread <= ALIGN_SCALE_TOL + EPS
        && missing.is_empty()
        && extra.is_empty()
        && dup.is_empty();
    let mut msg = format!(
        "정렬 기록 {}개, 점쌍 최소 {min_pairs}, 구역 간 스케일 차 {:.2}%, 잔차 중앙 최대 {max_fit:.3} m",
        arr.len(),
        spread * 100.0
    );
    if !missing.is_empty() {
        let _ = write!(msg, "; 정렬 기록 없는 구역 {}", brief(&missing));
    }
    if !extra.is_empty() {
        let _ = write!(msg, "; 출력에 없는 구역의 정렬 기록 {}", brief(&extra));
    }
    if !dup.is_empty() {
        let _ = write!(msg, "; 중복 정렬 기록 {}", brief(&dup));
    }
    Ok((pass, msg))
}

/// 긴 번호 목록은 앞 `LIST_HEAD` 개와 전체 개수만 보인다(깨진 입력에서 출력이 커지지 않게).
const LIST_HEAD: usize = 10;

fn brief<T: std::fmt::Debug>(v: &[T]) -> String {
    if v.len() <= LIST_HEAD {
        format!("{v:?}")
    } else {
        format!(
            "{:?} 외 {}개 (총 {}개)",
            &v[..LIST_HEAD],
            v.len() - LIST_HEAD,
            v.len()
        )
    }
}

fn brief_names(v: &[String]) -> String {
    if v.len() <= LIST_HEAD {
        v.join(",")
    } else {
        format!(
            "{} 외 {}개 (총 {}개)",
            v[..LIST_HEAD].join(","),
            v.len() - LIST_HEAD,
            v.len()
        )
    }
}

fn missing_of(expected: &BTreeSet<usize>, have: &BTreeMap<usize, PointCloud>) -> Vec<usize> {
    expected
        .iter()
        .filter(|k| !have.contains_key(k))
        .copied()
        .collect()
}

fn check_preview_vs_refined(
    p: &BTreeMap<usize, PointCloud>,
    r: &BTreeMap<usize, PointCloud>,
    expected: &BTreeSet<usize>,
) -> Result<(bool, String), String> {
    if p.is_empty() || r.is_empty() {
        return Ok((false, "구역 점군 없음".into()));
    }
    let mut worst_nn = 0.0f64;
    let mut worst_dz = 0.0f64;
    for (k, pc) in p {
        let Some(rc) = r.get(k) else {
            continue;
        };
        let pts = xyz(pc);
        let refp = xyz(rc);
        let nn = nn_median(&pts, &refp).unwrap_or(f64::INFINITY);
        let dz = height_pair_median(&pts, &refp, HEIGHT_PAIR_RADIUS_M).unwrap_or(f64::INFINITY);
        worst_nn = worst_nn.max(nn);
        worst_dz = worst_dz.max(dz);
    }
    let miss_p = missing_of(expected, p);
    let miss_r = missing_of(expected, r);
    let pass = miss_p.is_empty()
        && miss_r.is_empty()
        && worst_nn < NN_MEDIAN_MAX_M
        && worst_dz < HEIGHT_MEDIAN_MAX_M;
    let nn_s = if worst_nn >= NN_CAP_M {
        format!("> {NN_CAP_M:.3} m")
    } else {
        format!("{worst_nn:.3} m")
    };
    let mut m = format!(
        "{}개 구역, 최근접 중앙 최대 {nn_s}, 높이 차 중앙 최대 {worst_dz:.3} m",
        expected.len()
    );
    if !miss_p.is_empty() {
        let _ = write!(m, ", 초벌 없는 구역 {}", brief(&miss_p));
    }
    if !miss_r.is_empty() {
        let _ = write!(m, ", 정밀 없는 구역 {}", brief(&miss_r));
    }
    Ok((pass, m))
}

fn check_overlap(
    r: &BTreeMap<usize, PointCloud>,
    expected: &BTreeSet<usize>,
) -> Result<(bool, String), String> {
    if r.is_empty() {
        return Ok((false, "정밀 구역 0개".into()));
    }
    let miss = missing_of(expected, r);
    if !miss.is_empty() {
        return Ok((false, format!("정밀 없는 구역 {}", brief(&miss))));
    }
    let keys: Vec<usize> = r.keys().copied().collect();
    if keys.len() < 2 {
        return Ok((true, "해당 없음 (구역 1개)".into()));
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

/// manifest 의 스냅샷 한 줄. `step` 이 None 이면 최종.
#[derive(Clone, Debug, PartialEq)]
pub struct StepRow {
    pub step: Option<usize>,
    pub points: f64,
    pub preview_new_area: Option<f64>,
}

/// `snapshots[]` 를 읽는다. step 은 1 이상 정수 또는 `"final"` 만 받는다.
pub fn parse_steps(m: &Json) -> Result<Vec<StepRow>, String> {
    let arr = m
        .get("snapshots")
        .and_then(Json::as_array)
        .ok_or("snapshots 없음")?;
    let mut rows = Vec::new();
    for s in arr {
        let step = s.get("step").ok_or("step 없음")?;
        let k = match step {
            Json::Str(t) if t == "final" => None,
            j => match as_index(j) {
                Some(k) if k >= 1 => Some(k),
                _ => {
                    return Err(format!(
                        "step 형식 오류 {step:?}: 1 이상 정수 또는 \"final\" (SPEC §2)"
                    ))
                }
            },
        };
        rows.push(StepRow {
            step: k,
            points: num(s, &["points"])?,
            preview_new_area: s.get("preview_new_area").and_then(Json::as_f64),
        });
    }
    Ok(rows)
}

/// 스냅샷 파일 이름.
pub fn snapshot_file_name(step: Option<usize>) -> String {
    match step {
        Some(k) => format!("step_{k:02}_{k}regions.ply"),
        None => "step_final_all_refined.ply".to_string(),
    }
}

fn check_snapshots(
    rows: &[StepRow],
    snap_dir: &Path,
    expected: &BTreeSet<usize>,
    refined: Option<&BTreeMap<usize, PointCloud>>,
) -> Result<(bool, String), String> {
    if rows.is_empty() {
        return Ok((false, "스냅샷 0개".into()));
    }
    let mut problems = Vec::new();
    let mut ints: Vec<&StepRow> = rows.iter().filter(|r| r.step.is_some()).collect();
    ints.sort_by_key(|r| r.step);
    let finals: Vec<&StepRow> = rows.iter().filter(|r| r.step.is_none()).collect();

    // 기대 단계: 1..=구역 수 + final 하나.
    let want: Vec<usize> = (1..=expected.len()).collect();
    let have: Vec<usize> = ints.iter().filter_map(|r| r.step).collect();
    let miss_steps: Vec<usize> = want.iter().filter(|k| !have.contains(k)).copied().collect();
    if !miss_steps.is_empty() {
        problems.push(format!("manifest 에 없는 단계 {}", brief(&miss_steps)));
    }
    let over: Vec<usize> = have
        .iter()
        .filter(|&&k| k > expected.len())
        .copied()
        .collect();
    if !over.is_empty() {
        problems.push(format!(
            "구역 수 {} 초과 단계 {}",
            expected.len(),
            brief(&over)
        ));
    }
    if have.windows(2).any(|w| w[0] == w[1]) {
        problems.push("중복 단계".into());
    }
    if finals.len() != 1 {
        problems.push(format!("final 단계 {}개", finals.len()));
    }

    // 단조 증가: 정수 단계 사이에만 (같은 수 허용).
    for w in ints.windows(2) {
        if crate::stream::snapshot_count_decreased(w[0].points, w[1].points) {
            problems.push(format!(
                "점 수 감소 단계 {}→{}: {}→{}",
                w[0].step.unwrap_or(0),
                w[1].step.unwrap_or(0),
                w[0].points,
                w[1].points
            ));
        }
    }
    let mut min_area = f64::INFINITY;
    for r in &ints {
        let k = r.step.unwrap_or(0);
        if k >= 2 {
            let a = r.preview_new_area.unwrap_or(f64::NAN);
            min_area = min_area.min(if a.is_nan() { f64::NEG_INFINITY } else { a });
            if a.is_nan() || a <= 0.0 {
                problems.push(format!("단계 {k} 새 영역 {a}"));
            }
        }
    }
    // final = 정밀 구역 점 수 합. refined/ PLY 는 이미 6:1 추출된 점군이므로(SPEC §3.8
    // "모든 점군은 6:1 간격 추출") final 은 다시 추출하지 않은 그 합과 같아야 한다.
    if let (Some(fr), Some(rc)) = (finals.first(), refined) {
        if !rc.is_empty() {
            let sizes: Vec<usize> = rc.values().map(PointCloud::len).collect();
            let sum: usize = sizes.iter().sum();
            if sum as f64 != fr.points {
                problems.push(format!(
                    "final {} ≠ 정밀 점 수 합 {sum} ({sizes:?})",
                    fr.points
                ));
            }
        }
    }

    // 스냅샷 PLY: manifest 단계마다 있어야 하고, NaN 없음, 점 수 = manifest.
    let mut missing_files = Vec::new();
    let mut nan_files = Vec::new();
    let mut files = 0usize;
    let mut listed = BTreeSet::new();
    for r in rows {
        let n = snapshot_file_name(r.step);
        listed.insert(n.clone());
        let path = snap_dir.join(&n);
        if !path.exists() {
            missing_files.push(n);
            continue;
        }
        let c = read_ply_file(&path).map_err(|e| format!("{n}: {e}"))?;
        files += 1;
        if c.has_nan() {
            nan_files.push(n.clone());
        }
        if r.points != c.len() as f64 {
            problems.push(format!("{n} 점 수 {} ≠ manifest {}", c.len(), r.points));
        }
    }
    if let Ok(rd) = std::fs::read_dir(snap_dir) {
        let mut extra: Vec<String> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("step_") && n.ends_with(".ply") && !listed.contains(n))
            .collect();
        extra.sort();
        if !extra.is_empty() {
            problems.push(format!("manifest 에 없는 파일 {}", brief_names(&extra)));
        }
    }
    if !missing_files.is_empty() {
        problems.push(format!("빠진 파일 {}", brief_names(&missing_files)));
    }
    if !nan_files.is_empty() {
        problems.push(format!("NaN: {}", brief_names(&nan_files)));
    }
    let first = ints.first().map_or(f64::NAN, |r| r.points);
    let last = ints.last().map_or(f64::NAN, |r| r.points);
    let mut msg = format!("{}단계, PLY {files}개, 점 {first}→{last}", rows.len());
    if let Some(fr) = finals.first() {
        let _ = write!(msg, ", final {}", fr.points);
    }
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

/// `{prefix}{k:02}_pos{lo}-{hi}.ply` 이름에서 구역별 위치 범위 [lo, hi) 를 읽는다
/// (SPEC §3.5 구역 [start-OVL, start+SPAN+OVL), hi 는 포함 안 함).
pub fn region_positions(dir: &Path, prefix: &str) -> BTreeMap<usize, (usize, usize)> {
    let mut out = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = n.strip_prefix(prefix).and_then(|r| r.strip_suffix(".ply")) else {
            continue;
        };
        let Some((k, pos)) = rest.split_once("_pos") else {
            continue;
        };
        let Some((lo, hi)) = pos.split_once('-') else {
            continue;
        };
        if let (Ok(k), Ok(lo), Ok(hi)) = (k.parse(), lo.parse(), hi.parse()) {
            out.insert(k, (lo, hi));
        }
    }
    out
}

/// report.json 이 없을 때 구역 사진 수 항목에 붙이는 참고 값: 파일 이름의 구역별 위치 수
/// (hi − lo). 사진 수는 SPEC §2 출력에 없어 판정은 하지 않는다. 초벌·정밀 이름의 위치
/// 범위가 다르면 그 구역도 적는다.
fn positions_note(dir: &Path) -> String {
    let p = region_positions(&dir.join("preview"), "preview_");
    let r = region_positions(&dir.join("refined"), "refined_");
    let src = if r.is_empty() { &p } else { &r };
    if src.is_empty() {
        return String::new();
    }
    let counts: Vec<String> = src
        .iter()
        .map(|(k, (lo, hi))| format!("{k}:{}", hi.saturating_sub(*lo)))
        .collect();
    let mut s = format!(
        "; 파일 이름의 위치 수 [{}], 사진 수는 출력에 없음",
        counts.join(" ")
    );
    let differ: Vec<usize> = p
        .iter()
        .filter(|(k, v)| r.get(k).is_some_and(|w| w != *v))
        .map(|(k, _)| *k)
        .collect();
    if !differ.is_empty() {
        let _ = write!(s, "; 초벌·정밀 위치 범위가 다른 구역 {differ:?}");
    }
    s
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

fn usable(p: &[f64; 3]) -> bool {
    p.iter().all(|v| v.is_finite() && v.abs() <= COORD_LIMIT_M)
}

fn xyz(c: &PointCloud) -> Vec<[f64; 3]> {
    c.points
        .iter()
        .map(|p| [p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64])
        .filter(usable)
        .collect()
}

// ---------------------------------------------------------------- 최근접 (k-d 트리)

const LEAF: usize = 16;

struct Node {
    lo: [f64; 3],
    hi: [f64; 3],
    start: usize,
    end: usize,
    kids: Option<(usize, usize)>,
}

/// 2D(수평) 또는 3D 최근접 탐색용 k-d 트리. 노드마다 경계 상자를 두어,
/// 질의 반경 밖의 상자는 통째로 건너뛴다(빈 공간에서도 질의당 노드 몇 개만 본다).
pub struct NnIndex {
    dims: usize,
    pts: Vec<[f64; 3]>,
    nodes: Vec<Node>,
}

impl NnIndex {
    /// `dims` = 2 이면 z 를 무시한다. 유한하지 않거나 |좌표| > `COORD_LIMIT_M` 인 점은 뺀다.
    pub fn new(pts: &[[f64; 3]], dims: usize) -> Self {
        let mut t = NnIndex {
            dims,
            pts: pts.iter().copied().filter(usable).collect(),
            nodes: Vec::new(),
        };
        if !t.pts.is_empty() {
            t.build(0, t.pts.len());
        }
        t
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &self.pts[start..end] {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            lo,
            hi,
            start,
            end,
            kids: None,
        });
        if end - start > LEAF {
            let axis = (0..self.dims)
                .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
                .unwrap_or(0);
            let mid = (start + end) / 2;
            self.pts[start..end]
                .select_nth_unstable_by(mid - start, |a, b| a[axis].total_cmp(&b[axis]));
            let l = self.build(start, mid);
            let r = self.build(mid, end);
            self.nodes[id].kids = Some((l, r));
        }
        id
    }

    fn box_d2(&self, n: &Node, q: &[f64; 3]) -> f64 {
        (0..self.dims)
            .map(|a| {
                let d = (n.lo[a] - q[a]).max(q[a] - n.hi[a]).max(0.0);
                d * d
            })
            .sum()
    }

    fn d2(&self, a: &[f64; 3], b: &[f64; 3]) -> f64 {
        (0..self.dims).map(|k| (a[k] - b[k]).powi(2)).sum()
    }

    /// `max_r` 안의 최근접 점 (점, 거리). 없으면 None.
    pub fn nearest(&self, q: &[f64; 3], max_r: f64) -> Option<([f64; 3], f64)> {
        if self.nodes.is_empty() || !usable(q) {
            return None;
        }
        // 같은 거리의 상자·점은 더 볼 필요가 없다(`>=` 가지치기, `<` 갱신). 같은 좌표 점이
        // 많을 때 `>`/`<=` 로 두면 그 점들을 모두 훑어 질의당 O(n) 이 된다. 반경 경계의 점을
        // 그대로 받도록 시작 상한을 아주 조금 넓힌다.
        let mut best_d2 = max_r * max_r * (1.0 + 1e-12);
        let mut best: Option<usize> = None;
        let mut stack = vec![0usize];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id];
            if self.box_d2(n, q) >= best_d2 {
                continue;
            }
            match n.kids {
                None => {
                    for i in n.start..n.end {
                        let d = self.d2(q, &self.pts[i]);
                        if d < best_d2 {
                            best_d2 = d;
                            best = Some(i);
                        }
                    }
                }
                Some((l, r)) => {
                    // 가까운 쪽을 먼저 보도록 먼 쪽을 먼저 쌓는다.
                    let (dl, dr) = (
                        self.box_d2(&self.nodes[l], q),
                        self.box_d2(&self.nodes[r], q),
                    );
                    if dl <= dr {
                        stack.push(r);
                        stack.push(l);
                    } else {
                        stack.push(l);
                        stack.push(r);
                    }
                }
            }
        }
        best.map(|i| (self.pts[i], best_d2.sqrt()))
    }
}

fn stride_of(n: usize) -> usize {
    n.div_ceil(MAX_QUERIES).max(1)
}

/// 질의 점마다 기준 점군의 3D 최근접 거리(`NN_CAP_M` 에서 자름)의 중앙값.
pub fn nn_median(query: &[[f64; 3]], reference: &[[f64; 3]]) -> Option<f64> {
    let query: Vec<[f64; 3]> = query.iter().copied().filter(usable).collect();
    let t = NnIndex::new(reference, 3);
    if query.is_empty() || t.pts.is_empty() {
        return None;
    }
    let mut d: Vec<f64> = query
        .iter()
        .step_by(stride_of(query.len()))
        .map(|q| t.nearest(q, NN_CAP_M).map_or(NN_CAP_M, |(_, d)| d))
        .collect();
    Some(median(&mut d))
}

/// 질의 점마다 수평 `radius` 안의 수평 최근접 기준 점과 짝을 지어 |높이 차| 중앙값.
/// 짝이 하나도 없으면 None.
pub fn height_pair_median(query: &[[f64; 3]], reference: &[[f64; 3]], radius: f64) -> Option<f64> {
    let query: Vec<[f64; 3]> = query.iter().copied().filter(usable).collect();
    let t = NnIndex::new(reference, 2);
    if query.is_empty() || t.pts.is_empty() {
        return None;
    }
    let mut d: Vec<f64> = query
        .iter()
        .step_by(stride_of(query.len()))
        .filter_map(|q| t.nearest(q, radius).map(|(p, _)| (q[2] - p[2]).abs()))
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
    let v = parse_value(b, &mut i, 0)?;
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

/// 배열·객체 중첩 상한. 출력 파일은 깊이 3~4 이므로 넉넉하고, 깨진 파일의 깊은 중첩이
/// 재귀로 스택을 넘기지 않게 한다.
pub const JSON_MAX_DEPTH: usize = 64;

fn parse_value(b: &[u8], i: &mut usize, depth: usize) -> Result<Json, String> {
    skip_ws(b, i);
    if depth >= JSON_MAX_DEPTH && matches!(b.get(*i), Some(b'{' | b'[')) {
        return Err(format!("JSON 위치 {}: 중첩 깊이 {JSON_MAX_DEPTH} 초과", *i));
    }
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
                v.push((k, parse_value(b, i, depth + 1)?));
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
                v.push(parse_value(b, i, depth + 1)?);
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

    fn up_cross(json: &str) -> Result<(bool, String), String> {
        check_up_cross(&parse_json(json).unwrap())
    }

    fn up_cross_report(d0: &str, d1: &str) -> String {
        format!(
            r#"{{"up_cross_check":{{"threshold_deg":0.3,"regions":[
{{"region":0,"cameras":["F","R","L"],"diff_deg":[{d0}]}},
{{"region":1,"cameras":["F","R","L"],"diff_deg":[{d1}]}}]}}}}"#
        )
    }

    /// 손으로 만든 report: 정상(≤0.3°) 통과, 0.3~10° 경고(통과), 10° 초과 FAIL, 항목 없음 건너뜀(통과).
    #[test]
    fn up_cross_pass_warn_fail_skip() {
        let (ok, m) = up_cross(&up_cross_report("0.02, 0.03, 0.01", "0.25, null, 0.1")).unwrap();
        assert!(ok && !m.contains("경고"), "{m}");
        assert!(m.contains("0:0.030°") && m.contains("1:0.250°"), "{m}");
        // 시드 2 구역 2 실측 1.386° → 경고, 통과.
        let (ok, m) = up_cross(&up_cross_report("0.02, 0.03", "1.386, 0.5, 0.2")).unwrap();
        assert!(ok && m.contains("경고"), "{m}");
        // 경계: 정확히 10° 는 통과, 10.001° 는 FAIL.
        assert!(up_cross(&up_cross_report("10.0", "0.1")).unwrap().0);
        // 시드 3 구역 0 실측 69.509° → FAIL, 구역 번호 표시.
        let (ok, m) = up_cross(&up_cross_report("69.509, 47.0, 70.0", "0.25")).unwrap();
        assert!(
            !ok && m.contains("구역 0 ") && m.contains("0:70.000°"),
            "{m}"
        );
        let (ok, m) = up_cross(&up_cross_report("0.1", "10.001")).unwrap();
        assert!(!ok && m.contains("구역 1 "), "{m}");
        // 항목 없음 → 건너뜀(통과).
        let (ok, m) = up_cross(r#"{"registered":{"total":1}}"#).unwrap();
        assert!(ok && m.starts_with("건너뜀"), "{m}");
        // 값이 모두 null → 통과, 측정값 없음.
        let (ok, m) = up_cross(&up_cross_report("null", "")).unwrap();
        assert!(
            ok && m.starts_with("경고: 측정값 없음") && m.contains("구역 0,1"),
            "{m}"
        );
        // 일부 구역만 null → 측정된 구역은 판정하고 미측정 구역을 경고로 알린다.
        let (ok, m) = up_cross(&up_cross_report("null, null", "0.02")).unwrap();
        assert!(
            ok && m.starts_with("경고: 구역 0 미측정") && m.contains("1:0.020°"),
            "{m}"
        );
        let (ok, m) = up_cross(&up_cross_report("null", "10.5")).unwrap();
        assert!(!ok && m.contains("구역 1 "), "{m}");
        // 구역 목록이 비어도(regions: []) 건너뜀이 아니라 경고.
        let (ok, m) = up_cross(r#"{"up_cross_check":{"regions":[]}}"#).unwrap();
        assert!(ok && m.starts_with("경고: 측정값 없음"), "{m}");
        // 형식 오류는 오류.
        assert!(up_cross(r#"{"up_cross_check":{}}"#).is_err());
        assert!(up_cross(&up_cross_report("\"x\"", "0.1")).is_err());
    }

    #[test]
    fn up_cross_item_absent_report_is_skipped_pass() {
        let it = up_cross_item(&ReportJson::Absent);
        assert!(it.pass && it.decided && it.measured.starts_with("건너뜀"));
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
        let g = NnIndex::new(&pts, 3);
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

    /// 2만 × 2만 점 평면, 질의 점군이 25 m 위에 떠 있어 상한 안에 점이 없는 경우.
    /// 예전 격자 껍질 탐색은 같은 크기에서 57.7 s 걸렸다. 상한·상자 건너뛰기로 1 s 안.
    #[test]
    fn nn_median_far_offset_is_fast_and_capped() {
        let reference: Vec<[f64; 3]> = (0..20_000)
            .map(|i| [(i % 200) as f64 * 0.2, (i / 200) as f64 * 0.2, 0.0])
            .collect();
        let query: Vec<[f64; 3]> = reference.iter().map(|p| [p[0], p[1], 25.0]).collect();
        let t0 = std::time::Instant::now();
        let m = nn_median(&query, &reference).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        assert_eq!(m, NN_CAP_M);
        assert!(secs < 1.0, "{secs:.3} s");
    }

    /// x=1e30 같은 점이 섞여도 패닉 없이 그 점만 빼고 계산한다 (디버그 빌드 넘침 확인 포함).
    #[test]
    fn huge_coordinates_are_ignored() {
        let mut reference: Vec<[f64; 3]> = (0..100)
            .map(|i| [(i % 10) as f64, (i / 10) as f64, 0.0])
            .collect();
        reference.push([1e30, 0.0, 0.0]);
        reference.push([-1e19, 1e19, 5.0]);
        let mut query: Vec<[f64; 3]> = reference[..100].iter().map(|p| [p[0], p[1], 0.5]).collect();
        query.push([1e30, 1e30, 1e30]);
        assert!((nn_median(&query, &reference).unwrap() - 0.5).abs() < 1e-12);
        assert!((height_pair_median(&query, &reference, 2.0).unwrap() - 0.5).abs() < 1e-12);
        let t = NnIndex::new(&reference, 2);
        assert!(t.nearest(&[1e30, 0.0, 0.0], 2.0).is_none());
    }

    /// 같은 좌표 점 20만 개를 기준으로, 질의 20만 개(같은 좌표 10만 + 그 둘레 원 위 10만).
    /// 가지치기가 같은 거리 상자를 남기면 질의마다 20만 점을 다 훑어 165 s 걸렸다.
    /// 정답: 같은 좌표 질의는 0, 원 위 질의는 반지름 1.0 → 짝수 개의 중앙 = (0 + 1)/2 = 0.5.
    #[test]
    fn nn_median_many_identical_points_is_fast_and_exact() {
        let n = 200_000;
        let reference = vec![[10.0, 10.0, 0.0]; n];
        let query: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    [10.0, 10.0, 0.0]
                } else {
                    let a = i as f64 * 0.001;
                    [10.0 + a.cos(), 10.0 + a.sin(), 0.0]
                }
            })
            .collect();
        let t0 = std::time::Instant::now();
        let m = nn_median(&query, &reference).unwrap();
        let h = height_pair_median(&query, &reference, 2.0).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        assert!((m - 0.5).abs() < 1e-9, "{m}");
        assert_eq!(h, 0.0);
        assert!(secs < 1.0, "{secs:.3} s");
    }

    /// 같은 좌표·같은 거리 점이 많은 점군에서 브루트포스와 거리가 같고,
    /// 반경 정확히 경계의 점도 찾는다(시작 상한을 조금 넓힌 효과).
    #[test]
    fn nearest_with_ties_matches_brute_force() {
        let mut pts = Vec::new();
        for i in 0..40 {
            for _ in 0..30 {
                pts.push([(i % 5) as f64, (i / 5) as f64, 0.0]);
            }
        }
        let g = NnIndex::new(&pts, 3);
        for qx in 0..12 {
            for qy in 0..18 {
                let q = [qx as f64 * 0.5 - 0.5, qy as f64 * 0.5 - 0.5, 0.25];
                let bf = pts
                    .iter()
                    .map(|p| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + 0.0625).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let (_, d) = g.nearest(&q, 50.0).unwrap();
                assert!((d - bf).abs() < 1e-12, "{q:?}: {d} vs {bf}");
            }
        }
        let (_, d) = g.nearest(&[0.0, 0.0, 3.0], 3.0).unwrap();
        assert_eq!(d, 3.0);
        assert!(g.nearest(&[0.0, 0.0, 3.0], 2.999).is_none());
    }

    /// 깊은 중첩은 스택을 넘기지 않고 형식 오류. 깊이 64 바로 아래는 읽는다.
    #[test]
    fn json_depth_is_limited() {
        let ok = format!(
            "{}{}",
            "[".repeat(JSON_MAX_DEPTH),
            "]".repeat(JSON_MAX_DEPTH)
        );
        assert!(parse_json(&ok).is_ok());
        let bad = format!(
            "{}{}",
            "[".repeat(JSON_MAX_DEPTH + 1),
            "]".repeat(JSON_MAX_DEPTH + 1)
        );
        assert!(parse_json(&bad).unwrap_err().contains("중첩 깊이"));
        let huge = format!("{{\"snapshots\":{}", "[".repeat(300_000));
        assert!(parse_json(&huge).unwrap_err().contains("중첩 깊이"));
    }

    #[test]
    fn scale_spread_is_between_regions() {
        assert!(scale_spread(&[1.0, 1.0, 1.1]) <= ALIGN_SCALE_TOL + EPS);
        assert!(scale_spread(&[0.95, 1.0, 1.06]) > ALIGN_SCALE_TOL + EPS);
        assert!(scale_spread(&[0.9, 1.0, 1.1]) > ALIGN_SCALE_TOL + EPS);
        assert_eq!(scale_spread(&[0.0, 1.0]), f64::INFINITY);
    }

    #[test]
    fn step_must_be_integer_or_final() {
        let ok = parse_json(r#"{"snapshots":[{"step":1,"points":3},{"step":"final","points":4}]}"#)
            .unwrap();
        let rows = parse_steps(&ok).unwrap();
        assert_eq!(rows[0].step, Some(1));
        assert_eq!(rows[1].step, None);
        for bad in [r#""01""#, "1.5", "0", "-1", "null"] {
            let j =
                parse_json(&format!(r#"{{"snapshots":[{{"step":{bad},"points":3}}]}}"#)).unwrap();
            let e = parse_steps(&j).unwrap_err();
            assert!(e.contains("step 형식 오류"), "{bad}: {e}");
        }
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
