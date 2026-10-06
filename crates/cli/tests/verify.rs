//! `skylens-stream verify` 시험: 작은 통과용 출력 폴더를 만들고 항목마다 하나씩 깨뜨린다.

use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::ply::{write_ply_file, PointCloud, PointRecord};

#[derive(Clone)]
struct Fixture {
    /// 장면 사진 수 (report registered.total).
    total: usize,
    reg_preview: u32,
    reg_refined: u32,
    region_images: [u32; 2],
    reproj_refined: f64,
    /// 구역 0·1 의 정렬 값 (SPEC §3.7: 구역 0 도 자기 구역으로 정렬).
    align_pairs: [u32; 2],
    align_fit: [f64; 2],
    align_scale: [f64; 2],
    /// manifest `align` 에 쓸 구역 번호 (기본 [0, 1]; 2 이상은 구역 1 의 값을 쓴다).
    align_regions: &'static [usize],
    /// 초벌 점군 = 정밀 점군 xy 그대로, z 에 이 값을 더함.
    preview_dz: f32,
    /// 초벌 점군에 짝 없는 먼 점을 (격자 점 수 + 1)개 덧붙임 → 최근접 중앙이 커짐.
    preview_outliers: bool,
    /// 정밀 구역 1 의 높이 (구역 0 은 0).
    refined1_z: f32,
    snap_points: [usize; 3],
    snap_area: [f64; 3],
    snap_nan: bool,
    /// 지면 기울기 dz/dx (0 이면 평면).
    slope: f32,
    /// 초벌 점군만 x 방향으로 이만큼(m) 더 넓게 만든다.
    preview_extra_x: f32,
    /// 격자 간격·x/y 칸 수 (기본 0.5 m, 41×21).
    cell: f32,
    nx: usize,
    ny: usize,
}

impl Default for Fixture {
    fn default() -> Self {
        Fixture {
            total: 240,
            reg_preview: 240,
            reg_refined: 240,
            region_images: [42, 48],
            reproj_refined: 0.7,
            align_pairs: [1000, 2000],
            align_fit: [5.99, 1.0],
            align_scale: [1.0, 1.1],
            align_regions: &[0, 1],
            preview_dz: 1.99,
            preview_outliers: false,
            refined1_z: 0.29,
            // final 1722 = 정밀 구역 861 점 × 2 의 합 (refined/ PLY 는 이미 추출된 점군).
            snap_points: [100, 150, 1722],
            snap_area: [0.0, 12.5, 30.0],
            snap_nan: false,
            slope: 0.0,
            preview_extra_x: 0.0,
            cell: 0.5,
            nx: 41,
            ny: 21,
        }
    }
}

fn rec(x: f32, y: f32, z: f32) -> PointRecord {
    PointRecord {
        xyz: [x, y, z],
        normal: [0.0, 0.0, 1.0],
        rgb: [100, 120, 140],
    }
}

/// x = x0 + i·cell (i < nx), y = j·cell (j < ny), z = z0 + slope·x 격자.
fn grid(f: &Fixture, x0: f32, z0: f32, nx: usize) -> Vec<PointRecord> {
    let mut v = Vec::with_capacity(nx * f.ny);
    for i in 0..nx {
        for j in 0..f.ny {
            let x = x0 + i as f32 * f.cell;
            v.push(rec(x, j as f32 * f.cell, z0 + f.slope * x));
        }
    }
    v
}

fn write(path: &Path, points: Vec<PointRecord>) {
    write_ply_file(path, &PointCloud { points }).unwrap();
}

fn build(dir: &Path, f: &Fixture) {
    for sub in ["preview", "refined", "snapshots"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    std::fs::write(
        dir.join("report.json"),
        format!(
            r#"{{"registered":{{"total":{},"preview":{},"refined":{}}},
"reprojection_px":{{"preview":4.5,"refined":{}}},
"regions":[{{"region":0,"positions":14,"images":{}}},{{"region":1,"positions":16,"images":{}}}]}}"#,
            f.total,
            f.reg_preview,
            f.reg_refined,
            f.reproj_refined,
            f.region_images[0],
            f.region_images[1]
        ),
    )
    .unwrap();

    let names = [
        ("00", "pos0-14", 0.0f32, 0.0f32),
        ("01", "pos10-26", 15.0, f.refined1_z),
    ];
    for (k, pos, x0, z) in names {
        write(
            &dir.join(format!("refined/refined_{k}_{pos}.ply")),
            grid(f, x0, z, f.nx),
        );
        let extra = (f.preview_extra_x / f.cell).round() as usize;
        let mut p = grid(f, x0, z + f.preview_dz, f.nx + extra);
        if f.preview_outliers {
            let n = p.len() + 1;
            p.extend((0..n).map(|i| rec(500.0 + i as f32, 500.0, 0.0)));
        }
        write(&dir.join(format!("preview/preview_{k}_{pos}.ply")), p);
    }

    let mut snaps = Vec::new();
    let files = [
        "step_01_1regions.ply",
        "step_02_2regions.ply",
        "step_final_all_refined.ply",
    ];
    for (i, name) in files.iter().enumerate() {
        let n = f.snap_points[i];
        let mut pts: Vec<PointRecord> = (0..n).map(|j| rec(j as f32, 0.0, 1.0)).collect();
        if f.snap_nan && i == 1 {
            pts[3].xyz[2] = f32::NAN;
        }
        write(&dir.join("snapshots").join(name), pts);
        let step = if i == 2 {
            "\"final\"".to_string()
        } else {
            (i + 1).to_string()
        };
        snaps.push(format!(
            r#"{{"step":{step},"points":{n},"preview_new_area":{}}}"#,
            f.snap_area[i]
        ));
    }
    let align: Vec<String> = f
        .align_regions
        .iter()
        .map(|&r| {
            let i = r.min(1);
            format!(
                r#"{{"region":{r},"pairs":{},"fit_median_m":{},"scale":{}}}"#,
                f.align_pairs[i], f.align_fit[i], f.align_scale[i]
            )
        })
        .collect();
    std::fs::write(
        dir.join("snapshots/manifest.json"),
        format!(
            r#"{{"snapshots":[{}],"align":[{}]}}"#,
            snaps.join(","),
            align.join(",")
        ),
    )
    .unwrap();
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("skylens_verify_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(tag: &str, f: &Fixture) -> (i32, String) {
    run_with(tag, f, |_| {})
}

/// fixture 를 만든 뒤 `edit` 로 폴더를 고치고 verify 를 돌린다.
fn run_with(tag: &str, f: &Fixture, edit: impl FnOnce(&Path)) -> (i32, String) {
    let dir = tmp(tag);
    build(&dir, f);
    edit(&dir);
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("verify")
        .arg(&dir)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
    )
}

const ITEMS: [&str; 7] = [
    "registered",
    "region_images",
    "refined_reprojection",
    "preview_align",
    "preview_vs_refined",
    "refined_overlap",
    "snapshots",
];

fn status(out: &str, item: &str) -> &'static str {
    let line = out
        .lines()
        .find(|l| l.starts_with(&format!("| {item} |")))
        .unwrap_or_else(|| panic!("{item} 줄 없음:\n{out}"));
    if line.contains("| PASS |") {
        "PASS"
    } else {
        "FAIL"
    }
}

/// 기대: `failing` 항목만 FAIL, 나머지 PASS, 종료 코드 1.
fn expect_only_fail(tag: &str, f: &Fixture, failing: &str) {
    let (code, out) = run(tag, f);
    assert_eq!(code, 1, "{out}");
    for it in ITEMS {
        let want = if it == failing { "FAIL" } else { "PASS" };
        assert_eq!(status(&out, it), want, "{it}\n{out}");
    }
}

#[test]
fn passing_output_exits_zero() {
    let (code, out) = run("pass", &Fixture::default());
    assert_eq!(code, 0, "{out}");
    for it in ITEMS {
        assert_eq!(status(&out, it), "PASS", "{it}\n{out}");
    }
    assert!(out.contains("결과: 7/7 통과"), "{out}");
    assert!(out.contains("초벌 240/240, 정밀 240/240"), "{out}");
    assert!(out.contains("초벌·정밀 모두 전체 등록 (240/240)"), "{out}");
    assert!(out.contains("정밀 0.700 px"), "{out}");
    assert!(out.contains("점쌍 최소 1000"), "{out}");
    assert!(out.contains("구역 간 스케일 차 10.00%"), "{out}");
    assert!(out.contains("잔차 중앙 최대 5.990 m"), "{out}");
    // 초벌 = 정밀 + 1.99 m: 최근접·높이 차 모두 1.990 (f32 반올림).
    assert!(out.contains("최근접 중앙 최대 1.990 m"), "{out}");
    assert!(out.contains("높이 차 중앙 최대 1.990 m"), "{out}");
    assert!(out.contains("겹침 차 중앙 최대 0.290 m"), "{out}");
    assert!(out.contains("점 100→150, final 1722"), "{out}");
}

#[test]
fn registered_239_fails() {
    let f = Fixture {
        reg_preview: 239,
        ..Default::default()
    };
    expect_only_fail("reg_p", &f, "registered");
    let f = Fixture {
        reg_refined: 239,
        ..Default::default()
    };
    expect_only_fail("reg_r", &f, "registered");
}

#[test]
fn region_images_mismatch_fails() {
    let f = Fixture {
        region_images: [42, 47],
        ..Default::default()
    };
    expect_only_fail("img", &f, "region_images");
}

#[test]
fn reprojection_above_0_7_fails() {
    let f = Fixture {
        reproj_refined: 0.701,
        ..Default::default()
    };
    expect_only_fail("reproj", &f, "refined_reprojection");
}

#[test]
fn align_boundaries() {
    let f = Fixture {
        align_pairs: [999, 2000],
        ..Default::default()
    };
    expect_only_fail("pairs", &f, "preview_align");
    let f = Fixture {
        align_fit: [6.0, 1.0],
        ..Default::default()
    };
    expect_only_fail("fit", &f, "preview_align");
    let f = Fixture {
        align_scale: [1.0, 1.11],
        ..Default::default()
    };
    expect_only_fail("scale_hi", &f, "preview_align");
    // 구역 간 비 max/min − 1: 1/0.91 = 1.0989 통과, 1.06/0.95 = 1.116 과 1.1/0.9 = 1.222 실패.
    let f = Fixture {
        align_scale: [1.0, 0.91],
        ..Default::default()
    };
    assert_eq!(run("scale_lo_ok", &f).0, 0);
    let f = Fixture {
        align_scale: [0.95, 1.06],
        ..Default::default()
    };
    expect_only_fail("scale_spread", &f, "preview_align");
    let f = Fixture {
        align_scale: [0.9, 1.1],
        ..Default::default()
    };
    expect_only_fail("scale_lo", &f, "preview_align");
}

#[test]
fn preview_height_difference_2m_fails() {
    // 높이 차 2.0 m 는 "< 2 m" 를 어긴다 (최근접 2.0 m 는 < 3 m 라 통과해도 항목은 실패).
    let f = Fixture {
        preview_dz: 2.0,
        ..Default::default()
    };
    let (code, out) = run("dz2", &f);
    assert_eq!(code, 1);
    assert_eq!(status(&out, "preview_vs_refined"), "FAIL", "{out}");
    assert!(out.contains("높이 차 중앙 최대 2.000 m"), "{out}");
    assert!(out.contains("최근접 중앙 최대 2.000 m"), "{out}");
}

#[test]
fn preview_nearest_median_fails_with_far_points() {
    // 짝 없는 먼 점이 절반을 넘으면 최근접 중앙은 상한(> 6 m), 높이 차(짝 있는 점만)는 1.99 m 그대로.
    let f = Fixture {
        preview_outliers: true,
        ..Default::default()
    };
    expect_only_fail("nn", &f, "preview_vs_refined");
    let (_, out) = run("nn2", &f);
    assert!(out.contains("최근접 중앙 최대 > 6.000 m"), "{out}");
    assert!(out.contains("높이 차 중앙 최대 1.990 m"), "{out}");
}

#[test]
fn refined_overlap_0_3_fails() {
    let f = Fixture {
        refined1_z: 0.3,
        ..Default::default()
    };
    expect_only_fail("ovl", &f, "refined_overlap");
}

#[test]
fn snapshot_failures() {
    let f = Fixture {
        snap_points: [100, 90, 1722],
        ..Default::default()
    };
    expect_only_fail("mono", &f, "snapshots");
    // F-157: final 은 정밀 점 수 합(861 + 861 = 1722)과 같아야 한다. 1 점 차이도, 다시 6:1
    // 추출한 값(144 + 144 = 288)도 실패.
    for (tag, n) in [("final_sum", 1723), ("final_redecimated", 288)] {
        let f = Fixture {
            snap_points: [100, 150, n],
            ..Default::default()
        };
        expect_only_fail(tag, &f, "snapshots");
    }
    let (_, out) = run(
        "final_288_msg",
        &Fixture {
            snap_points: [100, 150, 288],
            ..Default::default()
        },
    );
    assert!(out.contains("final 288 ≠ 정밀 점 수 합 1722"), "{out}");
    let f = Fixture {
        snap_area: [0.0, 0.0, 30.0],
        ..Default::default()
    };
    expect_only_fail("area", &f, "snapshots");
    let f = Fixture {
        snap_nan: true,
        ..Default::default()
    };
    expect_only_fail("nan", &f, "snapshots");
}

#[test]
fn missing_folder_fails_every_item() {
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["verify", "/nonexistent/skylens_out"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let s = String::from_utf8(out.stdout).unwrap();
    for it in ITEMS {
        assert_eq!(status(&s, it), "FAIL", "{it}\n{s}");
    }
    assert!(s.contains("결과: 0/7 통과"), "{s}");
}

/// F-088·F-118: final 은 정밀 전부라 마지막 step(초벌 포함)보다 작을 수 있다. 같은 수 정체도 허용.
#[test]
fn final_smaller_than_last_step_passes() {
    let f = Fixture {
        snap_points: [100, 1800, 1722],
        ..Default::default()
    };
    let (code, out) = run("final_small", &f);
    assert_eq!(code, 0, "{out}");
    let f = Fixture {
        snap_points: [150, 150, 1722],
        ..Default::default()
    };
    assert_eq!(run("flat", &f).0, 0);
}

/// F-087: 구역 2개 × 정밀 9만 점(0.2 m 격자), 초벌 = 정밀 + 30 m → 10 s 안에 종료 코드 1.
#[test]
fn far_preview_finishes_quickly_and_fails() {
    let f = Fixture {
        cell: 0.2,
        nx: 300,
        ny: 300,
        preview_dz: 30.0,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let (code, out) = run("far30", &f);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "preview_vs_refined"), "FAIL", "{out}");
    assert!(out.contains("최근접 중앙 최대 > 6.000 m"), "{out}");
    assert!(secs < 10.0, "verify {secs:.2} s");
    eprintln!("far30 verify {secs:.2} s");
}

/// F-097: report 에 구역 2개, 디스크에 preview_00·refined_00 만, manifest 단계 1·2·final, 디스크에 step_01 만.
#[test]
fn missing_region_and_snapshot_files_fail() {
    let (code, out) = run_with("missing", &Fixture::default(), |d| {
        for n in [
            "preview/preview_01_pos10-26.ply",
            "refined/refined_01_pos10-26.ply",
            "snapshots/step_02_2regions.ply",
            "snapshots/step_final_all_refined.ply",
        ] {
            std::fs::remove_file(d.join(n)).unwrap();
        }
    });
    assert_eq!(code, 1, "{out}");
    for it in ["preview_vs_refined", "refined_overlap", "snapshots"] {
        assert_eq!(status(&out, it), "FAIL", "{it}\n{out}");
    }
    assert!(out.contains("초벌 없는 구역 [1]"), "{out}");
    assert!(out.contains("정밀 없는 구역 [1]"), "{out}");
    assert!(
        out.contains("빠진 파일 step_02_2regions.ply,step_final_all_refined.ply"),
        "{out}"
    );
}

/// F-097: manifest 에는 단계 1·final 만 있는데 구역은 2개 → 단계 2 누락.
#[test]
fn manifest_missing_step_fails() {
    let (code, out) = run_with("missing_step", &Fixture::default(), |d| {
        let p = d.join("snapshots/manifest.json");
        let m = std::fs::read_to_string(&p).unwrap();
        let m = m.replace(r#"{"step":2,"points":150,"preview_new_area":12.5},"#, "");
        std::fs::write(&p, m).unwrap();
        std::fs::remove_file(d.join("snapshots/step_02_2regions.ply")).unwrap();
    });
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "snapshots"), "FAIL", "{out}");
    assert!(out.contains("manifest 에 없는 단계 [2]"), "{out}");
}

/// F-066: step 이 문자열 "01" 이면 형식 오류로 FAIL.
#[test]
fn string_step_is_format_error() {
    let (code, out) = run_with("str_step", &Fixture::default(), |d| {
        let p = d.join("snapshots/manifest.json");
        let m = std::fs::read_to_string(&p).unwrap();
        std::fs::write(&p, m.replace(r#""step":1,"#, r#""step":"01","#)).unwrap();
    });
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "snapshots"), "FAIL", "{out}");
    assert!(out.contains("step 형식 오류"), "{out}");
}

/// F-118: refined 폴더가 비면 겹침 항목도 FAIL.
#[test]
fn empty_refined_fails_overlap() {
    let (code, out) = run_with("no_refined", &Fixture::default(), |d| {
        for e in std::fs::read_dir(d.join("refined")).unwrap() {
            std::fs::remove_file(e.unwrap().path()).unwrap();
        }
    });
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "refined_overlap"), "FAIL", "{out}");
    assert!(out.contains("정밀 구역 0개"), "{out}");
}

fn global_median_z(points: &[PointRecord]) -> f64 {
    let mut z: Vec<f64> = points.iter().map(|p| p.xyz[2] as f64).collect();
    z.sort_by(f64::total_cmp);
    z[z.len() / 2]
}

/// F-117: 초벌·정밀의 xy 범위가 다르고 지면이 기울어 있어 점군 전체 중앙값끼리는 2 m 넘게
/// 벌어지지만 같은 위치 짝 높이 차는 1.99 m → 통과해야 한다 (전체 중앙값 비교 구현은 실패).
#[test]
fn sloped_ground_with_different_extent_passes() {
    let f = Fixture {
        slope: 0.5,
        preview_extra_x: 10.0,
        ..Default::default()
    };
    let refined = grid(&f, 0.0, 0.0, f.nx);
    let preview = grid(&f, 0.0, f.preview_dz, f.nx + 20);
    let gap = global_median_z(&preview) - global_median_z(&refined);
    assert!(gap > 2.0, "전체 중앙값 차 {gap}");
    let (code, out) = run("slope", &f);
    assert_eq!(code, 0, "{out}");
    assert_eq!(status(&out, "preview_vs_refined"), "PASS", "{out}");
    assert!(out.contains("높이 차 중앙 최대 1.990 m"), "{out}");
}

/// F-089: SPEC §2 출력만 있는 폴더(report.json 없음) → 1~3 판정 불가, 4~7 은 측정값으로 PASS, 종료 코드 2.
#[test]
fn spec_outputs_only_marks_report_items_undecided() {
    let (code, out) = run_with("noreport", &Fixture::default(), |d| {
        std::fs::remove_file(d.join("report.json")).unwrap();
    });
    assert_eq!(code, 2, "{out}");
    for it in &ITEMS[..3] {
        let line = out
            .lines()
            .find(|l| l.starts_with(&format!("| {it} |")))
            .unwrap();
        assert!(line.contains("| 판정 불가 |"), "{it}\n{out}");
    }
    for it in &ITEMS[3..] {
        assert_eq!(status(&out, it), "PASS", "{it}\n{out}");
    }
    assert!(out.contains("결과: 4/7 통과"), "{out}");
    assert!(out.contains("판정 불가: 3개"), "{out}");
    assert!(out.contains("최근접 중앙 최대 1.990 m"), "{out}");
    // 사진 수는 출력에 없지만 위치 수는 파일 이름 pos0-14·pos10-26 (hi 미포함)에서 읽어 보여 준다.
    assert!(out.contains("파일 이름의 위치 수 [0:14 1:16]"), "{out}");
}

/// F-089: report.json 없이 판정 가능한 항목이 FAIL 이면 종료 코드 1.
#[test]
fn spec_outputs_only_with_failure_exits_one() {
    let f = Fixture {
        preview_dz: 2.5,
        ..Fixture::default()
    };
    let (code, out) = run_with("noreport_fail", &f, |d| {
        std::fs::remove_file(d.join("report.json")).unwrap();
    });
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "preview_vs_refined"), "FAIL", "{out}");
}

/// F-089: 형식이 깨진 report.json 은 판정 불가가 아니라 FAIL.
#[test]
fn broken_report_json_fails() {
    let (code, out) = run_with("badreport", &Fixture::default(), |d| {
        std::fs::write(d.join("report.json"), "{not json").unwrap();
    });
    assert_eq!(code, 1, "{out}");
    for it in &ITEMS[..3] {
        assert_eq!(status(&out, it), "FAIL", "{it}\n{out}");
    }
}

/// F-156: manifest `align` 의 구역 집합이 출력 구역 집합과 같아야 한다.
#[test]
fn align_records_must_cover_every_region() {
    // 구역 2개에 정렬 기록 1개(구역 0) → 구역 1 누락 FAIL, 종료 1.
    let f = Fixture {
        align_regions: &[0],
        ..Default::default()
    };
    expect_only_fail("align_missing", &f, "preview_align");
    let (code, out) = run("align_missing_msg", &f);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("정렬 기록 없는 구역 [1]"), "{out}");
    // 출력에 없는 구역 2 의 기록이 남아도 FAIL.
    let f = Fixture {
        align_regions: &[0, 1, 2],
        ..Default::default()
    };
    expect_only_fail("align_extra", &f, "preview_align");
    let (_, out) = run("align_extra_msg", &f);
    assert!(out.contains("출력에 없는 구역의 정렬 기록 [2]"), "{out}");
    // 같은 구역 기록이 두 번이면 FAIL (구역 1 누락도 함께).
    let f = Fixture {
        align_regions: &[0, 0],
        ..Default::default()
    };
    expect_only_fail("align_dup", &f, "preview_align");
    // 기본(구역 0·1 각 1개) 은 PASS.
    let (code, out) = run("align_ok", &Fixture::default());
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("정렬 기록 2개"), "{out}");
}

/// 구역 1개짜리 폴더를 직접 만든다(report.json 없음): 정밀 = `refined`, 초벌 = `preview`,
/// manifest = `manifest` 문자열, 스냅샷 PLY 는 `snaps` 이름마다 점 1개.
fn one_region_dir(
    tag: &str,
    preview: Vec<PointRecord>,
    refined: Vec<PointRecord>,
    manifest: &str,
    snaps: &[&str],
) -> PathBuf {
    let dir = tmp(tag);
    for sub in ["preview", "refined", "snapshots"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    write(&dir.join("preview/preview_00_pos0-14.ply"), preview);
    write(&dir.join("refined/refined_00_pos0-14.ply"), refined);
    for n in snaps {
        write(&dir.join("snapshots").join(n), vec![rec(0.0, 0.0, 0.0)]);
    }
    std::fs::write(dir.join("snapshots/manifest.json"), manifest).unwrap();
    dir
}

/// verify 를 돌려 (종료 코드, 표준 출력 바이트 수, 표준 출력, 걸린 초).
fn verify_timed(dir: &Path) -> (i32, usize, String, f64) {
    let t0 = std::time::Instant::now();
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("verify")
        .arg(dir)
        .output()
        .unwrap();
    let secs = t0.elapsed().as_secs_f64();
    std::fs::remove_dir_all(dir).unwrap();
    (
        out.status.code().unwrap(),
        out.stdout.len(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        secs,
    )
}

const ONE_STEP_MANIFEST: &str = r#"{"snapshots":[{"step":1,"points":1,"preview_new_area":0},{"step":"final","points":90000}],"align":[{"region":0,"pairs":2000,"fit_median_m":1.0,"scale":1.0}]}"#;

/// F-180: 정밀 9만 점이 모두 (10,10,0), 초벌은 그 둘레 8 m 정사각형 300×300 격자(z 0.5).
/// 정답: 초벌 점의 최근접은 모두 (10,10,0) 이고 거리 = √(r² + 0.25). 한 변 8 m 정사각형에서
/// 수평 거리 중앙 r ≈ √(32/π) = 3.19 m → 최근접 중앙 ≈ 3.23 m > 3 m 이므로 FAIL(종료 1).
/// 고치기 전 같은 입력에서 39.4 s. 상한 1 s 는 정답 비교와 별개의 시간 기준(같은 좌표 점이
/// 질의마다 잎 하나만 보게 되면 수십 ms 수준).
#[test]
fn identical_refined_points_finish_quickly() {
    let refined = vec![rec(10.0, 10.0, 0.0); 90_000];
    let mut preview = Vec::with_capacity(90_000);
    for i in 0..300 {
        for j in 0..300 {
            let x = 6.0 + 8.0 * i as f32 / 299.0;
            let y = 6.0 + 8.0 * j as f32 / 299.0;
            preview.push(rec(x, y, 0.5));
        }
    }
    let dir = one_region_dir(
        "same_xyz",
        preview,
        refined,
        ONE_STEP_MANIFEST,
        &["step_01_1regions.ply"],
    );
    let (code, _, out, secs) = verify_timed(&dir);
    eprintln!("same_xyz verify {secs:.2} s");
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "preview_vs_refined"), "FAIL", "{out}");
    let nn: f64 = out
        .split("최근접 중앙 최대 ")
        .nth(1)
        .and_then(|t| t.split(" m").next())
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or_else(|| panic!("{out}"));
    assert!((nn - 3.23).abs() < 0.05, "최근접 중앙 {nn}\n{out}");
    assert!(secs < 1.0, "verify {secs:.2} s");
}

/// F-181: manifest 의 step 하나가 3천만 → 예전에는 빠진 단계 3천만 개를 다 찍어 16 s·1.16 GB.
/// 이제 구역 수 + 1 에서 잘라 FAIL 하고 목록은 앞 몇 개와 개수만.
#[test]
fn huge_step_fails_with_small_output() {
    let m = r#"{"snapshots":[{"step":30000000,"points":1,"preview_new_area":1},{"step":"final","points":1}],"align":[{"region":0,"pairs":2000,"fit_median_m":1.0,"scale":1.0}]}"#;
    let g = vec![rec(0.0, 0.0, 0.0)];
    let dir = one_region_dir("huge_step", g.clone(), g, m, &[]);
    let (code, bytes, out, secs) = verify_timed(&dir);
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "snapshots"), "FAIL", "{out}");
    assert!(out.contains("구역 수 1 초과 단계 [30000000]"), "{out}");
    assert!(bytes < 1 << 20, "출력 {bytes} B");
    assert!(secs < 1.0, "verify {secs:.2} s");
}

/// F-181: manifest 가 `{"snapshots":` 뒤에 `[` 30만 개 → 예전에는 스택 넘침(종료 134).
/// 이제 형식 오류로 FAIL(종료 1), 표가 나온다.
#[test]
fn deeply_nested_manifest_fails_without_crash() {
    let m = format!("{{\"snapshots\":{}", "[".repeat(300_000));
    let g = vec![rec(0.0, 0.0, 0.0)];
    let dir = one_region_dir("deep_json", g.clone(), g, &m, &[]);
    let (code, bytes, out, secs) = verify_timed(&dir);
    assert_eq!(code, 1, "{out}");
    assert_eq!(status(&out, "snapshots"), "FAIL", "{out}");
    assert!(out.contains("중첩 깊이 64 초과"), "{out}");
    assert!(bytes < 1 << 20, "출력 {bytes} B");
    assert!(secs < 1.0, "verify {secs:.2} s");
}

#[test]
fn registered_criterion_shows_scene_photo_count() {
    let f = Fixture {
        total: 81,
        reg_preview: 81,
        reg_refined: 81,
        ..Fixture::default()
    };
    let (code, out) = run("total81", &f);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("초벌 81/81, 정밀 81/81"), "{out}");
    assert!(out.contains("초벌·정밀 모두 전체 등록 (81/81)"), "{out}");
    assert!(!out.contains("(240/240)"), "{out}");
}
