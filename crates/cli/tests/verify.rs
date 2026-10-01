//! `skylens-stream verify` 시험: 작은 통과용 출력 폴더를 만들고 항목마다 하나씩 깨뜨린다.

use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::ply::{write_ply_file, PointCloud, PointRecord};

#[derive(Clone)]
struct Fixture {
    reg_preview: u32,
    reg_refined: u32,
    region_images: [u32; 2],
    reproj_refined: f64,
    align_pairs: [u32; 3],
    align_fit: [f64; 3],
    align_scale: [f64; 3],
    /// 초벌 점군 = 정밀 점군 xy 그대로, z 에 이 값을 더함.
    preview_dz: f32,
    /// 초벌 점군에 짝 없는 먼 점을 (격자 점 수 + 1)개 덧붙임 → 최근접 중앙이 커짐.
    preview_outliers: bool,
    /// 정밀 구역 1 의 높이 (구역 0 은 0).
    refined1_z: f32,
    snap_points: [usize; 3],
    snap_area: [f64; 3],
    snap_nan: bool,
}

impl Default for Fixture {
    fn default() -> Self {
        Fixture {
            reg_preview: 240,
            reg_refined: 240,
            region_images: [42, 48],
            reproj_refined: 0.7,
            align_pairs: [1000, 1500, 2000],
            align_fit: [5.99, 2.0, 1.0],
            align_scale: [1.0, 1.0, 1.1],
            preview_dz: 1.99,
            preview_outliers: false,
            refined1_z: 0.29,
            snap_points: [100, 150, 200],
            snap_area: [0.0, 12.5, 30.0],
            snap_nan: false,
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

/// x ∈ [x0, x0+20], y ∈ [0, 10], 0.5 m 간격 격자.
fn grid(x0: f32, z: f32) -> Vec<PointRecord> {
    let mut v = Vec::new();
    for i in 0..=40 {
        for j in 0..=20 {
            v.push(rec(x0 + i as f32 * 0.5, j as f32 * 0.5, z));
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
            r#"{{"registered":{{"total":240,"preview":{},"refined":{}}},
"reprojection_px":{{"preview":4.5,"refined":{}}},
"regions":[{{"region":0,"positions":14,"images":{}}},{{"region":1,"positions":16,"images":{}}}]}}"#,
            f.reg_preview, f.reg_refined, f.reproj_refined, f.region_images[0], f.region_images[1]
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
            grid(x0, z),
        );
        let mut p = grid(x0, z + f.preview_dz);
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
    let align: Vec<String> = (0..3)
        .map(|r| {
            format!(
                r#"{{"region":{r},"pairs":{},"fit_median_m":{},"scale":{}}}"#,
                f.align_pairs[r], f.align_fit[r], f.align_scale[r]
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
    let dir = tmp(tag);
    build(&dir, f);
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
    assert!(out.contains("정밀 0.700 px"), "{out}");
    assert!(out.contains("점쌍 최소 1000"), "{out}");
    assert!(out.contains("스케일 최대 편차 10.00%"), "{out}");
    assert!(out.contains("잔차 중앙 최대 5.990 m"), "{out}");
    // 초벌 = 정밀 + 1.99 m: 최근접·높이 차 모두 1.990 (f32 반올림).
    assert!(out.contains("최근접 중앙 최대 1.990 m"), "{out}");
    assert!(out.contains("높이 차 중앙 최대 1.990 m"), "{out}");
    assert!(out.contains("겹침 차 중앙 최대 0.290 m"), "{out}");
    assert!(out.contains("점 100→200"), "{out}");
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
        align_pairs: [999, 1500, 2000],
        ..Default::default()
    };
    expect_only_fail("pairs", &f, "preview_align");
    let f = Fixture {
        align_fit: [6.0, 2.0, 1.0],
        ..Default::default()
    };
    expect_only_fail("fit", &f, "preview_align");
    let f = Fixture {
        align_scale: [1.0, 1.0, 1.11],
        ..Default::default()
    };
    expect_only_fail("scale_hi", &f, "preview_align");
    // 중앙 1.0 대비 −10% 는 통과, −11% 는 실패.
    let f = Fixture {
        align_scale: [1.0, 1.0, 0.9],
        ..Default::default()
    };
    assert_eq!(run("scale_lo_ok", &f).0, 0);
    let f = Fixture {
        align_scale: [1.0, 1.0, 0.89],
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
    // 짝 없는 먼 점이 절반을 넘으면 최근접 중앙은 상한 20 m, 높이 차(짝 있는 점만)는 1.99 m 그대로.
    let f = Fixture {
        preview_outliers: true,
        ..Default::default()
    };
    expect_only_fail("nn", &f, "preview_vs_refined");
    let (_, out) = run("nn2", &f);
    assert!(out.contains("최근접 중앙 최대 20.000 m"), "{out}");
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
        snap_points: [100, 100, 200],
        ..Default::default()
    };
    expect_only_fail("mono", &f, "snapshots");
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
