//! 구역 3개짜리 작은 합성 장면: 구역 순서 처리, 건너뛴 구역, 정지 구간, 겹침 중심 차이.

use std::path::{Path, PathBuf};

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};

const POSITIONS: usize = 26;

fn cfg() -> PipelineConfig {
    PipelineConfig {
        max_features: 600,
        dense_width: 80,
        hfov_deg: 65.0,
        ba_iters: 8,
    }
}

fn scene_dir(tag: &str) -> (PathBuf, Scene) {
    let root = std::env::temp_dir().join(format!("skylens_regions_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let scene = Scene::new(SceneConfig {
        positions: POSITIONS,
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    scene.write_dataset(&root.join("in")).unwrap();
    (root, scene)
}

fn dataset(root: &Path) -> skylens_core::dataset::Dataset {
    load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 8,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap()
}

fn flat(path: &Path) {
    image::RgbImage::from_pixel(320, 180, image::Rgb([90, 90, 90]))
        .save(path)
        .unwrap();
}

#[test]
fn regions_in_order_with_realign_and_one_line_per_photo() {
    let (root, _) = scene_dir("order");
    let ds = dataset(&root);
    let res = run_pipeline(&ds, &cfg(), &root.join("out")).unwrap();
    for i in &res.issues {
        eprintln!("issue {i}");
    }
    let report = std::fs::read_to_string(root.join("out/report.json")).unwrap();
    eprintln!("{report}");
    assert!(res.regions.len() >= 3, "구역 {}", res.regions.len());
    // 첫 구역 초벌이 두 번째 구역 도착보다 먼저 나간다.
    let pos = |s: &str| report.find(s).unwrap_or_else(|| panic!("{s}"));
    assert!(pos("coarse output region 0") < pos("arrive region 1"));
    assert!(pos("coarse output region 0") < pos("coarse output region 1"));
    assert!(report.contains("\"realign_count\""));
    assert!(!report.contains("\"realign_count\": 0,"), "재정렬 0회");
    // 사진마다 한 줄.
    let poses = std::fs::read_to_string(root.join("out/poses.txt")).unwrap();
    let lines = poses.lines().count();
    let mut names: Vec<&str> = poses
        .lines()
        .map(|l| l.split(' ').next().unwrap())
        .collect();
    names.sort();
    names.dedup();
    assert_eq!(lines, names.len(), "중복 줄");
    assert_eq!(lines, res.centers.len());
    assert!(lines >= 20 && lines <= ds.image_count(), "{lines}");
    assert!(!report.contains("\"overlap_center_diff_median_m\": null"));
    assert!(res
        .issues
        .iter()
        .any(|i| i.contains("겹침 구간 사진 중심 차이")));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn failed_middle_region_is_skipped_not_fatal() {
    let (root, _) = scene_dir("flatmid");
    let ds = dataset(&root);
    // 구역 1 = 위치 [6, 18).
    for p in 6..18 {
        for img in &ds.positions[p].images {
            flat(img);
        }
    }
    let res = run_pipeline(&ds, &cfg(), &root.join("out")).unwrap();
    for i in &res.issues {
        eprintln!("issue {i}");
    }
    assert!(
        res.issues.iter().any(|i| i.contains("구역 1 건너뜀")),
        "건너뜀 기록 없음"
    );
    let ids: Vec<usize> = res.regions.iter().map(|r| r.region).collect();
    assert!(ids.contains(&0) && ids.contains(&2), "{ids:?}");
    assert!(!ids.contains(&1));
    assert!(root
        .join("out/snapshots/step_final_all_refined.ply")
        .is_file());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn stationary_segment_is_error_or_issue() {
    let (root, _) = scene_dir("hover");
    let mut ds = dataset(&root);
    // 구역 1 [6,18) 의 GPS 를 한 점에 고정: 체공.
    let anchor = ds.positions[6].image_enu;
    for p in 6..18 {
        ds.positions[p].image_enu = anchor;
    }
    match run_pipeline(&ds, &cfg(), &root.join("out")) {
        Err(_) => {}
        Ok(res) => assert!(
            res.issues.iter().any(|i| i.contains("정지 구간")),
            "{:?}",
            res.issues
        ),
    }
    let _ = std::fs::remove_dir_all(&root);
}
