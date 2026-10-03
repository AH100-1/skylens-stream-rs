//! 3구역 합성 장면: 스냅샷 점 수 단조 증가·NaN 없음, 재정렬 잔차, 구역별 등록 현황.

use std::path::{Path, PathBuf};

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::ply::read_ply_file;
use skylens_core::stream::Manifest;
use skylens_core::synth::{Scene, SceneConfig};

fn setup(tag: &str) -> (PathBuf, skylens_core::dataset::Dataset) {
    let root = std::env::temp_dir().join(format!("skylens_pstream_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    Scene::new(SceneConfig {
        positions: 26,
        width: 320,
        height: 180,
        ..SceneConfig::default()
    })
    .write_dataset(&root.join("in"))
    .unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 8,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    (root, ds)
}

fn plies(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ply"))
        .collect();
    v.sort();
    v
}

#[test]
fn snapshots_grow_without_nan_and_realign_residuals_small() {
    let (root, ds) = setup("grow");
    let cfg = PipelineConfig {
        max_features: 600,
        dense_width: 80,
        hfov_deg: 65.0,
        ba_iters: 8,
    };
    let out = root.join("out");
    let res = run_pipeline(&ds, &cfg, &out).unwrap();
    let m =
        Manifest::from_json(&std::fs::read_to_string(out.join("snapshots/manifest.json")).unwrap())
            .unwrap();
    let pts: Vec<usize> = m.snapshots.iter().map(|s| s.points).collect();
    eprintln!("snapshot points {pts:?}");
    assert!(pts.len() >= 3);
    assert!(pts.windows(2).all(|w| w[1] >= w[0]), "단조 아님 {pts:?}");
    for sub in ["preview", "refined", "snapshots"] {
        for p in plies(&out.join(sub)) {
            let c = read_ply_file(&p).unwrap();
            assert!(!c.is_empty() && !c.has_nan(), "{p:?}");
        }
    }
    let report = std::fs::read_to_string(out.join("report.json")).unwrap();
    eprintln!("{report}");
    assert!(
        report.contains("realign refined"),
        "정밀 구역 재정렬 기록 없음"
    );
    assert!(report.contains("registration table region 0"));
    // 정밀 구역 재정렬 잔차 중앙값(m): 공유 3D 점이 같은 좌표계라 작아야 한다.
    let meds: Vec<f64> = report
        .split("realign refined ")
        .skip(1)
        .map(|t| {
            let t = t.split(" median ").nth(1).unwrap();
            t.split(' ').next().unwrap().parse::<f64>().unwrap()
        })
        .collect();
    eprintln!("refined realign medians {meds:?}");
    assert!(!meds.is_empty());
    assert!(meds.iter().all(|m| m.is_finite() && *m <= 0.3), "{meds:?}");
    assert!(res.regions.len() >= 3);
    let _ = std::fs::remove_dir_all(&root);
}
