//! 2구역 합성 장면 정렬 수치 측정(느림: `--ignored`). 환경 변수 SEED, SKYLENS_REGION_SIM3.

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};

#[test]
#[ignore]
fn two_region_alignment_numbers() {
    let seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let root = std::env::temp_dir().join(format!(
        "skylens_sim3_measure_{seed}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    Scene::new(SceneConfig {
        positions: 80,
        width: 320,
        height: 180,
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&root.join("in"))
    .unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 48,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    let cfg = PipelineConfig {
        max_features: 800,
        dense_width: 96,
        hfov_deg: 65.0,
        ba_iters: 15,
        ..PipelineConfig::default()
    };
    let res = run_pipeline(&ds, &cfg, &root.join("out")).unwrap();
    let mut sc = vec![];
    for a in &res.align {
        eprintln!(
            "ALIGN region {} pairs {} fit {:?} scale {:?}",
            a.region, a.pairs, a.fit_median_m, a.scale
        );
        sc.extend(a.scale);
    }
    let lo = sc.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("SPREAD {:.2}%", (hi / lo - 1.0) * 100.0);
    for i in &res.issues {
        eprintln!("ISSUE {i}");
    }
    let _ = std::fs::remove_dir_all(&root);
}
