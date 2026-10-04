//! 480x270 합성 장면에서 확대 채움 + 특징 3000 + 카메라 쌍 투표가 81장 모두 등록하는지 본다.
//! 오래 걸려 기본 시험에서 뺀다: `cargo test --release -p skylens-stream --test vote_upscale -- --ignored --nocapture`.

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};

#[test]
#[ignore = "480x270 전체 복원, 수 분 걸림"]
fn upscale_fill_with_pair_vote_registers_all_81() {
    let root = std::env::temp_dir().join(format!("skylens_vote_upscale_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    Scene::new(SceneConfig {
        width: 480,
        height: 270,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let ds = load_dataset(
        &input,
        DatasetConfig {
            stride: 3,
            span: 48,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    assert_eq!(ds.image_count(), 81);
    let cfg = PipelineConfig {
        max_features: 3000,
        upscale_fill: true,
        pair_vote: true,
        dense_width: 96,
        hfov_deg: 65.0,
        ba_iters: 15,
        ..PipelineConfig::default()
    };
    let res = run_pipeline(&ds, &cfg, &output).unwrap();
    eprintln!("registered {} of {}", res.centers.len(), ds.image_count());
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(res.centers.len(), 81, "등록 수");
}
