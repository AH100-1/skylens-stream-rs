//! 시드 5 기본 경로에서 정밀 모델에 등록되지 않은 사진을 찾는 진단(오래 걸려 기본으로는 돌리지 않는다).
//! 실행: `SKYLENS_TILT_SEED=5 cargo test --release -p skylens-stream --test seed5_last_frame -- --ignored --nocapture`

use std::collections::BTreeSet;
use std::process::Command;

use skylens_core::synth::{Scene, SceneConfig};

#[test]
#[ignore]
fn missing_photos_seed() {
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let root = std::env::temp_dir().join(format!("skylens_last_frame_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_REGION_DIAG", "1")
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0));
    let all: Vec<String> = std::fs::read_to_string(input.join("gps.txt"))
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().next().unwrap().to_string())
        .collect();
    let got: BTreeSet<String> = std::fs::read_to_string(output.join("poses.txt"))
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().next().unwrap().to_string())
        .collect();
    eprintln!("seed {seed} images {} registered {}", all.len(), got.len());
    for (i, n) in all.iter().enumerate() {
        if !got.contains(n) {
            eprintln!("MISSING index {i} name {n}");
        }
    }
    eprintln!("--- stdout ---\n{}", String::from_utf8_lossy(&run.stdout));
    let dir = std::env::temp_dir().join("seed5_last_frame_stderr.txt");
    std::fs::write(dir, &run.stderr).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}
