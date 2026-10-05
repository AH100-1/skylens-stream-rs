//! 시드별 기본 경로(인자 없는 synth → run → verify) 판정표 출력. 오래 걸려 기본으로는 돌리지 않는다.
//! 실행: `SKYLENS_TILT_SEED=4 cargo test --release -p skylens-stream --test seed_verify -- --ignored --nocapture`

use std::process::Command;

use skylens_core::synth::{Scene, SceneConfig};

#[test]
#[ignore]
fn default_path_seed_verify() {
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let root = std::env::temp_dir().join(format!("skylens_seed_verify_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let exe = env!("CARGO_BIN_EXE_skylens-stream");
    let run = Command::new(exe)
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_REGION_DIAG", "1")
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0));
    let v = Command::new(exe)
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    eprintln!("seed {seed}");
    eprintln!("{}", String::from_utf8_lossy(&v.stdout));
    eprintln!("verify exit {:?}", v.status.code());
    let _ = std::fs::remove_dir_all(&root);
}
