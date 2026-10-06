//! 시드5 기본 경로: 81/81 등록, verify 7/7, 반복 실행 poses 동일(SEED5_RUNS 회, 기본 3). 한 번에 약 6~9분.
//! 실행: `cargo test --release -p skylens-stream --test seed5_drop_trace -- --ignored --nocapture`

use std::process::Command;

use skylens_core::synth::{Scene, SceneConfig};

#[test]
#[ignore]
fn seed5_repeat_runs() {
    let root = std::env::temp_dir().join(format!("skylens_seed5_trace_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let input = root.join("in");
    Scene::new(SceneConfig {
        seed: 5,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let exe = env!("CARGO_BIN_EXE_skylens-stream");
    let runs: usize = std::env::var("SEED5_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let mut posesets = Vec::new();
    for r in 0..runs {
        let out = root.join(format!("out{r}"));
        let run = Command::new(exe)
            .args(["run", input.to_str().unwrap(), out.to_str().unwrap()])
            .env("SKYLENS_REGION_DIAG", "1")
            .output()
            .unwrap();
        std::fs::write(root.join(format!("stderr{r}.txt")), &run.stderr).unwrap();
        let v = Command::new(exe)
            .args(["verify", out.to_str().unwrap()])
            .output()
            .unwrap();
        let vs = String::from_utf8_lossy(&v.stdout).to_string();
        let poses = std::fs::read_to_string(out.join("poses.txt")).unwrap_or_default();
        assert_eq!(run.status.code(), Some(0));
        assert_eq!(v.status.code(), Some(0), "verify 7/7 이어야 함: {vs}");
        assert!(vs.contains("7/7 통과"), "{vs}");
        assert_eq!(poses.lines().count(), 81, "등록 81/81");
        eprintln!(
            "run {r} exit {:?} verify exit {:?} poses_lines {}",
            run.status.code(),
            v.status.code(),
            poses.lines().count()
        );
        eprintln!("{vs}");
        posesets.push(poses);
    }
    for r in 1..runs {
        assert_eq!(posesets[0], posesets[r], "run0 와 run{r} 의 poses 가 다름");
    }
    // 성공하면 임시 폴더를 지운다(실패하면 위 단언에서 멈춰 남는다).
    let _ = std::fs::remove_dir_all(&root);
}
