//! 다른 카메라 짝의 최소 대응 하한을 낮춘(`SKYLENS_CROSS_MIN_MATCHES`, 기본 10) 시드 기본 경로 시험(무시 시험).
//! 낮춘 짝도 RANSAC·정상 수·투표 검증은 그대로 거친다. 등록 수·up_cross 를 stderr 에 낸다.
//! 실행: `UPX_SEED=5 REG_MIN=78 UPX_MAX=10 cargo test --release -p skylens-stream --test seed5_cross_min -- --ignored --nocapture`

use std::path::PathBuf;
use std::process::Command;

use skylens_core::synth::{Scene, SceneConfig};

#[test]
#[ignore = "시드당 약 8분"]
fn seed_cross_min() {
    let seed: u64 = std::env::var("UPX_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let min: usize = std::env::var("REG_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let max_up: f64 = std::env::var("UPX_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(f64::INFINITY);
    let root: PathBuf =
        std::env::temp_dir().join(format!("skylens_crossmin_{seed}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    std::fs::create_dir_all(&input).unwrap();
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_REG_DEBUG", "1")
        .env(
            "SKYLENS_CROSS_MIN_MATCHES",
            std::env::var("SKYLENS_CROSS_MIN_MATCHES").unwrap_or_else(|_| "10".into()),
        )
        .env(
            "SKYLENS_PRUNE_CAM",
            std::env::var("SKYLENS_PRUNE_CAM").unwrap_or_else(|_| "2".into()),
        )
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&o.stderr);
    for l in stderr
        .lines()
        .filter(|l| l.starts_with("reg_debug") || l.starts_with("prune_dbg"))
    {
        eprintln!("{l}");
    }
    assert!(o.status.success(), "run 실패: {stderr}");
    let v = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    let vout = String::from_utf8_lossy(&v.stdout);
    eprintln!("{vout}");
    let reg_line = vout
        .lines()
        .find(|l| l.starts_with("| registered |"))
        .unwrap_or("");
    eprintln!("seed {seed} {reg_line}");
    let n: usize = reg_line
        .split('|')
        .nth(3)
        .and_then(|c| c.split('/').next())
        .and_then(|c| c.split_whitespace().last())
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    assert!(n >= min, "seed {seed} 등록 {n} < {min}");
    let up = vout.lines().find(|l| l.contains("up_cross")).unwrap_or("");
    eprintln!("seed {seed} {up}");
    let worst = up
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .filter_map(|t| t.parse::<f64>().ok())
        .fold(0.0f64, f64::max);
    assert!(
        worst <= max_up,
        "seed {seed} up_cross {worst} > {max_up}: {up}"
    );
    if std::env::var_os("KEEP_OUT").is_none() {
        let _ = std::fs::remove_dir_all(&root);
    }
}
