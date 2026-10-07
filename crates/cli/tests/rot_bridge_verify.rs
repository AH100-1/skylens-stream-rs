//! 회전 평균 덩어리 잇기(`SKYLENS_ROT_BRIDGE=1`)를 끄고 켠 기본 경로(인자 없는 synth 와 같은 장면, span 12)의
//! verify 결과를 시드별로 나란히 찍는 측정 시험. 오래 걸려 `#[ignore]`:
//! `RBV_SEEDS=3,1 RBV_MODES=off,on cargo test --release -p skylens-stream --test rot_bridge_verify -- --ignored --nocapture`
//! 출력은 `/tmp/rbv-<시드>-<on|off>/{in,out,run.txt,verify.txt}` 에 남긴다.

use std::path::PathBuf;
use std::process::Command;

use skylens_core::synth::{Scene, SceneConfig};

fn list(var: &str, default: &str) -> Vec<String> {
    std::env::var(var)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[test]
#[ignore = "기본 경로 run 을 시드·모드마다 한 번씩 돌린다(수 분)"]
fn rot_bridge_verify_table() {
    let exe = env!("CARGO_BIN_EXE_skylens-stream");
    for seed in list("RBV_SEEDS", "3,1") {
        let seed_n: u64 = seed.parse().unwrap();
        for mode in list("RBV_MODES", "off,on") {
            let base = PathBuf::from(format!("/tmp/rbv-{seed}-{mode}"));
            let _ = std::fs::remove_dir_all(&base);
            let (input, output) = (base.join("in"), base.join("out"));
            Scene::new(SceneConfig {
                seed: seed_n,
                ..SceneConfig::default()
            })
            .write_dataset(&input)
            .unwrap();
            let t0 = std::time::Instant::now();
            let mut run = Command::new(exe);
            run.args(["run", input.to_str().unwrap(), output.to_str().unwrap()]);
            if mode == "on" {
                run.env("SKYLENS_ROT_BRIDGE", "1");
            } else {
                run.env_remove("SKYLENS_ROT_BRIDGE");
            }
            let o = run.output().unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            std::fs::write(base.join("run.txt"), &o.stdout).unwrap();
            let secs = t0.elapsed().as_secs_f64();
            let v = Command::new(exe)
                .args(["verify", output.to_str().unwrap()])
                .output()
                .unwrap();
            std::fs::write(base.join("verify.txt"), &v.stdout).unwrap();
            eprintln!(
                "=== seed {seed} bridge {mode}: run {secs:.0} s, verify exit {:?}",
                v.status.code()
            );
            eprintln!("{}", String::from_utf8_lossy(&v.stdout));
        }
    }
}
