//! 시드 5 기본 경로에서 떨어진 카메라 덩어리 붙이기(`SKYLENS_ATTACH_COMPONENT=1`) 끔/켬 측정(무시 시험).
//! 등록 수·verify 통과 항목 수·카메라 2 중심 오차(정답 대비 중앙)를 출력하고 기준값을 단언한다.
//! 실행: `ATTACH=1 UPX_SEED=5 REG_MIN=54 cargo test --release -p skylens-stream --test cam_component_attach -- --ignored --nocapture`
//! (ATTACH=0 또는 미지정이면 끔). 회전은 poses.txt 에 없어 중심 오차로 대신 잰다.

use std::path::PathBuf;
use std::process::Command;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::synth::{Scene, SceneConfig};

fn geodetic(line: &str) -> Geodetic {
    let n: Vec<f64> = line
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

#[test]
#[ignore = "시드당 약 6분"]
fn seed_attach_component() {
    let seed: u64 = std::env::var("UPX_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let min: usize = std::env::var("REG_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let on = std::env::var("ATTACH").as_deref() == Ok("1");
    let root: PathBuf =
        std::env::temp_dir().join(format!("skylens_attach_{seed}_{on}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    std::fs::create_dir_all(&input).unwrap();
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skylens-stream"));
    cmd.args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_REG_DEBUG", "1");
    if on {
        cmd.env("SKYLENS_ATTACH_COMPONENT", "1");
    }
    let o = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&o.stderr);
    for l in stderr.lines().filter(|l| l.contains("attach")) {
        eprintln!("{l}");
    }
    assert!(o.status.success(), "run 실패: {stderr}");
    let v = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    let vout = String::from_utf8_lossy(&v.stdout);
    eprintln!("{vout}");
    let passed = vout.lines().filter(|l| l.contains("PASS")).count();
    let n: usize = vout
        .lines()
        .find(|l| l.starts_with("| registered |"))
        .and_then(|l| l.split('|').nth(3))
        .and_then(|c| c.split('/').next())
        .and_then(|c| c.split_whitespace().last())
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    // 카메라 2(이름 camL_*) 중심 오차: 정답은 정답 원점 기준이므로 첫 GPS 기준으로 옮긴다.
    let origin = geodetic(&std::fs::read_to_string(input.join("truth/origin.txt")).unwrap());
    let first = std::fs::read_to_string(input.join("gps.txt")).unwrap();
    let f: Vec<f64> = first
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let d = geodetic_to_enu(
        &origin,
        &Geodetic {
            lat_deg: f[0],
            lon_deg: f[1],
            alt: f[2],
        },
    );
    let shift = [-d.x, -d.y, -d.z];
    let mut errs = Vec::new();
    let truth = std::fs::read_to_string(input.join("truth/cameras.txt")).unwrap();
    let poses = std::fs::read_to_string(output.join("poses.txt")).unwrap();
    for l in poses.lines().filter(|l| l.starts_with("camL_")) {
        let p: Vec<&str> = l.split_whitespace().collect();
        let Some(t) = truth.lines().find(|t| t.starts_with(p[0])) else {
            continue;
        };
        let f: Vec<&str> = t.split_whitespace().collect();
        let nn: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, tt) = (&nn[..9], &nn[9..12]);
        let c: Vec<f64> = (0..3)
            .map(|j| -(r[j] * tt[0] + r[3 + j] * tt[1] + r[6 + j] * tt[2]))
            .collect();
        let e: f64 = (0..3)
            .map(|k| (p[1 + k].parse::<f64>().unwrap() - (c[k] + shift[k])).powi(2))
            .sum::<f64>()
            .sqrt();
        errs.push(e);
    }
    errs.sort_by(f64::total_cmp);
    let med = errs.get(errs.len() / 2).copied();
    eprintln!(
        "RESULT seed {seed} attach {on}: registered {n}, verify PASS lines {passed}, camera 2 registered {}, camera 2 center error median {med:?} m",
        errs.len()
    );
    assert!(n >= min, "seed {seed} 등록 {n} < {min}");
    if std::env::var_os("KEEP_OUT").is_none() {
        let _ = std::fs::remove_dir_all(&root);
    }
}
