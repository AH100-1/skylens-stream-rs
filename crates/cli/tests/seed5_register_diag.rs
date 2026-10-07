//! 시드 5 기본 경로에서 등록되지 않은 사진이 어느 구역·카메라·시점인지 가리는 진단 시험(무시 시험).
//! 환경 변수 `SKYLENS_REG_DEBUG` 를 켜 파이프라인이 구역별 미등록 사진의 실패 단계 정보를 stderr 로 내게 하고,
//! report.json 의 등록 표(events)와 issues 를 함께 출력한다. 등록 수는 REG_MIN 이상인지 단언한다.
//! 실행: `UPX_SEED=5 REG_MIN=54 cargo test --release -p skylens-stream --test seed5_register_diag -- --ignored --nocapture`

use std::path::PathBuf;
use std::process::Command;

use skylens_core::synth::{Scene, SceneConfig};

/// verify 표 `| registered | 판정 | 초벌 P/T, 정밀 R/T | ... |` 줄에서 초벌 등록 수 P 를 읽는다.
/// 줄이 없거나 형식이 다르면 None (호출하는 쪽이 실패시킨다).
fn parse_registered(vout: &str) -> Option<usize> {
    let line = vout.lines().find(|l| l.starts_with("| registered |"))?;
    line.split('|')
        .nth(3)?
        .split('/')
        .next()?
        .split_whitespace()
        .last()?
        .parse()
        .ok()
}

#[test]
fn parse_registered_reads_row_or_none() {
    let t = "| 항목 | 판정 | 측정 | 기준 |\n| registered | PASS | 초벌 240/240, 정밀 240/240 (x) | c |\n";
    assert_eq!(parse_registered(t), Some(240));
    let t = "| registered | FAIL | 초벌 54/240, 정밀 54/240 | c |";
    assert_eq!(parse_registered(t), Some(54));
    // 줄이 없거나 형식이 바뀌면 None.
    assert_eq!(parse_registered("| region_images | PASS | 1 | c |"), None);
    assert_eq!(parse_registered(""), None);
    assert_eq!(
        parse_registered("| registered | FAIL | 오류: x | c |"),
        None
    );
    assert_eq!(parse_registered("registered PASS 초벌 3/3"), None);
}

#[test]
#[ignore = "시드당 약 8분"]
fn seed_registration_diag() {
    let seed: u64 = std::env::var("UPX_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    // REG_MIN 이 없으면 최소한 등록 1장 이상이어야 한다(기본값 0 은 항상 참이라 쓰지 않는다).
    let min: Option<usize> = std::env::var("REG_MIN").ok().and_then(|s| s.parse().ok());
    let root: PathBuf =
        std::env::temp_dir().join(format!("skylens_regdiag_{seed}_{}", std::process::id()));
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
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&o.stderr);
    for l in stderr.lines().filter(|l| l.starts_with("reg_debug")) {
        eprintln!("{l}");
    }
    assert!(o.status.success(), "run 실패: {stderr}");
    let report = std::fs::read_to_string(output.join("report.json")).unwrap();
    for l in report.split("\",").chain(report.lines()) {
        if l.contains("registration table") || l.contains("건너뜀") {
            eprintln!("report: {l}");
        }
    }
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
    let n = parse_registered(&vout).unwrap_or_else(|| {
        panic!("seed {seed}: verify 출력에서 `| registered |` 줄의 등록 수를 읽지 못함:\n{vout}")
    });
    assert!(
        n >= min.unwrap_or(1),
        "seed {seed} 등록 {n} < {}",
        min.unwrap_or(1)
    );
    if std::env::var_os("KEEP_OUT").is_none() {
        let _ = std::fs::remove_dir_all(&root);
    }
}
