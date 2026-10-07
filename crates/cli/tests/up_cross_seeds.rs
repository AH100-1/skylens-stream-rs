//! 여덟째 항목 `up_cross` 문턱(10°)의 거짓 실패 점검(F-446): 시드별 기본 경로(인자 없는 synth → run → verify)를 돌려
//! 항목별 판정과 report.json 의 구역별 최대 어긋남(diff_deg)을 출력하고, 시드 3 구역 0 외에는 모두 문턱 이하임을 단언한다.
//! 오래 걸려(시드당 약 2~3분) 무시 시험이다.
//! 실행: `UPX_SEEDS=4,5 cargo test --release -p skylens-stream --test up_cross_seeds -- --ignored --nocapture`
//! (UPX_SEEDS 기본 "4,5", 쉼표 구분). synth CLI 에는 시드 인자가 없어 시드만 바꾼 기본 설정 장면을 라이브러리로 쓴다.

use std::path::PathBuf;
use std::process::Command;

use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::UP_CROSS_FAIL_DEG;

const ITEMS: [&str; 8] = [
    "registered",
    "region_images",
    "refined_reprojection",
    "preview_align",
    "preview_vs_refined",
    "refined_overlap",
    "snapshots",
    "up_cross",
];

/// 앞선 기록에서 이미 실패로 알려진 (시드, 구역): 시드 3 구역 0 (69.509°).
const KNOWN_FAIL: (u64, usize) = (3, 0);

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_upx_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cli(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

/// verify 표의 한 항목 (판정, 측정값).
fn item(table: &str, name: &str) -> (String, String) {
    let line = table
        .lines()
        .find(|l| l.starts_with(&format!("| {name} |")))
        .unwrap_or_else(|| panic!("{name} 줄 없음:\n{table}"));
    let c: Vec<&str> = line.split('|').map(str::trim).collect();
    (c[2].to_string(), c[3].to_string())
}

/// report.json 의 `"diff_deg":[...]` 배열마다 절댓값 최대 (구역 순서대로). 값이 모두 null 인 구역은 None(미측정).
fn region_maxes(report: &str) -> Vec<Option<f64>> {
    let compact: String = report.chars().filter(|c| !c.is_whitespace()).collect();
    compact
        .split("\"diff_deg\":[")
        .skip(1)
        .map(|s| {
            let body = &s[..s.find(']').expect("diff_deg 닫는 괄호 없음")];
            body.split(',')
                .filter_map(|t| t.parse::<f64>().ok())
                .map(f64::abs)
                .reduce(f64::max)
        })
        .collect()
}

fn seeds() -> Vec<u64> {
    std::env::var("UPX_SEEDS")
        .unwrap_or_else(|_| "4,5".into())
        .split(',')
        .map(|s| s.trim().parse().expect("UPX_SEEDS 는 쉼표로 구분한 정수"))
        .collect()
}

#[test]
#[ignore = "시드당 2~3분"]
fn up_cross_default_path_seeds() {
    let mut bad = Vec::new();
    for seed in seeds() {
        let t = TempDir::new(&format!("s{seed}"));
        let (input, output) = (t.0.join("in"), t.0.join("out"));
        let (i, o) = (input.to_str().unwrap(), output.to_str().unwrap());
        let scene = Scene::new(SceneConfig {
            seed,
            ..SceneConfig::default()
        });
        scene.write_dataset(&input).unwrap();
        let (code, _, stderr) = cli(&["run", i, o]);
        assert_eq!(code, 0, "seed {seed} run: {stderr}");
        let (vcode, vout, _) = cli(&["verify", o]);
        let mut pass = 0;
        let mut failed = Vec::new();
        for name in ITEMS {
            let (st, detail) = item(&vout, name);
            eprintln!("seed {seed} {name}: {st} | {detail}");
            if st == "PASS" {
                pass += 1;
            } else {
                failed.push(name);
            }
        }
        let report = std::fs::read_to_string(output.join("report.json")).unwrap();
        let maxes = region_maxes(&report);
        assert!(
            !maxes.is_empty(),
            "seed {seed}: report.json 에서 diff_deg 못 찾음:\n{report}"
        );
        eprintln!(
            "seed {seed}: verify {pass}/8 exit {vcode} failed {failed:?} up_cross 구역별 최대(deg, None=미측정) {maxes:.3?}"
        );
        eprintln!("seed {seed}: up_cross 만 실패 = {}", failed == ["up_cross"]);
        for (r, m) in maxes.iter().enumerate() {
            let Some(m) = m else {
                eprintln!("seed {seed} 구역 {r}: up_cross 미측정(diff_deg 전부 null)");
                continue;
            };
            if (seed, r) != KNOWN_FAIL && *m > UP_CROSS_FAIL_DEG {
                bad.push(format!("seed {seed} 구역 {r}: {m:.3}°"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "문턱 {UP_CROSS_FAIL_DEG}° 초과(시드 3 구역 0 제외): {bad:?}"
    );
}
