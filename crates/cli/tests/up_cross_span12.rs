//! 위 방향 교차 검사가 기본 경로(인자 없는 synth → run, span 12, 구역 3개)에서 거짓 초과를 내는지 재는 측정 시험.
//! 오래 걸려 `#[ignore]`: `cargo test --release -p skylens-stream --test up_cross_span12 -- --ignored --nocapture`.
//! 시드는 환경변수 `SPAN12_SEEDS`(쉼표 구분, 기본 "1,2,3"). 구역마다 report.json 의 어긋남과, 같은 사진에 정답 회전을
//! 넣었을 때의 어긋남(모델 오차가 없을 때의 기준)을 나란히 찍고, 구역 크기 하한 후보별로 거짓 초과가 몇 건 남는지 센다.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::align::{up_cross_check, UP_CROSS_WARN_DEG};
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::nalgebra::{Matrix3, Rotation3};
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::parse_json;

struct Row {
    seed: u64,
    region: usize,
    positions: usize,
    reported: Vec<Option<f64>>,
    truth: Vec<Option<f64>>,
    own: usize,
}

fn truth_rotations(input: &Path) -> HashMap<String, Rotation3<f64>> {
    let mut m = HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..16].iter().map(|s| s.parse().unwrap()).collect();
        let r = Matrix3::from_row_slice(&n);
        m.insert(f[0].to_string(), Rotation3::from_matrix_unchecked(r));
    }
    m
}

fn fmt(d: &Option<f64>) -> String {
    d.map_or("  -  ".into(), |v| format!("{v:6.3}"))
}

fn mx(v: &[Option<f64>]) -> f64 {
    v.iter().flatten().copied().fold(0.0, f64::max)
}

fn measure(seed: u64, base: &Path) -> Vec<Row> {
    let (input, output): (PathBuf, PathBuf) = (base.join("in"), base.join("out"));
    // 인자 없는 synth 와 같은 장면(960x540, 80 위치), 시드만 바꾼다.
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let t0 = std::time::Instant::now();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stdout = String::from_utf8_lossy(&o.stdout);
    eprintln!("seed {seed}: run {:.0} s", t0.elapsed().as_secs_f64());
    for l in stdout.lines().filter(|l| l.starts_with("region ")) {
        eprintln!("  {l}");
    }
    for l in stdout.lines().filter(|l| l.contains("위 방향 교차")) {
        eprintln!("  {l}");
    }
    let report = std::fs::read_to_string(output.join("report.json")).unwrap();
    let j = parse_json(&report).unwrap();
    let reg = j.get("registered").unwrap();
    eprintln!(
        "  registered total {} preview {} refined {}",
        reg.get("total").unwrap().as_f64().unwrap(),
        reg.get("preview").unwrap().as_f64().unwrap(),
        reg.get("refined").unwrap().as_f64().unwrap()
    );
    let positions: Vec<usize> = j
        .get("regions")
        .and_then(|r| r.as_array())
        .unwrap()
        .iter()
        .map(|r| r.get("positions").unwrap().as_f64().unwrap() as usize)
        .collect();
    let ds = load_dataset(&input, DatasetConfig::default()).unwrap();
    let chunks = ds.chunks();
    let truth = truth_rotations(&input);
    let uc = j.get("up_cross_check").unwrap();
    let mut rows = Vec::new();
    for r in uc.get("regions").and_then(|r| r.as_array()).unwrap() {
        let idx = r.get("region").unwrap().as_f64().unwrap() as usize;
        let reported: Vec<Option<f64>> = r
            .get("diff_deg")
            .and_then(|d| d.as_array())
            .unwrap()
            .iter()
            .map(|d| d.as_f64())
            .collect();
        let (mut rots, mut labels) = (Vec::new(), Vec::new());
        for p in chunks[idx].clone() {
            for c in 0..3 {
                let name = ds.positions[p].images[c]
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                rots.push(truth[&name]);
                labels.push(c);
            }
        }
        let t = up_cross_check(&rots, &labels).unwrap();
        rows.push(Row {
            seed,
            region: idx,
            positions: positions[idx],
            reported,
            truth: t.diff_deg.clone(),
            own: t.up_own.iter().flatten().count(),
        });
    }
    rows
}

#[test]
#[ignore = "기본 경로 run 을 시드마다 한 번씩 돌린다(수 분)"]
fn up_cross_span12_false_exceed() {
    let seeds: Vec<u64> = std::env::var("SPAN12_SEEDS")
        .unwrap_or_else(|_| "1,2,3".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let mut rows = Vec::new();
    for s in &seeds {
        let base = std::env::temp_dir().join(format!("skylens_upc12_{}_{s}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        rows.extend(measure(*s, &base));
        let _ = std::fs::remove_dir_all(&base);
    }
    eprintln!("\n시드 구역 위치수 | 보고 어긋남 F/R/L(도) | 정답 회전 어긋남 F/R/L | 자기 위 방향 있는 묶음 수 | 초과");
    for r in &rows {
        eprintln!(
            "{:>3} {:>3} {:>4} | {} {} {} | {} {} {} | {} | {}",
            r.seed,
            r.region,
            r.positions,
            fmt(&r.reported[0]),
            fmt(&r.reported[1]),
            fmt(&r.reported[2]),
            fmt(&r.truth[0]),
            fmt(&r.truth[1]),
            fmt(&r.truth[2]),
            r.own,
            if mx(&r.reported) > UP_CROSS_WARN_DEG {
                "예"
            } else {
                "아니오"
            }
        );
    }
    eprintln!("\n구역 위치 수 하한 후보: 검사하는 구역 수 / 그중 거짓 초과 구역 수(정답 회전으론 문턱 미만인데 보고는 초과)");
    for min_pos in [0usize, 8, 12, 14, 16, 20] {
        let kept: Vec<&Row> = rows.iter().filter(|r| r.positions >= min_pos).collect();
        let fp = kept
            .iter()
            .filter(|r| mx(&r.reported) > UP_CROSS_WARN_DEG && mx(&r.truth) <= UP_CROSS_WARN_DEG)
            .count();
        eprintln!(
            "  위치 >= {min_pos:>2}: 검사 {} / {} 구역, 거짓 초과 {fp}",
            kept.len(),
            rows.len()
        );
    }
    // 기준: 정답 회전을 넣으면 기본 장면에서 문턱을 넘지 않는다(검사 자체는 옳다).
    for r in &rows {
        assert!(
            mx(&r.truth) < UP_CROSS_WARN_DEG,
            "정답 어긋남 {:?}",
            r.truth
        );
    }
}
