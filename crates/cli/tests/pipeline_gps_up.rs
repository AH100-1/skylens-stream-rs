//! 구역 GPS 정렬의 연직 고정 옵션(`--gps-fixed-up`): 같은 합성 장면(구역 2개, 240장)을 옵션 끔/켬으로 돌려
//! verify 통과와 정답 대비 카메라 중심·정밀 점 표면 거리가 나빠지지 않음을 숫자로 고정한다.
//! 측정(4 코어 측정 기계, 시드 1): 끔 표면 중앙 0.151 m, 중심 중앙/최대 0.190/0.700 m, 켬 0.147 m, 0.192/0.523 m.
//! 상한은 켬 측정값의 약 1.2~1.4 배와 끔 대비 여유.

use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};

/// 켬 측정 최대 0.523 m, 끔 0.700 m.
const CENTER_MAX_LIMIT: f64 = 0.75;
/// 켬 측정 중앙 0.147 m, 끔 0.151 m.
const SURFACE_MED_LIMIT: f64 = 0.17;

/// README 의 `run` 공통 옵션(stride 제외).
const RUN_OPTS: [&str; 12] = [
    "--span",
    "48",
    "--ovl",
    "2",
    "--max-features",
    "800",
    "--dense-width",
    "96",
    "--hfov",
    "65",
    "--ba-iters",
    "15",
];

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_e2e_{tag}_{}", std::process::id()));
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
}

/// 한 줄 "위도 경도 고도" 세 수.
fn geodetic_of(fields: &[&str]) -> Geodetic {
    let n: Vec<f64> = fields.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

/// 정답 좌표(truth/origin.txt 기준) → 출력 좌표(첫 GPS 기준) 평행 이동량.
fn truth_to_output_shift(input: &Path) -> [f64; 3] {
    let o = std::fs::read_to_string(input.join("truth/origin.txt")).unwrap();
    let truth_origin = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gps = std::fs::read_to_string(input.join("gps.txt")).unwrap();
    let first: Vec<&str> = gps.lines().next().unwrap().split_whitespace().collect();
    let first_gps = geodetic_of(&first[1..4]);
    let d = geodetic_to_enu(&truth_origin, &first_gps);
    [d.x, d.y, d.z]
}

struct Metrics {
    registered: usize,
    center_med: f64,
    center_max: f64,
    points: usize,
    surface_med: f64,
    surface_p95: f64,
}

fn measure(input: &Path, output: &Path) -> Metrics {
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    let shift = truth_to_output_shift(input);
    // 정답 카메라 중심 C = -Rᵀ t (이름 뒤 내부 파라미터 6개, 그다음 R 행 우선 9개, t 3개).
    let mut truth = std::collections::HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, t) = (&n[..9], &n[9..12]);
        let c: Vec<f64> = (0..3)
            .map(|j| -(r[j] * t[0] + r[3 + j] * t[1] + r[6 + j] * t[2]))
            .collect();
        truth.insert(f[0].to_string(), [c[0], c[1], c[2]]);
    }
    let mut errs = Vec::new();
    for l in std::fs::read_to_string(output.join("poses.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let c: Vec<f64> = f[1..4].iter().map(|s| s.parse().unwrap()).collect();
        let t = truth[f[0]];
        let d: f64 = (0..3)
            .map(|k| (c[k] - (t[k] + shift[k])).powi(2))
            .sum::<f64>()
            .sqrt();
        errs.push(d);
    }
    let registered = errs.len();
    let center_max = errs.iter().copied().fold(0.0, f64::max);
    let center_med = median(&mut errs);

    let mut d = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(output.join("refined"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ply"))
        .collect();
    files.sort();
    for f in files {
        for p in read_ply_file(f).unwrap().points {
            let (x, y, z) = (
                p.xyz[0] as f64 - shift[0],
                p.xyz[1] as f64 - shift[1],
                p.xyz[2] as f64 - shift[2],
            );
            d.push((z - scene.surface_height(x, y)).abs());
        }
    }
    let points = d.len();
    Metrics {
        registered,
        center_med,
        center_max,
        points,
        surface_p95: percentile(&mut d, 0.95),
        surface_med: median(&mut d),
    }
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

/// "접두 123.456 m" 형태의 측정 문자열에서 접두 뒤 첫 숫자.
fn number_after(s: &str, prefix: &str) -> f64 {
    let at = s
        .find(prefix)
        .unwrap_or_else(|| panic!("{prefix} 없음: {s}"))
        + prefix.len();
    let t = &s[at..];
    let end = t
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .unwrap_or(t.len());
    t[..end].parse().unwrap()
}

fn run_case(tag: &str, scene: &Path, extra: &[&str]) -> (i32, String, Metrics) {
    let out = scene.parent().unwrap().join(format!("out_{tag}"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let mut args = vec!["run", scene_s, out_s, "--stride", "1"];
    args.extend(RUN_OPTS);
    args.extend(extra);
    let (code, so, se) = cli(&args);
    assert_eq!(code, 0, "{so}{se}");
    let (vcode, table, se) = cli(&["verify", out_s]);
    eprintln!("{tag}\n{table}verify exit {vcode} {se}");
    let m = measure(scene, &out);
    eprintln!(
        "{tag}: registered {} center med {:.3} max {:.3} surface med {:.3} p95 {:.3}",
        m.registered, m.center_med, m.center_max, m.surface_med, m.surface_p95
    );
    (vcode, table, m)
}

/// 옵션 켬은 verify 7/7 을 유지하고, 끔 대비 카메라 중심 최대·표면 중앙이 나빠지지 않는다(여유 포함).
/// 끔 기본 동작은 이 시험과 pipeline_regions 가 따로 지킨다.
#[test]
fn gps_fixed_up_does_not_regress() {
    let t = TempDir::new("gpsup");
    let scene = t.0.join("scene");
    let (code, so, se) = cli(&["synth", scene.to_str().unwrap(), "320", "180"]);
    assert_eq!(code, 0, "{so}{se}");
    let (c0, t0, off) = run_case("off", &scene, &[]);
    let (c1, t1, on) = run_case("on", &scene, &["--gps-fixed-up"]);
    assert_eq!(c0, 0, "{t0}");
    assert_eq!(c1, 0, "{t1}");
    for name in [
        "registered",
        "region_images",
        "refined_reprojection",
        "preview_align",
        "preview_vs_refined",
        "refined_overlap",
        "snapshots",
    ] {
        assert_eq!(item(&t1, name).0, "PASS", "{name}: {t1}");
    }
    let (_, ov) = item(&t1, "refined_overlap");
    assert!(number_after(&ov, "중앙 최대 ") < 0.2, "{ov}");
    assert_eq!(on.registered, off.registered, "등록 수");
    assert_eq!(on.registered, 240, "등록 수");
    assert!(
        on.points * 10 >= off.points * 9 && on.points * 10 <= off.points * 11,
        "점 수 켬 {} 끔 {}",
        on.points,
        off.points
    );
    // 끔 측정: 표면 중앙 0.151, 중심 중앙/최대 0.190/0.700. 켬 측정값은 아래 상한 근거(노트).
    assert!(
        on.center_max <= off.center_max + 0.05 && on.center_max < CENTER_MAX_LIMIT,
        "중심 최대 켬 {} 끔 {}",
        on.center_max,
        off.center_max
    );
    assert!(
        on.center_med <= off.center_med + 0.02,
        "중심 중앙 켬 {} 끔 {}",
        on.center_med,
        off.center_med
    );
    assert!(
        on.surface_med <= off.surface_med + 0.01 && on.surface_med < SURFACE_MED_LIMIT,
        "표면 중앙 켬 {} 끔 {}",
        on.surface_med,
        off.surface_med
    );
}
