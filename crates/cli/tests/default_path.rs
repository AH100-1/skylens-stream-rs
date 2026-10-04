//! 기본 경로: 인자 없는 `synth` → `run` → `verify` 를 프로세스로 돌려(F-343) verify 7/7·종료 코드 0,
//! 등록 81/81, 구역 수 ≥ 2, 합성 정답 대비 카메라 중심·점 표면 오차 중앙값을 숫자로 고정한다.
//! 구역은 SPAN 12·OVL 2 그대로이고, R·L 카메라 구역 창을 `cross_offset`(기본 24 위치) 뒤로 밀어
//! 구역 안에서 F–R·F–L 겹침 짝이 생긴다(연구 노트 experiments/region-camera-offset.md).
//! 정답 비교 방식은 `pipeline_e2e.rs` 와 같다(출력 좌표는 첫 GPS 원점, 정답은 `truth/origin.txt` 원점).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p =
            std::env::temp_dir().join(format!("skylens_default_path_{tag}_{}", std::process::id()));
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

    let scene = Scene::new(SceneConfig::default());
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

/// 기본 경로 한 번: 기본 장면 27위치 × 3대 = 81장, 3구역.
#[test]
fn default_path_registers_all_and_verifies() {
    let t = TempDir::new("default");
    let (scene, out) = (t.0.join("in"), t.0.join("out"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let (code, so, se) = cli(&["synth", scene_s]);
    assert_eq!(code, 0, "{so}{se}");
    assert!(so.contains("views 240"), "{so}");

    let start = Instant::now();
    let (code, so, se) = cli(&["run", scene_s, out_s]);
    let secs = start.elapsed().as_secs_f64();
    assert_eq!(code, 0, "{so}{se}");
    eprintln!("run secs {secs:.1}");
    let chunks = so.lines().find_map(|l| l.strip_prefix("chunks ")).unwrap();
    let chunks: usize = chunks.trim().parse().unwrap();
    assert!(chunks >= 2, "구역 수 {chunks}\n{so}");

    let (verify_code, table, se) = cli(&["verify", out_s]);
    eprintln!("{table}verify exit {verify_code} {se}");
    let m = measure(&scene, &out);
    eprintln!(
        "DEFAULT chunks {chunks} registered {} center med {:.3} max {:.3} points {} surface med {:.3} p95 {:.3}",
        m.registered, m.center_med, m.center_max, m.points, m.surface_med, m.surface_p95
    );
    assert_eq!(verify_code, 0, "{table}");
    assert!(table.contains("결과: 7/7 통과"), "{table}");
    let (_, reg) = item(&table, "registered");
    assert!(reg.contains("초벌 81/81, 정밀 81/81"), "{reg}");
    assert_eq!(m.registered, 81, "poses.txt 등록 수");
    let (_, pr) = item(&table, "preview_vs_refined");
    assert!(number_after(&pr, "최근접 중앙 최대 ") < 3.0, "{pr}");
    assert!(number_after(&pr, "높이 차 중앙 최대 ") < 2.0, "{pr}");
    let (_, ov) = item(&table, "refined_overlap");
    assert!(!ov.contains("해당 없음"), "겹침이 판정돼야 함: {ov}");
    // 실측(4코어 측정 기계): 중심 오차 중앙 0.307/최대 0.940 m, 정밀 점 표면 거리 중앙 0.482/95% 1.355 m (점 32419개).
    // 상한은 실측 x 1.2.
    assert!(m.center_med < 0.37, "중심 오차 중앙 {}", m.center_med);
    assert!(m.center_max < 1.13, "중심 오차 최대 {}", m.center_max);
    assert!(m.surface_med < 0.58, "점 중앙 {}", m.surface_med);
    assert!(m.surface_p95 < 1.63, "점 95% {}", m.surface_p95);
    assert!(
        m.points >= 32419 * 4 / 5 && m.points <= 32419 * 6 / 5,
        "점 수 {}",
        m.points
    );
}
