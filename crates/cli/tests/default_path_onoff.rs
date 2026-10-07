//! 회전 다리 옵션(`SKYLENS_ROT_BRIDGE`) 끔/켬을 같은 기본 경로(시드 지정, 기본 구역 크기)에서 번갈아 재는 측정 시험.
//! 모드 목록은 `ONOFF_MODES`(쉼표 구분, 0=끔 1=켬, 기본 "0,1,0,1"), 시드는 `ONOFF_SEED`(기본 1).
//! `cargo test --release -j 2 -p skylens-stream --test default_path_onoff -- --ignored --nocapture`

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};

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

    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
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

fn verify_pass(vout: &str) -> String {
    vout.lines()
        .find(|l| l.contains("결과:"))
        .unwrap_or("결과 줄 없음")
        .to_string()
}

#[test]
#[ignore]
fn bridge_on_off_default_path() {
    let seed: u64 = std::env::var("ONOFF_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let modes: Vec<String> = std::env::var("ONOFF_MODES")
        .unwrap_or_else(|_| "0,1,0,1".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let t = TempDir::new("onoff");
    let input = t.0.join("in");
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    for (k, mode) in modes.iter().enumerate() {
        let output = t.0.join(format!("out{k}"));
        let t0 = Instant::now();
        let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
            .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
            .env("SKYLENS_ROT_BRIDGE", mode)
            .output()
            .unwrap();
        let secs = t0.elapsed().as_secs_f64();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let (_, vout, _) = cli(&["verify", output.to_str().unwrap()]);
        let m = measure(&input, &output);
        eprintln!(
            "ONOFF seed {seed} run {k} bridge {mode} secs {secs:.0} registered {} center med {:.4} max {:.4} points {} surface med {:.4} p95 {:.4} verify [{}]",
            m.registered, m.center_med, m.center_max, m.points, m.surface_med, m.surface_p95,
            verify_pass(&vout)
        );
        let _ = std::fs::remove_dir_all(&output);
    }
}
