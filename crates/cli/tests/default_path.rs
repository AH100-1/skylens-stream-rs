//! 기본 경로: 인자 없는 `synth` → `run` → `verify` 를 프로세스로 돌려 7/7·종료 0 과 정답 대비 오차를 숫자로 고정한다(F-343).
//! 기본 구역 크기(span 12)라 구역이 여러 개이고, 구역을 복원할 때 구역 밖 앞·뒤 보조 사진(`HelperConfig`)이
//! 다른 카메라 짝을 이어 준다. 정답 비교 방식은 `pipeline_e2e` 와 같다(원점 차만큼 옮겨 비교, 회전은 재지 않음).

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

/// verify 표의 한 항목 (판정, 측정값).
fn item(table: &str, name: &str) -> (String, String) {
    let line = table
        .lines()
        .find(|l| l.starts_with(&format!("| {name} |")))
        .unwrap_or_else(|| panic!("{name} 줄 없음:\n{table}"));
    let c: Vec<&str> = line.split('|').map(str::trim).collect();
    (c[2].to_string(), c[3].to_string())
}

#[test]
fn default_args_synth_run_verify() {
    let t = TempDir::new("default_path");
    let (input, output) = (t.0.join("in"), t.0.join("out"));
    let (i, o) = (input.to_str().unwrap(), output.to_str().unwrap());
    assert_eq!(cli(&["synth", i]).0, 0);
    let t0 = Instant::now();
    let (code, stdout, stderr) = cli(&["run", i, o]);
    let run_secs = t0.elapsed().as_secs_f64();
    assert_eq!(code, 0, "{stderr}");
    let chunks: usize = stdout
        .lines()
        .find_map(|l| l.strip_prefix("chunks "))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(chunks >= 2, "기본 경로가 구역 하나로 합쳐짐: {chunks}");
    let (vcode, vout, _) = cli(&["verify", o]);
    assert_eq!(vcode, 0, "{vout}");
    assert!(vout.contains("결과: 7/7 통과"), "{vout}");
    let (st, reg) = item(&vout, "registered");
    assert_eq!(st, "PASS", "{vout}");
    assert!(
        reg.contains("초벌 81/81") && reg.contains("정밀 81/81"),
        "{reg}"
    );
    for name in ["preview_vs_refined", "refined_overlap", "snapshots"] {
        assert_eq!(item(&vout, name).0, "PASS", "{vout}");
    }
    let m = measure(&input, &output);
    eprintln!(
        "chunks {chunks} run {run_secs:.1}s registered {} center med {:.4} max {:.4} points {} surface med {:.4} p95 {:.4}",
        m.registered, m.center_med, m.center_max, m.points, m.surface_med, m.surface_p95
    );
    assert_eq!(m.registered, 81);
    assert!(
        m.center_med < 0.47,
        "카메라 중심 오차 중앙 {:.3} m",
        m.center_med
    );
    assert!(
        m.surface_med < 0.5,
        "점 표면 오차 중앙 {:.3} m",
        m.surface_med
    );
    assert!(m.points > 1000, "점 {}", m.points);
}

/// 진단: 환경 변수 SKY_IN·SKY_OUT 의 결과를 구역(refined/*.ply)별 점 표면 오차로 낸다.
#[test]
#[ignore]
fn diag_surface_by_region() {
    let (input, output) = (
        PathBuf::from(std::env::var("SKY_IN").unwrap()),
        PathBuf::from(std::env::var("SKY_OUT").unwrap()),
    );
    let shift = truth_to_output_shift(&input);
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    let mut files: Vec<_> = std::fs::read_dir(output.join("refined"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ply"))
        .collect();
    files.sort();
    let mut all = Vec::new();
    for f in files {
        let mut d = Vec::new();
        let (mut src, mut dst, mut signed) = (Vec::new(), Vec::new(), Vec::new());
        for p in read_ply_file(&f).unwrap().points {
            let (x, y, z) = (
                p.xyz[0] as f64 - shift[0],
                p.xyz[1] as f64 - shift[1],
                p.xyz[2] as f64 - shift[2],
            );
            let h = scene.surface_height(x, y);
            d.push((z - h).abs());
            signed.push(z - h);
            src.push(skylens_core::nalgebra::Vector3::new(x, y, z));
            dst.push(skylens_core::nalgebra::Vector3::new(x, y, h));
        }
        // 구역 모델 자체를 정답 표면 점(수직 투영)에 닮음 변환으로 맞춘 잔차: 전역 sim3 편향인지 내부 형상 오차인지 구분.
        // 구역 안 위치(분산이 큰 축 기준) 4 구간별 부호 있는 높이 오차 중앙: 편향이 구역 끝으로 갈수록 커지는지 본다.
        {
            let n = src.len() as f64;
            let mean = src
                .iter()
                .fold(skylens_core::nalgebra::Vector3::zeros(), |a, p| a + p)
                / n;
            let var = |k: usize| src.iter().map(|p| (p[k] - mean[k]).powi(2)).sum::<f64>();
            let ax = if var(0) >= var(1) { 0 } else { 1 };
            let mut idx: Vec<usize> = (0..src.len()).collect();
            idx.sort_by(|&a, &b| src[a][ax].total_cmp(&src[b][ax]));
            let q = idx.len() / 4;
            let bins: Vec<String> = (0..4)
                .map(|b| {
                    let hi = if b == 3 { idx.len() } else { (b + 1) * q };
                    let mut v: Vec<f64> = idx[b * q..hi].iter().map(|&i| signed[i]).collect();
                    format!("{:.3}", median(&mut v))
                })
                .collect();
            eprintln!("DIAG   axis {ax} signed_dz_by_quarter {}", bins.join(" "));
        }
        let sm = median(&mut signed);
        if let Some(sim) = skylens_core::align::umeyama(&src, &dst) {
            let mut res: Vec<f64> = src
                .iter()
                .zip(&dst)
                .map(|(a, b)| (sim.apply_point(a) - b).norm())
                .collect();
            eprintln!(
                "DIAG   signed_dz_med {sm:.4} sim3 s {:.4} rot {:.3}deg t_z {:.3} resid_med {:.4} resid_p90 {:.4}",
                sim.s,
                sim.r.angle().to_degrees(),
                sim.t.z,
                median(&mut res),
                percentile(&mut res, 0.9)
            );
        }
        all.extend(d.iter().copied());
        let n = d.len();
        let m = median(&mut d);
        let p90 = percentile(&mut d, 0.9);
        eprintln!(
            "DIAG {} n {n} med {m:.4} p90 {p90:.4}",
            f.file_name().unwrap().to_string_lossy()
        );
    }
    let m = measure(&input, &output);
    eprintln!(
        "DIAG center med {:.3} max {:.3}",
        m.center_med, m.center_max
    );
    eprintln!(
        "DIAG all med {:.4} p95 {:.4}",
        median(&mut all),
        percentile(&mut all, 0.95)
    );
}
