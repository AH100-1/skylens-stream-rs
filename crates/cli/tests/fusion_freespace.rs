//! 융합 선택 항목(자유공간 위반 검사, 중앙값 위치)을 단구역 끝까지 켜고 끈 수치.
//! 기본값은 바뀌지 않는다: 이 시험은 `SKYLENS_FUSION_FREESPACE`, `SKYLENS_FUSION_MEDIAN` 을 `run` 에만 넘겨
//! 점 수와 합성 정답 표면 거리(중앙·95%)를 표로 찍는다. 단구역 설정은 `pipeline_e2e.rs` 의 첫 시험과 같다.
//! 오래 걸려(조합당 run 수 분) `#[ignore]`: `cargo test --release --test fusion_freespace -- --ignored --nocapture`.

use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};

/// README 의 `run` 공통 옵션(stride 제외, BA 반복은 따로 붙인다: README 값 15).
const RUN_OPTS: [&str; 10] = [
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

fn cli(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(args)
        .envs(env.iter().copied())
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

/// (점 수, 표면 거리 중앙, 95%).
fn surface_stats(input: &Path, output: &Path) -> (usize, f64, f64) {
    let shift = truth_to_output_shift(input);
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
    (d.len(), median(&mut d), percentile(&mut d, 0.95))
}

/// 같은 합성 장면에 융합 설정만 바꿔 네 번 돌린다(기본 / a / b / a+b).
#[test]
#[ignore = "조합당 run 수 분"]
fn fusion_freespace_table() {
    let t = TempDir::new("ffs");
    let scene = t.0.join("scene");
    let scene_s = scene.to_str().unwrap();
    let (code, so, se) = cli(&["synth", scene_s, "320", "180"], &[]);
    assert_eq!(code, 0, "{so}{se}");
    let combos: [(&str, Vec<(&str, &str)>); 4] = [
        ("기본(끔)", vec![]),
        ("a 자유공간", vec![("SKYLENS_FUSION_FREESPACE", "1")]),
        ("b 중앙값", vec![("SKYLENS_FUSION_MEDIAN", "1")]),
        (
            "a+b",
            vec![
                ("SKYLENS_FUSION_FREESPACE", "1"),
                ("SKYLENS_FUSION_MEDIAN", "1"),
            ],
        ),
    ];
    println!("| 조합 | 점 수 | 표면 거리 중앙 (m) | 95% (m) |");
    println!("|---|---|---|---|");
    for (i, (name, env)) in combos.iter().enumerate() {
        let out = t.0.join(format!("out{i}"));
        let out_s = out.to_str().unwrap();
        let mut args = vec!["run", scene_s, out_s, "--stride", "2"];
        args.extend(RUN_OPTS);
        args.extend(["--ba-iters", "15"]);
        let (code, so, se) = cli(&args, env);
        assert_eq!(code, 0, "{so}{se}");
        let (n, med, p95) = surface_stats(&scene, &out);
        println!("| {name} | {n} | {med:.3} | {p95:.3} |");
        let _ = std::fs::remove_dir_all(&out);
    }
}
