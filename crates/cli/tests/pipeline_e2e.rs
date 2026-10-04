//! 명령행 끝까지 잇기: `synth` → `run` → `verify` 를 프로세스로 돌려, README 의 명령과 같은 설정에서
//! 출력 폴더 구성·verify 항목별 판정·합성 정답 대비 카메라 중심·점 오차를 숫자로 고정한다(F-273).
//!
//! 정답 비교: 정답은 `truth/cameras.txt`(원점 `truth/origin.txt` 기준 동-북-위)이고 출력은 첫 GPS 원점 좌표라서
//! 두 원점의 차만큼 옮겨 비교한다. 정밀 점은 `refined/*.ply` 전부, 정답 표면은 `Scene::surface_height`
//! (같은 기본 장면, 수직 거리 근사)다. 회전 오차는 재지 않는다: 출력(`poses.txt`)에 카메라 중심만 있고 회전이 없다.
//! 상한은 측정값의 약 1.2 배(근거는 연구 노트 experiments/pipeline-e2e.md).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};

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

struct Outcome {
    verify: String,
    verify_code: i32,
    m: Metrics,
}

/// README 명령 그대로: synth → run → verify. 출력 폴더 구성도 확인한다.
fn pipeline(tag: &str, stride: &str) -> Outcome {
    let t = TempDir::new(tag);
    let (scene, out) = (t.0.join("scene"), t.0.join("out"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let (code, so, se) = cli(&["synth", scene_s, "320", "180"]);
    assert_eq!(code, 0, "{so}{se}");
    assert!(so.contains("views 240"), "{so}");

    let mut args = vec!["run", scene_s, out_s, "--stride", stride];
    args.extend(RUN_OPTS);
    let start = Instant::now();
    let (code, so, se) = cli(&args);
    let secs = start.elapsed().as_secs_f64();
    assert_eq!(code, 0, "{so}{se}");
    eprintln!("run secs {secs:.1}");

    for d in ["preview", "refined", "snapshots"] {
        assert!(out.join(d).is_dir(), "{d} 폴더 없음");
    }
    for f in ["snapshots/manifest.json", "poses.txt", "report.json"] {
        assert!(out.join(f).is_file(), "{f} 없음");
    }
    let (verify_code, table, se) = cli(&["verify", out_s]);
    eprintln!("{table}verify exit {verify_code} {se}");
    let m = measure(&scene, &out);
    eprintln!(
        "E2E registered {} center med {:.3} max {:.3} points {} surface med {:.3} p95 {:.3}",
        m.registered, m.center_med, m.center_max, m.points, m.surface_med, m.surface_p95
    );
    Outcome {
        verify: table,
        verify_code,
        m,
    }
}

fn expect_items(o: &Outcome, expected: &[(&str, &str)]) {
    for (name, want) in expected {
        let (got, measured) = item(&o.verify, name);
        assert_eq!(&got, want, "{name}: {measured}");
    }
}

/// 단구역(README 첫 명령, stride 2 → 40위치 × 3대 = 120장). 4코어 부하 20 에서 run 약 2분.
/// 실측(초벌 다듬기 5회 기본, 반점 제거 뒤): verify 7/7 종료 코드 0, 중심 오차 중앙 0.312/최대 0.851 m,
/// 정밀 점 표면 거리 중앙 0.336/95% 0.932 m (점 10302개). 상한은 실측 x 1.2 (0.375, 1.021, 0.403, 1.118)를 넉넉히 올림.
#[test]
fn single_region_end_to_end() {
    let o = pipeline("single", "2");
    assert_eq!(o.m.registered, 120, "등록 수");
    assert_eq!(o.verify_code, 0, "{}", o.verify);
    expect_items(
        &o,
        &[
            ("registered", "PASS"),
            ("region_images", "PASS"),
            ("refined_reprojection", "PASS"),
            ("preview_align", "PASS"),
            // 통과: 최근접 중앙 최대 0.688 m (< 3), 높이 차 중앙 최대 0.564 m (< 2). 초벌 재투영 0.948 px.
            ("preview_vs_refined", "PASS"),
            ("refined_overlap", "PASS"),
            ("snapshots", "PASS"),
        ],
    );
    let (_, ov) = item(&o.verify, "refined_overlap");
    assert!(ov.contains("해당 없음"), "구역 1개는 겹침 해당 없음: {ov}");
    let (_, pr) = item(&o.verify, "preview_vs_refined");
    let h = number_after(&pr, "높이 차 중앙 최대 ");
    assert!(h < 2.0, "높이 차 상한: {pr}");
    assert!(number_after(&pr, "최근접 중앙 최대 ") < 3.0, "{pr}");
    assert!(o.m.center_med < 0.38, "중심 오차 중앙 {}", o.m.center_med);
    assert!(o.m.center_max < 1.05, "중심 오차 최대 {}", o.m.center_max);
    assert!(o.m.surface_med < 0.41, "점 중앙 {}", o.m.surface_med);
    assert!(o.m.surface_p95 < 1.12, "점 95% {}", o.m.surface_p95);
    assert!(
        o.m.points >= 10302 * 4 / 5 && o.m.points <= 10302 * 6 / 5,
        "점 수 {}",
        o.m.points
    );
}

/// 구역 2개(README 둘째 명령, stride 1 → 80위치 × 3대 = 240장): refined_overlap 이 실제로 판정된다.
/// 시험 전체 약 3 분(run 약 150 초). 실측(초벌 다듬기 5회 기본, 반점 제거 뒤): verify 7/7 종료 코드 0, 중심 오차 중앙 0.279/최대 3.152 m,
/// 정밀 점 표면 거리 중앙 0.431/95% 1.336 m (점 18575개). 상한은 실측 x 1.2 안팎.
/// preview_vs_refined 통과(최근접 0.970 m < 3, 높이 차 1.026 m < 2). preview_align 통과: 구역 간 점쌍을 구역의
/// 모든 이미지 관측으로 만들어 점쌍 최소 1178 (정렬 창 12장만 쓰면 324), 스케일 차 0.26%, 잔차 중앙 최대 0.735 m. 초벌 재투영 0.673 px.
#[test]
fn two_region_end_to_end() {
    let o = pipeline("two", "1");
    assert_eq!(o.m.registered, 240, "등록 수");
    assert_eq!(o.verify_code, 0, "{}", o.verify);
    expect_items(
        &o,
        &[
            ("registered", "PASS"),
            ("region_images", "PASS"),
            ("refined_reprojection", "PASS"),
            ("preview_align", "PASS"),
            ("preview_vs_refined", "PASS"),
            ("refined_overlap", "PASS"),
            ("snapshots", "PASS"),
        ],
    );
    let (_, ov) = item(&o.verify, "refined_overlap");
    assert!(!ov.contains("해당 없음"), "겹침이 판정돼야 함: {ov}");
    assert!(number_after(&ov, "중앙 최대 ") < 0.3, "{ov}");
    let (_, pr) = item(&o.verify, "preview_vs_refined");
    assert!(number_after(&pr, "높이 차 중앙 최대 ") < 2.0, "{pr}");
    assert!(number_after(&pr, "최근접 중앙 최대 ") < 3.0, "{pr}");
    let (_, pa) = item(&o.verify, "preview_align");
    assert!(number_after(&pa, "점쌍 최소 ") >= 1000.0, "{pa}");
    assert!(number_after(&pa, "구역 간 스케일 차 ") <= 1.0, "{pa}");
    assert!(number_after(&pa, "잔차 중앙 최대 ") < 6.0, "{pa}");
    assert!(o.m.center_med < 0.34, "중심 오차 중앙 {}", o.m.center_med);
    assert!(o.m.center_max < 3.75, "중심 오차 최대 {}", o.m.center_max);
    assert!(o.m.surface_med < 0.52, "점 중앙 {}", o.m.surface_med);
    assert!(o.m.surface_p95 < 1.70, "점 95% {}", o.m.surface_p95);
    // 현재 출력(반점 제거 뒤): 점 18575, 표면 거리 중앙 0.431 m, 95% 1.336 m.
    // 95% 상한 1.70 m 는 현재 출력 1.336 m 의 약 1.27 배.
    assert!(
        o.m.points >= 18575 * 4 / 5 && o.m.points <= 18575 * 6 / 5,
        "점 수 {}",
        o.m.points
    );
}
