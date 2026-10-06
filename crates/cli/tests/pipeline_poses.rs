//! 출력 폴더의 카메라 포즈 파일(`poses/refined_{k:02}.json`, `poses/preview_{k:02}.json`)을 합성 정답과 비교한다.
//!
//! 단구역 장면(README 첫 명령, stride 2 → 40위치 × 3대 = 120장)을 `synth` → `run` 으로 만들고,
//! 정밀 포즈의 회전 오차(도: 중앙·최대; 모든 카메라에 하나의 전역 회전을 최소제곱으로 맞춘 뒤)와
//! 카메라 중심 오차(m; 정답 원점과 첫 GPS 원점의 평행 이동만 보정)를 숫자로 단언한다.
//! 정답은 `truth/cameras.txt`(이름, 내부 파라미터 6개, R 행 우선 9개, t 3개)와 `truth/origin.txt`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::align::umeyama;
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::math::rotation_angle_between;
use skylens_core::nalgebra::{Matrix3, Rotation3, Vector3};
use skylens_core::poses_io::{rotation_of, PosesFile};

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
        let p = std::env::temp_dir().join(format!("skylens_poses_{tag}_{}", std::process::id()));
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

/// 빈 목록은 NaN.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

/// 빈 목록은 NaN(0 으로 찍히면 "잔차 0" 으로 읽힌다).
fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NAN, f64::max)
}

fn geodetic_of(fields: &[&str]) -> Geodetic {
    let n: Vec<f64> = fields.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

/// 정답 좌표 → 출력 좌표(첫 GPS 기준) 평행 이동량.
fn truth_to_output_shift(input: &Path) -> [f64; 3] {
    let o = std::fs::read_to_string(input.join("truth/origin.txt")).unwrap();
    let truth_origin = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gps = std::fs::read_to_string(input.join("gps.txt")).unwrap();
    let first: Vec<&str> = gps.lines().next().unwrap().split_whitespace().collect();
    let first_gps = geodetic_of(&first[1..4]);
    let d = geodetic_to_enu(&truth_origin, &first_gps);
    [d.x, d.y, d.z]
}

/// 정답: 이름 → (회전 세계→카메라, 중심).
fn truth_poses(input: &Path) -> BTreeMap<String, (Rotation3<f64>, [f64; 3])> {
    let mut m = BTreeMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, t) = (&n[..9], &n[9..12]);
        let c = [0, 1, 2].map(|j| -(r[j] * t[0] + r[3 + j] * t[1] + r[6 + j] * t[2]));
        let rot = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(r));
        m.insert(f[0].to_string(), (rot, c));
    }
    m
}

/// `poses/{kind}_*.json` 전부를 읽어 이름 → 포즈 (먼저 나온 파일 우선).
fn read_kind(
    out: &Path,
    kind: &str,
) -> (BTreeMap<String, skylens_core::poses_io::PoseEntry>, usize) {
    let mut files: Vec<_> = std::fs::read_dir(out.join("poses"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&format!("{kind}_")) && n.ends_with(".json"))
        })
        .collect();
    files.sort();
    let mut m = BTreeMap::new();
    for f in &files {
        let pf = PosesFile::from_json(&std::fs::read_to_string(f).unwrap()).unwrap();
        assert_eq!(pf.intrinsics.4, 320, "{}", f.display());
        assert_eq!(pf.intrinsics.5, 180, "{}", f.display());
        for e in pf.poses {
            m.entry(e.name.clone()).or_insert(e);
        }
    }
    (m, files.len())
}

/// 전역 회전 G(출력 세계 → 정답 세계)를 `R_out ≈ R_truth·G` 로 최소제곱 맞춘 뒤의 카메라별 회전 오차(도).
fn rotation_errors(pairs: &[(Rotation3<f64>, Rotation3<f64>)]) -> (Vec<f64>, Vec<f64>) {
    let mut m = Matrix3::zeros();
    for (out, truth) in pairs {
        m += truth.matrix().transpose() * out.matrix();
    }
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut g = u * vt;
    if g.determinant() < 0.0 {
        let mut u2 = u;
        u2.column_mut(2).neg_mut();
        g = u2 * vt;
    }
    let g = Rotation3::from_matrix_unchecked(g);
    let raw = pairs
        .iter()
        .map(|(o, t)| rotation_angle_between(o, t).to_degrees())
        .collect();
    let aligned = pairs
        .iter()
        .map(|(o, t)| rotation_angle_between(&(o * g.inverse()), t).to_degrees())
        .collect();
    (raw, aligned)
}

struct OffBound {
    region: usize,
    shared: usize,
    own_med: f64,
    own_max: f64,
    out_med: f64,
    out_max: f64,
    vs_refined: f64,
}

/// `off` 구역별 상한 = 실측 x 1.2 (4코어 측정 기계, 초벌 -> 정밀 정렬 직후 출력 좌표).
/// 구역 1: 초벌 자체(유사변환 후) 중앙 0.736/최대 1.234 m, 출력 대 정답 중앙 3.678/최대 6.348 m, 초벌 대 정밀 중앙 3.371~3.56 m.
/// 구역 2: 초벌 자체 0.556/1.217 m, 출력 대 정답 1.887/2.737 m, 초벌 대 정밀 1.572 m.
/// 구역 0 은 사진 14장이 정렬되지 못하고(중심이 거의 일직선) 표에 넣지 않는다.
const OFF_BOUNDS: [OffBound; 2] = [
    OffBound {
        region: 1,
        shared: 32,
        own_med: 0.89,
        own_max: 1.48,
        out_med: 4.42,
        out_max: 7.62,
        vs_refined: 4.3,
    },
    OffBound {
        region: 2,
        shared: 10,
        own_med: 0.67,
        own_max: 1.46,
        out_med: 2.27,
        out_max: 3.29,
        vs_refined: 1.89,
    },
];

/// 단구역 합성 장면. 실측(4코어 측정 기계): 회전 오차 정렬 전 중앙 1.140/최대 1.592 도, 정렬 후 중앙 0.309/최대 0.488 도,
/// 중심 오차 중앙 0.256/최대 0.713 m. 상한은 실측 x 1.2.
#[test]
fn refined_pose_rotation_and_center_errors() {
    let t = TempDir::new("single");
    let (scene, out) = (t.0.join("scene"), t.0.join("out"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let (code, so, se) = cli(&["synth", scene_s, "320", "180"]);
    assert_eq!(code, 0, "{so}{se}");
    let mut args = vec!["run", scene_s, out_s, "--stride", "2"];
    args.extend(RUN_OPTS);
    let (code, so, se) = cli(&args);
    assert_eq!(code, 0, "{so}{se}");

    let truth = truth_poses(&scene);
    let shift = truth_to_output_shift(&scene);
    let (refined, n_files) = read_kind(&out, "refined");
    assert_eq!(n_files, 1, "단구역은 포즈 파일 1개");
    assert_eq!(refined.len(), 120, "등록 수");
    let (preview, n_prev) = read_kind(&out, "preview");
    assert_eq!(n_prev, 1);
    assert!(preview.len() >= 100, "초벌 포즈 수 {}", preview.len());

    let mut pairs = Vec::new();
    let mut centers = Vec::new();
    for (name, e) in &refined {
        let (tr, tc) = &truth[name];
        let q = e.quat_wxyz;
        let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "쿼터니언 크기 {norm}");
        pairs.push((rotation_of(e), *tr));
        centers.push(
            (0..3)
                .map(|k| (e.center[k] - (tc[k] + shift[k])).powi(2))
                .sum::<f64>()
                .sqrt(),
        );
    }
    let (raw, aligned) = rotation_errors(&pairs);
    let (rm, rx) = (median(&raw), max(&raw));
    let (am, ax) = (median(&aligned), max(&aligned));
    let (cm, cx) = (median(&centers), max(&centers));
    eprintln!(
        "POSES refined n {} rot raw med {rm:.4} max {rx:.4} deg, aligned med {am:.4} max {ax:.4} deg, center med {cm:.3} max {cx:.3} m",
        refined.len()
    );
    // 상한은 실측 x 1.2(위 주석의 실측값).
    assert!(rm < 1.37, "정렬 전 회전 오차 중앙 {rm} 도");
    assert!(rx < 1.91, "정렬 전 회전 오차 최대 {rx} 도");
    assert!(am < 0.37, "정렬 후 회전 오차 중앙 {am} 도");
    assert!(ax < 0.59, "정렬 후 회전 오차 최대 {ax} 도");
    assert!(cm < 0.31, "중심 오차 중앙 {cm} m");
    assert!(cx < 0.86, "중심 오차 최대 {cx} m");

    // 초벌 포즈도 같은 형식이고 회전이 정답에 가깝다(느슨한 기준).
    let pp: Vec<_> = preview
        .iter()
        .map(|(n, e)| (rotation_of(e), truth[n].0))
        .collect();
    let (_, pa) = rotation_errors(&pp);
    eprintln!(
        "POSES preview n {} rot aligned med {:.4} max {:.4} deg",
        pp.len(),
        median(&pa),
        max(&pa)
    );
    // 실측: 정렬 후 중앙 0.258/최대 0.593 도. 상한은 실측 x 1.2.
    assert!(median(&pa) < 0.31, "초벌 회전 오차 중앙 {}", median(&pa));
    assert!(max(&pa) < 0.72, "초벌 회전 오차 최대 {}", max(&pa));
}

/// 기본 합성 장면(위치 수 기본값) + `--span 12 --coarse-back off`: 구역마다 초벌 포즈 파일의 사진 이름과 포즈가 짝이 맞아야 한다.
/// 정밀 다시 등록으로 구역 사진 목록이 바뀌어도 초벌 포즈는 초벌 때의 사진 번호와 짝지어 쓴다.
/// 같은 사진의 정밀 중심(출력 좌표)과 초벌 중심의 차이 중앙값을 구역별로 단언한다.
/// 또 초벌 중심을 정답과 직접 비교해 초벌 모델 자체 오차와 초벌 -> 정밀 정렬 잔차를 나눈다.
#[test]
fn coarse_back_off_preview_poses_stay_paired() {
    let t = TempDir::new("off");
    let (scene, out) = (t.0.join("scene"), t.0.join("out"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let (code, so, se) = cli(&["synth", scene_s, "320", "180"]);
    assert_eq!(code, 0, "{so}{se}");
    let args = [
        "run",
        scene_s,
        out_s,
        "--span",
        "12",
        "--coarse-back",
        "off",
        "--max-features",
        "800",
        "--dense-width",
        "96",
        "--hfov",
        "65",
        "--ba-iters",
        "15",
    ];
    let (code, so, se) = cli(&args);
    assert_eq!(code, 0, "{so}{se}");

    let truth = truth_poses(&scene);
    let shift = truth_to_output_shift(&scene);
    let registered: Vec<usize> = so
        .lines()
        .filter(|l| l.starts_with("region "))
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            let i = w.iter().position(|x| *x == "registered").unwrap();
            w[i + 1].split('/').next().unwrap().parse().unwrap()
        })
        .collect();
    let mut refined = BTreeMap::new();
    let mut n_regions = 0;
    for k in 0.. {
        let f = out.join("poses").join(format!("refined_{k:02}.json"));
        let Ok(txt) = std::fs::read_to_string(&f) else {
            break;
        };
        n_regions += 1;
        for e in PosesFile::from_json(&txt).unwrap().poses {
            refined.entry(e.name.clone()).or_insert(e);
        }
    }
    assert!(n_regions >= 3, "구역 수 {n_regions}");
    assert_eq!(registered.len(), n_regions);
    // 상한 표를 실제로 단언한 구역. 표 구역이 정렬되지 않거나 사라지면 아래에서 빠진 채 끝나므로 마지막에 잡는다.
    let mut asserted: Vec<usize> = Vec::new();
    for (k, &n_reg) in registered.iter().enumerate() {
        let f = out.join("poses").join(format!("preview_{k:02}.json"));
        let pf = PosesFile::from_json(&std::fs::read_to_string(&f).unwrap()).unwrap();
        let mut d = Vec::new();
        // 같은 사진(정밀 모델에도 등록된 것)의 정답 대비 중심 오차: 초벌(출력 좌표 그대로)·정밀.
        let (mut direct_shared, mut refined_shared) = (Vec::new(), Vec::new());
        // 초벌 전 사진의 (초벌 중심, 정답 중심).
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        let dist =
            |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        for e in &pf.poses {
            let tc = &truth[&e.name].1;
            let tco = [0, 1, 2].map(|i| tc[i] + shift[i]);
            src.push(Vector3::from(e.center));
            dst.push(Vector3::from(tco));
            // 정밀 모델에 등록되지 않은 사진은 비교에서 뺀다.
            let Some(r) = refined.get(&e.name) else {
                continue;
            };
            direct_shared.push(dist(e.center, tco));
            refined_shared.push(dist(r.center, tco));
            d.push(
                (0..3)
                    .map(|i| (e.center[i] - r.center[i]).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
        }
        // 초벌 포즈 수는 초벌 등록 수(`region k positions P registered R/N`)와 같아야 한다.
        assert_eq!(pf.poses.len(), n_reg, "구역 {k} 초벌 포즈 수");
        // 초벌 모델 자체 오차: 초벌 중심을 정답 중심에 유사변환으로 맞춘 뒤(전 사진) 남는 거리.
        let sim = umeyama(&src, &dst);
        if !pf.aligned {
            // 초벌이 정밀 모델과 겹치는 사진이 없어 정렬되지 못한 구역은 좌표가 달라 비교하지 않는다.
            let (own_med, own_max) = sim.as_ref().map_or((f64::NAN, f64::NAN), |sm| {
                let e: Vec<f64> = src
                    .iter()
                    .zip(&dst)
                    .map(|(a, b)| (sm.apply_point(a) - b).norm())
                    .collect();
                (median(&e), max(&e))
            });
            eprintln!(
                "POSES off preview_{k:02} n {} not aligned | coarse-own(sim-aligned to truth) med {own_med:.3} max {own_max:.3} m (nan = degenerate fit)",
                pf.poses.len()
            );
            continue;
        }
        let sim = sim.expect("초벌 유사변환");
        let own: Vec<f64> = src
            .iter()
            .zip(&dst)
            .map(|(a, b)| (sim.apply_point(a) - b).norm())
            .collect();
        assert!(!d.is_empty(), "구역 {k} 비교 가능한 사진 없음");
        eprintln!(
            "POSES off preview_{k:02} shared {} of {} | coarse-own(sim-aligned to truth, all {}) med {:.3} max {:.3} m scale {:.3} | coarse-as-output vs truth(shared) med {:.3} max {:.3} m | refined vs truth(shared) med {:.3} max {:.3} m",
            d.len(), pf.poses.len(), own.len(), median(&own), max(&own), sim.s,
            median(&direct_shared), max(&direct_shared), median(&refined_shared), max(&refined_shared)
        );
        let (md, mx) = (median(&d), max(&d));
        eprintln!(
            "POSES off preview_{k:02} n {} aligned {} center-vs-refined med {md:.3} max {mx:.3} m",
            pf.poses.len(),
            pf.aligned
        );
        // 구역별 상한 표(실측 x 1.2; 실측값은 아래 주석). 표에 없는 구역은 측정만 하고 단언하지 않는다.
        let Some(b) = OFF_BOUNDS.iter().find(|b| b.region == k) else {
            eprintln!("POSES off preview_{k:02} 상한 표에 없음: 측정만");
            continue;
        };
        asserted.push(k);
        assert_eq!(d.len(), b.shared, "구역 {k} 공유 사진 수");
        assert!(md < b.vs_refined, "구역 {k} 초벌-정밀 중심 차 중앙 {md} m");
        assert!(
            median(&own) < b.own_med,
            "구역 {k} 초벌 자체 오차 중앙 {} m",
            median(&own)
        );
        assert!(
            max(&own) < b.own_max,
            "구역 {k} 초벌 자체 오차 최대 {} m",
            max(&own)
        );
        let (dm, dx) = (median(&direct_shared), max(&direct_shared));
        assert!(dm < b.out_med, "구역 {k} 정답 대비 초벌 중심 중앙 {dm} m");
        assert!(dx < b.out_max, "구역 {k} 정답 대비 초벌 중심 최대 {dx} m");
    }
    // 표 구역이 정렬되지 않았거나 없으면 단언이 하나도 돌지 않으므로 여기서 실패시킨다.
    for b in &OFF_BOUNDS {
        assert!(
            asserted.contains(&b.region),
            "상한 표 구역 {} 이 단언되지 않음(정렬 안 됨 또는 구역 없음): 단언된 구역 {asserted:?}",
            b.region
        );
    }
}

/// `--coarse-back off` 한 번 실행: (임시 폴더, 장면, 출력, 표준 출력).
fn run_off(tag: &str) -> (TempDir, PathBuf, PathBuf, String) {
    let t = TempDir::new(tag);
    let (scene, out) = (t.0.join("scene"), t.0.join("out"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let (code, so, se) = cli(&["synth", scene_s, "320", "180"]);
    assert_eq!(code, 0, "{so}{se}");
    let args = [
        "run",
        scene_s,
        out_s,
        "--span",
        "12",
        "--coarse-back",
        "off",
        "--max-features",
        "800",
        "--dense-width",
        "96",
        "--hfov",
        "65",
        "--ba-iters",
        "15",
    ];
    let (code, so, se) = cli(&args);
    assert_eq!(code, 0, "{so}{se}");
    (t, scene, out, so)
}

/// 유사변환 `sim` 이 항등에서 얼마나 벗어났는지: (축척 차 %, 회전 도, `c` 에서의 이동 m).
fn deviation(sim: &skylens_core::align::Similarity, c: &Vector3<f64>) -> (f64, f64, f64) {
    (
        (sim.s - 1.0) * 100.0,
        sim.r.angle().to_degrees(),
        (sim.apply_point(c) - c).norm(),
    )
}

/// 점 집합의 주축 특잇값 비 (둘째/첫째, 셋째/첫째). 1 에 가까우면 고르게 퍼짐, 0 에 가까우면 일직선·평면.
fn spread_ratios(p: &[Vector3<f64>]) -> (f64, f64) {
    let n = p.len() as f64;
    let c = p.iter().fold(Vector3::zeros(), |a, x| a + x) / n;
    let mut m = Matrix3::zeros();
    for x in p {
        let d = x - c;
        m += d * d.transpose();
    }
    let sv = m.svd(false, false).singular_values;
    let mut v = [sv[0].sqrt(), sv[1].sqrt(), sv[2].sqrt()];
    v.sort_by(|a, b| b.total_cmp(a));
    (v[1] / v[0], v[2] / v[0])
}

/// 측정(`--ignored --nocapture`): `off` 에서 초벌 -> 정밀 정렬이 어느 단계에서 잔차를 내는지.
/// 출력 파일만으로 본다. 구역마다 같은 사진의 중심 쌍 (초벌 출력, 정밀 출력, 정답) 세 가지 사이의
/// 최적 유사변환(Umeyama)을 구해, 현재 출력이 그 최적에서 얼마나 벗어났는지(축척 %, 회전 도, 이동 m)와
/// 변환 뒤 남는 잔차(중앙/최대 m)를 표로 낸다. 구역 0 은 중심 배치의 퇴화(특잇값 비)를 낸다.
#[test]
#[ignore = "측정용: cargo test --release --test pipeline_poses align_stage_table -- --ignored --nocapture"]
fn align_stage_table() {
    let (_t, scene, out, so) = run_off("stages");
    for l in so.lines().filter(|l| l.contains("pairs")) {
        eprintln!("STAGE log {l}");
    }
    let truth = truth_poses(&scene);
    let shift = truth_to_output_shift(&scene);
    let (refined, _) = read_kind(&out, "refined");
    let (_, n_prev) = read_kind(&out, "preview");
    assert!(n_prev >= 3);
    eprintln!("STAGE region | step | n | scale diff % | rot deg | shift m | residual med m | residual max m");
    for k in 0..n_prev {
        let pf = PosesFile::from_json(
            &std::fs::read_to_string(out.join("poses").join(format!("preview_{k:02}.json")))
                .unwrap(),
        )
        .unwrap();
        let ctr = |e: &skylens_core::poses_io::PoseEntry| Vector3::from(e.center);
        let tru = |n: &str| {
            let c = truth[n].1;
            Vector3::new(c[0] + shift[0], c[1] + shift[1], c[2] + shift[2])
        };
        let (mut p, mut r, mut t) = (Vec::new(), Vec::new(), Vec::new());
        let mut all_p = Vec::new();
        let mut all_t = Vec::new();
        for e in &pf.poses {
            all_p.push(ctr(e));
            all_t.push(tru(&e.name));
            if let Some(re) = refined.get(&e.name) {
                p.push(ctr(e));
                r.push(ctr(re));
                t.push(tru(&e.name));
            }
        }
        let (sp, tp) = (spread_ratios(&all_p), spread_ratios(&all_t));
        eprintln!(
            "STAGE {k} | layout n {} aligned {} | preview centers sv2/sv1 {:.3} sv3/sv1 {:.3} | truth centers sv2/sv1 {:.3} sv3/sv1 {:.3}",
            all_p.len(), pf.aligned, sp.0, sp.1, tp.0, tp.1
        );
        let row = |step: &str, src: &[Vector3<f64>], dst: &[Vector3<f64>]| {
            let raw: Vec<f64> = src.iter().zip(dst).map(|(a, b)| (a - b).norm()).collect();
            match umeyama(src, dst) {
                Some(sm) => {
                    let (ds, dr, dt) = deviation(&sm, &src[0]);
                    let res: Vec<f64> = src
                        .iter()
                        .zip(dst)
                        .map(|(a, b)| (sm.apply_point(a) - b).norm())
                        .collect();
                    eprintln!(
                        "STAGE {k} | {step} | {} | {ds:+.2} | {dr:.3} | {dt:.3} | {:.3} | {:.3} | as-is med {:.3} max {:.3}",
                        src.len(), median(&res), max(&res), median(&raw), max(&raw)
                    );
                }
                None => eprintln!(
                    "STAGE {k} | {step} | {} | degenerate (no similarity) | as-is med {:.3} max {:.3}",
                    src.len(), median(&raw), max(&raw)
                ),
            }
        };
        // 초벌 출력 -> 정답 (공유 사진): 현재 출력이 중심 기준 최적에서 벗어난 정도 = 정렬 단계의 오차.
        row("preview-out -> truth (shared)", &p, &t);
        // 초벌 출력 -> 정밀 출력: 정렬이 초벌을 정밀 좌표에 제대로 얹었다면 항등 근처여야 한다.
        row("preview-out -> refined-out", &p, &r);
        // 정밀 출력 -> 정답: 정밀 모델 자체가 정답에 얼마나 가까운지(기준선).
        row("refined-out -> truth", &r, &t);
        // 초벌 전체(공유 아닌 사진 포함) -> 정답: 초벌 모델 자체 오차.
        row("preview-out -> truth (all)", &all_p, &all_t);
    }
}

/// 사진 이름 `camF_0036` 의 위치 번호(이름 끝 번호는 사진 번호, 위치마다 3장).
fn position_of(name: &str) -> usize {
    name.rsplit('_').next().unwrap().parse::<usize>().unwrap() / 3
}

/// 점 집합의 축별 범위 중 가장 긴 주축 방향 길이(m): 중심에서 주축으로 투영한 최대-최소.
fn principal_range(p: &[Vector3<f64>]) -> f64 {
    if p.len() < 2 {
        return 0.0;
    }
    let n = p.len() as f64;
    let c = p.iter().fold(Vector3::zeros(), |a, x| a + x) / n;
    let mut m = Matrix3::zeros();
    for x in p {
        let d = x - c;
        m += d * d.transpose();
    }
    let svd = m.svd(false, true);
    let vt = svd.v_t.unwrap();
    let mut best = (0.0, 0);
    for i in 0..3 {
        if svd.singular_values[i] > best.0 {
            best = (svd.singular_values[i], i);
        }
    }
    let axis = vt.row(best.1).transpose();
    let t: Vec<f64> = p.iter().map(|x| (x - c).dot(&axis)).collect();
    max(&t) - t.iter().copied().fold(f64::NAN, f64::min)
}

/// 측정(`--ignored --nocapture`): 점 쌍 쏠림·외삽 가설을 카메라 중심으로 대신 본다(`PAIRSPREAD` 줄).
///
/// 정렬에 쓰인 공유 3D 점 쌍(트랙 대응)은 출력 파일에 없고 시험에서 만들 수 없어 점 단위 분포는 얻지 못한다.
/// 대신 구역 안 사진 중심으로 같은 구조를 만든다. 구역 k 의 사진을 "겹침 띠"(`align_window`, 앞 구역과 겹치는 위치)와
/// 나머지로 나눠, (a) 띠 안 사진 비율·띠 범위 대 구역 범위·주축 비, (b) 띠 사진만으로 구한 유사변환을
/// 나머지 사진에 적용했을 때의 잔차(외삽)를 전체 사진으로 구한 변환의 잔차와 비교한다.
/// 띠 + 전체를 함께 넣는 경우는 중심 항을 더한 정렬에 해당한다. 모두 계산만 하며 제품 동작은 바꾸지 않는다.
#[test]
#[ignore = "측정용: cargo test --release --test pipeline_poses align_pair_spread -- --ignored --nocapture"]
fn align_pair_spread() {
    let (_t, scene, out, _so) = run_off("spread");
    let truth = truth_poses(&scene);
    let shift = truth_to_output_shift(&scene);
    let n_pos = truth.keys().map(|n| position_of(n) + 1).max().unwrap();
    let regions = skylens_core::stream::split_regions(n_pos, 12, 2);
    let (refined, _) = read_kind(&out, "refined");
    let (_, n_prev) = read_kind(&out, "preview");
    assert!(n_prev >= 3);
    let tru = |n: &str| {
        let c = truth[n].1;
        Vector3::new(c[0] + shift[0], c[1] + shift[1], c[2] + shift[2])
    };
    eprintln!("PAIRSPREAD region | cams | in-band | band/region range | sv2/sv1 band | sv2/sv1 region | fit set | fit residual med/max m | non-band residual med/max m | scale diff % | rot deg");
    for (k, r) in regions.iter().enumerate().take(n_prev) {
        let pf = PosesFile::from_json(
            &std::fs::read_to_string(out.join("poses").join(format!("preview_{k:02}.json")))
                .unwrap(),
        )
        .unwrap();
        let band = skylens_core::stream::align_window(r, 2, n_pos);
        let (mut all_p, mut all_t, mut is_band) = (Vec::new(), Vec::new(), Vec::new());
        let (mut ref_p, mut ref_r) = (Vec::new(), Vec::new());
        for e in &pf.poses {
            let pos = position_of(&e.name);
            all_p.push(Vector3::from(e.center));
            all_t.push(tru(&e.name));
            is_band.push(pos >= band.0 && pos < band.1);
            if let Some(re) = refined.get(&e.name) {
                ref_p.push(Vector3::from(e.center));
                ref_r.push(Vector3::from(re.center));
            }
        }
        let pick = |v: &[Vector3<f64>], want: bool| -> Vec<Vector3<f64>> {
            v.iter()
                .zip(&is_band)
                .filter(|(_, &b)| b == want)
                .map(|(x, _)| *x)
                .collect()
        };
        let (bp, bt) = (pick(&all_p, true), pick(&all_t, true));
        let (np, nt) = (pick(&all_p, false), pick(&all_t, false));
        let (sb, sa) = (spread_ratios(&bp), spread_ratios(&all_p));
        eprintln!(
            "PAIRSPREAD {k} | {} cams, aligned {} | in-band {} ({:.2}) | band/region principal range {:.1}/{:.1} m = {:.2} | sv2/sv1 band {:.3} region {:.3} | shared with refined {}",
            all_p.len(), pf.aligned, bp.len(), bp.len() as f64 / all_p.len().max(1) as f64,
            principal_range(&bp), principal_range(&all_p),
            principal_range(&bp) / principal_range(&all_p).max(1e-9), sb.0, sa.0, ref_p.len()
        );
        let line = |tag: &str,
                    fit_s: &[Vector3<f64>],
                    fit_d: &[Vector3<f64>],
                    extra_s: &[Vector3<f64>],
                    extra_d: &[Vector3<f64>]| {
            let mut s = fit_s.to_vec();
            let mut d = fit_d.to_vec();
            s.extend_from_slice(extra_s);
            d.extend_from_slice(extra_d);
            let Some(sm) = umeyama(&s, &d) else {
                eprintln!("PAIRSPREAD {k} | {tag} | degenerate");
                return;
            };
            let res = |a: &[Vector3<f64>], b: &[Vector3<f64>]| -> Vec<f64> {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (sm.apply_point(x) - y).norm())
                    .collect()
            };
            let (rf, rn) = (res(&s, &d), res(&np, &nt));
            eprintln!(
                "PAIRSPREAD {k} | {tag} | fit n {} | fit res {:.3}/{:.3} | non-band res {:.3}/{:.3} (n {}) | scale {:+.2} % rot {:.2} deg",
                s.len(), median(&rf), max(&rf), median(&rn), max(&rn), np.len(),
                (sm.s - 1.0) * 100.0, sm.r.angle().to_degrees()
            );
        };
        line("all cameras", &all_p, &all_t, &[], &[]);
        line("band cameras only", &bp, &bt, &[], &[]);
        line(
            "band + all cameras (center term added)",
            &bp,
            &bt,
            &all_p,
            &all_t,
        );
    }
}
