//! GPS 사전항 무게·강건화 비교(측정, 기본 동작 불변). 환경 변수 `SKYLENS_GPS_PRIOR` 로 조건을 주고
//! 기본 경로(축소 크기)를 시드 1·2·3 으로 돌려 구역별 카메라 간격 비, 구역 간 축척 차, 정밀 중심 오차를 낸다.
//! 실행: `cargo test --release -j 2 --test zone2_gps_prior -- --ignored --nocapture` (`ZGP_SEEDS=3` 처럼 줄일 수 있다).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::nalgebra::Vector3;
use skylens_core::stream::ALIGN_SCALE_TOL;
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::scale_spread;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_zgp_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // ZGP_KEEP=1 이면 출력 폴더를 남겨 읽기 쪽만 다시 돌릴 수 있게 한다.
        if std::env::var("ZGP_KEEP").is_ok() {
            eprintln!("ZONESCALE kept {}", self.0.display());
            return;
        }
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

fn geodetic_of(fields: &[&str]) -> Geodetic {
    let n: Vec<f64> = fields.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

fn truth_shift(input: &Path) -> Vector3<f64> {
    let o = std::fs::read_to_string(input.join("truth/origin.txt")).unwrap();
    let origin = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gps = std::fs::read_to_string(input.join("gps.txt")).unwrap();
    let first: Vec<&str> = gps.lines().next().unwrap().split_whitespace().collect();
    geodetic_to_enu(&origin, &geodetic_of(&first[1..4]))
}

/// 사진 번호(3 * 위치 + 카메라) → 정답 카메라 중심(GPS 좌표계).
fn truth_centers(input: &Path) -> BTreeMap<usize, Vector3<f64>> {
    let shift = truth_shift(input);
    let mut m = BTreeMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let frame: usize = f[0].rsplit('_').next().unwrap().parse().unwrap();
        let cam = ["camF", "camR", "camL"]
            .iter()
            .position(|c| f[0].starts_with(c))
            .unwrap();
        let n: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, t) = (&n[..9], &n[9..12]);
        let c = Vector3::new(
            -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
            -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
            -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
        );
        m.insert(frame * 3 + cam, c + shift);
    }
    m
}

#[allow(dead_code)]
struct Img {
    gid: usize,
    gps: Vector3<f64>,
    placed: Vector3<f64>,
    fin: Vector3<f64>,
}

fn v3(f: &[&str]) -> Vector3<f64> {
    let n: Vec<f64> = f.iter().map(|s| s.parse().unwrap()).collect();
    Vector3::new(n[0], n[1], n[2])
}

/// 구역 진단 파일: 사진마다 "I 번호 GPS 위치풀이(placed) 최종(fin)". 위치 평균을 쓰지 않는 기본 경로에서는 맞춤 전 중심이 NaN 이다.
fn read_zone(path: &Path) -> Vec<Img> {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut imgs = Vec::new();
    for l in txt.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f[0] == "I" {
            imgs.push(Img {
                gid: f[1].parse().unwrap(),
                gps: v3(&f[2..5]),
                placed: v3(&f[8..11]),
                fin: v3(&f[11..14]),
            });
        }
    }
    imgs
}

fn quant(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// 점 집합의 주축 표준편차(큰 것부터 3개)와 첫 주축 범위(m).
/// manifest.json 의 align 기록: 구역 → (점쌍, 잔차 중앙, 축척).
fn read_align(path: &Path) -> BTreeMap<usize, (f64, f64, f64)> {
    let txt = std::fs::read_to_string(path).unwrap();
    let a = txt.find("\"align\"").expect("align 없음");
    let body = &txt[a..];
    let num = |obj: &str, key: &str| -> f64 {
        let k = obj.find(&format!("\"{key}\"")).unwrap();
        let rest = obj[k..].split(':').nth(1).unwrap();
        rest.trim_start()
            .split(|c: char| c == ',' || c == '}' || c.is_whitespace())
            .next()
            .unwrap()
            .parse()
            .unwrap_or(f64::NAN)
    };
    let mut m = BTreeMap::new();
    for obj in body.split('{').skip(1) {
        let obj = obj.split('}').next().unwrap();
        if obj.contains("\"region\"") && obj.contains("\"scale\"") {
            m.insert(
                num(obj, "region") as usize,
                (
                    num(obj, "pairs"),
                    num(obj, "fit_median_m"),
                    num(obj, "scale"),
                ),
            );
        }
    }
    m
}

const ARGS: [&str; 12] = [
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

fn make_scene(dir: &Path, seed: u64) {
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        seed,
        ..SceneConfig::default()
    });
    scene.write_dataset(dir).unwrap();
}

struct Row {
    seed: u64,
    cond: String,
    rig: Vec<f64>,
    scales: Vec<f64>,
    spread: f64,
    center_med: f64,
    registered: usize,
    verify: String,
}

fn run_cond(scene_dir: &Path, base: &Path, seed: u64, cond: &str, spec: &str) -> Row {
    let (out, dump) = (base.join("out"), base.join("dump"));
    let mut args = vec!["run", scene_dir.to_str().unwrap(), out.to_str().unwrap()];
    args.extend(ARGS.iter());
    let mut env = vec![("SKYLENS_ZONE_DUMP", dump.to_str().unwrap())];
    if !spec.is_empty() {
        env.push(("SKYLENS_GPS_PRIOR", spec));
    }
    let (code, so, se) = cli(&args, &env);
    assert_eq!(code, 0, "{cond}: {so}{se}");
    let (vcode, vo, _) = cli(&["verify", out.to_str().unwrap()], &[]);
    let verify = format!(
        "exit {vcode} {}",
        vo.lines().find(|l| l.starts_with("결과")).unwrap_or("")
    );
    let truth = truth_centers(scene_dir);
    let shift = truth_shift(scene_dir);
    let _ = shift;
    let n_zones = (0..)
        .take_while(|zi| dump.join(format!("zone_scale_{zi:02}.txt")).exists())
        .count();
    let align = read_align(&out.join("snapshots/manifest.json"));
    let (mut rig, mut scales) = (vec![], vec![]);
    for zi in 0..n_zones {
        let imgs = read_zone(&dump.join(format!("zone_scale_{zi:02}.txt")));
        let ok: Vec<&Img> = imgs
            .iter()
            .filter(|i| i.placed.iter().all(|v| v.is_finite()) && truth.contains_key(&i.gid))
            .collect();
        let mut r = vec![];
        for (x, ix) in ok.iter().enumerate() {
            for iy in ok.iter().skip(x + 1) {
                if ix.gid / 3 == iy.gid / 3 {
                    r.push(
                        (ix.placed - iy.placed).norm() / (truth[&ix.gid] - truth[&iy.gid]).norm(),
                    );
                }
            }
        }
        rig.push(quant(&r, 0.5));
        scales.push(align.get(&zi).map_or(f64::NAN, |a| a.2));
    }
    // 정밀 중심 오차: poses.txt(첫 GPS 기준) 대 정답 + 첫 GPS 이동.
    let mut errs = vec![];
    for l in std::fs::read_to_string(out.join("poses.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let c = v3(&f[1..4]);
        let gid = {
            let frame: usize = f[0].rsplit('_').next().unwrap().parse().unwrap();
            let cam = ["camF", "camR", "camL"]
                .iter()
                .position(|p| f[0].starts_with(p))
                .unwrap();
            frame * 3 + cam
        };
        if let Some(t) = truth.get(&gid) {
            errs.push((c - t).norm());
        }
    }
    let spread = scale_spread(&scales);
    Row {
        seed,
        cond: cond.to_string(),
        rig,
        scales,
        spread,
        center_med: quant(&errs, 0.5),
        registered: errs.len(),
        verify,
    }
}

const CONDS: [(&str, &str); 9] = [
    ("base", ""),
    ("w0.25", "w=0.25"),
    ("w0.5", "w=0.5"),
    ("w2", "w=2"),
    ("huber1", "loss=huber,thr=1"),
    ("huber2", "loss=huber,thr=2"),
    ("cauchy1", "loss=cauchy,thr=1"),
    ("cauchy2", "loss=cauchy,thr=2"),
    ("post", "post=1"),
];

#[test]
#[ignore = "측정용: cargo test --release -j 2 --test zone2_gps_prior -- --ignored --nocapture"]
fn gps_prior_conditions() {
    let seeds: Vec<u64> = match std::env::var("ZGP_SEEDS") {
        Ok(s) => s.split(',').filter_map(|x| x.parse().ok()).collect(),
        Err(_) => vec![3, 1, 2],
    };
    let only = std::env::var("ZGP_CONDS").ok();
    let mut rows: Vec<Row> = vec![];
    for seed in seeds {
        let t = TempDir::new(&format!("s{seed}"));
        let scene = t.0.join("scene");
        make_scene(&scene, seed);
        let conds: Vec<&(&str, &str)> = CONDS
            .iter()
            .filter(|(n, _)| only.as_ref().is_none_or(|o| o.split(',').any(|x| x == *n)))
            .collect();
        for chunk in conds.chunks(3) {
            let hs: Vec<_> = chunk
                .iter()
                .map(|&&(n, spec)| {
                    let (scene, base) = (scene.clone(), t.0.join(n));
                    std::thread::spawn(move || run_cond(&scene, &base, seed, n, spec))
                })
                .collect();
            for h in hs {
                let r = h.join().unwrap();
                eprintln!(
                    "ZGP seed {} {:<11} rig {:.3?} scales {:.3?} spread {:.3} (tol {ALIGN_SCALE_TOL}) center_med {:.2} m reg {} verify [{}]",
                    r.seed, r.cond, r.rig, r.scales, r.spread, r.center_med, r.registered, r.verify
                );
                rows.push(r);
            }
        }
    }
    if let Some(b) = rows.iter().find(|r| r.seed == 1 && r.cond == "base") {
        assert!(b.spread > ALIGN_SCALE_TOL, "{:?}", b.scales);
    }
}
