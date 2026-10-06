//! 초벌 모델의 구역별 축척이 어디서 정해지는지 합성 정답과 비교해 나눠 본다(측정만, 기본 동작 불변).
//!
//! `SKYLENS_ZONE_DUMP` 로 구역마다 (GPS, 위치 평균 맞춤 전 중심, GPS 맞춤 뒤 중심, 최종 초벌 중심)을 받고,
//! 정답 카메라 중심(정답 좌표 + 첫 GPS 이동)과 최적 닮음 변환(중심 Umeyama)으로 축척을 비교한다.
//! 정답 위치로 같은 강건 맞춤을 돌린 축척, 구역 간 스케일 검사(`verify::ALIGN_SCALE_TOL`)와의 관계도 함께 낸다.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::align::umeyama;
use skylens_core::geo::{enu_to_geodetic, geodetic_to_enu, Geodetic};
use skylens_core::nalgebra::{Matrix3, Vector3};
use skylens_core::stream::{split_regions, ALIGN_SCALE_TOL};
use skylens_core::verify::scale_spread;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_zscale_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // ZONESCALE_KEEP=1 이면 출력 폴더를 남겨 읽기 쪽만 다시 돌릴 수 있게 한다.
        if std::env::var("ZONESCALE_KEEP").is_ok() {
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
        m.insert(frame / 3 * 3 + cam, c + shift);
    }
    m
}

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
fn spread(p: &[Vector3<f64>]) -> ([f64; 3], f64) {
    let n = p.len() as f64;
    let c = p.iter().fold(Vector3::zeros(), |a, x| a + x) / n;
    let mut m = Matrix3::zeros();
    for x in p {
        let d = x - c;
        m += d * d.transpose();
    }
    let e = m.symmetric_eigen();
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| e.eigenvalues[b].total_cmp(&e.eigenvalues[a]));
    let ax = e.eigenvectors.column(idx[0]).into_owned();
    let proj: Vec<f64> = p.iter().map(|x| (x - c).dot(&ax)).collect();
    let ext = proj.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - proj.iter().cloned().fold(f64::INFINITY, f64::min);
    (idx.map(|i| (e.eigenvalues[i] / n).max(0.0).sqrt()), ext)
}

fn sim_scale(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> f64 {
    umeyama(src, dst).map_or(f64::NAN, |s| s.s)
}

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

/// 합성 장면을 만든다. `exact` 이면 gps.txt 를 정답 카메라 위치로 바꿔 쓴다(잡음 없는 GPS).
fn make_scene(dir: &Path, exact: bool) {
    let (code, so, se) = cli(&["synth", dir.to_str().unwrap(), "320", "180"], &[]);
    assert_eq!(code, 0, "{so}{se}");
    if !exact {
        return;
    }
    let o = std::fs::read_to_string(dir.join("truth/origin.txt")).unwrap();
    let origin = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let mut cen: BTreeMap<String, Vector3<f64>> = BTreeMap::new();
    for l in std::fs::read_to_string(dir.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, t) = (&n[..9], &n[9..12]);
        cen.insert(
            f[0].to_string(),
            Vector3::new(
                -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
                -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
                -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
            ),
        );
    }
    let mut out = String::new();
    for l in std::fs::read_to_string(dir.join("gps.txt"))
        .unwrap()
        .lines()
    {
        let name = l.split_whitespace().next().unwrap();
        let g = enu_to_geodetic(&cen[name], &origin);
        out.push_str(&format!(
            "{name} {:.9} {:.9} {:.3}\n",
            g.lat_deg, g.lon_deg, g.alt
        ));
    }
    std::fs::write(dir.join("gps.txt"), out).unwrap();
}

struct RunOut {
    own: BTreeMap<usize, f64>,
    verify_lines: Vec<String>,
}

fn run_one(base: &Path, exact: bool, reuse: bool) -> RunOut {
    let (scene_dir, out, dump) = (base.join("scene"), base.join("out"), base.join("dump"));
    let mut verify_lines = Vec::new();
    if !reuse {
        make_scene(&scene_dir, exact);
        let mut args = vec!["run", scene_dir.to_str().unwrap(), out.to_str().unwrap()];
        args.extend(ARGS.iter().take(12));
        let (code, so, se) = cli(&args, &[("SKYLENS_ZONE_DUMP", dump.to_str().unwrap())]);
        assert_eq!(code, 0, "{so}{se}");
        for l in so.lines().filter(|l| l.starts_with("issue")) {
            verify_lines.push(l.to_string());
        }
        let (_, vo, _) = cli(&["verify", out.to_str().unwrap()], &[]);
        for l in vo
            .lines()
            .filter(|l| l.contains("정렬") || l.contains("align"))
        {
            verify_lines.push(format!("verify: {l}"));
        }
    }
    let truth = truth_centers(&scene_dir);
    let n_pos = truth.keys().map(|&i| i / 3 + 1).max().unwrap();
    let regions = split_regions(n_pos, 12, 2);
    let align = read_align(&out.join("snapshots/manifest.json"));
    let tag = if exact { "exact" } else { "noisy" };
    let mut own = BTreeMap::new();
    for (zi, reg) in regions.iter().enumerate() {
        let path = dump.join(format!("zone_scale_{zi:02}.txt"));
        assert!(path.exists(), "구역 {zi} 진단 없음");
        let imgs = read_zone(&path);
        let ok: Vec<&Img> = imgs
            .iter()
            .filter(|i| i.placed.iter().all(|v| v.is_finite()) && truth.contains_key(&i.gid))
            .collect();
        let tr: Vec<Vector3<f64>> = ok.iter().map(|i| truth[&i.gid]).collect();
        let gps: Vec<Vector3<f64>> = ok.iter().map(|i| i.gps).collect();
        let placed: Vec<Vector3<f64>> = ok.iter().map(|i| i.placed).collect();
        let fin: Vec<Vector3<f64>> = ok.iter().map(|i| i.fin).collect();
        let n_own = ok.iter().filter(|i| reg.contains(i.gid / 3)).count();
        let mut by_pos: BTreeMap<usize, (Vector3<f64>, f64)> = BTreeMap::new();
        for i in &ok {
            let e = by_pos.entry(i.gid / 3).or_insert((Vector3::zeros(), 0.0));
            e.0 += truth[&i.gid];
            e.1 += 1.0;
        }
        let pts: Vec<Vector3<f64>> = by_pos.values().map(|(s, n)| s / *n).collect();
        let path_len: f64 = pts.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
        let (sd, ext) = spread(&tr);
        let (sd_p, ext_p) = spread(&placed);
        let (sd_f, ext_f) = spread(&fin);
        let noise: Vec<f64> = gps.iter().zip(&tr).map(|(g, t)| (g - t).norm()).collect();
        // 기준선 길이 비(위치 풀이 / 정답) 중앙값: 같은 위치의 카메라 3대 사이(장비 안)와 같은 카메라의 이웃 위치 사이.
        let base_ratio = |pick: &dyn Fn(&Img) -> Vector3<f64>| -> (f64, f64, f64, f64) {
            let (mut rig, mut rig_t, mut nb, mut nb_t) = (vec![], vec![], vec![], vec![]);
            for (x, ix) in ok.iter().enumerate() {
                for iy in ok.iter().skip(x + 1) {
                    let same_pos = ix.gid / 3 == iy.gid / 3;
                    let same_cam_next = ix.gid % 3 == iy.gid % 3 && ix.gid.abs_diff(iy.gid) == 3;
                    if !(same_pos || same_cam_next) {
                        continue;
                    }
                    let lt = (truth[&ix.gid] - truth[&iy.gid]).norm();
                    let lp = (pick(ix) - pick(iy)).norm();
                    if same_pos {
                        rig.push(lp / lt);
                        rig_t.push(lt);
                    } else {
                        nb.push(lp / lt);
                        nb_t.push(lt);
                    }
                }
            }
            (
                quant(&rig, 0.5),
                quant(&rig_t, 0.5),
                quant(&nb, 0.5),
                quant(&nb_t, 0.5),
            )
        };
        let (bg, bp, bf) = (
            base_ratio(&|i| i.gps),
            base_ratio(&|i| i.placed),
            base_ratio(&|i| i.fin),
        );
        eprintln!(
            "ZONESCALE {tag} zone {zi} | baseline ratio (median model/truth) rig-internal (truth {:.2} m): gps {:.3} placed {:.3} final {:.3} | same-camera neighbours (truth {:.2} m): gps {:.3} placed {:.3} final {:.3}",
            bp.1, bg.0, bp.0, bf.0, bp.3, bg.2, bp.2, bf.2
        );
        let a = align.get(&zi).map_or(f64::NAN, |a| a.2);
        if a.is_finite() {
            own.insert(zi, a);
        }
        let pl_err: Vec<f64> = {
            let s = umeyama(&placed, &tr);
            placed
                .iter()
                .zip(&tr)
                .map(|(p, t)| {
                    s.as_ref()
                        .map_or(f64::NAN, |s| (s.apply_point(p) - t).norm())
                })
                .collect()
        };
        eprintln!(
            "ZONESCALE {tag} zone {zi} | photos {} own {n_own} | truth path {path_len:.1} m axis1 range {ext:.1} m std {:.2}/{:.2}/{:.2} | gps noise med {:.2} p90 {:.2} m",
            ok.len(), sd[0], sd[1], sd[2], quant(&noise, 0.5), quant(&noise, 0.9)
        );
        eprintln!(
            "ZONESCALE {tag} zone {zi} | center umeyama scale vs truth: gps {:.4} placed {:.4} final {:.4} | final vs placed {:.4} | axis1 range ratio placed {:.4} final {:.4} | std2 ratio placed {:.3} final {:.3} | placed resid after fit med {:.2} m | own align (coarse->refined) {a:.4}",
            sim_scale(&gps, &tr), sim_scale(&placed, &tr), sim_scale(&fin, &tr), sim_scale(&fin, &placed),
            ext_p / ext, ext_f / ext, sd_p[1] / sd[1], sd_f[1] / sd[1], quant(&pl_err, 0.5)
        );
    }
    RunOut { own, verify_lines }
}

#[test]
#[ignore = "측정용: cargo test --release --test zone2_scale -- --ignored --nocapture"]
fn zone_scale_origin() {
    let t = TempDir::new("zs");
    let reuse = std::env::var("ZONESCALE_REUSE").ok().map(PathBuf::from);
    let base = reuse.clone().unwrap_or_else(|| t.0.clone());
    let (b1, b2) = (base.join("noisy"), base.join("exact"));
    let (h1, h2) = {
        let (b1, b2, r1, r2) = (b1.clone(), b2.clone(), reuse.is_some(), reuse.is_some());
        (
            std::thread::spawn(move || run_one(&b1, false, r1)),
            std::thread::spawn(move || run_one(&b2, true, r2)),
        )
    };
    let (noisy, exact) = (h1.join().unwrap(), h2.join().unwrap());
    for (tag, r) in [("noisy", &noisy), ("exact", &exact)] {
        let sc: Vec<f64> = r.own.values().copied().collect();
        let sp = scale_spread(&sc);
        eprintln!(
            "ZONESCALE {tag} check own scales {:?} spread {sp:.4} tol {ALIGN_SCALE_TOL} pass {}",
            r.own,
            sp <= ALIGN_SCALE_TOL
        );
        for l in &r.verify_lines {
            eprintln!("ZONESCALE {tag} {l}");
        }
    }
    // 기준값: 잡음 GPS 기본 경로에서 구역 2 정렬 축척은 구역 1 보다 10 % 이상 크다(알려진 18 %).
    let (s1, s2) = (noisy.own[&1], noisy.own[&2]);
    assert!(s2 / s1 - 1.0 > ALIGN_SCALE_TOL, "{s1} {s2}");
}
