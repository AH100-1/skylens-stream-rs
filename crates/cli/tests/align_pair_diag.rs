//! 자기 구역 정렬에 쓰인 점 쌍의 잔차를 합성 정답 3D 점과 대응시켜 나눠 본다.
//!
//! `--coarse-back off` 로 실행하면서 환경 변수 `SKYLENS_PAIR_DUMP` 로 구역별 점 쌍 진단
//! (`own_pairs_{구역:02}.txt`: 초벌·정밀 좌표, 잔차, 안쪽 여부, 관측 사진 수, 삼각측량 각, 정밀 트랙 관측)을 받는다.
//! 정답 3D 점은 정밀 트랙 관측의 광선을 정답 카메라에서 합성 장면 표면에 쏘아 얻는다(관측들 중앙값).
//! 초벌 쪽·정밀 쪽 점 오차는 각 좌표계에서 정답으로의 최적 유사변환(강건) 뒤 |변환(점) - 정답| 이다.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::align::{umeyama, Similarity};
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use skylens_core::stream::{robust_fit, split_regions};
use skylens_core::synth::{Scene, SceneConfig};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_pairdiag_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // PAIRDIAG_KEEP=1 이면 출력 폴더를 남겨 읽기 쪽만 다시 돌릴 수 있게 한다.
        if std::env::var("PAIRDIAG_KEEP").is_ok() {
            eprintln!("PAIRDIAG kept {}", self.0.display());
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

/// 분위수(빈 목록은 NaN). `q` 는 0..=1.
fn quant(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
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

/// 사진 번호(파이프라인 안 번호) → (회전 세계→카메라, 중심 정답 좌표).
fn truth_cams(input: &Path) -> BTreeMap<u32, Cam> {
    let mut m = BTreeMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        // 사진 번호 = 3 * 위치 + 카메라(F, R, L). 위치 = 프레임 번호 / 3 (기본 stride 3, 첫 프레임 0).
        let frame: u32 = f[0].rsplit('_').next().unwrap().parse().unwrap();
        let cam = ["camF", "camR", "camL"]
            .iter()
            .position(|c| f[0].starts_with(c))
            .unwrap() as u32;
        let id = frame / 3 * 3 + cam;
        let n: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, t) = (&n[..9], &n[9..12]);
        let c = Vector3::new(
            -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
            -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
            -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
        );
        m.insert(
            id,
            (
                Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(r)),
                c,
                [0, 1, 2, 3].map(|i| f[1 + i].parse().unwrap()),
            ),
        );
    }
    m
}

/// 정답 카메라: 회전(세계→카메라), 중심(정답 좌표), 내부 파라미터 fx fy cx cy.
type Cam = (Rotation3<f64>, Vector3<f64>, [f64; 4]);

/// 정답 점으로 쓰는 쌍의 정답 광선 퍼짐 상한(m). 이보다 퍼지면 오대응·표면 경계로 보고 점 오차에서 뺀다. 0.9~1.0 m 에 몰린 쌍이 많아 더 낮추면 쌍 집합이 한쪽으로 치우친다(민감도 줄 참조).
const MAX_TRUTH_SPREAD: f64 = 1.0;

struct Row {
    image: u32,
    coarse: Vector3<f64>,
    refined: Vector3<f64>,
    resid: f64,
    inlier: bool,
    photos_r: usize,
    angle_c: f64,
    angle_r: f64,
    obs: Vec<(u32, [f64; 2])>,
    /// 정답 점(출력 좌표)과 관측 광선 정답 점들의 퍼짐(m).
    truth: Option<(Vector3<f64>, f64)>,
}

struct Dump {
    scope: (usize, usize),
    fit_scale: f64,
    fit_median: f64,
    rows: Vec<Row>,
}

fn nums(s: &str) -> Vec<f64> {
    s.split_whitespace().map(|x| x.parse().unwrap()).collect()
}

fn read_dump(path: &Path) -> Dump {
    let text = std::fs::read_to_string(path).unwrap();
    let mut it = text.lines();
    let h: Vec<&str> = it.next().unwrap().split_whitespace().collect();
    // # region R scope LO HI fit S MED
    let scope = (h[4].parse().unwrap(), h[5].parse().unwrap());
    let (fit_scale, fit_median) = if h[7] == "none" {
        (f64::NAN, f64::NAN)
    } else {
        (h[7].parse().unwrap(), h[8].parse().unwrap())
    };
    let rows = it
        .map(|l| {
            let p: Vec<&str> = l.split('|').collect();
            let a = nums(p[0]);
            let (c, r) = (nums(p[1]), nums(p[2]));
            let rs = nums(p[3]);
            let ph = nums(p[4]);
            let an = nums(p[5]);
            let o = nums(p[6]);
            Row {
                image: a[0] as u32,
                coarse: Vector3::new(c[0], c[1], c[2]),
                refined: Vector3::new(r[0], r[1], r[2]),
                resid: rs[0],
                inlier: rs[1] > 0.5,
                photos_r: ph[1] as usize,
                angle_c: an[0],
                angle_r: an[1],
                obs: o.chunks(4).map(|c| (c[0] as u32, [c[2], c[3]])).collect(),
                truth: None,
            }
        })
        .collect();
    Dump {
        scope,
        fit_scale,
        fit_median,
        rows,
    }
}

/// 정밀 트랙 관측마다 정답 카메라에서 표면으로 쏜 교점의 중앙값(성분별)과 가장 먼 교점까지 거리.
fn attach_truth(rows: &mut [Row], scene: &Scene, cams: &BTreeMap<u32, Cam>, shift: &Vector3<f64>) {
    for r in rows {
        let hits: Vec<Vector3<f64>> = r
            .obs
            .iter()
            .filter_map(|(img, p)| {
                let (rot, c, k) = cams.get(img)?;
                // 특징점 번호 규약(화소 중심이 정수) → 정규 좌표.
                let n = ((p[0] + 0.5 - k[2]) / k[0], (p[1] + 0.5 - k[3]) / k[1]);
                let d = rot.inverse() * Vector3::new(n.0, n.1, 1.0);
                let h = scene.intersect(&Point3::from(*c), &d.normalize())?;
                Some(h.point.coords + shift)
            })
            .collect();
        if hits.is_empty() {
            continue;
        }
        let comp = |k: usize| quant(&hits.iter().map(|h| h[k]).collect::<Vec<_>>(), 0.5);
        let med = Vector3::new(comp(0), comp(1), comp(2));
        let spread = hits.iter().map(|h| (h - med).norm()).fold(0.0, f64::max);
        r.truth = Some((med, spread));
    }
}

fn principal_axis(p: &[Vector3<f64>]) -> (Vector3<f64>, Vector3<f64>) {
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
    (c, vt.row(best.1).transpose())
}

fn spread_ratio(p: &[Vector3<f64>]) -> f64 {
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
    v[1] / v[0]
}

/// 점 쌍별 오차: 초벌 점·정밀 점이 각 최적 유사변환 뒤 정답에서 벗어난 거리.
struct Errs {
    /// 사용 쌍 번호 → (초벌 오차, 정밀 오차, 정밀 좌표 그대로 오차).
    e: Vec<(usize, f64, f64, f64)>,
    sim_c: Similarity,
    sim_r: Similarity,
}

fn point_errors(rows: &[Row], max_spread: f64) -> Option<Errs> {
    let used: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].truth.is_some_and(|t| t.1 < max_spread))
        .collect();
    let t: Vec<Vector3<f64>> = used.iter().map(|&i| rows[i].truth.unwrap().0).collect();
    let c: Vec<Vector3<f64>> = used.iter().map(|&i| rows[i].coarse).collect();
    let r: Vec<Vector3<f64>> = used.iter().map(|&i| rows[i].refined).collect();
    let sim_c = robust_fit(&c, &t)?.0;
    let sim_r = robust_fit(&r, &t)?.0;
    let e = used
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            (
                i,
                (sim_c.apply_point(&c[k]) - t[k]).norm(),
                (sim_r.apply_point(&r[k]) - t[k]).norm(),
                (r[k] - t[k]).norm(),
            )
        })
        .collect();
    Some(Errs { e, sim_c, sim_r })
}

/// 구간별 표 한 줄. `idx`: 구간에 든 쌍 번호.
fn bin_line(tag: &str, region: usize, rows: &[Row], errs: &Errs, idx: &[usize]) {
    let by: BTreeMap<usize, (f64, f64, f64)> =
        errs.e.iter().map(|&(i, c, r, a)| (i, (c, r, a))).collect();
    let sel: Vec<usize> = idx.iter().copied().filter(|i| by.contains_key(i)).collect();
    let resid: Vec<f64> = idx.iter().map(|&i| rows[i].resid).collect();
    let inl = idx.iter().filter(|&&i| rows[i].inlier).count();
    let ce: Vec<f64> = sel.iter().map(|i| by[i].0).collect();
    let re: Vec<f64> = sel.iter().map(|i| by[i].1).collect();
    let worse_c = sel.iter().filter(|i| by[i].0 > by[i].1).count();
    eprintln!(
        "PAIRDIAG {region} | {tag} | n {} | resid med {:.3} p90 {:.3} max {:.3} | inlier {:.0}% | truth-n {} coarse-err med {:.3} p90 {:.3} | refined-err med {:.3} p90 {:.3} | coarse worse {:.0}%",
        idx.len(), quant(&resid, 0.5), quant(&resid, 0.9), quant(&resid, 1.0),
        100.0 * inl as f64 / idx.len().max(1) as f64, sel.len(),
        quant(&ce, 0.5), quant(&ce, 0.9), quant(&re, 0.5), quant(&re, 0.9),
        100.0 * worse_c as f64 / sel.len().max(1) as f64
    );
}

/// 구간별 정답 광선 퍼짐 분포와 개수.
fn split_by_plain(
    name: &str,
    region: usize,
    rows: &[Row],
    key: &dyn Fn(usize) -> f64,
    edges: &[f64],
) {
    for b in 0..=edges.len() {
        let lo = if b == 0 {
            f64::NEG_INFINITY
        } else {
            edges[b - 1]
        };
        let hi = if b == edges.len() {
            f64::INFINITY
        } else {
            edges[b]
        };
        let idx: Vec<usize> = (0..rows.len())
            .filter(|&i| key(i) >= lo && key(i) < hi)
            .collect();
        let sp: Vec<f64> = idx
            .iter()
            .filter_map(|&i| rows[i].truth.map(|t| t.1))
            .collect();
        let inl = idx.iter().filter(|&&i| rows[i].inlier).count();
        eprintln!(
            "PAIRDIAG {region} | {name} [{lo:.1},{hi:.1}) | n {} | inlier {:.0}% | spread med {:.3} p90 {:.3}",
            idx.len(), 100.0 * inl as f64 / idx.len().max(1) as f64, quant(&sp, 0.5), quant(&sp, 0.9)
        );
    }
}

/// 값 목록을 경계로 나눠 구간마다 표 한 줄.
fn split_by(
    name: &str,
    region: usize,
    rows: &[Row],
    errs: &Errs,
    key: &dyn Fn(usize) -> f64,
    edges: &[f64],
) {
    for b in 0..=edges.len() {
        let lo = if b == 0 {
            f64::NEG_INFINITY
        } else {
            edges[b - 1]
        };
        let hi = if b == edges.len() {
            f64::INFINITY
        } else {
            edges[b]
        };
        let idx: Vec<usize> = (0..rows.len())
            .filter(|&i| key(i) >= lo && key(i) < hi)
            .collect();
        bin_line(
            &format!("{name} [{lo:.1},{hi:.1})"),
            region,
            rows,
            errs,
            &idx,
        );
    }
}

#[test]
#[ignore = "측정용: cargo test --release --test align_pair_diag -- --ignored --nocapture"]
fn own_align_pair_diag() {
    let t = TempDir::new("diag");
    // PAIRDIAG_REUSE=<PAIRDIAG_KEEP 으로 남긴 폴더> 이면 실행을 건너뛰고 읽기만 한다.
    let reuse = std::env::var("PAIRDIAG_REUSE").ok().map(PathBuf::from);
    let base = reuse.clone().unwrap_or_else(|| t.0.clone());
    let (scene_dir, out) = (base.join("scene"), base.join("out"));
    let dump = base.join("dump");
    if reuse.is_none() {
        let (scene_s, out_s) = (scene_dir.to_str().unwrap(), out.to_str().unwrap());
        let (code, so, se) = cli(&["synth", scene_s, "320", "180"], &[]);
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
        let (code, so, se) = cli(&args, &[("SKYLENS_PAIR_DUMP", dump.to_str().unwrap())]);
        assert_eq!(code, 0, "{so}{se}");
        for l in so.lines().filter(|l| l.contains("pairs")) {
            eprintln!("PAIRDIAG log {l}");
        }
    }

    let cfg = SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    };
    let scene = Scene::new(cfg);
    let cams = truth_cams(&scene_dir);
    let shift = truth_shift(&scene_dir);
    let n_pos = cams.keys().map(|&i| i as usize / 3 + 1).max().unwrap();
    let regions = split_regions(n_pos, 12, 2);

    for region in [1usize, 2] {
        let path = dump.join(format!("own_pairs_{region:02}.txt"));
        assert!(path.exists(), "구역 {region} 점 쌍 진단 없음");
        let mut d = read_dump(&path);
        attach_truth(&mut d.rows, &scene, &cams, &shift);
        let r = regions[region];
        eprintln!(
            "PAIRDIAG {region} | scope {:?} (region lo {} hi {}) | pairs {} | own fit scale {:.4} median {:.4} m | inliers {}",
            d.scope, r.lo, r.hi, d.rows.len(), d.fit_scale, d.fit_median,
            d.rows.iter().filter(|x| x.inlier).count()
        );
        let sp: Vec<f64> = d.rows.iter().filter_map(|x| x.truth.map(|t| t.1)).collect();
        eprintln!(
            "PAIRDIAG {region} | truth ray spread over track obs: med {:.3} p90 {:.3} m, share >1 m {:.1}% (n {})",
            quant(&sp, 0.5), quant(&sp, 0.9),
            100.0 * sp.iter().filter(|&&s| s > 1.0).count() as f64 / sp.len().max(1) as f64, sp.len()
        );
        // 정답 광선 퍼짐(정답 점의 믿을 만함)과 사진 위치별 퍼짐.
        let tspread = |i: usize| d.rows[i].truth.map_or(f64::NAN, |t| t.1);
        split_by_plain(
            "truth ray spread m",
            region,
            &d.rows,
            &tspread,
            &[0.1, 0.3, 0.9, 1.0],
        );
        split_by_plain(
            "pair image position",
            region,
            &d.rows,
            &|i| (d.rows[i].image / 3) as f64,
            &[22.0, 24.0, 26.0],
        );
        let errs = point_errors(&d.rows, MAX_TRUTH_SPREAD).expect("점 오차 변환 실패");
        for thr in [0.3, 0.1] {
            // 민감도: 정답 점을 더 엄격히 고르면 초벌->정답 맞춤이 어떻게 변하는가.
            if let Some(e) = point_errors(&d.rows, thr) {
                let ce: Vec<f64> = e.e.iter().map(|x| x.1).collect();
                let re: Vec<f64> = e.e.iter().map(|x| x.2).collect();
                eprintln!(
                    "PAIRDIAG {region} | sensitivity truth spread < {thr}: n {} coarse->truth scale {:.4} refined->truth scale {:.4} coarse-err med {:.3} refined-err med {:.3}",
                    e.e.len(), e.sim_c.s, e.sim_r.s, quant(&ce, 0.5), quant(&re, 0.5)
                );
            }
        }
        // 기준값(실측 2026-10-06, 4코어 측정 기계 x 여유): 정밀 쪽 점 오차는 작고 초벌 쪽이 크며,
        // 정답으로 본 초벌/정밀 축척비가 자기 구역 정렬 축척과 맞는다.
        {
            let ce: Vec<f64> = errs.e.iter().map(|x| x.1).collect();
            let re: Vec<f64> = errs.e.iter().map(|x| x.2).collect();
            let worse = errs.e.iter().filter(|x| x.1 > x.2).count() as f64 / errs.e.len() as f64;
            let implied = errs.sim_c.s / errs.sim_r.s;
            let (pairs_lo, pairs_hi, c_lo) = if region == 1 {
                (3000, 4200, 1.0)
            } else {
                (1400, 1900, 0.3)
            };
            assert!(
                (pairs_lo..pairs_hi).contains(&d.rows.len()),
                "구역 {region} 점 쌍 수 {}",
                d.rows.len()
            );
            assert!(
                quant(&re, 0.5) < 0.4,
                "구역 {region} 정밀 점 오차 중앙 {}",
                quant(&re, 0.5)
            );
            assert!(
                quant(&ce, 0.5) > c_lo,
                "구역 {region} 초벌 점 오차 중앙 {}",
                quant(&ce, 0.5)
            );
            assert!(worse > 0.85, "구역 {region} 초벌이 더 큰 쌍 비율 {worse}");
            if region == 2 {
                assert!(
                    (implied - d.fit_scale).abs() < 0.03,
                    "구역 2 정답 기준 축척비 {implied} 대 정렬 축척 {}",
                    d.fit_scale
                );
            }
        }
        let all: Vec<usize> = (0..d.rows.len()).collect();
        bin_line("ALL", region, &d.rows, &errs, &all);
        let inl: Vec<usize> = all.iter().copied().filter(|&i| d.rows[i].inlier).collect();
        bin_line("inliers", region, &d.rows, &errs, &inl);
        let out_i: Vec<usize> = all.iter().copied().filter(|&i| !d.rows[i].inlier).collect();
        bin_line("outliers", region, &d.rows, &errs, &out_i);
        eprintln!(
            "PAIRDIAG {region} | coarse->truth scale {:.4} refined->truth scale {:.4} | implied coarse->refined scale (coarse/refined) {:.4} vs own fit {:.4} | refined-as-is err med {:.3}",
            errs.sim_c.s, errs.sim_r.s, errs.sim_c.s / errs.sim_r.s, d.fit_scale,
            quant(&errs.e.iter().map(|e| e.3).collect::<Vec<_>>(), 0.5)
        );

        // 구역 안 위치(주축 투영, 구역 사진 정답 중심 기준), 높이, 관측 사진 수, 삼각측량 각.
        let region_cams: Vec<Vector3<f64>> = cams
            .iter()
            .filter(|(&i, _)| (i as usize / 3) >= r.lo && (i as usize / 3) < r.hi)
            .map(|(_, c)| c.1 + shift)
            .collect();
        let (cen, axis) = principal_axis(&region_cams);
        let proj: Vec<f64> = region_cams.iter().map(|c| (c - cen).dot(&axis)).collect();
        let (plo, phi) = (quant(&proj, 0.0), quant(&proj, 1.0));
        let truth_of = |i: usize| d.rows[i].truth.map_or(Vector3::zeros(), |t| t.0);
        let along = |i: usize| (truth_of(i) - cen).dot(&axis);
        let edges: Vec<f64> = (1..5).map(|k| plo + (phi - plo) * k as f64 / 5.0).collect();
        split_by(
            "along-axis m (cam range 5 bins)",
            region,
            &d.rows,
            &errs,
            &along,
            &edges,
        );
        let zs: Vec<f64> = errs.e.iter().map(|e| truth_of(e.0).z).collect();
        let zed = [quant(&zs, 1.0 / 3.0), quant(&zs, 2.0 / 3.0)];
        split_by(
            "height z m (terciles)",
            region,
            &d.rows,
            &errs,
            &|i| truth_of(i).z,
            &zed,
        );
        // 쌍을 맺은 관측 카메라에서 정답 점까지 거리(m): 초벌 점 오차가 거리에 비례하는지.
        let range = |i: usize| {
            let c = cams[&d.rows[i].image].1 + shift;
            d.rows[i].truth.map_or(f64::NAN, |t| (t.0 - c).norm())
        };
        split_by(
            "range camera-to-point m",
            region,
            &d.rows,
            &errs,
            &range,
            &[20.0, 30.0, 40.0, 60.0],
        );
        split_by(
            "photos in refined track",
            region,
            &d.rows,
            &errs,
            &|i| d.rows[i].photos_r as f64,
            &[3.0, 4.0, 5.0, 8.0],
        );
        split_by(
            "refined tri angle deg",
            region,
            &d.rows,
            &errs,
            &|i| d.rows[i].angle_r,
            &[2.0, 5.0, 10.0, 20.0],
        );
        split_by(
            "coarse tri angle deg",
            region,
            &d.rows,
            &errs,
            &|i| d.rows[i].angle_c,
            &[2.0, 5.0, 10.0, 20.0],
        );

        // 점 쌍 분포: 정답 점의 퍼짐과 정렬 창(앞 구역과 겹치는 위치) 안 비율.
        let tp: Vec<Vector3<f64>> = errs.e.iter().map(|e| truth_of(e.0)).collect();
        let ap: Vec<f64> = tp.iter().map(|p| (p - cen).dot(&axis)).collect();
        let band = skylens_core::stream::align_window(&r, 2, n_pos);
        let in_band = d
            .rows
            .iter()
            .filter(|x| {
                let p = (x.image / 3) as usize;
                p >= band.0 && p < band.1
            })
            .count();
        eprintln!(
            "PAIRDIAG {region} | pair truth points sv2/sv1 {:.3} | along-axis pair range {:.1} m vs cam range {:.1} m | pairs in align band {} ({:.0}%)",
            spread_ratio(&tp), quant(&ap, 1.0) - quant(&ap, 0.0), phi - plo, in_band,
            100.0 * in_band as f64 / d.rows.len().max(1) as f64
        );
        for (tag, pick) in [("first half of axis", true), ("second half of axis", false)] {
            let mid = 0.5 * (plo + phi);
            let idx: Vec<usize> = errs
                .e
                .iter()
                .map(|e| e.0)
                .filter(|&i| (along(i) < mid) == pick)
                .collect();
            let s: Vec<_> = idx.iter().map(|&i| d.rows[i].coarse).collect();
            let q: Vec<_> = idx.iter().map(|&i| d.rows[i].refined).collect();
            match (umeyama(&s, &q), robust_fit(&s, &q)) {
                (Some(u), Some(f)) => {
                    // 한쪽 절반의 점 쌍만으로 구한 변환을 다른 절반(정밀 점)에 적용했을 때 잔차(외삽).
                    let other: Vec<usize> = errs
                        .e
                        .iter()
                        .map(|e| e.0)
                        .filter(|&i| (along(i) < mid) != pick)
                        .collect();
                    let ex: Vec<f64> = other
                        .iter()
                        .map(|&i| (f.0.apply_point(&d.rows[i].coarse) - d.rows[i].refined).norm())
                        .collect();
                    eprintln!(
                        "PAIRDIAG {region} | fit on {tag}: n {} umeyama scale {:.4} robust scale {:.4} med {:.3} | applied to other half: resid med {:.3} p90 {:.3}",
                        idx.len(), u.s, f.0.s, f.2, quant(&ex, 0.5), quant(&ex, 0.9)
                    );
                }
                _ => eprintln!(
                    "PAIRDIAG {region} | fit on {tag}: degenerate (n {})",
                    idx.len()
                ),
            }
        }
        // 오차 큰 점 쌍을 빼고 다시 맞췄을 때 축척: 각 조건별.
        type Pick = fn(&Row) -> bool;
        let picks: [(&str, Pick); 4] = [
            ("all pairs (plain umeyama)", |_| true),
            ("refined angle >= 5 deg", |x| x.angle_r >= 5.0),
            ("refined photos >= 4", |x| x.photos_r >= 4),
            ("robust inliers only", |x| x.inlier),
        ];
        for (tag, f) in picks {
            let s: Vec<_> = d.rows.iter().filter(|x| f(x)).map(|x| x.coarse).collect();
            let q: Vec<_> = d.rows.iter().filter(|x| f(x)).map(|x| x.refined).collect();
            if let Some(u) = umeyama(&s, &q) {
                let res: Vec<f64> = s
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| (u.apply_point(a) - b).norm())
                    .collect();
                eprintln!(
                    "PAIRDIAG {region} | refit {tag}: n {} scale {:.4} rot {:.2} deg resid med {:.3} p90 {:.3}",
                    s.len(), u.s, u.r.angle().to_degrees(), quant(&res, 0.5), quant(&res, 0.9)
                );
            }
        }
    }
}
