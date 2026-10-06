//! 초벌 → 정밀 정렬에 같은 사진의 카메라 중심 쌍을 넣는 선택 옵션(`SKYLENS_ALIGN_CENTERS`, 기본 끔) 비교.
//!
//! 기본 경로(인자 없는 `synth` → `run`, 3구역)를 조건마다 돌려 구역 1·2 의 초벌 점 오차(정답 대비)·초벌 중심 오차·정렬 축척과
//! `verify` 의 초벌↔정밀 최근접·높이 차 중앙을 한 표로 낸다. 측정만 하고 어떤 기준도 기본값에 걸지 않는다(정답 복원 단언만 있다).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::align::Similarity;
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use skylens_core::stream::{align_region_centers, robust_fit, split_regions, CenterAlign};
use skylens_core::synth::{Scene, SceneConfig};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_pac_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if std::env::var("ALIGN_KEEP").is_ok() {
            eprintln!("ALIGN kept {}", self.0.display());
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

/// 정밀 트랙 관측마다 정답 카메라에서 표면으로 쏜 교점의 중앙값(성분별)과 가장 먼 교점까지 거리.
struct Row {
    coarse: Vector3<f64>,
    obs: Vec<(u32, [f64; 2])>,
    truth: Option<(Vector3<f64>, f64)>,
}

fn nums(s: &str) -> Vec<f64> {
    s.split_whitespace().map(|x| x.parse().unwrap()).collect()
}

fn read_rows(path: &Path) -> Vec<Row> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .skip(1)
        .map(|l| {
            let p: Vec<&str> = l.split('|').collect();
            let (c, o) = (nums(p[1]), nums(p[6]));
            Row {
                coarse: Vector3::new(c[0], c[1], c[2]),
                obs: o.chunks(4).map(|c| (c[0] as u32, [c[2], c[3]])).collect(),
                truth: None,
            }
        })
        .collect()
}

/// 적용된 정렬과 자기 구역 (사진 번호, 초벌 중심, 정밀 중심).
/// (사진 번호, 초벌 중심, 정밀 중심).
type CenterRow = (u32, Vector3<f64>, Vector3<f64>);

fn read_extra(path: &Path) -> (Similarity, Vec<CenterRow>) {
    let text = std::fs::read_to_string(path).unwrap();
    let (mut sim, mut cs) = (None, Vec::new());
    for l in text.lines() {
        let (tag, rest) = l.split_at(1);
        let n = nums(rest);
        if tag == "S" {
            sim = Some(Similarity {
                s: n[0],
                r: Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&n[1..10])),
                t: Vector3::new(n[10], n[11], n[12]),
            });
        } else {
            cs.push((
                n[0] as u32,
                Vector3::new(n[1], n[2], n[3]),
                Vector3::new(n[4], n[5], n[6]),
            ));
        }
    }
    (sim.expect("정렬 줄 없음"), cs)
}

fn stat(v: &[f64]) -> String {
    format!("{:.3}/{:.3}", quant(v, 0.5), quant(v, 0.9))
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

fn item(table: &str, name: &str) -> String {
    let line = table
        .lines()
        .find(|l| l.starts_with(&format!("| {name} |")))
        .unwrap_or_else(|| panic!("{name} 줄 없음:\n{table}"));
    line.split('|').map(str::trim).nth(3).unwrap().to_string()
}

fn pt(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}

/// 정답 닮음 복원: 점·중심 쌍이 정확하면 어떤 가중이든 닮음을 1e-9 안으로 복원한다.
#[test]
fn centers_alignment_recovers_truth_similarity() {
    let truth = Similarity {
        s: 1.18,
        r: Rotation3::from_euler_angles(0.2, -0.4, 0.9),
        t: pt(5.0, -3.0, 12.0),
    };
    let region = split_regions(36, 12, 2)[1];
    let src: Vec<Vector3<f64>> = (0..60)
        .map(|i| {
            let f = i as f64;
            pt(
                (f * 0.9).sin() * 20.0,
                (f * 1.7).cos() * 15.0,
                (f * 0.31).sin() * 4.0,
            )
        })
        .collect();
    let dst: Vec<_> = src.iter().map(|p| truth.apply_point(p)).collect();
    let pairs: Vec<_> = src.iter().copied().zip(dst.iter().copied()).collect();
    let cen: Vec<_> = pairs.iter().step_by(4).copied().collect();
    for mode in [
        CenterAlign::CentersOnly,
        CenterAlign::Weighted(1.0),
        CenterAlign::Weighted(5.0),
        CenterAlign::Weighted(20.0),
    ] {
        let (sim, _) = align_region_centers(&region, &pairs, &cen, mode);
        let sim = sim.unwrap_or_else(|| panic!("{mode:?} 퇴화"));
        for p in &src {
            assert!(
                (sim.apply_point(p) - truth.apply_point(p)).norm() < 1e-9,
                "{mode:?}"
            );
        }
    }
}

struct Cond {
    name: &'static str,
    env: Option<&'static str>,
}

#[test]
#[ignore = "측정용: cargo test --release -j 2 -p skylens-stream --test preview_align_centers -- --ignored --nocapture"]
fn preview_align_centers() {
    let t = TempDir::new("cmp");
    let base = std::env::var("ALIGN_REUSE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| t.0.clone());
    let input = base.join("in");
    if std::env::var("ALIGN_REUSE").is_err() {
        assert_eq!(cli(&["synth", input.to_str().unwrap()], &[]).0, 0);
    }
    let scene = Scene::new(SceneConfig::default());
    let cams = truth_cams(&input);
    let shift = truth_shift(&input);
    let conds = [
        Cond {
            name: "points only (current)",
            env: None,
        },
        Cond {
            name: "centers only",
            env: Some("only"),
        },
        Cond {
            name: "points + centers w1",
            env: Some("w1"),
        },
        Cond {
            name: "points + centers w5",
            env: Some("w5"),
        },
        Cond {
            name: "points + centers w20",
            env: Some("w20"),
        },
    ];
    for c in &conds {
        let tag = c.env.unwrap_or("off");
        let (out, dump) = (
            base.join(format!("out_{tag}")),
            base.join(format!("dump_{tag}")),
        );
        if std::env::var("ALIGN_REUSE").is_err() {
            let mut env = vec![("SKYLENS_PAIR_DUMP", dump.to_str().unwrap())];
            if let Some(e) = c.env {
                env.push(("SKYLENS_ALIGN_CENTERS", e));
            }
            let (code, so, se) = cli(
                &["run", input.to_str().unwrap(), out.to_str().unwrap()],
                &env,
            );
            assert_eq!(code, 0, "{so}{se}");
        }
        let (_, vout, _) = cli(&["verify", out.to_str().unwrap()], &[]);
        let pr = item(&vout, "preview_vs_refined");
        let (nn, hd) = (
            number_after(&pr, "최근접 중앙 최대 "),
            number_after(&pr, "높이 차 중앙 최대 "),
        );
        for region in [1usize, 2] {
            let mut rows = read_rows(&dump.join(format!("own_pairs_{region:02}.txt")));
            // 사진 번호 → 구역 안 쌍만 정답 광선 교점을 붙인다.
            for row in rows.iter_mut() {
                let hits: Vec<Vector3<f64>> = row
                    .obs
                    .iter()
                    .filter_map(|(img, p)| {
                        let (rot, cc, k) = cams.get(img)?;
                        let n = ((p[0] + 0.5 - k[2]) / k[0], (p[1] + 0.5 - k[3]) / k[1]);
                        let d = rot.inverse() * Vector3::new(n.0, n.1, 1.0);
                        let h = scene.intersect(&Point3::from(*cc), &d.normalize())?;
                        Some(h.point.coords + shift)
                    })
                    .collect();
                if hits.is_empty() {
                    continue;
                }
                let comp = |k: usize| quant(&hits.iter().map(|h| h[k]).collect::<Vec<_>>(), 0.5);
                let med = pt(comp(0), comp(1), comp(2));
                row.truth = Some((
                    med,
                    hits.iter().map(|h| (h - med).norm()).fold(0.0, f64::max),
                ));
            }
            let (sim, cs) = read_extra(&dump.join(format!("own_extra_{region:02}.txt")));
            let pe: Vec<f64> = rows
                .iter()
                .filter(|x| x.truth.is_some_and(|t| t.1 < MAX_TRUTH_SPREAD))
                .map(|x| (sim.apply_point(&x.coarse) - x.truth.unwrap().0).norm())
                .collect();
            let ce: Vec<f64> = cs
                .iter()
                .filter_map(|(g, c, _)| {
                    cams.get(g)
                        .map(|t| (sim.apply_point(c) - (t.1 + shift)).norm())
                })
                .collect();
            // 정렬 후 초벌 중심 ↔ 정밀 중심(정렬이 맞추는 대상) 거리.
            let cr: Vec<f64> = cs
                .iter()
                .map(|(_, c, q)| (sim.apply_point(c) - q).norm())
                .collect();
            let (a, b): (Vec<_>, Vec<_>) = cs
                .iter()
                .filter_map(|(g, c, _)| cams.get(g).map(|t| (*c, t.1 + shift)))
                .unzip();
            let best = robust_fit(&a, &b).map_or(f64::NAN, |f| {
                quant(
                    &a.iter()
                        .zip(&b)
                        .map(|(x, y)| (f.0.apply_point(x) - y).norm())
                        .collect::<Vec<_>>(),
                    0.5,
                )
            });
            eprintln!(
                "ALIGN | {:<24} | region {region} | point err m med/p90 {} (n {}) | coarse center err m med/p90 {} (best similarity med {best:.3}) | center gap to refined med {:.3} | scale {:.4} | verify nn med max {nn:.3} m, height diff med max {hd:.3} m",
                c.name, stat(&pe), pe.len(), stat(&ce), quant(&cr, 0.5), sim.s
            );
        }
    }
}
