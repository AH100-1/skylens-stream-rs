//! 구역 1 초벌 모델의 점 오차를 초벌 단계별(회전 평균·위치 평균·삼각측량)로 가르고, 정렬 쌍을 카메라 거리로 한정한 정렬과 비교한다.
//!
//! `--coarse-back off` 로 실행하며 환경 변수 `SKYLENS_STAGE_DUMP`(단계별 포즈·트랙 관측)와 `SKYLENS_PAIR_DUMP`(정렬 점 쌍)를 받는다.
//! 정답은 합성 장면(`truth/cameras.txt`, 장면 표면 광선 교점). 측정만 하고 제품 기본 동작은 바꾸지 않는다.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::align::Similarity;
use skylens_core::camera::{Camera, Intrinsics, Pose};
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::nalgebra::{Matrix3, Point3, Rotation3, Vector2, Vector3};
use skylens_core::pipeline::stand_in::triangulate_robust;
use skylens_core::stream::{robust_fit, split_regions};
use skylens_core::synth::{Scene, SceneConfig};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_stages_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // STAGES_KEEP=1 이면 출력 폴더를 남겨 읽기 쪽만 다시 돌릴 수 있게 한다.
        if std::env::var("STAGES_KEEP").is_ok() {
            eprintln!("STAGES kept {}", self.0.display());
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

/// 단계 덤프: 포즈 두 벌(P: 회전·위치 평균 직후, F: 초벌 최종)과 트랙 관측.
struct Stage {
    thr_px: f64,
    min_deg: f64,
    k: Intrinsics,
    p: BTreeMap<u32, Pose>,
    f: BTreeMap<u32, Pose>,
    tracks: Vec<Vec<(u32, [f64; 2])>>,
}

fn read_stage(path: &Path) -> Stage {
    let text = std::fs::read_to_string(path).unwrap();
    let mut st = Stage {
        thr_px: 0.0,
        min_deg: 0.0,
        k: Intrinsics::from_hfov(320, 180, 65f64.to_radians()),
        p: BTreeMap::new(),
        f: BTreeMap::new(),
        tracks: Vec::new(),
    };
    for l in text.lines() {
        let mut it = l.split_whitespace();
        match it.next() {
            Some("H") => {
                let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                st.thr_px = v[0];
                st.min_deg = v[1];
                st.k.fx = v[2];
                st.k.fy = v[3];
                st.k.cx = v[4];
                st.k.cy = v[5];
            }
            Some(t @ ("P" | "F")) => {
                let g: u32 = it.next().unwrap().parse().unwrap();
                let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                let r = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&v[..9]));
                let pose = Pose::from_center(r, &Point3::new(v[9], v[10], v[11]));
                if t == "P" { &mut st.p } else { &mut st.f }.insert(g, pose);
            }
            Some("T") => {
                let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                st.tracks
                    .push(v.chunks(4).map(|c| (c[0] as u32, [c[2], c[3]])).collect());
            }
            _ => {}
        }
    }
    st
}

/// 추정 포즈 집합의 좌표계를 정답 좌표계로 돌린다: 회전은 Σ Rᵢᵀ Tᵢ 의 극분해(전역 회전만), 중심은 그 회전 뒤 축척·이동만 최소제곱.
/// 좌표계 어긋남(전역 회전)을 회전 오차로 세지 않기 위한 것이다. 반환: 맞춘 포즈와 축척.
fn gauge_to_truth(
    ps: &BTreeMap<u32, Pose>,
    cams: &BTreeMap<u32, Cam>,
    shift: &Vector3<f64>,
) -> (BTreeMap<u32, Pose>, f64) {
    let mut m = Matrix3::zeros();
    for (g, p) in ps {
        if let Some(t) = truth_pose(cams, shift, *g) {
            m += p.rotation.matrix().transpose() * t.rotation.matrix();
        }
    }
    let sv = m.svd(true, true);
    let q = Rotation3::from_matrix_unchecked(sv.u.unwrap() * sv.v_t.unwrap());
    let ids: Vec<u32> = ps
        .keys()
        .copied()
        .filter(|g| cams.contains_key(g))
        .collect();
    let a: Vec<Vector3<f64>> = ids
        .iter()
        .map(|g| q.inverse() * ps[g].center().coords)
        .collect();
    let b: Vec<Vector3<f64>> = ids
        .iter()
        .map(|g| truth_pose(cams, shift, *g).unwrap().center().coords)
        .collect();
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().sum::<Vector3<f64>>() / n,
        b.iter().sum::<Vector3<f64>>() / n,
    );
    let num: f64 = a.iter().zip(&b).map(|(x, y)| (x - ma).dot(&(y - mb))).sum();
    let den: f64 = a.iter().map(|x| (x - ma).norm_squared()).sum();
    let s = num / den;
    let out = ps
        .iter()
        .map(|(g, p)| {
            let c = s * (q.inverse() * p.center().coords - ma) + mb;
            (*g, Pose::from_center(p.rotation * q, &Point3::from(c)))
        })
        .collect();
    (out, s)
}

fn stat(v: &[f64]) -> String {
    format!("{:.3}/{:.3}", quant(v, 0.5), quant(v, 0.9))
}

/// 정답 포즈(출력 좌표) 목록.
fn truth_pose(cams: &BTreeMap<u32, Cam>, shift: &Vector3<f64>, g: u32) -> Option<Pose> {
    let (r, c, _) = cams.get(&g)?;
    Some(Pose::from_center(*r, &Point3::from(*c + shift)))
}

fn truth_k(base: &Intrinsics, cams: &BTreeMap<u32, Cam>, g: u32) -> Intrinsics {
    let mut k = *base;
    let c = cams[&g].2;
    (k.fx, k.fy, k.cx, k.cy) = (c[0], c[1], c[2], c[3]);
    k
}

#[derive(Clone, Copy, PartialEq)]
enum RotSrc {
    Est,
    Truth,
}

/// 한 변형: 어느 포즈 집합의 회전·중심과 내부 파라미터를 쓰는가.
struct Variant {
    name: &'static str,
    rot: RotSrc,
    cen: RotSrc,
    /// 포즈 집합 선택(true: 단계 P, false: 최종 F).
    placed: bool,
    k_truth: bool,
}

/// 변형 하나로 모든 트랙을 삼각측량한다. 번호 순서는 트랙 순서, 실패는 None.
fn triangulate_variant(
    st: &Stage,
    cams: &BTreeMap<u32, Cam>,
    shift: &Vector3<f64>,
    v: &Variant,
) -> Vec<Option<Vector3<f64>>> {
    let poses = if v.placed { &st.p } else { &st.f };
    let pose_of = |g: u32| -> Option<Pose> {
        let e = poses.get(&g)?;
        let t = truth_pose(cams, shift, g)?;
        let r = if v.rot == RotSrc::Est {
            e.rotation
        } else {
            t.rotation
        };
        let c = if v.cen == RotSrc::Est {
            e.center()
        } else {
            t.center()
        };
        Some(Pose::from_center(r, &c))
    };
    st.tracks
        .iter()
        .map(|t| {
            let cs: Vec<(Camera, Vector2<f64>)> = t
                .iter()
                .filter_map(|&(g, p)| {
                    let k = if v.k_truth {
                        truth_k(&st.k, cams, g)
                    } else {
                        st.k
                    };
                    Some((
                        Camera {
                            intrinsics: k,
                            pose: pose_of(g)?,
                        },
                        Vector2::new(p[0] + 0.5, p[1] + 0.5),
                    ))
                })
                .collect();
            if cs.len() != t.len() || cs.len() < 2 {
                return None;
            }
            triangulate_robust(&cs, st.thr_px, st.min_deg).map(|r| r.0)
        })
        .collect()
}

fn run_pipeline(base: &Path) {
    let (scene_s, out_s) = (
        base.join("scene").to_str().unwrap().to_string(),
        base.join("out").to_str().unwrap().to_string(),
    );
    let (code, so, se) = cli(&["synth", &scene_s, "320", "180"], &[]);
    assert_eq!(code, 0, "{so}{se}");
    let args = [
        "run",
        &scene_s,
        &out_s,
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
    let (code, so, se) = cli(
        &args,
        &[
            ("SKYLENS_PAIR_DUMP", base.join("dump").to_str().unwrap()),
            ("SKYLENS_STAGE_DUMP", base.join("stage").to_str().unwrap()),
        ],
    );
    assert_eq!(code, 0, "{so}{se}");
}

fn max_c(v: &[f64]) -> f64 {
    quant(v, 1.0)
}

#[test]
#[ignore = "측정용: cargo test --release --test preview_error_stages -- --ignored --nocapture"]
fn preview_error_stages() {
    let t = TempDir::new("stages");
    let reuse = std::env::var("STAGES_REUSE").ok().map(PathBuf::from);
    let base = reuse.clone().unwrap_or_else(|| t.0.clone());
    if reuse.is_none() {
        run_pipeline(&base);
    }
    let (scene_dir, dump, stage) = (base.join("scene"), base.join("dump"), base.join("stage"));
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    let cams = truth_cams(&scene_dir);
    let shift = truth_shift(&scene_dir);
    let n_pos = cams.keys().map(|&i| i as usize / 3 + 1).max().unwrap();
    let regions = split_regions(n_pos, 12, 2);

    for region in [1usize, 2] {
        let mut st = read_stage(&stage.join(format!("stage_{region:02}.txt")));
        let raw_f = st.f.clone();
        let (gp, sp) = gauge_to_truth(&st.p, &cams, &shift);
        let (gf, sf) = gauge_to_truth(&st.f, &cams, &shift);
        st.p = gp;
        st.f = gf;
        eprintln!("STAGES {region} | gauge (rotation by polar fit, then scale/shift only): scale P {sp:.4} F {sf:.4}");
        let r = regions[region];
        let own = |g: u32| (g as usize / 3) >= r.lo && (g as usize / 3) < r.hi;
        eprintln!(
            "STAGES {region} | tracks {} | thr_px {:.2} min_deg {:.2} | poses P {} F {}",
            st.tracks.len(),
            st.thr_px,
            st.min_deg,
            st.p.len(),
            st.f.len()
        );
        // 1) 포즈 단계별 오차(자기 구역 사진).
        for (tag, ps) in [
            ("회전 평균+위치 평균 직후(P)", &st.p),
            ("초벌 최종(F)", &st.f),
        ] {
            let (mut rot, mut cen, mut raw) = (Vec::new(), Vec::new(), Vec::new());
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for (&g, p) in ps.iter().filter(|(g, _)| own(**g)) {
                let Some(tp) = truth_pose(&cams, &shift, g) else {
                    continue;
                };
                rot.push((p.rotation * tp.rotation.inverse()).angle().to_degrees());
                raw.push((p.center() - tp.center()).norm());
                a.push(p.center().coords);
                b.push(tp.center().coords);
            }
            let fit = robust_fit(&a, &b);
            if let Some(f) = &fit {
                for (x, y) in a.iter().zip(&b) {
                    cen.push((f.0.apply_point(x) - y).norm());
                }
            }
            eprintln!(
                "STAGES {region} | pose {tag} | n {} | rot err deg med/p90 {} max {:.2} | center err m (gauge-fit) {} | center err m (after robust similarity) {} scale {:.4}",
                rot.len(), stat(&rot), max_c(&rot), stat(&raw), stat(&cen),
                fit.as_ref().map_or(f64::NAN, |f| f.0.s)
            );
        }
        // 2) 삼각측량 변형.
        let variants = [
            Variant {
                name: "pipeline final (est rot, est center, est K)",
                rot: RotSrc::Est,
                cen: RotSrc::Est,
                placed: false,
                k_truth: false,
            },
            Variant {
                name: "(a) truth rot + est center",
                rot: RotSrc::Truth,
                cen: RotSrc::Est,
                placed: false,
                k_truth: false,
            },
            Variant {
                name: "(b) est rot + truth center",
                rot: RotSrc::Est,
                cen: RotSrc::Truth,
                placed: false,
                k_truth: false,
            },
            Variant {
                name: "(c) est poses + truth K",
                rot: RotSrc::Est,
                cen: RotSrc::Est,
                placed: false,
                k_truth: true,
            },
            Variant {
                name: "(d) truth poses + est K",
                rot: RotSrc::Truth,
                cen: RotSrc::Truth,
                placed: false,
                k_truth: false,
            },
            Variant {
                name: "(e) truth poses + truth K",
                rot: RotSrc::Truth,
                cen: RotSrc::Truth,
                placed: false,
                k_truth: true,
            },
        ];
        let pts: Vec<Vec<Option<Vector3<f64>>>> = variants
            .iter()
            .map(|v| triangulate_variant(&st, &cams, &shift, v))
            .collect();
        // 정답 점: 트랙 관측 광선 교점의 중앙값(광선 퍼짐 < 1 m).
        let mut rows: Vec<Row> = st
            .tracks
            .iter()
            .map(|t| Row {
                image: t[0].0,
                coarse: Vector3::zeros(),
                refined: Vector3::zeros(),
                resid: 0.0,
                inlier: true,
                photos_r: 0,
                angle_c: 0.0,
                angle_r: 0.0,
                obs: t.clone(),
                truth: None,
            })
            .collect();
        attach_truth(&mut rows, &scene, &cams, &shift);
        // 모든 변형이 풀고 정답 점이 있는 자기 구역 트랙만 쓴다.
        let common: Vec<usize> = (0..rows.len())
            .filter(|&i| {
                own(rows[i].image)
                    && rows[i].truth.is_some_and(|t| t.1 < MAX_TRUTH_SPREAD)
                    && pts.iter().all(|p| p[i].is_some())
            })
            .collect();
        eprintln!("STAGES {region} | common tracks {}", common.len());
        let mut meds = Vec::new();
        for (v, p) in variants.iter().zip(&pts) {
            let x: Vec<Vector3<f64>> = common.iter().map(|&i| p[i].unwrap()).collect();
            let tr: Vec<Vector3<f64>> = common.iter().map(|&i| rows[i].truth.unwrap().0).collect();
            let raw: Vec<f64> = x.iter().zip(&tr).map(|(a, b)| (a - b).norm()).collect();
            let (fit, e): (Option<Similarity>, Vec<f64>) = match robust_fit(&x, &tr) {
                Some(f) => {
                    let e = x
                        .iter()
                        .zip(&tr)
                        .map(|(a, b)| (f.0.apply_point(a) - b).norm())
                        .collect();
                    (Some(f.0), e)
                }
                None => (None, vec![]),
            };
            meds.push(quant(&e, 0.5));
            eprintln!(
                "STAGES {region} | tri {} | point err m after similarity med/p90 {} | raw {} | scale {:.4}",
                v.name, stat(&e), stat(&raw), fit.map_or(f64::NAN, |f| f.s)
            );
        }
        // 기준값(실측 2026-10-06, 4코어 측정 기계 x 여유): 정답 포즈면 점 오차 바닥이 작고, 구역 1 에서는 정답 중심만 줘도
        // 점 오차가 크게 줄지만 정답 회전만 주면 거의 안 준다(위치 오차가 지배).
        if region == 1 {
            assert!(meds[4] < 0.3, "정답 포즈 점 오차 바닥 {}", meds[4]);
            assert!(
                meds[2] < 0.5 * meds[0],
                "정답 중심 {} 대 초벌 최종 {}",
                meds[2],
                meds[0]
            );
            assert!(
                meds[1] > 0.7 * meds[0],
                "정답 회전 {} 대 초벌 최종 {}",
                meds[1],
                meds[0]
            );
        }
        // 위치 평균 직후 포즈(P)로 삼각측량한 점 오차(위치 다듬기 전 단계 비교).
        {
            let v = Variant {
                name: "P poses (before refine/snap)",
                rot: RotSrc::Est,
                cen: RotSrc::Est,
                placed: true,
                k_truth: false,
            };
            let p = triangulate_variant(&st, &cams, &shift, &v);
            let sel: Vec<usize> = common.iter().copied().filter(|&i| p[i].is_some()).collect();
            let x: Vec<Vector3<f64>> = sel.iter().map(|&i| p[i].unwrap()).collect();
            let tr: Vec<Vector3<f64>> = sel.iter().map(|&i| rows[i].truth.unwrap().0).collect();
            if let Some(f) = robust_fit(&x, &tr) {
                let e: Vec<f64> = x
                    .iter()
                    .zip(&tr)
                    .map(|(a, b)| (f.0.apply_point(a) - b).norm())
                    .collect();
                eprintln!(
                    "STAGES {region} | tri {} | n {} | point err m med/p90 {} scale {:.4}",
                    v.name,
                    sel.len(),
                    stat(&e),
                    f.0.s
                );
            }
        }

        // 3) 정렬 쌍 거리 한정.
        let d = read_dump(&dump.join(format!("own_pairs_{region:02}.txt")));
        let mut rows = d.rows;
        attach_truth(&mut rows, &scene, &cams, &shift);
        let est_dist: Vec<f64> = rows
            .iter()
            .map(|r| {
                raw_f
                    .get(&r.image)
                    .map_or(f64::NAN, |p| (r.coarse - p.center().coords).norm())
            })
            .collect();
        let true_dist: Vec<f64> = rows
            .iter()
            .map(|r| match (r.truth, cams.get(&r.image)) {
                (Some(t), Some(c)) => (t.0 - (c.1 + shift)).norm(),
                _ => f64::NAN,
            })
            .collect();
        let finite: Vec<f64> = est_dist.iter().copied().filter(|x| x.is_finite()).collect();
        eprintln!(
            "STAGES {region} | pair est camera distance (coarse pt - coarse cam) quantiles 10/50/90 {:.1}/{:.1}/{:.1} m, n {}",
            quant(&finite, 0.1), quant(&finite, 0.5), quant(&finite, 0.9), finite.len()
        );
        let scale_in_coarse = 1.0; // 거리 문턱은 초벌 좌표 기준 m(구역 2 는 초벌 축척 1.18 포함).
        let med_d = quant(&finite, 0.5);
        let p30 = quant(&finite, 0.3);
        let sel_sets: Vec<(String, Vec<usize>)> = {
            let all: Vec<usize> = (0..rows.len()).collect();
            let le = |dv: &[f64], th: f64| -> Vec<usize> {
                all.iter().copied().filter(|&i| dv[i] <= th).collect()
            };
            let tmed = quant(
                &true_dist
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .collect::<Vec<_>>(),
                0.5,
            );
            vec![
                ("all (current)".into(), all.clone()),
                (
                    format!("est dist lower 50% (<= {:.1} m)", med_d * scale_in_coarse),
                    le(&est_dist, med_d),
                ),
                (
                    format!("est dist lower 30% (<= {p30:.1} m)"),
                    le(&est_dist, p30),
                ),
                ("est dist <= 30 m".into(), le(&est_dist, 30.0)),
                ("est dist <= 40 m".into(), le(&est_dist, 40.0)),
                (
                    format!("truth dist lower 50% (<= {tmed:.1} m)"),
                    le(&true_dist, tmed),
                ),
            ]
        };
        // 평가 쌍: 정답 점이 있는 모든 쌍(선택과 무관, 같은 집합).
        let eval: Vec<usize> = (0..rows.len())
            .filter(|&i| rows[i].truth.is_some_and(|t| t.1 < MAX_TRUTH_SPREAD))
            .collect();
        let own_g: Vec<u32> = raw_f
            .keys()
            .copied()
            .filter(|&g| own(g) && cams.contains_key(&g))
            .collect();
        let cc: Vec<Vector3<f64>> = own_g.iter().map(|g| raw_f[g].center().coords).collect();
        let tc: Vec<Vector3<f64>> = own_g
            .iter()
            .map(|g| truth_pose(&cams, &shift, *g).unwrap().center().coords)
            .collect();
        if let Some(f) = robust_fit(&cc, &tc) {
            eprintln!("STAGES {region} | reference: best similarity of coarse centers to truth scale {:.4}", f.0.s);
        }
        for (name, sel) in &sel_sets {
            let src: Vec<Vector3<f64>> = sel.iter().map(|&i| rows[i].coarse).collect();
            let dst: Vec<Vector3<f64>> = sel.iter().map(|&i| rows[i].refined).collect();
            let Some((sim, inl, med)) = robust_fit(&src, &dst) else {
                eprintln!("STAGES {region} | align {name} | degenerate");
                continue;
            };
            let ce: Vec<f64> = own_g
                .iter()
                .zip(&cc)
                .zip(&tc)
                .map(|((_, c), t)| (sim.apply_point(c) - t).norm())
                .collect();
            let pe: Vec<f64> = eval
                .iter()
                .map(|&i| (sim.apply_point(&rows[i].coarse) - rows[i].truth.unwrap().0).norm())
                .collect();
            eprintln!(
                "STAGES {region} | align {name} | pairs {} inlier {} | scale {:.4} | resid med {:.3} | center err m med/p90 {} | point err m med/p90 {} (n {})",
                sel.len(), inl.iter().filter(|&&x| x).count(), sim.s, med, stat(&ce), stat(&pe), eval.len()
            );
        }
    }
}
