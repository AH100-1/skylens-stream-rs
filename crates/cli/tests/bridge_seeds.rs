//! 시드 1·2·3 의 세 구역 모두에서 카메라 묶음(F·R·L)별 정밀 포즈의 정답 대비 오차를 한 표로 내는 측정 시험.
//! 덩어리 잇기 옵션(`SKYLENS_ROT_BRIDGE=1`)의 끔·켬 비교용이며, 환경 변수는 자식 프로세스에 그대로 넘어간다.
//! `SKYLENS_ROT_BRIDGE=1 ZONE0_SEEDS=1 cargo test --release -j 2 -p skylens-stream --test bridge_seeds -- --ignored --nocapture`
//! 시드는 `ZONE0_SEEDS`(쉼표 구분, 기본 "1"). 출력 줄은 `TABLE` 로 시작한다.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use skylens_core::nalgebra::{Matrix3, Rotation3, Vector3};
use skylens_core::synth::{Scene, SceneConfig};

type Truth = HashMap<String, (Rotation3<f64>, Vector3<f64>)>;

fn truth_poses(input: &Path) -> Truth {
    let mut m = HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..19].iter().map(|s| s.parse().unwrap()).collect();
        let r = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&n[..9]));
        let t = Vector3::new(n[9], n[10], n[11]);
        m.insert(f[0].to_string(), (r, -(r.inverse() * t)));
    }
    m
}

struct Cam {
    cam: usize,
    r: Rotation3<f64>,
    c: Vector3<f64>,
    tr: Rotation3<f64>,
    tc: Vector3<f64>,
}

fn project(m: Matrix3<f64>) -> Rotation3<f64> {
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    Rotation3::from_matrix_unchecked(u * d * vt)
}

/// R_i ≈ Rt_i·Q 인 Q (구역 좌표계 → 정답 좌표계 회전의 역). 안 맞는 카메라를 두 번 덜어내며 다시 맞춘다.
fn fit_q(cams: &[&Cam], trim: bool) -> Rotation3<f64> {
    let mut keep: Vec<bool> = vec![true; cams.len()];
    let mut q = Rotation3::identity();
    for it in 0..if trim { 4 } else { 1 } {
        let mut s = Matrix3::zeros();
        for (c, _) in cams.iter().zip(&keep).filter(|(_, &k)| k) {
            s += c.tr.matrix().transpose() * c.r.matrix();
        }
        q = project(s);
        if it + 1 < 4 && trim {
            let mut e: Vec<(f64, usize)> = cams
                .iter()
                .enumerate()
                .map(|(i, c)| (ang(&(c.tr * q), &c.r), i))
                .collect();
            e.sort_by(|a, b| a.0.total_cmp(&b.0));
            keep = vec![false; cams.len()];
            for (_, i) in e.iter().take(cams.len().div_ceil(2)) {
                keep[*i] = true;
            }
        }
    }
    q
}

fn ang(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    (a.inverse() * b).angle().to_degrees()
}

fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    if s.is_empty() {
        f64::NAN
    } else {
        s[s.len() / 2]
    }
}
fn mx(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0, f64::max)
}

/// Umeyama 닮음(구역 → 정답)으로 중심 오차(정답 단위 m)를 구한다.
fn center_errors(cams: &[&Cam]) -> (f64, Vec<f64>) {
    let n = cams.len() as f64;
    let mu_a = cams.iter().map(|c| c.c).sum::<Vector3<f64>>() / n;
    let mu_b = cams.iter().map(|c| c.tc).sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    let mut va = 0.0;
    for c in cams {
        cov += (c.tc - mu_b) * (c.c - mu_a).transpose();
        va += (c.c - mu_a).norm_squared();
    }
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    let r = u * d * vt;
    let s = (svd
        .singular_values
        .component_mul(&Vector3::new(1.0, 1.0, d[(2, 2)])))
    .sum()
        / va;
    let e = cams
        .iter()
        .map(|c| (s * r * (c.c - mu_a) + mu_b - c.tc).norm())
        .collect();
    (s, e)
}

fn zone(seed: u64, base: &Path) {
    let (input, output) = (base.join("in"), base.join("out"));
    let dump = base.join("dump");
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let bridge = std::env::var("SKYLENS_ROT_BRIDGE").unwrap_or_default();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_DUMP_RPOSES", &dump)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let truth = truth_poses(&input);
    let mut zones: Vec<_> = std::fs::read_dir(&dump)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    zones.sort();
    for zp in zones {
        let mut cams = Vec::new();
        for l in std::fs::read_to_string(&zp).unwrap().lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            let g: usize = f[1].parse().unwrap();
            let n: Vec<f64> = f[2..14].iter().map(|s| s.parse().unwrap()).collect();
            let r = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&n[..9]));
            let t = Vector3::new(n[9], n[10], n[11]);
            let (tr, tc) = truth[f[0]];
            cams.push(Cam {
                cam: g % 3,
                r,
                c: -(r.inverse() * t),
                tr,
                tc,
            });
        }
        let all: Vec<&Cam> = cams.iter().collect();
        let zname = zp.file_stem().unwrap().to_string_lossy().into_owned();
        let q = fit_q(&all, true);
        let (_, ce) = center_errors(&all);
        for c in 0..3 {
            let re: Vec<f64> = cams
                .iter()
                .filter(|x| x.cam == c)
                .map(|x| ang(&(x.tr * q), &x.r))
                .collect();
            let pe: Vec<f64> = cams
                .iter()
                .zip(&ce)
                .filter(|(x, _)| x.cam == c)
                .map(|(_, e)| *e)
                .collect();
            eprintln!(
                "TABLE bridge={bridge:?} seed {seed} {zname} group {c} n {} rot_med {:.4} rot_max {:.4} pos_med {:.4}",
                re.len(),
                med(&re),
                mx(&re),
                med(&pe)
            );
        }
    }
}

#[test]
#[ignore = "기본 경로 run 을 시드마다 한 번씩 돌린다(수 분)"]
fn bridge_seeds_table() {
    let seeds: Vec<u64> = std::env::var("ZONE0_SEEDS")
        .unwrap_or_else(|_| "1".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    for s in seeds {
        let base = std::env::temp_dir().join(format!("skylens_bseed_{}_{s}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        zone(s, &base);
        let _ = std::fs::remove_dir_all(&base);
    }
}
