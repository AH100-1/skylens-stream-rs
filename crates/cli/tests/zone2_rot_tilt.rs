//! 짧은 마지막 구역(시드 4 구역 2)의 회전 평균 단계 연직 기울기 원인 가르기. 합성 → `run`
//! (SKYLENS_DIAG_PREVIEW=1, SKYLENS_DIAG_ROT=1) → 구역별 게이지 전 회전 R_i(세계→카메라)와 정답 R_true_i 로
//! S_i = R_true_iᵀ R_i 를 만든다(회전 평균이 맞으면 모든 i 에서 같은 세계 회전 S). 최종 포즈는 R_i gᵀ 이므로
//! 세계 좌표계 오차는 E = S gᵀ 이고 연직 기울기는 E·z 와 z 의 각이다.
//! - (a) 상대 회전 측정 편향: S_i 가 평균 S 에서 벗어난 각(전체·카메라별).
//! - (b) 간선 구조: `DIAGROT edges` 줄(카메라 쌍별 간선 수·잔차).
//! - (c) 게이지: Kabsch 직후 g 와 롤 선택 뒤 g 의 기울기, E 의 비행 축 둘레 몫.
//!
//! 실행: `SKYLENS_TILT_SEED=4 cargo test --release -p skylens-stream --test zone2_rot_tilt -- --ignored --nocapture`
//! 기본 동작은 바꾸지 않는다.

use std::collections::BTreeMap;
use std::process::Command;

use skylens_core::nalgebra::{Matrix3, Vector3};
use skylens_core::synth::{Scene, SceneConfig};

fn project(sum: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = sum.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    u * d * vt
}

fn angle_deg(r: &Matrix3<f64>) -> f64 {
    (((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0))
        .acos()
        .to_degrees()
}

fn tilt_deg(e: &Matrix3<f64>) -> f64 {
    (e * Vector3::z()).z.clamp(-1.0, 1.0).acos().to_degrees()
}

fn row9(f: &[&str]) -> Matrix3<f64> {
    let m: Vec<f64> = f.iter().map(|v| v.parse().unwrap()).collect();
    Matrix3::from_row_slice(&m)
}

#[test]
#[ignore]
fn zone2_rot_tilt() {
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let scene = Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    });
    let root = std::env::temp_dir().join(format!("skylens_z2rot_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    scene.write_dataset(&input).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_DIAG_PREVIEW", "1")
        .env("SKYLENS_DIAG_ROT", "1")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let _ = std::fs::remove_dir_all(&root);
    let truth: BTreeMap<usize, Matrix3<f64>> = scene
        .views
        .iter()
        .enumerate()
        .map(|(i, v)| (3 * v.position + i % 3, *v.camera.pose.rotation.matrix()))
        .collect();
    let mut kabsch: BTreeMap<usize, Matrix3<f64>> = BTreeMap::new();
    let mut fin: BTreeMap<usize, Matrix3<f64>> = BTreeMap::new();
    let mut raw: BTreeMap<usize, Vec<(usize, Matrix3<f64>)>> = BTreeMap::new();
    for l in String::from_utf8_lossy(&o.stderr).lines() {
        if l.starts_with("DIAGROT edges") {
            eprintln!("{l}");
        } else if let Some(r) = l.strip_prefix("DIAGROTG ") {
            let f: Vec<&str> = r.split_whitespace().collect();
            let region: usize = f[1].parse().unwrap();
            if f[0] == "kabsch" {
                kabsch.insert(region, row9(&f[2..11]));
            } else {
                fin.insert(region, row9(&f[2..11]));
                raw.insert(region, Vec::new());
            }
        } else if let Some(r) = l.strip_prefix("DIAGROTR ") {
            let f: Vec<&str> = r.split_whitespace().collect();
            let (region, gid): (usize, usize) = (f[0].parse().unwrap(), f[1].parse().unwrap());
            raw.get_mut(&region).unwrap().push((gid, row9(&f[2..11])));
        }
    }
    assert!(raw.len() >= 3, "구역 {} 개", raw.len());
    for (region, list) in &raw {
        let s_all: Vec<(usize, Matrix3<f64>)> = list
            .iter()
            .filter_map(|(g, r)| truth.get(g).map(|t| (*g % 3, t.transpose() * r)))
            .collect();
        let s_sum: Matrix3<f64> = s_all.iter().map(|x| x.1).sum();
        let s_mean = project(&s_sum);
        let (gk, gf) = (kabsch[region], fin[region]);
        // 카메라별 S 편향·퍼짐.
        let mut line = String::new();
        let mut spreads: Vec<f64> = s_all
            .iter()
            .map(|x| angle_deg(&(x.1 * s_mean.transpose())))
            .collect();
        spreads.sort_by(f64::total_cmp);
        for cam in 0..3 {
            let v: Vec<&Matrix3<f64>> = s_all.iter().filter(|x| x.0 == cam).map(|x| &x.1).collect();
            if v.is_empty() {
                continue;
            }
            let sum: Matrix3<f64> = v.iter().copied().sum();
            let m = project(&sum);
            line += &format!(
                " cam{cam} n {} bias_deg {:.3} bias_tilt_deg {:.3}",
                v.len(),
                angle_deg(&(m * s_mean.transpose())),
                tilt_deg(&(m * s_mean.transpose()))
            );
        }
        let e_f = s_mean * gf.transpose();
        let e_k = s_mean * gk.transpose();
        // 롤 선택이 더한 회전 gf gkᵀ 의 크기.
        let roll = gf * gk.transpose();
        eprintln!(
            "ZROT region {region} n {} spread_deg med {:.3} max {:.3} |{line} | tilt_deg kabsch {:.3} final {:.3} | roll_step_deg {:.3} | E_final_angle {:.3}",
            s_all.len(),
            spreads[spreads.len() / 2],
            spreads[spreads.len() - 1],
            tilt_deg(&e_k),
            tilt_deg(&e_f),
            angle_deg(&roll),
            angle_deg(&e_f)
        );
    }
}
