//! 기본 경로(인자 없는 synth → run → verify)의 구역별 연직 기울기 진단. 구역 정밀 모델의 사진별 회전을
//! 진단 출력(`SKYLENS_REGION_DIAG`)으로 받아 정답과 비교한다. 오래 걸려 기본으로는 돌리지 않는다.
//! 실행: `cargo test --release -p skylens-stream --test default_tilt -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use skylens_core::nalgebra::{Matrix3, Rotation3, Vector3};
use skylens_core::synth::{Scene, SceneConfig};

fn cli(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(args)
        .env("SKYLENS_REGION_DIAG", "1")
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

/// 정답 회전(세계→카메라), 사진 번호 gid = 고른 위치 * 3 + (F, R, L) 순서.
fn truth_rotations(input: &Path) -> HashMap<String, Matrix3<f64>> {
    let mut m = HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..16].iter().map(|s| s.parse().unwrap()).collect();
        m.insert(f[0].to_string(), Matrix3::from_row_slice(&n));
    }
    m
}

/// 기본 간격(stride 3)으로 고른 사진 목록의 번호 gid 에서 원본 위치 번호는 (gid / 3) * 3.
const STRIDE: usize = 3;

fn image_name(gid: usize) -> String {
    format!("cam{}_{:04}", ["F", "R", "L"][gid % 3], (gid / 3) * STRIDE)
}

/// 회전 행렬들의 평균 회전(극분해)과 단위 행렬 사이의 각(도).
fn mean_rotation_angle(ds: &[Matrix3<f64>]) -> f64 {
    let sum: Matrix3<f64> = ds.iter().sum();
    let svd = sum.svd(true, true);
    let mut r = svd.u.unwrap() * svd.v_t.unwrap();
    if r.determinant() < 0.0 {
        let mut u = svd.u.unwrap();
        u.column_mut(2).neg_mut();
        r = u * svd.v_t.unwrap();
    }
    Rotation3::from_matrix_unchecked(r).angle().to_degrees()
}

#[test]
#[ignore]
fn default_path_region_tilt() {
    let root = std::env::temp_dir().join(format!("skylens_tilt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    let (i, o) = (input.to_str().unwrap(), output.to_str().unwrap());
    // 시드는 환경 변수 SKYLENS_TILT_SEED (기본 1). 기본 합성 설정에서 시드만 바꾼다.
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let scene = Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    });
    scene.write_dataset(&input).unwrap();
    eprintln!("seed {seed}");
    let (code, stdout, stderr) = cli(&["run", i, o]);
    assert_eq!(code, 0, "{stderr}");
    for l in stdout.lines().filter(|l| l.starts_with("region ")) {
        eprintln!("{l}");
    }
    let (vcode, vout, _) = cli(&["verify", o]);
    eprintln!("verify exit {vcode}\n{vout}");
    let truth = truth_rotations(&input);
    // 구역마다 마지막으로 출력된 진단(정밀 모델 완성 직후)만 쓴다.
    let mut rots: HashMap<usize, HashMap<usize, (bool, Matrix3<f64>)>> = HashMap::new();
    let mut ctrs: HashMap<usize, HashMap<usize, Vector3<f64>>> = HashMap::new();
    for l in stderr.lines() {
        if l.starts_with("diag align") {
            eprintln!("{l}");
        }
        let Some(rest) = l.strip_prefix("diag rot region ") else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        let region: usize = f[0].parse().unwrap();
        let gid: usize = f[2].parse().unwrap();
        let own = f[4] == "1";
        let n: Vec<f64> = f[5..14].iter().map(|s| s.parse().unwrap()).collect();
        // 진단 출력은 nalgebra 행렬을 열 우선으로 내보낸다.
        if f.len() >= 17 {
            let c: Vec<f64> = f[14..17].iter().map(|s| s.parse().unwrap()).collect();
            ctrs.entry(region)
                .or_default()
                .insert(gid, Vector3::new(c[0], c[1], c[2]));
        }
        rots.entry(region)
            .or_default()
            .insert(gid, (own, Matrix3::from_column_slice(&n)));
    }
    let mut keys: Vec<_> = rots.keys().copied().collect();
    keys.sort();
    let up = Vector3::new(0.0, 0.0, 1.0);
    for k in keys {
        let r = &rots[&k];
        // 월드 회전 차 R_est^T R_true 의 평균 각, 그리고 연직(위) 방향 기울기:
        // 정답 위 방향 대비 추정 위 방향의 각 (카메라 좌표 위 = R * up 이 같도록 비교).
        let (mut diffs, mut tilts) = (Vec::new(), Vec::new());
        for (gid, (own, re)) in r {
            if !own {
                continue;
            }
            let rt = truth[&image_name(*gid)];
            diffs.push(re.transpose() * rt);
            let u_est = re.transpose() * (rt * up);
            tilts.push(u_est.angle(&up).to_degrees());
        }
        tilts.sort_by(f64::total_cmp);
        eprintln!(
            "tilt region {k} owned {} mean_rot_err_deg {:.3} up_tilt_median_deg {:.3}",
            diffs.len(),
            mean_rotation_angle(&diffs),
            tilts[tilts.len() / 2]
        );
    }
    // GPS 잡음과 척도 분리: 정답 중심(= -R^T t) 대비 (a) 잡음 GPS 의 거리 중앙값(완벽한 모델이 갖는 잔차),
    // (b) 추정 중심에 닮음 변환(Umeyama)을 맞춘 척도와 맞춘 뒤 잔차.
    let tc = |name: &str| -> Vector3<f64> {
        let mut t = Vector3::zeros();
        for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
            .unwrap()
            .lines()
        {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f[0] == name {
                let v: Vec<f64> = f[1..].iter().map(|s| s.parse().unwrap()).collect();
                t = Vector3::new(v[15], v[16], v[17]);
            }
        }
        -(truth[name].transpose() * t)
    };
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut rk: Vec<_> = ctrs.keys().copied().collect();
    rk.sort();
    for k in rk {
        let (mut e, mut t, mut noise) = (Vec::new(), Vec::new(), Vec::new());
        for (gid, c) in &ctrs[&k] {
            let name = image_name(*gid);
            let tcen = tc(&name);
            let vi = scene.views.iter().position(|v| v.name == name).unwrap();
            noise.push((scene.gps_enu[vi].coords - tcen).norm());
            e.push(*c);
            t.push(tcen);
        }
        let n = e.len() as f64;
        let me = e.iter().sum::<Vector3<f64>>() / n;
        let mt = t.iter().sum::<Vector3<f64>>() / n;
        let mut cov = Matrix3::zeros();
        let (mut ve, mut vt) = (0.0, 0.0);
        for (a, b) in e.iter().zip(&t) {
            cov += (b - mt) * (a - me).transpose() / n;
            ve += (a - me).norm_squared() / n;
            vt += (b - mt).norm_squared() / n;
        }
        let svd = cov.svd(true, true);
        let (u, vtm) = (svd.u.unwrap(), svd.v_t.unwrap());
        let mut d = Matrix3::identity();
        if (u * vtm).determinant() < 0.0 {
            d[(2, 2)] = -1.0;
        }
        let r = u * d * vtm;
        let scale = (svd
            .singular_values
            .component_mul(&Vector3::new(1.0, 1.0, d[(2, 2)])))
        .sum()
            / ve;
        let fit: Vec<f64> = e
            .iter()
            .zip(&t)
            .map(|(a, b)| (b - (scale * r * (a - me) + mt)).norm())
            .collect();
        let rms = (fit.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
        eprintln!(
            "scale region {k} n {} noise_only_gps_med {:.3} fit_scale {:.4} spread_ratio_est_over_truth {:.4} fit_resid_med {:.3} fit_resid_rms {:.3}",
            e.len(),
            med(noise),
            scale,
            (ve / vt).sqrt(),
            med(fit),
            rms
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}
