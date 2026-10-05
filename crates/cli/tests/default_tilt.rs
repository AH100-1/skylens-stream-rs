//! 기본 경로(인자 없는 synth → run → verify)의 구역별 연직 기울기 진단. 구역 정밀 모델의 사진별 회전을
//! 진단 출력(`SKYLENS_REGION_DIAG`)으로 받아 정답과 비교한다. 오래 걸려 기본으로는 돌리지 않는다.
//! 실행: `cargo test --release -p skylens-stream --test default_tilt -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use skylens_core::nalgebra::{Matrix3, Rotation3, Vector3};

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

/// 정답 회전(세계→카메라), 사진 번호 gid = 위치 * 3 + (F, R, L) 순서.
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

fn image_name(gid: usize) -> String {
    format!("cam{}_{:04}", ["F", "R", "L"][gid % 3], gid / 3)
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
    assert_eq!(cli(&["synth", i]).0, 0);
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
    let _ = std::fs::remove_dir_all(&root);
}
