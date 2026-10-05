//! 시드별 기본 경로(인자 없는 synth → run → verify) 판정표 출력. 오래 걸려 기본으로는 돌리지 않는다.
//! 실행: `SKYLENS_TILT_SEED=4 cargo test --release -p skylens-stream --test seed_verify -- --ignored --nocapture`

use std::collections::HashMap;
use std::process::Command;

use skylens_core::nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use skylens_core::synth::{Scene, SceneConfig};

#[test]
#[ignore]
fn default_path_seed_verify() {
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let root = std::env::temp_dir().join(format!("skylens_seed_verify_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let exe = env!("CARGO_BIN_EXE_skylens-stream");
    let run = Command::new(exe)
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_REGION_DIAG", "1")
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0));
    for l in String::from_utf8_lossy(&run.stderr).lines() {
        if l.starts_with("diag gps_align")
            || l.starts_with("diag detached")
            || l.starts_with("diag align region")
        {
            eprintln!("{l}");
        }
    }
    l_pose_errors(&input, &String::from_utf8_lossy(&run.stderr));
    let v = Command::new(exe)
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    eprintln!("seed {seed}");
    eprintln!("{}", String::from_utf8_lossy(&v.stdout));
    eprintln!("verify exit {:?}", v.status.code());
    let _ = std::fs::remove_dir_all(&root);
}

/// 구역 정밀 모델 진단(`diag rot region`, 구역마다 마지막 출력)의 자세를 정답과 비교해 구역·카메라 종류별
/// 회전 오차(도)와 위치 오차(m)의 중앙값·최대를 낸다. 위치는 정답을 첫 GPS 기준 좌표로 옮겨 비교한다.
fn l_pose_errors(input: &std::path::Path, stderr: &str) {
    let mut truth: HashMap<String, (Matrix3<f64>, Vector3<f64>)> = HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let v: Vec<f64> = f[1..].iter().map(|s| s.parse().unwrap()).collect();
        truth.insert(
            f[0].to_string(),
            (
                Matrix3::from_row_slice(&v[6..15]),
                Vector3::new(v[15], v[16], v[17]),
            ),
        );
    }
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let scene = Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    });
    let mut last: HashMap<(usize, usize), (Matrix3<f64>, Vector3<f64>)> = HashMap::new();
    for l in stderr.lines() {
        let Some(rest) = l.strip_prefix("diag rot region ") else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        if f[4] != "1" || f.len() < 17 {
            continue;
        }
        let n: Vec<f64> = f[5..14].iter().map(|s| s.parse().unwrap()).collect();
        let c: Vec<f64> = f[14..17].iter().map(|s| s.parse().unwrap()).collect();
        last.insert(
            (f[0].parse().unwrap(), f[2].parse().unwrap()),
            (
                Matrix3::from_column_slice(&n),
                Vector3::new(c[0], c[1], c[2]),
            ),
        );
    }
    let mut groups: HashMap<(usize, char), (Vec<f64>, Vec<f64>)> = HashMap::new();
    for ((region, gid), (re, c)) in &last {
        let name = format!("cam{}_{:04}", ["F", "R", "L"][gid % 3], (gid / 3) * 3);
        let Some((rt, t)) = truth.get(&name) else {
            continue;
        };
        let tc = scene.to_first_gps_frame(&Point3::from(-(rt.transpose() * t)));
        let ang = Rotation3::from_matrix_unchecked(re.transpose() * rt)
            .angle()
            .to_degrees();
        let e = groups
            .entry((*region, if gid % 3 == 2 { 'L' } else { 'M' }))
            .or_default();
        e.0.push(ang);
        e.1.push((c - tc.coords).norm());
    }
    let mut keys: Vec<_> = groups.keys().copied().collect();
    keys.sort();
    for k in keys {
        let (mut a, mut p) = groups[&k].clone();
        a.sort_by(f64::total_cmp);
        p.sort_by(f64::total_cmp);
        eprintln!(
            "lerr region {} cams {} n {} rot_deg med {:.2} max {:.2} pos_m med {:.2} max {:.2}",
            k.0,
            if k.1 == 'L' { "L" } else { "FR" },
            a.len(),
            a[a.len() / 2],
            a[a.len() - 1],
            p[p.len() / 2],
            p[p.len() - 1]
        );
    }
}
