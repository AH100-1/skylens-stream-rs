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
            || l.starts_with("diag align roll")
            || l.starts_with("diag detached")
            || l.starts_with("diag align region")
        {
            eprintln!("{l}");
        }
    }
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();
    l_pose_errors(&input, &stderr);
    stage_errors(&input, &stderr);
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

/// 단계별(`diag stage`, 마지막 `diag rot`) 자세를 정답과 비교한다. 카메라 종류(정면·오른쪽 / 왼쪽)마다 정답 대비 회전 오차
/// 중앙값(`abs`), 모든 카메라에 공통인 좌표계 회전(척도 평균, `global`), 그것을 뺀 나머지(`resid`)를 낸다.
/// 공통 회전이 크면 모델 전체가 기운 것(롤·좌표계), 나머지가 크면 사진끼리 어긋난 것이다.
fn stage_errors(input: &std::path::Path, stderr: &str) {
    for r in stage_rows(input, stderr) {
        eprintln!(
            "serr region {} {} {} cams {} n {} abs {:.2} global {:.2} resid {:.2}",
            r.region, r.tag, r.phase, r.cams, r.n, r.abs, r.global, r.resid
        );
    }
}

struct StageRow {
    region: usize,
    tag: String,
    phase: String,
    cams: &'static str,
    n: usize,
    abs: f64,
    global: f64,
    resid: f64,
}

fn stage_rows(input: &std::path::Path, stderr: &str) -> Vec<StageRow> {
    let mut truth: HashMap<String, Matrix3<f64>> = HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let v: Vec<f64> = f[1..].iter().map(|s| s.parse().unwrap()).collect();
        truth.insert(f[0].to_string(), Matrix3::from_row_slice(&v[6..15]));
    }
    // (구역, 이름, 단계) 별 마지막 호출의 {gid: 회전}. 단계 이름 순서는 처음 나온 순서를 지킨다.
    type Key = (usize, String, String);
    let mut order: Vec<Key> = Vec::new();
    let mut stages: HashMap<Key, HashMap<usize, Matrix3<f64>>> = HashMap::new();
    let mut cur: Option<Key> = None;
    for l in stderr.lines() {
        let (key, vals): (Key, Vec<&str>) = if let Some(rest) = l.strip_prefix("diag stage region ")
        {
            let f: Vec<&str> = rest.split_whitespace().collect();
            (
                (f[0].parse().unwrap(), f[1].into(), f[2].into()),
                f[4..].to_vec(),
            )
        } else if let Some(rest) = l.strip_prefix("diag rot region ") {
            let f: Vec<&str> = rest.split_whitespace().collect();
            ((f[0].parse().unwrap(), "final".into(), "final".into()), {
                let mut v = vec![f[2]];
                v.extend(&f[5..]);
                v
            })
        } else {
            continue;
        };
        if cur.as_ref() != Some(&key) {
            if !order.contains(&key) {
                order.push(key.clone());
            }
            stages.insert(key.clone(), HashMap::new());
            cur = Some(key.clone());
        }
        let gid: usize = vals[0].parse().unwrap();
        let n: Vec<f64> = vals[1..10].iter().map(|s| s.parse().unwrap()).collect();
        stages
            .get_mut(&key)
            .unwrap()
            .insert(gid, Matrix3::from_column_slice(&n));
    }
    let mut out = Vec::new();
    for key in order {
        for (cams, left) in [("FR", false), ("L", true)] {
            let errs: Vec<Matrix3<f64>> = stages[&key]
                .iter()
                .filter(|(g, _)| (*g % 3 == 2) == left)
                .filter_map(|(g, re)| {
                    let name = format!("cam{}_{:04}", ["F", "R", "L"][g % 3], (g / 3) * 3);
                    truth.get(&name).map(|rt| re.transpose() * rt)
                })
                .collect();
            if errs.is_empty() {
                continue;
            }
            let angle = |m: Matrix3<f64>| Rotation3::from_matrix_unchecked(m).angle().to_degrees();
            let mean = errs.iter().sum::<Matrix3<f64>>();
            let svd = mean.svd(true, true);
            let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
            let d = (u * vt).determinant().signum();
            let bar = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * vt;
            let med = |mut v: Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            out.push(StageRow {
                region: key.0,
                tag: key.1.clone(),
                phase: key.2.clone(),
                cams,
                n: errs.len(),
                abs: med(errs.iter().map(|e| angle(*e)).collect()),
                global: angle(bar),
                resid: med(errs.iter().map(|e| angle(e * bar.transpose())).collect()),
            });
        }
    }
    out
}

/// 구역 정밀 모델의 회전 오차 한도(정답 대비, 구역·카메라 종류별 중앙값, 도). 시드 5 에서 구역 0 은 롤 선택이 틀려
/// 14.7° 였고 수평 퍼짐 롤을 정밀 시작점에도 쓰면서 2.4° 가 됐다. 한도는 구역 0 에 3°, 나머지 구역에 4.5°.
/// 실행: `SKYLENS_TILT_SEED=5 cargo test --release -p skylens-stream --test seed_verify -- --ignored zone_rotation`
#[test]
#[ignore]
fn zone_rotation_error_limit() {
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let root = std::env::temp_dir().join(format!("skylens_zone_rot_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_REGION_DIAG", "1")
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();
    let rows: Vec<StageRow> = stage_rows(&input, &stderr)
        .into_iter()
        .filter(|r| r.tag == "final")
        .collect();
    let _ = std::fs::remove_dir_all(&root);
    assert!(rows.len() >= 3, "구역 최종 자세 {} 개", rows.len());
    for r in &rows {
        let limit = if r.region == 0 { 3.0 } else { 4.5 };
        eprintln!(
            "zone {} {} rot med {:.2} (limit {limit})",
            r.region, r.cams, r.abs
        );
        assert!(
            r.abs <= limit,
            "seed {seed} region {} {}: {:.2} deg > {limit}",
            r.region,
            r.cams,
            r.abs
        );
    }
}
