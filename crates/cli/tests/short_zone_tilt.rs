//! 짧은 마지막 구역의 연직 기울기가 생기는 단계를 가르는 진단. 합성 → `run`(SKYLENS_DIAG_PREVIEW=1) →
//! 단계별 포즈(회전 평균 직후 `rot`, 위치 평균 직후 `placed`, BA 직후 `ba`, GPS 정렬 직후 `gps`)를
//! 정답 회전과 비교해 구역별 연직 기울기(도)를 출력하고 `verify` 통과 수를 센다.
//! 실행: `SKYLENS_TILT_SEED=4 [SKYLENS_ALIGN_LINE_FIX=1] cargo test --release -p skylens-stream --test short_zone_tilt -- --ignored --nocapture`
//! 기본 동작은 바꾸지 않는다. 단계 기울기(`TILT`)는 단계 포즈 R_i(세계→카메라)와 정답 R_i 로 G_i = R_iᵀ·R_true_i 를 모아
//! 평균 회전 G 를 구한 뒤 G·z 와 z 의 각이다. 절댓값에는 카메라 축 규약 차가 섞여 구역 0·1 에서도 약 31° 가 나오므로
//! 구역 0 의 같은 단계와의 차로만 읽는다. 켬 판정은 `DIAGSIM own` 의 기울기를 쓴다.

use std::collections::BTreeMap;
use std::process::Command;

use skylens_core::nalgebra::{Matrix3, Rotation3, Vector3};
use skylens_core::synth::{Scene, SceneConfig};

/// 회전 행렬들의 평균 회전(SVD 투영)이 z 를 기울이는 각(도).
fn tilt_of(gs: &[Matrix3<f64>]) -> f64 {
    let sum: Matrix3<f64> = gs.iter().sum();
    let svd = sum.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    let g = u * d * vt;
    let gz: Vector3<f64> = g * Vector3::z();
    gz.z.clamp(-1.0, 1.0).acos().to_degrees()
}

#[test]
#[ignore]
fn short_zone_tilt() {
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let scene = Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    });
    let root = std::env::temp_dir().join(format!("skylens_tilt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    scene.write_dataset(&input).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_DIAG_PREVIEW", "1")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let truth: BTreeMap<usize, Matrix3<f64>> = scene
        .views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let cam = i % 3;
            let _ = v.cam;
            (3 * v.position + cam, *v.camera.pose.rotation.matrix())
        })
        .collect();
    // (구역, 단계) → G_i 목록. 같은 (구역, 단계)가 여러 번 나오면(재등록) 마지막 묶음만 쓴다.
    let mut seen: Vec<(String, usize, Vec<Matrix3<f64>>)> = Vec::new();
    let mut last_key: Option<(String, usize)> = None;
    for l in String::from_utf8_lossy(&o.stderr).lines() {
        if let Some(rest) = l.strip_prefix("DIAGSIM") {
            eprintln!("DIAGSIM{rest}");
            continue;
        }
        let Some(rest) = l.strip_prefix("DIAGPOSE ") else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        let (stage, region, gid) = (
            f[0].to_string(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
        );
        let m: Vec<f64> = f[3..12].iter().map(|v| v.parse().unwrap()).collect();
        let r = Matrix3::from_row_slice(&m);
        let Some(t) = truth.get(&gid) else { continue };
        let key = (stage.clone(), region);
        if last_key.as_ref() != Some(&key) || !seen.iter().any(|s| s.0 == stage && s.1 == region) {
            if last_key.as_ref() != Some(&key) {
                seen.retain(|s| !(s.0 == stage && s.1 == region));
                seen.push((stage, region, Vec::new()));
            }
            last_key = Some(key);
        }
        seen.last_mut().unwrap().2.push(r.transpose() * t);
    }
    let fix = std::env::var("SKYLENS_ALIGN_LINE_FIX").is_ok();
    eprintln!("TILT seed {seed} line_fix {fix}");
    let mut tilts: BTreeMap<(usize, String), f64> = BTreeMap::new();
    for (stage, region, gs) in &seen {
        let _ = Rotation3::<f64>::identity();
        tilts.insert((*region, stage.clone()), tilt_of(gs));
    }
    for ((region, stage), t) in &tilts {
        eprintln!("TILT region {region} stage {stage} tilt_deg {t:.3}");
    }
    let v = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&v.stdout);
    eprintln!("{text}");
    let _ = std::fs::remove_dir_all(&root);
    // 기준값: 초벌→자기 정밀 정렬 닮음 변환(DIAGSIM own)의 연직 기울기는 모든 구역에서 6° 안(정상 구간 0.9~3.8°, 켜지 않은 시드 4 구역 2 는 13.458°).
    if fix {
        let mut n = 0;
        for l in String::from_utf8_lossy(&o.stderr).lines() {
            if l.starts_with("DIAGSIM own") {
                let t: f64 = l
                    .split("tilt_deg ")
                    .nth(1)
                    .and_then(|x| x.split_whitespace().next())
                    .and_then(|x| x.parse().ok())
                    .unwrap();
                assert!(t < 6.0, "자기 정렬 기울기 {t:.3}° >= 6°: {l}");
                n += 1;
            }
        }
        assert!(n >= 3, "자기 정렬 기록 {n} 개");
    }
}
