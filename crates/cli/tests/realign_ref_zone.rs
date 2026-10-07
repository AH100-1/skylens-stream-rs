//! 재정렬 기준 구역 선택(`SKYLENS_REALIGN_REF=1`) 확인: 합성 장면(시드 4)을 기본 설정으로 `run` → `verify` 하고,
//! 정밀·초벌 점군의 정답 표면 대비 높이를 구역별로 잰다. 기준이 기울기가 가장 작은 구역이면 짧은 마지막
//! 구역(위치 22~27)의 기울어진 정밀 모델이 다른 구역을 기울이지 않으므로 verify 8/8 이고
//! 정밀 점군 높이 중앙값이 모든 구역에서 1 m 안이다(끔: 7/8, 높이 차 중앙 2.918 m).
//! 실행: `SKYLENS_GAP_SEED=4 cargo test --release -p skylens-stream --test realign_ref_zone -- --ignored --nocapture`
//! 시드당 약 10분.

use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::nalgebra::Point3;
use skylens_core::synth::{Scene, SceneConfig};

fn heights(dir: &Path, prefix: &str, scene: &Scene) -> Vec<(usize, Vec<f64>)> {
    let shift = scene.to_first_gps_frame(&Point3::new(0.0, 0.0, 0.0)).coords;
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if !n.starts_with(prefix) || !n.ends_with(".ply") {
            continue;
        }
        let Some(k) = n.split('_').nth(1).and_then(|s| s.parse::<usize>().ok()) else {
            continue;
        };
        let mut h: Vec<f64> = skylens_core::ply::read_ply_file(e.path())
            .unwrap()
            .points
            .iter()
            .map(|p| {
                let (x, y, z) = (
                    p.xyz[0] as f64 - shift.x,
                    p.xyz[1] as f64 - shift.y,
                    p.xyz[2] as f64 - shift.z,
                );
                z - scene.surface_height(x, y)
            })
            .filter(|v| v.is_finite())
            .collect();
        h.sort_by(f64::total_cmp);
        out.push((k, h));
    }
    out.sort_by_key(|x| x.0);
    out
}

fn q(h: &[f64], f: f64) -> f64 {
    h[((h.len() as f64 - 1.0) * f) as usize]
}

#[test]
#[ignore]
fn realign_ref_zone_keeps_regions_level() {
    let seed: u64 = std::env::var("SKYLENS_GAP_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let scene = Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    });
    let root: PathBuf =
        std::env::temp_dir().join(format!("skylens_refzone_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    scene.write_dataset(&input).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_REALIGN_REF", "1")
        .env("SKYLENS_DIAG_PREVIEW", "1")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    for l in String::from_utf8_lossy(&o.stderr).lines() {
        if l.starts_with("DIAGSIM") || l.starts_with("DIAGREF") {
            eprintln!("{l}");
        }
    }
    let v = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    let report = String::from_utf8_lossy(&v.stdout).into_owned();
    eprintln!("{report}");
    assert!(report.contains("결과: 8/8 통과"), "verify 8/8 이어야 한다");
    let pv_line = report
        .lines()
        .find(|l| l.contains("preview_vs_refined"))
        .unwrap();
    assert!(pv_line.contains("PASS"), "{pv_line}");

    let refined = heights(&output.join("refined"), "refined_", &scene);
    assert_eq!(refined.len(), 3, "정밀 구역 3개");
    for (k, h) in &refined {
        let (p05, p50, p95) = (q(h, 0.05), q(h, 0.5), q(h, 0.95));
        eprintln!("refined region {k} h p05 {p05:.2} p50 {p50:.2} p95 {p95:.2}");
        assert!(p50.abs() < 1.0, "구역 {k} 정밀 높이 중앙 {p50:.2} m");
        assert!(
            p05 > -3.5 && p95 < 3.5,
            "구역 {k} 5~95% {p05:.2}..{p95:.2} m"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}
