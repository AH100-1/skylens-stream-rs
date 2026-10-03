//! 끝까지 흐름(synth → run → verify)의 정답 대비 정확도: 구역 1·2개 × 시드 2개.
//! poses.txt·정밀/초벌 PLY 를 합성 정답 카메라·표면과 비교해 표로 출력하고 상한을 단언한다.
//! 회전 오차: 출력(poses.txt)에 회전이 없어 비교하지 못한다(중심만 기록됨).

use std::path::Path;

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::Point3;
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::{verify_dir, Report};

fn quant(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        return f64::NAN;
    }
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// 점군 하나의 (정답 표면까지 수직 거리 절댓값, 부호 있는 높이 차) 목록. 좌표는 첫 GPS 원점 기준.
fn surface_dev(scene: &Scene, path: &Path) -> (Vec<f64>, Vec<f64>) {
    let origin = scene.to_first_gps_frame(&Point3::new(0.0, 0.0, 0.0)).coords;
    let cloud = read_ply_file(path).unwrap();
    let signed: Vec<f64> = cloud
        .points
        .iter()
        .map(|p| {
            let (x, y, z) = (
                p.xyz[0] as f64 - origin.x,
                p.xyz[1] as f64 - origin.y,
                p.xyz[2] as f64 - origin.z,
            );
            z - scene.surface_height(x, y)
        })
        .collect();
    (signed.iter().map(|v| v.abs()).collect(), signed)
}

/// "접두 123.456" 형태의 측정 문자열에서 접두 뒤 첫 숫자.
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

struct Acc {
    images: usize,
    regions: usize,
    registered: usize,
    c_med: f64,
    c_p95: f64,
    c_max: f64,
    s_med: f64,
    s_p95: f64,
    height_diff: f64,
    report: Report,
}

fn run_case(positions: usize, seed: u64) -> Acc {
    let tag = format!("{}_{positions}_{seed}", std::process::id());
    let root = std::env::temp_dir().join(format!("skylens_acc_{tag}"));
    let _ = std::fs::remove_dir_all(&root);
    let (input, out) = (root.join("in"), root.join("out"));
    let scene = Scene::new(SceneConfig {
        positions,
        width: 320,
        height: 180,
        seed,
        ..SceneConfig::default()
    });
    scene.write_dataset(&input).unwrap();
    let ds = load_dataset(
        &input,
        DatasetConfig {
            stride: 1,
            span: 48,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    let cfg = PipelineConfig {
        max_features: 800,
        dense_width: 96,
        hfov_deg: 65.0,
        ba_iters: 15,
    };
    let res = run_pipeline(&ds, &cfg, &out).unwrap();
    let report = verify_dir(&out);
    eprintln!("[{tag}]\n{}", report.to_table());
    for i in &res.issues {
        eprintln!("[{tag}] issue {i}");
    }

    // poses.txt 의 중심 → 정답 중심.
    let poses = std::fs::read_to_string(out.join("poses.txt")).unwrap();
    let mut errs = Vec::new();
    let mut per_region_err: Vec<Vec<f64>> = vec![Vec::new(); res.regions.len().max(1)];
    for line in poses.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let c: Vec<f64> = f[1..4].iter().map(|s| s.parse().unwrap()).collect();
        let v = scene.views.iter().find(|v| v.name == f[0]).unwrap();
        let truth = scene.to_first_gps_frame(&v.camera.pose.center());
        let e = (Point3::new(c[0], c[1], c[2]) - truth).norm();
        errs.push(e);
        // 위치 번호 → 구역: 구역 경계는 report 위치 수 누적으로 근사.
        let pos: usize = f[0][f[0].len() - 4..].parse().unwrap();
        let mut acc = 0usize;
        let mut k = 0;
        for (ri, r) in res.regions.iter().enumerate() {
            k = ri;
            acc += r.positions.saturating_sub(2);
            if pos < acc {
                break;
            }
        }
        let last = per_region_err.len() - 1;
        per_region_err[k.min(last)].push(e);
    }
    // 구역별 초벌·정밀 점군의 정답 표면 편차(부호 있는 높이 중앙, 절댓값 중앙).
    #[allow(clippy::needless_range_loop)]
    for k in 0..res.regions.len() {
        for kind in ["preview", "refined"] {
            let dir = out.join(kind);
            let name = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .find(|n| n.starts_with(&format!("{kind}_{k:02}_")) && n.ends_with(".ply"));
            if let Some(n) = name {
                let (abs, signed) = surface_dev(&scene, &dir.join(n));
                eprintln!(
                    "REGION [{tag}] region {k} {kind}: points {} surface |dz| median {:.3} signed median {:+.3} (center err median of region cams {:.3})",
                    abs.len(),
                    quant(&abs, 0.5),
                    quant(&signed, 0.5),
                    quant(&per_region_err[k], 0.5)
                );
            }
        }
    }
    let (abs, _) = surface_dev(&scene, &out.join("snapshots/step_final_all_refined.ply"));
    assert!(abs.len() > 1000, "점 수 {}", abs.len());
    let hd = report
        .item("preview_vs_refined")
        .map(|i| number_after(&i.measured, "높이 차 중앙 최대 "))
        .unwrap_or(f64::NAN);
    let acc = Acc {
        images: ds.image_count(),
        regions: res.regions.len(),
        registered: errs.len(),
        c_med: quant(&errs, 0.5),
        c_p95: quant(&errs, 0.95),
        c_max: quant(&errs, 1.0),
        s_med: quant(&abs, 0.5),
        s_p95: quant(&abs, 0.95),
        height_diff: hd,
        report,
    };
    print_row(&tag, &acc);
    let _ = std::fs::remove_dir_all(&root);
    acc
}

fn print_row(tag: &str, a: &Acc) {
    eprintln!(
        "ACC [{tag}] regions {} registered {}/{} center med {:.3} p95 {:.3} max {:.3} m | rotation n/a | surface med {:.3} p95 {:.3} m | height diff {:.3} m | points-in-final-cloud see REGION",
        a.regions, a.registered, a.images, a.c_med, a.c_p95, a.c_max, a.s_med, a.s_p95, a.height_diff
    );
    for i in &a.report.items {
        eprintln!(
            "ACC [{tag}]   {} {} {}",
            i.name,
            if i.pass { "PASS" } else { "FAIL" },
            i.measured
        );
    }
}

fn passes(a: &Acc) -> Vec<&str> {
    a.report
        .items
        .iter()
        .filter(|i| i.pass)
        .map(|i| i.name)
        .collect()
}

/// 상한은 첫 측정값에 여유를 둔 느슨한 값이다(노트 참고).
#[allow(clippy::too_many_arguments)]
fn bounds(
    a: &Acc,
    regions: usize,
    c_med: f64,
    c_p95: f64,
    c_max: f64,
    s_med: f64,
    s_p95: f64,
    hd: f64,
) {
    assert_eq!(a.regions, regions);
    assert_eq!(a.registered, a.images, "등록 수");
    assert!(a.c_med < c_med, "중심 중앙 {}", a.c_med);
    assert!(a.c_p95 < c_p95, "중심 95% {}", a.c_p95);
    assert!(a.c_max < c_max, "중심 최대 {}", a.c_max);
    assert!(a.s_med < s_med, "표면 중앙 {}", a.s_med);
    assert!(a.s_p95 < s_p95, "표면 95% {}", a.s_p95);
    assert!(a.height_diff < hd, "높이 차 {}", a.height_diff);
    for n in [
        "registered",
        "region_images",
        "refined_reprojection",
        "snapshots",
    ] {
        assert!(passes(a).contains(&n), "{n} 통과해야 함");
    }
}

#[test]
fn accuracy_one_region_seed1() {
    let a = run_case(40, 1);
    bounds(&a, 1, 2.5, 7.0, 15.0, 9.5, 16.0, 6.5);
}

#[test]
fn accuracy_one_region_seed2() {
    let a = run_case(40, 2);
    bounds(&a, 1, 2.5, 7.0, 15.0, 9.5, 16.0, 6.5);
}

#[test]
fn accuracy_two_regions_seed1() {
    let a = run_case(80, 1);
    bounds(&a, 2, 2.5, 7.0, 15.0, 11.0, 28.0, 16.0);
}

#[test]
fn accuracy_two_regions_seed2() {
    let a = run_case(80, 2);
    bounds(&a, 2, 2.5, 7.0, 15.0, 11.0, 28.0, 16.0);
}
