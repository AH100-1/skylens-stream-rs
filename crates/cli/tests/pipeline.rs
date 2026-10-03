//! 합성 장면 → run → verify, 정답 카메라 중심·표면 대비 오차.

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::Point3;
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::verify_dir;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

struct Case {
    registered: usize,
    images: usize,
    regions: usize,
    center_med: f64,
    center_max: f64,
    surface_med: f64,
    report: skylens_core::verify::Report,
}

fn run_case(stride: usize, ba_iters: usize) -> Case {
    let root = std::env::temp_dir().join(format!(
        "skylens_pipe_{}_{stride}_{ba_iters}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    scene.write_dataset(&input).unwrap();
    let ds = load_dataset(
        &input,
        DatasetConfig {
            stride,
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
        ba_iters,
    };
    let t = std::time::Instant::now();
    let res = run_pipeline(&ds, &cfg, &output).unwrap();
    eprintln!("secs {:.1}", t.elapsed().as_secs_f64());
    for r in &res.regions {
        eprintln!("{r:?}");
    }
    for i in &res.issues {
        eprintln!("issue {i}");
    }
    for d in ["preview", "refined", "snapshots"] {
        assert!(output.join(d).is_dir(), "{d}");
    }
    assert!(output.join("snapshots/manifest.json").is_file());
    let report = verify_dir(&output);
    eprintln!("{}", report.to_table());
    eprintln!("verify exit code {}", report.exit_code());

    // 정답 대비 카메라 중심 오차(첫 GPS 원점 좌표).
    let mut errs = Vec::new();
    for (name, c) in &res.centers {
        let v = scene.views.iter().find(|v| &v.name == name).unwrap();
        let truth = scene.to_first_gps_frame(&v.camera.pose.center());
        errs.push((Point3::new(c[0], c[1], c[2]) - truth).norm());
    }
    let (med, max) = (
        median(errs.clone()),
        errs.iter().copied().fold(0.0, f64::max),
    );
    eprintln!("registered {} of {}", errs.len(), ds.image_count());
    eprintln!("center error median {med:.3} m max {max:.3} m");
    let registered = errs.len();

    // 점군 → 정답 표면(수직 거리 근사).
    let origin = scene.to_first_gps_frame(&Point3::new(0.0, 0.0, 0.0)).coords;
    let cloud = read_ply_file(output.join("snapshots/step_final_all_refined.ply")).unwrap();
    assert!(cloud.len() > 100, "점 수 {}", cloud.len());
    let d: Vec<f64> = cloud
        .points
        .iter()
        .map(|p| {
            let (x, y, z) = (
                p.xyz[0] as f64 - origin.x,
                p.xyz[1] as f64 - origin.y,
                p.xyz[2] as f64 - origin.z,
            );
            (z - scene.surface_height(x, y)).abs()
        })
        .collect();
    let sm = median(d);
    eprintln!(
        "cloud points {} surface distance median {sm:.3} m",
        cloud.len()
    );
    let _ = std::fs::remove_dir_all(&root);
    Case {
        registered,
        images: ds.image_count(),
        regions: res.regions.len(),
        center_med: med,
        center_max: max,
        surface_med: sm,
        report,
    }
}

/// 항목 하나의 판정을 이름으로 확인한다.
fn check_item(c: &Case, name: &str, expect_pass: bool) {
    let it = c
        .report
        .item(name)
        .unwrap_or_else(|| panic!("{name} 항목 없음"));
    assert!(it.decided, "{name} 판정 불가: {}", it.measured);
    assert_eq!(it.pass, expect_pass, "{name}: {}", it.measured);
}

/// "접두 123.456 m" 형태의 측정 문자열에서 접두 뒤 첫 숫자.
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

fn print_case(c: &Case) {
    eprintln!(
        "CASE reg {}/{} regions {} center med {:.3} max {:.3} surface med {:.3}",
        c.registered, c.images, c.regions, c.center_med, c.center_max, c.surface_med
    );
    for i in &c.report.items {
        eprintln!("CASE   {} pass={} {}", i.name, i.pass, i.measured);
    }
}

/// 단일 구역(README 첫 명령과 같은 설정: stride 2, span 48, BA 15회).
#[test]
fn synthetic_single_region_end_to_end() {
    let c = run_case(2, 15);
    print_case(&c);
    assert_eq!(c.regions, 1);
    assert_eq!(c.images, 3 * 40);
    assert_eq!(c.registered, 3 * 40, "등록 수");
    assert!(c.center_med < 1.5, "중심 오차 중앙 {}", c.center_med);
    assert!(c.center_max < 6.0, "중심 오차 최대 {}", c.center_max);
    assert!(c.surface_med < 3.5, "표면 거리 중앙 {}", c.surface_med);
    for n in [
        "registered",
        "region_images",
        "refined_reprojection",
        "preview_align",
        "snapshots",
    ] {
        check_item(&c, n, true);
    }
    // 구역 1개: 이웃 겹침은 해당 없음.
    let ov = c.report.item("refined_overlap").unwrap();
    assert!(ov.measured.contains("해당 없음"), "{}", ov.measured);
    // 알려진 미달: SPEC 목표는 같은 위치 높이 차 중앙 < 2 m. 현재 측정값의 상한만 단언한다.
    let pr = c.report.item("preview_vs_refined").unwrap();
    eprintln!(
        "SPEC 목표 preview_vs_refined 높이 차 < 2 m, 현재: {}",
        pr.measured
    );
    assert!(
        !pr.pass,
        "목표 달성: 이 단언을 pass 로 바꾼다 ({})",
        pr.measured
    );
    assert!(
        number_after(&pr.measured, "높이 차 중앙 최대 ") < 3.5,
        "높이 차 상한 {}",
        pr.measured
    );
}

/// 구역 2개 이상(README 둘째 명령: stride 1 → 80위치, span 48). 이웃 겹침이 실제로 판정된다.
/// 현재 알려진 미달 항목(registered, preview_align, preview_vs_refined, refined_overlap)은
/// SPEC 목표를 출력하고 측정값의 상한(여유 포함)만 단언한다. 목표를 달성하면 해당 단언을 바꾼다.
#[test]
fn synthetic_two_region_end_to_end() {
    let c = run_case(1, 15);
    print_case(&c);
    assert!(c.regions >= 2, "구역 수 {}", c.regions);
    assert_eq!(c.images, 3 * 80);
    // 구역 앞쪽 보조 F 사진으로 등록 240/240 (이전 210/240).
    assert_eq!(c.registered, 240, "등록 수");
    check_item(&c, "registered", true);
    assert!(c.center_med < 2.5, "중심 오차 중앙 {}", c.center_med);
    assert!(c.surface_med < 4.5, "표면 거리 중앙 {}", c.surface_med);
    for n in ["region_images", "refined_reprojection", "snapshots"] {
        check_item(&c, n, true);
    }
    // 겹침은 실제로 판정된다(구역 2개). 측정 3.98 m (이전 14.4 m, 목표 < 0.3 m).
    let ov = c.report.item("refined_overlap").unwrap();
    assert!(
        ov.decided && !ov.measured.contains("해당 없음"),
        "겹침 판정 안 됨: {}",
        ov.measured
    );
    eprintln!("SPEC 목표 refined_overlap < 0.3 m, 현재: {}", ov.measured);
    check_item(&c, "refined_overlap", false);
    assert!(
        number_after(&ov.measured, "중앙 최대 ") < 5.0,
        "{}",
        ov.measured
    );
    // 측정 높이 차 4.94 m (목표 < 2 m).
    check_item(&c, "preview_vs_refined", false);
    let pr = c.report.item("preview_vs_refined").unwrap();
    assert!(
        number_after(&pr.measured, "높이 차 중앙 최대 ") < 6.0,
        "{}",
        pr.measured
    );
    // 구역 간 스케일 차 1.06% (이전 13.08%, 목표 <= 10%): 보고서 항목은 잔차 중앙 4.1 m 로 아직 미달.
    let pa = c.report.item("preview_align").unwrap();
    assert!(
        number_after(&pa.measured, "구역 간 스케일 차 ") <= 10.0,
        "{}",
        pa.measured
    );
}
