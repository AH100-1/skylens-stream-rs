//! 합성 장면 → run → verify, 정답 카메라 중심·표면 대비 오차.

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::Point3;
use skylens_core::pipeline::{run_pipeline, DenseMethod, PipelineConfig, PositionMethod};
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
    /// report.json 의 초벌·정밀 재투영 오차(px).
    preview_px: f64,
    refined_px: f64,
    /// 실행 중 낸 알림(위치 평균 실패 후 GPS 최소제곱으로 되돌아가면 "위치 평균 실패" 가 들어간다).
    issues: Vec<String>,
    report: skylens_core::verify::Report,
}

fn run_case(stride: usize, ba_iters: usize, method: DenseMethod, preview_ba: usize) -> Case {
    run_case_with(
        stride,
        ba_iters,
        method,
        PositionMethod::GpsLeastSquares,
        preview_ba,
    )
}

fn run_case_with(
    stride: usize,
    ba_iters: usize,
    method: DenseMethod,
    position: PositionMethod,
    preview_ba: usize,
) -> Case {
    run_case_full(stride, ba_iters, method, position, preview_ba, 5)
}

fn run_case_full(
    stride: usize,
    ba_iters: usize,
    method: DenseMethod,
    position: PositionMethod,
    preview_ba: usize,
    preview_refine: usize,
) -> Case {
    // 시험은 한 프로세스에서 병렬로 돌고 일부는 인자가 같으므로, 시험 이름(스레드 이름)을 경로에 넣어 작업 폴더를 분리한다.
    let tag = std::thread::current()
        .name()
        .unwrap_or("main")
        .replace("::", "_");
    let root = std::env::temp_dir().join(format!(
        "skylens_pipe_{}_{tag}_{stride}_{ba_iters}_{method:?}_{position:?}_{preview_ba}_{preview_refine}",
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
        dense_method: method,
        hfov_deg: 65.0,
        ba_iters,
        position,
        preview_ba_iters: preview_ba,
        preview_refine_iters: preview_refine,
        ..PipelineConfig::default()
    };
    let t = std::time::Instant::now();
    let res = run_pipeline(&ds, &cfg, &output).unwrap();
    eprintln!("{method:?} secs {:.1}", t.elapsed().as_secs_f64());
    eprintln!(
        "{method:?} dense secs {:.2}",
        res.regions.iter().map(|r| r.secs_dense).sum::<f64>()
    );
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
    let rj = std::fs::read_to_string(output.join("report.json")).unwrap();
    let rp = &rj[rj.find("\"reprojection_px\"").expect("reprojection_px")..];
    let preview_px = number_after(rp, "\"preview\": ");
    let refined_px = number_after(rp, "\"refined\": ");
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
    let passed = report.items.iter().filter(|i| i.pass).count();
    eprintln!("verify passed {passed}/{}", report.items.len());

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
    let over1 = d.iter().filter(|&&v| v > 1.0).count() as f64 / d.len() as f64;
    let mut sorted = d.clone();
    sorted.sort_by(f64::total_cmp);
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
    let sm = median(d);
    eprintln!(
        "{method:?} verify pass {passed}/{} p95 {p95:.3} m",
        report.items.len()
    );
    eprintln!(
        "cloud points {} surface distance median {sm:.3} m, share over 1 m {:.1}%",
        cloud.len(),
        100.0 * over1
    );
    assert!(cloud.len() >= 5000, "점 수 {}", cloud.len());
    let _ = std::fs::remove_dir_all(&root);
    Case {
        registered,
        images: ds.image_count(),
        regions: res.regions.len(),
        center_med: med,
        center_max: max,
        surface_med: sm,
        preview_px,
        refined_px,
        issues: res.issues.clone(),
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

/// 단일 구역, 사진별 깊이를 패치매치로(점 수·표면 거리 기준은 스윕과 같은 틀).
#[test]
fn synthetic_single_region_end_to_end_patchmatch() {
    let c = run_case(2, 15, DenseMethod::PatchMatch, 0);
    print_case(&c);
    assert_eq!(c.regions, 1);
    assert_eq!(c.registered, 3 * 40, "등록 수");
    assert!(c.surface_med <= 1.0, "표면 거리 중앙 {}", c.surface_med);
}

/// 단일 구역(README 첫 명령과 같은 설정: stride 2, span 48, BA 15회).
#[test]
fn synthetic_single_region_end_to_end() {
    let c = run_case(2, 15, DenseMethod::Sweep, 0);
    print_case(&c);
    assert_eq!(c.regions, 1);
    assert_eq!(c.images, 3 * 40);
    assert_eq!(c.registered, 3 * 40, "등록 수");
    assert!(c.center_med < 1.5, "중심 오차 중앙 {}", c.center_med);
    assert!(c.center_max < 6.0, "중심 오차 최대 {}", c.center_max);
    // 밀집 경유 후 측정 0.336 m (기준 <= 1.0 m).
    assert!(c.surface_med <= 1.0, "표면 거리 중앙 {}", c.surface_med);
    for n in [
        "registered",
        "region_images",
        "refined_reprojection",
        "snapshots",
    ] {
        check_item(&c, n, true);
    }
    // 기본 설정(초벌 BA 0회, 다듬기 5회) 실측: verify 7/7.
    // preview_align 통과: 점쌍 2716, 잔차 중앙 최대 0.818 m (목표 < 6 m).
    check_item(&c, "preview_align", true);
    let pa = c.report.item("preview_align").unwrap();
    assert!(
        number_after(&pa.measured, "잔차 중앙 최대 ") < 6.0,
        "{}",
        pa.measured
    );
    // 구역 1개: 이웃 겹침은 해당 없음.
    let ov = c.report.item("refined_overlap").unwrap();
    assert!(ov.measured.contains("해당 없음"), "{}", ov.measured);
    // 통과(초벌 다듬기 5회): 최근접 중앙 최대 0.688 m (< 3 m), 높이 차 중앙 최대 0.564 m (< 2 m). 초벌 점 광선 각 20도 이상만 남김.
    check_item(&c, "preview_vs_refined", true);
    let pr = c.report.item("preview_vs_refined").unwrap();
    assert!(
        number_after(&pr.measured, "높이 차 중앙 최대 ") < 2.0,
        "{}",
        pr.measured
    );
    assert!(
        number_after(&pr.measured, "최근접 중앙 최대 ") < 3.0,
        "{}",
        pr.measured
    );
}

/// 구역 2개 이상(README 둘째 명령: stride 1 → 80위치, span 48), 기본 설정(초벌 BA 0회).
/// 이웃 겹침이 실제로 판정된다. 기본 설정 CLI 실측: verify 6/7, 종료 코드 1.
/// 알려진 미달 항목은 SPEC 목표를 출력하고 측정값을 그대로 단언한다(완화 아님).
/// 목표를 달성하면 해당 단언을 바꾼다.
#[test]
fn synthetic_two_region_end_to_end() {
    let c = run_case(1, 15, DenseMethod::Sweep, 0);
    print_case(&c);
    assert!(c.regions >= 2, "구역 수 {}", c.regions);
    assert_eq!(c.images, 3 * 80);
    assert_eq!(c.registered, 240, "등록 수");
    check_item(&c, "registered", true);
    assert!(c.center_med < 2.5, "중심 오차 중앙 {}", c.center_med);
    assert!(c.surface_med < 4.5, "표면 거리 중앙 {}", c.surface_med);
    for n in ["region_images", "refined_reprojection", "snapshots"] {
        check_item(&c, n, true);
    }
    // 겹침은 실제로 판정된다(구역 2개): 1쌍, 높이 차 중앙 최대 0.257 m (목표 < 0.3 m) 통과.
    check_item(&c, "refined_overlap", true);
    let ov = c.report.item("refined_overlap").unwrap();
    assert!(!ov.measured.contains("해당 없음"), "{}", ov.measured);
    assert!(
        number_after(&ov.measured, "중앙 최대 ") < 0.3,
        "{}",
        ov.measured
    );
    // 통과(초벌 중심 다듬기 5회): 높이 차 중앙 최대 1.026 m, 최근접 0.970 m (목표 높이 차 < 2 m, 최근접 < 3 m).
    check_item(&c, "preview_vs_refined", true);
    let pr = c.report.item("preview_vs_refined").unwrap();
    assert!(
        number_after(&pr.measured, "높이 차 중앙 최대 ") < 2.0,
        "{}",
        pr.measured
    );
    // 통과: 구역 간 대응을 구역의 모든 이미지 관측으로 만들어 점쌍 최소 1178 (창 안 12장만 쓰면 324),
    // 구역 간 스케일 차 0.26%, 잔차 중앙 최대 0.735 m (< 6 m). 점쌍 하한은 SPEC §4 의 1000.
    check_item(&c, "preview_align", true);
    let pa = c.report.item("preview_align").unwrap();
    assert!(
        number_after(&pa.measured, "점쌍 최소 ") >= 1000.0,
        "{}",
        pa.measured
    );
    assert!(
        number_after(&pa.measured, "구역 간 스케일 차 ") <= 1.0,
        "{}",
        pa.measured
    );
    assert!(
        number_after(&pa.measured, "잔차 중앙 최대 ") < 6.0,
        "{}",
        pa.measured
    );
}

/// 초벌 BA 옵션(0회 vs 8회)은 초벌 모델만 바꾸고 정밀 BA 시작점은 같다:
/// 정밀 카메라 중심 오차 중앙/최대가 같다(허용 오차 1e-3 m). 초벌 쪽 재투영 오차만 줄어든다.
#[test]
fn preview_ba_option_does_not_change_refined() {
    let a = run_case(2, 15, DenseMethod::Sweep, 0);
    let b = run_case(2, 15, DenseMethod::Sweep, 8);
    eprintln!(
        "preview_ba 0: center med {:.4} max {:.4} | 8: med {:.4} max {:.4}",
        a.center_med, a.center_max, b.center_med, b.center_max
    );
    assert!((a.center_med - b.center_med).abs() < 1e-3, "중앙");
    assert!((a.center_max - b.center_max).abs() < 1e-3, "최대");
    assert_eq!(a.registered, b.registered);
    // 판정 항목 refined_reprojection 은 정밀 쪽이라 둘 다 통과한다.
    check_item(&a, "refined_reprojection", true);
    check_item(&b, "refined_reprojection", true);
}

/// 위치 평균 경로(`--position translation-averaging` 과 같은 설정), 트랙은 `tracks::build_tracks`.
/// 실측(`PositionMethod::TranslationAveraging`, 초벌 BA 0, 다듬기 5): 등록 120/120, 중심 중앙 0.245/최대 0.508 m,
/// 표면 중앙 0.390 m, 되돌아감 없음.
/// 상한은 실측에 여유를 둔 값이고, 위치 평균이 실패해 GPS 최소제곱으로 되돌아가면 실패한다.
#[test]
fn synthetic_single_region_translation_averaging() {
    let c = run_case_with(
        2,
        15,
        DenseMethod::Sweep,
        PositionMethod::TranslationAveraging,
        0,
    );
    print_case(&c);
    eprintln!(
        "TA registered {} of {} center med {:.3} max {:.3} fallback {}",
        c.registered,
        c.images,
        c.center_med,
        c.center_max,
        c.issues.iter().any(|i| i.contains("위치 평균 실패"))
    );
    assert!(
        !c.issues.iter().any(|i| i.contains("위치 평균 실패")),
        "위치 평균이 GPS 최소제곱으로 되돌아감: {:?}",
        c.issues
    );
    assert_eq!(c.regions, 1);
    assert_eq!(c.registered, c.images, "등록 수");
    assert_eq!(c.registered, 120);
    assert!(c.center_med < 0.35, "중심 오차 중앙 {}", c.center_med);
    assert!(c.center_max < 0.75, "중심 오차 최대 {}", c.center_max);
    assert!(
        c.report.items.iter().all(|i| i.pass),
        "verify 전부 통과해야 함"
    );
    assert_eq!(c.report.items.len(), 7);
    let pr = c.report.item("preview_vs_refined").unwrap();
    assert!(
        number_after(&pr.measured, "높이 차 중앙 최대 ") < 0.2,
        "{}",
        pr.measured
    );
    assert!(
        number_after(&pr.measured, "최근접 중앙 최대 ") < 0.5,
        "{}",
        pr.measured
    );
}

/// `--preview-refine-iters` 0 과 5(기본)의 verify 결과를 항목별로 고정한다(F-313).
/// 위치 전용 다듬기는 BA 가 아니다(회전 고정, 카메라 중심만). 두 설정 수치는 README 의 '초벌 다듬기' 표.
#[test]
fn preview_refine_settings_verify_outcome() {
    let a = run_case_full(
        2,
        15,
        DenseMethod::Sweep,
        PositionMethod::GpsLeastSquares,
        0,
        0,
    );
    let b = run_case_full(
        2,
        15,
        DenseMethod::Sweep,
        PositionMethod::GpsLeastSquares,
        0,
        5,
    );
    for (tag, c) in [("refine 0", &a), ("refine 5", &b)] {
        eprintln!(
            "REFINE {tag}: preview reproj {:.3} px refined {:.3} px verify {}/{}",
            c.preview_px,
            c.refined_px,
            c.report.items.iter().filter(|i| i.pass).count(),
            c.report.items.len()
        );
        for i in &c.report.items {
            eprintln!("REFINE {tag}   {} pass={} {}", i.name, i.pass, i.measured);
        }
    }
    for (c, px_lo, px_hi) in [(&a, 2.5, 3.5), (&b, 0.7, 1.2)] {
        assert_eq!(c.registered, 120);
        // 단구역은 두 설정 모두 7/7 통과한다.
        assert!(
            c.report.items.iter().all(|i| i.pass),
            "verify 전부 통과해야 함"
        );
        assert_eq!(c.report.items.len(), 7);
        assert!(
            c.preview_px > px_lo && c.preview_px < px_hi,
            "초벌 재투영 {} px",
            c.preview_px
        );
        assert!((c.refined_px - 0.313).abs() < 0.05, "정밀 {}", c.refined_px);
    }
    // 정밀 쪽은 같고 초벌만 달라진다(위치 전용 다듬기는 정밀 시작점을 바꾸지 않는다).
    assert!((a.center_med - b.center_med).abs() < 1e-3);
    // 다듬기가 초벌 재투영을 2.5분의 1 미만으로 줄이고(실측 2.985 → 0.948 px) 초벌-정밀 차이를 줄인다.
    assert!(b.preview_px < a.preview_px / 2.5);
    let ha = number_after(
        &a.report.item("preview_vs_refined").unwrap().measured,
        "높이 차 중앙 최대 ",
    );
    let hb = number_after(
        &b.report.item("preview_vs_refined").unwrap().measured,
        "높이 차 중앙 최대 ",
    );
    assert!(hb < ha, "높이 차 {ha} -> {hb}");
}

/// 구역 2개에서 `--preview-refine-iters 0`: 초벌 재투영이 크고 preview_vs_refined 가 SPEC 기준에 못 미친다.
/// 알려진 미달을 측정값 그대로 단언한다(완화 아님). 다듬기 5회(기본)는 위 synthetic_two_region_end_to_end 가 통과를 고정한다.
#[test]
fn two_region_refine_off_verify_outcome() {
    let c = run_case_full(
        1,
        15,
        DenseMethod::Sweep,
        PositionMethod::GpsLeastSquares,
        0,
        0,
    );
    print_case(&c);
    eprintln!(
        "REFINE0 two-region preview reproj {:.3} px refined {:.3} px",
        c.preview_px, c.refined_px
    );
    assert_eq!(c.registered, 240);
    for n in [
        "registered",
        "region_images",
        "refined_reprojection",
        "snapshots",
        "refined_overlap",
    ] {
        check_item(&c, n, true);
    }
    // 실측: 초벌 재투영 3.209 px, preview_vs_refined 최근접 중앙 최대 4.535 m, 높이 차 4.767 m (목표 3 m, 2 m).
    assert!(c.preview_px > 2.5, "초벌 재투영 {}", c.preview_px);
    check_item(&c, "preview_vs_refined", false);
    let pr = c.report.item("preview_vs_refined").unwrap();
    assert!(
        number_after(&pr.measured, "높이 차 중앙 최대 ") > 2.0,
        "{}",
        pr.measured
    );
    assert!(
        number_after(&pr.measured, "최근접 중앙 최대 ") > 3.0,
        "{}",
        pr.measured
    );
    check_item(&c, "preview_align", true);
}
