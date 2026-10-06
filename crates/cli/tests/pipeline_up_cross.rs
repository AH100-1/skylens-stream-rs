//! 위 방향 교차 검사: 합성 기본 장면(README 첫 명령과 같은 단구역 설정, stride 2 → 120장)을 돌린
//! 결과 report.json 에 최대 어긋남이 남고 문턱 0.3° 미만이다(실측 0.089°: 카메라별 0.080/0.076/0.089°).
//! 작은 구역(위치 8곳)에서는 정밀 모델 자체가 어긋나 이 값이 수십 도까지 나오므로 이 설정을 쓴다.

use skylens_core::align::UP_CROSS_WARN_DEG;
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::parse_json;

#[test]
fn run_report_has_up_cross_check_below_threshold() {
    let root = std::env::temp_dir().join(format!("skylens_upcross_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    })
    .write_dataset(&root.join("in"))
    .unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 2,
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
        ..PipelineConfig::default()
    };
    let res = run_pipeline(&ds, &cfg, &root.join("out")).unwrap();
    let report = std::fs::read_to_string(root.join("out/report.json")).unwrap();
    eprintln!("{report}");
    let j = parse_json(&report).expect("report.json 형식");
    let uc = j.get("up_cross_check").expect("up_cross_check 항목");
    let max = uc
        .get("max_diff_deg")
        .and_then(|v| v.as_f64())
        .expect("최대 어긋남이 숫자로 찍혀야 한다");
    assert!(
        max.is_finite() && max < UP_CROSS_WARN_DEG,
        "최대 어긋남 {max}"
    );
    assert_eq!(
        res.up_cross.max_diff_deg.map(|m| (m - max).abs() < 1e-3),
        Some(true)
    );
    assert!(!res.up_cross.exceeds);
    assert_eq!(res.up_cross.regions.len(), 1);
    assert_eq!(res.up_cross.regions[0].1, vec![0, 1, 2]);
    assert!(res.up_cross.regions[0].2.iter().all(|d| d.is_some()));
    assert!(res
        .issues
        .iter()
        .any(|i| i.contains("위 방향 교차 검사") && !i.contains("초과")));
    let _ = std::fs::remove_dir_all(&root);
}
