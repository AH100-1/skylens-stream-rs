//! 초벌 등록이 구역 끝 hi 를 넘는 미래 위치를 읽지 않는지(F-348) 확인한다.
//! 기록 줄: `region N images A (own B + helper C) last position - hi L`.

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline_with, HelperConfig, PipelineConfig};
use skylens_core::pipeline_stream::StreamOptions;
use skylens_core::stream::split_regions;
use skylens_core::synth::{Scene, SceneConfig};

/// report.json 의 구역 입력 기록에서 (구역 번호, 보조 수, 마지막 위치 − hi) 를 읽는다.
fn region_lines(report: &str) -> Vec<(usize, usize, i64)> {
    let mut v = Vec::new();
    for part in report.split('"') {
        if !part.contains("last position - hi") {
            continue;
        }
        let w: Vec<&str> = part.split_whitespace().collect();
        let at = |key: &str| w.iter().position(|x| *x == key).unwrap();
        let region = w[at("region") + 1].parse().unwrap();
        let helper = w[at("helper") + 1].trim_end_matches(')').parse().unwrap();
        let lead = w[at("hi") + 1].parse().unwrap();
        v.push((region, helper, lead));
    }
    v
}

fn run(coarse_back: bool, tag: &str) -> (Vec<(usize, usize, i64)>, usize, usize) {
    let root = std::env::temp_dir().join(format!(
        "skylens_helper_latency_{tag}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let n_pos = 22;
    Scene::new(SceneConfig {
        positions: n_pos,
        width: 320,
        height: 180,
        ..SceneConfig::default()
    })
    .write_dataset(&root.join("in"))
    .unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 8,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    let cfg = PipelineConfig {
        max_features: 600,
        dense_width: 80,
        ba_iters: 6,
        helper: HelperConfig {
            coarse_back,
            ..HelperConfig::default()
        },
        ..PipelineConfig::default()
    };
    let out = root.join("out");
    let res = run_pipeline_with(&ds, &cfg, &out, StreamOptions::default()).unwrap();
    let report = std::fs::read_to_string(out.join("snapshots/report.json"))
        .or_else(|_| std::fs::read_to_string(out.join("report.json")))
        .unwrap();
    let _ = std::fs::remove_dir_all(&root);
    (
        region_lines(&report),
        res.regions.len(),
        split_regions(n_pos, 8, 2).len(),
    )
}

#[test]
fn coarse_without_back_helper_reads_no_future_position() {
    let (off, got, want) = run(false, "off");
    assert_eq!(got, want);
    assert_eq!(off.len(), want, "{off:?}");
    // 초벌 입력의 마지막 위치는 hi 바로 앞(-1) 이고 뒤쪽 보조는 없다. 앞쪽 F 보조는 첫 구역만 없다.
    for (i, &(r, helper, lead)) in off.iter().enumerate() {
        assert_eq!(r, i);
        assert_eq!(lead, -1, "구역 {r} 초벌이 미래 위치를 읽음: {off:?}");
        assert_eq!(helper == 0, r == 0, "{off:?}");
    }
    // 대조: 뒤쪽 보조를 초벌에도 넣으면 마지막 구역을 뺀 모든 구역이 hi 이후 위치를 읽는다.
    let (on, _, _) = run(true, "on");
    assert!(on[..on.len() - 1].iter().all(|x| x.2 > 0), "{on:?}");
}

/// 측정용 80위치 합성 장면 쓰기(기본 크기): `SKYLENS_SYNTH80_OUT=<폴더> cargo test --release --test helper_latency -- --ignored`
#[test]
#[ignore]
fn write_80_position_scene() {
    let out = std::env::var("SKYLENS_SYNTH80_OUT").expect("SKYLENS_SYNTH80_OUT 필요");
    Scene::new(SceneConfig {
        positions: 80,
        ..SceneConfig::default()
    })
    .write_dataset(std::path::Path::new(&out))
    .unwrap();
}
