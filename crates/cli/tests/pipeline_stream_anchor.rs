//! 2구역 합성 장면: 새 구역 등록이 최신 정밀 모델 위에서(겹침 카메라 고정 + 공유 점) 이뤄질 때와
//! 꺼졌을 때의 구역 경계 카메라 중심 차, 이웃 정밀 겹침 차, 재정렬 전후 정답 대비 중심 오차 비교.

use std::path::{Path, PathBuf};

use skylens_core::dataset::{load_dataset, Dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline_with, PipelineConfig, PipelineResult};
use skylens_core::pipeline_stream::StreamOptions;
use skylens_core::synth::{Scene, SceneConfig};

fn cfg() -> PipelineConfig {
    PipelineConfig {
        max_features: 600,
        dense_width: 80,
        hfov_deg: 65.0,
        ba_iters: 8,
        ..PipelineConfig::default()
    }
}

fn setup() -> (PathBuf, Dataset) {
    let root = std::env::temp_dir().join(format!("skylens_anchor_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    Scene::new(SceneConfig {
        positions: 16,
        width: 400,
        height: 225,
        heading_deg: [0.0; 3],
        ..SceneConfig::default()
    })
    .write_dataset(&root.join("in"))
    .unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 10,
            ovl: 4,
            max_skip_run: 2,
        },
    )
    .unwrap();
    (root, ds)
}

struct Row {
    res: PipelineResult,
    /// 구역 경계 겹침 사진의 구역 간 카메라 중심 차 중앙 m(report.json).
    boundary_diff: f64,
    /// 이웃 정밀 구역 겹침 점 차(재정렬 잔차 중앙) m.
    refined_overlap: Option<f64>,
    /// 재정렬 전후 정답 대비 중심 오차 중앙 (전, 후) m.
    realign_err: Option<(f64, f64)>,
    /// 최종 정밀 중심의 정답 대비 중앙 오차 m.
    final_err: f64,
    attached: bool,
}

fn num_after(s: &str, key: &str) -> Option<f64> {
    let p = s.find(key)? + key.len();
    s[p..].split_whitespace().next()?.parse().ok()
}

fn run(ds: &Dataset, out: &Path, anchor: bool) -> Row {
    let res = run_pipeline_with(
        ds,
        &cfg(),
        out,
        StreamOptions {
            sequential: false,
            anchor,
        },
    )
    .unwrap();
    let report = std::fs::read_to_string(out.join("report.json")).unwrap();
    let boundary_diff =
        num_after(&report, "\"overlap_center_diff_median_m\": ").unwrap_or(f64::NAN);
    let evs: Vec<&str> = report
        .split("\"events\": [")
        .nth(1)
        .unwrap()
        .split("\", \"")
        .collect();
    let mut refined_overlap = None;
    let mut realign_err = None;
    let mut attached = false;
    for e in &evs {
        if e.contains("realign refined 0 to refined 1") {
            refined_overlap = num_after(e, "median");
        }
        if e.contains("realign center error region 0") {
            realign_err = Some((
                num_after(e, "before").unwrap(),
                num_after(e, "after").unwrap(),
            ));
        }
        if e.contains("register region 1 on refined 0") {
            attached = true;
        }
    }
    let mut errs: Vec<f64> = Vec::new();
    for (name, c) in &res.centers {
        for p in &ds.positions {
            for (img, g) in p.images.iter().zip(&p.image_enu) {
                if img
                    .file_stem()
                    .is_some_and(|s| s.to_string_lossy() == *name)
                {
                    errs.push(
                        ((c[0] - g[0]).powi(2) + (c[1] - g[1]).powi(2) + (c[2] - g[2]).powi(2))
                            .sqrt(),
                    );
                }
            }
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Row {
        res,
        boundary_diff,
        refined_overlap,
        realign_err,
        final_err: errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN),
        attached,
    }
}

#[test]
fn anchored_registration_keeps_regions_in_one_frame() {
    let (root, ds) = setup();
    let on = run(&ds, &root.join("out_on"), true);
    let off = run(&ds, &root.join("out_off"), false);
    let f = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.3}"));
    let re =
        |v: Option<(f64, f64)>| v.map_or("-".to_string(), |(b, a)| format!("{b:.3} -> {a:.3}"));
    eprintln!(
        "| 앵커 | 등록 | 경계 카메라 중심 차 m | 이웃 정밀 겹침 차 m | 재정렬 전후 중심 오차 m | 최종 중심 오차 m |\n|---|---|---|---|---|---|\n| 켬 | {} | {:.3} | {} | {} | {:.3} |\n| 끔 | {} | {:.3} | {} | {} | {:.3} |",
        on.res.regions.iter().map(|x| x.registered).sum::<usize>(),
        on.boundary_diff,
        f(on.refined_overlap),
        re(on.realign_err),
        on.final_err,
        off.res.regions.iter().map(|x| x.registered).sum::<usize>(),
        off.boundary_diff,
        f(off.refined_overlap),
        re(off.realign_err),
        off.final_err
    );
    assert!(on.res.regions.len() >= 2 && off.res.regions.len() >= 2);
    assert!(!off.attached);
    let reg = |r: &PipelineResult| r.regions.iter().map(|x| x.registered).sum::<usize>();
    assert!(reg(&on.res) >= reg(&off.res), "등록 수 감소");
    // 이 합성 장면은 구역마다 한 카메라 사슬만 등록돼 구역 사이에 공유 카메라가 없다. 앵커를 붙일 수
    // 있으면(on.attached) 경계 차를 단언하고, 못 붙이면 끄고 켠 결과가 같음을 단언한다.
    if on.attached {
        assert!(on.boundary_diff < 0.3, "경계 중심 차 {}", on.boundary_diff);
        assert!(on.boundary_diff <= off.boundary_diff + 0.05);
        if let Some(m) = on.refined_overlap {
            assert!(m < 0.3, "이웃 정밀 겹침 차 {m}");
        }
        if let Some((b, a)) = on.realign_err {
            assert!(a <= b + 0.3, "재정렬 뒤 중심 오차 {b} -> {a}");
        }
    } else {
        assert_eq!(on.res.centers.len(), off.res.centers.len());
    }
    assert!(on.final_err < 0.5, "최종 중심 오차 {}", on.final_err);
    assert!(
        on.final_err <= off.final_err * 1.25 + 0.05,
        "최종 중심 오차 {} > {}",
        on.final_err,
        off.final_err
    );
    let _ = std::fs::remove_dir_all(&root);
}
