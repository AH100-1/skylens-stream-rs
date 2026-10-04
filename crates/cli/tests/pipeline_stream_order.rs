//! 2구역 합성 장면: 위치 단위 도착 → 초벌 즉시 출력 → 정밀 교체 → 재정렬의 사건 순서와
//! 순차 방식(구역마다 정밀이 끝난 뒤 다음 구역)과의 수치·시간 비교.

use std::path::{Path, PathBuf};

use skylens_core::dataset::{load_dataset, Dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline_with, PipelineConfig, PipelineResult};
use skylens_core::pipeline_stream::StreamOptions;
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::verify_dir;

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
    let root = std::env::temp_dir().join(format!("skylens_order_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    Scene::new(SceneConfig {
        positions: 16,
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
            span: 10,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    (root, ds)
}

#[derive(Debug, Clone)]
struct Ev {
    kind: String,
    region: usize,
    snapshot: Option<String>,
    points: usize,
}

/// manifest.json 의 events 줄을 읽는다.
fn events(out: &Path) -> Vec<Ev> {
    let t = std::fs::read_to_string(out.join("snapshots/manifest.json")).unwrap();
    let field = |l: &str, k: &str| -> String {
        let p = l.find(&format!("\"{k}\": ")).unwrap() + k.len() + 4;
        l[p..]
            .split([',', '}'])
            .next()
            .unwrap()
            .trim()
            .trim_matches('"')
            .to_string()
    };
    t.lines()
        .filter(|l| l.contains("\"kind\""))
        .map(|l| Ev {
            kind: field(l, "kind"),
            region: field(l, "region").parse().unwrap(),
            snapshot: match field(l, "snapshot").as_str() {
                "null" => None,
                s => Some(s.to_string()),
            },
            points: field(l, "points").parse().unwrap(),
        })
        .collect()
}

fn idx(ev: &[Ev], kind: &str, region: usize) -> usize {
    ev.iter()
        .position(|e| e.kind == kind && e.region == region)
        .unwrap_or_else(|| panic!("사건 없음 {kind} r{region}: {ev:?}"))
}

struct Run {
    res: PipelineResult,
    secs: f64,
    pass: usize,
    items: usize,
    center_err: f64,
}

fn run(ds: &Dataset, out: &Path, sequential: bool) -> Run {
    let t = std::time::Instant::now();
    let res = run_pipeline_with(
        ds,
        &cfg(),
        out,
        StreamOptions {
            sequential,
            ..StreamOptions::default()
        },
    )
    .unwrap();
    let secs = t.elapsed().as_secs_f64();
    let vr = verify_dir(out);
    let pass = vr.items.iter().filter(|i| i.pass).count();
    // 정밀 중심과 GPS 정답(합성 장면에서 GPS 는 잡음 없는 위치)의 중앙 오차.
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
    Run {
        res,
        secs,
        pass,
        items: vr.items.len(),
        center_err: errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN),
    }
}

#[test]
fn stream_order_events_and_numbers_match_sequential() {
    let (root, ds) = setup();
    let (so, qo) = (root.join("out_stream"), root.join("out_seq"));
    let s = run(&ds, &so, false);
    let q = run(&ds, &qo, true);
    assert!(s.res.regions.len() >= 2, "구역 {}", s.res.regions.len());
    let ev = events(&so);
    for e in &ev {
        eprintln!("{e:?}");
    }
    // 위치 단위 도착: 구역 0 의 위치들이 하나씩 적힌다.
    // 구역 1 에서 처음 새로 도착하는 위치 = 구역 0 의 끝(구역 0 사건 사이에는 그 앞 위치만 온다).
    let ds_lo1 = ev
        .iter()
        .position(|e| e.kind == "coarse_output" && e.region == 0)
        .map(|i| {
            ev[..i]
                .iter()
                .filter(|e| e.kind == "arrive_position")
                .count()
        })
        .unwrap();
    let arrive0: Vec<usize> = ev
        .iter()
        .filter(|e| e.kind == "arrive_position")
        .map(|e| e.region)
        .collect();
    assert!(arrive0.windows(2).all(|w| w[1] > w[0]), "{arrive0:?}");
    assert!(arrive0.len() >= 14 && arrive0.len() <= 16, "{arrive0:?}");
    // 순서: 구역 1 초벌 출력 전에 구역 0 초벌, 구역 0 정밀 교체는 구역 0 초벌 뒤, 마지막은 final.
    let c0 = idx(&ev, "coarse_output", 0);
    let c1 = idx(&ev, "coarse_output", 1);
    let r0 = idx(&ev, "refined_replace", 0);
    let r1 = idx(&ev, "refined_replace", 1);
    let first_pos_r1 = ev
        .iter()
        .position(|e| e.kind == "arrive_position" && e.region >= ds_lo1)
        .unwrap();
    assert!(c0 < first_pos_r1, "구역 0 초벌이 구역 1 도착보다 먼저");
    assert!(c0 < r0 && c0 < c1 && c1 < r1 && r0 < r1);
    assert_eq!(ev.last().unwrap().kind, "final");
    // 구역 0 정밀 교체 뒤 구역 1 쪽 재정렬 또는 구역 1 정밀 교체가 이어진다.
    assert!(ev.iter().any(|e| e.kind == "realign") || r1 > r0);
    // 시점별 스냅샷: 파일이 있고 NaN 없고, 정밀 교체 스냅샷 점 수가 초벌 직후보다 줄지 않는다(구역 0 기준 비어 있지 않음).
    for e in ev.iter().filter(|e| e.snapshot.is_some()) {
        let c = read_ply_file(so.join(e.snapshot.as_ref().unwrap())).unwrap();
        assert!(
            !c.is_empty() && !c.has_nan() && c.len() == e.points,
            "{e:?}"
        );
    }
    // 정밀 모델은 BA 의 이상치 제거로 초벌보다 점이 적을 수 있다(이 작은 장면은 초벌 18 점, 정밀 11 점).
    // 그래서 점 수 크기 비교 대신 구성을 정확히 단언한다: 각 시점 스냅샷의 점 수는
    // 그 시점에 정밀로 교체된 구역은 정밀 출력 파일, 나머지는 초벌 출력 파일의 점 수 합과 같다.
    let count = |dir: &str, region: usize| -> usize {
        let pre = format!("{dir}_{region:02}_");
        let f = std::fs::read_dir(so.join(dir))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.file_name().unwrap().to_string_lossy().starts_with(&pre))
            .unwrap();
        read_ply_file(f).unwrap().len()
    };
    let (n_pre, n_ref) = (
        [count("preview", 0), count("preview", 1)],
        [count("refined", 0), count("refined", 1)],
    );
    assert_eq!(ev[c0].points, n_pre[0], "구역 0 초벌 직후");
    // SPEC 순서 규칙: 구역이 차면 초벌을 앵커(직전 정밀)를 기다리지 않고 바로 낸다. 구역 1 초벌 시점에
    // 구역 0 은 아직 초벌이고, 구역 0 정밀 교체는 그 뒤다(main 과 같은 기대).
    assert!(c1 < r0, "구역 1 초벌이 구역 0 정밀 교체보다 먼저");
    assert_eq!(ev[c1].points, n_pre[0] + n_pre[1], "구역 1 초벌 직후");
    assert_eq!(ev[r0].points, n_ref[0] + n_pre[1], "구역 0 정밀 교체 직후");
    assert_eq!(ev[r1].points, n_ref[0] + n_ref[1], "구역 1 정밀 교체 직후");
    let last_snap = ev.iter().rev().find(|e| e.snapshot.is_some()).unwrap();
    assert_eq!(last_snap.points, n_ref[0] + n_ref[1], "{last_snap:?}");
    assert!(last_snap.points > 0, "{last_snap:?}");

    // 수치: 순차 방식과 같은 범위.
    let reg = |r: &PipelineResult| r.regions.iter().map(|x| x.registered).sum::<usize>();
    eprintln!(
        "| 방식 | 등록 | verify 통과/항목 | 정밀 중심 중앙 오차 m | 초 |\n|---|---|---|---|---|\n| 스트림 | {} | {}/{} | {:.4} | {:.1} |\n| 순차 | {} | {}/{} | {:.4} | {:.1} |",
        reg(&s.res), s.pass, s.items, s.center_err, s.secs,
        reg(&q.res), q.pass, q.items, q.center_err, q.secs
    );
    assert!(reg(&s.res) >= reg(&q.res), "등록 수 감소");
    assert!(s.pass >= q.pass, "verify 통과 수 감소");
    assert!(
        s.center_err <= q.center_err * 1.25 + 0.05,
        "중심 오차 {} > {}",
        s.center_err,
        q.center_err
    );
    // 순차 방식에서는 구역 0 정밀 교체가 구역 1 도착보다 먼저다.
    let sev = events(&qo);
    let r0q = idx(&sev, "refined_replace", 0);
    let a1q = sev
        .iter()
        .position(|e| e.kind == "arrive_position" && e.region >= ds_lo1)
        .unwrap();
    assert!(r0q < a1q, "순차 방식 순서");
    let _ = std::fs::remove_dir_all(&root);
}
