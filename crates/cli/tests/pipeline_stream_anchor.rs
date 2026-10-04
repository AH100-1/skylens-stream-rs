//! 2구역 합성 장면: 새 구역 등록이 최신 정밀 모델 위에서(겹침 카메라 고정 + 공유 점) 이뤄질 때와
//! 꺼졌을 때의 구역 경계 카메라 중심 차, 이웃 정밀 겹침 차, 재정렬 전후 정답 대비 중심 오차 비교.

use std::path::{Path, PathBuf};

use skylens_core::dataset::{load_dataset, Dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline_with, PipelineConfig, PipelineResult};
use skylens_core::pipeline_stream::StreamOptions;
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig, GPS_ORIGIN};
use skylens_core::verify::verify_dir;

fn cfg() -> PipelineConfig {
    PipelineConfig {
        max_features: 800,
        dense_width: 96,
        hfov_deg: 65.0,
        ba_iters: 15,
        ..PipelineConfig::default()
    }
}

/// 실측 편대 배치(기본 장면) 80 위치 × 3 대. 카메라 사이 짝은 F 기준 R·L 이 +20..=+40 위치 뒤라서
/// 구역이 그보다 짧으면 구역 안 카메라 사슬이 서로 이어지지 않는다(구역마다 한 카메라만 등록).
/// 그래서 구역을 48 위치로 잡는다(README 의 구역 2개 설정).
fn setup(seed: u64) -> (PathBuf, Dataset, Scene) {
    let root = std::env::temp_dir().join(format!("skylens_anchor_{seed}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let scene = Scene::new(SceneConfig {
        positions: 80,
        width: 320,
        height: 180,
        seed,
        ..SceneConfig::default()
    });
    scene.write_dataset(&root.join("in")).unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 48,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    (root, ds, scene)
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
    /// 앵커에 쓰인 공유(고정) 카메라 수, 공유 3D 점 쌍 수, sim3 잔차 중앙 m(구역 1 등록 기록).
    shared_cams: Option<usize>,
    shared_pts: Option<usize>,
    sim_resid: Option<f64>,
    final_err_max: f64,
    pass: usize,
    items: usize,
}

/// 정밀 점군의 정답 표면 대비 높이 편향(부호 있는 중앙 m). 이웃 구역 쌍의 겹침(수평 1 m 안 짝)에서 구역별
/// 편향과, 짝 높이 차의 부호 있는 중앙·절대 중앙을 낸다.
struct OverlapBias {
    pairs: usize,
    /// 앞·뒤 구역 점의 정답 대비 부호 있는 높이 차 중앙.
    bias_a: f64,
    bias_b: f64,
    /// (뒤 − 앞) 짝 높이 차의 부호 있는 중앙, 절대 중앙(verify 와 같은 값).
    signed: f64,
    abs: f64,
    /// 구역 전체 점의 편향 중앙.
    all_a: f64,
    all_b: f64,
    /// 정답 대비 높이 오차 절대 중앙(앞·뒤), 짝 두 점의 정답 표면 높이 차 절대 중앙(수평 어긋남 몫).
    abs_a: f64,
    abs_b: f64,
    slope_part: f64,
}

fn med(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn truth_dz(scene: &Scene, p: [f32; 3]) -> f64 {
    let first = scene.first_gps_origin();
    let g = skylens_core::geo::enu_to_geodetic(
        &skylens_core::nalgebra::Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64),
        &first,
    );
    let e = skylens_core::geo::geodetic_to_enu(&g, &GPS_ORIGIN);
    e.z - scene.surface_height(e.x, e.y)
}

fn overlap_bias(out: &Path, scene: &Scene, ka: usize, kb: usize) -> Option<OverlapBias> {
    let find = |k: usize| {
        let pre = format!("refined_{k:02}");
        std::fs::read_dir(out.join("refined"))
            .ok()?
            .filter_map(|e| e.ok())
            .find(|e| e.file_name().to_string_lossy().starts_with(&pre))
            .map(|e| read_ply_file(e.path()).unwrap())
    };
    let (ca, cb) = (find(ka)?, find(kb)?);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
    for (i, p) in cb.points.iter().enumerate() {
        grid.entry((p.xyz[0].floor() as i64, p.xyz[1].floor() as i64))
            .or_default()
            .push(i);
    }
    let (mut ba, mut bb, mut sg, mut sl) = (vec![], vec![], vec![], vec![]);
    for p in &ca.points {
        let (cx, cy) = (p.xyz[0].floor() as i64, p.xyz[1].floor() as i64);
        let mut best: Option<(f32, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &j in grid.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                    let q = &cb.points[j].xyz;
                    let d = ((p.xyz[0] - q[0]).powi(2) + (p.xyz[1] - q[1]).powi(2)).sqrt();
                    if d < 1.0 && best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, j));
                    }
                }
            }
        }
        if let Some((_, j)) = best {
            let q = cb.points[j].xyz;
            ba.push(truth_dz(scene, p.xyz));
            bb.push(truth_dz(scene, q));
            sg.push(q[2] as f64 - p.xyz[2] as f64);
            sl.push(
                (truth_dz(scene, q) - truth_dz(scene, p.xyz) - (q[2] as f64 - p.xyz[2] as f64))
                    .abs(),
            );
        }
    }
    let all = |c: &skylens_core::ply::PointCloud| {
        med(c
            .points
            .iter()
            .step_by(7)
            .map(|p| truth_dz(scene, p.xyz))
            .collect())
    };
    Some(OverlapBias {
        pairs: sg.len(),
        bias_a: med(ba.clone()),
        bias_b: med(bb.clone()),
        signed: med(sg.clone()),
        abs: med(sg.iter().map(|x| x.abs()).collect()),
        all_a: all(&ca),
        all_b: all(&cb),
        abs_a: med(ba.iter().map(|x| x.abs()).collect()),
        abs_b: med(bb.iter().map(|x| x.abs()).collect()),
        slope_part: med(sl),
    })
}

fn num_after(s: &str, key: &str) -> Option<f64> {
    let p = s.find(key)? + key.len();
    s[p..].split_whitespace().next()?.parse().ok()
}

fn run(ds: &Dataset, scene: &Scene, out: &Path, anchor: bool) -> Row {
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
    let (mut shared_cams, mut shared_pts, mut sim_resid) = (None, None, None);
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
            shared_pts = num_after(e, "shared points").map(|v| v as usize);
            sim_resid = num_after(e, "sim3 median");
            shared_cams = num_after(e, "shared cameras").map(|v| v as usize);
        }
    }
    // 정답 카메라 중심(첫 GPS 기준 좌표)과의 거리.
    let mut errs: Vec<f64> = Vec::new();
    for v in &scene.views {
        let stem = v.name.trim_end_matches(".jpg");
        if let Some(c) = res.centers.iter().find(|(n, _)| n == stem).map(|x| &x.1) {
            let t = scene.to_first_gps_frame(&v.camera.pose.center());
            errs.push(((c[0] - t.x).powi(2) + (c[1] - t.y).powi(2) + (c[2] - t.z).powi(2)).sqrt());
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let vr = verify_dir(out);
    eprintln!("{}", vr.to_table());
    for e in &evs {
        if e.contains("overlap cameras") {
            eprintln!("DIAG {e}");
        }
    }
    if let Some(b) = overlap_bias(out, scene, 0, 1) {
        eprintln!(
            "DIAG dense bias pairs {} region0 {:.3} region1 {:.3} signed(1-0) {:.3} abs {:.3} all0 {:.3} all1 {:.3} abs_err0 {:.3} abs_err1 {:.3} surface_pair_part {:.3}",
            b.pairs, b.bias_a, b.bias_b, b.signed, b.abs, b.all_a, b.all_b, b.abs_a, b.abs_b, b.slope_part
        );
    }
    Row {
        pass: vr.items.iter().filter(|i| i.pass).count(),
        items: vr.items.len(),
        final_err_max: errs.last().copied().unwrap_or(f64::NAN),
        shared_cams,
        shared_pts,
        sim_resid,
        res,
        boundary_diff,
        refined_overlap,
        realign_err,
        final_err: errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN),
        attached,
    }
}

/// 시드 목록은 환경변수 `ANCHOR_SEEDS`(쉼표)로 줄일 수 있다. 기본 1,2.
fn seeds() -> Vec<u64> {
    std::env::var("ANCHOR_SEEDS")
        .unwrap_or_else(|_| "1,2".into())
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

#[test]
fn anchored_registration_keeps_regions_in_one_frame() {
    let f = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.3}"));
    let re =
        |v: Option<(f64, f64)>| v.map_or("-".to_string(), |(b, a)| format!("{b:.3} -> {a:.3}"));
    let mut table = String::from(
        "| 시드 | 앵커 | 등록 | verify | 공유 카메라 | 공유 점 | sim3 잔차 m | 경계 카메라 중심 차 m | 이웃 정밀 겹침 차 m | 재정렬 전후 중심 오차 m | 최종 중심 오차 중앙/최대 m |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for seed in seeds() {
        let (root, ds, scene) = setup(seed);
        let on = run(&ds, &scene, &root.join("out_on"), true);
        if std::env::var("ANCHOR_ONLY").is_ok() {
            continue;
        }
        let off = run(&ds, &scene, &root.join("out_off"), false);
        for (tag, r) in [("켬", &on), ("끔", &off)] {
            table += &format!(
                "| {seed} | {tag} | {} | {}/{} | {} | {} | {} | {:.3} | {} | {} | {:.3} / {:.3} |\n",
                r.res.regions.iter().map(|x| x.registered).sum::<usize>(),
                r.pass,
                r.items,
                r.shared_cams.map_or("-".into(), |v| v.to_string()),
                r.shared_pts.map_or("-".into(), |v| v.to_string()),
                f(r.sim_resid),
                r.boundary_diff,
                f(r.refined_overlap),
                re(r.realign_err),
                r.final_err,
                r.final_err_max
            );
        }
        eprintln!("{table}");
        assert!(on.res.regions.len() >= 2 && off.res.regions.len() >= 2);
        assert!(!off.attached);
        // 구역 1 등록이 구역 0 정밀 모델에 실제로 붙는다: 겹침 카메라 2 대 이상 고정, 공유 점 충분.
        assert!(on.attached, "시드 {seed}: 앵커가 붙지 않음");
        assert!(
            on.shared_cams.unwrap() >= 2,
            "공유 카메라 {:?}",
            on.shared_cams
        );
        assert!(on.shared_pts.unwrap() >= 20, "공유 점 {:?}", on.shared_pts);
        assert!(on.sim_resid.unwrap() < 1.0, "sim3 잔차 {:?}", on.sim_resid);
        let reg = |r: &PipelineResult| r.regions.iter().map(|x| x.registered).sum::<usize>();
        assert!(reg(&on.res) >= reg(&off.res), "등록 수 감소");
        // 구역 겹침 점 차: 앵커가 끄기보다 나빠지지 않는다. 시드 2 는 아직 0.3 m 미만이 아니다(노트 참고).
        if let (Some(a), Some(b)) = (on.refined_overlap, off.refined_overlap) {
            assert!(a <= b + 0.05, "시드 {seed}: 이웃 정밀 겹침 차 {a} > {b}");
        }
        assert!(on.pass >= off.pass, "verify 통과 수 감소");
        assert!(on.final_err < 1.0, "최종 중심 오차 {}", on.final_err);
        assert!(
            on.final_err <= off.final_err * 1.25 + 0.05,
            "최종 중심 오차 {} > {}",
            on.final_err,
            off.final_err
        );
        if std::env::var("ANCHOR_KEEP").is_err() {
            let _ = std::fs::remove_dir_all(&root);
        } else {
            eprintln!("kept {}", root.display());
        }
    }
}

/// 보관한 출력 폴더(`ANALYZE_DIR`)의 겹침 높이 편향만 다시 계산한다. 시드는 `ANALYZE_SEED`.
#[test]
#[ignore]
fn analyze_kept_output() {
    let dir = std::env::var("ANALYZE_DIR").unwrap();
    let seed: u64 = std::env::var("ANALYZE_SEED").unwrap().parse().unwrap();
    let scene = Scene::new(SceneConfig {
        positions: 80,
        width: 320,
        height: 180,
        seed,
        ..SceneConfig::default()
    });
    let b = overlap_bias(Path::new(&dir), &scene, 0, 1).unwrap();
    eprintln!(
        "DIAG dense bias pairs {} region0 {:.3} region1 {:.3} signed(1-0) {:.3} abs {:.3} all0 {:.3} all1 {:.3} abs_err0 {:.3} abs_err1 {:.3} surface_pair_part {:.3}",
        b.pairs, b.bias_a, b.bias_b, b.signed, b.abs, b.all_a, b.all_b, b.abs_a, b.abs_b, b.slope_part
    );
}
