//! 초벌 회전 오차 진단(구역 0, README 와 같은 장면·설정): 회전 평균 입력 간선을 종류별(같은 카메라 위치
//! 간격 / 카메라 간)로 나눠 합성 정답 대비 상대 회전 오차를 재고, 간선 묶음을 빼거나 가중을 바꾼 판에서
//! 초벌 회전 오차 중앙을 다시 잰다. 오래 걸려 기본 시험에서 뺀다:
//! `cargo test --release -p skylens-stream --test pipeline_preview_rot -- --ignored --nocapture`.

use skylens_core::camera::Pose;
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::{Matrix3, Rotation3};
use skylens_core::pipeline::{preview_rotation_diag, PreviewOpts};
use skylens_core::synth::{Scene, SceneConfig};

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

struct Measured {
    /// 좌표계 무관(전역 회전 맞춘) 회전 평균 직후 회전 오차 중앙(도).
    free_med: f64,
    /// 좌표계 맞춤·위치 단계 뒤 초벌 포즈 회전 오차 중앙(도).
    placed_med: f64,
    placed_p95: f64,
}

struct Edge {
    kind: String,
    err_deg: f64,
    inliers: f64,
    parallax: f64,
}

fn run(stride: usize, specs: &[&str]) -> (Vec<Edge>, Vec<Measured>) {
    let root = std::env::temp_dir().join(format!("skylens_prerot_{stride}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    scene.write_dataset(&root).unwrap();
    let ds = load_dataset(
        &root,
        DatasetConfig {
            stride,
            span: 48,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    let hi = (48 + 2).min(ds.positions.len());
    let opts: Vec<PreviewOpts> = specs.iter().map(|s| PreviewOpts::parse(s)).collect();
    let (gids, edges, runs) = preview_rotation_diag(&ds, 0, hi, 800, 65.0, &opts).unwrap();
    let tp: Vec<Pose> = gids
        .iter()
        .map(|&g| {
            let name = ds.positions[g / 3].images[g % 3]
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let v = scene.views.iter().find(|v| v.name == name).unwrap();
            let c = scene.to_first_gps_frame(&v.camera.pose.center());
            Pose::from_center(v.camera.pose.rotation, &c)
        })
        .collect();
    let es: Vec<Edge> = edges
        .iter()
        .map(|e| {
            let truth = tp[e.j].rotation * tp[e.i].rotation.inverse();
            Edge {
                kind: e.gap.map_or("cross".to_string(), |g| format!("gap{g:02}")),
                err_deg: (e.rot * truth.inverse()).angle().to_degrees(),
                inliers: e.inliers as f64,
                parallax: e.parallax_deg,
            }
        })
        .collect();
    let ms = runs
        .iter()
        .map(|r| {
            let mut m = Matrix3::zeros();
            for (q, t) in r.rots.iter().zip(&tp) {
                if let Some(q) = q {
                    m += q.matrix().transpose() * t.rotation.matrix();
                }
            }
            let sv = m.svd(true, true);
            let q = Rotation3::from_matrix_unchecked(sv.u.unwrap() * sv.v_t.unwrap());
            let mut free: Vec<f64> = r
                .rots
                .iter()
                .zip(&tp)
                .filter_map(|(a, t)| {
                    Some(
                        (a.as_ref()? * q * t.rotation.inverse())
                            .angle()
                            .to_degrees(),
                    )
                })
                .collect();
            let mut placed: Vec<f64> = r
                .poses
                .iter()
                .zip(&tp)
                .filter_map(|(p, t)| {
                    Some(
                        (p.as_ref()?.rotation * t.rotation.inverse())
                            .angle()
                            .to_degrees(),
                    )
                })
                .collect();
            Measured {
                free_med: pct(&mut free, 0.5),
                placed_med: pct(&mut placed, 0.5),
                placed_p95: pct(&mut placed, 0.95),
            }
        })
        .collect();
    let _ = std::fs::remove_dir_all(&root);
    (es, ms)
}

fn edge_table(stride: usize, es: &[Edge]) {
    let mut kinds: Vec<&str> = es.iter().map(|e| e.kind.as_str()).collect();
    kinds.sort();
    kinds.dedup();
    eprintln!("EDGES stride {stride}: 종류 | 개수 | 상대 회전 오차 중앙° | 95%° | 정상 대응 중앙 | 시차 각 중앙°");
    for k in kinds {
        let sel: Vec<&Edge> = es.iter().filter(|e| e.kind == k).collect();
        let mut err: Vec<f64> = sel.iter().map(|e| e.err_deg).collect();
        let mut inl: Vec<f64> = sel.iter().map(|e| e.inliers).collect();
        let mut par: Vec<f64> = sel.iter().map(|e| e.parallax).collect();
        eprintln!(
            "EDGES stride {stride}: {k} | {} | {:.3} | {:.3} | {:.0} | {:.2}",
            sel.len(),
            pct(&mut err, 0.5),
            pct(&mut err, 0.95),
            pct(&mut inl, 0.5),
            pct(&mut par, 0.5)
        );
    }
}

const SPECS: [&str; 15] = [
    "",
    "passes=2",
    "min_inl=30,passes=2",
    "gap1w=0.3",
    "gap1w=0.3,passes=2",
    "dropgap=1",
    "dropgap=1+2",
    "dropgap=1+2+3",
    "dropgap=1+2+3+4+5",
    "minpar=1",
    "minpar=2",
    "w=par",
    "w=par,minpar=1",
    "nocross=1",
    "dropgap=16",
];

#[test]
#[ignore]
fn preview_rotation_by_edge_kind() {
    let specs: Vec<String> = match std::env::var("SKYLENS_ROT_SPECS") {
        Ok(s) => s.split(';').map(str::to_string).collect(),
        Err(_) => SPECS.iter().map(|s| s.to_string()).collect(),
    };
    let specs: Vec<&str> = specs.iter().map(String::as_str).collect();
    for stride in [1usize, 2] {
        let (es, ms) = run(stride, &specs);
        edge_table(stride, &es);
        let med_of = |kind: &str| {
            let mut v: Vec<f64> = es
                .iter()
                .filter(|e| e.kind == kind)
                .map(|e| e.err_deg)
                .collect();
            pct(&mut v, 0.5)
        };
        // 측정(구역 0): 짧은 간격 간선 상대 회전 오차 중앙 0.08°, 카메라 간 2.2~2.8°. 회전 평균 직후(전역 회전만 맞춘)
        // 오차는 stride 1 0.16° / stride 2 0.49° 로 작다: 초벌 포즈 회전 오차(2.75° / 0.49°)의 차는 좌표계 맞춤에서 온다.
        assert!(med_of("gap01") < 0.15, "{}", med_of("gap01"));
        assert!(med_of("cross") > 1.5, "{}", med_of("cross"));
        if specs[0].is_empty() {
            assert!(ms[0].free_med < 0.3 || stride == 2, "{}", ms[0].free_med);
            assert!(ms[0].free_med < 0.7, "{}", ms[0].free_med);
            if stride == 1 {
                assert!(
                    ms[0].placed_med > 2.0 && ms[0].placed_med < 3.3,
                    "{}",
                    ms[0].placed_med
                );
            } else {
                assert!(ms[0].placed_med < 0.7, "{}", ms[0].placed_med);
            }
        }
        for (s, m) in specs.iter().zip(&ms) {
            eprintln!(
                "ROT stride {stride} [{s}] free med {:.2} placed med {:.2} p95 {:.2}",
                m.free_med, m.placed_med, m.placed_p95
            );
        }
    }
}
