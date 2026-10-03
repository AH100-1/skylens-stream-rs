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
    let passed = report.items.iter().filter(|i| i.pass).count();

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

#[test]
fn explore() {
    for (st, ba) in [(2, 10), (2, 15), (1, 10), (1, 15)] {
        let t = std::time::Instant::now();
        let c = run_case(st, ba);
        eprintln!(
            "CASE stride {st} ba {ba}: reg {}/{} regions {} cmed {:.3} cmax {:.3} surf {:.3} secs {:.0}",
            c.registered, c.images, c.regions, c.center_med, c.center_max, c.surface_med, t.elapsed().as_secs_f64()
        );
        for i in &c.report.items {
            eprintln!(
                "CASE   {} pass={} decided={} {}",
                i.name, i.pass, i.decided, i.measured
            );
        }
    }
}
