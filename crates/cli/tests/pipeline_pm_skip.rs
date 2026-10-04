//! 측정용(무시 시험): 깊이 너비 480(층이 여럿)에서 PatchMatch 의 `skip_cost` 0 과 0.08 을 비교한다.
//! `run` 옵션에는 PatchMatch 설정을 넘기는 길이 없으므로, 밀집 단계(`region_cloud_with`)만 같은 설정
//! (이웃 8, 폭 480)으로 직접 부르되 깊이 추정기에서 `skip_cost` 를 바꾼다. 카메라는 합성 정답 자세다.
//! cargo test --release -p skylens-stream --test pipeline_pm_skip -- --ignored --nocapture

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use image::RgbImage;
use skylens_core::dense::{region_cloud_with, DenseConfig, DenseView, DepthView, SweepConfig};
use skylens_core::fusion::DepthMap;
use skylens_core::nalgebra::Vector2;
use skylens_core::patchmatch as pm;
use skylens_core::ply::PointCloud;
use skylens_core::synth::{Scene, SceneConfig};

static SKIP_BITS: AtomicU32 = AtomicU32::new(0);

fn estimator(r: &DepthView, nbrs: &[&DepthView], range: (f64, f64), _s: &SweepConfig) -> DepthMap {
    let to_view = |v: &DepthView| {
        let k = v.camera.intrinsics;
        let data: Vec<f32> = v.gray.iter().map(|g| g / 255.0).collect();
        pm::View {
            camera: v.camera,
            image: pm::GrayImage::new(k.width as usize, k.height as usize, data),
        }
    };
    let rv = to_view(r);
    let nv: Vec<pm::View> = nbrs.iter().map(|n| to_view(n)).collect();
    let cfg = pm::Config {
        skip_cost: f32::from_bits(SKIP_BITS.load(Ordering::Relaxed)),
        ..pm::Config::default()
    };
    let m = pm::estimate(&rv, &nv, range, &cfg);
    DepthMap {
        w: m.w,
        h: m.h,
        depth: m.depth,
        normal: m.normal,
        cost: m.cost,
    }
}

fn surface_dist(s: &Scene, p: &[f32; 3]) -> f64 {
    let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
    let mut d = (z - s.surface_height(x, y)).abs();
    for b in &s.buildings {
        let dx = (b.min.x - x).max(x - b.max.x).max(0.0);
        let dy = (b.min.y - y).max(y - b.max.y).max(0.0);
        let dz = (z - b.top).max(0.0);
        d = d.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    d
}

fn q(d: &[f64], f: f64) -> f64 {
    d[((d.len() as f64 * f) as usize).min(d.len() - 1)]
}

#[test]
#[ignore = "측정용(부하가 낮을 때 시간이 의미 있음)"]
fn skip_cost_at_width_480() {
    let (w, h) = (480u32, 270u32);
    let positions: usize = std::env::var("PM_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let s = Scene::new(SceneConfig {
        positions,
        width: w,
        height: h,
        ..SceneConfig::default()
    });
    let mut views = Vec::new();
    let mut sparse = Vec::new();
    for (vi, v) in s.views.iter().enumerate() {
        let (img, depth) = s.render(v);
        let rgb = RgbImage::from_raw(img.width, img.height, img.data).unwrap();
        if vi % 2 == 0 {
            for y in (4..h as usize).step_by(9) {
                for x in (4..w as usize).step_by(9) {
                    let d = depth[y * w as usize + x];
                    if d.is_finite() {
                        let p = v
                            .camera
                            .unproject(&Vector2::new(x as f64 + 0.5, y as f64 + 0.5), d as f64);
                        sparse.push([p.x, p.y, p.z]);
                    }
                }
            }
        }
        views.push(DenseView {
            camera: v.camera,
            image: rgb,
        });
    }
    let cfg = DenseConfig {
        max_width: w,
        ..DenseConfig::default()
    };
    for round in 0..2 {
        for skip in [0.0f32, 0.08] {
            SKIP_BITS.store(skip.to_bits(), Ordering::Relaxed);
            let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
            let t = Instant::now();
            let cloud: PointCloud =
                region_cloud_with(&views, &sparse, &cfg, estimator, &SweepConfig::default());
            let secs = t.elapsed().as_secs_f64();
            let mut d: Vec<f64> = cloud
                .points
                .iter()
                .map(|p| surface_dist(&s, &p.xyz))
                .collect();
            d.sort_by(f64::total_cmp);
            assert!(!d.is_empty());
            eprintln!(
                "회 {round} skip_cost {skip}: 사진 {} 점 {} 중앙 {:.4} 90% {:.4} 95% {:.4} 시간 {secs:.1} s 부하 {}",
                views.len(),
                d.len(),
                q(&d, 0.5),
                q(&d, 0.9),
                q(&d, 0.95),
                load.trim()
            );
        }
    }
}
