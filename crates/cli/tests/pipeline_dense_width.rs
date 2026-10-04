//! 밀집 폭(`--dense-width`)별 구역 하나의 밀집+융합 시간·점 수·정답 표면 거리(SPEC §3.6).
//!
//! 포즈·BA 단계는 거치지 않고 합성 정답 카메라와 정답 깊이에서 뽑은 희소 점을 입력으로 쓴다:
//! 밀집 단계(보정 → 이웃 → 시점별 PatchMatch → 융합)만 구간별로 잰다. 기본 장면 960×540,
//! 16 위치 × 3 대 = 48 장. 오래 걸려 무시 시험이다:
//! `cargo test --release -p skylens-stream --test pipeline_dense_width -- --ignored --nocapture`

use std::time::Instant;

use image::RgbImage;
use skylens_core::dense::{region_cloud_patchmatch, DenseConfig, DenseView};
use skylens_core::math::Vector2;
use skylens_core::ply::write_ply_file;
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::timing;

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

struct Row {
    width: u32,
    prepare: f64,
    neighbors: f64,
    estimate: f64,
    fuse: f64,
    write: f64,
    total: f64,
    points: usize,
    med: f64,
    p95: f64,
}

fn run_width(s: &Scene, views: &[DenseView], sparse: &[[f64; 3]], width: u32) -> Row {
    let cfg = DenseConfig {
        max_width: width,
        ..DenseConfig::default()
    };
    timing::reset();
    let t = Instant::now();
    let cloud = region_cloud_patchmatch(views, sparse, &cfg);
    let dense = t.elapsed().as_secs_f64();
    let path = std::env::temp_dir().join(format!("dense_width_{width}_{}.ply", std::process::id()));
    let tw = Instant::now();
    write_ply_file(&path, &cloud).unwrap();
    let write = tw.elapsed().as_secs_f64();
    let _ = std::fs::remove_file(&path);
    let mut d: Vec<f64> = cloud
        .points
        .iter()
        .map(|p| surface_dist(s, &p.xyz))
        .collect();
    d.sort_by(f64::total_cmp);
    let q = |f: f64| d[((d.len() as f64 * f) as usize).min(d.len() - 1)];
    Row {
        width,
        prepare: timing::total("dense_prepare"),
        neighbors: timing::total("dense_neighbors"),
        estimate: timing::total("dense_estimate"),
        fuse: timing::total("fusion"),
        write,
        total: dense + write,
        points: cloud.len(),
        med: q(0.5),
        p95: q(0.95),
    }
}

#[test]
#[ignore]
fn dense_width_table() {
    let (w, h) = (960u32, 540u32);
    let s = Scene::new(SceneConfig {
        positions: 16,
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
    assert_eq!(views.len(), 48);
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!(
        "cores {} load {}",
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        load.trim()
    );
    eprintln!("width | prepare | neighbors | estimate | fuse | write | total | points | med | p95");
    let mut rows = Vec::new();
    for width in [96, 240, 480, 960] {
        let r = run_width(&s, &views, &sparse, width);
        eprintln!(
            "{} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {:.4} | {:.4}",
            r.width,
            r.prepare,
            r.neighbors,
            r.estimate,
            r.fuse,
            r.write,
            r.total,
            r.points,
            r.med,
            r.p95
        );
        rows.push(r);
    }
    for r in &rows {
        assert!(r.points > 1000, "점 수 {} @ {}", r.points, r.width);
    }
    // 폭이 클수록 표면 거리 중앙값이 작다(해상도 이득).
    assert!(
        rows[3].med < rows[0].med,
        "{} vs {}",
        rows[3].med,
        rows[0].med
    );
}
