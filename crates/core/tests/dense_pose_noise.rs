//! 자세 잡음에서 이웃 선택 설정별 밀집 점군 표면 거리(무시 측정 + 수치 단언).
//!
//! 깊이 단계·반점 제거·융합은 공개 API 로 dense 모듈과 같은 순서로 다시 짠다
//! (이웃 선택만 설정을 바꿔 끼우기 위해).

use skylens_core::camera::{Camera, Pose};
use skylens_core::dense::{
    patchmatch_depth, remove_speckles, speckle_min_px, DenseView, DepthView, SweepConfig,
    SPECKLE_REL,
};
use skylens_core::fusion::{self, FusionConfig, FusionView};
use skylens_core::math::{Point3, Rotation3, Vector2, Vector3};
use skylens_core::ply::PointCloud;
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::undistort::undistort_to_long_side;
use skylens_core::view_selection::{self as vsel, NeighborConfig, SparsePoint};
use std::time::Instant;

const W: u32 = 480;

/// 16:9 장면의 높이.
fn height_for(w: u32) -> u32 {
    w * 9 / 16
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

fn scene_views(positions: usize, w: u32) -> (Scene, Vec<DenseView>, Vec<[f64; 3]>) {
    let h = height_for(w);
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
        let rgb = image::RgbImage::from_raw(img.width, img.height, img.data).unwrap();
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
    (s, views, sparse)
}

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

fn perturb(views: &[DenseView], pos_sigma: f64, rot_deg: f64, seed: u64) -> Vec<DenseView> {
    let mut rng = Lcg(seed);
    views
        .iter()
        .map(|v| {
            let c = v.camera.pose.center();
            let dc = Vector3::new(rng.normal(), rng.normal(), rng.normal()) * pos_sigma;
            let axis =
                Vector3::new(rng.normal(), rng.normal(), rng.normal()) * rot_deg.to_radians();
            let dr = Rotation3::from_scaled_axis(axis);
            let mut out = v.clone();
            out.camera.pose = Pose::from_center(dr * v.camera.pose.rotation, &(c + dc));
            out
        })
        .collect()
}

fn prepare(v: &DenseView, w: u32) -> DepthView {
    let k = v.camera.intrinsics.to_distorted();
    let u = undistort_to_long_side(&v.image, &k, w).unwrap();
    let rgb: Vec<[u8; 3]> = u.image.pixels().map(|p| p.0).collect();
    let gray = rgb
        .iter()
        .map(|c| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32)
        .collect();
    DepthView {
        camera: Camera {
            intrinsics: u.pinhole,
            pose: v.camera.pose,
        },
        gray,
        rgb,
        valid: u.valid,
    }
}

/// 이웃 설정만 바꿔 깊이→반점 제거→융합. (점군, 깊이 단계 초)
fn run(
    views: &[DenseView],
    sparse_pts: &[[f64; 3]],
    nc: &NeighborConfig,
    w: u32,
) -> (PointCloud, f64) {
    let t = Instant::now();
    let preps: Vec<DepthView> = views.iter().map(|v| prepare(v, w)).collect();
    let sparse: Vec<SparsePoint> = sparse_pts
        .iter()
        .map(|p| {
            let xyz = Point3::new(p[0], p[1], p[2]);
            let observers = preps
                .iter()
                .enumerate()
                .filter(|(_, v)| {
                    v.camera
                        .project(&xyz)
                        .is_some_and(|q| v.camera.intrinsics.contains(&q))
                })
                .map(|(i, _)| i)
                .collect();
            SparsePoint { xyz, observers }
        })
        .collect();
    let vs: Vec<vsel::View> = preps
        .iter()
        .enumerate()
        .map(|(i, v)| vsel::View {
            cam: v.camera,
            id: i,
        })
        .collect();
    let neighbors = vsel::select_neighbors_with(&vs, &sparse, 8, nc);
    let mut maps: Vec<_> = (0..preps.len())
        .map(|i| {
            let range = vsel::depth_range(&vs[i], &sparse);
            let nb: Vec<&DepthView> = neighbors[i].iter().map(|&j| &preps[j]).collect();
            patchmatch_depth(&preps[i], &nb, range, &SweepConfig::default())
        })
        .collect();
    for m in &mut maps {
        // 모듈과 같은 문턱 함수(면적 비례, 하한 4, 상한 400)를 쓴다.
        remove_speckles(m, SPECKLE_REL, speckle_min_px(m.w, m.h)).expect("지도 길이 정상");
    }
    let secs = t.elapsed().as_secs_f64();
    let fviews: Vec<FusionView> = preps
        .iter()
        .enumerate()
        .map(|(i, v)| FusionView {
            camera: v.camera,
            rgb: v.rgb.clone(),
            neighbors: neighbors[i].clone(),
            group: None,
        })
        .collect();
    let fcfg = FusionConfig {
        reproj_px: 1.0,
        depth_rel: 0.01,
        min_views: 3,
        normal_deg: 25.0,
        ..FusionConfig::default()
    };
    (
        fusion::try_fuse(&fviews, &maps, fcfg).unwrap_or_default(),
        secs,
    )
}

fn pct(d: &mut [f64], f: f64) -> f64 {
    d.sort_by(f64::total_cmp);
    d[((d.len() as f64 * f) as usize).min(d.len() - 1)]
}

/// 최소 각 없음(변경 전 기본 동작).
fn legacy() -> NeighborConfig {
    NeighborConfig {
        min_angle_deg: 0.0,
        ..NeighborConfig::default()
    }
}

fn surf(s: &Scene, cloud: &PointCloud) -> (f64, f64) {
    let mut d: Vec<f64> = cloud
        .points
        .iter()
        .map(|p| surface_dist(s, &p.xyz))
        .collect();
    (pct(&mut d, 0.5), pct(&mut d, 0.95))
}

fn settings() -> Vec<(&'static str, NeighborConfig)> {
    let base = legacy();
    let min = |m: f64| NeighborConfig {
        min_angle_deg: m,
        ..base
    };
    vec![
        ("현재 기본", base),
        ("최소 5도", min(5.0)),
        ("최소 8도", min(8.0)),
        ("최소 10도", min(10.0)),
        (
            "최적 12도(최소 8도)",
            NeighborConfig {
                min_angle_deg: 8.0,
                target_angle_deg: 12.0,
                sigma_below_deg: 4.0,
                ..base
            },
        ),
        (
            "자동(8도 상한, 25% 분위)",
            NeighborConfig {
                min_angle_deg: 8.0,
                auto_min_quantile: 0.25,
                ..base
            },
        ),
        (
            "자동(8도 상한, 50% 분위)",
            NeighborConfig {
                min_angle_deg: 8.0,
                auto_min_quantile: 0.5,
                ..base
            },
        ),
        (
            "8도 + 최소 6장 채움",
            NeighborConfig {
                min_angle_deg: 8.0,
                min_keep: 6,
                ..base
            },
        ),
        (
            "자동 50% + 최소 6장 채움",
            NeighborConfig {
                min_angle_deg: 8.0,
                auto_min_quantile: 0.5,
                min_keep: 6,
                ..base
            },
        ),
        (
            "최적 12도(최소 5도)",
            NeighborConfig {
                min_angle_deg: 5.0,
                target_angle_deg: 12.0,
                sigma_below_deg: 4.0,
                ..base
            },
        ),
    ]
}

/// 표: 자세 잡음 × 이웃 설정 → 표면 거리 중앙·95%·점 수·장당 시간(24장).
#[test]
#[ignore]
fn pose_noise_neighbor_table() {
    let (s, views, sparse) = scene_views(8, W);
    assert_eq!(views.len(), 24);
    let cases = [
        ("정답", 0.0, 0.0),
        ("0.05m 0.05도", 0.05, 0.05),
        ("0.2m 0.2도", 0.2, 0.2),
    ];
    for (cname, ps, rd) in cases {
        let vs = if ps > 0.0 {
            perturb(&views, ps, rd, 7)
        } else {
            views.clone()
        };
        for (sname, nc) in settings() {
            let (cloud, secs) = run(&vs, &sparse, &nc, W);
            let mut d: Vec<f64> = cloud
                .points
                .iter()
                .map(|p| surface_dist(&s, &p.xyz))
                .collect();
            let (med, p95) = (pct(&mut d, 0.5), pct(&mut d, 0.95));
            eprintln!(
                "TABLE {cname} | {sname} | points {} med {med:.4} p95 {p95:.4} s/img {:.2}",
                cloud.len(),
                secs / vs.len() as f64
            );
        }
    }
}

/// 최소 각 8도 설정(기본은 아님) 단언: 24장 장면, 기준은 실측(2026 측정 기계 4 코어)에 상한 ×1.2, 점 수 하한 ×0.8.
/// 실측: 정답 자세 중앙 0.0371 95% 0.1468 점 301570 / 잡음 0.05 m·0.05도 중앙 0.2857 95% 0.6148 점 197014.
/// 변경 전(최소 각 없음): 정답 0.0651·0.2778 / 잡음 0.4113·0.8417.
#[test]
#[ignore]
fn min_angle_8_robust_to_pose_noise() {
    let (s, views, sparse) = scene_views(8, W);
    let noisy = perturb(&views, 0.05, 0.05, 7);
    assert_eq!(NeighborConfig::default().min_angle_deg, 0.0);
    let new = NeighborConfig {
        min_angle_deg: 8.0,
        ..NeighborConfig::default()
    };
    let (c0, _) = run(&views, &sparse, &new, W);
    let (m0, p0) = surf(&s, &c0);
    let (c1, _) = run(&noisy, &sparse, &new, W);
    let (m1, p1) = surf(&s, &c1);
    let (l0, _) = run(&views, &sparse, &legacy(), W);
    let (lm0, _) = surf(&s, &l0);
    let (l1, _) = run(&noisy, &sparse, &legacy(), W);
    let (lm1, _) = surf(&s, &l1);
    eprintln!("ROBUST gt med {m0:.4} p95 {p0:.4} pts {} | noisy med {m1:.4} p95 {p1:.4} pts {} | legacy gt {lm0:.4} noisy {lm1:.4}", c0.len(), c1.len());
    assert!(m0 < 0.0371 * 1.2 && p0 < 0.1468 * 1.2, "정답 {m0} {p0}");
    assert!(m1 < 0.2857 * 1.2 && p1 < 0.6148 * 1.2, "잡음 {m1} {p1}");
    assert!(c0.len() as f64 > 301570.0 * 0.8 && c1.len() as f64 > 197014.0 * 0.8);
    assert!(m1 < 0.8 * lm1, "잡음 개선 {m1} vs {lm1}");
    assert!(m0 < 1.05 * lm0, "정답 악화 {m0} vs {lm0}");
}

/// 상시 시험: 12장·96×54 로 줄인 장면에서 최소 각 제한 경로(기본, 8도)를 돌려 점 수·표면 거리를 단언한다.
/// 문턱은 모듈의 `speckle_min_px` 하나를 쓴다(식이 둘로 갈라지지 않는다).
#[test]
fn small_scene_angle_paths_run_by_default() {
    let w = 96;
    assert_eq!(speckle_min_px(480, 270), 100);
    assert_eq!(speckle_min_px(960, 540), 400);
    assert_eq!(speckle_min_px(1920, 1080), 400);
    assert_eq!(speckle_min_px(w as usize, height_for(w) as usize), 6);
    let (s, views, sparse) = scene_views(4, w);
    assert_eq!(views.len(), 12);
    let new = NeighborConfig {
        min_angle_deg: 8.0,
        ..NeighborConfig::default()
    };
    for (name, nc) in [("기본", legacy()), ("최소 8도", new)] {
        let (cloud, _) = run(&views, &sparse, &nc, w);
        let (med, p95) = surf(&s, &cloud);
        eprintln!(
            "SMALL {name}: points {} med {med:.4} p95 {p95:.4}",
            cloud.len()
        );
        assert!(cloud.len() > 500, "{name}: 점 수 {}", cloud.len());
        assert!(!cloud.has_nan());
        assert!(med < 0.5 && p95 < 2.0, "{name}: 중앙 {med} 95% {p95}");
    }
}
