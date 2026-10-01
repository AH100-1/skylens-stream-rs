//! 합성 검증 장면 (SPEC §6).
//!
//! 지면 = 완만한 높낮이 + 상자 건물, 표면 색은 절차적 무늬. 드론은 고도 30m 에서 +x 방향 직선 비행,
//! 위치마다 카메라 3대(앞 F, 오른쪽 R, 왼쪽 L)가 아래로 기울어져 있다.
//! 정답 카메라·표면을 알고 있으므로 이후 단계의 오차를 직접 잴 수 있다.

use crate::camera::{Camera, Intrinsics, Pose};
use crate::math::{Matrix3, Point3, Rotation3, Vector2, Vector3};

/// 카메라 종류.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CamId {
    F,
    R,
    L,
}

impl CamId {
    pub const ALL: [CamId; 3] = [CamId::F, CamId::R, CamId::L];

    pub fn letter(self) -> char {
        match self {
            CamId::F => 'F',
            CamId::R => 'R',
            CamId::L => 'L',
        }
    }
}

/// 축 정렬 상자 건물: 바닥 (x0,y0)-(x1,y1), 지붕 높이 top.
#[derive(Clone, Copy, Debug)]
pub struct Building {
    pub min: Vector2<f64>,
    pub max: Vector2<f64>,
    pub top: f64,
}

/// 장면 설정.
#[derive(Clone, Debug)]
pub struct SceneConfig {
    pub positions: usize,
    pub spacing: f64,
    pub altitude: f64,
    pub width: u32,
    pub height: u32,
    pub hfov_deg: f64,
    /// 카메라 내려다보는 각(수평 아래, 도).
    pub tilt_deg: f64,
    pub gps_sigma: f64,
    pub seed: u64,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            positions: 80,
            spacing: 2.5,
            altitude: 30.0,
            width: 960,
            height: 540,
            hfov_deg: 70.0,
            tilt_deg: 50.0,
            gps_sigma: 1.5,
            seed: 1,
        }
    }
}

/// 한 장의 정답 정보.
#[derive(Clone, Debug)]
pub struct View {
    pub name: String,
    pub cam: CamId,
    pub position: usize,
    pub camera: Camera,
}

/// 합성 장면.
#[derive(Clone, Debug)]
pub struct Scene {
    pub config: SceneConfig,
    pub buildings: Vec<Building>,
    pub views: Vec<View>,
    /// 위치별 GPS (동-북-위, 미터): 정답 기체 중심 + 잡음.
    pub gps_enu: Vec<Point3<f64>>,
    /// 위치별 정답 기체 중심.
    pub rig_centers: Vec<Point3<f64>>,
}

/// 결정적 64비트 해시 (splitmix64).
fn hash64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn hash_unit(a: i64, b: i64, c: u64) -> f64 {
    let h = hash64((a as u64).wrapping_mul(0x0010_0000_01b3) ^ hash64(b as u64 ^ hash64(c)));
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// 표준 정규 난수 (Box–Muller, 결정적).
fn gauss(seed: u64, i: u64) -> f64 {
    let u1 = hash_unit(i as i64, 1, seed).max(1e-300);
    let u2 = hash_unit(i as i64, 2, seed);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// 격자 값 잡음(쌍선형 보간), 결과 [0,1).
fn value_noise(u: f64, v: f64, layer: u64) -> f64 {
    let (iu, iv) = (u.floor(), v.floor());
    let (fu, fv) = (u - iu, v - iv);
    let (su, sv) = (fu * fu * (3.0 - 2.0 * fu), fv * fv * (3.0 - 2.0 * fv));
    let (a, b) = (iu as i64, iv as i64);
    let n00 = hash_unit(a, b, layer);
    let n10 = hash_unit(a + 1, b, layer);
    let n01 = hash_unit(a, b + 1, layer);
    let n11 = hash_unit(a + 1, b + 1, layer);
    let x0 = n00 + (n10 - n00) * su;
    let x1 = n01 + (n11 - n01) * su;
    x0 + (x1 - x0) * sv
}

/// 지형 높이(건물 제외).
pub fn terrain_height(x: f64, y: f64) -> f64 {
    1.2 * (x / 23.0).sin() * (y / 31.0).cos() + 0.6 * (x / 11.0 + y / 17.0).sin()
}

const TERRAIN_MIN: f64 = -1.9;
const TERRAIN_MAX: f64 = 1.9;

/// 위치 (u,v) 의 무늬 색. 여러 축척 잡음을 섞어 특징점이 생기게 한다.
fn albedo(u: f64, v: f64, layer: u64) -> [u8; 3] {
    let n1 = value_noise(u / 0.4, v / 0.4, layer);
    let n2 = value_noise(u / 1.5, v / 1.5, layer + 7);
    let n3 = value_noise(u / 6.0, v / 6.0, layer + 13);
    let g = 0.45 * n1 + 0.35 * n2 + 0.2 * n3;
    let tint = value_noise(u / 9.0, v / 9.0, layer + 29);
    let r = (g * (0.7 + 0.5 * tint)).clamp(0.0, 1.0);
    let gg = g.clamp(0.0, 1.0);
    let b = (g * (1.2 - 0.5 * tint)).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (gg * 255.0) as u8, (b * 255.0) as u8]
}

/// 세계→카메라 회전: 카메라 z 축 = 보는 방향 d, x 축 = d × 위.
fn look_rotation(d: &Vector3<f64>) -> Rotation3<f64> {
    let z = d.normalize();
    let x = z.cross(&Vector3::z()).normalize();
    let y = z.cross(&x);
    Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
        x.transpose(),
        y.transpose(),
        z.transpose(),
    ]))
}

/// 표면 교차 결과.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f64,
    pub point: Point3<f64>,
    pub normal: Vector3<f64>,
    pub rgb: [u8; 3],
}

impl Scene {
    pub fn new(config: SceneConfig) -> Self {
        let length = config.spacing * (config.positions.max(1) - 1) as f64;
        // 경로 양옆과 위에 상자 건물을 흩어 놓는다.
        let mut buildings = Vec::new();
        let n_b = (length / 15.0).ceil() as usize + 2;
        for i in 0..n_b {
            let s = config.seed.wrapping_add(1000 + i as u64);
            let cx = -10.0 + (length + 20.0) * (i as f64 + 0.5) / n_b as f64;
            let side = if i % 2 == 0 { 1.0 } else { -1.0 };
            let cy = side * (8.0 + 18.0 * hash_unit(i as i64, 3, s));
            let hx = 2.5 + 3.0 * hash_unit(i as i64, 4, s);
            let hy = 2.5 + 3.0 * hash_unit(i as i64, 5, s);
            let top = terrain_height(cx, cy) + 4.0 + 8.0 * hash_unit(i as i64, 6, s);
            buildings.push(Building {
                min: Vector2::new(cx - hx, cy - hy),
                max: Vector2::new(cx + hx, cy + hy),
                top,
            });
        }

        let k = Intrinsics::from_hfov(config.width, config.height, config.hfov_deg.to_radians());
        let tilt = config.tilt_deg.to_radians();
        let (c, s) = (tilt.cos(), tilt.sin());
        let mut views = Vec::new();
        let mut rig_centers = Vec::new();
        let mut gps_enu = Vec::new();
        for p in 0..config.positions {
            let rig = Point3::new(config.spacing * p as f64, 0.0, config.altitude);
            rig_centers.push(rig);
            for cam in CamId::ALL {
                // 기체 위 카메라 간격 0.3m.
                let (dir, off) = match cam {
                    CamId::F => (Vector3::new(c, 0.0, -s), Vector3::new(0.3, 0.0, 0.0)),
                    CamId::R => (Vector3::new(0.0, -c, -s), Vector3::new(0.0, -0.3, 0.0)),
                    CamId::L => (Vector3::new(0.0, c, -s), Vector3::new(0.0, 0.3, 0.0)),
                };
                let center = rig + off;
                views.push(View {
                    name: format!("cam{}_{:04}", cam.letter(), p),
                    cam,
                    position: p,
                    camera: Camera {
                        intrinsics: k,
                        pose: Pose::from_center(look_rotation(&dir), &center),
                    },
                });
            }
            let noise = Vector3::new(
                gauss(config.seed, 3 * p as u64),
                gauss(config.seed, 3 * p as u64 + 1),
                gauss(config.seed, 3 * p as u64 + 2),
            ) * config.gps_sigma;
            gps_enu.push(rig + noise);
        }
        Self {
            config,
            buildings,
            views,
            gps_enu,
            rig_centers,
        }
    }

    /// 건물 포함 표면 높이.
    pub fn surface_height(&self, x: f64, y: f64) -> f64 {
        let mut h = terrain_height(x, y);
        for b in &self.buildings {
            if x >= b.min.x && x <= b.max.x && y >= b.min.y && y <= b.max.y {
                h = h.max(b.top);
            }
        }
        h
    }

    /// 광선 o + t·d (t > 0) 와 표면의 첫 교차.
    pub fn intersect(&self, o: &Point3<f64>, d: &Vector3<f64>) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        // 건물: 슬랩 교차. 바닥은 지형 아래(TERRAIN_MIN - 1)까지 내린다.
        for (bi, b) in self.buildings.iter().enumerate() {
            let lo = Vector3::new(b.min.x, b.min.y, TERRAIN_MIN - 1.0);
            let hi = Vector3::new(b.max.x, b.max.y, b.top);
            let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
            let mut axis = 0;
            let mut ok = true;
            for a in 0..3 {
                if d[a].abs() < 1e-15 {
                    if o[a] < lo[a] || o[a] > hi[a] {
                        ok = false;
                        break;
                    }
                    continue;
                }
                let (mut ta, mut tb) = ((lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]);
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                if ta > t0 {
                    t0 = ta;
                    axis = a;
                }
                t1 = t1.min(tb);
            }
            if !ok || t0 > t1 || t0 <= 0.0 {
                continue;
            }
            if best.is_some_and(|h| h.t <= t0) {
                continue;
            }
            let point = o + d * t0;
            let mut normal = Vector3::zeros();
            normal[axis] = -d[axis].signum();
            // 지붕은 (x,y), 벽은 (수평 좌표, z) 로 무늬를 입힌다.
            let rgb = match axis {
                2 => albedo(point.x, point.y, 100 + bi as u64),
                0 => albedo(point.y, point.z, 200 + bi as u64),
                _ => albedo(point.x, point.z, 300 + bi as u64),
            };
            if point.z < terrain_height(point.x, point.y) {
                continue; // 지형에 묻힌 벽 아래쪽
            }
            best = Some(Hit {
                t: t0,
                point,
                normal,
                rgb,
            });
        }

        // 지형: 높이 범위 안에서 전진 후 이분법.
        if d.z < 0.0 {
            let t_enter = ((TERRAIN_MAX - o.z) / d.z).max(0.0);
            let t_exit = (TERRAIN_MIN - o.z) / d.z;
            let t_limit = best.map_or(t_exit, |h| h.t.min(t_exit));
            let f = |t: f64| {
                let p = o + d * t;
                p.z - terrain_height(p.x, p.y)
            };
            let step = 0.2;
            let mut ta = t_enter;
            let mut fa = f(ta);
            while ta < t_limit {
                let tb = (ta + step).min(t_limit);
                let fb = f(tb);
                if fa > 0.0 && fb <= 0.0 {
                    let (mut l, mut r) = (ta, tb);
                    for _ in 0..50 {
                        let m = 0.5 * (l + r);
                        if f(m) > 0.0 {
                            l = m;
                        } else {
                            r = m;
                        }
                    }
                    let t = 0.5 * (l + r);
                    let point = o + d * t;
                    let e = 1e-4;
                    let gx = (terrain_height(point.x + e, point.y)
                        - terrain_height(point.x - e, point.y))
                        / (2.0 * e);
                    let gy = (terrain_height(point.x, point.y + e)
                        - terrain_height(point.x, point.y - e))
                        / (2.0 * e);
                    best = Some(Hit {
                        t,
                        point,
                        normal: Vector3::new(-gx, -gy, 1.0).normalize(),
                        rgb: albedo(point.x, point.y, 1),
                    });
                    break;
                }
                ta = tb;
                fa = fb;
            }
        }
        best
    }

    /// 한 장을 렌더한다: RGB 영상과 정답 깊이(카메라 z, 교차 없으면 NaN).
    pub fn render(&self, view: &View) -> (RgbImage, Vec<f32>) {
        let k = &view.camera.intrinsics;
        let (w, h) = (k.width as usize, k.height as usize);
        let rt = view.camera.pose.rotation.inverse();
        let o = view.camera.pose.center();
        let mut img = RgbImage {
            width: k.width,
            height: k.height,
            data: vec![0; w * h * 3],
        };
        let mut depth = vec![f32::NAN; w * h];
        for y in 0..h {
            for x in 0..w {
                let n = k.to_normalized(&Vector2::new(x as f64 + 0.5, y as f64 + 0.5));
                let dc = Vector3::new(n.x, n.y, 1.0);
                let d = rt * dc;
                let i = y * w + x;
                if let Some(hit) = self.intersect(&o, &d) {
                    // d 의 카메라 z 성분이 1 이므로 t 가 곧 깊이.
                    depth[i] = hit.t as f32;
                    let shade = 0.55
                        + 0.45
                            * hit
                                .normal
                                .dot(&Vector3::new(0.3, 0.2, 0.93).normalize())
                                .max(0.0);
                    for c in 0..3 {
                        img.data[3 * i + c] = (hit.rgb[c] as f64 * shade).min(255.0) as u8;
                    }
                } else {
                    img.data[3 * i..3 * i + 3].copy_from_slice(&[150, 190, 235]);
                }
            }
        }
        (img, depth)
    }
}

/// 단순 RGB 영상(행 우선, 화소당 3바이트).
#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_scene() -> Scene {
        Scene::new(SceneConfig {
            width: 160,
            height: 90,
            ..SceneConfig::default()
        })
    }

    #[test]
    fn view_count_and_names() {
        let s = small_scene();
        assert_eq!(s.views.len(), 240);
        assert_eq!(s.views[0].name, "camF_0000");
        assert_eq!(s.views[239].name, "camL_0079");
    }

    #[test]
    fn rotations_are_proper_and_tilted_down() {
        let s = small_scene();
        for v in &s.views {
            let m = v.camera.pose.rotation.matrix();
            assert!((m * m.transpose() - Matrix3::identity()).norm() < 1e-12);
            assert!((m.determinant() - 1.0).abs() < 1e-12);
            // 광축(세계) 의 z 성분 = -sin(50°).
            let axis = v.camera.pose.rotation.inverse() * Vector3::z();
            assert!((axis.z + 50f64.to_radians().sin()).abs() < 1e-12);
            // 카메라 y(아래) 축은 세계에서 아래쪽 성분을 가진다 (영상이 뒤집히지 않음).
            let down = v.camera.pose.rotation.inverse() * Vector3::y();
            assert!(down.z < 0.0);
        }
    }

    #[test]
    fn gps_noise_magnitude() {
        let s = small_scene();
        let n = s.gps_enu.len() as f64;
        let var: f64 = s
            .gps_enu
            .iter()
            .zip(&s.rig_centers)
            .map(|(g, c)| (g - c).norm_squared())
            .sum::<f64>()
            / (3.0 * n);
        let sigma = var.sqrt();
        // 설정 1.5m, 표본 240개 성분: 1.0~2.0m 안.
        eprintln!("gps_sigma_est {sigma:.3}");
        assert!((1.0..2.0).contains(&sigma), "sigma {sigma}");
    }

    #[test]
    fn rendered_depth_lies_on_surface() {
        let s = small_scene();
        let mut checked = 0;
        let mut worst: f64 = 0.0;
        for vi in [0usize, 1, 2, 121, 239] {
            let v = &s.views[vi];
            let (_, depth) = s.render(v);
            let w = v.camera.intrinsics.width as usize;
            for (i, &d) in depth.iter().enumerate().step_by(7) {
                if !d.is_finite() {
                    continue;
                }
                let px = Vector2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
                let x = v.camera.unproject(&px, d as f64);
                // 지형 위 또는 건물 표면 위 (벽이면 높이가 지형~지붕 사이).
                let terr = terrain_height(x.x, x.y);
                let on_terrain = (x.z - terr).abs();
                let on_building = s.buildings.iter().any(|b| {
                    let inside = x.x >= b.min.x - 1e-3
                        && x.x <= b.max.x + 1e-3
                        && x.y >= b.min.y - 1e-3
                        && x.y <= b.max.y + 1e-3;
                    inside && x.z <= b.top + 1e-3 && x.z >= terr - 1e-3
                });
                if !on_building {
                    worst = worst.max(on_terrain);
                }
                checked += 1;
            }
        }
        eprintln!("surface_checked {checked} worst {worst:.2e}");
        assert!(checked > 5000, "checked {checked}");
        // float32 깊이 저장 오차 포함 1cm 이내.
        assert!(worst < 0.01, "최대 표면 이탈 {worst} m");
    }

    #[test]
    fn surface_point_reprojects_to_its_pixel() {
        // 표면 점을 투영해 깊이 맵에서 같은 깊이를 읽는지 (투영·렌더 규약 일치).
        let s = small_scene();
        let v = &s.views[1]; // R 카메라
        let (_, depth) = s.render(v);
        let k = v.camera.intrinsics;
        let px = Vector2::new(80.5, 45.5);
        let d = depth[45 * k.width as usize + 80] as f64;
        let x = v.camera.unproject(&px, d);
        let back = v.camera.project(&x).unwrap();
        assert!((back - px).norm() < 1e-4);
    }

    #[test]
    fn image_has_texture() {
        let s = small_scene();
        let (img, depth) = s.render(&s.views[0]);
        let valid = depth.iter().filter(|d| d.is_finite()).count();
        assert!(
            valid as f64 > 0.9 * depth.len() as f64,
            "지면이 화면 대부분을 덮어야 함"
        );
        let g: Vec<f64> = img.data.chunks(3).map(|c| c[1] as f64).collect();
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        let std = (g.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / g.len() as f64).sqrt();
        eprintln!(
            "valid_frac {:.4} texture_std {std:.2}",
            valid as f64 / depth.len() as f64
        );
        assert!(std > 15.0, "무늬 표준편차 {std}");
    }

    #[test]
    fn buildings_visible_somewhere() {
        // 적어도 한 장에서 지붕 높이(지형 + 3m 이상) 점이 보여야 한다.
        let s = small_scene();
        let mut found = false;
        for v in s.views.iter().step_by(13) {
            let (_, depth) = s.render(v);
            let w = v.camera.intrinsics.width as usize;
            for (i, &d) in depth.iter().enumerate().step_by(5) {
                if d.is_finite() {
                    let x = v.camera.unproject(
                        &Vector2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5),
                        d as f64,
                    );
                    if x.z > terrain_height(x.x, x.y) + 3.0 {
                        found = true;
                        break;
                    }
                }
            }
            if found {
                break;
            }
        }
        assert!(found);
    }
}
