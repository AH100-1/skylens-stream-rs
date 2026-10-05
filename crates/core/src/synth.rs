//! 합성 검증 장면 (SPEC §6).
//!
//! 지면 = 완만한 높낮이 + 상자 건물, 표면 색은 절차적 무늬. 드론 3대가 고도 30m 에서 편대로
//! +x 방향 직선 비행하고, 드론마다 카메라 1대(F, R, L)가 아래로 기울어져 있다.
//! 기본값은 SPEC §1 실측 배치(편대 간격 약 10 m, 방향 F −3°·R +125°·L −116°, 기울기 60°,
//! 화각 65°, 위치 간 1.0 m). 예전 쉬운 배치(한 기체, ±90°, 2.5 m)는 [`SceneConfig::easy`].
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
    /// 위치 간 이동(m, +x 방향).
    pub spacing: f64,
    pub altitude: f64,
    pub width: u32,
    pub height: u32,
    pub hfov_deg: f64,
    /// 카메라별 내려다보는 각(수평 아래, 도). 순서는 [`CamId::ALL`] (F, R, L).
    pub tilt_deg: [f64; 3],
    /// 카메라별 방위각(도): 진행 방향(+x) 기준 수평 각, 오른쪽(−y)이 +.
    pub heading_deg: [f64; 3],
    /// 카메라(드론)별 편대 중심 기준 위치 (x 앞, y 왼쪽, z 위; m).
    pub offsets: [[f64; 3]; 3],
    /// GPS 잡음 표준편차(m, 축마다).
    pub gps_sigma: f64,
    /// 기체(카메라)별 GPS 치우침 표준편차(m, 축마다). 기체마다 한 번 뽑아 모든 위치에 같은 값을 더한다(0 이면 없음).
    pub gps_bias_sigma: f64,
    /// 사진마다 광축 둘레로 섞는 롤 잡음 표준편차(도). 0 이면 없음(짐벌 가정이 그대로 맞는다).
    pub roll_sigma_deg: f64,
    pub seed: u64,
}

/// SPEC §1 실측 편대 간격(F–R 9.8, F–L 10.6, R–L 10.5 m)을 만족하는 수평 삼각형,
/// 무게중심을 원점으로 둔 좌표 (x 앞, y 왼쪽).
///
/// 가정: 거리 세 개는 삼각형의 모양만 정하고 진행 방향에 대한 놓임은 정하지 않는다.
/// 여기서는 (1) 세 드론이 같은 고도, (2) R–L 변이 진행 방향에 수직, (3) F 가 그 변의 앞쪽에
/// 있다고 둔다. 그러면 R = (0, −h), L = (0, +h), h = R–L/2 = 5.25 m 이고
/// F 의 y = (FR² − FL²)/(4h) = −0.777 m, x = √(FR² − (y + h)²) = 8.720 m 이다(R–L 변 기준).
/// 실측 상대 위치가 생기면 이 가정을 그것으로 바꾼다.
fn formation_offsets() -> [[f64; 3]; 3] {
    let (d_fr, d_fl, d_rl) = (9.8f64, 10.6f64, 10.5f64);
    // R = (0, −d_rl/2), L = (0, +d_rl/2), F = (fx, fy).
    let h = d_rl / 2.0;
    let fy = (d_fr * d_fr - d_fl * d_fl) / (4.0 * h);
    let fx = (d_fr * d_fr - (fy + h) * (fy + h)).sqrt();
    let (gx, gy) = (fx / 3.0, fy / 3.0);
    [
        [fx - gx, fy - gy, 0.0],
        [-gx, -h - gy, 0.0],
        [-gx, h - gy, 0.0],
    ]
}

impl Default for SceneConfig {
    /// SPEC §1·§6 실측 편대 배치.
    fn default() -> Self {
        Self {
            positions: 80,
            spacing: 1.0,
            altitude: 30.0,
            width: 960,
            height: 540,
            hfov_deg: 65.0,
            tilt_deg: [60.0; 3],
            heading_deg: [-3.0, 125.0, -116.0],
            offsets: formation_offsets(),
            gps_sigma: 1.5,
            gps_bias_sigma: 0.0,
            roll_sigma_deg: 0.0,
            seed: 1,
        }
    }
}

impl SceneConfig {
    /// 예전 쉬운 배치: 한 기체에 카메라 3대(0.3 m 간격), F 정면·R/L ±90°,
    /// 위치 간 2.5 m, 기울기 50°, 화각 70°. 실측보다 시차가 커서 낙관적이다.
    pub fn easy() -> Self {
        Self {
            spacing: 2.5,
            hfov_deg: 70.0,
            tilt_deg: [50.0; 3],
            heading_deg: [0.0, 90.0, -90.0],
            offsets: [[0.3, 0.0, 0.0], [0.0, -0.3, 0.0], [0.0, 0.3, 0.0]],
            ..Self::default()
        }
    }

    /// 설정 검사. 기울기는 (0°, 90°) 열린 구간만 받는다: 90° 에서는 보는 방향이 연직이라
    /// 카메라 x 축(보는 방향 × 위)이 정해지지 않고(NaN), 0° 이하는 지면을 보지 않는다.
    pub fn validate(&self) -> Result<(), String> {
        for (cam, t) in CamId::ALL.iter().zip(self.tilt_deg) {
            if !(t > 0.0 && t < 90.0) {
                return Err(format!("{cam:?} 기울기 {t}° 는 (0, 90) 밖"));
            }
        }
        if !(self.hfov_deg > 0.0 && self.hfov_deg < 180.0) {
            return Err(format!("수평 화각 {}° 는 (0, 180) 밖", self.hfov_deg));
        }
        if !(self.spacing.is_finite() && self.altitude.is_finite() && self.altitude > 0.0) {
            return Err("위치 간 이동·고도가 유한하지 않거나 고도 ≤ 0".into());
        }
        if self.width == 0 || self.height == 0 {
            return Err("영상 크기 0".into());
        }
        Ok(())
    }

    /// 카메라 `cam` 의 세계 보는 방향(단위 벡터).
    pub fn view_dir(&self, cam: CamId) -> Vector3<f64> {
        let i = cam as usize;
        let (a, t) = (
            self.heading_deg[i].to_radians(),
            self.tilt_deg[i].to_radians(),
        );
        Vector3::new(t.cos() * a.cos(), -t.cos() * a.sin(), -t.sin())
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
    /// 장마다(`views` 와 같은 순서) GPS (동-북-위, 미터, [`GPS_ORIGIN`] 기준):
    /// 그 카메라를 단 드론의 정답 중심 + 잡음. 드론이 3대이므로 위치가 아니라 장 단위다.
    pub gps_enu: Vec<Point3<f64>>,
    /// 위치별 정답 편대 중심(세 드론 무게중심).
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
    /// 장면 생성. 설정이 잘못되면([`SceneConfig::validate`]) 멈춘다; 오류로 받으려면 [`Scene::try_new`].
    pub fn new(config: SceneConfig) -> Self {
        match Self::try_new(config) {
            Ok(s) => s,
            Err(e) => panic!("잘못된 장면 설정: {e}"),
        }
    }

    /// 설정을 검사한 뒤 장면 생성.
    pub fn try_new(config: SceneConfig) -> Result<Self, String> {
        config.validate()?;
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
        let mut views = Vec::new();
        let mut rig_centers = Vec::new();
        let mut gps_enu = Vec::new();
        for p in 0..config.positions {
            let rig = Point3::new(config.spacing * p as f64, 0.0, config.altitude);
            rig_centers.push(rig);
            for cam in CamId::ALL {
                let dir = config.view_dir(cam);
                let o = config.offsets[cam as usize];
                let center = rig + Vector3::new(o[0], o[1], o[2]);
                let i = views.len() as u64;
                let noise = Vector3::new(
                    gauss(config.seed, 3 * i),
                    gauss(config.seed, 3 * i + 1),
                    gauss(config.seed, 3 * i + 2),
                ) * config.gps_sigma;
                // 기체별 고정 치우침: 난수 열 번호를 위치 잡음과 겹치지 않게 큰 값에서 시작.
                let b = (1u64 << 40) + 3 * cam as u64;
                let bias = Vector3::new(
                    gauss(config.seed, b),
                    gauss(config.seed, b + 1),
                    gauss(config.seed, b + 2),
                ) * config.gps_bias_sigma;
                gps_enu.push(center + noise + bias);
                // 광축 둘레 롤 잡음: 난수 열 번호는 위 두 열과 겹치지 않게 더 큰 값에서 시작.
                let roll = gauss(config.seed, (1u64 << 41) + i) * config.roll_sigma_deg;
                let roll_rot = Rotation3::from_axis_angle(&Vector3::z_axis(), roll.to_radians());
                views.push(View {
                    name: format!("cam{}_{:04}", cam.letter(), p),
                    cam,
                    position: p,
                    camera: Camera {
                        intrinsics: k,
                        pose: Pose::from_center(roll_rot * look_rotation(&dir), &center),
                    },
                });
            }
        }
        Ok(Self {
            config,
            buildings,
            views,
            gps_enu,
            rig_centers,
        })
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
        use rayon::prelude::*;
        let mut data = vec![0u8; w * h * 3];
        let mut depth = vec![f32::NAN; w * h];
        data.par_chunks_mut(w * 3)
            .zip(depth.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (row, drow))| {
                for x in 0..w {
                    let n = k.to_normalized(&Vector2::new(x as f64 + 0.5, y as f64 + 0.5));
                    let d = rt * Vector3::new(n.x, n.y, 1.0);
                    if let Some(hit) = self.intersect(&o, &d) {
                        // d 의 카메라 z 성분이 1 이므로 t 가 곧 깊이.
                        drow[x] = hit.t as f32;
                        let light = Vector3::new(0.3, 0.2, 0.93).normalize();
                        let shade = 0.55 + 0.45 * hit.normal.dot(&light).max(0.0);
                        for c in 0..3 {
                            row[3 * x + c] = (hit.rgb[c] as f64 * shade).min(255.0) as u8;
                        }
                    } else {
                        row[3 * x..3 * x + 3].copy_from_slice(&[150, 190, 235]);
                    }
                }
            });
        let img = RgbImage {
            width: k.width,
            height: k.height,
            data,
        };
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
            // 광축(세계) 의 z 성분 = -sin(60°).
            let axis = v.camera.pose.rotation.inverse() * Vector3::z();
            assert!((axis.z + 60f64.to_radians().sin()).abs() < 1e-12);
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
            .zip(&s.views)
            .map(|(g, v)| (g - v.camera.pose.center()).norm_squared())
            .sum::<f64>()
            / (3.0 * n);
        let sigma = var.sqrt();
        // 설정 1.5m, 표본 720개 성분(장마다 드론 GPS): 1.0~2.0m 안.
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

/// 기준 GPS 원점(임의의 위경도). 장면 ENU 원점이 여기다.
pub const GPS_ORIGIN: crate::geo::Geodetic = crate::geo::Geodetic {
    lat_deg: 37.5,
    lon_deg: 127.0,
    alt: 50.0,
};

impl Scene {
    /// SPEC §2 출력 원점: 첫 장(`views[0]`)의 GPS 위경도·고도.
    pub fn first_gps_origin(&self) -> crate::geo::Geodetic {
        crate::geo::enu_to_geodetic(&self.gps_enu[0].coords, &GPS_ORIGIN)
    }

    /// 정답 좌표([`GPS_ORIGIN`] 기준 동-북-위)를 첫 GPS 기준 동-북-위로 옮긴다.
    pub fn to_first_gps_frame(&self, p: &Point3<f64>) -> Point3<f64> {
        let g = crate::geo::enu_to_geodetic(&p.coords, &GPS_ORIGIN);
        Point3::from(crate::geo::geodetic_to_enu(&g, &self.first_gps_origin()))
    }

    /// SPEC §1 입력 형식으로 폴더에 쓴다:
    /// `images/cam{F,R,L}_{번호:04}.jpg`, `gps.txt`(이름 위도 경도 고도),
    /// 정답 `truth/cameras.txt`(이름 fx fy cx cy w h, R 행 우선 9개, t 3개),
    /// 정답 원점 `truth/origin.txt`(`위도 경도 고도` 한 줄 = [`GPS_ORIGIN`]).
    /// 정답 좌표의 원점은 [`GPS_ORIGIN`] 이고 SPEC §2 출력 원점(첫 GPS)과 다르다;
    /// 비교할 때는 [`Scene::to_first_gps_frame`] 로 옮긴다.
    pub fn write_dataset(&self, dir: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let img_dir = dir.join("images");
        let truth_dir = dir.join("truth");
        std::fs::create_dir_all(&img_dir)?;
        std::fs::create_dir_all(&truth_dir)?;
        let mut gps = std::fs::File::create(dir.join("gps.txt"))?;
        let mut cams = std::fs::File::create(truth_dir.join("cameras.txt"))?;
        writeln!(
            std::fs::File::create(truth_dir.join("origin.txt"))?,
            "{:.9} {:.9} {:.3}",
            GPS_ORIGIN.lat_deg,
            GPS_ORIGIN.lon_deg,
            GPS_ORIGIN.alt
        )?;
        for (vi, v) in self.views.iter().enumerate() {
            let (img, _) = self.render(v);
            image::RgbImage::from_raw(img.width, img.height, img.data)
                .expect("버퍼 크기")
                .save(img_dir.join(format!("{}.jpg", v.name)))
                .map_err(std::io::Error::other)?;
            let g = crate::geo::enu_to_geodetic(&self.gps_enu[vi].coords, &GPS_ORIGIN);
            writeln!(
                gps,
                "{} {:.9} {:.9} {:.3}",
                v.name, g.lat_deg, g.lon_deg, g.alt
            )?;
            let k = &v.camera.intrinsics;
            let r = v.camera.pose.rotation.matrix();
            let t = v.camera.pose.translation;
            write!(
                cams,
                "{} {} {} {} {} {} {}",
                v.name, k.fx, k.fy, k.cx, k.cy, k.width, k.height
            )?;
            for i in 0..3 {
                for j in 0..3 {
                    write!(cams, " {}", r[(i, j)])?;
                }
            }
            writeln!(cams, " {} {} {}", t.x, t.y, t.z)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod dataset_tests {
    use super::*;
    use crate::geo::{geodetic_to_enu, Geodetic};

    #[test]
    fn written_gps_converts_back_to_enu() {
        let s = Scene::new(SceneConfig {
            positions: 4,
            width: 64,
            height: 36,
            ..SceneConfig::default()
        });
        let dir = std::env::temp_dir().join(format!("skylens_synth_{}", std::process::id()));
        s.write_dataset(&dir).unwrap();
        let gps = std::fs::read_to_string(dir.join("gps.txt")).unwrap();
        let lines: Vec<&str> = gps.lines().collect();
        assert_eq!(lines.len(), 12);
        let mut worst: f64 = 0.0;
        for (vi, (l, v)) in lines.iter().zip(&s.views).enumerate() {
            let f: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(f[0], v.name);
            let g = Geodetic {
                lat_deg: f[1].parse().unwrap(),
                lon_deg: f[2].parse().unwrap(),
                alt: f[3].parse().unwrap(),
            };
            let e = geodetic_to_enu(&g, &GPS_ORIGIN);
            worst = worst.max((e - s.gps_enu[vi].coords).norm());
        }
        // 위경도 소수 9자리(≈0.1mm) + 고도 1mm 반올림.
        assert!(worst < 2e-3, "GPS 왕복 오차 {worst} m");
        let img = image::open(dir.join("images/camR_0002.jpg")).unwrap();
        assert_eq!((img.width(), img.height()), (64, 36));
        let cams = std::fs::read_to_string(dir.join("truth/cameras.txt")).unwrap();
        assert_eq!(cams.lines().count(), 12);
        let origin = std::fs::read_to_string(dir.join("truth/origin.txt")).unwrap();
        let o: Vec<f64> = origin
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        assert_eq!(
            o,
            vec![GPS_ORIGIN.lat_deg, GPS_ORIGIN.lon_deg, GPS_ORIGIN.alt]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// F-024: 첫 GPS 를 원점으로 옮긴 정답 카메라 중심은 GPS 동-북-위와 잡음 수준으로 맞는다
    /// (원점을 옮기지 않으면 고도 30 m 만큼 어긋난다).
    #[test]
    fn truth_in_first_gps_frame_matches_gps() {
        for seed in [1u64, 2, 3, 4, 5] {
            let s = Scene::new(SceneConfig {
                seed,
                ..SceneConfig::default()
            });
            let o = s.first_gps_origin();
            let mut d: Vec<f64> = s
                .views
                .iter()
                .zip(&s.gps_enu)
                .map(|(v, g)| {
                    let gps_local =
                        geodetic_to_enu(&crate::geo::enu_to_geodetic(&g.coords, &GPS_ORIGIN), &o);
                    (s.to_first_gps_frame(&v.camera.pose.center()).coords - gps_local).norm()
                })
                .collect();
            d.sort_by(f64::total_cmp);
            let med = d[d.len() / 2];
            // 두 쪽 모두 같은 원점으로 옮겼으므로 차이 = 그 장의 GPS 잡음 하나.
            // σ=1.5 m 3축 잡음의 크기 중앙값 ≈ 1.54σ ≈ 2.3 m → 기준 3 m.
            eprintln!("seed {seed} median_truth_vs_gps {med:.3}");
            assert!(med < 3.0, "seed {seed} 중앙값 {med}");
            let raw_z = (s.views[0].camera.pose.center().z - 0.0).abs();
            assert!(raw_z > 29.0, "원점 이동 전에는 고도만큼 어긋나야 함");
        }
    }
}

#[cfg(test)]
mod formation_tests {
    use super::*;

    fn scene() -> Scene {
        Scene::new(SceneConfig {
            width: 160,
            height: 90,
            ..SceneConfig::default()
        })
    }

    fn horiz_heading_deg(v: &View) -> f64 {
        let a = v.camera.pose.rotation.inverse() * Vector3::z();
        // 진행 방향 +x 기준, 오른쪽(−y) 이 +.
        (-a.y).atan2(a.x).to_degrees()
    }

    /// F-029 확인 기준 (1): 편대 간격·위치 간 이동·방위각·기울기가 SPEC §1 실측값.
    #[test]
    fn default_is_measured_formation() {
        let s = scene();
        for p in 0..s.config.positions {
            let c: Vec<Point3<f64>> = (0..3)
                .map(|k| s.views[3 * p + k].camera.pose.center())
                .collect();
            let (fr, fl, rl) = (
                (c[0] - c[1]).norm(),
                (c[0] - c[2]).norm(),
                (c[1] - c[2]).norm(),
            );
            for d in [fr, fl, rl] {
                assert!((9.0..=11.0).contains(&d), "편대 간격 {d}");
            }
            assert!(
                (fr - 9.8).abs() < 1e-9 && (fl - 10.6).abs() < 1e-9 && (rl - 10.5).abs() < 1e-9
            );
            if p > 0 {
                for k in 0..3 {
                    let m = (s.views[3 * p + k].camera.pose.center()
                        - s.views[3 * (p - 1) + k].camera.pose.center())
                    .norm();
                    assert!((0.9..=1.1).contains(&m), "위치 간 이동 {m}");
                }
            }
        }
        for v in &s.views {
            let want = match v.cam {
                CamId::F => 0.0,
                CamId::R => 125.0,
                CamId::L => -116.0,
            };
            let h = horiz_heading_deg(v);
            assert!((h - want).abs() <= 5.0, "{} 방위각 {h}", v.name);
            let a = v.camera.pose.rotation.inverse() * Vector3::z();
            let tilt = (-a.z).asin().to_degrees();
            assert!((tilt - 60.0).abs() <= 3.0, "{} 기울기 {tilt}", v.name);
        }
    }

    /// 평지(z = 0) 화면 중앙 광선의 위치 1칸 삼각측량 각(도), 해석값.
    ///
    /// 카메라 높이 H(고도), 기울기 t, 방위각 a, 기선 b(위치 간 이동, 진행 방향 +x).
    /// 화면 중앙 광선 방향 d = (cos t cos a, −cos t sin a, −sin t) 이 평지에 닿는 거리는
    /// D = H / sin t, 기선과 광선(카메라→점) 사이 각 φ 는 cos φ = cos t cos a.
    /// 점 P 에서 두 카메라 중심을 보는 각은 삼각형 (C, C + b x̂, P) 에서
    ///   θ = atan2(b sin φ, D − b cos φ)  ≈  (b / H) · sin t · √(1 − cos² t cos² a).
    /// H = 30, t = 60°, b = 1.0 에서 정확한 식은 F(a = −3°) 1.454°, R(+125°) 1.571°,
    /// L(−116°) 1.603°. 근사식(b ≪ D)은 1.433°·1.585°·1.614° 로, 차이 b cos φ / D(최대 1.4%)는
    /// 둘째 카메라가 광선 방향으로 앞(F)·뒤(R·L)에 있어 거리가 줄거나 느는 몫이다.
    fn center_angle_flat_deg(cfg: &SceneConfig, cam: CamId) -> f64 {
        let d = cfg.view_dir(cam);
        let big_d = cfg.altitude / (-d.z);
        let cos_phi = d.x;
        let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
        (cfg.spacing * sin_phi)
            .atan2(big_d - cfg.spacing * cos_phi)
            .to_degrees()
    }

    /// 화면 9×5 격자 광선이 표면에 닿는 점마다 위치 p, p+1 같은 카메라 짝의 삼각측량 각(도).
    /// `flat` 이면 표면 대신 평지 z = 0 에 닿는 점을 쓴다(해석 기준 분포).
    fn grid_angles(s: &Scene, k: usize, flat: bool) -> Vec<f64> {
        let mut angles = Vec::new();
        for p in (0..s.config.positions - 1).step_by(7) {
            let (a, b) = (&s.views[3 * p + k], &s.views[3 * (p + 1) + k]);
            let (ca, cb) = (a.camera.pose.center(), b.camera.pose.center());
            let kk = a.camera.intrinsics;
            for gx in 0..9 {
                for gy in 0..5 {
                    let px = Vector2::new(
                        (gx as f64 + 0.5) * kk.width as f64 / 9.0,
                        (gy as f64 + 0.5) * kk.height as f64 / 5.0,
                    );
                    let n = kk.to_normalized(&px);
                    let d = a.camera.pose.rotation.inverse() * Vector3::new(n.x, n.y, 1.0);
                    let hit = if flat {
                        (d.z < 0.0).then(|| ca + d * (-ca.z / d.z))
                    } else {
                        s.intersect(&ca, &d).map(|h| h.point)
                    };
                    if let Some(q) = hit {
                        angles.push((ca - q).angle(&(cb - q)).to_degrees());
                    }
                }
            }
        }
        angles.sort_by(f64::total_cmp);
        angles
    }

    /// 표면 격자 중앙값과 평지 격자 중앙값(같은 광선을 z = 0 평지에 쏜 해석 분포)의 허용 폭(도).
    /// θ ∝ 1/D 이므로 표면 높이가 Δz 바뀌면 θ 는 약 θ·Δz/(H − Δz) 바뀐다. 지형 높낮이
    /// ±1.9 m([`TERRAIN_MIN`]·[`TERRAIN_MAX`])에서 θ ≈ 1.5° 면 1.5·1.9/28.1 = 0.10°.
    /// 건물 지붕은 격자 점의 일부라 중앙값을 조금만 민다.
    const TRI_TOL_FLAT_DEG: f64 = 0.1;

    /// 화면 중앙 해석값과 측정 중앙값의 허용 폭(도). 격자 광선은 화면 위(먼 점)·아래(가까운 점)로
    /// 거리가 달라 평지 격자 중앙값이 중앙 광선 값보다 F 0.05°·R 0.14°·L 0.17° 작다(시험 출력
    /// `flat_grid_median`). 여기에 지형 몫 ±0.10° 가 붙는데 실제로는 지붕·높은 지형이 각을 키우는
    /// 쪽이라 중앙 광선 쪽으로 돌아온다. 0.2° 는 두 몫 중 큰 쪽(0.17°)에 여유를 둔 값이다.
    /// 장면이 잘못 만들어지면(기선 2.5 m 면 θ 가 2.5 배, 고도가 10 m 틀리면 약 30%) 이 폭을 넘는다.
    const TRI_TOL_DEG: f64 = 0.2;

    /// F-029 확인 기준 (2)·F-115: 위치 1칸 같은 카메라 짝의 삼각측량 각 중앙값이
    /// 화면 중앙 해석값([`center_angle_flat_deg`]) ± [`TRI_TOL_DEG`] 안. 시드 1~5 모두.
    /// 해석 범위: F 1.25~1.65°, R 1.37~1.77°, L 1.40~1.80°. 함께 평지 격자 중앙값 ± [`TRI_TOL_FLAT_DEG`].
    #[test]
    fn one_step_triangulation_angle_is_small() {
        for seed in 1u64..=5 {
            let s = Scene::new(SceneConfig {
                width: 160,
                height: 90,
                seed,
                ..SceneConfig::default()
            });
            for (k, cam) in CamId::ALL.into_iter().enumerate() {
                let want = center_angle_flat_deg(&s.config, cam);
                let angles = grid_angles(&s, k, false);
                let flat = grid_angles(&s, k, true);
                let med = angles[angles.len() / 2];
                eprintln!(
                    "seed={seed} {cam:?} tri_angle n={} min={:.3} median={med:.3} max={:.3} center_analytic={want:.3} flat_grid_median={:.3}",
                    angles.len(),
                    angles[0],
                    angles[angles.len() - 1],
                    flat[flat.len() / 2]
                );
                assert!(
                    (med - want).abs() <= TRI_TOL_DEG,
                    "시드 {seed} {cam:?} 중앙 삼각측량 각 {med} (해석 {want})"
                );
                let flat_med = flat[flat.len() / 2];
                assert!(
                    (med - flat_med).abs() <= TRI_TOL_FLAT_DEG,
                    "시드 {seed} {cam:?} 중앙 삼각측량 각 {med} (평지 격자 {flat_med})"
                );
            }
        }
        // 해석값 자체를 숫자로 고정한다(식이 바뀌면 드러나게).
        let cfg = SceneConfig::default();
        for (cam, want) in [(CamId::F, 1.454), (CamId::R, 1.571), (CamId::L, 1.603)] {
            let got = center_angle_flat_deg(&cfg, cam);
            assert!((got - want).abs() < 1e-3, "{cam:?} 해석 {got}");
            // 근사식과는 1.5% 안.
            let d = cfg.view_dir(cam);
            let approx =
                (cfg.spacing / cfg.altitude * (-d.z) * (1.0 - d.x * d.x).sqrt()).to_degrees();
            assert!((got - approx).abs() / got < 0.015, "{cam:?} 근사 {approx}");
        }
        let easy = Scene::new(SceneConfig {
            width: 160,
            height: 90,
            ..SceneConfig::easy()
        });
        let (a, b) = (&easy.views[0], &easy.views[3]);
        let ca = a.camera.pose.center();
        let hit = easy
            .intersect(&ca, &(a.camera.pose.rotation.inverse() * Vector3::z()))
            .unwrap();
        let easy_ang = (ca - hit.point)
            .angle(&(b.camera.pose.center() - hit.point))
            .to_degrees();
        eprintln!(
            "easy_F center tri_angle {easy_ang:.3} analytic {:.3}",
            center_angle_flat_deg(&easy.config, CamId::F)
        );
        assert!(easy_ang > 2.0);
    }

    /// F-116: 기울기 (0°, 90°) 밖은 오류.
    #[test]
    fn invalid_tilt_is_rejected() {
        for t in [90.0, 0.0, -10.0, 120.0, f64::NAN] {
            let cfg = SceneConfig {
                positions: 2,
                width: 32,
                height: 18,
                tilt_deg: [t; 3],
                ..SceneConfig::default()
            };
            assert!(cfg.validate().is_err(), "기울기 {t}");
            assert!(Scene::try_new(cfg).is_err(), "기울기 {t}");
        }
        let one = SceneConfig {
            positions: 2,
            width: 32,
            height: 18,
            tilt_deg: [60.0, 90.0, 60.0],
            ..SceneConfig::default()
        };
        assert!(Scene::try_new(one).is_err());
        let ok = SceneConfig {
            positions: 2,
            width: 32,
            height: 18,
            tilt_deg: [89.0; 3],
            ..SceneConfig::default()
        };
        let s = Scene::try_new(ok).unwrap();
        assert!(s.views.iter().all(|v| v
            .camera
            .pose
            .rotation
            .matrix()
            .iter()
            .all(|x| x.is_finite())));
        assert!(std::panic::catch_unwind(|| Scene::new(SceneConfig {
            positions: 2,
            width: 32,
            height: 18,
            tilt_deg: [90.0; 3],
            ..SceneConfig::default()
        }))
        .is_err());
    }

    /// F-029·F-116: 편대 배치 수치를 정확히 단언한다. 거리 F–R 9.8·F–L 10.6·R–L 10.5 m,
    /// 위치 간 1.0 m, 방위각 −3°/125°/−116°, 기울기 60°, 같은 고도, 편대 놓임 가정
    /// (R–L 변이 진행 방향에 수직, F 가 그 변에서 8.720 m 앞).
    #[test]
    fn formation_exact_values() {
        let s = Scene::new(SceneConfig {
            positions: 5,
            width: 64,
            height: 36,
            ..SceneConfig::default()
        });
        for p in 0..s.config.positions {
            let c: Vec<Point3<f64>> = (0..3)
                .map(|k| s.views[3 * p + k].camera.pose.center())
                .collect();
            assert!(((c[0] - c[1]).norm() - 9.8).abs() < 1e-9);
            assert!(((c[0] - c[2]).norm() - 10.6).abs() < 1e-9);
            assert!(((c[1] - c[2]).norm() - 10.5).abs() < 1e-9);
            assert!(c.iter().all(|q| (q.z - 30.0).abs() < 1e-12), "같은 고도");
            assert!((c[1].x - c[2].x).abs() < 1e-12, "R–L 변이 진행 방향에 수직");
            assert!(
                ((c[0].x - c[1].x) - 8.720).abs() < 1e-3,
                "F 앞 {}",
                c[0].x - c[1].x
            );
            let g = (c[0].coords + c[1].coords + c[2].coords) / 3.0;
            assert!((g - s.rig_centers[p].coords).norm() < 1e-12);
            if p > 0 {
                for k in 0..3 {
                    let m = s.views[3 * p + k].camera.pose.center()
                        - s.views[3 * (p - 1) + k].camera.pose.center();
                    assert!(
                        (m - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12,
                        "위치 간 {m:?}"
                    );
                }
            }
        }
        for v in &s.views {
            let want = match v.cam {
                CamId::F => -3.0,
                CamId::R => 125.0,
                CamId::L => -116.0,
            };
            assert!(
                (horiz_heading_deg(v) - want).abs() < 1e-9,
                "{} 방위각",
                v.name
            );
            let a = v.camera.pose.rotation.inverse() * Vector3::z();
            assert!(((-a.z).asin().to_degrees() - 60.0).abs() < 1e-9);
        }
    }

    #[test]
    fn easy_layout_is_kept() {
        let s = Scene::new(SceneConfig {
            positions: 2,
            width: 64,
            height: 36,
            ..SceneConfig::easy()
        });
        let d = (s.views[0].camera.pose.center() - s.views[1].camera.pose.center()).norm();
        assert!((d - 0.3f64.hypot(0.3)).abs() < 1e-12);
        assert!((horiz_heading_deg(&s.views[1]) - 90.0).abs() < 1e-9);
    }
}
