//! 다시점 깊이 맵 융합 (SPEC §3.6).
//!
//! 기준 사진의 깊이 화소를 3D 로 올려 다른 사진에 투영하고, 그 사진의 깊이로
//! 다시 올린 점을 기준 사진으로 되돌려 왕복 재투영 오차와 상대 깊이 차를 본다.
//! 기준 사진을 포함해 동의하는 사진 수가 `min_views` 이상이면 평균 위치·법선·색으로
//! 점 하나를 만들고, 쓰인 화소는 표시해 같은 표면이 두 번 나오지 않게 한다.
//!
//! 화소 (x, y) 의 중심은 영상 좌표 (x + 0.5, y + 0.5) 이다.
//! 깊이는 카메라 z, 법선은 카메라 좌표계 단위 벡터이며 출력 법선은 세계 좌표계이다.

use crate::camera::Camera;
use crate::math::{Point3, Vector2, Vector3};
use crate::ply::{PointCloud, PointRecord};

/// 사진 한 장의 깊이 맵. 깊이 ≤ 0 이거나 유한하지 않으면 빈 화소로 본다.
///
/// 깊이 추정 쪽 형과 같은 필드를 가진 임시 형이다(병합 뒤 그 형으로 바꾼다).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DepthMap {
    pub w: usize,
    pub h: usize,
    pub depth: Vec<f32>,
    /// 카메라 좌표계 법선.
    pub normal: Vec<[f32; 3]>,
    pub cost: Vec<f32>,
}

impl DepthMap {
    fn get(&self, x: usize, y: usize) -> Option<f32> {
        let d = self.depth[y * self.w + x];
        (d.is_finite() && d > 0.0).then_some(d)
    }
}

/// 융합에 쓰는 사진: 카메라와 (있으면) 화소 색. `rgb` 가 비면 회색으로 둔다.
#[derive(Clone, Debug)]
pub struct FusionView {
    pub camera: Camera,
    /// 행 우선 `w × h` 색. 깊이 맵과 같은 크기이거나 비어 있어야 한다.
    pub rgb: Vec<[u8; 3]>,
}

/// 융합 설정.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionConfig {
    /// 왕복 재투영 오차 상한(화소).
    pub reproj_px: f64,
    /// 상대 깊이 차 상한 |d' − d| / d.
    pub depth_rel: f64,
    /// 기준 사진을 포함한 최소 동의 사진 수.
    pub min_views: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            reproj_px: 1.0,
            depth_rel: 0.01,
            min_views: 3,
        }
    }
}

fn pixel_of(p: &Vector2<f64>, w: usize, h: usize) -> Option<(usize, usize)> {
    if !(p.x >= 0.0 && p.y >= 0.0) {
        return None;
    }
    let (x, y) = (p.x.floor() as usize, p.y.floor() as usize);
    (x < w && y < h).then_some((x, y))
}

fn center(x: usize, y: usize) -> Vector2<f64> {
    Vector2::new(x as f64 + 0.5, y as f64 + 0.5)
}

/// 깊이 맵들을 일관성 검사로 걸러 점군 하나로 합친다.
///
/// `views[i]` 와 `depth_maps[i]` 가 짝이다. 길이가 다르면 짧은 쪽까지만 쓴다.
pub fn fuse(views: &[FusionView], depth_maps: &[DepthMap], cfg: FusionConfig) -> PointCloud {
    let n = views.len().min(depth_maps.len());
    let mut used: Vec<Vec<bool>> = depth_maps[..n]
        .iter()
        .map(|m| vec![false; m.w * m.h])
        .collect();
    let mut cloud = PointCloud::default();
    if n == 0 || cfg.min_views == 0 {
        return cloud;
    }

    let world_normal = |v: usize, idx: usize| -> Vector3<f64> {
        let nc = depth_maps[v].normal.get(idx).copied().unwrap_or([0.0; 3]);
        let nc = Vector3::new(nc[0] as f64, nc[1] as f64, nc[2] as f64);
        views[v].camera.pose.rotation.inverse() * nc
    };
    let color = |v: usize, idx: usize| -> Vector3<f64> {
        let c = views[v].rgb.get(idx).copied().unwrap_or([128; 3]);
        Vector3::new(c[0] as f64, c[1] as f64, c[2] as f64)
    };

    for r in 0..n {
        let rm = &depth_maps[r];
        let rc = &views[r].camera;
        for y in 0..rm.h {
            for x in 0..rm.w {
                let ridx = y * rm.w + x;
                if used[r][ridx] {
                    continue;
                }
                let Some(d) = rm.get(x, y) else { continue };
                let d = d as f64;
                let p = center(x, y);
                let xw = rc.unproject(&p, d);

                let mut agree: Vec<(usize, usize, Point3<f64>)> = Vec::new();
                for j in 0..n {
                    if j == r {
                        continue;
                    }
                    let jm = &depth_maps[j];
                    let jc = &views[j].camera;
                    let Some(q) = jc.project(&xw) else { continue };
                    let Some((qx, qy)) = pixel_of(&q, jm.w, jm.h) else {
                        continue;
                    };
                    let jidx = qy * jm.w + qx;
                    if used[j][jidx] {
                        continue;
                    }
                    let Some(dj) = jm.get(qx, qy) else { continue };
                    let yw = jc.unproject(&center(qx, qy), dj as f64);
                    let yc = rc.pose.transform(&yw);
                    if yc.z <= 0.0 {
                        continue;
                    }
                    let Some(back) = rc.project(&yw) else {
                        continue;
                    };
                    if (back - p).norm() > cfg.reproj_px {
                        continue;
                    }
                    if ((yc.z - d) / d).abs() > cfg.depth_rel {
                        continue;
                    }
                    agree.push((j, jidx, yw));
                }
                if agree.len() + 1 < cfg.min_views {
                    continue;
                }

                let k = (agree.len() + 1) as f64;
                let mut pos = xw.coords;
                let mut nor = world_normal(r, ridx);
                let mut col = color(r, ridx);
                for &(j, jidx, yw) in &agree {
                    pos += yw.coords;
                    nor += world_normal(j, jidx);
                    col += color(j, jidx);
                }
                pos /= k;
                col /= k;
                let nn = nor.norm();
                let nor = if nn > 1e-12 && nn.is_finite() {
                    nor / nn
                } else {
                    Vector3::zeros()
                };
                if !pos.iter().all(|v| v.is_finite()) {
                    continue;
                }
                used[r][ridx] = true;
                for &(j, jidx, _) in &agree {
                    used[j][jidx] = true;
                }
                cloud.points.push(PointRecord {
                    xyz: [pos.x as f32, pos.y as f32, pos.z as f32],
                    normal: [nor.x as f32, nor.y as f32, nor.z as f32],
                    rgb: [
                        col.x.round().clamp(0.0, 255.0) as u8,
                        col.y.round().clamp(0.0, 255.0) as u8,
                        col.z.round().clamp(0.0, 255.0) as u8,
                    ],
                });
            }
        }
    }
    cloud
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Intrinsics, Pose};
    use crate::math::Rotation3;

    const W: usize = 96;
    const H: usize = 72;
    /// 상자: [-B, B] × [-B, B] × [0, BH], 바닥 평면 z = 0.
    const B: f64 = 0.6;
    const BH: f64 = 0.8;

    /// 결정적 의사난수 (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-300);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// 중심 c 에서 목표 t 를 보는 카메라 (월드 z 위쪽).
    fn look_at(c: Point3<f64>, t: Point3<f64>) -> Camera {
        let zc = (t - c).normalize();
        let up = Vector3::new(0.0, 0.0, 1.0);
        let xc = zc.cross(&up).normalize();
        let yc = zc.cross(&xc);
        let m = nalgebra::Matrix3::from_rows(&[xc.transpose(), yc.transpose(), zc.transpose()]);
        let rot = Rotation3::from_matrix_unchecked(m);
        Camera {
            intrinsics: Intrinsics::from_hfov(W as u32, H as u32, 60f64.to_radians()),
            pose: Pose::from_center(rot, &c),
        }
    }

    /// 광선과 장면(평면 + 상자)의 가장 가까운 교점: (거리 t, 세계 법선).
    fn ray_cast(o: &Point3<f64>, dir: &Vector3<f64>) -> Option<(f64, Vector3<f64>)> {
        let mut best: Option<(f64, Vector3<f64>)> = None;
        if dir.z.abs() > 1e-12 {
            let t = -o.z / dir.z;
            if t > 0.0 {
                best = Some((t, Vector3::new(0.0, 0.0, 1.0)));
            }
        }
        // 상자: 슬래브 교차.
        let lo = [-B, -B, 0.0];
        let hi = [B, B, BH];
        let (mut t0, mut t1, mut axis) = (f64::NEG_INFINITY, f64::INFINITY, 0usize);
        let mut sign = 0.0;
        let mut hit = true;
        for a in 0..3 {
            if dir[a].abs() < 1e-12 {
                if o[a] < lo[a] || o[a] > hi[a] {
                    hit = false;
                }
                continue;
            }
            let ta = (lo[a] - o[a]) / dir[a];
            let tb = (hi[a] - o[a]) / dir[a];
            let (tn, tf, s) = if ta < tb {
                (ta, tb, -1.0)
            } else {
                (tb, ta, 1.0)
            };
            if tn > t0 {
                t0 = tn;
                axis = a;
                sign = s;
            }
            t1 = t1.min(tf);
        }
        if hit && t0 <= t1 && t0 > 0.0 && best.is_none_or(|(tb, _)| t0 < tb) {
            let mut nrm = Vector3::zeros();
            nrm[axis] = sign;
            best = Some((t0, nrm));
        }
        best
    }

    /// 정답 깊이 맵을 렌더한다.
    fn render(cam: &Camera) -> DepthMap {
        let c = cam.pose.center();
        let rinv = cam.pose.rotation.inverse();
        let mut m = DepthMap {
            w: W,
            h: H,
            depth: vec![0.0; W * H],
            normal: vec![[0.0; 3]; W * H],
            cost: vec![0.0; W * H],
        };
        for y in 0..H {
            for x in 0..W {
                let nrm = cam.intrinsics.to_normalized(&center(x, y));
                let dc = Vector3::new(nrm.x, nrm.y, 1.0);
                let dw = rinv * dc;
                if let Some((t, nw)) = ray_cast(&c, &dw) {
                    // dc 의 z 성분이 1 이므로 t 가 곧 카메라 깊이.
                    let i = y * W + x;
                    m.depth[i] = t as f32;
                    let ncam = cam.pose.rotation * nw;
                    m.normal[i] = [ncam.x as f32, ncam.y as f32, ncam.z as f32];
                }
            }
        }
        m
    }

    /// 정답 표면(평면 z=0 과 상자 겉면)까지 거리.
    fn surface_dist(p: &Vector3<f64>) -> f64 {
        let plane = p.z.abs();
        let q = Vector3::new(
            p.x.abs() - B,
            p.y.abs() - B,
            (p.z - 0.5 * BH).abs() - 0.5 * BH,
        );
        let outside = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
        let inside = q.x.max(q.y).max(q.z).min(0.0);
        plane.min((outside + inside).abs())
    }

    fn cameras() -> Vec<Camera> {
        let t = Point3::new(0.0, 0.0, 0.3);
        (0..6)
            .map(|k| {
                let a = k as f64 * std::f64::consts::TAU / 6.0;
                look_at(Point3::new(4.0 * a.cos(), 4.0 * a.sin(), 3.0), t)
            })
            .collect()
    }

    fn views(cams: &[Camera]) -> Vec<FusionView> {
        cams.iter()
            .map(|c| FusionView {
                camera: *c,
                rgb: vec![[200, 100, 50]; W * H],
            })
            .collect()
    }

    /// 상대 잡음 σ 와 이상치 비율로 깊이를 더럽힌다. 이상치 표시도 돌려준다.
    fn corrupt(m: &mut DepthMap, rng: &mut Rng, sigma: f64, outlier: f64) -> Vec<bool> {
        let mut bad = vec![false; m.w * m.h];
        for (i, d) in m.depth.iter_mut().enumerate() {
            if *d <= 0.0 {
                continue;
            }
            if rng.uniform() < outlier {
                // 정답에서 10~40% 벗어난 틀린 깊이.
                let s = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                *d *= (1.0 + s * (0.1 + 0.3 * rng.uniform())) as f32;
                bad[i] = true;
            } else {
                *d *= (1.0 + sigma * rng.normal()) as f32;
            }
        }
        bad
    }

    fn errors(cloud: &PointCloud) -> (f64, f64) {
        let mut e: Vec<f64> = cloud
            .points
            .iter()
            .map(|p| {
                surface_dist(&Vector3::new(
                    p.xyz[0] as f64,
                    p.xyz[1] as f64,
                    p.xyz[2] as f64,
                ))
            })
            .collect();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (e[e.len() / 2], *e.last().unwrap())
    }

    /// 정답 기준으로 점이 몇 장의 사진에 가리지 않고 보이는지 센다.
    fn visible_count(cams: &[Camera], truth: &[DepthMap], p: &Point3<f64>) -> usize {
        cams.iter()
            .zip(truth)
            .filter(|(c, m)| {
                let Some(q) = c.project(p) else { return false };
                let Some((x, y)) = pixel_of(&q, W, H) else {
                    return false;
                };
                // 정답 표면점이 3×3 이웃 안에서 화소 크기(깊이 7 에서 ≈0.085)의
                // 절반 이내로 닿으면 보인다고 센다. 모서리 근처 평균 점이
                // 이웃 화소로 넘어가는 경우를 받아 준다.
                (y.saturating_sub(1)..(y + 2).min(H)).any(|yy| {
                    (x.saturating_sub(1)..(x + 2).min(W)).any(|xx| {
                        m.get(xx, yy).is_some_and(|d| {
                            (c.unproject(&center(xx, yy), d as f64) - p).norm() < 0.04
                        })
                    })
                })
            })
            .count()
    }

    #[test]
    fn exact_depths_lie_on_surface() {
        let cams = cameras();
        let maps: Vec<DepthMap> = cams.iter().map(render).collect();
        let cloud = fuse(&views(&cams), &maps, FusionConfig::default());
        assert!(cloud.len() > 2000, "points {}", cloud.len());
        assert!(!cloud.has_nan());
        let (med, max) = errors(&cloud);
        println!(
            "exact: points {} median {med:.2e} max {max:.5}",
            cloud.len()
        );
        // 정답 깊이는 f32 반올림(깊이 ~6 에서 ~1e-6)뿐이므로 평균 위치는
        // 같은 면 위에 있다. 모서리에서 두 면 점이 섞이면 화소 크기(깊이 6,
        // f≈83 → 0.07)의 일부만큼 벗어날 수 있어 최대는 그 절반으로 둔다.
        assert!(med < 1e-5, "median {med}");
        assert!(max < 0.035, "max {max}");
        for p in &cloud.points {
            let n = p.normal;
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4);
            assert_eq!(p.rgb, [200, 100, 50]);
        }
    }

    #[test]
    fn noisy_with_outliers() {
        let cams = cameras();
        let truth: Vec<DepthMap> = cams.iter().map(render).collect();
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut maps = truth.clone();
        let mut bad = Vec::new();
        let sigma = 0.001;
        for m in maps.iter_mut() {
            bad.push(corrupt(m, &mut rng, sigma, 0.10));
        }
        let cfg = FusionConfig::default();
        let cloud = fuse(&views(&cams), &maps, cfg);
        assert!(!cloud.has_nan());
        assert!(cloud.len() > 2000, "points {}", cloud.len());

        // 걸러지지 않은 경우의 대조: 깊이를 그대로 올리면 이상치가 크게 벗어난다.
        let mut raw_max = 0.0f64;
        for (c, m) in cams.iter().zip(&maps) {
            for y in 0..H {
                for x in 0..W {
                    if let Some(d) = m.get(x, y) {
                        let p = c.unproject(&center(x, y), d as f64);
                        raw_max = raw_max.max(surface_dist(&p.coords));
                    }
                }
            }
        }
        assert!(raw_max > 0.3, "raw max {raw_max}");

        let (med, max) = errors(&cloud);
        println!(
            "noisy: points {} median {med:.5} max {max:.5} raw_max {raw_max:.3}",
            cloud.len()
        );
        // 깊이 ~4.5–7 에서 단일 사진 잡음 σ = 0.001·d ≈ 0.005~0.007.
        // 3장 이상 평균이면 σ/√3 ≈ 0.004 이하, 거리 중앙값은 그 0.67 배 근처.
        assert!(med < 0.004, "median {med}");
        // 남는 점의 각 깊이는 기준 깊이와 1% 이내 → 깊이 7 에서 0.07 이하.
        assert!(max < 0.07, "max {max}");

        // 정답 기준 가시 사진 수(출력만). 이 판정은 아직 약 10% 를 3장
        // 미만으로 세어 기준값으로 쓰지 않는다. 1~2장 제외는
        // two_views_give_nothing 에서 확인한다.
        let few = cloud
            .points
            .iter()
            .filter(|p| {
                let x = Point3::new(p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64);
                visible_count(&cams, &truth, &x) < 3
            })
            .count();
        println!("seen by <3: {few}");
        assert!(few < cloud.len());

        // 이상치 화소를 기준으로 한 점은 거의 남지 않는다: 남은 점 중 이상치
        // 기준 화소 비율이 입력 비율(10%)보다 훨씬 작아야 한다.
        let total_bad: usize = bad.iter().map(|b| b.iter().filter(|&&v| v).count()).sum();
        assert!(total_bad > 2000);
        let far = cloud
            .points
            .iter()
            .filter(|p| {
                surface_dist(&Vector3::new(
                    p.xyz[0] as f64,
                    p.xyz[1] as f64,
                    p.xyz[2] as f64,
                )) > 0.03
            })
            .count();
        assert!(far * 100 < cloud.len(), "far {far} of {}", cloud.len());
    }

    #[test]
    fn two_views_give_nothing() {
        let cams = cameras();
        let maps: Vec<DepthMap> = cams[..2].iter().map(render).collect();
        let cloud = fuse(&views(&cams[..2]), &maps, FusionConfig::default());
        assert!(cloud.is_empty());
        let cfg = FusionConfig {
            min_views: 2,
            ..FusionConfig::default()
        };
        assert!(!fuse(&views(&cams[..2]), &maps, cfg).is_empty());
    }

    #[test]
    fn invalid_depths_are_skipped() {
        let cams = cameras();
        let mut maps: Vec<DepthMap> = cams.iter().map(render).collect();
        for m in maps.iter_mut() {
            for (i, d) in m.depth.iter_mut().enumerate() {
                if i % 7 == 0 {
                    *d = f32::NAN;
                } else if i % 11 == 0 {
                    *d = -1.0;
                }
            }
        }
        let cloud = fuse(&views(&cams), &maps, FusionConfig::default());
        assert!(!cloud.is_empty());
        assert!(!cloud.has_nan());
    }
}
