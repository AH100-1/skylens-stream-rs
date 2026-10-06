//! 밀집 깊이 준비: 사진마다 이웃 사진 고르기와 깊이 탐색 범위.
//!
//! 이웃 점수(사진 i 에 대한 후보 j): 두 사진이 함께 본 희소 점 X 마다
//! `w_angle(θ) · w_scale(ρ)` 를 더한다.
//! - θ: X 에서 두 카메라 중심으로 가는 광선 사이 각. 목표 θ₀ = 10° 근처에서 1,
//!   θ < θ₀ 이면 폭 σ = 5°, θ > θ₀ 이면 σ = 15° 인 가우스 꼴로 줄어든다
//!   (작은 각은 깊이 분해능이 나쁘고 큰 각은 겉모습이 달라 정합이 어렵다).
//! - ρ: 두 사진에서 X 한 점이 차지하는 화소 크기의 비. 화소 크기는 **카메라 z(광축 방향 깊이) / 초점 거리**
//!   로 정한다(광선 길이가 아니다). 핀홀에서 X 근처 정면 평면 조각의 화소 축척이 z/f 이기 때문이다.
//!   `w_scale = (min/max)²` 이라 축척이 같을 때 1, 깊이가 2배면 0.25.
//!
//! 점수 상위 k 장(기본 [`DEFAULT_NEIGHBORS`] = 8)을 고른다. 점수 0 인 후보는 빼므로
//! 공유 점이 적으면 k 장보다 적을 수 있다. 좌표가 유한하지 않은 점이나 유한하지 않은 쌍 점수는 0 으로 친다.
//!
//! 깊이 범위는 [`try_depth_range`] 가 관측 점이 없으면 None 을 돌려주고, [`dense_jobs`] 는 그런 사진을
//! 밀집 깊이 작업에서 뺀다.
use std::collections::HashMap;

use crate::camera::Camera;
use crate::math::Point3;

/// 사진마다 고를 이웃 수 기본값.
pub const DEFAULT_NEIGHBORS: usize = 8;
/// 목표 광선 사이 각(도).
pub const TARGET_ANGLE_DEG: f64 = 10.0;
/// 목표보다 작은 각 쪽 감점 폭(도).
pub const SIGMA_BELOW_DEG: f64 = 5.0;
/// 목표보다 큰 각 쪽 감점 폭(도).
pub const SIGMA_ABOVE_DEG: f64 = 15.0;
/// 깊이 범위 분위(아래, 위).
pub const DEPTH_QUANTILES: (f64, f64) = (0.05, 0.95);
/// 깊이 범위 여유: 가까운 끝 × (1 − m), 먼 끝 × (1 + m).
pub const DEPTH_MARGIN: f64 = 0.1;

/// 등록된 사진 한 장.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// 왜곡 없는 핀홀 내부 파라미터와 세계→카메라 자세.
    pub cam: Camera,
    /// 사진 식별자. [`SparsePoint::observers`] 가 이 값을 가리킨다.
    pub id: usize,
}

/// 희소 복원 점 하나.
#[derive(Clone, Debug, PartialEq)]
pub struct SparsePoint {
    /// 세계 좌표.
    pub xyz: Point3<f64>,
    /// 이 점을 관측한 사진들의 [`View::id`].
    pub observers: Vec<usize>,
}

/// 광선 사이 각(라디안)에 대한 가중치.
pub fn angle_weight(theta: f64) -> f64 {
    let t0 = TARGET_ANGLE_DEG.to_radians();
    let s = if theta < t0 {
        SIGMA_BELOW_DEG
    } else {
        SIGMA_ABOVE_DEG
    }
    .to_radians();
    let d = (theta - t0) / s;
    (-0.5 * d * d).exp()
}

/// 축척 비 가중치: 두 화소 크기 a, b(> 0)에 대해 (min/max)².
pub fn scale_weight(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    let r = a.min(b) / a.max(b);
    r * r
}

/// 점 X 의 사진 안 화소 크기(카메라 z / 평균 초점 거리). 카메라 뒤면 None.
fn footprint(view: &View, x: &Point3<f64>) -> Option<f64> {
    let z = view.cam.pose.transform(x).z;
    let k = &view.cam.intrinsics;
    let f = z / (0.5 * (k.fx + k.fy));
    (z > 0.0 && f.is_finite()).then_some(f)
}

/// 두 사진이 함께 본 점 하나의 쌍 점수.
pub fn pair_score(a: &View, b: &View, x: &Point3<f64>) -> f64 {
    let (Some(fa), Some(fb)) = (footprint(a, x), footprint(b, x)) else {
        return 0.0;
    };
    let ra = a.cam.pose.center() - x;
    let rb = b.cam.pose.center() - x;
    let c = (ra.dot(&rb) / (ra.norm() * rb.norm())).clamp(-1.0, 1.0);
    let s = angle_weight(c.acos()) * scale_weight(fa, fb);
    if s.is_finite() {
        s
    } else {
        0.0
    }
}

/// 사진마다 이웃 k 장을 고른다.
///
/// 반환값 `out[i]` 는 `views[i]` 의 이웃들을 점수 내림차순으로 담은 `views` 안 위치(색인)다.
/// 관측자 id 가 `views` 에 없으면 무시한다.
pub fn select_neighbors(views: &[View], points: &[SparsePoint], k: usize) -> Vec<Vec<usize>> {
    let pos: HashMap<usize, usize> = views.iter().enumerate().map(|(i, v)| (v.id, i)).collect();
    let n = views.len();
    let mut score = vec![0.0f64; n * n];
    let mut obs = Vec::new();
    for p in points {
        if !(p.xyz.x.is_finite() && p.xyz.y.is_finite() && p.xyz.z.is_finite()) {
            continue;
        }
        obs.clear();
        obs.extend(p.observers.iter().filter_map(|id| pos.get(id).copied()));
        obs.sort_unstable();
        obs.dedup();
        for (a, &i) in obs.iter().enumerate() {
            for &j in &obs[a + 1..] {
                let s = pair_score(&views[i], &views[j], &p.xyz);
                score[i * n + j] += s;
                score[j * n + i] += s;
            }
        }
    }
    (0..n)
        .map(|i| {
            let mut cand: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i && score[i * n + j] > 0.0)
                .map(|j| (j, score[i * n + j]))
                .collect();
            cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            cand.truncate(k);
            cand.into_iter().map(|(j, _)| j).collect()
        })
        .collect()
}

/// 정렬된 표본의 선형 보간 분위.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let t = q * (sorted.len() - 1) as f64;
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (t - lo as f64)
}

/// 사진의 깊이 탐색 범위 (가까운 끝, 먼 끝). 깊이는 카메라 z.
///
/// 그 사진(`view.id`)이 관측한 점 중 카메라 앞에 있고 깊이가 유한한 점들에서
/// [`DEPTH_QUANTILES`] 분위를 잡고 [`DEPTH_MARGIN`] 만큼 넓힌다. 그런 점이 없으면 None.
pub fn try_depth_range(view: &View, points: &[SparsePoint]) -> Option<(f64, f64)> {
    try_depth_range_q(view, points, DEPTH_QUANTILES)
}

/// [`try_depth_range`] 에서 분위를 고를 수 있는 형태.
pub fn try_depth_range_q(view: &View, points: &[SparsePoint], q: (f64, f64)) -> Option<(f64, f64)> {
    let mut d: Vec<f64> = points
        .iter()
        .filter(|p| p.observers.contains(&view.id))
        .map(|p| view.cam.pose.transform(&p.xyz).z)
        .filter(|z| z.is_finite() && *z > 0.0)
        .collect();
    if d.is_empty() {
        return None;
    }
    d.sort_by(f64::total_cmp);
    let near = quantile(&d, q.0) * (1.0 - DEPTH_MARGIN);
    let far = quantile(&d, q.1) * (1.0 + DEPTH_MARGIN);
    Some((near, far))
}

/// [`try_depth_range`] 의 튜플 형태. 관측 점이 없으면 (0, 0) 이므로 호출 쪽은
/// `near < far` 를 확인하거나 [`dense_jobs`] 를 쓴다.
pub fn depth_range(view: &View, points: &[SparsePoint]) -> (f64, f64) {
    try_depth_range(view, points).unwrap_or((0.0, 0.0))
}

/// 밀집 깊이 작업 하나: 기준 사진, 이웃, 깊이 범위(모두 `views` 안 색인).
#[derive(Clone, Debug, PartialEq)]
pub struct DenseJob {
    pub view: usize,
    pub neighbors: Vec<usize>,
    pub range: (f64, f64),
}

/// 사진마다 이웃 k 장과 깊이 범위를 묶는다. 깊이 범위가 없거나(관측 없음) 이웃이 없는 사진은 뺀다.
pub fn dense_jobs(views: &[View], points: &[SparsePoint], k: usize) -> Vec<DenseJob> {
    let nb = select_neighbors(views, points, k);
    views
        .iter()
        .zip(nb)
        .enumerate()
        .filter_map(|(i, (v, neighbors))| {
            let range = try_depth_range(v, points)?;
            (!neighbors.is_empty()).then_some(DenseJob {
                view: i,
                neighbors,
                range,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Intrinsics, Pose};
    use crate::math::{Matrix3, Rotation3, Vector3};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// 고도 30 m, x 방향 1 m 간격 직선 비행, 연직 하방 촬영. 지면 점은 고도 0 ± 1 m.
    fn line_flight(n: usize) -> (Vec<View>, Vec<SparsePoint>) {
        let k = Intrinsics::from_hfov(1600, 1200, 70f64.to_radians());
        // 카메라 x = 세계 x, 카메라 y = −세계 y, 카메라 z = −세계 z(아래).
        let r = Rotation3::from_matrix_unchecked(Matrix3::new(
            1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0,
        ));
        let views: Vec<View> = (0..n)
            .map(|i| View {
                cam: Camera {
                    intrinsics: k,
                    pose: Pose::from_center(r, &Point3::new(i as f64, 0.0, 30.0)),
                },
                // id 를 위치와 다르게 두어 대응을 확인한다.
                id: 100 + i,
            })
            .collect();
        let mut rng = Lcg(7);
        let mut pts = Vec::new();
        for _ in 0..3000 {
            let x = Point3::new(
                -25.0 + (n as f64 + 50.0) * rng.next(),
                -25.0 + 50.0 * rng.next(),
                -1.0 + 2.0 * rng.next(),
            );
            let observers = views
                .iter()
                .filter(|v| v.cam.project(&x).is_some_and(|p| k.contains(&p)))
                .map(|v| v.id)
                .collect::<Vec<_>>();
            if observers.len() >= 2 {
                pts.push(SparsePoint { xyz: x, observers });
            }
        }
        (views, pts)
    }

    #[test]
    fn weights_shape() {
        assert!((angle_weight(10f64.to_radians()) - 1.0).abs() < 1e-12);
        assert!(angle_weight(1f64.to_radians()) < 0.25);
        assert!(angle_weight(40f64.to_radians()) < 0.2);
        assert!((scale_weight(2.0, 2.0) - 1.0).abs() < 1e-12);
        assert!((scale_weight(1.0, 2.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn far_baseline_beats_adjacent() {
        let (views, pts) = line_flight(21);
        let nb = select_neighbors(&views, &pts, DEFAULT_NEIGHBORS);
        let c = 10;
        let ranks = |j: usize| nb[c].iter().position(|&x| x == j);
        eprintln!(
            "center neighbors (offsets): {:?}",
            nb[c]
                .iter()
                .map(|&j| j as i64 - c as i64)
                .collect::<Vec<_>>()
        );
        assert_eq!(nb.len(), 21);
        assert_eq!(nb[c].len(), DEFAULT_NEIGHBORS);
        assert!(!nb[c].contains(&c));
        // 8칸(8 m) 이웃은 뽑히고, 1칸(1 m) 이웃(광선 각 약 2° 이하)은 그보다 아래거나 빠진다.
        for (far, near) in [(c - 8, c - 1), (c + 8, c + 1)] {
            let rf = ranks(far).expect("8칸 이웃이 뽑혀야 한다");
            assert!(
                ranks(near).is_none_or(|rn| rn > rf),
                "far {rf} near {:?}",
                ranks(near)
            );
        }
        // 1위는 5~9칸 떨어진 사진(광선 각 약 9~17°).
        let d0 = (nb[c][0] as i64 - c as i64).abs();
        assert!((5..=9).contains(&d0), "1위 간격 {d0}");
    }

    #[test]
    fn depth_range_covers_truth_quantiles() {
        let (mut views, mut pts) = line_flight(21);
        let v = views.remove(10);
        let mut truth: Vec<f64> = pts
            .iter()
            .filter(|p| p.observers.contains(&v.id))
            .map(|p| 30.0 - p.xyz.z)
            .collect();
        truth.sort_by(f64::total_cmp);
        let (q05, q95) = (quantile(&truth, 0.05), quantile(&truth, 0.95));
        // 이상점 1 %: 깊이 300 m 짜리 점이 섞여도 범위는 흔들리지 않아야 한다.
        let n_out = truth.len() / 100;
        for i in 0..n_out {
            pts.push(SparsePoint {
                xyz: Point3::new(10.0 + 0.01 * i as f64, 0.0, -270.0),
                observers: vec![v.id],
            });
        }
        let (near, far) = depth_range(&v, &pts);
        eprintln!("truth q05 {q05:.3} q95 {q95:.3} range {near:.3}..{far:.3}");
        assert!(near <= q05 && far >= q95);
        assert!(near >= 0.85 * q05 && far <= 1.15 * q95);
        assert_eq!(depth_range(&View { id: 999, ..v }, &pts), (0.0, 0.0));
        assert_eq!(try_depth_range(&View { id: 999, ..v }, &pts), None);
    }

    /// F-054: NaN 점 하나를 섞어도 이웃 목록이 같고, 관측 없는 사진은 밀집 작업에서 빠진다.
    #[test]
    fn non_finite_points_and_unobserved_views() {
        let (mut views, mut pts) = line_flight(21);
        let base = select_neighbors(&views, &pts, DEFAULT_NEIGHBORS);
        let all_ids: Vec<usize> = views.iter().map(|v| v.id).collect();
        pts.push(SparsePoint {
            xyz: Point3::new(f64::NAN, 0.0, 0.0),
            observers: all_ids.clone(),
        });
        pts.push(SparsePoint {
            xyz: Point3::new(5.0, f64::INFINITY, 0.0),
            observers: all_ids,
        });
        assert_eq!(select_neighbors(&views, &pts, DEFAULT_NEIGHBORS), base);
        let r = try_depth_range(&views[10], &pts).unwrap();
        assert!(r.0.is_finite() && r.1.is_finite() && r.0 < r.1);
        // 관측이 하나도 없는 사진.
        let mut lone = views[0];
        lone.id = 999;
        views.push(lone);
        let jobs = dense_jobs(&views, &pts, DEFAULT_NEIGHBORS);
        assert_eq!(jobs.len(), 21);
        assert!(jobs.iter().all(|j| j.view != 21 && j.range.0 < j.range.1));
    }

    /// 카메라 중심 `c` 에서 `target` 을 보는 회전(카메라 z = 보는 방향, y 는 아래쪽 성분).
    fn look_at(c: &Point3<f64>, target: &Point3<f64>, z_dir: Option<Vector3<f64>>) -> Pose {
        let z = z_dir.unwrap_or(target - c).normalize();
        let down = Vector3::new(0.0, 0.0, -1.0);
        let mut x = z.cross(&down);
        if x.norm() < 1e-9 {
            x = Vector3::x();
        }
        let x = x.normalize();
        let y = z.cross(&x);
        let r = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
            x.transpose(),
            y.transpose(),
            z.transpose(),
        ]));
        Pose::from_center(r, c)
    }

    /// F-110: 각은 같고(10°) 점까지 거리만 30 m 와 60 m 인 두 후보. 기준 사진은 점을 광축에서 25° 벗어나
    /// 보므로 z(30 m) 와 광선 길이(33.1 m)가 다르다. z 축척이면 점수 비 0.25, 광선 길이면 0.37.
    #[test]
    fn scale_ratio_uses_camera_depth() {
        let k = Intrinsics::from_hfov(1600, 1200, 70f64.to_radians());
        let x = Point3::origin();
        let alpha = 25f64.to_radians();
        let cr = Point3::new(0.0, 0.0, 30.0 / alpha.cos());
        let axis = Vector3::new(alpha.sin(), 0.0, -alpha.cos());
        let reference = View {
            cam: Camera {
                intrinsics: k,
                pose: look_at(&cr, &x, Some(axis)),
            },
            id: 0,
        };
        let t = 10f64.to_radians();
        let dir = Vector3::new(0.0, t.sin(), t.cos());
        let cand = |d: f64, id: usize| {
            let c = x + dir * d;
            View {
                cam: Camera {
                    intrinsics: k,
                    pose: look_at(&c, &x, None),
                },
                id,
            }
        };
        let (near, far) = (cand(30.0, 1), cand(60.0, 2));
        assert!((reference.cam.pose.transform(&x).z - 30.0).abs() < 1e-9);
        assert!(reference.cam.project(&x).is_some_and(|p| k.contains(&p)));
        let ratio = pair_score(&reference, &far, &x) / pair_score(&reference, &near, &x);
        eprintln!("scale_ratio far/near {ratio:.4}");
        assert!((ratio - 0.25).abs() <= 0.02, "{ratio}");
    }

    /// F-052: SPEC §1 실측 편대(드론 3대 약 10 m 간격, F −3°·R +125°·L −116°, 기울기 60°, 화각 65°,
    /// 위치 간 1 m) 40곳 × 3대. 가운데 위치 각 카메라 사진의 이웃 순위.
    #[test]
    fn formation_neighbors() {
        use crate::synth::{CamId, Scene, SceneConfig};
        let scene = Scene::new(SceneConfig {
            positions: 40,
            ..SceneConfig::default()
        });
        let views: Vec<View> = scene
            .views
            .iter()
            .enumerate()
            .map(|(i, v)| View {
                cam: v.camera,
                id: i,
            })
            .collect();
        let mut rng = Lcg(11);
        let mut pts = Vec::new();
        for _ in 0..40000 {
            let (gx, gy) = (-40.0 + 120.0 * rng.next(), -60.0 + 120.0 * rng.next());
            let x = Point3::new(gx, gy, scene.surface_height(gx, gy));
            let observers: Vec<usize> = views
                .iter()
                .filter(|v| {
                    let k = &v.cam.intrinsics;
                    v.cam.project(&x).is_some_and(|p| k.contains(&p))
                })
                .map(|v| v.id)
                .collect();
            if observers.len() >= 2 {
                pts.push(SparsePoint { xyz: x, observers });
            }
        }
        let nb = select_neighbors(&views, &pts, DEFAULT_NEIGHBORS);
        let c = 20;
        for cam in CamId::ALL {
            let me = scene
                .views
                .iter()
                .position(|v| v.cam == cam && v.position == c)
                .unwrap();
            let label: Vec<String> = nb[me]
                .iter()
                .map(|&j| {
                    let v = &scene.views[j];
                    format!("{}{:+}", v.cam.letter(), v.position as i64 - c as i64)
                })
                .collect();
            // 기하 예측: 사진 가운데 광선이 닿는 지면 점에서 같은 카메라가 b m 움직였을 때
            // 광선 사이 각이 10° 가 되는 b.
            let v0 = &scene.views[me];
            let c0 = v0.camera.pose.center();
            let dir = scene.config.view_dir(cam);
            let g = c0 + dir * ((c0.z - scene.surface_height(c0.x, c0.y)) / -dir.z);
            let step = Vector3::new(scene.config.spacing, 0.0, 0.0);
            let angle = |b: f64| {
                let (ra, rb) = (c0 - g, c0 + step * b - g);
                (ra.dot(&rb) / (ra.norm() * rb.norm())).acos().to_degrees()
            };
            let pred = (1..=20)
                .min_by(|&a, &b| {
                    (angle(a as f64) - TARGET_ANGLE_DEG)
                        .abs()
                        .total_cmp(&(angle(b as f64) - TARGET_ANGLE_DEG).abs())
                })
                .unwrap() as i64;
            eprintln!(
                "{} center neighbors {:?} predicted |offset| {pred} (1 step {:.2} deg, 8 steps {:.2} deg)",
                cam.letter(),
                label,
                angle(1.0),
                angle(8.0)
            );
            // 같은 카메라 간격별 분포: 공유 점 수, 공유 점 광선 사이 각의 중앙값, 점수 합, 점수 순위.
            let mut table = Vec::new();
            for d in (-12i64..=12).filter(|&d| d != 0) {
                let Some(j) = scene
                    .views
                    .iter()
                    .position(|v| v.cam == cam && v.position as i64 == c as i64 + d)
                else {
                    continue;
                };
                let (mut n_sh, mut sum, mut angs) = (0usize, 0.0, Vec::new());
                for p in pts
                    .iter()
                    .filter(|p| p.observers.contains(&me) && p.observers.contains(&j))
                {
                    n_sh += 1;
                    sum += pair_score(&views[me], &views[j], &p.xyz);
                    let (ra, rb) = (c0 - p.xyz, views[j].cam.pose.center() - p.xyz);
                    angs.push((ra.dot(&rb) / (ra.norm() * rb.norm())).acos().to_degrees());
                }
                angs.sort_by(f64::total_cmp);
                let med = angs.get(angs.len() / 2).copied().unwrap_or(f64::NAN);
                table.push((d, n_sh, med, sum));
            }
            let mut order: Vec<usize> = (0..table.len()).collect();
            order.sort_by(|&a, &b| table[b].3.total_cmp(&table[a].3));
            for (r, &t) in order.iter().enumerate() {
                let (d, n_sh, med, sum) = table[t];
                eprintln!(
                    "{} offset {d:+} rank {} shared {n_sh} median_angle {med:.2} score {sum:.1}",
                    cam.letter(),
                    r + 1
                );
            }
            // 1위 간격의 공유 점 광선 각 중앙값은 목표 10° 에서 ±2.5°(약 2칸) 안, 1칸 이웃은 24개 중 하위 4.
            let (_, _, med1, _) = table[order[0]];
            assert!((med1 - TARGET_ANGLE_DEG).abs() <= 2.5, "1위 중앙 각 {med1}");
            for (r, &t) in order.iter().enumerate() {
                if table[t].0.abs() == 1 {
                    assert!(r + 1 > order.len() - 4, "1칸 순위 {}", r + 1);
                }
            }
            let same: Vec<i64> = nb[me]
                .iter()
                .filter(|&&j| scene.views[j].cam == cam)
                .map(|&j| scene.views[j].position as i64 - c as i64)
                .collect();
            assert!(same.len() >= 6, "같은 카메라 {}", same.len());
            assert!(
                !same.contains(&1) && !same.contains(&-1),
                "1칸 이웃이 상위 8"
            );
            let top3 = &nb[me][..3];
            assert!(
                top3.iter().any(|&j| scene.views[j].cam == cam
                    && (6..=9).contains(&(scene.views[j].position as i64 - c as i64).abs())),
                "6~9칸 이웃이 상위 3 밖"
            );
            let first = &scene.views[nb[me][0]];
            assert_eq!(first.cam, cam);
            let d0 = (first.position as i64 - c as i64).abs();
            assert!((d0 - pred).abs() <= 1, "1위 간격 {d0} 예측 {pred}");
        }
    }
}
