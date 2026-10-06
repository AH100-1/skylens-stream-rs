//! 단구역 밀집 표면 95% 꼬리의 원인을 카메라별로 가른다(기준 카메라 camL 이 켬에서 나빠지는 이유).
//!
//! 한 번 돌리면 (1) 카메라 F/R/L 별 정밀 자세 회전 오차(출력 중심을 정답 중심에 강체 정렬한 뒤),
//! (2) 사진별 깊이 지도의 화소 오차(정답 카메라에서 같은 화소로 쏜 광선이 정답 장면과 만나는 깊이 대비),
//! (3) 점 오차를 수직 거리·건물 상자 포함 근사 3D 거리·기준 카메라 광선 깊이 차 세 가지로 낸다.
//! 그리고 같은 사진·희소 점으로 자세만 정답으로 바꿔 밀집 단계를 다시 돌려(전부 / camL 만) 비교한다.
//! `cargo test --release --test dense_caml_tail -- --ignored --nocapture` 출력이 연구 노트 표다.
//! 끔 실행은 특징 검출의 확대 문턱을 0 으로 둔 임시 빌드로 따로 돌렸다.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use skylens_core::camera::Camera;
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::dense::{region_cloud, view_diag_enable, view_diag_take, ViewDiagRecord};
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::math::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector2, Vector3};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};

const CAM: [&str; 3] = ["F", "R", "L"];

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_ct_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn quant(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() as f64 * q) as usize).min(s.len() - 1)]
}

fn frac_over(v: &[f64], t: f64) -> f64 {
    100.0 * v.iter().filter(|&&x| x > t).count() as f64 / v.len().max(1) as f64
}

fn geodetic_of(fields: &[&str]) -> Geodetic {
    let n: Vec<f64> = fields.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

fn truth_to_output_shift(input: &std::path::Path) -> Vector3<f64> {
    let o = std::fs::read_to_string(input.join("truth/origin.txt")).unwrap();
    let truth_origin = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gps = std::fs::read_to_string(input.join("gps.txt")).unwrap();
    let first: Vec<&str> = gps.lines().next().unwrap().split_whitespace().collect();
    let d = geodetic_to_enu(&truth_origin, &geodetic_of(&first[1..4]));
    Vector3::new(d.x, d.y, d.z)
}

/// 정답 표면까지 근사 3D 거리: 수직 차와 건물 상자(바깥 거리) 중 작은 값.
fn surface_dist(s: &Scene, p: &Point3<f64>) -> f64 {
    let mut d = (p.z - s.surface_height(p.x, p.y)).abs();
    for b in &s.buildings {
        let dx = (b.min.x - p.x).max(p.x - b.max.x).max(0.0);
        let dy = (b.min.y - p.y).max(p.y - b.max.y).max(0.0);
        let dz = (p.z - b.top).max(0.0);
        d = d.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    d
}

/// 출력 중심 → 정답 중심 강체 정렬 x_t = R x_o + t (Kabsch).
fn kabsch(a: &[Point3<f64>], b: &[Point3<f64>]) -> (Rotation3<f64>, Vector3<f64>) {
    let n = a.len() as f64;
    let ma = a.iter().map(|p| p.coords).sum::<Vector3<f64>>() / n;
    let mb = b.iter().map(|p| p.coords).sum::<Vector3<f64>>() / n;
    let mut h = Matrix3::zeros();
    for (p, q) in a.iter().zip(b) {
        h += (p.coords - ma) * (q.coords - mb).transpose();
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    if (vt.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = Rotation3::from_matrix_unchecked(vt.transpose() * d * u.transpose());
    (r, mb - r * ma)
}

/// 사진별 깊이 지도의 화소 오차: 정답 카메라에서 같은 화소로 쏜 광선의 정답 깊이 대비.
/// 반환: 카메라별 (상대 오차 부호 있는 값 목록, 절대 오차 m 목록, 표본 화소 수).
#[allow(clippy::type_complexity)]
fn depth_errors(
    rec: &ViewDiagRecord,
    truth_cam: &[Camera],
    scene: &Scene,
    shift: &Vector3<f64>,
) -> [(Vec<f64>, Vec<f64>, usize); 3] {
    const STEP: usize = 2;
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, Vec<f64>, Vec<f64>, usize)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..2)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let v = next.fetch_add(1, Ordering::Relaxed);
                        if v >= rec.cams.len() {
                            break;
                        }
                        let (m, k) = (&rec.maps[v], &rec.cams[v].intrinsics);
                        let tp = &truth_cam[v].pose;
                        let o = Point3::from(tp.center().coords - shift);
                        let rt = tp.rotation.inverse();
                        let (mut rel, mut abs, mut tot) = (Vec::new(), Vec::new(), 0usize);
                        for y in (0..m.h).step_by(STEP) {
                            for x in (0..m.w).step_by(STEP) {
                                let d = m.depth[y * m.w + x] as f64;
                                tot += 1;
                                if !(d.is_finite() && d > 0.0) {
                                    continue;
                                }
                                let n =
                                    k.to_normalized(&Vector2::new(x as f64 + 0.5, y as f64 + 0.5));
                                let dir = rt * Vector3::new(n.x, n.y, 1.0);
                                if let Some(h) = scene.intersect(&o, &dir) {
                                    rel.push((d - h.t) / h.t);
                                    abs.push((d - h.t).abs());
                                }
                            }
                        }
                        out.push((rec.src[v] % 3, rel, abs, tot));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut r: [(Vec<f64>, Vec<f64>, usize); 3] = Default::default();
    for (c, rel, abs, tot) in parts.into_iter().flatten() {
        r[c].0.extend(rel);
        r[c].1.extend(abs);
        r[c].2 += tot;
    }
    r
}

fn print_depth(label: &str, d: &[(Vec<f64>, Vec<f64>, usize); 3]) {
    println!("\n## 깊이 지도 화소 오차: {label}");
    println!("| 카메라 | 표본 화소 | 유효(교차 있음) | 상대 |오차| 중앙 | 상대 95% | 상대>5% | 절대 95% (m) | 절대>1 m | 상대 부호 중앙 |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for (c, (rel, abs, tot)) in d.iter().enumerate() {
        let ar: Vec<f64> = rel.iter().map(|x| x.abs()).collect();
        println!(
            "| cam{} | {tot} | {} ({:.1}%) | {:.4} | {:.4} | {:.1}% | {:.3} | {:.1}% | {:+.4} |",
            CAM[c],
            rel.len(),
            100.0 * rel.len() as f64 / (*tot).max(1) as f64,
            quant(&ar, 0.5),
            quant(&ar, 0.95),
            frac_over(&ar, 0.05),
            quant(abs, 0.95),
            frac_over(abs, 1.0),
            quant(rel, 0.5)
        );
    }
}

/// 점 오차 세 가지(수직, 근사 3D, 기준 카메라 광선 깊이 차), 점마다 기준 카메라와 함께.
struct PointErr {
    cam: usize,
    vert: f64,
    d3: f64,
    ray: f64,
}

/// `map`: 이 기록의 좌표 → 정답 장면 좌표.
fn point_errors(
    rec: &ViewDiagRecord,
    scene: &Scene,
    map: &dyn Fn(&Point3<f64>) -> Point3<f64>,
) -> Vec<PointErr> {
    let mut out = Vec::new();
    for (k, p) in rec.points.iter().enumerate() {
        let v = rec.point_view[k];
        if v == usize::MAX {
            continue;
        }
        let q = map(&Point3::new(p[0] as f64, p[1] as f64, p[2] as f64));
        let c = map(&rec.cams[v].pose.center());
        let dir = q - c;
        let dist = dir.norm();
        let ray = scene
            .intersect(&c, &(dir / dist))
            .map_or(f64::NAN, |h| (dist - h.t).abs());
        out.push(PointErr {
            cam: rec.src[v] % 3,
            vert: (q.z - scene.surface_height(q.x, q.y)).abs(),
            d3: surface_dist(scene, &q),
            ray,
        });
    }
    out
}

fn print_points(label: &str, e: &[PointErr]) {
    println!("\n## 점 오차: {label}");
    println!(
        "| 기준 | 점 수 | 수직 중앙/95%/>1 m | 3D 근사 중앙/95%/>1 m | 광선 깊이 중앙/95%/>1 m |"
    );
    println!("|---|---|---|---|---|");
    let row = |name: String, sel: Vec<&PointErr>| {
        let col = |f: fn(&PointErr) -> f64| {
            let v: Vec<f64> = sel.iter().map(|x| f(x)).filter(|x| x.is_finite()).collect();
            format!(
                "{:.3} / {:.3} / {:.1}%",
                quant(&v, 0.5),
                quant(&v, 0.95),
                frac_over(&v, 1.0)
            )
        };
        println!(
            "| {name} | {} | {} | {} | {} |",
            sel.len(),
            col(|x| x.vert),
            col(|x| x.d3),
            col(|x| x.ray)
        );
    };
    row("전체".into(), e.iter().collect());
    for (c, name) in CAM.iter().enumerate() {
        row(
            format!("cam{name}"),
            e.iter().filter(|x| x.cam == c).collect(),
        );
    }
}

fn p95_vert(e: &[PointErr], cam: Option<usize>) -> f64 {
    let v: Vec<f64> = e
        .iter()
        .filter(|x| cam.is_none_or(|c| x.cam == c))
        .map(|x| x.vert)
        .collect();
    quant(&v, 0.95)
}

fn cfg_scene() -> SceneConfig {
    SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    }
}

#[test]
#[ignore = "약 5분, 측정용: --ignored --nocapture"]
fn dense_caml_tail() {
    let t = TempDir::new("single");
    let (scene_dir, out) = (t.0.join("scene"), t.0.join("out"));
    let scene = Scene::new(cfg_scene());
    scene.write_dataset(&scene_dir).unwrap();
    let ds = load_dataset(
        &scene_dir,
        DatasetConfig {
            stride: 2,
            span: 48,
            ovl: 2,
            ..DatasetConfig::default()
        },
    )
    .unwrap();
    let pc = PipelineConfig {
        max_features: 800,
        dense_width: 96,
        hfov_deg: 65.0,
        ba_iters: 15,
        ..PipelineConfig::default()
    };
    view_diag_enable();
    run_pipeline(&ds, &pc, &out).unwrap();
    let recs = view_diag_take();
    let rec = recs.last().expect("밀집 기록 없음");
    let n = rec.cams.len();
    assert_eq!(n, 120, "등록 수");
    let shift = truth_to_output_shift(&scene_dir);

    // 정답 카메라(출력 좌표 = 장면 좌표 + shift). 내부 파라미터는 밀집 지도 기준 그대로.
    let truth_cam: Vec<Camera> = (0..n)
        .map(|v| {
            let g = rec.src[v];
            let name = ds.positions[g / 3].images[g % 3]
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let sv = scene.views.iter().find(|s| s.name == name).unwrap();
            let c = Point3::from(sv.camera.pose.center().coords + shift);
            Camera {
                intrinsics: rec.cams[v].intrinsics,
                pose: skylens_core::camera::Pose::from_center(sv.camera.pose.rotation, &c),
            }
        })
        .collect();

    // 1) 회전 오차.
    let oc: Vec<Point3<f64>> = rec.cams.iter().map(|c| c.pose.center()).collect();
    let tc: Vec<Point3<f64>> = truth_cam.iter().map(|c| c.pose.center()).collect();
    let (ra, ta) = kabsch(&oc, &tc);
    let ra_axis = UnitQuaternion::from_rotation_matrix(&ra)
        .scaled_axis()
        .map(f64::to_degrees);
    println!(
        "\n강체 정렬(출력→정답 중심): 회전 {:.3} 도 (동/북/위 성분 {:+.3} {:+.3} {:+.3}), 이동 {:.3} m",
        ra_axis.norm(),
        ra_axis.x,
        ra_axis.y,
        ra_axis.z,
        ta.norm()
    );
    println!("\n## 카메라별 회전 오차(도, 정렬 뒤) 와 정렬 뒤 중심 오차(m)");
    println!("| 카메라 | 장수 | 각 중앙 | 각 최대 | |x|(위아래 기울기) 중앙 | |y|(좌우) 중앙 | |z|(롤) 중앙 | 부호 평균 x/y/z | 중심 중앙/최대 |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut rot_med = [0.0; 3];
    for c in 0..3 {
        let (mut ang, mut ax, mut ay, mut az, mut ctr) = (vec![], vec![], vec![], vec![], vec![]);
        let mut sum = Vector3::zeros();
        for v in (0..n).filter(|&v| rec.src[v] % 3 == c) {
            let al = rec.cams[v].pose.rotation * ra.inverse();
            let e =
                UnitQuaternion::from_rotation_matrix(&(al * truth_cam[v].pose.rotation.inverse()))
                    .scaled_axis()
                    .map(|x| x.to_degrees());
            ang.push(e.norm());
            ax.push(e.x.abs());
            ay.push(e.y.abs());
            az.push(e.z.abs());
            sum += e;
            ctr.push((ra * oc[v] + ta - tc[v].coords).coords.norm());
        }
        let m = ang.len() as f64;
        rot_med[c] = quant(&ang, 0.5);
        println!(
            "| cam{} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:+.3} / {:+.3} / {:+.3} | {:.3} / {:.3} |",
            CAM[c],
            ang.len(),
            quant(&ang, 0.5),
            ang.iter().cloned().fold(0.0, f64::max),
            quant(&ax, 0.5),
            quant(&ay, 0.5),
            quant(&az, 0.5),
            sum.x / m,
            sum.y / m,
            sum.z / m,
            quant(&ctr, 0.5),
            ctr.iter().cloned().fold(0.0, f64::max)
        );
    }

    // 2) 출력 자세의 깊이 지도 오차.
    let d_out = depth_errors(rec, &truth_cam, &scene, &shift);
    print_depth("출력 자세(실행 그대로)", &d_out);

    // 3) 점 오차 세 가지.
    let p_shift = point_errors(rec, &scene, &|p| Point3::from(p.coords - shift));
    let p_kab = point_errors(rec, &scene, &|p| Point3::from((ra * p + ta).coords - shift));
    print_points("출력 자세, 이동만 보정(앞선 기록과 같은 방식)", &p_shift);
    print_points("출력 자세, 강체 정렬 보정", &p_kab);

    // 4) 자세만 정답으로 바꿔 밀집 단계를 다시(같은 사진·희소 점·설정).
    let rerun = |label: &str, only_cam: Option<usize>| -> ViewDiagRecord {
        let views: Vec<_> = rec
            .views
            .iter()
            .enumerate()
            .map(|(g, v)| {
                let mut v = v.clone();
                if only_cam.is_none_or(|c| g % 3 == c) {
                    let vi = rec.src.iter().position(|&s| s == g).unwrap();
                    v.camera.pose = truth_cam[vi].pose;
                }
                v
            })
            .collect();
        view_diag_enable();
        let cloud = region_cloud(&views, &rec.sparse, &rec.cfg);
        let r = view_diag_take().pop().expect("밀집 기록 없음");
        println!("\n(재실행 {label}: 밀집 점 {})", cloud.points.len());
        r
    };
    let mut reruns = Vec::new();
    for (label, only) in [("정답 자세 전부", None), ("camL 만 정답 자세", Some(2))] {
        let r = rerun(label, only);
        // 자세가 섞인 경우 깊이 오차는 화소 광선이 정답 카메라 기준이라 그대로 비교 가능하다.
        let d = depth_errors(&r, &truth_cam, &scene, &shift);
        print_depth(label, &d);
        let pe = point_errors(&r, &scene, &|p| {
            if only.is_none() {
                Point3::from(p.coords - shift)
            } else {
                Point3::from((ra * p + ta).coords - shift)
            }
        });
        print_points(label, &pe);
        reruns.push((d, pe));
    }

    // 기본(확대 켬) 실측: 정렬 뒤 camL 회전 각 중앙 0.726 도, camL 깊이 상대 95% 0.0185, 강체 정렬 뒤 점 표면 95% 0.637 m,
    // 정답 자세 재실행 95% 0.351 m, 이동만 보정한 근사 3D 95% 1.053 m. 상한은 실측의 약 1.2 배.
    let p3: Vec<f64> = p_shift.iter().map(|x| x.d3).collect();
    let rel_l: Vec<f64> = d_out[2].0.iter().map(|x| x.abs()).collect();
    eprintln!(
        "camL 회전 {:.3} 깊이 상대95 {:.4} 정렬 95 {:.3} 재실행 95 {:.3} 3D95 {:.3}",
        rot_med[2],
        quant(&rel_l, 0.95),
        p95_vert(&p_kab, None),
        p95_vert(&reruns[0].1, None),
        quant(&p3, 0.95)
    );
    assert!(rot_med[2] < 0.87, "camL 회전 오차 {}", rot_med[2]);
    assert!(quant(&rel_l, 0.95) < 0.0222, "camL 깊이 상대 95%");
    assert!(p95_vert(&p_kab, None) < 0.77, "정렬 뒤 표면 95%");
    assert!(p95_vert(&reruns[0].1, None) < 0.42, "정답 자세 재실행 95%");
    assert!(quant(&p3, 0.95) < 1.27, "근사 3D 95%");
}
