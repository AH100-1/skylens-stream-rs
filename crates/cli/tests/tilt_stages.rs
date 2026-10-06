//! 단구역 실행의 파이프라인 단계별 위 방향 오차와 좌표계 기울기를 잰다(측정만, 기본 동작 불변).
//!
//! 단계: 회전 평균 직후(`rot_avg`), 좌표계 맞춤+위치 평균 직후(`placed`), 위치 다듬기 직후
//! (`pos_refined`), 번들 조정 전(`ba_in`), 번들 조정 후(`ba_out`), GPS 정렬 후(`gps_aligned`),
//! 최종 출력 카메라(`final`). 각 단계 회전으로 `up_from_rotations` 를 구해 정답 회전으로 얻은
//! 위 방향과 비교한다. 좌표계는 (가) 카메라 중심 Kabsch, (나) 회전 집합의 평균 상대 회전으로 맞춘다.
//! `TILT_BA_ITERS`(기본 15)로 번들 조정 반복 수를 바꿔 반복 사이 값을 본다.
//! `cargo test --release -p skylens-cli --test tilt_stages -- --ignored --nocapture`

use skylens_core::align::up_from_rotations;
use skylens_core::camera::Pose;
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::math::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};
use skylens_core::pipeline::{
    run_pipeline, stage_diag_enable, stage_diag_refined_start, stage_diag_take, PipelineConfig,
};
use skylens_core::synth::{Scene, SceneConfig};

fn kabsch(a: &[Point3<f64>], b: &[Point3<f64>]) -> Rotation3<f64> {
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
    Rotation3::from_matrix_unchecked(vt.transpose() * d * u.transpose())
}

/// 행렬 합 `m` 에 가장 가까운 회전.
fn polar(m: &Matrix3<f64>) -> Rotation3<f64> {
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    Rotation3::from_matrix_unchecked(u * d * vt)
}

fn axis_deg(r: &Rotation3<f64>) -> Vector3<f64> {
    UnitQuaternion::from_rotation_matrix(r)
        .scaled_axis()
        .map(f64::to_degrees)
}

fn geodetic_of(f: &[&str]) -> Geodetic {
    let n: Vec<f64> = f.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

/// 위 방향 `u` 의 `truth_up` 대비 오차: (전체 각, 동쪽 축 둘레, 북쪽 축 둘레) 도.
fn up_err(u: &Vector3<f64>, truth_up: &Vector3<f64>) -> (f64, f64, f64) {
    let q = Rotation3::rotation_between(u, truth_up).unwrap_or_else(Rotation3::identity);
    let a = axis_deg(&q);
    (a.norm(), a.x, a.y)
}

struct Truth {
    tr: Vec<Rotation3<f64>>,
    tc: Vec<Point3<f64>>,
    cam: Vec<usize>,
}

/// 한 단계의 지표를 한 줄로 찍는다. 반환: (Kabsch 기준 위 방향 오차 전체 각, Kabsch 기울기 크기).
fn analyze(
    label: &str,
    poses: &[Option<Pose>],
    t: &Truth,
    a_override: Option<Rotation3<f64>>,
) -> (f64, f64) {
    let ids: Vec<usize> = (0..poses.len()).filter(|&i| poses[i].is_some()).collect();
    let rots: Vec<Rotation3<f64>> = ids.iter().map(|&i| poses[i].unwrap().rotation).collect();
    let tr: Vec<Rotation3<f64>> = ids.iter().map(|&i| t.tr[i]).collect();
    let ez = Vector3::z();
    let u = up_from_rotations(&rots).expect("up");
    // (나) 평균 상대 회전: R_i = tr_i A.
    let mut m = Matrix3::zeros();
    for (r, q) in rots.iter().zip(&tr) {
        m += q.inverse().matrix() * r.matrix();
    }
    let a_mean = polar(&m);
    let e_mean = up_err(&u, &(a_mean.inverse() * ez));
    // (가) 중심 Kabsch(회전 평균 직후 단계는 중심이 없으므로 바깥에서 준다).
    let a_kab = a_override.unwrap_or_else(|| {
        let c: Vec<Point3<f64>> = ids.iter().map(|&i| poses[i].unwrap().center()).collect();
        let tcs: Vec<Point3<f64>> = ids.iter().map(|&i| t.tc[i]).collect();
        kabsch(&c, &tcs)
    });
    let e_kab = up_err(&u, &(a_kab.inverse() * ez));
    let tilt = axis_deg(&a_kab);
    let mis = axis_deg(&(a_mean * a_kab.inverse()));
    // 회전 평균 단계는 모델 좌표계(좌표계 맞춤 전)라 동/북 성분과 기울기 축 분해에 뜻이 없다.
    if a_override.is_some() {
        println!(
            "| {label} | {:.3} (-) | {:.3} (-) | - | {:.3} |",
            e_kab.0,
            e_mean.0,
            mis.norm()
        );
    } else {
        println!(
            "| {label} | {:.3} ({:+.3}) | {:.3} ({:+.3}) | {:+.3}/{:+.3}/{:+.3} ({:.3}) | {:.3} |",
            e_kab.0,
            e_kab.1,
            e_mean.0,
            e_mean.1,
            tilt.x,
            tilt.y,
            tilt.z,
            tilt.norm(),
            mis.norm()
        );
    }
    // 카메라별 회전 편향 부호 평균: 세계축(동/북/위)과 카메라축(x 오른쪽/y 아래/z 광축).
    for c in 0..3 {
        let mut w = Vector3::zeros();
        let mut cm = Vector3::zeros();
        let mut k = 0.0;
        for (a, &i) in ids.iter().enumerate() {
            if t.cam[i] != c {
                continue;
            }
            let wi = tr[a].inverse() * rots[a] * a_kab.inverse();
            w += axis_deg(&wi);
            cm += axis_deg(&(tr[a] * wi * tr[a].inverse()));
            k += 1.0;
        }
        w /= k;
        cm /= k;
        println!(
            "    cam{} 세계(동/북/위) {:+.3}/{:+.3}/{:+.3}  카메라(x/y/z) {:+.3}/{:+.3}/{:+.3}",
            c, w.x, w.y, w.z, cm.x, cm.y, cm.z
        );
    }
    (e_kab.0, tilt.norm())
}

#[test]
#[ignore = "약 3분, 측정용"]
fn tilt_stages() {
    let ba_iters: usize = std::env::var("TILT_BA_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let dir = std::env::temp_dir().join(format!("skylens_ts_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (scene_dir, out) = (dir.join("scene"), dir.join("out"));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
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
        ba_iters,
        ..PipelineConfig::default()
    };
    skylens_core::dense::view_diag_enable();
    stage_diag_enable();
    // TILT_REFINED_START=1: 정밀 BA 시작점에도 위치 다듬기를 적용해 본다(대안 측정).
    stage_diag_refined_start(std::env::var("TILT_REFINED_START").is_ok());
    run_pipeline(&ds, &pc, &out).unwrap();
    let rec = skylens_core::dense::view_diag_take().pop().unwrap();
    let snaps = stage_diag_take();
    let n = rec.cams.len();
    assert_eq!(n, 120);

    let o = std::fs::read_to_string(scene_dir.join("truth/origin.txt")).unwrap();
    let to = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gtxt = std::fs::read_to_string(scene_dir.join("gps.txt")).unwrap();
    let first: Vec<&str> = gtxt.lines().next().unwrap().split_whitespace().collect();
    let d = geodetic_to_enu(&to, &geodetic_of(&first[1..4]));
    let shift = Vector3::new(d.x, d.y, d.z);
    let mut t = Truth {
        tr: vec![],
        tc: vec![],
        cam: vec![],
    };
    for v in 0..n {
        let g = rec.src[v];
        let name = ds.positions[g / 3].images[g % 3]
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let sv = scene.views.iter().find(|s| s.name == name).unwrap();
        t.tc.push(Point3::from(sv.camera.pose.center().coords + shift));
        t.tr.push(sv.camera.pose.rotation);
        t.cam.push(g % 3);
    }
    // GPS 위치 자체의 기울기: GPS 점을 정답 중심에 Kabsch. 중심 정렬이 GPS 에 묶이면 이보다 나아질 수 없다.
    let gps: Vec<Point3<f64>> = (0..n)
        .map(|v| {
            let g = rec.src[v];
            Point3::from(ds.positions[g / 3].image_enu[g % 3])
        })
        .collect();
    let gt = axis_deg(&kabsch(&gps, &t.tc));
    println!(
        "GPS 위치 자체의 Kabsch 기울기 동/북/위 {:+.3}/{:+.3}/{:+.3} (크기 {:.3}) 도",
        gt.x,
        gt.y,
        gt.z,
        gt.norm()
    );
    let tu = up_from_rotations(&t.tr).unwrap();
    println!(
        "BA 반복 {ba_iters}; 정답 자세에서 추정한 위 방향 편차 {:.3} 도",
        tu.angle(&Vector3::z()).to_degrees()
    );
    println!(
        "스냅숏 순서: {:?}",
        snaps.iter().map(|s| s.label).collect::<Vec<_>>()
    );
    // 마지막으로 나온 같은 이름의 스냅숏(정밀 BA 로 이어진 줄)을 쓴다.
    let last = |l: &str| {
        snaps
            .iter()
            .rev()
            .find(|s| s.label == l && s.poses.len() == n)
            .map(|s| s.poses.clone())
    };
    println!("\n| 단계 | 위 방향 오차 도 (동쪽 성분), 중심 Kabsch 기준 | 같은, 평균 상대 회전 기준 | 중심 Kabsch 기울기 동/북/위 (크기) | 회전·중심 좌표계 어긋남 |");
    println!("|---|---|---|---|---|");
    let placed = last("placed").expect("placed");
    // 회전 평균 단계의 좌표계: placed = rots · gᵀ 이므로 g = placedᵢ⁻¹ · rotsᵢ.
    let rot_avg = last("rot_avg").expect("rot_avg");
    let i0 = (0..n).find(|&i| placed[i].is_some()).unwrap();
    let g = placed[i0].unwrap().rotation.inverse() * rot_avg[i0].unwrap().rotation;
    let ac = {
        let c: Vec<Point3<f64>> = placed.iter().flatten().map(|p| p.center()).collect();
        let tcs: Vec<Point3<f64>> = (0..n)
            .filter(|&i| placed[i].is_some())
            .map(|i| t.tc[i])
            .collect();
        kabsch(&c, &tcs)
    };
    let mut rows = Vec::new();
    rows.push((
        "rot_avg",
        analyze(
            "rot_avg(A=placed 중심 Kabsch·g)",
            &rot_avg,
            &t,
            Some(ac * g),
        ),
        0,
    ));
    for l in ["placed", "pos_refined", "ba_in", "ba_out", "gps_aligned"] {
        if let Some(p) = last(l) {
            rows.push((l, analyze(l, &p, &t, None), 0));
        }
    }
    let fin: Vec<Option<Pose>> = rec.cams.iter().map(|c| Some(c.pose)).collect();
    rows.push(("final", analyze("final(출력 카메라)", &fin, &t, None), 0));
    // 최종 단계가 좌표계를 기울이는가: gps_aligned 와 final 의 중심 차이·회전 차이.
    if let Some(ga) = last("gps_aligned") {
        let (mut dc, mut dr) = (Vec::new(), Vec::new());
        #[allow(clippy::needless_range_loop)]
        for v in 0..n {
            let p = ga[v].unwrap();
            dc.push((p.center() - rec.cams[v].pose.center()).norm());
            dr.push(axis_deg(&(p.rotation * rec.cams[v].pose.rotation.inverse())).norm());
        }
        dc.sort_by(f64::total_cmp);
        dr.sort_by(f64::total_cmp);
        println!(
            "\ngps_aligned 와 final 의 차이: 중심 중앙 {:.4} m 최대 {:.4} m, 회전 중앙 {:.4} 도 최대 {:.4} 도",
            dc[n / 2], dc[n - 1], dr[n / 2], dr[n - 1]
        );
    }
    let _ = rows;
    let fe = analyze("final(재계산)", &fin, &t, None);
    assert!(fe.0.is_finite() && fe.1.is_finite());
    let _ = std::fs::remove_dir_all(&dir);
}
