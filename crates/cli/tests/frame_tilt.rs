//! 단구역 출력 좌표계 기울기(동쪽 축 둘레 약 0.75 도)와 GPS 정렬의 위 방향 추정을 잰다.
//!
//! 파이프라인 출력 카메라(이미 GPS 정렬된 좌표계)를 GPS 위치에 다시 정렬하는 방식으로
//! 위 방향 추정 방법을 바꿔 비교한다(위 방향 추정은 좌표계에 공변이므로 같은 결과를 준다).
//! `cargo test --release --test frame_tilt -- --ignored --nocapture`

use skylens_core::align::{
    align_to_enu_with, up_cross_check, up_from_rotations, GpsAlignConfig, Similarity,
    UP_CROSS_WARN_DEG,
};
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::math::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};

fn quant(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() as f64 * q) as usize).min(s.len() - 1)]
}

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

fn geodetic_of(f: &[&str]) -> Geodetic {
    let n: Vec<f64> = f.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

/// 위 방향 `u`(현재 출력 좌표)의 정답 위 방향 대비 오차: (전체 각, 동쪽 축 둘레, 북쪽 축 둘레) 도.
fn up_err(u: &Vector3<f64>, truth_up: &Vector3<f64>) -> (f64, f64, f64) {
    // 회전 u → truth_up 의 축각 성분(동 = x, 북 = y 둘레).
    let q = Rotation3::rotation_between(u, truth_up).unwrap_or_else(Rotation3::identity);
    let a = UnitQuaternion::from_rotation_matrix(&q)
        .scaled_axis()
        .map(f64::to_degrees);
    (a.norm(), a.x, a.y)
}

fn median_up(ups: &[Vector3<f64>]) -> Vector3<f64> {
    let m = |f: &dyn Fn(&Vector3<f64>) -> f64| {
        let mut v: Vec<f64> = ups.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    };
    Vector3::new(m(&|u| u.x), m(&|u| u.y), m(&|u| u.z)).normalize()
}

#[test]
#[ignore = "약 2분, 측정용"]
fn frame_tilt() {
    let dir = std::env::temp_dir().join(format!("skylens_ft_{}", std::process::id()));
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
        ba_iters: 15,
        ..PipelineConfig::default()
    };
    skylens_core::dense::view_diag_enable();
    run_pipeline(&ds, &pc, &out).unwrap();
    let rec = skylens_core::dense::view_diag_take().pop().unwrap();
    let n = rec.cams.len();
    assert_eq!(n, 120);

    // 정답 자세(출력 좌표 = 장면 좌표 + shift).
    let o = std::fs::read_to_string(scene_dir.join("truth/origin.txt")).unwrap();
    let to = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gtxt = std::fs::read_to_string(scene_dir.join("gps.txt")).unwrap();
    let first: Vec<&str> = gtxt.lines().next().unwrap().split_whitespace().collect();
    let d = geodetic_to_enu(&to, &geodetic_of(&first[1..4]));
    let shift = Vector3::new(d.x, d.y, d.z);
    let mut tc = Vec::new();
    let mut tr = Vec::new();
    let mut gps = Vec::new();
    for v in 0..n {
        let g = rec.src[v];
        let name = ds.positions[g / 3].images[g % 3]
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let sv = scene.views.iter().find(|s| s.name == name).unwrap();
        tc.push(Point3::from(sv.camera.pose.center().coords + shift));
        tr.push(sv.camera.pose.rotation);
        gps.push(ds.positions[g / 3].image_enu[g % 3]);
    }
    let src: Vec<Vector3<f64>> = rec.cams.iter().map(|c| c.pose.center().coords).collect();
    let rots: Vec<Rotation3<f64>> = rec.cams.iter().map(|c| c.pose.rotation).collect();
    let labels: Vec<usize> = (0..n).map(|v| rec.src[v] % 3).collect();
    let ez = Vector3::new(0.0, 0.0, 1.0);
    let oc: Vec<Point3<f64>> = src.iter().map(|c| Point3::from(*c)).collect();
    let ra0 = kabsch(&oc, &tc);
    let truth_up_in_out = ra0.inverse() * ez;
    let tu = up_from_rotations(&tr).unwrap();
    println!(
        "정답 자세에서 추정한 위 방향 편차 {:.3} 도",
        tu.angle(&ez).to_degrees()
    );

    let chk = up_cross_check(&rots, &labels).unwrap();
    println!(
        "묶음별 위 방향 어긋남(도): {:?}, 최대 {:?}",
        chk.diff_deg, chk.max_diff_deg
    );
    for (i, u) in chk.up_own.iter().enumerate() {
        if let Some(u) = u {
            let e = up_err(u, &truth_up_in_out);
            println!(
                "camera {} 자기 위 방향 오차 {:.3} (동 {:+.3} 북 {:+.3})",
                chk.labels[i], e.0, e.1, e.2
            );
        } else {
            println!("camera {} 자기 위 방향 없음(방위 한 방향)", chk.labels[i]);
        }
    }
    for (i, u) in chk.up_rest.iter().enumerate() {
        if let Some(u) = u {
            let e = up_err(u, &truth_up_in_out);
            println!(
                "camera {} 를 뺀 위 방향 오차 {:.3} (동 {:+.3} 북 {:+.3})",
                chk.labels[i], e.0, e.1, e.2
            );
        }
    }

    // 변형들.
    let all = up_from_rotations(&rots);
    let own: Vec<Vector3<f64>> = chk.up_own.iter().flatten().copied().collect();
    let med = (own.len() >= 2).then(|| median_up(&own));
    let dropped = chk
        .max_diff_deg
        .filter(|&m| m > UP_CROSS_WARN_DEG)
        .and_then(|_| {
            let worst = chk
                .diff_deg
                .iter()
                .enumerate()
                .filter_map(|(i, d)| d.map(|d| (i, d)))
                .max_by(|a, b| a.1.total_cmp(&b.1))?
                .0;
            chk.up_rest[worst]
        });
    // 희소 점 지면 평면 법선(점 구름 주성분의 최소 고유벡터, 위쪽 부호).
    let plane_up = {
        let pts: Vec<Vector3<f64>> = rec.sparse.iter().map(|p| Vector3::from(*p)).collect();
        let mu = pts.iter().sum::<Vector3<f64>>() / pts.len() as f64;
        let mut cov = Matrix3::zeros();
        for p in &pts {
            cov += (p - mu) * (p - mu).transpose();
        }
        let eig = cov.symmetric_eigen();
        let i = (0..3)
            .min_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]))
            .unwrap();
        let mut u: Vector3<f64> = eig.eigenvectors.column(i).into();
        if u.z < 0.0 {
            u = -u;
        }
        println!(
            "희소 점 {} 개, 평면 법선 오차 {:.3} 도",
            pts.len(),
            up_err(&u, &truth_up_in_out).0
        );
        Some(u)
    };
    let variants: Vec<(&str, Option<Vector3<f64>>)> = vec![
        ("희소 점 평면 법선", plane_up),
        ("전체 회전 위 방향(현재)", all),
        ("위 방향 고정 없음(GPS 만)", None),
        ("묶음별 위 방향 성분 중앙", med),
        ("교차 검사로 튄 묶음 제외", dropped),
    ];
    println!("\n| 방법 | 위 방향 오차(도) | 동쪽 축 | 정렬 뒤 좌표계 기울기 동/북/위 (도) | camF/R/L 회전 오차 중앙(도) |");
    println!("|---|---|---|---|---|");
    for (name, up) in variants {
        let acfg = GpsAlignConfig {
            up,
            ..Default::default()
        };
        let Some(a) = align_to_enu_with(&src, &gps, &acfg) else {
            println!("| {name} | 정렬 실패 또는 해당 없음 |");
            continue;
        };
        let sim: Similarity = a.sim;
        let nc: Vec<Point3<f64>> = src
            .iter()
            .map(|c| Point3::from(sim.apply_point(c)))
            .collect();
        let nr: Vec<Rotation3<f64>> = rots.iter().map(|r| *r * sim.r.inverse()).collect();
        let ra = kabsch(&nc, &tc);
        let ax = UnitQuaternion::from_rotation_matrix(&ra)
            .scaled_axis()
            .map(f64::to_degrees);
        let mut med_c = [0.0; 3];
        #[allow(clippy::needless_range_loop)]
        for c in 0..3 {
            let mut e = Vec::new();
            for v in (0..n).filter(|&v| labels[v] == c) {
                let al = nr[v] * ra.inverse();
                e.push(
                    UnitQuaternion::from_rotation_matrix(&(al * tr[v].inverse()))
                        .scaled_axis()
                        .norm()
                        .to_degrees(),
                );
            }
            med_c[c] = quant(&e, 0.5);
        }
        let ue = up.map(|u| up_err(&u, &truth_up_in_out));
        println!(
            "| {name} | {} | {} | {:+.3} / {:+.3} / {:+.3} | {:.3} / {:.3} / {:.3} |",
            ue.map_or("-".into(), |e| format!("{:.3}", e.0)),
            ue.map_or("-".into(), |e| format!("{:+.3}", e.1)),
            ax.x,
            ax.y,
            ax.z,
            med_c[0],
            med_c[1],
            med_c[2]
        );
    }
    // 실측: 현재 방법의 위 방향 오차 0.473 도(동쪽 축 +0.435), 묶음별 어긋남 최대 0.089 도,
    // 어느 묶음을 빼도 0.42~0.52 도, 자기 위 방향은 세 묶음 모두 없음. 한 묶음이 끌어당기는 것이 아니다.
    let e = up_err(&all.unwrap(), &truth_up_in_out);
    assert!((0.35..0.60).contains(&e.0), "위 방향 오차 {}", e.0);
    assert!(e.1 > 0.3, "동쪽 축 성분 {}", e.1);
    assert!(
        chk.max_diff_deg.unwrap() < UP_CROSS_WARN_DEG,
        "교차 검사 어긋남"
    );
    assert!(chk.up_own.iter().all(|u| u.is_none()));
    for u in chk.up_rest.iter().flatten() {
        let r = up_err(u, &truth_up_in_out).0;
        assert!((0.3..0.65).contains(&r), "묶음 제외 위 방향 오차 {r}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
