//! 구역 GPS 정렬이 남기는 기울기의 원인 분리(진단, 느림: `--ignored`).
//! 정렬 직전 자세·GPS 를 받아 같은 입력을 여러 변형(정답 중심, GPS 잡음 0/모의, 트리밍 없음, 연직 고정)으로 다시 맞춰 기울기를 잰다.

use skylens_core::align::{
    robust_similarity, similarity_fixed_up, umeyama, up_from_rotations, Similarity,
};
use skylens_core::camera::Pose;
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::{Matrix3, Rotation3, Vector3};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::pipeline_stream::{take_pre_tap, take_tap, TAP_ON};
use skylens_core::synth::{Scene, SceneConfig};

fn mean_rot(rs: &[Rotation3<f64>]) -> Rotation3<f64> {
    let mut m = Matrix3::zeros();
    for r in rs {
        m += r.matrix();
    }
    let sv = m.svd(true, true);
    let (u, vt) = (sv.u.unwrap(), sv.v_t.unwrap());
    let q = u * vt;
    let q = if q.determinant() < 0.0 {
        let mut u2 = u;
        u2.column_mut(2).neg_mut();
        u2 * vt
    } else {
        q
    };
    Rotation3::from_matrix_unchecked(q)
}

fn deg(r: &Rotation3<f64>) -> f64 {
    r.angle().to_degrees()
}

fn rv(r: &Rotation3<f64>) -> Vec<f64> {
    r.scaled_axis()
        .iter()
        .map(|x| (x.to_degrees() * 1000.0).round() / 1000.0)
        .collect()
}

/// 정렬 뒤 자세 오차 E = R_tᵀ R_after 의 평균 회전과 평균 제거 뒤 중앙값.
fn tilt(sim: &Similarity, poses: &[Pose], truth: &[Rotation3<f64>]) -> (f64, Vec<f64>, f64) {
    let es: Vec<Rotation3<f64>> = poses
        .iter()
        .zip(truth)
        .map(|(p, t)| t.inverse() * (p.rotation * sim.r.inverse()))
        .collect();
    let me = mean_rot(&es);
    let mut res: Vec<f64> = es.iter().map(|e| deg(&(me.inverse() * *e))).collect();
    res.sort_by(f64::total_cmp);
    (deg(&me), rv(&me), res[res.len() / 2])
}

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos()
    }
}

#[test]
#[ignore]
fn gps_tilt_causes() {
    let root = std::env::temp_dir().join(format!("skylens_tiltdiag_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let scene = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    scene.write_dataset(&root.join("in")).unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 48,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    let cfg = PipelineConfig {
        max_features: 800,
        dense_width: 96,
        hfov_deg: 65.0,
        ba_iters: 15,
        preview_ba_iters: 0,
        gps_fixed_up: std::env::var_os("GPS_UP").is_some(),
        ..PipelineConfig::default()
    };
    TAP_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = run_pipeline(&ds, &cfg, &root.join("out")).unwrap();
    let (post, _) = take_tap();
    let mut pre = take_pre_tap();
    pre.sort_by_key(|x| x.0);
    let view = |g: usize| {
        let name = ds.positions[g / 3].images[g % 3]
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        scene.views.iter().find(|v| v.name == name).unwrap()
    };
    for (slot, items) in &pre {
        let poses: Vec<Pose> = items.iter().map(|x| x.1).collect();
        let src: Vec<Vector3<f64>> = poses.iter().map(|p| p.center().coords).collect();
        let gps: Vec<Vector3<f64>> = items.iter().map(|x| x.2).collect();
        let tc: Vec<Vector3<f64>> = items
            .iter()
            .map(|x| {
                scene
                    .to_first_gps_frame(&view(x.0).camera.pose.center())
                    .coords
            })
            .collect();
        let tr: Vec<Rotation3<f64>> = items
            .iter()
            .map(|x| view(x.0).camera.pose.rotation)
            .collect();
        let n = items.len();
        // 중심 산포 고윳값 (정답 중심 기준)
        let mu = tc.iter().sum::<Vector3<f64>>() / n as f64;
        let mut cov = Matrix3::zeros();
        for c in &tc {
            cov += (c - mu) * (c - mu).transpose();
        }
        let ev = cov.symmetric_eigen();
        let mut l: Vec<f64> = ev
            .eigenvalues
            .iter()
            .map(|x| (x / n as f64).sqrt())
            .collect();
        l.sort_by(f64::total_cmp);
        let gz: Vec<f64> = gps.iter().zip(&tc).map(|(g, t)| g.z - t.z).collect();
        let gh: Vec<f64> = gps
            .iter()
            .zip(&tc)
            .map(|(g, t)| ((g.x - t.x).powi(2) + (g.y - t.y).powi(2)).sqrt())
            .collect();
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        eprintln!("=== 슬롯 {slot}: 사진 {n}, 정답 중심 표준편차 축별(작은→큰) {:.2} {:.2} {:.2} m, 고윳값 비 최소/최대 {:.4} 중간/최대 {:.4}", l[0], l[1], l[2], l[0] / l[2], l[1] / l[2]);
        eprintln!(
            "GPS-정답 중심: 수평 rms {:.2} m, 높이 rms {:.2} m",
            rms(&gh),
            rms(&gz)
        );
        let show = |name: &str, sim: Option<Similarity>| match sim {
            Some(s) => {
                let (a, v, m) = tilt(&s, &poses, &tr);
                eprintln!(
                    "{name:<34} 평균 기울기 {a:.3} deg {v:?} | 평균 제거 중앙 {m:.3} | 배율 {:.4}",
                    s.s
                );
            }
            None => eprintln!("{name:<34} 실패"),
        };
        let base = robust_similarity(&src, &gps, 3, 3.0);
        let inl = base.as_ref().map(|b| b.1.clone()).unwrap_or(vec![true; n]);
        eprintln!("정상 {} / {}", inl.iter().filter(|&&b| b).count(), n);
        show("기본(강건, GPS)", base.as_ref().map(|b| b.0));
        show("트리밍 없음(GPS)", umeyama(&src, &gps));
        show("정답 중심(잡음 0)", umeyama(&src, &tc));
        let mix1: Vec<Vector3<f64>> = gps
            .iter()
            .zip(&tc)
            .map(|(g, t)| Vector3::new(g.x, g.y, t.z))
            .collect();
        let mix2: Vec<Vector3<f64>> = gps
            .iter()
            .zip(&tc)
            .map(|(g, t)| Vector3::new(t.x, t.y, g.z))
            .collect();
        show("GPS 수평 + 정답 높이", umeyama(&src, &mix1));
        show("정답 수평 + GPS 높이", umeyama(&src, &mix2));
        // 모의 잡음: 정답 + N(0, sigma) 20회 평균 기울기
        for sg in [1.0, 2.0] {
            let mut rng = Rng(0x9E3779B97F4A7C15 ^ (*slot as u64 + 1));
            let mut angs = vec![];
            for _ in 0..20 {
                let d: Vec<Vector3<f64>> = tc
                    .iter()
                    .map(|t| t + Vector3::new(rng.n(), rng.n(), rng.n()) * sg)
                    .collect();
                if let Some(s) = umeyama(&src, &d) {
                    angs.push(tilt(&s, &poses, &tr).0);
                }
            }
            let m = angs.iter().sum::<f64>() / angs.len() as f64;
            angs.sort_by(f64::total_cmp);
            eprintln!(
                "정답 중심 + 모의 잡음 {sg} m (20회)   평균 기울기 평균 {m:.3} deg, 최대 {:.3}",
                angs[angs.len() - 1]
            );
        }
        // 연직 고정(사진 회전 평균에서 위 방향)
        let rots: Vec<Rotation3<f64>> = poses.iter().map(|p| p.rotation).collect();
        let up = up_from_rotations(&rots);
        eprintln!("up_from_rotations: {up:?}");
        if let Some(u) = up {
            let pick = |v: &[Vector3<f64>]| -> Vec<Vector3<f64>> {
                v.iter()
                    .zip(&inl)
                    .filter(|(_, &k)| k)
                    .map(|(x, _)| *x)
                    .collect()
            };
            show(
                "연직 고정(GPS, 정상 대응)",
                similarity_fixed_up(&pick(&src), &pick(&gps), &u),
            );
            show("연직 고정(정답 중심)", similarity_fixed_up(&src, &tc, &u));
        }
        // 실제 파이프라인 정렬 직후
        if let Some((_, ps)) = post.iter().find(|(r, _)| r == slot) {
            let es: Vec<Rotation3<f64>> = ps
                .iter()
                .map(|(g, p)| view(*g).camera.pose.rotation.inverse() * p.rotation)
                .collect();
            let me = mean_rot(&es);
            eprintln!(
                "파이프라인 정렬 직후(구역 {slot}) 평균 기울기 {:.3} deg {:?}",
                deg(&me),
                rv(&me)
            );
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
