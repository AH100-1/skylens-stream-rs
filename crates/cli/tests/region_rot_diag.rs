//! 두 구역 합성 장면에서 공유 사진별 구역 추정 회전 대비 정답 회전 오차 표(진단, 느림: `--ignored`).
//! 회전 표기: 정답 R_t, 추정 R(둘 다 세계→카메라). 세계 좌표계 오차 E = R_tᵀ R (구역 전체가 기울면 사진마다 같다).

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::{Matrix3, Rotation3, Vector3};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::pipeline_stream::{take_tap, TAP_ON};
use skylens_core::synth::{Scene, SceneConfig};
use std::collections::HashMap;

fn mean_rot(rs: &[Rotation3<f64>]) -> Rotation3<f64> {
    let mut m = Matrix3::zeros();
    for r in rs {
        m += r.matrix();
    }
    let sv = m.svd(true, true);
    let q = sv.u.unwrap() * sv.v_t.unwrap();
    let q = if q.determinant() < 0.0 {
        let mut u = sv.u.unwrap();
        u.column_mut(2).neg_mut();
        u * sv.v_t.unwrap()
    } else {
        q
    };
    Rotation3::from_matrix_unchecked(q)
}

fn deg(r: &Rotation3<f64>) -> f64 {
    r.angle().to_degrees()
}

fn rv(r: &Rotation3<f64>) -> Vector3<f64> {
    r.scaled_axis() * (180.0 / std::f64::consts::PI)
}

fn pct(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q) as usize]
}

#[test]
#[ignore]
fn shared_photo_rotation_table() {
    let root = std::env::temp_dir().join(format!("skylens_rotdiag_{}", std::process::id()));
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
        ..PipelineConfig::default()
    };
    TAP_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    let res = run_pipeline(&ds, &cfg, &root.join("out")).unwrap();
    let (poses, sims) = take_tap();
    eprintln!(
        "regions {} centers {}",
        res.regions.len(),
        res.centers.len()
    );
    let truth = |g: usize| -> Rotation3<f64> {
        let name = ds.positions[g / 3].images[g % 3]
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        scene
            .views
            .iter()
            .find(|v| v.name == name)
            .unwrap()
            .camera
            .pose
            .rotation
    };
    for stage in [
        "구역 좌표계(GPS 정렬 뒤, 연쇄 변환 전)",
        "최종(연쇄 변환 뒤)",
    ] {
        let fin = stage.starts_with("최종");
        let mut by_region: Vec<(usize, HashMap<usize, Rotation3<f64>>)> = vec![];
        for (ri, ps) in &poses {
            let sim = sims.iter().find(|(r, _)| r == ri).map(|(_, s)| s.r);
            by_region.push((
                *ri,
                ps.iter()
                    .map(|(g, p)| {
                        (
                            *g,
                            match (fin, sim) {
                                (true, Some(q)) => p.rotation * q.inverse(),
                                _ => p.rotation,
                            },
                        )
                    })
                    .collect(),
            ));
        }
        by_region.sort_by_key(|x| x.0);
        let (a, b) = (&by_region[0].1, &by_region[1].1);
        eprintln!("=== {stage}: 구역 {} / {}", by_region[0].0, by_region[1].0);
        let mut mean = vec![];
        for (ri, m) in &by_region {
            let es: Vec<Rotation3<f64>> = m.iter().map(|(g, r)| truth(*g).inverse() * *r).collect();
            let me = mean_rot(&es);
            let resid: Vec<f64> = es.iter().map(|e| deg(&(me.inverse() * *e))).collect();
            eprintln!(
                "구역 {ri}: 사진 {} 평균 오차 회전 {:.3} deg 벡터(ENU) {:?} | 평균 제거 뒤 중앙 {:.3} p90 {:.3} 최대 {:.3}",
                m.len(),
                deg(&me),
                rv(&me).iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
                pct(&resid, 0.5),
                pct(&resid, 0.9),
                pct(&resid, 1.0)
            );
            mean.push(me);
        }
        let mut shared: Vec<usize> = a.keys().filter(|g| b.contains_key(g)).copied().collect();
        shared.sort();
        eprintln!("공유 사진 {} 장", shared.len());
        eprintln!("gid 위치 | 오차0(deg) 오차1(deg) 구역간(deg) | 구역간 회전 벡터(deg x y z)");
        let mut rels = vec![];
        let mut ang = vec![];
        for g in &shared {
            let e0 = truth(*g).inverse() * a[g];
            let e1 = truth(*g).inverse() * b[g];
            let rel = e0.inverse() * e1; // 구역0 → 구역1 세계 좌표 회전 차
            ang.push(deg(&rel));
            eprintln!(
                "{:4} {:3} | {:.3} {:.3} {:.3} | {:?}",
                g,
                g / 3,
                deg(&e0),
                deg(&e1),
                deg(&rel),
                rv(&rel)
                    .iter()
                    .map(|x| (x * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
            rels.push(rel);
        }
        let mr = mean_rot(&rels);
        let disp: Vec<f64> = rels.iter().map(|r| deg(&(mr.inverse() * *r))).collect();
        let mut sq: Vec<f64> = ang.iter().map(|x| x * x).collect();
        sq.sort_by(|x, y| y.total_cmp(x));
        let tot: f64 = sq.iter().sum();
        let top5: f64 = sq.iter().take(5).sum();
        eprintln!(
            "구역간 회전 차: 중앙 {:.3} p90 {:.3} 최대 {:.3} | 평균 회전 {:.3} deg {:?} | 평균 제거 뒤 중앙 {:.3} p90 {:.3} 최대 {:.3} | 상위 5장 제곱합 비율 {:.3}",
            pct(&ang, 0.5),
            pct(&ang, 0.9),
            pct(&ang, 1.0),
            deg(&mr),
            rv(&mr).iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            pct(&disp, 0.5),
            pct(&disp, 0.9),
            pct(&disp, 1.0),
            top5 / tot.max(1e-12)
        );
        // 위치(경로 방향)에 따른 선형 변화: 구역간 회전 벡터 각 성분 대 위치 번호 최소제곱.
        let n = shared.len() as f64;
        let xs: Vec<f64> = shared.iter().map(|g| (g / 3) as f64).collect();
        let xm = xs.iter().sum::<f64>() / n;
        for ax in 0..3 {
            let ys: Vec<f64> = rels.iter().map(|r| rv(r)[ax]).collect();
            let ym = ys.iter().sum::<f64>() / n;
            let sxx: f64 = xs.iter().map(|x| (x - xm).powi(2)).sum();
            let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - xm) * (y - ym)).sum();
            let slope = sxy / sxx.max(1e-12);
            let rs: Vec<f64> = xs
                .iter()
                .zip(&ys)
                .map(|(x, y)| (y - ym - slope * (x - xm)).abs())
                .collect();
            let tot: Vec<f64> = ys.iter().map(|y| (y - ym).abs()).collect();
            eprintln!(
                "성분 {ax}: 평균 {ym:+.3} 위치당 기울기 {slope:+.4} deg/위치 | 평균 제거 |편차| 중앙 {:.3} 추세 제거 뒤 {:.3}",
                pct(&tot, 0.5),
                pct(&rs, 0.5)
            );
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
