//! 합성 단구역(40 위치 × 3 = 120 장) 정밀 포즈 정확도: 닮음 정렬 뒤 회전·중심 오차, 관측별 재투영 잔차,
//! 점별 삼각측량 각 분포. `measure_*` 는 후보별 수치를 찍는 측정(`--ignored --nocapture`)이고,
//! `refined_pose_accuracy_bounds` 는 기본 설정의 상한을 단언한다.

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::{Matrix3, Rotation3, Vector3};
use skylens_core::pipeline::{run_pipeline, BaRefine, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return f64::NAN;
    }
    v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
}

struct Stats {
    n: usize,
    rot_med: f64,
    rot_max: f64,
    ctr_med: f64,
    ctr_max: f64,
    res: [f64; 3],
    tri: [f64; 3],
    rms_note: String,
}

fn run(tag: &str, ba: BaRefine) -> Stats {
    let root = std::env::temp_dir().join(format!("skylens_pose_acc_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    let scene = Scene::new(SceneConfig {
        positions: 40,
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    scene.write_dataset(&input).unwrap();
    let ds = load_dataset(
        &input,
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
        ba_refine: ba,
        ..PipelineConfig::default()
    };
    let res = run_pipeline(&ds, &cfg, &output).unwrap();
    assert_eq!(res.regions.len(), 1, "단구역이어야 한다");
    let mut est = Vec::new();
    let mut tru = Vec::new();
    let mut tr_rot = Vec::new();
    for (name, p) in &res.poses {
        let v = scene.views.iter().find(|v| &v.name == name).unwrap();
        est.push(p.center().coords);
        tru.push(scene.to_first_gps_frame(&v.camera.pose.center()).coords);
        tr_rot.push(v.camera.pose.rotation);
    }
    // 정렬: 회전은 사진별 R_추정ᵀ·R_정답 의 화음 평균(SVD 사영), 축척·이동은 그 회전을 고정한 중심 최소제곱.
    // 중심만으로 맞추면 비행선이 거의 직선이라 축 둘레 회전이 정해지지 않는다.
    let mut m = Matrix3::zeros();
    for (i, (_, p)) in res.poses.iter().enumerate() {
        m += p.rotation.matrix().transpose() * tr_rot[i].matrix();
    }
    let svd = m.svd(true, true);
    let mut d = Matrix3::identity();
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let qt = Rotation3::from_matrix_unchecked(u * d * vt); // = Qᵀ
    let q = qt.inverse();
    let n = est.len() as f64;
    let (me, mt) = (
        est.iter().sum::<Vector3<f64>>() / n,
        tru.iter().sum::<Vector3<f64>>() / n,
    );
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..est.len() {
        let a = q * (est[i] - me);
        num += a.dot(&(tru[i] - mt));
        den += a.norm_squared();
    }
    let sc = num / den;
    let (mut rot, mut ctr) = (Vec::new(), Vec::new());
    for (i, (_, p)) in res.poses.iter().enumerate() {
        let r = p.rotation * qt;
        rot.push((r * tr_rot[i].inverse()).angle().to_degrees());
        let c: Vector3<f64> = sc * (q * (est[i] - me)) + mt;
        ctr.push((c - tru[i]).norm());
    }
    if std::env::var("POSE_DUMP").is_ok() {
        for (i, (nm, _)) in res.poses.iter().enumerate() {
            eprintln!("  {nm} rot {:.3} ctr {:.3}", rot[i], ctr[i]);
        }
    }
    let mut r2 = res.residuals_px.clone();
    let mut t2 = res.tri_angles_deg.clone();
    let s = Stats {
        n: rot.len(),
        rot_med: pct(&mut rot.clone(), 0.5),
        rot_max: pct(&mut rot, 1.0),
        ctr_med: pct(&mut ctr.clone(), 0.5),
        ctr_max: pct(&mut ctr, 1.0),
        res: [pct(&mut r2, 0.5), pct(&mut r2, 0.95), pct(&mut r2, 1.0)],
        tri: [pct(&mut t2, 0.05), pct(&mut t2, 0.5), pct(&mut t2, 0.95)],
        rms_note: format!("rms {:.3} px, 점 {}", res.regions[0].refined_rms, t2.len()),
    };
    eprintln!(
        "[{tag}] n {} 회전 중앙 {:.4}° 최대 {:.4}° | 중심 중앙 {:.3} m 최대 {:.3} m | 잔차 px 중앙 {:.3} 95% {:.3} 최대 {:.3} | 삼각각 5% {:.2} 중앙 {:.2} 95% {:.2} | {}",
        s.n, s.rot_med, s.rot_max, s.ctr_med, s.ctr_max, s.res[0], s.res[1], s.res[2],
        s.tri[0], s.tri[1], s.tri[2], s.rms_note
    );
    let _ = std::fs::remove_dir_all(&root);
    s
}

fn base() -> BaRefine {
    BaRefine::default()
}

#[test]
#[ignore]
fn measure_candidates() {
    use skylens_core::ba::Loss;
    let b = base();
    run("기본", b);
    // 내부 매개변수: 시험 장면은 hfov 65°·왜곡 0 이고 파이프라인도 같은 값을 쓰므로 정답과의 차는 0 이다.
    run(
        "손실 제곱",
        BaRefine {
            loss: Loss::Squared,
            ..b
        },
    );
    run(
        "손실 Huber 1",
        BaRefine {
            loss: Loss::Huber(1.0),
            ..b
        },
    );
    run(
        "손실 Cauchy 1",
        BaRefine {
            loss: Loss::Cauchy(1.0),
            ..b
        },
    );
    run(
        "손실 Cauchy 2",
        BaRefine {
            loss: Loss::Cauchy(2.0),
            ..b
        },
    );
    run("라운드 2", BaRefine { rounds: 2, ..b });
    run("라운드 3", BaRefine { rounds: 3, ..b });
    run(
        "라운드 3 + Cauchy 1",
        BaRefine {
            rounds: 3,
            loss: Loss::Cauchy(1.0),
            ..b
        },
    );
    run(
        "초점 정제",
        BaRefine {
            free_focal: true,
            ..b
        },
    );
    run(
        "라운드 3 + 초점",
        BaRefine {
            rounds: 3,
            free_focal: true,
            ..b
        },
    );
}

#[test]
#[ignore]
fn measure_default() {
    run("기본", base());
}

/// 기본 설정의 상한(측정값의 약 1.25 배, 근거는 연구 노트 experiments/pose-accuracy.md).
#[test]
fn refined_pose_accuracy_bounds() {
    let s = run("상한", base());
    assert_eq!(s.n, 120);
    assert!(s.rot_med < 1.6, "회전 중앙 {}", s.rot_med);
    assert!(s.rot_max < 12.0, "회전 최대 {}", s.rot_max);
    assert!(s.ctr_med < 0.63, "중심 중앙 {}", s.ctr_med);
    assert!(s.ctr_max < 7.0, "중심 최대 {}", s.ctr_max);
    assert!(s.res[1] < 0.7, "잔차 95% {}", s.res[1]);
    assert!(s.res[2] < 3.2, "잔차 최대 {}", s.res[2]);
    assert!(s.tri[0] > 3.0, "삼각각 5% {}", s.tri[0]);
}
