//! 정밀 BA 시작점 위치 다듬기 스위치(`stage_diag_refined_start`)의 끔/켬을 잡음 시드별로 잰다(측정만, 기본 동작 불변).
//!
//! 단구역(120장, `tilt_stages` 와 같은 설정)을 시드마다 끔/켬으로 돌리고, 기본 경로(기본 장면·기본 설정, 구역 여러 개)도
//! 한 번씩 돌린다. 지표: 위 방향 오차, 카메라별 회전 오차 중앙, 정밀 중심 오차 중앙/최대, 정밀 재투영, 밀집 점 표면 거리,
//! 등록 수, verify 통과, 실행 시간.
//! 환경변수: `RSS_SEEDS`(기본 `1,2,3,4`, 단구역 시드 목록, 빈 문자열이면 건너뜀), `RSS_DEFAULT`(기본 `1`, 기본 경로 시드, 빈 문자열이면 건너뜀).
//! `cargo test --release -p skylens-stream --test refine_start_seeds -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use skylens_core::align::up_from_rotations;
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::math::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};
use skylens_core::pipeline::{run_pipeline, stage_diag_refined_start, PipelineConfig};
use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::verify_dir;

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

fn angle_deg(r: &Rotation3<f64>) -> f64 {
    UnitQuaternion::from_rotation_matrix(r).angle().to_degrees()
}

fn geodetic_of(f: &[&str]) -> Geodetic {
    let n: Vec<f64> = f.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
}

#[derive(Clone, Debug, Default)]
struct Row {
    up: f64,
    rot: [f64; 3],
    c_med: f64,
    c_max: f64,
    rms: f64,
    s_med: f64,
    s_p95: f64,
    reg: usize,
    verify: String,
    secs: f64,
}

/// 한 번 돌려 지표를 잰다. `default_path` 면 기본 장면·기본 설정, 아니면 단구역 설정.
fn run_one(seed: u64, on: bool, default_path: bool) -> Row {
    let dir = std::env::temp_dir().join(format!(
        "skylens_rss_{}_{seed}_{on}_{default_path}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let (scene_dir, out) = (dir.join("scene"), dir.join("out"));
    for d in ["preview", "refined", "snapshots"] {
        std::fs::create_dir_all(out.join(d)).unwrap();
    }
    let mut sc = SceneConfig::default();
    if !default_path {
        sc.width = 320;
        sc.height = 180;
    }
    sc.seed = seed;
    let scene = Scene::new(sc);
    scene.write_dataset(&scene_dir).unwrap();
    let (dc, pc) = if default_path {
        (DatasetConfig::default(), PipelineConfig::default())
    } else {
        (
            DatasetConfig {
                stride: 2,
                span: 48,
                ovl: 2,
                ..DatasetConfig::default()
            },
            PipelineConfig {
                max_features: 800,
                dense_width: 96,
                hfov_deg: 65.0,
                ..PipelineConfig::default()
            },
        )
    };
    let ds = load_dataset(&scene_dir, dc).unwrap();
    skylens_core::dense::view_diag_enable();
    stage_diag_refined_start(on);
    let t0 = Instant::now();
    let res = run_pipeline(&ds, &pc, &out).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    stage_diag_refined_start(false);
    let recs = skylens_core::dense::view_diag_take();

    let o = std::fs::read_to_string(scene_dir.join("truth/origin.txt")).unwrap();
    let to = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gtxt = std::fs::read_to_string(scene_dir.join("gps.txt")).unwrap();
    let first: Vec<&str> = gtxt.lines().next().unwrap().split_whitespace().collect();
    let d = geodetic_to_enu(&to, &geodetic_of(&first[1..4]));
    let shift = Vector3::new(d.x, d.y, d.z);

    // 정밀 중심 오차: 출력 카메라 중심(동-북-위) 대 정답 + 이동량.
    let by_name: HashMap<&str, _> = scene.views.iter().map(|s| (s.name.as_str(), s)).collect();
    let mut ce: Vec<f64> = res
        .centers
        .iter()
        .map(|(n, c)| {
            let t = by_name[n.as_str()].camera.pose.center().coords + shift;
            (Vector3::new(c[0], c[1], c[2]) - t).norm()
        })
        .collect();
    let c_max = ce.iter().copied().fold(0.0, f64::max);
    let c_med = pct(&mut ce, 0.5);

    // 위 방향·회전: 구역 기록마다 구한 뒤 구역 평균(단구역이면 한 값).
    let (mut ups, mut rots): (Vec<f64>, Vec<Vec<f64>>) = (vec![], vec![vec![]; 3]);
    for rec in &recs {
        let n = rec.cams.len();
        let (mut tr, mut tc, mut cam) = (vec![], vec![], vec![]);
        for v in 0..n {
            let g = rec.src[v];
            let name = ds.positions[g / 3].images[g % 3]
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let sv = by_name[name.as_str()];
            tc.push(Point3::from(sv.camera.pose.center().coords + shift));
            tr.push(sv.camera.pose.rotation);
            cam.push(g % 3);
        }
        let rs: Vec<Rotation3<f64>> = rec.cams.iter().map(|c| c.pose.rotation).collect();
        let cs: Vec<Point3<f64>> = rec.cams.iter().map(|c| c.pose.center()).collect();
        let a = kabsch(&cs, &tc);
        let u = up_from_rotations(&rs).expect("up");
        let q = Rotation3::rotation_between(&u, &(a.inverse() * Vector3::z()))
            .unwrap_or_else(Rotation3::identity);
        ups.push(angle_deg(&q));
        for v in 0..n {
            rots[cam[v]].push(angle_deg(&(tr[v].inverse() * rs[v] * a.inverse())));
        }
    }
    let up = ups.iter().sum::<f64>() / ups.len() as f64;
    let mut rot = [0.0; 3];
    for c in 0..3 {
        rot[c] = pct(&mut rots[c], 0.5);
    }

    // 밀집 점 표면 거리.
    let mut sd = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(out.join("refined"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ply"))
        .collect();
    files.sort();
    for f in files {
        for p in read_ply_file(f).unwrap().points {
            let (x, y, z) = (
                p.xyz[0] as f64 - shift.x,
                p.xyz[1] as f64 - shift.y,
                p.xyz[2] as f64 - shift.z,
            );
            sd.push((z - scene.surface_height(x, y)).abs());
        }
    }
    let reg: usize = res.regions.iter().map(|r| r.registered).sum();
    let rms = res.regions.iter().map(|r| r.refined_rms).sum::<f64>() / res.regions.len() as f64;
    let rep = verify_dir(Path::new(&out));
    let pass = rep.items.iter().filter(|i| i.pass).count();
    let row = Row {
        up,
        rot,
        c_med,
        c_max,
        rms,
        s_med: pct(&mut sd, 0.5),
        s_p95: pct(&mut sd, 0.95),
        reg,
        verify: format!("{pass}/{}", rep.items.len()),
        secs,
    };
    let _ = std::fs::remove_dir_all(&dir);
    row
}

fn line(tag: &str, seed: u64, on: bool, r: &Row) {
    println!(
        "ROW {tag} seed {seed} {} | up {:.3} | rot {:.3}/{:.3}/{:.3} | center {:.4}/{:.4} | rms {:.3} | surf {:.4}/{:.4} | reg {} | verify {} | {:.0}s",
        if on { "켬" } else { "끔" },
        r.up, r.rot[0], r.rot[1], r.rot[2], r.c_med, r.c_max, r.rms, r.s_med, r.s_p95, r.reg, r.verify, r.secs
    );
}

fn seeds(var: &str, default: &str) -> Vec<u64> {
    std::env::var(var)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

#[test]
#[ignore = "시드당 끔/켬 두 번씩, 측정용"]
fn refine_start_seeds() {
    for seed in seeds("RSS_SEEDS", "1,2,3,4") {
        for on in [false, true] {
            let r = run_one(seed, on, false);
            line("single", seed, on, &r);
            assert!(r.up.is_finite() && r.reg > 0);
        }
    }
    for seed in seeds("RSS_DEFAULT", "1") {
        for on in [false, true] {
            let r = run_one(seed, on, true);
            line("default", seed, on, &r);
            assert!(r.up.is_finite() && r.reg > 0);
        }
    }
}
