//! 초벌-정밀 높이 차(`preview_vs_refined`)를 구역·비행 방향 위치(x, y 구간)별로 쪼개 보는 진단.
//! 쌍은 검증과 같게 만든다: 초벌 점마다 수평 2 m 안의 수평 최근접 정밀 점. 기본 동작은 바꾸지 않는다.
//! 실행: `SKYLENS_SPLIT_SEED=3 cargo test --release -p skylens-stream --test preview_split -- --ignored --nocapture`
//! SKYLENS_SPLIT_OUT=<이미 만든 출력 폴더> 이면 다시 돌리지 않고 그 폴더만 분석한다(같은 시드 필요).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::nalgebra::Point3;
use skylens_core::synth::{Scene, SceneConfig};

const RADIUS: f64 = 2.0;
const BINS: usize = 6;

fn med(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn files(dir: &Path) -> Vec<(usize, PathBuf)> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        // preview_{k:02}_*.ply / refined_{k:02}_*.ply
        if let Some(k) = n.split('_').nth(1).and_then(|s| s.parse().ok()) {
            if n.ends_with(".ply") {
                out.push((k, e.path()));
            }
        }
    }
    out.sort();
    out
}

fn load(path: &Path, scene: &Scene) -> Vec<[f64; 3]> {
    let shift = scene.to_first_gps_frame(&Point3::new(0.0, 0.0, 0.0)).coords;
    skylens_core::ply::read_ply_file(path)
        .unwrap()
        .points
        .iter()
        .map(|p| {
            [
                p.xyz[0] as f64 - shift.x,
                p.xyz[1] as f64 - shift.y,
                p.xyz[2] as f64 - shift.z,
            ]
        })
        .filter(|p| p.iter().all(|v| v.is_finite()))
        .collect()
}

struct Pair {
    q: [f64; 3],
    dz: f64, // 초벌 - 정밀
    hd: f64,
    hp: f64, // 초벌 점의 표면 상대 높이
    hr: f64, // 짝 정밀 점의 표면 상대 높이
}

fn pairs(scene: &Scene, pv: &[[f64; 3]], rf: &[[f64; 3]]) -> Vec<Pair> {
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let cell = |p: &[f64; 3]| {
        (
            (p[0] / RADIUS).floor() as i64,
            (p[1] / RADIUS).floor() as i64,
        )
    };
    for (i, p) in rf.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(i);
    }
    let mut out = Vec::new();
    for q in pv {
        let (cx, cy) = cell(q);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &i in grid.get(&(cx + dx, cy + dy)).map_or(&[][..], |v| v) {
                    let d = ((q[0] - rf[i][0]).powi(2) + (q[1] - rf[i][1]).powi(2)).sqrt();
                    if d <= RADIUS && best.is_none_or(|b| d < b.0) {
                        best = Some((d, i));
                    }
                }
            }
        }
        if let Some((d, i)) = best {
            let r = rf[i];
            out.push(Pair {
                q: *q,
                dz: q[2] - r[2],
                hd: d,
                hp: q[2] - scene.surface_height(q[0], q[1]),
                hr: r[2] - scene.surface_height(r[0], r[1]),
            });
        }
    }
    out
}

fn cloud_stats(name: &str, k: usize, c: &[[f64; 3]], scene: &Scene) {
    let mut h: Vec<f64> = c
        .iter()
        .map(|p| p[2] - scene.surface_height(p[0], p[1]))
        .collect();
    let n = h.len();
    let far = h.iter().filter(|v| v.abs() > 5.0).count();
    let hi = h.iter().filter(|v| **v > 5.0).count();
    h.sort_by(f64::total_cmp);
    let q = |f: f64| h[((n as f64 - 1.0) * f) as usize];
    eprintln!(
        "split cloud {name} region {k} n {n} h_p05 {:.2} p25 {:.2} p50 {:.2} p75 {:.2} p95 {:.2} frac_abs_gt5 {:.4} frac_gt5 {:.4}",
        q(0.05), q(0.25), q(0.5), q(0.75), q(0.95), far as f64 / n as f64, hi as f64 / n as f64
    );
}

/// 최소제곱 평면 dz = a + b*x + c*y (x, y 는 평균을 뺀 값).
fn plane(ps: &[&Pair]) -> (f64, f64, f64) {
    let n = ps.len() as f64;
    let mx = ps.iter().map(|p| p.q[0]).sum::<f64>() / n;
    let my = ps.iter().map(|p| p.q[1]).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy, mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in ps {
        let (x, y) = (p.q[0] - mx, p.q[1] - my);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxz += x * p.dz;
        syz += y * p.dz;
        sz += p.dz;
    }
    let det = sxx * syy - sxy * sxy;
    (
        sz / n,
        (sxz * syy - syz * sxy) / det,
        (syz * sxx - sxz * sxy) / det,
    )
}

fn row(label: &str, ps: &[&Pair]) {
    if ps.is_empty() {
        eprintln!("  {label} n 0");
        return;
    }
    let mut a: Vec<f64> = ps.iter().map(|p| p.dz.abs()).collect();
    let mut s: Vec<f64> = ps.iter().map(|p| p.dz).collect();
    let mut hp: Vec<f64> = ps.iter().map(|p| p.hp).collect();
    let mut hr: Vec<f64> = ps.iter().map(|p| p.hr).collect();
    let mut hd: Vec<f64> = ps.iter().map(|p| p.hd).collect();
    let gt2 = ps.iter().filter(|p| p.dz.abs() > 2.0).count() as f64 / ps.len() as f64;
    eprintln!(
        "  {label} n {} med_absdz {:.3} med_dz {:.3} frac_absdz_gt2 {:.3} med_hp {:.2} med_hr {:.2} med_hdist {:.2}",
        ps.len(), med(&mut a), med(&mut s), gt2, med(&mut hp), med(&mut hr), med(&mut hd)
    );
}

#[test]
#[ignore]
fn preview_refined_split() {
    let seed: u64 = std::env::var("SKYLENS_SPLIT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let scene = Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    });
    let reuse = std::env::var_os("SKYLENS_SPLIT_OUT").map(PathBuf::from);
    let root = std::env::temp_dir().join(format!("skylens_split_{}", std::process::id()));
    let output = match reuse {
        Some(o) => o,
        None => {
            let _ = std::fs::remove_dir_all(&root);
            let (input, output) = (root.join("in"), root.join("out"));
            scene.write_dataset(&input).unwrap();
            let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
                .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
                .output()
                .unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            let v = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
                .args(["verify", output.to_str().unwrap()])
                .output()
                .unwrap();
            eprintln!("{}", String::from_utf8_lossy(&v.stdout));
            output
        }
    };
    eprintln!("split seed {seed} out {}", output.display());
    let pf: HashMap<usize, PathBuf> = files(&output.join("preview")).into_iter().collect();
    let rf: HashMap<usize, PathBuf> = files(&output.join("refined")).into_iter().collect();
    let mut ks: Vec<usize> = pf.keys().copied().collect();
    ks.sort();
    for k in ks {
        let Some(rp) = rf.get(&k) else { continue };
        let (pv, rv) = (load(&pf[&k], &scene), load(rp, &scene));
        cloud_stats("preview", k, &pv, &scene);
        cloud_stats("refined", k, &rv, &scene);
        let ps = pairs(&scene, &pv, &rv);
        let all: Vec<&Pair> = ps.iter().collect();
        eprintln!(
            "split region {k} pairs {} of {} preview points",
            ps.len(),
            pv.len()
        );
        row("all", &all);
        let (a, b, c) = plane(&all);
        eprintln!("  plane_fit dz = {a:.3} + {b:.4}*dx + {c:.4}*dy  (per m)");
        // 이상값 층 제외: |표면 상대 높이| 가 5 m 이하인 쌍만.
        let core: Vec<&Pair> = ps
            .iter()
            .filter(|p| p.hp.abs() <= 5.0 && p.hr.abs() <= 5.0)
            .collect();
        row("both_within5m_of_surface", &core);
        let mixed: Vec<&Pair> = ps.iter().filter(|p| (p.hp > 0.5) != (p.hr > 0.5)).collect();
        row(
            "layer_mixed(one side >0.5m above surface, other not)",
            &mixed,
        );
        for (axis, name) in [(0usize, "x"), (1, "y")] {
            let lo = ps.iter().map(|p| p.q[axis]).fold(f64::INFINITY, f64::min);
            let hi = ps
                .iter()
                .map(|p| p.q[axis])
                .fold(f64::NEG_INFINITY, f64::max);
            let w = (hi - lo) / BINS as f64;
            for b in 0..BINS {
                let (l, h) = (lo + w * b as f64, lo + w * (b + 1) as f64);
                let sel: Vec<&Pair> = ps
                    .iter()
                    .filter(|p| p.q[axis] >= l && (p.q[axis] < h || b == BINS - 1))
                    .collect();
                row(&format!("bin_{name}{b} [{l:.0},{h:.0})"), &sel);
            }
        }
    }
    if std::env::var_os("SKYLENS_SPLIT_OUT").is_none()
        && std::env::var_os("SKYLENS_SPLIT_KEEP").is_none()
    {
        let _ = std::fs::remove_dir_all(&root);
    }
}
