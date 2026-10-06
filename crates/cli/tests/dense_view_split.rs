//! 밀집 점별 오차를 기준 사진의 '이웃 시점 구성'으로 나눠 본다(단구역, README 첫 명령과 같은 설정).
//!
//! 한 번 돌리면 밀집 단계(`dense::view_diag_*`)가 사진별 이웃 목록과 점마다의 기준 사진을 남긴다.
//! 확대 보충 특징 켬/끔 두 실행을 비교하려면 실행 하나의 이웃 목록을 `VIEW_SPLIT_SAVE=<파일>` 로 저장하고,
//! 다른 실행에서 `VIEW_SPLIT_OTHER=<파일>` 로 읽어 "바뀐 시점/안 바뀐 시점" 표를 낸다.
//! `cargo test --release --test dense_view_split -- --ignored --nocapture` 출력이 연구 노트 표다.
//! 켬은 기본값 그대로 돌리면 되고, 끔은 특징 검출의 확대 문턱을 0 으로 둔 빌드로 따로 돌려 저장했다.
//!
//! 점이 어느 기준 사진에서 나왔는지는 융합 점 순서(기준 사진 순서로 낸다)를 따라가며 점이 현재 기준
//! 사진의 깊이와 재투영·상대 깊이로 맞는지 보고 정하므로 근사다(맞는 사진이 없는 점은 '불명').

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::dense::{view_diag_enable, view_diag_take, ViewDiagRecord};
use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};
use skylens_core::synth::{Scene, SceneConfig};

const CAM: [&str; 3] = ["F", "R", "L"];
/// 깊이 단계가 비용 집계에 쓰는 이웃 수(`dense::SWEEP_NEIGHBORS`).
const DEPTH_NBRS: usize = 3;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_vs_{tag}_{}", std::process::id()));
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

fn geodetic_of(fields: &[&str]) -> Geodetic {
    let n: Vec<f64> = fields.iter().map(|s| s.parse().unwrap()).collect();
    Geodetic {
        lat_deg: n[0],
        lon_deg: n[1],
        alt: n[2],
    }
}

fn truth_to_output_shift(input: &Path) -> [f64; 3] {
    let o = std::fs::read_to_string(input.join("truth/origin.txt")).unwrap();
    let truth_origin = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gps = std::fs::read_to_string(input.join("gps.txt")).unwrap();
    let first: Vec<&str> = gps.lines().next().unwrap().split_whitespace().collect();
    let d = geodetic_to_enu(&truth_origin, &geodetic_of(&first[1..4]));
    [d.x, d.y, d.z]
}

fn load_other(path: &str) -> Vec<Vec<usize>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect()
}

fn dist_row(label: &str, e: &[f64]) -> String {
    let big = e.iter().filter(|&&x| x > 1.0).count();
    format!(
        "| {label} | {} | {:.3} | {:.3} | {} ({:.1}%) |",
        e.len(),
        quant(e, 0.5),
        quant(e, 0.95),
        big,
        100.0 * big as f64 / e.len().max(1) as f64
    )
}

/// 기준 사진 하나의 이웃 구성: 위치(위치 번호·카메라), 앞 3장 중 다른 카메라 수 등.
struct ViewInfo {
    cam: usize,
    pos: usize,
    diff_cam3: usize,
    diff_cam8: usize,
    n_nbrs: usize,
}

#[test]
#[ignore = "약 3분, 측정용: --ignored --nocapture"]
fn dense_view_split() {
    let t = TempDir::new("single");
    let (scene_dir, out) = (t.0.join("scene"), t.0.join("out"));
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
    view_diag_enable();
    run_pipeline(&ds, &pc, &out).unwrap();
    let recs: Vec<ViewDiagRecord> = view_diag_take();
    eprintln!(
        "records {:?}",
        recs.iter().map(|r| r.cams.len()).collect::<Vec<_>>()
    );
    // 정밀 단계가 마지막에 끝난다(초벌은 먼저).
    let rec = recs.last().expect("밀집 기록 없음");
    let n = rec.cams.len();
    assert_eq!(n, 120, "등록 수");
    let shift = truth_to_output_shift(&scene_dir);
    let truth = Scene::new(SceneConfig {
        width: 320,
        height: 180,
        ..SceneConfig::default()
    });
    let err: Vec<f64> = rec
        .points
        .iter()
        .map(|p| {
            let (x, y, z) = (
                p[0] as f64 - shift[0],
                p[1] as f64 - shift[1],
                p[2] as f64 - shift[2],
            );
            (z - truth.surface_height(x, y)).abs()
        })
        .collect();
    let all_p95 = quant(&err, 0.95);
    eprintln!(
        "밀집 점 {} 표면 중앙 {:.3} 95% {:.3}",
        err.len(),
        quant(&err, 0.5),
        all_p95
    );

    // 카메라별 중심 오차(정답 카메라 중심 C = -Rᵀt 와 출력 poses.txt 비교, 원점 차 보정).
    {
        let mut truth_c = std::collections::HashMap::new();
        for l in std::fs::read_to_string(scene_dir.join("truth/cameras.txt"))
            .unwrap()
            .lines()
        {
            let f: Vec<&str> = l.split_whitespace().collect();
            let v: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
            let c: Vec<f64> = (0..3)
                .map(|j| -(v[j] * v[9] + v[3 + j] * v[10] + v[6 + j] * v[11]))
                .collect();
            truth_c.insert(f[0].to_string(), c);
        }
        let mut by_cam: [Vec<f64>; 3] = Default::default();
        for l in std::fs::read_to_string(out.join("poses.txt"))
            .unwrap()
            .lines()
        {
            let f: Vec<&str> = l.split_whitespace().collect();
            let c: Vec<f64> = f[1..4].iter().map(|s| s.parse().unwrap()).collect();
            let tc = &truth_c[f[0]];
            let d = (0..3)
                .map(|k| (c[k] - (tc[k] + shift[k])).powi(2))
                .sum::<f64>()
                .sqrt();
            let cam = CAM
                .iter()
                .position(|n| f[0].contains(&format!("cam{n}")))
                .unwrap();
            by_cam[cam].push(d);
        }
        println!("\n## 카메라별 중심 오차(m)");
        println!("| 카메라 | 장수 | 중앙 | 최대 |");
        println!("|---|---|---|---|");
        for (c, e) in by_cam.iter().enumerate() {
            println!(
                "| cam{} | {} | {:.3} | {:.3} |",
                CAM[c],
                e.len(),
                quant(e, 0.5),
                e.iter().cloned().fold(0.0, f64::max)
            );
        }
    }

    if let Ok(path) = std::env::var("VIEW_SPLIT_SAVE") {
        let s: String = rec
            .neighbors
            .iter()
            .map(|nb| {
                nb.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
                    + "\n"
            })
            .collect();
        std::fs::write(path, s).unwrap();
    }

    // 사진 위치 → (위치 번호, 카메라): 등록 120/120 이면 입력 순서와 같다(src[i] = i).
    let info: Vec<ViewInfo> = (0..n)
        .map(|v| {
            let g = rec.src[v];
            let cam = g % 3;
            let nb = &rec.neighbors[v];
            let dc = |k: usize| {
                nb.iter()
                    .take(k)
                    .filter(|&&j| rec.src[j] % 3 != cam)
                    .count()
            };
            ViewInfo {
                cam,
                pos: g / 3,
                diff_cam3: dc(DEPTH_NBRS),
                diff_cam8: dc(8),
                n_nbrs: nb.len(),
            }
        })
        .collect();

    // 점별 값: 기준 사진, 앞 3장과의 평균 광선 각.
    let ray_deg = |v: usize, j: usize, p: &[f32; 3]| -> f64 {
        let x = skylens_core::math::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        let (a, b) = (rec.cams[v].pose.center() - x, rec.cams[j].pose.center() - x);
        (a.dot(&b) / (a.norm() * b.norm()))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    let mean_angle = |v: usize, p: &[f32; 3]| -> f64 {
        let nb: Vec<usize> = rec.neighbors[v].iter().take(DEPTH_NBRS).copied().collect();
        if nb.is_empty() {
            return f64::NAN;
        }
        nb.iter().map(|&j| ray_deg(v, j, p)).sum::<f64>() / nb.len() as f64
    };

    println!("\n## 이웃 구성별 점 오차 (기준 사진 카메라 x 앞 3장 중 다른 카메라 수)");
    println!("| 기준 카메라 | 다른 카메라 이웃(앞3) | 사진 수 | 점 수 | 표면 중앙 | 표면 95% | >1 m 점 |");
    println!("|---|---|---|---|---|---|---|");
    let mut grp: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
    let mut views_in: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (v, i) in info.iter().enumerate() {
        let _ = v;
        *views_in.entry((i.cam, i.diff_cam3)).or_default() += 1;
    }
    let mut unknown = Vec::new();
    for (k, &v) in rec.point_view.iter().enumerate() {
        if v == usize::MAX {
            unknown.push(err[k]);
            continue;
        }
        grp.entry((info[v].cam, info[v].diff_cam3))
            .or_default()
            .push(err[k]);
    }
    for ((cam, dc), e) in &grp {
        let big = e.iter().filter(|&&x| x > 1.0).count();
        println!(
            "| cam{} | {dc} | {} | {} | {:.3} | {:.3} | {big} ({:.1}%) |",
            CAM[*cam],
            views_in[&(*cam, *dc)],
            e.len(),
            quant(e, 0.5),
            quant(e, 0.95),
            100.0 * big as f64 / e.len() as f64
        );
    }
    println!(
        "기준 사진 불명 점 {} (오차 95% {:.3})",
        unknown.len(),
        quant(&unknown, 0.95)
    );
    println!(
        "이웃 수 분포: {:?}",
        info.iter().fold(BTreeMap::new(), |mut m, i| {
            *m.entry(i.n_nbrs).or_insert(0usize) += 1;
            m
        })
    );
    println!(
        "앞 8장 중 다른 카메라 수 분포: {:?}",
        info.iter().fold(BTreeMap::new(), |mut m, i| {
            *m.entry(i.diff_cam8).or_insert(0usize) += 1;
            m
        })
    );

    // 광선 각 구간별.
    println!("\n## 앞 3장 평균 광선 각 구간별 점 오차");
    println!("| 각(도) | 점 수 | 표면 중앙 | 표면 95% | >1 m 점 |");
    println!("|---|---|---|---|---|");
    let edges = [0.0, 3.0, 5.0, 8.0, 12.0, 20.0, 180.0];
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); edges.len() - 1];
    let mut angles_big = Vec::new();
    let mut angles_all = Vec::new();
    for (k, &v) in rec.point_view.iter().enumerate() {
        if v == usize::MAX {
            continue;
        }
        let a = mean_angle(v, &rec.points[k]);
        if !a.is_finite() {
            continue;
        }
        angles_all.push(a);
        if err[k] > 1.0 {
            angles_big.push(a);
        }
        let b = (0..bins.len()).find(|&b| a < edges[b + 1]).unwrap();
        bins[b].push(err[k]);
    }
    for (b, e) in bins.iter().enumerate() {
        if e.is_empty() {
            continue;
        }
        println!("{}", dist_row(&format!("{}~{}", edges[b], edges[b + 1]), e));
    }
    println!(
        "평균 각 중앙: 전체 {:.1} 도, >1 m 점 {:.1} 도",
        quant(&angles_all, 0.5),
        quant(&angles_big, 0.5)
    );

    // 켬/끔 비교.
    let mut changed3_share = None;
    if let Ok(path) = std::env::var("VIEW_SPLIT_OTHER") {
        let other = load_other(&path);
        assert_eq!(other.len(), n, "다른 실행의 사진 수");
        let diff = |a: &[usize], b: &[usize], k: usize| -> usize {
            let sa: HashSet<usize> = a.iter().take(k).copied().collect();
            let sb: HashSet<usize> = b.iter().take(k).copied().collect();
            sa.difference(&sb).count()
        };
        let d3: Vec<usize> = (0..n)
            .map(|v| diff(&rec.neighbors[v], &other[v], DEPTH_NBRS))
            .collect();
        let d8: Vec<usize> = (0..n)
            .map(|v| diff(&rec.neighbors[v], &other[v], 8))
            .collect();
        println!("\n## 이 실행 대 다른 실행: 시점별 이웃 집합 변화 (사진 {n}장)");
        for (name, d, kmax) in [("앞 3장", &d3, 3usize), ("앞 8장", &d8, 8)] {
            let hist: Vec<usize> = (0..=kmax)
                .map(|c| d.iter().filter(|&&x| x == c).count())
                .collect();
            println!("{name} 중 바뀐 장수별 사진 수 (0..={kmax}): {hist:?}");
        }
        println!("\n| 구분 | 사진 수 | 점 수 | 표면 중앙 | 표면 95% | >1 m 점 |");
        println!("|---|---|---|---|---|---|");
        let mut tail_total = 0usize;
        let mut tail_by: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
        for (name, d) in [("앞 3장", &d3), ("앞 8장", &d8)] {
            for (label, pick) in [
                (
                    "안 바뀜",
                    Box::new(|c: usize| c == 0) as Box<dyn Fn(usize) -> bool>,
                ),
                ("바뀜(>=1)", Box::new(|c: usize| c >= 1)),
                ("바뀜(>=2)", Box::new(|c: usize| c >= 2)),
            ] {
                let views = d.iter().filter(|&&c| pick(c)).count();
                let e: Vec<f64> = rec
                    .point_view
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v != usize::MAX && pick(d[v]))
                    .map(|(k, _)| err[k])
                    .collect();
                let big = e.iter().filter(|&&x| x > 1.0).count();
                println!(
                    "| {name} {label} | {views} | {} | {:.3} | {:.3} | {big} ({:.1}%) |",
                    e.len(),
                    quant(&e, 0.5),
                    quant(&e, 0.95),
                    100.0 * big as f64 / e.len().max(1) as f64
                );
                if name == "앞 3장" && label == "바뀜(>=1)" {
                    changed3_share = Some((big, e.len()));
                }
                let _ = (&mut tail_by, &mut tail_total);
            }
        }
        let total_big = err.iter().filter(|&&x| x > 1.0).count();
        if let Some((b, m)) = changed3_share {
            println!(
                "\n>1 m 점 {total_big} 개 중 앞 3장이 바뀐 기준 사진 몫 {b} ({:.1}%), 그 사진들의 점 몫 {:.1}%",
                100.0 * b as f64 / total_big as f64,
                100.0 * m as f64 / err.len() as f64
            );
        }
    }

    // >1 m 점이 많은 기준 사진 상위.
    println!("\n## >1 m 점이 많은 기준 사진 상위 12");
    println!(
        "| 사진(위치/카메라) | 점 수 | >1 m 점 | 95% | 앞 8 이웃(위치/카메라) | 앞3 평균각(도) |"
    );
    println!("|---|---|---|---|---|---|");
    let mut per: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
    for v in 0..n {
        per.push((v, Vec::new(), Vec::new()));
    }
    for (k, &v) in rec.point_view.iter().enumerate() {
        if v != usize::MAX {
            per[v].1.push(err[k]);
            per[v].2.push(mean_angle(v, &rec.points[k]));
        }
    }
    per.sort_by_key(|(_, e, _)| std::cmp::Reverse(e.iter().filter(|&&x| x > 1.0).count()));
    for (v, e, a) in per.iter().take(12) {
        let big = e.iter().filter(|&&x| x > 1.0).count();
        let nb: Vec<String> = rec.neighbors[*v]
            .iter()
            .map(|&j| format!("{}{}", info[j].pos, CAM[info[j].cam]))
            .collect();
        println!(
            "| {}{} | {} | {big} | {:.3} | {} | {:.1} |",
            info[*v].pos,
            CAM[info[*v].cam],
            e.len(),
            quant(e, 0.95),
            nb.join(" "),
            quant(a, 0.5)
        );
    }

    // 기본(확대 켬) 실측: 밀집 점 62256개(희소 점 덧붙이기 전), 표면 중앙 0.391, 95% 1.188 m.
    // 상한은 95% 실측의 약 1.2 배, 점 수는 실측 +-20%. 끔 빌드에서는 이 값이 달라(61069개, 0.950 m) 켬 기본에서만 맞다.
    assert!(
        rec.points.len() >= 62256 * 4 / 5 && rec.points.len() <= 62256 * 6 / 5,
        "점 수 {}",
        rec.points.len()
    );
    assert!(all_p95 < 1.43, "표면 95% {all_p95}");
}
