//! 위치 단위 도착 → 등록 → 초벌 출력 → 정밀 교체 → 재정렬 사건 순서와,
//! 공유 3D 점 닮음 변환(`chain_realign`)의 재정렬 전/후 정답 대비 오차.

use skylens_core::align::Similarity;
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::math::{Rotation3, Vector3};
use skylens_core::pipeline::{run_pipeline_with, PipelineConfig};
use skylens_core::pipeline_stream::StreamOptions;
use skylens_core::progressive::chain_realign;
use skylens_core::stream::{split_regions, Region, Track};
use skylens_core::synth::{Scene, SceneConfig};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// 합성 3구역(0..10, 8..18, 16..26): 구역마다 자기 좌표계(정답 좌표계에 닮음 변환을 걸어 둔 것)에
/// 같은 3D 점을 0.01 m 잡음으로 갖는다. 구역 0 과 2 는 겹치지 않아 구역 1 을 거친 연쇄가 필요하다.
#[test]
fn chain_realign_recovers_latest_frame_through_neighbour() {
    let mut rng = Lcg(7);
    let regions = [
        Region {
            index: 0,
            start: 0,
            lo: 0,
            hi: 10,
        },
        Region {
            index: 1,
            start: 10,
            lo: 8,
            hi: 18,
        },
        Region {
            index: 2,
            start: 20,
            lo: 16,
            hi: 26,
        },
    ];
    // 구역 r 의 자기 좌표 → 정답(최신 구역 2) 좌표 변환.
    let to_truth = [
        Similarity {
            s: 1.04,
            r: Rotation3::from_euler_angles(0.02, -0.03, 0.15),
            t: Vector3::new(6.0, -4.0, 2.0),
        },
        Similarity {
            s: 0.97,
            r: Rotation3::from_euler_angles(-0.01, 0.02, -0.08),
            t: Vector3::new(-3.0, 2.5, -1.0),
        },
        Similarity::identity(),
    ];
    let noise = 0.01;
    let mut tracks: [Vec<Track>; 3] = Default::default();
    // 점 묶음: (보는 구역들, 사진 위치 범위). feat 번호는 묶음 안 점마다 고유.
    let groups: [(&[usize], (usize, usize)); 4] = [
        (&[0], (0, 4)),
        (&[0, 1], (8, 10)),
        (&[1, 2], (16, 18)),
        (&[2], (22, 26)),
    ];
    let mut feat = 0u32;
    let mut truth0: Vec<Vector3<f64>> = Vec::new();
    for (seen_by, (plo, phi)) in groups {
        for _ in 0..40 {
            let x = Vector3::new(rng.next() * 60.0, rng.next() * 40.0, rng.next() * 8.0);
            let obs: Vec<(u32, u32)> = (plo..phi).map(|p| ((3 * p) as u32, feat)).collect();
            feat += 1;
            for &r in seen_by {
                let own = to_truth[r].inverse().apply_point(&x);
                let nz = Vector3::new(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5)
                    * (2.0 * noise);
                tracks[r].push(Track {
                    xyz: own + nz,
                    obs: obs.clone(),
                });
                if r == 0 {
                    truth0.push(x);
                }
            }
        }
    }
    let rms = |sim: Option<&Similarity>| -> f64 {
        let mut s = 0.0;
        for (t, x) in tracks[0].iter().zip(&truth0) {
            let y = sim.map_or(t.xyz, |m| m.apply_point(&t.xyz));
            s += (y - x).norm_squared();
        }
        (s / truth0.len() as f64).sqrt()
    };
    let items: Vec<Option<(Region, &[Track])>> = (0..3)
        .map(|i| Some((regions[i], tracks[i].as_slice())))
        .collect();
    let steps = chain_realign(&items, 2);
    assert_eq!(steps.len(), 2, "{steps:?}");
    assert_eq!((steps[0].region, steps[0].via), (1, 2));
    assert_eq!(
        (steps[1].region, steps[1].via),
        (0, 1),
        "구역 0 은 구역 1 을 거쳐 연쇄"
    );
    assert!(steps.iter().all(|s| s.pairs >= 40), "{steps:?}");
    let before = rms(None);
    let after = rms(Some(&steps[1].total));
    eprintln!(
        "재정렬 전 RMS {before:.4} m, 후 RMS {after:.4} m, 점쌍 {:?}",
        steps.iter().map(|s| s.pairs).collect::<Vec<_>>()
    );
    assert!(before > 5.0, "전 {before}");
    assert!(after < 0.03, "후 {after}");
    // 구역 1 도 최신 좌표계로.
    assert!((steps[0].total.s - 1.0 / to_truth[1].inverse().s).abs() < 0.01);
}

fn count_at(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

/// 3구역 합성 장면 전체 실행: 사건 순서와 최종 정밀 중심의 정답(GPS = 합성 정답) 대비 오차.
#[test]
fn arrival_order_and_realigned_centers() {
    let root = std::env::temp_dir().join(format!("skylens_arrival_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let n_pos = 26;
    Scene::new(SceneConfig {
        positions: n_pos,
        width: 320,
        height: 180,
        ..SceneConfig::default()
    })
    .write_dataset(&root.join("in"))
    .unwrap();
    let ds = load_dataset(
        &root.join("in"),
        DatasetConfig {
            stride: 1,
            span: 10,
            ovl: 2,
            max_skip_run: 2,
        },
    )
    .unwrap();
    let cfg = PipelineConfig {
        max_features: 600,
        dense_width: 80,
        hfov_deg: 65.0,
        ba_iters: 8,
        ..PipelineConfig::default()
    };
    let out = root.join("out");
    let t = std::time::Instant::now();
    let res = run_pipeline_with(&ds, &cfg, &out, StreamOptions::default()).unwrap();
    eprintln!("실행 {:.1} s", t.elapsed().as_secs_f64());
    let regs = split_regions(n_pos, 10, 2);
    assert!(res.regions.len() >= 3, "구역 {}", res.regions.len());

    let manifest = std::fs::read_to_string(out.join("snapshots/manifest.json")).unwrap();
    let kinds: Vec<(String, usize)> = manifest
        .lines()
        .filter(|l| l.contains("\"kind\""))
        .map(|l| {
            let f = |k: &str| {
                let p = l.find(&format!("\"{k}\": ")).unwrap() + k.len() + 4;
                l[p..]
                    .split([',', '}'])
                    .next()
                    .unwrap()
                    .trim()
                    .trim_matches('"')
                    .to_string()
            };
            (f("kind"), f("region").parse().unwrap())
        })
        .collect();
    let pos_of = |kind: &str, region: usize| {
        kinds
            .iter()
            .position(|(k, r)| k == kind && *r == region)
            .unwrap_or_else(|| panic!("사건 없음 {kind} {region}: {kinds:?}"))
    };
    // 위치는 하나씩 차례로 도착한다(오름차순, 한 번씩).
    let arrivals: Vec<usize> = kinds
        .iter()
        .filter(|(k, _)| k == "arrive_position")
        .map(|(_, p)| *p)
        .collect();
    assert!(
        arrivals.windows(2).all(|w| w[1] == w[0] + 1),
        "{arrivals:?}"
    );
    // 구역마다: 자기 위치 도착 → 등록 → 초벌 출력 → 정밀 교체, 그리고 초벌 출력은 다음 구역의 새 위치 도착보다 먼저.
    for (i, r) in regs.iter().enumerate().take(res.regions.len()) {
        let ri = r.index;
        let last_arrival = pos_of("arrive_position", r.hi - 1);
        let (reg, co, rf) = (
            pos_of("register", ri),
            pos_of("coarse_output", ri),
            pos_of("refined_replace", ri),
        );
        assert!(
            last_arrival < reg && reg < co && co < rf,
            "구역 {ri}: {kinds:?}"
        );
        if i > 0 {
            assert!(
                pos_of("coarse_output", regs[i - 1].index) < pos_of("arrive_position", r.hi - 1),
                "구역 {ri}"
            );
        }
        // 다음 구역의 등록은 이전 구역의 등록 뒤.
        if i + 1 < regs.len() {
            assert!(reg < pos_of("register", regs[i + 1].index));
        }
    }
    assert_eq!(kinds.last().unwrap().0, "final");
    // 재정렬: 정밀 교체 뒤에 realign 사건이 있고, 마지막 정밀 구역이 나온 뒤 이전 정밀 구역이 다시 맞춰진다.
    let last_ref = kinds
        .iter()
        .rposition(|(k, _)| k == "refined_replace")
        .unwrap();
    assert!(
        kinds[last_ref..].iter().any(|(k, _)| k == "realign"),
        "{kinds:?}"
    );

    // 보고서 사건 문장: 초벌 출력 직전에 최신 정밀 위로 맞춘 기록, 정밀 재정렬 기록과 잔차.
    let rep = std::fs::read_to_string(out.join("report.json")).unwrap_or_default();
    let log = if rep.is_empty() {
        manifest.clone()
    } else {
        rep
    };
    let n_ref_realign = count_at(&log, "realign refined");
    let n_coarse_on_refined = count_at(&log, "aligned to refined");
    eprintln!("정밀 재정렬 {n_ref_realign} 건, 최신 정밀 위 초벌 정렬 {n_coarse_on_refined} 건");
    for l in log
        .split("\", \"")
        .filter(|l| l.contains("realign") || l.contains("aligned to"))
    {
        eprintln!("  {l}");
    }
    assert!(n_ref_realign >= 1);
    let meds: Vec<f64> = log
        .split("\", \"")
        .filter(|l| l.contains("realign refined"))
        .map(|l| {
            let t = l.split("median ").nth(1).unwrap();
            t.split(' ').next().unwrap().parse().unwrap()
        })
        .collect();
    assert!(meds.iter().all(|m| *m < REALIGN_MEDIAN_BOUND), "{meds:?}");

    // 최종 정밀 중심 대 정답(합성 장면 GPS 는 잡음 없는 위치): 구역 0 이 맡은 앞쪽 위치 포함 전체.
    let mut errs: Vec<(usize, f64)> = Vec::new();
    for (name, c) in &res.centers {
        for (pi, p) in ds.positions.iter().enumerate() {
            for (img, g) in p.images.iter().zip(&p.image_enu) {
                if img
                    .file_stem()
                    .is_some_and(|s| s.to_string_lossy() == *name)
                {
                    let d = ((c[0] - g[0]).powi(2) + (c[1] - g[1]).powi(2) + (c[2] - g[2]).powi(2))
                        .sqrt();
                    errs.push((pi, d));
                }
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut early: Vec<f64> = errs
        .iter()
        .filter(|(p, _)| *p < regs[0].hi)
        .map(|(_, d)| *d)
        .collect();
    let mut all: Vec<f64> = errs.iter().map(|(_, d)| *d).collect();
    let (e_early, e_all) = (med(&mut early), med(&mut all));
    eprintln!(
        "정밀 중심 정답 대비 중앙 오차: 구역 0 위치 {e_early:.4} m, 전체 {e_all:.4} m ({} 장)",
        all.len()
    );
    assert!(e_early < EARLY_BOUND, "구역 0 오차 {e_early}");
    assert!(e_all < ALL_BOUND, "전체 오차 {e_all}");
    let _ = std::fs::remove_dir_all(&root);
}

/// 측정값(구역 0 위치 2.748 m, 전체 2.514 m; 중심에는 재정렬을 적용하지 않음)의 약 1.25 배.
/// 약한 합성 장면이라 절대 오차가 크다.
const EARLY_BOUND: f64 = 3.45;
const ALL_BOUND: f64 = 3.15;
/// 정밀 재정렬 잔차 중앙값(측정 0.280 m)의 상한.
const REALIGN_MEDIAN_BOUND: f64 = 0.6;
