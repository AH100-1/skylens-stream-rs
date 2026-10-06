//! 다구역 실행(`--span 12 --coarse-back off`)에서 초벌 출력(포즈 파일)이 정밀 출력과 같은 최신 좌표계에
//! 놓이는지 합성 정답과 비교한다. 정밀 출력은 새 정밀 구역이 나올 때마다 누적 재정렬 변환을 받는다.
//! 초벌 출력도 같은 누적 변환을 받아야 하므로, 구역별 초벌 카메라 중심의 정답 대비 거리(출력 좌표,
//! 정답 원점과 첫 GPS 원점의 평행 이동만 보정)와 같은 사진의 정밀 중심과의 거리를 숫자로 단언한다.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::poses_io::PosesFile;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_realign_{tag}_{}", std::process::id()));
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

fn cli(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0, f64::max)
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

/// 정답: 이름 → 카메라 중심(정답 좌표).
fn truth_centers(input: &Path) -> BTreeMap<String, [f64; 3]> {
    let mut m = BTreeMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, t) = (&n[..9], &n[9..12]);
        let c = [0, 1, 2].map(|j| -(r[j] * t[0] + r[3 + j] * t[1] + r[6 + j] * t[2]));
        m.insert(f[0].to_string(), c);
    }
    m
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}

/// 구역별 (정렬됨, 초벌 정답 대비 거리들, 정밀 정답 대비 거리들, 초벌-정밀 거리들).
type Row = (bool, Vec<f64>, Vec<f64>, Vec<f64>);

fn measure() -> Vec<Row> {
    let t = TempDir::new("off");
    let (scene, out) = (t.0.join("scene"), t.0.join("out"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let (code, so, se) = cli(&["synth", scene_s, "320", "180"]);
    assert_eq!(code, 0, "{so}{se}");
    let args = [
        "run",
        scene_s,
        out_s,
        "--span",
        "12",
        "--coarse-back",
        "off",
        "--max-features",
        "800",
        "--dense-width",
        "96",
        "--hfov",
        "65",
        "--ba-iters",
        "15",
    ];
    let (code, so, se) = cli(&args);
    assert_eq!(code, 0, "{so}{se}");
    let truth = truth_centers(&scene);
    let shift = truth_to_output_shift(&scene);
    let tc = |name: &str| {
        let c = truth[name];
        [0, 1, 2].map(|i| c[i] + shift[i])
    };
    let mut refined = BTreeMap::new();
    let mut n = 0;
    while let Ok(txt) = std::fs::read_to_string(out.join(format!("poses/refined_{n:02}.json"))) {
        n += 1;
        for e in PosesFile::from_json(&txt).unwrap().poses {
            refined.entry(e.name.clone()).or_insert(e);
        }
    }
    let mut rows = Vec::new();
    for k in 0..n {
        let txt = std::fs::read_to_string(out.join(format!("poses/preview_{k:02}.json"))).unwrap();
        let pf = PosesFile::from_json(&txt).unwrap();
        let (mut p_truth, mut r_truth, mut p_ref) = (Vec::new(), Vec::new(), Vec::new());
        for e in &pf.poses {
            let Some(r) = refined.get(&e.name) else {
                continue;
            };
            p_truth.push(dist(e.center, tc(&e.name)));
            r_truth.push(dist(r.center, tc(&e.name)));
            p_ref.push(dist(e.center, r.center));
        }
        rows.push((pf.aligned, p_truth, r_truth, p_ref));
    }
    rows
}

/// 구역 번호별 상한(실측 최악값 x 1.2): (초벌-정답 중앙, 최대, 정밀-정답 중앙, 최대, 초벌-정밀 중앙, 최대).
/// 실측(4코어 측정 기계, 4 단계 모두 같은 장면). 구역 1, 2 순서.
/// - 초벌에 누적 재정렬을 안 이을 때 구역 1: 초벌-정답 3.678/6.348, 초벌-정밀 3.371/5.389.
/// - 이을 때 구역 1: 초벌-정답 3.991/6.463, 초벌-정밀 3.558/5.492. 구역 2(최신)는 변환이 없어 같다: 1.887/2.737, 1.572/2.295.
const LIMITS: [(usize, [f64; 6]); 2] = [
    (1, [4.80, 7.80, 0.77, 1.21, 4.30, 6.60]),
    (2, [2.30, 3.30, 1.06, 1.21, 1.90, 2.80]),
];

#[test]
fn preview_follows_cumulative_realign() {
    let rows = measure();
    assert!(rows.len() >= 3, "구역 수 {}", rows.len());
    for (k, (aligned, p_truth, r_truth, p_ref)) in rows.iter().enumerate() {
        if !aligned || p_truth.is_empty() {
            continue;
        }
        let got = [
            median(p_truth),
            max(p_truth),
            median(r_truth),
            max(r_truth),
            median(p_ref),
            max(p_ref),
        ];
        eprintln!(
            "PREVIEW_REALIGN region {k} n {} preview_truth {:.3}/{:.3} refined_truth {:.3}/{:.3} preview_refined {:.3}/{:.3}",
            p_truth.len(),
            got[0], got[1], got[2], got[3], got[4], got[5]
        );
        if let Some((_, lim)) = LIMITS.iter().find(|(r, _)| *r == k) {
            for (g, l) in got.iter().zip(lim) {
                assert!(g <= l, "구역 {k}: {got:?} 가 상한 {lim:?} 초과");
            }
        }
    }
}
