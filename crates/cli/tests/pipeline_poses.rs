//! 출력 폴더의 카메라 포즈 파일(`poses/refined_{k:02}.json`, `poses/preview_{k:02}.json`)을 합성 정답과 비교한다.
//!
//! 단구역 장면(README 첫 명령, stride 2 → 40위치 × 3대 = 120장)을 `synth` → `run` 으로 만들고,
//! 정밀 포즈의 회전 오차(도: 중앙·최대; 모든 카메라에 하나의 전역 회전을 최소제곱으로 맞춘 뒤)와
//! 카메라 중심 오차(m; 정답 원점과 첫 GPS 원점의 평행 이동만 보정)를 숫자로 단언한다.
//! 정답은 `truth/cameras.txt`(이름, 내부 파라미터 6개, R 행 우선 9개, t 3개)와 `truth/origin.txt`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use skylens_core::geo::{geodetic_to_enu, Geodetic};
use skylens_core::math::rotation_angle_between;
use skylens_core::nalgebra::{Matrix3, Rotation3};
use skylens_core::poses_io::{rotation_of, PosesFile};

const RUN_OPTS: [&str; 12] = [
    "--span",
    "48",
    "--ovl",
    "2",
    "--max-features",
    "800",
    "--dense-width",
    "96",
    "--hfov",
    "65",
    "--ba-iters",
    "15",
];

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_poses_{tag}_{}", std::process::id()));
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

/// 정답 좌표 → 출력 좌표(첫 GPS 기준) 평행 이동량.
fn truth_to_output_shift(input: &Path) -> [f64; 3] {
    let o = std::fs::read_to_string(input.join("truth/origin.txt")).unwrap();
    let truth_origin = geodetic_of(&o.split_whitespace().collect::<Vec<_>>());
    let gps = std::fs::read_to_string(input.join("gps.txt")).unwrap();
    let first: Vec<&str> = gps.lines().next().unwrap().split_whitespace().collect();
    let first_gps = geodetic_of(&first[1..4]);
    let d = geodetic_to_enu(&truth_origin, &first_gps);
    [d.x, d.y, d.z]
}

/// 정답: 이름 → (회전 세계→카메라, 중심).
fn truth_poses(input: &Path) -> BTreeMap<String, (Rotation3<f64>, [f64; 3])> {
    let mut m = BTreeMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let (r, t) = (&n[..9], &n[9..12]);
        let c = [0, 1, 2].map(|j| -(r[j] * t[0] + r[3 + j] * t[1] + r[6 + j] * t[2]));
        let rot = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(r));
        m.insert(f[0].to_string(), (rot, c));
    }
    m
}

/// `poses/{kind}_*.json` 전부를 읽어 이름 → 포즈 (먼저 나온 파일 우선).
fn read_kind(
    out: &Path,
    kind: &str,
) -> (BTreeMap<String, skylens_core::poses_io::PoseEntry>, usize) {
    let mut files: Vec<_> = std::fs::read_dir(out.join("poses"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&format!("{kind}_")) && n.ends_with(".json"))
        })
        .collect();
    files.sort();
    let mut m = BTreeMap::new();
    for f in &files {
        let pf = PosesFile::from_json(&std::fs::read_to_string(f).unwrap()).unwrap();
        assert_eq!(pf.intrinsics.4, 320, "{}", f.display());
        assert_eq!(pf.intrinsics.5, 180, "{}", f.display());
        for e in pf.poses {
            m.entry(e.name.clone()).or_insert(e);
        }
    }
    (m, files.len())
}

/// 전역 회전 G(출력 세계 → 정답 세계)를 `R_out ≈ R_truth·G` 로 최소제곱 맞춘 뒤의 카메라별 회전 오차(도).
fn rotation_errors(pairs: &[(Rotation3<f64>, Rotation3<f64>)]) -> (Vec<f64>, Vec<f64>) {
    let mut m = Matrix3::zeros();
    for (out, truth) in pairs {
        m += truth.matrix().transpose() * out.matrix();
    }
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut g = u * vt;
    if g.determinant() < 0.0 {
        let mut u2 = u;
        u2.column_mut(2).neg_mut();
        g = u2 * vt;
    }
    let g = Rotation3::from_matrix_unchecked(g);
    let raw = pairs
        .iter()
        .map(|(o, t)| rotation_angle_between(o, t).to_degrees())
        .collect();
    let aligned = pairs
        .iter()
        .map(|(o, t)| rotation_angle_between(&(o * g.inverse()), t).to_degrees())
        .collect();
    (raw, aligned)
}

/// 단구역 합성 장면. 실측(4코어 측정 기계): 회전 오차 정렬 전 중앙 0.652/최대 1.535 도, 정렬 후 중앙 0.388/최대 1.135 도,
/// 중심 오차 중앙 0.312/최대 0.851 m. 상한은 실측 x 1.2.
#[test]
fn refined_pose_rotation_and_center_errors() {
    let t = TempDir::new("single");
    let (scene, out) = (t.0.join("scene"), t.0.join("out"));
    let (scene_s, out_s) = (scene.to_str().unwrap(), out.to_str().unwrap());
    let (code, so, se) = cli(&["synth", scene_s, "320", "180"]);
    assert_eq!(code, 0, "{so}{se}");
    let mut args = vec!["run", scene_s, out_s, "--stride", "2"];
    args.extend(RUN_OPTS);
    let (code, so, se) = cli(&args);
    assert_eq!(code, 0, "{so}{se}");

    let truth = truth_poses(&scene);
    let shift = truth_to_output_shift(&scene);
    let (refined, n_files) = read_kind(&out, "refined");
    assert_eq!(n_files, 1, "단구역은 포즈 파일 1개");
    assert_eq!(refined.len(), 120, "등록 수");
    let (preview, n_prev) = read_kind(&out, "preview");
    assert_eq!(n_prev, 1);
    assert!(preview.len() >= 100, "초벌 포즈 수 {}", preview.len());

    let mut pairs = Vec::new();
    let mut centers = Vec::new();
    for (name, e) in &refined {
        let (tr, tc) = &truth[name];
        let q = e.quat_wxyz;
        let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "쿼터니언 크기 {norm}");
        pairs.push((rotation_of(e), *tr));
        centers.push(
            (0..3)
                .map(|k| (e.center[k] - (tc[k] + shift[k])).powi(2))
                .sum::<f64>()
                .sqrt(),
        );
    }
    let (raw, aligned) = rotation_errors(&pairs);
    let (rm, rx) = (median(&raw), max(&raw));
    let (am, ax) = (median(&aligned), max(&aligned));
    let (cm, cx) = (median(&centers), max(&centers));
    eprintln!(
        "POSES refined n {} rot raw med {rm:.4} max {rx:.4} deg, aligned med {am:.4} max {ax:.4} deg, center med {cm:.3} max {cx:.3} m",
        refined.len()
    );
    assert!(am < 0.47, "정렬 후 회전 오차 중앙 {am} 도");
    assert!(ax < 1.4, "정렬 후 회전 오차 최대 {ax} 도");
    assert!(cm < 0.40, "중심 오차 중앙 {cm} m");
    assert!(cx < 1.02, "중심 오차 최대 {cx} m");

    // 초벌 포즈도 같은 형식이고 회전이 정답에 가깝다(느슨한 기준).
    let pp: Vec<_> = preview
        .iter()
        .map(|(n, e)| (rotation_of(e), truth[n].0))
        .collect();
    let (_, pa) = rotation_errors(&pp);
    eprintln!(
        "POSES preview n {} rot aligned med {:.4} max {:.4} deg",
        pp.len(),
        median(&pa),
        max(&pa)
    );
    // 실측: 정렬 후 중앙 0.491/최대 4.081 도.
    assert!(median(&pa) < 0.6, "초벌 회전 오차 중앙 {}", median(&pa));
    assert!(max(&pa) < 4.9, "초벌 회전 오차 최대 {}", max(&pa));
}
