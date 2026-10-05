//! 기본 경로(인자 없는 synth → run → verify)의 구역별 연직 기울기 진단. 구역 정밀 모델의 사진별 회전을
//! 진단 출력(`SKYLENS_REGION_DIAG`)으로 받아 정답과 비교한다. 오래 걸려 기본으로는 돌리지 않는다.
//! 실행: `cargo test --release -p skylens-stream --test default_tilt -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use skylens_core::nalgebra::{Matrix3, Rotation3, Vector3};
use skylens_core::synth::{Scene, SceneConfig};

fn cli(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(args)
        .env("SKYLENS_REGION_DIAG", "1")
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

/// 정답 회전(세계→카메라), 사진 번호 gid = 고른 위치 * 3 + (F, R, L) 순서.
fn truth_rotations(input: &Path) -> HashMap<String, Matrix3<f64>> {
    let mut m = HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..16].iter().map(|s| s.parse().unwrap()).collect();
        m.insert(f[0].to_string(), Matrix3::from_row_slice(&n));
    }
    m
}

/// 기본 간격(stride 3)으로 고른 사진 목록의 번호 gid 에서 원본 위치 번호는 (gid / 3) * 3.
const STRIDE: usize = 3;

fn image_name(gid: usize) -> String {
    format!("cam{}_{:04}", ["F", "R", "L"][gid % 3], (gid / 3) * STRIDE)
}

/// 회전 행렬들의 평균 회전(극분해)과 단위 행렬 사이의 각(도).
fn mean_rotation_angle(ds: &[Matrix3<f64>]) -> f64 {
    let sum: Matrix3<f64> = ds.iter().sum();
    let svd = sum.svd(true, true);
    let mut r = svd.u.unwrap() * svd.v_t.unwrap();
    if r.determinant() < 0.0 {
        let mut u = svd.u.unwrap();
        u.column_mut(2).neg_mut();
        r = u * svd.v_t.unwrap();
    }
    Rotation3::from_matrix_unchecked(r).angle().to_degrees()
}

#[test]
#[ignore]
fn default_path_region_tilt() {
    let root = std::env::temp_dir().join(format!("skylens_tilt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    let (i, o) = (input.to_str().unwrap(), output.to_str().unwrap());
    // 시드는 환경 변수 SKYLENS_TILT_SEED (기본 1). 기본 합성 설정에서 시드만 바꾼다.
    let seed: u64 = std::env::var("SKYLENS_TILT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let scene = Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    });
    scene.write_dataset(&input).unwrap();
    eprintln!("seed {seed}");
    // 추가 run 인자는 SKYLENS_TILT_RUN_ARGS (공백 구분, 예: "--pair-vote"). 없으면 인자 없는 run.
    let extra = std::env::var("SKYLENS_TILT_RUN_ARGS").unwrap_or_default();
    let mut args = vec!["run", i, o];
    args.extend(extra.split_whitespace());
    let (code, stdout, stderr) = cli(&args);
    assert_eq!(code, 0, "{stderr}");
    for l in stdout.lines().filter(|l| l.starts_with("region ")) {
        eprintln!("{l}");
    }
    let (vcode, vout, _) = cli(&["verify", o]);
    eprintln!("verify exit {vcode}\n{vout}");
    let truth = truth_rotations(&input);
    // 구역마다 마지막으로 출력된 진단(정밀 모델 완성 직후)만 쓴다.
    let mut rots: HashMap<usize, HashMap<usize, (bool, Matrix3<f64>)>> = HashMap::new();
    let mut ctrs: HashMap<usize, HashMap<usize, Vector3<f64>>> = HashMap::new();
    for l in stderr.lines() {
        if l.starts_with("diag align") {
            eprintln!("{l}");
        }
        let Some(rest) = l.strip_prefix("diag rot region ") else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        let region: usize = f[0].parse().unwrap();
        let gid: usize = f[2].parse().unwrap();
        let own = f[4] == "1";
        let n: Vec<f64> = f[5..14].iter().map(|s| s.parse().unwrap()).collect();
        // 진단 출력은 nalgebra 행렬을 열 우선으로 내보낸다.
        if f.len() >= 17 {
            let c: Vec<f64> = f[14..17].iter().map(|s| s.parse().unwrap()).collect();
            ctrs.entry(region)
                .or_default()
                .insert(gid, Vector3::new(c[0], c[1], c[2]));
        }
        rots.entry(region)
            .or_default()
            .insert(gid, (own, Matrix3::from_column_slice(&n)));
    }
    let mut keys: Vec<_> = rots.keys().copied().collect();
    keys.sort();
    let up = Vector3::new(0.0, 0.0, 1.0);
    for k in keys {
        let r = &rots[&k];
        // 월드 회전 차 R_est^T R_true 의 평균 각, 그리고 연직(위) 방향 기울기:
        // 정답 위 방향 대비 추정 위 방향의 각 (카메라 좌표 위 = R * up 이 같도록 비교).
        let (mut diffs, mut tilts) = (Vec::new(), Vec::new());
        for (gid, (own, re)) in r {
            if !own {
                continue;
            }
            let rt = truth[&image_name(*gid)];
            diffs.push(re.transpose() * rt);
            let u_est = re.transpose() * (rt * up);
            tilts.push(u_est.angle(&up).to_degrees());
        }
        tilts.sort_by(f64::total_cmp);
        eprintln!(
            "tilt region {k} owned {} mean_rot_err_deg {:.3} up_tilt_median_deg {:.3}",
            diffs.len(),
            mean_rotation_angle(&diffs),
            tilts[tilts.len() / 2]
        );
    }
    // GPS 잡음과 척도 분리: 정답 중심(= -R^T t) 대비 (a) 잡음 GPS 의 거리 중앙값(완벽한 모델이 갖는 잔차),
    // (b) 추정 중심에 닮음 변환(Umeyama)을 맞춘 척도와 맞춘 뒤 잔차.
    let tc = |name: &str| -> Vector3<f64> {
        let mut t = Vector3::zeros();
        for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
            .unwrap()
            .lines()
        {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f[0] == name {
                let v: Vec<f64> = f[1..].iter().map(|s| s.parse().unwrap()).collect();
                t = Vector3::new(v[15], v[16], v[17]);
            }
        }
        -(truth[name].transpose() * t)
    };
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut rk: Vec<_> = ctrs.keys().copied().collect();
    rk.sort();
    for k in rk {
        let (mut e, mut t, mut noise) = (Vec::new(), Vec::new(), Vec::new());
        for (gid, c) in &ctrs[&k] {
            let name = image_name(*gid);
            let tcen = tc(&name);
            let vi = scene.views.iter().position(|v| v.name == name).unwrap();
            noise.push((scene.gps_enu[vi].coords - tcen).norm());
            e.push(*c);
            t.push(tcen);
        }
        let n = e.len() as f64;
        let me = e.iter().sum::<Vector3<f64>>() / n;
        let mt = t.iter().sum::<Vector3<f64>>() / n;
        let mut cov = Matrix3::zeros();
        let (mut ve, mut vt) = (0.0, 0.0);
        for (a, b) in e.iter().zip(&t) {
            cov += (b - mt) * (a - me).transpose() / n;
            ve += (a - me).norm_squared() / n;
            vt += (b - mt).norm_squared() / n;
        }
        let svd = cov.svd(true, true);
        let (u, vtm) = (svd.u.unwrap(), svd.v_t.unwrap());
        let mut d = Matrix3::identity();
        if (u * vtm).determinant() < 0.0 {
            d[(2, 2)] = -1.0;
        }
        let r = u * d * vtm;
        let scale = (svd
            .singular_values
            .component_mul(&Vector3::new(1.0, 1.0, d[(2, 2)])))
        .sum()
            / ve;
        let fit: Vec<f64> = e
            .iter()
            .zip(&t)
            .map(|(a, b)| (b - (scale * r * (a - me) + mt)).norm())
            .collect();
        let rms = (fit.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
        eprintln!(
            "scale region {k} n {} noise_only_gps_med {:.3} fit_scale {:.4} spread_ratio_est_over_truth {:.4} fit_resid_med {:.3} fit_resid_rms {:.3}",
            e.len(),
            med(noise),
            scale,
            (ve / vt).sqrt(),
            med(fit),
            rms
        );
    }
    stage_report(&stderr, &truth, &tc);
    let _ = std::fs::remove_dir_all(&root);
}

/// 닮음 변환(Umeyama)을 추정 중심에 맞춘 뒤 정답과 비교: (척도, 잔차 중앙, 잔차 rms).
fn center_fit(e: &[Vector3<f64>], t: &[Vector3<f64>]) -> (f64, f64, f64) {
    let n = e.len() as f64;
    let me = e.iter().sum::<Vector3<f64>>() / n;
    let mt = t.iter().sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    let mut ve = 0.0;
    for (a, b) in e.iter().zip(t) {
        cov += (b - mt) * (a - me).transpose() / n;
        ve += (a - me).norm_squared() / n;
    }
    let svd = cov.svd(true, true);
    let (u, vtm) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    if (u * vtm).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = u * d * vtm;
    let scale = svd
        .singular_values
        .component_mul(&Vector3::new(1.0, 1.0, d[(2, 2)]))
        .sum()
        / ve;
    let mut fit: Vec<f64> = e
        .iter()
        .zip(t)
        .map(|(a, b)| (b - (scale * r * (a - me) + mt)).norm())
        .collect();
    let rms = (fit.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
    fit.sort_by(f64::total_cmp);
    (scale, fit[fit.len() / 2], rms)
}

/// 단계별 진단 줄(`diag stage ...`)을 정답과 비교한다. 같은 (구역, 호출, 단계)가 여러 번 나오면 마지막 것.
/// 회전 오차는 전역 회전 Q = polar(Σ R_eᵀ R_t) 를 맞춘 뒤의 각.
fn stage_report(
    stderr: &str,
    truth: &HashMap<String, Matrix3<f64>>,
    tc: &dyn Fn(&str) -> Vector3<f64>,
) {
    type Rows = Vec<(usize, Matrix3<f64>, Vector3<f64>)>;
    let mut groups: Vec<((usize, String, String), Rows)> = Vec::new();
    for l in stderr.lines() {
        if l.starts_with("diag first_ba")
            || l.starts_with("diag refined_ba")
            || l.starts_with("diag gps_align n")
        {
            eprintln!("{l}");
        }
        let Some(rest) = l.strip_prefix("diag stage region ") else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        let key = (f[0].parse().unwrap(), f[1].to_string(), f[2].to_string());
        let gid: usize = f[4].parse().unwrap();
        let n: Vec<f64> = f[5..14].iter().map(|s| s.parse().unwrap()).collect();
        let c: Vec<f64> = f[14..17].iter().map(|s| s.parse().unwrap()).collect();
        let row = (
            gid,
            Matrix3::from_column_slice(&n),
            Vector3::new(c[0], c[1], c[2]),
        );
        // 새 호출이 시작되면(같은 키가 이미 있고 이 gid 가 이미 있으면) 그룹을 새로 연다.
        match groups.iter_mut().rev().find(|(k, _)| *k == key) {
            Some((_, rows)) if !rows.iter().any(|r| r.0 == gid) => rows.push(row),
            _ => groups.push((key, vec![row])),
        }
    }
    pair_report(stderr, truth);
    for ((region, tag, phase), rows) in groups {
        let mut m = Matrix3::zeros();
        for (gid, re, _) in &rows {
            m += re.transpose() * truth[&image_name(*gid)];
        }
        let sv = m.svd(true, true);
        let mut q = sv.u.unwrap() * sv.v_t.unwrap();
        if q.determinant() < 0.0 {
            let mut u = sv.u.unwrap();
            u.column_mut(2).neg_mut();
            q = u * sv.v_t.unwrap();
        }
        let errs: Vec<f64> = rows
            .iter()
            .map(|(gid, re, _)| {
                let d = (re * q).transpose() * truth[&image_name(*gid)];
                Rotation3::from_matrix_unchecked(d).angle().to_degrees()
            })
            .collect();
        if phase == "rots" && tag == "coarse" && std::env::var("SKYLENS_TILT_VERBOSE").is_ok() {
            let v: Vec<String> = rows
                .iter()
                .zip(&errs)
                .map(|(r, e)| format!("{}:{e:.0}", r.0))
                .collect();
            eprintln!("rot_err_by_gid region {region} {}", v.join(" "));
        }
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        let max = errs.iter().cloned().fold(0.0, f64::max);
        // 전역 회전 Q 자체: 각과 연직(z) 기울기. 맞춘 뒤 오차에는 안 보이는 단계 간 전역 기울기를 본다.
        let q_angle = Rotation3::from_matrix_unchecked(q).angle().to_degrees();
        let q_tilt = (q * Vector3::z()).z.clamp(-1.0, 1.0).acos().to_degrees();
        let mut line = format!(
            "stage region {region} {tag} {phase} n {} rot_err_mean {mean:.3} max {max:.3} q_angle {q_angle:.2} q_up_tilt {q_tilt:.2}",
            rows.len()
        );
        if phase != "rots" {
            let e: Vec<Vector3<f64>> = rows.iter().map(|r| r.2).collect();
            let t: Vec<Vector3<f64>> = rows.iter().map(|r| tc(&image_name(r.0))).collect();
            let (sc, med, rms) = center_fit(&e, &t);
            line += &format!(" fit_scale {sc:.4} fit_resid_med {med:.3} rms {rms:.3}");
        }
        eprintln!("{line}");
    }
}

/// 입력 상대 회전(`diag pair`)을 정답 R_j R_iᵀ 와 비교: 구역별로 간선 수, 5° 넘게 어긋난 간선 수(전체/남긴 것),
/// 어긋난 간선의 정상 대응 수 중앙과 정상 간선의 정상 대응 수 중앙.
fn pair_report(stderr: &str, truth: &HashMap<String, Matrix3<f64>>) {
    type PairStat = (usize, usize, usize, Vec<usize>, Vec<usize>);
    let mut stats: std::collections::BTreeMap<usize, PairStat> = Default::default();
    for l in stderr.lines() {
        let Some(rest) = l.strip_prefix("diag pair region ") else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        if f[1] != "coarse" {
            continue;
        }
        let region: usize = f[0].parse().unwrap();
        let (gi, gj): (usize, usize) = (f[3].parse().unwrap(), f[4].parse().unwrap());
        let inl: usize = f[6].parse().unwrap();
        let keep = f[8] == "1";
        let n: Vec<f64> = f[9..18].iter().map(|s| s.parse().unwrap()).collect();
        let rel = Matrix3::from_column_slice(&n);
        let tr = truth[&image_name(gj)] * truth[&image_name(gi)].transpose();
        let err = Rotation3::from_matrix_unchecked(rel.transpose() * tr)
            .angle()
            .to_degrees();
        let e = stats.entry(region).or_default();
        e.0 += 1;
        if err > 5.0 {
            e.1 += 1;
            e.2 += usize::from(keep);
            e.3.push(inl);
        } else {
            e.4.push(inl);
        }
    }
    for (region, (n, bad, bad_kept, mut bi, mut gi)) in stats {
        bi.sort();
        gi.sort();
        eprintln!(
            "pairs region {region} total {n} bad_gt5deg {bad} bad_kept {bad_kept} bad_inl_med {:?} good_inl_med {:?}",
            bi.get(bi.len() / 2),
            gi.get(gi.len() / 2)
        );
    }
}
