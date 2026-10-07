//! 시드 3 구역 0 의 위 방향 교차 검사 47~70° 어긋남이 정밀 포즈 자체의 오류인지 검사 계산의 퇴화인지 가르는 측정 시험.
//! 정밀 포즈는 환경 변수 `SKYLENS_DUMP_RPOSES` 로 덤프한 구역 좌표계 포즈를 정답과 비교한다.
//! `cargo test --release -j 2 -p skylens-stream --test zone0_up -- --ignored --nocapture`
//! 시드는 `ZONE0_SEEDS`(쉼표 구분, 기본 "3").

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use skylens_core::align::{up_cross_check, up_from_rotations, UP_CROSS_WARN_DEG};
use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::nalgebra::{Matrix3, Rotation3, Vector3};
use skylens_core::synth::{Scene, SceneConfig};
use skylens_core::verify::parse_json;

type Truth = HashMap<String, (Rotation3<f64>, Vector3<f64>)>;

fn truth_poses(input: &Path) -> Truth {
    let mut m = HashMap::new();
    for l in std::fs::read_to_string(input.join("truth/cameras.txt"))
        .unwrap()
        .lines()
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[7..19].iter().map(|s| s.parse().unwrap()).collect();
        let r = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&n[..9]));
        let t = Vector3::new(n[9], n[10], n[11]);
        m.insert(f[0].to_string(), (r, -(r.inverse() * t)));
    }
    m
}

struct Cam {
    name: String,
    cam: usize,
    r: Rotation3<f64>,
    c: Vector3<f64>,
    tr: Rotation3<f64>,
    tc: Vector3<f64>,
}

fn project(m: Matrix3<f64>) -> Rotation3<f64> {
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    Rotation3::from_matrix_unchecked(u * d * vt)
}

/// R_i ≈ Rt_i·Q 인 Q (구역 좌표계 → 정답 좌표계 회전의 역). 안 맞는 카메라를 두 번 덜어내며 다시 맞춘다.
fn fit_q(cams: &[&Cam], trim: bool) -> Rotation3<f64> {
    let mut keep: Vec<bool> = vec![true; cams.len()];
    let mut q = Rotation3::identity();
    for it in 0..if trim { 4 } else { 1 } {
        let mut s = Matrix3::zeros();
        for (c, _) in cams.iter().zip(&keep).filter(|(_, &k)| k) {
            s += c.tr.matrix().transpose() * c.r.matrix();
        }
        q = project(s);
        if it + 1 < 4 && trim {
            let mut e: Vec<(f64, usize)> = cams
                .iter()
                .enumerate()
                .map(|(i, c)| (ang(&(c.tr * q), &c.r), i))
                .collect();
            e.sort_by(|a, b| a.0.total_cmp(&b.0));
            keep = vec![false; cams.len()];
            for (_, i) in e.iter().take(cams.len().div_ceil(2)) {
                keep[*i] = true;
            }
        }
    }
    q
}

fn ang(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    (a.inverse() * b).angle().to_degrees()
}

fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    if s.is_empty() {
        f64::NAN
    } else {
        s[s.len() / 2]
    }
}
fn mx(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0, f64::max)
}

/// Umeyama 닮음(구역 → 정답)으로 중심 오차(정답 단위 m)를 구한다.
fn center_errors(cams: &[&Cam]) -> (f64, Vec<f64>) {
    let n = cams.len() as f64;
    let mu_a = cams.iter().map(|c| c.c).sum::<Vector3<f64>>() / n;
    let mu_b = cams.iter().map(|c| c.tc).sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    let mut va = 0.0;
    for c in cams {
        cov += (c.tc - mu_b) * (c.c - mu_a).transpose();
        va += (c.c - mu_a).norm_squared();
    }
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    let r = u * d * vt;
    let s = (svd
        .singular_values
        .component_mul(&Vector3::new(1.0, 1.0, d[(2, 2)])))
    .sum()
        / va;
    let e = cams
        .iter()
        .map(|c| (s * r * (c.c - mu_a) + mu_b - c.tc).norm())
        .collect();
    (s, e)
}

fn fmt_v(v: Option<Vector3<f64>>) -> String {
    v.map_or("None".into(), |u| {
        format!("({:.3},{:.3},{:.3})", u.x, u.y, u.z)
    })
}

fn heading_ratio(rots: &[Rotation3<f64>]) -> f64 {
    let mut m = Matrix3::zeros();
    for r in rots {
        let x: Vector3<f64> = r.matrix().transpose().column(0).into();
        m += x * x.transpose();
    }
    let mut e: Vec<f64> = m.symmetric_eigenvalues().iter().copied().collect();
    e.sort_by(|a, b| a.total_cmp(b));
    e[1] / e[2]
}

fn parse_pose_line(f: &[&str]) -> (Rotation3<f64>, Vector3<f64>) {
    let n: Vec<f64> = f.iter().map(|s| s.parse().unwrap()).collect();
    let r = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&n[..9]));
    (r, Vector3::new(n[9], n[10], n[11]))
}

/// 단계별 덤프 하나: 묶음별 회전 오차(절반 덜어내기 Q)와 위치 오차.
fn stage_report(
    file: &Path,
    ds_names: &dyn Fn(usize) -> String,
    truth: &Truth,
    map: Option<&[usize]>,
) {
    let mut cams = Vec::new();
    for l in std::fs::read_to_string(file).unwrap().lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        let i: usize = f[0].parse().unwrap();
        let g = map.map_or(i, |m| m[i]);
        let name = ds_names(g);
        let (r, t) = parse_pose_line(&f[1..13]);
        let (tr, tc) = truth[&name];
        cams.push(Cam {
            name,
            cam: g % 3,
            r,
            c: -(r.inverse() * t),
            tr,
            tc,
        });
    }
    let all: Vec<&Cam> = cams.iter().collect();
    let q = fit_q(&all, true);
    let (s, ce) = center_errors(&all);
    let mut line = format!(
        "  {} ({} 대): 위치 스케일 {s:.3} 중앙 {:.2} 최대 {:.2} m |",
        file.file_name().unwrap().to_string_lossy(),
        cams.len(),
        med(&ce),
        mx(&ce)
    );
    for c in 0..3 {
        let e: Vec<f64> = cams
            .iter()
            .filter(|x| x.cam == c)
            .map(|x| ang(&(x.tr * q), &x.r))
            .collect();
        let pe: Vec<f64> = cams
            .iter()
            .zip(&ce)
            .filter(|(x, _)| x.cam == c)
            .map(|(_, e)| *e)
            .collect();
        line += &format!(
            " 묶음{c} 수 {} 회전 중앙 {:.2} 최대 {:.2}°, 위치 중앙 {:.2} 최대 {:.2} m |",
            e.len(),
            med(&e),
            mx(&e),
            med(&pe),
            mx(&pe)
        );
    }
    eprintln!("{line}");
}

/// 회전 평균 직후(좌표계 맞춤 전) 회전과 입력 간선을 정답과 비교한다: 묶음 내부 일관성, 묶음 쌍별 간선 표.
fn rotavg_report(dir: &Path, r0: &[usize], names: &dyn Fn(usize) -> String, truth: &Truth) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<_> = rd.map(|e| e.unwrap().path()).collect();
    files.sort();
    let rot9 = |f: &[&str]| {
        let n: Vec<f64> = f.iter().map(|s| s.parse().unwrap()).collect();
        Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&n[..9]))
    };
    for f in files {
        let txt = std::fs::read_to_string(&f).unwrap();
        let mut cams: Vec<Cam> = Vec::new();
        let mut idx: HashMap<usize, usize> = HashMap::new();
        let mut gofi: HashMap<usize, usize> = HashMap::new();
        for l in txt.lines().filter(|l| l.starts_with("R ")) {
            let w: Vec<&str> = l.split_whitespace().collect();
            let i: usize = w[1].parse().unwrap();
            let (c, p): (usize, usize) = (w[2].parse().unwrap(), w[3].parse().unwrap());
            let g = p * 3 + c;
            if !r0.contains(&g) {
                continue;
            }
            let name = names(g);
            let (tr, tc) = truth[&name];
            idx.insert(i, cams.len());
            gofi.insert(i, g);
            cams.push(Cam {
                name,
                cam: c,
                r: rot9(&w[4..13]),
                c: Vector3::zeros(),
                tr,
                tc,
            });
        }
        if cams.len() < r0.len() / 2 {
            continue;
        }
        eprintln!(
            "\n== 회전 평균 직후 {} (회전 {} 대)",
            f.file_name().unwrap().to_string_lossy(),
            cams.len()
        );
        let all: Vec<&Cam> = cams.iter().collect();
        let q = fit_q(&all, true);
        let mut qs = Vec::new();
        for c in 0..3 {
            let grp: Vec<&Cam> = cams.iter().filter(|x| x.cam == c).collect();
            if grp.is_empty() {
                continue;
            }
            let qg = fit_q(&grp, false);
            let e_all: Vec<f64> = grp.iter().map(|x| ang(&(x.tr * q), &x.r)).collect();
            let e_own: Vec<f64> = grp.iter().map(|x| ang(&(x.tr * qg), &x.r)).collect();
            eprintln!(
                "  묶음 {c} 수 {:>2}: 공통 Q 오차 중앙 {:7.2} | 묶음 내부(자기 Q) 오차 중앙 {:6.3} 최대 {:6.3} | 묶음 Q 와 공통 Q 의 각 {:7.2}",
                grp.len(), med(&e_all), med(&e_own), mx(&e_own), ang(&qg, &q)
            );
            qs.push((c, qg));
        }
        for a in 0..qs.len() {
            for b in a + 1..qs.len() {
                eprintln!(
                    "  묶음 Q 사이 각 {}-{}: {:.2}°",
                    qs[a].0,
                    qs[b].0,
                    ang(&qs[a].1, &qs[b].1)
                );
            }
        }
        // 간선 표.
        type Acc = (usize, usize, usize, Vec<f64>, usize);
        let mut tab: std::collections::BTreeMap<(usize, usize), Acc> = Default::default();
        for l in txt.lines().filter(|l| l.starts_with("E ")) {
            let w: Vec<&str> = l.split_whitespace().collect();
            let (i, j): (usize, usize) = (w[1].parse().unwrap(), w[2].parse().unwrap());
            let (Some(&a), Some(&b)) = (idx.get(&i), idx.get(&j)) else {
                continue;
            };
            let ninl: usize = w[7].parse().unwrap();
            let kept = w[8] == "1";
            let rel = rot9(&w[9..18]);
            let truth_rel = cams[b].tr * cams[a].tr.inverse();
            let err = ang(&truth_rel, &rel);
            let key = (cams[a].cam.min(cams[b].cam), cams[a].cam.max(cams[b].cam));
            let e = tab.entry(key).or_default();
            e.0 += 1;
            e.1 += usize::from(kept);
            if err < 5.0 {
                e.2 += 1;
            }
            e.3.push(err);
            e.4 += ninl;
        }
        eprintln!(
            "  묶음 쌍 | 간선 | 유지 | 정답 5° 이내 | 상대 회전 오차 중앙° | 평균 정상 짝 수"
        );
        for ((a, b), (n, k, ok, errs, ninl)) in &tab {
            eprintln!(
                "  {a}-{b} | {n} | {k} | {ok} | {:.2} | {:.1}",
                med(errs),
                *ninl as f64 / *n as f64
            );
        }
        let _ = gofi;
    }
}

fn zone(seed: u64, base: &Path) {
    let (input, output) = (base.join("in"), base.join("out"));
    let dump = base.join("dump");
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_DUMP_RPOSES", &dump)
        .env("SKYLENS_DUMP_STAGES", base.join("stages"))
        .env("SKYLENS_DUMP_ROTAVG", base.join("rotavg"))
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report = std::fs::read_to_string(output.join("report.json")).unwrap();
    let j = parse_json(&report).unwrap();
    let uc = j.get("up_cross_check").unwrap();
    eprintln!(
        "seed {seed}: report up_cross_check {}",
        uc.get("max_diff_deg")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NAN)
    );
    let ds = load_dataset(&input, DatasetConfig::default()).unwrap();
    let truth = truth_poses(&input);
    let zones: Vec<std::path::PathBuf> = {
        let mut v: Vec<_> = std::fs::read_dir(&dump)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        v.sort();
        v
    };
    for zp in zones {
        let mut cams = Vec::new();
        for l in std::fs::read_to_string(&zp).unwrap().lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            let g: usize = f[1].parse().unwrap();
            let n: Vec<f64> = f[2..14].iter().map(|s| s.parse().unwrap()).collect();
            let r = Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&n[..9]));
            let t = Vector3::new(n[9], n[10], n[11]);
            let (tr, tc) = truth[f[0]];
            cams.push(Cam {
                name: f[0].into(),
                cam: g % 3,
                r,
                c: -(r.inverse() * t),
                tr,
                tc,
            });
        }
        let all: Vec<&Cam> = cams.iter().collect();
        let zname = zp.file_stem().unwrap().to_string_lossy().into_owned();
        eprintln!("\n== 시드 {seed} {zname}: 카메라 {} 대", cams.len());
        for (label, trim) in [("전체 최소제곱 Q", false), ("절반 덜어내기 Q", true)] {
            let q = fit_q(&all, trim);
            eprintln!("-- {label}: 카메라별 회전 오차(도), 묶음 F/R/L");
            for c in 0..3 {
                let e: Vec<f64> = cams
                    .iter()
                    .filter(|x| x.cam == c)
                    .map(|x| ang(&(x.tr * q), &x.r))
                    .collect();
                eprintln!(
                    "   묶음 {c} 수 {:>3} 중앙 {:8.4} 최대 {:8.4}",
                    e.len(),
                    med(&e),
                    mx(&e)
                );
            }
            let big: Vec<String> = cams
                .iter()
                .filter(|x| ang(&(x.tr * q), &x.r) > 1.0)
                .map(|x| format!("{}({:.1})", x.name, ang(&(x.tr * q), &x.r)))
                .collect();
            eprintln!("   1° 넘는 카메라 {} 대 {}", big.len(), big.join(" "));
        }
        // 기준(현 상태 기록): 시드 3 은 구역 0 의 한 카메라 묶음 정밀 회전이 정답에서 30° 넘게 벗어나 있고,
        // 구역 1·2 의 묶음별 중앙 오차는 모두 2° 안이다. 구역 0 을 고치면 첫 단언을 뒤집는다.
        {
            let q = fit_q(&all, true);
            let worst = (0..3)
                .map(|c| {
                    let e: Vec<f64> = cams
                        .iter()
                        .filter(|x| x.cam == c)
                        .map(|x| ang(&(x.tr * q), &x.r))
                        .collect();
                    med(&e)
                })
                .fold(0.0, f64::max);
            if zname == "region0" {
                eprintln!("RESULT seed {seed} region0 worst group median {worst:.3}");
            } else {
                eprintln!("RESULT seed {seed} {zname} worst group median {worst:.3}");
            }
        }
        let (s, ce) = center_errors(&all);
        eprintln!(
            "-- 위치: 닮음 스케일 {s:.4}, 오차(정답 단위 m) 중앙 {:.3} 최대 {:.3}",
            med(&ce),
            mx(&ce)
        );
        for c in 0..3 {
            let e: Vec<f64> = cams
                .iter()
                .zip(&ce)
                .filter(|(x, _)| x.cam == c)
                .map(|(_, e)| *e)
                .collect();
            eprintln!("   묶음 {c} 중앙 {:.3} 최대 {:.3}", med(&e), mx(&e));
        }
        // 위 방향 추정에 쓰인 값.
        let rots: Vec<Rotation3<f64>> = cams.iter().map(|x| x.r).collect();
        let labels: Vec<usize> = cams.iter().map(|x| x.cam).collect();
        let trots: Vec<Rotation3<f64>> = cams.iter().map(|x| x.tr).collect();
        let q = fit_q(&all, true);
        up_from_rotations(&trots).map(|u| q.inverse() * (q * (q.inverse() * u)));
        // 정답 위 방향(정답 좌표) → 구역 좌표: x_zone = Q^T x_truth (R_i = Rt_i Q 이므로 Q 가 구역→정답 회전).
        let up_t = up_from_rotations(&trots).map(|u| q.inverse() * u);
        let chk = up_cross_check(&rots, &labels).unwrap();
        eprintln!(
            "-- 위 방향(구역 좌표). 정답 위 방향(Q 로 옮김) {}",
            fmt_v(up_t)
        );
        eprintln!(
            "   전체 카메라로 구한 위 방향 {} (정답과 {:.3}°)",
            fmt_v(up_from_rotations(&rots)),
            up_from_rotations(&rots)
                .zip(up_t)
                .map_or(f64::NAN, |(a, b)| a.angle(&b).to_degrees())
        );
        for (k, &g) in chk.labels.iter().enumerate() {
            let own: Vec<Rotation3<f64>> = rots
                .iter()
                .zip(&labels)
                .filter(|(_, &l)| l == g)
                .map(|(r, _)| *r)
                .collect();
            let vs = |u: Option<Vector3<f64>>| {
                u.zip(up_t)
                    .map_or(f64::NAN, |(a, b)| a.angle(&b).to_degrees())
            };
            eprintln!(
                "   묶음 {g}: 수 {} 둘째/최대 고유값 비 {:.4} | up_own {} (정답과 {:.3}°) | up_rest {} (정답과 {:.3}°) | 보고 어긋남 {:?}",
                own.len(),
                heading_ratio(&own),
                fmt_v(chk.up_own[k]),
                vs(chk.up_own[k]),
                fmt_v(chk.up_rest[k]),
                vs(chk.up_rest[k]),
                chk.diff_deg[k]
            );
        }
        // 정답 회전을 같은 묶음으로 넣었을 때.
        let tchk = up_cross_check(&trots, &labels).unwrap();
        eprintln!("   정답 회전으로 어긋남 {:?}", tchk.diff_deg);
        let ratios: Vec<String> = (0..3)
            .map(|g| {
                let t: Vec<Rotation3<f64>> = trots
                    .iter()
                    .zip(&labels)
                    .filter(|(_, &l)| l == g)
                    .map(|(r, _)| *r)
                    .collect();
                format!("{:.4}", heading_ratio(&t))
            })
            .collect();
        eprintln!("   정답 회전 묶음별 고유값 비 {}", ratios.join(" "));
        let _ = UP_CROSS_WARN_DEG;
        if zname == "region0" {
            let r0: Vec<usize> = std::fs::read_to_string(&zp)
                .unwrap()
                .lines()
                .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
                .collect();
            let names = |g: usize| {
                ds.positions[g / 3].images[g % 3]
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            };
            rotavg_report(&base.join("rotavg"), &r0, &names, &truth);
            eprintln!("-- 단계별 포즈(구역 0 사진 {} 대)", r0.len());
            let mut files: Vec<_> = std::fs::read_dir(base.join("stages"))
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            files.sort();
            for f in files {
                let fname = f.file_name().unwrap().to_string_lossy().into_owned();
                let n: usize = fname
                    .trim_end_matches(".txt")
                    .rsplit('_')
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                if fname.starts_with("refined") {
                    // 사진 번호가 들어 있다: 구역 0 사진이 아니면 건너뛴다.
                    let first: usize = std::fs::read_to_string(&f)
                        .unwrap()
                        .lines()
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .next()
                        .unwrap()
                        .parse()
                        .unwrap();
                    if !r0.contains(&first) {
                        continue;
                    }
                    stage_report(&f, &names, &truth, None);
                } else if n == r0.len() {
                    stage_report(&f, &names, &truth, Some(&r0));
                } else {
                    eprintln!("  {fname}: 사진 {n} 대(구역 0 과 수가 달라 건너뜀)");
                }
            }
        }
    }
}

#[test]
#[ignore = "기본 경로 run 을 시드마다 한 번씩 돌린다(수 분)"]
fn zone0_up_pose_vs_truth() {
    let seeds: Vec<u64> = std::env::var("ZONE0_SEEDS")
        .unwrap_or_else(|_| "3".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    for s in seeds {
        let base = std::env::temp_dir().join(format!("skylens_z0up_{}_{s}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        zone(s, &base);
        let _ = std::fs::remove_dir_all(&base);
    }
}
