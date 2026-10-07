//! 카메라 2 가 낀 교차 짝의 기하 검증 단계별 탈락을 합성 정답과 대조하는 진단(무시 시험).
//! 라이브러리가 `SKYLENS_CROSS_DEBUG` 로 낸 짝별 줄(`cross_dbg`, `cross_dbg_m`)을 모아, 합성 출력 폴더의
//! `truth/cameras.txt` 정답 포즈로 (1) 두 사진 지면(z=0) 시야 겹침 비율, (2) 비율 시험 통과 대응 중
//! 정답 에피폴라 기하와 맞는(Sampson 거리 < 문턱) 참 대응 수를 계산해 표로 낸다.
//! 실행: `UPX_SEED=5 cargo test --release -p skylens-stream --test cam2_cross_verify -- --ignored --nocapture`

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use skylens_core::matching::RansacConfig;
use skylens_core::synth::{Scene, SceneConfig};

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

struct Cam {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
    r: M3,
    t: V3,
}

fn mv(m: &M3, v: &V3) -> V3 {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}
fn mtv(m: &M3, v: &V3) -> V3 {
    [0, 1, 2].map(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}
fn mm(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}
fn tr(a: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = a[j][i];
        }
    }
    o
}

impl Cam {
    fn center(&self) -> V3 {
        let c = mtv(&self.r, &self.t);
        [-c[0], -c[1], -c[2]]
    }
    /// 화소(연속 좌표) → 세계 광선 방향.
    fn ray(&self, u: f64, v: f64) -> V3 {
        mtv(
            &self.r,
            &[(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0],
        )
    }
    fn project(&self, x: &V3) -> Option<(f64, f64)> {
        let c = mv(&self.r, x);
        let p = [c[0] + self.t[0], c[1] + self.t[1], c[2] + self.t[2]];
        (p[2] > 1e-6).then(|| {
            (
                self.fx * p[0] / p[2] + self.cx,
                self.fy * p[1] / p[2] + self.cy,
            )
        })
    }
    /// 이 사진 화소 격자의 지면(z=0) 점 중 `other` 사진 안에 보이는 비율.
    fn ground_overlap_into(&self, other: &Cam) -> f64 {
        let (nx, ny) = (48, 27);
        let (mut hit, mut tot) = (0usize, 0usize);
        let c = self.center();
        for a in 0..nx {
            for b in 0..ny {
                let (u, v) = (
                    (a as f64 + 0.5) / nx as f64 * self.w,
                    (b as f64 + 0.5) / ny as f64 * self.h,
                );
                tot += 1;
                let d = self.ray(u, v);
                if d[2] >= -1e-9 {
                    continue;
                }
                let s = -c[2] / d[2];
                let x = [c[0] + s * d[0], c[1] + s * d[1], 0.0];
                if let Some((p, q)) = other.project(&x) {
                    if p >= 0.0 && p < other.w && q >= 0.0 && q < other.h {
                        hit += 1;
                    }
                }
            }
        }
        hit as f64 / tot as f64
    }
}

fn load_truth(path: &std::path::Path) -> BTreeMap<String, Cam> {
    let mut out = BTreeMap::new();
    for l in std::fs::read_to_string(path).unwrap().lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n: Vec<f64> = f[1..].iter().map(|s| s.parse().unwrap()).collect();
        let r = [
            [n[6], n[7], n[8]],
            [n[9], n[10], n[11]],
            [n[12], n[13], n[14]],
        ];
        out.insert(
            f[0].to_string(),
            Cam {
                fx: n[0],
                fy: n[1],
                cx: n[2],
                cy: n[3],
                w: n[4],
                h: n[5],
                r,
                t: [n[15], n[16], n[17]],
            },
        );
    }
    out
}

fn name(cam: usize, pos: usize) -> String {
    format!("cam{}_{:04}", ['F', 'R', 'L'][cam], pos)
}

/// 정답 본질 행렬 기준 Sampson 거리(화소). 첫 사진 → 둘째 사진.
fn sampson_px(a: &Cam, b: &Cam, p: (f64, f64), q: (f64, f64)) -> f64 {
    let rab = mm(&b.r, &tr(&a.r));
    let rta = mv(&rab, &a.t);
    let t = [b.t[0] - rta[0], b.t[1] - rta[1], b.t[2] - rta[2]];
    let tx = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
    let e = mm(&tx, &rab);
    let x1 = [(p.0 - a.cx) / a.fx, (p.1 - a.cy) / a.fy, 1.0];
    let x2 = [(q.0 - b.cx) / b.fx, (q.1 - b.cy) / b.fy, 1.0];
    let ex1 = mv(&e, &x1);
    let etx2 = mtv(&e, &x2);
    let num = x2[0] * ex1[0] + x2[1] * ex1[1] + x2[2] * ex1[2];
    let den = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
    (num * num / den.max(1e-30)).sqrt() * a.fx
}

struct Row {
    a: (usize, usize),
    b: (usize, usize),
    ratio: usize,
    min: usize,
    outcome: String,
    n: usize,
    best_cnt: usize,
    top: usize,
    needed: usize,
    stage: usize,
    inl: usize,
    pts: Vec<((f64, f64), (f64, f64))>,
}

fn parse(stderr: &str) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for l in stderr.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        if l.starts_with("cross_dbg_m ") {
            let r = rows.last_mut().expect("요약 줄이 먼저");
            if let Some(body) = f.get(7) {
                r.pts = body
                    .split(';')
                    .map(|s| {
                        let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
                        ((v[0], v[1]), (v[2], v[3]))
                    })
                    .collect();
            }
        } else if l.starts_with("cross_dbg ") {
            let g = |k: &str| -> &str {
                let i = f.iter().position(|x| *x == k).unwrap();
                f[i + 1]
            };
            let gu = |k: &str| g(k).parse::<usize>().unwrap();
            let pair = |k: &str| {
                let i = f.iter().position(|x| *x == k).unwrap();
                (f[i + 1].parse().unwrap(), f[i + 2].parse().unwrap())
            };
            rows.push(Row {
                a: pair("a"),
                b: pair("b"),
                ratio: gu("ratio"),
                min: gu("min"),
                outcome: g("outcome").to_string(),
                n: gu("n"),
                best_cnt: gu("best_cnt"),
                top: gu("top"),
                needed: gu("needed"),
                stage: gu("stage"),
                inl: gu("inl"),
                pts: Vec::new(),
            });
        }
    }
    rows
}

#[test]
#[ignore = "시드당 약 8분"]
fn cam2_cross_verify() {
    let seed: u64 = std::env::var("UPX_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let root: PathBuf =
        std::env::temp_dir().join(format!("skylens_cam2xv_{seed}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (input, output) = (root.join("in"), root.join("out"));
    std::fs::create_dir_all(&input).unwrap();
    Scene::new(SceneConfig {
        seed,
        ..SceneConfig::default()
    })
    .write_dataset(&input)
    .unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["run", input.to_str().unwrap(), output.to_str().unwrap()])
        .env("SKYLENS_CROSS_DEBUG", "1")
        .env(
            "SKYLENS_CROSS_MIN_MATCHES",
            std::env::var("SKYLENS_CROSS_MIN_MATCHES").unwrap_or_else(|_| "10".into()),
        )
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    let _ = std::fs::write(root.join("stderr.txt"), &stderr);
    assert!(o.status.success(), "run 실패");
    let truth = load_truth(&input.join("truth/cameras.txt"));
    let th = RansacConfig::default().threshold_px;
    let mut rows = parse(&stderr);
    rows.sort_by_key(|r| (r.a.0.min(r.b.0), r.a.0.max(r.b.0), r.a.1, r.b.1));
    eprintln!("| 짝(카메라 위치 - 카메라 위치) | 대응 | 결과(탈락 단계) | n | 표본 최선 | 정밀화 최다 | 유의 기준 | 단계 | 최종 정상 | 겹침 a→b | 겹침 b→a | 정답 참 대응 |");
    eprintln!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let (ca, cb) = (&truth[&name(r.a.0, r.a.1)], &truth[&name(r.b.0, r.b.1)]);
        let true_n = r
            .pts
            .iter()
            .filter(|(p, q)| sampson_px(ca, cb, *p, *q) < th)
            .count();
        eprintln!(
            "| {}-{} {}-{} | {} (하한 {}) | {} | {} | {} | {} | {} | {} | {} | {:.2} | {:.2} | {} |",
            r.a.0,
            r.a.1,
            r.b.0,
            r.b.1,
            r.ratio,
            r.min,
            r.outcome,
            r.n,
            r.best_cnt,
            r.top,
            r.needed,
            r.stage,
            r.inl,
            ca.ground_overlap_into(cb),
            cb.ground_overlap_into(ca),
            true_n
        );
    }
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    for r in &rows {
        *by.entry(r.outcome.clone()).or_default() += 1;
    }
    eprintln!("outcome 집계 {by:?}");
    if std::env::var_os("KEEP_OUT").is_none() {
        let _ = std::fs::remove_dir_all(&root);
    }
}
