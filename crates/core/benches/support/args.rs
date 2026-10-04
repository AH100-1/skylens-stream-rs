//! `benches/pipeline.rs` 의 인자 해석. 순서 규칙을 시험(`tests/perf_structure.rs`)에서 확인할 수 있도록
//! 따로 둔다.
//!
//! 규칙: 묶음 인자(`--quick`, `--full`)는 놓인 자리와 상관없이 먼저 적용하고, 개별 인자
//! (`--positions`, `--width` 등)는 그 위에 덮어쓴다. 그래서 `--positions 20 --full` 과
//! `--full --positions 20` 은 같은 규모(위치 20, 나머지는 `--full`)가 된다. 묶음 인자를 둘 다 주면
//! 뒤에 준 것이 이긴다.

/// bench 실행 인자.
#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub positions: usize,
    pub width: u32,
    pub height: u32,
    pub repeat: usize,
    pub threads: usize,
    pub max_pairs: usize,
    pub ba_points: usize,
    pub json: Option<String>,
    pub mode: String,
    pub ba_tracks: usize,
    pub ba_iters: usize,
}

impl Default for Args {
    /// 인자 없는 실행: 120장(위치 40, 편대 짝 일정으로 카메라 간 겹침이 생기는 규모)·480×270·반복 3·번들 조정 점 3000.
    fn default() -> Self {
        Args {
            positions: 40,
            width: 480,
            height: 270,
            repeat: 3,
            threads: 0,
            max_pairs: 0,
            ba_points: 3000,
            json: None,
            mode: "pipeline".to_string(),
            ba_tracks: 100_000,
            ba_iters: 3,
        }
    }
}

/// 묶음 인자가 정하는 값.
fn apply_preset(a: &mut Args, name: &str) {
    match name {
        "--quick" => {
            let d = Args::default();
            a.positions = d.positions;
            a.width = d.width;
            a.height = d.height;
            a.repeat = d.repeat;
            a.ba_points = d.ba_points;
        }
        "--full" => {
            a.positions = 80;
            a.width = 960;
            a.height = 540;
            a.repeat = 3;
            a.ba_points = 20_000;
        }
        _ => unreachable!("묶음 인자가 아니다: {name}"),
    }
}

/// 인자 목록(프로그램 이름 뺀 것)을 해석한다. 잘못된 인자는 panic.
pub fn parse(argv: &[String]) -> Args {
    let mut a = Args::default();
    // 1단계: 묶음 인자만 놓인 순서대로 적용한다. 값을 받는 인자의 값 자리는 건너뛴다.
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--quick" | "--full" => apply_preset(&mut a, &argv[i]),
            "--bench" => {}
            _ => i += 1,
        }
        i += 1;
    }
    // 2단계: 개별 인자를 덮어쓴다.
    let mut positions_set = false;
    let num = |v: Option<&String>, name: &str| -> usize {
        v.and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("{name} 뒤에 0 이상의 정수가 필요하다"))
    };
    let mut i = 0;
    while i < argv.len() {
        let next = argv.get(i + 1);
        match argv[i].as_str() {
            "--positions" => {
                a.positions = num(next, "--positions");
                positions_set = true;
            }
            "--width" => a.width = num(next, "--width") as u32,
            "--height" => a.height = num(next, "--height") as u32,
            "--repeat" => a.repeat = num(next, "--repeat").max(1),
            "--threads" => a.threads = num(next, "--threads"),
            "--max-pairs" => a.max_pairs = num(next, "--max-pairs"),
            "--ba-points" => a.ba_points = num(next, "--ba-points"),
            "--ba-tracks" => a.ba_tracks = num(next, "--ba-tracks"),
            "--ba-iters" => a.ba_iters = num(next, "--ba-iters").max(1),
            "--json" => {
                a.json = Some(
                    next.cloned()
                        .unwrap_or_else(|| panic!("--json 뒤에 경로가 필요하다")),
                )
            }
            "--mode" => {
                a.mode = next
                    .cloned()
                    .unwrap_or_else(|| panic!("--mode 뒤에 값이 필요하다"))
            }
            "--quick" | "--full" => {
                positions_set = true;
                i += 1;
                continue;
            }
            // cargo bench 가 붙이는 인자는 무시한다.
            "--bench" => {
                i += 1;
                continue;
            }
            other => panic!("알 수 없는 인자: {other}"),
        }
        i += 2;
    }
    // 번들 조정 실제 규모 측정은 카메라 240(위치 80)이 목적이므로 위치를 따로 주지 않으면 80 으로 둔다.
    if a.mode == "ba-scale" && !positions_set {
        a.positions = 80;
    }
    a
}
