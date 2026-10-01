//! skylens-stream 명령행 도구: PLY 정보 확인, 합성 장면 생성.

use std::process::ExitCode;

use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};

const USAGE: &str = "사용법:\n  skylens-stream ply-info <파일.ply>\n  skylens-stream synth <출력 폴더> [폭 높이]\n  \
     (폭·높이는 16..=8192 정수, 둘 다 주거나 둘 다 생략)";

/// 합성 영상 폭·높이 하한(특징 검출 최소 크기)과 상한.
const MIN_SIDE: u32 = 16;
const MAX_SIDE: u32 = 8192;

fn parse_side(s: &str) -> Option<u32> {
    s.parse::<u32>()
        .ok()
        .filter(|v| (MIN_SIDE..=MAX_SIDE).contains(v))
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("오류: {msg}\n{USAGE}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["ply-info", path] => match read_ply_file(path) {
            Ok(cloud) => {
                println!("points {}", cloud.len());
                println!("nan {}", cloud.has_nan());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{path}: {e}");
                ExitCode::FAILURE
            }
        },
        ["synth", out, rest @ ..] => {
            let mut cfg = SceneConfig::default();
            match rest {
                [] => {}
                [w, h] => match (parse_side(w), parse_side(h)) {
                    (Some(w), Some(h)) => (cfg.width, cfg.height) = (w, h),
                    _ => {
                        return usage_error(&format!(
                            "폭·높이는 {MIN_SIDE}..={MAX_SIDE} 정수여야 함: {w} {h}"
                        ))
                    }
                },
                _ => {
                    return usage_error(&format!(
                        "synth 는 폭·높이를 둘 다 주거나 둘 다 생략해야 함 (받은 추가 인자 {}개)",
                        rest.len()
                    ))
                }
            }
            let scene = Scene::new(cfg);
            match scene.write_dataset(std::path::Path::new(out)) {
                Ok(()) => {
                    println!("views {}", scene.views.len());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{out}: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        ["--version"] => {
            println!("skylens-stream {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
