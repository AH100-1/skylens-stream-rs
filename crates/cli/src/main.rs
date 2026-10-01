//! skylens-stream 명령행 도구: PLY 정보 확인, 합성 장면 생성.

use std::process::ExitCode;

use skylens_core::ply::read_ply_file;
use skylens_core::synth::{Scene, SceneConfig};

const USAGE: &str =
    "사용법:\n  skylens-stream ply-info <파일.ply>\n  skylens-stream synth <출력 폴더> [폭 높이]";

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
            if let [w, h] = rest {
                match (w.parse(), h.parse()) {
                    (Ok(w), Ok(h)) => (cfg.width, cfg.height) = (w, h),
                    _ => {
                        eprintln!("{USAGE}");
                        return ExitCode::from(2);
                    }
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
