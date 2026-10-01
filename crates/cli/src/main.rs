//! skylens-stream 명령행 도구. 지금은 PLY 정보 확인만 한다.

use std::process::ExitCode;

use skylens_core::ply::read_ply_file;

const USAGE: &str = "사용법: skylens-stream ply-info <파일.ply>";

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
