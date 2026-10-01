//! `run` 하위 명령: 데이터셋을 읽고 구역 목록을 보이며 출력 폴더 구조를 만든다.

use std::path::Path;
use std::process::ExitCode;

use skylens_core::dataset::{load_dataset, DatasetConfig};

pub const USAGE: &str =
    "skylens-stream run <입력폴더> <출력폴더> [--stride N] [--span N] [--ovl N]";

/// 출력 폴더 아래에 만드는 하위 폴더.
pub const OUTPUT_DIRS: [&str; 3] = ["preview", "refined", "snapshots"];

fn parse_options(rest: &[&str]) -> Result<DatasetConfig, String> {
    let mut cfg = DatasetConfig::default();
    let mut it = rest.iter();
    while let Some(&key) = it.next() {
        let slot = match key {
            "--stride" => &mut cfg.stride,
            "--span" => &mut cfg.span,
            "--ovl" => &mut cfg.ovl,
            _ => return Err(format!("알 수 없는 옵션: {key}")),
        };
        let v = it.next().ok_or_else(|| format!("{key} 뒤에 값이 없음"))?;
        *slot = v
            .parse()
            .map_err(|_| format!("{key} 값이 자연수가 아님: {v}"))?;
    }
    if cfg.stride == 0 || cfg.span == 0 {
        return Err("--stride, --span 은 1 이상".into());
    }
    Ok(cfg)
}

pub fn run(input: &str, output: &str, rest: &[&str]) -> ExitCode {
    let cfg = match parse_options(rest) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}\n사용법:\n  {USAGE}");
            return ExitCode::from(2);
        }
    };
    let ds = match load_dataset(Path::new(input), cfg) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let out = Path::new(output);
    for d in OUTPUT_DIRS {
        if let Err(e) = std::fs::create_dir_all(out.join(d)) {
            eprintln!("{}: {e}", out.join(d).display());
            return ExitCode::FAILURE;
        }
    }
    let chunks = ds.chunks();
    println!("stride {} span {} ovl {}", cfg.stride, cfg.span, cfg.ovl);
    println!("positions {}", ds.positions.len());
    println!("images {}", ds.image_count());
    println!("chunks {}", chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        println!("chunk {i} {}..{}", c.start, c.end);
    }
    ExitCode::SUCCESS
}
