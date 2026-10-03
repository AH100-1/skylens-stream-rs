//! `run` 하위 명령: 데이터셋을 읽고 구역 목록을 보이며 출력 폴더 구조를 만든다.

use std::path::Path;
use std::process::ExitCode;

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline, PipelineConfig};

pub const USAGE: &str =
    "skylens-stream run <입력폴더> <출력폴더> [--stride N] [--span N] [--ovl N] [--max-skip-run N] [--max-features N] [--dense-width N] [--hfov DEG] [--ba-iters N] [--list-only]";

/// 출력 폴더 아래에 만드는 하위 폴더.
pub const OUTPUT_DIRS: [&str; 3] = ["preview", "refined", "snapshots"];

fn parse_options(rest: &[&str]) -> Result<(DatasetConfig, PipelineConfig, bool), String> {
    let mut cfg = DatasetConfig::default();
    let mut pc = PipelineConfig::default();
    let mut list_only = false;
    let mut it = rest.iter();
    while let Some(&key) = it.next() {
        if key == "--list-only" {
            list_only = true;
            continue;
        }
        if key == "--hfov" {
            let v = it.next().ok_or("--hfov 뒤에 값이 없음")?;
            pc.hfov_deg = v
                .parse()
                .ok()
                .filter(|h: &f64| *h > 1.0 && *h < 179.0)
                .ok_or_else(|| format!("--hfov 값이 1..179 도가 아님: {v}"))?;
            continue;
        }
        let slot = match key {
            "--max-features" => &mut pc.max_features,
            "--dense-width" => &mut pc.dense_width,
            "--ba-iters" => &mut pc.ba_iters,
            "--stride" => &mut cfg.stride,
            "--span" => &mut cfg.span,
            "--ovl" => &mut cfg.ovl,
            "--max-skip-run" => &mut cfg.max_skip_run,
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
    if pc.max_features == 0 || pc.dense_width < 8 {
        return Err("--max-features 는 1 이상, --dense-width 는 8 이상".into());
    }
    Ok((cfg, pc, list_only))
}

pub fn run(input: &str, output: &str, rest: &[&str]) -> ExitCode {
    let (cfg, pcfg, list_only) = match parse_options(rest) {
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
    let skipped = ds.skipped_frames();
    if skipped.is_empty() {
        println!("skipped 0");
    } else {
        let list: Vec<String> = skipped.iter().map(u32::to_string).collect();
        println!("skipped {} (frames {})", skipped.len(), list.join(","));
        for s in &ds.skipped {
            println!("skip frame {} missing {}", s.frame, s.missing.join(","));
        }
    }
    println!("chunks {}", chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        println!("chunk {i} {}..{}", c.start, c.end);
    }
    if list_only {
        return ExitCode::SUCCESS;
    }
    match run_pipeline(&ds, &pcfg, out) {
        Ok(res) => {
            for i in &res.issues {
                println!("issue {i}");
            }
            println!("done regions {}", res.regions.len());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
