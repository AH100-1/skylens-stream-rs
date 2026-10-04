//! `run` 하위 명령: 데이터셋을 읽고 구역 목록을 보이며 출력 폴더 구조를 만든다.

use std::path::Path;
use std::process::ExitCode;

use skylens_core::dataset::{load_dataset, DatasetConfig};
use skylens_core::pipeline::{run_pipeline, DenseMethod, PipelineConfig, PositionMethod};

pub const USAGE: &str =
    "skylens-stream run <입력폴더> <출력폴더> [--stride N] [--span N] [--ovl N] [--max-skip-run N] [--cross-offset N] [--max-features N] [--dense-width N] [--hfov DEG] [--ba-iters N] [--dense-method sweep|patchmatch] [--position gps|translation-averaging] [--preview-ba-iters N] [--preview-refine-iters N] [--gps-sigma-h M] [--gps-sigma-v M] [--tri-loose-frac F] [--tri-median-k K] [--tri-min-px PX] [--list-only]";

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
        if key == "--dense-method" {
            let v = it.next().ok_or("--dense-method 뒤에 값이 없음")?;
            pc.dense_method = match *v {
                "sweep" => DenseMethod::Sweep,
                "patchmatch" => DenseMethod::PatchMatch,
                _ => return Err(format!("--dense-method 값이 sweep|patchmatch 가 아님: {v}")),
            };
            continue;
        }
        if key == "--position" {
            let v = it.next().ok_or("--position 뒤에 값이 없음")?;
            pc.position = match *v {
                "gps" => PositionMethod::GpsLeastSquares,
                "translation-averaging" | "ta" => PositionMethod::TranslationAveraging,
                _ => {
                    return Err(format!(
                        "--position 값이 gps|translation-averaging 이 아님: {v}"
                    ))
                }
            };
            continue;
        }
        let fslot = match key {
            "--gps-sigma-h" => Some(&mut pc.gps_sigma_h),
            "--gps-sigma-v" => Some(&mut pc.gps_sigma_v),
            "--tri-loose-frac" => Some(&mut pc.tri_loose_frac),
            "--tri-median-k" => Some(&mut pc.tri_median_k),
            "--tri-min-px" => Some(&mut pc.tri_min_px),
            _ => None,
        };
        if let Some(slot) = fslot {
            let v = it.next().ok_or_else(|| format!("{key} 뒤에 값이 없음"))?;
            *slot = v
                .parse()
                .ok()
                .filter(|x: &f64| x.is_finite() && *x > 0.0)
                .ok_or_else(|| format!("{key} 값이 0 보다 큰 수가 아님: {v}"))?;
            continue;
        }
        let slot = match key {
            "--preview-ba-iters" => &mut pc.preview_ba_iters,
            "--preview-refine-iters" => &mut pc.preview_refine_iters,
            "--max-features" => &mut pc.max_features,
            "--dense-width" => &mut pc.dense_width,
            "--ba-iters" => &mut pc.ba_iters,
            "--stride" => &mut cfg.stride,
            "--span" => &mut cfg.span,
            "--ovl" => &mut cfg.ovl,
            "--max-skip-run" => &mut cfg.max_skip_run,
            "--cross-offset" => &mut pc.cross_offset,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<(DatasetConfig, PipelineConfig, bool), String> {
        parse_options(args)
    }

    #[test]
    fn defaults_when_no_options() {
        let (_, pc, lo) = parse(&[]).unwrap();
        let d = PipelineConfig::default();
        assert_eq!(pc.dense_method, d.dense_method);
        assert_eq!(pc.position, d.position);
        assert_eq!(pc.preview_ba_iters, d.preview_ba_iters);
        assert_eq!(pc.preview_refine_iters, d.preview_refine_iters);
        assert!(!lo);
    }

    #[test]
    fn pipeline_options_set_config() {
        let (_, pc, _) = parse(&[
            "--dense-method",
            "patchmatch",
            "--position",
            "translation-averaging",
            "--preview-ba-iters",
            "8",
            "--gps-sigma-h",
            "1.5",
            "--gps-sigma-v",
            "3",
            "--tri-loose-frac",
            "0.03",
            "--tri-median-k",
            "4",
            "--tri-min-px",
            "0.5",
        ])
        .unwrap();
        assert_eq!(pc.dense_method, DenseMethod::PatchMatch);
        assert_eq!(pc.position, PositionMethod::TranslationAveraging);
        assert_eq!(pc.preview_ba_iters, 8);
        assert_eq!((pc.gps_sigma_h, pc.gps_sigma_v), (1.5, 3.0));
        assert_eq!(
            (pc.tri_loose_frac, pc.tri_median_k, pc.tri_min_px),
            (0.03, 4.0, 0.5)
        );
        let (_, pc, _) = parse(&["--dense-method", "sweep", "--position", "gps"]).unwrap();
        assert_eq!(pc.dense_method, DenseMethod::Sweep);
        assert_eq!(pc.position, PositionMethod::GpsLeastSquares);
    }

    #[test]
    fn cross_offset_option() {
        let (_, pc, _) = parse(&[]).unwrap();
        assert_eq!(pc.cross_offset, 24);
        let (_, pc, _) = parse(&["--cross-offset", "0"]).unwrap();
        assert_eq!(pc.cross_offset, 0);
        assert!(parse(&["--cross-offset", "x"]).is_err());
    }

    #[test]
    fn bad_values_are_errors() {
        for args in [
            &["--dense-method", "x"][..],
            &["--dense-method"],
            &["--position", "x"],
            &["--gps-sigma-h", "0"],
            &["--gps-sigma-v", "-1"],
            &["--gps-sigma-h", "nan"],
            &["--tri-min-px", "abc"],
            &["--preview-ba-iters", "-1"],
            &["--preview-ba-iters", "1.5"],
        ] {
            assert!(parse(args).is_err(), "{args:?}");
        }
    }
}
