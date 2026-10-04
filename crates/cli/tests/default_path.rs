//! 기본 인자 그대로 `synth` → `run` → `verify` (F-343). 기본 구역 길이가 짧아 구역 사이 겹침이 없던 때
//! 3구역에서 등록이 61/81 로 빠지고 verify 가 5/7 이었다. 기본 경로는 960x540 240장 중 stride 3 → 27위치 81장, 1구역이다.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn cli(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

/// 약 2.5 분(4코어 부하 중 release): 일반 시험으로 둔다.
#[test]
fn default_synth_run_verify_passes_all_seven() {
    let root: PathBuf =
        std::env::temp_dir().join(format!("skylens_default_path_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let (inp, out) = (root.join("in"), root.join("out"));
    let (inp_s, out_s) = (inp.to_str().unwrap(), out.to_str().unwrap());

    let (code, so, se) = cli(&["synth", inp_s]);
    assert_eq!(code, 0, "{so}{se}");
    assert!(so.contains("views 240"), "{so}");

    let start = Instant::now();
    let (code, so, se) = cli(&["run", inp_s, out_s]);
    eprintln!("run secs {:.1}", start.elapsed().as_secs_f64());
    assert_eq!(code, 0, "{so}{se}");
    assert!(so.contains("images 81"), "{so}");
    assert!(so.contains("chunks 1"), "{so}");

    for d in ["preview", "refined", "snapshots"] {
        assert!(out.join(d).is_dir(), "{d} 폴더 없음");
    }
    assert!(out.join("snapshots/manifest.json").is_file());

    let (code, table, se) = cli(&["verify", out_s]);
    eprintln!("{table}{se}");
    assert_eq!(code, 0, "{table}");
    assert!(table.contains("7/7"), "{table}");
    assert_eq!(table.matches("| PASS |").count(), 7, "{table}");
    assert!(table.contains("초벌 81/81, 정밀 81/81"), "{table}");

    let poses = std::fs::read_to_string(out.join("poses.txt")).unwrap();
    assert_eq!(poses.lines().count(), 81, "등록 수");
    let _ = std::fs::remove_dir_all(&root);
}
