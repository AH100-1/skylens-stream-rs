use std::path::{Path, PathBuf};
use std::process::Command;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_run_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 프레임 0..frames, 8×8 jpg 와 gps.txt. GPS 는 사진마다 한 줄, 드론마다 위도 1e-4°(약 11 m) 차.
fn make_dataset(root: &Path, frames: u32) {
    let img = image::RgbImage::from_pixel(8, 8, image::Rgb([90, 120, 150]));
    let mut gps = String::new();
    for f in 0..frames {
        for (c, cam) in ["camF", "camR", "camL"].iter().enumerate() {
            gps += &format!(
                "{cam}_{f:04}.jpg {} {} 30.0\n",
                37.5 + c as f64 * 1e-4,
                127.0 + f as f64 * 1e-5
            );
        }
    }
    for cam in ["camF", "camR", "camL"] {
        let d = root.join("images").join(cam);
        std::fs::create_dir_all(&d).unwrap();
        for f in 0..frames {
            img.save(d.join(format!("{cam}_{f:04}.jpg"))).unwrap();
        }
    }
    std::fs::write(root.join("gps.txt"), gps).unwrap();
}

/// 데이터셋 읽기·구역 목록만 본다(복원은 `tests/pipeline.rs` 에서). 단색 사진이라 복원은 돌지 않는다.
fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("run")
        .args(args)
        .arg("--list-only")
        .output()
        .unwrap()
}

#[test]
fn run_lists_positions_and_chunks() {
    let t = TempDir::new("ok");
    let input = t.0.join("in");
    let output = t.0.join("out");
    // 프레임 0..40, STRIDE 3 → 0,3,…,39 의 14곳, 사진 42장.
    // SPAN 5, OVL 1 → start 0,5,10: [0,6) [4,11) [9,14).
    make_dataset(&input, 40);
    let out = run(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--span",
        "5",
        "--ovl",
        "1",
    ]);
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(
        out.status.success(),
        "{s}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(s.contains("positions 14\n"), "{s}");
    assert!(s.contains("images 42\n"), "{s}");
    assert!(s.contains("skipped 0\n"), "{s}");
    assert!(
        s.contains("chunks 3\nchunk 0 0..6\nchunk 1 4..11\nchunk 2 9..14\n"),
        "{s}"
    );
    for d in ["preview", "refined", "snapshots"] {
        assert!(output.join(d).is_dir(), "{d}");
    }
}

#[test]
fn run_defaults_and_stride_option() {
    let t = TempDir::new("stride");
    let input = t.0.join("in");
    make_dataset(&input, 10);
    let o = t.0.join("o");
    // 기본 STRIDE 3: 0,3,6,9 → 4곳, SPAN 12 → 구역 1개 0..4.
    let out = run(&[input.to_str().unwrap(), o.to_str().unwrap()]);
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(
        s.contains("positions 4\n") && s.contains("chunk 0 0..4\n"),
        "{s}"
    );
    // STRIDE 4: 0,4,8 → 3곳.
    let out = run(&[
        input.to_str().unwrap(),
        o.to_str().unwrap(),
        "--stride",
        "4",
    ]);
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("positions 3\n"), "{s}");
}

#[test]
fn run_reports_gps_error_line() {
    let t = TempDir::new("bad");
    let input = t.0.join("in");
    make_dataset(&input, 4);
    std::fs::write(
        input.join("gps.txt"),
        "camF_0000.jpg 37 127 1\ncamF_0003.jpg 37 127\n",
    )
    .unwrap();
    let out = run(&[input.to_str().unwrap(), t.0.join("o").to_str().unwrap()]);
    assert!(!out.status.success());
    let e = String::from_utf8(out.stderr).unwrap();
    assert!(e.contains("2번째 줄"), "{e}");
}

#[test]
fn run_rejects_bad_options_and_missing_input() {
    let t = TempDir::new("opt");
    let o = t.0.join("o");
    let out = run(&[t.0.to_str().unwrap(), o.to_str().unwrap(), "--span", "x"]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(&[t.0.to_str().unwrap(), o.to_str().unwrap(), "--stride", "0"]);
    assert_eq!(out.status.code(), Some(2));
    for bad in [
        ["--dense-method", "x"],
        ["--position", "x"],
        ["--gps-sigma-h", "0"],
    ] {
        let out = run(&[t.0.to_str().unwrap(), o.to_str().unwrap(), bad[0], bad[1]]);
        assert_eq!(out.status.code(), Some(2), "{bad:?}");
    }
    let out = run(&[t.0.join("nope").to_str().unwrap(), o.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(!o.exists());
}

#[test]
fn run_reports_skipped_frames() {
    let t = TempDir::new("skip");
    let input = t.0.join("in");
    make_dataset(&input, 40);
    // camR 6 이 빠짐 → 위치 13곳, 건너뜀 1.
    std::fs::remove_file(input.join("images/camR/camR_0006.jpg")).unwrap();
    let o = t.0.join("o");
    let out = run(&[input.to_str().unwrap(), o.to_str().unwrap()]);
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(out.status.success(), "{s}");
    assert!(s.contains("positions 13\n"), "{s}");
    assert!(
        s.contains("skipped 1 (frames 6)\nskip frame 6 missing camR\n"),
        "{s}"
    );
    // camL 20..39 이 빠짐 → 연속 7곳 건너뜀, 기본 허용 2 를 넘어 종료 코드 1.
    for f in 20..40 {
        std::fs::remove_file(input.join(format!("images/camL/camL_{f:04}.jpg"))).unwrap();
    }
    let out = run(&[input.to_str().unwrap(), o.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    let e = String::from_utf8(out.stderr).unwrap();
    assert!(e.contains("7곳 연속"), "{e}");
    // 허용을 늘리면 통과하고 7곳 + camR 6 을 모두 보고한다.
    let out = run(&[
        input.to_str().unwrap(),
        o.to_str().unwrap(),
        "--max-skip-run",
        "7",
    ]);
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(out.status.success(), "{s}");
    assert!(
        s.contains("positions 6\n") && s.contains("skipped 8 (frames 6,21,24,27,30,33,36,39)\n"),
        "{s}"
    );
}

#[test]
fn run_accepts_per_drone_gps() {
    // 같은 프레임 세 드론 GPS 가 서로 다름(약 11 m): 거부하지 않는다.
    let t = TempDir::new("pergps");
    let input = t.0.join("in");
    make_dataset(&input, 7);
    let out = run(&[input.to_str().unwrap(), t.0.join("o").to_str().unwrap()]);
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(
        out.status.success(),
        "{s}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(s.contains("positions 3\n"), "{s}");
}

#[test]
fn run_reports_frame_missing_from_all_cameras() {
    let t = TempDir::new("allgone");
    let input = t.0.join("in");
    make_dataset(&input, 40);
    for cam in ["camF", "camR", "camL"] {
        std::fs::remove_file(input.join(format!("images/{cam}/{cam}_0006.jpg"))).unwrap();
    }
    let o = t.0.join("o");
    let out = run(&[input.to_str().unwrap(), o.to_str().unwrap()]);
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(out.status.success(), "{s}");
    assert!(s.contains("positions 13\n"), "{s}");
    assert!(
        s.contains("skipped 1 (frames 6)\nskip frame 6 missing camF,camR,camL\n"),
        "{s}"
    );
}

#[test]
fn run_summarizes_huge_gap_as_range() {
    // 프레임 0 과 5000000 만 있고 허용 10^7: 출력은 구간 요약 한 줄이고 짧다.
    let t = TempDir::new("biggap");
    let input = t.0.join("in");
    let mut gps = String::new();
    for cam in ["camF", "camR", "camL"] {
        let d = input.join("images").join(cam);
        std::fs::create_dir_all(&d).unwrap();
        for f in [0u32, 5_000_000] {
            std::fs::write(d.join(format!("{cam}_{f:04}.jpg")), b"x").unwrap();
            gps += &format!("{cam}_{f:04}.jpg 37.5 127.0 30.0\n");
        }
    }
    std::fs::write(input.join("gps.txt"), gps).unwrap();
    let o = t.0.join("o");
    let t0 = std::time::Instant::now();
    let out = run(&[
        input.to_str().unwrap(),
        o.to_str().unwrap(),
        "--stride",
        "1",
        "--max-skip-run",
        "10000000",
    ]);
    let dt = t0.elapsed().as_secs_f64();
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(out.status.success(), "{s}");
    assert!(dt < 5.0, "{dt} s");
    assert!(s.len() < 1024, "{} bytes", s.len());
    assert!(
        s.contains("skipped 4999999 (frames 1..=4999999 step 1)\n")
            && s.contains("skip frames 1..=4999999 step 1 missing camF,camR,camL\n"),
        "{s}"
    );
}
