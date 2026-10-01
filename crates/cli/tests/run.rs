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

/// 프레임 0..frames, 8×8 jpg 와 gps.txt.
fn make_dataset(root: &Path, frames: u32) {
    let img = image::RgbImage::from_pixel(8, 8, image::Rgb([90, 120, 150]));
    for cam in ["camF", "camR", "camL"] {
        let d = root.join("images").join(cam);
        std::fs::create_dir_all(&d).unwrap();
        for f in 0..frames {
            img.save(d.join(format!("{cam}_{f:04}.jpg"))).unwrap();
        }
    }
    let gps: String = (0..frames)
        .map(|f| format!("camF_{f:04}.jpg 37.5 {} 30.0\n", 127.0 + f as f64 * 1e-5))
        .collect();
    std::fs::write(root.join("gps.txt"), gps).unwrap();
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("run")
        .args(args)
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
    let out = run(&[t.0.join("nope").to_str().unwrap(), o.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(!o.exists());
}
