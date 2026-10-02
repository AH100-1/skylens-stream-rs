//! synth 인자 검사 시험. 검사가 빠져 렌더가 시작되면 시간 제한에 걸려 실패한다.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// 인자 오류는 즉시 끝나야 한다. 이 시간 안에 끝나지 않으면 렌더가 시작된 것으로 본다.
const LIMIT: Duration = Duration::from_secs(10);

struct Done {
    code: Option<i32>,
    stderr: String,
}

fn run_in(cwd: Option<&Path>, args: &[&str]) -> Done {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skylens-stream"));
    cmd.args(args).stdout(Stdio::null()).stderr(Stdio::piped());
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let mut child = cmd.spawn().unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        if start.elapsed() > LIMIT {
            child.kill().ok();
            child.wait().ok();
            panic!("{args:?}: {LIMIT:?} 안에 끝나지 않음(인자 검사 없이 렌더가 시작됨)");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .ok();
    Done {
        code: status.code(),
        stderr,
    }
}

/// 시험마다 다른 임시 경로. 끝나면(실패해도) 지운다.
struct TempPath(PathBuf);

impl TempPath {
    fn new(tag: &str) -> Self {
        let p =
            std::env::temp_dir().join(format!("skylens_synth_args_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        Self(p)
    }

    fn s(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_usage_error(out: &Done, expect: &str) {
    let err = &out.stderr;
    assert_eq!(out.code, Some(2), "{err}");
    assert!(!err.contains("panicked"), "{err}");
    assert!(err.contains("사용법:"), "{err}");
    assert!(err.contains(expect), "{err}");
}

#[test]
fn zero_size_is_usage_error() {
    let dir = TempPath::new("zero");
    let out = run_in(None, &["synth", dir.s(), "0", "0"]);
    assert_usage_error(&out, "오류: 폭·높이는 16..=8192");
    assert!(!dir.0.exists());
}

#[test]
fn single_size_argument_is_usage_error() {
    let dir = TempPath::new("one");
    let out = run_in(None, &["synth", dir.s(), "64"]);
    assert_usage_error(&out, "추가 인자 1개");
    assert!(!dir.0.exists());
}

#[test]
fn three_size_arguments_is_usage_error() {
    let dir = TempPath::new("three");
    let out = run_in(None, &["synth", dir.s(), "64", "48", "5"]);
    assert_usage_error(&out, "추가 인자 3개");
    assert!(!dir.0.exists());
}

#[test]
fn out_of_range_and_non_numeric_sizes_are_usage_errors() {
    for (i, (w, h)) in [
        ("15", "64"),
        ("64", "8193"),
        ("-64", "64"),
        ("64", "x"),
        ("+20", "20"),
        ("20", "+20"),
        (" 20", "20"),
        ("", "20"),
    ]
    .into_iter()
    .enumerate()
    {
        let dir = TempPath::new(&format!("range{i}"));
        let out = run_in(None, &["synth", dir.s(), w, h]);
        assert_usage_error(&out, "오류: 폭·높이는 16..=8192");
        assert!(!dir.0.exists(), "{w:?} {h:?}");
    }
}

#[test]
fn empty_output_path_is_usage_error_and_writes_nothing() {
    let cwd = TempPath::new("emptycwd");
    std::fs::create_dir_all(&cwd.0).unwrap();
    for args in [&["synth", ""][..], &["synth", "", "64", "48"][..]] {
        let out = run_in(Some(&cwd.0), args);
        assert_usage_error(&out, "오류: synth 출력 폴더가 빈 문자열");
    }
    assert_eq!(std::fs::read_dir(&cwd.0).unwrap().count(), 0);
}

#[test]
fn help_prints_usage_to_stdout() {
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("--help")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.contains("사용법:") && s.contains("synth <출력 폴더> [폭 높이]"),
        "{s}"
    );
}
