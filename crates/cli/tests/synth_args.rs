use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(args)
        .output()
        .unwrap()
}

fn out_dir(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!("skylens_synth_args_{tag}_{}", std::process::id()))
        .to_string_lossy()
        .into_owned()
}

fn assert_usage_error(out: &Output, expect: &str) {
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(!err.contains("panicked"), "{err}");
    assert!(err.contains("사용법:"), "{err}");
    assert!(err.contains(expect), "{err}");
}

#[test]
fn zero_size_is_usage_error() {
    let dir = out_dir("zero");
    let out = run(&["synth", &dir, "0", "0"]);
    assert_usage_error(&out, "16..=8192");
    assert!(!std::path::Path::new(&dir).exists());
}

#[test]
fn single_size_argument_is_usage_error() {
    let dir = out_dir("one");
    let out = run(&["synth", &dir, "64"]);
    assert_usage_error(&out, "추가 인자 1개");
    assert!(!std::path::Path::new(&dir).exists());
}

#[test]
fn three_size_arguments_is_usage_error() {
    let dir = out_dir("three");
    let out = run(&["synth", &dir, "64", "48", "5"]);
    assert_usage_error(&out, "추가 인자 3개");
    assert!(!std::path::Path::new(&dir).exists());
}

#[test]
fn out_of_range_and_non_numeric_sizes_are_usage_errors() {
    for (w, h) in [("15", "64"), ("64", "8193"), ("-64", "64"), ("64", "x")] {
        let out = run(&["synth", &out_dir("range"), w, h]);
        assert_usage_error(&out, "16..=8192");
    }
}
