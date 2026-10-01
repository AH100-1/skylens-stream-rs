use std::process::Command;

use skylens_core::ply::{write_ply_file, PointCloud, PointRecord};

#[test]
fn ply_info_reports_point_count() {
    let cloud = PointCloud {
        points: (0..42)
            .map(|i| PointRecord {
                xyz: [i as f32, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                rgb: [1, 2, 3],
            })
            .collect(),
    };
    let dir = std::env::temp_dir().join(format!("skylens_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.ply");
    write_ply_file(&path, &cloud).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("ply-info")
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("points 42"), "{s}");
    assert!(s.contains("nan false"), "{s}");
}

#[test]
fn missing_file_fails() {
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .args(["ply-info", "/nonexistent/x.ply"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("skylens_cli_bad_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn huge_vertex_count_fails_cleanly() {
    let header = b"ply\nformat binary_little_endian 1.0\nelement vertex 10000000000\n\
property float x\nproperty float y\nproperty float z\nend_header\n";
    let path = write_tmp("big.ply", header);
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("ply-info")
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(!err.contains("panicked"), "{err}");
    assert!(err.contains("잘림"), "{err}");
}

#[test]
fn truncated_vertex_data_fails_cleanly() {
    let mut bytes = b"ply\nformat binary_little_endian 1.0\nelement vertex 4\n\
property float x\nproperty float y\nproperty float z\nend_header\n"
        .to_vec();
    bytes.extend_from_slice(&[0u8; 47]); // 48 바이트 필요
    let path = write_tmp("short.ply", &bytes);
    let out = Command::new(env!("CARGO_BIN_EXE_skylens-stream"))
        .arg("ply-info")
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("48 바이트 필요"));
}
