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
