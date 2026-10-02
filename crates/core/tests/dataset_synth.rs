//! 합성 장면 출력 폴더를 데이터 로더가 손대지 않고 그대로 읽는지 확인한다.
//!
//! 정답은 합성 설정에서 나온다: 위치 P 곳 × 카메라 3대 = 3P 장, gps.txt 는 사진마다 한 줄(3P 줄),
//! 각 위치의 GPS 동-북-위 좌표는 `Scene::gps_enu`.

use std::path::PathBuf;

use skylens_core::dataset::{load_dataset, parse_gps, DatasetConfig, CAMERAS};
use skylens_core::synth::{Scene, SceneConfig};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("skylens_dsynth_{tag}_{}", std::process::id()));
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

/// 기본 장면(80 위치)과 같은 배치·GPS, 렌더 크기만 작게(시험 시간 단축). 폴더 구조와 파일 수는
/// 렌더 크기와 무관하다.
fn small_default_scene() -> Scene {
    Scene::new(SceneConfig {
        width: 96,
        height: 54,
        ..SceneConfig::default()
    })
}

#[test]
fn loader_reads_synth_output_as_is() {
    let scene = small_default_scene();
    let n_pos = SceneConfig::default().positions;
    assert_eq!(n_pos, 80);
    let t = TempDir::new("asis");
    scene.write_dataset(&t.0).unwrap();

    // GPS 행 수: 사진마다 한 줄 → 80 × 3 = 240.
    let gps_text = std::fs::read_to_string(t.0.join("gps.txt")).unwrap();
    let gps = parse_gps(&gps_text).unwrap();
    assert_eq!(gps.len(), 240);

    // STRIDE 1: 모든 위치를 그대로 → 80 위치, 240 장, 카메라마다 80 장.
    let cfg = DatasetConfig {
        stride: 1,
        ..DatasetConfig::default()
    };
    let ds = load_dataset(&t.0, cfg).unwrap();
    assert_eq!(ds.positions.len(), 80);
    assert_eq!(ds.image_count(), 240);
    for (c, cam) in CAMERAS.iter().enumerate() {
        let n = ds
            .positions
            .iter()
            .filter(|p| p.images[c].is_file())
            .filter(|p| {
                p.images[c].file_name().unwrap().to_str().unwrap()
                    == format!("{cam}_{:04}.jpg", p.frame)
            })
            .count();
        assert_eq!(n, 80, "{cam}");
    }
    for (i, p) in ds.positions.iter().enumerate() {
        assert_eq!(p.index, i);
        assert_eq!(p.frame, i as u32);
        // 합성 출력은 평평한 구조: images/camF_0000.jpg.
        assert_eq!(
            p.images[0],
            t.0.join("images").join(format!("camF_{i:04}.jpg"))
        );
    }

    // 위치 좌표: 로더 원점은 gps.txt 첫 기록(= 위치 0 camF 의 GPS). 합성 GPS 는 장(드론)마다
    // 따로이므로 위치 i 카메라 c 의 정답은 gps_enu[3i+c] − gps_enu[0].
    // 허용 1 cm: gps.txt 기록 정밀도(위경도 1e-9° ≈ 0.1 mm, 고도 1 mm)와 두 접평면 원점 차
    // (수 m / 지구 반지름 × 이동 거리 200 m ≈ 0.1 mm)보다 충분히 크고, 위치 간격 2.5 m 보다 훨씬 작다.
    // 위치 좌표는 세 사진 좌표의 평균(편대 중심).
    for (i, p) in ds.positions.iter().enumerate() {
        let mut mean = skylens_core::nalgebra::Vector3::zeros();
        for c in 0..3 {
            let want = scene.gps_enu[3 * i + c] - scene.gps_enu[0];
            let err = (p.image_enu[c] - want).norm();
            assert!(err < 0.01, "위치 {i} 카메라 {c}: 오차 {err} m");
            mean += want / 3.0;
        }
        let err = (p.enu - mean).norm();
        assert!(err < 0.01, "위치 {i} 평균: 오차 {err} m");
    }
    assert!(ds.skipped.is_empty());

    // 기본 설정(STRIDE 3, SPAN 12, OVL 2): 프레임 0,3,…,78 → 27 위치, 81 장.
    // 구역 start 0,12,24 → [0,14) [10,26) [22,27). 24 + OVL < 27 이라 꼬리 구역은 앞 구역 끝 26
    // 너머 위치 26 을 가지므로 남는다(SPEC §3.5 범위 그대로).
    let ds3 = load_dataset(&t.0, DatasetConfig::default()).unwrap();
    assert_eq!(ds3.positions.len(), 27);
    assert_eq!(ds3.image_count(), 81);
    assert_eq!(ds3.positions.last().unwrap().frame, 78);
    assert_eq!(ds3.chunks(), vec![0..14, 10..26, 22..27]);
}

#[test]
fn synth_output_moved_into_camera_folders_reads_same() {
    // 같은 합성 출력을 카메라별 하위 폴더로 옮겨도 같은 결과.
    let scene = Scene::new(SceneConfig {
        positions: 10,
        width: 64,
        height: 36,
        ..SceneConfig::default()
    });
    let t = TempDir::new("sub");
    scene.write_dataset(&t.0).unwrap();
    let cfg = DatasetConfig {
        stride: 1,
        ..DatasetConfig::default()
    };
    let flat = load_dataset(&t.0, cfg).unwrap();
    let images = t.0.join("images");
    for cam in CAMERAS {
        std::fs::create_dir_all(images.join(cam)).unwrap();
        for f in 0..10 {
            let name = format!("{cam}_{f:04}.jpg");
            std::fs::rename(images.join(&name), images.join(cam).join(&name)).unwrap();
        }
    }
    let sub = load_dataset(&t.0, cfg).unwrap();
    assert_eq!(flat.positions.len(), 10);
    assert_eq!(sub.positions.len(), 10);
    assert_eq!(sub.image_count(), 30);
    for (a, b) in flat.positions.iter().zip(&sub.positions) {
        assert_eq!(a.frame, b.frame);
        assert_eq!(a.enu, b.enu);
        assert_eq!(a.image_enu, b.image_enu);
        for (c, cam) in CAMERAS.iter().enumerate() {
            let name = a.images[c].file_name().unwrap();
            assert_eq!(b.images[c], images.join(cam).join(name));
        }
    }
}
