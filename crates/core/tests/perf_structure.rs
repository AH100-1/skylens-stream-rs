//! 연산량 기반 구조 단언. 벽시계 시간은 기계 부하에 따라 흔들리므로(`#[ignore]` 시간 시험은
//! 기본 실행에서 빠진다) 일하는 양을 정하는 구조를 숫자로 고정해 성능 회귀를 잡는다.
//! 시간 측정 자체는 `examples/bench_stages.rs` 가 맡는다.

use skylens_core::features::{detect, DetectorConfig, GrayImage, Keypoint};
use skylens_core::matching::{
    adaptive_iterations, candidate_pairs, ratio_match, MIN_RANSAC_ITERS, PAIR_CROSS, PAIR_POW2_MAX,
    PAIR_TEMPORAL,
};
use skylens_core::synth::{CamId, Scene, SceneConfig};

/// 3 카메라 × `positions` 위치의 (카메라, 위치) 목록.
fn rig_views(positions: usize) -> Vec<(usize, usize)> {
    (0..positions)
        .flat_map(|p| (0..3).map(move |c| (c, p)))
        .collect()
}

/// 짝 수는 위치 수에 선형이어야 한다(전수 짝 O(n²) 로 되돌아가면 매칭 시간이 240장에서 약 9 배).
///
/// 위치 P, 카메라 3대, temporal=5, cross=4, pow2_max=16 일 때 해석적으로
/// - 같은 카메라: 간격 1..=5 와 8, 16 → 카메라마다 Σ(P−d) = 7P − (15 + 24)
/// - 다른 카메라: 카메라 쌍 3개 × [간격 0 은 P, 간격 1..=4 는 양방향 2(P−d)] = 3(9P − 20)
///
/// 합 = 48P − 177. P = 80(240장) → 3663 짝, 장당 이웃 상한 2·7 + 2·9 = 32.
#[test]
fn candidate_pairs_grow_linearly_with_positions() {
    assert_eq!((PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX), (5, 4, 16));
    for p in [20usize, 40, 80, 160] {
        let pairs = candidate_pairs(&rig_views(p), PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);
        assert_eq!(pairs.len(), 48 * p - 177, "위치 {p}");
    }
    let views = rig_views(80);
    let pairs = candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);
    assert_eq!(pairs.len(), 3663);
    let mut degree = vec![0usize; views.len()];
    for &(i, j) in &pairs {
        degree[i] += 1;
        degree[j] += 1;
    }
    assert!(degree.iter().all(|&d| d <= 32));
    // 경로 가운데 영상은 상한에 닿는다(가장자리만 덜하다).
    assert_eq!(degree[3 * 40], 32);
}

/// 시험용 실측 배치 장면의 한 장(480×270, F 카메라, 위치 10).
fn test_image() -> GrayImage {
    let scene = Scene::new(SceneConfig {
        positions: 20,
        width: 480,
        height: 270,
        ..SceneConfig::default()
    });
    let v = scene
        .views
        .iter()
        .find(|v| v.cam == CamId::F && v.position == 10)
        .expect("F 카메라 위치 10");
    let (rgb, _) = scene.render(v);
    GrayImage::from_rgb(rgb.width as usize, rgb.height as usize, &rgb.data)
}

/// 특징 수 상한이 지켜져야 짝 하나의 기술자 비교 수(|A|·|B|, 전수 탐색)가 `max_features²` 로 묶인다.
#[test]
fn feature_cap_bounds_descriptor_comparisons() {
    let img = test_image();
    let full = detect(&img, &DetectorConfig::default());
    // 상한이 실제로 작동하는 영상이어야 시험이 의미가 있다.
    let cap = 200;
    assert!(full.len() > cap, "검출 {} 개", full.len());
    let capped = detect(
        &img,
        &DetectorConfig {
            max_features: cap,
            ..DetectorConfig::default()
        },
    );
    assert!(capped.len() <= cap && capped.len() >= cap * 9 / 10);
    // 상호 매칭 결과는 양쪽 수의 최솟값을 넘지 못한다.
    let feats = skylens_core::features::detect_and_describe(
        &img,
        &DetectorConfig {
            max_features: cap,
            ..DetectorConfig::default()
        },
    );
    let m = ratio_match(&feats, &feats[..cap / 2], 0.8, true);
    assert!(m.len() <= cap / 2);
}

/// 검출 층 수: 옥타브 o 개, 옥타브당 간격 s 면 가장 큰 스케일은
/// σ0 · 2^(o−1) · 2^((s+1)/s) (맨 위 옥타브의 맨 위 DoG 층 + 부화소 보정 반 층 여유) 이하다.
/// 옥타브를 늘리면 층(흐림 횟수 o·(s+3))이 늘고, 큰 스케일 특징이 나타난다.
#[test]
fn detected_scales_follow_octave_count() {
    let img = test_image();
    let max_sigma = |kps: &[Keypoint]| kps.iter().map(|k| k.sigma).fold(0f32, f32::max);
    for octaves in [1usize, 2, 4] {
        let cfg = DetectorConfig {
            octaves,
            ..DetectorConfig::default()
        };
        let kps = detect(&img, &cfg);
        assert!(!kps.is_empty());
        let bound = cfg.sigma0 * 2f32.powi(octaves as i32 - 1) * 2f32.powf(4.0 / 3.0);
        let m = max_sigma(&kps);
        assert!(m <= bound, "옥타브 {octaves}: 최대 σ {m} > {bound}");
        if octaves > 1 {
            // 한 옥타브 아래의 상한을 넘는 특징이 있어야 위 옥타브가 실제로 돈 것이다.
            let lower = cfg.sigma0 * 2f32.powi(octaves as i32 - 2) * 2f32.powf(4.0 / 3.0);
            assert!(m > lower, "옥타브 {octaves}: 최대 σ {m} ≤ {lower}");
        }
    }
}

/// RANSAC 반복 수 = ⌈ln(1−p) / ln(1−wˢ)⌉ (Fischler & Bolles 1981), [50, max] 로 자름.
/// w=0.5, s=5, p=0.999: ln(0.001)/ln(1−1/32) = 6.9078/0.031749 = 217.6 → 218.
/// w=0.8, s=8: 6.9078/0.18364 = 37.6 → 바닥 50. w=0.1, s=5: 약 69 만 → 상한 2000.
#[test]
fn ransac_iteration_counts() {
    assert_eq!(MIN_RANSAC_ITERS, 50);
    assert_eq!(adaptive_iterations(0.5, 5, 0.999, 2000), 218);
    assert_eq!(adaptive_iterations(0.8, 8, 0.999, 2000), 50);
    assert_eq!(adaptive_iterations(0.1, 5, 0.999, 2000), 2000);
}

/// FNV-1a 64 비트.
fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

/// 검출 결과 회귀: 고정 합성 영상(480×270)의 특징 개수와 (x, y, σ, 응답, 방향) 비트 해시.
/// 흐림 누산 순서·극값 판정·정렬이 바뀌면 값이 달라진다. 기대값은 main 08d5248 에서 한 번 계산했다.
#[test]
fn detection_hash_regression() {
    let img = test_image();
    let kps = detect(&img, &DetectorConfig::default());
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for k in &kps {
        for v in [k.x, k.y, k.sigma, k.response, k.angle] {
            fnv(&mut h, &v.to_bits().to_le_bytes());
        }
    }
    println!("검출 {} 개, 해시 {h:#018x}", kps.len());
    assert_eq!((kps.len(), h), (EXPECTED_COUNT, EXPECTED_HASH));
}

const EXPECTED_COUNT: usize = 753;
const EXPECTED_HASH: u64 = 0x618d_ef58_7be0_2b14;
