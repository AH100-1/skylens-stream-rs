//! 연산량 기반 구조 단언. 벽시계 시간은 기계 부하에 따라 흔들리므로 여기서는 시간을 재지 않고,
//! 일하는 양을 정하는 구조(짝 수, 짝당 기술자 비교 수 상한, 검출 층 수, E RANSAC 최소 반복)를 숫자로
//! 고정해 성능 회귀를 잡는다. 시간 측정 자체는 `benches/pipeline.rs` 가 맡는다.
//! 검출 결과 해시 회귀 시험은 `features.rs` 시험 모듈 한 곳에서만 고정한다(여기에 두지 않는다).

use skylens_core::features::{detect, DetectorConfig, GrayImage, Keypoint};
use skylens_core::matching::{candidate_pairs, PAIR_CROSS, PAIR_POW2_MAX, PAIR_TEMPORAL};
use skylens_core::synth::{CamId, Scene, SceneConfig};
use skylens_core::two_view::ESSENTIAL_MIN_ITERS;

#[path = "../benches/support/args.rs"]
#[allow(dead_code)]
mod bench_args;

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
    test_image_at(CamId::F, 10)
}

/// 시험용 실측 배치 장면(위치 20곳, 480×270)의 (카메라, 위치) 한 장.
fn test_image_at(cam: CamId, position: usize) -> GrayImage {
    let scene = Scene::new(SceneConfig {
        positions: 20,
        width: 480,
        height: 270,
        ..SceneConfig::default()
    });
    let v = scene
        .views
        .iter()
        .find(|v| v.cam == cam && v.position == position)
        .expect("시험 장면에 있는 카메라·위치");
    let (rgb, _) = scene.render(v);
    GrayImage::from_rgb(rgb.width as usize, rgb.height as usize, &rgb.data)
}

/// 특징 수 상한이 지켜져야 짝 하나의 기술자 비교 수(|A|·|B|, 전수 탐색)가 `max_features²` 로 묶인다.
/// 상한이 실제로 작동하는(상한 없이 검출하면 상한보다 많은) 영상 세 장으로 짝마다 확인한다.
#[test]
fn feature_cap_bounds_descriptor_comparisons() {
    let cap = 200;
    let capped_cfg = DetectorConfig {
        max_features: cap,
        ..DetectorConfig::default()
    };
    let imgs = [
        test_image_at(CamId::F, 10),
        test_image_at(CamId::F, 11),
        test_image_at(CamId::R, 10),
    ];
    let mut counts = Vec::new();
    for img in &imgs {
        let full = detect(img, &DetectorConfig::default()).len();
        assert!(full > cap, "상한 없이 검출 {full} 개 ≤ 상한 {cap}");
        let n = detect(img, &capped_cfg).len();
        // 상한 아래로 너무 많이 잘리면 상한이 아니라 다른 것이 수를 정하고 있다.
        assert!(n <= cap && n >= cap * 9 / 10, "상한 {cap} 에서 검출 {n} 개");
        counts.push(n);
    }
    for a in 0..counts.len() {
        for b in a + 1..counts.len() {
            assert!(counts[a] * counts[b] <= cap * cap);
        }
    }
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

/// E RANSAC 의 최소 반복 수. 축소 실행에서 E RANSAC 이 짝당 가장 비싼 구간이고(짝당 수십 ms),
/// 정상 비율이 높아 적응 반복 수가 바닥으로 내려가는 짝이 대부분이라 이 값이 그 구간 시간을 거의 그대로 정한다.
/// 100 은 `two_view::ransac_essential` 이 쌍둥이 해도 표본에 나오도록 둔 바닥값이다(조기 포기 계수와 함께 정했다. 적응 반복 수
/// ⌈ln(1−p)/ln(1−w⁵)⌉ 은 w=0.8·p=0.999 에서 18 이라 바닥이 없으면 대부분의 짝이 20 회 안팎에서 끝난다).
/// 이 값을 바꾸면 E RANSAC 시간이 비례해 바뀌므로 시간 측정 노트와 함께 고친다.
#[test]
fn essential_ransac_min_iterations_fixed() {
    assert_eq!(ESSENTIAL_MIN_ITERS, 100);
}

fn argv(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_string).collect()
}

/// bench 묶음 인자(`--full`·`--quick`)는 놓인 자리와 상관없이 먼저 적용되고 개별 인자가 이긴다.
/// 예전에는 `--positions 20 --full` 이 위치 80 으로 바뀌어 줄인 측정이 전체 규모로 돌았다.
#[test]
fn bench_args_preset_order_does_not_matter() {
    let a = bench_args::parse(&argv("--positions 20 --full"));
    let b = bench_args::parse(&argv("--full --positions 20"));
    assert_eq!(a, b);
    assert_eq!(a.positions, 20);
    // 나머지는 `--full` 값.
    assert_eq!(
        (a.width, a.height, a.repeat, a.ba_points),
        (960, 540, 3, 20_000)
    );

    // 해상도·반복도 같은 규칙.
    let c = bench_args::parse(&argv("--width 320 --height 180 --repeat 1 --full --bench"));
    let d = bench_args::parse(&argv("--bench --full --width 320 --height 180 --repeat 1"));
    assert_eq!(c, d);
    assert_eq!(
        (c.positions, c.width, c.height, c.repeat),
        (80, 320, 180, 1)
    );

    let e = bench_args::parse(&argv("--ba-points 500 --quick"));
    assert_eq!(e.ba_points, 500);
    assert_eq!(e.positions, 8);

    // 묶음끼리는 뒤의 것이 이긴다.
    assert_eq!(bench_args::parse(&argv("--quick --full")).positions, 80);
    assert_eq!(bench_args::parse(&argv("--full --quick")).positions, 8);
}

/// 인자 없는 실행은 빠른 규모(24장·480×270)이고, 240장은 `--full` 을 줄 때만이다.
#[test]
fn bench_args_default_is_quick_scale() {
    let a = bench_args::parse(&[]);
    assert_eq!(a, bench_args::parse(&argv("--quick")));
    assert_eq!((a.positions, a.width, a.height), (8, 480, 270));
    assert_eq!(bench_args::parse(&argv("--full")).positions, 80);
    // ba-scale 은 위치를 따로 주지 않으면 80, 주면 그 값(묶음 인자도 위치를 준 것으로 본다).
    assert_eq!(bench_args::parse(&argv("--mode ba-scale")).positions, 80);
    assert_eq!(
        bench_args::parse(&argv("--positions 4 --mode ba-scale")).positions,
        4
    );
    assert_eq!(
        bench_args::parse(&argv("--mode ba-scale --quick")).positions,
        8
    );
}
