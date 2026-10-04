# skylens-stream-rs

[한국어](#한국어) · [English](#english)

---

## 한국어

드론 3대 편대(드론마다 카메라 1대, 모두 3대) 영상에서 **점진적으로 나아지는 3D 점군**을 만드는 Rust 도구.

새 구역이 들어올 때마다 빠른 초벌 점군을 먼저 보여 주고, 정밀 계산이 끝난 구역은 정밀본으로 바꿔 끼운다.
화면에는 단계마다 "이전 구역은 정밀본 + 최신 구역은 초벌" 점군이 나간다.

> 개발 중이다. 지금 명령행 도구에는 `ply-info`·`synth`·`run`·`verify` 가 있고, `run` 이 아래 단계를 묶어 초벌·정밀·스냅샷을 낸다. 점진적 점군 생성은 아래 라이브러리 단계(특징점 → 매칭 → 두 시점 자세 → 회전 평균 → 번들 조정 → GPS 정렬 → 구역 분할 → 밀집 준비·융합 → 초벌 정렬·스냅샷 = 점진 스트림)를 묶어 만들어 가는 중이다.

### 설치

```bash
cargo build --release
# 실행 파일: target/release/skylens-stream
# 빌드 없이 바로 실행: cargo run --release -p skylens-stream -- <명령>
```

Rust stable 1.88 이상이 필요하다.

### 입력

```
<데이터>/
  images/camF_0000.jpg ...   # 앞 카메라 (camF_<위치 번호 4자리>.jpg)
  images/camR_0000.jpg ...   # 오른쪽
  images/camL_0000.jpg ...   # 왼쪽
  gps.txt                    # 한 줄에: 확장자 없는 이미지 이름(camF_0000) 위도 경도(도) 고도(m)
  truth/cameras.txt          # synth 출력에만: 정답 카메라
  truth/origin.txt           # synth 출력에만: 정답 좌표 원점
```

`gps.txt` 의 이름에는 확장자를 붙이지 않는다(`camF_0000.jpg` 가 아니라 `camF_0000`).
`synth` 출력도 같은 구조이며, 정답 카메라 `truth/cameras.txt`(한 줄에: 이름 fx fy cx cy 폭 높이, 세계→카메라 R 행 우선 9개, t 3개)와
정답 좌표 원점 `truth/origin.txt`(`위도 경도 고도` 한 줄)가 더 있다.

좌표계: 출력 좌표는 첫 GPS(첫 영상의 위도·경도·고도)를 원점으로 하는 동-북-위 지역 직교 좌표(미터)다(SPEC §2).
`truth/cameras.txt` 의 정답 좌표는 첫 GPS 가 아니라 `truth/origin.txt` 를 원점으로 하는 동-북-위(미터)라서, 결과와 비교할 때는
라이브러리의 `Scene::to_first_gps_frame` 으로 첫 GPS 원점 좌표로 옮긴다(옮기지 않으면 기본 장면에서 고도만 약 27 m 어긋난다).

### 사용법

```bash
# 점군 파일의 점 개수와 NaN 여부 출력 ("points <개수>", "nan <true|false>")
# 읽을 수 없거나 잘린 파일이면 오류 메시지와 종료 코드 1
skylens-stream ply-info <파일.ply>

# 시험용 합성 장면 만들기 (영상 + GPS + 정답 카메라, 기본 960×540, 3대 × 80위치 = 240장)
# 폭·높이는 16..=8192 이며 숫자만 쓴다(`+20`·`-20` 불가). 둘 다 주거나 둘 다 생략한다
# 출력 폴더가 빈 문자열이거나 폭·높이가 맞지 않으면 아무것도 쓰지 않고 사용법과 종료 코드 2
# 끝나면 만든 영상 수를 출력 ("views 240"). 쓰기 실패면 오류 메시지와 종료 코드 1
skylens-stream synth <출력 폴더> [폭 높이]

# 출력 폴더 검사(SPEC §4 검증 기준 일곱 항목을 PASS/FAIL/판정 불가 표로 출력)
# 종료 코드: FAIL 이 하나라도 있으면 1, FAIL 없이 판정 불가가 있으면 2, 모두 PASS 면 0
# 등록 사진 수·구역 사진 수·재투영 오차는 출력 폴더의 report.json 이 있을 때만 판정한다
skylens-stream verify <출력 폴더>

# 판 번호 출력 ("skylens-stream 0.1.0")
skylens-stream --version

# 사용법을 표준 출력으로 내고 종료 코드 0 (-h 도 같음)
skylens-stream --help
```

인자 없이 실행하거나 모르는 명령이면 사용법을 표준 오류로 내고 종료 코드 2 로 끝난다.

### 합성 장면으로 한 번 돌려 보기

`synth` → `run` → `verify` 를 끝까지 잇는다. 인자 없이 돌리면 구역은 SPAN 12·OVL 2 이고, 카메라 사이 겹침이 12~40 위치 떨어진 짝에서 생기므로 R·L 카메라의 구역 창을 `--cross-offset`(기본 24 위치)만큼 뒤로 밀어 한 구역 안에서 세 카메라가 이어진다(기본 장면 3구역, 81/81, verify 7/7: `skylens-stream synth d/in && skylens-stream run d/in d/out && skylens-stream verify d/out`, 시험 `crates/cli/tests/default_path.rs`). 아래 명령은 `--span 48` 로 구역을 크게 잡는 설정이다.
같은 명령을 `crates/cli/tests/pipeline_e2e.rs` 가 프로세스로 돌려 아래 수치에 상한을 건다(`cargo test --release -p skylens-stream --test pipeline_e2e`, 구역 2개 시험 포함 약 3~6 분).

```bash
skylens-stream synth scene 320 180
skylens-stream run scene out --stride 2 --span 48 --ovl 2 --max-features 800 --dense-width 96 --hfov 65 --ba-iters 15
skylens-stream verify out
# 구역 2개(80위치 전부, 이웃 구역 겹침 항목 판정)
skylens-stream run scene out2 --stride 1 --span 48 --ovl 2 --max-features 800 --dense-width 96 --hfov 65 --ba-iters 15
skylens-stream verify out2
```

출력 폴더에는 `preview/`·`refined/`·`snapshots/`(와 `snapshots/manifest.json`)·`poses.txt`·`report.json` 이 생긴다. 4코어 기계(부하 약 20)에서 실측:

| | 단구역 (`out`, 120장) | 구역 2개 (`out2`, 240장) |
|---|---|---|
| run 시간 | 약 30 초 (부하 낮을 때) | 332 초 (부하 22), 시험 전체 179 초 (부하 낮을 때) |
| verify | 7/7, 종료 코드 0 | 6/7, 종료 코드 1 |
| 미달 항목 | 없음 (preview_vs_refined 최근접 중앙 1.416 m, 높이 차 중앙 1.553 m 통과) | preview_vs_refined (최근접 4.923 m, 높이 차 5.829 m). preview_align 은 통과(점쌍 최소 1221, 스케일 차 0.05%, 잔차 5.130 m) |
| refined_overlap | 해당 없음 (구역 1개) | PASS, 1쌍 겹침 차 중앙 최대 0.257 m |
| 재투영 (초벌 → 정밀) | 3.026 → 0.300 px | 3.232 → 0.282 px |
| 카메라 중심 오차 중앙 / 최대 | 0.329 / 2.912 m | 0.290 / 3.115 m |
| 정밀 점 → 정답 표면 중앙 / 95% | 0.475 / 1.465 m (점 11054) | 0.494 / 3.022 m (점 19855) |

오차는 정답 카메라 `truth/cameras.txt`·정답 표면(`Scene::surface_height`, 수직 거리)과 `poses.txt`·`refined/*.ply` 를 비교한 값이다(정답 원점과 첫 GPS 원점 차를 옮겨서). 회전 오차는 `poses.txt` 에 카메라 중심만 있어 재지 못한다.

### 라이브러리: 특징점

```rust
use skylens_core::features::{detect_and_describe, DetectorConfig, GrayImage};

// 버퍼 길이가 폭×높이×3 이 아니면 Err(ImageSizeError). 회색조·RGBA 는 try_from_gray·try_from_rgba.
// from_rgb 는 같은 검사를 하되 길이가 틀리면 메시지와 함께 패닉한다.
// 밝기 값(f32) 버퍼가 이미 있으면 try_from_vec(폭, 높이, data). 필드는 비공개이고
// width()·height()·data() 로 읽는다(길이가 폭×높이와 다른 영상은 만들 수 없다).
let img = GrayImage::try_from_rgb(width, height, &rgb_bytes)?;
let feats = detect_and_describe(&img, &DetectorConfig::default());
for f in &feats {
    // f.kp: x, y(화소 번호 규약: 화소 (i, j) 의 중심이 (i, j)), sigma(스케일), angle(라디안), response
    //       카메라 픽셀 좌표(화소 중심 = i + 0.5)로는 +0.5, 정규 좌표로는 k.index_to_normalized
    // f.desc: 단위 길이 128차원 기술자
}
// 흐림·극값 탐색·기술자는 rayon 으로 병렬 실행된다(스레드 수와 무관하게 결과가 같다).
// 스레드 수는 RAYON_NUM_THREADS 환경 변수로 정한다.
```

### 라이브러리: 매칭과 기하 검증

```rust
use skylens_core::matching::{candidate_pairs, ratio_match, ransac_fundamental, RansacConfig};
use skylens_core::matching::{PAIR_CROSS, PAIR_POW2_MAX, PAIR_TEMPORAL};
use skylens_core::features::Feature;

// 매칭할 영상 짝: views[k] = (카메라 번호, 촬영 위치 번호).
// 같은 카메라는 위치 차 1..=5 와 PAIR_POW2_MAX(16) 이하의 2의 거듭제곱(8, 16), 다른 카메라는 위치 차 0..=4.
let image_pairs = candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);

// 비율 0.8, 양방향 확인. 결과는 a 인덱스 순, 거리 계산은 한 번만 하며 병렬로 돈다.
let pairs = ratio_match(&feats_a, &feats_b, 0.8, true);
// 특징점 좌표(화소 번호) → 카메라 픽셀 좌표(화소 중심 = i + 0.5)
let px = |f: &Feature| nalgebra::Vector2::new(f.kp.x as f64 + 0.5, f.kp.y as f64 + 0.5);
let x1: Vec<_> = pairs.iter().map(|&(i, _)| px(&feats_a[i])).collect();
let x2: Vec<_> = pairs.iter().map(|&(_, j)| px(&feats_b[j])).collect();
if let Some((f, inliers)) = ransac_fundamental(&x1, &x2, &RansacConfig::default()) {
    // f: 기본 행렬(x2ᵀ F x1 = 0), inliers[k]: pairs[k] 가 기하적으로 맞는지
}
// None: 점 8개 미만, 길이 불일치, NaN·무한대 좌표, 동일선상 등 퇴화 배치,
// 또는 정상 비율이 min_inlier_ratio(기본 0.2) 미만. 문턱 threshold_px 는 px 단위,
// sampson_error 는 제곱 거리(px²)를 돌려준다.
```

시간 측정(1920×1080 한 장 검출, 7300×7300 매칭): `cargo test --release -- --ignored --test-threads=1 --nocapture timing`
구간별 시간(합성 장면, 기본 8위치 × 3대 = 24장 480×270, `--full` 이면 240장 960×540): `cargo bench --bench pipeline -- --quick`(인자 설명은 `crates/core/benches/pipeline.rs` 머리말).
표의 구간은 합성 렌더·특징 검출·짝 생성·비율 매칭·RANSAC F(8점)·RANSAC E(5점)·두 시점 자세·회전 평균(정답 그래프)·회전 평균(검증 결과)·번들 조정이다.
회전 평균은 정답 상대 회전 그래프와 검증된 짝에서 얻은 그래프 두 가지로 잰다.
표를 파일로도 남기려면 `cargo bench --bench pipeline -- --quick --json <파일>`.

### 라이브러리: 두 시점 상대 자세와 삼각측량

```rust
use skylens_core::two_view::{essential_from_fundamental, refine_relative_pose, triangulate};
// 최소 해법이 필요하면 essential_5pt(&n1[..5], &n2[..5]) 가 본질 행렬 후보(최대 10개)를 준다.

// f, inliers: 위 RANSAC 결과. k: 카메라 내부 파라미터(Intrinsics, 왜곡 계수 `dist` 포함).
// k.to_normalized 가 역왜곡까지 한다. 역왜곡 수렴 여부가 필요하면 k.unproject(p)(실패 시 None).
let n1: Vec<_> = x1.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let n2: Vec<_> = x2.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let e = essential_from_fundamental(&f, &k, &k);
if let Some(rp) = refine_relative_pose(&e, &n1, &n2, 50).filter(|rp| rp.translation_observable) {
    // 카메라 1 = [I|0], 카메라 2 = [R|t] (t 는 단위 길이, 스케일 미정)
    let x = triangulate(&rp.rotation, &rp.translation, &n1[0], &n2[0]); // 카메라 1 좌표계의 3D 점
}
// refine_relative_pose 는 선형 E 의 분해 후보 여러 곳에서 Sampson 비용 LM 을 돌려 비용이 가장 작은 해를
// 키랄리티로 골라 준다(선형 해 하나만 다듬으면 회전–이동 혼동 국소 최소에 갇힐 수 있음).
// 선형 해만 필요하면 recover_pose(&e, &n1, &n2), 시작점을 직접 주려면 refine_pose 를 쓴다.
// 둘 다 길이 불일치, NaN 입력에서 None.
// 순수 회전·아주 짧은 기선(이동 방향을 관측할 수 없음)이면 translation_observable = false 이고
// rotation 만 믿을 수 있다(translation 은 0). 회전 평균에는 이 회전도 쓸 수 있다.
```

내부 파라미터를 아는 짝은 5점 본질 행렬 RANSAC 으로 바로 검증할 수 있다(지면처럼 평면에 가까운 장면에서도 동작).

```rust
use skylens_core::matching::RansacConfig;
use skylens_core::two_view::ransac_essential_candidates;
// n1, n2: 모든 대응의 정규화 좌표(k.to_normalized). 문턱 cfg.threshold_px 는 픽셀, k.fx 로 환산한다.
let cands = ransac_essential_candidates(&n1, &n2, k.fx, &RansacConfig::default());
// 평면 장면은 두 시점만으로 두 겹 모호하므로 후보(E, 정상 표시)를 최대 4개 돌려준다.
// 정상 짝 대부분이 한 평면(호모그래피)으로 설명되면 그 평면에서 벗어난 짝은 정상에서 뺀다.
// 정답 후보 고르기는 셋째 시점 등 다른 정보로 한다. 하나만 필요하면 ransac_essential(...) 이 첫 후보를 준다.
```

### 라이브러리: 회전 평균

```rust
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};

// 간선 (i, j): rotation = R_j R_iᵀ (recover_pose 의 회전과 같은 규약), weight = 정상 대응 수 등
let edges = vec![RelativeRotation { i: 0, j: 1, rotation: pose01.rotation, weight: 120.0 } /* ... */];
if let Some(res) = average_rotations(num_views, &edges, &AveragingConfig::default()) {
    // res.rotations[v]: 세계→카메라 v 회전(기준 정점 = 단위 회전). 기준과 이어지지 않으면 None
    // res.inliers[k], res.residuals_rad[k]: 간선별 정상 표시와 잔차 각(rad)
    // 이상치 문턱은 max(5°, 6 × 잔차로 추정한 잡음 σ̂) 이고 실제 값은 res.outlier_threshold_rad
}
// None: 정점 0개, 범위 밖 번호, 쓸 수 있는 간선 없음. NaN 회전·0 이하 가중치·자기 간선은 무시.
```

### 라이브러리: 번들 조정

```rust
use skylens_core::ba::{bundle_adjust, BaOptions, BaProblem, Loss, Observation};

// groups[g]: 내부 파라미터 그룹(DistortedIntrinsics: fx fy cx cy + 왜곡 k1 k2 p1 p2), 카메라 폴더마다 하나
// poses[c]: 카메라 c 의 세계→카메라 포즈(x_c = R x + t), camera_group[c]: c 가 속한 그룹 번호
// points[p]: 3D 점 초기값. 관측 픽셀은 카메라 픽셀 좌표(화소 중심 = i + 0.5)
let observations = vec![Observation { camera: 0, point: 0, pixel: nalgebra::Vector2::new(512.5, 300.5) } /* ... */];
let mut problem = BaProblem { groups, poses, camera_group, points, observations };
let opts = BaOptions {
    loss: Loss::Huber(2.0), // Squared, Huber(δ), Cauchy(δ) — δ 는 픽셀
    // 자유 파라미터 마스크, 순서는 INTRINSIC_NAMES = [fx, fy, cx, cy, k1, k2, p1, p2]
    default_free_intrinsics: [true, true, true, true, true, true, false, false],
    fixed_cameras: vec![0], // 포즈를 고정할 카메라(게이지)
    ..BaOptions::default()  // 최대 50회, 트랙(점) 최대 100 000개
};
let report = bundle_adjust(&mut problem, &opts); // problem 의 포즈·점·내부 파라미터를 제자리에서 고친다
println!("RMS {:.3} → {:.3} px, {} 회", report.initial_rms, report.final_rms, report.iterations);
```

점 블록을 슈어 보수로 소거해 카메라 쪽 축소 계통만 푸는 희소 Levenberg–Marquardt 다.
그룹마다 따로 마스크를 주려면 `free_intrinsics[g]`(비어 있으면 모든 그룹에 `default_free_intrinsics`).
점은 관측 수가 많은 것부터(같으면 번호 순) `max_tracks` 개만 쓰고, 선택되지 않은 점은 그대로 둔다.
`max_iterations = 0` 이면 비용만 평가하고(`report.refined = false`), 1 이상이면 고친다.
`report` 에는 그 밖에 쓴 카메라·트랙·관측 수, 처음·마지막 비용, 수렴 여부가 있다.

### 라이브러리: 밀집 준비·융합·점진 스트림

명령행에는 아직 묶여 있지 않은 단계들이다.

- `skylens_core::align`: 닮음 변환 `Similarity { s, r, t }`(`identity`·`apply_point`·`apply_normal`(회전만)·`inverse`·`compose`·`to_matrix4`).
  `umeyama(&src, &dst)` 는 대응점 최소제곱 닮음 변환, `robust_similarity(&src, &dst, iters, floor_m)` 는 반복 트리밍
  (임계 max(3 × 잔차 중앙값, `floor_m`))으로 `(변환, 정상 표시, 잔차 중앙값)` 을 준다.
  `align_to_enu(&centers, &enu, max_residual_m)` 은 정밀 포즈의 카메라 중심을 동-북-위 좌표에 1회 정렬하고 잔차가 상한을 넘는 대응을 빼고 다시 푼다.
  `gps_align(&centers, &gps, &origin)` 은 위경도(`geo::Geodetic`)를 `origin` 기준 동-북-위로 바꾼 뒤 상한 `GPS_MAX_RESIDUAL_M`(3 m)로 `align_to_enu` 를 부른다.
  결과 `GpsAlignment` 에는 `sim`·`inliers`·`residuals`·`median_residual` 이 있다. 상한은 3 m 고정이고, 정상 대응이 3개 미만으로 줄면 첫 추정을 그대로 쓴다.
  `None` 은 길이 불일치, 대응 3개 미만, NaN·무한대, 카메라 중심이 한 점이거나 일직선인 퇴화 배치일 때다(`umeyama` 와 같은 조건).
  SPEC §2 출력 좌표로 정렬하려면 `origin` 에 첫 GPS 를 준다(합성 장면은 `Scene::first_gps_origin()`). `truth/origin.txt` 는 정답 좌표의 원점일 뿐 출력 원점이 아니다.
- `skylens_core::undistort`: `undistort_to_long_side(&img, &k, long_side)` 가 왜곡 있는 사진을 긴 변 `long_side` 화소의 핀홀 사진과 새 `Intrinsics` 로 바꾼다(원본 밖 화소는 0, 쌍선형 보간).
- `skylens_core::view_selection`: `select_neighbors(&views, &points, k)` 가 희소 점을 함께 본 사진들 가운데 사진마다 이웃 최대 k 장(기본 `DEFAULT_NEIGHBORS` = 8)을 고르고, `depth_range(&view, &points)` 가 깊이 탐색 범위를 준다.
- `skylens_core::fusion`: `fuse(&views, &depth_maps, FusionConfig)` 가 사진별 깊이 맵을 왕복 재투영 검사로 합쳐 `PointCloud` 를 만든다(동의 사진 수 `min_views` 이상).
- `skylens_core::stream`: `split_regions(위치 수, span, overlap)` 로 구역을 나누고, `align_region` 으로 초벌 구역을 정밀본에 닮음 변환 정렬, `build_snapshots` 로 단계별 점군("이전 구역 정밀본 + 최신 구역 초벌")을 만든 뒤 `write_outputs(출력 폴더, ...)` 가 다음을 쓴다.

```
<출력>/
  preview/preview_00_pos0-14.ply ...   # 구역별 초벌 (pos<시작>-<끝, 끝 미포함>)
  refined/refined_00_pos0-14.ply ...   # 구역별 정밀본
  snapshots/step_01_1regions.ply ...   # 단계별 화면용 점군
  snapshots/step_final_all_refined.ply
  snapshots/manifest.json              # 단계별 점 수, 정렬 수치
```

### PLY 형식

`ply-info` 가 읽고 라이브러리(`skylens_core::ply`)가 쓰는 형식은 이진 little-endian PLY 다.
쓸 때는 점마다 `x y z nx ny nz`(float32) + `red green blue`(uint8) 이다. 읽을 때는 `vertex` 원소가 첫 원소여야 하고,
속성을 이름으로 찾으므로 순서가 달라도 되며, 법선·색이 없으면 0 으로 채운다.
헤더에는 `format binary_little_endian 1.0` 줄이 정확히 한 번 있어야 하고, 헤더 한 줄은 4096 바이트,
헤더 전체는 64 KiB 를 넘을 수 없다(넘으면 그 이상 읽지 않고 오류).

---

## English

A Rust tool that builds a **progressively refined 3D point cloud** from the video of a three-drone formation (one camera per drone, three cameras in all).

Each time a new region arrives, a fast preview cloud is shown first; once the accurate solve for a region finishes, its preview is swapped for the refined cloud.
At every step the output is "refined clouds for earlier regions + preview for the newest region".

> Work in progress. The command-line tool currently has `ply-info`, `synth`, `run` and `verify`; `run` chains the stages below into preview, refined and snapshot outputs. The progressive point cloud is being assembled from the library stages below (features → matching → two-view pose → rotation averaging → bundle adjustment → GPS alignment → region split → dense preparation and fusion → preview alignment and snapshots = progressive stream).

### Build

```bash
cargo build --release
# binary: target/release/skylens-stream
# or build and run in one step: cargo run --release -p skylens-stream -- <command>
```

Requires stable Rust 1.88 or newer.

### Input

```
<data>/
  images/camF_0000.jpg ...   # front camera (camF_<4-digit position>.jpg)
  images/camR_0000.jpg ...   # right camera
  images/camL_0000.jpg ...   # left camera
  gps.txt                    # one line per image: image name without extension (camF_0000) latitude longitude (deg) altitude (m)
  truth/cameras.txt          # synth output only: ground-truth cameras
  truth/origin.txt           # synth output only: ground-truth coordinate origin
```

Names in `gps.txt` carry no extension (`camF_0000`, not `camF_0000.jpg`).
`synth` writes the same layout plus ground-truth cameras in `truth/cameras.txt` (one line per image: name fx fy cx cy width height, 9 entries of the world→camera R row-major, 3 entries of t) and
the ground-truth coordinate origin in `truth/origin.txt` (one line: `latitude longitude altitude`).

Coordinates: output coordinates are local east-north-up Cartesian coordinates (metres) whose origin is the first GPS fix (latitude, longitude and altitude of the first image) (SPEC §2).
The ground-truth coordinates in `truth/cameras.txt` are east-north-up (metres) about `truth/origin.txt`, not about the first GPS fix, so before comparing them with results
move them into the first-GPS frame with `Scene::to_first_gps_frame` in the library (without it the default scene is off by about 27 m in altitude alone).

### Usage

```bash
# Print the point count and whether any NaN is present ("points <count>", "nan <true|false>")
# Unreadable or truncated files print an error and exit with code 1
skylens-stream ply-info <file.ply>

# Generate a synthetic test scene (images + GPS + ground-truth cameras; default 960×540, 3 cameras × 80 positions = 240 images)
# width and height are in 16..=8192, digits only (no `+20` or `-20`), given together or both omitted
# An empty output path or invalid width/height writes nothing and prints usage with exit code 2
# Prints the number of images written when done ("views 240"); write errors print a message and exit with code 1
skylens-stream synth <output dir> [width height]

# Check an output folder (prints the seven SPEC §4 criteria as a PASS/FAIL/undecided table)
# Exit code: 1 if any FAIL, 2 if no FAIL but some undecided, 0 if all PASS
# Registered images, region images and reprojection error are decided only when report.json is in the output folder
skylens-stream verify <output dir>

# Run a synthetic scene end to end (synth -> run -> verify); cross-camera overlap only appears between positions 12-40 apart, so use regions of 40+ positions (--span 48)
skylens-stream synth scene 320 180
skylens-stream run scene out --stride 2 --span 48 --ovl 2 --max-features 800 --dense-width 96 --hfov 65 --ba-iters 15
skylens-stream verify out
# two or more regions (neighbor-overlap item is judged): --stride 1 uses all 80 positions
skylens-stream run scene out2 --stride 1 --span 48 --ovl 2 --max-features 800 --dense-width 96 --hfov 65 --ba-iters 15
skylens-stream verify out2

# Print the version ("skylens-stream 0.1.0")
skylens-stream --version

# Print the usage to standard output and exit with code 0 (-h does the same)
skylens-stream --help
```

Running with no arguments or an unknown command prints the usage to standard error and exits with code 2.

### Library: features

```rust
use skylens_core::features::{detect_and_describe, DetectorConfig, GrayImage};

// Err(ImageSizeError) unless the buffer length is width×height×3. Use try_from_gray / try_from_rgba for
// other layouts. from_rgb performs the same check but panics with a message on a length mismatch.
// For an existing f32 intensity buffer use try_from_vec(width, height, data). Fields are private;
// read them with width(), height() and data() (an image whose length differs from width×height cannot be built).
let img = GrayImage::try_from_rgb(width, height, &rgb_bytes)?;
let feats = detect_and_describe(&img, &DetectorConfig::default());
for f in &feats {
    // f.kp: x, y (pixel-index convention: the centre of pixel (i, j) is (i, j)), sigma (scale), angle (radians), response
    //       add 0.5 for camera pixel coordinates (pixel centre = i + 0.5), or use k.index_to_normalized
    // f.desc: unit-length 128-dim descriptor
}
// Blurring, extremum search and descriptors run in parallel with rayon (results do not depend on the
// thread count). Set the thread count with the RAYON_NUM_THREADS environment variable.
```

### Library: matching and geometric verification

```rust
use skylens_core::matching::{candidate_pairs, ratio_match, ransac_fundamental, RansacConfig};
use skylens_core::matching::{PAIR_CROSS, PAIR_POW2_MAX, PAIR_TEMPORAL};
use skylens_core::features::Feature;

// Image pairs to match: views[k] = (camera index, capture position index).
// Same camera: position gap 1..=5 plus powers of two up to PAIR_POW2_MAX (16), i.e. 8 and 16; different cameras: gap 0..=4.
let image_pairs = candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);

// ratio 0.8, mutual check. Results are in a-index order; distances are computed once, in parallel.
let pairs = ratio_match(&feats_a, &feats_b, 0.8, true);
// keypoint coordinates (pixel index) → camera pixel coordinates (pixel centre = i + 0.5)
let px = |f: &Feature| nalgebra::Vector2::new(f.kp.x as f64 + 0.5, f.kp.y as f64 + 0.5);
let x1: Vec<_> = pairs.iter().map(|&(i, _)| px(&feats_a[i])).collect();
let x2: Vec<_> = pairs.iter().map(|&(_, j)| px(&feats_b[j])).collect();
if let Some((f, inliers)) = ransac_fundamental(&x1, &x2, &RansacConfig::default()) {
    // f: fundamental matrix (x2ᵀ F x1 = 0), inliers[k]: whether pairs[k] is geometrically consistent
}
// None: fewer than 8 points, length mismatch, NaN/infinite coordinates, degenerate layouts such as
// collinear points, or an inlier ratio below min_inlier_ratio (default 0.2). threshold_px is in px;
// sampson_error returns the squared distance (px²).
```

Timing (detection on one 1920×1080 image, 7300×7300 matching): `cargo test --release -- --ignored --test-threads=1 --nocapture timing`
Per-stage timing (synthetic scene, default 8 positions × 3 cameras = 24 images at 480×270, 240 images at 960×540 with `--full`): `cargo bench --bench pipeline -- --quick` (arguments are described at the top of `crates/core/benches/pipeline.rs`).
The table rows are synthetic render (합성 렌더), feature detection (특징 검출), pair generation (짝 생성), ratio matching (비율 매칭), RANSAC F (8-point), RANSAC E (5-point), two-view pose (두 시점 자세), rotation averaging on the ground-truth graph (회전 평균(정답 그래프)), rotation averaging on the verified result (회전 평균(검증 결과)) and bundle adjustment (번들 조정).
Rotation averaging is timed twice: on the ground-truth relative-rotation graph and on the graph from verified pairs.
To also save the table as a file, `cargo bench --bench pipeline -- --quick --json <file>`.

### Library: two-view relative pose and triangulation

```rust
use skylens_core::two_view::{essential_from_fundamental, refine_relative_pose, triangulate};
// For a minimal solver, essential_5pt(&n1[..5], &n2[..5]) returns essential-matrix candidates (up to 10).

// f, inliers: the RANSAC result above. k: camera intrinsics (Intrinsics, including the distortion coefficients `dist`).
// k.to_normalized also removes the distortion. If you need to know whether undistortion converged, use k.unproject(p) (None on failure).
let n1: Vec<_> = x1.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let n2: Vec<_> = x2.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let e = essential_from_fundamental(&f, &k, &k);
if let Some(rp) = refine_relative_pose(&e, &n1, &n2, 50).filter(|rp| rp.translation_observable) {
    // camera 1 = [I|0], camera 2 = [R|t] (t has unit length; scale is unknown)
    let x = triangulate(&rp.rotation, &rp.translation, &n1[0], &n2[0]); // 3D point in camera 1 coordinates
}
// refine_relative_pose runs Sampson-cost LM from several decompositions of the linear E and picks the
// lowest-cost pose by cheirality (refining only the linear pose can get stuck in a rotation–translation
// ambiguity local minimum). Use recover_pose(&e, &n1, &n2) for the linear pose only, or refine_pose to
// supply your own start. Both return None on length mismatch or NaN input.
// For pure rotation or a very short baseline (translation unobservable), translation_observable = false
// and only rotation is reliable (translation is zero). That rotation can still feed rotation averaging.
```

With known intrinsics, a pair can be verified directly with 5-point essential-matrix RANSAC (works for near-planar scenes such as the ground).

```rust
use skylens_core::matching::RansacConfig;
use skylens_core::two_view::ransac_essential_candidates;
// n1, n2: normalized coordinates of all matches (k.to_normalized). cfg.threshold_px is in pixels, converted with k.fx.
let cands = ransac_essential_candidates(&n1, &n2, k.fx, &RansacConfig::default());
// A planar scene is two-fold ambiguous from two views, so up to 4 candidates (E, inlier mask) are returned.
// When most inliers are explained by one plane (homography), matches off that plane are dropped from the inliers.
// Pick the right candidate with other information such as a third view. ransac_essential(...) returns the first candidate.
```

### Library: rotation averaging

```rust
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};

// edge (i, j): rotation = R_j R_iᵀ (same convention as recover_pose), weight = e.g. inlier count
let edges = vec![RelativeRotation { i: 0, j: 1, rotation: pose01.rotation, weight: 120.0 } /* ... */];
if let Some(res) = average_rotations(num_views, &edges, &AveragingConfig::default()) {
    // res.rotations[v]: world→camera rotation of view v (reference view = identity), None if not connected
    // res.inliers[k], res.residuals_rad[k]: per-edge inlier flag and residual angle (rad)
    // the outlier threshold is max(5°, 6 × noise σ̂ estimated from residuals); the value used is res.outlier_threshold_rad
}
// None: zero views, out-of-range index, or no usable edge. NaN rotations, non-positive weights and self edges are ignored.
```

### Library: bundle adjustment

```rust
use skylens_core::ba::{bundle_adjust, BaOptions, BaProblem, Loss, Observation};

// groups[g]: intrinsics group (DistortedIntrinsics: fx fy cx cy + distortion k1 k2 p1 p2), one per camera folder
// poses[c]: world→camera pose of camera c (x_c = R x + t), camera_group[c]: group index of camera c
// points[p]: initial 3D points. Observed pixels are camera pixel coordinates (pixel centre = i + 0.5)
let observations = vec![Observation { camera: 0, point: 0, pixel: nalgebra::Vector2::new(512.5, 300.5) } /* ... */];
let mut problem = BaProblem { groups, poses, camera_group, points, observations };
let opts = BaOptions {
    loss: Loss::Huber(2.0), // Squared, Huber(δ), Cauchy(δ) — δ in pixels
    // free-parameter mask, order INTRINSIC_NAMES = [fx, fy, cx, cy, k1, k2, p1, p2]
    default_free_intrinsics: [true, true, true, true, true, true, false, false],
    fixed_cameras: vec![0], // cameras whose pose is held fixed (gauge)
    ..BaOptions::default()  // at most 50 iterations, at most 100 000 tracks (points)
};
let report = bundle_adjust(&mut problem, &opts); // updates poses, points and intrinsics of problem in place
println!("RMS {:.3} → {:.3} px, {} iterations", report.initial_rms, report.final_rms, report.iterations);
```

This is sparse Levenberg–Marquardt that eliminates the point blocks with the Schur complement and solves only the reduced camera system.
For a per-group mask set `free_intrinsics[g]` (when empty, `default_free_intrinsics` applies to every group).
Only `max_tracks` points are used, those with the most observations first (ties by index); unselected points are left unchanged.
With `max_iterations = 0` the cost is only evaluated (`report.refined = false`); with 1 or more the problem is refined.
`report` also holds the numbers of cameras, tracks and observations used, the initial and final cost, and whether it converged.

### Library: dense preparation, fusion and progressive stream

These stages are not wired into the command line yet.

- `skylens_core::align`: the similarity transform `Similarity { s, r, t }` (`identity`, `apply_point`, `apply_normal` (rotation only), `inverse`, `compose`, `to_matrix4`).
  `umeyama(&src, &dst)` is the least-squares similarity from point correspondences, and `robust_similarity(&src, &dst, iters, floor_m)` trims iteratively
  (threshold max(3 × median residual, `floor_m`)) and returns `(transform, inlier flags, median residual)`.
  `align_to_enu(&centers, &enu, max_residual_m)` aligns the camera centres of the refined poses to east-north-up coordinates once, drops correspondences whose residual exceeds the limit and solves again.
  `gps_align(&centers, &gps, &origin)` converts latitude/longitude (`geo::Geodetic`) to east-north-up about `origin` and calls `align_to_enu` with the limit `GPS_MAX_RESIDUAL_M` (3 m).
  The result `GpsAlignment` holds `sim`, `inliers`, `residuals` and `median_residual`. The limit is fixed at 3 m, and if fewer than 3 inliers remain the first estimate is kept.
  It returns `None` for a length mismatch, fewer than 3 correspondences, NaN/infinite values, or a degenerate layout where the camera centres coincide or lie on a line (the same conditions as `umeyama`).
  To align into the SPEC §2 output frame, pass the first GPS fix as `origin` (for the synthetic scene, `Scene::first_gps_origin()`). `truth/origin.txt` is only the origin of the ground-truth coordinates, not the output origin.
- `skylens_core::undistort`: `undistort_to_long_side(&img, &k, long_side)` turns a distorted image into a pinhole image whose long side is `long_side` pixels, plus the new `Intrinsics` (pixels outside the source are 0, bilinear interpolation).
- `skylens_core::view_selection`: `select_neighbors(&views, &points, k)` picks up to k neighbours per image among the images that share sparse points (default `DEFAULT_NEIGHBORS` = 8), and `depth_range(&view, &points)` gives the depth search range.
- `skylens_core::fusion`: `fuse(&views, &depth_maps, FusionConfig)` merges per-image depth maps with a round-trip reprojection check into a `PointCloud` (at least `min_views` agreeing images).
- `skylens_core::stream`: `split_regions(positions, span, overlap)` splits regions, `align_region` aligns a preview region to the refined clouds with a similarity transform, `build_snapshots` builds the per-step clouds ("refined earlier regions + preview of the newest region"), and `write_outputs(output dir, ...)` writes:

```
<output>/
  preview/preview_00_pos0-14.ply ...   # per-region preview (pos<start>-<end, exclusive>)
  refined/refined_00_pos0-14.ply ...   # per-region refined cloud
  snapshots/step_01_1regions.ply ...   # per-step clouds for display
  snapshots/step_final_all_refined.ply
  snapshots/manifest.json              # point counts per step, alignment figures
```

### PLY format

`ply-info` reads, and the library (`skylens_core::ply`) writes, binary little-endian PLY.
Written files have `x y z nx ny nz` (float32) + `red green blue` (uint8) per point. When reading, `vertex` must be the first element;
properties are looked up by name, so their order may differ, and missing normals or colours are filled with 0.
The header must contain the line `format binary_little_endian 1.0` exactly once; a header line may not exceed 4096 bytes
and the whole header may not exceed 64 KiB (reading stops there with an error).

---

## License / 라이선스

MIT
