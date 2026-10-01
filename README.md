# skylens-stream-rs

[한국어](#한국어) · [English](#english)

---

## 한국어

드론 편대(카메라 3대) 영상에서 **점진적으로 나아지는 3D 점군**을 만드는 Rust 도구.

새 구역이 들어올 때마다 빠른 초벌 점군을 먼저 보여 주고, 정밀 계산이 끝난 구역은 정밀본으로 바꿔 끼운다.
화면에는 단계마다 "이전 구역은 정밀본 + 최신 구역은 초벌" 점군이 나간다.

> 개발 중이다. 아래 명령 중 아직 없는 것은 표시해 둔다.

### 설치

```bash
cargo build --release
# 실행 파일: target/release/skylens-stream
```

Rust stable(1.80 이상)이 필요하다.

### 입력

```
<데이터>/
  images/camF/camF_0000.jpg ...   # 앞 카메라
  images/camR/camR_0000.jpg ...   # 오른쪽
  images/camL/camL_0000.jpg ...   # 왼쪽
  gps.txt                         # 한 줄에: 이미지이름 위도 경도 고도
```

### 사용법

```bash
# 점군 파일 정보 보기
skylens-stream ply-info <파일.ply>

# 시험용 합성 장면 만들기 (정답 카메라·점·GPS 포함, 기본 해상도는 작게)
skylens-stream synth <출력 폴더> [폭 높이]

# 점진적 점군 생성 (개발 중)
skylens-stream run <데이터> -o <출력> [--stride 3] [--span 12] [--overlap 2]

# 결과 검증 (개발 중) — 실패하면 종료 코드 1
skylens-stream verify <출력>
```

### 라이브러리: 특징점

```rust
use skylens_core::features::{detect_and_describe, DetectorConfig, GrayImage};

let img = GrayImage::from_rgb(width, height, &rgb_bytes);
let feats = detect_and_describe(&img, &DetectorConfig::default());
for f in &feats {
    // f.kp: x, y(화소), sigma(스케일), angle(라디안), response
    // f.desc: 단위 길이 128차원 기술자
}
```

### 라이브러리: 매칭과 기하 검증

```rust
use skylens_core::matching::{candidate_pairs, ratio_match, ransac_fundamental, RansacConfig};
use skylens_core::matching::{PAIR_CROSS, PAIR_POW2_MAX, PAIR_TEMPORAL};

// 매칭할 영상 짝: views[k] = (카메라 번호, 촬영 위치 번호).
// 같은 카메라는 위치 차 1..=5 와 2의 거듭제곱(8, 16, 32, …), 다른 카메라는 위치 차 0..=4.
let image_pairs = candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);

// 비율 0.8, 양방향 확인
let pairs = ratio_match(&feats_a, &feats_b, 0.8, true);
let x1: Vec<_> = pairs.iter().map(|&(i, _)| nalgebra::Vector2::new(feats_a[i].kp.x as f64, feats_a[i].kp.y as f64)).collect();
let x2: Vec<_> = pairs.iter().map(|&(_, j)| nalgebra::Vector2::new(feats_b[j].kp.x as f64, feats_b[j].kp.y as f64)).collect();
if let Some((f, inliers)) = ransac_fundamental(&x1, &x2, &RansacConfig::default()) {
    // f: 기본 행렬(x2ᵀ F x1 = 0), inliers[k]: pairs[k] 가 기하적으로 맞는지
}
// None: 점 8개 미만, 길이 불일치, NaN·무한대 좌표, 동일선상 등 퇴화 배치,
// 또는 정상 비율이 min_inlier_ratio(기본 0.2) 미만. 문턱 threshold_px 는 px 단위,
// sampson_error 는 제곱 거리(px²)를 돌려준다.
```

### 라이브러리: 두 시점 상대 자세와 삼각측량

```rust
use skylens_core::two_view::{essential_from_fundamental, recover_pose, refine_pose, triangulate};
// 최소 해법이 필요하면 essential_5pt(&n1[..5], &n2[..5]) 가 본질 행렬 후보(최대 10개)를 준다.

// f, inliers: 위 RANSAC 결과. k: 카메라 내부 파라미터(Intrinsics).
let n1: Vec<_> = x1.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let n2: Vec<_> = x2.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let e = essential_from_fundamental(&f, &k, &k);
if let Some(rp) = recover_pose(&e, &n1, &n2).filter(|rp| rp.translation_observable) {
    // 카메라 1 = [I|0], 카메라 2 = [R|t] (t 는 단위 길이, 스케일 미정)
    let (r, t) = refine_pose(&rp.rotation, &rp.translation, &n1, &n2, 50);
    let x = triangulate(&r, &t, &n1[0], &n2[0]); // 카메라 1 좌표계의 3D 점
}
// recover_pose 는 길이 불일치, NaN 입력에서 None.
// 순수 회전·아주 짧은 기선(이동 방향을 관측할 수 없음)이면 translation_observable = false 이고
// rotation 만 믿을 수 있다(translation 은 0). 회전 평균에는 이 회전도 쓸 수 있다.
```

### 라이브러리: 회전 평균

```rust
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};

// 간선 (i, j): rotation = R_j R_iᵀ (recover_pose 의 회전과 같은 규약), weight = 정상 대응 수 등
let edges = vec![RelativeRotation { i: 0, j: 1, rotation: pose01.rotation, weight: 120.0 } /* ... */];
if let Some(res) = average_rotations(num_views, &edges, &AveragingConfig::default()) {
    // res.rotations[v]: 세계→카메라 v 회전(기준 정점 = 단위 회전). 기준과 이어지지 않으면 None
    // res.inliers[k], res.residuals_rad[k]: 간선별 정상 표시와 잔차 각(rad), 기본 이상치 문턱 5°
}
// None: 정점 0개, 범위 밖 번호, 쓸 수 있는 간선 없음. NaN 회전·0 이하 가중치·자기 간선은 무시.
```

### 출력

```
<출력>/
  preview/preview_00_pos0-14.ply ...   # 구역별 초벌
  refined/refined_00_pos0-14.ply ...   # 구역별 정밀본
  snapshots/step_01_1regions.ply ...   # 단계별 화면용 점군
  snapshots/step_final_all_refined.ply
  snapshots/manifest.json              # 단계별 점 수, 정렬 수치
```

PLY 는 이진 little-endian, 점마다 `x y z nx ny nz`(float32) + `red green blue`(uint8).
좌표는 첫 GPS 를 원점으로 하는 동-북-위(미터)다. CloudCompare 등에서 `step_01` 부터 순서대로 열면 공간이 자라며 또렷해지는 과정을 볼 수 있다.

---

## English

A Rust tool that builds a **progressively refined 3D point cloud** from drone-formation video (three cameras per drone).

Each time a new region arrives, a fast preview cloud is shown first; once the accurate solve for a region finishes, its preview is swapped for the refined cloud.
At every step the output is "refined clouds for earlier regions + preview for the newest region".

> Work in progress. Commands that do not exist yet are marked below.

### Build

```bash
cargo build --release
# binary: target/release/skylens-stream
```

Requires stable Rust 1.80 or newer.

### Input

```
<data>/
  images/camF/camF_0000.jpg ...   # front camera
  images/camR/camR_0000.jpg ...   # right camera
  images/camL/camL_0000.jpg ...   # left camera
  gps.txt                         # one line per image: name latitude longitude altitude
```

### Usage

```bash
# Show point count and bounds of a PLY file
skylens-stream ply-info <file.ply>

# Generate a synthetic test scene (ground-truth cameras, points and GPS; small default resolution)
skylens-stream synth <output dir> [width height]

# Build the progressive point cloud (in progress)
skylens-stream run <data> -o <output> [--stride 3] [--span 12] [--overlap 2]

# Verify the result (in progress) — exits with code 1 on failure
skylens-stream verify <output>
```

### Library: features

```rust
use skylens_core::features::{detect_and_describe, DetectorConfig, GrayImage};

let img = GrayImage::from_rgb(width, height, &rgb_bytes);
let feats = detect_and_describe(&img, &DetectorConfig::default());
for f in &feats {
    // f.kp: x, y (pixels), sigma (scale), angle (radians), response
    // f.desc: unit-length 128-dim descriptor
}
```

### Library: matching and geometric verification

```rust
use skylens_core::matching::{candidate_pairs, ratio_match, ransac_fundamental, RansacConfig};
use skylens_core::matching::{PAIR_CROSS, PAIR_POW2_MAX, PAIR_TEMPORAL};

// Image pairs to match: views[k] = (camera index, capture position index).
// Same camera: position gap 1..=5 plus powers of two (8, 16, 32, …); different cameras: gap 0..=4.
let image_pairs = candidate_pairs(&views, PAIR_TEMPORAL, PAIR_CROSS, PAIR_POW2_MAX);

// ratio 0.8, mutual check
let pairs = ratio_match(&feats_a, &feats_b, 0.8, true);
let x1: Vec<_> = pairs.iter().map(|&(i, _)| nalgebra::Vector2::new(feats_a[i].kp.x as f64, feats_a[i].kp.y as f64)).collect();
let x2: Vec<_> = pairs.iter().map(|&(_, j)| nalgebra::Vector2::new(feats_b[j].kp.x as f64, feats_b[j].kp.y as f64)).collect();
if let Some((f, inliers)) = ransac_fundamental(&x1, &x2, &RansacConfig::default()) {
    // f: fundamental matrix (x2ᵀ F x1 = 0), inliers[k]: whether pairs[k] is geometrically consistent
}
// None: fewer than 8 points, length mismatch, NaN/infinite coordinates, degenerate layouts such as
// collinear points, or an inlier ratio below min_inlier_ratio (default 0.2). threshold_px is in px;
// sampson_error returns the squared distance (px²).
```

### Library: two-view relative pose and triangulation

```rust
use skylens_core::two_view::{essential_from_fundamental, recover_pose, refine_pose, triangulate};
// For a minimal solver, essential_5pt(&n1[..5], &n2[..5]) returns essential-matrix candidates (up to 10).

// f, inliers: the RANSAC result above. k: camera intrinsics (Intrinsics).
let n1: Vec<_> = x1.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let n2: Vec<_> = x2.iter().zip(&inliers).filter(|(_, &ok)| ok).map(|(p, _)| k.to_normalized(p)).collect();
let e = essential_from_fundamental(&f, &k, &k);
if let Some(rp) = recover_pose(&e, &n1, &n2).filter(|rp| rp.translation_observable) {
    // camera 1 = [I|0], camera 2 = [R|t] (t has unit length; scale is unknown)
    let (r, t) = refine_pose(&rp.rotation, &rp.translation, &n1, &n2, 50);
    let x = triangulate(&r, &t, &n1[0], &n2[0]); // 3D point in camera 1 coordinates
}
// recover_pose returns None on length mismatch or NaN input.
// For pure rotation or a very short baseline (translation unobservable), translation_observable = false
// and only rotation is reliable (translation is zero). That rotation can still feed rotation averaging.
```

### Library: rotation averaging

```rust
use skylens_core::rotation_averaging::{average_rotations, AveragingConfig, RelativeRotation};

// edge (i, j): rotation = R_j R_iᵀ (same convention as recover_pose), weight = e.g. inlier count
let edges = vec![RelativeRotation { i: 0, j: 1, rotation: pose01.rotation, weight: 120.0 } /* ... */];
if let Some(res) = average_rotations(num_views, &edges, &AveragingConfig::default()) {
    // res.rotations[v]: world→camera rotation of view v (reference view = identity), None if not connected
    // res.inliers[k], res.residuals_rad[k]: per-edge inlier flag and residual angle (rad), default outlier threshold 5°
}
// None: zero views, out-of-range index, or no usable edge. NaN rotations, non-positive weights and self edges are ignored.
```

### Output

```
<output>/
  preview/preview_00_pos0-14.ply ...   # per-region preview
  refined/refined_00_pos0-14.ply ...   # per-region refined cloud
  snapshots/step_01_1regions.ply ...   # per-step clouds for display
  snapshots/step_final_all_refined.ply
  snapshots/manifest.json              # point counts per step, alignment figures
```

PLY files are binary little-endian with `x y z nx ny nz` (float32) and `red green blue` (uint8) per point.
Coordinates are local East-North-Up in metres with the first GPS fix as origin. Open `step_01` onward in order in a viewer such as CloudCompare to watch the scene grow and sharpen.

---

## License / 라이선스

MIT
