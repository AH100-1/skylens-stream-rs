# skylens-stream-rs

드론 편대(카메라 3대) 영상에서 **점진적으로 나아지는 3D 점군**을 만드는 Rust 도구.

새 구역이 들어올 때마다 빠른 초벌 점군을 먼저 보여 주고, 정밀 계산이 끝난 구역은 정밀본으로 바꿔 끼운다.
화면에는 단계마다 "이전 구역은 정밀본 + 최신 구역은 초벌" 점군이 나간다.

> 개발 중이다. 아래 명령 중 아직 없는 것은 표시해 둔다.

## 설치

```bash
cargo build --release
# 실행 파일: target/release/skylens-stream
```

Rust stable(1.80 이상)이 필요하다.

## 입력

```
<데이터>/
  images/camF/camF_0000.jpg ...   # 앞 카메라
  images/camR/camR_0000.jpg ...   # 오른쪽
  images/camL/camL_0000.jpg ...   # 왼쪽
  gps.txt                         # 한 줄에: 이미지이름 위도 경도 고도
```

## 사용법

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
use skylens_core::matching::{ratio_match, ransac_fundamental, RansacConfig};

// 비율 0.8, 양방향 확인
let pairs = ratio_match(&feats_a, &feats_b, 0.8, true);
let x1: Vec<_> = pairs.iter().map(|&(i, _)| nalgebra::Vector2::new(feats_a[i].kp.x as f64, feats_a[i].kp.y as f64)).collect();
let x2: Vec<_> = pairs.iter().map(|&(_, j)| nalgebra::Vector2::new(feats_b[j].kp.x as f64, feats_b[j].kp.y as f64)).collect();
if let Some((f, inliers)) = ransac_fundamental(&x1, &x2, &RansacConfig::default()) {
    // f: 기본 행렬(x2ᵀ F x1 = 0), inliers[k]: pairs[k] 가 기하적으로 맞는지
}
```

## 출력

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

## 라이선스

MIT
