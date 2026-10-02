//! 왜곡 보정: 왜곡 있는 사진을 긴 변 `long_side` 화소의 핀홀(왜곡 없음) 사진으로 바꾼다.
//!
//! 화소 좌표 규약은 저장소 전체와 같다([`crate::camera`]): 화소 번호 i 의 중심은 연속 좌표 i + 0.5,
//! 영상은 연속 좌표 [0, W] × [0, H] 를 덮는다. 축척 s 로 줄이면 연속 좌표는 그대로 곱해진다.
//! - 새 내부 파라미터: fx' = fx·s, fy' = fy·s, cx' = cx·s, cy' = cy·s (연속 좌표 주점).
//!   주점을 화소 번호 좌표 c_idx = c − 0.5 로 쓰는 쪽에서는 같은 식이 c_idx' = (c_idx + 0.5)·s − 0.5 가 된다.
//! - 출력 화소 번호 u' 는 연속 좌표 u' + 0.5 에서의 표본이다. 그 원본 연속 좌표
//!   p = fx·d.x + cx (d 는 왜곡 적용 정규 좌표)를 화소 번호 좌표 p − 0.5 로 바꿔 보간한다.
//!
//! 축소비 r = 1/s 가 1 보다 크면 원본을 가우스 σ = [`ANTIALIAS_K`]·√(r² − 1) 로 먼저 흐려
//! 출력 나이퀴스트를 넘는 무늬가 다른 주파수로 접혀 남지 않게 한다(무아레 방지).
//!
//! 원본 영역 [0, W] × [0, H] 밖으로 나가는 출력 화소는 값 0, 마스크 `false` 다.
//! 가장자리 반 화소 띠(화소 번호 좌표 [−0.5, 0) 와 (W − 1, W − 0.5])는 끝 화소 값으로 채우고 유효로 둔다.
use image::{ImageBuffer, Pixel};

use crate::camera::Intrinsics;
use crate::distortion::DistortedIntrinsics;
use crate::math::Vector2;

/// 저역 통과 세기: 원본 가우스 σ = K·√(r² − 1) (r = 축소비).
pub const ANTIALIAS_K: f64 = 0.6;

/// 왜곡 보정 입력 오류.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UndistortError {
    /// 원본 너비·높이 또는 목표 긴 변이 0.
    EmptyImage,
}

impl std::fmt::Display for UndistortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyImage => write!(f, "empty image or zero target size"),
        }
    }
}

impl std::error::Error for UndistortError {}

/// 출력 화소 → 원본 연속 좌표 역사상 표. 카메라(내부 파라미터·원본 크기)마다 한 번 만든다.
#[derive(Clone, Debug)]
pub struct UndistortMap {
    /// 출력 핀홀 내부 파라미터.
    pub pinhole: Intrinsics,
    /// 원본 너비·높이.
    pub source_size: (u32, u32),
    /// 행 우선, 출력 화소 (u', v') 중심 (u' + 0.5, v' + 0.5) 의 원본 연속 좌표. 길이 width·height.
    pub table: Vec<[f64; 2]>,
    /// 행 우선 유효 화소 마스크: 원본 영역 안이면 true.
    pub valid: Vec<bool>,
    /// 원본에 거는 저역 통과 가우스 σ(원본 화소). 0 이면 걸지 않는다.
    pub sigma: f64,
}

/// 보정 결과: 사진, 유효 마스크(행 우선), 핀홀 내부 파라미터.
#[derive(Clone, Debug)]
pub struct Undistorted<P: Pixel<Subpixel = u8>> {
    pub image: ImageBuffer<P, Vec<u8>>,
    pub valid: Vec<bool>,
    pub pinhole: Intrinsics,
}

impl<P: Pixel<Subpixel = u8>> Undistorted<P> {
    /// 유효하지 않은 화소 수.
    pub fn invalid_count(&self) -> usize {
        self.valid.iter().filter(|v| !**v).count()
    }
}

/// 원본 크기와 내부 파라미터로 출력 핀홀 내부 파라미터를 만든다. 크기 0 이면 오류.
pub fn pinhole_for_long_side(
    width: u32,
    height: u32,
    k: &DistortedIntrinsics,
    long_side: u32,
) -> Result<Intrinsics, UndistortError> {
    if width == 0 || height == 0 || long_side == 0 {
        return Err(UndistortError::EmptyImage);
    }
    let s = long_side as f64 / width.max(height) as f64;
    let w = ((width as f64 * s).round() as u32).max(1);
    let h = ((height as f64 * s).round() as u32).max(1);
    let sx = w as f64 / width as f64;
    let sy = h as f64 / height as f64;
    Ok(Intrinsics {
        fx: k.fx * sx,
        fy: k.fy * sy,
        cx: k.cx * sx,
        cy: k.cy * sy,
        width: w,
        height: h,
        dist: crate::distortion::Distortion::default(),
    })
}

/// 축소비 r 에 대한 저역 통과 σ(원본 화소).
pub fn antialias_sigma(r: f64) -> f64 {
    if r > 1.0 {
        ANTIALIAS_K * (r * r - 1.0).sqrt()
    } else {
        0.0
    }
}

impl UndistortMap {
    pub fn new(
        width: u32,
        height: u32,
        k: &DistortedIntrinsics,
        long_side: u32,
    ) -> Result<Self, UndistortError> {
        let pinhole = pinhole_for_long_side(width, height, k, long_side)?;
        let n = (pinhole.width * pinhole.height) as usize;
        let mut table = Vec::with_capacity(n);
        let mut valid = Vec::with_capacity(n);
        let (fw, fh) = (width as f64, height as f64);
        for v in 0..pinhole.height {
            for u in 0..pinhole.width {
                let out = Vector2::new(u as f64 + 0.5, v as f64 + 0.5);
                let p = Self::source_of(&pinhole, k, &out);
                table.push([p.x, p.y]);
                valid.push(p.x >= 0.0 && p.y >= 0.0 && p.x <= fw && p.y <= fh);
            }
        }
        let r = (width as f64 / pinhole.width as f64).max(height as f64 / pinhole.height as f64);
        Ok(Self {
            pinhole,
            source_size: (width, height),
            table,
            valid,
            sigma: antialias_sigma(r),
        })
    }

    /// 출력 연속 좌표 하나의 원본 연속 좌표(표 없이 바로 계산).
    pub fn source_of(
        pinhole: &Intrinsics,
        k: &DistortedIntrinsics,
        out: &Vector2<f64>,
    ) -> Vector2<f64> {
        let d = k.dist.distort(&pinhole.to_normalized(out));
        Vector2::new(k.fx * d.x + k.cx, k.fy * d.y + k.cy)
    }

    /// 표를 쌍선형 보간해 출력 연속 좌표의 원본 연속 좌표를 얻는다.
    /// 출력 화소 중심이 이루는 격자 [0.5, w − 0.5] × [0.5, h − 0.5] 밖이면 None.
    pub fn lookup(&self, out: &Vector2<f64>) -> Option<Vector2<f64>> {
        let w = self.pinhole.width as usize;
        let h = self.pinhole.height as usize;
        let s = bilinear(w, h, out.x - 0.5, out.y - 0.5, |i, c| self.table[i][c], 2)?;
        Some(Vector2::new(s[0], s[1]))
    }
}

/// 격자 값 `get(색인, 채널)` 의 쌍선형 보간(화소 번호 좌표). 격자 밖이면 None.
fn bilinear(
    w: usize,
    h: usize,
    x: f64,
    y: f64,
    get: impl Fn(usize, usize) -> f64,
    channels: usize,
) -> Option<[f64; 4]> {
    if w == 0 || h == 0 || !(x >= 0.0 && y >= 0.0 && x <= (w - 1) as f64 && y <= (h - 1) as f64) {
        return None;
    }
    let x0 = (x.floor() as usize).min(w.saturating_sub(2));
    let y0 = (y.floor() as usize).min(h.saturating_sub(2));
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let (ax, ay) = (x - x0 as f64, y - y0 as f64);
    let mut out = [0.0; 4];
    for (c, o) in out.iter_mut().enumerate().take(channels) {
        let top = get(y0 * w + x0, c) * (1.0 - ax) + get(y0 * w + x1, c) * ax;
        let bot = get(y1 * w + x0, c) * (1.0 - ax) + get(y1 * w + x1, c) * ax;
        *o = top * (1.0 - ay) + bot * ay;
    }
    Some(out)
}

/// 가로·세로 분리 가우스 흐림(가장자리는 끝 화소 반복). 채널 교차 배열.
fn gaussian_blur(src: &[f32], w: usize, h: usize, nc: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let rad = (3.0 * sigma).ceil() as i64;
    let mut kern: Vec<f32> = (-rad..=rad)
        .map(|d| (-0.5 * (d as f64 / sigma).powi(2)).exp() as f32)
        .collect();
    let sum: f32 = kern.iter().sum();
    kern.iter_mut().for_each(|x| *x /= sum);
    let mut tmp = vec![0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc {
                let mut acc = 0f32;
                for (t, kv) in kern.iter().enumerate() {
                    let xx = (x as i64 + t as i64 - rad).clamp(0, w as i64 - 1) as usize;
                    acc += kv * src[(y * w + xx) * nc + c];
                }
                tmp[(y * w + x) * nc + c] = acc;
            }
        }
    }
    let mut out = vec![0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc {
                let mut acc = 0f32;
                for (t, kv) in kern.iter().enumerate() {
                    let yy = (y as i64 + t as i64 - rad).clamp(0, h as i64 - 1) as usize;
                    acc += kv * tmp[(yy * w + x) * nc + c];
                }
                out[(y * w + x) * nc + c] = acc;
            }
        }
    }
    out
}

/// 실수 채널 교차 배열 원본(`source_size`, `nc` 채널)을 역사상 표로 다시 표본화한다.
/// 저역 통과를 먼저 걸고, 무효 화소는 0 이다. 출력 길이 = 출력 화소 수 × nc.
pub fn remap_f32(src: &[f32], nc: usize, map: &UndistortMap) -> Vec<f32> {
    let (sw, sh) = (map.source_size.0 as usize, map.source_size.1 as usize);
    assert!(nc <= 4, "채널 수 {nc} > 4");
    assert_eq!(src.len(), sw * sh * nc, "원본 크기");
    let blurred = gaussian_blur(src, sw, sh, nc, map.sigma);
    let mut buf = vec![0f32; map.table.len() * nc];
    for (i, (p, ok)) in map.table.iter().zip(&map.valid).enumerate() {
        if !ok {
            continue;
        }
        // 연속 좌표 → 화소 번호 좌표, 가장자리 반 화소 띠는 끝 화소로 붙인다.
        let x = (p[0] - 0.5).clamp(0.0, (sw - 1) as f64);
        let y = (p[1] - 0.5).clamp(0.0, (sh - 1) as f64);
        if let Some(v) = bilinear(sw, sh, x, y, |q, c| blurred[q * nc + c] as f64, nc) {
            for c in 0..nc {
                buf[i * nc + c] = v[c] as f32;
            }
        }
    }
    buf
}

/// 역사상 표로 8비트 원본을 다시 표본화한다.
pub fn remap<P>(img: &ImageBuffer<P, Vec<u8>>, map: &UndistortMap) -> ImageBuffer<P, Vec<u8>>
where
    P: Pixel<Subpixel = u8>,
{
    assert_eq!(img.dimensions(), map.source_size, "원본 크기");
    let nc = P::CHANNEL_COUNT as usize;
    let src: Vec<f32> = img.as_raw().iter().map(|&x| x as f32).collect();
    let out = remap_f32(&src, nc, map);
    let buf = out
        .iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    ImageBuffer::from_raw(map.pinhole.width, map.pinhole.height, buf).expect("버퍼 크기")
}

/// 왜곡 있는 사진을 긴 변 `long_side` 의 핀홀 사진으로 바꾸고 유효 마스크·새 내부 파라미터를 돌려준다.
/// 화소 형은 8비트 채널(최대 4채널)이면 무엇이든 된다. 크기 0 입력은 오류.
pub fn undistort_to_long_side<P>(
    img: &ImageBuffer<P, Vec<u8>>,
    k: &DistortedIntrinsics,
    long_side: u32,
) -> Result<Undistorted<P>, UndistortError>
where
    P: Pixel<Subpixel = u8>,
{
    let map = UndistortMap::new(img.width(), img.height(), k, long_side)?;
    Ok(Undistorted {
        image: remap(img, &map),
        valid: map.valid.clone(),
        pinhole: map.pinhole,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Camera, Pose};
    use crate::distortion::Distortion;
    use crate::math::{Point3, Rotation3, Vector3};
    use image::{GrayImage, Luma};

    fn intr() -> DistortedIntrinsics {
        DistortedIntrinsics {
            fx: 1300.0,
            fy: 1295.0,
            cx: 803.0,
            cy: 596.0,
            dist: Distortion {
                k1: -0.12,
                k2: 0.03,
                p1: 0.001,
                p2: -0.0015,
            },
        }
    }

    fn plain(fx: f64, cx: f64, cy: f64, k1: f64) -> DistortedIntrinsics {
        DistortedIntrinsics {
            fx,
            fy: fx,
            cx,
            cy,
            dist: Distortion {
                k1,
                ..Distortion::default()
            },
        }
    }

    #[test]
    fn output_size_and_intrinsics() {
        let p = pinhole_for_long_side(1600, 1200, &intr(), 960).unwrap();
        assert_eq!((p.width, p.height), (960, 720));
        assert!((p.fx - 1300.0 * 0.6).abs() < 1e-12);
        assert!((p.cx - 803.0 * 0.6).abs() < 1e-12);
        let q = pinhole_for_long_side(1200, 1600, &intr(), 960).unwrap();
        assert_eq!((q.width, q.height), (720, 960));
        // 기하 중심 주점은 기하 중심으로 간다(F-060: 1920×1080 → 480, 270).
        let c = pinhole_for_long_side(1920, 1080, &plain(1500.0, 960.0, 540.0, 0.0), 960).unwrap();
        assert_eq!((c.cx, c.cy), (480.0, 270.0));
    }

    #[test]
    fn zero_size_is_rejected() {
        let k = intr();
        assert_eq!(
            UndistortMap::new(0, 0, &k, 960).unwrap_err(),
            UndistortError::EmptyImage
        );
        assert!(UndistortMap::new(16, 0, &k, 960).is_err());
        assert!(UndistortMap::new(16, 16, &k, 0).is_err());
        let img = GrayImage::new(0, 0);
        assert!(undistort_to_long_side(&img, &k, 960).is_err());
    }

    /// 표 보간 좌표 → 원본 → 정규 좌표 역변환 → 출력 좌표 왕복.
    #[test]
    fn map_roundtrip() {
        let k = intr();
        let map = UndistortMap::new(1600, 1200, &k, 960).unwrap();
        let mut worst: f64 = 0.0;
        for i in 0..400 {
            let out = Vector2::new(
                0.5 + (i as f64 * 37.13) % 959.0,
                0.5 + (i as f64 * 23.71) % 719.0,
            );
            let src = map.lookup(&out).unwrap();
            let n = k.unproject(&src).unwrap();
            let back = map.pinhole.to_pixel(&n);
            worst = worst.max((back - out).norm());
        }
        eprintln!("map_roundtrip_worst_px {worst:.3e}");
        // 표 보간 오차: 화소 간 왜곡 2차 변화량 수준(1e-3 px 미만).
        assert!(worst < 1e-3, "{worst}");
    }

    /// 저장소 규약(화소 중심 i + 0.5)으로 정규 좌표계 가우스 반점을 그린 왜곡 사진을 보정하고,
    /// 반점 무게중심(화소 번호 + 0.5)이 돌려받은 핀홀의 정답 투영과 맞는지 본다. 최대 차를 돌려준다.
    fn blob_error(w: u32, h: u32, k: &DistortedIntrinsics, sigma_out: f64) -> f64 {
        let pose = Pose::from_center(
            Rotation3::from_euler_angles(0.05, -0.03, 0.2),
            &Point3::new(0.0, 0.0, -30.0),
        );
        let world: Vec<Point3<f64>> = [
            (0.0, 0.0),
            (-12.0, -6.0),
            (11.0, 5.5),
            (-13.0, 6.0),
            (12.5, -5.0),
            (5.0, -2.0),
        ]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect();
        let centers: Vec<Vector2<f64>> = world
            .iter()
            .map(|x| {
                let c: Vector3<f64> = pose.transform(x);
                Vector2::new(c.x / c.z, c.y / c.z)
            })
            .collect();
        let s = 960.0 / w.max(h) as f64;
        let sigma_n = sigma_out / (k.fx * s);
        let img = GrayImage::from_fn(w, h, |u, v| {
            let n = k
                .unproject(&Vector2::new(u as f64 + 0.5, v as f64 + 0.5))
                .unwrap();
            let val: f64 = centers
                .iter()
                .map(|c| (-(n - c).norm_squared() / (2.0 * sigma_n * sigma_n)).exp())
                .sum();
            Luma([(250.0 * val).round().min(255.0) as u8])
        });
        let res = undistort_to_long_side(&img, k, 960).unwrap();
        let cam = Camera {
            intrinsics: res.pinhole,
            pose,
        };
        let mut worst: f64 = 0.0;
        for x in &world {
            let p = cam.project(x).unwrap();
            let r = (4.0 * sigma_out).ceil() as i64;
            let (mut sw, mut sx, mut sy) = (0.0, 0.0, 0.0);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (u, v) = (p.x.floor() as i64 + dx, p.y.floor() as i64 + dy);
                    let val = res.image.get_pixel(u as u32, v as u32)[0] as f64;
                    sw += val;
                    sx += val * (u as f64 + 0.5);
                    sy += val * (v as f64 + 0.5);
                }
            }
            worst = worst.max((Vector2::new(sx / sw, sy / sw) - p).norm());
        }
        worst
    }

    /// F-060: 1920×1080(fx 1500, 주점 기하 중심) 왜곡 0·k1 −0.12, 2048×1152(fx 1609) k1 −0.12,
    /// 그리고 접선 왜곡까지 있는 1600×1200 을 960 으로 보정.
    #[test]
    fn blobs_land_on_pinhole_projection() {
        let cases = [
            ("1920 k1=0", 1920, 1080, plain(1500.0, 960.0, 540.0, 0.0)),
            (
                "1920 k1=-0.12",
                1920,
                1080,
                plain(1500.0, 960.0, 540.0, -0.12),
            ),
            (
                "2048 k1=-0.12",
                2048,
                1152,
                plain(1609.0, 1024.0, 576.0, -0.12),
            ),
            ("1600 full", 1600, 1200, intr()),
        ];
        for (name, w, h, k) in cases {
            let e = blob_error(w, h, &k, 3.0);
            eprintln!("blob_centroid_worst_px {name} {e:.4}");
            assert!(e < 0.01, "{name} {e}");
        }
    }

    /// F-053: 원본 화소 번호를 값으로 쓰는 선형 램프(실수). 왜곡 0 이면 출력 화소 u' 의 값은
    /// (u' + 0.5)/s − 0.5 여야 한다.
    #[test]
    fn linear_ramp_matches_center_convention() {
        for (w, h, fx) in [(1920u32, 1080u32, 1500.0), (2048, 1152, 1609.0)] {
            let k = plain(fx, w as f64 / 2.0, h as f64 / 2.0, 0.0);
            let map = UndistortMap::new(w, h, &k, 960).unwrap();
            let s = map.pinhole.width as f64 / w as f64;
            let mut src = Vec::with_capacity((w * h * 2) as usize);
            for v in 0..h {
                for u in 0..w {
                    src.push(u as f32);
                    src.push(v as f32);
                }
            }
            let out = remap_f32(&src, 2, &map);
            let (ow, oh) = (map.pinhole.width as usize, map.pinhole.height as usize);
            let mut worst: f64 = 0.0;
            // 가장자리 끝 화소 반복의 영향이 없는 안쪽(흐림 반경 + 여유).
            let m = 8;
            for v in m..oh - m {
                for u in m..ow - m {
                    let i = v * ow + u;
                    let eu = (u as f64 + 0.5) / s - 0.5;
                    let ev = (v as f64 + 0.5) / s - 0.5;
                    worst = worst
                        .max((out[2 * i] as f64 - eu).abs())
                        .max((out[2 * i + 1] as f64 - ev).abs());
                }
            }
            eprintln!("ramp_worst_src_px {w}x{h} {worst:.4}");
            assert!(worst < 0.05, "{w} {worst}");
        }
    }

    /// F-092: 2048×1152, 주기 3 px·진폭 100 사인 줄무늬, 왜곡 0, 960 으로 축소.
    /// 출력 나이퀴스트(원본 주기 4.27 px)를 넘는 무늬라 출력에 남으면 안 된다.
    #[test]
    fn stripes_above_nyquist_are_removed() {
        let k = plain(1609.0, 1024.0, 576.0, 0.0);
        let img = GrayImage::from_fn(2048, 1152, |u, _| {
            let x = u as f64 + 0.5;
            Luma([(128.0 + 100.0 * (std::f64::consts::TAU * x / 3.0).sin()).round() as u8])
        });
        let res = undistort_to_long_side(&img, &k, 960).unwrap();
        let vals: Vec<f64> = res.image.pixels().map(|p| p[0] as f64).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let sd = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        eprintln!("stripe_output_std {sd:.3} mean {mean:.2}");
        assert!(sd <= 5.0, "{sd}");
    }

    /// F-111: 2048×1152, k1 +0.08(원본 밖 모서리), 균일 밝기 200.
    /// 무효 화소 수 = 출력 0 화소 수, 유효 화소는 모두 200.
    #[test]
    fn valid_mask_marks_outside() {
        let k = plain(1609.0, 1024.0, 576.0, 0.08);
        let img = GrayImage::from_pixel(2048, 1152, Luma([200]));
        let res = undistort_to_long_side(&img, &k, 960).unwrap();
        let zeros = res.image.pixels().filter(|p| p[0] == 0).count();
        let bad_valid = res
            .image
            .pixels()
            .zip(&res.valid)
            .filter(|(p, ok)| **ok && p[0] != 200)
            .count();
        eprintln!(
            "invalid {} zeros {zeros} of {}",
            res.invalid_count(),
            res.valid.len()
        );
        assert!(res.invalid_count() > 0);
        assert_eq!(res.invalid_count(), zeros);
        assert_eq!(bad_valid, 0);
    }

    #[test]
    fn zero_distortion_rgb_keeps_content() {
        let mut k = intr();
        k.dist = Distortion::default();
        let img = image::RgbImage::from_fn(1600, 1200, |u, v| {
            image::Rgb([(u / 8) as u8, (v / 5) as u8, 128])
        });
        let res = undistort_to_long_side(&img, &k, 960).unwrap();
        // 왜곡이 없으면 출력 (u', v') 는 원본 화소 번호 ((u'+0.5)/0.6 − 0.5, ...) 근방 값.
        let px = res.image.get_pixel(300, 200);
        let su = (300.5 / 0.6 - 0.5) / 8.0;
        assert!((px[0] as f64 - su).abs() <= 1.0, "{:?} {su}", px);
        assert_eq!(px[2], 128);
        assert_eq!((res.pinhole.width, res.pinhole.height), (960, 720));
        assert!(res.valid.iter().all(|v| *v));
    }
}
