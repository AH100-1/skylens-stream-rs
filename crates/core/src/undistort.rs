//! 왜곡 보정: 왜곡 있는 사진을 긴 변 `long_side` 화소의 핀홀(왜곡 없음) 사진으로 바꾼다.
//!
//! 화소 좌표 규약: 화소 (i, j) 의 값은 연속 좌표 (i, j) 에서의 표본이다.
//! 축척 s 로 줄일 때 좌표는 u' = (u + 0.5)·s − 0.5 로 옮긴다.
//!
//! 새 핀홀 내부 파라미터는 원래 fx, fy, cx, cy 를 같은 축척으로 옮긴 것이다(왜곡만 제거).
//! 출력 화소마다 역사상 표(출력 → 원본 좌표)를 한 번 만들고, 원본을 쌍선형 보간해 채운다.
//! 원본 밖으로 나가는 출력 화소는 0 이다.
use image::{ImageBuffer, Pixel};

use crate::camera::Intrinsics;
use crate::distortion::DistortedIntrinsics;
use crate::math::Vector2;

/// 출력 화소 → 원본 화소 좌표 역사상 표.
#[derive(Clone, Debug)]
pub struct UndistortMap {
    /// 출력 핀홀 내부 파라미터.
    pub pinhole: Intrinsics,
    /// 행 우선 `(x, y)` 원본 좌표, 길이 width·height.
    pub table: Vec<[f64; 2]>,
}

/// 원본 크기와 내부 파라미터로 출력 핀홀 내부 파라미터를 만든다.
pub fn pinhole_for_long_side(
    width: u32,
    height: u32,
    k: &DistortedIntrinsics,
    long_side: u32,
) -> Intrinsics {
    let s = long_side as f64 / width.max(height) as f64;
    let w = ((width as f64 * s).round() as u32).max(1);
    let h = ((height as f64 * s).round() as u32).max(1);
    let sx = w as f64 / width as f64;
    let sy = h as f64 / height as f64;
    Intrinsics {
        fx: k.fx * sx,
        fy: k.fy * sy,
        cx: (k.cx + 0.5) * sx - 0.5,
        cy: (k.cy + 0.5) * sy - 0.5,
        width: w,
        height: h,
        dist: crate::distortion::Distortion::default(),
    }
}

impl UndistortMap {
    pub fn new(width: u32, height: u32, k: &DistortedIntrinsics, long_side: u32) -> Self {
        let pinhole = pinhole_for_long_side(width, height, k, long_side);
        let mut table = Vec::with_capacity((pinhole.width * pinhole.height) as usize);
        for v in 0..pinhole.height {
            for u in 0..pinhole.width {
                let p = Self::source_of(&pinhole, k, &Vector2::new(u as f64, v as f64));
                table.push([p.x, p.y]);
            }
        }
        Self { pinhole, table }
    }

    /// 출력 연속 좌표 하나의 원본 좌표(표 없이 바로 계산).
    pub fn source_of(
        pinhole: &Intrinsics,
        k: &DistortedIntrinsics,
        out: &Vector2<f64>,
    ) -> Vector2<f64> {
        let d = k.dist.distort(&pinhole.to_normalized(out));
        Vector2::new(k.fx * d.x + k.cx, k.fy * d.y + k.cy)
    }

    /// 표를 쌍선형 보간해 출력 연속 좌표의 원본 좌표를 얻는다. 표 밖이면 None.
    pub fn lookup(&self, out: &Vector2<f64>) -> Option<Vector2<f64>> {
        let w = self.pinhole.width as usize;
        let h = self.pinhole.height as usize;
        let s = bilinear(w, h, out.x, out.y, |i, c| self.table[i][c], 2)?;
        Some(Vector2::new(s[0], s[1]))
    }
}

/// 격자 값 `get(색인, 채널)` 의 쌍선형 보간. 격자 밖이면 None.
fn bilinear(
    w: usize,
    h: usize,
    x: f64,
    y: f64,
    get: impl Fn(usize, usize) -> f64,
    channels: usize,
) -> Option<[f64; 4]> {
    if !(x >= 0.0 && y >= 0.0 && x <= (w - 1) as f64 && y <= (h - 1) as f64) {
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

/// 왜곡 있는 사진을 긴 변 `long_side` 의 핀홀 사진으로 바꾸고 새 내부 파라미터를 돌려준다.
/// 화소 형은 8비트 채널(최대 4채널)이면 무엇이든 된다.
pub fn undistort_to_long_side<P>(
    img: &ImageBuffer<P, Vec<u8>>,
    k: &DistortedIntrinsics,
    long_side: u32,
) -> (ImageBuffer<P, Vec<u8>>, Intrinsics)
where
    P: Pixel<Subpixel = u8>,
{
    let map = UndistortMap::new(img.width(), img.height(), k, long_side);
    (remap(img, &map), map.pinhole)
}

/// 역사상 표로 원본을 다시 표본화한다.
pub fn remap<P>(img: &ImageBuffer<P, Vec<u8>>, map: &UndistortMap) -> ImageBuffer<P, Vec<u8>>
where
    P: Pixel<Subpixel = u8>,
{
    let nc = P::CHANNEL_COUNT as usize;
    assert!(nc <= 4, "채널 수 {nc} > 4");
    let (sw, sh) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    let (w, h) = (map.pinhole.width, map.pinhole.height);
    let mut buf = vec![0u8; w as usize * h as usize * nc];
    for (i, src) in map.table.iter().enumerate() {
        if let Some(v) = bilinear(sw, sh, src[0], src[1], |p, c| raw[p * nc + c] as f64, nc) {
            for c in 0..nc {
                buf[i * nc + c] = v[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    ImageBuffer::from_raw(w, h, buf).expect("버퍼 크기")
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

    #[test]
    fn output_size_and_intrinsics() {
        let p = pinhole_for_long_side(1600, 1200, &intr(), 960);
        assert_eq!((p.width, p.height), (960, 720));
        assert!((p.fx - 1300.0 * 0.6).abs() < 1e-12);
        assert!((p.cx - (803.5 * 0.6 - 0.5)).abs() < 1e-12);
        let q = pinhole_for_long_side(1200, 1600, &intr(), 960);
        assert_eq!((q.width, q.height), (720, 960));
    }

    /// 표 보간 좌표 → 원본 → 정규 좌표 역변환 → 출력 좌표 왕복.
    #[test]
    fn map_roundtrip() {
        let k = intr();
        let map = UndistortMap::new(1600, 1200, &k, 960);
        let mut worst: f64 = 0.0;
        for i in 0..400 {
            let out = Vector2::new((i as f64 * 37.13) % 959.0, (i as f64 * 23.71) % 719.0);
            let src = map.lookup(&out).unwrap();
            let n = k.unproject(&src).unwrap();
            let back = map.pinhole.to_pixel(&n);
            worst = worst.max((back - out).norm());
        }
        eprintln!("map_roundtrip_worst_px {worst:.3e}");
        // 표 보간 오차: 화소 간 왜곡 2차 변화량 수준(1e-3 px 미만).
        assert!(worst < 1e-3, "{worst}");
    }

    /// 정답 3D 점 주변에 정규 좌표계 가우스 반점을 그린 왜곡 사진을 만들고,
    /// 보정 사진의 반점 무게중심이 핀홀 투영과 맞는지 본다.
    #[test]
    fn blobs_land_on_pinhole_projection() {
        let k = intr();
        let (w, h) = (1600u32, 1200u32);
        let pose = Pose::from_center(
            Rotation3::from_euler_angles(0.05, -0.03, 0.2),
            &Point3::new(0.0, 0.0, -30.0),
        );
        let world: Vec<Point3<f64>> = [
            (0.0, 0.0),
            (-12.0, -8.0),
            (11.0, 7.5),
            (-13.0, 8.0),
            (12.5, -7.0),
            (5.0, -2.0),
        ]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect();
        // 정규 좌표 반점 중심(카메라 좌표 광선 방향).
        let centers: Vec<Vector2<f64>> = world
            .iter()
            .map(|x| {
                let c: Vector3<f64> = pose.transform(x);
                Vector2::new(c.x / c.z, c.y / c.z)
            })
            .collect();
        // 출력에서 σ ≈ 3 px 가 되도록 정규 좌표 σ 를 잡는다.
        let sigma_n = 3.0 / (k.fx * 0.6);
        let img = GrayImage::from_fn(w, h, |u, v| {
            let n = k.unproject(&Vector2::new(u as f64, v as f64)).unwrap();
            let val: f64 = centers
                .iter()
                .map(|c| (-(n - c).norm_squared() / (2.0 * sigma_n * sigma_n)).exp())
                .sum();
            Luma([(250.0 * val).round().min(255.0) as u8])
        });
        let (out, pin) = undistort_to_long_side(&img, &k, 960);
        let cam = Camera {
            intrinsics: pin,
            pose,
        };
        let mut worst: f64 = 0.0;
        for x in &world {
            let p = cam.project(x).unwrap();
            let r = 12i64;
            let (mut sw, mut sx, mut sy) = (0.0, 0.0, 0.0);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (u, v) = (p.x.round() as i64 + dx, p.y.round() as i64 + dy);
                    let val = out.get_pixel(u as u32, v as u32)[0] as f64;
                    sw += val;
                    sx += val * u as f64;
                    sy += val * v as f64;
                }
            }
            let e = (Vector2::new(sx / sw, sy / sw) - p).norm();
            worst = worst.max(e);
        }
        eprintln!("blob_centroid_worst_px {worst:.4}");
        assert!(worst < 0.05, "{worst}");
    }

    #[test]
    fn zero_distortion_rgb_keeps_content() {
        let mut k = intr();
        k.dist = Distortion::default();
        let img = image::RgbImage::from_fn(1600, 1200, |u, v| {
            image::Rgb([(u / 7) as u8, (v / 5) as u8, 128])
        });
        let (out, pin) = undistort_to_long_side(&img, &k, 960);
        // 왜곡이 없으면 출력 (u', v') = 원본 ((u'+0.5)/0.6 − 0.5, ...) 의 선형 보간.
        let px = out.get_pixel(300, 200);
        let su = (300.5 / 0.6 - 0.5) / 7.0;
        assert!((px[0] as f64 - su).abs() <= 1.0, "{:?} {su}", px);
        assert_eq!(px[2], 128);
        assert_eq!((pin.width, pin.height), (960, 720));
    }
}
