//! 다중 스케일 가우시안 차분(DoG) 극값 검출 (Lowe 2004, SPEC §3.1).
//!
//! 검출기(부화소·부스케일 정밀화) → 방향 할당 → 128차원 기술자.

use rayon::prelude::*;

/// 영상 버퍼 길이가 `폭 × 높이 × 채널` 과 다를 때의 오류.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSizeError {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub len: usize,
}

impl std::fmt::Display for ImageSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "영상 버퍼 길이 {} 가 {}x{}x{} 와 맞지 않음",
            self.len, self.width, self.height, self.channels
        )
    }
}

impl std::error::Error for ImageSizeError {}

/// 단일 채널 f32 영상(행 우선).
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    /// RGB 8비트 → [0,1] 밝기. `rgb.len()` 이 `width * height * 3` 이 아니면 패닉한다
    /// (오류로 받으려면 [`GrayImage::try_from_rgb`]).
    pub fn from_rgb(width: usize, height: usize, rgb: &[u8]) -> Self {
        Self::try_from_rgb(width, height, rgb).unwrap_or_else(|e| panic!("from_rgb: {e}"))
    }

    /// RGB 8비트 → [0,1] 밝기. 버퍼 길이가 맞지 않으면 오류.
    pub fn try_from_rgb(width: usize, height: usize, rgb: &[u8]) -> Result<Self, ImageSizeError> {
        Self::from_interleaved(width, height, rgb, 3)
    }

    /// RGBA 8비트 → [0,1] 밝기(알파 무시). 버퍼 길이가 맞지 않으면 오류.
    pub fn try_from_rgba(width: usize, height: usize, rgba: &[u8]) -> Result<Self, ImageSizeError> {
        Self::from_interleaved(width, height, rgba, 4)
    }

    /// 회색조 8비트 → [0,1] 밝기. 버퍼 길이가 맞지 않으면 오류.
    pub fn try_from_gray(width: usize, height: usize, gray: &[u8]) -> Result<Self, ImageSizeError> {
        Self::from_interleaved(width, height, gray, 1)
    }

    fn from_interleaved(
        width: usize,
        height: usize,
        buf: &[u8],
        channels: usize,
    ) -> Result<Self, ImageSizeError> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(channels));
        if expected != Some(buf.len()) {
            return Err(ImageSizeError {
                width,
                height,
                channels,
                len: buf.len(),
            });
        }
        let data = buf
            .chunks_exact(channels)
            .map(|c| {
                if channels >= 3 {
                    (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0
                } else {
                    c[0] as f32 / 255.0
                }
            })
            .collect();
        Ok(Self {
            width,
            height,
            data,
        })
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// 2배 축소(짝수 화소 추출).
    pub fn downsample(&self) -> Self {
        let (w, h) = (self.width / 2, self.height / 2);
        let mut out = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                out.data[y * w + x] = self.at(2 * x, 2 * y);
            }
        }
        out
    }
}

/// 분리형 가우시안 흐림, 경계는 가장자리 복제.
pub fn gaussian_blur(img: &GrayImage, sigma: f32) -> GrayImage {
    let r = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut k: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let (w, h) = (img.width as isize, img.height as isize);
    // 행 단위 병렬. 가로는 가장자리를 복제해 덧댄 행으로, 세로는 행 전체를 한꺼번에 누산한다.
    // 화소마다 탭을 같은 순서로 더하므로 결과는 화소별 직접 계산과 비트 단위로 같다.
    let (wu, ru) = (img.width, r as usize);
    let mut tmp = GrayImage::new(img.width, img.height);
    tmp.data
        .par_chunks_mut(wu.max(1))
        .zip(img.data.par_chunks(wu.max(1)))
        .for_each(|(row, src)| {
            let padded: Vec<f32> = (-r..w + r)
                .map(|x| src[x.clamp(0, w - 1) as usize])
                .collect();
            for (x, o) in row.iter_mut().enumerate() {
                let win = &padded[x..x + 2 * ru + 1];
                let mut acc = 0.0;
                for (kv, v) in k.iter().zip(win) {
                    acc += kv * v;
                }
                *o = acc;
            }
        });
    let mut out = GrayImage::new(img.width, img.height);
    out.data
        .par_chunks_mut(wu.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            for (j, kv) in k.iter().enumerate() {
                let yy = (y as isize + j as isize - r).clamp(0, h - 1) as usize;
                let src = &tmp.data[yy * wu..(yy + 1) * wu];
                for (o, v) in row.iter_mut().zip(src) {
                    *o += kv * v;
                }
            }
        });
    out
}

/// 검출 설정.
#[derive(Clone, Copy, Debug)]
pub struct DetectorConfig {
    pub octaves: usize,
    /// 옥타브당 스케일 간격 수 s (DoG 층 s+2, 가우시안 층 s+3).
    pub scales: usize,
    pub sigma0: f32,
    /// |DoG| 최소값 (밝기 [0,1] 기준).
    pub contrast: f32,
    /// 주곡률 비 상한 r (가장자리 응답 제거, tr²/det < (r+1)²/r).
    pub edge_ratio: f32,
    pub max_features: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            octaves: 4,
            scales: 3,
            sigma0: 1.6,
            contrast: 0.01,
            edge_ratio: 10.0,
            max_features: 8192,
        }
    }
}

/// 특징점: 원본 영상 좌표, 스케일(σ), DoG 응답.
///
/// (x, y) 는 카메라 모형([`crate::camera`])과 같은 화소 중심 규약의 연속 좌표다:
/// 화소 (i, j) 의 중심이 (i + 0.5, j + 0.5), 영상 전체는 [0, w] × [0, h].
/// 그래서 [`crate::camera::Intrinsics::to_normalized`] 에 보정 없이 넣는다
/// ([`Keypoint::pixel`]). 정수 화소 번호(배열 첨자)가 필요하면 `x.floor()` 다.
///
/// σ 는 극값이 난 DoG 층 D = G(kσ) − G(σ) 의 아래쪽 가우시안 σ 다.
/// 반지름 σ_b 인 가우시안 덩어리에서는 σ ≈ σ_b·k^(-1/2) 로 나온다.
#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub sigma: f32,
    pub response: f32,
    /// 주 방향(라디안, 영상 좌표 x→y, [0, 2π)).
    pub angle: f32,
}

impl Keypoint {
    /// 화소 중심 규약 연속 좌표 (x, y). `Intrinsics::to_normalized`·`Camera::unproject` 의 입력 규약과 같다.
    pub fn pixel(&self) -> nalgebra::Vector2<f64> {
        nalgebra::Vector2::new(self.x as f64, self.y as f64)
    }
}

/// 방향 히스토그램 칸 수 (10° 간격).
const ORI_BINS: usize = 36;

/// 특징점 주변 기울기 방향 히스토그램에서 주 방향들을 구한다 (Lowe 2004 §5).
///
/// `img` 는 특징점 스케일로 흐린 영상, (x, y)·`sigma` 는 그 영상의 화소 단위이며
/// (x, y) 는 화소 번호 규약(화소 (i, j) 의 중심이 (i, j))이다. [`Keypoint`] 좌표를 넣으려면 0.5 를 뺀다.
/// 가중치 창 σ_w = 1.5σ, 반지름 3σ_w. 36칸 히스토그램을 [1,1,1]/3 로 6번 평활하고,
/// 최댓값의 80% 이상인 극대마다 포물선 보간한 방향을 낸다.
pub fn dominant_orientations(img: &GrayImage, x: f32, y: f32, sigma: f32) -> Vec<f32> {
    orientations_with(
        img.width,
        img.height,
        |ux, uy| pixel_grad(img, ux, uy),
        x,
        y,
        sigma,
    )
}

/// 화소 (ux, uy) 의 중앙 차분 기울기 크기와 `atan2(gy, gx)`. 테두리 화소는 호출하지 않는다.
#[inline]
fn pixel_grad(img: &GrayImage, ux: usize, uy: usize) -> (f32, f32) {
    let gx = img.at(ux + 1, uy) - img.at(ux - 1, uy);
    let gy = img.at(ux, uy + 1) - img.at(ux, uy - 1);
    ((gx * gx + gy * gy).sqrt(), gy.atan2(gx))
}

/// 한 층 영상의 화소별 기울기(크기, atan2)를 미리 계산해 둔 표. 특징점마다 창이 겹치므로
/// 층마다 한 번만 계산한다. 값은 `pixel_grad` 와 같은 식이라 비트 단위로 같다.
struct GradTable {
    width: usize,
    g: Vec<(f32, f32)>,
}

impl GradTable {
    fn new(img: &GrayImage) -> Self {
        let (w, h) = (img.width, img.height);
        let mut g = vec![(0f32, 0f32); w * h];
        g.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
            if y == 0 || y + 1 >= h {
                return;
            }
            for (x, o) in row.iter_mut().enumerate().take(w - 1).skip(1) {
                *o = pixel_grad(img, x, y);
            }
        });
        Self { width: w, g }
    }

    #[inline]
    fn at(&self, ux: usize, uy: usize) -> (f32, f32) {
        self.g[uy * self.width + ux]
    }
}

fn orientations_with(
    width: usize,
    height: usize,
    grad: impl Fn(usize, usize) -> (f32, f32),
    x: f32,
    y: f32,
    sigma: f32,
) -> Vec<f32> {
    let sw = 1.5 * sigma;
    let r = (3.0 * sw).round() as isize;
    let (xi, yi) = (x.round() as isize, y.round() as isize);
    let (w, h) = (width as isize, height as isize);
    let mut hist = [0f32; ORI_BINS];
    let two_pi = std::f32::consts::TAU;
    for dy in -r..=r {
        for dx in -r..=r {
            let (px, py) = (xi + dx, yi + dy);
            if px < 1 || py < 1 || px >= w - 1 || py >= h - 1 {
                continue;
            }
            let (mag, at) = grad(px as usize, py as usize);
            let (fx, fy) = (px as f32 - x, py as f32 - y);
            let wgt = (-(fx * fx + fy * fy) / (2.0 * sw * sw)).exp();
            let ang = at.rem_euclid(two_pi);
            let bin = ((ang / two_pi * ORI_BINS as f32).round() as usize) % ORI_BINS;
            hist[bin] += wgt * mag;
        }
    }
    for _ in 0..6 {
        let prev = hist;
        for i in 0..ORI_BINS {
            hist[i] =
                (prev[(i + ORI_BINS - 1) % ORI_BINS] + prev[i] + prev[(i + 1) % ORI_BINS]) / 3.0;
        }
    }
    let max = hist.iter().cloned().fold(0.0, f32::max);
    if max <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..ORI_BINS {
        let (l, c, rr) = (
            hist[(i + ORI_BINS - 1) % ORI_BINS],
            hist[i],
            hist[(i + 1) % ORI_BINS],
        );
        if c >= 0.8 * max && c > l && c > rr {
            let off = 0.5 * (l - rr) / (l - 2.0 * c + rr);
            out.push(((i as f32 + off) / ORI_BINS as f32 * two_pi).rem_euclid(two_pi));
        }
    }
    out
}

/// 부화소·부스케일로 정밀화한 극값.
struct Refined {
    xi: usize,
    yi: usize,
    layer: usize,
    x: f32,
    y: f32,
    /// 연속 층 좌표(σ = σ0 k^s).
    s: f32,
    value: f32,
}

/// 3차원 2차 근사로 극값 위치를 정밀화한다 (Brown & Lowe 2002; Lowe 2004 §4).
/// D(x+δ) ≈ D + gᵀδ + ½δᵀHδ → δ = −H⁻¹g, D̂ = D + ½gᵀδ.
/// |δ| 의 어느 성분이 0.5 를 넘으면 이웃 표본으로 옮겨 최대 5번 반복한다.
fn refine_extremum(
    dog: &[GrayImage],
    mut x: usize,
    mut y: usize,
    mut l: usize,
    s: usize,
) -> Option<Refined> {
    let (w, h) = (dog[0].width, dog[0].height);
    for _ in 0..5 {
        let (dm, d0, dp) = (&dog[l - 1], &dog[l], &dog[l + 1]);
        let v = d0.at(x, y);
        let g = nalgebra::Vector3::new(
            0.5 * (d0.at(x + 1, y) - d0.at(x - 1, y)),
            0.5 * (d0.at(x, y + 1) - d0.at(x, y - 1)),
            0.5 * (dp.at(x, y) - dm.at(x, y)),
        );
        let hxx = d0.at(x + 1, y) + d0.at(x - 1, y) - 2.0 * v;
        let hyy = d0.at(x, y + 1) + d0.at(x, y - 1) - 2.0 * v;
        let hss = dp.at(x, y) + dm.at(x, y) - 2.0 * v;
        let hxy = 0.25
            * (d0.at(x + 1, y + 1) - d0.at(x - 1, y + 1) - d0.at(x + 1, y - 1)
                + d0.at(x - 1, y - 1));
        let hxs = 0.25 * (dp.at(x + 1, y) - dp.at(x - 1, y) - dm.at(x + 1, y) + dm.at(x - 1, y));
        let hys = 0.25 * (dp.at(x, y + 1) - dp.at(x, y - 1) - dm.at(x, y + 1) + dm.at(x, y - 1));
        let hm = nalgebra::Matrix3::new(hxx, hxy, hxs, hxy, hyy, hys, hxs, hys, hss);
        let delta = -(hm.try_inverse()? * g);
        if delta.iter().all(|d| d.abs() <= 0.5) {
            return Some(Refined {
                xi: x,
                yi: y,
                layer: l,
                x: x as f32 + delta.x,
                y: y as f32 + delta.y,
                s: l as f32 + delta.z,
                value: v + 0.5 * g.dot(&delta),
            });
        }
        if !delta.iter().all(|d| d.is_finite()) {
            return None;
        }
        let step = |p: usize, d: f32| p as isize + d.round() as isize;
        let (nx, ny, nl) = (step(x, delta.x), step(y, delta.y), step(l, delta.z));
        if nx < 1
            || ny < 1
            || nl < 1
            || nx >= w as isize - 1
            || ny >= h as isize - 1
            || nl > s as isize
        {
            return None;
        }
        (x, y, l) = (nx as usize, ny as usize, nl as usize);
    }
    None
}

/// DoG 극값 검출 (기술자 없이 특징점만).
pub fn detect(img: &GrayImage, cfg: &DetectorConfig) -> Vec<Keypoint> {
    detect_and_describe(img, cfg)
        .into_iter()
        .map(|f| f.kp)
        .collect()
}

/// 기술자 길이: 4×4 칸 × 8 방향.
pub const DESC_LEN: usize = 128;

/// 특징점 + 단위 길이 기술자.
#[derive(Clone, Debug)]
pub struct Feature {
    pub kp: Keypoint,
    pub desc: [f32; DESC_LEN],
}

/// 128차원 기울기 방향 기술자 (Lowe 2004 §6).
///
/// 주 방향으로 돌린 4×4 칸(칸 너비 3σ), 칸마다 8방향 히스토그램.
/// 가우시안 가중(σ = 칸 2개 = 창 너비의 절반), 위치 2축·방향 1축 삼선형 보간.
/// 단위 길이로 정규화 → 0.2 로 자르기 → 다시 정규화.
/// (x, y) 는 [`dominant_orientations`] 와 같이 `img` 의 화소 번호 규약이다.
pub fn describe(img: &GrayImage, x: f32, y: f32, sigma: f32, angle: f32) -> [f32; DESC_LEN] {
    describe_with(
        img.width,
        img.height,
        |ux, uy| pixel_grad(img, ux, uy),
        x,
        y,
        sigma,
        angle,
    )
}

fn describe_with(
    width: usize,
    height: usize,
    grad: impl Fn(usize, usize) -> (f32, f32),
    x: f32,
    y: f32,
    sigma: f32,
    angle: f32,
) -> [f32; DESC_LEN] {
    const NC: usize = 4;
    const NO: usize = 8;
    let cell = 3.0 * sigma;
    let r = (cell * std::f32::consts::SQRT_2 * (NC as f32 + 1.0) * 0.5).round() as isize;
    let (c, sn) = (angle.cos(), angle.sin());
    let (xi, yi) = (x.round() as isize, y.round() as isize);
    let (w, h) = (width as isize, height as isize);
    let two_pi = std::f32::consts::TAU;
    let mut hist = [0f32; DESC_LEN];
    for dy in -r..=r {
        for dx in -r..=r {
            let (px, py) = (xi + dx, yi + dy);
            if px < 1 || py < 1 || px >= w - 1 || py >= h - 1 {
                continue;
            }
            let (fx, fy) = (px as f32 - x, py as f32 - y);
            // 특징점 틀로 회전(−angle), 칸 단위로.
            let u = (c * fx + sn * fy) / cell;
            let v = (-sn * fx + c * fy) / cell;
            let (bu, bv) = (u + NC as f32 / 2.0 - 0.5, v + NC as f32 / 2.0 - 0.5);
            if bu <= -1.0 || bv <= -1.0 || bu >= NC as f32 || bv >= NC as f32 {
                continue;
            }
            let (mag, at) = grad(px as usize, py as usize);
            let ori = (at - angle).rem_euclid(two_pi);
            let bo = ori / two_pi * NO as f32;
            let wgt = (-(u * u + v * v) / (2.0 * (NC as f32 / 2.0).powi(2))).exp();
            let m = mag * wgt;
            let (u0, v0, o0) = (bu.floor(), bv.floor(), bo.floor());
            let (du, dv, dob) = (bu - u0, bv - v0, bo - o0);
            for (iv, wv) in [(v0 as isize, 1.0 - dv), (v0 as isize + 1, dv)] {
                if iv < 0 || iv >= NC as isize {
                    continue;
                }
                for (iu, wu) in [(u0 as isize, 1.0 - du), (u0 as isize + 1, du)] {
                    if iu < 0 || iu >= NC as isize {
                        continue;
                    }
                    for (io, wo) in [(o0 as usize % NO, 1.0 - dob), ((o0 as usize + 1) % NO, dob)] {
                        hist[(iv as usize * NC + iu as usize) * NO + io] += m * wv * wu * wo;
                    }
                }
            }
        }
    }
    let normalize = |h: &mut [f32; DESC_LEN]| {
        let n = h.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n > 0.0 {
            h.iter_mut().for_each(|v| *v /= n);
        }
    };
    normalize(&mut hist);
    hist.iter_mut().for_each(|v| *v = v.min(0.2));
    normalize(&mut hist);
    hist
}

/// DoG 극값 검출 + 방향 + 기술자. 입력은 이미 σ≈0.5 로 흐려진 영상으로 가정한다.
///
/// 특징점 좌표는 화소 중심 규약([`Keypoint`])으로 내보낸다. 옥타브 o 의 화소 j 는
/// 원본 화소 2^o·j 를 뽑은 것([`GrayImage::downsample`])이라 그 중심은 원본 연속 좌표
/// 2^o·j + 0.5 에 있다. 따라서 옥타브 안 화소 번호 좌표 u 는 2^o·u + 0.5 로 옮긴다.
pub fn detect_and_describe(img: &GrayImage, cfg: &DetectorConfig) -> Vec<Feature> {
    let s = cfg.scales;
    let kstep = 2f32.powf(1.0 / s as f32);
    let mut base = gaussian_blur(img, (cfg.sigma0 * cfg.sigma0 - 0.25).max(0.01).sqrt());
    let mut out = Vec::new();
    for o in 0..cfg.octaves {
        if base.width < 16 || base.height < 16 {
            break;
        }
        // 가우시안 층: σ_i = σ0 k^i, 이전 층에서 증분 흐림.
        let mut gauss = vec![base.clone()];
        for i in 1..s + 3 {
            let prev = cfg.sigma0 * kstep.powi(i as i32 - 1);
            let cur = prev * kstep;
            gauss.push(gaussian_blur(
                &gauss[i - 1],
                (cur * cur - prev * prev).sqrt(),
            ));
        }
        let dog: Vec<GrayImage> = gauss
            .windows(2)
            .map(|p| GrayImage {
                width: p[0].width,
                height: p[0].height,
                data: p[1]
                    .data
                    .iter()
                    .zip(&p[0].data)
                    .map(|(a, b)| a - b)
                    .collect(),
            })
            .collect();
        // 정밀화된 극값은 층 1..=s 에 있다(refine_extremum). 그 층들의 기울기 표.
        let grads: Vec<GradTable> = (0..s + 3)
            .into_par_iter()
            .map(|i| {
                if (1..=s).contains(&i) {
                    GradTable::new(&gauss[i])
                } else {
                    GradTable {
                        width: 0,
                        g: Vec::new(),
                    }
                }
            })
            .collect();
        let (w, h) = (base.width, base.height);
        let scale = (1usize << o) as f32;
        let edge_thr = (cfg.edge_ratio + 1.0).powi(2) / cfg.edge_ratio;
        for l in 1..=s {
            let (d0, d1, d2) = (&dog[l - 1], &dog[l], &dog[l + 1]);
            // 행 단위 병렬, 행 순서대로 이어 붙여 직렬과 같은 순서를 지킨다.
            let rows: Vec<Vec<Feature>> = (1..h - 1)
                .into_par_iter()
                .map(|y| {
                    let mut out = Vec::new();
                    for x in 1..w - 1 {
                        let v = d1.at(x, y);
                        if v.abs() < cfg.contrast {
                            continue;
                        }
                        let mut is_max = true;
                        let mut is_min = true;
                        'n: for d in [d0, d1, d2] {
                            for yy in y - 1..=y + 1 {
                                for xx in x - 1..=x + 1 {
                                    if std::ptr::eq(d, d1) && xx == x && yy == y {
                                        continue;
                                    }
                                    let u = d.at(xx, yy);
                                    is_max &= v > u;
                                    is_min &= v < u;
                                    if !is_max && !is_min {
                                        break 'n;
                                    }
                                }
                            }
                        }
                        if !(is_max || is_min) {
                            continue;
                        }
                        let Some(r) = refine_extremum(&dog, x, y, l, s) else {
                            continue;
                        };
                        if r.value.abs() < cfg.contrast {
                            continue;
                        }
                        let d = &dog[r.layer];
                        let (xi, yi) = (r.xi, r.yi);
                        let c = d.at(xi, yi);
                        let dxx = d.at(xi + 1, yi) + d.at(xi - 1, yi) - 2.0 * c;
                        let dyy = d.at(xi, yi + 1) + d.at(xi, yi - 1) - 2.0 * c;
                        let dxy = 0.25
                            * (d.at(xi + 1, yi + 1) - d.at(xi - 1, yi + 1) - d.at(xi + 1, yi - 1)
                                + d.at(xi - 1, yi - 1));
                        let tr = dxx + dyy;
                        let det = dxx * dyy - dxy * dxy;
                        if det <= 0.0 || tr * tr / det >= edge_thr {
                            continue;
                        }
                        let sig_oct = cfg.sigma0 * kstep.powf(r.s);
                        let gt = &grads[r.layer];
                        let grad = |ux, uy| gt.at(ux, uy);
                        for angle in orientations_with(w, h, grad, r.x, r.y, sig_oct) {
                            out.push(Feature {
                                kp: Keypoint {
                                    x: r.x * scale + 0.5,
                                    y: r.y * scale + 0.5,
                                    sigma: sig_oct * scale,
                                    response: r.value,
                                    angle,
                                },
                                desc: describe_with(w, h, grad, r.x, r.y, sig_oct, angle),
                            });
                        }
                    }
                    out
                })
                .collect();
            out.extend(rows.into_iter().flatten());
        }
        base = gauss[s].downsample();
    }
    out.sort_by(|a, b| b.kp.response.abs().total_cmp(&a.kp.response.abs()));
    out.truncate(cfg.max_features);
    out
}

#[cfg(test)]
mod image_buffer_tests {
    use super::*;

    #[test]
    fn rejects_mismatched_buffer_lengths() {
        let e = GrayImage::try_from_rgb(10, 10, &[0u8; 30]).unwrap_err();
        assert_eq!((e.len, e.channels), (30, 3));
        assert!(GrayImage::try_from_rgb(10, 10, &[0u8; 301]).is_err());
        assert!(GrayImage::try_from_rgba(10, 10, &[0u8; 300]).is_err());
        assert!(GrayImage::try_from_gray(10, 10, &[0u8; 99]).is_err());
        assert!(GrayImage::try_from_rgb(usize::MAX, 2, &[]).is_err());
    }

    #[test]
    fn channel_layouts_agree() {
        let gray: Vec<u8> = (0..12).map(|i| (i * 20) as u8).collect();
        let rgb: Vec<u8> = gray.iter().flat_map(|&v| [v, v, v]).collect();
        let rgba: Vec<u8> = gray.iter().flat_map(|&v| [v, v, v, 7]).collect();
        let a = GrayImage::try_from_gray(4, 3, &gray).unwrap();
        let b = GrayImage::try_from_rgb(4, 3, &rgb).unwrap();
        let c = GrayImage::try_from_rgba(4, 3, &rgba).unwrap();
        for i in 0..12 {
            assert!((a.data[i] - b.data[i]).abs() < 1e-6);
            assert!((a.data[i] - c.data[i]).abs() < 1e-6);
        }
        assert!((a.data[11] - 220.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "영상 버퍼 길이 30")]
    fn from_rgb_panics_with_message() {
        let _ = GrayImage::from_rgb(10, 10, &[0u8; 30]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector2;

    /// 중심 (cx,cy), 표준편차 s 인 밝은 가우시안 덩어리. 카메라 규약으로 그린다:
    /// 화소 (x, y) 의 값은 연속 좌표 (x + 0.5, y + 0.5) 에서 잰다.
    fn blob(w: usize, h: usize, cx: f32, cy: f32, s: f32) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r2 = (x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2);
                img.data[y * w + x] = 0.2 + 0.6 * (-r2 / (2.0 * s * s)).exp();
            }
        }
        img
    }

    #[test]
    fn blur_preserves_mean_and_constant() {
        let img = GrayImage {
            width: 20,
            height: 10,
            data: vec![0.7; 200],
        };
        let b = gaussian_blur(&img, 2.0);
        assert!(b.data.iter().all(|v| (v - 0.7).abs() < 1e-5));
    }

    #[test]
    fn detects_blob_location_and_scale() {
        // 가우시안 덩어리 σ_b 의 DoG(정규화 라플라시안) 극값 스케일은 σ ≈ σ_b.
        for (sb, cx, cy) in [(4.0f32, 61.0f32, 47.0f32), (8.0, 70.0, 66.0)] {
            let img = blob(140, 120, cx, cy, sb);
            let kps = detect(&img, &DetectorConfig::default());
            let best = kps.first().expect("검출 없음");
            let pos_err = ((best.x - cx).powi(2) + (best.y - cy).powi(2)).sqrt();
            let ratio = best.sigma / sb;
            eprintln!("blob sb={sb} pos_err={pos_err:.2} sigma_ratio={ratio:.3}");
            assert!(
                pos_err <= 2.0 * (best.sigma / 1.6).max(1.0),
                "위치 오차 {pos_err}"
            );
            // 정밀화 후 σ·√k ≈ σ_b (k = 2^(1/3)), 3% 이내.
            let corrected = ratio * 2f32.powf(1.0 / 6.0);
            assert!((corrected - 1.0).abs() < 0.03, "보정 스케일 비 {corrected}");
            // 부화소 정밀화: 정수 중심에서 0.1 px 이내.
            assert!(pos_err < 0.1, "위치 오차 {pos_err}");
        }
    }

    #[test]
    fn refines_subpixel_center() {
        // 정수 격자에서 벗어난 중심: 정밀화 없이는 최대 0.5 px 오차.
        for (cx, cy) in [(61.3f32, 47.6f32), (60.75, 48.2)] {
            let img = blob(140, 120, cx, cy, 4.0);
            let best = detect(&img, &DetectorConfig::default())[0];
            let err = ((best.x - cx).powi(2) + (best.y - cy).powi(2)).sqrt();
            eprintln!("subpixel ({cx},{cy}) err={err:.3}");
            assert!(err < 0.1, "부화소 오차 {err}");
        }
    }

    #[test]
    fn keypoints_follow_camera_pixel_convention() {
        // 카메라 규약으로 그린 덩어리의 검출 좌표가 보정 없이 연속 좌표 중심과 0.1 px 안에서 맞는다.
        // 주점 (w/2, h/2) 은 화소 경계라 화소 번호 규약 출력이면 오차가 √2·0.5 = 0.71 px 가 된다.
        // 주점에서 벗어난 점은 카메라 투영으로 만든다(세 번째는 둘째 옥타브(σ_b = 8)에서 나온다).
        use crate::camera::{Camera, Intrinsics, Pose};
        use crate::math::{Point3, Rotation3};
        let (w, h) = (160u32, 120u32);
        let k = Intrinsics::from_hfov(w, h, 60f64.to_radians());
        let cam = Camera {
            intrinsics: k,
            pose: Pose::from_center(Rotation3::identity(), &Point3::origin()),
        };
        let off = cam.project(&Point3::new(0.83, -0.41, 10.0)).unwrap();
        let cases = [
            (k.cx, k.cy, 4.0f32),
            (off.x, off.y, 4.0),
            (k.cx + 3.25, k.cy - 1.75, 8.0),
        ];
        for (cx, cy, sb) in cases {
            let img = blob(w as usize, h as usize, cx as f32, cy as f32, sb);
            let best = detect(&img, &DetectorConfig::default())[0];
            let err = (best.pixel() - Vector2::new(cx, cy)).norm();
            eprintln!("camera convention ({cx:.3},{cy:.3}) σ_b={sb} err={err:.4}");
            assert!(err < 0.1, "({cx}, {cy}) 위치 오차 {err} px");
            // 정규 좌표도 투영 전 방향과 맞는다.
            let n = k.to_normalized(&best.pixel());
            let n0 = k.to_normalized(&Vector2::new(cx, cy));
            assert!((n - n0).norm() * k.fx < 0.1);
        }
    }

    fn angle_diff(a: f32, b: f32) -> f32 {
        let d = (a - b).rem_euclid(std::f32::consts::TAU);
        d.min(std::f32::consts::TAU - d)
    }

    #[test]
    fn orientation_follows_rotation() {
        // 밝기가 θ 방향으로 증가하는 경사면 + 무관한 덩어리 → 주 방향 ≈ θ.
        let mut worst = 0f32;
        for k in 0..12 {
            let th = k as f32 * 30f32.to_radians() + 0.1;
            let (cx, cy) = (64.0f32, 64.0f32);
            // dominant_orientations 는 화소 번호 규약이라 덩어리 중심을 화소 (64, 64) 의 중심에 둔다.
            let mut img = blob(128, 128, cx + 0.5, cy + 0.5, 6.0);
            for y in 0..128 {
                for x in 0..128 {
                    let t = (x as f32 - cx) * th.cos() + (y as f32 - cy) * th.sin();
                    img.data[y * 128 + x] += 0.02 * t;
                }
            }
            let g = gaussian_blur(&img, 2.0);
            let oris = dominant_orientations(&g, cx, cy, 4.0);
            let best = oris
                .iter()
                .map(|&a| angle_diff(a, th))
                .fold(f32::INFINITY, f32::min);
            worst = worst.max(best);
        }
        eprintln!("orientation worst err={:.2} deg", worst.to_degrees());
        assert!(
            worst.to_degrees() < 3.0,
            "방향 오차 {}°",
            worst.to_degrees()
        );
    }

    /// 결정적 난수 덩어리 무늬 (선형 합동 생성기).
    fn texture(w: usize, h: usize, seed: u64) -> GrayImage {
        let mut st = seed;
        let mut rnd = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as f32 / (1u64 << 31) as f32
        };
        let blobs: Vec<[f32; 4]> = (0..400)
            .map(|_| [rnd() * 400.0, rnd() * 400.0, 1.5 + rnd() * 5.0, rnd() - 0.5])
            .collect();
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // 영상 중심 기준 정답 좌표계(-200..200).
                let (px, py) = (
                    x as f32 - w as f32 / 2.0 + 200.0,
                    y as f32 - h as f32 / 2.0 + 200.0,
                );
                let mut v = 0.5;
                for b in &blobs {
                    let d2 = (px - b[0]).powi(2) + (py - b[1]).powi(2);
                    if d2 < 16.0 * b[2] * b[2] {
                        v += 0.5 * b[3] * (-d2 / (2.0 * b[2] * b[2])).exp();
                    }
                }
                img.data[y * w + x] = v;
            }
        }
        img
    }

    /// 중심 기준 회전 θ·축척 sc 로 다시 그린다(정답 좌표 함수 사용, 보간 오차 없음).
    fn texture_warped(w: usize, h: usize, seed: u64, th: f32, sc: f32) -> GrayImage {
        let big = texture(800, 800, seed);
        let mut img = GrayImage::new(w, h);
        let (c, s) = (th.cos(), th.sin());
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 - w as f32 / 2.0, y as f32 - h as f32 / 2.0);
                // 역사상: 원본 좌표 = R(−θ)·p / sc.
                let ox = (c * fx + s * fy) / sc + 400.0;
                let oy = (-s * fx + c * fy) / sc + 400.0;
                let (x0, y0) = (ox.floor() as usize, oy.floor() as usize);
                let (ax, ay) = (ox - x0 as f32, oy - y0 as f32);
                let at = |xx: usize, yy: usize| big.at(xx.min(799), yy.min(799));
                img.data[y * w + x] = (1.0 - ay) * ((1.0 - ax) * at(x0, y0) + ax * at(x0 + 1, y0))
                    + ay * ((1.0 - ax) * at(x0, y0 + 1) + ax * at(x0 + 1, y0 + 1));
            }
        }
        img
    }

    /// `texture_warped` 의 회전·축척 중심을 특징점 좌표(화소 중심 규약)로 적은 것.
    /// 영상은 화소 번호 (w/2, h/2) 를 중심으로 다시 그렸으므로 연속 좌표로는 +0.5 다.
    fn warp_center(w: usize, h: usize) -> (f32, f32) {
        (w as f32 / 2.0 + 0.5, h as f32 / 2.0 + 0.5)
    }

    /// 최근접/차근접 비율 검사 매칭 → (정답 2 px 이내 비율, 매칭 수, 원본 특징 수).
    fn match_accuracy(th: f32, sc: f32) -> (f32, usize, usize) {
        let (w, h) = (240usize, 240usize);
        let a = detect_and_describe(
            &texture_warped(w, h, 7, 0.0, 1.0),
            &DetectorConfig::default(),
        );
        let b = detect_and_describe(&texture_warped(w, h, 7, th, sc), &DetectorConfig::default());
        let (c, s) = (th.cos(), th.sin());
        let (mut good, mut n) = (0, 0);
        for fa in &a {
            let mut best = (f32::INFINITY, f32::INFINITY, 0usize);
            for (j, fb) in b.iter().enumerate() {
                let d: f32 = fa
                    .desc
                    .iter()
                    .zip(&fb.desc)
                    .map(|(p, q)| (p - q).powi(2))
                    .sum();
                if d < best.0 {
                    best = (d, best.0, j);
                } else if d < best.1 {
                    best.1 = d;
                }
            }
            if best.0 >= 0.8 * 0.8 * best.1 {
                continue;
            }
            n += 1;
            let (ox, oy) = warp_center(w, h);
            let (fx, fy) = (fa.kp.x - ox, fa.kp.y - oy);
            let (ex, ey) = (sc * (c * fx - s * fy) + ox, sc * (s * fx + c * fy) + oy);
            let kb = b[best.2].kp;
            if ((kb.x - ex).powi(2) + (kb.y - ey).powi(2)).sqrt() < 2.0 {
                good += 1;
            }
        }
        (good as f32 / n.max(1) as f32, n, a.len())
    }

    /// 재검출률: 변환 후에도 영상 안쪽(가장자리 16 px 제외)에 들어오는 원본 특징점 중
    /// 정답 위치 2 px 이내에 특징점이 다시 검출된 비율 → (비율, 분모).
    fn repeatability(th: f32, sc: f32) -> (f32, usize) {
        let (w, h) = (240usize, 240usize);
        let cfg = DetectorConfig::default();
        let a = detect(&texture_warped(w, h, 7, 0.0, 1.0), &cfg);
        let b = detect(&texture_warped(w, h, 7, th, sc), &cfg);
        let (c, s) = (th.cos(), th.sin());
        let (mut hit, mut n) = (0, 0);
        let (ox, oy) = warp_center(w, h);
        for ka in &a {
            let (fx, fy) = (ka.x - ox, ka.y - oy);
            let (ex, ey) = (sc * (c * fx - s * fy) + ox, sc * (s * fx + c * fy) + oy);
            if ex < 16.0 || ey < 16.0 || ex > w as f32 - 16.0 || ey > h as f32 - 16.0 {
                continue;
            }
            n += 1;
            if b.iter()
                .any(|kb| ((kb.x - ex).powi(2) + (kb.y - ey).powi(2)).sqrt() < 2.0)
            {
                hit += 1;
            }
        }
        (hit as f32 / n.max(1) as f32, n)
    }

    #[test]
    fn repeatability_under_rotation_and_scale() {
        for (deg, sc, min_rate) in [
            (0.0f32, 1.0f32, 0.99f32),
            (30.0, 1.0, 0.85),
            (90.0, 1.0, 0.95),
            (45.0, 0.8, 0.70),
            (0.0, 1.25, 0.70),
        ] {
            let (rate, n) = repeatability(deg.to_radians(), sc);
            eprintln!("rot={deg} scale={sc} repeat={rate:.3} of {n}");
            assert!(n >= 30, "비교 대상 {n}");
            assert!(rate >= min_rate, "재검출률 {rate}");
        }
    }

    #[test]
    fn descriptor_unit_length() {
        let img = texture_warped(160, 160, 3, 0.0, 1.0);
        let f = detect_and_describe(&img, &DetectorConfig::default());
        assert!(!f.is_empty());
        for x in &f {
            let n: f32 = x.desc.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4);
            assert!(x.desc.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn matching_under_rotation_and_scale() {
        for (deg, sc) in [
            (0.0f32, 1.0f32),
            (30.0, 1.0),
            (90.0, 1.0),
            (45.0, 0.8),
            (0.0, 1.25),
        ] {
            let (acc, n, total) = match_accuracy(deg.to_radians(), sc);
            eprintln!("rot={deg} scale={sc} matches={n}/{total} precision={acc:.3}");
            assert!(n >= 40, "매칭 수 {n}");
            assert!(acc >= 0.95, "정확도 {acc}");
        }
    }

    /// 합성 드론 장면에서 같은 카메라의 이웃 위치 두 장을 매칭하고, 정답 깊이로
    /// 옮긴 위치와 비교한다 → (정답 2 px 이내 비율, 매칭 수).
    fn scene_match_accuracy(step: usize) -> (f32, usize) {
        use crate::synth::{Scene, SceneConfig};
        let scene = Scene::new(SceneConfig {
            width: 480,
            height: 270,
            ..SceneConfig::default()
        });
        let va = &scene.views[0];
        let vb = scene
            .views
            .iter()
            .find(|v| v.cam == va.cam && v.position == va.position + step)
            .unwrap();
        let cfg = DetectorConfig::default();
        let (ia, da) = scene.render(va);
        let (ib, _) = scene.render(vb);
        let fa = detect_and_describe(&GrayImage::from_rgb(480, 270, &ia.data), &cfg);
        let fb = detect_and_describe(&GrayImage::from_rgb(480, 270, &ib.data), &cfg);
        let (mut good, mut n) = (0, 0);
        for f in &fa {
            let mut best = (f32::INFINITY, f32::INFINITY, 0usize);
            for (j, g) in fb.iter().enumerate() {
                let d: f32 = f
                    .desc
                    .iter()
                    .zip(&g.desc)
                    .map(|(p, q)| (p - q).powi(2))
                    .sum();
                if d < best.0 {
                    best = (d, best.0, j);
                } else if d < best.1 {
                    best.1 = d;
                }
            }
            if best.0 >= 0.8 * 0.8 * best.1 {
                continue;
            }
            let (px, py) = (f.kp.x.floor() as usize, f.kp.y.floor() as usize);
            let z = da[py.min(269) * 480 + px.min(479)];
            if !z.is_finite() {
                continue;
            }
            // 특징점 좌표는 카메라 모형과 같은 화소 중심 규약이라 그대로 넣는다.
            let pa = f.kp.pixel();
            let Some(e) = vb.camera.project(&va.camera.unproject(&pa, z as f64)) else {
                continue;
            };
            n += 1;
            let k = fb[best.2].kp;
            if (k.pixel() - e).norm() < 2.0 {
                good += 1;
            }
        }
        (good as f32 / n.max(1) as f32, n)
    }

    #[test]
    fn matching_on_synthetic_drone_views() {
        for step in [1usize, 3] {
            let (acc, n) = scene_match_accuracy(step);
            eprintln!("scene step={step} matches={n} precision={acc:.3}");
            assert!(n >= 250, "매칭 수 {n}");
            assert!(acc >= 0.95, "정확도 {acc}");
        }
    }

    fn scene_image(w: usize, h: usize) -> GrayImage {
        use crate::synth::{Scene, SceneConfig};
        let scene = Scene::new(SceneConfig {
            width: w as u32,
            height: h as u32,
            ..SceneConfig::default()
        });
        let (img, _) = scene.render(&scene.views[0]);
        GrayImage::from_rgb(w, h, &img.data)
    }

    #[test]
    fn parallel_detection_matches_single_thread() {
        // 병렬 흐림·극값 탐색이 스레드 1개 실행과 비트 단위로 같은 특징(순서 포함)을 낸다.
        let img = scene_image(480, 270);
        let cfg = DetectorConfig::default();
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| detect_and_describe(&img, &cfg));
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| detect_and_describe(&img, &cfg));
        assert!(one.len() > 100, "특징 {}", one.len());
        assert_eq!(one.len(), many.len());
        for (p, q) in one.iter().zip(&many) {
            assert_eq!(
                (p.kp.x, p.kp.y, p.kp.sigma, p.kp.angle),
                (q.kp.x, q.kp.y, q.kp.sigma, q.kp.angle)
            );
            assert_eq!(p.desc, q.desc);
        }
    }

    #[test]
    #[ignore = "시간 측정: cargo test --release -- --ignored --test-threads=1 detection_timing"]
    fn detection_timing() {
        // F-014 확인 기준: 합성 1920×1080 한 장 검출 ≤ 0.4 s.
        let img = scene_image(1920, 1080);
        let cfg = DetectorConfig::default();
        let _ = detect_and_describe(&img, &cfg); // 예열
        let t = std::time::Instant::now();
        let f = detect_and_describe(&img, &cfg);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "detect_and_describe 1920x1080: {:.3} s, 특징 {}, 스레드 {}",
            dt,
            f.len(),
            rayon::current_num_threads()
        );
        assert!(dt <= 0.4, "검출 {dt:.3} s > 0.4 s");
    }

    /// 검출 결과의 FNV-1a 64비트 해시: 특징마다 x, y(각각 `off` 를 더한 f32), σ, 응답, 방향,
    /// 기술자 128개의 비트를 차례로 섞는다(순서 포함).
    fn features_hash(f: &[Feature], off: f32) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: f32| {
            for byte in v.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        for x in f {
            let k = &x.kp;
            for v in [k.x + off, k.y + off, k.sigma, k.response, k.angle] {
                mix(v);
            }
            x.desc.iter().for_each(|&v| mix(v));
        }
        h
    }

    #[test]
    fn detection_regression_hash() {
        // 고정 합성 영상(scene_image 480×270, 첫 시점)의 검출 결과(개수 + 좌표·σ·응답·방향·기술자 비트,
        // 순서 포함)를 상수로 고정한다. 흐림 누산 순서·극값 판정·정밀화·방향·기술자 계산을 바꾸면
        // 비트가 달라져 이 시험이 실패한다(스레드 비교 시험은 이전 구현과의 차이를 잡지 못한다).
        //
        // 기대값의 근거: 화소 중심 규약 출력(x, y 에 +0.5)을 넣은 커밋의 구현으로 계산했다.
        // 그 직전 main(08d5248, 화소 번호 규약)의 출력은 745개, 해시 0x45f7_d13c_594d_4054 이고,
        // 그 출력의 x, y 에 f32 로 0.5 를 더해 같은 방식으로 섞은 해시가 아래 기대값과 같다.
        // 즉 규약 변경은 좌표를 정확히 +0.5 옮긴 것뿐이고 개수·순서·σ·응답·방향·기술자는 그대로다.
        let f = detect_and_describe(&scene_image(480, 270), &DetectorConfig::default());
        let got = (f.len(), features_hash(&f, 0.0));
        eprintln!("detection hash: {} {:#x}", got.0, got.1);
        assert_eq!(got, EXPECT_DETECTION);
    }
    const EXPECT_DETECTION: (usize, u64) = (745, 0x2a1c_72f3_9c5e_a950);

    #[test]
    fn flat_image_has_no_features() {
        let img = GrayImage {
            width: 64,
            height: 64,
            data: vec![0.5; 64 * 64],
        };
        assert!(detect(&img, &DetectorConfig::default()).is_empty());
    }
}
