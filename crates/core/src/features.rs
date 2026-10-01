//! 다중 스케일 가우시안 차분(DoG) 극값 검출 (Lowe 2004, SPEC §3.1).
//!
//! 진행 상황: 검출기만 있다. 방향 할당·128차원 기술자는 다음 단계.

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

    /// RGB 8비트 → [0,1] 밝기.
    pub fn from_rgb(width: usize, height: usize, rgb: &[u8]) -> Self {
        let data = rgb
            .chunks_exact(3)
            .map(|c| (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0)
            .collect();
        Self {
            width,
            height,
            data,
        }
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
    let mut tmp = GrayImage::new(img.width, img.height);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let xx = (x + j as isize - r).clamp(0, w - 1);
                acc += kv * img.data[(y * w + xx) as usize];
            }
            tmp.data[(y * w + x) as usize] = acc;
        }
    }
    let mut out = GrayImage::new(img.width, img.height);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let yy = (y + j as isize - r).clamp(0, h - 1);
                acc += kv * tmp.data[(yy * w + x) as usize];
            }
            out.data[(y * w + x) as usize] = acc;
        }
    }
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
#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub sigma: f32,
    pub response: f32,
}

/// DoG 극값 검출. 입력은 이미 σ≈0.5 로 흐려진 영상으로 가정한다.
pub fn detect(img: &GrayImage, cfg: &DetectorConfig) -> Vec<Keypoint> {
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
        let (w, h) = (base.width, base.height);
        let scale = (1usize << o) as f32;
        let edge_thr = (cfg.edge_ratio + 1.0).powi(2) / cfg.edge_ratio;
        for l in 1..=s {
            let (d0, d1, d2) = (&dog[l - 1], &dog[l], &dog[l + 1]);
            for y in 1..h - 1 {
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
                    let dxx = d1.at(x + 1, y) + d1.at(x - 1, y) - 2.0 * v;
                    let dyy = d1.at(x, y + 1) + d1.at(x, y - 1) - 2.0 * v;
                    let dxy = 0.25
                        * (d1.at(x + 1, y + 1) - d1.at(x - 1, y + 1) - d1.at(x + 1, y - 1)
                            + d1.at(x - 1, y - 1));
                    let tr = dxx + dyy;
                    let det = dxx * dyy - dxy * dxy;
                    if det <= 0.0 || tr * tr / det >= edge_thr {
                        continue;
                    }
                    out.push(Keypoint {
                        x: x as f32 * scale,
                        y: y as f32 * scale,
                        sigma: cfg.sigma0 * kstep.powi(l as i32) * scale,
                        response: v,
                    });
                }
            }
        }
        base = gauss[s].downsample();
    }
    out.sort_by(|a, b| b.response.abs().total_cmp(&a.response.abs()));
    out.truncate(cfg.max_features);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 중심 (cx,cy), 표준편차 s 인 밝은 가우시안 덩어리.
    fn blob(w: usize, h: usize, cx: f32, cy: f32, s: f32) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
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
            // 스케일 표본 간격 2^(1/3) 이내.
            assert!((0.75..1.35).contains(&ratio), "스케일 비 {ratio}");
        }
    }

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
