//! 입력 데이터셋 읽기: 세 카메라 사진 폴더와 GPS 텍스트, 구역 분할.
//!
//! 폴더 구조: `images/cam{F,R,L}/cam{F,R,L}_{번호:04}.jpg`, `gps.txt`.
//! `gps.txt` 는 `이름 위도 경도 고도` 한 줄씩이며, 이름 끝의 숫자를 프레임 번호로 본다
//! (`camF_0003.jpg`, `0003` 모두 프레임 3). 빈 줄과 `#` 로 시작하는 줄은 건너뛴다.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::geo::{geodetic_to_enu, Geodetic};
use crate::math::Vector3;

/// 카메라 이름 순서: 앞, 오른쪽, 왼쪽.
pub const CAMERAS: [&str; 3] = ["camF", "camR", "camL"];

/// 데이터셋 설정.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatasetConfig {
    /// 프레임 간격.
    pub stride: usize,
    /// 구역 하나의 위치 수.
    pub span: usize,
    /// 구역 앞뒤로 겹치는 위치 수.
    pub ovl: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            stride: 3,
            span: 12,
            ovl: 2,
        }
    }
}

/// 읽기 오류.
#[derive(Debug)]
pub enum DatasetError {
    Io(PathBuf, std::io::Error),
    /// 필요한 파일·폴더가 없음.
    Missing(PathBuf),
    /// GPS 줄 형식 오류(줄 번호는 1부터).
    GpsFormat {
        line: usize,
        msg: String,
    },
    /// 선택된 프레임의 GPS 가 없음.
    GpsMissing {
        frame: u32,
    },
    /// 설정 값 오류.
    Config(String),
    /// 쓸 수 있는 위치가 없음.
    Empty,
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Self::Missing(p) => write!(f, "파일 없음: {}", p.display()),
            Self::GpsFormat { line, msg } => write!(f, "gps.txt {line}번째 줄: {msg}"),
            Self::GpsMissing { frame } => write!(f, "GPS 누락: 프레임 {frame}"),
            Self::Config(m) => write!(f, "설정 오류: {m}"),
            Self::Empty => write!(f, "세 카메라가 모두 있는 위치가 없음"),
        }
    }
}

impl std::error::Error for DatasetError {}

/// GPS 한 줄.
#[derive(Clone, Debug, PartialEq)]
pub struct GpsRecord {
    pub name: String,
    pub frame: u32,
    pub geo: Geodetic,
    /// 줄 번호(1부터).
    pub line: usize,
}

/// 위치 하나: 같은 프레임의 세 카메라 사진과 GPS.
#[derive(Clone, Debug)]
pub struct Position {
    /// 위치 번호(0부터).
    pub index: usize,
    /// 원래 프레임 번호.
    pub frame: u32,
    /// `CAMERAS` 순서의 사진 경로.
    pub images: [PathBuf; 3],
    pub geo: Geodetic,
    /// 동-북-위 좌표(m), 원점은 gps.txt 의 첫 기록.
    pub enu: Vector3<f64>,
}

/// 읽은 데이터셋.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub root: PathBuf,
    pub config: DatasetConfig,
    pub origin: Geodetic,
    pub positions: Vec<Position>,
}

impl Dataset {
    /// 사진 수(위치 수 × 3).
    pub fn image_count(&self) -> usize {
        self.positions.len() * CAMERAS.len()
    }

    /// 이 데이터셋의 구역 목록.
    pub fn chunks(&self) -> Vec<Range<usize>> {
        chunk_ranges(self.positions.len(), self.config.span, self.config.ovl)
    }
}

/// 구역 분할: start = 0, SPAN, 2·SPAN, … 마다 위치 [start-OVL, start+SPAN+OVL) 를 한 구역으로,
/// 범위는 0..n 으로 자른다. `span == 0` 이면 빈 목록.
pub fn chunk_ranges(n: usize, span: usize, ovl: usize) -> Vec<Range<usize>> {
    if span == 0 {
        return Vec::new();
    }
    (0..n)
        .step_by(span)
        .map(|s| s.saturating_sub(ovl)..(s + span + ovl).min(n))
        .collect()
}

/// 이름 끝의 숫자(확장자 제외)를 프레임 번호로.
fn frame_of_name(name: &str) -> Option<u32> {
    let stem = match name.rsplit_once('.') {
        Some((s, ext)) if !ext.is_empty() && !ext.chars().all(|c| c.is_ascii_digit()) => s,
        _ => name,
    };
    let start = stem
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_digit())
        .last()
        .map(|(i, _)| i)?;
    stem[start..].parse().ok()
}

/// gps.txt 내용 해석.
pub fn parse_gps(text: &str) -> Result<Vec<GpsRecord>, DatasetError> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let err = |msg: String| DatasetError::GpsFormat { line, msg };
        let f: Vec<&str> = t.split_whitespace().collect();
        if f.len() != 4 {
            return Err(err(format!(
                "항목 4개(이름 위도 경도 고도)가 아니라 {}개",
                f.len()
            )));
        }
        let num = |s: &str, what: &str| -> Result<f64, DatasetError> {
            match s.parse::<f64>() {
                Ok(v) if v.is_finite() => Ok(v),
                _ => Err(err(format!("{what} 값이 수가 아님: {s}"))),
            }
        };
        let (lat, lon, alt) = (num(f[1], "위도")?, num(f[2], "경도")?, num(f[3], "고도")?);
        if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
            return Err(err(format!("위경도 범위 밖: {lat} {lon}")));
        }
        let frame =
            frame_of_name(f[0]).ok_or_else(|| err(format!("이름에 번호 없음: {}", f[0])))?;
        out.push(GpsRecord {
            name: f[0].to_string(),
            frame,
            geo: Geodetic {
                lat_deg: lat,
                lon_deg: lon,
                alt,
            },
            line,
        });
    }
    Ok(out)
}

/// 카메라 폴더에서 `{cam}_{번호}.jpg` 프레임 번호 모음.
fn scan_camera(dir: &Path, cam: &str) -> Result<BTreeSet<u32>, DatasetError> {
    if !dir.is_dir() {
        return Err(DatasetError::Missing(dir.to_path_buf()));
    }
    let rd = std::fs::read_dir(dir).map_err(|e| DatasetError::Io(dir.to_path_buf(), e))?;
    let prefix = format!("{cam}_");
    let mut set = BTreeSet::new();
    for ent in rd {
        let ent = ent.map_err(|e| DatasetError::Io(dir.to_path_buf(), e))?;
        let name = ent.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(num) = name
            .strip_prefix(&prefix)
            .and_then(|r| r.strip_suffix(".jpg"))
        else {
            continue;
        };
        if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = num.parse() {
                set.insert(n);
            }
        }
    }
    Ok(set)
}

/// 사진 경로.
pub fn image_path(root: &Path, cam: &str, frame: u32) -> PathBuf {
    root.join("images")
        .join(cam)
        .join(format!("{cam}_{frame:04}.jpg"))
}

/// 데이터셋 읽기.
///
/// 세 카메라 중 하나라도 있는 프레임 가운데 가장 작은 번호 f0 에서 시작해 f0, f0+STRIDE, …
/// 를 고르고, 그중 세 카메라가 모두 있는 프레임만 위치 0, 1, … 로 매긴다.
/// 고른 위치에 GPS 가 없거나, 폴더·gps.txt 가 없거나, gps.txt 형식이 틀리면 Err.
pub fn load_dataset(root: &Path, config: DatasetConfig) -> Result<Dataset, DatasetError> {
    if config.stride == 0 {
        return Err(DatasetError::Config("STRIDE 는 1 이상".into()));
    }
    if config.span == 0 {
        return Err(DatasetError::Config("SPAN 은 1 이상".into()));
    }
    let images = root.join("images");
    if !images.is_dir() {
        return Err(DatasetError::Missing(images));
    }
    let mut sets = Vec::with_capacity(3);
    for cam in CAMERAS {
        sets.push(scan_camera(&images.join(cam), cam)?);
    }
    let gps_path = root.join("gps.txt");
    if !gps_path.is_file() {
        return Err(DatasetError::Missing(gps_path));
    }
    let text = std::fs::read_to_string(&gps_path).map_err(|e| DatasetError::Io(gps_path, e))?;
    let gps = parse_gps(&text)?;
    let origin = gps.first().ok_or(DatasetError::Empty)?.geo;
    let mut by_frame: BTreeMap<u32, &GpsRecord> = BTreeMap::new();
    for g in &gps {
        if let Some(prev) = by_frame.get(&g.frame) {
            if prev.geo != g.geo {
                return Err(DatasetError::GpsFormat {
                    line: g.line,
                    msg: format!(
                        "프레임 {} 이 {}번째 줄과 다른 값으로 중복",
                        g.frame, prev.line
                    ),
                });
            }
        } else {
            by_frame.insert(g.frame, g);
        }
    }

    let all: BTreeSet<u32> = sets.iter().flatten().copied().collect();
    let (Some(&first), Some(&last)) = (all.first(), all.last()) else {
        return Err(DatasetError::Empty);
    };
    let mut positions = Vec::new();
    for frame in (first..=last).step_by(config.stride) {
        if !sets.iter().all(|s| s.contains(&frame)) {
            continue;
        }
        let g = by_frame
            .get(&frame)
            .ok_or(DatasetError::GpsMissing { frame })?;
        positions.push(Position {
            index: positions.len(),
            frame,
            images: CAMERAS.map(|c| image_path(root, c, frame)),
            geo: g.geo,
            enu: geodetic_to_enu(&g.geo, &origin),
        });
    }
    if positions.is_empty() {
        return Err(DatasetError::Empty);
    }
    Ok(Dataset {
        root: root.to_path_buf(),
        config,
        origin,
        positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!("skylens_ds_{tag}_{}", std::process::id()));
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

    /// 프레임 0..frames 의 사진(빈 파일)과 GPS 를 만든다. `skip` 의 (카메라, 프레임) 은 빼고.
    fn make(root: &Path, frames: u32, skip: &[(usize, u32)], gps_skip: &[u32]) {
        for (ci, cam) in CAMERAS.iter().enumerate() {
            let d = root.join("images").join(cam);
            std::fs::create_dir_all(&d).unwrap();
            for f in 0..frames {
                if !skip.contains(&(ci, f)) {
                    std::fs::write(d.join(format!("{cam}_{f:04}.jpg")), b"x").unwrap();
                }
            }
        }
        let mut s = String::new();
        for f in 0..frames {
            if !gps_skip.contains(&f) {
                s += &format!("camF_{f:04}.jpg 37.0 {} 50.0\n", 127.0 + f as f64 * 1e-5);
            }
        }
        std::fs::write(root.join("gps.txt"), s).unwrap();
    }

    #[test]
    fn chunks_80_12_2() {
        let c = chunk_ranges(80, 12, 2);
        assert_eq!(
            c,
            vec![0..14, 10..26, 22..38, 34..50, 46..62, 58..74, 70..80]
        );
    }

    #[test]
    fn chunks_edges() {
        assert_eq!(chunk_ranges(0, 12, 2), Vec::<Range<usize>>::new());
        assert_eq!(chunk_ranges(5, 12, 2), vec![0..5]);
        assert_eq!(chunk_ranges(12, 12, 2), vec![0..12]);
        assert_eq!(chunk_ranges(13, 12, 2), vec![0..13, 10..13]);
        assert_eq!(chunk_ranges(10, 4, 0), vec![0..4, 4..8, 8..10]);
    }

    #[test]
    fn frame_names() {
        assert_eq!(frame_of_name("camF_0003.jpg"), Some(3));
        assert_eq!(frame_of_name("0012"), Some(12));
        assert_eq!(frame_of_name("img"), None);
    }

    #[test]
    fn stride_selection_and_enu() {
        let t = TempDir::new("stride");
        // 프레임 0..10, 프레임 6 의 camR 없음 → 0,3,9 (6 제외).
        make(&t.0, 10, &[(1, 6)], &[]);
        let ds = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        let frames: Vec<u32> = ds.positions.iter().map(|p| p.frame).collect();
        assert_eq!(frames, vec![0, 3, 9]);
        assert_eq!(ds.image_count(), 9);
        assert_eq!(ds.positions[2].index, 2);
        assert!(ds.positions[0].enu.norm() < 1e-6);
        // 경도 9e-5 도, 위도 37 도: 동쪽 약 6378137·cos37°·9e-5·π/180 ≈ 8.0 m.
        let e = ds.positions[2].enu;
        assert!((e.x - 8.0).abs() < 0.05, "{e}");
        assert!(e.y.abs() < 0.01 && e.z.abs() < 0.01, "{e}");
        // STRIDE 1 이면 0..10 중 6 빠져 9곳.
        let cfg = DatasetConfig {
            stride: 1,
            ..Default::default()
        };
        assert_eq!(load_dataset(&t.0, cfg).unwrap().positions.len(), 9);
    }

    #[test]
    fn missing_gps_is_error() {
        let t = TempDir::new("gpsmiss");
        make(&t.0, 7, &[], &[3]);
        match load_dataset(&t.0, DatasetConfig::default()) {
            Err(DatasetError::GpsMissing { frame: 3 }) => {}
            r => panic!("{r:?}"),
        }
    }

    #[test]
    fn missing_folder_and_file() {
        let t = TempDir::new("missing");
        assert!(matches!(
            load_dataset(&t.0, DatasetConfig::default()),
            Err(DatasetError::Missing(_))
        ));
        make(&t.0, 4, &[], &[]);
        std::fs::remove_dir_all(t.0.join("images/camL")).unwrap();
        assert!(matches!(
            load_dataset(&t.0, DatasetConfig::default()),
            Err(DatasetError::Missing(p)) if p.ends_with("camL")
        ));
        make(&t.0, 4, &[], &[]);
        std::fs::remove_file(t.0.join("gps.txt")).unwrap();
        assert!(matches!(
            load_dataset(&t.0, DatasetConfig::default()),
            Err(DatasetError::Missing(p)) if p.ends_with("gps.txt")
        ));
    }

    #[test]
    fn gps_format_errors_carry_line() {
        let cases = [
            ("a_0 1 2 3\n\na_1 1 2\n", 3),
            ("# c\na_0 x 2 3\n", 2),
            ("a_0 1 2 3\na_1 95 2 3\n", 2),
            ("noname 1 2 3\n", 1),
            ("a_0 1 2 3\na_0 1 2 4\n", 2),
        ];
        for (text, want) in cases {
            match parse_gps(text).and_then(|g| {
                // 중복 검사는 load 에서 하므로 여기서 흉내.
                for (i, a) in g.iter().enumerate() {
                    for b in &g[..i] {
                        if a.frame == b.frame && a.geo != b.geo {
                            return Err(DatasetError::GpsFormat {
                                line: a.line,
                                msg: String::new(),
                            });
                        }
                    }
                }
                Ok(g)
            }) {
                Err(DatasetError::GpsFormat { line, .. }) => assert_eq!(line, want, "{text}"),
                r => panic!("{text}: {r:?}"),
            }
        }
    }

    #[test]
    fn bad_gps_in_dataset() {
        let t = TempDir::new("badgps");
        make(&t.0, 4, &[], &[]);
        std::fs::write(
            t.0.join("gps.txt"),
            "camF_0000.jpg 37 127 1\ncamF_0003.jpg 37 abc 1\n",
        )
        .unwrap();
        match load_dataset(&t.0, DatasetConfig::default()) {
            Err(e @ DatasetError::GpsFormat { line: 2, .. }) => {
                assert!(e.to_string().contains("2번째 줄"), "{e}")
            }
            r => panic!("{r:?}"),
        }
    }

    #[test]
    fn zero_stride_rejected() {
        let t = TempDir::new("zero");
        make(&t.0, 3, &[], &[]);
        let cfg = DatasetConfig {
            stride: 0,
            ..Default::default()
        };
        assert!(matches!(
            load_dataset(&t.0, cfg),
            Err(DatasetError::Config(_))
        ));
    }
}
