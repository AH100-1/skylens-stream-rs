//! 입력 데이터셋 읽기: 세 카메라 사진 폴더와 GPS 텍스트, 구역 분할.
//!
//! 폴더 구조: 사진은 `images/cam{F,R,L}_{번호:04}.jpg`(평평한 구조, 합성 출력·README 형식) 또는
//! `images/cam{F,R,L}/cam{F,R,L}_{번호:04}.jpg`(카메라별 하위 폴더) 중 어느 쪽이든, 둘을 섞어도
//! 읽는다. 같은 카메라·같은 프레임 사진이 두 곳에 다 있으면 오류. 번호는 `{:04}` 표기 그대로여야
//! 한다(`camF_3.jpg`, `camF_00003.jpg` 는 오류). 그리고 `gps.txt`.
//!
//! `gps.txt` 는 `이름 위도 경도 고도` 한 줄씩이며, 이름 끝의 숫자를 프레임 번호로 본다.
//! 이름이 `cam{F,R,L}_` 로 시작하면 그 카메라(드론)의 GPS, 아니면(`0003` 등) 세 카메라 공통이다.
//! 드론 3대 편대라 같은 프레임이라도 카메라마다 GPS 가 다르다. 카메라별 줄이 공통 줄보다 우선한다.
//! 같은 (카메라, 프레임)·같은 공통 프레임에 다른 값이 두 번 나오면 오류. 빈 줄과 `#` 줄은 건너뛴다.

use std::collections::BTreeMap;
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
    /// 카메라가 빠져 건너뛰는 고른 프레임이 이 수보다 많이 연속되면 오류.
    /// 기본 2: 두 곳을 건너뛰어도 앞뒤 위치 차가 3칸이라 §3.2 시간 이웃(1..5)에 남는다.
    pub max_skip_run: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            stride: 3,
            span: 12,
            ovl: 2,
            max_skip_run: 2,
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
    /// gps.txt 에 기록이 하나도 없음.
    GpsEmpty,
    /// 선택된 프레임의 그 카메라 GPS 가 없음.
    GpsMissing {
        camera: &'static str,
        frame: u32,
    },
    /// 설정 값 오류.
    Config(String),
    /// 쓸 수 있는 위치가 없음.
    Empty,
    /// 같은 카메라·프레임의 사진이 두 경로에 있음(평평한 구조와 하위 폴더 양쪽 등).
    DuplicateImage {
        first: PathBuf,
        second: PathBuf,
    },
    /// 번호가 `{:04}` 표기가 아닌 사진 이름(`camF_3.jpg`, `camF_00003.jpg`).
    BadImageName(PathBuf),
    /// 카메라가 빠져 건너뛴 프레임이 허용보다 길게 연속됨.
    SkipRun {
        first: u32,
        last: u32,
        count: usize,
        limit: usize,
    },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Self::Missing(p) => write!(f, "파일 없음: {}", p.display()),
            Self::GpsFormat { line, msg } => write!(f, "gps.txt {line}번째 줄: {msg}"),
            Self::GpsEmpty => write!(f, "gps.txt 기록 없음"),
            Self::GpsMissing { camera, frame } => {
                write!(f, "GPS 누락: {camera} 프레임 {frame}")
            }
            Self::Config(m) => write!(f, "설정 오류: {m}"),
            Self::Empty => write!(f, "세 카메라가 모두 있는 위치가 없음"),
            Self::DuplicateImage { first, second } => write!(
                f,
                "같은 사진이 두 곳에 있음: {} , {}",
                first.display(),
                second.display()
            ),
            Self::BadImageName(p) => {
                write!(f, "사진 번호가 4자리 표기가 아님: {}", p.display())
            }
            Self::SkipRun {
                first,
                last,
                count,
                limit,
            } => write!(
                f,
                "카메라가 빠진 프레임 {count}곳 연속(프레임 {first}..={last}, 허용 {limit})"
            ),
        }
    }
}

impl std::error::Error for DatasetError {}

/// GPS 한 줄.
#[derive(Clone, Debug, PartialEq)]
pub struct GpsRecord {
    pub name: String,
    /// 이름의 카메라(`CAMERAS` 번호). `None` 이면 세 카메라 공통.
    pub camera: Option<usize>,
    pub frame: u32,
    pub geo: Geodetic,
    /// 줄 번호(1부터).
    pub line: usize,
}

/// (카메라, 프레임) → GPS 찾기표.
#[derive(Clone, Debug, Default)]
pub struct GpsTable {
    per_camera: [BTreeMap<u32, (Geodetic, usize)>; 3],
    common: BTreeMap<u32, (Geodetic, usize)>,
}

impl GpsTable {
    /// 기록을 모은다. 같은 (카메라, 프레임) 또는 같은 공통 프레임에 다른 값이 다시 나오면
    /// 뒤 줄 번호로 `GpsFormat`. 같은 값의 반복은 허용.
    pub fn build(records: &[GpsRecord]) -> Result<Self, DatasetError> {
        let mut t = Self::default();
        for r in records {
            let map = match r.camera {
                Some(c) => &mut t.per_camera[c],
                None => &mut t.common,
            };
            if let Some(&(geo, line)) = map.get(&r.frame) {
                if geo != r.geo {
                    let who = r.camera.map_or("공통", |c| CAMERAS[c]);
                    return Err(DatasetError::GpsFormat {
                        line: r.line,
                        msg: format!(
                            "{who} 프레임 {} 이 {line}번째 줄과 다른 값으로 중복",
                            r.frame
                        ),
                    });
                }
            } else {
                map.insert(r.frame, (r.geo, r.line));
            }
        }
        Ok(t)
    }

    /// 카메라 `cam`(`CAMERAS` 번호)·프레임의 GPS: 카메라별 기록, 없으면 공통 기록.
    pub fn get(&self, cam: usize, frame: u32) -> Option<Geodetic> {
        self.per_camera[cam]
            .get(&frame)
            .or_else(|| self.common.get(&frame))
            .map(|&(g, _)| g)
    }
}

/// 위치 하나: 같은 프레임의 세 카메라 사진과 카메라별 GPS.
#[derive(Clone, Debug)]
pub struct Position {
    /// 위치 번호(0부터).
    pub index: usize,
    /// 원래 프레임 번호.
    pub frame: u32,
    /// `CAMERAS` 순서의 사진 경로(실제로 읽은 파일).
    pub images: [PathBuf; 3],
    /// 위치(편대 중심) GPS: 세 카메라 GPS 의 평균.
    pub geo: Geodetic,
    /// 위치(편대 중심) 동-북-위 좌표(m): `image_enu` 의 평균. 원점은 gps.txt 의 첫 기록.
    pub enu: Vector3<f64>,
    /// `CAMERAS` 순서의 카메라(드론)별 GPS.
    pub image_geo: [Geodetic; 3],
    /// `CAMERAS` 순서의 카메라(드론)별 동-북-위 좌표(m).
    pub image_enu: [Vector3<f64>; 3],
}

/// 고른 프레임 가운데 카메라가 빠져 위치가 되지 못한 것.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkippedFrame {
    pub frame: u32,
    /// 빠진 카메라 이름.
    pub missing: Vec<&'static str>,
}

/// 읽은 데이터셋.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub root: PathBuf,
    pub config: DatasetConfig,
    pub origin: Geodetic,
    pub positions: Vec<Position>,
    /// 카메라가 빠져 건너뛴 프레임(프레임 순).
    pub skipped: Vec<SkippedFrame>,
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

    /// 건너뛴 프레임 번호.
    pub fn skipped_frames(&self) -> Vec<u32> {
        self.skipped.iter().map(|s| s.frame).collect()
    }
}

/// 구역 분할: start = 0, SPAN, 2·SPAN, … 마다 위치 [start-OVL, start+SPAN+OVL) 를 한 구역으로,
/// 범위는 0..n 으로 자른다. start ≥ 1 인 구역은 start + OVL < n 일 때만 만든다. 그렇지 않은
/// 꼬리(start 부터 남은 위치 수 ≤ OVL)는 앞 구역 끝이 이미 n 이라 앞 구역 안에 통째로 들어가므로
/// 따로 두지 않는다. 남는 구역은 모두 SPEC 범위 그대로이고, 구역 i≥1 은 앞 구역 끝 너머 위치를
/// 1개 이상 가지며, 합집합은 0..n. 80/12/2 → 7구역, 26/12/2 → [0..14, 10..26].
/// `span == 0` 이면 빈 목록.
pub fn chunk_ranges(n: usize, span: usize, ovl: usize) -> Vec<Range<usize>> {
    if span == 0 {
        return Vec::new();
    }
    (0..n)
        .step_by(span)
        .filter(|&s| s == 0 || s + ovl < n)
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

/// 이름(경로면 마지막 부분)의 카메라: `cam{F,R,L}_` 로 시작하면 그 번호.
fn camera_of_name(name: &str) -> Option<usize> {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    CAMERAS
        .iter()
        .position(|cam| base.strip_prefix(cam).is_some_and(|r| r.starts_with('_')))
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
            camera: camera_of_name(f[0]),
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

/// 폴더 하나에서 `{cam}_{번호}.jpg` 파일을 찾아 프레임 번호 → 경로 모음에 더한다.
/// 번호가 `{:04}` 표기가 아니면 `BadImageName`, 이미 있는 프레임이면 `DuplicateImage`.
fn scan_into(dir: &Path, cam: &str, out: &mut BTreeMap<u32, PathBuf>) -> Result<(), DatasetError> {
    let rd = std::fs::read_dir(dir).map_err(|e| DatasetError::Io(dir.to_path_buf(), e))?;
    let prefix = format!("{cam}_");
    let mut found: Vec<(u32, PathBuf)> = Vec::new();
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
        if num.is_empty() || !num.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let path = ent.path();
        if !path.is_file() {
            continue;
        }
        let n: u32 = match num.parse() {
            Ok(n) if format!("{n:04}") == num => n,
            _ => return Err(DatasetError::BadImageName(path)),
        };
        found.push((n, path));
    }
    // 디렉터리 순서에 기대지 않도록 정렬해 오류 메시지를 결정적으로 만든다.
    found.sort();
    for (n, path) in found {
        if let Some(first) = out.get(&n) {
            return Err(DatasetError::DuplicateImage {
                first: first.clone(),
                second: path,
            });
        }
        out.insert(n, path);
    }
    Ok(())
}

/// 한 카메라의 사진: 평평한 `images/{cam}_*.jpg` 와 하위 폴더 `images/{cam}/{cam}_*.jpg` 를 함께 본다.
/// 하위 폴더가 없고 평평한 사진도 하나 없으면 `Missing(images/{cam})`.
fn scan_camera(images: &Path, cam: &str) -> Result<BTreeMap<u32, PathBuf>, DatasetError> {
    let mut map = BTreeMap::new();
    scan_into(images, cam, &mut map)?;
    let sub = images.join(cam);
    if sub.is_dir() {
        scan_into(&sub, cam, &mut map)?;
    } else if map.is_empty() {
        return Err(DatasetError::Missing(sub));
    }
    Ok(map)
}

/// 카메라별 하위 폴더 구조에서의 사진 경로. 실제로 읽은 경로는 `Position::images`.
pub fn image_path(root: &Path, cam: &str, frame: u32) -> PathBuf {
    root.join("images")
        .join(cam)
        .join(format!("{cam}_{frame:04}.jpg"))
}

/// 데이터셋 읽기.
///
/// 세 카메라 중 하나라도 있는 프레임 가운데 가장 작은 번호 f0 에서 시작해, 실제로 있는 프레임 중
/// (f − f0) 가 STRIDE 의 배수인 것을 고른다. 그중 세 카메라가 모두 있는 프레임만 위치 0, 1, … 로
/// 매기고, 카메라가 빠진 프레임은 `Dataset::skipped` 에 남긴다. 건너뛴 프레임이
/// `max_skip_run` 보다 많이 연속되면(앞·뒤 끝 포함) `SkipRun`.
/// 고른 위치의 어느 카메라 GPS 가 없거나, 폴더·gps.txt 가 없거나, gps.txt 형식이 틀리면 Err.
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
        sets.push(scan_camera(&images, cam)?);
    }
    let gps_path = root.join("gps.txt");
    if !gps_path.is_file() {
        return Err(DatasetError::Missing(gps_path));
    }
    let text = std::fs::read_to_string(&gps_path).map_err(|e| DatasetError::Io(gps_path, e))?;
    let gps = parse_gps(&text)?;
    let origin = gps.first().ok_or(DatasetError::GpsEmpty)?.geo;
    let table = GpsTable::build(&gps)?;

    let all: std::collections::BTreeSet<u32> =
        sets.iter().flat_map(|m| m.keys()).copied().collect();
    let Some(&f0) = all.first() else {
        return Err(DatasetError::Empty);
    };
    let stride = config.stride as u64;
    let mut positions = Vec::new();
    let mut skipped = Vec::new();
    // 진행 중인 건너뜀 연속: (첫 프레임, 마지막 프레임, 개수).
    let mut run: Option<(u32, u32, usize)> = None;
    let check_run = |run: Option<(u32, u32, usize)>| match run {
        Some((first, last, count)) if count > config.max_skip_run => Err(DatasetError::SkipRun {
            first,
            last,
            count,
            limit: config.max_skip_run,
        }),
        _ => Ok(()),
    };
    for &frame in all.iter().filter(|&&f| u64::from(f - f0) % stride == 0) {
        let missing: Vec<&'static str> = (0..3)
            .filter(|&c| !sets[c].contains_key(&frame))
            .map(|c| CAMERAS[c])
            .collect();
        if !missing.is_empty() {
            run = Some(match run {
                Some((first, _, n)) => (first, frame, n + 1),
                None => (frame, frame, 1),
            });
            skipped.push(SkippedFrame { frame, missing });
            continue;
        }
        check_run(run.take())?;
        let mut geo = [origin; 3];
        for (c, g) in geo.iter_mut().enumerate() {
            *g = table.get(c, frame).ok_or(DatasetError::GpsMissing {
                camera: CAMERAS[c],
                frame,
            })?;
        }
        let image_enu = geo.map(|g| geodetic_to_enu(&g, &origin));
        positions.push(Position {
            index: positions.len(),
            frame,
            images: [0, 1, 2].map(|c| sets[c][&frame].clone()),
            geo: Geodetic {
                lat_deg: geo.iter().map(|g| g.lat_deg).sum::<f64>() / 3.0,
                lon_deg: geo.iter().map(|g| g.lon_deg).sum::<f64>() / 3.0,
                alt: geo.iter().map(|g| g.alt).sum::<f64>() / 3.0,
            },
            enu: (image_enu[0] + image_enu[1] + image_enu[2]) / 3.0,
            image_geo: geo,
            image_enu,
        });
    }
    if positions.is_empty() {
        return Err(DatasetError::Empty);
    }
    check_run(run)?;
    Ok(Dataset {
        root: root.to_path_buf(),
        config,
        origin,
        positions,
        skipped,
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
        // 카메라(드론)마다 위도를 1e-4° 씩 달리 쓴다: 북쪽 약 11.1 m 간격.
        let mut s = String::new();
        for f in 0..frames {
            if !gps_skip.contains(&f) {
                for (ci, cam) in CAMERAS.iter().enumerate() {
                    s += &format!(
                        "{cam}_{f:04}.jpg {} {} 50.0\n",
                        37.0 + ci as f64 * 1e-4,
                        127.0 + f as f64 * 1e-5
                    );
                }
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
        assert_eq!(chunk_ranges(1, 12, 2), vec![0..1]);
        assert_eq!(chunk_ranges(5, 12, 2), vec![0..5]);
        assert_eq!(chunk_ranges(12, 12, 2), vec![0..12]);
        // start 12 의 꼬리 10..13·10..14 는 앞 구역 0..13·0..14 안에 들어가므로 만들지 않는다.
        assert_eq!(chunk_ranges(13, 12, 2), vec![0..13]);
        assert_eq!(chunk_ranges(14, 12, 2), vec![0..14]);
        // 12 + 2 < 15 부터 꼬리 구역이 앞 구역 끝 너머 새 위치를 가진다.
        assert_eq!(chunk_ranges(15, 12, 2), vec![0..14, 10..15]);
        assert_eq!(chunk_ranges(17, 12, 2), vec![0..14, 10..17]);
        assert_eq!(chunk_ranges(26, 12, 2), vec![0..14, 10..26]);
        assert_eq!(chunk_ranges(27, 12, 2), vec![0..14, 10..26, 22..27]);
        // 80 곳 전후: 81·82·86 곳은 7구역(마지막 구역만 늘어남), 87 곳부터 8구역.
        let head = vec![0..14, 10..26, 22..38, 34..50, 46..62, 58..74];
        for (n, tail) in [(81, vec![70..81]), (82, vec![70..82]), (86, vec![70..86])] {
            let mut want = head.clone();
            want.extend(tail);
            assert_eq!(chunk_ranges(n, 12, 2), want, "n={n}");
        }
        let mut want = head.clone();
        want.extend([70..86, 82..87]);
        assert_eq!(chunk_ranges(87, 12, 2), want);
        // OVL 0: 꼬리가 1곳이어도 새 위치라 남긴다.
        assert_eq!(chunk_ranges(10, 4, 0), vec![0..4, 4..8, 8..10]);
        assert_eq!(chunk_ranges(9, 4, 0), vec![0..4, 4..8, 8..9]);
        // SPAN 4, OVL 2: 10 곳이면 start 8 꼬리(8+2=10)는 앞 구역 2..10 안이라 버린다.
        assert_eq!(chunk_ranges(10, 4, 2), vec![0..6, 2..10]);
        assert_eq!(chunk_ranges(11, 4, 2), vec![0..6, 2..10, 6..11]);
        assert_eq!(chunk_ranges(10, 0, 2), Vec::<Range<usize>>::new());
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
        assert_eq!(ds.skipped_frames(), vec![6]);
        assert_eq!(ds.skipped[0].missing, vec!["camR"]);
        assert_eq!(ds.image_count(), 9);
        assert_eq!(ds.positions[2].index, 2);
        assert!(ds.positions[0].image_enu[0].norm() < 1e-6);
        // 경도 9e-5 도, 위도 37 도: 동쪽 약 6378137·cos37°·9e-5·π/180 ≈ 8.0 m.
        let e = ds.positions[2].image_enu[0];
        assert!((e.x - 8.0).abs() < 0.05, "{e}");
        assert!(e.y.abs() < 0.01 && e.z.abs() < 0.01, "{e}");
        // STRIDE 1 이면 0..10 중 6 빠져 9곳.
        let cfg = DatasetConfig {
            stride: 1,
            ..Default::default()
        };
        let ds1 = load_dataset(&t.0, cfg).unwrap();
        assert_eq!(ds1.positions.len(), 9);
        assert_eq!(ds1.skipped_frames(), vec![6]);
    }

    #[test]
    fn missing_gps_is_error() {
        let t = TempDir::new("gpsmiss");
        make(&t.0, 7, &[], &[3]);
        match load_dataset(&t.0, DatasetConfig::default()) {
            Err(DatasetError::GpsMissing {
                camera: "camF",
                frame: 3,
            }) => {}
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
        ];
        for (text, want) in cases {
            match parse_gps(text) {
                Err(DatasetError::GpsFormat { line, .. }) => assert_eq!(line, want, "{text}"),
                r => panic!("{text}: {r:?}"),
            }
        }
    }

    #[test]
    fn gps_camera_of_name() {
        let g = parse_gps("camF_0001.jpg 1 2 3\ncamL_0001 1 2 3\n0001 1 2 3\ncamX_0001 1 2 3\n")
            .unwrap();
        let cams: Vec<Option<usize>> = g.iter().map(|r| r.camera).collect();
        assert_eq!(cams, vec![Some(0), Some(2), None, None]);
        assert!(g.iter().all(|r| r.frame == 1));
    }

    #[test]
    fn gps_table_duplicates() {
        // 카메라마다 다른 값: 허용. 같은 값 반복: 허용.
        let ok =
            parse_gps("camF_0000 1 2 3\ncamR_0000 1.1 2 3\ncamF_0000 1 2 3\n0000 5 5 5\n").unwrap();
        let t = GpsTable::build(&ok).unwrap();
        assert_eq!(t.get(0, 0).unwrap().lat_deg, 1.0);
        assert_eq!(t.get(1, 0).unwrap().lat_deg, 1.1);
        // camL 은 카메라별 기록이 없어 공통 줄.
        assert_eq!(t.get(2, 0).unwrap().lat_deg, 5.0);
        assert!(t.get(0, 1).is_none());
        // 같은 (카메라, 프레임)에 다른 값: 뒤 줄 번호로 오류.
        for (text, want) in [
            ("camF_0000 1 2 3\ncamR_0000 1 2 3\ncamF_0000 1 2 4\n", 3),
            ("0007 1 2 3\n0007 1 2 3.5\n", 2),
        ] {
            match GpsTable::build(&parse_gps(text).unwrap()) {
                Err(DatasetError::GpsFormat { line, .. }) => assert_eq!(line, want, "{text}"),
                r => panic!("{text}: {r:?}"),
            }
        }
    }

    #[test]
    fn duplicate_gps_in_dataset_is_error() {
        let t = TempDir::new("dupgps");
        make(&t.0, 4, &[], &[]);
        let mut text = std::fs::read_to_string(t.0.join("gps.txt")).unwrap();
        text += "camR_0003.jpg 37.5 127 50.0\n";
        std::fs::write(t.0.join("gps.txt"), &text).unwrap();
        let want = text.lines().count();
        match load_dataset(&t.0, DatasetConfig::default()) {
            Err(e @ DatasetError::GpsFormat { .. }) => {
                assert!(matches!(e, DatasetError::GpsFormat { line, .. } if line == want));
                assert!(e.to_string().contains("camR 프레임 3"), "{e}");
            }
            r => panic!("{r:?}"),
        }
    }

    #[test]
    fn per_camera_gps_accepted() {
        // 같은 프레임의 세 드론 GPS 가 위도 1e-4° 씩(북쪽 약 11.1 m) 다르다.
        let t = TempDir::new("percam");
        make(&t.0, 7, &[], &[]);
        let ds = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        assert_eq!(ds.positions.len(), 3);
        // 위도 1e-4° 의 남북 거리: 자오선 곡률 반지름 M = a(1−e²)/(1−e² sin²φ)^1.5, φ=37°.
        let (a, e2) = (6378137.0_f64, 6.694379990141317e-3_f64);
        let s2 = 37.0_f64.to_radians().sin().powi(2);
        let m = a * (1.0 - e2) / (1.0 - e2 * s2).powf(1.5);
        let d = m * 1e-4_f64.to_radians();
        assert!((d - 11.09).abs() < 0.02, "{d}");
        for p in &ds.positions {
            assert_eq!(p.image_geo[1].lat_deg, 37.0001);
            assert!((p.geo.lat_deg - 37.0001).abs() < 1e-12);
            for (i, j, k) in [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)] {
                let got = (p.image_enu[j] - p.image_enu[i]).norm();
                assert!((got - k * d).abs() < 0.1, "위치 {} {i}-{j}: {got}", p.index);
            }
            // 위치 좌표는 세 드론 평균 = 가운데(camR) 드론 자리.
            assert!((p.enu - p.image_enu[1]).norm() < 0.01, "{}", p.enu);
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

    /// 하위 폴더 사진을 평평한 위치로 옮긴다(카메라 하나만, 또는 모두).
    fn flatten(root: &Path, cams: &[&str]) {
        for cam in cams {
            let d = root.join("images").join(cam);
            for e in std::fs::read_dir(&d).unwrap() {
                let e = e.unwrap();
                std::fs::rename(e.path(), root.join("images").join(e.file_name())).unwrap();
            }
            std::fs::remove_dir(&d).unwrap();
        }
    }

    #[test]
    fn flat_and_mixed_layouts_read_same() {
        // 프레임 0..9, STRIDE 3 → 0,3,6 의 3곳.
        let t = TempDir::new("flat");
        make(&t.0, 9, &[], &[]);
        // 읽은 직후에 경로가 실제 파일인지 본다(다음 단계에서 파일을 옮기므로).
        let check = |d: &Dataset| {
            assert_eq!(
                d.positions.iter().map(|p| p.frame).collect::<Vec<_>>(),
                vec![0, 3, 6]
            );
            assert_eq!(d.image_count(), 9);
            for p in &d.positions {
                for (c, img) in p.images.iter().enumerate() {
                    assert!(img.is_file(), "{}", img.display());
                    let want = format!("{}_{:04}.jpg", CAMERAS[c], p.frame);
                    assert_eq!(img.file_name().unwrap().to_str().unwrap(), want);
                }
            }
        };
        let sub = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        check(&sub);
        flatten(&t.0, &["camR"]);
        let mixed = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        check(&mixed);
        flatten(&t.0, &["camF", "camL"]);
        let flat = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        check(&flat);
        assert_eq!(
            mixed.positions[1].images[1],
            t.0.join("images/camR_0003.jpg")
        );
        assert_eq!(
            mixed.positions[1].images[0],
            t.0.join("images/camF/camF_0003.jpg")
        );
        assert_eq!(
            flat.positions[2].images[2],
            t.0.join("images/camL_0006.jpg")
        );
    }

    #[test]
    fn same_image_in_both_layouts_is_error() {
        let t = TempDir::new("dup");
        make(&t.0, 4, &[], &[]);
        std::fs::write(t.0.join("images/camR_0002.jpg"), b"x").unwrap();
        match load_dataset(&t.0, DatasetConfig::default()) {
            Err(DatasetError::DuplicateImage { first, second }) => {
                let mut v = [first, second];
                v.sort();
                assert_eq!(
                    v,
                    [
                        t.0.join("images/camR/camR_0002.jpg"),
                        t.0.join("images/camR_0002.jpg")
                    ]
                );
            }
            other => panic!("중복 오류가 아님: {other:?}"),
        }
    }

    #[test]
    fn flat_layout_missing_camera() {
        // 평평한 구조에서 camL 사진이 하나도 없으면 Missing(images/camL).
        let t = TempDir::new("flatmiss");
        make(&t.0, 4, &[], &[]);
        flatten(&t.0, &["camF", "camR"]);
        std::fs::remove_dir_all(t.0.join("images/camL")).unwrap();
        assert!(matches!(
            load_dataset(&t.0, DatasetConfig::default()),
            Err(DatasetError::Missing(p)) if p.ends_with("camL")
        ));
    }

    #[test]
    fn non_four_digit_names_rejected() {
        // `camF_3.jpg`·`camF_00003.jpg` 만 있는 폴더: 없는 `camF_0003.jpg` 를 만들지 않고 Err.
        for bad in ["camF_3.jpg", "camF_00003.jpg", "camF_012.jpg"] {
            let t = TempDir::new("digits");
            make(&t.0, 4, &[(0, 3)], &[]);
            let p = t.0.join("images/camF").join(bad);
            std::fs::write(&p, b"x").unwrap();
            match load_dataset(&t.0, DatasetConfig::default()) {
                Err(e @ DatasetError::BadImageName(_)) => {
                    assert!(matches!(&e, DatasetError::BadImageName(q) if *q == p));
                    assert!(e.to_string().contains(bad), "{e}");
                }
                r => panic!("{bad}: {r:?}"),
            }
        }
        // 4자리 넘는 번호는 `{:04}` 표기 그대로(앞자리 0 없음)면 받는다. 모든 경로가 실제 파일.
        let t = TempDir::new("digits5");
        make(&t.0, 1, &[], &[]);
        let mut gps = std::fs::read_to_string(t.0.join("gps.txt")).unwrap();
        for cam in CAMERAS {
            std::fs::write(t.0.join(format!("images/{cam}/{cam}_12345.jpg")), b"x").unwrap();
            gps += &format!("{cam}_12345.jpg 37 127.1 50\n");
        }
        std::fs::write(t.0.join("gps.txt"), gps).unwrap();
        let cfg = DatasetConfig {
            stride: 12345,
            ..Default::default()
        };
        let ds = load_dataset(&t.0, cfg).unwrap();
        assert_eq!(
            ds.positions.iter().map(|p| p.frame).collect::<Vec<_>>(),
            vec![0, 12345]
        );
        assert!(ds
            .positions
            .iter()
            .flat_map(|p| &p.images)
            .all(|q| q.exists()));
    }

    #[test]
    fn missing_camera_frames_reported_or_rejected() {
        // camR 6 만 빠짐: 위치 0,3,9,…,39 (13곳), skipped == [6].
        let t = TempDir::new("skip1");
        make(&t.0, 40, &[(1, 6)], &[]);
        let ds = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        assert_eq!(ds.positions.len(), 13);
        assert_eq!(ds.skipped_frames(), vec![6]);
        assert_eq!(
            ds.skipped,
            vec![SkippedFrame {
                frame: 6,
                missing: vec!["camR"]
            }]
        );
        // camL 20..39 빠짐: 고른 프레임 21,24,…,39 의 7곳이 연속으로 빠짐 → 기본(2)에서 오류.
        let t = TempDir::new("skiptail");
        let skip: Vec<(usize, u32)> = (20..40).map(|f| (2, f)).collect();
        make(&t.0, 40, &skip, &[]);
        match load_dataset(&t.0, DatasetConfig::default()) {
            Err(DatasetError::SkipRun {
                first: 21,
                last: 39,
                count: 7,
                limit: 2,
            }) => {}
            r => panic!("{r:?}"),
        }
        // 허용을 늘리면 읽되 빠진 7곳을 모두 보고한다.
        let cfg = DatasetConfig {
            max_skip_run: 7,
            ..Default::default()
        };
        let ds = load_dataset(&t.0, cfg).unwrap();
        assert_eq!(ds.positions.len(), 7);
        assert_eq!(ds.skipped_frames(), vec![21, 24, 27, 30, 33, 36, 39]);
        // 연속 2곳(허용 2)은 통과, 3곳은 오류. 앞 끝의 연속도 센다.
        let t = TempDir::new("skiprun");
        make(&t.0, 40, &[(0, 0), (1, 3), (2, 12), (2, 15)], &[]);
        let ds = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        assert_eq!(ds.skipped_frames(), vec![0, 3, 12, 15]);
        assert_eq!(ds.positions[0].frame, 6);
        let t = TempDir::new("skiprun3");
        make(&t.0, 40, &[(0, 0), (1, 3), (2, 6)], &[]);
        assert!(matches!(
            load_dataset(&t.0, DatasetConfig::default()),
            Err(DatasetError::SkipRun {
                first: 0,
                last: 6,
                count: 3,
                ..
            })
        ));
    }

    #[test]
    fn chunk_tail_always_adds_new_positions() {
        for n in 1..=200 {
            for span in [1, 4, 12] {
                for ovl in [0, 2, 5] {
                    let c = chunk_ranges(n, span, ovl);
                    let ctx = format!("{n} {span} {ovl}: {c:?}");
                    assert_eq!(c[0].start, 0, "{ctx}");
                    assert_eq!(c.last().unwrap().end, n, "{ctx}");
                    for w in c.windows(2) {
                        // 빈틈 없이 이어지고, 뒤 구역은 앞 구역 끝 너머 위치를 1개 이상 가진다.
                        assert!(w[1].start <= w[0].end, "{ctx}");
                        assert!(w[1].end > w[0].end, "{ctx}");
                        // OVL ≥ SPAN 이면 앞쪽 여러 구역이 0 에서 시작할 수 있다(SPEC 범위 그대로).
                        assert!(w[1].start >= w[0].start, "{ctx}");
                    }
                    // 남은 구역은 모두 SPEC §3.5 범위 그대로(i 번째 구역의 start = i·SPAN).
                    for (i, r) in c.iter().enumerate() {
                        let s = i * span;
                        assert_eq!(*r, s.saturating_sub(ovl)..(s + span + ovl).min(n), "{ctx}");
                    }
                    // 버려진 start 는 앞 구역 안에 통째로 들어가는 것뿐이다.
                    let dropped = n.div_ceil(span) - c.len();
                    for k in c.len()..c.len() + dropped {
                        let s = k * span;
                        let prev = c.last().unwrap();
                        assert!(
                            s.saturating_sub(ovl) >= prev.start && n <= prev.end,
                            "{ctx}"
                        );
                    }
                }
            }
        }
        // OVL ≥ SPAN 이라도 같은 구역이 두 번 나오지 않는다.
        assert_eq!(chunk_ranges(2, 1, 5), vec![0..2]);
    }

    #[test]
    fn empty_gps_message() {
        let t = TempDir::new("emptygps");
        make(&t.0, 4, &[], &[]);
        std::fs::write(t.0.join("gps.txt"), "# 기록 없음\n\n").unwrap();
        match load_dataset(&t.0, DatasetConfig::default()) {
            Err(e @ DatasetError::GpsEmpty) => assert_eq!(e.to_string(), "gps.txt 기록 없음"),
            r => panic!("{r:?}"),
        }
    }

    #[test]
    fn huge_frame_gap_is_fast() {
        // 프레임 0 과 4000000000 만: 공백을 한 칸씩 돌지 않고 있는 프레임에서 고른다.
        let t = TempDir::new("gap");
        make(&t.0, 1, &[], &[]);
        let mut gps = std::fs::read_to_string(t.0.join("gps.txt")).unwrap();
        for cam in CAMERAS {
            std::fs::write(t.0.join(format!("images/{cam}/{cam}_4000000000.jpg")), b"x").unwrap();
            gps += &format!("{cam}_4000000000.jpg 37 127.1 50\n");
        }
        std::fs::write(t.0.join("gps.txt"), gps).unwrap();
        let t0 = std::time::Instant::now();
        let cfg = DatasetConfig {
            stride: 1,
            ..Default::default()
        };
        let ds = load_dataset(&t.0, cfg).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        assert!(dt < 1.0, "{dt} s");
        assert_eq!(
            ds.positions.iter().map(|p| p.frame).collect::<Vec<_>>(),
            vec![0, 4000000000]
        );
        // STRIDE 3: 4000000000 − 0 은 3 의 배수가 아니라 빠진다(3·1333333333 = 3999999999).
        let ds3 = load_dataset(&t.0, DatasetConfig::default()).unwrap();
        assert_eq!(ds3.positions.len(), 1);
    }
}
