//! 점군 PLY 입출력 (SPEC §2).
//!
//! 쓰기 형식은 고정: `binary_little_endian 1.0`, vertex 속성
//! x y z nx ny nz (float32) + red green blue (uint8).
//! 읽기는 vertex 원소의 스칼라 속성을 이름으로 찾으므로 속성 순서·추가 속성에 견딘다.
//! 법선·색이 없으면 0 으로 채운다.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// 점 하나: 위치, 법선, 색.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointRecord {
    pub xyz: [f32; 3],
    pub normal: [f32; 3],
    pub rgb: [u8; 3],
}

/// 점군.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloud {
    pub points: Vec<PointRecord>,
}

impl PointCloud {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn has_nan(&self) -> bool {
        self.points
            .iter()
            .any(|p| p.xyz.iter().chain(p.normal.iter()).any(|v| !v.is_finite()))
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// SPEC 형식으로 쓴다.
pub fn write_ply<W: Write>(w: &mut W, cloud: &PointCloud) -> io::Result<()> {
    write!(
        w,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property float nx\nproperty float ny\nproperty float nz\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        cloud.len()
    )?;
    let mut buf = Vec::with_capacity(cloud.len() * 27);
    for p in &cloud.points {
        for v in p.xyz.iter().chain(p.normal.iter()) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&p.rgb);
    }
    w.write_all(&buf)
}

pub fn write_ply_file(path: impl AsRef<Path>, cloud: &PointCloud) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_ply(&mut w, cloud)?;
    w.flush()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn read(self, b: &[u8]) -> f64 {
        match self {
            Self::I8 => b[0] as i8 as f64,
            Self::U8 => b[0] as f64,
            Self::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            Self::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            Self::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Self::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Self::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Self::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()),
        }
    }
}

/// 이진 리틀엔디언 PLY 의 vertex 원소를 읽는다. vertex 가 첫 원소여야 한다.
pub fn read_ply<R: Read>(r: R) -> io::Result<PointCloud> {
    let mut r = BufReader::new(r);
    let mut line = String::new();
    let mut next_line = |r: &mut BufReader<R>| -> io::Result<String> {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(invalid("헤더가 끝나기 전에 파일이 끝남"));
        }
        Ok(line.trim().to_string())
    };

    if next_line(&mut r)? != "ply" {
        return Err(invalid("PLY 매직 없음"));
    }
    let mut count: Option<usize> = None;
    let mut in_vertex = false;
    let mut props: Vec<(String, Scalar)> = Vec::new();
    loop {
        let l = next_line(&mut r)?;
        let tok: Vec<&str> = l.split_whitespace().collect();
        match tok.as_slice() {
            ["end_header"] => break,
            ["format", fmt, _] => {
                if *fmt != "binary_little_endian" {
                    return Err(invalid(format!("지원하지 않는 형식: {fmt}")));
                }
            }
            ["comment", ..] | ["obj_info", ..] => {}
            ["element", name, n] => {
                if count.is_some() && in_vertex {
                    // vertex 뒤의 다른 원소는 무시한다 (데이터는 vertex 만 읽음).
                    in_vertex = false;
                } else if *name == "vertex" {
                    if count.is_some() {
                        return Err(invalid("vertex 원소가 둘 이상"));
                    }
                    count = Some(n.parse().map_err(|_| invalid("vertex 개수 해석 실패"))?);
                    in_vertex = true;
                } else if count.is_none() {
                    return Err(invalid("vertex 가 첫 원소가 아님"));
                }
            }
            ["property", "list", ..] if in_vertex => {
                return Err(invalid("vertex 의 list 속성은 지원하지 않음"));
            }
            ["property", ty, name] if in_vertex => {
                let s = Scalar::parse(ty).ok_or_else(|| invalid(format!("모르는 타입: {ty}")))?;
                props.push((name.to_string(), s));
            }
            ["property", ..] => {}
            _ => return Err(invalid(format!("해석할 수 없는 헤더 줄: {l}"))),
        }
    }
    let count = count.ok_or_else(|| invalid("vertex 원소 없음"))?;

    let mut offsets = Vec::with_capacity(props.len());
    let mut stride = 0;
    for (_, s) in &props {
        offsets.push(stride);
        stride += s.size();
    }
    let find = |name: &str| props.iter().position(|(n, _)| n == name);
    let xyz_idx = ["x", "y", "z"].map(find);
    if xyz_idx.iter().any(Option::is_none) {
        return Err(invalid("x/y/z 속성 없음"));
    }
    let n_idx = ["nx", "ny", "nz"].map(find);
    let c_idx = ["red", "green", "blue"].map(find);

    // 헤더의 개수를 믿고 한 번에 할당하지 않는다: 크기를 넘침 검사하고,
    // 실제로 읽힌 만큼만 버퍼를 키운다(잘린 파일·거대한 개수 → InvalidData).
    let total = count
        .checked_mul(stride)
        .ok_or_else(|| invalid(format!("vertex 개수가 너무 큼: {count}")))?;
    let mut data = Vec::with_capacity(total.min(1 << 24));
    let got = r.by_ref().take(total as u64).read_to_end(&mut data)?;
    if got < total {
        return Err(invalid(format!(
            "vertex 데이터가 잘림: {total} 바이트 필요, {got} 바이트 있음"
        )));
    }
    let get = |row: &[u8], i: usize| props[i].1.read(&row[offsets[i]..]);
    let points = data
        .chunks_exact(stride.max(1))
        .take(count)
        .map(|row| {
            let mut p = PointRecord {
                xyz: [0.0; 3],
                normal: [0.0; 3],
                rgb: [0; 3],
            };
            for k in 0..3 {
                p.xyz[k] = get(row, xyz_idx[k].unwrap()) as f32;
                if let Some(i) = n_idx[k] {
                    p.normal[k] = get(row, i) as f32;
                }
                if let Some(i) = c_idx[k] {
                    p.rgb[k] = get(row, i).clamp(0.0, 255.0) as u8;
                }
            }
            p
        })
        .collect();
    Ok(PointCloud { points })
}

pub fn read_ply_file(path: impl AsRef<Path>) -> io::Result<PointCloud> {
    read_ply(File::open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 결정적 의사 난수(xorshift).
    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 24) as f32
    }

    fn sample_cloud(n: usize) -> PointCloud {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let points = (0..n)
            .map(|_| {
                let xyz = [
                    rng(&mut s) * 200.0 - 100.0,
                    rng(&mut s) * 200.0 - 100.0,
                    rng(&mut s) * 40.0,
                ];
                let v = [rng(&mut s) - 0.5, rng(&mut s) - 0.5, rng(&mut s) + 0.1];
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                PointRecord {
                    xyz,
                    normal: v.map(|c| c / len),
                    rgb: [
                        (rng(&mut s) * 255.0) as u8,
                        (rng(&mut s) * 255.0) as u8,
                        (rng(&mut s) * 255.0) as u8,
                    ],
                }
            })
            .collect();
        PointCloud { points }
    }

    #[test]
    fn roundtrip_bit_exact() {
        let cloud = sample_cloud(10_000);
        let mut buf = Vec::new();
        write_ply(&mut buf, &cloud).unwrap();
        let back = read_ply(&buf[..]).unwrap();
        assert_eq!(back.len(), 10_000);
        // float32 를 그대로 쓰고 읽으므로 비트 단위로 같아야 한다.
        assert_eq!(back, cloud);
        assert!(!back.has_nan());
    }

    #[test]
    fn file_size_matches_spec_layout() {
        let cloud = sample_cloud(1234);
        let mut buf = Vec::new();
        write_ply(&mut buf, &cloud).unwrap();
        let header_end = buf.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        // 점당 float32×6 + uint8×3 = 27바이트.
        assert_eq!(buf.len() - header_end, 1234 * 27);
    }

    #[test]
    fn empty_cloud_roundtrip() {
        let cloud = PointCloud::default();
        let mut buf = Vec::new();
        write_ply(&mut buf, &cloud).unwrap();
        assert_eq!(read_ply(&buf[..]).unwrap(), cloud);
    }

    #[test]
    fn file_roundtrip() {
        let cloud = sample_cloud(500);
        let dir = std::env::temp_dir().join(format!("skylens_ply_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.ply");
        write_ply_file(&path, &cloud).unwrap();
        assert_eq!(read_ply_file(&path).unwrap(), cloud);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reads_xyz_only_double_with_reordered_props() {
        // 속성 순서가 다르고 타입이 double 이며 법선·색이 없는 파일.
        let mut buf = b"ply\nformat binary_little_endian 1.0\ncomment test\nelement vertex 2\n\
property double z\nproperty double x\nproperty double y\nend_header\n"
            .to_vec();
        for v in [3.0f64, 1.0, 2.0, -6.0, -4.0, -5.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let c = read_ply(&buf[..]).unwrap();
        assert_eq!(c.points[0].xyz, [1.0, 2.0, 3.0]);
        assert_eq!(c.points[1].xyz, [-4.0, -5.0, -6.0]);
        assert_eq!(c.points[1].normal, [0.0; 3]);
    }

    #[test]
    fn rejects_ascii_and_truncated() {
        let ascii = b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n";
        assert!(read_ply(&ascii[..]).is_err());
        let cloud = sample_cloud(10);
        let mut buf = Vec::new();
        write_ply(&mut buf, &cloud).unwrap();
        buf.truncate(buf.len() - 5);
        assert!(read_ply(&buf[..]).is_err());
    }

    fn header_only(count: &str, props: &str) -> Vec<u8> {
        format!("ply\nformat binary_little_endian 1.0\nelement vertex {count}\n{props}end_header\n")
            .into_bytes()
    }

    #[test]
    fn rejects_huge_vertex_count_without_allocating() {
        let xyz = "property float x\nproperty float y\nproperty float z\n";
        // 120 GB 를 요구하는 헤더 + 데이터 없음.
        let e = read_ply(&header_only("10000000000", xyz)[..]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("잘림"), "{e}");
        // count * 12 가 usize 를 넘는 헤더.
        let e = read_ply(&header_only("683212743470724134000", xyz)[..]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        let e = read_ply(&header_only("1537228672809129302", xyz)[..]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("너무 큼"), "{e}");
    }

    #[test]
    fn rejects_short_vertex_data() {
        let xyz = "property float x\nproperty float y\nproperty float z\n";
        let mut buf = header_only("3", xyz);
        buf.extend_from_slice(&[0u8; 35]); // 36 바이트 필요
        let e = read_ply(&buf[..]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        buf.push(0);
        assert_eq!(read_ply(&buf[..]).unwrap().len(), 3);
    }

    #[test]
    fn detects_nan() {
        let mut cloud = sample_cloud(3);
        assert!(!cloud.has_nan());
        cloud.points[1].xyz[2] = f32::NAN;
        assert!(cloud.has_nan());
    }
}
