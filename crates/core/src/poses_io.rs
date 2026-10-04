//! 카메라 포즈 파일 `poses/{preview,refined}_{k:02}.json` 읽기·쓰기.
//!
//! 한 파일은 구역 하나의 사진들이다. 사진마다 이름, 회전 쿼터니언 `[w, x, y, z]`(세계 → 카메라),
//! 카메라 중심(동-북-위 m)을 가진다. 좌표계는 같은 구역 점군 파일(`preview/`, `refined/`)과 같다.
//! 내부 파라미터는 파일 머리에 한 번 쓴다(화소 단위, 왜곡 없음).

use std::path::Path;

use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion};

use crate::align::Similarity;
use crate::camera::{Intrinsics, Pose};
use crate::stream::Json;

/// 사진 하나의 포즈.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseEntry {
    pub name: String,
    /// 세계 → 카메라 회전, `[w, x, y, z]`, w ≥ 0.
    pub quat_wxyz: [f64; 4],
    /// 카메라 중심, 동-북-위 m.
    pub center: [f64; 3],
}

/// 포즈 파일 내용.
#[derive(Clone, Debug, PartialEq)]
pub struct PosesFile {
    /// (fx, fy, cx, cy, 너비, 높이).
    pub intrinsics: (f64, f64, f64, f64, u32, u32),
    /// 닮음 변환으로 정렬된 좌표인가(초벌에서 정렬 실패면 false).
    pub aligned: bool,
    pub poses: Vec<PoseEntry>,
}

/// 닮음 변환 `x' = s·R·x + t` 를 받은 좌표계에서의 포즈: 회전 `R_wc·Rᵀ`, 중심 `S(C)`.
pub fn transform_pose(sim: &Similarity, p: &Pose) -> Pose {
    let c = sim.apply_point(&p.center().coords);
    Pose::from_center(p.rotation * sim.r.inverse(), &c.into())
}

pub fn entry(name: &str, p: &Pose) -> PoseEntry {
    let q = UnitQuaternion::from_rotation_matrix(&p.rotation);
    let mut w = [q.w, q.i, q.j, q.k];
    if w[0] < 0.0 {
        w = w.map(|x| -x);
    }
    let c = p.center();
    PoseEntry {
        name: name.to_string(),
        quat_wxyz: w,
        center: [c.x, c.y, c.z],
    }
}

/// 쿼터니언 → 회전 행렬(세계 → 카메라).
pub fn rotation_of(e: &PoseEntry) -> Rotation3<f64> {
    let [w, x, y, z] = e.quat_wxyz;
    let q = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
    let m: Matrix3<f64> = q.to_rotation_matrix().into_inner();
    Rotation3::from_matrix_unchecked(m)
}

impl PosesFile {
    pub fn new(k: &Intrinsics, aligned: bool, poses: Vec<PoseEntry>) -> Self {
        Self {
            intrinsics: (k.fx, k.fy, k.cx, k.cy, k.width, k.height),
            aligned,
            poses,
        }
    }

    pub fn to_json(&self) -> String {
        let (fx, fy, cx, cy, w, h) = self.intrinsics;
        let rows: Vec<String> = self
            .poses
            .iter()
            .map(|e| {
                let n = e.name.replace('\\', "/").replace('"', "'");
                format!(
                    "    {{\"image\": \"{n}\", \"quat_wxyz\": [{:?}, {:?}, {:?}, {:?}], \"center_enu\": [{:?}, {:?}, {:?}]}}",
                    e.quat_wxyz[0], e.quat_wxyz[1], e.quat_wxyz[2], e.quat_wxyz[3],
                    e.center[0], e.center[1], e.center[2]
                )
            })
            .collect();
        format!(
            "{{\n  \"rotation\": \"world_to_camera\",\n  \"aligned\": {},\n  \"intrinsics\": {{\"fx\": {fx:?}, \"fy\": {fy:?}, \"cx\": {cx:?}, \"cy\": {cy:?}, \"width\": {w}, \"height\": {h}}},\n  \"poses\": [\n{}\n  ]\n}}\n",
            self.aligned,
            rows.join(",\n")
        )
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let j = Json::parse(text)?;
        let num = |o: &Json, k: &str| match o.get(k) {
            Some(Json::Num(v)) => Ok(*v),
            _ => Err(format!("숫자 항목 없음: {k}")),
        };
        let arr = |o: &Json, k: &str, n: usize| -> Result<Vec<f64>, String> {
            match o.get(k) {
                Some(Json::Arr(a)) if a.len() == n => a
                    .iter()
                    .map(|v| match v {
                        Json::Num(x) => Ok(*x),
                        _ => Err(format!("{k}: 숫자가 아님")),
                    })
                    .collect(),
                _ => Err(format!("{k}: 길이 {n} 배열이 아님")),
            }
        };
        let ki = j.get("intrinsics").ok_or("intrinsics 없음")?;
        let intrinsics = (
            num(ki, "fx")?,
            num(ki, "fy")?,
            num(ki, "cx")?,
            num(ki, "cy")?,
            num(ki, "width")? as u32,
            num(ki, "height")? as u32,
        );
        let aligned = matches!(j.get("aligned"), Some(Json::Bool(true)));
        let Some(Json::Arr(list)) = j.get("poses") else {
            return Err("poses 배열 없음".into());
        };
        let mut poses = Vec::new();
        for o in list {
            let Some(Json::Str(name)) = o.get("image") else {
                return Err("image 이름 없음".into());
            };
            let q = arr(o, "quat_wxyz", 4)?;
            let c = arr(o, "center_enu", 3)?;
            poses.push(PoseEntry {
                name: name.clone(),
                quat_wxyz: [q[0], q[1], q[2], q[3]],
                center: [c[0], c[1], c[2]],
            });
        }
        Ok(Self {
            intrinsics,
            aligned,
            poses,
        })
    }
}

/// `poses/` 폴더를 만들고 `{kind}_{k:02}.json` 을 쓴다.
pub fn write_poses_file(
    out: &Path,
    kind: &str,
    region: usize,
    file: &PosesFile,
) -> Result<(), String> {
    let dir = out.join("poses");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{kind}_{region:02}.json"));
    std::fs::write(&path, file.to_json()).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};

    fn pose() -> Pose {
        let r = Rotation3::from_euler_angles(0.3, -1.2, 2.5);
        Pose::from_center(r, &Point3::new(10.0, -4.5, 80.25))
    }

    #[test]
    fn json_roundtrip_keeps_rotation_and_center() {
        let p = pose();
        let k = Intrinsics::from_hfov(320, 180, 65f64.to_radians());
        let f = PosesFile::new(&k, true, vec![entry("a_01", &p)]);
        let back = PosesFile::from_json(&f.to_json()).unwrap();
        assert_eq!(back, f);
        let r = rotation_of(&back.poses[0]);
        assert!((r.matrix() - p.rotation.matrix()).norm() < 1e-12);
        assert!(back.poses[0].quat_wxyz[0] >= 0.0);
    }

    #[test]
    fn transform_matches_point_mapping() {
        let sim = Similarity {
            s: 1.7,
            r: Rotation3::from_euler_angles(0.1, 0.2, -0.9),
            t: Vector3::new(3.0, 2.0, -1.0),
        };
        let p = pose();
        let q = transform_pose(&sim, &p);
        // 세계 점 x 와 변환된 점 S(x) 는 같은 방향·(스케일 배) 깊이로 보인다.
        let x = Point3::new(1.0, 2.0, 3.0);
        let a = p.transform(&x);
        let b = q.transform(&Point3::from(sim.apply_point(&x.coords)));
        assert!((b - a * sim.s).norm() < 1e-9);
        assert!((q.center().coords - sim.apply_point(&p.center().coords)).norm() < 1e-9);
    }

    #[test]
    fn bad_json_is_error() {
        assert!(PosesFile::from_json("{}").is_err());
        assert!(PosesFile::from_json("{\"intrinsics\": {}, \"poses\": []}").is_err());
    }
}
