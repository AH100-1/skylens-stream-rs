//! 구역 순서 처리의 보조 논리: 이미 내보낸 정밀 구역을 최신 정밀 모델 좌표계로 다시 맞추는
//! 닮음 변환(공유 3D 점 대응)과 등록 현황 표.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::align::Similarity;
use crate::camera::Pose;
use crate::math::Vector3;
use crate::ply::{write_ply_file, PointCloud};
use crate::progressive::{cross_align, overlap_window};
use crate::stream::{
    apply_cloud, decimate, remove_ghosts, RadiusIndex, Region, Track, DECIMATE_EVERY,
    GHOST_RADIUS_M,
};

/// 실행 방식. `sequential` 이면 구역마다 정밀(BA)이 끝난 뒤에 다음 구역을 받는다(비교용 기준 흐름).
/// `anchor` 이면 새 구역 등록이 직전까지의 최신 정밀 모델 위에서 이뤄진다: 겹침 위치 카메라 포즈를
/// 정밀 값으로 고정하고 공유 3D 점으로 같은 좌표계에 붙인다(기본 켬).
#[derive(Clone, Copy, Debug)]
pub struct StreamOptions {
    pub sequential: bool,
    pub anchor: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            sequential: false,
            anchor: true,
        }
    }
}

/// 최신 정밀 모델에 새 구역을 붙이는 계획: 닮음 변환과 고정할 카메라(구역 안 사진 번호, 정밀 포즈).
pub struct AnchorPlan {
    pub sim: Similarity,
    pub fixed: Vec<(usize, Pose)>,
    pub pairs: usize,
    pub median_m: f64,
}

/// 새 구역의 초벌 트랙 `ta` 를 정밀 모델(`prev` 트랙, 사진별 포즈 `prev_poses`) 에 붙일 계획을 세운다.
/// 변환은 겹침 위치 범위의 공유 3D 점 대응(5회 트리밍)으로, 고정 카메라는 등록된 사진 중 정밀 포즈가
/// 있는 것. 고정 카메라가 2 대 미만이거나 점 대응이 모자라면 이유를 담은 `Err`.
pub fn plan_anchor(
    ta: &[Track],
    prev: &[Track],
    prev_poses: &HashMap<usize, Pose>,
    regions: (&Region, &Region),
    gids: &[usize],
    registered: &[bool],
) -> Result<AnchorPlan, String> {
    let win = overlap_window(regions.0, regions.1);
    let (sim, pairs, median_m) = cross_align(ta, prev, win).ok_or_else(|| {
        format!(
            "겹침 위치 {}..{} 의 공유 3D 점 대응이 모자라거나 퇴화함",
            win.0, win.1
        )
    })?;
    let fixed: Vec<(usize, Pose)> = gids
        .iter()
        .enumerate()
        .filter(|&(a, g)| registered[a] && g / 3 >= win.0 && g / 3 < win.1)
        .filter_map(|(a, g)| prev_poses.get(g).map(|p| (a, *p)))
        .collect();
    if fixed.len() < 2 {
        return Err(format!(
            "겹침 위치 {}..{} 에서 등록된 정밀 카메라 {} 대 < 2 (공유 점 {pairs} 쌍)",
            win.0,
            win.1,
            fixed.len()
        ));
    }
    Ok(AnchorPlan {
        sim,
        fixed,
        pairs,
        median_m,
    })
}

/// 카메라 중심 `centers`(사진 번호 → 중심)에 닮음 변환 `sim`(없으면 항등)을 적용한 뒤 정답 `truth` 와의 중앙 거리(m).
pub fn center_error_median(
    centers: &BTreeMap<usize, [f64; 3]>,
    sim: Option<&Similarity>,
    truth: impl Fn(usize) -> Vector3<f64>,
) -> Option<f64> {
    let mut e: Vec<f64> = centers
        .iter()
        .map(|(g, c)| {
            let v = Vector3::new(c[0], c[1], c[2]);
            let v = sim.map_or(v, |s| s.apply_point(&v));
            (v - truth(*g)).norm()
        })
        .collect();
    if e.is_empty() {
        return None;
    }
    e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(e[e.len() / 2])
}

/// 한 시점의 구역 상태(스냅샷 합성용 빌림).
pub struct LiveRegion<'a> {
    pub region: Region,
    pub coarse: &'a PointCloud,
    pub coarse_sim: Option<Similarity>,
    pub refined: Option<&'a PointCloud>,
    pub refined_sim: Option<Similarity>,
}

/// 흐름 중 한 사건(manifest `events` 로 남는다).
#[derive(Clone, Debug)]
pub struct LiveEvent {
    pub seq: usize,
    pub secs: f64,
    pub kind: String,
    pub region: usize,
    pub snapshot: Option<String>,
    pub points: usize,
}

/// 사건 순서와 시점별 스냅샷(`snapshots/live/ev_NNN_*.ply`)을 모은다.
pub struct LiveLog {
    out: PathBuf,
    pub events: Vec<LiveEvent>,
}

impl LiveLog {
    pub fn new(out: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(out.join("snapshots/live"))?;
        Ok(Self {
            out: out.to_path_buf(),
            events: Vec::new(),
        })
    }

    /// 스냅샷 없이 사건만 적는다.
    pub fn note(&mut self, secs: f64, kind: &str, region: usize) {
        let seq = self.events.len();
        self.events.push(LiveEvent {
            seq,
            secs,
            kind: kind.to_string(),
            region,
            snapshot: None,
            points: 0,
        });
    }

    /// 사건을 적고 지금 상태(정밀 구역 + 정밀이 아직 없는 초벌 구역, 잔상 1.5 m 걸러냄)를 스냅샷으로 쓴다.
    pub fn snapshot(
        &mut self,
        secs: f64,
        kind: &str,
        region: usize,
        state: &[LiveRegion],
    ) -> Result<(), String> {
        let cloud = crate::timing::timed("align_ghost", || compose(state));
        let seq = self.events.len();
        let name = format!("snapshots/live/ev_{seq:03}_{kind}_r{region}.ply");
        crate::timing::timed("write", || write_ply_file(self.out.join(&name), &cloud))
            .map_err(|e| format!("스냅샷 쓰기 실패: {e}"))?;
        self.events.push(LiveEvent {
            seq,
            secs,
            kind: kind.to_string(),
            region,
            snapshot: Some(name),
            points: cloud.len(),
        });
        Ok(())
    }

    /// 최종 `manifest.json` 에 `events` 배열을 덧붙인다(스냅샷 목록·align 은 그대로).
    pub fn finish(&self, out: &Path) -> Result<(), String> {
        let path = out.join("snapshots/manifest.json");
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let body = text.trim_end();
        let body = body
            .strip_suffix('}')
            .ok_or("manifest 형식 오류")?
            .trim_end();
        let rows: Vec<String> = self
            .events
            .iter()
            .map(|e| {
                format!(
                    "    {{\"seq\": {}, \"secs\": {:.2}, \"kind\": \"{}\", \"region\": {}, \"snapshot\": {}, \"points\": {}}}",
                    e.seq,
                    e.secs,
                    e.kind,
                    e.region,
                    e.snapshot
                        .as_ref()
                        .map_or("null".to_string(), |s| format!("\"{s}\"")),
                    e.points
                )
            })
            .collect();
        let new = format!("{body},\n  \"events\": [\n{}\n  ]\n}}\n", rows.join(",\n"));
        std::fs::write(&path, new).map_err(|e| e.to_string())
    }
}

/// 현재 내보낸 상태를 한 점군으로 합친다: 정밀 구역은 최신 좌표계 변환 뒤, 정밀이 없는 초벌 구역은
/// 정밀 점 전부에서 `GHOST_RADIUS_M` 안에 있는 점을 뺀 뒤 `DECIMATE_EVERY`:1 간격 추출.
pub fn compose(state: &[LiveRegion]) -> PointCloud {
    let mut index = RadiusIndex::new(GHOST_RADIUS_M);
    let mut out = PointCloud { points: Vec::new() };
    let moved: Vec<Option<PointCloud>> = state
        .iter()
        .map(|r| {
            r.refined.map(|c| match &r.refined_sim {
                Some(s) => apply_cloud(s, c),
                None => c.clone(),
            })
        })
        .collect();
    for c in moved.iter().flatten() {
        index.insert_cloud(c);
        out.points.extend(decimate(c, DECIMATE_EVERY).points);
    }
    for (r, m) in state.iter().zip(&moved) {
        if m.is_some() {
            continue;
        }
        let shown = match &r.coarse_sim {
            Some(s) => apply_cloud(s, r.coarse),
            None => r.coarse.clone(),
        };
        out.points
            .extend(decimate(&remove_ghosts(&shown, &index), DECIMATE_EVERY).points);
    }
    out
}

/// 정밀 구역 `old` 를 방금 나온 정밀 구역 `newest` 의 좌표계로 옮기는 닮음 변환.
/// 같은 사진·같은 특징 번호의 정밀 점 대응을 겹침 위치 범위에서 모아 5회 트리밍으로 맞춘다.
/// 반환: (변환, 점쌍 수, 잔차 중앙값 m). 겹침이 없거나 짝이 모자라면 `None`.
pub fn realign_refined(
    old: (&Region, &[Track]),
    newest: (&Region, &[Track]),
) -> Option<(Similarity, usize, f64)> {
    cross_align(old.1, newest.1, overlap_window(old.0, newest.0))
}

/// 카메라(0..3)마다 등록된 사진 수와 빠진 위치 목록. `gids[a]` = 전역 사진 번호(3·위치+카메라).
pub fn missing_by_camera(gids: &[usize], registered: &[bool]) -> [(usize, Vec<usize>); 3] {
    let mut out: [(usize, Vec<usize>); 3] = Default::default();
    for (a, g) in gids.iter().enumerate() {
        if registered[a] {
            out[g % 3].0 += 1;
        } else {
            out[g % 3].1.push(g / 3);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_anchor_recovers_frame_and_fixes_overlap_cameras() {
        use crate::math::{Point3, Rotation3};
        // 정밀 모델(참): 점 40개, 위치 0..12. 새 구역 초벌은 같은 점을 다른 좌표계(닮음 변환)에서 본다.
        let truth = Similarity {
            s: 1.3,
            r: Rotation3::from_euler_angles(0.02, -0.03, 0.4),
            t: Vector3::new(5.0, -2.0, 1.0),
        };
        let inv = |v: &Vector3<f64>| truth.r.inverse() * (v - truth.t) / truth.s;
        let pts: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let f = i as f64;
                Vector3::new(f * 1.7 % 11.0, f * 2.9 % 13.0, (f * 0.7) % 3.0)
            })
            .collect();
        let mk = |f: &dyn Fn(&Vector3<f64>) -> Vector3<f64>| -> Vec<Track> {
            pts.iter()
                .enumerate()
                .map(|(i, p)| Track {
                    xyz: f(p),
                    obs: vec![
                        (3 * (6 + i as u32 % 6), i as u32),
                        (3 * (8 + i as u32 % 3), i as u32),
                    ],
                })
                .collect()
        };
        let prev = mk(&|p| *p);
        let ta = mk(&inv);
        let (ra, rb) = (
            Region {
                index: 1,
                start: 6,
                lo: 6,
                hi: 14,
            },
            Region {
                index: 0,
                start: 0,
                lo: 0,
                hi: 10,
            },
        );
        let gids: Vec<usize> = (6..14).map(|p| 3 * p).collect();
        let registered = vec![true; gids.len()];
        let poses: HashMap<usize, Pose> = gids
            .iter()
            .map(|&g| {
                (
                    g,
                    Pose::from_center(Rotation3::identity(), &Point3::new(g as f64, 0.0, 30.0)),
                )
            })
            .collect();
        let plan = plan_anchor(&ta, &prev, &poses, (&ra, &rb), &gids, &registered).unwrap();
        let err = pts
            .iter()
            .map(|p| (plan.sim.apply_point(&inv(p)) - p).norm())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "복원 오차 {err}");
        assert!(plan.pairs >= 30, "{}", plan.pairs);
        // 겹침 위치 6..10 의 사진만 고정된다(위치당 1장 x 4).
        assert_eq!(plan.fixed.len(), 4);
        assert!(plan.fixed.iter().all(|&(a, _)| gids[a] / 3 < 10));
        // 등록된 사진이 2장 미만이면 붙이지 않는다.
        let none = vec![false; gids.len()];
        assert!(plan_anchor(&ta, &prev, &poses, (&ra, &rb), &gids, &none).is_err());
    }

    #[test]
    fn center_error_median_applies_similarity() {
        let c: BTreeMap<usize, [f64; 3]> = (0..5).map(|g| (g, [g as f64, 0.0, 0.0])).collect();
        let truth = |g: usize| Vector3::new(g as f64 + 2.0, 0.0, 0.0);
        assert!((center_error_median(&c, None, truth).unwrap() - 2.0).abs() < 1e-12);
        let shift = Similarity {
            t: Vector3::new(2.0, 0.0, 0.0),
            ..Similarity::identity()
        };
        assert!(center_error_median(&c, Some(&shift), truth).unwrap() < 1e-12);
    }

    #[test]
    fn missing_table_counts_per_camera() {
        let gids: Vec<usize> = (0..4)
            .flat_map(|p| (0..3).map(move |c| 3 * p + c))
            .collect();
        let reg: Vec<bool> = gids.iter().map(|g| !(g % 3 == 2 && g / 3 >= 2)).collect();
        let t = missing_by_camera(&gids, &reg);
        assert_eq!((t[0].0, t[1].0, t[2].0), (4, 4, 2));
        assert_eq!(t[2].1, vec![2, 3]);
    }
}
