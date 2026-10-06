//! 구역 순서 처리의 보조 논리: 이미 내보낸 정밀 구역을 최신 정밀 모델 좌표계로 다시 맞추는
//! 닮음 변환(공유 3D 점 대응)과 등록 현황 표.

use std::path::{Path, PathBuf};

use crate::align::Similarity;
use crate::ply::{write_ply_file, PointCloud};
use crate::progressive::{cross_align, overlap_window};
use crate::stream::{
    apply_cloud, decimate, remove_ghosts, RadiusIndex, Region, Track, DECIMATE_EVERY,
    GHOST_RADIUS_M,
};

/// 실행 방식. `sequential` 이면 구역마다 정밀(BA)이 끝난 뒤에 다음 구역을 받는다(비교용 기준 흐름).
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamOptions {
    pub sequential: bool,
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

/// 최종 초벌 출력에 쓸 변환: 자기 정밀 구역 좌표로 옮기는 `own` 뒤에, 그 정밀 구역이 받은
/// 누적 재정렬 변환 `rsim`(최신 정밀 좌표계로)을 이어 붙인다. `own` 이 없으면(정렬 실패) `None`.
pub fn preview_final_sim(own: Option<Similarity>, rsim: Option<Similarity>) -> Option<Similarity> {
    match (own, rsim) {
        (Some(o), Some(r)) => Some(r.compose(&o)),
        (o, _) => o,
    }
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
    fn preview_final_sim_chains_own_then_cumulative() {
        use crate::math::Vector3;
        let own = Similarity {
            s: 2.0,
            r: crate::nalgebra::Rotation3::identity(),
            t: Vector3::new(1.0, 0.0, 0.0),
        };
        let cum = Similarity {
            s: 1.0,
            r: crate::nalgebra::Rotation3::identity(),
            t: Vector3::new(0.0, 5.0, 0.0),
        };
        let p = Vector3::new(1.0, 1.0, 1.0);
        let f = preview_final_sim(Some(own), Some(cum)).unwrap();
        assert!((f.apply_point(&p) - cum.apply_point(&own.apply_point(&p))).norm() < 1e-12);
        assert_eq!(preview_final_sim(Some(own), None).unwrap().t, own.t);
        assert!(preview_final_sim(None, Some(cum)).is_none());
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
