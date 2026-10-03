//! 구역 순서 처리의 보조 논리: 이미 내보낸 정밀 구역을 최신 정밀 모델 좌표계로 다시 맞추는
//! 닮음 변환(공유 3D 점 대응)과 등록 현황 표.

use crate::align::Similarity;
use crate::progressive::{cross_align, overlap_window};
use crate::stream::{Region, Track};

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
