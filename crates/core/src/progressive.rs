//! 구역을 위치 순서대로 처리할 때 쓰는 보조 논리: 초벌 점을 최신 정밀 모델 좌표계로 옮기는
//! 닮음 변환, 구역 소유 범위, 정지 구간 검사, 겹침 구간 중심 차이.

use crate::align::Similarity;
use crate::math::Vector3;
use crate::stream::{point_pairs, robust_fit, Region, Track};

/// 재정렬 한 번의 기록.
#[derive(Clone, Debug, PartialEq)]
pub struct ReAlign {
    /// 이 변환이 만들어진 순간(실행 시작 후 초).
    pub secs: f64,
    /// 옮겨지는 초벌 구역.
    pub region: usize,
    /// 기준이 된 정밀 구역.
    pub target: usize,
    pub pairs: usize,
    /// 트리밍 뒤 잔차 중앙값(m).
    pub median_m: f64,
    pub scale: f64,
}

/// 초벌 트랙과 정밀 트랙의 공유 3D 점(같은 사진·같은 특징 번호)으로 닮음 변환을 구한다.
/// `window` 는 점쌍에 쓰는 사진 위치 범위 `[lo, hi)`. 짝이 3개 미만이거나 퇴화하면 `None`.
/// 반환: (변환, 점쌍 수, 잔차 중앙값).
pub fn cross_align(
    coarse: &[Track],
    refined: &[Track],
    window: (usize, usize),
) -> Option<(Similarity, usize, f64)> {
    if window.0 >= window.1 {
        return None;
    }
    let pairs = point_pairs(coarse, refined, |i| (i / 3) as usize, window);
    let (src, dst): (Vec<Vector3<f64>>, Vec<Vector3<f64>>) = pairs.iter().copied().unzip();
    let (sim, _, med) = robust_fit(&src, &dst)?;
    Some((sim, pairs.len(), med))
}

/// 두 구역이 함께 가진 위치 범위.
pub fn overlap_window(a: &Region, b: &Region) -> (usize, usize) {
    (a.lo.max(b.lo), a.hi.min(b.hi))
}

/// 구역마다 "자기" 위치 범위 `[own_lo, own_hi)`: 구역 i 는 `start_i` 부터 다음 구역 `start` 직전까지,
/// 첫 구역은 0 부터, 마지막 구역은 끝까지.
pub fn own_ranges(regions: &[Region], n_positions: usize) -> Vec<(usize, usize)> {
    (0..regions.len())
        .map(|i| {
            let lo = if i == 0 { 0 } else { regions[i].start };
            let hi = regions.get(i + 1).map_or(n_positions, |r| r.start);
            (lo, hi)
        })
        .collect()
}

/// GPS 기준선이 좌표계 정렬(Kabsch)에 쓸 만한가. `views[i] = (카메라 번호, 위치 번호)`.
/// 같은 카메라끼리 기준선이 3 m 를 넘는 짝(= 실제로 움직인 짝)이 3개 미만이면 정지·체공 구간으로 보고
/// 오류를 낸다. 편대 안 카메라 간 간격(약 10 m)은 움직임이 아니므로 세지 않는다.
pub fn check_motion(
    gps: &[Vector3<f64>],
    views: &[(usize, usize)],
    pairs: &[(usize, usize)],
) -> Result<(), String> {
    let n = pairs
        .iter()
        .filter(|&&(i, j)| views[i].0 == views[j].0 && (gps[j] - gps[i]).norm() > 3.0)
        .count();
    if n < 3 {
        return Err(format!(
            "같은 카메라 GPS 기준선 3 m 초과 짝 {n} 개 < 3: 정지 구간이라 좌표계를 정할 수 없음"
        ));
    }
    Ok(())
}

/// 값 목록의 중앙값(빈 목록은 `None`).
pub fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// 연쇄 재정렬 한 칸: 구역 `region` 을 `via` 를 거쳐 최신 구역 좌표계로 옮기는 누적 변환.
#[derive(Clone, Debug)]
pub struct ChainStep {
    pub region: usize,
    pub via: usize,
    pub total: Similarity,
    pub pairs: usize,
    pub median_m: f64,
}

/// 정밀 구역들(`items[j] = Some((구역, 정밀 트랙))`)을 최신 정밀 구역 `k` 의 좌표계로 옮기는 변환을 구한다.
/// 같은 사진·같은 특징의 공유 3D 점으로 겹치는 이웃끼리 닮음 변환을 구하고, 최신 구역과 겹치지 않는
/// 구역은 겹치는 이웃을 거쳐 변환을 연쇄 합성한다(너비 우선). 최신 구역 자신은 결과에 없다.
pub fn chain_realign(items: &[Option<(Region, &[Track])>], k: usize) -> Vec<ChainStep> {
    let mut total: Vec<Option<Similarity>> = vec![None; items.len()];
    total[k] = Some(Similarity::identity());
    let mut out = Vec::new();
    let mut queue = std::collections::VecDeque::from([k]);
    while let Some(m) = queue.pop_front() {
        let Some((rm, tm)) = items[m] else { continue };
        for j in 0..items.len() {
            let Some((rj, tj)) = items[j] else { continue };
            if total[j].is_some() {
                continue;
            }
            let Some((s, n, med)) = cross_align(tj, tm, overlap_window(&rj, &rm)) else {
                continue;
            };
            let acc = total[m].as_ref().unwrap().compose(&s);
            out.push(ChainStep {
                region: j,
                via: m,
                total: acc,
                pairs: n,
                median_m: med,
            });
            total[j] = Some(acc);
            queue.push_back(j);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_ranges_partition_positions() {
        let rs = crate::stream::split_regions(26, 8, 2);
        let own = own_ranges(&rs, 26);
        assert_eq!(own[0].0, 0);
        assert_eq!(own.last().unwrap().1, 26);
        for w in own.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
    }

    #[test]
    fn overlap_window_is_intersection() {
        let a = Region {
            index: 0,
            start: 0,
            lo: 0,
            hi: 10,
        };
        let b = Region {
            index: 1,
            start: 8,
            lo: 6,
            hi: 20,
        };
        assert_eq!(overlap_window(&a, &b), (6, 10));
    }

    #[test]
    fn hovering_is_rejected_and_motion_accepted() {
        let views: Vec<(usize, usize)> = (0..6).map(|i| (0, i)).collect();
        let pairs: Vec<(usize, usize)> = (0..5).map(|i| (i, i + 1)).collect();
        let still: Vec<Vector3<f64>> = (0..6)
            .map(|i| Vector3::new(0.1 * i as f64, 0.0, 0.0))
            .collect();
        assert!(check_motion(&still, &views, &pairs).is_err());
        let moving: Vec<Vector3<f64>> = (0..6)
            .map(|i| Vector3::new(8.0 * i as f64, 0.0, 0.0))
            .collect();
        assert!(check_motion(&moving, &views, &pairs).is_ok());
    }

    #[test]
    fn cross_align_recovers_shift() {
        let mk = |shift: f64| -> Vec<Track> {
            (0..60)
                .map(|k| Track {
                    xyz: Vector3::new(k as f64, (k * k % 7) as f64, (k % 5) as f64 * 2.0 + shift),
                    obs: vec![(3 * (k as u32 % 4), k as u32)],
                })
                .collect()
        };
        let (a, b) = (mk(0.0), mk(1.5));
        let (sim, n, med) = cross_align(&a, &b, (0, 4)).unwrap();
        assert_eq!(n, 60);
        assert!(med < 1e-6, "{med}");
        assert!((sim.apply_point(&a[0].xyz) - b[0].xyz).norm() < 1e-6);
        assert!(cross_align(&a, &b, (2, 2)).is_none());
    }

    #[test]
    fn median_basic() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![]), None);
    }
}
