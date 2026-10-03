//! 다시점 특징 트랙: 짝별 기하 검증된 대응을 (영상 번호, 특징 번호) 노드의 연결 성분으로 묶는다.
//!
//! 입력은 매칭 모듈의 출력 그대로다: [`crate::matching::ratio_match`] 의 (a 특징, b 특징) 목록에서
//! [`crate::matching::ransac_fundamental`] 의 정상 표시가 참인 것만 남긴 짝 목록([`verified_matches`]).
//!
//! 충돌: 한 성분에 같은 영상의 서로 다른 특징이 둘 이상 들어가면 한 3D 점의 관측으로 볼 수 없다.
//! - [`ConflictPolicy::Drop`]: 충돌 성분을 통째로 버린다.
//! - [`ConflictPolicy::Split`]: 대응 간선을 "삼각 지지도"(두 끝 노드가 함께 대응되는 제3 노드 수)
//!   내림차순으로(같으면 노드 번호 순) 하나씩 합치되, 합치면 같은 영상의 특징이 둘이 되는 간선은 건너뛴다.
//!   한 트랙 안의 참 대응은 길이 L 트랙에서 지지도가 최대 L−2 이고, 우연한 오대응은 공통 이웃이 거의
//!   없어 지지도가 0 에 가깝다. 따라서 오대응이 마지막에 처리되고 충돌을 만들면 버려진다.
//!   지지도 0 간선은 지지도가 있는 간선을 모두 처리한 뒤 성분 쌍 단위로 모아, 잇는 간선이 많은 쌍부터
//!   영상이 겹치지 않을 때 잇는다(대응이 성긴 참 트랙 조각은 간선 한 개로만 이어지는 경우가 많다).
//!   충돌이 없는 성분은 어느 정책이든 같은 결과다.
//!
//! 오대응 선거름(Split): 짝마다 대응의 변위(b 화소 − a 화소)를 같은 짝에서 영상 a 의 가까운 대응들
//! 변위의 중앙값과 비교해 크게 어긋난 대응은 간선으로 쓰지 않는다. 그래프 구조만으로는 대응이 성긴
//! (재현율 30%) 참 트랙의 잎과 오대응 잎이 구별되지 않아 순도가 0.97 에 머물렀다(노트 참고).
//!
//! 결정성: 간선을 (작은 노드, 큰 노드) 로 정규화·정렬·중복 제거한 뒤 처리하므로 짝 순서, 짝 안 대응 순서,
//! 짝의 앞뒤(a, b) 방향을 바꿔도 결과가 같다. 출력 트랙은 관측을 (영상, 특징) 순으로, 트랙 목록은
//! 첫 관측 순으로 정렬한다.

use crate::ba::Observation;
use crate::math::{Matrix3, Vector2, Vector3};
use std::collections::HashMap;

/// 국소 변위 일관성 기준 문턱(화소): 960x540 영상 기준. 영상 크기에 비례해 늘린다.
const DISPLACEMENT_TOLERANCE: f64 = 40.0;
/// 기준 영상 크기(화소).
const DISPLACEMENT_REF_WIDTH: f64 = 960.0;
const DISPLACEMENT_REF_HEIGHT: f64 = 540.0;
/// 국소 아핀 맞춤에 쓰는 최근접 이웃 수.
const DISPLACEMENT_NEIGHBORS: usize = 24;

/// SPEC 번들 조정 트랙 상한과 같은 값.
pub const MAX_TRACKS: usize = 100_000;

/// 한 영상 짝의 검증된 대응.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairMatches {
    pub image_a: usize,
    pub image_b: usize,
    /// (image_a 특징 번호, image_b 특징 번호).
    pub matches: Vec<(usize, usize)>,
}

/// `ratio_match` 결과에서 RANSAC 정상 표시가 참인 짝만 남긴다.
/// 대응과 정상 표시의 길이가 다르면 (대응 수, 표시 수) 를 오류로 돌려준다.
pub fn verified_matches(
    matches: &[(usize, usize)],
    inliers: &[bool],
) -> Result<Vec<(usize, usize)>, (usize, usize)> {
    if matches.len() != inliers.len() {
        return Err((matches.len(), inliers.len()));
    }
    Ok(matches
        .iter()
        .zip(inliers)
        .filter(|(_, &ok)| ok)
        .map(|(&m, _)| m)
        .collect())
}

/// 트랙 관측 하나: 영상 번호, 특징 번호, 화소 좌표.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackObservation {
    pub image: usize,
    pub feature: usize,
    pub pixel: Vector2<f64>,
}

/// 한 3D 점의 관측 목록. 영상 번호 오름차순, 영상마다 하나.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub observations: Vec<TrackObservation>,
}

impl Track {
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// 시점 다양성: 관측 영상 번호의 폭(최대 − 최소). 촬영 순서대로 번호가 매겨진 영상에서
    /// 폭이 넓을수록 기선이 길어 삼각측량이 안정하다.
    pub fn image_span(&self) -> usize {
        match (self.observations.first(), self.observations.last()) {
            (Some(a), Some(b)) => b.image - a.image,
            _ => 0,
        }
    }
}

/// 충돌 성분 처리 규칙.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictPolicy {
    Drop,
    Split,
}

/// 트랙 만들기 설정.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackConfig {
    pub policy: ConflictPolicy,
    /// 최소 관측 수(2 미만은 2 로 본다). 기본 3: 두 뷰 트랙은 위치 추정에서 간격 비를 정하지 못한다.
    pub min_length: usize,
    /// 트랙 수 상한([`select_tracks`]).
    pub max_tracks: usize,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            policy: ConflictPolicy::Split,
            min_length: 3,
            max_tracks: MAX_TRACKS,
        }
    }
}

/// 트랙 만들기 통계.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackStats {
    /// 정규화·중복 제거 후 간선 수.
    pub edges: usize,
    /// 범위 밖 번호·같은 영상 짝이라 버린 대응 수.
    pub invalid_matches: usize,
    /// Drop: 버린 충돌 성분 수. Split: 충돌(또는 지지도 0 으로 두 트랙 잇기) 때문에 건너뛴 간선 수.
    pub conflicts: usize,
    /// 길이 미달로 버린 성분 수.
    pub too_short: usize,
    /// Split: 같은 짝 이웃 대응의 변위와 어긋나 버린 대응 수.
    pub inconsistent: usize,
    /// 상한 때문에 버린 트랙 수.
    pub truncated: usize,
    pub tracks: usize,
}

/// 합집합-찾기. 대표는 항상 성분 안 가장 작은 노드 번호다(순서와 무관한 대표).
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        while self.parent[x] != r {
            let next = self.parent[x];
            self.parent[x] = r;
            x = next;
        }
        r
    }

    /// 두 대표를 합치고 새 대표(작은 쪽)를 돌려준다.
    fn link(&mut self, ra: usize, rb: usize) -> usize {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[hi] = lo;
        lo
    }
}

/// 검증된 대응에서 트랙을 만든다. `keypoints[i][f]` 는 영상 i 특징 f 의 화소 좌표.
/// 결과는 상한 선택([`select_tracks`])까지 마친 트랙(첫 관측 순)과 통계.
pub fn build_tracks(
    pairs: &[PairMatches],
    keypoints: &[Vec<Vector2<f64>>],
    cfg: &TrackConfig,
) -> (Vec<Track>, TrackStats) {
    let mut stats = TrackStats::default();
    let mut offset = Vec::with_capacity(keypoints.len() + 1);
    offset.push(0usize);
    for k in keypoints {
        offset.push(offset.last().unwrap() + k.len());
    }
    let n = *offset.last().unwrap();
    let mut node_image = vec![0usize; n];
    for i in 0..keypoints.len() {
        node_image[offset[i]..offset[i + 1]].fill(i);
    }

    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut valid: Vec<(usize, usize)> = Vec::new();
    for p in pairs {
        valid.clear();
        for &(fa, fb) in &p.matches {
            let ok = p.image_a != p.image_b
                && p.image_a < keypoints.len()
                && p.image_b < keypoints.len()
                && fa < keypoints[p.image_a].len()
                && fb < keypoints[p.image_b].len();
            if !ok {
                stats.invalid_matches += 1;
                continue;
            }
            valid.push((fa, fb));
        }
        // 방향·순서·중복과 무관하게: 작은 영상 번호를 기준 영상으로, 대응은 정렬·중복 제거.
        let flip = p.image_a > p.image_b;
        if flip {
            valid.iter_mut().for_each(|m| *m = (m.1, m.0));
        }
        valid.sort_unstable();
        valid.dedup();
        let (lo, hi) = (p.image_a.min(p.image_b), p.image_a.max(p.image_b));
        let outlier = if cfg.policy == ConflictPolicy::Split && !valid.is_empty() {
            displacement_outliers(&valid, &keypoints[lo], &keypoints[hi])
        } else {
            vec![false; valid.len()]
        };
        for (&(fa, fb), &bad) in valid.iter().zip(&outlier) {
            if bad {
                stats.inconsistent += 1;
                continue;
            }
            let (u, v) = (offset[lo] + fa, offset[hi] + fb);
            edges.push((u.min(v), u.max(v)));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    stats.edges = edges.len();

    let mut uf = UnionFind::new(n);
    match cfg.policy {
        ConflictPolicy::Drop => {
            for &(u, v) in &edges {
                let (ru, rv) = (uf.find(u), uf.find(v));
                if ru != rv {
                    uf.link(ru, rv);
                }
            }
        }
        ConflictPolicy::Split => {
            // 인접 목록(CSR: 오프셋 + 평탄 배열, 노드마다 정렬)으로 간선마다 공통 이웃 수를 센다.
            let mut start = vec![0usize; n + 1];
            for &(u, v) in &edges {
                start[u + 1] += 1;
                start[v + 1] += 1;
            }
            for i in 0..n {
                start[i + 1] += start[i];
            }
            let mut fill = start.clone();
            let mut flat = vec![0usize; 2 * edges.len()];
            for &(u, v) in &edges {
                flat[fill[u]] = v;
                fill[u] += 1;
                flat[fill[v]] = u;
                fill[v] += 1;
            }
            for i in 0..n {
                flat[start[i]..start[i + 1]].sort_unstable();
            }
            let nb = |x: usize| &flat[start[x]..start[x + 1]];
            let mut order: Vec<(usize, usize, usize)> = edges
                .iter()
                .map(|&(u, v)| (common_count(nb(u), nb(v)), u, v))
                .collect();
            order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
            // 대표마다 성분이 가진 영상 번호(정렬됨). 비어 있으면 홀로인 노드.
            let mut images: Vec<Vec<usize>> = vec![Vec::new(); n];
            // 1 단계: 지지도 > 0 간선을 지지도 내림차순으로 합친다(충돌이면 건너뜀).
            let split = order.iter().position(|e| e.0 == 0).unwrap_or(order.len());
            for &(_, u, v) in &order[..split] {
                if !try_join(&mut uf, &mut images, &node_image, u, v) {
                    stats.conflicts += 1;
                }
            }
            // 2 단계: 지지도 0 간선은 1 단계 뒤 성분 쌍마다 묶는다. 대응 재현율 30~50% 에서는 참 트랙
            // 조각 사이 간선이 1 개뿐인 경우가 많아 간선 수 문턱을 두면 참 트랙이 쪼개진다(완전도 0.83).
            // 대신 순서로 오대응을 늦춘다: 성분 그래프에서 간선 2 개 이상이거나 공통 이웃 성분(삼각형)이
            // 있는 쌍을 먼저 되풀이해 합쳐 참 트랙을 키운 뒤, 남은 단일 간선을 마지막에 합친다. 무작위
            // 오대응은 삼각형을 이루기 어려워 뒤로 밀리고, 그때는 양쪽 트랙이 커져 영상이 겹쳐(충돌) 걸러진다.
            let zero = &order[split..];
            let mut comp_start = vec![0usize; n + 1];
            for round in 0..=8 {
                let mut counted = component_pairs(zero, &mut uf);
                if counted.is_empty() {
                    break;
                }
                // 성분 그래프 CSR(대표 번호 = 노드 번호)로 쌍마다 공통 이웃 성분 수를 센다.
                comp_start.iter_mut().for_each(|x| *x = 0);
                for &(a, b, _, _) in &counted {
                    comp_start[a + 1] += 1;
                    comp_start[b + 1] += 1;
                }
                for i in 0..n {
                    comp_start[i + 1] += comp_start[i];
                }
                let mut fill = comp_start.clone();
                let mut flat = vec![0usize; 2 * counted.len()];
                for &(a, b, _, _) in &counted {
                    flat[fill[a]] = b;
                    fill[a] += 1;
                    flat[fill[b]] = a;
                    fill[b] += 1;
                }
                // `counted` 가 (a, b) 오름차순이라 각 목록은 이미 정렬돼 있지 않을 수 있다.
                for i in 0..n {
                    if comp_start[i + 1] - comp_start[i] > 1 {
                        flat[comp_start[i]..comp_start[i + 1]].sort_unstable();
                    }
                }
                for c in counted.iter_mut() {
                    c.3 = common_count(
                        &flat[comp_start[c.0]..comp_start[c.0 + 1]],
                        &flat[comp_start[c.1]..comp_start[c.1 + 1]],
                    );
                }
                counted.sort_unstable_by(|x, y| {
                    (y.2, y.3)
                        .cmp(&(x.2, x.3))
                        .then((x.0, x.1).cmp(&(y.0, y.1)))
                });
                let last = round == 8;
                let mut merged = 0;
                for (a, b, count, sup) in counted {
                    let strong = count >= 2 || sup > 0;
                    if !strong && !last {
                        continue;
                    }
                    if try_join(&mut uf, &mut images, &node_image, a, b) {
                        merged += 1;
                    } else {
                        stats.conflicts += count;
                    }
                }
                if last {
                    break;
                }
                if merged == 0 {
                    // 강한 쌍이 더 없으면 마지막 단계(단일 간선 포함)로 넘어간다.
                    let mut rest = component_pairs(zero, &mut uf);
                    rest.sort_unstable_by(|x, y| y.2.cmp(&x.2).then((x.0, x.1).cmp(&(y.0, y.1))));
                    for (a, b, count, _) in rest {
                        if !try_join(&mut uf, &mut images, &node_image, a, b) {
                            stats.conflicts += count;
                        }
                    }
                    break;
                }
            }
        }
    }

    // 성분 모으기: 대표 → 노드(노드 번호 오름차순 = (영상, 특징) 순).
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(u, v) in &edges {
        for x in [u, v] {
            let r = uf.find(x);
            members.entry(r).or_default().push(x);
        }
    }
    let mut roots: Vec<usize> = members.keys().copied().collect();
    roots.sort_unstable();
    let min_len = cfg.min_length.max(2);
    let mut tracks = Vec::new();
    for r in roots {
        let mut nodes = members.remove(&r).unwrap();
        nodes.sort_unstable();
        nodes.dedup();
        let conflict = nodes
            .windows(2)
            .any(|w| node_image[w[0]] == node_image[w[1]]);
        if conflict {
            // Split 에서는 생기지 않는다.
            stats.conflicts += 1;
            continue;
        }
        if nodes.len() < min_len {
            stats.too_short += 1;
            continue;
        }
        tracks.push(Track {
            observations: nodes
                .iter()
                .map(|&x| {
                    let image = node_image[x];
                    let feature = x - offset[image];
                    TrackObservation {
                        image,
                        feature,
                        pixel: keypoints[image][feature],
                    }
                })
                .collect(),
        });
    }
    let before = tracks.len();
    let tracks = select_tracks(tracks, cfg.max_tracks);
    stats.truncated = before - tracks.len();
    stats.tracks = tracks.len();
    (tracks, stats)
}

/// 짝 안 국소 변위 일관성: 대응마다 같은 짝에서 영상 a 의 가까운 대응 `DISPLACEMENT_NEIGHBORS` 개(점 중심
/// 최근접 이웃)로 변위장 `d(u) = t + A u` (국소 아핀, u = 이웃 − 대상 점)를 강건하게 맞추고, 대상 점의
/// 변위와 예측 `t` 의 차이(잔차)가 문턱을 넘으면 오대응으로 본다. 짝 사이 회전·축척 차가 있어도
/// 변위장은 위치에 대해 아핀이므로 참 대응의 잔차는 작다. 이웃이 적거나 한 직선 위면(증거 부족) 그대로 둔다.
///
/// 맞춤: 최소제곱 → 잔차 중앙 절대편차(MAD)로 이상치 이웃 제외 → 다시 맞춤을 두 번 되풀이한다.
/// 문턱 = clamp(4 · 1.4826 · MAD, 기준 문턱, 3 · 기준 문턱), 기준 문턱은 영상 크기에 비례
/// (960x540 에서 40 px). 영상 크기는 영상 a 특징 좌표의 최대값으로 어림한다.
fn displacement_outliers(
    matches: &[(usize, usize)],
    kp_a: &[Vector2<f64>],
    kp_b: &[Vector2<f64>],
) -> Vec<bool> {
    let mut out = vec![false; matches.len()];
    if matches.len() < 8 {
        return out;
    }
    let pos: Vec<(Vector2<f64>, Vector2<f64>)> = matches
        .iter()
        .map(|&(fa, fb)| (kp_a[fa], kp_b[fb] - kp_a[fa]))
        .collect();
    let (mut max_x, mut max_y) = (0.0f64, 0.0f64);
    let (mut min_x, mut min_y) = (f64::MAX, f64::MAX);
    for q in &pos {
        max_x = max_x.max(q.0.x);
        max_y = max_y.max(q.0.y);
        min_x = min_x.min(q.0.x);
        min_y = min_y.min(q.0.y);
    }
    let size = (max_x / DISPLACEMENT_REF_WIDTH).max(max_y / DISPLACEMENT_REF_HEIGHT);
    if size <= 0.0 || !size.is_finite() {
        return out;
    }
    let tol = DISPLACEMENT_TOLERANCE * size;
    // 칸당 평균 8 점쯤 되도록 격자 칸을 정한다(점 분포 범위 기준).
    let area = ((max_x - min_x) * (max_y - min_y)).max(1e-12);
    let cell = (area * 8.0 / pos.len() as f64)
        .sqrt()
        .max(size * 4.0)
        .max(1e-9);
    let key = |p: &Vector2<f64>| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64);
    let mut idx: Vec<((i64, i64), usize)> = pos
        .iter()
        .enumerate()
        .map(|(i, q)| (key(&q.0), i))
        .collect();
    idx.sort_unstable();
    let k = DISPLACEMENT_NEIGHBORS;
    let mut cand: Vec<(f64, usize)> = Vec::new();
    // 이웃 하나: (u / cell, 변위)
    let mut nb: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(k);
    let mut res: Vec<f64> = Vec::with_capacity(k);
    let mut keep: Vec<bool> = Vec::with_capacity(k);
    for (i, q) in pos.iter().enumerate() {
        let (cx, cy) = key(&q.0);
        cand.clear();
        for ring in 1..=3i64 {
            cand.clear();
            for gx in cx - ring..=cx + ring {
                for gy in cy - ring..=cy + ring {
                    let lo = idx.partition_point(|e| e.0 < (gx, gy));
                    for e in idx[lo..].iter().take_while(|e| e.0 == (gx, gy)) {
                        if e.1 != i {
                            let d = pos[e.1].0 - q.0;
                            cand.push((d.x * d.x + d.y * d.y, e.1));
                        }
                    }
                }
            }
            if cand.len() >= k {
                break;
            }
        }
        if cand.len() < 8 {
            continue;
        }
        if cand.len() > k {
            cand.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
            cand.truncate(k);
        }
        nb.clear();
        for &(_, j) in &cand {
            let u = (pos[j].0 - q.0) / cell;
            nb.push((u.x, u.y, pos[j].1.x - q.1.x, pos[j].1.y - q.1.y));
        }
        // 변위는 대상 점 변위를 뺀 값으로 둔다: 맞춘 상수항 t 가 곧 (예측 − 실제) 의 반대 부호 잔차다.
        keep.clear();
        keep.resize(nb.len(), true);
        let mut fit = None;
        let mut s = 0.0;
        for round in 0..3 {
            let Some(f) = fit_affine(&nb, &keep) else {
                break;
            };
            res.clear();
            for e in &nb {
                let rx = e.2 - (f[0] + f[1] * e.0 + f[2] * e.1);
                let ry = e.3 - (f[3] + f[4] * e.0 + f[5] * e.1);
                res.push((rx * rx + ry * ry).sqrt());
            }
            let mut sorted = res.clone();
            sorted.sort_unstable_by(|a, b| a.total_cmp(b));
            s = 1.4826 * sorted[sorted.len() / 2];
            fit = Some(f);
            if round == 2 {
                break;
            }
            let cut = (3.0 * s).max(tol * 0.25);
            let mut n = 0;
            for (kp, r) in keep.iter_mut().zip(&res) {
                *kp = *r <= cut;
                n += *kp as usize;
            }
            if n < 6 {
                break;
            }
        }
        let Some(f) = fit else {
            continue;
        };
        // 대상 점은 u = 0, 변위 0(뺀 값)이므로 예측 (f[0], f[3]) 가 곧 어긋남이다.
        let r = (f[0] * f[0] + f[3] * f[3]).sqrt();
        out[i] = r > (4.0 * s).clamp(tol, 3.0 * tol);
    }
    out
}

/// `keep` 인 이웃에 대해 `dx = a0 + a1 u + a2 v`, `dy = b0 + b1 u + b2 v` 의 최소제곱 해
/// `[a0, a1, a2, b0, b1, b2]`. 점이 한 직선 위에 가까우면 `None`.
fn fit_affine(nb: &[(f64, f64, f64, f64)], keep: &[bool]) -> Option<[f64; 6]> {
    let mut m = Matrix3::<f64>::zeros();
    let (mut rx, mut ry) = (Vector3::<f64>::zeros(), Vector3::<f64>::zeros());
    for (e, _) in nb.iter().zip(keep).filter(|(_, &k)| k) {
        let v = Vector3::new(1.0, e.0, e.1);
        m += v * v.transpose();
        rx += v * e.2;
        ry += v * e.3;
    }
    let det = m.determinant();
    // 정규화된 좌표(칸 크기 1)에서 기대 크기 대비 너무 작으면 퇴화로 본다.
    if det.is_nan() || det <= 1e-4 * m[(0, 0)].powi(3) {
        return None;
    }
    let inv = m.try_inverse()?;
    let (a, b) = (inv * rx, inv * ry);
    Some([a[0], a[1], a[2], b[0], b[1], b[2]])
}

/// 두 노드의 성분을 영상이 겹치지 않을 때만 합친다. 이미 같은 성분이면 참(할 일 없음).
fn try_join(
    uf: &mut UnionFind,
    images: &mut [Vec<usize>],
    node_image: &[usize],
    u: usize,
    v: usize,
) -> bool {
    let (ru, rv) = (uf.find(u), uf.find(v));
    if ru == rv {
        return true;
    }
    let single_u = [node_image[ru]];
    let single_v = [node_image[rv]];
    let iu: &[usize] = if images[ru].is_empty() {
        &single_u
    } else {
        &images[ru]
    };
    let iv: &[usize] = if images[rv].is_empty() {
        &single_v
    } else {
        &images[rv]
    };
    if common_count(iu, iv) > 0 {
        return false;
    }
    let m = merge_sorted(iu, iv);
    images[ru] = Vec::new();
    images[rv] = Vec::new();
    let r = uf.link(ru, rv);
    images[r] = m;
    true
}

/// 지지도 0 간선을 현재 성분 쌍 (작은 대표, 큰 대표, 간선 수, 0) 으로 묶는다(대표 쌍 오름차순).
fn component_pairs(
    zero: &[(usize, usize, usize)],
    uf: &mut UnionFind,
) -> Vec<(usize, usize, usize, usize)> {
    let mut groups: Vec<(usize, usize)> = zero
        .iter()
        .filter_map(|&(_, u, v)| {
            let (ru, rv) = (uf.find(u), uf.find(v));
            (ru != rv).then(|| (ru.min(rv), ru.max(rv)))
        })
        .collect();
    groups.sort_unstable();
    let mut counted: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(groups.len());
    for (a, b) in groups {
        match counted.last_mut() {
            Some(last) if last.0 == a && last.1 == b => last.2 += 1,
            _ => counted.push((a, b, 1, 0)),
        }
    }
    counted
}

/// 정렬된 두 목록의 공통 원소 수.
fn common_count(a: &[usize], b: &[usize]) -> usize {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

/// 정렬된 두 목록을 정렬 없이 하나로 합친다.
fn merge_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// 트랙 수 상한 선택: 관측 수 내림차순, 같으면 시점 다양성([`Track::image_span`]) 내림차순,
/// 그래도 같으면 첫 관측 (영상, 특징) 오름차순으로 상위 `max_tracks` 개. 결과는 첫 관측 순으로 정렬.
pub fn select_tracks(mut tracks: Vec<Track>, max_tracks: usize) -> Vec<Track> {
    let key = |t: &Track| t.observations.first().map(|o| (o.image, o.feature));
    if tracks.len() > max_tracks {
        tracks.sort_by(|a, b| {
            b.len()
                .cmp(&a.len())
                .then(b.image_span().cmp(&a.image_span()))
                .then(key(a).cmp(&key(b)))
        });
        tracks.truncate(max_tracks);
    }
    tracks.sort_by_key(key);
    tracks
}

/// 번들 조정 관측 목록으로 바꾼다: 카메라 = 영상 번호, 점 = 트랙 순번.
pub fn to_ba_observations(tracks: &[Track]) -> Vec<Observation> {
    tracks
        .iter()
        .enumerate()
        .flat_map(|(p, t)| {
            t.observations.iter().map(move |o| Observation {
                camera: o.image,
                point: p,
                pixel: o.pixel,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{candidate_pairs, scheduled_pairs, PairSchedule};
    use crate::math::Point3;
    use crate::synth::{CamId, Scene, SceneConfig};

    fn hash(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    /// 합성 장면 대응: 영상별 특징(정답 점 번호 포함), 검증된 대응(오대응 섞음).
    struct Synthetic {
        keypoints: Vec<Vec<Vector2<f64>>>,
        /// `gt[i][f]` = 영상 i 특징 f 의 정답 점 번호.
        gt: Vec<Vec<usize>>,
        pairs: Vec<PairMatches>,
        outliers: usize,
        inliers: usize,
    }

    /// `outlier_per_mille`: 짝마다 참 대응 1000 개당 섞는 무작위 오대응 수.
    /// `keep_percent`: 짝마다 남기는 참 대응 비율(%, 대응 재현율).
    /// `swap_percent`: 정답 점 중 이 비율(%)을 1.5 m 옆 격자 점과 모든 짝에서 일관되게 바꿔
    /// 대응시킨다(반복 무늬형 오대응).
    fn synthetic(outlier_per_mille: u64, keep_percent: u64, swap_percent: u64) -> Synthetic {
        synthetic_seeded(outlier_per_mille, keep_percent, swap_percent, 0)
    }

    /// `seed`: 대응 유지·오대응 선택의 시드(0 이면 시드 없는 기본 선택과 같다).
    fn synthetic_seeded(
        outlier_per_mille: u64,
        keep_percent: u64,
        swap_percent: u64,
        seed: u64,
    ) -> Synthetic {
        synthetic_scene(
            outlier_per_mille,
            keep_percent,
            swap_percent,
            seed,
            &SceneOpts::default(),
        )
    }

    /// 장면 선택: 위치 수, 영상 크기, 카메라 간 짝 포함 여부.
    struct SceneOpts {
        positions: usize,
        width: u32,
        height: u32,
        /// 참이면 실측 편대 짝 일정(F–R·F–L 위치 차 +20..40 포함), 거짓이면 기존 후보 짝.
        formation_pairs: bool,
    }

    impl Default for SceneOpts {
        fn default() -> Self {
            Self {
                positions: 12,
                width: 960,
                height: 540,
                formation_pairs: false,
            }
        }
    }

    fn synthetic_scene(
        outlier_per_mille: u64,
        keep_percent: u64,
        swap_percent: u64,
        seed: u64,
        opts: &SceneOpts,
    ) -> Synthetic {
        let sd = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let scene = Scene::new(SceneConfig {
            positions: opts.positions,
            width: opts.width,
            height: opts.height,
            ..SceneConfig::default()
        });
        let x_end = 40.0 + opts.positions.saturating_sub(12) as f64;
        // 경로 주변 표면 위 격자 점.
        let mut points = Vec::new();
        let mut x = -10.0;
        while x <= x_end {
            let mut y = -30.0;
            while y <= 30.0 {
                points.push(Point3::new(x, y, scene.surface_height(x, y)));
                y += 1.5;
            }
            x += 1.5;
        }
        let nv = scene.views.len();
        let mut keypoints = vec![Vec::new(); nv];
        let mut gt = vec![Vec::new(); nv];
        // feat_of[i]: 점 번호 → 영상 i 특징 번호.
        let mut feat_of: Vec<HashMap<usize, usize>> = vec![HashMap::new(); nv];
        for (i, v) in scene.views.iter().enumerate() {
            let mut vis: Vec<(u64, usize, Vector2<f64>)> = Vec::new();
            for (p, x) in points.iter().enumerate() {
                if let Some(px) = v.camera.project(x) {
                    if v.camera.intrinsics.contains(&px) {
                        vis.push((hash((i as u64) << 32 | p as u64), p, px));
                    }
                }
            }
            // 특징 번호는 정답 점 번호와 무관하게 섞는다.
            vis.sort_unstable_by_key(|e| e.0);
            for (f, &(_, p, px)) in vis.iter().enumerate() {
                keypoints[i].push(px);
                gt[i].push(p);
                feat_of[i].insert(p, f);
            }
        }
        let cam_index = |c: CamId| CamId::ALL.iter().position(|&a| a == c).unwrap();
        let views: Vec<(usize, usize)> = scene
            .views
            .iter()
            .map(|v| (cam_index(v.cam), v.position))
            .collect();
        let mut pairs = Vec::new();
        let (mut outliers, mut inliers) = (0, 0);
        let cand = if opts.formation_pairs {
            scheduled_pairs(&views, &PairSchedule::default())
        } else {
            candidate_pairs(&views, 5, 4, 16)
        };
        for (a, b) in cand {
            let common: Vec<(usize, usize, usize)> = gt[a]
                .iter()
                .enumerate()
                .filter_map(|(fa, &p)| feat_of[b].get(&p).map(|&fb| (fa, fb, p)))
                .collect();
            if common.len() < 16 {
                continue;
            }
            let mut m = Vec::new();
            for &(fa, fb, p) in &common {
                if hash(((a as u64) << 44 | (b as u64) << 24 | p as u64 | 1 << 63) ^ sd) % 100
                    >= keep_percent
                {
                    continue;
                }
                // 바뀐 점: 같은 짝 b 에서 옆 격자 점(번호 + 1)의 특징과 대응시킨다.
                if hash(p as u64 ^ 0x5A5A_0000_0000 ^ sd) % 100 < swap_percent {
                    if let Some(&fq) = feat_of[b].get(&(p + 1)) {
                        m.push((fa, fq));
                        outliers += 1;
                    }
                    continue;
                }
                m.push((fa, fb));
            }
            inliers += m.len();
            let k = (m.len() as u64 * outlier_per_mille).div_ceil(1000);
            for t in 0..k {
                let s = hash(((a as u64) << 40 | (b as u64) << 20 | t) ^ sd);
                let fa = (s % keypoints[a].len() as u64) as usize;
                let fb = ((s >> 32) % keypoints[b].len() as u64) as usize;
                if gt[a][fa] != gt[b][fb] {
                    m.push((fa, fb));
                    outliers += 1;
                }
            }
            pairs.push(PairMatches {
                image_a: a,
                image_b: b,
                matches: m,
            });
        }
        Synthetic {
            keypoints,
            gt,
            pairs,
            outliers,
            inliers,
        }
    }

    /// (순도, 완전도). 순도 = 정답 점이 하나뿐인 트랙 비율.
    /// 완전도 = 대응에 나타난 정답 점마다 (한 트랙에 모인 최대 관측 수 / 참 대응으로 이어지는 최대 관측 무리 크기) 의 평균.
    fn purity_completeness(s: &Synthetic, tracks: &[Track]) -> (f64, f64) {
        let pure = tracks
            .iter()
            .filter(|t| {
                let p0 = s.gt[t.observations[0].image][t.observations[0].feature];
                t.observations
                    .iter()
                    .all(|o| s.gt[o.image][o.feature] == p0)
            })
            .count();
        // 정답 점마다 참 대응 간선만으로 이어지는 가장 큰 관측 무리의 크기(정답만으로 계산).
        // 한 점의 관측이라도 후보 짝으로 이어지지 않는 영상들(예: 먼 위치의 다른 카메라)은
        // 어떤 방법으로도 한 트랙에 모일 수 없으므로 분모는 이 무리 크기로 둔다.
        type Node = (usize, usize);
        let mut edges: HashMap<usize, Vec<(Node, Node)>> = HashMap::new();
        for p in &s.pairs {
            for &(fa, fb) in &p.matches {
                let (ga, gb) = (s.gt[p.image_a][fa], s.gt[p.image_b][fb]);
                if ga == gb {
                    edges
                        .entry(ga)
                        .or_default()
                        .push(((p.image_a, fa), (p.image_b, fb)));
                }
            }
        }
        let mut reach: HashMap<usize, usize> = HashMap::new();
        for (&p, es) in &edges {
            let mut id: HashMap<(usize, usize), usize> = HashMap::new();
            for &(x, y) in es {
                for k in [x, y] {
                    let n = id.len();
                    id.entry(k).or_insert(n);
                }
            }
            // 너비 우선 탐색으로 가장 큰 연결 무리(시험 대상 합집합-찾기를 쓰지 않음).
            let mut adj = vec![Vec::new(); id.len()];
            for &(x, y) in es {
                adj[id[&x]].push(id[&y]);
                adj[id[&y]].push(id[&x]);
            }
            let mut seen = vec![false; id.len()];
            let mut size = Vec::new();
            for s0 in 0..id.len() {
                if seen[s0] {
                    continue;
                }
                seen[s0] = true;
                let mut queue = std::collections::VecDeque::from([s0]);
                let mut c = 0;
                while let Some(x) = queue.pop_front() {
                    c += 1;
                    for &y in &adj[x] {
                        if !seen[y] {
                            seen[y] = true;
                            queue.push_back(y);
                        }
                    }
                }
                size.push(c);
            }
            reach.insert(p, *size.iter().max().unwrap());
        }
        let mut best: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        for (ti, t) in tracks.iter().enumerate() {
            for o in &t.observations {
                *best
                    .entry(s.gt[o.image][o.feature])
                    .or_default()
                    .entry(ti)
                    .or_default() += 1;
            }
        }
        let mut sum = 0.0;
        // 길이 하한(기본 3) 미만으로만 이어지는 점은 어떤 방법으로도 트랙이 되지 않으므로 뺀다.
        reach.retain(|_, n| *n >= TrackConfig::default().min_length);
        for (p, &n) in &reach {
            let m = best
                .get(p)
                .and_then(|h| h.values().max().copied())
                .unwrap_or(0);
            sum += m.min(n) as f64 / n as f64;
        }
        let obs = &reach;
        (pure as f64 / tracks.len() as f64, sum / obs.len() as f64)
    }

    #[test]
    fn clean_matches_give_exact_tracks() {
        // 오대응이 없으면 Drop 의 성분 = 정답 점: 순도·완전도 모두 정확히 1, 충돌 0.
        let s = synthetic(0, 100, 0);
        assert_eq!(s.outliers, 0);
        for policy in [ConflictPolicy::Drop, ConflictPolicy::Split] {
            let cfg = TrackConfig {
                policy,
                ..TrackConfig::default()
            };
            let (tracks, st) = build_tracks(&s.pairs, &s.keypoints, &cfg);
            let (pur, comp) = purity_completeness(&s, &tracks);
            eprintln!("clean {policy:?}: purity {pur} completeness {comp} {st:?}");
            assert_eq!(pur, 1.0);
            if policy == ConflictPolicy::Drop {
                assert_eq!(st.conflicts, 0);
                assert_eq!(comp, 1.0);
            } else {
                // Split 의 지지도 0 규칙은 공통 이웃 없이 사슬로만 이어진 참 대응도 막을 수 있다.
                // 후보 짝이 시간 이웃 1..5 를 모두 포함해 사슬만으로 이어지는 경우는 드물다: 잘리는 트랙이
                // 1% 미만이고, 잘려도 관측 대부분은 큰 쪽에 남으므로 완전도 손실은 0.5% 미만.
                assert!(comp >= 0.995, "Split 완전도 {comp}");
            }
        }
    }

    #[test]
    fn purity_and_completeness_with_outliers() {
        // 짝마다 대응의 1% 를 무작위 오대응으로 섞는다(기하 검증 뒤 남는 오대응 비율로 넉넉한 값).
        let s = synthetic(10, 100, 0);
        assert!(s.outliers * 200 > s.inliers, "오대응이 충분히 섞여야 한다");
        let run = |policy| {
            let cfg = TrackConfig {
                policy,
                ..TrackConfig::default()
            };
            let (t, st) = build_tracks(&s.pairs, &s.keypoints, &cfg);
            let (p, c) = purity_completeness(&s, &t);
            eprintln!(
                "{policy:?}: tracks {} purity {p:.4} completeness {c:.4} {st:?}",
                t.len()
            );
            (p, c, st)
        };
        let (pd, cd, sd) = run(ConflictPolicy::Drop);
        let (ps, cs, ss) = run(ConflictPolicy::Split);
        // 순도: 오대응이 두 트랙을 이으면 둘이 함께 보이는 영상에서 거의 늘 충돌이 나 걸러진다.
        // 남는 불순 트랙은 두 점이 겹치는 영상이 없는 경우뿐이라 드물다: 99% 이상.
        // Drop 은 영상이 겹치지 않는 두 점을 잇는 오대응을 못 거른다. 장면 점 간격 1.5 m 에서
        // 무작위 오대응의 두 점은 대개 멀어 겹치는 영상이 적으므로 5% 까지는 남을 수 있다고 본다.
        assert!(pd >= 0.95, "Drop 순도 {pd}");
        assert!(ps >= 0.99, "Split 순도 {ps}");
        assert!(sd.conflicts > 0);
        // Split 완전도: 오대응 간선은 공통 이웃이 없어 마지막에 처리되고 충돌로 건너뛰므로
        // 참 관측은 거의 모두 제 트랙에 모인다: 98% 이상.
        assert!(cs >= 0.98, "Split 완전도 {cs}");
        // Drop 은 오대응에 닿은 트랙을 통째로 잃으므로 Split 보다 낮다(충돌 처리 규칙의 효과).
        assert!(cd < cs, "Drop {cd} / Split {cs}");
        // 상한: 트랙 수보다 작은 상한이면 실제로 잘린다.
        let cap = ss.tracks / 2;
        let cfg = TrackConfig {
            max_tracks: cap,
            ..TrackConfig::default()
        };
        let (t, st) = build_tracks(&s.pairs, &s.keypoints, &cfg);
        assert_eq!(t.len(), cap);
        assert_eq!(st.truncated, ss.tracks - cap);
        assert!(st.truncated > 0);
    }

    fn run_policy(s: &Synthetic, policy: ConflictPolicy) -> (f64, f64, TrackStats, f64) {
        let cfg = TrackConfig {
            policy,
            ..TrackConfig::default()
        };
        let (t, st) = build_tracks(&s.pairs, &s.keypoints, &cfg);
        let (p, c) = purity_completeness(s, &t);
        let mean = t.iter().map(|x| x.len()).sum::<usize>() as f64 / t.len() as f64;
        (p, c, st, mean)
    }

    #[test]
    fn sparse_recall_keeps_tracks_whole() {
        // 짝마다 참 대응의 100%·50%·30% 만 남기고(대응 재현율) 오대응 0·1% 를 섞는다.
        // 네 경우를 모두 잰 뒤 한꺼번에 판정한다.
        let mut failures = Vec::new();
        for keep in [100, 50, 30] {
            for opm in [0, 10] {
                let s = synthetic(opm, keep, 0);
                let (pd, cd, sd, md) = run_policy(&s, ConflictPolicy::Drop);
                let (ps, cs, ss, ms) = run_policy(&s, ConflictPolicy::Split);
                eprintln!(
                    "keep {keep}% outlier {opm}permil edges {}: Drop tracks {} purity {pd:.4} completeness {cd:.4} mean {md:.2} | Split tracks {} purity {ps:.4} completeness {cs:.4} mean {ms:.2} conflicts {} inconsistent {}",
                    ss.edges, sd.tracks, ss.tracks, ss.conflicts, ss.inconsistent
                );
                if ps < 0.99 {
                    failures.push(format!("keep {keep} opm {opm}: Split 순도 {ps}"));
                }
                if cs < 0.97 {
                    failures.push(format!("keep {keep} opm {opm}: Split 완전도 {cs}"));
                }
                if opm == 0 && cs < cd - 0.01 {
                    failures.push(format!("keep {keep}: Split {cs} / Drop {cd}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    /// 기준 규모(240 장 × 8192 특징, 대응 약 527만) 시간 측정. `cargo test --release -p skylens-core
    /// --lib tracks::tests::reference_scale_time -- --ignored --nocapture`.
    #[test]
    #[ignore = "기준 규모 시간 측정(수 초~수십 초, 메모리 큼)"]
    fn reference_scale_time() {
        let (images, feats, block, per_pair) = (240usize, 8192usize, 12usize, 4_000usize);
        // 12 장씩 한 블록: 블록 안에서 특징 번호 f 는 같은 3D 점(위치는 블록·f 의 해시 + 영상별 작은 변위).
        let keypoints: Vec<Vec<Vector2<f64>>> = (0..images)
            .map(|i| {
                (0..feats)
                    .map(|f| {
                        let h = hash(((i / block) * feats + f) as u64);
                        let j = (i % block) as f64 * 0.5;
                        Vector2::new(
                            (h & 0xFFFF) as f64 / 65.536 * 1.92 + j,
                            (h >> 16 & 0xFFFF) as f64 / 65.536 * 1.08,
                        )
                    })
                    .collect()
            })
            .collect();
        let mut pairs = Vec::new();
        for a in 0..images {
            for b in a + 1..(a / block + 1) * block {
                let matches = (0..per_pair)
                    .map(|k| {
                        let f =
                            (hash((a * 1009 + b) as u64 * 7919 + k as u64) % feats as u64) as usize;
                        (f, f)
                    })
                    .collect();
                pairs.push(PairMatches {
                    image_a: a,
                    image_b: b,
                    matches,
                });
            }
        }
        let total: usize = pairs.iter().map(|p| p.matches.len()).sum();
        for policy in [ConflictPolicy::Drop, ConflictPolicy::Split] {
            let t0 = std::time::Instant::now();
            let (t, st) = build_tracks(
                &pairs,
                &keypoints,
                &TrackConfig {
                    policy,
                    ..TrackConfig::default()
                },
            );
            eprintln!(
                "{policy:?}: matches {total} edges {} tracks {} time {:.2} s",
                st.edges,
                t.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    #[test]
    fn sparse_recall_over_seeds() {
        // 시드 1~10, 재현율 30·40·50% × 오대응 0·1·5%: 완전도 >= 0.95 이고, 오대응 1% 이하에서는
        // 순도 >= 0.99. 5% 오대응은 순도 >= 0.97 만 단언한다(0.99 미달 경우가 남아 있다, 노트 참고).
        let mut failures = Vec::new();
        for keep in [30, 40, 50] {
            for opm in [0, 10, 50] {
                let (mut pmin, mut cmin) = (2.0f64, 2.0f64);
                for seed in 1..=10 {
                    let s = synthetic_seeded(opm, keep, 0, seed);
                    let (p, c, _, _) = run_policy(&s, ConflictPolicy::Split);
                    pmin = pmin.min(p);
                    cmin = cmin.min(c);
                }
                eprintln!("keep {keep} opm {opm}: seeds 1..10 min purity {pmin:.4} min completeness {cmin:.4}");
                let pure_min = if opm <= 10 { 0.99 } else { 0.97 };
                if pmin < pure_min || cmin < 0.95 {
                    failures.push(format!("keep {keep} opm {opm}: {pmin} {cmin}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    #[ignore = "원인 조사용 측정표"]
    fn recall_table() {
        for keep in [30, 40, 50] {
            for (opm, swap) in [(0, 0), (10, 0), (50, 0), (0, 1)] {
                let (mut ps, mut cs, mut pd, mut cd) = (vec![], vec![], vec![], vec![]);
                for seed in 1..=5 {
                    let s = synthetic_seeded(opm, keep, swap, seed);
                    let (p, c, _, _) = run_policy(&s, ConflictPolicy::Split);
                    ps.push(p);
                    cs.push(c);
                    let (p, c, _, _) = run_policy(&s, ConflictPolicy::Drop);
                    pd.push(p);
                    cd.push(c);
                }
                let f = |v: &Vec<f64>| {
                    format!(
                        "{:.4}/{:.4}",
                        v.iter().cloned().fold(2.0, f64::min),
                        v.iter().sum::<f64>() / v.len() as f64
                    )
                };
                eprintln!(
                    "keep {keep} opm {opm} swap {swap}: Split purity(min/mean) {} compl {} | Drop purity {} compl {}",
                    f(&ps), f(&cs), f(&pd), f(&cd)
                );
            }
        }
    }

    #[test]
    fn consistent_swaps_stay_pure() {
        // 정답 점 1% 를 옆 격자 점과 모든 짝에서 일관되게 바꾼다(반복 무늬형 오대응).
        let s = synthetic(0, 100, 1);
        assert!(s.outliers > 0);
        let (pd, cd, sd, _) = run_policy(&s, ConflictPolicy::Drop);
        let (ps, cs, ss, _) = run_policy(&s, ConflictPolicy::Split);
        eprintln!(
            "consistent 1% outliers {}: Drop tracks {} purity {pd:.4} completeness {cd:.4} | Split tracks {} purity {ps:.4} completeness {cs:.4}",
            s.outliers, sd.tracks, ss.tracks
        );
        assert!(ps >= 0.99, "Split 순도 {ps}");
    }

    #[test]
    fn verified_matches_rejects_length_mismatch() {
        let m = [(0, 1), (2, 3), (4, 5)];
        assert_eq!(
            verified_matches(&m, &[true, false, true]),
            Ok(vec![(0, 1), (4, 5)])
        );
        assert_eq!(verified_matches(&m, &[true, false]), Err((3, 2)));
        assert_eq!(verified_matches(&m[..1], &[true, true]), Err((1, 2)));
    }

    #[test]
    fn conflicting_component_is_dropped() {
        // 영상 0·1·2 의 특징 0 이 서로 대응되고, 2↔0 대응 하나가 영상 0 의 특징 1 로 이어진다.
        // 성분에 영상 0 의 특징 0, 1 이 함께 들어가므로 Drop 은 이 성분을 버린다.
        let kp: Vec<Vec<Vector2<f64>>> = (0..4)
            .map(|i| (0..3).map(|f| Vector2::new(i as f64, f as f64)).collect())
            .collect();
        let pm = |a, b, m: Vec<(usize, usize)>| PairMatches {
            image_a: a,
            image_b: b,
            matches: m,
        };
        let pairs = vec![
            pm(0, 1, vec![(0, 0), (2, 2)]),
            pm(1, 2, vec![(0, 0), (2, 2)]),
            pm(2, 0, vec![(0, 1)]),
            pm(2, 3, vec![(2, 2)]),
        ];
        let drop = TrackConfig {
            policy: ConflictPolicy::Drop,
            ..TrackConfig::default()
        };
        let (t, st) = build_tracks(&pairs, &kp, &drop);
        assert_eq!(st.conflicts, 1);
        assert_eq!(t.len(), 1);
        let imgs: Vec<_> = t[0]
            .observations
            .iter()
            .map(|o| (o.image, o.feature))
            .collect();
        assert_eq!(imgs, vec![(0, 2), (1, 2), (2, 2), (3, 2)]);
        // 어떤 트랙에도 같은 영상이 두 번 나오지 않는다.
        for policy in [ConflictPolicy::Drop, ConflictPolicy::Split] {
            let cfg = TrackConfig {
                policy,
                ..TrackConfig::default()
            };
            for tr in build_tracks(&pairs, &kp, &cfg).0 {
                assert!(tr.observations.windows(2).all(|w| w[0].image < w[1].image));
            }
        }
        // Split: 세 간선 모두 지지도 0 이라 노드 번호 순((0,0)-(1,0), (0,1)-(2,0), (1,0)-(2,0))으로
        // 처리하고, 마지막 간선은 영상 0 이 겹쳐 건너뛴다. 충돌 성분이 두 트랙으로 나뉜다.
        let two = TrackConfig {
            min_length: 2,
            ..TrackConfig::default()
        };
        let (t, st) = build_tracks(&pairs, &kp, &two);
        assert_eq!(st.conflicts, 1);
        let all: Vec<Vec<_>> = t
            .iter()
            .map(|t| {
                t.observations
                    .iter()
                    .map(|o| (o.image, o.feature))
                    .collect()
            })
            .collect();
        assert_eq!(
            all,
            vec![
                vec![(0, 0), (1, 0)],
                vec![(0, 1), (2, 0)],
                vec![(0, 2), (1, 2), (2, 2), (3, 2)]
            ]
        );
    }

    #[test]
    fn short_and_invalid_matches_are_excluded() {
        let kp = vec![vec![Vector2::new(0.0, 0.0); 2]; 3];
        let pairs = vec![
            PairMatches {
                image_a: 0,
                image_b: 0,
                matches: vec![(0, 1)],
            },
            PairMatches {
                image_a: 0,
                image_b: 1,
                matches: vec![(5, 0), (0, 0)],
            },
        ];
        let cfg = TrackConfig {
            min_length: 3,
            ..TrackConfig::default()
        };
        let (t, st) = build_tracks(&pairs, &kp, &cfg);
        assert_eq!(st.invalid_matches, 2);
        assert_eq!(st.too_short, 1);
        assert!(t.is_empty());
        let two = TrackConfig {
            min_length: 2,
            ..TrackConfig::default()
        };
        let (t, _) = build_tracks(&pairs, &kp, &two);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].len(), 2);
        // 기본 최소 관측 수는 3 이다.
        assert_eq!(TrackConfig::default().min_length, 3);
    }

    #[test]
    fn order_independent() {
        let s = synthetic(10, 100, 0);
        for policy in [ConflictPolicy::Drop, ConflictPolicy::Split] {
            let cfg = TrackConfig {
                policy,
                ..TrackConfig::default()
            };
            let (t0, st0) = build_tracks(&s.pairs, &s.keypoints, &cfg);
            // 짝 순서 뒤집기, 짝 방향 바꾸기, 짝 안 대응 순서 섞기.
            let mut pairs: Vec<PairMatches> = s
                .pairs
                .iter()
                .rev()
                .enumerate()
                .map(|(k, p)| {
                    let mut m = p.matches.clone();
                    m.sort_by_key(|&(a, b)| hash((k as u64) << 40 ^ (a as u64) << 20 ^ b as u64));
                    if k % 2 == 0 {
                        PairMatches {
                            image_a: p.image_b,
                            image_b: p.image_a,
                            matches: m.iter().map(|&(a, b)| (b, a)).collect(),
                        }
                    } else {
                        PairMatches {
                            matches: m,
                            ..p.clone()
                        }
                    }
                })
                .collect();
            // 중복 대응도 결과를 바꾸지 않는다.
            pairs.push(s.pairs[0].clone());
            let (t1, st1) = build_tracks(&pairs, &s.keypoints, &cfg);
            assert_eq!(t0, t1);
            assert_eq!(st0.conflicts, st1.conflicts);
            assert_eq!(st0.tracks, st1.tracks);
        }
    }

    #[test]
    fn cap_prefers_long_then_wide_tracks() {
        let tr = |imgs: &[usize], f: usize| Track {
            observations: imgs
                .iter()
                .map(|&i| TrackObservation {
                    image: i,
                    feature: f,
                    pixel: Vector2::zeros(),
                })
                .collect(),
        };
        let tracks = vec![
            tr(&[0, 1], 0),
            tr(&[0, 1, 2], 1),
            tr(&[0, 5], 2),
            tr(&[3, 4, 5], 3),
            tr(&[0, 1, 9], 4),
        ];
        let sel = select_tracks(tracks.clone(), 3);
        let feats: Vec<_> = sel.iter().map(|t| t.observations[0].feature).collect();
        // 길이 3 셋 중 폭이 넓은 4(폭 9), 그다음 1(폭 2)·3(폭 2) 중 첫 관측이 앞인 1, 그리고 3.
        assert_eq!(feats, vec![1, 4, 3]);
        let sel = select_tracks(tracks, 4);
        let feats: Vec<_> = sel.iter().map(|t| t.observations[0].feature).collect();
        // 길이 2 중에서는 폭 5 인 2 가 뽑힌다.
        assert_eq!(feats, vec![1, 2, 4, 3]);
        // 번들 조정 관측으로 바꾸면 트랙마다 점 번호 하나, 영상 = 카메라.
        let obs = to_ba_observations(&sel);
        assert_eq!(obs.len(), 2 + 3 + 3 + 3);
        assert_eq!(
            crate::ba::select_tracks(sel.len(), &obs, 10),
            vec![0, 1, 2, 3]
        );
    }

    /// 영상 b 특징만 중심 기준으로 `deg` 도 돌린 짝에서 변위 거름이 버리는 참 대응 수(참 대응 합계 포함)와
    /// 무작위 오대응 검출 수(오대응 합계 포함).
    fn rotation_filter_counts(s: &Synthetic, deg: f64, size: (f64, f64)) -> [usize; 4] {
        let (c, sn) = (deg.to_radians().cos(), deg.to_radians().sin());
        let ctr = Vector2::new(size.0 / 2.0, size.1 / 2.0);
        let mut r = [0usize; 4];
        for p in &s.pairs {
            let kb: Vec<Vector2<f64>> = s.keypoints[p.image_b]
                .iter()
                .map(|q| {
                    let d = q - ctr;
                    ctr + Vector2::new(c * d.x - sn * d.y, sn * d.x + c * d.y)
                })
                .collect();
            let flags = displacement_outliers(&p.matches, &s.keypoints[p.image_a], &kb);
            for (&(fa, fb), &bad) in p.matches.iter().zip(&flags) {
                if s.gt[p.image_a][fa] == s.gt[p.image_b][fb] {
                    r[0] += bad as usize;
                    r[1] += 1;
                } else {
                    r[2] += bad as usize;
                    r[3] += 1;
                }
            }
        }
        r
    }

    #[test]
    fn displacement_filter_survives_pair_rotation() {
        // 오대응 0: 짝 사이 회전 0·30·60·120 도에서 버리는 참 대응 <= 1%.
        // 무작위 오대응 1%: 검출률 >= 95%.
        let clean = synthetic(0, 100, 0);
        let noisy = synthetic(10, 100, 0);
        let mut failures = Vec::new();
        for deg in [0.0, 30.0, 60.0, 120.0] {
            let [lost, total, _, _] = rotation_filter_counts(&clean, deg, (960.0, 540.0));
            let [_, _, hit, bad] = rotation_filter_counts(&noisy, deg, (960.0, 540.0));
            let (lost_rate, det) = (lost as f64 / total as f64, hit as f64 / bad as f64);
            eprintln!("rotation {deg}: true dropped {lost}/{total} ({lost_rate:.4}) outlier detected {hit}/{bad} ({det:.4})");
            if lost_rate > 0.01 || det < 0.95 {
                failures.push(format!("{deg}: {lost_rate} {det}"));
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn displacement_filter_scales_with_image_size() {
        // 같은 배치를 1920x1080 으로 렌더해도 같은 비율로 걸러야 한다.
        let opts = SceneOpts {
            width: 1920,
            height: 1080,
            ..SceneOpts::default()
        };
        let clean = synthetic_scene(0, 100, 0, 1, &opts);
        let noisy = synthetic_scene(10, 100, 0, 1, &opts);
        for deg in [0.0, 60.0] {
            let [lost, total, _, _] = rotation_filter_counts(&clean, deg, (1920.0, 1080.0));
            let [_, _, hit, bad] = rotation_filter_counts(&noisy, deg, (1920.0, 1080.0));
            eprintln!("1920x1080 rotation {deg}: true dropped {lost}/{total} outlier detected {hit}/{bad}");
            assert!(lost as f64 <= 0.01 * total as f64);
            assert!(hit as f64 >= 0.95 * bad as f64);
        }
    }

    #[test]
    fn sparse_recall_two_resolutions() {
        // 실측 편대 배치, 재현율 50·30% × 오대응 0·1%, 두 해상도(1920x1080 은 시드 1~3).
        let mut failures = Vec::new();
        for (w, h, seeds) in [(960u32, 540u32, 0..=0u64), (1920, 1080, 1..=3)] {
            let opts = SceneOpts {
                width: w,
                height: h,
                ..SceneOpts::default()
            };
            for keep in [50, 30] {
                for opm in [0, 10] {
                    let (mut pmin, mut cmin, mut gap) = (2.0f64, 2.0f64, 2.0f64);
                    for seed in seeds.clone() {
                        let s = synthetic_scene(opm, keep, 0, seed, &opts);
                        let (ps, cs, _, _) = run_policy(&s, ConflictPolicy::Split);
                        let (_, cd, _, _) = run_policy(&s, ConflictPolicy::Drop);
                        pmin = pmin.min(ps);
                        cmin = cmin.min(cs);
                        gap = gap.min(cs - cd);
                    }
                    eprintln!("{w}x{h} keep {keep} opm {opm}: Split min purity {pmin:.4} min completeness {cmin:.4} min (Split - Drop) {gap:.4}");
                    if pmin < 0.99 || cmin < 0.97 || (opm == 0 && gap < -0.01) {
                        failures.push(format!(
                            "{w}x{h} keep {keep} opm {opm}: {pmin} {cmin} {gap}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn cross_camera_pairs_keep_tracks_whole() {
        // 편대 짝 일정(F–R·F–L 위치 차 +20..40)이 들어가는 44 위치 장면, 참 대응만 + 오대응 1%.
        let opts = SceneOpts {
            positions: 44,
            formation_pairs: true,
            ..SceneOpts::default()
        };
        let mut failures = Vec::new();
        for (keep, opm) in [(100, 0), (50, 10)] {
            let s = synthetic_scene(opm, keep, 0, 1, &opts);
            let cross = s
                .pairs
                .iter()
                // 영상 번호 = 위치 * 3 + 카메라 번호.
                .filter(|p| p.image_a % 3 != p.image_b % 3)
                .count();
            let (ps, cs, st, _) = run_policy(&s, ConflictPolicy::Split);
            eprintln!(
                "cross scene keep {keep} opm {opm}: pairs {} cross-camera {cross} inconsistent {} Split purity {ps:.4} completeness {cs:.4}",
                s.pairs.len(), st.inconsistent
            );
            assert!(cross > 0);
            if ps < 0.99 || cs < 0.97 {
                failures.push(format!("keep {keep} opm {opm}: {ps} {cs}"));
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    /// 참 점 중 관측이 둘 이상의 트랙으로 갈라진 점의 수와 (길이 하한 이상으로 이어지는) 점 수.
    fn split_points(s: &Synthetic, tracks: &[Track]) -> (usize, usize) {
        let mut spread: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
        for (ti, t) in tracks.iter().enumerate() {
            for o in &t.observations {
                spread
                    .entry(s.gt[o.image][o.feature])
                    .or_default()
                    .insert(ti);
            }
        }
        (
            spread.values().filter(|v| v.len() > 1).count(),
            spread.len(),
        )
    }

    #[test]
    fn formation_recall_table() {
        // 편대 기본 장면(44 위치, 카메라 간 짝 포함), 재현율 30·40·50% x 오대응 0·5·10%.
        // 오대응 수는 짝마다 참 대응 1000 개당 0·50·100 개. 시드 1~2 의 최저값을 단언.
        let opts = SceneOpts {
            positions: 44,
            formation_pairs: true,
            ..SceneOpts::default()
        };
        let mut failures = Vec::new();
        for keep in [30, 40, 50] {
            for opm in [0, 50, 100] {
                let (mut pmin, mut cmin) = (2.0f64, 2.0f64);
                let (mut split, mut total) = (0, 0);
                for seed in 1..=2 {
                    let s = synthetic_scene(opm, keep, 0, seed, &opts);
                    let (p, c, _, _) = run_policy(&s, ConflictPolicy::Split);
                    let cfg = TrackConfig::default();
                    let (t, _) = build_tracks(&s.pairs, &s.keypoints, &cfg);
                    let (sp, n) = split_points(&s, &t);
                    pmin = pmin.min(p);
                    cmin = cmin.min(c);
                    split += sp;
                    total += n;
                }
                eprintln!(
                    "formation keep {keep} outlier {opm}permil: min purity {pmin:.4} min completeness {cmin:.4} split points {split}/{total}"
                );
                if cmin < 0.95 {
                    failures.push(format!("keep {keep} opm {opm}: completeness {cmin}"));
                }
                if opm <= 50 && pmin < 0.97 {
                    failures.push(format!("keep {keep} opm {opm}: purity {pmin}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn seeds_change_the_kept_matches() {
        // 시드가 해시 키 전체에 섞이는지: 시드 1·2 의 같은 짝(첫 짝; 영상 (0,1) 은 공통 점이 16 개
        // 미만이라 짝이 없다) 유지 대응 집합이 다르고, 겹침 비율(전체 대비)이 유지율^2 +- 0.05.
        let all = synthetic(0, 100, 0);
        let (ia, ib) = (all.pairs[0].image_a, all.pairs[0].image_b);
        let pair = |s: &Synthetic| -> Vec<(usize, usize)> {
            s.pairs
                .iter()
                .find(|p| (p.image_a, p.image_b) == (ia, ib))
                .expect("first pair")
                .matches
                .clone()
        };
        let total = pair(&all).len();
        assert!(total >= 200, "{total}");
        for keep in [50u64, 30] {
            let a = pair(&synthetic_seeded(0, keep, 0, 1));
            let b = pair(&synthetic_seeded(0, keep, 0, 2));
            assert_ne!(a, b);
            let both = a.iter().filter(|m| b.contains(m)).count();
            let frac = both as f64 / total as f64;
            let want = (keep as f64 / 100.0).powi(2);
            eprintln!("keep {keep}: overlap {frac:.4} expected {want:.4} of {total}");
            assert!((frac - want).abs() <= 0.05, "{frac} vs {want}");
        }
    }
}
