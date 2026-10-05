//! 카메라 쌍마다 위치 차에 따른 정답 시야 겹침(진단).
use nalgebra::Vector2;
use skylens_core::synth::{CamId, Scene, SceneConfig};

/// A 영상의 격자 광선이 표면에 닿고 B 영상 안에 보이며 B 에서 가려지지 않는 비율.
fn overlap(scene: &Scene, a: CamId, pa: usize, b: CamId, pb: usize) -> f64 {
    let find = |c: CamId, p: usize| {
        scene
            .views
            .iter()
            .find(|v| v.cam == c && v.position == p)
            .unwrap()
            .camera
    };
    let (ca, cb) = (find(a, pa), find(b, pb));
    let (oa, ob) = (ca.pose.center(), cb.pose.center());
    let (w, h) = (scene.config.width as f64, scene.config.height as f64);
    let (mut tot, mut vis) = (0usize, 0usize);
    let mut y = h / 36.0;
    while y < h {
        let mut x = w / 64.0;
        while x < w {
            tot += 1;
            let dir = ca.unproject(&Vector2::new(x, y), 1.0) - oa;
            if let Some(hit) = scene.intersect(&oa, &dir.normalize()) {
                if let Some(q) = cb.project(&hit.point) {
                    if cb.intrinsics.contains(&q) {
                        let dv = hit.point - ob;
                        if scene
                            .intersect(&ob, &dv.normalize())
                            .is_some_and(|h2| (h2.t - dv.norm()).abs() < 0.3)
                        {
                            vis += 1;
                        }
                    }
                }
            }
            x += w / 32.0;
        }
        y += h / 18.0;
    }
    vis as f64 / tot as f64
}

#[test]
fn overlap_table() {
    let scene = Scene::new(SceneConfig {
        width: 480,
        height: 270,
        positions: 80,
        seed: 1,
        ..SceneConfig::default()
    });
    let base = 10usize;
    for (a, b) in [
        (CamId::F, CamId::R),
        (CamId::F, CamId::L),
        (CamId::R, CamId::L),
    ] {
        let mut row = Vec::new();
        for d in -10i32..=50 {
            let pb = base as i32 + d;
            if !(0..80).contains(&pb) {
                continue;
            }
            row.push((d, overlap(&scene, a, base, b, pb as usize)));
        }
        let best = row
            .iter()
            .cloned()
            .fold((0, -1.0), |m, v| if v.1 > m.1 { v } else { m });
        let s: Vec<String> = row
            .iter()
            .step_by(4)
            .map(|(d, o)| format!("{d}:{:.0}%", 100.0 * o))
            .collect();
        eprintln!(
            "{a:?}-{b:?} 최대 d={} 겹침 {:.1}% | {}",
            best.0,
            100.0 * best.1,
            s.join(" ")
        );
    }
}

/// 위치 간 이동 3 m(`--stride 3`)에서 일정이 고른 짝의 정답 겹침: 예전(위치 차 20..40)은 이동 거리 60 m 이상이라
/// 겹침이 거의 없고, 이동 거리로 줄인 일정은 F–R·F–L 모두 평균 15% 이상.
#[test]
fn scaled_schedule_picks_overlapping_pairs() {
    use skylens_core::matching::CrossSchedule;
    let scene = Scene::new(SceneConfig {
        width: 480,
        height: 270,
        positions: 80,
        seed: 1,
        ..SceneConfig::default()
    });
    let stride = 3usize;
    let mean = |sch: CrossSchedule, other: CamId| {
        let CrossSchedule::Formation {
            right_min,
            left_min,
            max,
            step,
        } = sch
        else {
            unreachable!()
        };
        let lo = if other == CamId::R {
            right_min
        } else {
            left_min
        };
        let mut v = Vec::new();
        let mut d = lo;
        while d <= max {
            for base in [0usize, 2] {
                if (base + d) * stride < 80 {
                    v.push(overlap(
                        &scene,
                        CamId::F,
                        base * stride,
                        other,
                        (base + d) * stride,
                    ));
                }
            }
            d += step;
        }
        v.iter().sum::<f64>() / v.len().max(1) as f64
    };
    let new = CrossSchedule::FORMATION.scaled(stride as f64);
    for other in [CamId::R, CamId::L] {
        let (old_o, new_o) = (mean(CrossSchedule::FORMATION, other), mean(new, other));
        eprintln!(
            "F-{other:?}: 예전 평균 겹침 {:.1}% 새 {:.1}%",
            100.0 * old_o,
            100.0 * new_o
        );
        assert!(new_o >= 0.15, "F-{other:?} 새 일정 겹침 {new_o}");
        assert!(new_o > old_o + 0.05, "F-{other:?} 새 {new_o} 예전 {old_o}");
    }
    assert_eq!(
        new,
        CrossSchedule::Formation {
            right_min: 7,
            left_min: 7,
            max: 13,
            step: 1
        }
    );
    assert_eq!(
        CrossSchedule::FORMATION.scaled(1.0),
        CrossSchedule::FORMATION
    );
}
