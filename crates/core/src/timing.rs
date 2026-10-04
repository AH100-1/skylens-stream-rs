//! 흐름 단계별 벽시계 시간 누적. 한 번의 `run` 동안 단계 이름별로 초를 더하고 끝에 `timing.json` 으로 쓴다.
//! 정밀(BA) 쪽 단계는 다른 스레드에서 돌아 메인 흐름과 겹치므로 단계 합은 전체 벽시계보다 클 수 있다.

use std::sync::Mutex;
use std::time::Instant;

static ACC: Mutex<Vec<(&'static str, f64, u32)>> = Mutex::new(Vec::new());

/// 누적을 비운다(실행 시작 때).
pub fn reset() {
    if let Ok(mut a) = ACC.lock() {
        a.clear();
    }
}

/// 단계 `name` 에 `secs` 초를 더한다.
pub fn add(name: &'static str, secs: f64) {
    if let Ok(mut a) = ACC.lock() {
        match a.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += secs;
                e.2 += 1;
            }
            None => a.push((name, secs, 1)),
        }
    }
}

/// `f` 를 돌리고 걸린 시간을 단계 `name` 에 더한다.
pub fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    add(name, t.elapsed().as_secs_f64());
    r
}

/// 지금까지의 누적을 JSON 으로: 전체 벽시계와 단계별 `secs`·`calls`.
pub fn to_json(wall_secs: f64) -> String {
    let rows: Vec<String> = ACC
        .lock()
        .map(|a| {
            a.iter()
                .map(|(n, s, c)| {
                    format!("    {{\"stage\": \"{n}\", \"secs\": {s:.3}, \"calls\": {c}}}")
                })
                .collect()
        })
        .unwrap_or_default();
    format!(
        "{{\n  \"wall_secs\": {wall_secs:.3},\n  \"stages\": [\n{}\n  ]\n}}\n",
        rows.join(",\n")
    )
}

/// 단계별 누적을 사람이 읽는 표로: 초 큰 순, 단계 이름·초·호출 수. 한 줄에 하나.
pub fn table() -> Vec<String> {
    let mut rows: Vec<(&'static str, f64, u32)> = ACC.lock().map(|a| a.clone()).unwrap_or_default();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    rows.iter()
        .map(|(n, s, c)| format!("timing {n:<22} {s:>9.3} s {c:>6} calls"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_by_name() {
        add("t_a", 1.0);
        add("t_a", 0.5);
        let v = timed("t_b", || 7);
        assert_eq!(v, 7);
        let j = to_json(2.0);
        assert!(
            j.contains("\"stage\": \"t_a\", \"secs\": 1.500, \"calls\": 2"),
            "{j}"
        );
        assert!(j.contains("\"stage\": \"t_b\""), "{j}");
        assert!(j.contains("\"wall_secs\": 2.000"));
        let t = table();
        let (a, b) = (
            t.iter().position(|l| l.contains("t_a")).unwrap(),
            t.iter().position(|l| l.contains("t_b")).unwrap(),
        );
        assert!(a < b, "{t:?}");
    }
}
