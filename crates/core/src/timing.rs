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

type Row = (&'static str, f64, u32);

fn snapshot() -> Vec<Row> {
    ACC.lock().map(|a| a.clone()).unwrap_or_default()
}

/// 지금까지의 누적을 JSON 으로: 전체 벽시계와 단계별 `secs`·`calls`.
pub fn to_json(wall_secs: f64) -> String {
    json_from(wall_secs, &snapshot())
}

/// 스냅숏 `rows` 로 JSON 을 만든다(전역 누적을 읽지 않는다).
fn json_from(wall_secs: f64, rows: &[Row]) -> String {
    let rows: Vec<String> = rows
        .iter()
        .map(|(n, s, c)| format!("    {{\"stage\": \"{n}\", \"secs\": {s:.3}, \"calls\": {c}}}"))
        .collect();
    format!(
        "{{\n  \"wall_secs\": {wall_secs:.3},\n  \"stages\": [\n{}\n  ]\n}}\n",
        rows.join(",\n")
    )
}

/// 단계별 누적을 사람이 읽는 표로: 초 큰 순, 단계 이름·초·호출 수. 한 줄에 하나.
pub fn table() -> Vec<String> {
    table_from(&snapshot())
}

/// 스냅숏 `rows` 로 표를 만든다(전역 누적을 읽지 않는다).
fn table_from(rows: &[Row]) -> Vec<String> {
    let mut rows = rows.to_vec();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    rows.iter()
        .map(|(n, s, c)| format!("timing {n:<22} {s:>9.3} s {c:>6} calls"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: [Row; 2] = [("t_b", 0.25, 1), ("t_a", 1.5, 2)];

    #[test]
    fn json_from_snapshot() {
        let j = json_from(2.0, &ROWS);
        assert!(
            j.contains("\"stage\": \"t_a\", \"secs\": 1.500, \"calls\": 2"),
            "{j}"
        );
        assert!(
            j.contains("\"stage\": \"t_b\", \"secs\": 0.250, \"calls\": 1"),
            "{j}"
        );
        assert!(j.contains("\"wall_secs\": 2.000"), "{j}");
        assert!(json_from(0.0, &[]).contains("\"stages\": [\n\n  ]"));
    }

    #[test]
    fn table_sorted_largest_first() {
        let t = table_from(&ROWS);
        assert_eq!(t.len(), 2);
        assert_eq!(
            t[0],
            format!("timing {:<22} {:>9.3} s {:>6} calls", "t_a", 1.5, 2)
        );
        assert!(t[1].contains("t_b"), "{t:?}");
    }
}
