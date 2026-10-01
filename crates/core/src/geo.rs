//! WGS84 측지 좌표 ↔ 지구중심(ECEF) ↔ 지역 동-북-위(ENU) 변환.
//!
//! 수식은 표준 WGS84 타원체의 측지↔ECEF 변환과 ECEF↔ENU 회전이다.

use crate::math::{Matrix3, Vector3};

const A: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;
const E2: f64 = F * (2.0 - F);

/// 위도·경도(도), 타원체 고도(m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geodetic {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt: f64,
}

pub fn geodetic_to_ecef(g: &Geodetic) -> Vector3<f64> {
    let (lat, lon) = (g.lat_deg.to_radians(), g.lon_deg.to_radians());
    let n = A / (1.0 - E2 * lat.sin().powi(2)).sqrt();
    Vector3::new(
        (n + g.alt) * lat.cos() * lon.cos(),
        (n + g.alt) * lat.cos() * lon.sin(),
        (n * (1.0 - E2) + g.alt) * lat.sin(),
    )
}

/// 반복법(위도 고정점 반복). 지표 근처에서 수 회에 1e-12 rad 수렴.
pub fn ecef_to_geodetic(x: &Vector3<f64>) -> Geodetic {
    let lon = x.y.atan2(x.x);
    let p = (x.x * x.x + x.y * x.y).sqrt();
    let mut lat = x.z.atan2(p * (1.0 - E2));
    let mut alt = 0.0;
    for _ in 0..10 {
        let n = A / (1.0 - E2 * lat.sin().powi(2)).sqrt();
        alt = p / lat.cos() - n;
        lat = x.z.atan2(p * (1.0 - E2 * n / (n + alt)));
    }
    Geodetic {
        lat_deg: lat.to_degrees(),
        lon_deg: lon.to_degrees(),
        alt,
    }
}

/// 원점 기준 ECEF→ENU 회전 (행 = 동, 북, 위 단위 벡터).
fn enu_rotation(origin: &Geodetic) -> Matrix3<f64> {
    let (lat, lon) = (origin.lat_deg.to_radians(), origin.lon_deg.to_radians());
    let (sl, cl, so, co) = (lat.sin(), lat.cos(), lon.sin(), lon.cos());
    Matrix3::new(-so, co, 0.0, -sl * co, -sl * so, cl, cl * co, cl * so, sl)
}

pub fn geodetic_to_enu(g: &Geodetic, origin: &Geodetic) -> Vector3<f64> {
    enu_rotation(origin) * (geodetic_to_ecef(g) - geodetic_to_ecef(origin))
}

pub fn enu_to_geodetic(e: &Vector3<f64>, origin: &Geodetic) -> Geodetic {
    ecef_to_geodetic(&(geodetic_to_ecef(origin) + enu_rotation(origin).transpose() * e))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIGIN: Geodetic = Geodetic {
        lat_deg: 37.5,
        lon_deg: 127.0,
        alt: 50.0,
    };

    #[test]
    fn ecef_known_values() {
        // 적도·본초자오선: (a, 0, 0).
        let e = geodetic_to_ecef(&Geodetic {
            lat_deg: 0.0,
            lon_deg: 0.0,
            alt: 0.0,
        });
        assert!((e - Vector3::new(A, 0.0, 0.0)).norm() < 1e-6);
        // 북극: z = b = a(1-f).
        let n = geodetic_to_ecef(&Geodetic {
            lat_deg: 90.0,
            lon_deg: 0.0,
            alt: 0.0,
        });
        assert!((n.z - A * (1.0 - F)).abs() < 1e-6);
    }

    #[test]
    fn enu_roundtrip_mm() {
        let mut worst: f64 = 0.0;
        for i in 0..100 {
            let e = Vector3::new(
                (i as f64 * 37.1) % 400.0 - 200.0,
                (i as f64 * 13.7) % 300.0 - 150.0,
                (i as f64 * 3.3) % 60.0,
            );
            let g = enu_to_geodetic(&e, &ORIGIN);
            worst = worst.max((geodetic_to_enu(&g, &ORIGIN) - e).norm());
        }
        assert!(worst < 1e-6, "왕복 오차 {worst} m");
    }

    #[test]
    fn north_and_up_directions() {
        // 위도를 조금 올리면 북(+y)으로, 고도를 올리면 위(+z)로 간다.
        let mut g = ORIGIN;
        g.lat_deg += 1e-4;
        let e = geodetic_to_enu(&g, &ORIGIN);
        assert!(e.y > 11.0 && e.y < 11.2 && e.x.abs() < 1e-3, "{e:?}");
        let mut g = ORIGIN;
        g.alt += 10.0;
        let e = geodetic_to_enu(&g, &ORIGIN);
        assert!((e - Vector3::new(0.0, 0.0, 10.0)).norm() < 1e-6);
    }
}
