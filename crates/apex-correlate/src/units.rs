//! Source-unit parsing and conversion to registry-canonical units.
//!
//! A mapping config declares each source column's unit as a free-form string
//! (`"km/h"`, `"percent"`, `"deg"`, …). On import we convert every sample into
//! the channel's **registry unit** (the unit named in the `# columns:` line).
//!
//! The conversion is routed through [`Unit::si_factor`]: a value in the source
//! unit is scaled to the registry's canonical base for that dimension, then
//! divided by the *target* registry unit's own `si_factor`. For units that
//! exist in the registry we delegate to `Unit::si_factor()` directly, so the
//! conversion basis is provably identical to the registry's own (e.g. `rpm` is
//! canonical-for-`rpm` on both sides, so `rpm → rpm` is exactly identity even
//! though rpm is dimensionally an angular velocity).

use apex_telemetry::Unit;

/// Multiplicative factor converting a value in `source_unit` into the registry
/// `target` unit: `value_target = value_source * conversion_factor(..)`.
///
/// Returns `None` if `source_unit` is not recognized. Recognized units are the
/// registry unit symbols (`m`, `m/s`, `km/h`, `rad/s`, `g`, `1/m`, `rad`,
/// `deg`, `N`, `s`, `°C`, `rpm`) plus common raw-telemetry spellings and the
/// non-registry units `percent` (→ 0–1 fraction) and `ms` (→ s). An empty
/// string, `"-"`, `"1"`, `"none"`, `"fraction"`, and `"ratio"` all mean
/// dimensionless identity.
pub fn conversion_factor(source_unit: &str, target: Unit) -> Option<f64> {
    let src_to_canonical = source_canonical_factor(source_unit)?;
    Some(src_to_canonical / target.si_factor())
}

/// Factor from `source_unit` to the registry-canonical base for its dimension.
/// Registry units delegate to [`Unit::si_factor`]; percent/ms are handled
/// locally relative to their dimensionless / second base.
fn source_canonical_factor(source_unit: &str) -> Option<f64> {
    let s = source_unit.trim();
    if let Some(unit) = unit_from_symbol(s) {
        return Some(unit.si_factor());
    }
    match s.to_ascii_lowercase().as_str() {
        "percent" | "pct" | "%" => Some(0.01),
        "ms" | "millisecond" | "milliseconds" => Some(1.0e-3),
        // Dimensionless identity spellings.
        "" | "-" | "1" | "none" | "fraction" | "frac" | "ratio" => Some(1.0),
        _ => None,
    }
}

/// Map a unit string onto a registry [`Unit`], accepting the canonical symbol
/// and a few common aliases. Case-sensitive for the canonical symbols (`N` is
/// newton), case-insensitive for the word aliases.
fn unit_from_symbol(s: &str) -> Option<Unit> {
    // Canonical registry symbols first (case-sensitive).
    let canonical = match s {
        "m" => Some(Unit::Meter),
        "m/s" => Some(Unit::MeterPerSecond),
        "km/h" => Some(Unit::KilometerPerHour),
        "rad/s" => Some(Unit::RadPerSecond),
        "g" => Some(Unit::G),
        "1/m" => Some(Unit::RadPerMeter),
        "rad" => Some(Unit::Radian),
        "deg" => Some(Unit::Degree),
        "N" => Some(Unit::Newton),
        "s" => Some(Unit::Second),
        "°C" => Some(Unit::Celsius),
        "rpm" => Some(Unit::Rpm),
        _ => None,
    };
    if canonical.is_some() {
        return canonical;
    }
    // Word aliases (case-insensitive).
    match s.to_ascii_lowercase().as_str() {
        "mps" | "meter_per_second" => Some(Unit::MeterPerSecond),
        "kph" | "kmh" | "kmph" => Some(Unit::KilometerPerHour),
        "degree" | "degrees" => Some(Unit::Degree),
        "radian" | "radians" => Some(Unit::Radian),
        "newton" | "newtons" => Some(Unit::Newton),
        "sec" | "second" | "seconds" => Some(Unit::Second),
        "degc" | "celsius" => Some(Unit::Celsius),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < EPS, "{a} vs {b}");
    }

    #[test]
    fn kmh_to_ms() {
        // speed's registry unit is m/s.
        let f = conversion_factor("km/h", Unit::MeterPerSecond).unwrap();
        approx(f, 1.0 / 3.6);
        approx(216.0 * f, 60.0);
    }

    #[test]
    fn deg_to_rad() {
        let f = conversion_factor("deg", Unit::Radian).unwrap();
        approx(f, std::f64::consts::PI / 180.0);
        approx(180.0 * f, std::f64::consts::PI);
    }

    #[test]
    fn percent_to_fraction() {
        // throttle's registry unit is dimensionless (Unit::None).
        let f = conversion_factor("percent", Unit::None).unwrap();
        approx(f, 0.01);
        approx(100.0 * f, 1.0);
    }

    #[test]
    fn ms_to_s() {
        let f = conversion_factor("ms", Unit::Second).unwrap();
        approx(f, 1.0e-3);
        approx(2500.0 * f, 2.5);
    }

    #[test]
    fn identity_spellings() {
        for u in ["", "-", "1", "none", "fraction", "m/s", "rad", "s"] {
            let target = match u {
                "m/s" => Unit::MeterPerSecond,
                "rad" => Unit::Radian,
                "s" => Unit::Second,
                _ => Unit::None,
            };
            approx(conversion_factor(u, target).unwrap(), 1.0);
        }
    }

    #[test]
    fn rpm_is_identity_to_rpm() {
        // rpm is dimensionally angular velocity, but the registry treats rpm as
        // canonical-for-rpm (si_factor 1.0), so rpm → rpm must be exact identity.
        approx(conversion_factor("rpm", Unit::Rpm).unwrap(), 1.0);
        approx(
            11_000.0 * conversion_factor("rpm", Unit::Rpm).unwrap(),
            11_000.0,
        );
    }

    #[test]
    fn unknown_unit_is_none() {
        assert!(conversion_factor("furlongs", Unit::Meter).is_none());
    }
}
