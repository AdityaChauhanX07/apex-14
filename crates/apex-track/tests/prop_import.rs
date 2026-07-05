//! Property-based tests for track import robustness (apex-track).
//!
//! Two families:
//!  A1  Malformed input never panics — the CSV importer and JSON loader must
//!      return `Ok`/`Err`, never unwind, for arbitrary bytes, arbitrary text,
//!      and "almost-valid" CSV (valid header + rows with NaN/inf/empty/wrong
//!      column count/huge/negative fields).
//!  A2  Well-formed input succeeds and the processed track satisfies its
//!      geometric invariants (finite arc length > 0, strictly monotone
//!      stations, finite curvature, positive widths).
//!
//! Determinism: proptest's default RNG + on-by-default failure persistence
//! (`proptest-regressions/`). Any counterexample found during development is
//! committed as a regression fixture.

use apex_track::{parse_track_json, parse_tumftm_csv, Track};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared invariant checker for a processed Track (A2).
// ---------------------------------------------------------------------------

fn assert_track_invariants(track: &Track) -> Result<(), TestCaseError> {
    prop_assert!(
        track.total_length.is_finite() && track.total_length > 0.0,
        "total_length not finite/positive: {}",
        track.total_length
    );
    prop_assert!(!track.segments.is_empty(), "no segments");

    for (i, seg) in track.segments.iter().enumerate() {
        prop_assert!(
            seg.curvature.is_finite(),
            "curvature[{i}] not finite: {}",
            seg.curvature
        );
        prop_assert!(
            seg.width_left > 0.0 && seg.width_right > 0.0,
            "width[{i}] not positive: L={} R={}",
            seg.width_left,
            seg.width_right
        );
        if i > 0 {
            prop_assert!(
                seg.s > track.segments[i - 1].s,
                "stations not strictly monotone at {i}: {} !> {}",
                seg.s,
                track.segments[i - 1].s
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A1 — malformed input never panics
// ---------------------------------------------------------------------------

/// A single CSV field: numeric, special-float text, empty, non-numeric, or
/// overflowing — the values that stress `f64::from_str` and the column logic.
fn csv_field() -> impl Strategy<Value = String> {
    prop_oneof![
        (-1.0e6f64..1.0e6).prop_map(|v| format!("{v:?}")),
        (-1.0e12f64..1.0e12).prop_map(|v| format!("{v:?}")),
        Just("NaN".to_string()),
        Just("inf".to_string()),
        Just("-inf".to_string()),
        Just(String::new()),
        Just("   ".to_string()),
        Just("1e400".to_string()), // overflows to inf on parse
        Just("1e308".to_string()),
        Just("not_a_number".to_string()),
    ]
}

/// A CSV row: 0..7 comma-joined fields (wrong column counts included).
fn csv_row() -> impl Strategy<Value = String> {
    proptest::collection::vec(csv_field(), 0..7).prop_map(|f| f.join(","))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Arbitrary bytes (UTF-8-lossy) to both importers: no panic.
    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        let s = String::from_utf8_lossy(&bytes);
        let _ = parse_track_json(&s);
        let _ = parse_tumftm_csv(&s, "fuzz");
    }

    /// Arbitrary Unicode text (control chars, weird planes) to both importers.
    #[test]
    fn arbitrary_text_never_panics(s in any::<String>()) {
        let _ = parse_track_json(&s);
        let _ = parse_tumftm_csv(&s, "fuzz");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// "Almost-valid" CSV: valid header + rows of adversarial fields. Must
    /// resolve to Ok or Err, never panic.
    #[test]
    fn almost_valid_csv_never_panics(rows in proptest::collection::vec(csv_row(), 0..40)) {
        let mut s = String::from("x_m,y_m,w_tr_right_m,w_tr_left_m\n");
        for r in &rows {
            s.push_str(r);
            s.push('\n');
        }
        let _ = parse_tumftm_csv(&s, "fuzz");
    }

    /// "Almost-valid" JSON: a points array whose entries have random/missing
    /// coordinates. Must resolve to Ok or Err, never panic.
    #[test]
    fn almost_valid_json_never_panics(
        pts in proptest::collection::vec(
            (any::<Option<f64>>(), any::<Option<f64>>()),
            0..40,
        )
    ) {
        let body: Vec<String> = pts
            .iter()
            .map(|(x, y)| {
                let xs = x.map(|v| format!("{v:?}")).unwrap_or_else(|| "0".into());
                let ys = y.map(|v| format!("{v:?}")).unwrap_or_else(|| "0".into());
                format!("{{\"x\":{xs},\"y\":{ys}}}")
            })
            .collect();
        let json = format!("{{\"name\":\"f\",\"points\":[{}]}}", body.join(","));
        let _ = parse_track_json(&json);
    }
}

// ---------------------------------------------------------------------------
// A2 — well-formed input succeeds and satisfies invariants
// ---------------------------------------------------------------------------

/// A smooth closed centerline: `n` points on a (possibly eccentric) ellipse
/// with per-point positive widths. Consecutive points are always distinct
/// (distinct angles, positive radii), so arc length is strictly monotone.
/// Returns `(points, closed)` where each point is `(x, y, width_left, width_right)`.
fn valid_track() -> impl Strategy<Value = (Vec<(f64, f64, f64, f64)>, bool)> {
    (4usize..40, 20.0f64..500.0, 20.0f64..500.0, any::<bool>())
        .prop_flat_map(|(n, rx, ry, closed)| {
            (
                Just(rx),
                Just(ry),
                Just(closed),
                proptest::collection::vec((2.0f64..15.0, 2.0f64..15.0), n),
            )
        })
        .prop_map(|(rx, ry, closed, widths)| {
            let n = widths.len();
            let pts = (0..n)
                .map(|i| {
                    let t = i as f64 / n as f64 * std::f64::consts::TAU;
                    let (wl, wr) = widths[i];
                    (rx * t.cos(), ry * t.sin(), wl, wr)
                })
                .collect();
            (pts, closed)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// The JSON loader accepts a well-formed centerline and the processed
    /// track satisfies all geometric invariants.
    #[test]
    fn wellformed_json_imports_and_is_valid((pts, closed) in valid_track()) {
        let body: Vec<String> = pts
            .iter()
            .map(|(x, y, wl, wr)| {
                format!(
                    "{{\"x\":{x:?},\"y\":{y:?},\"width_left\":{wl:?},\"width_right\":{wr:?}}}"
                )
            })
            .collect();
        let json = format!(
            "{{\"name\":\"P\",\"closed\":{closed},\"points\":[{}]}}",
            body.join(",")
        );
        let track = parse_track_json(&json).map_err(|e| {
            TestCaseError::fail(format!("well-formed JSON failed to import: {e}"))
        })?;
        assert_track_invariants(&track)?;
    }

    /// The TUMFTM CSV importer accepts the same well-formed centerline (columns
    /// `x, y, w_right, w_left`) and the processed track satisfies all invariants.
    #[test]
    fn wellformed_csv_imports_and_is_valid((pts, _closed) in valid_track()) {
        let mut csv = String::from("x_m,y_m,w_tr_right_m,w_tr_left_m\n");
        for (x, y, wl, wr) in &pts {
            csv.push_str(&format!("{x:?},{y:?},{wr:?},{wl:?}\n"));
        }
        let track = parse_tumftm_csv(&csv, "P").map_err(|e| {
            TestCaseError::fail(format!("well-formed CSV failed to import: {e}"))
        })?;
        assert_track_invariants(&track)?;
    }
}
