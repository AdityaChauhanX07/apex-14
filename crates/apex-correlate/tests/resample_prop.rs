//! Property test: resampling an already-uniform signal onto its own step is an
//! identity within FP tolerance.

use std::collections::BTreeMap;

use apex_correlate::{GridKind, Telemetry};
use apex_telemetry::ChannelId;
use proptest::prelude::*;

proptest! {
    #[test]
    fn uniform_resample_is_identity(
        n in 2usize..40,
        step in 0.1f64..50.0,
        start in -100.0f64..100.0,
        slope in -10.0f64..10.0,
        intercept in -100.0f64..100.0,
    ) {
        // Build a uniform axis and an affine signal on it.
        let axis: Vec<f64> = (0..n).map(|k| start + step * k as f64).collect();
        let signal: Vec<f64> = axis.iter().map(|x| slope * x + intercept).collect();

        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, axis.clone());
        channels.insert(ChannelId::Speed, signal.clone());
        let t = Telemetry { grid: GridKind::S, channels, metadata: Vec::new() };

        let r = t.resample_to_s(step, f64::INFINITY).unwrap();
        let rs = r.channel(ChannelId::S).unwrap();
        let rv = r.channel(ChannelId::Speed).unwrap();

        prop_assert_eq!(rs.len(), n);
        // Tolerance scales with magnitude to absorb FP error over the span.
        for k in 0..n {
            let scale = 1.0 + axis[k].abs() + signal[k].abs();
            prop_assert!((rs[k] - axis[k]).abs() <= 1e-9 * scale,
                "axis[{}]: {} vs {}", k, rs[k], axis[k]);
            prop_assert!((rv[k] - signal[k]).abs() <= 1e-9 * scale,
                "signal[{}]: {} vs {}", k, rv[k], signal[k]);
        }
    }
}
