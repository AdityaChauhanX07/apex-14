//! Property-based tests for the g_z pathway (apex-physics).
//!
//! Contract (docs/design/envelope-qss/gz-pathway.md): every normal-load site
//! gained a `*_with_gz` variant taking an imposed vertical acceleration, and the
//! pre-existing form now delegates with `g_z = GRAVITY`. The invariant these
//! properties pin: for ANY valid operating point, the default path
//! (`g_z = GRAVITY`) is BIT-IDENTICAL to the baseline — asserted via raw
//! `to_bits()`, not tolerance. This is the randomized complement to the
//! fixed-input bitwise tests inside the crate.
//!
//! Determinism: proptest default RNG + on-by-default `proptest-regressions/`.

use apex_physics::car_params::GRAVITY;
use apex_physics::CarParams;
use proptest::prelude::*;

fn base_of(calibrated: bool) -> CarParams {
    if calibrated {
        CarParams::f1_2024_calibrated()
    } else {
        CarParams::default()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// The default g_z path reproduces every baseline normal-load site bit-for-bit
    /// across random valid operating points.
    #[test]
    fn gz_default_equals_baseline_bitwise(
        calibrated in any::<bool>(),
        speed in 0.0f64..120.0,
        a_x in -30.0f64..30.0,
        a_y in -40.0f64..40.0,
        roll_frac in 0.0f64..1.0,
    ) {
        let c = base_of(calibrated);

        // point-mass grip budget
        prop_assert_eq!(
            c.max_grip_force(speed).to_bits(),
            c.max_grip_force_with_gz(speed, GRAVITY).to_bits()
        );

        // single-track axle loads (f64 and autodiff-generic mirror)
        let (f0, r0) = c.axle_loads(speed, a_x);
        let (fg, rg) = c.axle_loads_with_gz(speed, a_x, GRAVITY);
        prop_assert_eq!(f0.to_bits(), fg.to_bits());
        prop_assert_eq!(r0.to_bits(), rg.to_bits());

        let (gf0, gr0) = c.axle_loads_generic::<f64>(speed, a_x);
        let (gfg, grg) = c.axle_loads_generic_with_gz::<f64>(speed, a_x, GRAVITY);
        prop_assert_eq!(gf0.to_bits(), gfg.to_bits());
        prop_assert_eq!(gr0.to_bits(), grg.to_bits());

        // four-corner loads
        let a = c.corner_loads(speed, a_x, a_y, roll_frac);
        let b = c.corner_loads_with_gz(speed, a_x, a_y, roll_frac, GRAVITY);
        for i in 0..4 {
            prop_assert_eq!(a[i].to_bits(), b[i].to_bits());
        }
    }
}
