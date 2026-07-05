//! Property-based tests for the tire model (apex-physics/tire).
//!
//!  C1 Sign symmetry        — F_y is odd in slip angle, F_x is odd in slip
//!                            ratio (this Pacejka model has NO Sh/Sv shift or
//!                            camber terms, so oddness is structural/exact).
//!  C2 Saturation           — |F| <= mu_eff·Fz for pure axes and combined
//!                            resultant. The ceiling uses the load-sensitive
//!                            EFFECTIVE mu (see note below).
//!  C3 C1 continuity        — the smooth combined-slip surface has matching
//!                            left/right finite-difference slopes across the
//!                            friction-saturation transition (no derivative jump).
//!  C4 Load monotonicity    — at fixed slip, |F| is non-decreasing in Fz over
//!                            the increasing branch of the load-sensitivity
//!                            curve (boundary documented).
//!  C5 Thermal grip factor  — the temperature grip multiplier stays in [0, 1]
//!                            and finite over a generous temperature range.
//!
//! Determinism: proptest default RNG + on-by-default `proptest-regressions/`.

use apex_physics::tire::{TireThermalParams, TireThermalState};
use apex_physics::PacejkaTire;
use proptest::prelude::*;

/// f1_default tire with the friction coefficient and load sensitivity overridden.
fn tire_with(mu: f64, load_sens: f64) -> PacejkaTire {
    let mut t = PacejkaTire::f1_default();
    t.lateral.mu = mu;
    t.longitudinal.mu = mu;
    t.load_sensitivity = load_sens;
    t
}

// ---------------------------------------------------------------------------
// C1 — sign symmetry (odd in slip)
// ---------------------------------------------------------------------------
//
// magic_formula(slip) = D·sin(C·atan(B·slip - E·(B·slip - atan(B·slip)))). With
// no horizontal (Sh) / vertical (Sv) shift and no camber term, every factor is
// odd in `slip` while D is even, so the force is exactly odd. Asserted to 1e-12
// relative (it is bit-exact in practice).

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn lateral_force_odd_in_slip_angle(
        a in -1.5f64..1.5,
        fz in 50.0f64..20_000.0,
        mu in 0.3f64..3.0,
        ls in 0.0f64..0.3,
    ) {
        let t = tire_with(mu, ls);
        let fp = t.lateral_force(a, fz);
        let fm = t.lateral_force(-a, fz);
        let scale = fp.abs().max(1.0);
        prop_assert!(
            (fp + fm).abs() <= 1e-12 * scale,
            "F_y(-a) != -F_y(a): fp={fp} fm={fm}"
        );
    }

    #[test]
    fn longitudinal_force_odd_in_slip_ratio(
        k in -1.5f64..1.5,
        fz in 50.0f64..20_000.0,
        mu in 0.3f64..3.0,
        ls in 0.0f64..0.3,
    ) {
        let t = tire_with(mu, ls);
        let fp = t.longitudinal_force(k, fz);
        let fm = t.longitudinal_force(-k, fz);
        let scale = fp.abs().max(1.0);
        prop_assert!(
            (fp + fm).abs() <= 1e-12 * scale,
            "F_x(-k) != -F_x(k): fp={fp} fm={fm}"
        );
    }
}

// ---------------------------------------------------------------------------
// C2 — saturation
// ---------------------------------------------------------------------------
//
// The peak factor is D = mu_eff·Fz and |sin(...)| <= 1, so |F| <= mu_eff·Fz
// exactly. NOTE: the ceiling is the load-sensitive EFFECTIVE mu, not the base
// coefficient mu — for Fz < Fz_nominal load sensitivity RAISES mu above base,
// so the base-mu ceiling is intentionally exceeded there. Testing against
// effective mu is the correct, model-faithful bound.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn forces_bounded_by_effective_mu(
        a in -2.0f64..2.0,
        k in -2.0f64..2.0,
        fz in 50.0f64..20_000.0,
        mu in 0.3f64..3.0,
        ls in 0.0f64..0.3,
    ) {
        let t = tire_with(mu, ls);
        let ceil = t.effective_mu(mu, fz) * fz; // == f_max (lat and lon mu equal)
        let eps = 1e-9 * ceil.max(1.0);

        prop_assert!(t.lateral_force(a, fz).abs() <= ceil + eps);
        prop_assert!(t.longitudinal_force(k, fz).abs() <= ceil + eps);

        let r = t.combined_forces_smooth(a, k, fz);
        let resultant = (r.fx * r.fx + r.fy * r.fy).sqrt();
        prop_assert!(
            resultant <= ceil + eps,
            "combined resultant {resultant} exceeds ceiling {ceil}"
        );
    }
}

// ---------------------------------------------------------------------------
// C3 — C1 continuity of the smooth combined-slip surface
// ---------------------------------------------------------------------------
//
// Probe fy(alpha) and fx(kappa) on a fine grid spanning the friction-saturation
// transition (where combined_forces_smooth's smooth clamp lives) and require the
// left and right one-sided finite-difference slopes to agree — a discrete C1
// check. The bound is scaled to the local slope magnitude plus a floor tied to
// the peak force f_max (to cover inflection points where the slope crosses zero
// but curvature is nonzero). A hard if/else clamp would jump by O(cornering
// stiffness) here and fail; the smooth clamp's slopes differ only by O(h·f'').

const C3_H: f64 = 1e-5;
const C3_REL: f64 = 5e-3; // relative to |slope|
const C3_ABS_FRAC: f64 = 2e-4; // relative to f_max (N per unit slip)

/// Max left/right slope discrepancy of `f` at `x` (one grid point).
fn slope_jump(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let left = (f(x) - f(x - C3_H)) / C3_H;
    let right = (f(x + C3_H) - f(x)) / C3_H;
    (left - right).abs()
}

/// Returns `(max_discrepancy, bound_that_held)` over the transition grid.
fn probe_c1(tire: &PacejkaTire, fz: f64, steps: usize) -> Result<(), TestCaseError> {
    let f_max = tire.effective_mu(tire.lateral.mu, fz) * fz;
    let abs_floor = C3_ABS_FRAC * f_max;
    let span = 0.35; // covers pure -> saturated for f1 coefficients
    for i in 0..=steps {
        for j in 0..=steps {
            let a = -span + 2.0 * span * (i as f64) / (steps as f64);
            let k = -span + 2.0 * span * (j as f64) / (steps as f64);

            let fy_jump = slope_jump(|aa| tire.combined_forces_smooth(aa, k, fz).fy, a);
            let fx_jump = slope_jump(|kk| tire.combined_forces_smooth(a, kk, fz).fx, k);

            // local slope magnitudes for the relative term
            let fy_slope = (tire.combined_forces_smooth(a + C3_H, k, fz).fy
                - tire.combined_forces_smooth(a - C3_H, k, fz).fy)
                / (2.0 * C3_H);
            let fx_slope = (tire.combined_forces_smooth(a, k + C3_H, fz).fx
                - tire.combined_forces_smooth(a, k - C3_H, fz).fx)
                / (2.0 * C3_H);

            let fy_bound = abs_floor + C3_REL * fy_slope.abs();
            let fx_bound = abs_floor + C3_REL * fx_slope.abs();
            prop_assert!(
                fy_jump <= fy_bound,
                "fy slope jump {fy_jump} > {fy_bound} at (a={a}, k={k}, fz={fz})"
            );
            prop_assert!(
                fx_jump <= fx_bound,
                "fx slope jump {fx_jump} > {fx_bound} at (a={a}, k={k}, fz={fz})"
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn smooth_combined_is_c1_across_transition(fz in 500.0f64..12_000.0) {
        let tire = PacejkaTire::f1_default();
        probe_c1(&tire, fz, 24)?;
    }
}

/// Deeper, deterministic sweep (finer grid, more loads) — run on demand.
#[test]
#[ignore = "slow exhaustive C1 sweep; run with --ignored"]
fn smooth_combined_is_c1_dense() {
    let tire = PacejkaTire::f1_default();
    for &fz in &[500.0, 1500.0, 3000.0, 5000.0, 8000.0, 12_000.0] {
        probe_c1(&tire, fz, 120).expect("dense C1 sweep");
    }
}

// ---------------------------------------------------------------------------
// C4 — load-sensitivity monotonicity
// ---------------------------------------------------------------------------
//
// |F| = mu_eff(Fz)·Fz·|shape(slip)|, and shape is independent of Fz, so |F| is
// monotone in Fz iff D(Fz) = mu_eff(Fz)·Fz is. D rises then falls; the turning
// point is Fz_turn = Fz_nom·(1 + 1/load_sensitivity)/2 (for f1_default:
// 4000·(1+10)/2 = 22_000 N), beyond which load sensitivity INTENTIONALLY makes
// grip fall with load (peak-load rollover). We therefore assert monotonicity
// only on the physical increasing branch [_, 0.9·Fz_turn]; the rollover past
// Fz_turn is documented model behavior, not a violated property.

fn fz_turn(t: &PacejkaTire) -> f64 {
    t.fz_nominal * (1.0 + 1.0 / t.load_sensitivity) / 2.0
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn force_magnitude_nondecreasing_in_load(
        slip in 0.01f64..0.5,
        lateral in any::<bool>(),
        f_lo in 300.0f64..10_000.0,
        extra in 0.0f64..9_000.0,
    ) {
        let tire = PacejkaTire::f1_default();
        let cap = 0.9 * fz_turn(&tire); // stay on the increasing branch
        let fz1 = f_lo.min(cap - 1.0);
        let fz2 = (fz1 + extra).min(cap);
        prop_assume!(fz2 > fz1);

        let force = |fz: f64| {
            if lateral {
                tire.lateral_force(slip, fz).abs()
            } else {
                tire.longitudinal_force(slip, fz).abs()
            }
        };
        let f1 = force(fz1);
        let f2 = force(fz2);
        prop_assert!(
            f2 >= f1 - 1e-6 * f1.max(1.0),
            "|F| decreased with load on increasing branch: F({fz1})={f1} F({fz2})={f2}"
        );
    }
}

// ---------------------------------------------------------------------------
// C5 — thermal grip factor bounded and finite
// ---------------------------------------------------------------------------
//
// grip_factor maps surface temperature onto a dimensionless multiplier in
// [0, 1] (here mu_max == 1: it scales the base grip, it is not an absolute mu).
// grip_multiplier folds in a wear factor in [0.7, 1], so the product also stays
// in [0, 1]. Both must be finite for any temperature in a generous range.

fn thermal_params(which: u8) -> TireThermalParams {
    match which % 3 {
        0 => TireThermalParams::f1_soft(),
        1 => TireThermalParams::f1_medium(),
        _ => TireThermalParams::f1_hard(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn grip_factor_in_unit_interval(
        temp in -10_000.0f64..10_000.0,
        which in any::<u8>(),
    ) {
        let p = thermal_params(which);
        let g = p.grip_factor(temp);
        prop_assert!(g.is_finite(), "grip_factor not finite at {temp} C: {g}");
        prop_assert!((0.0..=1.0).contains(&g), "grip_factor {g} out of [0,1] at {temp} C");
    }

    #[test]
    fn grip_multiplier_in_unit_interval(
        temp in -10_000.0f64..10_000.0,
        wear in 0.0f64..1.0,
        which in any::<u8>(),
    ) {
        let p = thermal_params(which);
        let state = TireThermalState {
            surface_temp: temp,
            carcass_temp: temp,
            wear,
        };
        let g = state.grip_multiplier(&p);
        prop_assert!(g.is_finite(), "grip_multiplier not finite: {g}");
        prop_assert!((0.0..=1.0).contains(&g), "grip_multiplier {g} out of [0,1]");
    }
}
