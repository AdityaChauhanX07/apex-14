//! Integration tests for the g-g-g envelope: thread-count-independent
//! determinism and C1 continuity of the interpolated constraint across grid
//! cell boundaries.

use apex_math::Dual;
use apex_physics::{
    AeroModel, CarParams, Envelope, EnvelopeGridSpec, PacejkaTire, SuspensionSystem,
};
use proptest::prelude::*;

fn rig() -> (CarParams, PacejkaTire, SuspensionSystem, AeroModel) {
    (
        CarParams::default(),
        PacejkaTire::f1_default(),
        SuspensionSystem::f1_default(),
        AeroModel::f1_default(),
    )
}

fn spec() -> EnvelopeGridSpec {
    EnvelopeGridSpec {
        theta_res: 16,
        v_min: 10.0,
        v_max: 80.0,
        v_res: 6,
        gz_min: 9.0,
        gz_max: 12.0,
        gz_res: 4,
        max_accel: 80.0,
        coarse_steps: 48,
        bisect_tol: 1e-3,
    }
}

/// The Rayon-parallel sweep must produce byte-identical envelopes regardless of
/// the worker-thread count (order-preserving reduction over rays).
#[test]
fn byte_identical_across_thread_counts() {
    let (c, t, s, a) = rig();

    let gen_with = |threads: usize| -> Vec<u8> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            Envelope::generate(&c, &t, &s, &a, spec())
                .unwrap()
                .to_bytes()
        })
    };

    let one = gen_with(1);
    let four = gen_with(4);
    let eight = gen_with(8);
    assert_eq!(
        one, four,
        "1-thread vs 4-thread envelopes must be byte-identical"
    );
    assert_eq!(
        four, eight,
        "4-thread vs 8-thread envelopes must be byte-identical"
    );
}

proptest! {
    // Each case regenerates a small envelope (~tens of ms), so cap the count.
    #![proptest_config(ProptestConfig::with_cases(20))]

    // C1: the interpolated boundary radius `rho` has a continuous first
    // derivative across every cell boundary. We check the theta-derivative at an
    // interior theta node, approached from the left and right cells, at random
    // in-range (v, g_z). Value continuity is basis-exact; the substantive check
    // is that the two one-sided theta-derivatives agree.
    #[test]
    fn constraint_c1_across_theta_cell_boundaries(
        node in 1usize..15,
        v in 12.0f64..78.0,
        gz in 9.2f64..11.8,
    ) {
        // Regenerating per case is wasteful but keeps the test self-contained and
        // the spec is small (~1500 nodes, tens of ms).
        let (c, t, s, a) = rig();
        let env = Envelope::generate(&c, &t, &s, &a, spec()).unwrap();
        let step = std::f64::consts::TAU / spec().theta_res as f64;
        let theta_c = node as f64 * step;
        let eps = step * 1e-5;

        let left = env_rho_dual_theta(&env, theta_c - eps, v, gz);
        let right = env_rho_dual_theta(&env, theta_c + eps, v, gz);

        // Value continuity: the two one-sided samples differ only by the true
        // slope over the 2*eps gap, so scale the tolerance to that slope.
        let slope = left.dual.abs().max(right.dual.abs()) + 1.0;
        prop_assert!((left.real - right.real).abs() < slope * 4.0 * eps + 1e-6);
        // Derivative continuity — the C1 property.
        prop_assert!(
            (left.dual - right.dual).abs() < 1e-2,
            "theta-derivative jump at node {node}: {} vs {}",
            left.dual,
            right.dual
        );
    }

    // C1 across a speed cell boundary, approached from both adjacent v-cells.
    #[test]
    fn constraint_c1_across_v_cell_boundaries(
        node in 1usize..5,
        theta in 0.0f64..std::f64::consts::TAU,
        gz in 9.2f64..11.8,
    ) {
        let (c, t, s, a) = rig();
        let env = Envelope::generate(&c, &t, &s, &a, spec()).unwrap();
        let sp = spec();
        let step = (sp.v_max - sp.v_min) / (sp.v_res - 1) as f64;
        let v_c = sp.v_min + node as f64 * step;
        let eps = step * 1e-5;

        let left = env_rho_dual_v(&env, theta, v_c - eps, gz);
        let right = env_rho_dual_v(&env, theta, v_c + eps, gz);

        let slope = left.dual.abs().max(right.dual.abs()) + 1.0;
        prop_assert!((left.real - right.real).abs() < slope * 4.0 * eps + 1e-6);
        prop_assert!(
            (left.dual - right.dual).abs() < 1e-2,
            "v-derivative jump at node {node}: {} vs {}",
            left.dual,
            right.dual
        );
    }
}

/// `rho` with the theta axis seeded as the dual variable.
fn env_rho_dual_theta(env: &Envelope, theta: f64, v: f64, gz: f64) -> Dual {
    // Re-derive via rho_grad's machinery: evaluate the grid directly.
    let (val, grad) = env.rho_grad(theta, v, gz);
    let _ = val;
    Dual::new(env.rho(theta, v, gz), grad[0])
}

/// `rho` with the speed axis seeded as the dual variable.
fn env_rho_dual_v(env: &Envelope, theta: f64, v: f64, gz: f64) -> Dual {
    let (_, grad) = env.rho_grad(theta, v, gz);
    Dual::new(env.rho(theta, v, gz), grad[1])
}
