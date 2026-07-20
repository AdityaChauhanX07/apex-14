//! Regression tests for the block-tridiagonal KKT preconditioner
//! (`apex_optimizer::precond`), exercised through the envelope OCP.
//!
//! These use the **synthetic** `silverstone_circuit` deliberately: the real
//! imported circuits are gitignored (TUMFTM-derived), and a test that depends on
//! them cannot run in CI. `silverstone_tuned_reaches_tight` sets the same
//! precedent. The full real-circuit measurement lives in the `#[ignore]`d
//! `kkt_precond_sweep` harness and is written up in
//! `docs/design/dynamic-ocp/kkt-precond.md`.

use apex_optimizer::envelope_ocp::{EnvelopeOcp, EnvelopeOcpConfig, EnvelopeOcpResult};
use apex_optimizer::ipm::{IpmConfig, Preconditioner};
use apex_physics::{
    AeroModel, CarParams, Envelope, EnvelopeGridSpec, PacejkaTire, SuspensionSystem,
};
use apex_track::{build_track, silverstone_circuit, Track};

/// `N` chosen from the sweep for the **widest two-sided margin**: at 56 nodes
/// Jacobi fails catastrophically (`eq ~ 26`, `ineq ~ 28`) while BlockTridiag is
/// machine-tight (`eq ~ 2e-5`, `ineq ~ 4e-5`) — about four orders of magnitude
/// of separation on either side of the asserted bound.
const N_DISCRIMINATING: usize = 56;

/// Quantitative feasibility bound, asserted **instead of `IpmStatus`**.
///
/// This applies the CI-marginality lesson from `real-track-convergence.md`
/// Part C: near the barrier floor, libm rounding differences between platforms
/// can flip the terminal *status* between `Optimal` and `MaxIter` without the
/// solution quality being marginal at all. So assert the physically meaningful
/// property (near-feasibility), sized to sit far from both sides' measured
/// residuals.
const FEAS_BOUND: f64 = 1e-2;

fn setup() -> (Track, CarParams, Envelope) {
    let (pts, closed) = silverstone_circuit();
    let track = build_track("silverstone", &pts, closed);
    let car = CarParams::f1_2024_calibrated();
    let spec = EnvelopeGridSpec {
        v_min: 5.0,
        v_max: 90.0,
        ..EnvelopeGridSpec::default()
    };
    let env = Envelope::generate(
        &car,
        &PacejkaTire::f1_default(),
        &SuspensionSystem::f1_default(),
        &AeroModel::f1_default(),
        spec,
    )
    .expect("envelope generates");
    (track, car, env)
}

fn solve_at(
    n: usize,
    base: IpmConfig,
    track: &Track,
    car: &CarParams,
    env: &Envelope,
) -> EnvelopeOcpResult {
    let cfg = EnvelopeOcpConfig {
        n_nodes: n,
        ..EnvelopeOcpConfig::default()
    };
    EnvelopeOcp::new(cfg, track, car, env).solve(&IpmConfig {
        max_iterations: 1500,
        constraint_tol: 5e-3,
        ..base
    })
}

fn median_cg(r: &EnvelopeOcpResult) -> usize {
    let mut cg: Vec<usize> = r
        .log
        .iter()
        .map(|l| l.cg_iters)
        .filter(|&c| c > 0)
        .collect();
    cg.sort_unstable();
    assert!(!cg.is_empty(), "no Newton steps logged");
    cg[cg.len() / 2]
}

/// The headline: at a mesh where the Jacobi-preconditioned solver cannot reach
/// feasibility, the block-tridiagonal preconditioner does.
#[test]
fn blocktridiag_converges_where_jacobi_fails() {
    let (track, car, env) = setup();
    let r = solve_at(
        N_DISCRIMINATING,
        EnvelopeOcp::recommended_block_ip_config(),
        &track,
        &car,
        &env,
    );
    assert!(
        r.eq_violation <= FEAS_BOUND && r.ineq_violation <= FEAS_BOUND,
        "BlockTridiag should reach near-feasibility at N={N_DISCRIMINATING}: \
         status={:?}, eq={:.2e}, ineq={:.2e}",
        r.status,
        r.eq_violation,
        r.ineq_violation
    );
    assert!(r.lap_time.is_finite() && r.lap_time > 0.0);
}

/// Companion, in the spirit of `silverstone_untuned_still_fails_near_feasibility`:
/// Jacobi must still **fail** the same bound at the same `N`. Without this, a
/// future change that made Jacobi converge here would silently rob the test
/// above of its meaning.
#[test]
fn jacobi_still_fails_at_the_discriminating_mesh() {
    let (track, car, env) = setup();
    let r = solve_at(
        N_DISCRIMINATING,
        EnvelopeOcp::recommended_ip_config(),
        &track,
        &car,
        &env,
    );
    assert!(
        r.eq_violation > FEAS_BOUND || r.ineq_violation > FEAS_BOUND,
        "Jacobi unexpectedly reached near-feasibility at N={N_DISCRIMINATING} \
         (status={:?}, eq={:.2e}, ineq={:.2e}) -- \
         blocktridiag_converges_where_jacobi_fails no longer discriminates the \
         preconditioners; pick a new N from the kkt_precond_sweep harness",
        r.status,
        r.eq_violation,
        r.ineq_violation
    );
}

/// The block preconditioner collapses inner CG from the saturated 250-iteration
/// cap to single digits. This is the mechanism the whole change rests on, so it
/// is asserted directly rather than inferred from the convergence result.
#[test]
fn blocktridiag_collapses_inner_cg_iterations() {
    let (track, car, env) = setup();
    let block = median_cg(&solve_at(
        N_DISCRIMINATING,
        EnvelopeOcp::recommended_block_ip_config(),
        &track,
        &car,
        &env,
    ));
    let jacobi = median_cg(&solve_at(
        N_DISCRIMINATING,
        EnvelopeOcp::recommended_ip_config(),
        &track,
        &car,
        &env,
    ));

    // Measured: block ~4, jacobi ~195-250. The bounds are loose enough to absorb
    // platform numerics while still pinning an order-of-magnitude effect.
    assert!(
        block <= 25,
        "BlockTridiag median CG {block} should be small (measured ~4)"
    );
    assert!(
        jacobi >= 4 * block.max(1),
        "Jacobi median CG {jacobi} should far exceed BlockTridiag's {block}"
    );
}

/// The determinism contract must hold under the new preconditioner to exactly
/// the same standard as under Jacobi: identical inputs, bitwise-identical
/// iterate history and solution.
#[test]
fn blocktridiag_is_bitwise_deterministic() {
    let (track, car, env) = setup();
    let a = solve_at(
        32,
        EnvelopeOcp::recommended_block_ip_config(),
        &track,
        &car,
        &env,
    );
    let b = solve_at(
        32,
        EnvelopeOcp::recommended_block_ip_config(),
        &track,
        &car,
        &env,
    );

    assert_eq!(a.log.len(), b.log.len(), "log lengths differ");
    for (x, y) in a.log.iter().zip(&b.log) {
        assert_eq!(
            x.mu.to_bits(),
            y.mu.to_bits(),
            "mu differs at iter {}",
            x.iter
        );
        assert_eq!(x.primal_eq_inf.to_bits(), y.primal_eq_inf.to_bits());
        assert_eq!(x.dual_inf.to_bits(), y.dual_inf.to_bits());
        assert_eq!(x.alpha_primal.to_bits(), y.alpha_primal.to_bits());
        assert_eq!(x.cg_iters, y.cg_iters);
    }
    for (x, y) in a.speeds.iter().zip(&b.speeds) {
        assert_eq!(x.to_bits(), y.to_bits(), "speed differs");
    }
    for (x, y) in a.offsets.iter().zip(&b.offsets) {
        assert_eq!(x.to_bits(), y.to_bits(), "offset differs");
    }
}

/// Opting into `BlockTridiag` on a problem that supplies **no** block structure
/// must fall back to Jacobi and produce bit-identical results — the graceful
/// degradation the design promises.
#[test]
fn unstructured_problem_falls_back_to_jacobi_bit_identically() {
    use apex_math::{CsrBuilder, CsrMatrix};
    use apex_optimizer::ipm::solve_ipm;
    use apex_optimizer::nlp::{NlpEvaluator, NlpProblem};

    // min 0.5*||x||^2 s.t. x0 + x1 = 2, box-bounded. No `block_structure` override.
    struct Eq;
    impl NlpEvaluator for Eq {
        fn objective(&self, x: &[f64]) -> f64 {
            0.5 * (x[0] * x[0] + x[1] * x[1])
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0], x[1]]
        }
        fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] - 2.0]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            let mut b = CsrBuilder::new(1, 2);
            b.add(0, 0, 1.0);
            b.add(0, 1, 1.0);
            b.build()
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
        fn objective_hessian_vec(&self, _x: &[f64], v: &[f64]) -> Vec<f64> {
            v.to_vec()
        }
    }

    let problem = NlpProblem {
        n_vars: 2,
        n_eq: 1,
        n_ineq: 0,
        lower_bounds: vec![-10.0; 2],
        upper_bounds: vec![10.0; 2],
    };
    let jac = solve_ipm(&problem, &Eq, &[0.0, 0.0], &IpmConfig::default());
    let blk = solve_ipm(
        &problem,
        &Eq,
        &[0.0, 0.0],
        &IpmConfig {
            preconditioner: Preconditioner::BlockTridiag,
            ..IpmConfig::default()
        },
    );
    assert_eq!(jac.iterations, blk.iterations);
    for (a, b) in jac.x.iter().zip(&blk.x) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "fallback should be bit-identical to Jacobi"
        );
    }
}
