//! Measurement harness for the block-tridiagonal KKT preconditioner.
//!
//! This is the instrument that produced the CG-iteration-versus-`N` table in
//! `docs/design/dynamic-ocp/kkt-precond.md`. It is `#[ignore]`d because it is a
//! multi-minute sweep over real-circuit data that is **gitignored** (TUMFTM-derived,
//! see `tracks/README.md`), so it cannot run in CI. It skips cleanly when the
//! track file is absent rather than failing — the same discipline
//! `silverstone_tuned_reaches_tight` follows by using the synthetic circuit.
//!
//! Run with:
//! ```text
//! cargo test -p apex-optimizer --release --test kkt_precond_sweep -- --ignored --nocapture
//! ```

use apex_optimizer::envelope_ocp::{EnvelopeOcp, EnvelopeOcpConfig};
use apex_optimizer::ipm::{IpmConfig, Preconditioner};
use apex_physics::{
    qss_lap_sim, AeroModel, CarParams, Envelope, EnvelopeGridSpec, PacejkaTire, SuspensionSystem,
};
use apex_track::Track;

/// Load the real (imported, gitignored) Silverstone, or `None` if absent.
fn real_silverstone() -> Option<Track> {
    let path = std::path::Path::new("../../tracks/silverstone.json");
    if !path.exists() {
        return None;
    }
    apex_track::load_track_json(path).ok()
}

/// The CLI's calibrated setup: `f1_2024_calibrated` car with the aero-bridged
/// envelope, matching `real-track-convergence.md` §B.3.
fn calibrated_setup() -> (CarParams, Envelope) {
    let car = CarParams::f1_2024_calibrated();
    let aero = AeroModel::f1_default().scaled_for_car(&car);
    let spec = EnvelopeGridSpec {
        v_min: 5.0,
        v_max: 90.0,
        ..EnvelopeGridSpec::default()
    };
    let env = Envelope::generate(
        &car,
        &PacejkaTire::f1_default(),
        &SuspensionSystem::f1_default(),
        &aero,
        spec,
    )
    .expect("envelope generates");
    (car, env)
}

struct Row {
    n: usize,
    precond: &'static str,
    status: String,
    eq: f64,
    ineq: f64,
    lap: f64,
    cg_median: usize,
    cg_max: usize,
    cg_steps: usize,
    outer: usize,
    wall_ms: u128,
}

fn run_one(track: &Track, car: &CarParams, env: &Envelope, n: usize, p: Preconditioner) -> Row {
    let cfg = EnvelopeOcpConfig {
        n_nodes: n,
        ..EnvelopeOcpConfig::default()
    };
    let ocp = EnvelopeOcp::new(cfg, track, car, env);
    // Each preconditioner gets its own recommended config: BlockTridiag needs
    // the larger `reg` (see `recommended_block_ip_config`), because an exact
    // solve resolves the JeqᵀJeq null space that truncated CG regularized away.
    let base = match p {
        Preconditioner::Jacobi => EnvelopeOcp::recommended_ip_config(),
        Preconditioner::BlockTridiag => EnvelopeOcp::recommended_block_ip_config(),
    };
    let ip = IpmConfig {
        max_iterations: 1500,
        ..base
    };
    let t0 = std::time::Instant::now();
    let r = ocp.solve(&ip);
    let wall_ms = t0.elapsed().as_millis();

    // Only rows from genuine Newton steps carry a CG count; schedule-advance and
    // termination rows log `cg_iters: 0`.
    let mut cg: Vec<usize> = r
        .log
        .iter()
        .map(|l| l.cg_iters)
        .filter(|&c| c > 0)
        .collect();
    cg.sort_unstable();
    let cg_median = if cg.is_empty() { 0 } else { cg[cg.len() / 2] };
    let cg_max = cg.last().copied().unwrap_or(0);

    Row {
        n,
        precond: match p {
            Preconditioner::Jacobi => "Jacobi",
            Preconditioner::BlockTridiag => "BlockTridiag",
        },
        status: format!("{:?}", r.status),
        eq: r.eq_violation,
        ineq: r.ineq_violation,
        lap: r.lap_time,
        cg_median,
        cg_max,
        cg_steps: cg.len(),
        outer: r.iterations,
        wall_ms,
    }
}

#[test]
#[ignore = "multi-minute sweep over gitignored real-circuit data; produces the kkt-precond.md table"]
fn silverstone_cg_scaling_sweep() {
    let Some(track) = real_silverstone() else {
        eprintln!("SKIP: tracks/silverstone.json absent (gitignored import)");
        return;
    };
    let (car, env) = calibrated_setup();
    let qss = qss_lap_sim(&track, &car);
    println!(
        "\nreal Silverstone, calibrated car (aero-bridged). fixed-line QSS = {:.3} s",
        qss.lap_time
    );
    println!(
        "\n| N | precond | status | eq | ineq | lap (s) | CG med | CG max | steps | outer | wall (ms) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    for &n in &[24usize, 32, 40, 48, 64, 96, 128] {
        for &p in &[Preconditioner::Jacobi, Preconditioner::BlockTridiag] {
            let r = run_one(&track, &car, &env, n, p);
            println!(
                "| {} | {} | {} | {:.2e} | {:.2e} | {:.3} | {} | {} | {} | {} | {} |",
                r.n,
                r.precond,
                r.status,
                r.eq,
                r.ineq,
                r.lap,
                r.cg_median,
                r.cg_max,
                r.cg_steps,
                r.outer,
                r.wall_ms
            );
        }
    }
}

/// Mesh-convergence re-sweep: does the lap-time objective stabilize with `N`
/// once the conditioning wall is out of the way? The deferred bar from
/// `real-track-convergence.md` §B.6 is "monotone with the last delta < 1 %".
#[test]
#[ignore = "multi-minute sweep over gitignored real-circuit data"]
fn silverstone_mesh_convergence_blocktridiag() {
    let Some(track) = real_silverstone() else {
        eprintln!("SKIP: tracks/silverstone.json absent (gitignored import)");
        return;
    };
    let (car, env) = calibrated_setup();
    println!("\n| N | precond | status | eq | ineq | lap (s) | delta vs prev |");
    println!("|---|---|---|---|---|---|---|");
    for &p in &[Preconditioner::Jacobi, Preconditioner::BlockTridiag] {
        let mut prev: Option<f64> = None;
        for &n in &[24usize, 32, 40, 48, 64, 96, 128, 192, 256] {
            let r = run_one(&track, &car, &env, n, p);
            let delta = prev
                .map(|q: f64| format!("{:+.2} %", 100.0 * (r.lap - q) / q))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "| {} | {} | {} | {:.2e} | {:.2e} | {:.3} | {} |",
                r.n, r.precond, r.status, r.eq, r.ineq, r.lap, delta
            );
            prev = Some(r.lap);
        }
    }
}
