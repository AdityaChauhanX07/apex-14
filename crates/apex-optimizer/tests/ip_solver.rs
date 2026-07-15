//! Interior-point solver acceptance tests on the collocation problem:
//! matching Gauss-Newton where GN works, resolving the documented GN bound
//! deadlock, and determinism.

use apex_optimizer::collocation::{CollocationConfig, CollocationMethod, CollocationOptimizer};
use apex_optimizer::ipm::IpmConfig;
use apex_optimizer::GaussNewtonConfig;
use apex_physics::{qss_lap_sim, CarParams};
use apex_track::{build_track, circle_track, oval_track};

/// Collocation-tuned interior-point configuration: a small objective weight
/// (the lap-time objective is near-linear; feasibility dominates), converging
/// the dynamics defects to `1e-6`.
fn collocation_ipm() -> IpmConfig {
    IpmConfig {
        max_iterations: 300,
        obj_weight: 1e-2,
        constraint_tol: 1e-6,
        ..IpmConfig::default()
    }
}

/// On the circle — the one non-trivial track GN converges on cleanly — the
/// interior-point solver must reach a feasible trajectory whose lap time agrees
/// with GN (and with QSS). This is the "IP matches GN where GN works" check.
#[test]
fn ip_matches_gn_on_circle() {
    let (pts, closed) = circle_track(100.0, 12.0, 200);
    let track = build_track("circle", &pts, closed);
    let car = CarParams::default();
    let qss = qss_lap_sim(&track, &car).lap_time;

    let config = CollocationConfig {
        n_nodes: 30,
        closed: true,
        ..CollocationConfig::default()
    };
    let opt = CollocationOptimizer::new(config, &track, &car);

    let gn = opt.optimize_gn(&GaussNewtonConfig::default());
    assert!(gn.converged, "GN baseline should converge on the circle");

    let ip = opt.optimize_ip(&collocation_ipm());
    assert!(
        ip.eq_violation < 1e-4,
        "IP eq_violation {} should be feasible",
        ip.eq_violation
    );
    // IP lap time agrees with both GN and QSS.
    assert!(
        (ip.lap_time - gn.lap_time).abs() / gn.lap_time < 0.03,
        "IP lap {} vs GN lap {}",
        ip.lap_time,
        gn.lap_time
    );
    assert!(
        (ip.lap_time - qss).abs() / qss < 0.05,
        "IP lap {} vs QSS {}",
        ip.lap_time,
        qss
    );
}

/// THE HEADLINE: the documented Gauss-Newton bound-deadlock configuration
/// (Hermite-Simpson, N=50, calibrated car, oval — `f_drive` pins at
/// `max_drive_force` on the straights). GN floors at `eq_violation ≈ 0.3–0.7`;
/// the interior-point solver must drive it to `<= 1e-6`.
///
/// See `docs/design/gn-solver-bound-deadlock.md` and the wall-time record in
/// `docs/design/envelope-qss/ip-solver.md`.
#[test]
fn ip_resolves_gn_bound_deadlock() {
    let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
    let track = build_track("oval", &pts, closed);
    let car = CarParams::f1_2024_calibrated();

    let config = CollocationConfig {
        n_nodes: 50,
        closed: true,
        method: CollocationMethod::HermiteSimpson,
        ..CollocationConfig::default()
    };
    let opt = CollocationOptimizer::new(config, &track, &car);

    // GN deadlocks (documented). Confirm it floors far above tolerance.
    let gn = opt.optimize_gn(&GaussNewtonConfig {
        max_iterations: 100,
        constraint_tol: 1e-3,
        ..GaussNewtonConfig::default()
    });
    assert!(
        gn.eq_violation > 1e-2,
        "GN is expected to deadlock well above tolerance, got {}",
        gn.eq_violation
    );

    // IP resolves it.
    let t0 = std::time::Instant::now();
    let ip = opt.optimize_ip(&collocation_ipm());
    let wall = t0.elapsed();
    eprintln!(
        "deadlock: GN eq_violation={:.3e} -> IP eq_violation={:.3e} in {:.1} ms",
        gn.eq_violation,
        ip.eq_violation,
        wall.as_secs_f64() * 1e3
    );
    assert!(
        ip.eq_violation <= 1e-6,
        "IP must resolve the deadlock to <= 1e-6, got {}",
        ip.eq_violation
    );
    assert!(ip.converged, "IP should report convergence");
    // Every state stays finite.
    for &v in &ip.speeds {
        assert!(v.is_finite());
    }
}

/// Determinism: the interior-point solve on the (large, real) collocation
/// problem is bitwise-reproducible.
#[test]
fn ip_collocation_determinism_bitwise() {
    let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
    let track = build_track("oval", &pts, closed);
    let car = CarParams::f1_2024_calibrated();
    let config = CollocationConfig {
        n_nodes: 40,
        closed: true,
        method: CollocationMethod::HermiteSimpson,
        ..CollocationConfig::default()
    };
    let opt = CollocationOptimizer::new(config, &track, &car);
    let cfg = collocation_ipm();

    let a = opt.optimize_ip(&cfg);
    let b = opt.optimize_ip(&cfg);
    assert_eq!(a.speeds.len(), b.speeds.len());
    for (x, y) in a.speeds.iter().zip(&b.speeds) {
        assert_eq!(x.to_bits(), y.to_bits(), "speeds not bitwise-identical");
    }
    for (x, y) in a.drive_forces.iter().zip(&b.drive_forces) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "drive forces not bitwise-identical"
        );
    }
    for (x, y) in a.stations.iter().zip(&b.stations) {
        assert_eq!(x.to_bits(), y.to_bits(), "stations not bitwise-identical");
    }
}
