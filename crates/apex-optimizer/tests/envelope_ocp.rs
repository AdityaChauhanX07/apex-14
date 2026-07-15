//! Validation for the envelope free-trajectory OCP (`apex_optimizer::envelope_ocp`).
//!
//! These assert *objective values*, not just feasibility: the analytic circle
//! lap time against the closed form, monotone improvement over the fixed-line
//! QSS on every track, corner-cutting (the line reaches the track edge on an
//! oval), and a Silverstone cross-check — plus solver determinism and IP-log
//! sanity.
//!
//! All solves use a point-mass-like car (aero/drag/load-sensitivity off) so the
//! circle has a closed-form optimum. The OCP is opt-in (`--solver ip`); the
//! existing golden/regression suite is untouched.

use apex_optimizer::envelope_ocp::{EnvelopeOcp, EnvelopeOcpConfig};
use apex_optimizer::ipm::{IpmConfig, IpmStatus};
use apex_physics::{
    qss_lap_sim, AeroModel, CarParams, Envelope, EnvelopeGridSpec, PacejkaTire, SuspensionSystem,
    GRAVITY,
};
use apex_track::{build_track, circle_track, oval_track, silverstone_circuit, Track};

fn point_mass_car() -> (CarParams, PacejkaTire, SuspensionSystem, AeroModel) {
    let car = CarParams {
        lift_coeff: 0.0,
        drag_coeff: 0.0,
        cog_height: 0.05,
        max_drive_force: 1e7,
        max_brake_force: 1e7,
        ..CarParams::default()
    };
    let mut tire = PacejkaTire::f1_default();
    tire.load_sensitivity = 0.0;
    let mut aero = AeroModel::f1_default();
    aero.cl_front_base = 0.0;
    aero.cl_rear_base = 0.0;
    let susp = SuspensionSystem::f1_default();
    (car, tire, susp, aero)
}

fn envelope(car: &CarParams) -> Envelope {
    let (_, tire, susp, aero) = point_mass_car();
    let spec = EnvelopeGridSpec {
        v_min: 5.0,
        v_max: 90.0,
        ..EnvelopeGridSpec::default()
    };
    Envelope::generate(car, &tire, &susp, &aero, spec).unwrap()
}

/// Analytic circle: on a circle of radius `R`, half-width `w`, the fastest line
/// hugs the inner edge (radius `R - w`) at the speed the envelope allows there;
/// with aero off `rho` is speed-independent so the lap time is closed form.
#[test]
fn circle_matches_closed_form() {
    let radius = 100.0;
    let width = 8.0;
    let (pts, closed) = circle_track(radius, width, 240);
    let track = build_track("circle", &pts, closed);
    let (car, _, _, _) = point_mass_car();
    let env = envelope(&car);

    let cfg = EnvelopeOcpConfig {
        n_nodes: 60,
        ..EnvelopeOcpConfig::default()
    };
    let ocp = EnvelopeOcp::new(cfg, &track, &car, &env);
    // The circle caps `rho_max` at `3e4`, below the shared real-circuit
    // `rho_max = 3e6`. `rho_max` is a problem-scale knob (see
    // `recommended_ip_config`): at `3e6` the stiff equality penalty overwhelms
    // the objective that migrates `n` to the track edge, so the racing line
    // freezes near the centerline (n stays within ~1 m of the 4 m half-width)
    // even though the solve reports feasible. At `3e4` the line reaches the
    // inner edge and the lap matches the closed form. See
    // `docs/design/envelope-qss/real-track-convergence.md` (mesh-robustness).
    let ip = IpmConfig {
        max_iterations: 800,
        rho_max: 3e4,
        ..EnvelopeOcp::recommended_ip_config()
    };
    let r = ocp.solve(&ip);

    // closed form: eps safety margin, inner radius R - w_left.
    let eps = 0.01;
    let r_inner = radius - width / 2.0;
    let rho = env.rho(std::f64::consts::FRAC_PI_2, 30.0, GRAVITY); // v-independent (aero off)
    let rho_eff = (1.0 - eps) * rho;
    let t_star = 2.0 * std::f64::consts::PI * (r_inner / rho_eff).sqrt();

    assert_eq!(r.status, IpmStatus::Optimal, "circle should converge");
    let rel = (r.lap_time - t_star).abs() / t_star;
    assert!(
        rel < 0.02,
        "circle lap {:.4} vs analytic {:.4} (rel {:.4})",
        r.lap_time,
        t_star,
        rel
    );
    // the line hugs the inner edge (n = +w_left)
    let n_max = r.offsets.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        n_max > width / 2.0 - 0.1,
        "line should hug inner edge, n_max={n_max}"
    );

    // monotone: beats the fixed-line (centerline) QSS.
    let qss = qss_lap_sim(&track, &car);
    assert!(
        r.lap_time < qss.lap_time,
        "OCP {:.3} should beat QSS {:.3}",
        r.lap_time,
        qss.lap_time
    );

    // IP-log sanity: non-empty, mu anneals downward, and the final logged
    // point is feasible (the converged equality residual is small). eq is NOT
    // monotone across iterations — it rises during penalty ramps — so we check
    // the endpoint, not per-iteration decrease.
    assert!(!r.log.is_empty(), "IP log should be populated");
    assert!(
        r.log.first().unwrap().mu >= r.log.last().unwrap().mu,
        "mu should anneal downward"
    );
    assert!(
        r.log.last().unwrap().primal_eq_inf < 1e-3,
        "final equality residual should be feasible, got {}",
        r.log.last().unwrap().primal_eq_inf
    );
}

/// Mesh-robustness regression (docs/design/envelope-qss/real-track-convergence.md):
/// `rho_max` is a problem-scale knob, not a universal constant. The shared
/// real-circuit `rho_max = 3e6` **freezes the circle's racing line** — the stiff
/// equality penalty overwhelms the objective that would migrate `n` to the inner
/// edge — so the line stays near the centerline even though the solve reports
/// feasible. This locks the reason `circle_matches_closed_form` overrides
/// `rho_max` down to `3e4` (where the line reaches the edge, checked there). It
/// is also why no single `rho_max` unifies the gentle synthetic tracks with the
/// real circuits that need the high ceiling.
#[test]
fn circle_high_rho_freezes_line() {
    let (pts, closed) = circle_track(100.0, 8.0, 240);
    let track = build_track("circle", &pts, closed);
    let (car, _, _, _) = point_mass_car();
    let env = envelope(&car);
    let cfg = EnvelopeOcpConfig {
        n_nodes: 60,
        ..EnvelopeOcpConfig::default()
    };
    let ocp = EnvelopeOcp::new(cfg, &track, &car, &env);

    // Shared config: rho_max = 3e6.
    let hi = ocp.solve(&IpmConfig {
        max_iterations: 800,
        ..EnvelopeOcp::recommended_ip_config()
    });
    // Scale-matched config: rho_max = 3e4.
    let lo = ocp.solve(&IpmConfig {
        max_iterations: 800,
        rho_max: 3e4,
        ..EnvelopeOcp::recommended_ip_config()
    });

    let n_max = |r: &apex_optimizer::envelope_ocp::EnvelopeOcpResult| {
        r.offsets.iter().cloned().fold(f64::MIN, f64::max)
    };
    // At rho_max = 3e4 the line reaches the inner edge (half-width 4 m); at 3e6
    // it is frozen near the centerline. The gap is the whole point.
    assert!(
        n_max(&lo) > 3.5,
        "rho_max=3e4 line should reach the inner edge, n_max={}",
        n_max(&lo)
    );
    assert!(
        n_max(&hi) < 2.0,
        "rho_max=3e6 line should be frozen off the edge, n_max={}",
        n_max(&hi)
    );
}

/// Corner-cutting: on an oval the optimal line reaches *both* track edges
/// (turn-in wide, apex tight), and beats the centerline QSS.
#[test]
fn oval_corner_cutting_and_monotone() {
    let (pts, closed) = oval_track(200.0, 80.0, 12.0, 240);
    let track = build_track("oval", &pts, closed);
    let (car, _, _, _) = point_mass_car();
    let env = envelope(&car);

    let cfg = EnvelopeOcpConfig {
        n_nodes: 40,
        ..EnvelopeOcpConfig::default()
    };
    let ocp = EnvelopeOcp::new(cfg, &track, &car, &env);
    let r = ocp.solve(&EnvelopeOcp::recommended_ip_config());

    assert_eq!(r.status, IpmStatus::Optimal, "oval should converge");
    let (wl, wr) = track.width_at(0.0);
    let n_max = r.offsets.iter().cloned().fold(f64::MIN, f64::max);
    let n_min = r.offsets.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        n_max > wl - 0.2,
        "line should reach the left edge (n_max={n_max}, wl={wl})"
    );
    assert!(
        n_min < -wr + 0.2,
        "line should reach the right edge (n_min={n_min}, wr={wr})"
    );

    let qss = qss_lap_sim(&track, &car);
    assert!(
        r.lap_time < qss.lap_time,
        "OCP {:.3} should beat QSS {:.3}",
        r.lap_time,
        qss.lap_time
    );
}

/// Silverstone 2D cross-check. The synthetic circuit has curvature
/// discontinuities that a coarse trapezoidal mesh cannot resolve to tight
/// feasibility, so this asserts the run completes and improves on QSS rather
/// than a closed-form value (see `free-trajectory-ocp.md`).
#[test]
fn silverstone_cross_check() {
    let (pts, closed) = silverstone_circuit();
    let track: Track = build_track("silverstone", &pts, closed);
    let (car, _, _, _) = point_mass_car();
    let env = envelope(&car);

    let cfg = EnvelopeOcpConfig {
        n_nodes: 60,
        ..EnvelopeOcpConfig::default()
    };
    let ocp = EnvelopeOcp::new(cfg, &track, &car, &env);
    let ip = IpmConfig {
        max_iterations: 800,
        constraint_tol: 5e-3, // physical: mm on n, ~0.005 rad on xi, mm/s on v
        ..EnvelopeOcp::recommended_ip_config()
    };
    let r = ocp.solve(&ip);
    let qss = qss_lap_sim(&track, &car);

    assert!(r.lap_time.is_finite() && r.lap_time > 0.0, "finite lap");
    for v in r.speeds.iter().chain(r.offsets.iter()) {
        assert!(v.is_finite());
    }
    assert!(
        r.lap_time < qss.lap_time,
        "OCP {:.2} should improve on QSS {:.2}",
        r.lap_time,
        qss.lap_time
    );
    // the line uses the track width in the corners
    let n_max = r.offsets.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        n_max > 3.0,
        "line should move toward the edges (n_max={n_max})"
    );
}

/// The solve is bitwise deterministic across repeats (matrix-free CG + rayon,
/// sequential reductions).
#[test]
fn determinism_bitwise() {
    let (pts, closed) = circle_track(100.0, 8.0, 240);
    let track = build_track("circle", &pts, closed);
    let (car, _, _, _) = point_mass_car();
    let env = envelope(&car);
    let cfg = EnvelopeOcpConfig {
        n_nodes: 40,
        ..EnvelopeOcpConfig::default()
    };
    let ip = IpmConfig {
        max_iterations: 300,
        ..EnvelopeOcp::recommended_ip_config()
    };
    let a = EnvelopeOcp::new(cfg.clone(), &track, &car, &env).solve(&ip);
    let b = EnvelopeOcp::new(cfg, &track, &car, &env).solve(&ip);

    assert_eq!(a.offsets.len(), b.offsets.len());
    for (x, y) in a.speeds.iter().zip(&b.speeds) {
        assert_eq!(x.to_bits(), y.to_bits(), "speeds not bitwise-identical");
    }
    for (x, y) in a.offsets.iter().zip(&b.offsets) {
        assert_eq!(x.to_bits(), y.to_bits(), "offsets not bitwise-identical");
    }
    assert_eq!(a.lap_time.to_bits(), b.lap_time.to_bits());
}

/// Regression for the Part-A convergence finding
/// (docs/design/envelope-qss/real-track-convergence.md): the documented
/// "MaxIter on Silverstone" is **not** a curvature-discontinuity limit but a
/// mistuned augmented-Lagrangian schedule. The shared config
/// (`recommended_ip_config`: `al_contract = 0.1` to favour multiplier updates
/// over penalty growth, `rho_max = 3e6`) reaches *machine-tight* feasibility at
/// a coarse mesh (N=36). This uses the synthetic `silverstone_circuit` and the
/// calibrated aero-on car (as the CLI builds it) so it needs no gitignored data.
///
/// It deliberately runs the *same* synthetic circuit that `silverstone_cross_check`
/// only asserts "improves" on, and shows the shared config converges it tight.
#[test]
fn silverstone_tuned_reaches_tight() {
    let (pts, closed) = silverstone_circuit();
    let track: Track = build_track("silverstone", &pts, closed);
    let car = CarParams::f1_2024_calibrated();
    // The envelope the CLI builds: full load-sensitive, speed-dependent aero.
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
    .unwrap();

    let cfg = EnvelopeOcpConfig {
        n_nodes: 36,
        ..EnvelopeOcpConfig::default()
    };
    let ocp = EnvelopeOcp::new(cfg, &track, &car, &env);
    // The shared config (al_contract = 0.1, rho_max = 3e6) with only the
    // iteration budget and coarse-mesh tolerance overridden.
    let ip = IpmConfig {
        max_iterations: 1500,
        constraint_tol: 5e-3,
        ..EnvelopeOcp::recommended_ip_config()
    };
    let r = ocp.solve(&ip);

    assert_eq!(
        r.status,
        IpmStatus::Optimal,
        "tuned config should converge (eq={:.2e}, ineq={:.2e})",
        r.eq_violation,
        r.ineq_violation
    );
    assert!(
        r.eq_violation <= 5e-3 && r.ineq_violation <= 5e-3,
        "should be tight-feasible: eq={:.2e}, ineq={:.2e}",
        r.eq_violation,
        r.ineq_violation
    );
    // and it still beats the fixed-line QSS
    let qss = qss_lap_sim(&track, &car);
    assert!(r.lap_time < qss.lap_time && r.lap_time.is_finite());
}
