//! Apex-14 collocation racing-line optimizer demo binary.
//!
//! Runs the Augmented Lagrangian (AL), Gauss-Newton (GN), and direct
//! defect-correction solvers on each track and exports the GN result.

use std::path::Path;

use apex_optimizer::{
    CollocationConfig, CollocationOptimizer, DetailedTelemetry, DirectSolverConfig,
    GaussNewtonConfig, OptimizationResult, SolverConfig,
};
use apex_physics::{
    qss_lap_sim, qss_lap_sim_tire, AeroModel, CarParams, PacejkaTire, SuspensionSystem,
};
use apex_telemetry::{export_columns_csv, render_track_svg};
use apex_track::Track;

/// Per-track outcome: the QSS baseline and the three solvers' results.
struct Outcome {
    qss_lap: f64,
    al: OptimizationResult,
    gn: OptimizationResult,
    direct: OptimizationResult,
}

/// Optimize one track with all three solvers, print a summary, and export the
/// GN result as CSV + SVG. File names are derived from `label`.
fn run_track(
    label: &str,
    track: &Track,
    car: &CarParams,
    collocation: CollocationConfig,
    al_solver: &SolverConfig,
    gn_solver: &GaussNewtonConfig,
    direct_solver: &DirectSolverConfig,
) -> Result<Outcome, Box<dyn std::error::Error>> {
    let slug = label.to_lowercase();
    let csv_path = format!("opt_{}_telemetry.csv", slug);
    let svg_path = format!("opt_{}_track.svg", slug);
    let svg_title = format!("Apex-14 — Optimized {}", label);

    let qss_lap = qss_lap_sim(track, car).lap_time;

    println!("Optimizing: {} (N={} nodes)...", label, collocation.n_nodes);
    let optimizer = CollocationOptimizer::new(collocation, track, car);

    let al = optimizer.optimize(al_solver);
    let gn = optimizer.optimize_gn(gn_solver);
    let direct = optimizer.optimize_direct(direct_solver);

    println!("  QSS baseline: {:.3}s", qss_lap);
    println!(
        "  AL solver:     {:.3}s | eq_viol {:.2e} | converged: {}",
        al.lap_time, al.eq_violation, al.converged
    );
    println!(
        "  GN solver:     {:.3}s | eq_viol {:.2e} | converged: {}",
        gn.lap_time, gn.eq_violation, gn.converged
    );
    println!(
        "  Direct solver: {:.3}s | eq_viol {:.2e} | converged: {}",
        direct.lap_time, direct.eq_violation, direct.converged
    );

    // Export the GN result (the best-conditioned solution overall).
    export_optimized(&gn, &csv_path)?;
    println!("  Telemetry exported to {}", csv_path);
    render_track_svg(Path::new(&svg_path), track, &gn.speeds, &svg_title)?;
    println!("  Track SVG exported to {}", svg_path);
    println!();

    Ok(Outcome {
        qss_lap,
        al,
        gn,
        direct,
    })
}

/// Print a summary of the 14-DOF forward simulation.
fn print_forward_sim(tele: &DetailedTelemetry) {
    let max_abs = |xs: &[f64]| xs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let max_susp_mm = [
        &tele.suspension_fl,
        &tele.suspension_fr,
        &tele.suspension_rl,
        &tele.suspension_rr,
    ]
    .iter()
    .map(|c| max_abs(c))
    .fold(0.0_f64, f64::max)
        * 1000.0;
    let rh_min = tele
        .ride_height_front
        .iter()
        .chain(&tele.ride_height_rear)
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let rh_max = tele
        .ride_height_front
        .iter()
        .chain(&tele.ride_height_rear)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "  Phase B (full 14-DOF forward sim): {:.3}s lap",
        tele.lap_time
    );
    println!("    Top speed:         {:.1} km/h", max_abs(&tele.speed) * 3.6);
    println!("    Max lateral g:     {:.2}", max_abs(&tele.lateral_g));
    println!("    Max roll:          {:.3} deg", max_abs(&tele.roll).to_degrees());
    println!("    Max pitch:         {:.3} deg", max_abs(&tele.pitch).to_degrees());
    println!("    Max suspension:    {:.1} mm", max_susp_mm);
    println!(
        "    Ride height range: {:.1} - {:.1} mm",
        rh_min * 1000.0,
        rh_max * 1000.0
    );
}

/// Export the detailed 14-DOF forward-simulation telemetry as columnar CSV.
fn export_detailed(
    tele: &DetailedTelemetry,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let roll_deg: Vec<f64> = tele.roll.iter().map(|r| r.to_degrees()).collect();
    let pitch_deg: Vec<f64> = tele.pitch.iter().map(|p| p.to_degrees()).collect();
    let speed_kph: Vec<f64> = tele.speed.iter().map(|v| v * 3.6).collect();
    export_columns_csv(
        Path::new(path),
        &[
            ("t", &tele.time),
            ("s", &tele.s),
            ("speed_kph", &speed_kph),
            ("lateral_offset", &tele.lateral_offset),
            ("roll_deg", &roll_deg),
            ("pitch_deg", &pitch_deg),
            ("susp_fl", &tele.suspension_fl),
            ("susp_fr", &tele.suspension_fr),
            ("susp_rl", &tele.suspension_rl),
            ("susp_rr", &tele.suspension_rr),
            ("fz_fl", &tele.fz_fl),
            ("fz_fr", &tele.fz_fr),
            ("fz_rl", &tele.fz_rl),
            ("fz_rr", &tele.fz_rr),
            ("lateral_g", &tele.lateral_g),
            ("longitudinal_g", &tele.longitudinal_g),
            ("ride_height_front", &tele.ride_height_front),
            ("ride_height_rear", &tele.ride_height_rear),
        ],
    )
}

/// Export the optimized racing line as columnar CSV.
fn export_optimized(
    result: &OptimizationResult,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let v_kph: Vec<f64> = result.speeds.iter().map(|v| v * 3.6).collect();
    export_columns_csv(
        Path::new(path),
        &[
            ("s", &result.stations),
            ("n", &result.offsets),
            ("v_kph", &v_kph),
            ("f_drive", &result.drive_forces),
            ("curvature_cmd", &result.curvature_cmds),
        ],
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Apex-14 — Collocation Racing Line Optimizer");
    println!("===========================================");
    println!();

    let car = CarParams::default();

    let al_solver = SolverConfig {
        max_outer_iter: 30,
        max_inner_iter: 200,
        constraint_tol: 1e-3,
        print_interval: 0,
        ..SolverConfig::default()
    };
    let gn_solver = GaussNewtonConfig {
        max_iterations: 50,
        constraint_tol: 1e-3,
        damping: 0.5,
        regularization: 1e-4,
        print_interval: 0,
        ..GaussNewtonConfig::default()
    };
    let direct_solver = DirectSolverConfig {
        max_iterations: 200,
        constraint_tol: 1e-3,
        damping: 0.6,
        print_interval: 0,
    };

    // --- Oval track ---
    let (oval_pts, oval_closed) = apex_track::oval_track(500.0, 80.0, 12.0, 300);
    let oval_track = apex_track::build_track("Oval", &oval_pts, oval_closed);
    let oval_collocation = CollocationConfig {
        n_nodes: 50,
        closed: true,
        ..CollocationConfig::default()
    };
    let oval = run_track(
        "Oval",
        &oval_track,
        &car,
        oval_collocation.clone(),
        &al_solver,
        &gn_solver,
        &direct_solver,
    )?;

    // 7-DOF tire model on the oval: Pacejka combined-slip forces with
    // four-corner load-sensitive grip, warm-started from the tire-aware QSS so
    // the seed speed profile is feasible for the tire model.
    let tire = PacejkaTire::f1_default();
    let grip_qss = qss_lap_sim(&oval_track, &car).lap_time;
    let tire_qss = qss_lap_sim_tire(&oval_track, &car, &tire, 0.55).lap_time;
    let sd_optimizer = CollocationOptimizer::new(oval_collocation, &oval_track, &car);
    let sd_solver = GaussNewtonConfig {
        max_iterations: 30,
        ..gn_solver.clone()
    };
    let sd = sd_optimizer.optimize_seven_dof(&tire, &sd_solver);
    println!("Oval with 7-DOF tire model (tire-aware QSS warm start):");
    println!(
        "  QSS warm starts: grip-circle {:.3}s -> tire-aware {:.3}s (load sensitivity {:+.1}%)",
        grip_qss,
        tire_qss,
        100.0 * (tire_qss - grip_qss) / grip_qss
    );
    println!(
        "  7-DOF tire model: {:.3}s | eq_viol {:.2e} | converged: {}",
        sd.lap_time, sd.eq_violation, sd.converged
    );
    println!();

    // --- 14-DOF two-phase pipeline ---
    // Phase A optimizes the racing line with the ride-height-coupled 14-DOF grip
    // budget; Phase B forward-simulates the full 14-DOF model along that line.
    let suspension = SuspensionSystem::f1_default();
    let aero = AeroModel::f1_default();
    let fd_solver = GaussNewtonConfig {
        max_iterations: 30,
        ..gn_solver.clone()
    };

    // Phase A on the oval (the requested optimize_fourteen_dof run).
    let fd_oval_cfg = CollocationConfig {
        n_nodes: 50,
        closed: true,
        ..CollocationConfig::default()
    };
    let fd_oval = CollocationOptimizer::new(fd_oval_cfg, &oval_track, &car);
    let fd_oval_opt = fd_oval.optimize_fourteen_dof(&tire, &suspension, &aero, &fd_solver);
    println!("Oval with 14-DOF force model (Phase A, reduced optimization):");
    println!(
        "  14-DOF grip budget: {:.3}s | eq_viol {:.2e} | converged: {}",
        fd_oval_opt.lap_time, fd_oval_opt.eq_violation, fd_oval_opt.converged
    );
    println!();

    // Phase A + B on a tight circle. The simple path-tracking controller is
    // robust on constant-curvature cornering at moderate speed, so the forward
    // sim returns clean suspension / roll / ride-height telemetry. (High-speed
    // straights and straight-to-corner transitions, as on the oval, would need an
    // MPC-class controller to forward-track stably.)
    let (fd_circle_pts, fd_circle_closed) = apex_track::circle_track(30.0, 8.0, 200);
    let fd_circle_track = apex_track::build_track("Circle-30", &fd_circle_pts, fd_circle_closed);
    let fd_circle_cfg = CollocationConfig {
        n_nodes: 30,
        closed: true,
        ..CollocationConfig::default()
    };
    let fd_circle = CollocationOptimizer::new(fd_circle_cfg, &fd_circle_track, &car);
    let (fd_opt, fd_tele) =
        fd_circle.optimize_fourteen_dof_full(&tire, &suspension, &aero, &fd_solver);

    println!("Tight circle (R=30m) with 14-DOF two-phase pipeline:");
    println!(
        "  Phase A (reduced 14-DOF opt): {:.3}s | eq_viol {:.2e} | converged: {}",
        fd_opt.lap_time, fd_opt.eq_violation, fd_opt.converged
    );
    print_forward_sim(&fd_tele);
    export_detailed(&fd_tele, "opt_circle_14dof_telemetry.csv")?;
    println!("  Detailed telemetry exported to opt_circle_14dof_telemetry.csv");
    println!();

    // --- Circle track ---
    let (circle_pts, circle_closed) = apex_track::circle_track(100.0, 12.0, 200);
    let circle = apex_track::build_track("Circle", &circle_pts, circle_closed);
    let circle_collocation = CollocationConfig {
        n_nodes: 30,
        closed: true,
        ..CollocationConfig::default()
    };
    let circle = run_track(
        "Circle",
        &circle,
        &car,
        circle_collocation,
        &al_solver,
        &gn_solver,
        &direct_solver,
    )?;

    // --- Comparison ---
    println!("--- Comparison (lap time | eq_viol) ---");
    print_comparison("Oval", &oval);
    print_comparison("Circle", &circle);

    Ok(())
}

fn print_comparison(label: &str, o: &Outcome) {
    println!(
        "{:8} QSS {:.3}s | AL {:.3}s ({:.1e}) | GN {:.3}s ({:.1e}) | Direct {:.3}s ({:.1e})",
        format!("{}:", label),
        o.qss_lap,
        o.al.lap_time,
        o.al.eq_violation,
        o.gn.lap_time,
        o.gn.eq_violation,
        o.direct.lap_time,
        o.direct.eq_violation
    );
}
