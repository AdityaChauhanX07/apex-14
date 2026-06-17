//! Apex-14 collocation racing-line optimizer demo binary.
//!
//! Runs the Augmented Lagrangian (AL), Gauss-Newton (GN), and direct
//! defect-correction solvers on each track and exports the GN result.

use std::path::Path;

use apex_optimizer::{
    CollocationConfig, CollocationOptimizer, DirectSolverConfig, GaussNewtonConfig,
    OptimizationResult, SolverConfig,
};
use apex_physics::{qss_lap_sim, qss_lap_sim_tire, CarParams, PacejkaTire};
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
