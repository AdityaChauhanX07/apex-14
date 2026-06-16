//! Apex-14 collocation racing-line optimizer demo binary.

use std::path::Path;

use apex_optimizer::{CollocationConfig, CollocationOptimizer, OptimizationResult, SolverConfig};
use apex_physics::{qss_lap_sim, CarParams};
use apex_telemetry::{export_columns_csv, render_track_svg};
use apex_track::Track;

/// Optimize one track, print a summary, and export CSV + SVG.
/// Returns `(qss_lap_time, optimized_lap_time)`.
fn run_track(
    label: &str,
    track: &Track,
    car: &CarParams,
    collocation: CollocationConfig,
    solver: &SolverConfig,
    csv_path: &str,
    svg_path: &str,
    svg_title: &str,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let qss_lap = qss_lap_sim(track, car).lap_time;

    println!("Optimizing: {} (N={} nodes)...", label, collocation.n_nodes);
    let optimizer = CollocationOptimizer::new(collocation, track, car);
    let result = optimizer.optimize(solver);

    let top = result.speeds.iter().cloned().fold(f64::MIN, f64::max);
    let min = result.speeds.iter().cloned().fold(f64::MAX, f64::min);

    println!(
        "  Lap time: {:.3}s (QSS baseline: {:.3}s)",
        result.lap_time, qss_lap
    );
    println!("  Converged: {}", result.converged);
    println!("  Speed range: {:.1} - {:.1} km/h", min * 3.6, top * 3.6);

    export_optimized(&result, csv_path)?;
    println!("  Telemetry exported to {}", csv_path);

    render_track_svg(Path::new(svg_path), track, &result.speeds, svg_title)?;
    println!("  Track SVG exported to {}", svg_path);
    println!();

    Ok((qss_lap, result.lap_time))
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

    // --- Oval track ---
    let (oval_pts, oval_closed) = apex_track::oval_track(500.0, 80.0, 12.0, 300);
    let oval = apex_track::build_track("Oval", &oval_pts, oval_closed);
    let oval_collocation = CollocationConfig {
        n_nodes: 50,
        closed: true,
        ..CollocationConfig::default()
    };
    let oval_solver = SolverConfig {
        max_outer_iter: 30,
        max_inner_iter: 200,
        constraint_tol: 1e-3,
        print_interval: 10,
        ..SolverConfig::default()
    };
    let (oval_qss, oval_opt) = run_track(
        "Oval",
        &oval,
        &car,
        oval_collocation,
        &oval_solver,
        "opt_oval_telemetry.csv",
        "opt_oval_track.svg",
        "Apex-14 — Optimized Oval",
    )?;

    // --- Circle track ---
    let (circle_pts, circle_closed) = apex_track::circle_track(100.0, 12.0, 200);
    let circle = apex_track::build_track("Circle", &circle_pts, circle_closed);
    let circle_collocation = CollocationConfig {
        n_nodes: 30,
        closed: true,
        ..CollocationConfig::default()
    };
    let circle_solver = SolverConfig {
        max_outer_iter: 20,
        max_inner_iter: 200,
        constraint_tol: 1e-3,
        print_interval: 10,
        ..SolverConfig::default()
    };
    let (circle_qss, circle_opt) = run_track(
        "Circle",
        &circle,
        &car,
        circle_collocation,
        &circle_solver,
        "opt_circle_telemetry.csv",
        "opt_circle_track.svg",
        "Apex-14 — Optimized Circle",
    )?;

    // --- Comparison ---
    let pct = |qss: f64, opt: f64| 100.0 * (opt - qss) / qss;
    println!("--- Comparison ---");
    println!(
        "Oval:   QSS {:.3}s → Optimized {:.3}s ({:+.1}%)",
        oval_qss,
        oval_opt,
        pct(oval_qss, oval_opt)
    );
    println!(
        "Circle: QSS {:.3}s → Optimized {:.3}s ({:+.1}%)",
        circle_qss,
        circle_opt,
        pct(circle_qss, circle_opt)
    );

    Ok(())
}
