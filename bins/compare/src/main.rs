#![deny(unsafe_code)]
//! Apex-14 model-fidelity comparison: runs every model fidelity on the same
//! oval track and prints a structured comparison plus the 14-DOF chassis
//! dynamics recovered by the forward simulation.

use apex_optimizer::{
    optimize_with_refinement, CollocationConfig, CollocationOptimizer, DetailedTelemetry,
    GaussNewtonConfig, MeshRefinementConfig, OptimizationResult,
};
use apex_physics::{
    qss_lap_sim, qss_lap_sim_tire, AeroModel, CarParams, PacejkaTire, QssResult, SuspensionSystem,
};
use apex_track::{build_track, circle_track, oval_track, Track};

const G: f64 = 9.81;
const ROLL_STIFFNESS_FRONT: f64 = 0.55;

/// One row of the comparison table.
struct Row {
    name: &'static str,
    lap: f64,
    top_kph: f64,
    min_kph: f64,
    max_lat_g: f64,
    note: &'static str,
}

fn min_max(speeds: &[f64]) -> (f64, f64) {
    let top = speeds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = speeds.iter().cloned().fold(f64::INFINITY, f64::min);
    (top, min)
}

/// Summarize a QSS result. `lateral_gs` is already in g (v²·κ / g).
fn row_from_qss(name: &'static str, r: &QssResult) -> Row {
    let (top, min) = min_max(&r.speeds);
    let max_lat_g = r.lateral_gs.iter().map(|a| a.abs()).fold(0.0, f64::max);
    Row {
        name,
        lap: r.lap_time,
        top_kph: top * 3.6,
        min_kph: min * 3.6,
        max_lat_g,
        note: "",
    }
}

/// Summarize a collocation result. Lateral g uses the path (track) curvature at
/// each node — `v²·κ_track / g` — to match the QSS metric. (The optimizer's
/// curvature *command* can overshoot at unconverged transition nodes, so it is
/// not a faithful measure of the cornering load.)
fn row_from_opt(name: &'static str, r: &OptimizationResult, track: &Track) -> Row {
    let (top, min) = min_max(&r.speeds);
    let max_lat_g = r
        .speeds
        .iter()
        .zip(&r.stations)
        .map(|(v, s)| (v * v * track.curvature_at(*s)).abs())
        .fold(0.0, f64::max)
        / G;
    Row {
        name,
        lap: r.lap_time,
        top_kph: top * 3.6,
        min_kph: min * 3.6,
        max_lat_g,
        note: "",
    }
}

/// Whether a forward-sim telemetry trace represents a completed, sane lap.
fn forward_completed(tele: &DetailedTelemetry) -> bool {
    let finite = tele.speed.iter().all(|v| v.is_finite())
        && tele.roll.iter().all(|r| r.is_finite())
        && tele.lap_time.is_finite();
    let (top, _) = min_max(&tele.speed);
    finite && top < 200.0 && tele.lap_time < 250.0
}

fn row_from_forward(name: &'static str, tele: &DetailedTelemetry) -> Row {
    let (top, min) = min_max(&tele.speed);
    let max_lat_g = tele.lateral_g.iter().map(|g| g.abs()).fold(0.0, f64::max);
    if forward_completed(tele) {
        Row {
            name,
            lap: tele.lap_time,
            top_kph: top * 3.6,
            min_kph: min * 3.6,
            max_lat_g,
            note: "",
        }
    } else {
        Row {
            name,
            lap: tele.lap_time,
            top_kph: f64::NAN,
            min_kph: f64::NAN,
            max_lat_g: f64::NAN,
            note: "diverged (controller cannot track oval transitions)",
        }
    }
}

fn fmt_cell(v: f64) -> String {
    if v.is_finite() {
        format!("{:.1}", v)
    } else {
        "   --".to_string()
    }
}

fn print_table(track_len: f64, rows: &[Row]) {
    println!("=== Apex-14 — Model Fidelity Comparison ===");
    println!(
        "Track: Oval (500m straights, R=80m corners, {:.0}m total)",
        track_len
    );
    println!();
    println!(
        "{:<24} | {:>12} | {:>16} | {:>16} | {:>9}",
        "Model", "Lap Time (s)", "Top Speed (km/h)", "Min Speed (km/h)", "Max Lat g"
    );
    println!(
        "{}-+-{}-+-{}-+-{}-+-{}",
        "-".repeat(24),
        "-".repeat(12),
        "-".repeat(16),
        "-".repeat(16),
        "-".repeat(9)
    );
    for r in rows {
        let lat = if r.max_lat_g.is_finite() {
            format!("{:.2}", r.max_lat_g)
        } else {
            "  --".to_string()
        };
        println!(
            "{:<24} | {:>12.3} | {:>16} | {:>16} | {:>9}",
            r.name,
            r.lap,
            fmt_cell(r.top_kph),
            fmt_cell(r.min_kph),
            lat
        );
        if !r.note.is_empty() {
            println!("    note: {}", r.note);
        }
    }
}

fn summarize_chassis(tele: &DetailedTelemetry) -> (f64, f64, f64) {
    let max_abs = |xs: &[f64]| xs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let max_roll = max_abs(&tele.roll).to_degrees();
    let max_pitch = max_abs(&tele.pitch).to_degrees();
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
    (max_roll, max_pitch, max_susp_mm)
}

fn main() {
    let car = CarParams::default();
    let tire = PacejkaTire::f1_default();
    let suspension = SuspensionSystem::f1_default();
    let aero = AeroModel::f1_default();

    // (a) Oval track.
    let (pts, closed) = oval_track(500.0, 80.0, 12.0, 300);
    let oval: Track = build_track("Oval", &pts, closed);

    let gn = GaussNewtonConfig {
        max_iterations: 40,
        constraint_tol: 1e-3,
        print_interval: 0,
        ..GaussNewtonConfig::default()
    };
    let coll_cfg = CollocationConfig {
        n_nodes: 50,
        closed: true,
        ..CollocationConfig::default()
    };

    // (b) QSS grip circle (point-mass baseline).
    let qss_grip = qss_lap_sim(&oval, &car);
    // (c) QSS tire-aware.
    let qss_tire = qss_lap_sim_tire(&oval, &car, &tire, ROLL_STIFFNESS_FRONT);

    // (d) Collocation, point-mass.
    let opt = CollocationOptimizer::new(coll_cfg.clone(), &oval, &car);
    let coll_pm = opt.optimize_gn(&gn);
    // (e) Collocation, 7-DOF tire model.
    let coll_7dof = opt.optimize_seven_dof(&tire, &gn);
    // (f) Collocation, 14-DOF force model.
    let coll_14dof = opt.optimize_fourteen_dof(&tire, &suspension, &aero, &gn);
    // (g) 14-DOF forward simulation along the optimized oval line.
    let (_oval_opt, oval_fwd) =
        opt.optimize_fourteen_dof_full(&tire, &suspension, &aero, &gn);

    let rows = [
        row_from_qss("QSS (grip circle)", &qss_grip),
        row_from_qss("QSS (tire-aware)", &qss_tire),
        row_from_opt("Collocation (point-mass)", &coll_pm, &oval),
        row_from_opt("Collocation (7-DOF tire)", &coll_7dof, &oval),
        row_from_opt("Collocation (14-DOF)", &coll_14dof, &oval),
        row_from_forward("14-DOF Forward Sim", &oval_fwd),
    ];

    print_table(oval.total_length, &rows);

    // Key observations.
    let tire_vs_grip = 100.0 * (qss_tire.lap_time - qss_grip.lap_time) / qss_grip.lap_time;
    let fd_vs_pm = 100.0 * (coll_14dof.lap_time - coll_pm.lap_time) / coll_pm.lap_time;

    println!();
    println!("Key Observations:");
    println!(
        "- QSS tire-aware is {:+.1}% vs grip circle (load sensitivity reduces total grip)",
        tire_vs_grip
    );
    println!(
        "- 14-DOF force model shows {:+.1}% vs point-mass (ride-height aero + suspension)",
        fd_vs_pm
    );

    // The oval forward sim diverges (the simple controller cannot track the
    // high-speed straight-to-corner transitions), so the controller-tracking
    // margin is measured on a constant-curvature circle where it is stable.
    let (cpts, cclosed) = circle_track(30.0, 8.0, 200);
    let circle = build_track("Circle-30", &cpts, cclosed);
    let circle_cfg = CollocationConfig {
        n_nodes: 30,
        closed: true,
        ..CollocationConfig::default()
    };
    let circle_opt = CollocationOptimizer::new(circle_cfg, &circle, &car);
    let (circle_phase_a, circle_fwd) =
        circle_opt.optimize_fourteen_dof_full(&tire, &suspension, &aero, &gn);

    let fwd_vs_opt =
        100.0 * (circle_fwd.lap_time - circle_phase_a.lap_time) / circle_phase_a.lap_time;
    if forward_completed(&oval_fwd) {
        let oval_margin = 100.0 * (oval_fwd.lap_time - coll_14dof.lap_time) / coll_14dof.lap_time;
        println!(
            "- Forward sim is {:+.1}% vs optimized on the oval (controller tracking margin)",
            oval_margin
        );
    } else {
        println!(
            "- Forward sim is {:+.1}% vs optimized on a tight circle (controller tracking margin;",
            fwd_vs_opt
        );
        println!("  the oval forward sim diverges on straight-to-corner transitions)");
    }

    // (h) Chassis dynamics, from whichever forward sim completed.
    let (chassis_src, chassis) = if forward_completed(&oval_fwd) {
        ("oval", summarize_chassis(&oval_fwd))
    } else {
        ("R=30 circle", summarize_chassis(&circle_fwd))
    };
    println!();
    println!("14-DOF Chassis Dynamics (from {} forward sim):", chassis_src);
    println!("  Max roll:    {:.3} deg", chassis.0);
    println!("  Max pitch:   {:.3} deg", chassis.1);
    println!("  Max susp:    {:.1} mm", chassis.2);

    // Mesh refinement demonstration (coarse -> fine warm starting).
    let mr_cfg = MeshRefinementConfig {
        mesh_sequence: vec![25, 50],
        ..MeshRefinementConfig::default()
    };
    let refined = optimize_with_refinement(&oval, &car, &mr_cfg);
    println!();
    println!("Mesh refinement (point-mass collocation, coarse -> fine):");
    for lvl in &refined.level_results {
        println!(
            "  N={:>3}: {:.3}s | eq_viol {:.2e} | converged: {}",
            lvl.n_nodes, lvl.lap_time, lvl.eq_violation, lvl.converged
        );
    }
}
