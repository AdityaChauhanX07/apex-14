#![deny(unsafe_code)]
//! Apex-14 Quasi-Steady-State lap simulator demo binary.

use std::path::Path;

use apex_integrator::{rk45_adaptive_step, AdaptiveConfig};
use apex_physics::{
    qss_lap_sim, AeroModel, CarParams, FourteenDofModel, PacejkaTire, QssResult, SuspensionSystem,
};
use apex_telemetry::{export_qss_csv, render_track_svg};
use apex_track::{
    build_track, circle_track, monza_circuit, normalize_angle, oval_track, silverstone_circuit,
    Track,
};

/// Standard gravity (m/s²) for reporting lateral acceleration in g.
const GRAVITY: f64 = 9.81;

/// Summary statistics for a single QSS run.
struct LapStats {
    lap_time: f64,
    top_speed: f64,
    min_speed: f64,
    max_lat_g: f64,
    max_lon_g: f64,
}

fn summarize(result: &QssResult) -> LapStats {
    let top_speed = result.speeds.iter().cloned().fold(f64::MIN, f64::max);
    let min_speed = result.speeds.iter().cloned().fold(f64::MAX, f64::min);
    let max_lat_g = result
        .lateral_gs
        .iter()
        .map(|g| g.abs())
        .fold(0.0, f64::max);
    let max_lon_g = result
        .longitudinal_gs
        .iter()
        .map(|g| g.abs())
        .fold(0.0, f64::max);

    LapStats {
        lap_time: result.lap_time,
        top_speed,
        min_speed,
        max_lat_g,
        max_lon_g,
    }
}

fn print_results(stats: &LapStats) {
    println!("  Lap time:            {:.3} s", stats.lap_time);
    println!("  Top speed:           {:.1} km/h", stats.top_speed * 3.6);
    println!("  Min speed:           {:.1} km/h", stats.min_speed * 3.6);
    println!("  Max lateral g:       {:.2} g", stats.max_lat_g);
    println!("  Max longitudinal g:  {:.2} g", stats.max_lon_g);
}

fn print_car_stats(params: &CarParams) {
    println!("Car parameters:");
    println!("  Mass:                {:.1} kg", params.mass);
    println!("  Downforce coeff:     {:.2}", params.lift_coeff);
    println!("  Tire mu:             {:.2}", params.tire_mu);
    println!("  Max drive force:     {:.0} N", params.max_drive_force);
    println!("  Max brake force:     {:.0} N", params.max_brake_force);
}

/// Summary of a 14-DOF forward (transient) simulation around a track.
struct ForwardSimResult {
    lap_time: f64,
    distance_traveled: f64,
    max_speed: f64,
    max_lateral_g: f64,
    max_roll_deg: f64,
    max_pitch_deg: f64,
    max_suspension_mm: f64,
    n_steps: usize,
}

/// Look up the QSS target speed nearest arc length `s` (wrapping by lap length).
fn target_speed_at(qss: &QssResult, s: f64, total_length: f64) -> f64 {
    let n = qss.speeds.len();
    if n == 0 || total_length <= 0.0 {
        return 30.0;
    }
    let frac = (s / total_length).rem_euclid(1.0);
    let idx = ((frac * n as f64) as usize).min(n - 1);
    qss.speeds[idx]
}

/// Drive the 14-DOF car around `track` with a simple centerline-tracking
/// controller, integrating with the adaptive RK45 stepper. The throttle/brake
/// targets follow the precomputed QSS speed profile.
fn simulate_fourteen_dof(
    track: &Track,
    params: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
) -> ForwardSimResult {
    // Speed profile target from the quasi-steady-state solver.
    let qss = qss_lap_sim(track, params);

    // Anchor the model's static trim to the starting speed.
    let start_speed = 30.0;
    let model = FourteenDofModel::new(params, tire, suspension, aero, start_speed);

    // --- initial state (24 elements) ---
    let (x0, y0) = track.position_at(0.0);
    let psi0 = track.heading_at(0.0);
    let z_eq = model.equilibrium_travel(); // static-trim suspension travel

    let mut state = [0.0f64; 24];
    state[0] = x0;
    state[1] = y0;
    state[2] = aero.design_ride_height + params.cog_height; // CoG height above ground
    state[5] = psi0;
    state[6] = start_speed; // vx
    let wheel_omega = start_speed / params.wheel_radius;
    for w in state.iter_mut().skip(12).take(4) {
        *w = wheel_omega;
    }
    state[16..20].copy_from_slice(&z_eq); // suspension at static equilibrium
                                          // suspension velocities (20..24) and all remaining rates already 0.

    // --- adaptive integrator configuration ---
    let config = AdaptiveConfig {
        atol: 1e-5,
        rtol: 1e-5,
        dt_min: 1e-7,
        dt_max: 0.005, // >= 200 Hz to resolve suspension dynamics
        ..AdaptiveConfig::default()
    };

    let total_length = track.total_length;
    let safety_time = 300.0;

    let mut t = 0.0;
    let mut s = 0.0;
    let mut dt = config.dt_max;

    let mut n_steps = 0usize;
    let mut distance = 0.0;
    let mut max_speed = 0.0f64;
    let mut max_lat_g = 0.0f64;
    let mut max_roll = 0.0f64;
    let mut max_pitch = 0.0f64;
    let mut max_susp = 0.0f64;

    while s < total_length && t < safety_time {
        // --- controller ---
        let kappa = track.curvature_at(s);
        let (tx, ty) = track.position_at(s);
        let track_heading = track.heading_at(s);

        // signed lateral offset from centerline (left of heading is positive)
        let dx = state[0] - tx;
        let dy = state[1] - ty;
        let n_offset = -dx * track_heading.sin() + dy * track_heading.cos();
        let heading_error = normalize_angle(state[5] - track_heading);

        // off-track recovery: slow down and steer harder
        let (wl, wr) = track.width_at(s);
        let half_width = 0.5 * (wl + wr);
        let off_track = n_offset.abs() > half_width;
        let (k_lat, k_head, speed_scale) = if off_track {
            (1.0, 3.0, 0.5)
        } else {
            (0.5, 2.0, 1.0)
        };

        // steering: curvature feedforward minus centerline/heading feedback
        let delta =
            (kappa * params.wheelbase - k_lat * n_offset - k_head * heading_error).clamp(-0.5, 0.5);

        // throttle / brake from the QSS target speed
        let target = target_speed_at(&qss, s, total_length) * speed_scale;
        let vx = state[6];
        let (torque, brake) = if vx < target * 0.95 {
            (3000.0, 0.0)
        } else if vx > target * 1.05 {
            (0.0, 0.3)
        } else {
            // coast: small torque to offset drag and hold speed
            (params.drag_force(vx) * params.wheel_radius, 0.0)
        };
        let control = [delta, torque, brake];

        // --- adaptive step ---
        let step = rk45_adaptive_step(&model, &state, &control, t, dt, &config);
        let at_floor = dt <= config.dt_min * (1.0 + 1e-9);
        if step.accepted || at_floor {
            let speed = (state[6] * state[6] + state[7] * state[7]).sqrt();
            t += dt;
            s += state[6] * dt; // advance track progress by longitudinal speed
            distance += speed * dt;
            state = step.state;
            n_steps += 1;

            if !state.iter().all(|v| v.is_finite()) {
                break; // diverged — stop and report what we have
            }

            // telemetry maxima
            max_speed = max_speed.max(speed);
            let lat_g = (state[6] * state[11]).abs() / GRAVITY; // v·yaw_rate ≈ a_lat
            max_lat_g = max_lat_g.max(lat_g);
            max_roll = max_roll.max(state[3].abs());
            max_pitch = max_pitch.max(state[4].abs());
            for &z in &state[16..20] {
                max_susp = max_susp.max(z.abs());
            }
        }
        dt = step.dt_next;
    }

    ForwardSimResult {
        lap_time: t,
        distance_traveled: distance,
        max_speed,
        max_lateral_g: max_lat_g,
        max_roll_deg: max_roll.to_degrees(),
        max_pitch_deg: max_pitch.to_degrees(),
        max_suspension_mm: max_susp * 1000.0,
        n_steps,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Apex-14 — Quasi-Steady-State Lap Simulator");
    println!("==========================================");
    println!();

    let params = CarParams::default();

    // --- Oval track ---
    let (oval_points, oval_closed) = oval_track(1000.0, 100.0, 12.0, 500);
    let oval: Track = build_track("Oval", &oval_points, oval_closed);
    println!("Track: Oval (1000m straights, R=100m corners)");
    println!("Track length: {:.1} m", oval.total_length);
    println!();

    print_car_stats(&params);
    println!();

    let oval_result = qss_lap_sim(&oval, &params);
    let oval_stats = summarize(&oval_result);
    println!("Oval results:");
    print_results(&oval_stats);
    export_qss_csv(Path::new("qss_oval_telemetry.csv"), &oval, &oval_result)?;
    println!("Telemetry exported to qss_oval_telemetry.csv");
    render_track_svg(
        Path::new("qss_oval_track.svg"),
        &oval,
        &oval_result.speeds,
        "Apex-14 — Oval (R=100m)",
    )?;
    println!("Track SVG exported to qss_oval_track.svg");
    println!();

    // --- 14-DOF forward simulation (Oval) ---
    let tire = PacejkaTire::f1_default();
    let suspension = SuspensionSystem::f1_default();
    let aero = AeroModel::f1_default();
    let fwd = simulate_fourteen_dof(&oval, &params, &tire, &suspension, &aero);

    println!("14-DOF Forward Simulation: Oval");
    println!("  Lap time: {:.3}s", fwd.lap_time);
    println!("  Top speed: {:.1} km/h", fwd.max_speed * 3.6);
    println!("  Max lateral g: {:.2}", fwd.max_lateral_g);
    println!("  Max roll: {:.3} deg", fwd.max_roll_deg);
    println!("  Max pitch: {:.3} deg", fwd.max_pitch_deg);
    println!(
        "  Max suspension compression: {:.1} mm",
        fwd.max_suspension_mm
    );
    println!("  Distance traveled: {:.1} m", fwd.distance_traveled);
    println!("  Integration steps: {} (accepted)", fwd.n_steps);
    println!();

    println!("--- Model Comparison: Oval ---");
    println!("  QSS (2-DOF):  {:.3}s", oval_stats.lap_time);
    println!("  14-DOF sim:   {:.3}s", fwd.lap_time);
    println!();

    // --- Circle track ---
    let (circle_points, circle_closed) = circle_track(100.0, 12.0, 200);
    let circle: Track = build_track("Circle", &circle_points, circle_closed);
    println!("Track: Circle (R=100m)");
    println!("Track length: {:.1} m", circle.total_length);
    println!();

    let circle_result = qss_lap_sim(&circle, &params);
    let circle_stats = summarize(&circle_result);
    println!("Circle results:");
    print_results(&circle_stats);
    export_qss_csv(
        Path::new("qss_circle_telemetry.csv"),
        &circle,
        &circle_result,
    )?;
    println!("Telemetry exported to qss_circle_telemetry.csv");
    render_track_svg(
        Path::new("qss_circle_track.svg"),
        &circle,
        &circle_result.speeds,
        "Apex-14 — Circle (R=100m)",
    )?;
    println!("Track SVG exported to qss_circle_track.svg");
    println!();

    // --- Silverstone Circuit ---
    let (silver_points, silver_closed) = silverstone_circuit();
    let silverstone = build_track("Silverstone", &silver_points, silver_closed);
    let silver_stats = run_circuit(
        &silverstone,
        &params,
        "Silverstone Circuit",
        "qss_silverstone_telemetry.csv",
        "qss_silverstone_track.svg",
        "Apex-14 — Silverstone",
    )?;

    // --- Monza Circuit ---
    let (monza_points, monza_closed) = monza_circuit();
    let monza = build_track("Monza", &monza_points, monza_closed);
    let monza_stats = run_circuit(
        &monza,
        &params,
        "Monza Circuit",
        "qss_monza_telemetry.csv",
        "qss_monza_track.svg",
        "Apex-14 — Monza",
    )?;

    // --- Comparison ---
    println!("--- Comparison ---");
    println!(
        "Oval:        {:.3}s lap | {:.1} - {:.1} km/h speed range",
        oval_stats.lap_time,
        oval_stats.min_speed * 3.6,
        oval_stats.top_speed * 3.6
    );
    println!(
        "Circle:      {:.3}s lap | {:.1} km/h constant speed",
        circle_stats.lap_time,
        circle_stats.top_speed * 3.6
    );
    println!(
        "Silverstone: {:.3}s lap | {:.1} - {:.1} km/h speed range",
        silver_stats.lap_time,
        silver_stats.min_speed * 3.6,
        silver_stats.top_speed * 3.6
    );
    println!(
        "Monza:       {:.3}s lap | {:.1} - {:.1} km/h speed range",
        monza_stats.lap_time,
        monza_stats.min_speed * 3.6,
        monza_stats.top_speed * 3.6
    );

    Ok(())
}

/// Run the simulation for one circuit: print results, export CSV and SVG.
fn run_circuit(
    track: &Track,
    params: &CarParams,
    title: &str,
    csv_path: &str,
    svg_path: &str,
    svg_title: &str,
) -> Result<LapStats, Box<dyn std::error::Error>> {
    println!("Track: {}", title);
    println!("Track length: {:.1} m", track.total_length);
    println!();

    let result = qss_lap_sim(track, params);
    let stats = summarize(&result);
    println!("{} results:", title);
    print_results(&stats);
    export_qss_csv(Path::new(csv_path), track, &result)?;
    println!("Telemetry exported to {}", csv_path);
    render_track_svg(Path::new(svg_path), track, &result.speeds, svg_title)?;
    println!("Track SVG exported to {}", svg_path);
    println!();

    Ok(stats)
}
