#![deny(unsafe_code)]
//! Apex-14 Quasi-Steady-State lap simulator demo binary.

use std::path::Path;

use apex_integrator::{rk45_adaptive_step, AdaptiveConfig};
use apex_physics::{
    qss_lap_sim, AeroModel, CarParams, FourteenDofModel, LqrController, PacejkaTire, QssResult,
    SpeedController, SuspensionSystem,
};
use apex_telemetry::{export_qss_csv, render_track_svg};
use apex_track::{
    build_track, circle_track, load_tumftm_csv, monza_circuit, normalize_angle, oval_track,
    silverstone_circuit, Track,
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
    /// Arc length at which the integration diverged (state became non-finite),
    /// if it did. `None` means the lap completed (or hit the safety timeout).
    diverged_at: Option<f64>,
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

/// QSS speed at arc length `s` (wrapping by lap length), looked up against the
/// QSS distance grid. `qss_distances` is sorted ascending, so we binary-search
/// for the enclosing segment.
fn qss_speed_at(qss_speeds: &[f64], qss_distances: &[f64], s: f64) -> f64 {
    let n = qss_speeds.len();
    if n == 0 {
        return 30.0;
    }
    let total_length = qss_distances.last().copied().unwrap_or(0.0);
    if total_length <= 0.0 {
        return qss_speeds[0];
    }
    let sq = s.rem_euclid(total_length);
    let idx = match qss_distances.binary_search_by(|d| d.partial_cmp(&sq).unwrap()) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };
    qss_speeds[idx.min(n - 1)]
}

/// Preview-braking target speed: look ahead by `preview_time * current_speed`
/// meters, find the slowest QSS speed in that window, and target a `grip_margin`
/// fraction of it. This makes the controller start slowing *before* it reaches a
/// slow corner rather than reacting at the apex. The grip margin accounts for the
/// 14-DOF car (with load transfer and roll) having less usable grip than the
/// point-mass QSS profile assumes.
///
/// Returns the minimum of the current-position target and the look-ahead minimum.
fn target_speed_with_preview(
    qss_speeds: &[f64],
    qss_distances: &[f64],
    current_s: f64,
    current_speed: f64,
    car: &CarParams,
    preview_time: f64,
    grip_margin: f64,
) -> f64 {
    if qss_speeds.is_empty() {
        return 30.0;
    }

    // Target at the current position.
    let here = qss_speed_at(qss_speeds, qss_distances, current_s);

    // Look-ahead window. Floor it at an estimated full-brake distance so the
    // preview never collapses to zero at low speed and we always see the next
    // corner in time to react.
    let brake_decel = (car.max_brake_force / car.mass).max(1.0);
    let brake_dist = current_speed * current_speed / (2.0 * brake_decel);
    let look = (preview_time * current_speed).max(brake_dist);

    let samples = 20;
    let mut min_ahead = here;
    for i in 0..=samples {
        let ds = look * i as f64 / samples as f64;
        let v = qss_speed_at(qss_speeds, qss_distances, current_s + ds);
        min_ahead = min_ahead.min(v);
    }

    min_ahead.min(here) * grip_margin
}

/// Drive the 14-DOF car around `track` with a simple proportional
/// centerline-tracking controller, integrating with the adaptive RK45 stepper.
/// The throttle/brake targets follow the precomputed QSS speed profile.
///
/// This is the original baseline controller, kept for comparison against the
/// LQR + PID controller in [`simulate_fourteen_dof_lqr`].
fn simulate_fourteen_dof_proportional(
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
    let mut diverged_at = None;

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
                diverged_at = Some(s); // diverged — stop and report what we have
                break;
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
        diverged_at,
    }
}

/// Drive the 14-DOF car around `track` using the LQR steering controller and
/// PID speed controller, integrating with the adaptive RK45 stepper. The speed
/// target follows the precomputed QSS profile.
fn simulate_fourteen_dof_lqr(
    track: &Track,
    params: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
) -> ForwardSimResult {
    let qss = qss_lap_sim(track, params);

    let start_speed = 30.0;
    let model = FourteenDofModel::new(params, tire, suspension, aero, start_speed);

    let lqr = LqrController::default();
    let mut speed_ctrl = SpeedController::f1_default();

    // --- initial state (24 elements) ---
    let (x0, y0) = track.position_at(0.0);
    let psi0 = track.heading_at(0.0);
    let z_eq = model.equilibrium_travel();

    let mut state = [0.0f64; 24];
    state[0] = x0;
    state[1] = y0;
    state[2] = aero.design_ride_height + params.cog_height;
    state[5] = psi0;
    state[6] = start_speed; // vx
    let wheel_omega = start_speed / params.wheel_radius;
    for w in state.iter_mut().skip(12).take(4) {
        *w = wheel_omega;
    }
    state[16..20].copy_from_slice(&z_eq);

    let config = AdaptiveConfig {
        atol: 1e-5,
        rtol: 1e-5,
        dt_min: 1e-7,
        dt_max: 0.005,
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
    let mut diverged_at = None;

    while s < total_length && t < safety_time {
        // --- error signals relative to the reference path at arc length s ---
        let vx = state[6];
        let vy = state[7];
        let omega_z = state[11];
        let psi = state[5];

        let (tx, ty) = track.position_at(s);
        let psi_track = track.heading_at(s);
        let kappa = track.curvature_at(s);
        let kappa_ahead = track.curvature_at(s + lqr.preview_distance);

        // signed perpendicular distance from the centerline (left = positive)
        let e_lat = -(state[0] - tx) * psi_track.sin() + (state[1] - ty) * psi_track.cos();
        let e_heading = normalize_angle(psi - psi_track);
        // rates: lateral from body-frame velocity + heading coupling, heading
        // from yaw rate minus the rate the road heading itself turns.
        let e_lat_dot = vy + vx * e_heading;
        let e_heading_dot = omega_z - vx * kappa;

        // --- LQR steering + PID speed control ---
        let delta = lqr.compute_steering(
            params,
            vx,
            e_lat,
            e_heading,
            e_lat_dot,
            e_heading_dot,
            kappa,
            kappa_ahead,
        );
        // Run the PID at a fixed control rate, not the integrator's adaptive
        // dt: feeding a 1e-7 s step into the derivative term makes it explode.
        // Preview braking looks 2 s ahead and targets 80% of the QSS speed.
        let target =
            target_speed_with_preview(&qss.speeds, &qss.distances, s, vx, params, 2.0, 0.80);
        // Drive-torque ceiling handed to the controller. Two limits apply: the
        // engine's physical maximum (T = F_max·r), and a *traction* limit so a
        // rear-wheel-drive car doesn't spin its rears when the PID asks for full
        // torque. Without the traction cap the rear tires are driven past their
        // slip-ratio peak, the wheels run away (ω·r ≫ vx), the rear loses grip,
        // and the car snaps into a spin — exactly the divergence we're fixing.
        // The cap is a fraction of the rear-axle grip torque (μ·F_z_rear·r); the
        // fraction is well under 1 because most of that grip is needed laterally
        // through a corner and the tire goes unstable well before the peak.
        const TRACTION_MARGIN: f64 = 0.15;
        let (_fz_front, fz_rear) = params.axle_loads(vx, 0.0);
        let traction_torque = TRACTION_MARGIN * params.tire_mu * fz_rear * params.wheel_radius;
        let physical_torque = params.max_drive_force * params.wheel_radius;
        let max_drive_torque = physical_torque.min(traction_torque);
        let (torque, brake) = speed_ctrl.compute(
            target,
            vx,
            config.dt_max,
            max_drive_torque,
            params.max_brake_force,
        );
        let control = [delta, torque, brake];

        // --- adaptive step ---
        let step = rk45_adaptive_step(&model, &state, &control, t, dt, &config);
        let at_floor = dt <= config.dt_min * (1.0 + 1e-9);
        if step.accepted || at_floor {
            let speed = (state[6] * state[6] + state[7] * state[7]).sqrt();
            t += dt;
            // Advance arc-length progress by the along-track speed component.
            // `s` is kept unwrapped so `s < total_length` terminates the lap;
            // the track queries wrap internally via `locate`, so closed-track
            // lookups at `s` and `s + preview` remain correct.
            s += vx * e_heading.cos() * dt;
            distance += speed * dt;
            state = step.state;
            n_steps += 1;

            // Treat non-finite OR clearly unphysical states as divergence: no
            // real lap exceeds ~200 m/s or rolls/pitches past 90 degrees.
            let unphysical = speed > 200.0
                || state[3].abs() > std::f64::consts::FRAC_PI_2
                || state[4].abs() > std::f64::consts::FRAC_PI_2;
            if !state.iter().all(|v| v.is_finite()) || unphysical {
                diverged_at = Some(s);
                break;
            }

            max_speed = max_speed.max(speed);
            let lat_g = (state[6] * state[11]).abs() / GRAVITY;
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
        diverged_at,
    }
}

/// One-line completion status for a forward sim: `(completed)` if the lap
/// finished, or `(diverged at X%)` with the fraction of the lap reached.
fn lap_status(result: &ForwardSimResult, total_length: f64) -> String {
    match result.diverged_at {
        Some(s) if total_length > 0.0 => {
            format!("(diverged at {:.0}%)", 100.0 * s / total_length)
        }
        Some(_) => "(diverged)".to_string(),
        None => "(completed)".to_string(),
    }
}

/// Print a 14-DOF forward-sim summary, or a divergence notice if the
/// integration blew up partway around the lap.
fn print_forward_result(result: &ForwardSimResult, total_length: f64) {
    if let Some(s) = result.diverged_at {
        let pct = if total_length > 0.0 {
            100.0 * s / total_length
        } else {
            0.0
        };
        println!("  Diverged at s = {:.1} m ({:.1}%)", s, pct);
        return;
    }
    println!("  Lap time: {:.3}s", result.lap_time);
    println!("  Top speed: {:.1} km/h", result.max_speed * 3.6);
    println!("  Max lateral g: {:.2}", result.max_lateral_g);
    println!("  Max roll: {:.3} deg", result.max_roll_deg);
    println!("  Max pitch: {:.3} deg", result.max_pitch_deg);
    println!(
        "  Max suspension compression: {:.1} mm",
        result.max_suspension_mm
    );
    println!("  Distance traveled: {:.1} m", result.distance_traveled);
    println!("  Integration steps: {} (accepted)", result.n_steps);
}

/// Obtain a Silverstone track for the 14-DOF run. Prefers real TUMFTM CSV data
/// if a local copy is found; otherwise falls back to the hardcoded approximate
/// circuit. Returns the track and a short description of the data source.
fn load_silverstone_track() -> (Track, &'static str) {
    const CANDIDATES: [&str; 3] = [
        "tracks/Silverstone.csv",
        "racetrack-database/tracks/Silverstone.csv",
        "/tmp/tracks/tracks/Silverstone.csv",
    ];
    for path in CANDIDATES {
        let p = Path::new(path);
        if p.exists() {
            if let Ok(track) = load_tumftm_csv(p, "Silverstone") {
                return (track, "TUMFTM CSV");
            }
        }
    }
    let (pts, closed) = silverstone_circuit();
    (
        build_track("Silverstone", &pts, closed),
        "hardcoded circuit",
    )
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
    // Old proportional controller (kept for comparison) and the new LQR + PID.
    let fwd_prop = simulate_fourteen_dof_proportional(&oval, &params, &tire, &suspension, &aero);
    let fwd = simulate_fourteen_dof_lqr(&oval, &params, &tire, &suspension, &aero);

    println!("14-DOF Forward Simulation: Oval (LQR + PID controller)");
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

    println!("--- Controller Comparison: Oval ---");
    println!(
        "  Old (proportional): {:.3}s lap | max roll {:.3} deg {}",
        fwd_prop.lap_time,
        fwd_prop.max_roll_deg,
        lap_status(&fwd_prop, oval.total_length)
    );
    println!(
        "  New (LQR + PID):    {:.3}s lap | max roll {:.3} deg {}",
        fwd.lap_time,
        fwd.max_roll_deg,
        lap_status(&fwd, oval.total_length)
    );
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

    // --- 14-DOF forward simulation (Silverstone) ---
    // Key validation of the new controller on a real circuit geometry. If the
    // integration diverges, report where instead of crashing.
    let (silver_track, source) = load_silverstone_track();
    println!(
        "14-DOF Forward Simulation: Silverstone ({}, LQR + PID controller)",
        source
    );
    let silver_fwd = simulate_fourteen_dof_lqr(&silver_track, &params, &tire, &suspension, &aero);
    print_forward_result(&silver_fwd, silver_track.total_length);
    println!();

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
