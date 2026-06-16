//! Apex-14 Quasi-Steady-State lap simulator demo binary.

use std::path::Path;

use apex_physics::{qss_lap_sim, CarParams, QssResult};
use apex_telemetry::export_qss_csv;
use apex_track::{build_track, circle_track, oval_track, Track};

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
    export_qss_csv(Path::new("qss_circle_telemetry.csv"), &circle, &circle_result)?;
    println!("Telemetry exported to qss_circle_telemetry.csv");
    println!();

    // --- Comparison ---
    println!("--- Comparison ---");
    println!(
        "Oval:   {:.3}s lap | {:.1} - {:.1} km/h speed range",
        oval_stats.lap_time,
        oval_stats.min_speed * 3.6,
        oval_stats.top_speed * 3.6
    );
    println!(
        "Circle: {:.3}s lap | {:.1} km/h constant speed",
        circle_stats.lap_time,
        circle_stats.top_speed * 3.6
    );

    Ok(())
}
