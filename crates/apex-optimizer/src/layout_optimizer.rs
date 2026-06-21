//! Track layout optimization using CMA-ES.
//!
//! Generates optimal track layouts by optimizing control point
//! positions to maximize racing quality subject to design constraints.

use std::f64::consts::PI;

use apex_physics::CarParams;
use apex_track::{check_constraints, ConstraintViolation, Track, TrackConstraints, TrackLayout};

use crate::cmaes::{CmaEs, CmaEsConfig};
use crate::racing_quality::{compute_racing_quality, RacingQuality};

/// Fitness returned for a candidate that cannot be evaluated (invalid layout,
/// failed track conversion, or failed QSS). Large enough to be rejected.
const INVALID_FITNESS: f64 = 1e6;

/// Configuration for track layout optimization.
#[derive(Debug, Clone)]
pub struct LayoutOptConfig {
    /// Number of control points for the track layout.
    pub n_control_points: usize,
    /// CMA-ES configuration.
    pub cmaes_config: CmaEsConfig,
    /// Design constraints.
    pub constraints: TrackConstraints,
    /// Car parameters for lap time computation.
    pub car: CarParams,
    /// Scale of the initial layout (radius of the initial circle, m).
    pub initial_radius: f64,
}

impl Default for LayoutOptConfig {
    fn default() -> Self {
        Self {
            n_control_points: 10,
            cmaes_config: CmaEsConfig {
                max_generations: 50,
                initial_sigma: 0.3,
                ..CmaEsConfig::default()
            },
            constraints: TrackConstraints::default(),
            car: CarParams::default(),
            initial_radius: 300.0,
        }
    }
}

/// Result of track layout optimization.
#[derive(Debug, Clone)]
pub struct LayoutOptResult {
    /// The optimized track layout.
    pub layout: TrackLayout,
    /// Racing quality of the optimized layout.
    pub quality: RacingQuality,
    /// Constraint violation (should be 0 or near 0).
    pub violation: ConstraintViolation,
    /// Number of CMA-ES generations run.
    pub generations: usize,
    /// Convergence history: `(generation, fitness)` pairs.
    pub history: Vec<(usize, f64)>,
}

/// Optimize a track layout.
///
/// Uses CMA-ES to search for control point positions that maximize the
/// overtaking opportunity score while satisfying all constraints. The initial
/// layout is a rough circle of the configured radius, which the optimizer then
/// deforms.
pub fn optimize_layout(config: &LayoutOptConfig) -> LayoutOptResult {
    let n = config.n_control_points;

    // Initial layout: control points on a circle.
    let mut initial_params = Vec::with_capacity(n * 3);
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / n as f64;
        initial_params.push(config.initial_radius * angle.cos());
        initial_params.push(config.initial_radius * angle.sin());
        initial_params.push(12.0); // default width
    }

    // Bounds: x and y within a large range, width between min and max.
    let mut bounds = Vec::with_capacity(n * 3);
    let pos_range = config.initial_radius * 3.0;
    for _ in 0..n {
        bounds.push((-pos_range, pos_range)); // x
        bounds.push((-pos_range, pos_range)); // y
        bounds.push((config.constraints.min_width, config.constraints.max_width));
        // width
    }

    let mut cmaes = CmaEs::new(initial_params, bounds, config.cmaes_config.clone());
    let mut history = Vec::new();

    loop {
        let candidates = cmaes.ask();
        let fitnesses: Vec<f64> = candidates
            .iter()
            .map(|params| evaluate_layout(params, n, config))
            .collect();

        cmaes.tell(&fitnesses);

        let gen = cmaes.generation();
        let best_fit = cmaes.best_fitness();
        log::info!(
            "Gen {:>3} | fitness: {:.2} | sigma: {:.4}",
            gen,
            best_fit,
            cmaes.sigma()
        );
        history.push((gen, best_fit));

        if cmaes.should_stop() {
            break;
        }
    }

    // Build the best layout found.
    let best_params = cmaes.best_params();
    let layout = TrackLayout::from_params("optimized", best_params, n)
        .unwrap_or_else(|| TrackLayout::new("optimized", Vec::new()));

    let (quality, violation) = match layout.to_track() {
        Some(track) => {
            let q =
                compute_racing_quality(&track, &config.car, config.constraints.min_straight_length);
            let points = extract_track_points(&track);
            let lt = q.as_ref().map(|r| r.lap_time);
            let v = check_constraints(&track, &points, &config.constraints, lt);
            (q, Some(v))
        }
        None => (None, None),
    };

    LayoutOptResult {
        layout,
        quality: quality.unwrap_or(RacingQuality {
            overtaking_score: 0.0,
            braking_zones: 0,
            drs_straights: 0,
            mean_straight_length: 0.0,
            speed_range: 0.0,
            lap_time: 0.0,
        }),
        violation: violation.unwrap_or(ConstraintViolation {
            total_penalty: f64::INFINITY,
            boundary_penalty: 0.0,
            intersection_penalty: 0.0,
            lap_time_penalty: 0.0,
            radius_penalty: 0.0,
            length_penalty: 0.0,
            clearance_penalty: 0.0,
            feasible: false,
        }),
        generations: cmaes.generation(),
        history,
    }
}

/// Evaluate a single layout candidate.
///
/// Returns a fitness value to MINIMIZE: `-overtaking_score + penalties`, or a
/// large constant for invalid/self-intersecting/QSS-failed candidates.
fn evaluate_layout(params: &[f64], n_points: usize, config: &LayoutOptConfig) -> f64 {
    let layout = match TrackLayout::from_params("candidate", params, n_points) {
        Some(l) => l,
        None => return INVALID_FITNESS,
    };

    let track = match layout.to_track() {
        Some(t) => t,
        None => return INVALID_FITNESS,
    };

    let points = extract_track_points(&track);

    // Cheap pre-check first: skip the (relatively expensive) QSS pass on tracks
    // that already self-intersect.
    let pre_check = check_constraints(&track, &points, &config.constraints, None);
    if pre_check.intersection_penalty > 0.0 {
        return INVALID_FITNESS + pre_check.total_penalty;
    }

    let quality =
        match compute_racing_quality(&track, &config.car, config.constraints.min_straight_length) {
            Some(q) => q,
            None => return INVALID_FITNESS,
        };

    let violation = check_constraints(&track, &points, &config.constraints, Some(quality.lap_time));

    // Minimize: a high overtaking score lowers fitness; penalties raise it.
    -quality.overtaking_score + violation.total_penalty
}

/// Extract `(x, y)` centerline positions from a track for constraint checking.
fn extract_track_points(track: &Track) -> Vec<(f64, f64)> {
    track.segments.iter().map(|s| (s.x, s.y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small, fast optimization config for tests.
    fn test_config(n_points: usize, max_gen: usize) -> LayoutOptConfig {
        LayoutOptConfig {
            n_control_points: n_points,
            cmaes_config: CmaEsConfig {
                max_generations: max_gen,
                initial_sigma: 0.3,
                ..CmaEsConfig::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_optimize_layout_produces_valid() {
        let config = test_config(6, 5);
        let result = optimize_layout(&config);

        assert_eq!(
            result.layout.control_points.len(),
            6,
            "layout should keep the requested control-point count"
        );
        let track = result.layout.to_track().expect("optimized layout converts");
        for seg in &track.segments {
            assert!(
                seg.x.is_finite() && seg.y.is_finite() && seg.curvature.is_finite(),
                "optimized geometry must be finite"
            );
        }
        assert!(
            result.generations <= 5,
            "should respect the generation budget"
        );
    }

    #[test]
    fn test_optimize_layout_respects_boundary() {
        // A generous boundary that comfortably contains the bounded search box
        // (control points are clamped to +-3*initial_radius = +-900 m).
        let mut config = test_config(8, 5);
        config.constraints.boundary = vec![
            (-1500.0, -1500.0),
            (1500.0, -1500.0),
            (1500.0, 1500.0),
            (-1500.0, 1500.0),
        ];
        let result = optimize_layout(&config);

        assert_eq!(
            result.violation.boundary_penalty, 0.0,
            "optimized track should stay within the generous boundary"
        );
    }

    #[test]
    fn test_evaluate_layout_rejects_bad() {
        let config = test_config(4, 5);
        // Bowtie control points -> the interpolated centerline self-intersects.
        let bad = [
            0.0, 0.0, 12.0, // p0
            100.0, 100.0, 12.0, // p1
            100.0, 0.0, 12.0, // p2
            0.0, 100.0, 12.0, // p3
        ];
        let fitness = evaluate_layout(&bad, 4, &config);
        assert!(
            fitness > 1e5,
            "a self-intersecting layout should be heavily penalized, got {fitness}"
        );
    }
}
