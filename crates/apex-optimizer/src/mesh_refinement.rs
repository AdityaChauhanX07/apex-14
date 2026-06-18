//! Progressive mesh refinement for the collocation optimizer.
//!
//! Solve on a coarse mesh first (cheap), interpolate the solution onto a finer
//! mesh, and solve again from that warm start. The coarse solve is fast and
//! gives the fine solve a near-feasible starting point, which converges far more
//! reliably than a cold start on the fine mesh.

use apex_physics::CarParams;
use apex_track::Track;

use crate::collocation::{
    CollocationConfig, CollocationMethod, CollocationOptimizer, OptimizationResult,
};
use crate::gauss_newton::GaussNewtonConfig;

/// Configuration for mesh refinement.
#[derive(Debug, Clone)]
pub struct MeshRefinementConfig {
    /// Sequence of node counts to use, from coarse to fine.
    /// Example: [30, 60, 120] means solve at 30 nodes, refine to 60, then 120.
    pub mesh_sequence: Vec<usize>,
    /// Solver configuration for each mesh level.
    /// If shorter than mesh_sequence, the last config is reused.
    pub solver_configs: Vec<GaussNewtonConfig>,
}

impl Default for MeshRefinementConfig {
    fn default() -> Self {
        let coarse = GaussNewtonConfig {
            max_iterations: 30,
            constraint_tol: 1e-2, // relaxed for the coarse mesh
            ..GaussNewtonConfig::default()
        };
        let fine = GaussNewtonConfig {
            max_iterations: 50,
            constraint_tol: 1e-3, // tighter for the fine mesh
            ..GaussNewtonConfig::default()
        };
        MeshRefinementConfig {
            mesh_sequence: vec![30, 60, 120],
            solver_configs: vec![coarse, fine],
        }
    }
}

/// Result of mesh-refined optimization.
#[derive(Debug, Clone)]
pub struct RefinedResult {
    /// Results at each mesh level (coarse to fine).
    pub level_results: Vec<LevelResult>,
    /// Final optimization result (from the finest mesh).
    pub final_result: OptimizationResult,
}

/// Summary of the solve at one mesh level.
#[derive(Debug, Clone)]
pub struct LevelResult {
    pub n_nodes: usize,
    pub lap_time: f64,
    pub eq_violation: f64,
    pub converged: bool,
}

/// Pick the solver config for `level`, reusing the last one if the list is
/// shorter than the mesh sequence (and falling back to the default if empty).
fn solver_for_level(config: &MeshRefinementConfig, level: usize) -> GaussNewtonConfig {
    if config.solver_configs.is_empty() {
        return GaussNewtonConfig::default();
    }
    let idx = level.min(config.solver_configs.len() - 1);
    config.solver_configs[idx].clone()
}

/// Build a collocation config for `n_nodes` on `track` using collocation
/// `method`.
fn level_config(n_nodes: usize, track: &Track, method: CollocationMethod) -> CollocationConfig {
    CollocationConfig {
        n_nodes,
        closed: track.is_closed,
        method,
        ..CollocationConfig::default()
    }
}

/// Collocation method for refinement `level` out of `n_levels`.
///
/// Coarse meshes use the robust second-order trapezoidal scheme — on a sparse
/// mesh the fourth-order Hermite-Simpson midpoint cannot resolve sharp curvature
/// transitions and converges poorly. The finest mesh, where the nodes resolve the
/// geometry, switches to Hermite-Simpson for its higher accuracy. This is the
/// standard "robust low order to seed, high order to polish" refinement ladder.
fn method_for_level(level: usize, n_levels: usize) -> CollocationMethod {
    if level + 1 == n_levels {
        CollocationMethod::HermiteSimpson
    } else {
        CollocationMethod::Trapezoidal
    }
}

fn level_summary(n_nodes: usize, result: &OptimizationResult) -> LevelResult {
    LevelResult {
        n_nodes,
        lap_time: result.lap_time,
        eq_violation: result.eq_violation,
        converged: result.converged,
    }
}

/// Run the collocation optimizer with progressive mesh refinement.
///
/// 1. Solve on the coarsest mesh using the QSS warm start.
/// 2. Interpolate the coarse solution onto the next finer mesh.
/// 3. Solve on the finer mesh using the interpolated solution as warm start.
/// 4. Repeat until the finest mesh.
pub fn optimize_with_refinement(
    track: &Track,
    car: &CarParams,
    config: &MeshRefinementConfig,
) -> RefinedResult {
    // Guarantee at least one mesh level so a final result always exists.
    let sequence: Vec<usize> = if config.mesh_sequence.is_empty() {
        vec![30]
    } else {
        config.mesh_sequence.clone()
    };

    let n_levels = sequence.len();

    // Coarsest level: cold start from the QSS warm start.
    let opt0 = CollocationOptimizer::new(
        level_config(sequence[0], track, method_for_level(0, n_levels)),
        track,
        car,
    );
    let mut current = opt0.optimize_gn(&solver_for_level(config, 0));
    let mut level_results = vec![level_summary(sequence[0], &current)];

    // Finer levels: interpolate the previous solution as the warm start.
    for (level, &n_nodes) in sequence.iter().enumerate().skip(1) {
        let opt = CollocationOptimizer::new(
            level_config(n_nodes, track, method_for_level(level, n_levels)),
            track,
            car,
        );
        let x0 = opt.initial_guess_from_result(&current);
        current = opt.optimize_gn_from(&x0, &solver_for_level(config, level));
        level_results.push(level_summary(n_nodes, &current));
    }

    RefinedResult {
        level_results,
        final_result: current,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, circle_track, oval_track};

    fn circle_track_50() -> Track {
        let (pts, closed) = circle_track(50.0, 12.0, 200);
        build_track("circle", &pts, closed)
    }

    /// Build a synthetic constant-speed result with `n` nodes on `track`.
    fn constant_speed_result(track: &Track, n: usize, speed: f64) -> OptimizationResult {
        let stations: Vec<f64> = (0..n)
            .map(|k| track.total_length * (k as f64) / ((n - 1) as f64))
            .collect();
        let curvature_cmds: Vec<f64> = stations.iter().map(|&s| track.curvature_at(s)).collect();
        let dt: Vec<f64> = (0..n - 1)
            .map(|k| (stations[k + 1] - stations[k]) / speed)
            .collect();
        let lap_time = dt.iter().sum();
        OptimizationResult {
            speeds: vec![speed; n],
            offsets: vec![0.0; n],
            headings: vec![0.0; n],
            stations,
            drive_forces: vec![0.0; n],
            curvature_cmds,
            time_steps: dt,
            lap_time,
            eq_violation: 0.0,
            converged: true,
        }
    }

    #[test]
    fn interpolation_preserves_constant_speed() {
        let track = circle_track_50();
        let coarse = constant_speed_result(&track, 10, 22.0);

        let n_fine = 20;
        let car = CarParams::default();
        let fine_opt = CollocationOptimizer::new(
            level_config(n_fine, &track, CollocationMethod::HermiteSimpson),
            &track,
            &car,
        );
        let x = fine_opt.initial_guess_from_result(&coarse);

        // Decision-variable layout: s = x[0..N], v = x[2N..3N].
        let stations = &x[0..n_fine];
        let speeds = &x[2 * n_fine..3 * n_fine];

        for &v in speeds {
            assert!((v - 22.0).abs() < 1e-6, "interpolated speed {} != 22", v);
        }

        // Stations evenly spaced over the lap.
        let step = track.total_length / (n_fine as f64 - 1.0);
        for (i, &s) in stations.iter().enumerate() {
            let expected = step * i as f64;
            assert!(
                (s - expected).abs() < 1e-6,
                "station {} = {} != {}",
                i,
                s,
                expected
            );
        }
    }

    #[test]
    fn refinement_on_circle() {
        let track = circle_track_50();
        let car = CarParams::default();
        let config = MeshRefinementConfig {
            mesh_sequence: vec![10, 20, 40],
            ..MeshRefinementConfig::default()
        };
        let refined = optimize_with_refinement(&track, &car, &config);

        assert_eq!(refined.level_results.len(), 3);
        for lvl in &refined.level_results {
            assert!(
                lvl.lap_time.is_finite() && lvl.lap_time > 0.0,
                "lap {}",
                lvl.lap_time
            );
        }

        // The fine solve converges to tolerance. (Absolute eq_violation is not
        // comparable across mesh sizes — a finer mesh has more trapezoidal
        // defects — so the meaningful check is that the final solve is feasible
        // to the fine-level constraint tolerance, not that it beats the coarse
        // mesh's tiny near-exact residual.)
        let final_viol = refined.final_result.eq_violation;
        assert!(
            final_viol.is_finite() && final_viol < 1e-3,
            "final eq_viol {} should be converged below the fine tolerance",
            final_viol
        );

        // The circle is easy: all level lap times agree to within 5%.
        let first = refined.level_results[0].lap_time;
        for lvl in &refined.level_results {
            assert!(
                (lvl.lap_time - first).abs() / first < 0.05,
                "lap {} at N={} differs from {} by >5%",
                lvl.lap_time,
                lvl.n_nodes,
                first
            );
        }
    }

    #[test]
    fn refinement_beats_cold_start_on_oval() {
        let (pts, closed) = oval_track(500.0, 80.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();

        let config = MeshRefinementConfig {
            mesh_sequence: vec![20, 40],
            ..MeshRefinementConfig::default()
        };
        let refined = optimize_with_refinement(&track, &car, &config);

        // Cold start directly at N=40 with the same fine-level solver config and
        // the same (Hermite-Simpson) method the refinement ladder ends on.
        let fine_solver = solver_for_level(&config, 1);
        let cold = CollocationOptimizer::new(
            level_config(40, &track, CollocationMethod::HermiteSimpson),
            &track,
            &car,
        );
        let cold_result = cold.optimize_gn(&fine_solver);

        let warm_viol = refined.final_result.eq_violation;
        assert!(refined.final_result.lap_time.is_finite());
        // The warm-started fine solve should reach a point at least as feasible.
        assert!(
            warm_viol <= cold_result.eq_violation + 1e-6,
            "mesh-refined eq_viol {} should be <= cold-start eq_viol {}",
            warm_viol,
            cold_result.eq_violation
        );
    }

    #[test]
    fn single_level_matches_plain_gn() {
        let track = circle_track_50();
        let car = CarParams::default();

        let solver = GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-3,
            ..GaussNewtonConfig::default()
        };
        let config = MeshRefinementConfig {
            mesh_sequence: vec![50],
            solver_configs: vec![solver.clone()],
        };
        let refined = optimize_with_refinement(&track, &car, &config);
        assert_eq!(refined.level_results.len(), 1);

        // A single level must match a plain optimize_gn at the same N. A lone
        // level is the finest, so it uses Hermite-Simpson.
        let plain = CollocationOptimizer::new(
            level_config(50, &track, CollocationMethod::HermiteSimpson),
            &track,
            &car,
        )
        .optimize_gn(&solver);

        assert!(
            (refined.final_result.lap_time - plain.lap_time).abs() < 1e-9,
            "single-level lap {} != plain GN lap {}",
            refined.final_result.lap_time,
            plain.lap_time
        );
        assert!((refined.final_result.eq_violation - plain.eq_violation).abs() < 1e-9);
    }
}
