//! Setup evaluation pipeline for car setup optimization.
//!
//! Evaluates car setups by running the QSS lap simulation and returning the lap
//! time as the fitness value (lower is better), and drives the [`CmaEs`]
//! optimizer over the [`SetupSpace`] to search for a faster setup.

use apex_physics::{export_car_toml, qss_lap_sim, CarParams};
use apex_track::Track;

use crate::cmaes::{CmaEs, CmaEsConfig};
use crate::setup::SetupSpace;

/// Configuration for setup evaluation.
pub struct SetupEvalConfig {
    /// Track to evaluate on.
    pub track: Track,
    /// Base car parameters (non-tunable fields come from here).
    pub base_car: CarParams,
    /// Setup parameter space definition.
    pub space: SetupSpace,
}

impl SetupEvalConfig {
    /// Create an evaluation config for a given track and base car, using the
    /// standard F1 setup space.
    pub fn new(track: Track, base_car: CarParams) -> Self {
        Self {
            track,
            base_car,
            space: SetupSpace::f1_standard(),
        }
    }
}

/// Evaluate a single parameter vector.
///
/// Returns the lap time in seconds (lower is better), or [`f64::INFINITY`] if
/// the simulation produces a non-finite or non-positive lap time.
pub fn evaluate_setup(params: &[f64], config: &SetupEvalConfig) -> f64 {
    let car = config.space.apply(&config.base_car, params);
    let result = qss_lap_sim(&config.track, &car);
    let t = result.lap_time;
    if t.is_finite() && t > 0.0 {
        t
    } else {
        f64::INFINITY
    }
}

/// Evaluate a batch of parameter vectors.
///
/// Uses Rayon for parallel evaluation when the `parallel` feature is enabled,
/// otherwise evaluates sequentially. Returns fitness values corresponding 1:1 to
/// each input parameter vector.
pub fn evaluate_batch(candidates: &[Vec<f64>], config: &SetupEvalConfig) -> Vec<f64> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        candidates
            .par_iter()
            .map(|params| evaluate_setup(params, config))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        candidates
            .iter()
            .map(|params| evaluate_setup(params, config))
            .collect()
    }
}

/// Record of a single generation's progress.
#[derive(Debug, Clone)]
pub struct GenerationRecord {
    /// Generation number.
    pub generation: usize,
    /// Best fitness (lap time, s) found through this generation.
    pub best_fitness: f64,
    /// Step size at this generation.
    pub sigma: f64,
}

/// Result of a setup optimization run.
#[derive(Debug, Clone)]
pub struct SetupOptResult {
    /// Lap time with the baseline setup (s).
    pub baseline_time: f64,
    /// Best lap time found (s); never worse than `baseline_time`.
    pub best_time: f64,
    /// Best parameter vector (the baseline if no improvement was found).
    pub best_params: Vec<f64>,
    /// Time improvement over baseline (s), always `>= 0`.
    pub improvement: f64,
    /// Number of generations run.
    pub generations: usize,
    /// Convergence history, one record per generation.
    pub history: Vec<GenerationRecord>,
}

/// Run the full setup optimization.
///
/// Evaluates the baseline setup, then drives [`CmaEs`] over the
/// [`SetupSpace`] for up to `cmaes_config.max_generations` generations. The
/// returned `best_time`/`best_params` are guaranteed to be no worse than the
/// baseline: if the search never beats the baseline, the baseline is returned.
pub fn optimize_setup(config: &SetupEvalConfig, cmaes_config: CmaEsConfig) -> SetupOptResult {
    let baseline = config.space.baseline_vec();
    let bounds = config.space.bounds();

    // Evaluate the baseline setup.
    let baseline_time = evaluate_setup(&baseline, config);

    // Create the CMA-ES optimizer, starting from the baseline setup.
    let mut cmaes = CmaEs::new(baseline.clone(), bounds, cmaes_config);

    let mut history: Vec<GenerationRecord> = Vec::new();

    loop {
        let candidates = cmaes.ask();
        let fitnesses = evaluate_batch(&candidates, config);
        cmaes.tell(&fitnesses);

        let gen = cmaes.generation();
        let best = cmaes.best_fitness();

        log::info!(
            "Gen {:>3} | best_time: {:.3}s | sigma: {:.4} | baseline: {:.3}s | improvement: {:.3}s",
            gen,
            best,
            cmaes.sigma(),
            baseline_time,
            baseline_time - best
        );

        history.push(GenerationRecord {
            generation: gen,
            best_fitness: best,
            sigma: cmaes.sigma(),
        });

        if cmaes.should_stop() {
            log::info!("CMA-ES converged after {} generations", gen);
            break;
        }
    }

    // Never report a setup worse than the baseline.
    let (best_time, best_params) = if cmaes.best_fitness() <= baseline_time {
        (cmaes.best_fitness(), cmaes.best_params().to_vec())
    } else {
        (baseline_time, baseline)
    };

    SetupOptResult {
        baseline_time,
        best_time,
        improvement: baseline_time - best_time,
        best_params,
        generations: cmaes.generation(),
        history,
    }
}

/// Export an optimized setup to a car-config TOML string.
///
/// Applies `params` to `base_car` and serializes the result with
/// [`apex_physics::export_car_toml`], so the output is a valid car config that
/// can be loaded with `--car`. A header documents the track, lap time, and the
/// tuned setup parameters.
///
/// Note: `tire_radial_stiffness` has no field in the car-config schema (and is
/// unused by the QSS objective), so it appears only in the documentation header,
/// not in the loadable body.
pub fn export_setup_toml(
    space: &SetupSpace,
    base_car: &CarParams,
    params: &[f64],
    track_name: &str,
    lap_time: f64,
) -> String {
    let car = space.apply(base_car, params);

    let mut toml = String::new();
    toml.push_str(&format!("# Optimized setup for {track_name}\n"));
    toml.push_str(&format!("# Lap time: {lap_time:.3}s\n#\n"));
    toml.push_str("# Tuned setup parameters:\n");
    for (i, def) in space.params().iter().enumerate() {
        let val = params.get(i).copied().unwrap_or(def.baseline);
        toml.push_str(&format!(
            "#   {:<22} {:>12.3} {:<6} (range: {:.3} - {:.3})\n",
            def.name, val, def.unit, def.min, def.max
        ));
    }
    toml.push('\n');
    toml.push_str(&export_car_toml(
        &car,
        &format!("Optimized for {track_name}"),
    ));
    toml
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, oval_track};

    /// Build a closed oval track with long straights (so drag matters).
    fn oval() -> Track {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 500);
        build_track("oval", &pts, closed)
    }

    fn config() -> SetupEvalConfig {
        SetupEvalConfig::new(oval(), CarParams::default())
    }

    #[test]
    fn test_evaluate_setup_returns_finite() {
        let cfg = config();
        let t = evaluate_setup(&cfg.space.baseline_vec(), &cfg);
        assert!(t.is_finite(), "baseline lap time should be finite");
        assert!(
            (10.0..300.0).contains(&t),
            "baseline lap time {t}s out of expected range"
        );
    }

    #[test]
    fn test_evaluate_setup_different_params_different_time() {
        let cfg = config();
        let baseline = cfg.space.baseline_vec();
        let t0 = evaluate_setup(&baseline, &cfg);

        // Crank drag (index 0) to its maximum: on an oval with long straights
        // this should reduce top speed and change the lap time.
        let mut high_drag = baseline.clone();
        high_drag[0] = cfg.space.params()[0].max;
        let t1 = evaluate_setup(&high_drag, &cfg);

        assert!(t1.is_finite());
        assert!(
            (t0 - t1).abs() > 1e-3,
            "different drag should change lap time: {t0}s vs {t1}s"
        );
    }

    #[test]
    fn test_optimize_setup_improves() {
        let cfg = config();
        let cmaes_config = CmaEsConfig {
            max_generations: 10,
            ..Default::default()
        };
        let result = optimize_setup(&cfg, cmaes_config);
        assert!(result.best_time.is_finite());
        assert!(
            result.best_time <= result.baseline_time + 1e-9,
            "optimizer should not make the setup worse: {} vs {}",
            result.best_time,
            result.baseline_time
        );
        assert!(result.improvement >= -1e-9, "improvement should be >= 0");
        assert_eq!(result.best_params.len(), cfg.space.dim());
        assert!(!result.history.is_empty(), "history should be recorded");
    }

    #[test]
    fn test_export_toml_contains_params() {
        let cfg = config();
        let params = cfg.space.baseline_vec();
        let toml = export_setup_toml(&cfg.space, &cfg.base_car, &params, "Oval", 42.0);
        assert!(!toml.is_empty());
        // Parameter names appear in the documentation header.
        for def in cfg.space.params() {
            assert!(
                toml.contains(&def.name),
                "TOML missing parameter {}",
                def.name
            );
        }
        // A tuned value (drag) appears in the loadable body and is re-parseable.
        assert!(toml.contains("drag_coeff"));
        let parsed = apex_physics::parse_car_toml(&toml, &CarParams::default())
            .expect("exported setup TOML should be a valid car config");
        assert!((parsed.drag_coeff - params[0]).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_batch_length() {
        let cfg = config();
        let baseline = cfg.space.baseline_vec();
        let candidates: Vec<Vec<f64>> = (0..5).map(|_| baseline.clone()).collect();
        let fitnesses = evaluate_batch(&candidates, &cfg);
        assert_eq!(fitnesses.len(), 5);
        for f in fitnesses {
            assert!(f.is_finite(), "batch fitness should be finite");
        }
    }
}
