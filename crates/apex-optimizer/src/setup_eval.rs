//! Setup evaluation pipeline for car setup optimization.
//!
//! Evaluates car setups by running the QSS lap simulation and returning the lap
//! time as the fitness value (lower is better), and drives the [`CmaEs`]
//! optimizer over the [`SetupSpace`] to search for a faster setup.

use std::path::PathBuf;

use apex_physics::{
    export_car_toml, qss_lap_sim, AeroModel, CarParams, Envelope, EnvelopeGridSpec, PacejkaTire,
    SuspensionSystem,
};
use apex_track::Track;

use crate::cmaes::{CmaEs, CmaEsConfig};
use crate::envelope_ocp::{EnvelopeOcp, EnvelopeOcpConfig};
use crate::ipm::IpmConfig;
use crate::setup::SetupSpace;

/// Which lap-time model scores a setup candidate.
///
/// The default [`InnerObjective::Qss`] preserves the historical behavior. The
/// opt-in [`InnerObjective::Envelope`] uses the load-sensitive envelope
/// free-trajectory OCP; its absolute lap time is *not* mesh-converged, but at a
/// fixed `N` the ranking of setups is stable (the rank-stability gate,
/// `docs/design/envelope-qss/setup-envelope.md`), which is all CMA-ES needs.
#[derive(Debug, Clone)]
pub enum InnerObjective {
    /// Fixed-line quasi-steady-state lap (the friction-circle model). Fast; the
    /// default so no existing behavior changes.
    Qss,
    /// Envelope free-trajectory OCP at a **fixed** mesh (`nodes`), solved with
    /// the shared real-circuit IP config. Load-sensitive (responds to CoG height
    /// and weight distribution, which the point-mass QSS ignores) but blind to
    /// `lift_coeff`/`aero_balance` (the envelope's downforce comes from the
    /// `AeroModel`, not the `CarParams` aero fields — see setup-envelope.md).
    Envelope {
        /// OCP node count, held fixed across the whole search (ranking is only
        /// mesh-stable *at fixed `N`*; a per-candidate `N` would reshuffle).
        nodes: usize,
        /// Optional content-hash cache directory for generated envelopes. `None`
        /// regenerates every time (no disk side effects — used by tests).
        cache_dir: Option<PathBuf>,
    },
}

/// Configuration for setup evaluation.
pub struct SetupEvalConfig {
    /// Track to evaluate on.
    pub track: Track,
    /// Base car parameters (non-tunable fields come from here).
    pub base_car: CarParams,
    /// Setup parameter space definition.
    pub space: SetupSpace,
    /// Which lap-time model scores each candidate (default [`InnerObjective::Qss`]).
    pub inner: InnerObjective,
}

impl SetupEvalConfig {
    /// Create an evaluation config for a given track and base car, using the
    /// standard F1 setup space and the fixed-line QSS objective.
    pub fn new(track: Track, base_car: CarParams) -> Self {
        Self {
            track,
            base_car,
            space: SetupSpace::f1_standard(),
            inner: InnerObjective::Qss,
        }
    }

    /// Set the inner objective (builder style).
    pub fn with_inner(mut self, inner: InnerObjective) -> Self {
        self.inner = inner;
        self
    }
}

/// Penalty returned for a setup whose envelope OCP does not reach tight
/// feasibility (or whose envelope cannot be generated). Chosen far above any
/// physical lap time so a non-converged candidate always ranks below every
/// feasible one; the constraint residual is added so that, among non-converged
/// candidates, the less-infeasible ones rank better (a mild gradient back toward
/// feasibility).
const ENVELOPE_REJECT_PENALTY: f64 = 1.0e4;

/// Feasibility tolerance for the envelope inner loop (SI). Matches the CLI /
/// gate: `eq` and `ineq` violations must both be `<= 5e-3`.
const ENVELOPE_TIGHT_TOL: f64 = 5.0e-3;

/// Evaluate a single parameter vector under the configured inner objective.
///
/// Returns the lap time in seconds (lower is better). QSS returns
/// [`f64::INFINITY`] for a non-finite lap; the envelope path returns a large
/// finite penalty (never the un-converged lap, which is optimistically biased by
/// coarse-mesh over-cutting and would mislead the search toward infeasible
/// setups — see [`ENVELOPE_REJECT_PENALTY`]).
pub fn evaluate_setup(params: &[f64], config: &SetupEvalConfig) -> f64 {
    let car = config.space.apply(&config.base_car, params);
    match &config.inner {
        InnerObjective::Qss => {
            let t = qss_lap_sim(&config.track, &car).lap_time;
            if t.is_finite() && t > 0.0 {
                t
            } else {
                f64::INFINITY
            }
        }
        InnerObjective::Envelope { nodes, cache_dir } => {
            evaluate_setup_envelope(&car, &config.track, *nodes, cache_dir.as_deref())
        }
    }
}

/// Envelope free-trajectory OCP objective for one setup. Regenerates the g-g-g
/// envelope for the (setup-modified) car — its content hash changes with the
/// setup, so a fresh envelope is mandatory — then solves the OCP at the fixed
/// mesh and returns the lap time, or [`ENVELOPE_REJECT_PENALTY`] if it does not
/// reach tight feasibility.
fn evaluate_setup_envelope(
    car: &CarParams,
    track: &Track,
    nodes: usize,
    cache_dir: Option<&std::path::Path>,
) -> f64 {
    let spec = EnvelopeGridSpec {
        v_min: 5.0,
        v_max: 90.0,
        ..EnvelopeGridSpec::default()
    };
    let tire = PacejkaTire::f1_default();
    let susp = SuspensionSystem::f1_default();
    let aero = AeroModel::f1_default();
    let env = match cache_dir {
        Some(dir) => Envelope::generate_cached(car, &tire, &susp, &aero, spec, dir).map(|(e, _)| e),
        None => Envelope::generate(car, &tire, &susp, &aero, spec),
    };
    let env = match env {
        Ok(e) => e,
        Err(_) => return ENVELOPE_REJECT_PENALTY,
    };

    let cfg = EnvelopeOcpConfig {
        n_nodes: nodes,
        ..EnvelopeOcpConfig::default()
    };
    let ip = IpmConfig {
        max_iterations: 1500,
        constraint_tol: ENVELOPE_TIGHT_TOL,
        ..EnvelopeOcp::recommended_ip_config()
    };
    let r = EnvelopeOcp::new(cfg, track, car, &env).solve(&ip);

    let tight = r.eq_violation <= ENVELOPE_TIGHT_TOL && r.ineq_violation <= ENVELOPE_TIGHT_TOL;
    if tight && r.lap_time.is_finite() && r.lap_time > 0.0 {
        r.lap_time
    } else {
        ENVELOPE_REJECT_PENALTY + r.eq_violation + r.ineq_violation
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
    use apex_track::{build_track, oval_track, silverstone_circuit};

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

    // --- envelope inner-loop integration (docs/design/envelope-qss/setup-envelope.md) ---

    /// The synthetic Silverstone circuit + calibrated car: the fast, reliable
    /// case the envelope OCP reaches tight feasibility on at a coarse mesh
    /// (mirrors `envelope_ocp::silverstone_tuned_reaches_tight`). Needs no
    /// gitignored real-track data.
    fn syn_silverstone() -> Track {
        let (pts, closed) = silverstone_circuit();
        build_track("silverstone", &pts, closed)
    }

    /// Params that reproduce the calibrated car's own tuned fields (so `apply`
    /// is ~identity and the car stays the known-converging calibrated one).
    fn calibrated_params() -> (CarParams, Vec<f64>) {
        let car = CarParams::f1_2024_calibrated();
        let params = SetupSpace::f1_standard().extract(&car);
        (car, params)
    }

    #[test]
    fn envelope_inner_evaluates_finite_and_below_penalty() {
        let (car, params) = calibrated_params();
        let cfg =
            SetupEvalConfig::new(syn_silverstone(), car).with_inner(InnerObjective::Envelope {
                nodes: 24,
                cache_dir: None,
            });
        let t = evaluate_setup(&params, &cfg);
        assert!(
            t.is_finite() && t > 0.0,
            "envelope lap should be finite: {t}"
        );
        assert!(
            t < ENVELOPE_REJECT_PENALTY,
            "calibrated Silverstone should converge tight (got {t})"
        );
        assert!((50.0..120.0).contains(&t), "envelope lap {t}s out of range");
    }

    #[test]
    fn envelope_objective_is_load_sensitive_where_qss_is_blind() {
        // The payoff: CoG height (index 4) moves the envelope lap but NOT the
        // point-mass QSS lap — the reason envelope- and QSS-optimized setups
        // differ. Both CoG values must still converge (below the penalty).
        let (car, base) = calibrated_params();
        let space = SetupSpace::f1_standard();
        let qss_cfg = SetupEvalConfig::new(syn_silverstone(), car.clone());
        let env_cfg =
            SetupEvalConfig::new(syn_silverstone(), car).with_inner(InnerObjective::Envelope {
                nodes: 24,
                cache_dir: None,
            });

        let mut lo = base.clone();
        let mut hi = base.clone();
        lo[4] = space.params()[4].min; // cog_height low (0.25)
        hi[4] = space.params()[4].max; // cog_height high (0.35)

        let (q_lo, q_hi) = (evaluate_setup(&lo, &qss_cfg), evaluate_setup(&hi, &qss_cfg));
        let (e_lo, e_hi) = (evaluate_setup(&lo, &env_cfg), evaluate_setup(&hi, &env_cfg));

        assert!(
            (q_lo - q_hi).abs() < 1e-6,
            "point-mass QSS is blind to CoG height: {q_lo} vs {q_hi}"
        );
        assert!(
            e_lo < ENVELOPE_REJECT_PENALTY && e_hi < ENVELOPE_REJECT_PENALTY,
            "both CoG values should converge: {e_lo}, {e_hi}"
        );
        assert!(
            (e_lo - e_hi).abs() > 1e-3,
            "envelope should respond to CoG height: {e_lo} vs {e_hi}"
        );
    }

    #[test]
    fn envelope_penalizes_nonconverging_setup() {
        // The Simple Oval sample has sharp low-speed hairpins the coarse envelope
        // OCP cannot make feasible (docs/analysis.md), so the calibrated car
        // never reaches tight feasibility -> the reject penalty, NOT a (low,
        // over-cut) un-converged lap that would mislead the search.
        let track =
            apex_track::load_track_json(std::path::Path::new("../../tracks/oval_simple.json"))
                .expect("load tracks/oval_simple.json");
        let cfg = SetupEvalConfig::new(track, CarParams::f1_2024_calibrated()).with_inner(
            InnerObjective::Envelope {
                nodes: 24,
                cache_dir: None,
            },
        );
        let t = evaluate_setup(&cfg.space.baseline_vec(), &cfg);
        assert!(
            t >= ENVELOPE_REJECT_PENALTY,
            "non-converging setup must be penalized, got {t}"
        );
    }

    #[test]
    fn envelope_inner_optimize_is_deterministic() {
        // Seeded CMA-ES + deterministic envelope inner loop -> bitwise-identical
        // argmin across two independent runs.
        let cmaes_config = CmaEsConfig {
            population_size: Some(3),
            max_generations: 1,
            seed: 7,
            ..Default::default()
        };
        let run = || {
            let cfg = SetupEvalConfig::new(syn_silverstone(), CarParams::f1_2024_calibrated())
                .with_inner(InnerObjective::Envelope {
                    nodes: 24,
                    cache_dir: None,
                });
            optimize_setup(&cfg, cmaes_config.clone())
        };
        let a = run();
        let b = run();
        assert_eq!(
            a.best_time.to_bits(),
            b.best_time.to_bits(),
            "best_time differs"
        );
        assert_eq!(a.best_params.len(), b.best_params.len());
        for (i, (x, y)) in a.best_params.iter().zip(&b.best_params).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "best_params[{i}] differs");
        }
    }
}
