//! Race strategy optimization under uncertainty.
//!
//! Uses CMA-ES to optimize pit stop timing for a target car,
//! evaluating each candidate strategy via Monte Carlo simulation.
//! The objective is to maximize expected championship points.

use apex_optimizer::{CmaEs, CmaEsConfig};

use crate::config::{PlannedStop, RaceConfig, RaceEntry, RaceStrategy, TireCompound};
use crate::monte_carlo::monte_carlo_race;

/// Configuration for strategy optimization.
#[derive(Debug, Clone)]
pub struct StrategyOptConfig {
    /// Number of Monte Carlo sims per strategy evaluation.
    pub n_sims_per_eval: usize,
    /// CMA-ES configuration.
    pub cmaes_config: CmaEsConfig,
    /// Base RNG seed for reproducibility.
    pub seed: u64,
    /// Number of stops to optimize (1 or 2).
    pub n_stops: usize,
    /// Compound sequence for a 1-stop: `[start, after_stop1]`.
    /// For a 2-stop: `[start, after_stop1, after_stop2]`.
    pub compounds: Vec<TireCompound>,
}

impl Default for StrategyOptConfig {
    fn default() -> Self {
        Self {
            n_sims_per_eval: 100,
            cmaes_config: CmaEsConfig {
                max_generations: 30,
                initial_sigma: 0.3,
                ..CmaEsConfig::default()
            },
            seed: 42,
            n_stops: 1,
            compounds: vec![TireCompound::Medium, TireCompound::Hard],
        }
    }
}

/// Result of strategy optimization.
#[derive(Debug, Clone)]
pub struct StrategyOptResult {
    /// Optimized pit stop laps.
    pub stop_laps: Vec<usize>,
    /// Tire compounds used (start + after each stop).
    pub compounds: Vec<TireCompound>,
    /// Expected points with optimized strategy.
    pub expected_points: f64,
    /// Expected position with optimized strategy.
    pub expected_position: f64,
    /// Win probability with optimized strategy.
    pub win_prob: f64,
    /// Expected points with the original strategy (for comparison).
    pub baseline_points: f64,
    /// Number of CMA-ES generations run.
    pub generations: usize,
}

/// Optimize race strategy for a target car.
///
/// Tries to maximize expected championship points for the specified car
/// by optimizing pit stop timing. Other cars keep their original strategies.
pub fn optimize_strategy(
    race_config: &RaceConfig,
    entries: &[RaceEntry],
    target_car: usize,
    opt_config: &StrategyOptConfig,
) -> StrategyOptResult {
    let n_laps = race_config.n_laps;

    // Evaluate baseline strategy
    let baseline_mc = monte_carlo_race(
        race_config,
        entries,
        opt_config.n_sims_per_eval,
        opt_config.seed,
    );
    let baseline_points = baseline_mc.expected_points[target_car];

    // Define bounds for pit laps (as fractions of race distance)
    // 1-stop: [0.2, 0.8] (pit between 20% and 80% of race)
    // 2-stop: [0.15, 0.55] for stop1, [0.45, 0.85] for stop2
    let (initial_mean, bounds) = match opt_config.n_stops {
        1 => (vec![0.5], vec![(0.2, 0.8)]),
        2 => (vec![0.33, 0.67], vec![(0.15, 0.55), (0.45, 0.85)]),
        _ => (vec![0.5], vec![(0.2, 0.8)]),
    };

    let mut cmaes = CmaEs::new(initial_mean, bounds, opt_config.cmaes_config.clone());

    loop {
        let candidates = cmaes.ask();
        // All candidates in a generation share a seed offset so the comparison
        // between them is fair (same Monte Carlo noise), but it varies per
        // generation to avoid overfitting a single noise realization.
        let gen_seed = opt_config
            .seed
            .wrapping_add(cmaes.generation() as u64 * 1000);
        let fitnesses: Vec<f64> = candidates
            .iter()
            .map(|params| {
                // Convert fractional laps to integer stop laps
                let stop_laps = params_to_stop_laps(params, n_laps, opt_config.n_stops);

                // Build modified strategy
                let strategy = build_strategy(&stop_laps, &opt_config.compounds);

                // Create modified entries with the new strategy for target car
                let mut modified = entries.to_vec();
                modified[target_car].strategy = strategy;

                // Run Monte Carlo -- use a different seed offset per generation
                // to avoid correlation, but keep it deterministic
                let mc =
                    monte_carlo_race(race_config, &modified, opt_config.n_sims_per_eval, gen_seed);

                // Minimize negative expected points (CMA-ES minimizes)
                -mc.expected_points[target_car]
            })
            .collect();

        cmaes.tell(&fitnesses);

        log::info!(
            "Gen {:>3} | best expected pts: {:.2} | sigma: {:.4}",
            cmaes.generation(),
            -cmaes.best_fitness(),
            cmaes.sigma()
        );

        if cmaes.should_stop() {
            break;
        }
    }

    // Extract best result
    let best_params = cmaes.best_params();
    let stop_laps = params_to_stop_laps(best_params, n_laps, opt_config.n_stops);

    // Run a final MC with more sims for accurate stats
    let final_strategy = build_strategy(&stop_laps, &opt_config.compounds);
    let mut final_entries = entries.to_vec();
    final_entries[target_car].strategy = final_strategy;
    let final_mc = monte_carlo_race(
        race_config,
        &final_entries,
        opt_config.n_sims_per_eval * 5,
        opt_config.seed.wrapping_add(999_999),
    );

    StrategyOptResult {
        stop_laps,
        compounds: opt_config.compounds.clone(),
        expected_points: final_mc.expected_points[target_car],
        expected_position: final_mc.expected_position[target_car],
        win_prob: final_mc.win_probs[target_car],
        baseline_points,
        generations: cmaes.generation(),
    }
}

/// Convert CMA-ES parameters (fractions) to integer pit stop laps.
fn params_to_stop_laps(params: &[f64], n_laps: usize, n_stops: usize) -> Vec<usize> {
    let mut laps: Vec<usize> = params
        .iter()
        .take(n_stops)
        .map(|&frac| {
            let lap = (frac * n_laps as f64).round() as usize;
            lap.clamp(2, n_laps.saturating_sub(1)) // don't pit on lap 1 or last lap
        })
        .collect();
    laps.sort(); // ensure stops are in order
                 // Ensure minimum gap between stops (at least 5 laps)
    for i in 1..laps.len() {
        if laps[i] <= laps[i - 1] + 4 {
            laps[i] = laps[i - 1] + 5;
        }
    }
    laps
}

/// Build a [`RaceStrategy`] from stop laps and a compound sequence.
fn build_strategy(stop_laps: &[usize], compounds: &[TireCompound]) -> RaceStrategy {
    let start_compound = compounds.first().copied().unwrap_or(TireCompound::Medium);
    let stops = stop_laps
        .iter()
        .enumerate()
        .map(|(i, &lap)| PlannedStop {
            lap,
            compound: compounds.get(i + 1).copied().unwrap_or(TireCompound::Hard),
        })
        .collect();
    RaceStrategy {
        start_compound,
        stops,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RaceStrategy;

    /// A simple entry with the given base lap time and a 1-stop strategy.
    fn entry(name: &str, base: f64) -> RaceEntry {
        RaceEntry {
            name: name.to_string(),
            base_lap_time: base,
            lap_time_variance: 0.15,
            tire_deg_factor: 1.0,
            pit_crew_variance: 0.0,
            strategy: RaceStrategy {
                stops: vec![PlannedStop {
                    lap: 5,
                    compound: TireCompound::Hard,
                }],
                start_compound: TireCompound::Medium,
            },
            driver_skill: 0.9,
        }
    }

    fn five_car_grid() -> Vec<RaceEntry> {
        vec![
            entry("A", 90.0),
            entry("B", 90.5),
            entry("C", 91.0),
            entry("D", 91.5),
            entry("E", 92.0),
        ]
    }

    #[test]
    fn test_params_to_stop_laps() {
        // 1-stop at the midpoint of a 52-lap race -> lap 26.
        let laps = params_to_stop_laps(&[0.5], 52, 1);
        assert_eq!(laps, vec![26]);

        // 2-stop at thirds -> ~17 and ~35.
        let laps = params_to_stop_laps(&[0.33, 0.67], 52, 2);
        assert_eq!(laps.len(), 2);
        assert_eq!(laps[0], 17);
        assert_eq!(laps[1], 35);
    }

    #[test]
    fn test_build_strategy() {
        let strategy = build_strategy(&[20], &[TireCompound::Soft, TireCompound::Medium]);
        assert_eq!(strategy.start_compound, TireCompound::Soft);
        assert_eq!(strategy.stops.len(), 1);
        assert_eq!(strategy.stops[0].lap, 20);
        assert_eq!(strategy.stops[0].compound, TireCompound::Medium);

        // Two stops with a full 3-compound sequence.
        let strategy = build_strategy(
            &[15, 35],
            &[TireCompound::Soft, TireCompound::Medium, TireCompound::Hard],
        );
        assert_eq!(strategy.stops.len(), 2);
        assert_eq!(strategy.stops[0].compound, TireCompound::Medium);
        assert_eq!(strategy.stops[1].compound, TireCompound::Hard);
    }

    #[test]
    fn test_stop_laps_minimum_gap() {
        // Two near-identical fractions must be separated by at least 5 laps.
        let laps = params_to_stop_laps(&[0.5, 0.51], 52, 2);
        assert_eq!(laps.len(), 2);
        assert!(
            laps[1] >= laps[0] + 5,
            "stops should be at least 5 laps apart, got {laps:?}"
        );
    }

    #[test]
    fn test_optimize_strategy_returns_valid() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 10);
        let opt = StrategyOptConfig {
            n_sims_per_eval: 20,
            cmaes_config: CmaEsConfig {
                max_generations: 3,
                initial_sigma: 0.3,
                ..CmaEsConfig::default()
            },
            seed: 7,
            n_stops: 1,
            compounds: vec![TireCompound::Medium, TireCompound::Hard],
        };

        let result = optimize_strategy(&cfg, &entries, 0, &opt);

        assert_eq!(result.stop_laps.len(), 1);
        for &lap in &result.stop_laps {
            assert!(
                lap >= 2 && lap < cfg.n_laps,
                "stop lap {lap} should be within race distance (2..{})",
                cfg.n_laps
            );
        }
        assert!(
            result.expected_points.is_finite(),
            "expected points should be finite"
        );
        assert!(
            result.expected_position.is_finite() && result.expected_position > 0.0,
            "expected position should be a valid finite position"
        );
        assert!(
            (0.0..=1.0).contains(&result.win_prob),
            "win probability {} out of [0, 1]",
            result.win_prob
        );
        assert!(
            result.generations <= 3,
            "should respect the generation budget"
        );
    }

    #[test]
    fn test_optimized_not_worse() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 12);
        let opt = StrategyOptConfig {
            n_sims_per_eval: 30,
            cmaes_config: CmaEsConfig {
                max_generations: 4,
                initial_sigma: 0.3,
                ..CmaEsConfig::default()
            },
            seed: 11,
            n_stops: 1,
            compounds: vec![TireCompound::Medium, TireCompound::Hard],
        };

        let result = optimize_strategy(&cfg, &entries, 0, &opt);

        // The optimizer should not make the target car significantly worse than
        // its baseline strategy. A small tolerance absorbs Monte Carlo noise
        // between the baseline and the higher-sim final evaluation.
        assert!(
            result.expected_points >= result.baseline_points - 1.0,
            "optimized points {:.2} should not be much worse than baseline {:.2}",
            result.expected_points,
            result.baseline_points
        );
    }
}
