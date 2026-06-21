//! Monte Carlo race simulation framework.
//!
//! Runs thousands of stochastic race simulations in parallel and
//! aggregates the results into probability distributions for
//! finishing positions, win likelihood, and expected points.

use rayon::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{RaceConfig, RaceEntry};
use crate::sim::{simulate_race_stochastic, RaceResult};

/// Aggregated results from a Monte Carlo race simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Number of simulations run.
    pub n_sims: usize,
    /// Number of entries (cars).
    pub n_cars: usize,
    /// Win probability for each car (index matches entry order).
    pub win_probs: Vec<f64>,
    /// Podium (top 3) probability for each car.
    pub podium_probs: Vec<f64>,
    /// Points finish (top 10) probability for each car.
    pub points_probs: Vec<f64>,
    /// Expected (mean) finishing position for each car.
    pub expected_position: Vec<f64>,
    /// Expected championship points for each car (F1 scoring).
    pub expected_points: Vec<f64>,
    /// Position distribution: `position_counts[car][pos] = count`.
    /// `pos` is 0-indexed (0 = 1st place).
    pub position_counts: Vec<Vec<usize>>,
    /// DNF count per car across all simulations.
    pub dnf_counts: Vec<usize>,
    /// Total safety car deployments across all simulations.
    pub total_sc_deployments: usize,
    /// Mean number of safety cars per race.
    pub mean_sc_per_race: f64,
}

/// Run N Monte Carlo race simulations in parallel.
///
/// Each simulation uses a deterministic RNG seeded from the base seed
/// plus the simulation index, ensuring reproducibility.
///
/// Uses Rayon for parallel execution across CPU cores.
pub fn monte_carlo_race(
    config: &RaceConfig,
    entries: &[RaceEntry],
    n_sims: usize,
    seed: u64,
) -> MonteCarloResult {
    let n_cars = entries.len();

    // Run simulations in parallel
    let results: Vec<RaceResult> = (0..n_sims)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_race_stochastic(&mut rng, config, entries)
        })
        .collect();

    // Aggregate results
    aggregate_results(&results, n_cars, n_sims)
}

/// Aggregate individual race results into Monte Carlo statistics.
fn aggregate_results(results: &[RaceResult], n_cars: usize, n_sims: usize) -> MonteCarloResult {
    let mut position_counts = vec![vec![0usize; n_cars]; n_cars];
    let mut dnf_counts = vec![0usize; n_cars];
    let mut total_points = vec![0.0f64; n_cars];
    let mut total_position = vec![0.0f64; n_cars];
    let mut win_count = vec![0usize; n_cars];
    let mut podium_count = vec![0usize; n_cars];
    let mut points_count = vec![0usize; n_cars];
    let mut total_sc = 0usize;

    for result in results {
        for (pos, &car_idx) in result.finishing_order.iter().enumerate() {
            if pos < n_cars {
                position_counts[car_idx][pos] += 1;
            }
            total_position[car_idx] += (pos + 1) as f64;
            total_points[car_idx] += result.points_for(car_idx);

            if pos == 0 {
                win_count[car_idx] += 1;
            }
            if pos < 3 {
                podium_count[car_idx] += 1;
            }
            if pos < 10 {
                points_count[car_idx] += 1;
            }
        }

        // Count DNFs by car index from the end-of-race car states.
        for (car_idx, state) in result.car_states.iter().enumerate() {
            if state.retired {
                dnf_counts[car_idx] += 1;
            }
        }

        // Safety car deployments are carried on the race result.
        total_sc += result.sc_count;
    }

    let n = n_sims as f64;
    MonteCarloResult {
        n_sims,
        n_cars,
        win_probs: win_count.iter().map(|&c| c as f64 / n).collect(),
        podium_probs: podium_count.iter().map(|&c| c as f64 / n).collect(),
        points_probs: points_count.iter().map(|&c| c as f64 / n).collect(),
        expected_position: total_position.iter().map(|&t| t / n).collect(),
        expected_points: total_points.iter().map(|&t| t / n).collect(),
        position_counts,
        dnf_counts,
        total_sc_deployments: total_sc,
        mean_sc_per_race: total_sc as f64 / n,
    }
}

/// Format Monte Carlo results as a human-readable report.
pub fn format_report(result: &MonteCarloResult, entries: &[RaceEntry]) -> String {
    let mut report = String::new();
    report.push_str(&format!(
        "Monte Carlo Race Simulation ({} races)\n",
        result.n_sims
    ));
    report.push_str(&format!("{}\n\n", "=".repeat(60)));

    // Sort by expected position for display
    let mut order: Vec<usize> = (0..result.n_cars).collect();
    order.sort_by(|&a, &b| {
        result.expected_position[a]
            .partial_cmp(&result.expected_position[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    report.push_str(&format!(
        "{:<20} {:>6} {:>8} {:>8} {:>8} {:>6} {:>6}\n",
        "Driver", "E[Pos]", "Win%", "Pod%", "Pts%", "E[Pts]", "DNF%"
    ));
    report.push_str(&format!("{}\n", "-".repeat(70)));

    for &idx in &order {
        let name = &entries[idx].name;
        let epos = result.expected_position[idx];
        let win = result.win_probs[idx] * 100.0;
        let pod = result.podium_probs[idx] * 100.0;
        let pts = result.points_probs[idx] * 100.0;
        let epts = result.expected_points[idx];
        let dnf = result.dnf_counts[idx] as f64 / result.n_sims as f64 * 100.0;

        report.push_str(&format!(
            "{name:<20} {epos:>6.1} {win:>7.1}% {pod:>7.1}% {pts:>7.1}% {epts:>6.1} {dnf:>5.1}%\n"
        ));
    }

    report.push_str(&format!(
        "\nSafety cars: {:.1} per race average\n",
        result.mean_sc_per_race
    ));

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RaceStrategy, TireCompound};

    /// A simple entry with the given base lap time, no planned stops, and no
    /// stochastic variance of its own (so race outcomes are driven by pace and
    /// the configured event probabilities).
    fn entry(name: &str, base: f64) -> RaceEntry {
        RaceEntry {
            name: name.to_string(),
            base_lap_time: base,
            lap_time_variance: 0.15,
            tire_deg_factor: 1.0,
            pit_crew_variance: 0.0,
            strategy: RaceStrategy {
                stops: vec![],
                start_compound: TireCompound::Medium,
            },
            driver_skill: 0.9,
        }
    }

    /// Five cars with a clear, monotonic pace spread.
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
    fn test_monte_carlo_100_sims() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 10);
        let result = monte_carlo_race(&cfg, &entries, 100, 42);

        assert_eq!(result.n_sims, 100);
        assert_eq!(result.n_cars, 5);
        assert_eq!(result.win_probs.len(), 5);
        assert_eq!(result.podium_probs.len(), 5);
        assert_eq!(result.points_probs.len(), 5);
        assert_eq!(result.expected_position.len(), 5);
        assert_eq!(result.expected_points.len(), 5);

        for &p in result
            .win_probs
            .iter()
            .chain(&result.podium_probs)
            .chain(&result.points_probs)
        {
            assert!((0.0..=1.0).contains(&p), "probability {p} out of [0, 1]");
        }
    }

    #[test]
    fn test_win_probs_sum_to_one() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 10);
        let result = monte_carlo_race(&cfg, &entries, 200, 7);

        let sum: f64 = result.win_probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "win probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_faster_car_wins_more() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 12);
        let result = monte_carlo_race(&cfg, &entries, 500, 123);

        let fastest = result.win_probs[0];
        for (i, &p) in result.win_probs.iter().enumerate().skip(1) {
            assert!(
                fastest >= p,
                "fastest car (idx 0, {fastest:.3}) should win at least as often as car {i} ({p:.3})"
            );
        }
        assert!(
            fastest > 0.5,
            "the clearly fastest car should win a majority of races, got {fastest:.3}"
        );
    }

    #[test]
    fn test_expected_position_ordered() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 12);
        let result = monte_carlo_race(&cfg, &entries, 500, 555);

        // Expected positions should monotonically increase with base lap time.
        for i in 1..result.n_cars {
            assert!(
                result.expected_position[i] >= result.expected_position[i - 1],
                "car {i} (E[pos] {:.2}) should not finish ahead of faster car {} (E[pos] {:.2})",
                result.expected_position[i],
                i - 1,
                result.expected_position[i - 1]
            );
        }
    }

    #[test]
    fn test_points_system() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 12);
        let result = monte_carlo_race(&cfg, &entries, 500, 88);

        // With only 5 cars, every finisher scores, so the dominant winner should
        // average close to the 25-point maximum.
        assert!(
            result.expected_points[0] > 20.0,
            "winner-caliber car should average near 25 points, got {:.1}",
            result.expected_points[0]
        );
        // The slowest car of five never reaches the podium points but still
        // collects the 5th-place 10 points when it finishes; it should clearly
        // trail the leader.
        assert!(
            result.expected_points[0] > result.expected_points[4],
            "faster car should earn more points: {:.1} vs {:.1}",
            result.expected_points[0],
            result.expected_points[4]
        );
    }

    #[test]
    fn test_backmarker_low_points() {
        // A 20-car grid: the backmarker should rarely score, so its expected
        // points should be near zero.
        let entries = crate::config::default_f1_grid(90.0);
        let cfg = RaceConfig::silverstone_default();
        let result = monte_carlo_race(&cfg, &entries, 200, 314);

        let backmarker = result.n_cars - 1;
        assert!(
            result.expected_points[backmarker] < 2.0,
            "backmarker should average near 0 points, got {:.2}",
            result.expected_points[backmarker]
        );
    }

    #[test]
    fn test_reproducible_with_same_seed() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 10);
        let a = monte_carlo_race(&cfg, &entries, 100, 2024);
        let b = monte_carlo_race(&cfg, &entries, 100, 2024);

        assert_eq!(a.win_probs, b.win_probs);
        assert_eq!(a.expected_position, b.expected_position);
        assert_eq!(a.position_counts, b.position_counts);
        assert_eq!(a.dnf_counts, b.dnf_counts);
        assert_eq!(a.total_sc_deployments, b.total_sc_deployments);
    }

    #[test]
    fn test_different_seeds_differ() {
        // Use the full grid with the default event config so safety cars, DNFs,
        // overtakes, and lap variance can reshuffle outcomes between seeds.
        let entries = crate::config::default_f1_grid(90.0);
        let cfg = RaceConfig::silverstone_default();
        let a = monte_carlo_race(&cfg, &entries, 100, 1);
        let b = monte_carlo_race(&cfg, &entries, 100, 2);

        assert!(
            a.position_counts != b.position_counts || a.expected_points != b.expected_points,
            "different seeds should produce different aggregate statistics"
        );
    }

    #[test]
    fn test_format_report_nonempty() {
        let entries = five_car_grid();
        let cfg = RaceConfig::for_track(5000.0, 10);
        let result = monte_carlo_race(&cfg, &entries, 50, 9);
        let report = format_report(&result, &entries);

        assert!(!report.is_empty(), "report should not be empty");
        for e in &entries {
            assert!(
                report.contains(&e.name),
                "report should mention driver {}",
                e.name
            );
        }
        assert!(
            report.contains("Monte Carlo"),
            "report should contain a header"
        );
    }
}
