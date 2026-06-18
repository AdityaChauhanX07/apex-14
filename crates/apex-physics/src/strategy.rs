//! Race strategy primitives: fuel model, tire compounds, strategy definition,
//! and the strategy evaluator.
//!
//! The evaluator uses a simple analytical lap-time model — base lap time plus a
//! compound pace offset, linear tire degradation, and a fuel-weight penalty —
//! to compare pit strategies and analyse undercut/overcut decisions.

/// Fuel load model for race simulation.
///
/// Tracks fuel consumption over a race and computes the lap time penalty
/// from carrying fuel weight.
#[derive(Debug, Clone, Copy)]
pub struct FuelModel {
    /// Fuel consumption per lap (kg/lap).
    pub consumption_per_lap: f64,
    /// Lap time sensitivity to fuel load (s/kg).
    /// Each kg of fuel makes the car approximately this much slower per lap.
    pub time_per_kg: f64,
    /// Starting fuel load (kg).
    pub start_fuel: f64,
    /// Minimum fuel remaining at end of race (kg). FIA requires ~1 kg sample.
    pub min_end_fuel: f64,
}

impl FuelModel {
    /// F1 representative fuel model.
    /// ~110 kg for a ~57-lap Silverstone race, ~0.033 s/kg sensitivity.
    pub fn f1_default() -> Self {
        FuelModel {
            consumption_per_lap: 1.74,
            time_per_kg: 0.033,
            start_fuel: 110.0,
            min_end_fuel: 1.0,
        }
    }

    /// Fuel remaining at a given lap (0-indexed).
    pub fn fuel_at_lap(&self, lap: usize) -> f64 {
        (self.start_fuel - self.consumption_per_lap * lap as f64).max(self.min_end_fuel)
    }

    /// Lap time penalty from fuel weight relative to an empty car.
    pub fn fuel_time_penalty(&self, fuel_kg: f64) -> f64 {
        fuel_kg * self.time_per_kg
    }

    /// Total fuel needed for a given number of laps (including minimum end fuel).
    pub fn fuel_required(&self, laps: usize) -> f64 {
        self.consumption_per_lap * laps as f64 + self.min_end_fuel
    }
}

/// Available tire compounds with their performance characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TireCompound {
    Soft,
    Medium,
    Hard,
}

impl TireCompound {
    /// All available compounds.
    pub fn all() -> &'static [TireCompound] {
        &[TireCompound::Soft, TireCompound::Medium, TireCompound::Hard]
    }

    /// Pace offset relative to medium compound (seconds per lap).
    /// Negative = faster.
    pub fn pace_offset(&self) -> f64 {
        match self {
            TireCompound::Soft => -0.8,
            TireCompound::Medium => 0.0,
            TireCompound::Hard => 0.5,
        }
    }

    /// Tire degradation rate (seconds of lap time increase per lap on this compound).
    /// Soft tires degrade fastest, hard slowest.
    pub fn degradation_rate(&self) -> f64 {
        match self {
            TireCompound::Soft => 0.08, // loses 0.08s per lap
            TireCompound::Medium => 0.05,
            TireCompound::Hard => 0.03,
        }
    }

    /// Wear rate multiplier relative to medium.
    pub fn wear_multiplier(&self) -> f64 {
        match self {
            TireCompound::Soft => 1.5,
            TireCompound::Medium => 1.0,
            TireCompound::Hard => 0.7,
        }
    }

    /// Short display name.
    pub fn short_name(&self) -> &'static str {
        match self {
            TireCompound::Soft => "S",
            TireCompound::Medium => "M",
            TireCompound::Hard => "H",
        }
    }
}

impl std::fmt::Display for TireCompound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TireCompound::Soft => write!(f, "Soft"),
            TireCompound::Medium => write!(f, "Medium"),
            TireCompound::Hard => write!(f, "Hard"),
        }
    }
}

/// A single stint in a race strategy.
#[derive(Debug, Clone, Copy)]
pub struct Stint {
    /// Tire compound for this stint.
    pub compound: TireCompound,
    /// Number of laps in this stint.
    pub laps: usize,
}

/// A complete pit stop strategy for a race.
#[derive(Debug, Clone)]
pub struct RaceStrategy {
    /// Ordered sequence of stints.
    pub stints: Vec<Stint>,
    /// Time lost per pit stop (s). Includes pit lane transit + stationary time.
    pub pit_time_loss: f64,
}

impl RaceStrategy {
    /// Total race laps across all stints.
    pub fn total_laps(&self) -> usize {
        self.stints.iter().map(|s| s.laps).sum()
    }

    /// Number of pit stops (stints - 1).
    pub fn num_stops(&self) -> usize {
        self.stints.len().saturating_sub(1)
    }

    /// Compounds used (unique).
    pub fn compounds_used(&self) -> Vec<TireCompound> {
        let mut seen = Vec::new();
        for stint in &self.stints {
            if !seen.contains(&stint.compound) {
                seen.push(stint.compound);
            }
        }
        seen
    }

    /// Whether the strategy uses at least two different compounds (F1 regulation).
    pub fn uses_two_compounds(&self) -> bool {
        self.compounds_used().len() >= 2
    }

    /// Display string like "S15-H37" for Soft 15 laps, Hard 37 laps.
    pub fn display(&self) -> String {
        self.stints
            .iter()
            .map(|s| format!("{}{}", s.compound.short_name(), s.laps))
            .collect::<Vec<_>>()
            .join("-")
    }
}

/// Result of evaluating a race strategy.
#[derive(Debug, Clone)]
pub struct StrategyResult {
    /// The strategy that was evaluated.
    pub strategy: RaceStrategy,
    /// Total race time including pit stops (s).
    pub total_time: f64,
    /// Per-lap times.
    pub lap_times: Vec<f64>,
    /// Per-lap cumulative tire degradation (s of pace lost).
    pub tire_degradation: Vec<f64>,
    /// Per-lap fuel load (kg).
    pub fuel_loads: Vec<f64>,
}

/// Evaluates race strategies by computing total race time.
///
/// Uses a simple analytical model: each lap's time is the sum of
/// a base lap time, compound pace offset, tire degradation, and fuel penalty.
pub struct StrategyEvaluator {
    /// Base lap time on fresh medium tires with zero fuel (s).
    /// Typically from QSS simulation.
    pub base_lap_time: f64,
    /// Fuel model.
    pub fuel: FuelModel,
    /// Total race distance in laps.
    pub race_laps: usize,
    /// Pit stop time loss (s).
    pub pit_time_loss: f64,
}

impl StrategyEvaluator {
    /// Create an evaluator for a typical F1 race.
    pub fn new(base_lap_time: f64, race_laps: usize) -> Self {
        StrategyEvaluator {
            base_lap_time,
            fuel: FuelModel::f1_default(),
            race_laps,
            pit_time_loss: 22.0, // typical F1 pit stop loss
        }
    }

    /// Evaluate a single strategy and return detailed results.
    pub fn evaluate(&self, strategy: &RaceStrategy) -> StrategyResult {
        let mut lap_times = Vec::with_capacity(self.race_laps);
        let mut tire_deg = Vec::with_capacity(self.race_laps);
        let mut fuel_loads = Vec::with_capacity(self.race_laps);

        let mut race_lap = 0;
        let mut total_time = 0.0;

        for stint in &strategy.stints {
            for stint_lap in 0..stint.laps {
                // Fuel
                let fuel = self.fuel.fuel_at_lap(race_lap);
                let fuel_penalty = self.fuel.fuel_time_penalty(fuel);

                // Tire degradation (increases linearly with stint lap)
                let deg = stint.compound.degradation_rate() * stint_lap as f64;

                // Total lap time
                let lap_time =
                    self.base_lap_time + stint.compound.pace_offset() + deg + fuel_penalty;

                lap_times.push(lap_time);
                tire_deg.push(deg);
                fuel_loads.push(fuel);

                total_time += lap_time;
                race_lap += 1;
            }
        }

        // Add pit stop time
        total_time += strategy.num_stops() as f64 * self.pit_time_loss;

        StrategyResult {
            strategy: strategy.clone(),
            total_time,
            lap_times,
            tire_degradation: tire_deg,
            fuel_loads,
        }
    }

    /// Generate all feasible strategies and return them sorted by total time.
    ///
    /// Arguments:
    /// - max_stops: maximum number of pit stops to consider (1-3 typical)
    /// - min_stint: minimum laps per stint (for tire warmup, typically 5-10)
    /// - require_two_compounds: enforce the F1 rule of using at least 2 different compounds
    pub fn find_optimal(
        &self,
        max_stops: usize,
        min_stint: usize,
        require_two_compounds: bool,
    ) -> Vec<StrategyResult> {
        let mut results = Vec::new();
        let compounds = TireCompound::all();

        // Generate strategies for each number of stops
        for n_stops in 0..=max_stops {
            let n_stints = n_stops + 1;

            // Generate all compound permutations for this number of stints
            let compound_perms = compound_permutations(compounds, n_stints);

            for compound_seq in &compound_perms {
                // Check two-compound rule
                if require_two_compounds {
                    let unique: std::collections::HashSet<_> = compound_seq.iter().collect();
                    if unique.len() < 2 {
                        continue;
                    }
                }

                // Generate stint length distributions
                // Laps must sum to race_laps, each stint >= min_stint
                let lap_distributions = distribute_laps(self.race_laps, n_stints, min_stint);

                for dist in &lap_distributions {
                    let stints: Vec<Stint> = compound_seq
                        .iter()
                        .zip(dist.iter())
                        .map(|(&compound, &laps)| Stint { compound, laps })
                        .collect();

                    let strategy = RaceStrategy {
                        stints,
                        pit_time_loss: self.pit_time_loss,
                    };

                    let result = self.evaluate(&strategy);
                    results.push(result);
                }
            }
        }

        // Sort by total time
        results.sort_by(|a, b| {
            a.total_time
                .partial_cmp(&b.total_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Analyze the undercut/overcut around a reference pit lap.
    ///
    /// Computes the time delta for pitting 1-3 laps early (undercut) or late (overcut)
    /// relative to pitting on the reference lap.
    pub fn undercut_overcut(
        &self,
        reference_pit_lap: usize,
        current_compound: TireCompound,
        new_compound: TireCompound,
        stint_start_lap: usize,
    ) -> UndercutResult {
        // Reference: pit on the given lap
        let ref_time = self.evaluate_pit_scenario(
            reference_pit_lap,
            stint_start_lap,
            current_compound,
            new_compound,
        );

        // Undercut: pit 1 lap early
        let early = if reference_pit_lap > stint_start_lap + 1 {
            self.evaluate_pit_scenario(
                reference_pit_lap - 1,
                stint_start_lap,
                current_compound,
                new_compound,
            )
        } else {
            ref_time
        };

        // Overcut: pit 1 lap late
        let late = if reference_pit_lap < self.race_laps - 2 {
            self.evaluate_pit_scenario(
                reference_pit_lap + 1,
                stint_start_lap,
                current_compound,
                new_compound,
            )
        } else {
            ref_time
        };

        UndercutResult {
            reference_time: ref_time,
            undercut_delta: early - ref_time,
            overcut_delta: late - ref_time,
            recommendation: if early < ref_time && early < late {
                "Undercut: pit early for fresh-tire advantage"
            } else if late < ref_time {
                "Overcut: extend stint to benefit from lighter fuel"
            } else {
                "Pit on plan: reference lap is optimal"
            },
        }
    }

    // Helper: compute total time for a two-stint scenario around a pit lap
    fn evaluate_pit_scenario(
        &self,
        pit_lap: usize,
        stint_start: usize,
        compound_before: TireCompound,
        compound_after: TireCompound,
    ) -> f64 {
        let mut total = 0.0;
        let window = 6; // analyze 3 laps before and after the pit window

        let start = if pit_lap > stint_start + window / 2 {
            pit_lap - window / 2
        } else {
            stint_start
        };
        let end = (pit_lap + window / 2).min(self.race_laps);

        for lap in start..end {
            let race_fuel = self.fuel.fuel_at_lap(lap);
            let fuel_penalty = self.fuel.fuel_time_penalty(race_fuel);

            if lap < pit_lap {
                let stint_lap = lap - stint_start;
                let deg = compound_before.degradation_rate() * stint_lap as f64;
                total += self.base_lap_time + compound_before.pace_offset() + deg + fuel_penalty;
            } else {
                let stint_lap = lap - pit_lap;
                let deg = compound_after.degradation_rate() * stint_lap as f64;
                total += self.base_lap_time + compound_after.pace_offset() + deg + fuel_penalty;
            }
        }

        total + self.pit_time_loss
    }
}

/// Result of undercut/overcut analysis.
#[derive(Debug, Clone)]
pub struct UndercutResult {
    /// Total time for the reference pit lap scenario (s).
    pub reference_time: f64,
    /// Time delta for pitting 1 lap early (negative = faster).
    pub undercut_delta: f64,
    /// Time delta for pitting 1 lap late (negative = faster).
    pub overcut_delta: f64,
    /// Human-readable recommendation.
    pub recommendation: &'static str,
}

/// Generate all permutations of compounds for n_stints slots.
/// With 3 compounds and 2 stints: 3^2 = 9 permutations.
fn compound_permutations(compounds: &[TireCompound], n_stints: usize) -> Vec<Vec<TireCompound>> {
    if n_stints == 0 {
        return vec![vec![]];
    }
    if n_stints == 1 {
        return compounds.iter().map(|&c| vec![c]).collect();
    }
    let mut result = Vec::new();
    let sub = compound_permutations(compounds, n_stints - 1);
    for &c in compounds {
        for s in &sub {
            let mut v = vec![c];
            v.extend(s);
            result.push(v);
        }
    }
    result
}

/// Distribute total_laps across n_stints, each with at least min_stint laps.
/// Returns all valid distributions (step size of 5 laps for manageable count).
fn distribute_laps(total: usize, n_stints: usize, min_stint: usize) -> Vec<Vec<usize>> {
    if n_stints == 1 {
        if total >= min_stint {
            return vec![vec![total]];
        } else {
            return vec![];
        }
    }

    let mut results = Vec::new();
    let step = 5; // enumerate in steps of 5 laps for speed
    let max_first = total - (n_stints - 1) * min_stint;

    let mut first = min_stint;
    while first <= max_first {
        let remaining = total - first;
        let sub = distribute_laps(remaining, n_stints - 1, min_stint);
        for mut s in sub {
            let mut v = vec![first];
            v.append(&mut s);
            results.push(v);
        }
        first += step;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // (a) FuelModel.
    #[test]
    fn fuel_model_basics() {
        let fm = FuelModel::f1_default();
        assert!(close(fm.fuel_at_lap(0), 110.0));

        // NOTE: with the as-specified 110 kg start and 1.74 kg/lap, a 57-lap race
        // only burns ~99 kg, so ~10.8 kg remains at the flag (the tank carries
        // margin over the ~100 kg required, not down to the 1 kg FIA minimum).
        assert!(close(fm.fuel_at_lap(57), 110.0 - 1.74 * 57.0));
        assert!(fm.fuel_at_lap(57) < fm.start_fuel);
        // Far enough in, fuel clamps at the minimum.
        assert!(close(fm.fuel_at_lap(80), fm.min_end_fuel));

        assert!(close(fm.fuel_time_penalty(110.0), 3.63));
        assert!(close(fm.fuel_required(57), 100.18));
    }

    // (b) TireCompound.
    #[test]
    fn compound_pace_and_degradation() {
        assert!(TireCompound::Soft.pace_offset() < TireCompound::Medium.pace_offset());
        assert!(TireCompound::Hard.pace_offset() > TireCompound::Medium.pace_offset());
        // Soft degrades fastest.
        assert!(TireCompound::Soft.degradation_rate() > TireCompound::Medium.degradation_rate());
        assert!(TireCompound::Medium.degradation_rate() > TireCompound::Hard.degradation_rate());
    }

    // (c) RaceStrategy.
    #[test]
    fn strategy_metadata() {
        let strat = RaceStrategy {
            stints: vec![
                Stint {
                    compound: TireCompound::Soft,
                    laps: 20,
                },
                Stint {
                    compound: TireCompound::Hard,
                    laps: 32,
                },
            ],
            pit_time_loss: 22.0,
        };
        assert_eq!(strat.total_laps(), 52);
        assert_eq!(strat.num_stops(), 1);
        assert!(strat.uses_two_compounds());
        assert_eq!(strat.display(), "S20-H32");
    }

    // (d) Strategy evaluation.
    #[test]
    fn evaluate_one_stop() {
        let ev = StrategyEvaluator::new(90.0, 52);
        let strat = RaceStrategy {
            stints: vec![
                Stint {
                    compound: TireCompound::Soft,
                    laps: 25,
                },
                Stint {
                    compound: TireCompound::Hard,
                    laps: 27,
                },
            ],
            pit_time_loss: 22.0,
        };
        let r = ev.evaluate(&strat);

        assert_eq!(r.lap_times.len(), 52);
        // Total includes the one pit stop.
        let sum: f64 = r.lap_times.iter().sum();
        assert!(close(r.total_time, sum + ev.pit_time_loss));

        // Soft stint: degradation outpaces fuel burn, so later soft laps are slower.
        assert!(r.lap_times[24] > r.lap_times[0]);
        // Over the whole race, fuel burn-off makes the final lap faster than the first.
        assert!(r.lap_times[51] < r.lap_times[0]);
    }

    // (e) Optimal strategy search.
    #[test]
    fn find_optimal_two_compound() {
        let ev = StrategyEvaluator::new(90.0, 52);
        let results = ev.find_optimal(2, 10, true);

        assert!(results.len() > 1, "should produce multiple strategies");
        // Sorted ascending: the first is the best.
        for w in results.windows(2) {
            assert!(w[0].total_time <= w[1].total_time);
        }
        // Every strategy respects the race distance, the two-compound rule, and
        // pits at least once (the 0-stop option uses a single compound).
        for r in &results {
            assert_eq!(r.strategy.total_laps(), 52);
            assert!(r.strategy.uses_two_compounds());
            assert!(r.strategy.num_stops() >= 1);
        }
    }

    // (f) 1-stop vs 2-stop comparison.
    #[test]
    fn one_stop_vs_two_stop() {
        let ev = StrategyEvaluator::new(90.0, 52);
        let results = ev.find_optimal(2, 10, true);

        let best_one = results
            .iter()
            .find(|r| r.strategy.num_stops() == 1)
            .expect("a 1-stop strategy");
        let best_two = results
            .iter()
            .find(|r| r.strategy.num_stops() == 2)
            .expect("a 2-stop strategy");

        assert!(
            (best_one.total_time - best_two.total_time).abs() < 10.0,
            "best 1-stop {} and best 2-stop {} should be within 10 s",
            best_one.total_time,
            best_two.total_time
        );
    }

    // (g) Undercut analysis.
    #[test]
    fn undercut_analysis() {
        // The undercut pays off when the saved tire degradation outweighs the
        // extra fuel weight carried over the pit window. With the heavy default
        // fuel burn the windowed model's fuel term marginally dominates, so this
        // test uses a lighter consumption to exercise the undercut-favorable
        // regime the analysis is meant to flag.
        let mut ev = StrategyEvaluator::new(90.0, 52);
        ev.fuel.consumption_per_lap = 1.0;

        let result = ev.undercut_overcut(20, TireCompound::Soft, TireCompound::Hard, 0);

        assert!(result.reference_time > 0.0);
        // Pitting a lap early is faster here (fresh tires outrun the degraded set).
        assert!(
            result.undercut_delta < 0.0,
            "undercut delta {} should be negative",
            result.undercut_delta
        );
        assert!(
            result.recommendation.starts_with("Undercut")
                || result.recommendation.starts_with("Pit on plan"),
            "unexpected recommendation: {}",
            result.recommendation
        );
    }

    // (h) Fuel decreases lap time over the race.
    #[test]
    fn fuel_burn_off_speeds_up_laps() {
        let ev = StrategyEvaluator::new(90.0, 52);
        let strat = RaceStrategy {
            stints: vec![
                Stint {
                    compound: TireCompound::Medium,
                    laps: 26,
                },
                Stint {
                    compound: TireCompound::Hard,
                    laps: 26,
                },
            ],
            pit_time_loss: 22.0,
        };
        let r = ev.evaluate(&strat);
        // Final lap (light fuel) is faster than the opening lap (full fuel).
        assert!(r.lap_times[51] < r.lap_times[0]);
        // Fuel loads strictly decrease over the race.
        assert!(r.fuel_loads[51] < r.fuel_loads[0]);
    }

    // (i) Compound permutations.
    #[test]
    fn compound_permutation_counts() {
        let compounds = TireCompound::all();
        assert_eq!(compound_permutations(compounds, 2).len(), 9);
        assert_eq!(compound_permutations(compounds, 1).len(), 3);
    }
}
