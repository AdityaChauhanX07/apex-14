//! Lap-by-lap race simulation engine.
//!
//! Simulates a full race for N cars, tracking positions, pit stops,
//! tire wear, and fuel consumption. This is the deterministic core;
//! probabilistic events are layered on top in the events module.

use std::cmp::Ordering;

use crate::config::{RaceConfig, RaceEntry, TireCompound};

/// State of a single car during the race.
#[derive(Debug, Clone)]
pub struct CarState {
    /// Current position (1-indexed).
    pub position: usize,
    /// Cumulative race time (s).
    pub race_time: f64,
    /// Current tire compound.
    pub compound: TireCompound,
    /// Laps on the current set of tires.
    pub tire_age: usize,
    /// Number of pit stops completed.
    pub pit_stops: usize,
    /// Remaining fuel (kg).
    pub fuel_remaining: f64,
    /// Whether the car has retired (DNF).
    pub retired: bool,
    /// Whether the car pitted this lap.
    pub in_pit: bool,
    /// Lap times recorded (s).
    pub lap_times: Vec<f64>,
    /// Index into the strategy's planned stops (next stop to execute).
    pub next_stop_index: usize,
}

/// Complete result of a race simulation.
#[derive(Debug, Clone)]
pub struct RaceResult {
    /// Car indices in finishing order (winner first).
    pub finishing_order: Vec<usize>,
    /// Car states at the end of the race.
    pub car_states: Vec<CarState>,
    /// Total race time for the winner (s).
    pub winner_time: f64,
    /// Number of laps completed.
    pub laps_completed: usize,
}

impl RaceResult {
    /// Get the finishing position (1-indexed) of a car by index.
    pub fn position_of(&self, car_index: usize) -> Option<usize> {
        self.finishing_order
            .iter()
            .position(|&idx| idx == car_index)
            .map(|p| p + 1)
    }

    /// Get championship points for a car by index.
    ///
    /// Uses standard F1 points: 25, 18, 15, 12, 10, 8, 6, 4, 2, 1 for the top
    /// ten finishers, and 0 otherwise.
    pub fn points_for(&self, car_index: usize) -> f64 {
        let points_table = [25.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0];
        match self.position_of(car_index) {
            Some(pos) if pos <= 10 => points_table[pos - 1],
            _ => 0.0,
        }
    }
}

/// Simulate a complete race (deterministic, no random events).
///
/// Runs lap-by-lap, computing each lap time from base performance, the current
/// tire compound's fresh pace and degradation (scaled by the entry's
/// `tire_deg_factor`), the fuel load, and any pit stop on that lap. Returns the
/// complete race result with cars sorted into finishing order.
pub fn simulate_race(config: &RaceConfig, entries: &[RaceEntry]) -> RaceResult {
    let n_cars = entries.len();

    // Initialize car states in grid order.
    let mut states: Vec<CarState> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| CarState {
            position: i + 1,
            race_time: 0.0,
            compound: entry.strategy.start_compound,
            tire_age: 0,
            pit_stops: 0,
            fuel_remaining: config.start_fuel_kg,
            retired: false,
            in_pit: false,
            lap_times: Vec::with_capacity(config.n_laps),
            next_stop_index: 0,
        })
        .collect();

    // Simulate lap by lap.
    for lap in 1..=config.n_laps {
        for (state, entry) in states.iter_mut().zip(entries.iter()) {
            if state.retired {
                continue;
            }

            // A planned stop fires when the next unexecuted stop targets this lap.
            let pitting = entry
                .strategy
                .stops
                .get(state.next_stop_index)
                .is_some_and(|stop| stop.lap == lap);

            // Lap time from the CURRENT compound (the one run this lap), its
            // degradation over the tires' current age, and the fuel load.
            let tire_deg =
                state.compound.degradation_rate() * state.tire_age as f64 * entry.tire_deg_factor;
            let compound_offset = state.compound.pace_offset();
            let fuel_effect =
                (config.start_fuel_kg - state.fuel_remaining) * config.fuel_time_factor;

            let mut total_lap_time = entry.base_lap_time + compound_offset + tire_deg - fuel_effect;

            // A pit stop adds the pit-lane transit plus stationary time.
            if pitting {
                total_lap_time += config.pit_loss_time + config.pit_stop_time;
            }

            // Advance this car's running totals.
            state.race_time += total_lap_time;
            state.fuel_remaining = (state.fuel_remaining - config.fuel_per_lap).max(0.0);
            state.tire_age += 1;
            state.lap_times.push(total_lap_time);
            state.in_pit = pitting;

            // Apply the pit stop at the end of the lap: fit the new compound and
            // reset tire age so the next lap starts on fresh rubber.
            if pitting {
                if let Some(stop) = entry.strategy.stops.get(state.next_stop_index) {
                    state.compound = stop.compound;
                    state.tire_age = 0;
                    state.pit_stops += 1;
                    state.next_stop_index += 1;
                }
            }
        }
    }

    // Determine finishing order: running cars by ascending race time, retired
    // cars last.
    let mut indices: Vec<usize> = (0..n_cars).collect();
    indices.sort_by(|&a, &b| match (states[a].retired, states[b].retired) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => states[a]
            .race_time
            .partial_cmp(&states[b].race_time)
            .unwrap_or(Ordering::Equal),
    });

    // Assign 1-indexed finishing positions.
    for (pos, &car_idx) in indices.iter().enumerate() {
        states[car_idx].position = pos + 1;
    }

    let winner_time = indices.first().map(|&i| states[i].race_time).unwrap_or(0.0);

    RaceResult {
        finishing_order: indices,
        car_states: states,
        winner_time,
        laps_completed: config.n_laps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_f1_grid, PlannedStop, RaceStrategy};

    /// A race config for short test races on a generic track.
    fn test_config(n_laps: usize) -> RaceConfig {
        RaceConfig::for_track(5000.0, n_laps)
    }

    /// A simple entry with the given base lap time and no planned stops.
    fn entry_no_stop(name: &str, base: f64, start: TireCompound) -> RaceEntry {
        RaceEntry {
            name: name.to_string(),
            base_lap_time: base,
            lap_time_variance: 0.0,
            tire_deg_factor: 1.0,
            pit_crew_variance: 0.0,
            strategy: RaceStrategy {
                stops: vec![],
                start_compound: start,
            },
            driver_skill: 0.9,
        }
    }

    #[test]
    fn test_simulate_basic() {
        // Three cars, identical setup except base pace, no pit stops.
        let entries = vec![
            entry_no_stop("A", 90.0, TireCompound::Medium),
            entry_no_stop("B", 91.0, TireCompound::Medium),
            entry_no_stop("C", 92.0, TireCompound::Medium),
        ];
        let result = simulate_race(&test_config(10), &entries);
        assert_eq!(result.finishing_order.len(), 3);
        assert_eq!(
            result.finishing_order[0], 0,
            "the fastest car (index 0) should win"
        );
        assert_eq!(result.position_of(0), Some(1));
    }

    #[test]
    fn test_pit_stop_adds_time() {
        // Same base pace; car B pits on lap 5, car A never pits.
        let car_a = entry_no_stop("A", 90.0, TireCompound::Medium);
        let mut car_b = entry_no_stop("B", 90.0, TireCompound::Medium);
        car_b.strategy.stops = vec![PlannedStop {
            lap: 5,
            compound: TireCompound::Medium,
        }];

        let result = simulate_race(&test_config(10), &[car_a, car_b]);

        assert_eq!(
            result.car_states[1].pit_stops, 1,
            "car B should have pitted"
        );
        assert!(
            result.car_states[1].race_time > result.car_states[0].race_time,
            "the pitting car should be slower over 10 laps: B {:.1}s vs A {:.1}s",
            result.car_states[1].race_time,
            result.car_states[0].race_time
        );
        assert_eq!(result.finishing_order[0], 0, "non-pitting car A wins");
    }

    #[test]
    fn test_tire_degradation_effect() {
        // Isolate tire wear by removing the fuel effect (no fuel burn).
        let mut cfg = test_config(20);
        cfg.fuel_per_lap = 0.0;
        let entries = vec![entry_no_stop("A", 90.0, TireCompound::Soft)];

        let result = simulate_race(&cfg, &entries);
        let laps = &result.car_states[0].lap_times;
        assert_eq!(laps.len(), 20);
        assert!(
            laps[19] > laps[0],
            "soft tires should degrade: last {:.3}s vs first {:.3}s",
            laps[19],
            laps[0]
        );
    }

    #[test]
    fn test_fuel_effect() {
        // Isolate fuel by removing tire degradation (tire_deg_factor = 0).
        let cfg = test_config(20);
        let mut entry = entry_no_stop("A", 90.0, TireCompound::Medium);
        entry.tire_deg_factor = 0.0;

        let result = simulate_race(&cfg, &[entry]);
        let laps = &result.car_states[0].lap_times;
        assert!(
            laps[0] > laps[19],
            "fuel burn-off should speed the car up: first {:.3}s vs last {:.3}s",
            laps[0],
            laps[19]
        );
    }

    #[test]
    fn test_strategy_advantage() {
        // Car A makes a 1-stop (medium -> hard at lap 10); car B runs the whole
        // race on medium with no stop. Over only 20 laps the strategies should
        // produce a meaningfully different race time.
        let mut car_a = entry_no_stop("A", 90.0, TireCompound::Medium);
        car_a.strategy.stops = vec![PlannedStop {
            lap: 10,
            compound: TireCompound::Hard,
        }];
        let car_b = entry_no_stop("B", 90.0, TireCompound::Medium);

        let result = simulate_race(&test_config(20), &[car_a, car_b]);
        let delta = (result.car_states[0].race_time - result.car_states[1].race_time).abs();
        assert!(
            delta > 5.0,
            "the strategies should differ meaningfully, delta {delta:.1}s"
        );
    }

    #[test]
    fn test_full_grid_completes() {
        let cfg = RaceConfig::silverstone_default();
        let entries = default_f1_grid(90.0);
        let result = simulate_race(&cfg, &entries);

        assert_eq!(result.finishing_order.len(), 20);
        assert!(
            result.car_states.iter().all(|c| !c.retired),
            "no car should retire in a deterministic race"
        );
        assert_eq!(result.laps_completed, 52);
        assert!(
            (3500.0..5000.0).contains(&result.winner_time),
            "winner time {:.0}s out of expected range",
            result.winner_time
        );
    }

    #[test]
    fn test_points_scoring() {
        let cfg = RaceConfig::silverstone_default();
        let entries = default_f1_grid(90.0);
        let result = simulate_race(&cfg, &entries);

        let winner = result.finishing_order[0];
        let second = result.finishing_order[1];
        let eleventh = result.finishing_order[10];

        assert_eq!(result.points_for(winner), 25.0, "winner scores 25");
        assert_eq!(result.points_for(second), 18.0, "second scores 18");
        assert_eq!(result.points_for(eleventh), 0.0, "11th scores 0");
    }
}
