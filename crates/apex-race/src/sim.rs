//! Lap-by-lap race simulation engine.
//!
//! Simulates a full race for N cars, tracking positions, pit stops,
//! tire wear, and fuel consumption. This is the deterministic core;
//! probabilistic events are layered on top in the events module.

use std::cmp::Ordering;

use rand::Rng;
use rand_distr::StandardNormal;

use crate::config::{RaceConfig, RaceEntry, TireCompound};
use crate::events::{generate_lap_events, EventState};

/// Lap time added to every car under a Virtual Safety Car (s). The fixed delta
/// preserves the gaps between cars (unlike a full safety car).
const VSC_TIME_DELTA: f64 = 10.0;
/// Lap time penalty for running dry-compound tires on a wet track (s/lap). Wet
/// compounds are not modeled, so a wet track always incurs this penalty.
const WET_DRY_PENALTY: f64 = 5.0;
/// On a successful pass, the overtaking car moves this far ahead of the car it
/// passed (s), and the passed car takes [`OVERTAKE_DIRTY_AIR_PENALTY`].
const OVERTAKE_PASS_MARGIN: f64 = 0.05;
/// Dirty-air time penalty applied to a car that has just been passed (s).
const OVERTAKE_DIRTY_AIR_PENALTY: f64 = 0.10;
/// Floor on any single lap time (s), guarding against negative lap times from
/// large random variance draws.
const MIN_LAP_TIME: f64 = 1.0;

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
    /// Number of full safety car deployments during the race. Always 0 for the
    /// deterministic [`simulate_race`]; populated by [`simulate_race_stochastic`].
    pub sc_count: usize,
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

/// Build the starting car states for a race (grid order, full fuel, start tires).
fn init_states(config: &RaceConfig, entries: &[RaceEntry]) -> Vec<CarState> {
    entries
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
        .collect()
}

/// Green-flag "clean" lap time for a car: base pace plus the current compound's
/// fresh-tire offset and degradation (scaled by `tire_deg_factor`), minus the
/// fuel-burn benefit. Excludes pit stops, events, and random variance.
fn clean_lap_time(config: &RaceConfig, entry: &RaceEntry, state: &CarState) -> f64 {
    let tire_deg =
        state.compound.degradation_rate() * state.tire_age as f64 * entry.tire_deg_factor;
    let fuel_effect = (config.start_fuel_kg - state.fuel_remaining) * config.fuel_time_factor;
    entry.base_lap_time + state.compound.pace_offset() + tire_deg - fuel_effect
}

/// Advance a car's running totals by one completed lap, then apply its pit stop
/// if `pitting` (fit the new compound and reset tire age).
fn advance_car(
    config: &RaceConfig,
    entry: &RaceEntry,
    state: &mut CarState,
    lap_time: f64,
    pitting: bool,
) {
    state.race_time += lap_time;
    state.fuel_remaining = (state.fuel_remaining - config.fuel_per_lap).max(0.0);
    state.tire_age += 1;
    state.lap_times.push(lap_time);
    state.in_pit = pitting;

    if pitting {
        if let Some(stop) = entry.strategy.stops.get(state.next_stop_index) {
            state.compound = stop.compound;
            state.tire_age = 0;
            state.pit_stops += 1;
            state.next_stop_index += 1;
        }
    }
}

/// Sort cars into finishing order and assign positions.
///
/// Running cars come first, ordered by ascending race time. Retired cars come
/// last, ordered by laps completed (more laps = classified ahead).
fn finalize_result(mut states: Vec<CarState>, config: &RaceConfig) -> RaceResult {
    let n_cars = states.len();
    let mut indices: Vec<usize> = (0..n_cars).collect();
    indices.sort_by(|&a, &b| match (states[a].retired, states[b].retired) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => states[a]
            .race_time
            .partial_cmp(&states[b].race_time)
            .unwrap_or(Ordering::Equal),
        // Both retired: the car that completed more laps is classified ahead.
        (true, true) => states[b].lap_times.len().cmp(&states[a].lap_times.len()),
    });

    for (pos, &car_idx) in indices.iter().enumerate() {
        states[car_idx].position = pos + 1;
    }

    // The winner is the leading still-running car (or the leader if all retired).
    let winner_time = indices
        .iter()
        .copied()
        .find(|&i| !states[i].retired)
        .or_else(|| indices.first().copied())
        .map(|i| states[i].race_time)
        .unwrap_or(0.0);

    RaceResult {
        finishing_order: indices,
        car_states: states,
        winner_time,
        laps_completed: config.n_laps,
        sc_count: 0,
    }
}

/// Simulate a complete race (deterministic, no random events).
///
/// Runs lap-by-lap, computing each lap time from base performance, the current
/// tire compound's fresh pace and degradation (scaled by the entry's
/// `tire_deg_factor`), the fuel load, and any pit stop on that lap. Returns the
/// complete race result with cars sorted into finishing order.
pub fn simulate_race(config: &RaceConfig, entries: &[RaceEntry]) -> RaceResult {
    let mut states = init_states(config, entries);

    for lap in 1..=config.n_laps {
        for (state, entry) in states.iter_mut().zip(entries.iter()) {
            if state.retired {
                continue;
            }
            let pitting = entry
                .strategy
                .stops
                .get(state.next_stop_index)
                .is_some_and(|stop| stop.lap == lap);

            let mut lap_time = clean_lap_time(config, entry, state);
            if pitting {
                lap_time += config.pit_loss_time + config.pit_stop_time;
            }
            advance_car(config, entry, state, lap_time, pitting);
        }
    }

    finalize_result(states, config)
}

/// Attempt overtakes between adjacent cars in the running order.
///
/// Cars are ordered by cumulative race time; when a car is within
/// `overtake_gap_threshold` of the one ahead, it attempts a pass with
/// probability `overtake_base_prob * driver_skill`. A successful pass moves the
/// overtaking car just ahead and applies a small dirty-air penalty to the car it
/// passed, so track position can change beyond pure lap-time differences. Only
/// called under green-flag conditions.
fn process_overtaking<R: Rng>(
    rng: &mut R,
    states: &mut [CarState],
    entries: &[RaceEntry],
    config: &RaceConfig,
) {
    let mut order: Vec<usize> = (0..states.len()).filter(|&i| !states[i].retired).collect();
    order.sort_by(|&a, &b| {
        states[a]
            .race_time
            .partial_cmp(&states[b].race_time)
            .unwrap_or(Ordering::Equal)
    });

    for pair in order.windows(2) {
        let ahead = pair[0];
        let behind = pair[1];
        let gap = states[behind].race_time - states[ahead].race_time;
        if gap > 0.0 && gap <= config.overtake_gap_threshold {
            let prob = config.overtake_base_prob * entries[behind].driver_skill;
            if rng.random::<f64>() < prob {
                // Complete the pass: the overtaker slots just ahead; the passed
                // car loses a little time to dirty air.
                states[behind].race_time = states[ahead].race_time - OVERTAKE_PASS_MARGIN;
                states[ahead].race_time += OVERTAKE_DIRTY_AIR_PENALTY;
            }
        }
    }
}

/// Simulate a race with probabilistic events.
///
/// Uses the provided RNG to generate per-lap events (safety car, VSC,
/// mechanical DNFs, weather transitions, driver errors), adds lap-time variance,
/// and processes overtaking. The simulation is fully deterministic for a given
/// seeded RNG. This is the function used by the Monte Carlo framework.
///
/// Event handling per lap:
/// - **Safety car**: all running cars lap at `safety_car_pace`, so gaps stop
///   growing (the field is effectively neutralized) for `safety_car_laps`.
/// - **VSC**: every car adds a fixed [`VSC_TIME_DELTA`] for `vsc_laps`, holding
///   gaps roughly constant.
/// - **DNF**: affected cars retire and take no further laps.
/// - **Rain**: while wet, every car takes [`WET_DRY_PENALTY`] (wet compounds are
///   not modeled).
/// - **Driver error / variance**: green-flag laps add a normal lap-time jitter
///   plus any error penalty.
/// - **Overtaking**: processed at green-flag lap ends (see [`process_overtaking`]).
pub fn simulate_race_stochastic<R: Rng>(
    rng: &mut R,
    config: &RaceConfig,
    entries: &[RaceEntry],
) -> RaceResult {
    let n_cars = entries.len();
    let mut states = init_states(config, entries);
    let mut event_state = EventState::new();

    for lap in 1..=config.n_laps {
        // Roll this lap's events against the carried-over event state.
        let retired_flags: Vec<bool> = states.iter().map(|s| s.retired).collect();
        let events = generate_lap_events(rng, config, &event_state, n_cars, &retired_flags);

        // Update race-level state from the new events.
        if events.safety_car_start {
            event_state.safety_car_remaining = config.safety_car_laps;
            event_state.total_sc += 1;
        }
        if events.vsc_start {
            event_state.vsc_remaining = config.vsc_laps;
            event_state.total_vsc += 1;
        }
        if events.rain_start {
            event_state.is_wet = true;
        }
        if events.rain_stop {
            event_state.is_wet = false;
        }
        for &car in &events.dnf_cars {
            states[car].retired = true;
        }

        let under_sc = event_state.safety_car_remaining > 0;
        let under_vsc = !under_sc && event_state.vsc_remaining > 0;

        for car_idx in 0..n_cars {
            if states[car_idx].retired {
                continue;
            }
            let entry = &entries[car_idx];
            let pitting = entry
                .strategy
                .stops
                .get(states[car_idx].next_stop_index)
                .is_some_and(|stop| stop.lap == lap);

            // Base lap time for the conditions this lap.
            let mut lap_time = if under_sc {
                // Field neutralized: everyone laps at the safety-car pace.
                config.safety_car_pace
            } else {
                let mut t = clean_lap_time(config, entry, &states[car_idx]);
                if event_state.is_wet {
                    t += WET_DRY_PENALTY;
                }
                if under_vsc {
                    t += VSC_TIME_DELTA;
                } else {
                    // Green flag: lap-to-lap jitter plus any driver error.
                    let jitter: f64 = rng.sample(StandardNormal);
                    t += jitter * entry.lap_time_variance;
                    if let Some(pos) = events.error_cars.iter().position(|&c| c == car_idx) {
                        t += events.error_penalties[pos];
                    }
                }
                t
            };

            if pitting {
                let pit_jitter: f64 = rng.sample(StandardNormal);
                lap_time += config.pit_loss_time
                    + config.pit_stop_time
                    + pit_jitter * entry.pit_crew_variance;
            }
            lap_time = lap_time.max(MIN_LAP_TIME);

            advance_car(config, entry, &mut states[car_idx], lap_time, pitting);
        }

        // Wind down active cautions by one lap.
        if event_state.safety_car_remaining > 0 {
            event_state.safety_car_remaining -= 1;
        }
        if event_state.vsc_remaining > 0 {
            event_state.vsc_remaining -= 1;
        }

        // Overtaking only happens under green-flag racing.
        if !under_sc && !under_vsc {
            process_overtaking(rng, &mut states, entries, config);
        }
    }

    let mut result = finalize_result(states, config);
    result.sc_count = event_state.total_sc;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_f1_grid, PlannedStop, RaceStrategy};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// A race config for short test races on a generic track.
    fn test_config(n_laps: usize) -> RaceConfig {
        RaceConfig::for_track(5000.0, n_laps)
    }

    /// A config with every random-event probability zeroed.
    fn quiet_config(n_laps: usize) -> RaceConfig {
        let mut cfg = test_config(n_laps);
        cfg.safety_car_prob = 0.0;
        cfg.vsc_prob = 0.0;
        cfg.dnf_prob = 0.0;
        cfg.rain_prob = 0.0;
        cfg.driver_error_prob = 0.0;
        cfg
    }

    /// Spread (s) between the leading and trailing still-running car.
    fn field_spread(result: &RaceResult) -> f64 {
        let times: Vec<f64> = result
            .car_states
            .iter()
            .filter(|c| !c.retired)
            .map(|c| c.race_time)
            .collect();
        let max = times.iter().cloned().fold(f64::MIN, f64::max);
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        max - min
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

    #[test]
    fn test_stochastic_race_completes() {
        let cfg = RaceConfig::silverstone_default();
        let entries = default_f1_grid(90.0);
        let mut rng = StdRng::seed_from_u64(12345);
        let result = simulate_race_stochastic(&mut rng, &cfg, &entries);
        assert_eq!(result.finishing_order.len(), 20);
        assert_eq!(result.laps_completed, 52);
        assert!(result.winner_time > 0.0);
    }

    #[test]
    fn test_sc_bunches_field() {
        let entries = default_f1_grid(90.0);

        // Green race (no events) vs a race with a guaranteed safety car each lap.
        let green = quiet_config(40);
        let mut sc = quiet_config(40);
        sc.safety_car_prob = 1.0;

        let mut rng_green = StdRng::seed_from_u64(99);
        let green_result = simulate_race_stochastic(&mut rng_green, &green, &entries);
        let mut rng_sc = StdRng::seed_from_u64(99);
        let sc_result = simulate_race_stochastic(&mut rng_sc, &sc, &entries);

        let green_spread = field_spread(&green_result);
        let sc_spread = field_spread(&sc_result);
        assert!(
            sc_spread < green_spread,
            "safety car should compress the field: SC spread {sc_spread:.1}s vs green {green_spread:.1}s"
        );
    }

    #[test]
    fn test_dnf_reduces_finishers() {
        let entries = default_f1_grid(90.0);
        // Moderate DNF rate over a short race so some cars finish and some retire.
        let mut cfg = quiet_config(10);
        cfg.dnf_prob = 0.1;
        let mut rng = StdRng::seed_from_u64(2024);
        let result = simulate_race_stochastic(&mut rng, &cfg, &entries);

        let retired = result.car_states.iter().filter(|c| c.retired).count();
        assert!(retired > 0, "high DNF rate should retire some cars");

        // Retired cars must be classified behind every running car.
        let last_running = result
            .finishing_order
            .iter()
            .position(|&i| result.car_states[i].retired);
        if let Some(first_retired_pos) = last_running {
            for &i in &result.finishing_order[first_retired_pos..] {
                assert!(
                    result.car_states[i].retired,
                    "retired cars must occupy the back of the order"
                );
            }
        }
    }

    #[test]
    fn test_driver_error_adds_time() {
        // Two cars, no lap variance, on a flat config: errors are the only
        // stochastic effect, so they can only add time.
        let entries = vec![
            entry_no_stop("A", 90.0, TireCompound::Medium),
            entry_no_stop("B", 90.0, TireCompound::Medium),
        ];

        let base = quiet_config(15);
        let mut with_errors = base.clone();
        with_errors.driver_error_prob = 1.0;

        let mut rng0 = StdRng::seed_from_u64(5);
        let clean = simulate_race_stochastic(&mut rng0, &base, &entries);
        let mut rng1 = StdRng::seed_from_u64(5);
        let errored = simulate_race_stochastic(&mut rng1, &with_errors, &entries);

        assert!(
            errored.car_states[0].race_time > clean.car_states[0].race_time,
            "constant driver errors should add time: {:.1}s vs {:.1}s",
            errored.car_states[0].race_time,
            clean.car_states[0].race_time
        );
    }

    #[test]
    fn test_variance_produces_different_results() {
        let cfg = RaceConfig::silverstone_default();
        let entries = default_f1_grid(90.0);

        let mut rng_a = StdRng::seed_from_u64(1);
        let a = simulate_race_stochastic(&mut rng_a, &cfg, &entries);
        let mut rng_b = StdRng::seed_from_u64(2);
        let b = simulate_race_stochastic(&mut rng_b, &cfg, &entries);

        assert!(
            a.finishing_order != b.finishing_order || (a.winner_time - b.winner_time).abs() > 1e-9,
            "different seeds should produce different races"
        );
    }

    #[test]
    fn test_stochastic_reproducible() {
        // Same seed must yield identical results (reproducibility guarantee).
        let cfg = RaceConfig::silverstone_default();
        let entries = default_f1_grid(90.0);
        let mut rng_a = StdRng::seed_from_u64(777);
        let a = simulate_race_stochastic(&mut rng_a, &cfg, &entries);
        let mut rng_b = StdRng::seed_from_u64(777);
        let b = simulate_race_stochastic(&mut rng_b, &cfg, &entries);
        assert_eq!(a.finishing_order, b.finishing_order);
        assert!((a.winner_time - b.winner_time).abs() < 1e-12);
    }

    #[test]
    fn test_deterministic_still_works() {
        // The original deterministic simulator is unchanged: identical runs
        // produce identical results, and the fastest car wins.
        let entries = default_f1_grid(90.0);
        let cfg = RaceConfig::silverstone_default();
        let r1 = simulate_race(&cfg, &entries);
        let r2 = simulate_race(&cfg, &entries);
        assert_eq!(r1.finishing_order, r2.finishing_order);
        assert!((r1.winner_time - r2.winner_time).abs() < 1e-12);
        assert_eq!(r1.finishing_order[0], 0, "the fastest car still wins");
    }
}
