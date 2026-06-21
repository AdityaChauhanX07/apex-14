//! Random event generation for race simulation.
//!
//! Generates safety cars, VSC periods, DNFs, weather changes,
//! and driver errors using per-lap probability rolls.

use rand::Rng;

use crate::config::RaceConfig;

/// Events that occurred during a single lap.
#[derive(Debug, Clone, Default)]
pub struct LapEvents {
    /// Safety car deployed this lap.
    pub safety_car_start: bool,
    /// Virtual safety car deployed this lap.
    pub vsc_start: bool,
    /// Indices of cars that suffered a mechanical DNF this lap.
    pub dnf_cars: Vec<usize>,
    /// Whether it started raining this lap (dry->wet transition).
    pub rain_start: bool,
    /// Whether rain stopped this lap (wet->dry transition).
    pub rain_stop: bool,
    /// Indices of cars that made a driver error this lap.
    pub error_cars: Vec<usize>,
    /// Time penalty for each driver error (s), aligned with `error_cars`.
    pub error_penalties: Vec<f64>,
}

/// Ongoing race-level event state, carried across laps.
#[derive(Debug, Clone)]
pub struct EventState {
    /// Remaining laps of active full safety car (0 = none).
    pub safety_car_remaining: usize,
    /// Remaining laps of active VSC (0 = none).
    pub vsc_remaining: usize,
    /// Whether the track is currently wet.
    pub is_wet: bool,
    /// Total safety car deployments this race.
    pub total_sc: usize,
    /// Total VSC deployments this race.
    pub total_vsc: usize,
}

impl EventState {
    /// Create initial event state (no active events, dry track).
    pub fn new() -> Self {
        Self {
            safety_car_remaining: 0,
            vsc_remaining: 0,
            is_wet: false,
            total_sc: 0,
            total_vsc: 0,
        }
    }

    /// Whether any safety car (full or virtual) is currently active.
    pub fn under_caution(&self) -> bool {
        self.safety_car_remaining > 0 || self.vsc_remaining > 0
    }
}

impl Default for EventState {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate random events for a single lap.
///
/// `retired` flags which cars are already out (they cannot DNF or err again).
/// Safety car and VSC only start when no caution is currently active. The error
/// penalty for each affected car is drawn uniformly from `[1, 3)` seconds.
pub fn generate_lap_events<R: Rng>(
    rng: &mut R,
    config: &RaceConfig,
    event_state: &EventState,
    n_cars: usize,
    retired: &[bool],
) -> LapEvents {
    let mut events = LapEvents::default();

    // Safety car (only if no SC/VSC already active).
    if !event_state.under_caution() && rng.random::<f64>() < config.safety_car_prob {
        events.safety_car_start = true;
    }

    // VSC (only if no caution active and no SC just started).
    if !event_state.under_caution()
        && !events.safety_car_start
        && rng.random::<f64>() < config.vsc_prob
    {
        events.vsc_start = true;
    }

    // Mechanical DNF for each still-running car.
    for (car_idx, &is_retired) in retired.iter().enumerate().take(n_cars) {
        if !is_retired && rng.random::<f64>() < config.dnf_prob {
            events.dnf_cars.push(car_idx);
        }
    }

    // Weather transition (toggles wet/dry).
    if config.rain_prob > 0.0 && rng.random::<f64>() < config.rain_prob {
        if event_state.is_wet {
            events.rain_stop = true;
        } else {
            events.rain_start = true;
        }
    }

    // Driver errors for each still-running car.
    for (car_idx, &is_retired) in retired.iter().enumerate().take(n_cars) {
        if !is_retired && rng.random::<f64>() < config.driver_error_prob {
            events.error_cars.push(car_idx);
            events.error_penalties.push(rng.random_range(1.0..3.0));
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// A config with all event probabilities zeroed, for controlled tests.
    fn quiet_config() -> RaceConfig {
        let mut cfg = RaceConfig::silverstone_default();
        cfg.safety_car_prob = 0.0;
        cfg.vsc_prob = 0.0;
        cfg.dnf_prob = 0.0;
        cfg.rain_prob = 0.0;
        cfg.driver_error_prob = 0.0;
        cfg
    }

    #[test]
    fn test_generate_no_events() {
        let cfg = quiet_config();
        let state = EventState::new();
        let mut rng = StdRng::seed_from_u64(1);
        let retired = vec![false; 5];
        let events = generate_lap_events(&mut rng, &cfg, &state, 5, &retired);
        assert!(!events.safety_car_start);
        assert!(!events.vsc_start);
        assert!(!events.rain_start && !events.rain_stop);
        assert!(events.dnf_cars.is_empty());
        assert!(events.error_cars.is_empty());
    }

    #[test]
    fn test_generate_sc_probability() {
        let state = EventState::new();
        let retired = vec![false; 3];

        let mut always = quiet_config();
        always.safety_car_prob = 1.0;
        let mut rng = StdRng::seed_from_u64(7);
        let ev = generate_lap_events(&mut rng, &always, &state, 3, &retired);
        assert!(ev.safety_car_start, "SC should always fire at prob 1.0");

        let never = quiet_config(); // safety_car_prob = 0.0
        let mut rng = StdRng::seed_from_u64(7);
        let ev = generate_lap_events(&mut rng, &never, &state, 3, &retired);
        assert!(!ev.safety_car_start, "SC should never fire at prob 0.0");
    }

    #[test]
    fn test_dnf_generates_for_active_cars() {
        let mut cfg = quiet_config();
        cfg.dnf_prob = 1.0;
        let state = EventState::new();
        let mut rng = StdRng::seed_from_u64(3);
        // 5 cars, two already retired (indices 2 and 3).
        let retired = vec![false, false, true, true, false];
        let ev = generate_lap_events(&mut rng, &cfg, &state, 5, &retired);
        assert_eq!(ev.dnf_cars, vec![0, 1, 4], "only active cars can DNF");
    }

    #[test]
    fn test_event_state_under_caution() {
        let mut s = EventState::new();
        assert!(!s.under_caution());
        s.safety_car_remaining = 2;
        assert!(s.under_caution());
        s.safety_car_remaining = 0;
        s.vsc_remaining = 1;
        assert!(s.under_caution());
    }

    #[test]
    fn test_no_sc_during_existing_sc() {
        let mut cfg = quiet_config();
        cfg.safety_car_prob = 1.0;
        cfg.vsc_prob = 1.0;
        let mut state = EventState::new();
        state.safety_car_remaining = 2; // SC already active
        let mut rng = StdRng::seed_from_u64(11);
        let retired = vec![false; 3];
        let ev = generate_lap_events(&mut rng, &cfg, &state, 3, &retired);
        assert!(!ev.safety_car_start, "no new SC while one is active");
        assert!(!ev.vsc_start, "no VSC while a caution is active");
    }
}
