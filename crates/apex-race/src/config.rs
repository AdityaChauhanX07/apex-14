//! Race configuration and entry definitions.

use serde::{Deserialize, Serialize};

// Reuse the tire compound model from apex-physics rather than duplicating it.
// Its `degradation_rate()` and `pace_offset()` methods supply the per-compound
// degradation and fresh-tire pace used by the race simulation.
pub use apex_physics::TireCompound;

/// Configuration for a race.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceConfig {
    /// Number of race laps.
    pub n_laps: usize,
    /// Time lost entering and exiting the pit lane (s).
    pub pit_loss_time: f64,
    /// Time stationary in the pit box for a tire change (s).
    pub pit_stop_time: f64,
    /// Track length (m), derived from the track.
    pub track_length: f64,
    /// Fuel load at race start (kg).
    pub start_fuel_kg: f64,
    /// Fuel consumption per lap (kg/lap).
    pub fuel_per_lap: f64,
    /// Lap time benefit per kg of fuel burned (s/kg). Lighter car = faster.
    pub fuel_time_factor: f64,

    /// Safety car probability per lap (0.0-1.0). Default: 0.02 (~1 per race).
    pub safety_car_prob: f64,
    /// Virtual safety car probability per lap. Default: 0.01.
    pub vsc_prob: f64,
    /// Mechanical DNF probability per car per lap. Default: 0.001.
    pub dnf_prob: f64,
    /// Probability of a weather transition per lap (dry->wet or wet->dry).
    /// Default: 0.0 (dry race).
    pub rain_prob: f64,
    /// Driver error probability per lap (causes ~1-3s time loss). Default: 0.02.
    pub driver_error_prob: f64,
    /// Safety car duration in laps. Default: 3.
    pub safety_car_laps: usize,
    /// VSC duration in laps. Default: 2.
    pub vsc_laps: usize,
    /// Safety car pace (seconds per lap, typically slow). Default: 120.0.
    pub safety_car_pace: f64,
    /// Gap threshold for an overtaking attempt (s). Default: 1.0.
    pub overtake_gap_threshold: f64,
    /// Base overtaking probability when within the gap threshold. Default: 0.3.
    pub overtake_base_prob: f64,
}

impl RaceConfig {
    /// Default config for a ~5.9km track (Silverstone-like), 52 laps.
    pub fn silverstone_default() -> Self {
        Self {
            n_laps: 52,
            pit_loss_time: 20.0,
            pit_stop_time: 2.5,
            track_length: 5891.0,
            start_fuel_kg: 110.0,
            fuel_per_lap: 2.1,
            fuel_time_factor: 0.035,
            safety_car_prob: 0.02,
            vsc_prob: 0.01,
            dnf_prob: 0.001,
            rain_prob: 0.0,
            driver_error_prob: 0.02,
            safety_car_laps: 3,
            vsc_laps: 2,
            safety_car_pace: 120.0,
            overtake_gap_threshold: 1.0,
            overtake_base_prob: 0.3,
        }
    }

    /// Create a config for a given track with default race parameters.
    pub fn for_track(track_length: f64, n_laps: usize) -> Self {
        Self {
            n_laps,
            track_length,
            ..Self::silverstone_default()
        }
    }
}

impl apex_math::ContentHash for RaceConfig {
    /// Encode all 17 result-determining fields in declaration order. The
    /// destructure forces any new field to be handled here before it compiles.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let RaceConfig {
            n_laps,
            pit_loss_time,
            pit_stop_time,
            track_length,
            start_fuel_kg,
            fuel_per_lap,
            fuel_time_factor,
            safety_car_prob,
            vsc_prob,
            dnf_prob,
            rain_prob,
            driver_error_prob,
            safety_car_laps,
            vsc_laps,
            safety_car_pace,
            overtake_gap_threshold,
            overtake_base_prob,
        } = self;
        w.usize(*n_laps);
        w.f64(*pit_loss_time);
        w.f64(*pit_stop_time);
        w.f64(*track_length);
        w.f64(*start_fuel_kg);
        w.f64(*fuel_per_lap);
        w.f64(*fuel_time_factor);
        w.f64(*safety_car_prob);
        w.f64(*vsc_prob);
        w.f64(*dnf_prob);
        w.f64(*rain_prob);
        w.f64(*driver_error_prob);
        w.usize(*safety_car_laps);
        w.usize(*vsc_laps);
        w.f64(*safety_car_pace);
        w.f64(*overtake_gap_threshold);
        w.f64(*overtake_base_prob);
    }
}

/// Content hash of the race configuration, under domain `"race"`.
pub fn race_config_hash(cfg: &RaceConfig) -> apex_math::Hash {
    apex_math::content_hash("race", cfg)
}

/// A single planned pit stop.
#[derive(Debug, Clone)]
pub struct PlannedStop {
    /// Target lap number to pit.
    pub lap: usize,
    /// Tire compound to fit.
    pub compound: TireCompound,
}

/// Planned pit stop strategy.
#[derive(Debug, Clone)]
pub struct RaceStrategy {
    /// Planned pit stops, in lap order.
    pub stops: Vec<PlannedStop>,
    /// Starting tire compound.
    pub start_compound: TireCompound,
}

/// A car entry in the race.
#[derive(Debug, Clone)]
pub struct RaceEntry {
    /// Driver/team name for display.
    pub name: String,
    /// Base lap time on fresh tires with full fuel (s).
    /// Typically from QSS lap simulation.
    pub base_lap_time: f64,
    /// Standard deviation of lap-to-lap time variation (s).
    pub lap_time_variance: f64,
    /// Tire degradation rate multiplier (1.0 = normal, >1 = harder on tires).
    pub tire_deg_factor: f64,
    /// Pit crew time variance std dev (s).
    pub pit_crew_variance: f64,
    /// Planned pit strategy.
    pub strategy: RaceStrategy,
    /// Driver skill factor (0.0-1.0). Affects overtaking probability.
    pub driver_skill: f64,
}

/// Create a default 20-car F1-like grid.
///
/// Returns entries with a realistic performance spread: the fastest car (index
/// 0) sets the reference pace and the rest are offset behind it, spanning a
/// top-team to backmarker gap of roughly 1.7s per lap. Strategies alternate
/// between a 1-stop (medium→hard) and a 2-stop (soft→medium→hard).
pub fn default_f1_grid(base_lap_time: f64) -> Vec<RaceEntry> {
    let names = [
        "Verstappen",
        "Perez", // Top team 1
        "Hamilton",
        "Russell", // Top team 2
        "Leclerc",
        "Sainz", // Top team 3
        "Norris",
        "Piastri", // Strong midfield
        "Alonso",
        "Stroll", // Midfield
        "Gasly",
        "Ocon", // Midfield
        "Tsunoda",
        "Lawson", // Lower midfield
        "Bottas",
        "Zhou", // Lower midfield
        "Magnussen",
        "Hulkenberg", // Backmarker
        "Albon",
        "Sargeant", // Backmarker
    ];
    // Performance deltas from the fastest car (seconds per lap).
    let deltas = [
        0.0, 0.3, // team 1
        0.2, 0.4, // team 2
        0.3, 0.5, // team 3
        0.5, 0.6, // strong mid
        0.8, 1.0, // mid
        0.9, 1.0, // mid
        1.1, 1.2, // lower mid
        1.2, 1.3, // lower mid
        1.5, 1.6, // back
        1.5, 1.7, // back
    ];
    let skills = [
        0.98, 0.85, 0.95, 0.90, 0.92, 0.88, 0.93, 0.87, 0.88, 0.75, 0.82, 0.80, 0.83, 0.78, 0.80,
        0.72, 0.78, 0.76, 0.82, 0.68,
    ];

    let mut entries = Vec::with_capacity(20);
    for i in 0..20 {
        // Alternate strategies: 1-stop medium->hard vs 2-stop soft->medium->hard.
        let strategy = if i % 2 == 0 {
            RaceStrategy {
                start_compound: TireCompound::Medium,
                stops: vec![PlannedStop {
                    lap: 25,
                    compound: TireCompound::Hard,
                }],
            }
        } else {
            RaceStrategy {
                start_compound: TireCompound::Soft,
                stops: vec![
                    PlannedStop {
                        lap: 15,
                        compound: TireCompound::Medium,
                    },
                    PlannedStop {
                        lap: 35,
                        compound: TireCompound::Hard,
                    },
                ],
            }
        };

        entries.push(RaceEntry {
            name: names[i].to_string(),
            base_lap_time: base_lap_time + deltas[i],
            lap_time_variance: 0.15,
            tire_deg_factor: 1.0,
            pit_crew_variance: 0.3,
            strategy,
            driver_skill: skills[i],
        });
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_grid_size() {
        let grid = default_f1_grid(90.0);
        assert_eq!(grid.len(), 20, "default grid should have 20 cars");
    }

    #[test]
    fn test_default_grid_ordered() {
        // The fastest car heads the grid (index 0 has the minimum base lap time),
        // and the field spans a realistic spread behind it. The intermediate
        // order is intentionally non-monotonic (teammates and overlapping
        // midfield), so we assert the pole pace and the overall span, not a
        // strict sort.
        let grid = default_f1_grid(90.0);
        let fastest = grid[0].base_lap_time;
        for entry in &grid {
            assert!(
                entry.base_lap_time >= fastest - 1e-12,
                "{} ({}) should not be faster than the pole car",
                entry.name,
                entry.base_lap_time
            );
        }
        let slowest = grid
            .iter()
            .map(|e| e.base_lap_time)
            .fold(f64::MIN, f64::max);
        assert!(
            (slowest - fastest) > 1.0,
            "field spread {:.2}s should be over 1s",
            slowest - fastest
        );
    }

    #[test]
    fn test_silverstone_config() {
        let cfg = RaceConfig::silverstone_default();
        assert_eq!(cfg.n_laps, 52);
        assert!(cfg.pit_loss_time > 0.0);
        assert!(cfg.pit_stop_time > 0.0);
    }

    #[test]
    fn test_tire_compound_ordering() {
        // Soft is the fastest on fresh tires (most negative pace offset) and the
        // hard the slowest; soft degrades fastest, hard the slowest.
        assert!(TireCompound::Soft.pace_offset() < TireCompound::Medium.pace_offset());
        assert!(TireCompound::Medium.pace_offset() < TireCompound::Hard.pace_offset());
        assert!(TireCompound::Soft.degradation_rate() > TireCompound::Medium.degradation_rate());
        assert!(TireCompound::Medium.degradation_rate() > TireCompound::Hard.degradation_rate());
    }
}
