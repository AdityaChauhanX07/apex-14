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
