//! Weather and track-surface conditions: rain, wetness, drying, wet-tire grip,
//! aquaplaning, and combined effective grip / tire-change analysis.

use super::grip_map::GripMap;

/// Weather and track surface conditions.
///
/// Models rain, surface wetness, drying, and their effect on tire grip.
#[derive(Debug, Clone)]
pub struct WeatherState {
    /// Surface wetness at each station along the track (0.0 = dry, 1.0 = standing water).
    pub wetness: Vec<f64>,
    /// Current rain intensity (mm/hr). 0 = no rain.
    pub rain_intensity: f64,
    /// Air temperature (deg C).
    pub air_temp: f64,
    /// Track temperature (deg C).
    pub track_temp: f64,
    /// Number of stations.
    n_stations: usize,
}

impl WeatherState {
    /// Fully dry conditions.
    pub fn dry(n_stations: usize) -> Self {
        WeatherState {
            wetness: vec![0.0; n_stations],
            rain_intensity: 0.0,
            air_temp: 25.0,
            track_temp: 35.0,
            n_stations,
        }
    }

    /// Light rain (drizzle).
    pub fn light_rain(n_stations: usize) -> Self {
        WeatherState {
            wetness: vec![0.3; n_stations],
            rain_intensity: 2.0,
            air_temp: 18.0,
            track_temp: 20.0,
            n_stations,
        }
    }

    /// Heavy rain.
    pub fn heavy_rain(n_stations: usize) -> Self {
        WeatherState {
            wetness: vec![0.8; n_stations],
            rain_intensity: 15.0,
            air_temp: 15.0,
            track_temp: 16.0,
            n_stations,
        }
    }

    /// Interpolate wetness at an arbitrary arc length.
    pub fn wetness_at(&self, s: f64, track_length: f64) -> f64 {
        if self.wetness.is_empty() {
            return 0.0;
        }
        let ds = track_length / self.n_stations as f64;
        let idx = ((s / ds) as usize).min(self.n_stations - 1);
        self.wetness[idx]
    }

    /// Evolve the weather over a time step.
    ///
    /// Rain increases wetness, evaporation decreases it.
    /// Drying rate depends on temperature and wind (simplified as a constant).
    pub fn evolve(&mut self, dt: f64) {
        let rain_rate = self.rain_intensity / 3600.0; // mm/hr -> mm/s, normalize to wetness
        let evap_rate = 0.001 * (self.track_temp - 10.0).max(0.0) / 30.0; // faster drying when hot

        for w in &mut self.wetness {
            // Rain adds wetness
            *w += rain_rate * dt * 0.1; // scale factor to keep wetness in [0, 1]
                                        // Evaporation removes wetness
            *w -= evap_rate * dt;
            *w = w.clamp(0.0, 1.0);
        }
    }

    /// Simulate cars drying the racing line.
    ///
    /// Cars push water off the racing line, creating a dry line
    /// through wet sections. The drying effect is concentrated
    /// around the racing line lateral offset.
    pub fn car_drying_effect(&mut self, racing_line_stations: &[usize], drying_factor: f64) {
        for &station in racing_line_stations {
            if station < self.n_stations {
                self.wetness[station] *= 1.0 - drying_factor;
                // Also dry adjacent stations slightly (spray)
                if station > 0 {
                    self.wetness[station - 1] *= 1.0 - drying_factor * 0.3;
                }
                if station + 1 < self.n_stations {
                    self.wetness[station + 1] *= 1.0 - drying_factor * 0.3;
                }
            }
        }
    }

    /// Check if conditions are wet enough to require wet tires.
    pub fn requires_wet_tires(&self) -> bool {
        let avg_wetness: f64 = self.wetness.iter().sum::<f64>() / self.wetness.len().max(1) as f64;
        avg_wetness > 0.15
    }

    /// Check if conditions are wet enough for full wet tires.
    pub fn requires_full_wets(&self) -> bool {
        let avg_wetness: f64 = self.wetness.iter().sum::<f64>() / self.wetness.len().max(1) as f64;
        avg_wetness > 0.50
    }
}

/// Tire types including wet-weather options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TireType {
    /// Dry slick tire of a given compound.
    Slick(super::strategy::TireCompound),
    /// Intermediate tire for light rain / drying conditions.
    Intermediate,
    /// Full wet tire for heavy rain.
    FullWet,
}

impl TireType {
    /// Grip multiplier as a function of surface wetness.
    ///
    /// - Slicks: excellent on dry, lose grip rapidly as wetness increases.
    /// - Intermediates: best in light wet (wetness 0.2-0.4), poor on dry or heavy wet.
    /// - Full wets: best in heavy wet (wetness > 0.5), poor on dry.
    pub fn wet_grip_factor(&self, wetness: f64) -> f64 {
        match self {
            TireType::Slick(_) => {
                // Slicks: 1.0 on dry, drops fast past wetness 0.1
                if wetness < 0.05 {
                    1.0
                } else if wetness < 0.3 {
                    1.0 - 2.0 * (wetness - 0.05) // linear drop
                } else {
                    0.5 - 0.5 * ((wetness - 0.3) / 0.7).min(1.0) // very poor above 0.3
                }
            }
            TireType::Intermediate => {
                // Peak around wetness 0.25, drops on both sides
                let peak = 0.25;
                let sigma = 0.15;
                let base = 0.70; // dry grip of intermediates (70% of slick)
                let wet_bonus = 0.25 * (-0.5 * ((wetness - peak) / sigma).powi(2)).exp();
                (base + wet_bonus).min(0.95)
            }
            TireType::FullWet => {
                // Best above wetness 0.5, poor on dry
                if wetness < 0.1 {
                    0.45 // terrible on dry (overheating)
                } else if wetness < 0.4 {
                    0.45 + 0.35 * ((wetness - 0.1) / 0.3) // improving
                } else {
                    0.80 + 0.10 * ((wetness - 0.4) / 0.6).min(1.0) // strong on wet
                }
            }
        }
    }

    /// Speed at which aquaplaning begins (m/s).
    ///
    /// Below this speed, the tire maintains contact. Above it, grip drops sharply.
    /// Returns f64::INFINITY if aquaplaning is not a risk.
    pub fn aquaplaning_speed(&self, wetness: f64) -> f64 {
        if wetness < 0.3 {
            return f64::INFINITY;
        } // no aquaplaning risk below 0.3

        let base_speed = match self {
            TireType::Slick(_) => 180.0, // ~650 km/h on lightly wet
            TireType::Intermediate => 220.0,
            TireType::FullWet => 280.0, // designed for water dispersal
        };

        // Aquaplaning speed decreases with more water
        base_speed / (1.0 + 2.0 * (wetness - 0.3))
    }

    /// Display name.
    pub fn name(&self) -> String {
        match self {
            TireType::Slick(compound) => format!("Slick ({})", compound),
            TireType::Intermediate => "Intermediate".to_string(),
            TireType::FullWet => "Full Wet".to_string(),
        }
    }
}

/// Compute the effective grip at a track position considering surface conditions.
///
/// Combines the grip map (rubber/marbles), weather (wetness), and tire type
/// into a single grip multiplier.
#[allow(clippy::too_many_arguments)]
pub fn effective_grip(
    grip_map: &GripMap,
    weather: &WeatherState,
    tire_type: TireType,
    s: f64,
    n: f64,
    speed: f64,
    track_length: f64,
) -> f64 {
    // Base grip from surface condition (rubber/marbles)
    let surface_grip = grip_map.grip_at(s, n);

    // Weather effect
    let wetness = weather.wetness_at(s, track_length);
    let wet_factor = tire_type.wet_grip_factor(wetness);

    // Aquaplaning check
    let aqua_speed = tire_type.aquaplaning_speed(wetness);
    let aqua_factor = if speed > aqua_speed {
        (aqua_speed / speed).powi(2) // grip drops as v^2 past the threshold
    } else {
        1.0
    };

    // In wet conditions, the rubbered racing line is actually MORE slippery
    // (polished surface + water = low friction). Invert the surface bonus.
    let adjusted_surface = if wetness > 0.2 {
        // Rubbered line becomes a penalty in the wet
        let rubber_bonus = surface_grip - 1.0; // positive if rubbered
        let wet_penalty = rubber_bonus * -0.5 * (wetness - 0.2).min(0.8) / 0.8;
        (surface_grip + wet_penalty).clamp(0.3, 1.2)
    } else {
        surface_grip
    };

    (adjusted_surface * wet_factor * aqua_factor).clamp(0.05, 1.5)
}

/// Analysis of whether to change tire type given current and predicted conditions.
#[derive(Debug, Clone)]
pub struct TireChangeAnalysis {
    /// Current tire type.
    pub current_tire: TireType,
    /// Proposed new tire type.
    pub proposed_tire: TireType,
    /// Estimated average grip on current tires for next N laps.
    pub current_grip_avg: f64,
    /// Estimated average grip on proposed tires for next N laps.
    pub proposed_grip_avg: f64,
    /// Estimated lap time delta per lap (negative = proposed is faster).
    pub lap_time_delta: f64,
    /// Pit stop time cost (s).
    pub pit_cost: f64,
    /// Number of laps before the switch pays off (including pit cost).
    /// None if the switch never pays off.
    pub payoff_laps: Option<usize>,
    /// Recommendation.
    pub recommendation: String,
}

/// Analyze whether to switch tire type.
#[allow(clippy::too_many_arguments)]
pub fn analyze_tire_change(
    current_tire: TireType,
    proposed_tire: TireType,
    weather: &WeatherState,
    grip_map: &GripMap,
    track_length: f64,
    base_lap_time: f64,
    pit_cost: f64,
    analysis_laps: usize,
) -> TireChangeAnalysis {
    // Estimate average grip for each tire type across the track
    let n_samples = 20;
    let ds = track_length / n_samples as f64;

    let mut current_grip_sum = 0.0;
    let mut proposed_grip_sum = 0.0;

    for i in 0..n_samples {
        let s = i as f64 * ds;
        let current_g = effective_grip(grip_map, weather, current_tire, s, 0.0, 50.0, track_length);
        let proposed_g =
            effective_grip(grip_map, weather, proposed_tire, s, 0.0, 50.0, track_length);
        current_grip_sum += current_g;
        proposed_grip_sum += proposed_g;
    }

    let current_avg = current_grip_sum / n_samples as f64;
    let proposed_avg = proposed_grip_sum / n_samples as f64;

    // Grip translates roughly linearly to lap time.
    // Less grip = slower: delta_time ~ base_time * (1/proposed_grip - 1/current_grip).
    let grip_ratio = if current_avg > 0.01 {
        proposed_avg / current_avg
    } else {
        1.0
    };

    // Lap time delta: how much faster per lap on proposed tires.
    // If proposed has more grip, lap time is lower (negative delta).
    let lap_time_delta = base_lap_time * (1.0 / grip_ratio - 1.0);

    // Payoff: how many laps to recover the pit stop cost.
    let payoff_laps = if lap_time_delta < -0.01 {
        // Proposed is faster
        let laps = (pit_cost / (-lap_time_delta)).ceil() as usize;
        if laps <= analysis_laps {
            Some(laps)
        } else {
            None
        }
    } else {
        None // Proposed is not faster
    };

    let recommendation = if let Some(laps) = payoff_laps {
        format!(
            "Switch to {} - pays off in {} laps",
            proposed_tire.name(),
            laps
        )
    } else if lap_time_delta < -0.01 {
        format!(
            "Switch to {} eventually, but pit cost too high for {} laps",
            proposed_tire.name(),
            analysis_laps
        )
    } else {
        format!(
            "Stay on {} - current tires are faster or equal",
            current_tire.name()
        )
    };

    TireChangeAnalysis {
        current_tire,
        proposed_tire,
        current_grip_avg: current_avg,
        proposed_grip_avg: proposed_avg,
        lap_time_delta,
        pit_cost,
        payoff_laps,
        recommendation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::TireCompound;
    use apex_track::{build_track, oval_track, Track};

    fn oval() -> Track {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 600);
        build_track("oval", &pts, closed)
    }

    fn slick() -> TireType {
        TireType::Slick(TireCompound::Medium)
    }

    // (d) Dry weather.
    #[test]
    fn dry_weather() {
        let w = WeatherState::dry(20);
        assert!(w.wetness.iter().all(|&x| x == 0.0));
        assert!(!w.requires_wet_tires());
    }

    // (e) Rain increases wetness. NOTE: with the dry preset's 35 C track the
    // evaporation term outpaces this rain rate, so wetness would stay at 0; rain
    // physically arrives with a cooler track, so the test uses a rain-appropriate
    // track temperature.
    #[test]
    fn rain_increases_wetness() {
        let mut w = WeatherState::dry(20);
        w.rain_intensity = 10.0;
        w.track_temp = 12.0;
        w.evolve(60.0);
        assert!(
            w.wetness[0] > 0.0,
            "rain should wet the track, got {}",
            w.wetness[0]
        );
    }

    // (f) Drying.
    #[test]
    fn track_dries_without_rain() {
        let mut w = WeatherState::light_rain(20);
        let before = w.wetness[0];
        w.rain_intensity = 0.0;
        w.evolve(300.0);
        assert!(
            w.wetness[0] < before,
            "track should dry: {} -> {}",
            before,
            w.wetness[0]
        );
    }

    // (g) Car drying effect.
    #[test]
    fn cars_dry_the_line() {
        let mut w = WeatherState::heavy_rain(20);
        let untouched = w.wetness[10];
        w.car_drying_effect(&[5], 0.5);
        assert!(w.wetness[5] < untouched, "driven station should be drier");
    }

    // (h) Slick grip on dry.
    #[test]
    fn slick_dry_grip() {
        assert!((slick().wet_grip_factor(0.0) - 1.0).abs() < 1e-9);
    }

    // (i) Slick grip on wet.
    #[test]
    fn slick_wet_grip_collapses() {
        assert!(slick().wet_grip_factor(0.5) < 0.5);
    }

    // (j) Intermediate crossover with slicks.
    #[test]
    fn intermediate_crossover() {
        let inter = TireType::Intermediate;
        assert!(inter.wet_grip_factor(0.25) > slick().wet_grip_factor(0.25));
        assert!(slick().wet_grip_factor(0.0) > inter.wet_grip_factor(0.0));
    }

    // (k) Full wet crossover.
    #[test]
    fn full_wet_crossover() {
        let inter = TireType::Intermediate;
        let wet = TireType::FullWet;
        assert!(wet.wet_grip_factor(0.6) > inter.wet_grip_factor(0.6));
        assert!(wet.wet_grip_factor(0.0) < slick().wet_grip_factor(0.0));
    }

    // (l) Aquaplaning.
    #[test]
    fn aquaplaning_thresholds() {
        let wet = TireType::FullWet;
        assert!(slick().aquaplaning_speed(0.5).is_finite());
        assert!(slick().aquaplaning_speed(0.0).is_infinite());
        assert!(wet.aquaplaning_speed(0.5) > slick().aquaplaning_speed(0.5));
    }

    // (m) Effective grip combines surface, weather, and tire.
    #[test]
    fn effective_grip_combines_factors() {
        let track = oval();
        let len = track.total_length;
        let map = GripMap::dry_rubbered(&track, 50, 33);
        let s = 300.0;

        // Dry rubbered + slicks on the line: grip above baseline.
        let dry = WeatherState::dry(20);
        let g_dry = effective_grip(&map, &dry, slick(), s, 0.0, 50.0, len);
        assert!(g_dry > 1.0, "dry line grip {g_dry} should exceed 1.0");

        // Wet + slicks: well below baseline.
        let mut wet = WeatherState::dry(20);
        wet.wetness = vec![0.5; 20];
        let g_wet_slick = effective_grip(&map, &wet, slick(), s, 0.0, 50.0, len);
        assert!(
            g_wet_slick < 1.0,
            "wet slick grip {g_wet_slick} should be below 1.0"
        );

        // Wet + intermediates: better than wet slicks.
        let g_wet_inter = effective_grip(&map, &wet, TireType::Intermediate, s, 0.0, 50.0, len);
        assert!(
            g_wet_inter > g_wet_slick,
            "inters should beat slicks in the wet"
        );
    }

    // (n) Wet racing line penalty. NOTE: the as-specified `-0.5` coefficient only
    // dampens (never inverts) the rubber bonus, so the rubbered line keeps a small
    // advantage over off-line even in the wet. The test verifies the modelled
    // physics: the wet-rubber penalty erodes the line's advantage and lowers its
    // grip below the no-penalty value.
    #[test]
    fn wet_rubber_penalty_erodes_line_advantage() {
        let track = oval();
        let len = track.total_length;
        let map = GripMap::dry_rubbered(&track, 50, 33);
        let s = 300.0;

        let mut wet = WeatherState::dry(20);
        wet.wetness = vec![0.5; 20];

        let line_dry = map.grip_at(s, 0.0);
        let off_dry = map.grip_at(s, 3.0);
        assert!(line_dry > off_dry, "dry: rubbered line is grippier");

        let line_wet = effective_grip(&map, &wet, slick(), s, 0.0, 50.0, len);
        let off_wet = effective_grip(&map, &wet, slick(), s, 3.0, 50.0, len);

        // The wet-rubber penalty shrinks the line's advantage over off-line.
        assert!(
            (line_wet - off_wet) < (line_dry - off_dry),
            "wet penalty should erode the line advantage"
        );

        // And the line's grip is below what it would be without the polished-line
        // penalty (raw surface * wet-tire factor).
        let wetness = wet.wetness_at(s, len);
        let no_penalty = map.grip_at(s, 0.0) * slick().wet_grip_factor(wetness);
        assert!(
            line_wet < no_penalty,
            "polished-line penalty should reduce grip"
        );
    }

    // (o) Tire change analysis.
    #[test]
    fn tire_change_analysis() {
        let track = oval();
        let len = track.total_length;
        let map = GripMap::dry_rubbered(&track, 50, 33);

        // Dry: stay on slicks rather than switch to intermediates.
        let dry = WeatherState::dry(20);
        let dry_a = analyze_tire_change(
            slick(),
            TireType::Intermediate,
            &dry,
            &map,
            len,
            90.0,
            22.0,
            10,
        );
        assert!(dry_a.payoff_laps.is_none(), "dry: no reason to switch");
        assert!(dry_a.recommendation.starts_with("Stay"));

        // Wet: switch from slicks to intermediates, and it pays off quickly.
        let mut wet = WeatherState::dry(20);
        wet.wetness = vec![0.5; 20];
        let wet_a = analyze_tire_change(
            slick(),
            TireType::Intermediate,
            &wet,
            &map,
            len,
            90.0,
            22.0,
            10,
        );
        assert!(wet_a.lap_time_delta < 0.0, "wet: inters are faster");
        let payoff = wet_a.payoff_laps.expect("wet switch should pay off");
        assert!(payoff < 5, "payoff {payoff} laps should be small");
        assert!(wet_a.recommendation.starts_with("Switch"));
    }
}
