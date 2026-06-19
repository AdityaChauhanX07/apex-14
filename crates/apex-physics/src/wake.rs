//! Aerodynamic wake model for car-to-car interaction (slipstream and dirty air)
//! plus a lightweight multi-car track state for gaps, wake exposure, and
//! collision checks.

/// Aerodynamic wake model for car-to-car interaction.
///
/// When a car follows another closely, two effects occur:
/// - Drag reduction (slipstream): the lead car punches a hole in the air,
///   reducing the following car's aerodynamic drag.
/// - Downforce loss (dirty air): the turbulent wake disrupts the following
///   car's aerodynamic surfaces, reducing downforce.
///
/// Both effects depend on the gap distance and lateral offset between the cars.
/// The wake decays with distance and spreads laterally.
#[derive(Debug, Clone, Copy)]
pub struct WakeModel {
    /// Maximum distance behind the lead car where effects are felt (m).
    pub effect_range: f64,
    /// Maximum drag reduction when directly behind at close range (fraction, 0.0-1.0).
    /// 0.30 means drag drops to 70% of its clean-air value.
    pub max_drag_reduction: f64,
    /// Maximum downforce loss when directly behind at close range (fraction, 0.0-1.0).
    /// 0.45 means downforce drops to 55% of its clean-air value.
    pub max_downforce_loss: f64,
    /// Half-width of the wake cone at one car length behind (m).
    pub wake_half_width_base: f64,
    /// Rate at which the wake cone spreads with distance (m/m).
    /// wake_half_width = base + spread_rate * gap
    pub wake_spread_rate: f64,
    /// Length of the lead car (m). Used for close-proximity scaling.
    pub car_length: f64,
}

impl WakeModel {
    /// Representative F1 wake model based on published CFD and wind tunnel data.
    pub fn f1_default() -> Self {
        WakeModel {
            effect_range: 60.0,        // effects felt up to 60m behind
            max_drag_reduction: 0.30,  // 30% drag reduction at very close range
            max_downforce_loss: 0.45,  // 45% downforce loss at very close range
            wake_half_width_base: 1.2, // ~car width at 1 car length
            wake_spread_rate: 0.08,    // wake spreads ~8cm per meter of gap
            car_length: 5.7,           // F1 car length
        }
    }

    /// Compute the aerodynamic wake effect on a following car.
    ///
    /// Returns (drag_factor, downforce_factor) where:
    /// - drag_factor: multiplier on the follower's drag coefficient.
    ///   < 1.0 means reduced drag (slipstream benefit).
    /// - downforce_factor: multiplier on the follower's downforce.
    ///   < 1.0 means reduced downforce (dirty air penalty).
    ///
    /// Arguments:
    /// - gap: longitudinal distance behind the lead car (m).
    ///   Positive means the follower is behind. Negative or zero means no effect.
    /// - lateral_offset: absolute lateral distance between car centerlines (m).
    pub fn wake_effect(&self, gap: f64, lateral_offset: f64) -> (f64, f64) {
        // No effect if not behind or beyond range
        if gap <= 0.0 || gap > self.effect_range {
            return (1.0, 1.0);
        }

        // Longitudinal decay: effect is strongest close behind, decays with distance.
        // Use an inverse-distance model with a saturation at very close range.
        // At 1 car length: full effect. At effect_range: near zero.
        let normalized_gap = gap / self.effect_range;
        let longitudinal_factor = (1.0 - normalized_gap).powi(2); // quadratic decay

        // Lateral decay: Gaussian profile centered on the lead car's path.
        // The wake cone widens with distance.
        let wake_width = self.wake_half_width_base + self.wake_spread_rate * gap;
        let lateral_factor = (-0.5 * (lateral_offset / wake_width).powi(2)).exp();

        // Combined effect
        let total_factor = longitudinal_factor * lateral_factor;

        let drag_factor = 1.0 - self.max_drag_reduction * total_factor;
        let downforce_factor = 1.0 - self.max_downforce_loss * total_factor;

        (
            drag_factor.clamp(0.1, 1.0),
            downforce_factor.clamp(0.1, 1.0),
        )
    }

    /// Compute the combined wake effect from multiple cars ahead.
    ///
    /// Takes a slice of (gap, lateral_offset) pairs for each car ahead.
    /// The combined effect uses a maximum-of-effects model: the follower
    /// sees the strongest effect from whichever car is most influential.
    /// (Superposition would double-count overlapping wakes.)
    pub fn combined_wake_effect(&self, cars_ahead: &[(f64, f64)]) -> (f64, f64) {
        let mut min_drag = 1.0f64;
        let mut min_downforce = 1.0f64;

        for &(gap, lateral) in cars_ahead {
            let (df, dff) = self.wake_effect(gap, lateral);
            min_drag = min_drag.min(df);
            min_downforce = min_downforce.min(dff);
        }

        (min_drag, min_downforce)
    }

    /// Compute the speed advantage from slipstream on a straight.
    ///
    /// Given the follower's clean-air terminal velocity, compute the
    /// new terminal velocity with reduced drag.
    ///
    /// v_terminal = sqrt(F_drive / (0.5 * rho * Cd * A))
    /// With drag_factor applied to Cd:
    /// v_new = v_clean / sqrt(drag_factor)
    pub fn slipstream_speed_gain(&self, clean_terminal_speed: f64, drag_factor: f64) -> f64 {
        if drag_factor <= 0.01 {
            return clean_terminal_speed * 3.0;
        } // safety clamp
        clean_terminal_speed / drag_factor.sqrt() - clean_terminal_speed
    }

    /// Estimate the cornering speed penalty from downforce loss.
    ///
    /// Returns the fraction by which the maximum cornering speed is reduced.
    /// With less downforce, the car has less grip and must slow down in corners.
    ///
    /// Approximate: for a downforce-dominated car, corner speed scales roughly
    /// as sqrt(downforce_factor) (since grip ~ downforce and v^2 ~ grip/curvature).
    pub fn corner_speed_penalty(&self, downforce_factor: f64) -> f64 {
        1.0 - downforce_factor.sqrt()
    }
}

/// State of a single car on track.
#[derive(Debug, Clone)]
pub struct OnTrackCar {
    /// Arc length position on the track centerline (m).
    pub s: f64,
    /// Lateral offset from centerline (m, positive = left).
    pub n: f64,
    /// Speed (m/s).
    pub speed: f64,
    /// Car identifier (0-indexed).
    pub id: usize,
    /// Car parameters.
    pub params: super::CarParams,
}

/// Manages multiple cars on the same track.
#[derive(Debug, Clone)]
pub struct MultiCarState {
    /// All cars on track.
    pub cars: Vec<OnTrackCar>,
    /// Track total length (m), for wrapping.
    pub track_length: f64,
    /// Wake model for aero interaction.
    pub wake: WakeModel,
}

impl MultiCarState {
    pub fn new(track_length: f64, wake: WakeModel) -> Self {
        MultiCarState {
            cars: Vec::new(),
            track_length,
            wake,
        }
    }

    /// Add a car at a given position and speed.
    pub fn add_car(&mut self, s: f64, n: f64, speed: f64, params: super::CarParams) -> usize {
        let id = self.cars.len();
        self.cars.push(OnTrackCar {
            s,
            n,
            speed,
            id,
            params,
        });
        id
    }

    /// Compute the gap between two cars (in meters along the track).
    ///
    /// Returns the distance car_behind must travel to reach car_ahead.
    /// Always positive, accounts for track wrapping on closed circuits.
    pub fn gap_meters(&self, car_ahead: usize, car_behind: usize) -> f64 {
        let s_ahead = self.cars[car_ahead].s;
        let s_behind = self.cars[car_behind].s;
        let mut gap = s_ahead - s_behind;
        if gap < 0.0 {
            gap += self.track_length;
        }
        gap
    }

    /// Compute the time gap between two cars.
    /// gap_time = gap_meters / speed_behind
    pub fn gap_time(&self, car_ahead: usize, car_behind: usize) -> f64 {
        let gap = self.gap_meters(car_ahead, car_behind);
        let speed = self.cars[car_behind].speed.max(1.0);
        gap / speed
    }

    /// Find all cars ahead of and within range of a given car.
    ///
    /// Returns [(car_id, gap_meters, lateral_offset)] for each car within range.
    pub fn cars_ahead_within_range(&self, car_id: usize, range: f64) -> Vec<(usize, f64, f64)> {
        let mut result = Vec::new();
        for other in &self.cars {
            if other.id == car_id {
                continue;
            }
            let gap = self.gap_meters(other.id, car_id);
            if gap > 0.0 && gap <= range {
                let lateral = (self.cars[car_id].n - other.n).abs();
                result.push((other.id, gap, lateral));
            }
        }
        result
    }

    /// Compute the wake effect experienced by a given car.
    ///
    /// Finds all cars ahead within the wake range and computes
    /// the combined aerodynamic effect.
    pub fn wake_effect_on(&self, car_id: usize) -> (f64, f64) {
        let ahead = self.cars_ahead_within_range(car_id, self.wake.effect_range);
        let pairs: Vec<(f64, f64)> = ahead.iter().map(|&(_, gap, lat)| (gap, lat)).collect();
        self.wake.combined_wake_effect(&pairs)
    }

    /// Advance all cars by one time step.
    ///
    /// Each car moves forward by speed * dt, wrapping around the track.
    pub fn advance(&mut self, dt: f64) {
        for car in &mut self.cars {
            car.s += car.speed * dt;
            if car.s >= self.track_length {
                car.s -= self.track_length;
            }
        }
    }

    /// Check if two cars are within collision distance.
    ///
    /// Collision if longitudinal gap < car_length AND lateral overlap.
    pub fn is_collision(
        &self,
        car_a: usize,
        car_b: usize,
        car_length: f64,
        car_width: f64,
    ) -> bool {
        let gap = self
            .gap_meters(car_a, car_b)
            .min(self.gap_meters(car_b, car_a));
        let lateral = (self.cars[car_a].n - self.cars[car_b].n).abs();
        gap < car_length && lateral < car_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CarParams;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn is_one(x: f64) -> bool {
        (x - 1.0).abs() < 1e-12
    }

    // (a) Directly behind at close range: strong slipstream + dirty air.
    #[test]
    fn wake_close_behind_is_strong() {
        let w = WakeModel::f1_default();
        let (drag, downforce) = w.wake_effect(5.0, 0.0);
        assert!(drag < 0.85, "drag factor {drag} should be well below 1.0");
        assert!(
            downforce < 0.75,
            "downforce factor {downforce} should be well below 1.0"
        );
        assert!(
            downforce < drag,
            "downforce loss should exceed drag reduction"
        );
    }

    // (b) Large gap and beyond range.
    #[test]
    fn wake_weakens_with_distance() {
        let w = WakeModel::f1_default();
        let (drag50, df50) = w.wake_effect(50.0, 0.0);
        assert!(
            drag50 > 0.95 && df50 > 0.95,
            "weak effect at 50m: {drag50}, {df50}"
        );

        let (drag70, df70) = w.wake_effect(70.0, 0.0);
        assert!(
            is_one(drag70) && is_one(df70),
            "beyond range must be neutral"
        );
    }

    // (c) Lateral offset weakens the effect.
    #[test]
    fn wake_weakens_off_center() {
        let w = WakeModel::f1_default();
        let (drag0, _) = w.wake_effect(10.0, 0.0);
        let (drag3, _) = w.wake_effect(10.0, 3.0);
        let (drag10, df10) = w.wake_effect(10.0, 10.0);

        assert!(
            drag0 < drag3,
            "on-center {drag0} should be stronger than 3m off {drag3}"
        );
        assert!(
            drag3 < drag10,
            "3m off {drag3} should be stronger than 10m off {drag10}"
        );
        // Far outside the cone: essentially clean air.
        assert!(
            drag10 > 0.99 && df10 > 0.99,
            "10m off should be near-neutral"
        );
    }

    // (d) Not behind / zero gap: no effect.
    #[test]
    fn wake_no_effect_when_not_behind() {
        let w = WakeModel::f1_default();
        let (d1, f1) = w.wake_effect(-5.0, 0.0);
        assert!(is_one(d1) && is_one(f1));
        let (d2, f2) = w.wake_effect(0.0, 0.0);
        assert!(is_one(d2) && is_one(f2));
    }

    // (e) Combined effect from multiple cars: the strongest dominates.
    #[test]
    fn combined_wake_takes_strongest() {
        let w = WakeModel::f1_default();
        let close_effect = w.wake_effect(5.0, 0.0);
        let combined = w.combined_wake_effect(&[(5.0, 0.0), (40.0, 0.0)]);
        assert!(close(combined.0, close_effect.0));
        assert!(close(combined.1, close_effect.1));
    }

    // (f) Slipstream speed gain.
    #[test]
    fn slipstream_gain_is_positive() {
        let w = WakeModel::f1_default();
        let gain = w.slipstream_speed_gain(90.0, 0.70);
        assert!(gain > 0.0);
        // 90 / sqrt(0.70) - 90 ~= 17.57 m/s
        assert!((gain - (90.0 / 0.70_f64.sqrt() - 90.0)).abs() < 1e-9);
        assert!((gain - 17.57).abs() < 0.1, "gain {gain}");
    }

    // (g) Corner speed penalty.
    #[test]
    fn corner_penalty_from_downforce_loss() {
        let w = WakeModel::f1_default();
        let penalty = w.corner_speed_penalty(0.55);
        assert!((penalty - (1.0 - 0.55_f64.sqrt())).abs() < 1e-9);
        assert!((penalty - 0.259).abs() < 0.01, "penalty {penalty}");
    }

    fn two_car_state(s_a: f64, s_b: f64, track: f64) -> MultiCarState {
        let mut st = MultiCarState::new(track, WakeModel::f1_default());
        st.add_car(s_a, 0.0, 80.0, CarParams::default());
        st.add_car(s_b, 0.0, 80.0, CarParams::default());
        st
    }

    // (h) Gap computation.
    #[test]
    fn gap_meters_basic() {
        let st = two_car_state(800.0, 200.0, 1000.0);
        // Car 0 at 800, car 1 at 200.
        assert!(close(st.gap_meters(0, 1), 600.0)); // 0 ahead of 1
        assert!(close(st.gap_meters(1, 0), 400.0)); // 1 ahead of 0 (wrap)
    }

    // (i) Wrap-around gap.
    #[test]
    fn gap_meters_wraps() {
        let st = two_car_state(950.0, 50.0, 1000.0);
        // Car 1 (s=50) is ahead of car 0 (s=950) after the wrap: ~100m.
        assert!(close(st.gap_meters(1, 0), 100.0));
    }

    // (j) Cars ahead within range.
    #[test]
    fn cars_ahead_filtered_by_range() {
        let mut st = MultiCarState::new(1000.0, WakeModel::f1_default());
        st.add_car(0.0, 0.0, 80.0, CarParams::default()); // car 0 (behind)
        st.add_car(10.0, 0.0, 80.0, CarParams::default()); // car 1 (gap 10)
        st.add_car(80.0, 0.0, 80.0, CarParams::default()); // car 2 (gap 80, beyond range)

        let ahead = st.cars_ahead_within_range(0, 60.0);
        assert_eq!(ahead.len(), 1, "only car 1 is within 60m");
        assert_eq!(ahead[0].0, 1);
        assert!(close(ahead[0].1, 10.0));
    }

    // (k) Collision detection.
    #[test]
    fn collision_detection() {
        // Same s, same n -> collision.
        let st = two_car_state(500.0, 500.0, 1000.0);
        assert!(st.is_collision(0, 1, 5.7, 2.0));

        // Same s, 3m lateral apart -> no collision (wider than the car).
        let mut st2 = MultiCarState::new(1000.0, WakeModel::f1_default());
        st2.add_car(500.0, 0.0, 80.0, CarParams::default());
        st2.add_car(500.0, 3.0, 80.0, CarParams::default());
        assert!(!st2.is_collision(0, 1, 5.7, 2.0));

        // Same n, 10m apart longitudinally -> no collision (longer than the car).
        let st3 = two_car_state(500.0, 510.0, 1000.0);
        assert!(!st3.is_collision(0, 1, 5.7, 2.0));
    }

    // (l) Advance wraps around the track.
    #[test]
    fn advance_wraps() {
        let mut st = MultiCarState::new(1000.0, WakeModel::f1_default());
        st.add_car(990.0, 0.0, 50.0, CarParams::default());
        st.advance(1.0);
        assert!(close(st.cars[0].s, 40.0), "s = {}", st.cars[0].s);
    }

    // (m) Wake exposure: follower in dirty air, leader in clean air.
    #[test]
    fn wake_effect_on_follower_and_leader() {
        let mut st = MultiCarState::new(1000.0, WakeModel::f1_default());
        st.add_car(10.0, 0.0, 80.0, CarParams::default()); // car 0 (leader)
        st.add_car(0.0, 0.0, 80.0, CarParams::default()); // car 1 (10m behind)

        let (drag, downforce) = st.wake_effect_on(1);
        assert!(
            drag < 1.0 && downforce < 1.0,
            "follower should be in dirty air"
        );

        let (ld, lf) = st.wake_effect_on(0);
        assert!(is_one(ld) && is_one(lf), "leader sees clean air");
    }
}
