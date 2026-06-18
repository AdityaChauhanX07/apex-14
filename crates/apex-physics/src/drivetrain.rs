//! Drivetrain models: engine torque curve, gearbox, and a combined powertrain
//! with automatic gear selection.

/// Engine model with a torque-vs-RPM curve.
///
/// The torque curve defines the maximum available torque at each engine speed.
/// Actual torque delivered is controlled by throttle position (0.0 to 1.0).
#[derive(Debug, Clone)]
pub struct Engine {
    /// Torque curve as (rpm, torque_nm) pairs, sorted by RPM ascending.
    /// Linear interpolation between points.
    pub torque_curve: Vec<(f64, f64)>,
    /// Maximum engine RPM (rev limiter).
    pub max_rpm: f64,
    /// Idle RPM.
    pub idle_rpm: f64,
}

impl Engine {
    /// Create a representative 2024-era F1 power unit.
    ///
    /// 1.6L turbo V6 hybrid producing approximately 710 kW (950 HP) combined.
    /// The torque curve peaks around 12,000 RPM with a broad plateau.
    pub fn f1_2024() -> Self {
        Engine {
            torque_curve: vec![
                (4000.0, 300.0),
                (6000.0, 420.0),
                (8000.0, 500.0),
                (10000.0, 540.0),
                (11000.0, 550.0),
                (12000.0, 545.0),
                (13000.0, 520.0),
                (14000.0, 480.0),
                (15000.0, 400.0),
            ],
            max_rpm: 15000.0,
            idle_rpm: 4000.0,
        }
    }

    /// Interpolate the maximum torque at a given RPM.
    /// Below idle: returns idle torque. Above max: returns zero (rev limiter).
    pub fn max_torque_at_rpm(&self, rpm: f64) -> f64 {
        if rpm <= self.idle_rpm {
            return self.torque_curve.first().map(|&(_, t)| t).unwrap_or(0.0);
        }
        if rpm >= self.max_rpm {
            return 0.0;
        }

        // Linear interpolation
        for window in self.torque_curve.windows(2) {
            let (rpm0, t0) = window[0];
            let (rpm1, t1) = window[1];
            if rpm >= rpm0 && rpm <= rpm1 {
                let frac = (rpm - rpm0) / (rpm1 - rpm0);
                return t0 + frac * (t1 - t0);
            }
        }

        // Past the last point but below max_rpm
        self.torque_curve.last().map(|&(_, t)| t).unwrap_or(0.0)
    }

    /// Compute the torque delivered at a given RPM and throttle position.
    /// throttle: 0.0 (closed) to 1.0 (wide open).
    pub fn torque(&self, rpm: f64, throttle: f64) -> f64 {
        self.max_torque_at_rpm(rpm) * throttle.clamp(0.0, 1.0)
    }

    /// Compute power output at a given RPM and throttle (watts).
    pub fn power(&self, rpm: f64, throttle: f64) -> f64 {
        let torque = self.torque(rpm, throttle);
        torque * rpm * std::f64::consts::PI / 30.0 // P = T * omega, omega = rpm * 2*pi/60
    }

    /// Find the RPM that produces peak power at wide-open throttle.
    pub fn peak_power_rpm(&self) -> f64 {
        let mut best_rpm = self.idle_rpm;
        let mut best_power = 0.0f64;

        // Sweep in 100 RPM steps
        let mut rpm = self.idle_rpm;
        while rpm <= self.max_rpm {
            let p = self.power(rpm, 1.0);
            if p > best_power {
                best_power = p;
                best_rpm = rpm;
            }
            rpm += 100.0;
        }
        best_rpm
    }
}

/// Sequential gearbox with fixed ratios.
///
/// Each gear ratio converts engine RPM to wheel RPM via:
/// wheel_rpm = engine_rpm / (gear_ratio * final_drive)
#[derive(Debug, Clone)]
pub struct Gearbox {
    /// Gear ratios (engine_rpm / gearbox_output_rpm), indexed by gear number.
    /// Index 0 = first gear (highest ratio), last = top gear (lowest ratio).
    pub ratios: Vec<f64>,
    /// Final drive ratio (gearbox output to wheel).
    pub final_drive: f64,
    /// Drivetrain efficiency (0.0 to 1.0). Accounts for friction losses.
    pub efficiency: f64,
    /// Time to complete a gear shift (s). During a shift, no torque is transmitted.
    pub shift_time: f64,
}

impl Gearbox {
    /// Create a representative F1 8-speed gearbox.
    ///
    /// The final drive is geared so that top gear (total ratio 0.82 * 6.70 =
    /// 5.49) reaches the 15,000-rpm limiter at ~94 m/s (~340 km/h) on 0.330 m
    /// wheels — a realistic F1 top speed.
    pub fn f1_2024() -> Self {
        Gearbox {
            ratios: vec![3.20, 2.40, 1.90, 1.55, 1.28, 1.08, 0.93, 0.82],
            final_drive: 6.70,
            efficiency: 0.95,
            shift_time: 0.05, // 50ms shift (modern F1 is ~20ms, but model is simpler)
        }
    }

    /// Number of gears.
    pub fn num_gears(&self) -> usize {
        self.ratios.len()
    }

    /// Total ratio from engine to wheel for a given gear.
    pub fn total_ratio(&self, gear: usize) -> f64 {
        self.ratios[gear.min(self.ratios.len() - 1)] * self.final_drive
    }

    /// Compute engine RPM from wheel angular velocity and gear.
    /// wheel_omega: wheel angular velocity (rad/s).
    pub fn engine_rpm(&self, wheel_omega: f64, gear: usize) -> f64 {
        let omega_engine = wheel_omega * self.total_ratio(gear);
        omega_engine * 30.0 / std::f64::consts::PI // convert rad/s to RPM
    }

    /// Compute wheel torque from engine torque and gear.
    pub fn wheel_torque(&self, engine_torque: f64, gear: usize) -> f64 {
        engine_torque * self.total_ratio(gear) * self.efficiency
    }

    /// Determine the optimal gear for a given wheel speed and engine.
    ///
    /// Selects the gear that produces maximum wheel torque at the current speed.
    /// Avoids over-revving (stays below engine max_rpm).
    pub fn optimal_gear(&self, engine: &Engine, wheel_omega: f64) -> usize {
        let mut best_gear = 0;
        let mut best_torque = f64::NEG_INFINITY;

        for gear in 0..self.ratios.len() {
            let rpm = self.engine_rpm(wheel_omega, gear);
            if rpm > engine.max_rpm || rpm < engine.idle_rpm {
                continue;
            }
            let engine_torque = engine.max_torque_at_rpm(rpm);
            let wheel_t = self.wheel_torque(engine_torque, gear);
            if wheel_t > best_torque {
                best_torque = wheel_t;
                best_gear = gear;
            }
        }

        best_gear
    }

    /// Compute the vehicle speed at which a gear hits the rev limiter.
    /// Returns speed in m/s.
    pub fn gear_max_speed(&self, engine: &Engine, gear: usize, wheel_radius: f64) -> f64 {
        let omega_engine_max = engine.max_rpm * std::f64::consts::PI / 30.0;
        let omega_wheel = omega_engine_max / self.total_ratio(gear);
        omega_wheel * wheel_radius
    }
}

/// Complete powertrain: engine + gearbox with automatic gear selection.
#[derive(Debug, Clone)]
pub struct Powertrain {
    pub engine: Engine,
    pub gearbox: Gearbox,
    current_gear: usize,
}

impl Powertrain {
    pub fn f1_2024() -> Self {
        Powertrain {
            engine: Engine::f1_2024(),
            gearbox: Gearbox::f1_2024(),
            current_gear: 0,
        }
    }

    /// Get the current gear (0-indexed).
    pub fn current_gear(&self) -> usize {
        self.current_gear
    }

    /// Compute the drive torque at the wheels given throttle position and wheel speed.
    ///
    /// Automatically selects the optimal gear. Returns the total wheel torque
    /// accounting for gear ratio and drivetrain efficiency.
    pub fn drive_torque(&mut self, throttle: f64, wheel_omega: f64) -> f64 {
        // Select optimal gear
        self.current_gear = self.gearbox.optimal_gear(&self.engine, wheel_omega);

        let rpm = self.gearbox.engine_rpm(wheel_omega, self.current_gear);
        let engine_torque = self.engine.torque(rpm, throttle);
        self.gearbox.wheel_torque(engine_torque, self.current_gear)
    }

    /// Compute the maximum possible wheel torque at a given wheel speed.
    pub fn max_drive_torque(&mut self, wheel_omega: f64) -> f64 {
        self.drive_torque(1.0, wheel_omega)
    }

    /// Get the top speed of the car in the highest gear (m/s).
    pub fn top_speed(&self, wheel_radius: f64) -> f64 {
        let top_gear = self.gearbox.num_gears() - 1;
        self.gearbox.gear_max_speed(&self.engine, top_gear, wheel_radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // --- Engine torque curve ---

    #[test]
    fn engine_torque_curve_key_points() {
        let e = Engine::f1_2024();
        // At idle (4000 RPM): 300 N*m (first point).
        assert!(approx(e.max_torque_at_rpm(4000.0), 300.0, 1e-9));
        // At peak (11000 RPM): 550 N*m.
        assert!(approx(e.max_torque_at_rpm(11000.0), 550.0, 1e-9));
        // At redline (15000 RPM): rev limiter cuts to 0.
        assert_eq!(e.max_torque_at_rpm(15000.0), 0.0);
        // Just below redline (14999 RPM): near the last curve point (~400).
        assert!(
            (e.max_torque_at_rpm(14999.0) - 400.0).abs() < 5.0,
            "near-redline torque {}",
            e.max_torque_at_rpm(14999.0)
        );
    }

    #[test]
    fn engine_above_max_rpm_is_zero() {
        let e = Engine::f1_2024();
        assert_eq!(e.max_torque_at_rpm(15001.0), 0.0);
        assert_eq!(e.max_torque_at_rpm(20000.0), 0.0);
    }

    #[test]
    fn engine_below_idle_returns_idle_torque() {
        let e = Engine::f1_2024();
        assert!(approx(e.max_torque_at_rpm(2000.0), 300.0, 1e-9));
        assert!(approx(e.max_torque_at_rpm(0.0), 300.0, 1e-9));
    }

    #[test]
    fn engine_interpolates_between_points() {
        let e = Engine::f1_2024();
        // 7000 RPM sits between (6000, 420) and (8000, 500): expect the midpoint.
        let t = e.max_torque_at_rpm(7000.0);
        assert!((420.0..=500.0).contains(&t), "torque {} out of range", t);
        assert!(approx(t, 460.0, 1e-9), "interpolated torque {}", t);
    }

    // --- Engine power ---

    #[test]
    fn engine_peak_power_in_realistic_range() {
        let e = Engine::f1_2024();
        let rpm = e.peak_power_rpm();
        let peak_power = e.power(rpm, 1.0);

        // The reported peak RPM should indeed maximize power over the sweep.
        let mut sweep = e.idle_rpm;
        let mut max_seen = 0.0f64;
        while sweep <= e.max_rpm {
            max_seen = max_seen.max(e.power(sweep, 1.0));
            sweep += 100.0;
        }
        assert!(approx(peak_power, max_seen, 1e-6), "peak {peak_power} vs swept {max_seen}");

        // 600-800 kW for the F1 preset.
        assert!(
            (600_000.0..=800_000.0).contains(&peak_power),
            "peak power {peak_power} W out of range"
        );
    }

    #[test]
    fn engine_power_at_idle_below_peak() {
        let e = Engine::f1_2024();
        let idle_power = e.power(e.idle_rpm, 1.0);
        let peak_power = e.power(e.peak_power_rpm(), 1.0);
        assert!(
            idle_power < 0.5 * peak_power,
            "idle power {idle_power} should be well below peak {peak_power}"
        );
    }

    // --- Throttle scaling ---

    #[test]
    fn throttle_scales_torque_linearly() {
        let e = Engine::f1_2024();
        let rpm = 10000.0;
        let full = e.torque(rpm, 1.0);
        assert!(approx(e.torque(rpm, 0.5), 0.5 * full, 1e-9));
        assert_eq!(e.torque(rpm, 0.0), 0.0);
    }

    #[test]
    fn throttle_is_clamped() {
        let e = Engine::f1_2024();
        let rpm = 10000.0;
        let full = e.torque(rpm, 1.0);
        // Over-range throttle is clamped to [0, 1].
        assert!(approx(e.torque(rpm, 1.5), full, 1e-9));
        assert_eq!(e.torque(rpm, -0.5), 0.0);
    }

    // --- Gearbox ratios ---

    #[test]
    fn gearbox_ratio_ordering() {
        let g = Gearbox::f1_2024();
        // First gear is the highest total ratio.
        assert!(approx(g.total_ratio(0), 3.20 * 6.70, 1e-9));
        // Top gear is the lowest.
        assert!(approx(g.total_ratio(g.num_gears() - 1), 0.82 * 6.70, 1e-9));
        assert!(g.total_ratio(0) > g.total_ratio(g.num_gears() - 1));
    }

    #[test]
    fn gearbox_lower_gear_higher_rpm() {
        let g = Gearbox::f1_2024();
        let wheel_omega = 50.0; // rad/s
        let rpm_first = g.engine_rpm(wheel_omega, 0);
        let rpm_second = g.engine_rpm(wheel_omega, 1);
        assert!(
            rpm_first > rpm_second,
            "first-gear rpm {rpm_first} should exceed second {rpm_second}"
        );
    }

    // --- Optimal gear selection ---

    #[test]
    fn optimal_gear_low_speed_is_first() {
        let e = Engine::f1_2024();
        let g = Gearbox::f1_2024();
        // Walking pace: ~1.4 m/s -> wheel_omega ~4.2 rad/s at r=0.33.
        let wheel_omega = 1.4 / 0.330;
        assert_eq!(g.optimal_gear(&e, wheel_omega), 0, "low speed should pick first gear");
    }

    #[test]
    fn optimal_gear_high_speed_is_higher() {
        let e = Engine::f1_2024();
        let g = Gearbox::f1_2024();
        // ~80 m/s wheel speed.
        let wheel_omega = 80.0 / 0.330;
        let gear = g.optimal_gear(&e, wheel_omega);
        assert!(gear > 0, "high speed should pick a higher gear, got {gear}");
    }

    #[test]
    fn optimal_gear_never_over_revs() {
        let e = Engine::f1_2024();
        let g = Gearbox::f1_2024();
        // Sweep a range of wheel speeds; the selected gear must keep rpm in range.
        let mut v = 2.0;
        while v <= 100.0 {
            let wheel_omega = v / 0.330;
            let gear = g.optimal_gear(&e, wheel_omega);
            let rpm = g.engine_rpm(wheel_omega, gear);
            // Gear 0 is the fallback when nothing is feasible; only assert the
            // limiter when a genuinely feasible gear exists.
            let feasible = (0..g.num_gears()).any(|gg| {
                let r = g.engine_rpm(wheel_omega, gg);
                r >= e.idle_rpm && r <= e.max_rpm
            });
            if feasible {
                assert!(
                    rpm <= e.max_rpm + 1e-6,
                    "v={v}: gear {gear} over-revs at {rpm} rpm"
                );
            }
            v += 1.0;
        }
    }

    // --- Top speed ---

    #[test]
    fn top_speed_matches_gearing() {
        // With the 6.70 final drive (top total ratio 0.82 * 6.70 = 5.49) and the
        // 15000-rpm limiter on 0.33 m wheels, the rev-limited top speed is ~94 m/s
        // (~340 km/h) — a realistic F1 figure.
        let p = Powertrain::f1_2024();
        let v = p.top_speed(0.330);

        let expected = 15000.0 * std::f64::consts::PI / 30.0 / (0.82 * 6.70) * 0.330;
        assert!(approx(v, expected, 1e-6), "top speed {v} vs expected {expected}");
        assert!(
            (85.0..=100.0).contains(&v),
            "top speed {v} m/s ({:.0} km/h) out of expected band",
            v * 3.6
        );
    }

    // --- Wheel torque ---

    #[test]
    fn wheel_torque_higher_in_first_gear() {
        let g = Gearbox::f1_2024();
        let engine_torque = 550.0; // peak
        let first = g.wheel_torque(engine_torque, 0);
        let top = g.wheel_torque(engine_torque, g.num_gears() - 1);

        // First gear multiplies torque the most.
        assert!(approx(first, 550.0 * 3.20 * 6.70 * 0.95, 1e-6));
        assert!(first > top, "first {first} should exceed top {top}");
    }

    // --- Powertrain ---

    #[test]
    fn powertrain_auto_selects_gears() {
        let mut p = Powertrain::f1_2024();

        // Low speed, full throttle: low gear, high torque. (20 m/s keeps first
        // gear above idle; below ~14 m/s the tall first gear falls under idle.)
        let low_omega = 20.0 / 0.330;
        let low_torque = p.drive_torque(1.0, low_omega);
        let low_gear = p.current_gear();

        // High speed, full throttle: higher gear, lower torque.
        let high_omega = 85.0 / 0.330;
        let high_torque = p.drive_torque(1.0, high_omega);
        let high_gear = p.current_gear();

        assert!(low_torque > 0.0, "low-speed torque {low_torque}");
        assert!(high_torque > 0.0, "high-speed torque {high_torque}");
        assert!(
            low_torque > high_torque,
            "low-gear torque {low_torque} should exceed high-gear {high_torque}"
        );
        assert!(
            high_gear > low_gear,
            "gear should increase with speed: {low_gear} -> {high_gear}"
        );
    }

    #[test]
    fn powertrain_max_drive_torque_is_full_throttle() {
        let mut p = Powertrain::f1_2024();
        let omega = 40.0 / 0.330;
        let max_t = p.max_drive_torque(omega);
        let mut p2 = Powertrain::f1_2024();
        let full_t = p2.drive_torque(1.0, omega);
        assert!(approx(max_t, full_t, 1e-9));
    }
}
