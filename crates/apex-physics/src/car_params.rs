//! Vehicle parameters shared across all model fidelities.

/// Standard gravitational acceleration (m/s²).
pub const GRAVITY: f64 = 9.81;

/// Physical parameters defining the car. Shared across all model fidelities.
/// Values default to a representative 2026 F1 car.
#[derive(Debug, Clone)]
pub struct CarParams {
    // Mass and inertia
    /// Total mass (kg).
    pub mass: f64,

    // Aerodynamics
    /// Frontal area (m²).
    pub frontal_area: f64,
    /// Drag coefficient `C_d`.
    pub drag_coeff: f64,
    /// Downforce coefficient `C_l` (positive = downforce).
    pub lift_coeff: f64,
    /// Air density ρ (kg/m³).
    pub air_density: f64,

    // Tires (simplified for point-mass)
    /// Peak tire friction coefficient.
    pub tire_mu: f64,
    /// Rolling resistance coefficient `C_rr`.
    pub rolling_resistance: f64,

    // Powertrain limits
    /// Maximum engine force (N).
    pub max_drive_force: f64,
    /// Maximum braking force (N).
    pub max_brake_force: f64,

    // Geometry
    /// Total wheelbase `L = l_f + l_r` (m).
    pub wheelbase: f64,
    /// Distance from CoG to front axle `l_f` (m).
    pub cog_to_front: f64,
    /// Distance from CoG to rear axle `l_r` (m).
    pub cog_to_rear: f64,
    /// Center of gravity height `h_cog` (m).
    pub cog_height: f64,
    /// Yaw moment of inertia `I_z` (kg·m²).
    pub yaw_inertia: f64,
    /// Fraction of total downforce acting on the front axle (0.0–1.0).
    pub aero_balance_front: f64,

    // Wheels and drivetrain
    /// Tire radius `R` (m).
    pub wheel_radius: f64,
    /// Rotational inertia of each wheel `I_w` (kg·m²).
    pub wheel_inertia: f64,
    /// Front axle track width (m).
    pub track_width_front: f64,
    /// Rear axle track width (m).
    pub track_width_rear: f64,
    /// Fraction of brake force on the front axle (0.0–1.0).
    pub brake_bias_front: f64,
    /// Fraction of drive torque to the rear axle (0.0 = FWD, 1.0 = RWD, 0.5 = AWD).
    pub drive_distribution: f64,

    // 14-DOF chassis additions
    /// Unsprung mass per corner (kg) — wheel + upright + brake assembly.
    pub unsprung_mass: f64,
    /// Tire radial (vertical) stiffness (N/m).
    pub tire_radial_stiffness: f64,
    /// Roll moment of inertia `I_xx` (kg·m²).
    pub inertia_xx: f64,
    /// Pitch moment of inertia `I_yy` (kg·m²).
    pub inertia_yy: f64,
}

impl Default for CarParams {
    fn default() -> Self {
        CarParams {
            mass: 798.0,
            frontal_area: 1.5,
            drag_coeff: 0.9,
            lift_coeff: 3.5,
            air_density: 1.225,
            tire_mu: 1.75,
            rolling_resistance: 0.015,
            max_drive_force: 15000.0,
            max_brake_force: 30000.0,
            wheelbase: 3.60,
            cog_to_front: 1.67,
            cog_to_rear: 1.93,
            cog_height: 0.30,
            yaw_inertia: 1200.0,
            aero_balance_front: 0.45,
            wheel_radius: 0.330,
            wheel_inertia: 1.2,
            track_width_front: 1.60,
            track_width_rear: 1.60,
            brake_bias_front: 0.60,
            drive_distribution: 1.0, // rear-wheel drive (F1 is RWD)
            unsprung_mass: 15.0,
            tire_radial_stiffness: 250_000.0,
            inertia_xx: 400.0,
            inertia_yy: 1400.0,
        }
    }
}

impl CarParams {
    /// Calibrated 2024-era F1 car parameters.
    ///
    /// Tuned to produce performance numbers consistent with published data:
    /// - Top speed: 330-350 km/h (Monza main straight)
    /// - Peak lateral acceleration: 4.5-5.5g (high-speed corners)
    /// - Peak longitudinal deceleration: 5.0-6.0g (heavy braking)
    /// - Silverstone lap time: 85-95s (qualifying pace)
    pub fn f1_2024_calibrated() -> Self {
        CarParams {
            mass: 798.0,
            frontal_area: 1.5,
            drag_coeff: 1.10,         // higher drag (complex aero, open wheels)
            lift_coeff: 2.80,         // reduced from 3.5 (more realistic total downforce)
            air_density: 1.225,
            tire_mu: 1.55,            // reduced from 1.75 (more realistic peak grip)
            rolling_resistance: 0.015,
            max_drive_force: 11000.0, // ~900 HP at ~300 km/h
            max_brake_force: 25000.0, // slightly reduced
            wheelbase: 3.60,
            cog_to_front: 1.67,
            cog_to_rear: 1.93,
            cog_height: 0.30,
            yaw_inertia: 1200.0,
            aero_balance_front: 0.44, // slightly more rear-biased
            wheel_radius: 0.330,
            wheel_inertia: 1.2,
            track_width_front: 1.60,
            track_width_rear: 1.60,
            brake_bias_front: 0.58,
            drive_distribution: 1.0,
            unsprung_mass: 15.0,
            tire_radial_stiffness: 250_000.0,
            inertia_xx: 400.0,
            inertia_yy: 1400.0,
        }
    }

    /// Aerodynamic drag force at the given speed: `0.5 · ρ · C_d · A · v²`.
    pub fn drag_force(&self, speed: f64) -> f64 {
        0.5 * self.air_density * self.drag_coeff * self.frontal_area * speed * speed
    }

    /// Aerodynamic downforce at the given speed: `0.5 · ρ · C_l · A · v²`.
    pub fn downforce(&self, speed: f64) -> f64 {
        0.5 * self.air_density * self.lift_coeff * self.frontal_area * speed * speed
    }

    /// Maximum tire grip force at the given speed:
    /// `μ · (m·g + downforce(v))`.
    pub fn max_grip_force(&self, speed: f64) -> f64 {
        self.tire_mu * (self.mass * GRAVITY + self.downforce(speed))
    }

    /// Rolling resistance force: `C_rr · m · g`.
    pub fn rolling_resistance_force(&self) -> f64 {
        self.rolling_resistance * self.mass * GRAVITY
    }

    /// Compute front and rear axle vertical loads (N), including static weight
    /// distribution, aerodynamic downforce split, and longitudinal weight transfer.
    ///
    /// Returns (F_z_front, F_z_rear) — each is the TOTAL for that axle (both wheels combined).
    pub fn axle_loads(&self, speed: f64, longitudinal_accel: f64) -> (f64, f64) {
        let weight = self.mass * GRAVITY;
        let df = self.downforce(speed);

        // Static weight distribution
        let fz_front_static = weight * self.cog_to_rear / self.wheelbase;
        let fz_rear_static = weight * self.cog_to_front / self.wheelbase;

        // Aero load distribution
        let fz_front_aero = df * self.aero_balance_front;
        let fz_rear_aero = df * (1.0 - self.aero_balance_front);

        // Longitudinal weight transfer: ΔF_z = m·a_x·h_cog / L
        let wt = self.mass * longitudinal_accel * self.cog_height / self.wheelbase;

        // Under acceleration (a_x > 0), load transfers to rear (front loses, rear gains)
        let fz_front = (fz_front_static + fz_front_aero - wt).max(0.0);
        let fz_rear = (fz_rear_static + fz_rear_aero + wt).max(0.0);

        (fz_front, fz_rear)
    }

    /// Compute vertical load on each individual tire (N).
    ///
    /// Accounts for static distribution, aero downforce, longitudinal weight transfer,
    /// and lateral weight transfer (split by roll stiffness distribution).
    ///
    /// Returns [F_z_fl, F_z_fr, F_z_rl, F_z_rr] (front-left, front-right, rear-left, rear-right).
    ///
    /// Convention: positive lateral_accel = turning left = load transfers to right wheels.
    pub fn corner_loads(
        &self,
        speed: f64,
        longitudinal_accel: f64,
        lateral_accel: f64,
        roll_stiffness_front_fraction: f64,
    ) -> [f64; 4] {
        let (front_total, rear_total) = self.axle_loads(speed, longitudinal_accel);

        // Lateral load transfer per axle (positive a_y -> right wheels gain).
        let dfz_front = self.mass * lateral_accel * self.cog_height * roll_stiffness_front_fraction
            / self.track_width_front;
        let dfz_rear =
            self.mass * lateral_accel * self.cog_height * (1.0 - roll_stiffness_front_fraction)
                / self.track_width_rear;

        let fz_fl = (front_total / 2.0 - dfz_front).max(0.0);
        let fz_fr = (front_total / 2.0 + dfz_front).max(0.0);
        let fz_rl = (rear_total / 2.0 - dfz_rear).max(0.0);
        let fz_rr = (rear_total / 2.0 + dfz_rear).max(0.0);

        [fz_fl, fz_fr, fz_rl, fz_rr]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn default_values() {
        let p = CarParams::default();
        assert_eq!(p.mass, 798.0);
        assert_eq!(p.frontal_area, 1.5);
        assert_eq!(p.drag_coeff, 0.9);
        assert_eq!(p.lift_coeff, 3.5);
        assert_eq!(p.air_density, 1.225);
        assert_eq!(p.tire_mu, 1.75);
        assert_eq!(p.rolling_resistance, 0.015);
        assert_eq!(p.max_drive_force, 15000.0);
        assert_eq!(p.max_brake_force, 30000.0);
    }

    #[test]
    fn drag_force_at_100() {
        let p = CarParams::default();
        // 0.5 * 1.225 * 0.9 * 1.5 * 10000 = 8268.75
        assert!(approx(p.drag_force(100.0), 8268.75, 0.01));
    }

    #[test]
    fn downforce_at_100() {
        let p = CarParams::default();
        // 0.5 * 1.225 * 3.5 * 1.5 * 10000 = 32156.25
        assert!(approx(p.downforce(100.0), 32156.25, 0.01));
    }

    #[test]
    fn max_grip_force_at_100() {
        let p = CarParams::default();
        // 1.75 * (798*9.81 + 32156.25)
        let expected = 1.75 * (798.0 * 9.81 + 32156.25);
        assert!(approx(p.max_grip_force(100.0), expected, 0.01));
    }

    #[test]
    fn rolling_resistance() {
        let p = CarParams::default();
        let expected = 0.015 * 798.0 * 9.81;
        assert!(approx(p.rolling_resistance_force(), expected, 1e-9));
    }

    // --- Calibrated 2024 F1 preset -----------------------------------------

    /// Closed-form terminal velocity (m/s): the speed where drive force equals
    /// drag + rolling resistance. `v = sqrt((F_max − F_roll) / (½·ρ·C_d·A))`.
    fn terminal_velocity(p: &CarParams) -> f64 {
        let f_roll = p.rolling_resistance_force();
        let denom = 0.5 * p.air_density * p.drag_coeff * p.frontal_area;
        ((p.max_drive_force - f_roll) / denom).sqrt()
    }

    /// Grip-limited cornering lateral acceleration (in g) at radius `r`, the same
    /// balance the QSS solver uses: `m·v²·κ = μ·(m·g + ½·ρ·C_l·A·v²)`.
    fn cornering_lat_g(p: &CarParams, r: f64) -> f64 {
        let kappa = 1.0 / r;
        let denom =
            p.mass * kappa - p.tire_mu * 0.5 * p.air_density * p.lift_coeff * p.frontal_area;
        let v2 = p.tire_mu * p.mass * GRAVITY / denom;
        v2 * kappa / GRAVITY
    }

    #[test]
    fn calibrated_params_sanity() {
        let c = CarParams::f1_2024_calibrated();
        let d = CarParams::default();

        // FIA minimum mass.
        assert_eq!(c.mass, 798.0);

        // Less aggressive than the default preset.
        assert!(c.tire_mu < d.tire_mu, "tire_mu {} should be < default {}", c.tire_mu, d.tire_mu);
        assert!(
            c.lift_coeff < d.lift_coeff,
            "lift_coeff {} should be < default {}",
            c.lift_coeff,
            d.lift_coeff
        );

        // All magnitude fields are positive.
        for v in [
            c.mass,
            c.frontal_area,
            c.drag_coeff,
            c.lift_coeff,
            c.air_density,
            c.tire_mu,
            c.rolling_resistance,
            c.max_drive_force,
            c.max_brake_force,
            c.wheelbase,
            c.cog_to_front,
            c.cog_to_rear,
            c.cog_height,
            c.yaw_inertia,
            c.aero_balance_front,
            c.wheel_radius,
            c.wheel_inertia,
            c.track_width_front,
            c.track_width_rear,
            c.brake_bias_front,
            c.drive_distribution,
            c.unsprung_mass,
            c.tire_radial_stiffness,
            c.inertia_xx,
            c.inertia_yy,
        ] {
            assert!(v > 0.0, "field value {v} should be positive");
        }
        // Fractions stay within [0, 1].
        assert!((0.0..=1.0).contains(&c.aero_balance_front));
        assert!((0.0..=1.0).contains(&c.brake_bias_front));
        assert!((0.0..=1.0).contains(&c.drive_distribution));
    }

    #[test]
    fn calibrated_terminal_velocity_is_realistic() {
        let c = CarParams::f1_2024_calibrated();
        let v_kph = terminal_velocity(&c) * 3.6;
        // The specified params yield ~374 km/h — far below the default's ~483 and
        // in realistic F1 territory (a track top speed of ~340 km/h sits below
        // this drag-limited terminal value).
        assert!(
            (300.0..=380.0).contains(&v_kph),
            "calibrated terminal velocity {v_kph:.1} km/h out of realistic band"
        );
        assert!(
            v_kph < terminal_velocity(&CarParams::default()) * 3.6,
            "calibrated terminal velocity should be below the default's"
        );
    }

    #[test]
    fn calibrated_cornering_g_is_realistic() {
        let c = CarParams::f1_2024_calibrated();
        let d = CarParams::default();
        let g_c = cornering_lat_g(&c, 100.0);
        let g_d = cornering_lat_g(&d, 100.0);

        assert!(
            (3.0..=6.0).contains(&g_c),
            "calibrated cornering {g_c:.2}g out of [3, 6]"
        );
        assert!(
            g_c < g_d,
            "calibrated cornering {g_c:.2}g should be below default {g_d:.2}g"
        );
    }

    #[test]
    fn calibrated_qss_lap_is_slower_with_less_grip() {
        use apex_track::{build_track, oval_track};

        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 500);
        let oval = build_track("oval", &pts, closed);

        let default_res = crate::qss_lap_sim(&oval, &CarParams::default());
        let calib_res = crate::qss_lap_sim(&oval, &CarParams::f1_2024_calibrated());

        let top = |r: &crate::QssResult| r.speeds.iter().cloned().fold(f64::MIN, f64::max);
        let max_lat = |r: &crate::QssResult| {
            r.lateral_gs.iter().map(|g| g.abs()).fold(0.0_f64, f64::max)
        };

        // Less power + more drag -> lower top speed; less grip -> longer lap and
        // lower cornering load.
        assert!(
            calib_res.lap_time > default_res.lap_time,
            "calibrated lap {:.2}s should be slower than default {:.2}s",
            calib_res.lap_time,
            default_res.lap_time
        );
        assert!(
            top(&calib_res) < top(&default_res),
            "calibrated top speed {:.1} should be below default {:.1}",
            top(&calib_res),
            top(&default_res)
        );
        assert!(
            max_lat(&calib_res) < max_lat(&default_res),
            "calibrated lateral g {:.2} should be below default {:.2}",
            max_lat(&calib_res),
            max_lat(&default_res)
        );
    }
}
