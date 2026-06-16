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
        }
    }
}

impl CarParams {
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
        let dfz_front =
            self.mass * lateral_accel * self.cog_height * roll_stiffness_front_fraction
                / self.track_width_front;
        let dfz_rear = self.mass * lateral_accel * self.cog_height
            * (1.0 - roll_stiffness_front_fraction)
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
}
