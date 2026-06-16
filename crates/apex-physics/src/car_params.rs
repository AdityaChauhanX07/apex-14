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
