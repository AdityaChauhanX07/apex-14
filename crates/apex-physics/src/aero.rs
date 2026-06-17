//! Ride-height-sensitive aerodynamics model.
//!
//! F1 ground-effect downforce depends strongly on ride height: too high and the
//! floor loses ground effect, too low and it stalls. This module captures that
//! with separate front/rear downforce maps plus pitch-sensitive drag.

/// Aerodynamic model with ride-height-sensitive downforce.
///
/// F1 ground effect downforce is a strong function of ride height: too high
/// and you lose ground effect, too low and the floor stalls. This model
/// captures that behavior with separate front and rear downforce maps.
#[derive(Debug, Clone, Copy)]
pub struct AeroModel {
    /// Base drag coefficient at design ride height.
    pub cd_base: f64,
    /// Frontal area (m²).
    pub frontal_area: f64,
    /// Air density (kg/m³).
    pub air_density: f64,

    /// Base front downforce coefficient (at design ride height).
    pub cl_front_base: f64,
    /// Base rear downforce coefficient (at design ride height).
    pub cl_rear_base: f64,

    /// Design ride height — the height at which the aero coefficients are nominal (m).
    pub design_ride_height: f64,
    /// Ride height at which the floor stalls and downforce collapses (m).
    /// Below this height, downforce drops sharply.
    pub stall_ride_height: f64,
    /// Ride height above which ground effect is negligible (m).
    /// Above this, downforce asymptotes to a reduced baseline.
    pub high_ride_height: f64,

    /// Sensitivity of drag to pitch angle (rad⁻¹).
    /// Positive pitch (nose up) increases drag slightly.
    pub drag_pitch_sensitivity: f64,
}

impl AeroModel {
    /// Returns an F1-representative aero model.
    pub fn f1_default() -> Self {
        AeroModel {
            cd_base: 0.9,
            frontal_area: 1.5,
            air_density: 1.225,
            cl_front_base: 1.575,      // 45% of total C_l = 3.5
            cl_rear_base: 1.925,       // 55% of total C_l = 3.5
            design_ride_height: 0.030, // 30mm
            stall_ride_height: 0.010,  // 10mm — floor stalls below this
            high_ride_height: 0.060,   // 60mm — ground effect fading
            drag_pitch_sensitivity: 0.5,
        }
    }

    /// Compute the ride-height efficiency factor for downforce.
    ///
    /// Returns a multiplier in [0, 1] that scales the base downforce coefficient:
    /// - At design ride height: returns 1.0 (full downforce).
    /// - Above design: gradually decreases toward ~0.5 at high_ride_height.
    /// - Below design: increases slightly (more ground effect) down to a peak,
    ///   then collapses below stall_ride_height.
    ///
    /// The shape is a smooth curve that captures:
    /// 1. Increasing ground effect as ride height decreases toward the design point.
    /// 2. A sharp stall below the critical height.
    /// 3. Diminishing ground effect above the design point.
    pub fn ride_height_factor(&self, ride_height: f64) -> f64 {
        let h = ride_height;
        let h_design = self.design_ride_height;
        let h_stall = self.stall_ride_height;
        let h_high = self.high_ride_height;

        if h <= 0.0 {
            // At or below ground: fully stalled
            return 0.0;
        }

        if h < h_stall {
            // Below stall: downforce collapses. Use smooth ramp from 0 to stall value.
            let t = h / h_stall; // 0 to 1
                                 // Smooth ramp: t² gives a gentle onset
            return 0.3 * t * t; // peaks at 0.3 at the stall height
        }

        if h <= h_design {
            // Between stall and design: ground effect increasing
            // Interpolate from 0.3 (at stall) to 1.0 (at design)
            let t = (h - h_stall) / (h_design - h_stall); // 0 to 1
                                                          // Smooth interpolation using cubic Hermite
            let smooth_t = t * t * (3.0 - 2.0 * t);
            return 0.3 + 0.7 * smooth_t;
        }

        // Above design: ground effect fading
        // Interpolate from 1.0 (at design) to 0.5 (at high and beyond)
        let t = ((h - h_design) / (h_high - h_design)).min(1.0); // 0 to 1
        let smooth_t = t * t * (3.0 - 2.0 * t);
        1.0 - 0.5 * smooth_t // 1.0 at design, 0.5 at high
    }

    /// Compute the dynamic pressure: q = 0.5 · ρ · v².
    pub fn dynamic_pressure(&self, speed: f64) -> f64 {
        0.5 * self.air_density * speed * speed
    }

    /// Compute aerodynamic forces and moments.
    ///
    /// Arguments:
    /// - speed: vehicle speed (m/s)
    /// - front_ride_height: average ride height at front axle (m)
    /// - rear_ride_height: average ride height at rear axle (m)
    /// - pitch_angle: chassis pitch angle (rad, positive = nose up)
    ///
    /// Returns an AeroForces struct.
    pub fn compute(
        &self,
        speed: f64,
        front_ride_height: f64,
        rear_ride_height: f64,
        pitch_angle: f64,
    ) -> AeroForces {
        let q = self.dynamic_pressure(speed);
        let area = self.frontal_area;

        // Front and rear downforce with ride-height sensitivity
        let front_factor = self.ride_height_factor(front_ride_height);
        let rear_factor = self.ride_height_factor(rear_ride_height);

        let downforce_front = q * area * self.cl_front_base * front_factor;
        let downforce_rear = q * area * self.cl_rear_base * rear_factor;

        // Drag with pitch sensitivity
        let cd = self.cd_base * (1.0 + self.drag_pitch_sensitivity * pitch_angle.abs());
        let drag = q * area * cd;

        // Aerodynamic pitch moment: difference between front and rear downforce
        // creates a pitch moment about the CoG. Not computed here — it's handled
        // by the chassis dynamics using the application points.

        AeroForces {
            drag,
            downforce_front,
            downforce_rear,
            downforce_total: downforce_front + downforce_rear,
        }
    }
}

/// Result of aerodynamic force computation.
#[derive(Debug, Clone, Copy)]
pub struct AeroForces {
    /// Aerodynamic drag force (N), opposing motion.
    pub drag: f64,
    /// Front axle downforce (N).
    pub downforce_front: f64,
    /// Rear axle downforce (N).
    pub downforce_rear: f64,
    /// Total downforce (N).
    pub downforce_total: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::car_params::CarParams;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ride_height_factor_key_points() {
        let a = AeroModel::f1_default();
        assert!(approx(a.ride_height_factor(0.030), 1.0, 1e-9), "design");
        assert!(approx(a.ride_height_factor(0.0), 0.0, 1e-9), "ground");
        assert!(approx(a.ride_height_factor(0.010), 0.3, 1e-9), "stall");
        assert!(approx(a.ride_height_factor(0.060), 0.5, 1e-9), "high");
        assert!(
            approx(a.ride_height_factor(0.100), 0.5, 1e-9),
            "above high (capped)"
        );
    }

    #[test]
    fn ride_height_factor_monotonic_up_to_design() {
        let a = AeroModel::f1_default();
        // sample from just above 0 up to the design height; factor must increase
        let mut prev = a.ride_height_factor(1e-4);
        let mut h = 1e-3;
        while h <= 0.030 + 1e-12 {
            let f = a.ride_height_factor(h);
            assert!(
                f >= prev - 1e-9,
                "not monotonic at h={}: {} < {}",
                h,
                f,
                prev
            );
            prev = f;
            h += 1e-3;
        }
    }

    #[test]
    fn ride_height_factor_smooth_no_jumps() {
        let a = AeroModel::f1_default();
        // adjacent samples differ by a small amount (continuity across branches)
        let mut h = 0.001;
        let mut prev = a.ride_height_factor(h);
        while h <= 0.080 {
            h += 0.001;
            let f = a.ride_height_factor(h);
            assert!((f - prev).abs() < 0.1, "jump at h={}: {} vs {}", h, f, prev);
            prev = f;
        }
    }

    #[test]
    fn downforce_at_design() {
        let a = AeroModel::f1_default();
        let f = a.compute(100.0, 0.030, 0.030, 0.0);
        let q = 0.5 * 1.225 * 10000.0;
        let area = 1.5;
        assert!(
            approx(f.downforce_front, q * area * 1.575, 1e-6),
            "front {}",
            f.downforce_front
        );
        assert!(
            approx(f.downforce_rear, q * area * 1.925, 1e-6),
            "rear {}",
            f.downforce_rear
        );
        assert!(
            approx(f.downforce_total, q * area * 3.5, 1e-6),
            "total {}",
            f.downforce_total
        );
    }

    #[test]
    fn ride_height_sensitivity() {
        let a = AeroModel::f1_default();
        let design = a.compute(100.0, 0.030, 0.030, 0.0);
        let raised_rear = a.compute(100.0, 0.030, 0.050, 0.0);

        // front unchanged, rear reduced, total reduced
        assert!(approx(
            raised_rear.downforce_front,
            design.downforce_front,
            1e-9
        ));
        assert!(
            raised_rear.downforce_rear < design.downforce_rear,
            "rear reduced"
        );
        assert!(
            raised_rear.downforce_total < design.downforce_total,
            "total reduced"
        );
    }

    #[test]
    fn floor_stall() {
        let a = AeroModel::f1_default();
        let design = a.compute(100.0, 0.030, 0.030, 0.0);
        let stalled = a.compute(100.0, 0.005, 0.030, 0.0);

        // front (below stall) collapses to < 20% of design front
        assert!(
            stalled.downforce_front < 0.20 * design.downforce_front,
            "front {} should be < 20% of {}",
            stalled.downforce_front,
            design.downforce_front
        );
        // rear unaffected
        assert!(
            approx(stalled.downforce_rear, design.downforce_rear, 1e-9),
            "rear unaffected"
        );
    }

    #[test]
    fn drag_pitch_sensitivity() {
        let a = AeroModel::f1_default();
        let flat = a.compute(100.0, 0.030, 0.030, 0.0);
        let pitched = a.compute(100.0, 0.030, 0.030, 0.1);
        let q = 0.5 * 1.225 * 10000.0;
        assert!(
            approx(flat.drag, q * 1.5 * 0.9, 1e-6),
            "flat drag {}",
            flat.drag
        );
        // 0.5 * 0.1 = 5% more drag
        assert!(
            approx(pitched.drag / flat.drag, 1.05, 1e-9),
            "ratio {}",
            pitched.drag / flat.drag
        );
    }

    #[test]
    fn zero_speed_zero_forces() {
        let a = AeroModel::f1_default();
        let f = a.compute(0.0, 0.030, 0.030, 0.0);
        assert_eq!(f.drag, 0.0);
        assert_eq!(f.downforce_front, 0.0);
        assert_eq!(f.downforce_rear, 0.0);
        assert_eq!(f.downforce_total, 0.0);
    }

    #[test]
    fn consistency_with_car_params() {
        let a = AeroModel::f1_default();
        let car = CarParams::default();
        let f = a.compute(100.0, 0.030, 0.030, 0.0);
        let cp = car.downforce(100.0);
        assert!(
            (f.downforce_total - cp).abs() / cp < 0.01,
            "aero total {} vs CarParams {}",
            f.downforce_total,
            cp
        );
    }
}
