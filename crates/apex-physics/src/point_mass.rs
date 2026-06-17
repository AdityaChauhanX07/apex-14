//! The 2-DOF point-mass model in curvilinear (track-relative) coordinates.

use apex_integrator::OdeSystem;

use crate::car_params::CarParams;

/// Point-mass vehicle model with curvilinear state.
///
/// State vector:
/// - `state[0]` = `s`: distance along the track centerline (m)
/// - `state[1]` = `n`: lateral offset from centerline (m, positive = left)
/// - `state[2]` = `v`: speed magnitude (m/s)
/// - `state[3]` = `alpha`: heading relative to the track tangent (rad)
///
/// Control vector:
/// - `control[0]` = `f_drive`: net longitudinal force (N; positive accelerates)
/// - `control[1]` = `curvature_cmd`: commanded path curvature (1/m)
pub struct PointMassModel<'a> {
    /// Vehicle parameters.
    pub params: &'a CarParams,
    /// Track curvature κ(s) at the current position; set externally before each
    /// evaluation.
    pub track_curvature: f64,
}

impl OdeSystem<4, 2> for PointMassModel<'_> {
    fn derivatives(&self, state: &[f64; 4], control: &[f64; 2], _t: f64) -> [f64; 4] {
        let n = state[1];
        let v = state[2];
        let alpha = state[3];

        let f_drive = control[0];
        let curv_cmd = control[1];

        let kappa = self.track_curvature;

        // Longitudinal forces
        let f_drag = self.params.drag_force(v);
        let f_roll = self.params.rolling_resistance_force();

        // Guard against division by zero when v ≈ 0
        let v_safe = v.max(0.1);

        // Curvilinear equations of motion
        let ds_dt = v_safe * alpha.cos() / (1.0 - n * kappa);
        let dn_dt = v_safe * alpha.sin();
        let dv_dt = (f_drive - f_drag - f_roll) / self.params.mass;
        let dalpha_dt = curv_cmd * v_safe - kappa * ds_dt;

        [ds_dt, dn_dt, dv_dt, dalpha_dt]
    }
}

impl PointMassModel<'_> {
    /// Returns the grip utilization ratio for the given state and control.
    ///
    /// Values up to `1.0` are within the tire grip circle; values above `1.0`
    /// exceed it.
    pub fn grip_utilization(&self, state: &[f64; 4], control: &[f64; 2]) -> f64 {
        let v = state[2];
        let f_drive = control[0];
        let curv_cmd = control[1];

        let f_lon = f_drive;
        let f_lat = self.params.mass * v * v * curv_cmd;
        let f_grip_max = self.params.max_grip_force(v);

        (f_lon * f_lon + f_lat * f_lat).sqrt() / f_grip_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_integrator::rk4_integrate;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn straight_line_full_throttle_derivatives() {
        let params = CarParams::default();
        let model = PointMassModel {
            params: &params,
            track_curvature: 0.0,
        };

        let state = [0.0, 0.0, 50.0, 0.0];
        let control = [15000.0, 0.0];
        let d = model.derivatives(&state, &control, 0.0);

        assert!(approx(d[0], 50.0, 1e-9), "ds_dt = {}", d[0]);
        assert!(approx(d[1], 0.0, 1e-9), "dn_dt = {}", d[1]);

        let expected_dv =
            (15000.0 - params.drag_force(50.0) - params.rolling_resistance_force()) / params.mass;
        assert!(approx(d[2], expected_dv, 1e-9), "dv_dt = {}", d[2]);

        assert!(approx(d[3], 0.0, 1e-9), "dalpha_dt = {}", d[3]);
    }

    #[test]
    fn straight_line_integration_gains_speed() {
        let params = CarParams::default();
        let model = PointMassModel {
            params: &params,
            track_curvature: 0.0,
        };

        // start from rest (v = 0.1 to avoid singularity), full throttle, 5 s
        let initial = [0.0, 0.0, 0.1, 0.0];
        let control = [15000.0, 0.0];
        let final_state = rk4_integrate(&model, &initial, &control, 0.001, 5000);

        assert!(
            final_state[2] > 50.0,
            "speed only reached {}",
            final_state[2]
        );
        assert!(
            final_state[0] > 100.0,
            "distance only reached {}",
            final_state[0]
        );
        assert!(
            approx(final_state[1], 0.0, 1e-9),
            "n drifted to {}",
            final_state[1]
        );
        assert!(
            approx(final_state[3], 0.0, 1e-9),
            "alpha drifted to {}",
            final_state[3]
        );
    }

    #[test]
    fn grip_utilization_sanity() {
        let params = CarParams::default();
        let model = PointMassModel {
            params: &params,
            track_curvature: 0.0,
        };

        // 80 m/s, moderate throttle, turning
        let state = [0.0, 0.0, 80.0, 0.0];
        let control = [5000.0, 0.01];
        let u = model.grip_utilization(&state, &control);
        assert!(u > 0.0 && u < 2.0, "utilization {}", u);

        // at rest with no forces, utilization ~ 0
        let rest = [0.0, 0.0, 0.0, 0.0];
        let no_force = [0.0, 0.0];
        let u0 = model.grip_utilization(&rest, &no_force);
        assert!(approx(u0, 0.0, 1e-9), "rest utilization {}", u0);
    }

    #[test]
    fn approaches_terminal_velocity() {
        let params = CarParams::default();
        let model = PointMassModel {
            params: &params,
            track_curvature: 0.0,
        };

        // full throttle on a straight for 30 s
        let initial = [0.0, 0.0, 0.1, 0.0];
        let control = [15000.0, 0.0];
        let final_state = rk4_integrate(&model, &initial, &control, 0.001, 30000);

        // v_terminal: drag(v) + rolling = F_max
        // 0.5 ρ Cd A v² = F_max - F_roll
        let f_roll = params.rolling_resistance_force();
        let k = 0.5 * params.air_density * params.drag_coeff * params.frontal_area;
        let v_terminal = ((params.max_drive_force - f_roll) / k).sqrt();

        let rel_err = (final_state[2] - v_terminal).abs() / v_terminal;
        assert!(
            rel_err < 0.01,
            "speed {} vs terminal {} (rel err {})",
            final_state[2],
            v_terminal,
            rel_err
        );
    }
}
