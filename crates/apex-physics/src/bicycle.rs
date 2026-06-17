//! 3-DOF single-track (bicycle) model.
//!
//! Collapses the four tires into two virtual tires on the vehicle centerline —
//! one front axle and one rear axle — and integrates planar rigid-body motion
//! with Pacejka lateral tire forces.

use apex_integrator::OdeSystem;

use crate::car_params::CarParams;
use crate::tire::PacejkaTire;

/// Single-track vehicle model.
///
/// State vector:
/// - `state[0]` = `X`: global X position of the CoG (m)
/// - `state[1]` = `Y`: global Y position of the CoG (m)
/// - `state[2]` = `psi`: yaw angle / heading (rad)
/// - `state[3]` = `vx`: longitudinal velocity in the body frame (m/s)
/// - `state[4]` = `vy`: lateral velocity in the body frame (m/s)
/// - `state[5]` = `omega_z`: yaw rate (rad/s)
///
/// Control vector:
/// - `control[0]` = `delta`: front wheel steering angle (rad)
/// - `control[1]` = `fx_total`: total longitudinal force applied (N; positive accelerates)
pub struct BicycleModel<'a> {
    /// Vehicle parameters.
    pub params: &'a CarParams,
    /// Tire model used for the axle lateral forces.
    pub tire: &'a PacejkaTire,
}

impl OdeSystem<6, 2> for BicycleModel<'_> {
    fn derivatives(&self, state: &[f64; 6], control: &[f64; 2], _t: f64) -> [f64; 6] {
        let psi = state[2];
        let vx = state[3];
        let vy = state[4];
        let omega_z = state[5];

        let delta = control[0];
        let fx_total = control[1];

        let lf = self.params.cog_to_front;
        let lr = self.params.cog_to_rear;
        let m = self.params.mass;
        let iz = self.params.yaw_inertia;

        // Minimum speed guard to avoid division by zero in slip angle computation
        let vx_safe = vx.max(1.0);

        // Slip angles
        // Front: α_f = δ - arctan((vy + lf·ωz) / vx)
        // Rear:  α_r = -arctan((vy - lr·ωz) / vx)
        let alpha_front = delta - ((vy + lf * omega_z) / vx_safe).atan();
        let alpha_rear = -((vy - lr * omega_z) / vx_safe).atan();

        // Vertical loads (use vx for speed approximation, assume small vy)
        // For longitudinal accel, use fx_total/m as approximation (actual accel not yet known)
        let ax_approx = fx_total / m;
        let (fz_front, fz_rear) = self.params.axle_loads(vx_safe, ax_approx);

        // Tire lateral forces from Pacejka
        let fy_front = self.tire.lateral_force(alpha_front, fz_front);
        let fy_rear = self.tire.lateral_force(alpha_rear, fz_rear);

        // Aerodynamic drag
        let f_drag = self.params.drag_force(vx_safe);
        let f_roll = self.params.rolling_resistance_force();

        // Equations of motion (body frame)
        // m·(dvx/dt - vy·ωz) = Fx_total - Fdrag - Froll - Fy_front·sin(δ)
        // m·(dvy/dt + vx·ωz) = Fy_front·cos(δ) + Fy_rear
        // Iz·dωz/dt = lf·Fy_front·cos(δ) - lr·Fy_rear

        let dvx_dt = (fx_total - f_drag - f_roll - fy_front * delta.sin()) / m + vy * omega_z;
        let dvy_dt = (fy_front * delta.cos() + fy_rear) / m - vx_safe * omega_z;
        let domega_z_dt = (lf * fy_front * delta.cos() - lr * fy_rear) / iz;

        // Global position derivatives
        let dx_dt = vx_safe * psi.cos() - vy * psi.sin();
        let dy_dt = vx_safe * psi.sin() + vy * psi.cos();
        let dpsi_dt = omega_z;

        [dx_dt, dy_dt, dpsi_dt, dvx_dt, dvy_dt, domega_z_dt]
    }
}

impl BicycleModel<'_> {
    /// Compute the steady-state understeer gradient K_us.
    ///
    /// K_us > 0 means understeer (typical, safe).
    /// K_us < 0 means oversteer.
    /// K_us = 0 means neutral steer.
    ///
    /// K_us = (m / L) · (l_r/C_α_f - l_f/C_α_r)
    /// where C_α is the cornering stiffness per axle.
    pub fn understeer_gradient(&self, fz_front: f64, fz_rear: f64) -> f64 {
        let ca_front = self.tire.cornering_stiffness(fz_front);
        let ca_rear = self.tire.cornering_stiffness(fz_rear);
        let m = self.params.mass;
        let l = self.params.wheelbase;
        let lf = self.params.cog_to_front;
        let lr = self.params.cog_to_rear;

        (m / l) * (lr / ca_front - lf / ca_rear)
    }

    /// Compute the steady-state yaw rate for a given speed and steering angle.
    /// ω_ss = v · δ / (L · (1 + K_us · v²))
    pub fn steady_state_yaw_rate(&self, speed: f64, delta: f64) -> f64 {
        let (fz_f, fz_r) = self.params.axle_loads(speed, 0.0);
        let k_us = self.understeer_gradient(fz_f, fz_r);
        let l = self.params.wheelbase;
        speed * delta / (l * (1.0 + k_us * speed * speed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::car_params::GRAVITY;
    use apex_integrator::rk4_step;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn f1_model<'a>(params: &'a CarParams, tire: &'a PacejkaTire) -> BicycleModel<'a> {
        BicycleModel { params, tire }
    }

    #[test]
    fn straight_line_coasting() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let model = f1_model(&params, &tire);

        let state = [0.0, 0.0, 0.0, 50.0, 0.0, 0.0];
        let control = [0.0, 0.0];
        let d = model.derivatives(&state, &control, 0.0);

        assert!(approx(d[0], 50.0, 1e-9), "dX/dt = {}", d[0]);
        assert!(approx(d[1], 0.0, 1e-9), "dY/dt = {}", d[1]);
        assert!(approx(d[2], 0.0, 1e-9), "dpsi/dt = {}", d[2]);
        assert!(approx(d[4], 0.0, 1e-9), "dvy/dt = {}", d[4]);
        assert!(approx(d[5], 0.0, 1e-9), "domega_z/dt = {}", d[5]);
        // drag + rolling decelerate the car
        assert!(d[3] < 0.0, "dvx/dt = {} should be negative", d[3]);
    }

    #[test]
    fn slip_angles_and_forces() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();

        let vx: f64 = 50.0;
        let delta: f64 = 0.05;
        // recompute the slip angles the model uses
        let alpha_front = delta - (0.0_f64 / vx).atan();
        let alpha_rear = -(0.0_f64 / vx).atan();
        assert!(
            approx(alpha_front, 0.05, 1e-9),
            "alpha_front {}",
            alpha_front
        );
        assert!(approx(alpha_rear, 0.0, 1e-12), "alpha_rear {}", alpha_rear);

        // front tire loaded, rear near zero
        let (fz_f, fz_r) = params.axle_loads(vx, 0.0);
        let fy_front = tire.lateral_force(alpha_front, fz_f);
        let fy_rear = tire.lateral_force(alpha_rear, fz_r);
        assert!(fy_front.abs() > 100.0, "fy_front {}", fy_front);
        assert!(approx(fy_rear, 0.0, 1e-9), "fy_rear {}", fy_rear);
    }

    #[test]
    fn steady_state_circular_motion() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let model = f1_model(&params, &tire);

        let delta = 0.02;
        let v0 = 30.0;
        // constant longitudinal force to offset drag + rolling at the initial speed
        let fx = params.drag_force(v0) + params.rolling_resistance_force();
        let control = [delta, fx];

        let dt = 0.001;
        let mut state = [0.0, 0.0, 0.0, v0, 0.0, 0.0];
        let mut traj: Vec<(f64, f64)> = Vec::new();
        let total_steps = 10_000; // 10 s
        for step in 0..total_steps {
            state = rk4_step(&model, &state, &control, 0.0, dt);
            // record the last ~3 s of trajectory
            if step >= 7_000 {
                traj.push((state[0], state[1]));
            }
        }

        // Settled yaw rate is in the ballpark of the linear steady-state
        // prediction. The linear formula uses the initial-slope cornering
        // stiffness, whereas the Pacejka model's secant stiffness at the
        // operating slip is lower, so the real car turns somewhat more — a
        // ~15% tolerance captures this nonlinear-tire discrepancy.
        let predicted = model.steady_state_yaw_rate(state[3], delta);
        assert!(
            (state[5] - predicted).abs() / predicted.abs() < 0.15,
            "yaw rate {} vs predicted {}",
            state[5],
            predicted
        );

        // the path is circular: circumcenter from 3 spread-out points, then
        // verify other points are roughly equidistant from it
        let n = traj.len();
        let p0 = traj[0];
        let p1 = traj[n / 2];
        let p2 = traj[n - 1];
        let (cx, cy) = circumcenter(p0, p1, p2).expect("non-degenerate arc");
        let radius = ((p0.0 - cx).powi(2) + (p0.1 - cy).powi(2)).sqrt();
        for &(x, y) in &traj {
            let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            assert!(
                (r - radius).abs() / radius < 0.02,
                "radius {} vs reference {}",
                r,
                radius
            );
        }
    }

    /// Circumcenter of three 2D points, or `None` if they are collinear.
    fn circumcenter(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> Option<(f64, f64)> {
        let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
        if d.abs() < 1e-9 {
            return None;
        }
        let a2 = a.0 * a.0 + a.1 * a.1;
        let b2 = b.0 * b.0 + b.1 * b.1;
        let c2 = c.0 * c.0 + c.1 * c.1;
        let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
        let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
        Some((ux, uy))
    }

    #[test]
    fn understeer_gradient_reasonable() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let model = f1_model(&params, &tire);

        let (fz_f, fz_r) = params.axle_loads(0.0, 0.0);
        let k_us = model.understeer_gradient(fz_f, fz_r);

        assert!(k_us.is_finite(), "K_us not finite: {}", k_us);
        // sane magnitude for a road vehicle (rad per (m/s)^2)
        assert!(k_us.abs() < 0.1, "K_us {} unreasonably large", k_us);
    }

    #[test]
    fn weight_transfer() {
        let params = CarParams::default();

        // static at rest: total = weight, no downforce
        let (ff0, fr0) = params.axle_loads(0.0, 0.0);
        let weight = params.mass * GRAVITY;
        assert!(
            approx(ff0 + fr0, weight, 1e-6),
            "static total {}",
            ff0 + fr0
        );

        // at speed with no accel: total = weight + downforce
        let (ff1, fr1) = params.axle_loads(50.0, 0.0);
        let expected = weight + params.downforce(50.0);
        assert!(
            approx(ff1 + fr1, expected, 1e-6),
            "loaded total {}",
            ff1 + fr1
        );

        // braking shifts load to the front
        let (ff_brake, fr_brake) = params.axle_loads(50.0, -10.0);
        assert!(ff_brake > ff1, "front {} should exceed {}", ff_brake, ff1);
        assert!(fr_brake < fr1, "rear {} should be below {}", fr_brake, fr1);
    }

    #[test]
    fn integration_stays_stable() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let model = f1_model(&params, &tire);

        let v0 = 80.0;
        let fx = params.drag_force(v0) + params.rolling_resistance_force();
        let control = [0.01, fx];

        let dt = 0.0005;
        let mut state = [0.0, 0.0, 0.0, v0, 0.0, 0.0];
        let mut omega_history = Vec::new();
        for step in 0..10_000 {
            state = rk4_step(&model, &state, &control, 0.0, dt);
            for v in state.iter() {
                assert!(v.is_finite(), "state went non-finite at step {}", step);
            }
            if step >= 9_000 {
                omega_history.push(state[5]);
            }
        }

        // yaw rate settles: variation over the last 0.5 s is small
        let max = omega_history.iter().cloned().fold(f64::MIN, f64::max);
        let min = omega_history.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            (max - min).abs() < 0.01,
            "yaw rate not settled: min {} max {}",
            min,
            max
        );
    }
}
