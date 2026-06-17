//! 7-DOF vehicle model: chassis planar motion (3 DOF) plus four independent
//! wheel spin states (4 DOF), with per-corner load transfer and combined-slip
//! Pacejka tire forces.

use apex_integrator::OdeSystem;

use crate::car_params::CarParams;
use crate::tire::PacejkaTire;

/// 7-DOF vehicle model.
///
/// State vector (10 elements):
/// - `state[0]` = `X`: global X position (m)
/// - `state[1]` = `Y`: global Y position (m)
/// - `state[2]` = `psi`: yaw angle (rad)
/// - `state[3]` = `vx`: body-frame longitudinal velocity (m/s)
/// - `state[4]` = `vy`: body-frame lateral velocity (m/s)
/// - `state[5]` = `omega_z`: yaw rate (rad/s)
/// - `state[6]` = `omega_fl`: front-left wheel angular velocity (rad/s)
/// - `state[7]` = `omega_fr`: front-right wheel angular velocity (rad/s)
/// - `state[8]` = `omega_rl`: rear-left wheel angular velocity (rad/s)
/// - `state[9]` = `omega_rr`: rear-right wheel angular velocity (rad/s)
///
/// Control vector (3 elements):
/// - `control[0]` = `delta`: front steering angle (rad)
/// - `control[1]` = `torque_drive`: total engine torque (N·m)
/// - `control[2]` = `brake_pressure`: brake pressure (0.0–1.0, scaled to `max_brake_force`)
pub struct SevenDofModel<'a> {
    /// Vehicle parameters.
    pub params: &'a CarParams,
    /// Tire model.
    pub tire: &'a PacejkaTire,
    /// Front roll-stiffness fraction `K_roll_f / (K_roll_f + K_roll_r)`.
    pub roll_stiffness_front_fraction: f64,
}

impl OdeSystem<10, 3> for SevenDofModel<'_> {
    fn derivatives(&self, state: &[f64; 10], control: &[f64; 3], _t: f64) -> [f64; 10] {
        let p = self.params;

        let psi = state[2];
        let vx = state[3];
        let vy = state[4];
        let omega_z = state[5];
        let omega_w = [state[6], state[7], state[8], state[9]];

        let delta = control[0];
        let torque_drive = control[1];
        let brake_pressure = control[2];

        let lf = p.cog_to_front;
        let lr = p.cog_to_rear;
        let twf = p.track_width_front;
        let twr = p.track_width_rear;
        let r = p.wheel_radius;
        let m = p.mass;
        let iz = p.yaw_inertia;
        let iw = p.wheel_inertia;

        let vx_safe = vx.max(1.0);

        // Vertical loads from longitudinal (approx 0) and lateral (centripetal) accel.
        let ax_approx = 0.0;
        let ay_approx = vx_safe * omega_z;
        let fz = p.corner_loads(
            vx_safe,
            ax_approx,
            ay_approx,
            self.roll_stiffness_front_fraction,
        );

        // Wheel layout: [FL, FR, RL, RR]
        let x_off = [lf, lf, -lr, -lr];
        let y_off = [twf / 2.0, -twf / 2.0, twr / 2.0, -twr / 2.0];
        let is_front = [true, true, false, false];

        let (cos_d, sin_d) = (delta.cos(), delta.sin());

        let mut total_fx = 0.0;
        let mut total_fy = 0.0;
        let mut total_mz = 0.0;
        let mut domega = [0.0; 4];

        for i in 0..4 {
            // hub velocity in body frame
            let v_hub_x = vx - y_off[i] * omega_z;
            let v_hub_y = vy + x_off[i] * omega_z;

            // tire-frame longitudinal force, plus body-frame force components
            let (fx_tire, fx_body, fy_body) = if is_front[i] {
                // rotate hub velocity into the steered wheel frame
                let v_tire_x = v_hub_x * cos_d + v_hub_y * sin_d;
                let v_tire_y = -v_hub_x * sin_d + v_hub_y * cos_d;
                let slip_angle = -(v_tire_y / v_tire_x.abs().max(1.0)).atan();
                let slip_ratio = (omega_w[i] * r - v_tire_x) / v_tire_x.abs().max(1.0);
                let f = self.tire.combined_forces(slip_angle, slip_ratio, fz[i]);
                let fx_body = f.fx * cos_d - f.fy * sin_d;
                let fy_body = f.fx * sin_d + f.fy * cos_d;
                (f.fx, fx_body, fy_body)
            } else {
                let slip_angle = -(v_hub_y / v_hub_x.abs().max(1.0)).atan();
                let slip_ratio = (omega_w[i] * r - v_hub_x) / v_hub_x.abs().max(1.0);
                let f = self.tire.combined_forces(slip_angle, slip_ratio, fz[i]);
                (f.fx, f.fx, f.fy)
            };

            total_fx += fx_body;
            total_fy += fy_body;
            total_mz += x_off[i] * fy_body - y_off[i] * fx_body;

            // wheel spin dynamics
            let t_drive = if is_front[i] {
                torque_drive * (1.0 - p.drive_distribution) / 2.0
            } else {
                torque_drive * p.drive_distribution / 2.0
            };
            let bias = if is_front[i] {
                p.brake_bias_front
            } else {
                1.0 - p.brake_bias_front
            };
            let t_brake_mag = brake_pressure * p.max_brake_force * r * bias / 2.0;
            // brake torque always opposes wheel spin
            let t_brake_signed = t_brake_mag * omega_w[i].signum();
            domega[i] = (t_drive - t_brake_signed - fx_tire * r) / iw;
        }

        // aerodynamic + rolling drag act along the body longitudinal axis
        total_fx -= p.drag_force(vx_safe) + p.rolling_resistance_force();

        // chassis derivatives
        let dvx = total_fx / m + vy * omega_z;
        let dvy = total_fy / m - vx_safe * omega_z;
        let domega_z = total_mz / iz;

        // global position
        let dx = vx_safe * psi.cos() - vy * psi.sin();
        let dy = vx_safe * psi.sin() + vy * psi.cos();
        let dpsi = omega_z;

        [
            dx, dy, dpsi, dvx, dvy, domega_z, domega[0], domega[1], domega[2], domega[3],
        ]
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

    fn model<'a>(params: &'a CarParams, tire: &'a PacejkaTire) -> SevenDofModel<'a> {
        SevenDofModel {
            params,
            tire,
            roll_stiffness_front_fraction: 0.55,
        }
    }

    /// State with the chassis at speed `vx` and all wheels rolling without slip.
    fn rolling_state(vx: f64, r: f64) -> [f64; 10] {
        let w = vx / r;
        [0.0, 0.0, 0.0, vx, 0.0, 0.0, w, w, w, w]
    }

    #[test]
    fn straight_line_coast() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let m = model(&params, &tire);

        let state = rolling_state(50.0, params.wheel_radius);
        let control = [0.0, 0.0, 0.0];
        let d = m.derivatives(&state, &control, 0.0);

        assert!(approx(d[0], 50.0, 1e-9), "dX/dt {}", d[0]);
        assert!(approx(d[1], 0.0, 1e-9), "dY/dt {}", d[1]);
        assert!(approx(d[2], 0.0, 1e-9), "dpsi/dt {}", d[2]);
        assert!(approx(d[4], 0.0, 1e-6), "dvy/dt {}", d[4]);
        assert!(approx(d[5], 0.0, 1e-6), "domega_z/dt {}", d[5]);
        // drag decelerates
        assert!(d[3] < 0.0, "dvx/dt {} should be negative", d[3]);
        // wheels in equilibrium (no net torque at zero slip)
        for (k, dk) in d[6..10].iter().enumerate() {
            assert!(approx(*dk, 0.0, 1e-6), "wheel {} spin deriv {}", k + 6, dk);
        }
    }

    #[test]
    fn straight_line_acceleration() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let m = model(&params, &tire);

        let state = rolling_state(30.0, params.wheel_radius);
        let control = [0.0, 5000.0, 0.0]; // drive torque, RWD
        let d = m.derivatives(&state, &control, 0.0);

        // rear wheels spin up from drive torque, fronts (no drive) stay ~0
        assert!(
            d[8] > 0.0,
            "rear-left spin deriv {} should be positive",
            d[8]
        );
        assert!(
            d[9] > 0.0,
            "rear-right spin deriv {} should be positive",
            d[9]
        );
        assert!(approx(d[6], 0.0, 1e-6), "front-left spin deriv {}", d[6]);
        assert!(approx(d[7], 0.0, 1e-6), "front-right spin deriv {}", d[7]);

        // integrate: the car should actually gain speed
        let mut s = state;
        for _ in 0..1000 {
            s = rk4_step(&m, &s, &control, 0.0, 0.0005);
        }
        assert!(s[3] > 30.0, "car did not accelerate: vx {}", s[3]);
        for v in s.iter() {
            assert!(v.is_finite(), "state went non-finite");
        }
    }

    #[test]
    fn braking() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let m = model(&params, &tire);

        let state = rolling_state(50.0, params.wheel_radius);
        let control = [0.0, 0.0, 0.5]; // half brake pressure
        let d = m.derivatives(&state, &control, 0.0);

        // every wheel decelerates under braking
        for (k, dk) in d[6..10].iter().enumerate() {
            assert!(
                *dk < 0.0,
                "wheel {} spin deriv {} should be negative",
                k + 6,
                dk
            );
        }
        // front brakes harder (60% bias) -> more negative
        assert!(
            d[6] < d[8],
            "front-left {} should brake harder than rear-left {}",
            d[6],
            d[8]
        );
        assert!(
            d[7] < d[9],
            "front-right {} should brake harder than rear-right {}",
            d[7],
            d[9]
        );

        // integrate: the car should slow significantly
        let mut s = state;
        for _ in 0..1000 {
            s = rk4_step(&m, &s, &control, 0.0, 0.0005);
        }
        assert!(s[3] < 45.0, "car did not brake: vx {}", s[3]);
    }

    #[test]
    fn cornering_is_stable() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let m = model(&params, &tire);

        // drive torque to roughly offset drag at 30 m/s
        let drive =
            (params.drag_force(30.0) + params.rolling_resistance_force()) * params.wheel_radius;
        let control = [0.02, drive, 0.0];

        let mut s = rolling_state(30.0, params.wheel_radius);
        for _ in 0..6000 {
            s = rk4_step(&m, &s, &control, 0.0, 0.0005);
            for v in s.iter() {
                assert!(v.is_finite(), "state went non-finite during cornering");
            }
        }
        // a left steer produces a positive yaw rate, and the car is turning
        assert!(s[5].abs() > 1e-3, "yaw rate {} should be nonzero", s[5]);
        assert!(
            s[5] > 0.0,
            "left steer should give positive yaw rate, got {}",
            s[5]
        );
    }

    #[test]
    fn corner_loads_symmetry() {
        let params = CarParams::default();
        let rsf = 0.55;

        // no lateral accel -> left/right symmetric
        let flat = params.corner_loads(50.0, 0.0, 0.0, rsf);
        assert!(
            approx(flat[0], flat[1], 1e-9),
            "FL {} FR {}",
            flat[0],
            flat[1]
        );
        assert!(
            approx(flat[2], flat[3], 1e-9),
            "RL {} RR {}",
            flat[2],
            flat[3]
        );

        // total equals weight + downforce
        let total: f64 = flat.iter().sum();
        let expected = params.mass * GRAVITY + params.downforce(50.0);
        assert!(
            approx(total, expected, 1e-6),
            "total {} vs {}",
            total,
            expected
        );

        // left turn (positive a_y) -> right wheels gain load
        let turning = params.corner_loads(50.0, 0.0, 15.0, rsf);
        assert!(
            turning[1] > turning[0],
            "FR {} should exceed FL {}",
            turning[1],
            turning[0]
        );
        assert!(
            turning[3] > turning[2],
            "RR {} should exceed RL {}",
            turning[3],
            turning[2]
        );
        let total_turn: f64 = turning.iter().sum();
        assert!(
            approx(total_turn, expected, 1e-6),
            "turning total {}",
            total_turn
        );
    }

    #[test]
    fn wheel_spin_equilibrium() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let m = model(&params, &tire);

        let mut s = rolling_state(50.0, params.wheel_radius);
        let control = [0.0, 0.0, 0.0];
        for _ in 0..1000 {
            s = rk4_step(&m, &s, &control, 0.0, 0.0005);
        }
        // wheels still roughly match ground speed (no runaway spin or lock)
        let r = params.wheel_radius;
        for k in 6..10 {
            let surface_speed = s[k] * r;
            assert!(
                (surface_speed - s[3]).abs() < 2.0,
                "wheel {} surface speed {} vs vx {}",
                k,
                surface_speed,
                s[3]
            );
        }
    }
}
