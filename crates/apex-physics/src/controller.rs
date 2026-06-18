//! Trajectory-tracking controllers: an LQR steering controller and a PID speed
//! controller.
//!
//! The steering controller linearizes the lateral path-tracking error dynamics
//! around the current speed and solves the continuous algebraic Riccati
//! equation (CARE) for the optimal feedback gains, adding a curvature
//! feedforward term. The speed controller is a straightforward PID with
//! anti-windup that maps a speed error onto drive torque or brake pressure.

use crate::CarParams;

// ---------------------------------------------------------------------------
// 4x4 dense matrix helpers
//
// The lateral error model has four states, which is too large for the shared
// `Mat3` type, so we use small fixed-size helpers here. They are private to the
// module and operate on `[[f64; 4]; 4]`.
// ---------------------------------------------------------------------------

type Mat4 = [[f64; 4]; 4];

/// Matrix product `a * b`.
fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[i][k] * b[k][j];
            }
            *cell = sum;
        }
    }
    out
}

/// Transpose.
fn mat_transpose(a: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[j][i] = v;
        }
    }
    out
}

/// Frobenius norm of `a - b`.
fn frob_diff(a: &Mat4, b: &Mat4) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = a[i][j] - b[i][j];
            sum += d * d;
        }
    }
    sum.sqrt()
}

/// Solve the continuous algebraic Riccati equation for a 4x4 system with a
/// single input:
///
/// ```text
/// A' P + P A - P B R^-1 B' P + Q = 0
/// ```
///
/// The solver discretizes the system with a small Euler step (`Ad = I + A·dt`,
/// `Bd = B·dt`, `Qd = Q·dt`, `Rd = R·dt`) and runs the discrete Riccati value
/// iteration to its fixed point. As `dt → 0` that fixed point solves the CARE.
/// With a single input the `(R + Bd'·P·Bd)` term is a scalar, so its inverse is
/// trivial.
///
/// Returns the solution `P` and the continuous-time optimal gain
/// `K = R^-1 B' P` (a row vector applied as `u = -K x`). Returns `None` if the
/// iteration fails to converge or diverges (e.g. an unstabilizable system).
pub fn solve_care_4x4(
    a: &[[f64; 4]; 4],
    b: &[[f64; 4]; 1],
    q: &[[f64; 4]; 4],
    r: f64,
) -> Option<([[f64; 4]; 4], [f64; 4])> {
    if r <= 0.0 {
        return None;
    }

    const DT: f64 = 0.01;
    const MAX_ITERS: usize = 200_000;
    // Relative tolerance: once P is large, the per-iteration change plateaus at
    // round-off proportional to ‖P‖, so an absolute threshold is unreachable.
    const TOL_REL: f64 = 1e-10;

    // Discretized system.
    let mut ad: Mat4 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            ad[i][j] = if i == j { 1.0 } else { 0.0 } + a[i][j] * DT;
        }
    }
    let bd = [b[0][0] * DT, b[0][1] * DT, b[0][2] * DT, b[0][3] * DT];
    let rd = r * DT;

    let ad_t = mat_transpose(&ad);

    // Initialize P with the (scaled) state cost.
    let mut p: Mat4 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            p[i][j] = q[i][j] * DT;
        }
    }

    for _ in 0..MAX_ITERS {
        // bt_p = Bd' P   (1x4)
        let mut bt_p = [0.0; 4];
        for (j, slot) in bt_p.iter_mut().enumerate() {
            let mut s = 0.0;
            for i in 0..4 {
                s += bd[i] * p[i][j];
            }
            *slot = s;
        }
        // scalar S = Rd + Bd' P Bd
        let mut s_scalar = rd;
        for j in 0..4 {
            s_scalar += bt_p[j] * bd[j];
        }
        if s_scalar.abs() < 1e-300 {
            return None;
        }
        // w = Bd' P Ad   (1x4)
        let mut w = [0.0; 4];
        for (j, slot) in w.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..4 {
                acc += bt_p[k] * ad[k][j];
            }
            *slot = acc;
        }

        // Ad' P Ad
        let ad_t_p = mat_mul(&ad_t, &p);
        let ad_t_p_ad = mat_mul(&ad_t_p, &ad);

        // P_new = Qd + Ad'PAd - (1/S) w' w
        let mut p_new: Mat4 = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                p_new[i][j] = q[i][j] * DT + ad_t_p_ad[i][j] - (w[i] * w[j]) / s_scalar;
            }
        }
        // Symmetrize to suppress numerical drift. The transpose-style access
        // (both [i][j] and [j][i]) doesn't map cleanly onto an iterator.
        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            for j in (i + 1)..4 {
                let avg = 0.5 * (p_new[i][j] + p_new[j][i]);
                p_new[i][j] = avg;
                p_new[j][i] = avg;
            }
        }

        // Bail out on numerical blow-up (unstabilizable system).
        for row in &p_new {
            for &v in row {
                if !v.is_finite() || v.abs() > 1e18 {
                    return None;
                }
            }
        }

        let delta = frob_diff(&p_new, &p);
        let scale = frob_diff(&p_new, &[[0.0; 4]; 4]) + 1.0;
        p = p_new;
        if delta < TOL_REL * scale {
            // Continuous-time gain K = R^-1 B' P.
            let mut k = [0.0; 4];
            for (j, slot) in k.iter_mut().enumerate() {
                let mut acc = 0.0;
                for i in 0..4 {
                    acc += b[0][i] * p[i][j];
                }
                *slot = acc / r;
            }
            return Some((p, k));
        }
    }

    None
}

/// LQR-based steering controller for trajectory tracking.
///
/// Linearizes the lateral dynamics around the current operating point
/// (speed, road curvature) and computes the optimal steering correction
/// to minimize a weighted combination of lateral error, heading error,
/// and steering effort.
#[derive(Debug, Clone)]
pub struct LqrController {
    /// Lateral error weight in the Q matrix.
    pub q_lateral: f64,
    /// Heading error weight in the Q matrix.
    pub q_heading: f64,
    /// Lateral rate weight.
    pub q_lateral_rate: f64,
    /// Heading rate weight.
    pub q_heading_rate: f64,
    /// Steering effort weight (R matrix, scalar).
    pub r_steering: f64,
    /// How far ahead on the track to read curvature for feedforward (m).
    pub preview_distance: f64,
}

impl Default for LqrController {
    fn default() -> Self {
        LqrController {
            q_lateral: 10.0,
            q_heading: 50.0,
            q_lateral_rate: 1.0,
            q_heading_rate: 5.0,
            // Steering-effort weight. Tuned for the F1-class control authority
            // (the dynamic-bicycle B matrix is large, ~300), so a small R would
            // make the optimal feedback saturate on any non-trivial error.
            r_steering: 1000.0,
            preview_distance: 20.0,
        }
    }
}

impl LqrController {
    /// Compute the steering angle for trajectory tracking.
    ///
    /// Arguments:
    /// - `params`: car parameters (wheelbase, mass, etc.)
    /// - `speed`: current longitudinal speed (m/s)
    /// - `lateral_error`: signed lateral offset from desired path (m, positive = left of path)
    /// - `heading_error`: heading error relative to path tangent (rad)
    /// - `lateral_error_rate`: rate of change of lateral error (m/s)
    /// - `heading_error_rate`: rate of change of heading error (rad/s)
    /// - `track_curvature`: curvature at current position (1/m)
    /// - `track_curvature_ahead`: curvature at preview distance ahead (1/m)
    ///
    /// Returns: steering angle (rad), clamped to +/- 0.5 rad.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_steering(
        &self,
        params: &CarParams,
        speed: f64,
        lateral_error: f64,
        heading_error: f64,
        lateral_error_rate: f64,
        heading_error_rate: f64,
        track_curvature: f64,
        track_curvature_ahead: f64,
    ) -> f64 {
        let v = speed.max(3.0); // minimum speed for linearization validity
        let l = params.wheelbase;
        let m = params.mass;
        let iz = params.yaw_inertia;
        let lf = params.cog_to_front;
        let lr = params.cog_to_rear;

        // Dynamic-bicycle lateral path-tracking error model (Rajamani, *Vehicle
        // Dynamics and Control*, Ch. 2). Unlike the purely kinematic form, the
        // lateral-rate state is independent here (tire forces drive it), so the
        // system is controllable by steering and the CARE has a finite
        // stabilizing solution.
        //
        //   x = [e_lat, e_heading, e_lat_rate, e_heading_rate]
        //
        // Per-axle cornering stiffness is estimated as C = k·μ·Fz, where k ≈
        // B·C from the Pacejka curve slope and Fz is the axle load at this
        // speed (static weight + downforce). This matches the cornering
        // stiffness used elsewhere in the tire model (C_α = B·C·D, D = μ·Fz).
        const CORNERING_COEFF: f64 = 16.0; // representative Pacejka B·C
        let (fz_front, fz_rear) = params.axle_loads(v, 0.0);
        let caf = CORNERING_COEFF * params.tire_mu * fz_front; // front axle
        let car = CORNERING_COEFF * params.tire_mu * fz_rear; // rear axle

        let a = [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                (caf + car) / m,
                -(caf + car) / (m * v),
                (-caf * lf + car * lr) / (m * v),
            ],
            [
                0.0,
                (caf * lf - car * lr) / iz,
                -(caf * lf - car * lr) / (iz * v),
                -(caf * lf * lf + car * lr * lr) / (iz * v),
            ],
        ];

        // Steering enters the lateral- and yaw-acceleration equations. Stored as
        // a 1-row array whose single row is the 4-element input vector.
        let b = [[0.0, 0.0, caf / m, caf * lf / iz]];

        let q = [
            [self.q_lateral, 0.0, 0.0, 0.0],
            [0.0, self.q_heading, 0.0, 0.0],
            [0.0, 0.0, self.q_lateral_rate, 0.0],
            [0.0, 0.0, 0.0, self.q_heading_rate],
        ];

        // Solve CARE for the optimal gains.
        let gains = match solve_care_4x4(&a, &b, &q, self.r_steering) {
            Some((_, k)) => k,
            None => {
                // Fallback to proportional control if CARE doesn't converge.
                let delta_fb = -0.5 * lateral_error - 2.0 * heading_error;
                let delta_ff = track_curvature * l;
                return (delta_ff + delta_fb).clamp(-0.5, 0.5);
            }
        };

        let state = [
            lateral_error,
            heading_error,
            lateral_error_rate,
            heading_error_rate,
        ];

        // LQR feedback: u = -K x.
        let delta_fb = -(gains[0] * state[0]
            + gains[1] * state[1]
            + gains[2] * state[2]
            + gains[3] * state[3]);

        // Feedforward: steer to match the (previewed) track curvature.
        let kappa_ff = track_curvature
            + self.preview_distance / v.max(10.0) * (track_curvature_ahead - track_curvature);
        let delta_ff = kappa_ff * l;

        // Total steering command, clamped to physical limits.
        (delta_ff + delta_fb).clamp(-0.5, 0.5)
    }
}

/// PID speed controller mapping a speed error onto drive torque or brake
/// pressure, with integral anti-windup.
#[derive(Debug, Clone)]
pub struct SpeedController {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Derivative gain.
    pub kd: f64,
    /// Clamp on the integral term (anti-windup).
    pub integral_limit: f64,
    integral: f64,
    prev_error: f64,
}

impl SpeedController {
    /// Create a controller with explicit gains.
    pub fn new(kp: f64, ki: f64, kd: f64, integral_limit: f64) -> Self {
        SpeedController {
            kp,
            ki,
            kd,
            integral_limit,
            integral: 0.0,
            prev_error: 0.0,
        }
    }

    /// A reasonable default tuning for the F1-class car parameters.
    pub fn f1_default() -> Self {
        SpeedController::new(500.0, 50.0, 100.0, 5000.0)
    }

    /// Compute drive torque and brake pressure from a speed error.
    ///
    /// Returns `(torque_drive, brake_pressure)`:
    /// - `torque_drive`: engine torque (N·m), 0 when braking
    /// - `brake_pressure`: 0.0-1.0, 0 when accelerating
    pub fn compute(&mut self, target_speed: f64, current_speed: f64, dt: f64) -> (f64, f64) {
        let error = target_speed - current_speed;

        // Update integral with anti-windup.
        self.integral += error * dt;
        self.integral = self
            .integral
            .clamp(-self.integral_limit, self.integral_limit);

        // Derivative.
        let derivative = if dt > 1e-10 {
            (error - self.prev_error) / dt
        } else {
            0.0
        };
        self.prev_error = error;

        // PID output (force-like quantity).
        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;

        if output > 0.0 {
            // Accelerate: convert to torque (clamp to a reasonable range).
            let torque = output.clamp(0.0, 10000.0);
            (torque, 0.0)
        } else {
            // Brake: convert to brake pressure (0-1 range).
            let brake = (-output / 30000.0).clamp(0.0, 1.0);
            // Bleed the integral to avoid windup during braking.
            self.integral *= 0.9;
            (0.0, brake)
        }
    }

    /// Reset the controller state (e.g. at lap start).
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, rel: f64, abs: f64) -> bool {
        (a - b).abs() <= abs + rel * b.abs()
    }

    #[test]
    fn care_double_integrator() {
        // Double integrator on states (0, 1) embedded in 4x4; states (2, 3)
        // are decoupled and heavily damped. Penalizing position only (Q on
        // state 0) is the classic case whose LQR gain is K = [1, sqrt(2)].
        let a = [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -10.0, 0.0],
            [0.0, 0.0, 0.0, -10.0],
        ];
        let b = [[0.0, 1.0, 0.0, 0.0]];
        let q = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];

        let (_, k) = solve_care_4x4(&a, &b, &q, 1.0).expect("CARE should converge");

        // Within 20% of the analytic double-integrator gains.
        assert!(approx(k[0], 1.0, 0.20, 0.0), "k0 = {}", k[0]);
        assert!(
            approx(k[1], std::f64::consts::SQRT_2, 0.20, 0.0),
            "k1 = {}",
            k[1]
        );
    }

    #[test]
    fn lqr_straight_line_corrects_lateral_error() {
        let params = CarParams::default();
        let ctrl = LqrController::default();
        // 1 m to the left of the path at 50 m/s, no other error.
        let delta = ctrl.compute_steering(&params, 50.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        // Steer right (negative) to correct a leftward offset.
        assert!(delta < 0.0, "expected negative steering, got {delta}");
        // And only a gentle nudge, not full lock, at this speed.
        assert!(
            delta.abs() < 0.4,
            "expected a small correction, got {delta}"
        );
    }

    #[test]
    fn lqr_curvature_feedforward() {
        let params = CarParams::default();
        let ctrl = LqrController::default();
        // Perfectly on the path, but the track curves left at kappa = 0.01.
        let delta = ctrl.compute_steering(&params, 50.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01);
        let expected = 0.01 * params.wheelbase; // ~0.036 rad
        assert!(
            approx(delta, expected, 0.05, 1e-6),
            "delta = {delta}, expected ~{expected}"
        );
    }

    #[test]
    fn lqr_heading_correction() {
        let params = CarParams::default();
        let ctrl = LqrController::default();
        // Pointed 0.1 rad off the path tangent, no lateral error, no curvature.
        let delta = ctrl.compute_steering(&params, 50.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0);
        // Must steer back against the heading error.
        assert!(delta < 0.0, "expected negative steering, got {delta}");
    }

    #[test]
    fn lqr_steering_clamped() {
        let params = CarParams::default();
        let ctrl = LqrController::default();
        // Absurd 100 m lateral error must still respect the steering limits.
        let delta = ctrl.compute_steering(&params, 50.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            (-0.5..=0.5).contains(&delta),
            "steering {delta} outside [-0.5, 0.5]"
        );
    }

    #[test]
    fn speed_controller_accelerates() {
        let mut pid = SpeedController::f1_default();
        let (torque, brake) = pid.compute(100.0, 80.0, 0.01);
        assert!(torque > 0.0, "expected positive torque, got {torque}");
        assert_eq!(brake, 0.0);
    }

    #[test]
    fn speed_controller_brakes() {
        let mut pid = SpeedController::f1_default();
        let (torque, brake) = pid.compute(50.0, 80.0, 0.01);
        assert_eq!(torque, 0.0);
        assert!(brake > 0.0, "expected positive brake, got {brake}");
    }

    #[test]
    fn speed_controller_at_target() {
        let mut pid = SpeedController::f1_default();
        // Prime the derivative term with a steady-state step.
        let _ = pid.compute(80.0, 80.0, 0.01);
        let (torque, brake) = pid.compute(80.0, 80.0, 0.01);
        // At target the command should be essentially nothing.
        assert!(torque < 1.0, "torque {torque} should be ~0");
        assert!(brake < 1e-3, "brake {brake} should be ~0");
    }

    #[test]
    fn speed_controller_reset_clears_state() {
        let mut pid = SpeedController::f1_default();
        let _ = pid.compute(100.0, 50.0, 0.1);
        pid.reset();
        // After reset, a zero-error step yields zero integral/derivative action.
        let (torque, brake) = pid.compute(80.0, 80.0, 0.01);
        assert_eq!(torque, 0.0);
        assert_eq!(brake, 0.0);
    }
}
