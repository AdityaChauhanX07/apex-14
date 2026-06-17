//! Full 14-DOF chassis model: 6 chassis DOF (X, Y, Z, roll, pitch, yaw) plus
//! four wheel-spin states and four suspension-travel states (24 state variables).
//!
//! The model is anchored to a precomputed static-trim equilibrium (solved for a
//! reference speed at construction): the tire vertical loads come from the tire
//! radial stiffness acting on the deviation of suspension travel and chassis
//! corner height from that trim. This keeps the heave/pitch/roll balance
//! self-consistent with the ride-height-sensitive aero, so the car sits still at
//! the trim point and responds correctly to braking, cornering, and bumps.

use apex_integrator::OdeSystem;

use crate::aero::AeroModel;
use crate::car_params::{CarParams, GRAVITY};
use crate::suspension::SuspensionSystem;
use crate::tire::PacejkaTire;

/// Full 14-DOF vehicle model.
///
/// State vector (24 elements):
/// 0..6   X, Y, Z, phi(roll), theta(pitch), psi(yaw)
/// 6..12  vx, vy, vz, omega_x, omega_y, omega_z
/// 12..16 wheel spins: omega_fl, omega_fr, omega_rl, omega_rr
/// 16..20 suspension travel: z_s_fl, z_s_fr, z_s_rl, z_s_rr
/// 20..24 suspension velocity: dz_s_fl, dz_s_fr, dz_s_rl, dz_s_rr
///
/// Control vector (3 elements): delta, torque_drive, brake_pressure.
pub struct FourteenDofModel<'a> {
    /// Vehicle parameters.
    pub params: &'a CarParams,
    /// Tire model.
    pub tire: &'a PacejkaTire,
    /// Suspension system.
    pub suspension: &'a SuspensionSystem,
    /// Aerodynamics model.
    pub aero: &'a AeroModel,
    /// Reference equilibrium suspension travel `[fl, fr, rl, rr]`.
    z_s_eq: [f64; 4],
    /// Reference equilibrium tire vertical loads `[fl, fr, rl, rr]` (N).
    f_z_eq: [f64; 4],
}

impl<'a> FourteenDofModel<'a> {
    /// Create a model, solving the static-trim equilibrium for `reference_speed`.
    pub fn new(
        params: &'a CarParams,
        tire: &'a PacejkaTire,
        suspension: &'a SuspensionSystem,
        aero: &'a AeroModel,
        reference_speed: f64,
    ) -> Self {
        let (z_s_eq, f_z_eq) = solve_trim(params, suspension, aero, reference_speed);
        FourteenDofModel {
            params,
            tire,
            suspension,
            aero,
            z_s_eq,
            f_z_eq,
        }
    }

    /// Equilibrium suspension travel `[fl, fr, rl, rr]` at the reference speed.
    pub fn equilibrium_travel(&self) -> [f64; 4] {
        self.z_s_eq
    }

    /// Front/rear ride heights implied by the suspension travel in `state`.
    fn ride_heights(&self, state: &[f64; 24]) -> (f64, f64) {
        let front = self.aero.design_ride_height - 0.5 * (state[16] + state[17]);
        let rear = self.aero.design_ride_height - 0.5 * (state[18] + state[19]);
        (front, rear)
    }

    /// Aerodynamic forces for the current state (ride heights from suspension).
    pub fn aero_forces(&self, state: &[f64; 24]) -> crate::aero::AeroForces {
        let (front_rh, rear_rh) = self.ride_heights(state);
        self.aero
            .compute(state[6].max(1.0), front_rh, rear_rh, state[4])
    }

    /// Front/rear ride heights `(front, rear)` (m) implied by `state`.
    pub fn ride_heights_of(&self, state: &[f64; 24]) -> (f64, f64) {
        self.ride_heights(state)
    }

    /// Per-corner tire vertical loads `[fl, fr, rl, rr]` (N) for `state`,
    /// from the tire radial stiffness acting about the static trim. This is the
    /// same load the dynamics use internally, exposed for telemetry.
    pub fn tire_loads(&self, state: &[f64; 24]) -> [f64; 4] {
        let p = self.params;
        let h = p.cog_height;
        let lf = p.cog_to_front;
        let lr = p.cog_to_rear;
        let twf = p.track_width_front;
        let twr = p.track_width_rear;
        let k_tire = p.tire_radial_stiffness;
        let z_chassis = state[2];
        let phi = state[3];
        let theta = state[4];
        let z_s = [state[16], state[17], state[18], state[19]];
        let x_off = [lf, lf, -lr, -lr];
        let y_off = [twf / 2.0, -twf / 2.0, twr / 2.0, -twr / 2.0];
        std::array::from_fn(|i| {
            let d_chassis = (z_chassis - h) - x_off[i] * theta + y_off[i] * phi;
            let comp_change = -((z_s[i] - self.z_s_eq[i]) + d_chassis);
            (self.f_z_eq[i] + k_tire * comp_change).max(0.0)
        })
    }
}

/// Solve the symmetric static trim (front/rear suspension travel) at `speed`:
/// find compressions where the vertical force and pitch moment both vanish, with
/// ride-height-sensitive aero. Returns (z_s `[4]`, tire loads `[4]`).
fn solve_trim(
    params: &CarParams,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    speed: f64,
) -> ([f64; 4], [f64; 4]) {
    let m = params.mass;
    let mu = params.unsprung_mass;
    let g = GRAVITY;
    let lf = params.cog_to_front;
    let lr = params.cog_to_rear;

    // per-corner tire load as a function of front/rear compression
    let f_zf = |zf: f64| -suspension.front_left.spring_force(zf) + mu * g;
    let f_zr = |zr: f64| -suspension.rear_left.spring_force(zr) + mu * g;

    // residuals: G1 = vertical balance, G2 = pitch balance (moment about CoG)
    let residual = |zf: f64, zr: f64| -> (f64, f64) {
        let front_rh = aero.design_ride_height - zf;
        let rear_rh = aero.design_ride_height - zr;
        let af = aero.compute(speed, front_rh, rear_rh, 0.0);
        let ff = f_zf(zf);
        let fr = f_zr(zr);
        let g1 = 2.0 * ff + 2.0 * fr - m * g - af.downforce_total;
        let g2 = lf * 2.0 * ff - lr * 2.0 * fr;
        (g1, g2)
    };

    // initial guess from linear springs supporting the static axle loads
    let mut zf = (m * g * lr / (lf + lr) / 2.0) / suspension.front_left.spring_rate;
    let mut zr = (m * g * lf / (lf + lr) / 2.0) / suspension.rear_left.spring_rate;

    let eps = 1e-7;
    for _ in 0..40 {
        let (g1, g2) = residual(zf, zr);
        // 2x2 numerical Jacobian
        let (g1f, g2f) = residual(zf + eps, zr);
        let (g1r, g2r) = residual(zf, zr + eps);
        let j11 = (g1f - g1) / eps;
        let j21 = (g2f - g2) / eps;
        let j12 = (g1r - g1) / eps;
        let j22 = (g2r - g2) / eps;
        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-30 {
            break;
        }
        // solve J * dz = -g
        let dzf = -(j22 * g1 - j12 * g2) / det;
        let dzr = -(-j21 * g1 + j11 * g2) / det;
        zf += dzf;
        zr += dzr;
        if dzf.abs() + dzr.abs() < 1e-12 {
            break;
        }
    }

    let z = [zf, zf, zr, zr];
    let fz = [f_zf(zf), f_zf(zf), f_zr(zr), f_zr(zr)];
    (z, fz)
}

impl OdeSystem<24, 3> for FourteenDofModel<'_> {
    fn derivatives(&self, state: &[f64; 24], control: &[f64; 3], _t: f64) -> [f64; 24] {
        let p = self.params;

        // --- (a) unpack state ---
        let z_chassis = state[2];
        let phi = state[3];
        let theta = state[4];
        let psi = state[5];
        let vx = state[6];
        let vy = state[7];
        let vz = state[8];
        let omega_x = state[9];
        let omega_y = state[10];
        let omega_z = state[11];
        let omega_w = [state[12], state[13], state[14], state[15]];
        let z_s = [state[16], state[17], state[18], state[19]];
        let dz_s = [state[20], state[21], state[22], state[23]];

        let delta = control[0];
        let torque_drive = control[1];
        let brake_pressure = control[2];

        // --- (b) speed guard ---
        let vx_safe = vx.max(1.0);

        let m = p.mass;
        let mu = p.unsprung_mass;
        let g = GRAVITY;
        let h = p.cog_height;
        let r = p.wheel_radius;
        let iw = p.wheel_inertia;
        let lf = p.cog_to_front;
        let lr = p.cog_to_rear;
        let twf = p.track_width_front;
        let twr = p.track_width_rear;
        let k_tire = p.tire_radial_stiffness;

        // wheel layout [FL, FR, RL, RR]
        let x_off = [lf, lf, -lr, -lr];
        let y_off = [twf / 2.0, -twf / 2.0, twr / 2.0, -twr / 2.0];
        let is_front = [true, true, false, false];

        // --- (c) suspension forces ---
        let f_susp = self.suspension.forces(&z_s, &dz_s);

        // --- (d) tire vertical loads from radial stiffness about the trim ---
        // tire compression change = (suspension compresses) - (chassis corner rises)
        // Tire deflection rises when the wheel drops relative to the ground.
        // The wheel height is `w = z_s + chassis_corner - const`, so increasing
        // suspension compression or raising the chassis lifts the wheel and
        // unloads the tire: ΔF_z = -K_tire·((z_s - z_s_eq) + Δchassis_corner).
        // Chassis-corner vertical height: pitch theta (about +y) drops the front
        // (-x·theta), roll phi (about +x) raises the left (+y·phi).
        let mut f_z_tire = [0.0; 4];
        for i in 0..4 {
            let d_chassis = (z_chassis - h) - x_off[i] * theta + y_off[i] * phi;
            let comp_change = -((z_s[i] - self.z_s_eq[i]) + d_chassis);
            f_z_tire[i] = (self.f_z_eq[i] + k_tire * comp_change).max(0.0);
        }

        // --- aero (ride-height sensitive) ---
        let (front_rh, rear_rh) = self.ride_heights(state);
        let aero = self.aero.compute(vx_safe, front_rh, rear_rh, theta);

        // --- (e,f) tire slip forces, transformed to body frame ---
        let (cos_d, sin_d) = (delta.cos(), delta.sin());
        let mut fx_body = [0.0; 4];
        let mut fy_body = [0.0; 4];
        let mut fx_tire = [0.0; 4];
        for i in 0..4 {
            let v_hub_x = vx - y_off[i] * omega_z;
            let v_hub_y = vy + x_off[i] * omega_z;
            let (slip_angle, slip_ratio, ft_x, ft_y) = if is_front[i] {
                let v_tx = v_hub_x * cos_d + v_hub_y * sin_d;
                let v_ty = -v_hub_x * sin_d + v_hub_y * cos_d;
                let sa = -(v_ty / v_tx.abs().max(1.0)).atan();
                let sr = (omega_w[i] * r - v_tx) / v_tx.abs().max(1.0);
                let f = self.tire.combined_forces_smooth(sa, sr, f_z_tire[i]);
                (sa, sr, f.fx, f.fy)
            } else {
                let sa = -(v_hub_y / v_hub_x.abs().max(1.0)).atan();
                let sr = (omega_w[i] * r - v_hub_x) / v_hub_x.abs().max(1.0);
                let f = self.tire.combined_forces_smooth(sa, sr, f_z_tire[i]);
                (sa, sr, f.fx, f.fy)
            };
            let _ = (slip_angle, slip_ratio);
            fx_tire[i] = ft_x;
            if is_front[i] {
                fx_body[i] = ft_x * cos_d - ft_y * sin_d;
                fy_body[i] = ft_x * sin_d + ft_y * cos_d;
            } else {
                fx_body[i] = ft_x;
                fy_body[i] = ft_y;
            }
        }

        // --- (h) total forces ---
        let total_fx: f64 = fx_body.iter().sum::<f64>() - aero.drag - p.rolling_resistance_force();
        let total_fy: f64 = fy_body.iter().sum();
        let total_fz: f64 = f_z_tire.iter().sum();

        // --- (i) moments about CoG via r × F, r_i = (x_i, y_i, -h) ---
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut mz = 0.0;
        for i in 0..4 {
            mx += y_off[i] * f_z_tire[i] + h * fy_body[i];
            my += -h * fx_body[i] - x_off[i] * f_z_tire[i];
            mz += x_off[i] * fy_body[i] - y_off[i] * fx_body[i];
        }

        // --- (j) chassis translational derivatives ---
        let dvx = total_fx / m + vy * omega_z - vz * omega_y;
        let dvy = total_fy / m - vx_safe * omega_z + vz * omega_x;
        let dvz = (total_fz - m * g - aero.downforce_total) / m + vx_safe * omega_y - vy * omega_x;

        // --- (k) chassis rotational derivatives (Euler) ---
        let ixx = p.inertia_xx;
        let iyy = p.inertia_yy;
        let izz = p.yaw_inertia;
        let domega_x = (mx - (izz - iyy) * omega_y * omega_z) / ixx;
        let domega_y = (my - (ixx - izz) * omega_x * omega_z) / iyy;
        let domega_z = (mz - (iyy - ixx) * omega_x * omega_y) / izz;

        // --- (m) wheel spin derivatives ---
        let mut domega_w = [0.0; 4];
        for i in 0..4 {
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
            let t_brake = brake_pressure * p.max_brake_force * r * bias / 2.0 * omega_w[i].signum();
            domega_w[i] = (t_drive - t_brake - fx_tire[i] * r) / iw;
        }

        // --- (n) suspension derivatives ---
        // ddz_s = chassis-corner vertical accel - wheel (unsprung) vertical accel
        let mut ddz_s = [0.0; 4];
        for i in 0..4 {
            // second derivative of d_chassis = (z-h) - x·theta + y·phi
            let a_chassis_corner = dvz - x_off[i] * domega_y + y_off[i] * domega_x;
            let a_wheel = (f_z_tire[i] + f_susp[i] - mu * g) / mu;
            // z_s = w - chassis_corner + const, so d²z_s = a_wheel - a_chassis_corner
            ddz_s[i] = a_wheel - a_chassis_corner;
        }

        // --- (l) global position / attitude derivatives ---
        let dx = vx_safe * psi.cos() - vy * psi.sin();
        let dy = vx_safe * psi.sin() + vy * psi.cos();
        let dz = vz;
        let dphi = omega_x;
        let dtheta = omega_y;
        let dpsi = omega_z;

        [
            dx,
            dy,
            dz,
            dphi,
            dtheta,
            dpsi, // 0..6
            dvx,
            dvy,
            dvz,
            domega_x,
            domega_y,
            domega_z, // 6..12
            domega_w[0],
            domega_w[1],
            domega_w[2],
            domega_w[3], // 12..16
            dz_s[0],
            dz_s[1],
            dz_s[2],
            dz_s[3], // 16..20 (suspension travel rates)
            ddz_s[0],
            ddz_s[1],
            ddz_s[2],
            ddz_s[3], // 20..24
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::AeroModel;
    use crate::suspension::SuspensionSystem;
    use apex_integrator::rk4_step;

    struct Rig {
        params: CarParams,
        tire: PacejkaTire,
        susp: SuspensionSystem,
        aero: AeroModel,
    }

    fn rig() -> Rig {
        Rig {
            params: CarParams::default(),
            tire: PacejkaTire::f1_default(),
            susp: SuspensionSystem::f1_default(),
            aero: AeroModel::f1_default(),
        }
    }

    /// Build the static-equilibrium state at `speed`.
    fn equilibrium_state(model: &FourteenDofModel, speed: f64) -> [f64; 24] {
        let z = model.equilibrium_travel();
        let r = model.params.wheel_radius;
        let w = speed / r;
        let mut s = [0.0; 24];
        s[2] = model.params.cog_height; // Z at reference
        s[6] = speed; // vx
        s[12] = w;
        s[13] = w;
        s[14] = w;
        s[15] = w;
        s[16] = z[0];
        s[17] = z[1];
        s[18] = z[2];
        s[19] = z[3];
        s
    }

    #[test]
    fn static_equilibrium_is_quiescent() {
        let rg = rig();
        let model = FourteenDofModel::new(&rg.params, &rg.tire, &rg.susp, &rg.aero, 50.0);
        let s = equilibrium_state(&model, 50.0);
        let d = model.derivatives(&s, &[0.0, 0.0, 0.0], 0.0);

        assert!(d[8].abs() < 0.05, "dvz/dt {}", d[8]); // heave
        assert!(d[9].abs() < 0.05, "domega_x/dt {}", d[9]); // roll
        assert!(d[10].abs() < 0.05, "domega_y/dt {}", d[10]); // pitch
        for (k, dk) in d[20..24].iter().enumerate() {
            assert!(dk.abs() < 0.5, "susp accel {} = {}", k + 20, dk);
        }
        for (k, dk) in d[12..16].iter().enumerate() {
            assert!(dk.abs() < 1e-3, "wheel spin {} = {}", k + 12, dk);
        }
        assert!(d[6] < 0.0, "dvx/dt {} should be negative (drag)", d[6]);
    }

    #[test]
    fn braking_induces_pitch() {
        let rg = rig();
        let model = FourteenDofModel::new(&rg.params, &rg.tire, &rg.susp, &rg.aero, 50.0);
        let s = equilibrium_state(&model, 50.0);
        let control = [0.0, 0.0, 0.5];

        // Tire braking force (and thus the pitch couple) develops as the wheels
        // slow and slip builds, so integrate and check the pitch angle changes.
        let mut st = s;
        let mut max_pitch_rate = 0.0_f64;
        for _ in 0..2500 {
            st = rk4_step(&model, &st, &control, 0.0, 0.0002);
            max_pitch_rate = max_pitch_rate.max(st[10].abs());
        }
        assert!(
            max_pitch_rate > 1e-3,
            "pitch rate should become nonzero: {}",
            max_pitch_rate
        );
        assert!(
            (st[4] - s[4]).abs() > 1e-4,
            "pitch angle should change: {} -> {}",
            s[4],
            st[4]
        );
        for v in st.iter() {
            assert!(v.is_finite(), "state went non-finite");
        }
    }

    #[test]
    fn cornering_induces_roll() {
        let rg = rig();
        let model = FourteenDofModel::new(&rg.params, &rg.tire, &rg.susp, &rg.aero, 30.0);
        let s = equilibrium_state(&model, 30.0);
        let drive = (rg.params.drag_force(30.0) + rg.params.rolling_resistance_force())
            * rg.params.wheel_radius;
        let control = [0.02, drive, 0.0];

        let d = model.derivatives(&s, &control, 0.0);
        assert!(d[9].abs() > 1e-3, "roll accel {} should be nonzero", d[9]);

        let mut st = s;
        for _ in 0..3000 {
            st = rk4_step(&model, &st, &control, 0.0, 0.0002);
            for v in st.iter() {
                assert!(v.is_finite(), "non-finite during cornering");
            }
        }
        assert!(
            (st[3] - s[3]).abs() > 1e-4,
            "roll angle should change: {} -> {}",
            s[3],
            st[3]
        );
    }

    #[test]
    fn suspension_oscillation_damps() {
        let rg = rig();
        let model = FourteenDofModel::new(&rg.params, &rg.tire, &rg.susp, &rg.aero, 50.0);
        let mut s = equilibrium_state(&model, 50.0);
        let eq_fl = s[16];
        s[16] += 0.005; // displace FL by 5mm

        let initial_dev = (s[16] - eq_fl).abs();
        for _ in 0..5000 {
            s = rk4_step(&model, &s, &[0.0, 0.0, 0.0], 0.0, 0.0002);
        }
        let final_dev = (s[16] - eq_fl).abs();
        assert!(
            final_dev < initial_dev,
            "FL deviation should damp: {} -> {}",
            initial_dev,
            final_dev
        );
    }

    #[test]
    fn ride_height_sensitive_downforce() {
        let rg = rig();
        let model = FourteenDofModel::new(&rg.params, &rg.tire, &rg.susp, &rg.aero, 50.0);

        // Place all corners at the design ride height (z_s = 0 -> ride = design),
        // where ground effect is at its peak.
        let mut at_design = equilibrium_state(&model, 50.0);
        for z in &mut at_design[16..20] {
            *z = 0.0;
        }
        // Raise the car well above design (extend suspension 20mm): in the
        // above-design regime ground effect fades, so downforce must drop.
        let mut raised = at_design;
        for z in &mut raised[16..20] {
            *z -= 0.020;
        }

        let base = model.aero_forces(&at_design);
        let high = model.aero_forces(&raised);
        assert!(
            high.downforce_total < base.downforce_total,
            "above design, raising ride height should reduce downforce: {} vs {}",
            high.downforce_total,
            base.downforce_total
        );
    }

    #[test]
    fn straight_line_drag_matches_lower_fidelity() {
        let rg = rig();
        let model = FourteenDofModel::new(&rg.params, &rg.tire, &rg.susp, &rg.aero, 50.0);
        let s = equilibrium_state(&model, 50.0);
        let d = model.derivatives(&s, &[0.0, 0.0, 0.0], 0.0);

        // dvx ≈ -(drag + rolling)/m on a straight, like the simpler models
        let expected =
            -(rg.params.drag_force(50.0) + rg.params.rolling_resistance_force()) / rg.params.mass;
        // aero drag here uses the model's (slightly reduced) downforce-coupled Cd,
        // but on a straight the drag is the dominant term and should be close.
        assert!(
            (d[6] - expected).abs() < 1.0,
            "dvx {} vs expected {}",
            d[6],
            expected
        );
    }
}
