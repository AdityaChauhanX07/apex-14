//! Fixed-budget real-time integrator.
//!
//! Designed for 1kHz simulation loops where each frame must complete
//! within a wall-clock budget. Uses RK4 sub-steps for accuracy,
//! with an Euler fallback if the budget (number of steps, not wall time)
//! is exhausted.

use apex_integrator::{rk4_step, OdeSystem};

/// Fixed-budget integrator for real-time simulation.
///
/// Advances the simulation by `target_dt` each frame using up to
/// `max_substeps` RK4 sub-steps. If the budget is exhausted before
/// reaching `target_dt`, the remaining time is covered by a single
/// forward Euler step (less accurate but bounded).
///
/// For 1kHz operation: target_dt = 0.001s, max_substeps = 4,
/// substep_dt = 0.00025s (0.25ms each).
pub struct RealtimeIntegrator {
    /// Target simulation time step per frame (s).
    target_dt: f64,
    /// Maximum number of RK4 sub-steps per frame.
    max_substeps: usize,
    /// Time step for each RK4 sub-step (s).
    substep_dt: f64,
}

impl RealtimeIntegrator {
    /// Create a 1kHz real-time integrator.
    ///
    /// Uses 4 RK4 sub-steps of 0.25ms each to advance by 1ms per frame.
    pub fn new_1khz() -> Self {
        Self {
            target_dt: 0.001,
            max_substeps: 4,
            substep_dt: 0.00025,
        }
    }

    /// Create a custom real-time integrator.
    ///
    /// `target_dt` is the simulation time to advance per frame.
    /// `max_substeps` is the maximum number of RK4 sub-steps.
    /// `substep_dt` is computed as `target_dt / max_substeps`.
    ///
    /// When `max_substeps` is zero, `substep_dt` is set to `target_dt` (it is
    /// never used, since the RK4 loop performs no iterations and the whole
    /// frame is covered by the Euler fallback).
    pub fn new(target_dt: f64, max_substeps: usize) -> Self {
        let substep_dt = if max_substeps == 0 {
            target_dt
        } else {
            target_dt / max_substeps as f64
        };
        Self {
            target_dt,
            max_substeps,
            substep_dt,
        }
    }

    /// The target time step per frame (s).
    pub fn target_dt(&self) -> f64 {
        self.target_dt
    }

    /// The time step of each RK4 sub-step (s).
    pub fn substep_dt(&self) -> f64 {
        self.substep_dt
    }

    /// Advance the state by one real-time frame.
    ///
    /// Performs up to `max_substeps` RK4 steps of size `substep_dt`.
    /// If any time remains after the budget is exhausted, a single
    /// forward Euler step covers the remainder.
    ///
    /// `system` is any [`OdeSystem`] with `N` state variables and `M` control
    /// inputs; `t` is the current simulation time, `state` the current state,
    /// and `control` the (constant over the frame) control input. Returns the
    /// new state after advancing by `target_dt`.
    pub fn step<const N: usize, const M: usize, S: OdeSystem<N, M>>(
        &self,
        system: &S,
        t: f64,
        state: &[f64; N],
        control: &[f64; M],
    ) -> [f64; N] {
        let mut current = *state;
        let mut t_remaining = self.target_dt;
        let mut steps_taken = 0;

        // RK4 sub-steps until the budget is spent or the frame is covered.
        while t_remaining > 1e-10 && steps_taken < self.max_substeps {
            let dt = t_remaining.min(self.substep_dt);
            let current_t = t + (self.target_dt - t_remaining);
            current = rk4_step(system, &current, control, current_t, dt);
            t_remaining -= dt;
            steps_taken += 1;
        }

        // Forward Euler fallback for any remaining time.
        if t_remaining > 1e-10 {
            let current_t = t + (self.target_dt - t_remaining);
            let deriv = system.derivatives(&current, control, current_t);
            for i in 0..N {
                current[i] += deriv[i] * t_remaining;
            }
        }

        current
    }

    /// Returns the number of sub-steps that would be needed to exactly
    /// cover target_dt (always <= max_substeps for the RK4 portion).
    pub fn substeps_needed(&self) -> usize {
        self.max_substeps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_physics::{AeroModel, CarParams, FourteenDofModel, PacejkaTire, SuspensionSystem};

    /// dx/dt = -x (exponential decay). One state, one (unused) control.
    struct ExpDecay;

    impl OdeSystem<1, 1> for ExpDecay {
        fn derivatives(&self, state: &[f64; 1], _control: &[f64; 1], _t: f64) -> [f64; 1] {
            [-state[0]]
        }
    }

    /// Build the static-equilibrium state for the 14-DOF model at `speed`.
    fn equilibrium_state(model: &FourteenDofModel, speed: f64) -> [f64; 24] {
        let z = model.equilibrium_travel();
        let r = model.params.wheel_radius;
        let w = speed / r;
        let mut s = [0.0; 24];
        s[2] = model.params.cog_height;
        s[6] = speed;
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
    fn test_new_1khz() {
        let integ = RealtimeIntegrator::new_1khz();
        assert_eq!(integ.target_dt, 0.001);
        assert_eq!(integ.max_substeps, 4);
        assert_eq!(integ.substep_dt, 0.00025);
    }

    #[test]
    fn test_new_custom() {
        let integ = RealtimeIntegrator::new(0.01, 10);
        assert_eq!(integ.target_dt, 0.01);
        assert_eq!(integ.max_substeps, 10);
        assert!((integ.substep_dt - 0.001).abs() < 1e-15);
    }

    #[test]
    fn test_step_simple_ode() {
        let integ = RealtimeIntegrator::new_1khz();
        let system = ExpDecay;
        let result = integ.step(&system, 0.0, &[1.0], &[0.0]);
        let exact = (-0.001_f64).exp(); // ≈ 0.999
        assert!(
            (result[0] - exact).abs() < 1e-6,
            "result {} not within 1e-6 of {}",
            result[0],
            exact
        );
    }

    #[test]
    fn test_step_preserves_total_dt() {
        let integ = RealtimeIntegrator::new_1khz();
        let system = ExpDecay;
        let mut state = [1.0];
        let mut t = 0.0;
        for _ in 0..1000 {
            state = integ.step(&system, t, &state, &[0.0]);
            t += integ.target_dt();
        }
        let exact = (-1.0_f64).exp(); // ≈ 0.3679
        assert!(
            (state[0] - exact).abs() < 1e-4,
            "after 1s, value {} not within 1e-4 of {}",
            state[0],
            exact
        );
    }

    #[test]
    fn test_euler_fallback() {
        let system = ExpDecay;

        // (a) Pure Euler: max_substeps = 0 means the RK4 loop runs zero times
        // and the whole frame is a single forward-Euler step.
        // dx/dt = -x at x = 1, dt = 0.001 -> 1 - 1*0.001 = 0.999 exactly.
        let euler_only = RealtimeIntegrator::new(0.001, 0);
        let result = euler_only.step(&system, 0.0, &[1.0], &[0.0]);
        assert_eq!(result[0], 0.999);

        // (b) Budget exhausted mid-frame: 2 RK4 steps of 0.001s cover 0.002s,
        // then 0.001s remains and is Euler-stepped. Result stays finite and
        // close to the analytic solution exp(-0.003).
        let partial = RealtimeIntegrator {
            target_dt: 0.003,
            max_substeps: 2,
            substep_dt: 0.001,
        };
        let result = partial.step(&system, 0.0, &[1.0], &[0.0]);
        let exact = (-0.003_f64).exp();
        assert!(result[0].is_finite());
        assert!(
            (result[0] - exact).abs() < 1e-3,
            "partial-budget result {} not within 1e-3 of {}",
            result[0],
            exact
        );
    }

    #[test]
    fn test_with_14dof_model() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let susp = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();
        let model = FourteenDofModel::new(&params, &tire, &susp, &aero, 50.0);

        let state = equilibrium_state(&model, 50.0);
        let integ = RealtimeIntegrator::new_1khz();
        let result = integ.step(&model, 0.0, &state, &[0.0, 0.0, 0.0]);

        assert!(
            result.iter().all(|v| v.is_finite()),
            "14-DOF step produced non-finite state"
        );
    }

    #[test]
    fn test_1000_frames_stability() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let susp = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();
        let model = FourteenDofModel::new(&params, &tire, &susp, &aero, 50.0);

        let mut state = equilibrium_state(&model, 50.0);
        let integ = RealtimeIntegrator::new_1khz();
        let mut t = 0.0;
        for _ in 0..1000 {
            state = integ.step(&model, t, &state, &[0.0, 0.0, 0.0]);
            t += integ.target_dt();
        }
        assert!(
            state.iter().all(|v| v.is_finite()),
            "14-DOF simulation went non-finite over 1000 frames"
        );
    }

    #[test]
    fn test_deterministic() {
        let integ = RealtimeIntegrator::new_1khz();
        let system = ExpDecay;
        let a = integ.step(&system, 0.0, &[1.0], &[0.0]);
        let b = integ.step(&system, 0.0, &[1.0], &[0.0]);
        assert_eq!(a, b, "identical inputs must produce bit-identical output");
    }
}
