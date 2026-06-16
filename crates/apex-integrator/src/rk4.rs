//! Fixed-step classical 4th-order Runge-Kutta (RK4) integration.

use crate::traits::OdeSystem;

/// Perform a single 4th-order Runge-Kutta integration step.
///
/// Returns the new state after advancing by time step `dt`.
pub fn rk4_step<const N: usize, const M: usize>(
    system: &impl OdeSystem<N, M>,
    state: &[f64; N],
    control: &[f64; M],
    t: f64,
    dt: f64,
) -> [f64; N] {
    let half = dt / 2.0;

    // k1 = f(t, y)
    let k1 = system.derivatives(state, control, t);

    // k2 = f(t + dt/2, y + dt/2 * k1)
    let mut tmp = [0.0; N];
    for i in 0..N {
        tmp[i] = state[i] + half * k1[i];
    }
    let k2 = system.derivatives(&tmp, control, t + half);

    // k3 = f(t + dt/2, y + dt/2 * k2)
    for i in 0..N {
        tmp[i] = state[i] + half * k2[i];
    }
    let k3 = system.derivatives(&tmp, control, t + half);

    // k4 = f(t + dt, y + dt * k3)
    for i in 0..N {
        tmp[i] = state[i] + dt * k3[i];
    }
    let k4 = system.derivatives(&tmp, control, t + dt);

    // y_next = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    let sixth = dt / 6.0;
    let mut next = [0.0; N];
    for i in 0..N {
        next[i] = state[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    next
}

/// Integrate the system from `t = 0` for `n_steps` steps of size `dt`,
/// holding the control input constant.
///
/// Returns the final state.
pub fn rk4_integrate<const N: usize, const M: usize>(
    system: &impl OdeSystem<N, M>,
    initial_state: &[f64; N],
    control: &[f64; M],
    dt: f64,
    n_steps: usize,
) -> [f64; N] {
    let mut state = *initial_state;
    let mut t = 0.0;
    for _ in 0..n_steps {
        state = rk4_step(system, &state, control, t, dt);
        t += dt;
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// dx/dt = -k * x  (1 state, dummy 1-element control)
    struct ExpDecay {
        k: f64,
    }

    impl OdeSystem<1, 1> for ExpDecay {
        fn derivatives(&self, state: &[f64; 1], _control: &[f64; 1], _t: f64) -> [f64; 1] {
            [-self.k * state[0]]
        }
    }

    /// Simple harmonic oscillator: state = [position, velocity].
    /// dx/dt = v, dv/dt = -omega² * x
    struct Harmonic {
        omega: f64,
    }

    impl OdeSystem<2, 1> for Harmonic {
        fn derivatives(&self, state: &[f64; 2], _control: &[f64; 1], _t: f64) -> [f64; 2] {
            [state[1], -self.omega * self.omega * state[0]]
        }
    }

    #[test]
    fn exponential_decay_accuracy() {
        let system = ExpDecay { k: 1.0 };
        let result = rk4_integrate(&system, &[1.0], &[0.0], 0.001, 1000);
        let exact = (-1.0_f64).exp(); // e^-1 ≈ 0.36788
        assert!(
            (result[0] - exact).abs() < 1e-8,
            "result {} not within 1e-8 of {}",
            result[0],
            exact
        );
    }

    #[test]
    fn harmonic_oscillator_one_period() {
        let omega = 2.0 * PI; // period = 1 second
        let system = Harmonic { omega };
        // integrate for one full period: t = 1.0 with dt = 0.0001 -> 10000 steps
        let result = rk4_integrate(&system, &[1.0, 0.0], &[0.0], 0.0001, 10000);
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "position {} did not return to 1.0",
            result[0]
        );
        assert!(
            (result[1] - 0.0).abs() < 1e-6,
            "velocity {} did not return to 0.0",
            result[1]
        );
    }

    #[test]
    fn zero_dt_returns_original_state() {
        let system = Harmonic { omega: 3.0 };
        let state = [0.7, -1.2];
        let result = rk4_step(&system, &state, &[0.0], 0.0, 0.0);
        assert_eq!(result, state);
    }

    #[test]
    fn fourth_order_convergence() {
        let system = ExpDecay { k: 1.0 };
        let exact = (-1.0_f64).exp();

        // dt = 0.01 -> 100 steps to reach t = 1.0
        let coarse = rk4_integrate(&system, &[1.0], &[0.0], 0.01, 100);
        let err_coarse = (coarse[0] - exact).abs();

        // dt = 0.001 -> 1000 steps to reach t = 1.0
        let fine = rk4_integrate(&system, &[1.0], &[0.0], 0.001, 1000);
        let err_fine = (fine[0] - exact).abs();

        // 4th order: 10x smaller step -> ~10^4 smaller error
        let ratio = err_coarse / err_fine;
        assert!(
            (5000.0..=15000.0).contains(&ratio),
            "convergence ratio {} outside expected range",
            ratio
        );
    }
}
