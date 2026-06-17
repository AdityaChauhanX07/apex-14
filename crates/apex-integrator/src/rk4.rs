//! Fixed-step classical 4th-order Runge-Kutta (RK4) integration.

use apex_math::Float;

use crate::traits::{OdeSystem, OdeSystemGeneric};

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

/// Perform a single 4th-order Runge-Kutta step over any [`Float`] type.
///
/// Same algorithm as [`rk4_step`], but generic so it can carry [`Dual`]
/// values for sensitivity/Jacobian computation. Fixed-size arrays are built
/// with [`std::array::from_fn`] because `[T::zero(); N]` is not a const
/// expression for a generic `T`.
pub fn rk4_step_generic<T: Float, const N: usize, const M: usize>(
    system: &impl OdeSystemGeneric<T, N, M>,
    state: &[T; N],
    control: &[T; M],
    t: T,
    dt: T,
) -> [T; N] {
    let half_dt = dt * T::from_f64(0.5);
    let two = T::from_f64(2.0);
    let sixth_dt = dt / T::from_f64(6.0);

    let k1 = system.derivatives_generic(state, control, t);

    let tmp: [T; N] = std::array::from_fn(|i| state[i] + half_dt * k1[i]);
    let k2 = system.derivatives_generic(&tmp, control, t + half_dt);

    let tmp: [T; N] = std::array::from_fn(|i| state[i] + half_dt * k2[i]);
    let k3 = system.derivatives_generic(&tmp, control, t + half_dt);

    let tmp: [T; N] = std::array::from_fn(|i| state[i] + dt * k3[i]);
    let k4 = system.derivatives_generic(&tmp, control, t + dt);

    std::array::from_fn(|i| state[i] + sixth_dt * (k1[i] + two * k2[i] + two * k3[i] + k4[i]))
}

/// Integrate a generic system from `t = 0` for `n_steps` steps of size `dt`,
/// holding the control input constant.
pub fn rk4_integrate_generic<T: Float, const N: usize, const M: usize>(
    system: &impl OdeSystemGeneric<T, N, M>,
    initial_state: &[T; N],
    control: &[T; M],
    dt: T,
    n_steps: usize,
) -> [T; N] {
    let mut state: [T; N] = *initial_state;
    let mut t = T::zero();
    for _ in 0..n_steps {
        state = rk4_step_generic(system, &state, control, t, dt);
        t = t + dt;
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

    // --- generic / autodiff tests ---

    use apex_math::{Dual, Float};

    /// Generic exponential decay: dx/dt = -k·x.
    struct GenericExpDecay {
        k: f64,
    }

    impl<T: Float> OdeSystemGeneric<T, 1, 1> for GenericExpDecay {
        fn derivatives_generic(&self, state: &[T; 1], _control: &[T; 1], _t: T) -> [T; 1] {
            [state[0] * T::from_f64(-self.k)]
        }
    }

    /// Decay where the rate k arrives via the control vector (so it can be a Dual
    /// variable we differentiate against).
    struct DecayWithDualK;

    impl OdeSystemGeneric<Dual, 1, 1> for DecayWithDualK {
        fn derivatives_generic(
            &self,
            state: &[Dual; 1],
            control: &[Dual; 1],
            _t: Dual,
        ) -> [Dual; 1] {
            [state[0] * (-control[0])] // control[0] is k
        }
    }

    #[test]
    fn generic_f64_matches_concrete() {
        let concrete = ExpDecay { k: 1.0 };
        let generic = GenericExpDecay { k: 1.0 };

        let r_concrete = rk4_integrate(&concrete, &[1.0], &[0.0], 0.001, 1000);
        let r_generic = rk4_integrate_generic::<f64, 1, 1>(&generic, &[1.0], &[0.0], 0.001, 1000);

        assert!(
            (r_concrete[0] - r_generic[0]).abs() < 1e-12,
            "concrete {} vs generic {}",
            r_concrete[0],
            r_generic[0]
        );
    }

    #[test]
    fn dual_sensitivity_to_initial_condition() {
        // x(t) = x0·e^(-kt); dx(1)/dx0 = e^(-1).
        let system = GenericExpDecay { k: 1.0 };
        let initial = [Dual::variable(1.0)];
        let control = [Dual::constant(0.0)];
        let result =
            rk4_integrate_generic(&system, &initial, &control, Dual::constant(0.001), 1000);

        let expected = (-1.0_f64).exp();
        assert!(
            (result[0].real - expected).abs() < 1e-6,
            "value {}",
            result[0].real
        );
        assert!(
            (result[0].dual - expected).abs() < 1e-6,
            "dx/dx0 {}",
            result[0].dual
        );
    }

    #[test]
    fn dual_sensitivity_to_parameter() {
        // dx(t)/dk = -t·x0·e^(-kt); at t=1, x0=1, k=1 -> -e^(-1).
        let system = DecayWithDualK;
        let initial = [Dual::constant(1.0)];
        let control = [Dual::variable(1.0)]; // differentiate w.r.t. k
        let result =
            rk4_integrate_generic(&system, &initial, &control, Dual::constant(0.001), 1000);

        let expected = -(-1.0_f64).exp(); // ≈ -0.3679
        assert!(
            (result[0].real - (-1.0_f64).exp()).abs() < 1e-6,
            "value {}",
            result[0].real
        );
        assert!(
            (result[0].dual - expected).abs() < 1e-6,
            "dx/dk {}",
            result[0].dual
        );
    }
}
