//! Adaptive step-size integration via the Dormand-Prince RK45 method.
//!
//! The Dormand-Prince pair computes an embedded 4th- and 5th-order Runge-Kutta
//! solution in a single 7-stage pass. The difference between the two estimates
//! is a local error estimate that drives a step-size controller: large error
//! shrinks the step, small error grows it, so smooth regions take big steps and
//! stiff/fast regions take small ones.

use crate::traits::OdeSystem;

/// Result of a single adaptive step.
#[derive(Debug, Clone)]
pub struct AdaptiveStepResult<const N: usize> {
    /// The 5th-order state computed for this step (keep it only when accepted).
    pub state: [f64; N],
    /// The time step that was actually used.
    pub dt_used: f64,
    /// The suggested time step for the next step.
    pub dt_next: f64,
    /// Whether the step was accepted (error within tolerance).
    pub accepted: bool,
    /// The estimated local error (tolerance-weighted RMS norm).
    pub error_estimate: f64,
}

/// Configuration for the adaptive integrator.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Absolute error tolerance per state variable.
    pub atol: f64,
    /// Relative error tolerance per state variable.
    pub rtol: f64,
    /// Minimum allowed time step (s).
    pub dt_min: f64,
    /// Maximum allowed time step (s).
    pub dt_max: f64,
    /// Safety factor for step size adjustment (typically 0.8-0.9).
    pub safety: f64,
    /// Maximum step size growth factor per step.
    pub max_growth: f64,
    /// Maximum step size shrink factor per step.
    pub max_shrink: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        AdaptiveConfig {
            atol: 1e-6,
            rtol: 1e-6,
            dt_min: 1e-8,
            dt_max: 0.1,
            safety: 0.9,
            max_growth: 5.0,
            max_shrink: 0.2,
        }
    }
}

/// Perform a single Dormand-Prince RK45 step with error estimation.
///
/// Returns `(y5th, y4th, error_norm)` where `error_norm` is the plain L2 norm
/// of the embedded error `y5th - y4th`. The 5th-order estimate `y5th` is the
/// one to propagate; `y4th` exists only to form the error estimate. Tolerance
/// weighting is applied by the caller (it needs `atol`/`rtol`).
pub fn rk45_step<const N: usize, const M: usize>(
    system: &impl OdeSystem<N, M>,
    state: &[f64; N],
    control: &[f64; M],
    t: f64,
    dt: f64,
) -> ([f64; N], [f64; N], f64) {
    // --- Dormand-Prince nodes ---
    const C2: f64 = 1.0 / 5.0;
    const C3: f64 = 3.0 / 10.0;
    const C4: f64 = 4.0 / 5.0;
    const C5: f64 = 8.0 / 9.0;
    // c6 = c7 = 1.0

    // --- stage coefficients (Butcher tableau, lower triangle) ---
    const A21: f64 = 1.0 / 5.0;

    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;

    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;

    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;

    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    // --- 5th-order weights (= 7th stage row, b7 = 0) ---
    const B1: f64 = 35.0 / 384.0;
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;

    // --- 4th-order weights (for error estimation) ---
    const BS1: f64 = 5179.0 / 57600.0;
    const BS3: f64 = 7571.0 / 16695.0;
    const BS4: f64 = 393.0 / 640.0;
    const BS5: f64 = -92097.0 / 339200.0;
    const BS6: f64 = 187.0 / 2100.0;
    const BS7: f64 = 1.0 / 40.0;

    let k1 = system.derivatives(state, control, t);

    let tmp: [f64; N] = std::array::from_fn(|i| state[i] + dt * A21 * k1[i]);
    let k2 = system.derivatives(&tmp, control, t + C2 * dt);

    let tmp: [f64; N] = std::array::from_fn(|i| state[i] + dt * (A31 * k1[i] + A32 * k2[i]));
    let k3 = system.derivatives(&tmp, control, t + C3 * dt);

    let tmp: [f64; N] =
        std::array::from_fn(|i| state[i] + dt * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]));
    let k4 = system.derivatives(&tmp, control, t + C4 * dt);

    let tmp: [f64; N] = std::array::from_fn(|i| {
        state[i] + dt * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i])
    });
    let k5 = system.derivatives(&tmp, control, t + C5 * dt);

    let tmp: [f64; N] = std::array::from_fn(|i| {
        state[i] + dt * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i])
    });
    let k6 = system.derivatives(&tmp, control, t + dt);

    // 5th-order solution (the 7th stage node sits at t + dt with input = y5th)
    let y5: [f64; N] = std::array::from_fn(|i| {
        state[i] + dt * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i])
    });

    // FSAL stage: k7 = f(t + dt, y5th), reused as next step's k1.
    let k7 = system.derivatives(&y5, control, t + dt);

    // 4th-order solution for the embedded error estimate.
    let y4: [f64; N] = std::array::from_fn(|i| {
        state[i]
            + dt * (BS1 * k1[i]
                + BS3 * k3[i]
                + BS4 * k4[i]
                + BS5 * k5[i]
                + BS6 * k6[i]
                + BS7 * k7[i])
    });

    let error_l2 = y5
        .iter()
        .zip(y4.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();

    (y5, y4, error_l2)
}

/// Tolerance-weighted RMS error norm used by the step-size controller:
/// `sqrt(mean((y5 - y4) / (atol + rtol·max(|y_old|, |y5|)))²)`.
fn weighted_error_norm<const N: usize>(
    y5: &[f64; N],
    y4: &[f64; N],
    y_old: &[f64; N],
    config: &AdaptiveConfig,
) -> f64 {
    let terms: [f64; N] = std::array::from_fn(|i| {
        let e = y5[i] - y4[i];
        let scale = config.atol + config.rtol * y_old[i].abs().max(y5[i].abs());
        let r = e / scale;
        r * r
    });
    (terms.iter().sum::<f64>() / N as f64).sqrt()
}

/// Perform one adaptive step: take an RK45 step of size `dt`, decide whether to
/// accept it against the tolerances, and suggest the next step size.
pub fn rk45_adaptive_step<const N: usize, const M: usize>(
    system: &impl OdeSystem<N, M>,
    state: &[f64; N],
    control: &[f64; M],
    t: f64,
    dt: f64,
    config: &AdaptiveConfig,
) -> AdaptiveStepResult<N> {
    let (y5, y4, _) = rk45_step(system, state, control, t, dt);
    let err = weighted_error_norm(&y5, &y4, state, config);
    let accepted = err <= 1.0;

    // Standard PI-free controller: factor = safety · err^(-1/5), clamped.
    let factor = if err > 1e-15 {
        config.safety * (1.0 / err).powf(0.2)
    } else {
        config.max_growth
    };
    let factor = factor.clamp(config.max_shrink, config.max_growth);
    let dt_next = (dt * factor).clamp(config.dt_min, config.dt_max);

    AdaptiveStepResult {
        state: y5,
        dt_used: dt,
        dt_next,
        accepted,
        error_estimate: err,
    }
}

/// Integrate from `t_start` to `t_end` using adaptive step sizing.
///
/// Returns `(t_final, state, accepted_steps, rejected_steps)`.
pub fn rk45_integrate<const N: usize, const M: usize>(
    system: &impl OdeSystem<N, M>,
    initial_state: &[f64; N],
    control: &[f64; M],
    t_start: f64,
    t_end: f64,
    config: &AdaptiveConfig,
) -> (f64, [f64; N], usize, usize) {
    let mut t = t_start;
    let mut state = *initial_state;
    let mut dt = config.dt_max.min((t_end - t_start) / 10.0).max(config.dt_min);

    let mut accepted_steps = 0;
    let mut rejected_steps = 0;

    while t < t_end - config.dt_min {
        // Clamp the final step so we land exactly on t_end.
        if t + dt > t_end {
            dt = t_end - t;
        }

        let result = rk45_adaptive_step(system, &state, control, t, dt, config);

        // Force acceptance at the floor so we cannot stall indefinitely.
        let at_floor = dt <= config.dt_min * (1.0 + 1e-9);
        if result.accepted || at_floor {
            t += dt;
            state = result.state;
            accepted_steps += 1;
        } else {
            rejected_steps += 1;
        }

        dt = result.dt_next;
    }

    (t, state, accepted_steps, rejected_steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rk4::rk4_integrate;
    use std::f64::consts::PI;

    /// dx/dt = -k·x  (1 state, dummy 1-element control).
    struct ExpDecay {
        k: f64,
    }

    impl OdeSystem<1, 1> for ExpDecay {
        fn derivatives(&self, state: &[f64; 1], _control: &[f64; 1], _t: f64) -> [f64; 1] {
            [-self.k * state[0]]
        }
    }

    /// Simple harmonic oscillator: state = [position, velocity].
    struct Harmonic {
        omega: f64,
    }

    impl OdeSystem<2, 1> for Harmonic {
        fn derivatives(&self, state: &[f64; 2], _control: &[f64; 1], _t: f64) -> [f64; 2] {
            [state[1], -self.omega * self.omega * state[0]]
        }
    }

    /// Damped suspension-like oscillator: x'' + c·x' + k·x = 0.
    struct DampedOsc {
        k: f64,
        c: f64,
    }

    impl OdeSystem<2, 1> for DampedOsc {
        fn derivatives(&self, state: &[f64; 2], _control: &[f64; 1], _t: f64) -> [f64; 2] {
            [state[1], -self.k * state[0] - self.c * state[1]]
        }
    }

    #[test]
    fn exponential_decay_accuracy() {
        let system = ExpDecay { k: 1.0 };
        let config = AdaptiveConfig::default();
        let (t_final, state, accepted, rejected) =
            rk45_integrate(&system, &[1.0], &[0.0], 0.0, 1.0, &config);

        let exact = (-1.0_f64).exp();
        assert!((t_final - 1.0).abs() < config.dt_min, "t_final {}", t_final);
        assert!(
            (state[0] - exact).abs() < 1e-6,
            "result {} not within 1e-6 of {}",
            state[0],
            exact
        );
        // Far fewer steps than fixed RK4 needs for comparable accuracy (~1000).
        assert!(
            accepted < 50,
            "took {} accepted steps, expected far fewer",
            accepted
        );
        assert!(rejected <= 2, "smooth problem should rarely reject: {}", rejected);
    }

    #[test]
    fn harmonic_oscillator_one_period() {
        let omega = 2.0 * PI; // period = 1 second
        let system = Harmonic { omega };
        let config = AdaptiveConfig {
            atol: 1e-10,
            rtol: 1e-10,
            ..AdaptiveConfig::default()
        };
        let (_, state, _, _) = rk45_integrate(&system, &[1.0, 0.0], &[0.0], 0.0, 1.0, &config);
        assert!(
            (state[0] - 1.0).abs() < 1e-6,
            "position {} did not return to 1.0",
            state[0]
        );
        assert!(
            state[1].abs() < 1e-5,
            "velocity {} did not return to 0.0",
            state[1]
        );
    }

    #[test]
    fn stiff_oscillator_damps_with_variable_steps() {
        let system = DampedOsc {
            k: 200000.0 / 15.0,
            c: 8000.0 / 15.0,
        };
        let config = AdaptiveConfig {
            dt_max: 0.02,
            dt_min: 1e-7,
            ..AdaptiveConfig::default()
        };

        // Drive the adaptive loop directly so we can observe the step sizes.
        let mut t = 0.0;
        let mut state: [f64; 2] = [0.005, 0.0];
        let initial = state[0].abs();
        let mut dt = config.dt_max.min(0.5 / 10.0).max(config.dt_min);
        let mut dts: Vec<f64> = Vec::new();

        while t < 0.5 - config.dt_min {
            if t + dt > 0.5 {
                dt = 0.5 - t;
            }
            let r = rk45_adaptive_step(&system, &state, &[0.0], t, dt, &config);
            let at_floor = dt <= config.dt_min * (1.0 + 1e-9);
            if r.accepted || at_floor {
                t += dt;
                state = r.state;
                dts.push(dt);
            }
            dt = r.dt_next;
        }

        // Solution is damped.
        assert!(
            state[0].abs() < initial,
            "should damp: |x_final| {} >= |x_initial| {}",
            state[0].abs(),
            initial
        );

        // Step size genuinely varies (small while resolving the fast mode,
        // larger once it has decayed).
        let dt_max_used = dts.iter().cloned().fold(0.0_f64, f64::max);
        let dt_min_used = dts.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            dt_max_used / dt_min_used > 1.5,
            "step size should vary: max {} min {}",
            dt_max_used,
            dt_min_used
        );
    }

    #[test]
    fn step_adaptation_rejects_rarely_on_smooth_problems() {
        // Smooth problem: very few rejections, few steps.
        let smooth = ExpDecay { k: 1.0 };
        let config = AdaptiveConfig::default();
        let (_, _, smooth_accepted, smooth_rejected) =
            rk45_integrate(&smooth, &[1.0], &[0.0], 0.0, 1.0, &config);
        assert!(smooth_rejected <= 2, "smooth rejected {}", smooth_rejected);

        // Stiff problem over the same span needs more accepted steps.
        let stiff = DampedOsc {
            k: 200000.0 / 15.0,
            c: 8000.0 / 15.0,
        };
        let stiff_config = AdaptiveConfig {
            dt_max: 0.02,
            dt_min: 1e-7,
            ..AdaptiveConfig::default()
        };
        let (_, _, stiff_accepted, _) =
            rk45_integrate(&stiff, &[0.005, 0.0], &[0.0], 0.0, 1.0, &stiff_config);
        assert!(
            stiff_accepted > smooth_accepted,
            "stiff {} should need more steps than smooth {}",
            stiff_accepted,
            smooth_accepted
        );
    }

    #[test]
    fn exact_endpoint() {
        let system = ExpDecay { k: 1.0 };
        let config = AdaptiveConfig::default();
        let (t_final, _, _, _) = rk45_integrate(&system, &[1.0], &[0.0], 0.0, 1.0, &config);
        assert!(
            (t_final - 1.0).abs() < config.dt_min,
            "t_final {} not within dt_min of 1.0",
            t_final
        );
    }

    #[test]
    fn consistency_with_rk4() {
        let system = ExpDecay { k: 1.0 };
        let config = AdaptiveConfig::default();
        let (_, adaptive, _, _) = rk45_integrate(&system, &[1.0], &[0.0], 0.0, 1.0, &config);

        let fixed = rk4_integrate(&ExpDecay { k: 1.0 }, &[1.0], &[0.0], 0.001, 1000);

        assert!(
            (adaptive[0] - fixed[0]).abs() < 1e-4,
            "adaptive {} vs RK4 {}",
            adaptive[0],
            fixed[0]
        );
    }
}
