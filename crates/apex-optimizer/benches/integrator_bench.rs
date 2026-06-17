//! Integrator throughput benchmarks: fixed-step RK4 (2-DOF and 10-state) and
//! adaptive Dormand-Prince RK45. Verifies the "fast inner loop" claims.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use apex_integrator::{rk4_integrate, rk4_step, rk45_adaptive_step, AdaptiveConfig, OdeSystem};
use apex_physics::{CarParams, PointMassModel};

/// A 10-state system of coupled linear oscillators, standing in for the size of
/// the 7-DOF model (10 state variables). One dummy control input.
struct CoupledOscillators;

impl OdeSystem<10, 1> for CoupledOscillators {
    fn derivatives(&self, state: &[f64; 10], _control: &[f64; 1], _t: f64) -> [f64; 10] {
        // 5 oscillators (position, velocity pairs) with nearest-neighbour coupling.
        let mut d = [0.0; 10];
        for i in 0..5 {
            let x = state[2 * i];
            let v = state[2 * i + 1];
            let left = if i > 0 { state[2 * (i - 1)] } else { 0.0 };
            let right = if i < 4 { state[2 * (i + 1)] } else { 0.0 };
            d[2 * i] = v;
            d[2 * i + 1] = -2.0 * x + 0.3 * (left + right) - 0.05 * v;
        }
        d
    }
}

fn bench_integrator(c: &mut Criterion) {
    let car = CarParams::default();
    // Point mass on a straight at 50 m/s: state = [s, n, v, alpha].
    let pm = PointMassModel {
        params: &car,
        track_curvature: 0.0,
    };
    let pm_state = [0.0, 0.0, 50.0, 0.0];
    let pm_control = [2000.0, 0.0];

    let osc = CoupledOscillators;
    let osc_state = [0.5, 0.0, 0.3, 0.0, -0.2, 0.0, 0.1, 0.0, -0.4, 0.0];
    let osc_control = [0.0];

    let mut group = c.benchmark_group("integrator");

    group.bench_function("rk4_step_2dof", |b| {
        b.iter(|| {
            rk4_step(
                black_box(&pm),
                black_box(&pm_state),
                black_box(&pm_control),
                black_box(0.0),
                black_box(0.01),
            )
        })
    });

    group.bench_function("rk4_step_10state", |b| {
        b.iter(|| {
            rk4_step(
                black_box(&osc),
                black_box(&osc_state),
                black_box(&osc_control),
                black_box(0.0),
                black_box(0.01),
            )
        })
    });

    group.bench_function("rk4_integrate_2dof_1000steps", |b| {
        b.iter(|| {
            rk4_integrate(
                black_box(&pm),
                black_box(&pm_state),
                black_box(&pm_control),
                black_box(0.001),
                black_box(1000),
            )
        })
    });

    let cfg = AdaptiveConfig::default();
    group.bench_function("rk45_step_2dof", |b| {
        b.iter(|| {
            rk45_adaptive_step(
                black_box(&pm),
                black_box(&pm_state),
                black_box(&pm_control),
                black_box(0.0),
                black_box(0.01),
                black_box(&cfg),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_integrator);
criterion_main!(benches);
