//! Per-fidelity dynamics RHS (state-derivative) cost.
//!
//! One `derivatives()` evaluation for each model fidelity — point-mass (4
//! states), single-track / bicycle (6), four-wheel / 7-DOF (10), and full
//! 14-DOF (24) — all on the SAME car (`CarParams::default`) and tire, at a
//! representative ~50 m/s cornering state. This is hot-path (a): the inner cost
//! that every integrator step and every collocation defect pays, isolated per
//! fidelity so the fidelity-vs-cost tradeoff is directly comparable.
//!
//! The 14-DOF derivative is also benched (from a different angle) in
//! suspension_bench.rs; here it sits alongside the lower fidelities for a
//! like-for-like comparison.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use apex_integrator::OdeSystem;
use apex_physics::{
    AeroModel, BicycleModel, CarParams, FourteenDofModel, PacejkaTire, PointMassModel,
    SevenDofModel, SuspensionSystem,
};

fn bench_dynamics_rhs(c: &mut Criterion) {
    let car = CarParams::default();
    let tire = PacejkaTire::f1_default();
    let suspension = SuspensionSystem::f1_default();
    let aero = AeroModel::f1_default();

    const V: f64 = 50.0; // shared reference speed (m/s)
    let wheel_omega = V / car.wheel_radius;

    let mut group = c.benchmark_group("dynamics_rhs");

    // --- point-mass (4 states, curvilinear) ---
    {
        let model = PointMassModel {
            params: &car,
            track_curvature: 0.01,
        };
        // [s, n, v, alpha]
        let state = [0.0, 0.0, V, 0.0];
        // [f_drive, curvature_cmd]
        let control = [1500.0, 0.01];
        group.bench_function("point_mass_4", |b| {
            b.iter(|| model.derivatives(black_box(&state), black_box(&control), black_box(0.0)))
        });
    }

    // --- single-track / bicycle (6 states) ---
    {
        let model = BicycleModel {
            params: &car,
            tire: &tire,
        };
        // [X, Y, psi, vx, vy, omega_z]
        let state = [0.0, 0.0, 0.0, V, 0.0, 0.1];
        // [delta, fx_total]
        let control = [0.03, 1500.0];
        group.bench_function("single_track_6", |b| {
            b.iter(|| model.derivatives(black_box(&state), black_box(&control), black_box(0.0)))
        });
    }

    // --- four-wheel / 7-DOF (10 states) ---
    {
        let model = SevenDofModel {
            params: &car,
            tire: &tire,
            roll_stiffness_front_fraction: 0.5,
        };
        // [X, Y, psi, vx, vy, omega_z, omega_fl, omega_fr, omega_rl, omega_rr]
        let state = [
            0.0,
            0.0,
            0.0,
            V,
            0.0,
            0.1,
            wheel_omega,
            wheel_omega,
            wheel_omega,
            wheel_omega,
        ];
        // [steer, drive_torque, brake_torque]
        let control = [0.03, 1500.0, 0.0];
        group.bench_function("four_wheel_10", |b| {
            b.iter(|| model.derivatives(black_box(&state), black_box(&control), black_box(0.0)))
        });
    }

    // --- full 14-DOF (24 states) ---
    {
        let model = FourteenDofModel::new(&car, &tire, &suspension, &aero, V);
        let z_eq = model.equilibrium_travel();
        let mut state = [0.0f64; 24];
        state[2] = aero.design_ride_height + car.cog_height; // chassis CoG height
        state[6] = V; // vx
        state[11] = 0.3; // yaw rate (cornering)
        for w in state.iter_mut().skip(12).take(4) {
            *w = wheel_omega;
        }
        state[16..20].copy_from_slice(&z_eq);
        let control = [0.04, 1500.0, 0.0]; // steer, drive torque, brake
        group.bench_function("fourteen_dof_24", |b| {
            b.iter(|| model.derivatives(black_box(&state), black_box(&control), black_box(0.0)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dynamics_rhs);
criterion_main!(benches);
