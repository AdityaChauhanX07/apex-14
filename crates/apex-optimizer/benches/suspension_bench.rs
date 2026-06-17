//! Suspension and full-vehicle benchmarks: per-corner suspension forces, the
//! static-equilibrium Newton solve, and a single evaluation of the full 14-DOF
//! model derivatives (the most expensive per-step computation in the project).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use apex_integrator::OdeSystem;
use apex_physics::{AeroModel, CarParams, FourteenDofModel, PacejkaTire, SuspensionSystem};

fn bench_suspension(c: &mut Criterion) {
    let suspension = SuspensionSystem::f1_default();

    // Representative corner displacements (mm-scale) and velocities.
    let z = [0.015, 0.013, 0.018, 0.016];
    let dz = [0.05, -0.04, 0.03, -0.02];
    // Quarter-car-ish loads for the static-equilibrium solve.
    let loads = [2200.0, 2400.0, 2600.0, 2800.0];

    let mut group = c.benchmark_group("suspension");

    group.bench_function("forces", |b| {
        b.iter(|| suspension.forces(black_box(&z), black_box(&dz)))
    });

    group.bench_function("static_equilibrium", |b| {
        b.iter(|| suspension.static_equilibrium(black_box(&loads)))
    });

    group.finish();

    // --- full 14-DOF derivatives ---
    let car = CarParams::default();
    let tire = PacejkaTire::f1_default();
    let aero = AeroModel::f1_default();
    let model = FourteenDofModel::new(&car, &tire, &suspension, &aero, 50.0);

    // A representative cornering state near the static trim at 50 m/s.
    let z_eq = model.equilibrium_travel();
    let mut state = [0.0f64; 24];
    state[2] = aero.design_ride_height + car.cog_height; // chassis CoG height
    state[6] = 50.0; // vx
    state[11] = 0.3; // yaw rate (cornering)
    let wheel_omega = 50.0 / car.wheel_radius;
    for w in state.iter_mut().skip(12).take(4) {
        *w = wheel_omega;
    }
    state[16..20].copy_from_slice(&z_eq);
    let control = [0.04, 1500.0, 0.0]; // steer, drive torque, brake

    let mut group = c.benchmark_group("fourteen_dof");
    group.bench_function("derivatives", |b| {
        b.iter(|| model.derivatives(black_box(&state), black_box(&control), black_box(0.0)))
    });
    group.finish();
}

criterion_group!(benches, bench_suspension);
criterion_main!(benches);
