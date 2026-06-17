//! Pacejka tire model benchmarks: pure lateral, combined slip (hard and smooth),
//! and the forward-mode auto-diff (`Dual`) cost relative to `f64`. Verifies the
//! "Dual ≤ 2.5× f64" claim from the project plan.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use apex_math::Dual;
use apex_physics::PacejkaTire;

fn bench_tire(c: &mut Criterion) {
    let tire = PacejkaTire::f1_default();

    // Representative operating point: 5° slip angle, 8% slip ratio, 4 kN load.
    let slip_angle = 0.0873_f64; // ~5 degrees
    let slip_ratio = 0.08_f64;
    let fz = 4000.0_f64;

    let mut group = c.benchmark_group("tire");

    group.bench_function("lateral_force_f64", |b| {
        b.iter(|| tire.lateral_force(black_box(slip_angle), black_box(fz)))
    });

    group.bench_function("combined_forces", |b| {
        b.iter(|| tire.combined_forces(black_box(slip_angle), black_box(slip_ratio), black_box(fz)))
    });

    group.bench_function("combined_forces_smooth", |b| {
        b.iter(|| {
            tire.combined_forces_smooth(black_box(slip_angle), black_box(slip_ratio), black_box(fz))
        })
    });

    group.bench_function("lateral_force_dual", |b| {
        b.iter(|| {
            tire.lateral_force_generic(
                black_box(Dual::variable(slip_angle)),
                black_box(Dual::constant(fz)),
            )
        })
    });

    // The same lateral-force evaluation over f64 for a direct cost comparison
    // (the generic path through `T: Float` instantiated at f64).
    group.bench_function("lateral_force_generic_f64", |b| {
        b.iter(|| tire.lateral_force_generic::<f64>(black_box(slip_angle), black_box(fz)))
    });

    group.finish();
}

criterion_group!(benches, bench_tire);
criterion_main!(benches);
