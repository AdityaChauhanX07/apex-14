//! Operating-point trim benchmarks: a single `solve_operating_point` call and a
//! Rayon batch of 1000 points. The envelope sweep calls this thousands of times,
//! so these numbers set the sweep budget (recorded in
//! `docs/design/envelope-qss/trim-solver.md`).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use apex_physics::car_params::GRAVITY;
use apex_physics::{
    solve_operating_point, AeroModel, CarParams, OperatingPoint, PacejkaTire, SuspensionSystem,
};

fn rig() -> (CarParams, PacejkaTire, SuspensionSystem, AeroModel) {
    (
        CarParams::default(),
        PacejkaTire::f1_default(),
        SuspensionSystem::f1_default(),
        AeroModel::f1_default(),
    )
}

/// A deterministic spread of 1000 operating points over (v, a_x, a_y), fixed
/// g_z. Mimics a slice of an envelope grid.
fn grid_1000() -> Vec<OperatingPoint> {
    let mut pts = Vec::with_capacity(1000);
    for iv in 0..10 {
        for iax in 0..10 {
            for iay in 0..10 {
                let v = 10.0 + iv as f64 * 8.0; // 10..82 m/s
                let a_x = -15.0 + iax as f64 * 3.0; // -15..12 m/s²
                let a_y = iay as f64 * 3.0; // 0..27 m/s²
                pts.push(OperatingPoint {
                    v,
                    a_x,
                    a_y,
                    g_z: GRAVITY,
                });
            }
        }
    }
    pts
}

fn bench_trim(c: &mut Criterion) {
    let (car, tire, susp, aero) = rig();

    let mut group = c.benchmark_group("trim");

    // Single symmetric (delegated) solve — the straight-line fast path.
    group.bench_function("single_straight_line", |b| {
        let op = OperatingPoint {
            v: 60.0,
            ..Default::default()
        };
        b.iter(|| solve_operating_point(&car, &tire, &susp, black_box(&aero), black_box(op)))
    });

    // Single general (3-DOF Newton) solve — combined braking + cornering.
    group.bench_function("single_combined", |b| {
        let op = OperatingPoint {
            v: 55.0,
            a_x: -8.0,
            a_y: 20.0,
            g_z: GRAVITY,
        };
        b.iter(|| solve_operating_point(&car, &tire, &susp, black_box(&aero), black_box(op)))
    });

    let grid = grid_1000();

    // Batch of 1000, sequential.
    group.bench_function("batch_1000_sequential", |b| {
        b.iter(|| {
            grid.iter()
                .map(|&op| {
                    solve_operating_point(&car, &tire, &susp, &aero, op)
                        .unwrap()
                        .residual
                })
                .sum::<f64>()
        })
    });

    // Batch of 1000, Rayon-parallel.
    group.bench_function("batch_1000_rayon", |b| {
        b.iter(|| {
            grid.par_iter()
                .map(|&op| {
                    solve_operating_point(&car, &tire, &susp, &aero, op)
                        .unwrap()
                        .residual
                })
                .sum::<f64>()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_trim);
criterion_main!(benches);
