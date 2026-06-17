//! Optimizer inner-loop benchmarks: the equality-constraint Jacobian assembly
//! (forward-mode auto-diff vs. finite differences) and a single Gauss-Newton
//! iteration. The Jacobian assembly is the dominant per-iteration cost.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use apex_optimizer::{CollocationConfig, CollocationOptimizer, GaussNewtonConfig};
use apex_physics::CarParams;
use apex_track::{build_track, circle_track};

/// Build a closed circle track and an optimizer with `n_nodes` nodes.
fn make_optimizer(n_nodes: usize) -> (apex_track::Track, CarParams, CollocationConfig) {
    let (pts, closed) = circle_track(100.0, 12.0, 200);
    let track = build_track("circle", &pts, closed);
    let car = CarParams::default();
    let config = CollocationConfig {
        n_nodes,
        closed: true,
        ..CollocationConfig::default()
    };
    (track, car, config)
}

/// Dense numerical Jacobian of the equality residuals via central differences.
fn numerical_jacobian(opt: &CollocationOptimizer, x: &[f64]) -> Vec<f64> {
    let n_vars = x.len();
    let n_eq = opt.equality_count();
    let mut jac = vec![0.0; n_eq * n_vars];
    let eps = 1e-7;
    let mut xp = x.to_vec();
    for j in 0..n_vars {
        let orig = xp[j];
        xp[j] = orig + eps;
        let fp = opt.equality_residuals(&xp);
        xp[j] = orig - eps;
        let fm = opt.equality_residuals(&xp);
        xp[j] = orig;
        for i in 0..n_eq {
            jac[i * n_vars + j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    jac
}

fn bench_optimizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer");

    // --- Jacobian assembly: auto-diff vs numerical (N = 50) ---
    let (track, car, config) = make_optimizer(50);
    let opt = CollocationOptimizer::new(config, &track, &car);
    let x = opt.warm_start();

    group.bench_function("equality_jacobian_autodiff_n50", |b| {
        b.iter(|| black_box(opt.equality_jacobian(black_box(&x))))
    });

    group.bench_function("equality_jacobian_numerical_n50", |b| {
        b.iter(|| black_box(numerical_jacobian(black_box(&opt), black_box(&x))))
    });

    // --- single Gauss-Newton iteration (N = 30) ---
    let (track30, car30, config30) = make_optimizer(30);
    let opt30 = CollocationOptimizer::new(config30, &track30, &car30);
    let gn_one = GaussNewtonConfig {
        max_iterations: 1,
        print_interval: 0,
        ..GaussNewtonConfig::default()
    };

    group.bench_function("gn_single_iteration_n30", |b| {
        b.iter(|| black_box(opt30.optimize_gn(black_box(&gn_one))))
    });

    group.finish();
}

criterion_group!(benches, bench_optimizer);
criterion_main!(benches);
