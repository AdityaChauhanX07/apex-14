# Apex-14

**A deterministic vehicle dynamics engine and minimum-time racing line optimizer, written from scratch in Rust.**

## What This Does

Apex-14 takes a racetrack definition and a mathematical model of an F1 car and computes the
theoretical minimum lap time along with the exact racing line required to achieve it. It couples a
multi-fidelity vehicle dynamics solver — ranging from a 2-DOF point mass up to a planned 14-DOF full
chassis — with a direct collocation trajectory optimizer. Everything is built from first principles:
no physics engine, no linear algebra crate, no off-the-shelf optimizer. The tire model, automatic
differentiation, sparse matrices, ODE integrator, and nonlinear program solver are all implemented in
this repository.

## Technical Highlights

- Pacejka Magic Formula tire model with combined slip (friction ellipse) and load sensitivity
- 4th-order Runge-Kutta integrator with fixed-size arrays and zero-allocation inner loops
- Forward-mode automatic differentiation engine using dual numbers
- Generic `Float` trait — the same physics code computes forces (`f64`) and exact Jacobians (`Dual`)
- Direct collocation with trapezoidal defects for trajectory optimization
- Gauss-Newton solver with a conjugate-gradient inner loop exploiting banded Jacobian sparsity
- Compressed Sparse Row (CSR) matrix implementation for efficient Jacobian assembly
- Quasi-steady-state (QSS) lap simulator used to warm-start the optimizer

## Vehicle Models

The dynamics layer is intentionally multi-fidelity: the same track and tire code drives every model,
so results can be validated at low fidelity before adding complexity.

| Model        | DOF  | State Variables                                  | Key Features                                              |
|--------------|------|--------------------------------------------------|----------------------------------------------------------|
| Point Mass   | 2    | `[s, n, v, alpha]`                               | Curvilinear coordinates, grip-circle limit               |
| Bicycle      | 3    | `[X, Y, psi, vx, vy, omega_z]`                  | Single-track, per-axle loads, understeer gradient        |
| Four-Wheel   | 7    | chassis (6) + four wheel speeds                  | Per-corner load transfer, combined-slip tire forces      |
| Full Chassis | 14   | chassis + suspension + ride-height states        | Suspension, ride-height-sensitive aero — *In Development* |

## Architecture

The workspace is a directed acyclic graph of focused crates. `apex-math` sits at the root with no
internal dependencies; everything else builds on it.

```
                 apex-math  (vectors, matrices, dual numbers, Float trait, CSR sparse)
                /    |    \
   apex-integrator  |   apex-track  (geometry, curvature, circuit generators)
        |           |    /    |
        |        apex-physics  (tire models, QSS, point-mass/bicycle/7-DOF)
        |          /   |    \
   apex-optimizer ---/    apex-telemetry  (CSV + SVG export)
   (collocation NLP, Gauss-Newton, augmented Lagrangian)
```

- `apex-physics` depends on `apex-math`, `apex-integrator`, and `apex-track`.
- `apex-telemetry` depends on `apex-math`, `apex-physics`, and `apex-track`.
- `apex-optimizer` depends on `apex-math`, `apex-integrator`, `apex-physics`, and `apex-track`.

## Sample Output

Run `cargo run --release --bin simulate` to generate QSS lap times and SVG track visualizations for
the oval, circle, Silverstone, and Monza circuits. Each track produces a CSV telemetry file and a
speed-colored SVG of the racing line.

Run `cargo run --release --bin optimize` to run the collocation optimizer on the oval and circle
tracks, comparing the augmented-Lagrangian and Gauss-Newton solvers side by side.

Representative results:

- Silverstone QSS lap time of approximately 68 s on the default car parameters.
- On the circle, the Gauss-Newton optimizer converges to an equality-constraint violation of
  `2.6e-6` (a dynamically consistent trajectory).
- Switching the collocation Jacobian from finite differences to forward-mode automatic
  differentiation gave a roughly 25x speedup of the optimization binary (~31 s to ~1.3 s).

## Build & Test

```sh
cargo build --release

cargo test --workspace          # 161 tests, zero warnings

cargo clippy -- -D warnings     # enforced zero-warning policy
```

## Project Structure

```
crates/
  apex-math          Vectors, 3x3 matrices, dual numbers, the Float trait, and CSR sparse matrices.
  apex-integrator    Generic fixed-step RK4 ODE integrator and the OdeSystem trait.
  apex-track         Track geometry: arc length, heading, curvature, queries, and circuit generators.
  apex-physics       Car parameters, Pacejka tire models, QSS lap simulator, and vehicle dynamics models.
  apex-telemetry     CSV telemetry export and standalone SVG racing-line rendering.
  apex-optimizer     NLP definition, collocation formulation, augmented-Lagrangian and Gauss-Newton solvers.

bins/
  simulate           Runs the QSS lap simulator across several circuits and exports CSV + SVG.
  optimize           Runs the collocation optimizer and compares solvers on the oval and circle.
```

The workspace is approximately 7,937 lines of Rust across six library crates and two binaries.

## Mathematical References

- H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed. Butterworth-Heinemann, 2012.
- W. F. Milliken and D. L. Milliken, *Race Car Vehicle Dynamics*. SAE International, 1995.
- J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*,
  2nd ed. SIAM, 2010.
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed. Springer, 2006.

## Roadmap

- 14-DOF full chassis model with suspension and ride-height-sensitive aerodynamics
- SQP solver upgrade for robust convergence on complex circuits
- Adaptive mesh refinement for the collocation discretization
- Real-time telemetry dashboard
