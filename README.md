# Apex-14

Minimum-time lap simulation and racing line optimization for open-wheel race cars.

Apex-14 computes optimal racing lines and lap times using nonlinear vehicle dynamics models coupled with direct collocation trajectory optimization. It supports four levels of model fidelity and includes an interactive telemetry viewer.

## Quick Start

```bash
cargo build --release
cargo run --release --bin simulate    # lap simulation with telemetry export
cargo run --release --bin optimize    # trajectory optimization
cargo run --release --bin compare     # model fidelity comparison
cargo run --release --bin viewer      # interactive telemetry viewer
```

## Features

**Vehicle Dynamics** - Four model fidelities sharing a common tire and track interface:

| Model | States | Use Case |
|-------|--------|----------|
| Point-mass | 4 | Fast lap estimation, optimizer prototyping |
| Single-track | 6 | Slip angle analysis, understeer characterization |
| Four-wheel | 10 | Combined-slip tire forces, per-corner load transfer |
| Full chassis | 24 | Suspension dynamics, ride-height aerodynamics, transient response |

**Trajectory Optimization** - Direct collocation minimizing lap time subject to vehicle dynamics, grip limits, and track boundaries. Trapezoidal and Hermite-Simpson transcriptions with automatic Jacobian computation via forward-mode dual-number differentiation.

**Tire Model** - Pacejka Magic Formula with combined slip, load sensitivity, and smooth (C1) friction saturation for gradient-based optimization.

**Track Import** - Native JSON format and TUMFTM racetrack database CSV import (25 real circuits including Silverstone, Monza, Spa, Barcelona).

**Interactive Viewer** - Real-time track map with speed-colored racing line, synchronized telemetry plots (speed, lateral/longitudinal g, curvature), and bidirectional cursor tracking.

**Calibrated Parameters** - Includes a 2024-era open-wheel preset calibrated against published performance data (320 km/h top speed, 2.5g cornering on medium-speed corners).

## Usage

### Lap Simulation

```bash
cargo run --release --bin simulate
```

Runs the quasi-steady-state lap simulator on oval, circle, Silverstone, and Monza circuits. Produces CSV telemetry and SVG track visualizations. Also runs a transient 14-DOF forward simulation with LQR steering and predictive speed control.

### Trajectory Optimization

```bash
cargo run --release --bin optimize
```

Solves the minimum-time optimal control problem using direct collocation. Compares augmented Lagrangian, Gauss-Newton, and direct correction solvers.

### Model Comparison

```bash
cargo run --release --bin compare
```

Runs all model fidelities on the same track with both default and calibrated car parameters. Outputs a comparison table showing the effect of tire model, load transfer, and ride-height aerodynamics on lap time.

### Interactive Viewer

```bash
cargo run --release --bin viewer
```

Opens a desktop application with a track map and telemetry plots. Select circuits from the dropdown, toggle boundary and racing line overlays, zoom and pan the track view, and hover to see speed at any point synchronized across all telemetry channels.

### Importing Real Tracks

```bash
# Clone the TUMFTM racetrack database (LGPL-3.0)
git clone https://github.com/TUMFTM/racetrack-database.git

# Load in code
let track = apex_track::load_tumftm_csv(Path::new("racetrack-database/tracks/Silverstone.csv"), "Silverstone")?;
```

See `tracks/README.md` for the JSON track format and import details.

## Project Structure

```
crates/
  apex-math          Linear algebra, dual numbers, sparse matrices
  apex-integrator    RK4 and adaptive RK45 ODE solvers
  apex-track         Track geometry, circuit generators, file import
  apex-physics       Tire models, vehicle dynamics (2/3/7/14-DOF), controllers
  apex-telemetry     CSV and SVG export
  apex-optimizer     Collocation NLP, solvers, mesh refinement
  apex-viewer        Interactive egui-based telemetry viewer

bins/
  simulate           Lap simulation and telemetry export
  optimize           Trajectory optimization
  compare            Model fidelity comparison
  viewer             Interactive viewer
```

## Documentation

Mathematical derivations and validation data are in the `docs/` directory:

- `docs/math/equations_of_motion.md` - Vehicle model derivations (point-mass through 14-DOF)
- `docs/math/pacejka.md` - Tire model theory and implementation
- `docs/math/collocation.md` - Optimal control transcription and solver architecture
- `docs/analysis.md` - Model fidelity comparison with quantitative results

## Build Requirements

- Rust stable toolchain (edition 2021)
- No external C/C++ dependencies

```bash
cargo test --workspace              # run all tests
cargo clippy -- -D warnings         # lint check
cargo bench                         # performance benchmarks
```

## References

- H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., Butterworth-Heinemann, 2012
- W. F. Milliken and D. L. Milliken, *Race Car Vehicle Dynamics*, SAE International, 1995
- J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed., SIAM, 2010
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006

## License

MIT
