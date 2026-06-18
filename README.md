# Apex-14

Minimum-time lap simulation and racing line optimization for open-wheel race cars.

Apex-14 computes optimal racing lines and lap times using nonlinear vehicle dynamics models coupled with direct collocation trajectory optimization. It supports four levels of model fidelity, a tire fitting pipeline, thermal degradation modeling, and an interactive telemetry viewer.

## Quick Start

```bash
cargo build --release
apex-14 qss --calibrated                          # lap simulation on default oval
apex-14 qss --track tracks/test_circle.json        # custom track
apex-14 optimize --calibrated --hermite-simpson     # trajectory optimization
apex-14 car-info --calibrated                       # show car parameters
apex-14 tracks                                      # list available tracks
```

Or use the standalone binaries directly:

```bash
cargo run --release --bin simulate    # full lap simulation suite
cargo run --release --bin optimize    # optimizer comparison
cargo run --release --bin compare     # model fidelity comparison
cargo run --release --bin viewer      # interactive telemetry viewer
cargo run --release --bin validate    # validation against published data
```

## Features

**Vehicle Dynamics** - four model fidelities sharing a common tire and track interface:

| Model | States | Use Case |
|-------|--------|----------|
| Point-mass | 4 | Fast lap estimation, optimizer prototyping |
| Single-track | 6 | Slip angle analysis, understeer characterization |
| Four-wheel | 10 | Combined-slip tire forces, per-corner load transfer |
| Full chassis | 24 | Suspension dynamics, ride-height aerodynamics, transient response |

**Trajectory Optimization** - direct collocation minimizing lap time subject to vehicle dynamics, grip limits, and track boundaries. Trapezoidal and Hermite-Simpson transcriptions with automatic Jacobian computation via forward-mode dual-number differentiation. Progressive mesh refinement with coarse-to-fine solution interpolation.

**Tire Model** - Pacejka Magic Formula with combined slip, load sensitivity, and smooth (C1) friction saturation. Includes a Levenberg-Marquardt coefficient fitter for fitting Pacejka parameters to raw tire test data.

**Thermal Tire Model** - temperature-dependent grip with surface and carcass heat transfer, three compound presets (soft, medium, hard), and cumulative wear degradation over race stints.

**Powertrain** - engine torque curve with RPM-dependent power delivery, 8-speed sequential gearbox with automatic gear selection, and drivetrain efficiency modeling.

**Controllers** - LQR steering controller with curvature feedforward and preview, PID speed controller with predictive braking and traction limiting.

**Track Import** - native JSON format and TUMFTM racetrack database CSV import (25 real circuits including Silverstone, Monza, Spa, Barcelona).

**Car Configuration** - TOML-based car parameter files with overlay semantics. Missing fields fall back to a base preset, so partial configs work.

**Interactive Viewer** - real-time track map with speed-colored racing line, synchronized telemetry plots (speed, lateral/longitudinal g, curvature), and bidirectional cursor tracking.

**Calibrated Parameters** - 2024-era open-wheel preset validated against published Silverstone data. Top speed within 5% of published values.

## Usage

### Unified CLI

```bash
# Lap simulation
apex-14 qss --track tracks/test_circle.json --car cars/f1_2024_calibrated.toml
apex-14 qss --track tracks/test_circle.json --csv telemetry.csv --svg track.svg

# Trajectory optimization
apex-14 optimize --track tracks/test_circle.json --nodes 80 --hermite-simpson --calibrated

# Import real track data
apex-14 import-track -i Silverstone.csv -o tracks/silverstone.json -n Silverstone

# Car parameter management
apex-14 car-info --calibrated
apex-14 car-info --car cars/f3_car.toml --export my_car.toml
```

### Importing Real Tracks

```bash
git clone https://github.com/TUMFTM/racetrack-database.git
apex-14 import-track -i racetrack-database/tracks/Silverstone.csv -o tracks/silverstone.json -n Silverstone
apex-14 qss --track tracks/silverstone.json --calibrated
```

See `tracks/README.md` for the track file format.

### Car Configuration

Define cars in TOML. All fields are optional - missing fields use the base preset.

```toml
[car]
name = "My Car"
mass = 798.0

[aero]
drag_coeff = 1.10
lift_coeff = 2.80

[tires]
mu = 1.55
```

See `cars/README.md` for the full schema and sample files.

## Project Structure

```
crates/
  apex-math          Linear algebra, dual numbers, sparse matrices
  apex-integrator    RK4 and adaptive RK45 ODE solvers
  apex-track         Track geometry, circuit generators, file import
  apex-physics       Tire models, thermal model, powertrain, vehicle dynamics, controllers
  apex-telemetry     CSV and SVG export
  apex-optimizer     Collocation NLP, solvers, mesh refinement
  apex-viewer        Interactive egui-based telemetry viewer

bins/
  apex-cli           Unified CLI (installed as apex-14)
  simulate           Lap simulation and telemetry export
  optimize           Trajectory optimization with solver comparison
  compare            Model fidelity comparison
  viewer             Interactive telemetry viewer
  validate           Validation against published F1 data

cars/                TOML car configuration files
tracks/              Track files (JSON and sample data)
docs/                Mathematical derivations and validation reports
```

## Documentation

- `docs/math/equations_of_motion.md` - vehicle model derivations (point-mass through 14-DOF)
- `docs/math/pacejka.md` - tire model theory and implementation
- `docs/math/collocation.md` - optimal control transcription and solver architecture
- `docs/analysis.md` - model fidelity comparison with quantitative results
- `docs/validation/silverstone.md` - validation against published F1 qualifying data
- `docs/validation/methodology.md` - validation approach and acceptance criteria

## Development

```bash
git config core.hooksPath .githooks   # enable auto-format pre-commit hook
cargo test --workspace                # 304 tests
cargo clippy -- -D warnings           # lint check
cargo bench                           # criterion benchmarks
cargo fmt --check                     # format check
```

Or run the setup script:

```bash
./setup.sh
```

## References

- H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., Butterworth-Heinemann, 2012
- W. F. Milliken and D. L. Milliken, *Race Car Vehicle Dynamics*, SAE International, 1995
- J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed., SIAM, 2010
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006
- R. Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
