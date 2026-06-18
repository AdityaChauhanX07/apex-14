# Car Configuration Files

TOML files defining vehicle parameters. All fields are optional - missing fields use the base preset's defaults.

## Usage

```rust
use apex_physics::{CarParams, load_car_toml};
let params = load_car_toml(Path::new("cars/f1_2024_calibrated.toml"), &CarParams::default())?;
```

## File Format

```toml
[car]
name = "My Car"
mass = 798.0              # kg
max_drive_force = 11000.0 # N
max_brake_force = 25000.0 # N

[aero]
frontal_area = 1.5        # m^2
drag_coeff = 1.10
lift_coeff = 2.80
aero_balance_front = 0.44 # fraction (0.0-1.0)

[tires]
mu = 1.55                 # peak friction coefficient
rolling_resistance = 0.015

[geometry]
wheelbase = 3.60          # m
cog_to_front = 1.67       # m
cog_to_rear = 1.93        # m
cog_height = 0.30         # m
wheel_radius = 0.330      # m
```
