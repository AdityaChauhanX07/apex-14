# Validation Methodology

## Approach

Apex-14 is validated by comparing simulation output against published Formula 1 performance data. The comparison uses:

- **Track geometry**: TUMFTM racetrack database centerline coordinates with measured track widths, or parametric approximations when real data is unavailable.
- **Vehicle parameters**: The `f1_2024_calibrated` preset, tuned to approximate a 2024-era F1 car.
- **Reference data**: Published qualifying lap times, speed trap data, and onboard telemetry analyses from Formula 1 broadcasts and technical publications.

## Metrics

Each validation compares these quantities:

| Metric | Source | Acceptable error |
|--------|--------|-----------------|
| Lap time | Published qualifying results | Within 15% |
| Top speed | Speed trap data | Within 10% |
| Minimum corner speed | Onboard analysis | Within 20% |
| Lateral g envelope | Telemetry analysis | Within 1.0g |

## Limitations

- The vehicle model uses a single set of parameters for the entire lap. Real F1 cars adjust differential, brake bias, and engine maps corner by corner.
- Aerodynamic parameters are constant. Real downforce varies with DRS activation and ride height changes.
- Tire grip is assumed constant over the lap. Thermal degradation is modeled but not integrated into the lap simulation.
- The track surface is assumed flat. Real circuits have elevation changes and camber that significantly affect cornering speeds.
- The quasi-steady-state (QSS) solver evaluates speed along the **centerline**, not the racing line a real car drives. A racing line straightens corners (larger effective radius), so centerline QSS is systematically conservative through fast, wide corners.

## Reference Data Sources

Published qualifying results from formula1.com and FIA timing sheets. Speed trap data from F1 broadcasts. Cornering speed and g-force estimates from published technical analyses (e.g., Scarbs Technical, Craig Scarborough; Mark Hughes race analyses).
