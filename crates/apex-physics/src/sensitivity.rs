//! Parameter sensitivity analysis for the QSS lap simulator.
//!
//! Runs many perturbed simulations to quantify how each car parameter affects
//! lap time, via one-at-a-time (OAT) sweeps and Monte Carlo sampling. The QSS
//! runs are dispatched across threads with rayon.

/// Identifies a tunable parameter in CarParams.
#[derive(Debug, Clone)]
pub struct ParameterDef {
    /// Human-readable name.
    pub name: String,
    /// Nominal (baseline) value.
    pub nominal: f64,
    /// Minimum value for perturbation.
    pub min: f64,
    /// Maximum value for perturbation.
    pub max: f64,
}

/// Standard set of parameters to analyze for an F1 car.
pub fn f1_parameter_set(params: &super::CarParams) -> Vec<ParameterDef> {
    vec![
        ParameterDef {
            name: "Mass".into(),
            nominal: params.mass,
            min: params.mass * 0.95,
            max: params.mass * 1.05,
        },
        ParameterDef {
            name: "Drag coeff".into(),
            nominal: params.drag_coeff,
            min: params.drag_coeff * 0.90,
            max: params.drag_coeff * 1.10,
        },
        ParameterDef {
            name: "Lift coeff".into(),
            nominal: params.lift_coeff,
            min: params.lift_coeff * 0.90,
            max: params.lift_coeff * 1.10,
        },
        ParameterDef {
            name: "Tire mu".into(),
            nominal: params.tire_mu,
            min: params.tire_mu * 0.90,
            max: params.tire_mu * 1.10,
        },
        ParameterDef {
            name: "Max drive force".into(),
            nominal: params.max_drive_force,
            min: params.max_drive_force * 0.90,
            max: params.max_drive_force * 1.10,
        },
        ParameterDef {
            name: "Max brake force".into(),
            nominal: params.max_brake_force,
            min: params.max_brake_force * 0.90,
            max: params.max_brake_force * 1.10,
        },
        ParameterDef {
            name: "Frontal area".into(),
            nominal: params.frontal_area,
            min: params.frontal_area * 0.95,
            max: params.frontal_area * 1.05,
        },
        ParameterDef {
            name: "CoG height".into(),
            nominal: params.cog_height,
            min: params.cog_height * 0.80,
            max: params.cog_height * 1.20,
        },
        ParameterDef {
            name: "Aero balance".into(),
            nominal: params.aero_balance_front,
            min: 0.40,
            max: 0.50,
        },
        ParameterDef {
            name: "Rolling resistance".into(),
            nominal: params.rolling_resistance,
            min: params.rolling_resistance * 0.80,
            max: params.rolling_resistance * 1.20,
        },
    ]
}

/// Apply a parameter value to CarParams by name.
/// Returns a modified copy.
fn apply_parameter(params: &super::CarParams, name: &str, value: f64) -> super::CarParams {
    let mut p = params.clone();
    match name {
        "Mass" => p.mass = value,
        "Drag coeff" => p.drag_coeff = value,
        "Lift coeff" => p.lift_coeff = value,
        "Tire mu" => p.tire_mu = value,
        "Max drive force" => p.max_drive_force = value,
        "Max brake force" => p.max_brake_force = value,
        "Frontal area" => p.frontal_area = value,
        "CoG height" => p.cog_height = value,
        "Aero balance" => p.aero_balance_front = value,
        "Rolling resistance" => p.rolling_resistance = value,
        _ => {}
    }
    p
}

/// Result of varying a single parameter while holding others at nominal.
#[derive(Debug, Clone)]
pub struct OatResult {
    /// Parameter name.
    pub name: String,
    /// Nominal value.
    pub nominal: f64,
    /// Parameter values tested.
    pub values: Vec<f64>,
    /// Corresponding lap times.
    pub lap_times: Vec<f64>,
    /// Sensitivity: d(lap_time)/d(parameter) at nominal, estimated via central difference.
    pub sensitivity: f64,
    /// Normalized sensitivity: percentage lap time change per percentage parameter change.
    pub sensitivity_pct: f64,
    /// Nominal lap time.
    pub nominal_lap_time: f64,
}

/// Run one-at-a-time sensitivity analysis.
///
/// For each parameter, sweep from min to max in `n_samples` steps,
/// run QSS at each point, and compute the sensitivity.
pub fn oat_sensitivity(
    track: &apex_track::Track,
    base_params: &super::CarParams,
    parameters: &[ParameterDef],
    n_samples: usize,
) -> Vec<OatResult> {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let nominal_time = super::qss_lap_sim(track, base_params).lap_time;

    parameters
        .iter()
        .map(|param| {
            // Generate sample points
            let values: Vec<f64> = (0..n_samples)
                .map(|i| {
                    param.min + (param.max - param.min) * i as f64 / (n_samples - 1).max(1) as f64
                })
                .collect();

            // Run QSS for each sample point (parallel when rayon is enabled,
            // sequential otherwise -- e.g. on wasm32 without threads).
            #[cfg(feature = "parallel")]
            let lap_times: Vec<f64> = values
                .par_iter()
                .map(|&val| {
                    let modified = apply_parameter(base_params, &param.name, val);
                    super::qss_lap_sim(track, &modified).lap_time
                })
                .collect();
            #[cfg(not(feature = "parallel"))]
            let lap_times: Vec<f64> = values
                .iter()
                .map(|&val| {
                    let modified = apply_parameter(base_params, &param.name, val);
                    super::qss_lap_sim(track, &modified).lap_time
                })
                .collect();

            // Compute sensitivity via central difference at nominal
            // Find the two points closest to nominal on either side
            let eps = (param.max - param.min) * 0.01;
            let params_plus = apply_parameter(base_params, &param.name, param.nominal + eps);
            let params_minus = apply_parameter(base_params, &param.name, param.nominal - eps);
            let time_plus = super::qss_lap_sim(track, &params_plus).lap_time;
            let time_minus = super::qss_lap_sim(track, &params_minus).lap_time;
            let sensitivity = (time_plus - time_minus) / (2.0 * eps);

            // Normalized: (dT/T) / (dP/P) = (dT/dP) * (P/T)
            let sensitivity_pct = sensitivity * param.nominal / nominal_time * 100.0;

            OatResult {
                name: param.name.clone(),
                nominal: param.nominal,
                values,
                lap_times,
                sensitivity,
                sensitivity_pct,
                nominal_lap_time: nominal_time,
            }
        })
        .collect()
}

/// Result of Monte Carlo sensitivity analysis.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Number of samples run.
    pub n_samples: usize,
    /// All lap times from the samples.
    pub lap_times: Vec<f64>,
    /// Mean lap time.
    pub mean: f64,
    /// Standard deviation of lap time.
    pub std_dev: f64,
    /// 5th percentile.
    pub percentile_5: f64,
    /// 95th percentile.
    pub percentile_95: f64,
    /// Nominal lap time (all parameters at baseline).
    pub nominal: f64,
    /// Correlation between each parameter and lap time.
    /// Sorted by absolute correlation (most correlated first).
    pub correlations: Vec<(String, f64)>,
}

/// Run Monte Carlo sensitivity analysis.
///
/// Samples all parameters simultaneously from uniform distributions
/// within their min/max ranges. Uses a deterministic pseudo-random
/// sequence for reproducibility.
pub fn monte_carlo_sensitivity(
    track: &apex_track::Track,
    base_params: &super::CarParams,
    parameters: &[ParameterDef],
    n_samples: usize,
    seed: u64,
) -> MonteCarloResult {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let nominal_time = super::qss_lap_sim(track, base_params).lap_time;

    // Generate parameter samples using a simple deterministic PRNG
    // (avoid adding rand as a dependency - use a basic LCG)
    let samples: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| {
            parameters
                .iter()
                .enumerate()
                .map(|(j, param)| {
                    // Deterministic pseudo-random: hash of (seed, sample_index, param_index)
                    let hash = simple_hash(seed, i as u64, j as u64);
                    let t = (hash as f64) / (u64::MAX as f64); // uniform [0, 1)
                    param.min + t * (param.max - param.min)
                })
                .collect()
        })
        .collect();

    // Run QSS for each sample (parallel when rayon is enabled, sequential
    // otherwise -- e.g. on wasm32 without threads).
    #[cfg(feature = "parallel")]
    let lap_times: Vec<f64> = samples
        .par_iter()
        .map(|sample| {
            let mut modified = base_params.clone();
            for (j, param) in parameters.iter().enumerate() {
                modified = apply_parameter(&modified, &param.name, sample[j]);
            }
            super::qss_lap_sim(track, &modified).lap_time
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let lap_times: Vec<f64> = samples
        .iter()
        .map(|sample| {
            let mut modified = base_params.clone();
            for (j, param) in parameters.iter().enumerate() {
                modified = apply_parameter(&modified, &param.name, sample[j]);
            }
            super::qss_lap_sim(track, &modified).lap_time
        })
        .collect();

    // Statistics
    let n = lap_times.len() as f64;
    let mean = lap_times.iter().sum::<f64>() / n;
    let variance = lap_times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    // Percentiles
    let mut sorted = lap_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_5 = sorted[(0.05 * n) as usize];
    let percentile_95 = sorted[(0.95 * n).min(n - 1.0) as usize];

    // Correlations: Pearson correlation between each parameter and lap time
    let mut correlations: Vec<(String, f64)> = parameters
        .iter()
        .enumerate()
        .map(|(j, param)| {
            let param_values: Vec<f64> = samples.iter().map(|s| s[j]).collect();
            let corr = pearson_correlation(&param_values, &lap_times);
            (param.name.clone(), corr)
        })
        .collect();

    // Sort by absolute correlation (most influential first)
    correlations.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    MonteCarloResult {
        n_samples,
        lap_times,
        mean,
        std_dev,
        percentile_5,
        percentile_95,
        nominal: nominal_time,
        correlations,
    }
}

/// Simple deterministic hash for pseudo-random sampling (no rand dependency).
fn simple_hash(seed: u64, i: u64, j: u64) -> u64 {
    let mut h = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    h = h.wrapping_add(i.wrapping_mul(2862933555777941757));
    h = h.wrapping_add(j.wrapping_mul(3037000493));
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Pearson correlation coefficient between two vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

/// Generate a tornado chart SVG showing parameter sensitivities.
pub fn tornado_chart_svg(
    results: &[OatResult],
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Sort by absolute sensitivity_pct (most sensitive first)
    let mut sorted: Vec<&OatResult> = results.iter().collect();
    sorted.sort_by(|a, b| {
        b.sensitivity_pct
            .abs()
            .partial_cmp(&a.sensitivity_pct.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = sorted.len();
    let bar_height = 30.0;
    let gap = 8.0;
    let label_width = 180.0;
    let chart_width = 400.0;
    let margin = 20.0;
    let total_width = label_width + chart_width + margin * 2.0 + 80.0;
    let total_height = margin * 2.0 + n as f64 * (bar_height + gap) + 40.0;

    let max_sensitivity = sorted
        .iter()
        .map(|r| r.sensitivity_pct.abs())
        .fold(0.0f64, f64::max)
        .max(0.1);

    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"##,
        total_width, total_height, total_width, total_height
    ));
    svg.push_str(&format!(
        r##"<rect width="{}" height="{}" fill="#1a1a2e"/>"##,
        total_width, total_height
    ));

    // Title
    svg.push_str(&format!(
        r##"<text x="{}" y="{}" fill="white" font-size="16" font-family="sans-serif">Sensitivity Analysis (% lap time / % parameter)</text>"##,
        margin,
        margin + 16.0
    ));

    let chart_left = label_width + margin;
    let center_x = chart_left + chart_width / 2.0;

    // Center line
    let chart_top = margin + 35.0;
    let chart_bottom = chart_top + n as f64 * (bar_height + gap);
    svg.push_str(&format!(
        r##"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#666" stroke-width="1"/>"##,
        center_x, chart_top, center_x, chart_bottom
    ));

    for (i, result) in sorted.iter().enumerate() {
        let y = chart_top + i as f64 * (bar_height + gap);
        let bar_width = (result.sensitivity_pct.abs() / max_sensitivity) * (chart_width / 2.0);

        // Color: red for positive sensitivity (more = slower), blue for negative (more = faster)
        let color = if result.sensitivity_pct > 0.0 {
            "#e74c3c"
        } else {
            "#3498db"
        };

        let (bar_x, bar_w) = if result.sensitivity_pct > 0.0 {
            (center_x, bar_width)
        } else {
            (center_x - bar_width, bar_width)
        };

        // Bar
        svg.push_str(&format!(
            r##"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" rx="3"/>"##,
            bar_x, y, bar_w, bar_height, color
        ));

        // Label
        svg.push_str(&format!(
            r##"<text x="{:.1}" y="{:.1}" fill="#ccc" font-size="13" font-family="sans-serif" text-anchor="end">{}</text>"##,
            chart_left - 8.0,
            y + bar_height / 2.0 + 4.5,
            result.name
        ));

        // Value
        svg.push_str(&format!(
            r##"<text x="{:.1}" y="{:.1}" fill="white" font-size="12" font-family="sans-serif">{:+.2}%</text>"##,
            if result.sensitivity_pct > 0.0 {
                bar_x + bar_w + 5.0
            } else {
                bar_x - 5.0
            },
            y + bar_height / 2.0 + 4.5,
            result.sensitivity_pct
        ));
    }

    svg.push_str("</svg>");

    let mut file = std::fs::File::create(path)?;
    file.write_all(svg.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CarParams;
    use apex_track::{build_track, oval_track, Track};

    fn test_oval() -> Track {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 300);
        build_track("oval", &pts, closed)
    }

    fn one_param(params: &CarParams, name: &str) -> Vec<ParameterDef> {
        f1_parameter_set(params)
            .into_iter()
            .filter(|p| p.name == name)
            .collect()
    }

    // (a) Mass increase slows the car.
    #[test]
    fn oat_mass_slows_car() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = one_param(&car, "Mass");
        let results = oat_sensitivity(&track, &car, &params, 5);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].sensitivity > 0.0,
            "mass sensitivity {} should be positive",
            results[0].sensitivity
        );
        assert!(results[0].sensitivity_pct > 0.0);
    }

    // (b) Tire mu increase speeds up the car.
    #[test]
    fn oat_grip_speeds_car() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = one_param(&car, "Tire mu");
        let results = oat_sensitivity(&track, &car, &params, 5);
        assert!(
            results[0].sensitivity < 0.0,
            "tire-mu sensitivity {} should be negative",
            results[0].sensitivity
        );
    }

    // (c) Tire mu is among the most sensitive parameters.
    #[test]
    fn oat_grip_is_highly_sensitive() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = f1_parameter_set(&car);
        let mut results = oat_sensitivity(&track, &car, &params, 5);
        results.sort_by(|a, b| {
            b.sensitivity_pct
                .abs()
                .partial_cmp(&a.sensitivity_pct.abs())
                .unwrap()
        });
        let top3: Vec<&str> = results.iter().take(3).map(|r| r.name.as_str()).collect();
        assert!(
            top3.contains(&"Tire mu"),
            "Tire mu should be in the top 3 most sensitive, got {top3:?}"
        );
    }

    // (d) Monte Carlo mean is close to nominal.
    #[test]
    fn monte_carlo_mean_near_nominal() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = f1_parameter_set(&car);
        let mc = monte_carlo_sensitivity(&track, &car, &params, 500, 42);
        let rel = (mc.mean - mc.nominal).abs() / mc.nominal;
        assert!(
            rel < 0.03,
            "mean {} vs nominal {} ({:.2}% off)",
            mc.mean,
            mc.nominal,
            rel * 100.0
        );
    }

    // (e) Monte Carlo spread is reasonable.
    #[test]
    fn monte_carlo_std_reasonable() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = f1_parameter_set(&car);
        let mc = monte_carlo_sensitivity(&track, &car, &params, 500, 42);
        let rel_std = mc.std_dev / mc.nominal;
        assert!(
            (0.005..=0.10).contains(&rel_std),
            "relative std {rel_std} should be in [0.5%, 10%]"
        );
    }

    // (f) Monte Carlo correlations have the right signs and a clear driver.
    #[test]
    fn monte_carlo_correlations_make_sense() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = f1_parameter_set(&car);
        let mc = monte_carlo_sensitivity(&track, &car, &params, 500, 7);

        let find = |name: &str| mc.correlations.iter().find(|(n, _)| n == name).unwrap().1;
        assert!(find("Tire mu") < 0.0, "tire mu should correlate negatively");
        assert!(find("Mass") > 0.0, "mass should correlate positively");

        // The most influential parameter (first, since sorted by |r|) is clear.
        assert!(
            mc.correlations[0].1.abs() > 0.3,
            "top correlation {:?} should have |r| > 0.3",
            mc.correlations[0]
        );
    }

    // (g) Reproducibility: same seed -> identical, different seed -> different.
    #[test]
    fn monte_carlo_reproducible() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = f1_parameter_set(&car);

        let a = monte_carlo_sensitivity(&track, &car, &params, 200, 123);
        let b = monte_carlo_sensitivity(&track, &car, &params, 200, 123);
        assert_eq!(a.lap_times, b.lap_times, "same seed must reproduce");
        assert_eq!(a.mean, b.mean);

        let c = monte_carlo_sensitivity(&track, &car, &params, 200, 999);
        assert_ne!(a.lap_times, c.lap_times, "different seed must differ");
    }

    // (h) Tornado chart SVG output.
    #[test]
    fn tornado_svg_written() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = f1_parameter_set(&car);
        let results = oat_sensitivity(&track, &car, &params, 5);

        let path = std::env::temp_dir().join(format!("apex_tornado_{}.svg", std::process::id()));
        tornado_chart_svg(&results, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("<svg"), "should start with <svg");
        assert!(content.contains("</svg>"), "should contain closing tag");
        assert!(
            content.len() > 500,
            "svg {} bytes, expected > 500",
            content.len()
        );

        let _ = std::fs::remove_file(&path);
    }

    // (i) Parallel Monte Carlo completes and yields finite results.
    #[test]
    fn monte_carlo_parallel_completes() {
        let track = test_oval();
        let car = CarParams::f1_2024_calibrated();
        let params = f1_parameter_set(&car);
        let mc = monte_carlo_sensitivity(&track, &car, &params, 1000, 1);
        assert_eq!(mc.lap_times.len(), 1000);
        assert!(mc.lap_times.iter().all(|t| t.is_finite()));
        assert!(mc.mean.is_finite() && mc.std_dev.is_finite());
    }
}
