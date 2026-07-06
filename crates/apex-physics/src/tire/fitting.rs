//! Pacejka coefficient fitting from raw tire test data.
//!
//! Takes force-vs-slip measurements at various vertical loads and fits the
//! Magic Formula coefficients (`B`, `C`, `μ`, `E`) by nonlinear least squares.
//! The Levenberg-Marquardt solver itself lives in [`apex_math::lm`]; this module
//! is a consumer, supplying the residuals through [`ResidualProvider`].

use apex_math::lm::{levenberg_marquardt, LmConfig, ResidualProvider};

use super::pacejka::{PacejkaCoeffs, PacejkaTire};

/// A single measurement point from a tire testing machine.
#[derive(Debug, Clone, Copy)]
pub struct TireTestPoint {
    /// Slip angle (rad). Used for lateral force fitting.
    pub slip_angle: f64,
    /// Slip ratio (dimensionless). Used for longitudinal force fitting.
    pub slip_ratio: f64,
    /// Vertical load on the tire (N).
    pub fz: f64,
    /// Measured lateral force (N).
    pub fy: f64,
    /// Measured longitudinal force (N).
    pub fx: f64,
}

/// A collection of tire test measurements.
#[derive(Debug, Clone)]
pub struct TireTestData {
    /// Descriptive name for this dataset.
    pub name: String,
    /// Individual measurement points.
    pub points: Vec<TireTestPoint>,
}

/// Parse tire test data from CSV.
///
/// Expected columns: slip_angle_deg,slip_ratio,fz_kn,fy_n,fx_n
/// Slip angle is in degrees (converted to radians internally).
/// Vertical load is in kilonewtons (converted to newtons internally).
pub fn parse_tire_test_csv(
    csv_content: &str,
    name: &str,
) -> Result<TireTestData, Box<dyn std::error::Error>> {
    let mut points = Vec::new();

    for line in csv_content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("slip") {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 {
            return Err(format!("Expected 5 columns, got {}: {}", parts.len(), line).into());
        }

        let slip_angle_deg: f64 = parts[0].trim().parse()?;
        let slip_ratio: f64 = parts[1].trim().parse()?;
        let fz_kn: f64 = parts[2].trim().parse()?;
        let fy: f64 = parts[3].trim().parse()?;
        let fx: f64 = parts[4].trim().parse()?;

        points.push(TireTestPoint {
            slip_angle: slip_angle_deg.to_radians(),
            slip_ratio,
            fz: fz_kn * 1000.0,
            fy,
            fx,
        });
    }

    if points.is_empty() {
        return Err("No data points found".into());
    }

    Ok(TireTestData {
        name: name.to_string(),
        points,
    })
}

/// Report on the quality of a Pacejka coefficient fit.
#[derive(Debug, Clone)]
pub struct FitReport {
    /// Root mean square error (N).
    pub rmse: f64,
    /// Coefficient of determination (1.0 = perfect fit).
    pub r_squared: f64,
    /// Maximum absolute error across all points (N).
    pub peak_error: f64,
    /// Number of data points used in the fit.
    pub n_points: usize,
    /// The fitted coefficients.
    pub coeffs: PacejkaCoeffs,
}

/// Fits Pacejka Magic Formula coefficients to tire test data.
///
/// Uses the Levenberg-Marquardt algorithm (a blend of gradient descent
/// and Gauss-Newton) to minimize the sum of squared residuals between
/// measured and predicted tire forces.
pub struct TireFitter<'a> {
    pub data: &'a TireTestData,
}

impl<'a> TireFitter<'a> {
    pub fn new(data: &'a TireTestData) -> Self {
        TireFitter { data }
    }

    /// Fit lateral (cornering) force coefficients.
    ///
    /// Uses only data points where |slip_ratio| < 0.01 (approximately pure cornering).
    /// Fits B, C, mu, E coefficients by minimizing sum of (Fy_measured - Fy_pacejka)^2.
    pub fn fit_lateral(&self) -> FitReport {
        // Filter to pure cornering points
        let pure_lat: Vec<&TireTestPoint> = self
            .data
            .points
            .iter()
            .filter(|p| p.slip_ratio.abs() < 0.01 && p.fz > 100.0)
            .collect();

        if pure_lat.is_empty() {
            return FitReport {
                rmse: f64::INFINITY,
                r_squared: 0.0,
                peak_error: f64::INFINITY,
                n_points: 0,
                coeffs: PacejkaCoeffs::f1_lateral(),
            };
        }

        // Levenberg-Marquardt over [B, C, mu, E], initial guess = F1 defaults.
        let provider = PacejkaResidual {
            points: pure_lat.clone(),
            lateral: true,
        };
        let res = levenberg_marquardt(&provider, &[12.0, 1.5, 1.75, -0.5], &fit_config());
        let coeffs = coeffs_from(&res.params);
        self.compute_fit_report(&pure_lat, &coeffs, true)
    }

    /// Fit longitudinal (traction/braking) force coefficients.
    ///
    /// Uses only data points where |slip_angle| < 0.01 rad (approximately pure longitudinal).
    pub fn fit_longitudinal(&self) -> FitReport {
        let pure_lon: Vec<&TireTestPoint> = self
            .data
            .points
            .iter()
            .filter(|p| p.slip_angle.abs() < 0.01 && p.fz > 100.0)
            .collect();

        if pure_lon.is_empty() {
            return FitReport {
                rmse: f64::INFINITY,
                r_squared: 0.0,
                peak_error: f64::INFINITY,
                n_points: 0,
                coeffs: PacejkaCoeffs::f1_longitudinal(),
            };
        }

        // Levenberg-Marquardt over [B, C, mu, E], initial guess = F1 defaults.
        let provider = PacejkaResidual {
            points: pure_lon.clone(),
            lateral: false,
        };
        let res = levenberg_marquardt(&provider, &[14.0, 1.65, 1.70, -0.3], &fit_config());
        let coeffs = coeffs_from(&res.params);
        self.compute_fit_report(&pure_lon, &coeffs, false)
    }

    // Compute fit quality metrics
    fn compute_fit_report(
        &self,
        points: &[&TireTestPoint],
        coeffs: &PacejkaCoeffs,
        is_lateral: bool,
    ) -> FitReport {
        let params = [coeffs.b, coeffs.c, coeffs.mu, coeffs.e];
        let tire_for_report = make_fit_tire(&params, is_lateral);

        let n = points.len();
        let mut sse = 0.0;
        let mut peak_error = 0.0f64;
        let mut mean_measured = 0.0;

        for p in points {
            let measured = if is_lateral { p.fy } else { p.fx };
            mean_measured += measured;
        }
        mean_measured /= n as f64;

        let mut ss_total = 0.0;
        for p in points {
            let measured = if is_lateral { p.fy } else { p.fx };
            let predicted = if is_lateral {
                tire_for_report.lateral_force(p.slip_angle, p.fz)
            } else {
                tire_for_report.longitudinal_force(p.slip_ratio, p.fz)
            };
            let error = measured - predicted;
            sse += error * error;
            peak_error = peak_error.max(error.abs());
            ss_total += (measured - mean_measured).powi(2);
        }

        let rmse = (sse / n as f64).sqrt();
        let r_squared = if ss_total > 0.0 {
            1.0 - sse / ss_total
        } else {
            0.0
        };

        FitReport {
            rmse,
            r_squared,
            peak_error,
            n_points: n,
            coeffs: *coeffs,
        }
    }
}

/// Build a fitting tire (load sensitivity disabled to isolate the MF
/// coefficients) with the given `[B, C, mu, E]` on the chosen axis.
fn make_fit_tire(params: &[f64], lateral: bool) -> PacejkaTire {
    let mut tire = PacejkaTire::f1_default();
    let coeffs = coeffs_from(params);
    if lateral {
        tire.lateral = coeffs;
    } else {
        tire.longitudinal = coeffs;
    }
    tire.load_sensitivity = 0.0;
    tire
}

/// `[B, C, mu, E]` -> [`PacejkaCoeffs`].
fn coeffs_from(params: &[f64]) -> PacejkaCoeffs {
    PacejkaCoeffs {
        b: params[0],
        c: params[1],
        mu: params[2],
        e: params[3],
    }
}

/// LM schedule matching the original in-module fitter (200 iterations,
/// multiplicative damping from 0.01, 1e-8 step tolerance).
fn fit_config() -> LmConfig {
    LmConfig {
        max_iter: 200,
        initial_lambda: 0.01,
        step_tol: 1e-8,
        cost_tol: 0.0,
    }
}

/// Residual provider for Magic-Formula fitting: `residual = measured - predicted`
/// force at each test point, with box bounds on `[B, C, mu, E]`.
struct PacejkaResidual<'a> {
    points: Vec<&'a TireTestPoint>,
    lateral: bool,
}

impl ResidualProvider for PacejkaResidual<'_> {
    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let tire = make_fit_tire(params, self.lateral);
        self.points
            .iter()
            .map(|p| {
                if self.lateral {
                    p.fy - tire.lateral_force(p.slip_angle, p.fz)
                } else {
                    p.fx - tire.longitudinal_force(p.slip_ratio, p.fz)
                }
            })
            .collect()
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        // B in [1,30], C in [0.5,3], mu in [0.5,3], E in [-5,2].
        vec![(1.0, 30.0), (0.5, 3.0), (0.5, 3.0), (-5.0, 2.0)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `true` if `fitted` is within `frac` (relative) of `expected`.
    fn rel_close(fitted: f64, expected: f64, frac: f64) -> bool {
        (fitted - expected).abs() <= frac * expected.abs()
    }

    /// Build a tire that evaluates the Magic Formula with the given lateral
    /// coefficients and no load sensitivity (so generated data matches what the
    /// fitter assumes).
    fn gen_tire_lateral(b: f64, c: f64, mu: f64, e: f64) -> PacejkaTire {
        let mut tire = PacejkaTire::f1_default();
        tire.lateral = PacejkaCoeffs { b, c, mu, e };
        tire.load_sensitivity = 0.0;
        tire
    }

    fn gen_tire_longitudinal(b: f64, c: f64, mu: f64, e: f64) -> PacejkaTire {
        let mut tire = PacejkaTire::f1_default();
        tire.longitudinal = PacejkaCoeffs { b, c, mu, e };
        tire.load_sensitivity = 0.0;
        tire
    }

    /// Synthetic pure-cornering data: sweep slip angle over each load.
    fn synth_lateral(b: f64, c: f64, mu: f64, e: f64, loads: &[f64]) -> TireTestData {
        let tire = gen_tire_lateral(b, c, mu, e);
        let mut points = Vec::new();
        for &fz in loads {
            let mut a = -0.3;
            while a <= 0.3 + 1e-9 {
                points.push(TireTestPoint {
                    slip_angle: a,
                    slip_ratio: 0.0,
                    fz,
                    fy: tire.lateral_force(a, fz),
                    fx: 0.0,
                });
                a += 0.01;
            }
        }
        TireTestData {
            name: "synthetic-lateral".to_string(),
            points,
        }
    }

    /// Synthetic pure-longitudinal data: sweep slip ratio over each load.
    fn synth_longitudinal(b: f64, c: f64, mu: f64, e: f64, loads: &[f64]) -> TireTestData {
        let tire = gen_tire_longitudinal(b, c, mu, e);
        let mut points = Vec::new();
        for &fz in loads {
            let mut sr = -0.3;
            while sr <= 0.3 + 1e-9 {
                points.push(TireTestPoint {
                    slip_angle: 0.0,
                    slip_ratio: sr,
                    fz,
                    fy: 0.0,
                    fx: tire.longitudinal_force(sr, fz),
                });
                sr += 0.01;
            }
        }
        TireTestData {
            name: "synthetic-longitudinal".to_string(),
            points,
        }
    }

    #[test]
    fn parse_csv_counts_and_converts_units() {
        let csv = "\
# a synthetic tire sweep
slip_angle_deg,slip_ratio,fz_kn,fy_n,fx_n
0.0,0.0,4.0,0.0,0.0
1.0,0.0,4.0,800.0,0.0
2.0,0.0,4.0,1600.0,0.0
3.0,0.0,4.0,2300.0,0.0
4.0,0.0,4.0,2900.0,0.0
5.0,0.0,4.0,3400.0,0.0
-1.0,0.0,4.0,-800.0,0.0
-2.0,0.0,4.0,-1600.0,0.0
-3.0,0.0,4.0,-2300.0,0.0
-4.0,0.0,4.0,-2900.0,0.0
-5.0,0.0,4.0,-3400.0,0.0
6.0,0.0,5.0,4000.0,0.0
";
        let data = parse_tire_test_csv(csv, "test").unwrap();
        assert_eq!(data.name, "test");
        assert_eq!(data.points.len(), 12, "header and comment must be skipped");

        // First data row: 0 deg, 4 kN.
        assert!((data.points[0].slip_angle - 0.0).abs() < 1e-12);
        assert!((data.points[0].fz - 4000.0).abs() < 1e-9, "kN -> N");

        // Second data row: 1 deg -> radians.
        assert!(
            (data.points[1].slip_angle - 1.0_f64.to_radians()).abs() < 1e-12,
            "deg -> rad"
        );
        // Last row: 5 kN -> 5000 N.
        assert!((data.points[11].fz - 5000.0).abs() < 1e-9);
    }

    #[test]
    fn parse_csv_rejects_malformed() {
        // Too few columns.
        assert!(parse_tire_test_csv("1.0,0.0,4.0\n", "bad").is_err());
        // No data rows at all.
        assert!(
            parse_tire_test_csv("slip_angle_deg,slip_ratio,fz_kn,fy_n,fx_n\n", "empty").is_err()
        );
    }

    #[test]
    fn recover_lateral_coefficients() {
        // Known truth.
        let (b, c, mu, e) = (10.0, 1.4, 1.6, -0.8);
        let data = synth_lateral(b, c, mu, e, &[4000.0]);
        let report = TireFitter::new(&data).fit_lateral();

        assert!(
            rel_close(report.coeffs.b, b, 0.10),
            "B = {}",
            report.coeffs.b
        );
        assert!(
            rel_close(report.coeffs.c, c, 0.10),
            "C = {}",
            report.coeffs.c
        );
        assert!(
            rel_close(report.coeffs.mu, mu, 0.10),
            "mu = {}",
            report.coeffs.mu
        );
        assert!(
            rel_close(report.coeffs.e, e, 0.10),
            "E = {}",
            report.coeffs.e
        );

        assert!(report.r_squared > 0.99, "R^2 = {}", report.r_squared);
        assert!(report.rmse < 50.0, "RMSE = {}", report.rmse);
    }

    #[test]
    fn recover_lateral_coefficients_multi_load() {
        let (b, c, mu, e) = (10.0, 1.4, 1.6, -0.8);
        let data = synth_lateral(b, c, mu, e, &[3000.0, 4000.0, 5000.0]);
        let report = TireFitter::new(&data).fit_lateral();

        assert!(
            rel_close(report.coeffs.b, b, 0.10),
            "B = {}",
            report.coeffs.b
        );
        assert!(
            rel_close(report.coeffs.c, c, 0.10),
            "C = {}",
            report.coeffs.c
        );
        assert!(
            rel_close(report.coeffs.mu, mu, 0.10),
            "mu = {}",
            report.coeffs.mu
        );
        assert!(
            rel_close(report.coeffs.e, e, 0.10),
            "E = {}",
            report.coeffs.e
        );
        assert!(report.r_squared > 0.99, "R^2 = {}", report.r_squared);
    }

    #[test]
    fn recover_longitudinal_coefficients() {
        let (b, c, mu, e) = (14.0, 1.65, 1.7, -0.3);
        let data = synth_longitudinal(b, c, mu, e, &[4000.0]);
        let report = TireFitter::new(&data).fit_longitudinal();

        assert!(
            rel_close(report.coeffs.b, b, 0.10),
            "B = {}",
            report.coeffs.b
        );
        assert!(
            rel_close(report.coeffs.c, c, 0.10),
            "C = {}",
            report.coeffs.c
        );
        assert!(
            rel_close(report.coeffs.mu, mu, 0.10),
            "mu = {}",
            report.coeffs.mu
        );
        assert!(
            rel_close(report.coeffs.e, e, 0.10),
            "E = {}",
            report.coeffs.e
        );
        assert!(report.r_squared > 0.99, "R^2 = {}", report.r_squared);
    }

    #[test]
    fn fit_tolerates_noise() {
        let (b, c, mu, e) = (10.0, 1.4, 1.6, -0.8);
        let mut data = synth_lateral(b, c, mu, e, &[3000.0, 4000.0, 5000.0]);
        // Multiplicative, deterministic "noise" of +/- 5%.
        for (i, p) in data.points.iter_mut().enumerate() {
            p.fy *= 1.0 + 0.05 * (i as f64).sin();
        }

        let report = TireFitter::new(&data).fit_lateral();
        assert!(
            rel_close(report.coeffs.b, b, 0.20),
            "B = {}",
            report.coeffs.b
        );
        assert!(
            rel_close(report.coeffs.c, c, 0.20),
            "C = {}",
            report.coeffs.c
        );
        assert!(
            rel_close(report.coeffs.mu, mu, 0.20),
            "mu = {}",
            report.coeffs.mu
        );
        assert!(
            rel_close(report.coeffs.e, e, 0.20),
            "E = {}",
            report.coeffs.e
        );
        assert!(report.r_squared > 0.90, "R^2 = {}", report.r_squared);
    }

    #[test]
    fn empty_data_returns_defaults() {
        let data = TireTestData {
            name: "empty".to_string(),
            points: Vec::new(),
        };
        let report = TireFitter::new(&data).fit_lateral();

        assert!(report.rmse.is_infinite(), "RMSE should be infinite");
        assert_eq!(report.n_points, 0);
        let def = PacejkaCoeffs::f1_lateral();
        assert_eq!(report.coeffs.b, def.b);
        assert_eq!(report.coeffs.c, def.c);
        assert_eq!(report.coeffs.mu, def.mu);
        assert_eq!(report.coeffs.e, def.e);
    }

    #[test]
    fn lateral_fit_ignores_pure_longitudinal_data() {
        // Only longitudinal points (slip ratio well away from zero, so none pass
        // the pure-cornering filter): fit_lateral must fall back to defaults.
        let tire = gen_tire_longitudinal(14.0, 1.65, 1.7, -0.3);
        let mut points = Vec::new();
        let mut sr = 0.05;
        while sr <= 0.3 + 1e-9 {
            points.push(TireTestPoint {
                slip_angle: 0.0,
                slip_ratio: sr,
                fz: 4000.0,
                fy: 0.0,
                fx: tire.longitudinal_force(sr, 4000.0),
            });
            sr += 0.01;
        }
        let data = TireTestData {
            name: "longitudinal-only".to_string(),
            points,
        };
        let report = TireFitter::new(&data).fit_lateral();

        assert!(report.rmse.is_infinite(), "RMSE should be infinite");
        assert_eq!(report.n_points, 0);
        let def = PacejkaCoeffs::f1_lateral();
        assert_eq!(report.coeffs.b, def.b);
    }

    #[test]
    fn fit_report_metrics_on_clean_data() {
        let data = synth_lateral(10.0, 1.4, 1.6, -0.8, &[4000.0]);
        let report = TireFitter::new(&data).fit_lateral();

        assert!(report.rmse < 50.0, "RMSE = {}", report.rmse);
        assert!(
            report.peak_error < 100.0,
            "peak error = {}",
            report.peak_error
        );
        assert!(report.r_squared > 0.99, "R^2 = {}", report.r_squared);
        assert!(report.n_points > 0);
    }
}
