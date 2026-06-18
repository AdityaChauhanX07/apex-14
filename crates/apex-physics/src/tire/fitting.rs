//! Pacejka coefficient fitting from raw tire test data.
//!
//! Takes force-vs-slip measurements at various vertical loads and fits the
//! Magic Formula coefficients (`B`, `C`, `μ`, `E`) by nonlinear least squares
//! using the Levenberg-Marquardt algorithm.

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

        // Levenberg-Marquardt optimization
        // Decision variables: [B, C, mu, E]
        // Initial guess from the F1 defaults
        let mut params = [12.0, 1.5, 1.75, -0.5];
        let mut lambda = 0.01; // LM damping parameter

        for _iter in 0..200 {
            // Compute residuals and Jacobian
            let (residuals, jacobian) =
                self.compute_lateral_residuals_and_jacobian(&pure_lat, &params);

            let n = residuals.len();
            let m = 4; // number of parameters

            // Normal equations: (J^T J + lambda * diag(J^T J)) * delta = -J^T r
            // Compute J^T J (4x4) and J^T r (4x1)
            let mut jtj = [[0.0f64; 4]; 4];
            let mut jtr = [0.0f64; 4];

            for i in 0..n {
                for p in 0..m {
                    jtr[p] += jacobian[i][p] * residuals[i];
                    for q in 0..m {
                        jtj[p][q] += jacobian[i][p] * jacobian[i][q];
                    }
                }
            }

            // Add LM damping to diagonal
            for (p, row) in jtj.iter_mut().enumerate() {
                row[p] *= 1.0 + lambda;
            }

            // Solve 4x4 system using Gaussian elimination
            let delta = match solve_4x4(&jtj, &jtr) {
                Some(d) => d,
                None => break, // singular, stop
            };

            // Trial step
            let trial = [
                params[0] - delta[0],
                params[1] - delta[1],
                params[2] - delta[2],
                params[3] - delta[3],
            ];

            // Enforce parameter bounds
            let trial = [
                trial[0].clamp(1.0, 30.0), // B: positive, reasonable range
                trial[1].clamp(0.5, 3.0),  // C: positive, shape factor
                trial[2].clamp(0.5, 3.0),  // mu: positive friction
                trial[3].clamp(-5.0, 2.0), // E: can be negative
            ];

            // Compare costs
            let cost_current: f64 = residuals.iter().map(|r| r * r).sum();
            let trial_residuals = self.compute_lateral_residuals(&pure_lat, &trial);
            let cost_trial: f64 = trial_residuals.iter().map(|r| r * r).sum();

            if cost_trial < cost_current {
                params = trial;
                lambda *= 0.5; // reduce damping (more Gauss-Newton)
                lambda = lambda.max(1e-10);
            } else {
                lambda *= 2.0; // increase damping (more gradient descent)
                lambda = lambda.min(1e6);
            }

            // Convergence check
            let step_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            if step_norm < 1e-8 {
                break;
            }
        }

        // Compute final fit quality
        let coeffs = PacejkaCoeffs {
            b: params[0],
            c: params[1],
            mu: params[2],
            e: params[3],
        };

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

        // Same LM optimization but for longitudinal forces
        let mut params = [14.0, 1.65, 1.70, -0.3];
        let mut lambda = 0.01;

        for _iter in 0..200 {
            let (residuals, jacobian) =
                self.compute_longitudinal_residuals_and_jacobian(&pure_lon, &params);

            let n = residuals.len();
            let m = 4;

            let mut jtj = [[0.0f64; 4]; 4];
            let mut jtr = [0.0f64; 4];

            for i in 0..n {
                for p in 0..m {
                    jtr[p] += jacobian[i][p] * residuals[i];
                    for q in 0..m {
                        jtj[p][q] += jacobian[i][p] * jacobian[i][q];
                    }
                }
            }

            for (p, row) in jtj.iter_mut().enumerate() {
                row[p] *= 1.0 + lambda;
            }

            let delta = match solve_4x4(&jtj, &jtr) {
                Some(d) => d,
                None => break,
            };

            let trial = [
                (params[0] - delta[0]).clamp(1.0, 30.0),
                (params[1] - delta[1]).clamp(0.5, 3.0),
                (params[2] - delta[2]).clamp(0.5, 3.0),
                (params[3] - delta[3]).clamp(-5.0, 2.0),
            ];

            let cost_current: f64 = residuals.iter().map(|r| r * r).sum();
            let trial_residuals = self.compute_longitudinal_residuals(&pure_lon, &trial);
            let cost_trial: f64 = trial_residuals.iter().map(|r| r * r).sum();

            if cost_trial < cost_current {
                params = trial;
                lambda *= 0.5;
                lambda = lambda.max(1e-10);
            } else {
                lambda *= 2.0;
                lambda = lambda.min(1e6);
            }

            let step_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            if step_norm < 1e-8 {
                break;
            }
        }

        let coeffs = PacejkaCoeffs {
            b: params[0],
            c: params[1],
            mu: params[2],
            e: params[3],
        };

        self.compute_fit_report(&pure_lon, &coeffs, false)
    }

    // Helper: compute residuals for lateral fitting
    fn compute_lateral_residuals(&self, points: &[&TireTestPoint], params: &[f64; 4]) -> Vec<f64> {
        let tire = self.make_tire(params);
        points
            .iter()
            .map(|p| {
                let fy_pred = tire.lateral_force(p.slip_angle, p.fz);
                p.fy - fy_pred
            })
            .collect()
    }

    // Helper: compute residuals AND Jacobian for lateral fitting using auto-diff
    fn compute_lateral_residuals_and_jacobian(
        &self,
        points: &[&TireTestPoint],
        params: &[f64; 4],
    ) -> (Vec<f64>, Vec<[f64; 4]>) {
        let tire = self.make_tire(params);
        let n = points.len();
        let mut residuals = Vec::with_capacity(n);
        let mut jacobian = Vec::with_capacity(n);

        for p in points {
            let fy_pred = tire.lateral_force(p.slip_angle, p.fz);
            residuals.push(p.fy - fy_pred);

            // Compute Jacobian by finite differences on the Pacejka parameters
            // (auto-diff over the coefficients would require making them generic,
            // which is complex; FD on 4 parameters is cheap and accurate enough)
            let eps = 1e-6;
            let mut jac_row = [0.0; 4];
            for k in 0..4 {
                let mut p_plus = *params;
                p_plus[k] += eps;
                let tire_plus = self.make_tire(&p_plus);
                let fy_plus = tire_plus.lateral_force(p.slip_angle, p.fz);

                let mut p_minus = *params;
                p_minus[k] -= eps;
                let tire_minus = self.make_tire(&p_minus);
                let fy_minus = tire_minus.lateral_force(p.slip_angle, p.fz);

                // Negative because residual = measured - predicted
                // d(residual)/d(param) = -d(predicted)/d(param)
                jac_row[k] = -(fy_plus - fy_minus) / (2.0 * eps);
            }
            jacobian.push(jac_row);
        }

        (residuals, jacobian)
    }

    // Same helpers for longitudinal
    fn compute_longitudinal_residuals(
        &self,
        points: &[&TireTestPoint],
        params: &[f64; 4],
    ) -> Vec<f64> {
        let tire = self.make_tire_lon(params);
        points
            .iter()
            .map(|p| {
                let fx_pred = tire.longitudinal_force(p.slip_ratio, p.fz);
                p.fx - fx_pred
            })
            .collect()
    }

    fn compute_longitudinal_residuals_and_jacobian(
        &self,
        points: &[&TireTestPoint],
        params: &[f64; 4],
    ) -> (Vec<f64>, Vec<[f64; 4]>) {
        let tire = self.make_tire_lon(params);
        let n = points.len();
        let mut residuals = Vec::with_capacity(n);
        let mut jacobian = Vec::with_capacity(n);

        for p in points {
            let fx_pred = tire.longitudinal_force(p.slip_ratio, p.fz);
            residuals.push(p.fx - fx_pred);

            let eps = 1e-6;
            let mut jac_row = [0.0; 4];
            for k in 0..4 {
                let mut p_plus = *params;
                p_plus[k] += eps;
                let tire_plus = self.make_tire_lon(&p_plus);
                let fx_plus = tire_plus.longitudinal_force(p.slip_ratio, p.fz);

                let mut p_minus = *params;
                p_minus[k] -= eps;
                let tire_minus = self.make_tire_lon(&p_minus);
                let fx_minus = tire_minus.longitudinal_force(p.slip_ratio, p.fz);

                jac_row[k] = -(fx_plus - fx_minus) / (2.0 * eps);
            }
            jacobian.push(jac_row);
        }

        (residuals, jacobian)
    }

    // Create a PacejkaTire from lateral fit parameters
    fn make_tire(&self, params: &[f64; 4]) -> PacejkaTire {
        let mut tire = PacejkaTire::f1_default();
        tire.lateral = PacejkaCoeffs {
            b: params[0],
            c: params[1],
            mu: params[2],
            e: params[3],
        };
        tire.load_sensitivity = 0.0; // disable during fitting to isolate the MF coefficients
        tire
    }

    // Create a PacejkaTire from longitudinal fit parameters
    fn make_tire_lon(&self, params: &[f64; 4]) -> PacejkaTire {
        let mut tire = PacejkaTire::f1_default();
        tire.longitudinal = PacejkaCoeffs {
            b: params[0],
            c: params[1],
            mu: params[2],
            e: params[3],
        };
        tire.load_sensitivity = 0.0;
        tire
    }

    // Compute fit quality metrics
    fn compute_fit_report(
        &self,
        points: &[&TireTestPoint],
        coeffs: &PacejkaCoeffs,
        is_lateral: bool,
    ) -> FitReport {
        let tire_for_report = if is_lateral {
            self.make_tire(&[coeffs.b, coeffs.c, coeffs.mu, coeffs.e])
        } else {
            self.make_tire_lon(&[coeffs.b, coeffs.c, coeffs.mu, coeffs.e])
        };

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

/// Solve a 4x4 linear system Ax = b using Gaussian elimination with partial pivoting.
//
// The forward-elimination and back-substitution sweeps are inherently
// index-based (pivoting, triangular access), so the range loops are clearer than
// iterator equivalents here.
#[allow(clippy::needless_range_loop)]
fn solve_4x4(a: &[[f64; 4]; 4], b: &[f64; 4]) -> Option<[f64; 4]> {
    let mut aug = [[0.0; 5]; 4];
    for i in 0..4 {
        aug[i][..4].copy_from_slice(&a[i][..4]);
        aug[i][4] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..4 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate below
        for row in (col + 1)..4 {
            let factor = aug[row][col] / aug[col][col];
            for j in col..5 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        x[i] = aug[i][4];
        for j in (i + 1)..4 {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Some(x)
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
