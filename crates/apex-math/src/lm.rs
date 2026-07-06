//! Generic Levenberg-Marquardt least-squares with identifiability diagnostics.
//!
//! A caller implements [`ResidualProvider`] (residual vector + per-parameter
//! bounds; the Jacobian defaults to central finite differences) and calls
//! [`levenberg_marquardt`]. The solver minimizes `Σ rᵢ(p)²` subject to box
//! bounds using Marquardt's multiplicative-diagonal damping, and returns not
//! just the fitted parameters but the statistics needed to judge whether they
//! are *identifiable*: the normal matrix `JᵀJ`, its condition number, the
//! parameter covariance `(JᵀJ)⁻¹·σ²`, per-parameter standard errors, the
//! correlation matrix, and flags for strongly-correlated pairs / weakly-pinned
//! parameters.
//!
//! Pure `std` + this crate only (no rand, no threads), so it is wasm-safe —
//! apex-math is in the web-viewer graph.

// The dense linear-algebra routines below (Gaussian elimination with pivoting,
// Gauss-Jordan inversion, cyclic Jacobi rotations) are inherently index-based —
// pivot/triangular/symmetric access patterns read clearer as range loops than as
// iterator chains.
#![allow(clippy::needless_range_loop)]

/// Something whose sum-of-squared residuals can be minimized over `params`.
pub trait ResidualProvider {
    /// Residual vector `r(params)` (e.g. `measured − predicted`). Its length is
    /// the number of observations and must not depend on `params`.
    fn residuals(&self, params: &[f64]) -> Vec<f64>;

    /// Inclusive `(lo, hi)` bound per parameter (same length as `params`).
    fn bounds(&self) -> Vec<(f64, f64)>;

    /// Jacobian `Jᵢₖ = ∂rᵢ/∂pₖ`. Defaults to central finite differences; override
    /// for an analytic or forward-dual Jacobian.
    fn jacobian(&self, params: &[f64]) -> Vec<Vec<f64>> {
        numerical_jacobian(self, params)
    }
}

/// Central-difference Jacobian of a [`ResidualProvider`].
pub fn numerical_jacobian<P: ResidualProvider + ?Sized>(p: &P, params: &[f64]) -> Vec<Vec<f64>> {
    let m = params.len();
    let n = p.residuals(params).len();
    let mut jac = vec![vec![0.0; m]; n];
    for k in 0..m {
        let h = (1e-6 * params[k].abs()).max(1e-8);
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        pp[k] += h;
        pm[k] -= h;
        let rp = p.residuals(&pp);
        let rm = p.residuals(&pm);
        for (i, row) in jac.iter_mut().enumerate() {
            row[k] = (rp[i] - rm[i]) / (2.0 * h);
        }
    }
    jac
}

/// Per-iteration trace entry.
#[derive(Debug, Clone, Copy)]
pub struct LmIteration {
    /// Iteration index (0-based).
    pub iter: usize,
    /// Sum of squared residuals *after* this iteration's accepted step.
    pub cost: f64,
    /// Euclidean norm of the accepted parameter step (0 if the step was rejected).
    pub step_norm: f64,
    /// Damping parameter `λ` after this iteration.
    pub lambda: f64,
    /// Whether the trial step reduced the cost and was accepted.
    pub accepted: bool,
}

/// LM tuning. [`Default`] mirrors the legacy tire-fit schedule.
#[derive(Debug, Clone, Copy)]
pub struct LmConfig {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Initial damping `λ`.
    pub initial_lambda: f64,
    /// Stop when the accepted step norm falls below this.
    pub step_tol: f64,
    /// Stop when the relative cost improvement falls below this (0 disables).
    pub cost_tol: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        LmConfig {
            max_iter: 200,
            initial_lambda: 0.01,
            step_tol: 1e-8,
            cost_tol: 0.0,
        }
    }
}

/// Fit outcome plus identifiability diagnostics.
#[derive(Debug, Clone)]
pub struct LmResult {
    /// Fitted parameters.
    pub params: Vec<f64>,
    /// Final sum of squared residuals.
    pub cost: f64,
    /// Initial sum of squared residuals (at the starting guess).
    pub initial_cost: f64,
    /// Number of observations (residual length).
    pub n_residuals: usize,
    /// Per-iteration trace.
    pub iterations: Vec<LmIteration>,
    /// `true` if a tolerance (step/cost) triggered before `max_iter`.
    pub converged: bool,
    /// Normal matrix `JᵀJ` at the solution.
    pub jtj: Vec<Vec<f64>>,
    /// Condition number of `JᵀJ` (`λ_max / λ_min`; `∞` if singular).
    pub condition_number: f64,
    /// Parameter covariance `(JᵀJ)⁻¹ · σ²` (`σ²` = cost / dof).
    pub covariance: Vec<Vec<f64>>,
    /// Per-parameter standard errors (`√diag(covariance)`).
    pub std_errors: Vec<f64>,
    /// Parameter correlation matrix.
    pub correlations: Vec<Vec<f64>>,
    /// Parameter-index pairs with `|correlation| > 0.95` (i < j, |corr|).
    pub weak_pairs: Vec<(usize, usize, f64)>,
    /// Indices whose std error exceeds 50% of `|value|` (weakly identifiable).
    pub weak_params: Vec<usize>,
    /// Indices sitting on (within 1e-6 relative of) a bound.
    pub bound_pinned: Vec<usize>,
}

/// Minimize `Σ r²` over box-bounded parameters with Levenberg-Marquardt.
pub fn levenberg_marquardt<P: ResidualProvider + ?Sized>(
    provider: &P,
    initial: &[f64],
    config: &LmConfig,
) -> LmResult {
    let m = initial.len();
    let bounds = provider.bounds();
    assert_eq!(bounds.len(), m, "bounds length must match parameter count");

    let clamp = |p: &[f64]| -> Vec<f64> {
        p.iter()
            .zip(&bounds)
            .map(|(&v, &(lo, hi))| v.clamp(lo, hi))
            .collect()
    };

    let mut params = clamp(initial);
    let mut lambda = config.initial_lambda;
    let initial_cost: f64 = sum_sq(&provider.residuals(&params));
    let mut cost = initial_cost;
    let mut iterations = Vec::new();
    let mut converged = false;

    for iter in 0..config.max_iter {
        let residuals = provider.residuals(&params);
        let jac = provider.jacobian(&params);
        let n = residuals.len();

        // JᵀJ and Jᵀr.
        let mut jtj = vec![vec![0.0; m]; m];
        let mut jtr = vec![0.0; m];
        for i in 0..n {
            for p in 0..m {
                jtr[p] += jac[i][p] * residuals[i];
                for q in 0..m {
                    jtj[p][q] += jac[i][p] * jac[i][q];
                }
            }
        }

        // Marquardt multiplicative-diagonal damping.
        let mut damped = jtj.clone();
        for (p, row) in damped.iter_mut().enumerate() {
            row[p] *= 1.0 + lambda;
        }

        let delta = match solve_linear(&damped, &jtr) {
            Some(d) => d,
            None => break, // singular normal matrix
        };

        let trial: Vec<f64> = clamp(
            &params
                .iter()
                .zip(&delta)
                .map(|(&p, &d)| p - d)
                .collect::<Vec<_>>(),
        );
        let trial_cost = sum_sq(&provider.residuals(&trial));

        let step_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        let accepted = trial_cost < cost;
        if accepted {
            let rel_improve = if cost > 0.0 {
                (cost - trial_cost) / cost
            } else {
                0.0
            };
            params = trial;
            cost = trial_cost;
            lambda = (lambda * 0.5).max(1e-10);
            iterations.push(LmIteration {
                iter,
                cost,
                step_norm,
                lambda,
                accepted,
            });
            if step_norm < config.step_tol
                || (config.cost_tol > 0.0 && rel_improve < config.cost_tol)
            {
                converged = true;
                break;
            }
        } else {
            lambda = (lambda * 2.0).min(1e6);
            iterations.push(LmIteration {
                iter,
                cost,
                step_norm: 0.0,
                lambda,
                accepted,
            });
            if step_norm < config.step_tol {
                converged = true;
                break;
            }
        }
    }

    // ---- identifiability diagnostics at the solution ----
    let residuals = provider.residuals(&params);
    let jac = provider.jacobian(&params);
    let n = residuals.len();
    let mut jtj = vec![vec![0.0; m]; m];
    for i in 0..n {
        for p in 0..m {
            for q in 0..m {
                jtj[p][q] += jac[i][p] * jac[i][q];
            }
        }
    }

    let eig = jacobi_eigenvalues(&jtj);
    let lam_max = eig.iter().cloned().fold(f64::MIN, f64::max);
    let lam_min = eig.iter().cloned().fold(f64::MAX, f64::min);
    let condition_number = if lam_min > 0.0 {
        lam_max / lam_min
    } else {
        f64::INFINITY
    };

    let dof = n.saturating_sub(m);
    let sigma2 = if dof > 0 { cost / dof as f64 } else { f64::NAN };
    let covariance = match invert_sym(&jtj) {
        Some(inv) => inv
            .iter()
            .map(|row| row.iter().map(|v| v * sigma2).collect())
            .collect(),
        None => vec![vec![f64::INFINITY; m]; m],
    };
    let std_errors: Vec<f64> = (0..m)
        .map(|i| {
            let v = covariance[i][i];
            if v.is_finite() && v >= 0.0 {
                v.sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let mut correlations = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let denom = std_errors[i] * std_errors[j];
            correlations[i][j] = if denom.is_finite() && denom > 0.0 {
                (covariance[i][j] / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
    }

    let mut weak_pairs = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            if correlations[i][j].abs() > 0.95 {
                weak_pairs.push((i, j, correlations[i][j].abs()));
            }
        }
    }
    let weak_params: Vec<usize> = (0..m)
        .filter(|&i| std_errors[i] > 0.5 * params[i].abs() && params[i].abs() > 0.0)
        .collect();
    let bound_pinned: Vec<usize> = (0..m)
        .filter(|&i| {
            let (lo, hi) = bounds[i];
            near(params[i], lo) || near(params[i], hi)
        })
        .collect();

    LmResult {
        params,
        cost,
        initial_cost,
        n_residuals: n,
        iterations,
        converged,
        jtj,
        condition_number,
        covariance,
        std_errors,
        correlations,
        weak_pairs,
        weak_params,
        bound_pinned,
    }
}

fn sum_sq(r: &[f64]) -> f64 {
    r.iter().map(|x| x * x).sum()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * (1.0 + b.abs())
}

/// Solve `A x = b` (general, partial-pivoting Gaussian elimination). `None` if
/// singular.
pub fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > best {
                best = aug[row][col].abs();
                pivot = row;
            }
        }
        if best < 1e-14 {
            return None;
        }
        aug.swap(col, pivot);
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        x[i] = s / aug[i][i];
    }
    Some(x)
}

/// Invert a symmetric matrix via Gauss-Jordan. `None` if singular.
fn invert_sym(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    // Augment [A | I].
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();

    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col][col].abs();
        for row in (col + 1)..n {
            if m[row][col].abs() > best {
                best = m[row][col].abs();
                pivot = row;
            }
        }
        if best < 1e-14 {
            return None;
        }
        m.swap(col, pivot);
        let d = m[col][col];
        for j in 0..2 * n {
            m[col][j] /= d;
        }
        for row in 0..n {
            if row != col {
                let f = m[row][col];
                if f != 0.0 {
                    for j in 0..2 * n {
                        m[row][j] -= f * m[col][j];
                    }
                }
            }
        }
    }
    Some(m.iter().map(|row| row[n..2 * n].to_vec()).collect())
}

/// Eigenvalues of a small symmetric matrix via the cyclic Jacobi method.
fn jacobi_eigenvalues(a: &[Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a[0][0]];
    }
    let mut m: Vec<Vec<f64>> = a.to_vec();
    for _sweep in 0..100 {
        // Largest off-diagonal magnitude.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i][j] * m[i][j];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols p,q.
                for k in 0..n {
                    let mkp = m[k][p];
                    let mkq = m[k][q];
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p][k];
                    let mqk = m[q][k];
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
            }
        }
    }
    (0..n).map(|i| m[i][i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear model r_i = y_i - (a*x_i + b); recovering (a,b) is a convex LSQ
    /// with a known closed-form answer.
    struct Line {
        xs: Vec<f64>,
        ys: Vec<f64>,
    }
    impl ResidualProvider for Line {
        fn residuals(&self, p: &[f64]) -> Vec<f64> {
            self.xs
                .iter()
                .zip(&self.ys)
                .map(|(&x, &y)| y - (p[0] * x + p[1]))
                .collect()
        }
        fn bounds(&self) -> Vec<(f64, f64)> {
            vec![(-100.0, 100.0), (-100.0, 100.0)]
        }
    }

    #[test]
    fn recovers_line() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let (a, b) = (2.5, -1.3);
        let ys: Vec<f64> = xs.iter().map(|&x| a * x + b).collect();
        let line = Line { xs, ys };
        let res = levenberg_marquardt(&line, &[0.0, 0.0], &LmConfig::default());
        assert!((res.params[0] - a).abs() < 1e-6, "a={}", res.params[0]);
        assert!((res.params[1] - b).abs() < 1e-6, "b={}", res.params[1]);
        assert!(res.cost < 1e-12);
        assert!(res.converged);
        // Perfect fit ⇒ tiny std errors.
        assert!(res.std_errors[0] < 1e-3);
    }

    #[test]
    fn bounds_are_enforced() {
        // Truth a=2.5 but cap a at 1.0 ⇒ the fit must pin to the bound.
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.5 * x - 1.3).collect();
        struct Capped {
            xs: Vec<f64>,
            ys: Vec<f64>,
        }
        impl ResidualProvider for Capped {
            fn residuals(&self, p: &[f64]) -> Vec<f64> {
                self.xs
                    .iter()
                    .zip(&self.ys)
                    .map(|(&x, &y)| y - (p[0] * x + p[1]))
                    .collect()
            }
            fn bounds(&self) -> Vec<(f64, f64)> {
                vec![(-1.0, 1.0), (-100.0, 100.0)]
            }
        }
        let m = Capped { xs, ys };
        let res = levenberg_marquardt(&m, &[0.0, 0.0], &LmConfig::default());
        assert!(res.params[0] <= 1.0 + 1e-9);
        assert!(
            res.bound_pinned.contains(&0),
            "param 0 should be bound-pinned"
        );
    }

    #[test]
    fn detects_correlated_parameters() {
        // r_i = y_i - (a + b) * x_i : a and b are perfectly non-identifiable
        // individually (only their sum matters) ⇒ correlation ≈ -1.
        struct Sum {
            xs: Vec<f64>,
            ys: Vec<f64>,
        }
        impl ResidualProvider for Sum {
            fn residuals(&self, p: &[f64]) -> Vec<f64> {
                self.xs
                    .iter()
                    .zip(&self.ys)
                    .map(|(&x, &y)| y - (p[0] + p[1]) * x)
                    .collect()
            }
            fn bounds(&self) -> Vec<(f64, f64)> {
                vec![(-100.0, 100.0), (-100.0, 100.0)]
            }
        }
        let xs: Vec<f64> = (1..15).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 3.0 * x).collect();
        let m = Sum { xs, ys };
        let res = levenberg_marquardt(&m, &[0.5, 0.5], &LmConfig::default());
        assert!((res.params[0] + res.params[1] - 3.0).abs() < 1e-4);
        assert!(
            !res.weak_pairs.is_empty(),
            "should flag the correlated pair"
        );
        assert!(res.condition_number > 1e6, "cond {}", res.condition_number);
    }

    #[test]
    fn eig_and_solve_basics() {
        let a = [vec![2.0, 0.0], vec![0.0, 3.0]];
        let mut e = jacobi_eigenvalues(&a);
        e.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((e[0] - 2.0).abs() < 1e-9 && (e[1] - 3.0).abs() < 1e-9);
        let x = solve_linear(&[vec![2.0, 1.0], vec![1.0, 3.0]], &[3.0, 5.0]).unwrap();
        // 2a+b=3, a+3b=5 -> a=0.8, b=1.4
        assert!((x[0] - 0.8).abs() < 1e-9 && (x[1] - 1.4).abs() < 1e-9);
    }
}
