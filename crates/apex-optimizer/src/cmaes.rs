//! Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
//!
//! A derivative-free optimizer for continuous, possibly non-convex
//! optimization problems with 5-50 parameters. Used for car setup
//! optimization where the objective (lap time) is expensive to evaluate.
//!
//! This is the *separable* variant (sep-CMA-ES): the covariance matrix is
//! diagonal, so sampling and adaptation are O(n) per coordinate and no matrix
//! decomposition is needed. It works well for moderate dimensions (≈10-30) and
//! axis-aligned problems.
//!
//! ## Coordinates
//!
//! Internally the optimizer works in a normalized space where each dimension is
//! mapped to `[0, 1]` via its bounds (`normalized = (real - min) / (max - min)`).
//! This makes the standard CMA-ES constants (which assume a roughly unit-scale,
//! isotropic search) valid even when the real parameters span very different
//! magnitudes, and lets [`CmaEsConfig::initial_sigma`] be interpreted directly
//! as a fraction of each parameter's range. [`CmaEs::ask`] returns candidates in
//! real coordinates; [`CmaEs::best_params`] is likewise in real coordinates.

use std::cmp::Ordering;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// CMA-ES configuration.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Population size (number of candidates per generation).
    /// `None` selects the default `4 + floor(3 * ln(n))`.
    pub population_size: Option<usize>,
    /// Initial step size (fraction of each parameter's range). Default: 0.3.
    pub initial_sigma: f64,
    /// Maximum number of generations. Default: 100.
    pub max_generations: usize,
    /// Stop if the best fitness has not improved by this much over the last 10
    /// generations. Default: 1e-4.
    pub stagnation_threshold: f64,
    /// Stop if the step size drops below this value. Default: 1e-8.
    pub min_sigma: f64,
    /// Seed for the internal RNG used to sample candidates. Default: 42.
    /// A fixed seed makes an optimization run reproducible.
    pub seed: u64,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            population_size: None, // auto from dimension
            initial_sigma: 0.3,
            max_generations: 100,
            stagnation_threshold: 1e-4,
            min_sigma: 1e-8,
            seed: 42,
        }
    }
}

impl apex_math::ContentHash for CmaEsConfig {
    /// Encode the result-determining fields. `seed` is deliberately EXCLUDED
    /// (bound to `_`) so two configs differing only in seed hash identically —
    /// the content hash is seed-independent, and the seed is tracked
    /// separately from config identity. The destructure forces any new field
    /// to be handled here before it compiles.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let CmaEsConfig {
            population_size,
            initial_sigma,
            max_generations,
            stagnation_threshold,
            min_sigma,
            seed: _, // excluded from content identity by design
        } = self;
        // Option<usize>: tag 0 = None, tag 1 = Some(v).
        match population_size {
            None => w.tag(0),
            Some(v) => {
                w.tag(1);
                w.usize(*v);
            }
        }
        w.f64(*initial_sigma);
        w.usize(*max_generations);
        w.f64(*stagnation_threshold);
        w.f64(*min_sigma);
    }
}

/// CMA-ES optimizer state (separable / diagonal covariance).
pub struct CmaEs {
    /// Problem dimension.
    dim: usize,
    /// Population size (lambda).
    lambda: usize,
    /// Number of parents (top candidates used for updates).
    mu: usize,
    /// Normalized recombination weights (sum to 1).
    weights: Vec<f64>,
    /// Variance-effective selection mass.
    mueff: f64,
    /// Mean of the search distribution (normalized coordinates).
    mean: Vec<f64>,
    /// Overall step size.
    sigma: f64,
    /// Diagonal of the covariance matrix (per-dimension variances).
    cov: Vec<f64>,
    /// Evolution path for step-size adaptation.
    ps: Vec<f64>,
    /// Evolution path for covariance adaptation.
    pc: Vec<f64>,
    /// Step-size cumulation rate.
    cs: f64,
    /// Step-size damping.
    damps: f64,
    /// Covariance-path cumulation rate.
    cc: f64,
    /// Rank-one covariance learning rate.
    c1: f64,
    /// Rank-mu covariance learning rate.
    cmu: f64,
    /// Expected length of a standard normal vector, `E||N(0, I)||`.
    chi_n: f64,
    /// Generation counter.
    generation: usize,
    /// Best fitness seen so far.
    best_fitness: f64,
    /// Best parameters seen so far (real coordinates).
    best_params: Vec<f64>,
    /// Best fitness at the end of each generation (for stagnation detection).
    fitness_history: Vec<f64>,
    /// Candidates from the most recent [`ask`](CmaEs::ask) (normalized).
    candidates: Vec<Vec<f64>>,
    /// Parameter bounds `(min, max)` per dimension (real coordinates).
    bounds: Vec<(f64, f64)>,
    /// Configuration.
    config: CmaEsConfig,
    /// Random number generator (seeded for reproducibility).
    rng: StdRng,
}

impl CmaEs {
    /// Create a new CMA-ES optimizer.
    ///
    /// `initial_mean` is the starting point (typically the baseline parameters)
    /// in real coordinates. `bounds` are `(min, max)` for each dimension and must
    /// have the same length as `initial_mean`.
    pub fn new(initial_mean: Vec<f64>, bounds: Vec<(f64, f64)>, config: CmaEsConfig) -> Self {
        let dim = initial_mean.len();
        let n = dim as f64;

        let lambda = config
            .population_size
            .unwrap_or_else(|| 4 + (3.0 * n.ln()).floor() as usize)
            .max(2);
        let mu = (lambda / 2).max(1);

        // Super-linear recombination weights, normalized to sum to 1.
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let wsum: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= wsum;
        }
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Standard CMA-ES adaptation constants (Hansen, "The CMA Evolution
        // Strategy: A Tutorial").
        let cs = (mueff + 2.0) / (n + mueff + 5.0);
        let cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n);
        let c1 = 2.0 / ((n + 1.3).powi(2) + mueff);
        let cmu = (2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0).powi(2) + mueff))
            .min(1.0 - c1)
            .max(0.0);
        let damps = 1.0 + 2.0 * (((mueff - 1.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let chi_n = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));

        // Normalize the initial mean into [0, 1] per dimension.
        let mean: Vec<f64> = initial_mean
            .iter()
            .zip(&bounds)
            .map(|(&x, &(lo, hi))| {
                let range = (hi - lo).max(1e-12);
                ((x - lo) / range).clamp(0.0, 1.0)
            })
            .collect();

        Self {
            dim,
            lambda,
            mu,
            weights,
            mueff,
            mean,
            sigma: config.initial_sigma,
            cov: vec![1.0; dim],
            ps: vec![0.0; dim],
            pc: vec![0.0; dim],
            cs,
            damps,
            cc,
            c1,
            cmu,
            chi_n,
            generation: 0,
            best_fitness: f64::INFINITY,
            best_params: initial_mean,
            fitness_history: Vec::new(),
            candidates: Vec::new(),
            bounds,
            rng: StdRng::seed_from_u64(config.seed),
            config,
        }
    }

    /// Map a normalized point to real coordinates.
    fn denormalize(&self, xn: &[f64]) -> Vec<f64> {
        xn.iter()
            .zip(&self.bounds)
            .map(|(&v, &(lo, hi))| lo + v * (hi - lo))
            .collect()
    }

    /// Sample the next generation of candidate solutions.
    ///
    /// Returns `lambda` candidate parameter vectors in real coordinates to be
    /// evaluated. The normalized samples are retained internally for the
    /// subsequent [`tell`](CmaEs::tell).
    // Per-coordinate loops index several parallel arrays (mean, cov, ...) by the
    // same index, so a range loop is clearer than zipping them all.
    #[allow(clippy::needless_range_loop)]
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        let mut norm = Vec::with_capacity(self.lambda);
        for _ in 0..self.lambda {
            let mut x = vec![0.0; self.dim];
            for i in 0..self.dim {
                let z: f64 = self.rng.sample(StandardNormal);
                let step = self.sigma * self.cov[i].max(0.0).sqrt() * z;
                x[i] = (self.mean[i] + step).clamp(0.0, 1.0);
            }
            norm.push(x);
        }
        let real: Vec<Vec<f64>> = norm.iter().map(|xn| self.denormalize(xn)).collect();
        self.candidates = norm;
        real
    }

    /// Report fitness values for the candidates from the last [`ask`](CmaEs::ask).
    ///
    /// `fitnesses` must have length `lambda` and correspond 1:1 to the candidates
    /// returned by [`ask`](CmaEs::ask). Updates the distribution (mean, step
    /// size, and diagonal covariance) and the best-so-far record. If `fitnesses`
    /// is the wrong length, or [`ask`](CmaEs::ask) was not called first, the call
    /// is a no-op.
    // Per-coordinate loops index several parallel arrays (mean, old_mean, ps,
    // pc, cov, ...) by the same index, so range loops are clearer than zipping.
    #[allow(clippy::needless_range_loop)]
    pub fn tell(&mut self, fitnesses: &[f64]) {
        if self.candidates.len() != self.lambda || fitnesses.len() != self.lambda {
            return;
        }

        // Rank candidates by fitness (ascending = best first).
        let mut order: Vec<usize> = (0..self.lambda).collect();
        order.sort_by(|&a, &b| {
            fitnesses[a]
                .partial_cmp(&fitnesses[b])
                .unwrap_or(Ordering::Equal)
        });

        let old_mean = self.mean.clone();
        let sigma_old = self.sigma;

        // Recombination: weighted mean of the top mu candidates.
        let mut new_mean = vec![0.0; self.dim];
        for (k, &w) in self.weights.iter().enumerate() {
            let cand = &self.candidates[order[k]];
            for i in 0..self.dim {
                new_mean[i] += w * cand[i];
            }
        }
        self.mean = new_mean;

        // Step-size evolution path and cumulative step-size adaptation (CSA).
        let cs_factor = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        let mut ps_norm2 = 0.0;
        for i in 0..self.dim {
            let denom = sigma_old * self.cov[i].max(1e-30).sqrt();
            let disp = (self.mean[i] - old_mean[i]) / denom;
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + cs_factor * disp;
            ps_norm2 += self.ps[i] * self.ps[i];
        }
        let ps_norm = ps_norm2.sqrt();
        self.sigma *= ((self.cs / self.damps) * (ps_norm / self.chi_n - 1.0)).exp();

        // Covariance evolution path and diagonal covariance update.
        let pc_factor = (self.cc * (2.0 - self.cc) * self.mueff).sqrt();
        for i in 0..self.dim {
            let disp = (self.mean[i] - old_mean[i]) / sigma_old;
            self.pc[i] = (1.0 - self.cc) * self.pc[i] + pc_factor * disp;

            // Rank-mu: weighted variance of the selected steps.
            let mut rank_mu = 0.0;
            for (k, &w) in self.weights.iter().enumerate() {
                let y = (self.candidates[order[k]][i] - old_mean[i]) / sigma_old;
                rank_mu += w * y * y;
            }

            self.cov[i] = (1.0 - self.c1 - self.cmu) * self.cov[i]
                + self.c1 * self.pc[i] * self.pc[i]
                + self.cmu * rank_mu;
            // Guard against numerical drift to non-positive variance.
            self.cov[i] = self.cov[i].max(1e-30);
        }

        // Track the best candidate seen (in real coordinates).
        let best_idx = order[0];
        if fitnesses[best_idx] < self.best_fitness {
            self.best_fitness = fitnesses[best_idx];
            self.best_params = self.denormalize(&self.candidates[best_idx]);
        }

        self.generation += 1;
        self.fitness_history.push(self.best_fitness);
    }

    /// Current generation number.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Best fitness found so far.
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Best parameters found so far (real coordinates).
    pub fn best_params(&self) -> &[f64] {
        &self.best_params
    }

    /// Current step size.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Population size (number of candidates per generation).
    pub fn population_size(&self) -> usize {
        self.lambda
    }

    /// Number of parents (top candidates recombined into the new mean).
    pub fn num_parents(&self) -> usize {
        self.mu
    }

    /// Whether the optimizer should stop.
    ///
    /// Returns true when the generation budget is exhausted, the step size has
    /// collapsed below [`CmaEsConfig::min_sigma`], or the best fitness has
    /// stagnated (improved by less than [`CmaEsConfig::stagnation_threshold`]
    /// over the last 10 generations).
    pub fn should_stop(&self) -> bool {
        if self.generation >= self.config.max_generations {
            return true;
        }
        if self.sigma < self.config.min_sigma {
            return true;
        }
        let h = &self.fitness_history;
        if h.len() >= 10 {
            let past = h[h.len() - 10];
            if (past - self.best_fitness).abs() < self.config.stagnation_threshold {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function: f(x) = sum(x_i^2), global minimum 0 at the origin.
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    /// 2-D Rosenbrock: f(x, y) = (1 - x)^2 + 100 (y - x^2)^2, minimum 0 at (1, 1).
    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    /// Run a full ask/tell optimization loop on `f` and return the optimizer.
    fn optimize(
        f: impl Fn(&[f64]) -> f64,
        mean: Vec<f64>,
        bounds: Vec<(f64, f64)>,
        config: CmaEsConfig,
    ) -> CmaEs {
        let max_gen = config.max_generations;
        let mut es = CmaEs::new(mean, bounds, config);
        for _ in 0..max_gen {
            let candidates = es.ask();
            let fitnesses: Vec<f64> = candidates.iter().map(|c| f(c)).collect();
            es.tell(&fitnesses);
        }
        es
    }

    #[test]
    fn test_cmaes_sphere() {
        let config = CmaEsConfig {
            max_generations: 50,
            ..Default::default()
        };
        let es = optimize(sphere, vec![5.0, 5.0, 5.0], vec![(-10.0, 10.0); 3], config);
        assert!(
            es.best_fitness() < 0.1,
            "sphere should converge below 0.1, got {}",
            es.best_fitness()
        );
    }

    #[test]
    fn test_cmaes_rosenbrock() {
        let config = CmaEsConfig {
            max_generations: 100,
            ..Default::default()
        };
        let es = optimize(rosenbrock, vec![-1.0, -1.0], vec![(-5.0, 5.0); 2], config);
        assert!(
            es.best_fitness() < 1.0,
            "rosenbrock should reach below 1.0, got {}",
            es.best_fitness()
        );
    }

    #[test]
    fn test_cmaes_respects_bounds() {
        // Sphere with all dimensions bounded to [0, 10]: the unconstrained
        // optimum (the origin) sits at the lower bound, so the result should be
        // near 0 and never negative.
        let config = CmaEsConfig {
            max_generations: 60,
            ..Default::default()
        };
        let es = optimize(sphere, vec![5.0, 5.0, 5.0], vec![(0.0, 10.0); 3], config);
        for (i, &v) in es.best_params().iter().enumerate() {
            assert!(v >= 0.0, "param {i} = {v} should not be negative");
            assert!(v < 0.5, "param {i} = {v} should be near the lower bound");
        }
    }

    #[test]
    fn test_cmaes_ask_returns_correct_count() {
        let es_config = CmaEsConfig::default();
        let mut es = CmaEs::new(vec![0.0; 5], vec![(-1.0, 1.0); 5], es_config);
        let lambda = es.population_size();
        let candidates = es.ask();
        assert_eq!(candidates.len(), lambda);
        for c in &candidates {
            assert_eq!(c.len(), 5);
        }
    }

    #[test]
    fn test_cmaes_should_stop() {
        let config = CmaEsConfig {
            max_generations: 200,
            ..Default::default()
        };
        let mut es = CmaEs::new(vec![5.0, 5.0], vec![(-10.0, 10.0); 2], config);
        let mut gens = 0;
        while !es.should_stop() {
            let candidates = es.ask();
            let fitnesses: Vec<f64> = candidates.iter().map(|c| sphere(c)).collect();
            es.tell(&fitnesses);
            gens += 1;
            assert!(gens <= 200, "optimizer failed to terminate");
        }
        assert!(es.should_stop(), "should_stop must hold after the loop");
    }

    #[test]
    fn test_cmaes_same_seed_reproducible() {
        // Two optimizers built with the same seed must emit byte-identical
        // candidate sequences across successive asks.
        let make = || {
            CmaEs::new(
                vec![0.0; 4],
                vec![(-1.0, 1.0); 4],
                CmaEsConfig {
                    seed: 7,
                    ..Default::default()
                },
            )
        };
        let mut a = make();
        let mut b = make();
        for gen in 0..5 {
            assert_eq!(
                a.ask(),
                b.ask(),
                "same seed must yield identical candidates (generation {gen})"
            );
        }
    }

    #[test]
    fn test_cmaes_different_seed_differs() {
        // A different seed must actually reach the RNG: the first generation of
        // candidates should differ, proving the seed is wired through.
        let mut a = CmaEs::new(
            vec![0.0; 4],
            vec![(-1.0, 1.0); 4],
            CmaEsConfig {
                seed: 1,
                ..Default::default()
            },
        );
        let mut b = CmaEs::new(
            vec![0.0; 4],
            vec![(-1.0, 1.0); 4],
            CmaEsConfig {
                seed: 2,
                ..Default::default()
            },
        );
        assert_ne!(
            a.ask(),
            b.ask(),
            "different seeds must yield different candidates"
        );
    }

    #[test]
    fn test_cmaes_tell_wrong_length_is_noop() {
        let mut es = CmaEs::new(vec![0.0; 3], vec![(-1.0, 1.0); 3], CmaEsConfig::default());
        es.ask();
        let gen_before = es.generation();
        es.tell(&[0.0]); // wrong length
        assert_eq!(es.generation(), gen_before, "bad tell should be a no-op");
    }
}
