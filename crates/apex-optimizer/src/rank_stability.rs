//! Rank-correlation statistics for the setup-envelope rank-stability gate.
//!
//! The envelope-OCP lap time is constraint-tight but **not** mesh-converged
//! (see `docs/design/envelope-qss/real-track-convergence.md`): absolute lap
//! times swing ~20 % across node counts `N`. A CMA-ES setup search does not need
//! absolute accuracy — only that the *ranking* of setups is stable at a fixed
//! `N` (and does not reshuffle between the meshes that are tight-feasible). These
//! functions quantify that: Spearman's `rho` and Kendall's `tau` between two
//! lap-time vectors indexed by setup variant.
//!
//! See `docs/design/envelope-qss/setup-envelope.md` for the gate result.

/// Fractional ranks of `v` (0-based), with **ties averaged** (so tied values
/// share the mean of the ranks they span). Used by [`spearman`].
fn average_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).expect("NaN in rank input"));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Extent of the tie group [i, j).
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // Average of positions i..j (0-based).
        let avg = (i..j).sum::<usize>() as f64 / (j - i) as f64;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation of two equal-length vectors. Returns `0.0` if either has
/// zero variance (degenerate — no ordering information).
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "pearson: length mismatch");
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    let denom = (va * vb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
    }
}

/// Spearman's rank correlation `rho` between two lap-time vectors: Pearson
/// correlation of their tie-averaged ranks. `+1` = identical ordering, `-1` =
/// reversed, `0` = unrelated.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "spearman: length mismatch");
    pearson(&average_ranks(a), &average_ranks(b))
}

/// Kendall's `tau` (tau-b, tie-corrected) between two vectors: over all pairs,
/// `(concordant - discordant) / sqrt((P - Ta)(P - Tb))` where `P` is the total
/// pair count and `Ta`/`Tb` are the tied-pair counts within each input. `+1` =
/// identical ordering, `-1` = reversed.
pub fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "kendall_tau: length mismatch");
    let n = a.len();
    let (mut concordant, mut discordant) = (0i64, 0i64);
    let (mut ties_a, mut ties_b) = (0i64, 0i64);
    for i in 0..n {
        for j in (i + 1)..n {
            let da = (a[i] - a[j]).partial_cmp(&0.0).expect("NaN");
            let db = (b[i] - b[j]).partial_cmp(&0.0).expect("NaN");
            use std::cmp::Ordering::Equal;
            if da == Equal && db == Equal {
                ties_a += 1;
                ties_b += 1;
            } else if da == Equal {
                ties_a += 1;
            } else if db == Equal {
                ties_b += 1;
            } else if da == db {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total = (n * (n.saturating_sub(1)) / 2) as i64;
    let denom = (((total - ties_a) * (total - ties_b)) as f64).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (concordant - discordant) as f64 / denom
    }
}

/// The ordering of variant indices from best (smallest value) to worst.
/// Deterministic: ties break by index (stable sort).
pub fn ranking(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).expect("NaN in ranking input"));
    idx
}

/// Count of adjacent transpositions needed to turn ranking `a` into ranking `b`
/// — i.e. the number of discordant pairs (Kendall distance). `0` = identical
/// order; `1` = a single adjacent swap. A compact way to state the gate's
/// "at most one adjacent swap in N" criterion.
pub fn discordant_pairs(a: &[f64], b: &[f64]) -> usize {
    assert_eq!(a.len(), b.len(), "discordant_pairs: length mismatch");
    let n = a.len();
    let mut d = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let sa = (a[i] - a[j]).partial_cmp(&0.0).expect("NaN");
            let sb = (b[i] - b[j]).partial_cmp(&0.0).expect("NaN");
            use std::cmp::Ordering::Equal;
            if sa != Equal && sb != Equal && sa != sb {
                d += 1;
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_order_is_one() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 20.0, 30.0, 40.0, 50.0]; // monotone transform
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-12);
        assert!((kendall_tau(&a, &b) - 1.0).abs() < 1e-12);
        assert_eq!(discordant_pairs(&a, &b), 0);
    }

    #[test]
    fn reversed_order_is_minus_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        assert!((spearman(&a, &b) + 1.0).abs() < 1e-12);
        assert!((kendall_tau(&a, &b) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn one_adjacent_swap_in_eight() {
        // b swaps the two middle-ranked (near-tied) elements of a.
        // This is the exact gate scenario: >= 0.9 Spearman, one discordant pair.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0];
        assert_eq!(discordant_pairs(&a, &b), 1);
        // Spearman for one adjacent swap in n=8: 1 - 6*2/(8*63) = 0.97619...
        let rho = spearman(&a, &b);
        assert!(
            rho > 0.9,
            "one adjacent swap should keep rho > 0.9, got {rho}"
        );
        assert!((rho - 0.976190).abs() < 1e-4, "rho = {rho}");
        // Kendall tau: (28 - 2)/28 = 0.9286.
        let tau = kendall_tau(&a, &b);
        assert!((tau - 0.928571).abs() < 1e-4, "tau = {tau}");
    }

    #[test]
    fn spearman_handles_ties() {
        // Tied values must get averaged ranks; perfectly correlated with ties.
        let a = [1.0, 1.0, 2.0, 3.0];
        let b = [10.0, 10.0, 20.0, 30.0];
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ranking_is_deterministic_best_first() {
        let v = [3.0, 1.0, 2.0];
        assert_eq!(ranking(&v), vec![1, 2, 0]); // smallest (1.0 at idx 1) first
    }

    #[test]
    fn zero_variance_returns_zero() {
        let a = [5.0, 5.0, 5.0];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(spearman(&a, &b), 0.0);
        assert_eq!(kendall_tau(&a, &b), 0.0);
    }
}
