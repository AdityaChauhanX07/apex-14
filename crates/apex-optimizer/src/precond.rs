//! Block-structured preconditioners for the interior-point solver's condensed
//! SPD Newton system.
//!
//! The interior-point solver ([`crate::ipm`]) condenses its Newton system to a
//! symmetric positive-definite operator
//!
//! ```text
//! M = w_f·H_f + rho·JeqᵀJeq + Jineqᵀ Σ_I Jineq + diag(Σ_L + Σ_U) + reg·I
//! ```
//!
//! and solves it matrix-free by preconditioned conjugate gradient. Until now the
//! only preconditioner was scalar (Jacobi) `1/diag(M)`, and
//! `docs/design/envelope-qss/real-track-convergence.md` §B.2 established that it
//! is exhausted: the periodic first-difference collocation Jacobian gives
//! `JeqᵀJeq` a **graph-Laplacian** structure whose condition number grows like
//! `N²`, CG saturates its iteration cap at every mesh, and the primal freezes.
//!
//! # Why block-tridiagonal
//!
//! A trapezoidal collocation defect couples exactly **two adjacent nodes**. So
//! if the decision variables are grouped by node, `JeqᵀJeq` — and therefore `M` —
//! is **block-tridiagonal** (plus a periodic corner from the flying-lap wrap).
//! Block-tridiagonal is the sparsity of a 1-D Laplacian, and a block-Thomas
//! (block-LDLᵀ) sweep inverts it **exactly**. That is the point: the documented
//! `N²` growth exists *because* the operator is a 1-D Laplacian, so a
//! preconditioner that inverts that structure exactly should leave a
//! preconditioned system whose conditioning does not grow with `N`.
//!
//! Whether it actually does is a measurement, not an assertion — see
//! `docs/design/dynamic-ocp/kkt-precond.md` for the CG-iteration-versus-`N`
//! table this module was built to produce.
//!
//! # Layout: index mapping, not physical reordering
//!
//! The envelope OCP (and the collocation NLP) lay variables out
//! **block-contiguous by quantity** (`[n | xi | v | a_x | kappa]`), so a node's
//! variables sit at stride `N`. Rather than repack them node-major — which would
//! permute every floating-point reduction in the solver and so move every
//! committed number on the *default* path — this module takes an explicit
//! [`BlockStructure`] (a list of global variable indices per node) and
//! gathers/scatters through it. The arithmetic the Jacobi path performs is
//! byte-for-byte unchanged.
//!
//! # The periodic wrap — kept, via Woodbury
//!
//! A flying lap closes the mesh, so the last node couples back to the first and
//! the operator is block-tridiagonal *plus a corner*. `recon.md` §3.2 proposed
//! **dropping** that corner, reasoning that a rank-`nb` perturbation out of
//! `N·nb` is asymptotically irrelevant.
//!
//! **That was measured and it is wrong.** Dropping the corner turns a periodic
//! ring into a line, and the inverse of a tridiagonal operator is *dense* — so a
//! structurally tiny change to the operator is a global change to its inverse.
//! Measured on real Silverstone at `N = 32`, the wrap-dropped preconditioner had
//! a relative inverse error of `3e-1` rising to `2.3` as `rho` grew, CG still
//! saturated its cap, and the solve was **worse** than plain Jacobi. The
//! reasoning error was about *size* where it needed to be about *conditioning*.
//!
//! So the corner is retained **exactly**, by a Sherman-Morrison-Woodbury
//! correction on top of the line factorization. Writing the corner as
//! `Z S Zᵀ` with `Z` selecting the first and last blocks and
//! `S = [[0, W], [Wᵀ, 0]]`,
//!
//! ```text
//! M⁻¹ = T⁻¹ - (T⁻¹Z) (I + S·ZᵀT⁻¹Z)⁻¹ S (ZᵀT⁻¹)
//! ```
//!
//! which needs `2·nb` extra line solves to form `T⁻¹Z` and one dense
//! `2nb x 2nb` solve — and never inverts `S`, which may well be singular. The
//! result inverts `M` exactly (to round-off) for the collocation problems this
//! targets, so CG should converge in a handful of iterations rather than
//! saturating. `preconditioner_inverts_operator_exactly` locks that.
//!
//! # No new dependencies
//!
//! Dense per-block inverses come from [`apex_math::solve_linear`] (partial-
//! pivoting Gaussian elimination, already public). Everything else is `f64`
//! arithmetic in fixed sequential order: no threads, no allocation-order
//! dependence, no RNG — so the determinism contract `ipm` documents carries over
//! unchanged.

use apex_math::lm::solve_linear;
use apex_math::CsrMatrix;

/// Node-contiguous grouping of decision variables, supplied by an
/// [`NlpEvaluator`](crate::nlp::NlpEvaluator) that has collocation structure.
///
/// `blocks[b]` lists the global variable indices belonging to node `b`, in a
/// fixed per-node order (the same quantity must occupy the same slot in every
/// block). Blocks must be **uniform in size**, **disjoint**, and together
/// **cover every variable** — [`BlockStructure::validate`] checks all three, and
/// the solver silently falls back to Jacobi if any fails, so a malformed
/// structure degrades rather than corrupts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockStructure {
    /// Per-node global variable indices.
    pub blocks: Vec<Vec<usize>>,
}

impl BlockStructure {
    /// Build a structure whose block `b` is `[stride_starts[i] + b; ...]` — the
    /// block-contiguous-by-quantity layout used by the envelope OCP and the
    /// collocation NLP, where quantity `i` occupies `[i*n_nodes, (i+1)*n_nodes)`.
    pub fn strided(n_nodes: usize, n_quantities: usize) -> BlockStructure {
        let blocks = (0..n_nodes)
            .map(|b| (0..n_quantities).map(|q| q * n_nodes + b).collect())
            .collect();
        BlockStructure { blocks }
    }

    /// Number of blocks (nodes).
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Uniform block size, or `None` if the blocks differ in length or there are
    /// none.
    pub fn block_size(&self) -> Option<usize> {
        let first = self.blocks.first()?.len();
        if first == 0 || self.blocks.iter().any(|b| b.len() != first) {
            return None;
        }
        Some(first)
    }

    /// Check that the blocks are uniform, in range, disjoint, and cover all
    /// `n_vars` variables. Returns the per-variable `(block, slot)` map on
    /// success.
    pub fn validate(&self, n_vars: usize) -> Option<Vec<(usize, usize)>> {
        let nb = self.block_size()?;
        if nb * self.n_blocks() != n_vars || self.n_blocks() < 2 {
            return None;
        }
        let mut map = vec![(usize::MAX, usize::MAX); n_vars];
        for (b, block) in self.blocks.iter().enumerate() {
            for (slot, &var) in block.iter().enumerate() {
                if var >= n_vars || map[var].0 != usize::MAX {
                    return None; // out of range, or assigned twice
                }
                map[var] = (b, slot);
            }
        }
        if map.iter().any(|&(b, _)| b == usize::MAX) {
            return None; // some variable uncovered
        }
        Some(map)
    }
}

/// Dense row-major `nb x nb` block.
type Block = Vec<f64>;

#[inline]
fn at(m: &[f64], nb: usize, i: usize, j: usize) -> f64 {
    m[i * nb + j]
}

/// `out = a * b` for row-major `nb x nb` blocks.
fn matmul(a: &[f64], b: &[f64], nb: usize) -> Block {
    let mut out = vec![0.0; nb * nb];
    for i in 0..nb {
        for k in 0..nb {
            let aik = at(a, nb, i, k);
            if aik == 0.0 {
                continue;
            }
            for j in 0..nb {
                out[i * nb + j] += aik * at(b, nb, k, j);
            }
        }
    }
    out
}

/// `out = aᵀ * b` for row-major `nb x nb` blocks.
fn matmul_t(a: &[f64], b: &[f64], nb: usize) -> Block {
    let mut out = vec![0.0; nb * nb];
    for k in 0..nb {
        for i in 0..nb {
            let aki = at(a, nb, k, i);
            if aki == 0.0 {
                continue;
            }
            for j in 0..nb {
                out[i * nb + j] += aki * at(b, nb, k, j);
            }
        }
    }
    out
}

/// `out = m * v` for a row-major `nb x nb` block.
fn matvec(m: &[f64], v: &[f64], nb: usize) -> Vec<f64> {
    m.chunks_exact(nb)
        .map(|row| row.iter().zip(v).map(|(&a, &b)| a * b).sum())
        .collect()
}

/// Average a block with its transpose. The assembled blocks are symmetric in
/// exact arithmetic; Gaussian-elimination inverses are not, and CG's convergence
/// theory needs a symmetric preconditioner, so both are symmetrized explicitly.
fn symmetrize(m: &mut [f64], nb: usize) {
    for i in 0..nb {
        for j in (i + 1)..nb {
            let avg = 0.5 * (m[i * nb + j] + m[j * nb + i]);
            m[i * nb + j] = avg;
            m[j * nb + i] = avg;
        }
    }
}

/// Dense inverse of a row-major `nb x nb` block, column by column via
/// [`solve_linear`]. `None` if singular. Makes no symmetry assumption.
fn invert_n(m: &[f64], nb: usize) -> Option<Block> {
    let rows: Vec<Vec<f64>> = (0..nb).map(|i| m[i * nb..(i + 1) * nb].to_vec()).collect();
    let mut inv = vec![0.0; nb * nb];
    for c in 0..nb {
        let mut e = vec![0.0; nb];
        e[c] = 1.0;
        let col = solve_linear(&rows, &e)?;
        for (r, &val) in col.iter().enumerate() {
            inv[r * nb + c] = val;
        }
    }
    Some(inv)
}

/// [`invert_n`] for a block known to be symmetric, symmetrizing the result.
fn invert(m: &[f64], nb: usize) -> Option<Block> {
    let mut inv = invert_n(m, nb)?;
    symmetrize(&mut inv, nb);
    Some(inv)
}

/// A factorized block-tridiagonal approximation of the condensed IP operator.
///
/// Built once per Newton step from the constraint Jacobians and barrier weights,
/// then applied once per CG iteration.
#[derive(Debug, Clone)]
pub struct BlockTridiag {
    nb: usize,
    n_blocks: usize,
    /// `blocks[b][slot]` -> global variable index.
    blocks: Vec<Vec<usize>>,
    /// Upper off-diagonal coupling `C_b = M[block b, block b+1]`, length `n_blocks - 1`.
    c: Vec<Block>,
    /// `L_b = C_{b-1}ᵀ D_{b-1}⁻¹`, indexed `1..n_blocks` (slot 0 unused).
    l: Vec<Block>,
    /// `D_b⁻¹`, length `n_blocks`.
    dinv: Vec<Block>,
    /// Periodic-wrap correction, absent when the mesh is not closed (or when
    /// `n_blocks == 2`, where the wrap coincides with the ordinary coupling and
    /// is already carried by `c[0]`).
    wrap: Option<WrapCorrection>,
}

/// Sherman-Morrison-Woodbury data restoring the periodic corner on top of the
/// line factorization. See the module docs for the identity.
#[derive(Debug, Clone)]
struct WrapCorrection {
    /// Global variable indices selected by `Z`: block 0's, then the last block's.
    z_idx: Vec<usize>,
    /// `S = [[0, W], [Wᵀ, 0]]`, row-major `2nb x 2nb`.
    s: Vec<f64>,
    /// Columns of `T⁻¹Z`; `tz[p]` has length `n_vars`.
    tz: Vec<Vec<f64>>,
    /// `(I + S·ZᵀT⁻¹Z)⁻¹`, row-major `2nb x 2nb`.
    kinv: Vec<f64>,
}

impl BlockTridiag {
    /// Assemble and factorize the preconditioner.
    ///
    /// Contributions, matching [`crate::ipm`]'s condensed operator term for term:
    /// - `rho · JeqᵀJeq` — each row of `j_eq` contributes the rank-1 outer product
    ///   `rho·vvᵀ`, which (because a defect row touches only adjacent nodes) lands
    ///   entirely inside the block-tridiagonal band. Captured **exactly**, except
    ///   for the deliberately dropped periodic-wrap cross terms (see module docs).
    /// - `Jineqᵀ Σ_I Jineq` — same treatment, weighted by `sig_i[row]`.
    /// - `diag_extra` — `Σ_L + Σ_U + reg`, added to the block diagonals.
    ///
    /// The objective Hessian `w_f·H_f` is **not** included: it is only available
    /// matrix-free and defaults to zero for the collocation problems this
    /// targets. Omitting a PSD term from a preconditioner is safe (it stays SPD)
    /// and is noted as a limitation in the design doc.
    ///
    /// Returns `None` if any diagonal block is numerically singular, so the
    /// caller can fall back to Jacobi deterministically.
    pub fn assemble(
        structure: &BlockStructure,
        var_map: &[(usize, usize)],
        j_eq: &CsrMatrix,
        rho: f64,
        j_ineq: &CsrMatrix,
        sig_i: &[f64],
        diag_extra: &[f64],
    ) -> Option<BlockTridiag> {
        let nb = structure.block_size()?;
        let n_blocks = structure.n_blocks();

        let mut diag: Vec<Block> = vec![vec![0.0; nb * nb]; n_blocks];
        let mut c: Vec<Block> = vec![vec![0.0; nb * nb]; n_blocks.saturating_sub(1)];
        // `W = M[block 0, block n_blocks-1]`, the periodic corner. Only a
        // distinct structure when `n_blocks > 2`; at `n_blocks == 2` the wrap
        // coincides with the ordinary `c[0]` coupling and is already carried
        // there.
        let has_wrap = n_blocks > 2;
        let mut w_corner: Block = vec![0.0; nb * nb];
        let last = n_blocks - 1;

        // Accumulate w · Jᵀ J over the rows of one Jacobian.
        let mut accumulate = |j: &CsrMatrix, weight: &dyn Fn(usize) -> f64| {
            for row in 0..j.nrows() {
                let w = weight(row);
                if w == 0.0 {
                    continue;
                }
                let (vals, cols) = j.row_entries(row);
                for (&vi, &ci) in vals.iter().zip(cols) {
                    let (bi, si) = var_map[ci];
                    for (&vj, &cj) in vals.iter().zip(cols) {
                        let (bj, sj) = var_map[cj];
                        let contrib = w * vi * vj;
                        if bi == bj {
                            diag[bi][si * nb + sj] += contrib;
                        } else if bj == bi + 1 {
                            // Upper coupling. The transposed pair (bj, bi) is
                            // visited on another iteration of this same double
                            // loop and is skipped, so `c[bi]` accrues exactly
                            // once — no double counting.
                            c[bi][si * nb + sj] += contrib;
                        } else if has_wrap && bi == 0 && bj == last {
                            // Periodic corner `M[0, last]`; its mirror
                            // `(last, 0)` is skipped for the same reason.
                            w_corner[si * nb + sj] += contrib;
                        }
                        // A coupling wider than one node would be dropped here.
                        // Trapezoidal collocation produces none; a wider stencil
                        // (e.g. Hermite-Simpson with midpoint variables) would
                        // need this revisited.
                    }
                }
            }
        };

        accumulate(j_eq, &|_| rho);
        accumulate(j_ineq, &|row| sig_i.get(row).copied().unwrap_or(0.0));

        for (var, &(b, slot)) in var_map.iter().enumerate() {
            diag[b][slot * nb + slot] += diag_extra[var];
        }
        for d in diag.iter_mut() {
            symmetrize(d, nb);
        }

        // Block LDLᵀ (block-Thomas) forward sweep.
        let mut dinv: Vec<Block> = Vec::with_capacity(n_blocks);
        let mut l: Vec<Block> = vec![vec![0.0; nb * nb]; n_blocks];
        let mut d_cur = diag[0].clone();
        symmetrize(&mut d_cur, nb);
        dinv.push(invert(&d_cur, nb)?);
        for b in 1..n_blocks {
            let cprev = &c[b - 1];
            // L_b = C_{b-1}ᵀ D_{b-1}⁻¹
            let lb = matmul_t(cprev, &dinv[b - 1], nb);
            // D_b = A_b - L_b C_{b-1}
            let lc = matmul(&lb, cprev, nb);
            let mut db: Block = (0..nb * nb).map(|i| diag[b][i] - lc[i]).collect();
            symmetrize(&mut db, nb);
            dinv.push(invert(&db, nb)?);
            l[b] = lb;
        }

        let mut bt = BlockTridiag {
            nb,
            n_blocks,
            blocks: structure.blocks.clone(),
            c,
            l,
            dinv,
            wrap: None,
        };

        // --- periodic-wrap (Woodbury) correction ---
        if has_wrap && w_corner.iter().any(|&v| v != 0.0) {
            let n_vars = var_map.len();
            let m2 = 2 * nb;
            // Z selects block 0's variables then the last block's.
            let mut z_idx = structure.blocks[0].clone();
            z_idx.extend_from_slice(&structure.blocks[last]);

            // S = [[0, W], [Wᵀ, 0]]
            let mut s = vec![0.0; m2 * m2];
            for i in 0..nb {
                for j in 0..nb {
                    s[i * m2 + (nb + j)] = w_corner[i * nb + j];
                    s[(nb + i) * m2 + j] = w_corner[j * nb + i];
                }
            }

            // T⁻¹Z: one line solve per column of Z.
            let mut tz: Vec<Vec<f64>> = Vec::with_capacity(m2);
            for p in 0..m2 {
                let mut e = vec![0.0; n_vars];
                e[z_idx[p]] = 1.0;
                let mut col = vec![0.0; n_vars];
                bt.line_solve(&e, &mut col);
                tz.push(col);
            }

            // G = Zᵀ(T⁻¹Z), then K = I + S·G.
            let mut g = vec![0.0; m2 * m2];
            for (q, col) in tz.iter().enumerate() {
                for (p, &gi) in z_idx.iter().enumerate() {
                    g[p * m2 + q] = col[gi];
                }
            }
            let sg = matmul(&s, &g, m2);
            let mut k = sg;
            for i in 0..m2 {
                k[i * m2 + i] += 1.0;
            }
            // A singular K means the corrected operator is singular; fall back
            // rather than emit garbage.
            let kinv = invert_n(&k, m2)?;

            bt.wrap = Some(WrapCorrection { z_idx, s, tz, kinv });
        }

        Some(bt)
    }

    /// Solve `T z = r` for the block-tridiagonal *line* part (wrap excluded), by
    /// block forward/back substitution.
    fn line_solve(&self, r: &[f64], z: &mut [f64]) {
        let nb = self.nb;
        // Gather the residual into per-block vectors.
        let mut y: Vec<Vec<f64>> = self
            .blocks
            .iter()
            .map(|block| block.iter().map(|&v| r[v]).collect())
            .collect();

        // Forward: y_b = r_b - L_b y_{b-1}
        for b in 1..self.n_blocks {
            let t = matvec(&self.l[b], &y[b - 1], nb);
            for (yi, ti) in y[b].iter_mut().zip(&t) {
                *yi -= ti;
            }
        }

        // Back: z_{last} = D⁻¹ y_{last};  z_b = D_b⁻¹ (y_b - C_b z_{b+1})
        let last = self.n_blocks - 1;
        let mut zb = matvec(&self.dinv[last], &y[last], nb);
        for (slot, &var) in self.blocks[last].iter().enumerate() {
            z[var] = zb[slot];
        }
        for b in (0..last).rev() {
            let cz = matvec(&self.c[b], &zb, nb);
            let rhs: Vec<f64> = y[b].iter().zip(&cz).map(|(&yi, &ci)| yi - ci).collect();
            zb = matvec(&self.dinv[b], &rhs, nb);
            for (slot, &var) in self.blocks[b].iter().enumerate() {
                z[var] = zb[slot];
            }
        }
    }

    /// Apply `z = M⁻¹ r`: a line solve, plus the Woodbury correction that
    /// restores the periodic corner.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        self.line_solve(r, z);
        let Some(w) = &self.wrap else { return };

        // z <- a - (T⁻¹Z) · K⁻¹ · S · (Zᵀ a),  with a = T⁻¹r already in `z`.
        let m2 = w.z_idx.len();
        let zt_a: Vec<f64> = w.z_idx.iter().map(|&i| z[i]).collect();
        let s_zt_a = matvec(&w.s, &zt_a, m2);
        let y = matvec(&w.kinv, &s_zt_a, m2);
        for (p, col) in w.tz.iter().enumerate() {
            let yp = y[p];
            if yp == 0.0 {
                continue;
            }
            for (zi, &ci) in z.iter_mut().zip(col.iter()) {
                *zi -= yp * ci;
            }
        }
    }

    /// Reconstruct, densely, the operator this preconditioner inverts —
    /// **test and diagnostics only**, `O(n_vars²)` memory.
    ///
    /// For a trapezoidal periodic collocation problem this is the *exact*
    /// condensed operator `M` (minus the objective Hessian, which is matrix-free
    /// and zero for these problems): every coupling is either intra-node,
    /// between adjacent nodes, or the periodic corner. A wider stencil would be
    /// silently truncated here, matching what [`BlockTridiag::assemble`] does.
    #[doc(hidden)]
    pub fn to_dense_operator(
        structure: &BlockStructure,
        var_map: &[(usize, usize)],
        j_eq: &CsrMatrix,
        rho: f64,
        j_ineq: &CsrMatrix,
        sig_i: &[f64],
        diag_extra: &[f64],
    ) -> Vec<Vec<f64>> {
        assert!(structure.block_size().is_some(), "uniform blocks required");
        let n = var_map.len();
        let last = structure.n_blocks() - 1;
        let mut dense = vec![vec![0.0; n]; n];
        let mut add = |j: &CsrMatrix, weight: &dyn Fn(usize) -> f64| {
            for row in 0..j.nrows() {
                let w = weight(row);
                let (vals, cols) = j.row_entries(row);
                for (&vi, &ci) in vals.iter().zip(cols) {
                    let (bi, _) = var_map[ci];
                    for (&vj, &cj) in vals.iter().zip(cols) {
                        let (bj, _) = var_map[cj];
                        let adjacent = bi == bj || bj == bi + 1 || bi == bj + 1;
                        let wrapped = (bi == 0 && bj == last) || (bj == 0 && bi == last);
                        if adjacent || wrapped {
                            dense[ci][cj] += w * vi * vj;
                        }
                    }
                }
            }
        };
        add(j_eq, &|_| rho);
        add(j_ineq, &|row| sig_i.get(row).copied().unwrap_or(0.0));
        for var in 0..n {
            dense[var][var] += diag_extra[var];
        }
        dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_math::CsrBuilder;

    fn strided_map(n_nodes: usize, nq: usize) -> (BlockStructure, Vec<(usize, usize)>) {
        let s = BlockStructure::strided(n_nodes, nq);
        let map = s.validate(n_nodes * nq).expect("valid structure");
        (s, map)
    }

    /// A small periodic trapezoidal-like equality Jacobian: each row couples
    /// node `i` and node `(i+1) % n`, exactly the envelope OCP's stencil.
    fn periodic_jeq(n_nodes: usize, nq: usize, seed: u64) -> CsrMatrix {
        let mut rng = seed;
        let mut next = || {
            // deterministic LCG, values in [-1, 1)
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let n_eq = nq * n_nodes;
        let mut b = CsrBuilder::new(n_eq, n_nodes * nq);
        for i in 0..n_nodes {
            let j = (i + 1) % n_nodes;
            for comp in 0..nq {
                let row = nq * i + comp;
                for q in 0..nq {
                    b.add(row, q * n_nodes + i, next());
                    b.add(row, q * n_nodes + j, next());
                }
            }
        }
        b.build()
    }

    fn node_local_jineq(n_nodes: usize, nq: usize) -> CsrMatrix {
        let mut b = CsrBuilder::new(n_nodes, n_nodes * nq);
        for k in 0..n_nodes {
            for q in 0..nq {
                b.add(k, q * n_nodes + k, 0.3 + 0.1 * q as f64);
            }
        }
        b.build()
    }

    #[test]
    fn strided_structure_validates_and_maps() {
        let (s, map) = strided_map(4, 3);
        assert_eq!(s.n_blocks(), 4);
        assert_eq!(s.block_size(), Some(3));
        // variable 6 = quantity 1, node 2 (quantity stride is n_nodes = 4)
        assert_eq!(map[6], (2, 1));
        // every variable covered exactly once
        assert!(map.iter().all(|&(b, _)| b != usize::MAX));
    }

    #[test]
    fn validate_rejects_malformed_structures() {
        // ragged blocks
        let ragged = BlockStructure {
            blocks: vec![vec![0, 1], vec![2]],
        };
        assert!(ragged.validate(3).is_none());
        // duplicate variable
        let dup = BlockStructure {
            blocks: vec![vec![0, 1], vec![1, 2]],
        };
        assert!(dup.validate(4).is_none());
        // wrong total size
        let (s, _) = strided_map(4, 3);
        assert!(s.validate(11).is_none());
        // out-of-range index
        let oob = BlockStructure {
            blocks: vec![vec![0, 9], vec![2, 3]],
        };
        assert!(oob.validate(4).is_none());
    }

    /// The assembled blocks must reproduce the banded part of the dense
    /// operator entry for entry — this is the correctness anchor for the
    /// assembly loop (including the no-double-counting claim on `c[b]`).
    #[test]
    fn block_assembly_matches_dense_reference() {
        let (n_nodes, nq) = (6, 3);
        let (s, map) = strided_map(n_nodes, nq);
        let j_eq = periodic_jeq(n_nodes, nq, 0xA11CE);
        let j_ineq = node_local_jineq(n_nodes, nq);
        let sig: Vec<f64> = (0..n_nodes).map(|k| 1.0 + 0.5 * k as f64).collect();
        let diag_extra: Vec<f64> = (0..n_nodes * nq).map(|i| 0.7 + 0.01 * i as f64).collect();
        let rho = 12.5;

        let bt = BlockTridiag::assemble(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra)
            .expect("assembles");
        let dense =
            BlockTridiag::to_dense_operator(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra);

        // The dense reference must be symmetric — the block form assumes it.
        let n = n_nodes * nq;
        for (i, row) in dense.iter().enumerate() {
            for (j, &vij) in row.iter().enumerate() {
                assert!(
                    (vij - dense[j][i]).abs() < 1e-9,
                    "dense operator not symmetric at ({i},{j})"
                );
            }
        }

        // The real check: applying the preconditioner must invert the dense
        // banded operator. M_approx · (M_approx⁻¹ r) == r.
        let r: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let mut z = vec![0.0; n];
        bt.apply(&r, &mut z);
        for (i, &ri) in r.iter().enumerate() {
            let mz: f64 = (0..n).map(|j| dense[i][j] * z[j]).sum();
            assert!(
                (mz - ri).abs() < 1e-6 * (1.0 + ri.abs()),
                "row {i}: M·z = {mz}, expected r = {ri}"
            );
        }
    }

    /// Block-Thomas must agree with a general dense solve of the *same*
    /// assembled system, to solver precision.
    #[test]
    fn block_thomas_matches_solve_linear() {
        let (n_nodes, nq) = (8, 3);
        let (s, map) = strided_map(n_nodes, nq);
        let j_eq = periodic_jeq(n_nodes, nq, 0xBEEF);
        let j_ineq = node_local_jineq(n_nodes, nq);
        let sig = vec![1.3; n_nodes];
        let diag_extra = vec![0.9; n_nodes * nq];
        let rho = 40.0;

        let bt = BlockTridiag::assemble(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra)
            .expect("assembles");
        let dense =
            BlockTridiag::to_dense_operator(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra);

        let n = n_nodes * nq;
        let r: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut z_block = vec![0.0; n];
        bt.apply(&r, &mut z_block);
        let z_dense = solve_linear(&dense, &r).expect("dense solve");

        for i in 0..n {
            assert!(
                (z_block[i] - z_dense[i]).abs() < 1e-7 * (1.0 + z_dense[i].abs()),
                "component {i}: block-Thomas {} vs dense {}",
                z_block[i],
                z_dense[i]
            );
        }
    }

    /// The assembled operator must be symmetric positive definite — the property
    /// CG requires of both the system and the preconditioner. It is PSD by
    /// construction (a sum of rank-1 outer products `w·vvᵀ`) and made strictly
    /// definite by the barrier/regularization diagonal; this verifies that
    /// holds numerically, including through the periodic corner.
    #[test]
    fn operator_is_spd() {
        let (n_nodes, nq) = (10, 3);
        let (s, map) = strided_map(n_nodes, nq);
        let j_eq = periodic_jeq(n_nodes, nq, 0xD15EA5E);
        let j_ineq = node_local_jineq(n_nodes, nq);
        let sig = vec![2.0; n_nodes];
        let diag_extra = vec![1e-3; n_nodes * nq]; // small: SPD must come from the PSD terms
        let rho = 100.0;

        let dense =
            BlockTridiag::to_dense_operator(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra);
        let n = n_nodes * nq;

        // symmetric
        for (i, row) in dense.iter().enumerate() {
            for (j, &vij) in row.iter().enumerate() {
                assert!((vij - dense[j][i]).abs() < 1e-9, "asymmetry at ({i},{j})");
            }
        }
        // positive definite on a spread of probe vectors
        for seed in 0..12u64 {
            let x: Vec<f64> = (0..n)
                .map(|i| (((i as u64 * 31 + seed * 17) % 23) as f64 - 11.0) + 0.5)
                .collect();
            let quad: f64 = (0..n)
                .map(|i| x[i] * (0..n).map(|j| dense[i][j] * x[j]).sum::<f64>())
                .sum();
            assert!(quad > 0.0, "xᵀMx = {quad} not positive for seed {seed}");
        }

        // and the factorization succeeds (no singular diagonal block)
        assert!(
            BlockTridiag::assemble(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra).is_some(),
            "SPD system should factorize"
        );
    }

    /// The headline property: with the periodic corner retained via Woodbury,
    /// the preconditioner inverts the operator **exactly** (to round-off), not
    /// approximately. This is the test that would have caught the wrap-dropping
    /// design error before it reached a measurement run — the same probe the
    /// solver's `verbose > 1` diagnostic reports.
    #[test]
    fn preconditioner_inverts_operator_exactly() {
        for &(n_nodes, nq) in &[(6usize, 3usize), (16, 5), (33, 5)] {
            let (s, map) = strided_map(n_nodes, nq);
            let j_eq = periodic_jeq(n_nodes, nq, 0x5EED + n_nodes as u64);
            let j_ineq = node_local_jineq(n_nodes, nq);
            let sig = vec![1.7; n_nodes];
            let diag_extra = vec![0.05; n_nodes * nq];
            let rho = 5e3; // stiff, as the AL penalty becomes late in a solve

            let bt = BlockTridiag::assemble(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra)
                .expect("assembles");
            let dense =
                BlockTridiag::to_dense_operator(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra);

            let n = n_nodes * nq;
            let v: Vec<f64> = (0..n).map(|i| ((i % 7) as f64 - 3.0) + 0.5).collect();
            // r = M v, then z = P^-1 r should recover v.
            let r: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| dense[i][j] * v[j]).sum())
                .collect();
            let mut z = vec![0.0; n];
            bt.apply(&r, &mut z);

            let num: f64 = z
                .iter()
                .zip(&v)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            let den: f64 = v.iter().map(|&b| b * b).sum::<f64>().sqrt();
            assert!(
                num / den < 1e-8,
                "N={n_nodes} nq={nq}: inverse error {:.3e} — the preconditioner is not exact",
                num / den
            );
        }
    }

    /// The periodic corner is load-bearing, not a rounding detail: a
    /// preconditioner built from a mesh whose wrap coupling has been zeroed
    /// out is a materially worse inverse of the true (wrapped) operator. This
    /// pins the measured finding that `recon.md`'s "drop the wrap" would have
    /// produced a preconditioner that does not invert what CG is solving.
    #[test]
    fn dropping_the_wrap_would_degrade_the_inverse() {
        let (n_nodes, nq) = (24, 5);
        let (s, map) = strided_map(n_nodes, nq);
        let j_eq = periodic_jeq(n_nodes, nq, 0xC0FFEE);
        let j_ineq = node_local_jineq(n_nodes, nq);
        let sig = vec![1.7; n_nodes];
        let diag_extra = vec![0.05; n_nodes * nq];
        let rho = 5e3;
        let n = n_nodes * nq;

        let dense =
            BlockTridiag::to_dense_operator(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra);
        let v: Vec<f64> = (0..n).map(|i| ((i % 7) as f64 - 3.0) + 0.5).collect();
        let r: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| dense[i][j] * v[j]).sum())
            .collect();

        let rel_err = |bt: &BlockTridiag| {
            let mut z = vec![0.0; n];
            bt.apply(&r, &mut z);
            let num: f64 = z
                .iter()
                .zip(&v)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            let den: f64 = v.iter().map(|&b| b * b).sum::<f64>().sqrt();
            num / den
        };

        let full = BlockTridiag::assemble(&s, &map, &j_eq, rho, &j_ineq, &sig, &diag_extra)
            .expect("assembles");
        let mut line_only = full.clone();
        line_only.wrap = None; // exactly what "drop the wrap" would build

        let e_full = rel_err(&full);
        let e_line = rel_err(&line_only);
        assert!(
            e_full < 1e-8,
            "wrapped preconditioner should be exact, got {e_full:.3e}"
        );
        assert!(
            e_line > 1e3 * e_full.max(1e-16),
            "dropping the wrap should visibly degrade the inverse: full {e_full:.3e} vs line-only {e_line:.3e}"
        );
    }

    /// Applying the preconditioner twice to the same input must be bitwise
    /// identical — the determinism contract, at the module level.
    #[test]
    fn apply_is_bitwise_deterministic() {
        let (n_nodes, nq) = (7, 3);
        let (s, map) = strided_map(n_nodes, nq);
        let j_eq = periodic_jeq(n_nodes, nq, 0xFEED);
        let j_ineq = node_local_jineq(n_nodes, nq);
        let sig = vec![1.1; n_nodes];
        let diag_extra = vec![0.4; n_nodes * nq];

        let build =
            || BlockTridiag::assemble(&s, &map, &j_eq, 7.0, &j_ineq, &sig, &diag_extra).unwrap();
        let a = build();
        let b = build();
        let n = n_nodes * nq;
        let r: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
        let (mut za, mut zb) = (vec![0.0; n], vec![0.0; n]);
        a.apply(&r, &mut za);
        b.apply(&r, &mut zb);
        for i in 0..n {
            assert_eq!(za[i].to_bits(), zb[i].to_bits(), "component {i} differs");
        }
    }

    /// A singular diagonal block must return `None` rather than producing
    /// garbage, so the solver can fall back deterministically.
    #[test]
    fn singular_block_returns_none() {
        let (n_nodes, nq) = (4, 2);
        let (s, map) = strided_map(n_nodes, nq);
        let empty_eq = CsrMatrix::zeros(0, n_nodes * nq);
        let empty_ineq = CsrMatrix::zeros(0, n_nodes * nq);
        // no constraint contribution and a zero diagonal => singular blocks
        let diag_extra = vec![0.0; n_nodes * nq];
        assert!(
            BlockTridiag::assemble(&s, &map, &empty_eq, 1.0, &empty_ineq, &[], &diag_extra)
                .is_none(),
            "singular assembly should return None"
        );
    }
}
