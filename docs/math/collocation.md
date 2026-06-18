# Direct Collocation - Formulation and Implementation

This document records the trajectory-optimization method implemented in Apex-14. It is a reference
appendix: it states the optimal control problem, its transcription to a finite nonlinear program
(NLP), and the structure that makes the NLP cheap to solve. Notation is plain Unicode and ASCII math,
consistent with `equations_of_motion.md`.

---

## 1. The Minimum-Time Problem

The goal is the fastest lap: find the state and control trajectories that traverse the track in
minimum time without violating physics or staying on the road.

```
minimize    T = ∫ dt           (total lap time)
subject to  ẋ = f(x, u, s)      (vehicle dynamics)
            −w_right(s) ≤ n ≤ w_left(s)   (track boundaries)
            ‖F_tire‖ ≤ F_grip(x)          (tire grip limit)
            x(0) = x(L)         (periodicity, for a closed lap)
```

The state `x = [s, n, v, α]` and control `u = [F_drive, κ_cmd]` are the curvilinear point-mass
variables (see `equations_of_motion.md` §1); the 7-DOF and 14-DOF formulations keep the same
4-state frame but compute `F_grip` from richer tire/aero models.

---

## 2. Transcription to NLP

Direct collocation turns the continuous optimal control problem into a finite-dimensional NLP by
sampling it at `N` nodes along the track:

- Discretize the track into `N` nodes at arc lengths `s_0 … s_{N-1}`.
- The state and control *values at each node* become the decision variables.
- The continuous dynamics `ẋ = f` become algebraic equality constraints - the **defects** - one per
  state per interval, enforcing that consecutive nodes are consistent with `f`.
- The continuous inequality constraints (boundaries, grip) are enforced pointwise at each node.

The result is a static optimization over a finite vector of numbers, which a Newton-type solver can
attack directly. "Direct" means we discretize first and optimize the discretized problem (as opposed
to indirect methods that derive the optimality conditions analytically first).

---

## 3. Decision Variable Layout

Apex-14 packs all node values into one flat vector. For `N` nodes (block-contiguous):

```
[ s_0..s_{N-1} | n_0..n_{N-1} | v_0..v_{N-1} | α_0..α_{N-1}
  | F_drive_0..F_drive_{N-1} | κ_cmd_0..κ_cmd_{N-1} | dt_0..dt_{N-2} ]
```

Block offsets: `s = 0`, `n = N`, `v = 2N`, `α = 3N`, `F_drive = 4N`, `κ_cmd = 5N`, `dt = 6N`. There
are `N` values for each of the six state/control quantities and `N−1` time steps (one per interval),
for a total of **7N − 1** decision variables. This is the layout in
`crates/apex-optimizer/src/collocation.rs` (`pack` / `unpack`).

---

## 4. Trapezoidal Defects

Over interval `k` (between nodes `k` and `k+1`), the dynamics are enforced with the trapezoidal rule:
approximate the integral of `f` by the average of its endpoint values.

```
d_k = x_{k+1} − x_k − (dt_k / 2)·( f(x_k, u_k) + f(x_{k+1}, u_{k+1}) ) = 0
```

There are 4 defects per interval (one per state component) and `N−1` intervals, giving `4(N−1)`
defect equations; a closed lap adds 4 periodicity equations (`s` wrap, and `n`, `v`, `α` matching).

Why trapezoidal rather than Euler or Hermite-Simpson:

- **Accuracy.** Trapezoidal is second-order accurate (local error `O(dt³)`), versus first-order for
  forward Euler. For the modest node counts used here it captures the dynamics far better than Euler
  without Hermite-Simpson's extra midpoint collocation.
- **Stability.** It is an implicit, A-stable rule, which behaves well on the stiff-ish grip-limited
  segments where speeds change quickly.
- **Jacobian structure.** Each defect couples only the two adjacent nodes `k` and `k+1`, which keeps
  the constraint Jacobian banded and cheap (Section 5). Hermite-Simpson would add midpoint variables
  and widen the stencil.

---

## 5. Banded Jacobian Structure

Because defect `d_k` depends only on nodes `k` and `k+1`, the constraint Jacobian is extremely
sparse. Each scalar defect row touches at most **13** variables: the 6 quantities at node `k`
(`s, n, v, α, F_drive, κ_cmd`), the same 6 at node `k+1`, and the single `dt_k` - `6 + 6 + 1 = 13`.

Concretely, at `N = 100` for a closed lap:

- variables: `7N − 1 = 699`
- equality constraints: `4(N−1) + 4 = 400`
- dense Jacobian entries: `400 × 699 = 279,600`
- structural non-zeros: `≈ 13 × 4(N−1) + 8 ≈ 5,156`
- density: `5,156 / 279,600 ≈ 1.8 %`

So fewer than two percent of the Jacobian is non-zero, and the non-zeros sit in a narrow band along
the diagonal. Storing and factorizing only those entries (in CSR form, `apex_math::CsrMatrix`) is the
difference between a tractable solve and an intractable one as `N` grows.

---

## 6. Automatic Differentiation for Jacobians

The defects need exact derivatives with respect to the 13 local variables. Apex-14 uses forward-mode
automatic differentiation with dual numbers (`apex_math::Dual`, see `equations_of_motion.md` notes on
the `Float` trait). To fill the Jacobian for interval `k`:

1. Seed one of the 13 local variables as `Dual::variable` (derivative part = 1) and the rest as
   `Dual::constant`.
2. Evaluate the four defects through the *generic* dynamics (`seven_dof_derivatives_generic`, etc.).
3. Read each defect's derivative from the `.dual` field of the result.
4. Repeat for each of the 13 variables, writing the non-zeros into the banded CSR matrix.

This is **exact** to machine precision - no truncation error and no step-size tuning, unlike finite
differences. It also exploits the sparsity directly: 13 evaluations per interval, versus `2·(7N−1)`
full-defect evaluations for a dense central-difference Jacobian. The benchmark suite measures the
auto-diff equality Jacobian at N = 50 at ~32 µs versus ~1.7 ms for the numerical one - roughly **52×
faster** - while also being more accurate.

---

## 7. Warm Starting

The NLP is non-convex: it has many local minima and large flat regions where a cold start (e.g. all
zeros, or a straight-line guess) simply fails to converge. A good initial guess is not a nicety here,
it is a requirement.

Apex-14 warm-starts from the quasi-steady-state (QSS) lap simulation. QSS marches the track
forward (acceleration-limited) and backward (braking-limited) and takes the minimum with the
cornering-limited speed, producing a speed profile that is already approximately dynamically
consistent. Interpolating it onto the node mesh gives the optimizer a starting point near the
feasible manifold, from which Newton steps converge.

For the 7-DOF and 14-DOF tire formulations a **tire-aware** QSS is used
(`qss_lap_sim_tire`). The plain grip-circle QSS over-estimates grip (it ignores load sensitivity), so
its speed profile is mildly *infeasible* for the load-sensitive tire model; seeding from the
tire-aware QSS starts inside the achievable grip envelope and avoids an immediate constraint
violation at the first iteration.

---

## 8. Mesh Refinement

A single fine-mesh solve is both expensive and fragile. Mesh refinement
(`mesh_refinement::optimize_with_refinement`) solves coarse-to-fine instead:

1. Solve at a small `N` (cheap) from the QSS warm start.
2. Interpolate that solution onto the next finer mesh
   (`CollocationOptimizer::initial_guess_from_result`): each fine node's state and control are
   linearly interpolated from the surrounding coarse nodes, and `dt` is recomputed from the finer
   spacing.
3. Re-solve on the finer mesh from the interpolated warm start.
4. Repeat to the finest mesh.

The coarse solve is fast and captures the qualitative shape of the trajectory; the fine solve then
only has to add resolution rather than discover the solution from scratch. This improves both speed
and robustness. (Note: absolute defect violations are not comparable across mesh sizes - a finer mesh
has more, finer defects - so convergence is judged per-level against the level's tolerance.)

---

## 9. Solver Architecture

Three solvers are implemented, each a different trade-off for the same NLP:

- **Augmented Lagrangian** (`solver::solve_nlp`). General-purpose: it folds the constraints into the
  objective with penalty terms and Lagrange-multiplier updates, solved by projected gradient descent.
  Robust and broadly applicable, but converges slowly on stiff equality constraints - the penalty
  must grow large before the defects are tightly satisfied.
- **Gauss-Newton with CG** (`gauss_newton::solve_gauss_newton`). Linearizes the constraints and takes
  a damped least-squares step, solving the normal equations with conjugate gradient on the sparse
  Jacobian. Cheap per iteration and fast on well-conditioned problems (e.g. the constant-curvature
  circle, where it converges tightly). This is the default working solver.
- **Sequential Defect Correction** (`direct_solver::solve_direct`). A direct, physics-based repair
  that corrects each trapezoidal interval in sequence rather than solving a global linear system.
  Best on smooth tracks where local corrections propagate cleanly.

On easy geometries (the circle) all three agree and converge tightly; on tracks with sharp curvature
transitions (the oval's straight-to-corner steps) none fully drives every defect to machine zero -
they reach a sensible, near-feasible trajectory rather than a tight optimum. This honest limitation is
recorded in `docs/analysis.md`.
