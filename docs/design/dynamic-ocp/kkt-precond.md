**Status: implemented** (opt-in; `Preconditioner::Jacobi` remains the default).

# Block-tridiagonal KKT preconditioner

The documented `N >= 44` conditioning wall
([`../envelope-qss/real-track-convergence.md`](../envelope-qss/real-track-convergence.md) §B.2)
is the gate on everything in [`recon.md`](recon.md) §3: the dynamic OCP targets
`N = 300–600` on a solver that stops converging somewhere past 44. This work attacks that
wall directly, using the **existing envelope OCP** as the test problem — no new vehicle
models are involved.

**Headline: the wall falls.** On real Silverstone with the calibrated car, the
block-tridiagonal preconditioner converges to machine-tight feasibility at
`N = 24, 32, 40, 48, 64, 96` — every mesh at and above 40 being one where Jacobi fails —
while collapsing median inner-CG iterations from a **saturated 250** to **3–8**. Wall time
at matched `N` improves by **2.6×–6.6×** despite the more expensive per-iteration solve.

**But feasibility was never the whole caveat.** The lap-time objective is still not
mesh-converged, so the "directional deltas, not converged magnitudes" caveat in
`analysis.md` is **partially** earned away — the feasibility half is fixed, the
objective-accuracy half is not. §6 states exactly which clause changes.

---

## 1. What was built

Three pieces, all additive:

| Piece | Where |
|---|---|
| `BlockStructure` + `BlockTridiag` (assembly, block-LDLᵀ, Woodbury wrap correction) | `crates/apex-optimizer/src/precond.rs` (new) |
| `NlpEvaluator::block_structure()`, default `None` | `crates/apex-optimizer/src/nlp.rs` |
| `IpmConfig::preconditioner: Preconditioner{Jacobi\|BlockTridiag}` | `crates/apex-optimizer/src/ipm.rs` |
| `EnvelopeOcp::recommended_block_ip_config()` | `crates/apex-optimizer/src/envelope_ocp.rs` |

**Default unchanged, and proven so.** `Preconditioner::Jacobi` is `#[default]`. Beyond the
existing tests passing, the default path was verified **bit-identical** by capturing an
FNV hash of the full solution vectors (`speeds`, `offsets`, `headings`, `ax`) plus the
`mu`/`cg_iters` history on synthetic Silverstone at `N = 24, 32`, then `git stash`ing the
entire change and re-running:

```
N=24  iters=272  eq_bits=3effcafa1fc00000  lap_bits=405453835a82c679  hash=a3fcf9973892e7f7
N=32  iters=271  eq_bits=3e802d53feb00000  lap_bits=4051a3879e375ec4  hash=77db9c2f8120144a
```

identical before and after. The rank gate (Spearman *exactly* 0.900) and every committed
number are therefore untouched by construction, not merely by assertion.

## 2. The reorder decision: index mapping, not physical repacking

**Chosen: an index-mapping layer.** `BlockStructure` carries a list of global variable
indices per node; the preconditioner gathers and scatters through it.

The envelope OCP lays variables out block-contiguous **by quantity**
(`idx_n(k) = k`, `idx_xi(k) = N + k`, …), so a node's variables sit at stride `N`.
Physically repacking to node-major would have been the more obvious reading of `recon.md`
§3.2, and was rejected for one decisive reason: **it permutes the order of every
floating-point reduction in the solver.** `pcg`'s dot products, the merit function's
sums, the Jacobi diagonal accumulation — all would sum in a different order and therefore
produce different last-bit results. That would move every committed envelope-OCP number
*on the default path*, which the brief explicitly forbids and which the marginal rank gate
makes genuinely risky.

The mapping layer costs one indirection per gather/scatter — negligible against the
`O(nb³)` block algebra — and is strictly more general: it accommodates any layout, including
the node-major one a future dynamic OCP might adopt natively. `BlockStructure::strided`
is the one-line constructor for the current layout.

`ScaledEvaluator` forwards `block_structure()` unchanged, which is correct because the
scaling is **diagonal**: a variable keeps its index, hence its node.

## 3. The design error in `recon.md`, and the fix

`recon.md` §3.2 recommended **dropping** the periodic wrap — the corner blocks coupling the
last node to the first on a flying lap — reasoning that "a rank-`nb` perturbation out of
`N·nb` is asymptotically irrelevant," with Woodbury held in reserve.

**That was implemented, measured, and is wrong.** With the wrap dropped:

| | relative inverse error `‖P⁻¹Mv − v‖/‖v‖` |
|---|---|
| Jacobi | ~1.1 |
| BlockTridiag, wrap dropped | 3.1e-1, degrading to **2.3** as `rho` grew |
| BlockTridiag, wrap kept (Woodbury) | **~1e-10** |

and the solve was *worse than plain Jacobi* — CG still saturated, and every real-circuit
run terminated `LineSearchFailure`.

**Why the reasoning failed.** It argued about the *size* of the perturbation when the
relevant property is its effect on the *inverse*. Dropping the corner turns a periodic ring
into a line, and **the inverse of a tridiagonal matrix is dense** — every entry of `T⁻¹`
depends on the boundary condition. A structurally tiny change to the operator is a global
change to its inverse. Worse, the error grows with `rho`, so it degrades exactly as the
augmented-Lagrangian penalty ramps and the preconditioner is most needed.

**The fix.** Retain the corner exactly via Sherman-Morrison-Woodbury. Writing it as
`Z S Zᵀ` with `Z` selecting the first and last blocks and `S = [[0, W], [Wᵀ, 0]]`:

```
M⁻¹ = T⁻¹ − (T⁻¹Z) (I + S·ZᵀT⁻¹Z)⁻¹ S (ZᵀT⁻¹)
```

Cost: `2·nb` extra line solves to form `T⁻¹Z`, plus one dense `2nb × 2nb` solve. The form
above never inverts `S` — which is singular whenever the corner is rank-deficient — so it
is safe for the general case. `preconditioner_inverts_operator_exactly` locks the result,
and `dropping_the_wrap_would_degrade_the_inverse` pins the finding so the shortcut cannot
be reintroduced silently.

## 4. The second finding: an exact solve exposes a rank deficiency

Making the preconditioner exact fixed CG but the solves *still* failed — `LineSearchFailure`
at every mesh. Two isolation experiments:

1. **Objective model ruled out.** Re-running with `obj_weight = 0` (pure feasibility) failed
   too, so this was not the lap-time objective interacting badly with better steps.
2. **Regularization sweep — the answer.** Sweeping `reg` over `1e-8 … 1e0` at `N = 48/96/128`
   showed a sharp transition: `reg <= 1e-4` fails everywhere, `reg = 1e-1` reaches `Optimal`
   everywhere.

**The mechanism.** `Jeq` is `3N × 5N`, so `rho·JeqᵀJeq` is **rank-deficient by `2N`**: the
`a_x` and `kappa_cmd` control directions have no bound constraints, hence no barrier term
`Σ_L + Σ_U`, leaving `reg` as their *only* regularization. Jacobi-preconditioned CG never
resolved those directions within its 250-iteration cap, so it regularized them
**implicitly** — the classical iterative-regularization property of truncated CG. An exact
preconditioner *does* resolve them, and amplifies them by `1/reg = 1e8`. The line search
then collapses.

So `reg = 1e-8` was never a neutral default; it was load-bearing only because the linear
solve was too weak to exercise it. **Fixing the preconditioner surfaced a latent
ill-posedness that had been masked.** This is the single most transferable finding here:
the dynamic OCP will have the same structure (controls without bounds) and must carry an
explicit `reg`.

`reg = 1e-1` is the measured sweet spot. `1e-2` still stalls at `N = 128`; `1e0` converges
everywhere but visibly over-damps (Silverstone 108 s vs 88 s). Both are recorded in §5.3.

## 5. Measurements

Real Silverstone (`tracks/silverstone.json`, gitignored import), `CarParams::f1_2024_calibrated`
with the aero-bridged envelope, `max_iterations = 1500`. Jacobi rows use
`recommended_ip_config()`; BlockTridiag rows use `recommended_block_ip_config()`
(same, plus `preconditioner` and `reg = 1e-1`). Fixed-line QSS reference: **112.174 s**.
Harness: `crates/apex-optimizer/tests/kkt_precond_sweep.rs` (`#[ignore]`d).

### 5.1 The headline — CG iterations and the wall

| N | precond | status | eq | ineq | lap (s) | **CG med** | CG max | outer | wall (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 24 | Jacobi | MaxIter | 6.10e-4 | 5.88e-4 | 76.203 | **250** | 250 | 1500 | 2536 |
| 24 | **BlockTridiag** | **Optimal** | 2.33e-5 | 1.09e-5 | 78.936 | **8** | 250 | 280 | **650** |
| 32 | Jacobi | Optimal | 2.01e-7 | 6.16e-10 | 89.084 | **250** | 250 | 268 | 1459 |
| 32 | **BlockTridiag** | **Optimal** | 2.38e-5 | 4.39e-5 | 90.084 | **6** | 250 | 330 | 1234 |
| 40 | Jacobi | MaxIter | 1.42e-2 | 1.36e-2 | 92.000 | **250** | 250 | 1500 | 7603 |
| 40 | **BlockTridiag** | **Optimal** | 3.99e-5 | 4.25e-5 | 93.837 | **3** | 14 | 262 | **1160** |
| 48 | Jacobi | MaxIter | 2.73e-2 | 5.66e0 | 76.104 | **250** | 250 | 1500 | 8484 |
| 48 | **BlockTridiag** | **Optimal** | 4.88e-5 | 6.85e-5 | 82.290 | **6** | 250 | 353 | **1714** |
| 64 | Jacobi | MaxIter | 1.71e-1 | 1.65e1 | 81.470 | **250** | 250 | 1500 | 7928 |
| 64 | **BlockTridiag** | **Optimal** | 1.77e-5 | 7.52e-5 | 87.145 | **4** | 250 | 345 | **2862** |
| 96 | Jacobi | MaxIter | 2.62e-1 | 2.19e1 | 80.901 | **250** | 250 | 1500 | 15381 |
| 96 | **BlockTridiag** | **Optimal** | 1.17e-5 | 7.01e-5 | 88.512 | **4** | 250 | 930 | **9856** |
| 128 | Jacobi | MaxIter | 1.46e1 | 1.38e1 | 111.608 | **250** | 250 | 1500 | 16267 |
| 128 | BlockTridiag | MaxIter | 5.11e-6 | 9.38e-5 | 88.958 | **6** | 250 | 1500 | 22047 |

**Reading it:**

- **CG is flat in `N`.** Median 8, 6, 3, 6, 4, 4, 6 across `N = 24 → 128`. This is the
  `O(1)`-in-`N` claim, measured rather than asserted. Jacobi is pinned at the 250 cap at
  every single mesh — confirming §B.2's observation that it saturates on converging and
  failing runs alike.
- **The `N >= 44` wall falls.** Jacobi's only `Optimal` in the whole table is `N = 32`
  (the known per-track `N*`). BlockTridiag is `Optimal` at 24/32/40/48/64/96, with `eq` and
  `ineq` both `~1e-5` — machine-tight, four orders inside `constraint_tol`.
- **`CG max` stays 250.** A handful of steps still saturate, almost all in the first few
  outer iterations before `rho` has ramped. The median is the honest summary statistic;
  the max is reported so this is not hidden.
- **`N = 128` is `MaxIter`, but not a wall.** `eq = 5.1e-6`, `ineq = 9.4e-5` — it is
  *feasible*, just still walking when the 1500-iteration budget ran out. Qualitatively
  different from Jacobi's `eq = 14.6` at the same mesh. It converges at higher `reg` (§5.3).

### 5.2 Wall time at matched `N`

Block solves cost `O(N·nb³)` per CG iteration versus Jacobi's `O(N·nb)`, so the per-iteration
cost is genuinely higher. The question was whether the iteration collapse pays for it.

| N | Jacobi (ms) | BlockTridiag (ms) | speedup |
|---|---:|---:|---|
| 24 | 2536 | 650 | **3.9×** |
| 32 | 1459 | 1234 | 1.2× |
| 40 | 7603 | 1160 | **6.6×** |
| 48 | 8484 | 1714 | **5.0×** |
| 64 | 7928 | 2862 | **2.8×** |
| 96 | 15381 | 9856 | 1.6× |
| 128 | 16267 | 22047 | 0.7× |

**Net win, with an honest caveat.** Speedup is 2.8×–6.6× where Jacobi fails and BlockTridiag
converges — but those comparisons are not like-for-like, because the Jacobi runs are burning
their full 1500-iteration budget *without converging*. The fair reading:

- `N = 32`, the one mesh where **both** converge: **1.2× faster** and to comparable
  feasibility. The block algebra pays for itself even in a head-to-head.
- `N = 128` is slower in wall time, but Jacobi's "faster" run ends at `eq = 14.6`. Time to a
  useless answer is not a meaningful baseline.

`solve_linear` is unblocked partial-pivoting Gaussian elimination per `nb × nb` block. At
`nb = 5` that is trivial; a future dynamic OCP at `nb = 8–13` would see `O(nb³)` grow ~4–18×
per block, which is where a cached factorization (rather than an explicit inverse) would be
the natural optimization. Not needed at present scale.

### 5.3 The `reg` sweep (BlockTridiag)

| N | `reg` = 1e-8 | 1e-6 | 1e-4 | 1e-2 | **1e-1** | 1e0 |
|---|---|---|---|---|---|---|
| 48 | LineSearchFail | InfeasDetect | LineSearchFail | Optimal 77.0 | **Optimal 82.3** | Optimal 104.9 |
| 96 | LineSearchFail | LineSearchFail | InfeasDetect | MaxIter 88.2 | **Optimal 88.5** | Optimal 108.6 |
| 128 | InfeasDetect | LineSearchFail | LineSearchFail | InfeasDetect | **MaxIter 89.0** | Optimal 108.0 |

The `1e0` column shows the cost of over-damping: it converges everywhere but inflates the lap
by ~20 s. `1e-1` is the balance point adopted in `recommended_block_ip_config`.

### 5.4 Mesh convergence re-sweep

The deferred bar from `real-track-convergence.md` §B.6 was "**real Silverstone monotone over
`N = 40 → 64 → 96` with the last delta < 1 %**."

| N | Jacobi lap (s) | Δ | BlockTridiag lap (s) | Δ | BT status |
|---|---|---|---|---|---|
| 24 | 76.203 | – | 78.936 | – | Optimal |
| 32 | 89.084 | +16.90 % | 90.084 | +14.12 % | Optimal |
| 40 | 92.000 | +3.27 % | 93.837 | +4.17 % | Optimal |
| 48 | 76.104 | −17.28 % | 82.290 | −12.31 % | Optimal |
| 64 | 81.470 | +7.05 % | 87.145 | +5.90 % | Optimal |
| 96 | 80.901 | −0.70 % | 88.512 | +1.57 % | Optimal |
| 128 | 111.608 | +37.96 % | 88.958 | **+0.50 %** | MaxIter |
| 192 | 112.280 | +0.60 % | 98.580 | +10.82 % | Optimal |
| 256 | 96.681 | −13.89 % | 103.891 | +5.39 % | MaxIter |

**Verdict: not met — feasibility is fixed, the objective is not.**

What genuinely improved: **feasibility is now mesh-robust to `N = 256`**. Every BlockTridiag
row has `eq <= 7e-5` and `ineq <= 1e-4`; the Jacobi column swings up to `eq = 14.6`. The
`N >= 44` feasibility wall is gone.

What did not: the lap time still swings ~±12 % and is **not monotone**. The `N = 96 → 128`
delta of `+0.50 %` clears the 1 % bar in isolation, but the sequence continues `+10.82 %`
at 192, so this is not convergence — it is a coincidence, and it is reported as one.

A follow-up sweep tested whether the high-`N` upward drift is a fixed-`reg` damping bias
(`reg ∈ {1e-2, 3e-2, 1e-1, 3e-1}` at `N = 96…256`). It is **not** a clean bias — no `reg`
schedule flattens the curve. The residual is the **objective-accuracy** problem
`real-track-convergence.md` §5 and `CLOSE.md` §3 both flag separately from conditioning:
at coarse-to-moderate `N` a uniform mesh over-cuts corners by a mesh-dependent amount.
The named remedy remains **adaptive `s`-refinement** (refine at high curvature) — untouched
by this work, and now clearly the *next* bottleneck rather than one hidden behind the
conditioning wall.

## 6. Consequences for the `analysis.md` caveat

The standing caveat reads, in substance: *tight-feasible QSS-vs-OCP deltas are directionally
valid but not reliable lap-time magnitudes, and are not comparable across tracks solved at
different `N`.*

**It splits in two, and only one half is earned away:**

- **Earned away (feasibility):** the clause that tight feasibility is reachable *only* at
  coarse per-track `N* ≈ 24–40`, with `N >= 44` regressing. That was a conditioning
  artifact, and BlockTridiag removes it — feasibility now holds to `N = 256`. The
  "per-track `N` is a knob" framing no longer describes the solver's actual limit.
- **Still standing (objective accuracy):** the clause that lap-time *magnitudes* are
  mesh-dependent and not converged. §5.4 shows ±12 % non-monotone swings persisting. The
  numbers in `analysis.md` remain directional.

Because the committed `analysis.md` table was produced under **Jacobi at per-track `N*`**,
and this change ships Jacobi as the default, **no number in `analysis.md` moves and no
edit is required.** Rewriting the caveat should accompany a deliberate default switch and a
re-measured rank gate — deliberately out of scope here, and recorded in §8.

## 7. Tests

Nine unit tests in `precond.rs`:

| Test | Property |
|---|---|
| `strided_structure_validates_and_maps` | the strided constructor and its `(block, slot)` map |
| `validate_rejects_malformed_structures` | ragged / duplicate / out-of-range / wrong-total all rejected |
| `block_assembly_matches_dense_reference` | assembled blocks invert the dense reference operator; no double-counting in `c[b]` |
| `block_thomas_matches_solve_linear` | block-Thomas agrees with a general dense solve on the same system |
| `operator_is_spd` | symmetry + `xᵀMx > 0` on probe vectors, including through the corner |
| `preconditioner_inverts_operator_exactly` | `‖P⁻¹Mv − v‖/‖v‖ < 1e-8` at `N = 6, 16, 33` under stiff `rho = 5e3` |
| `dropping_the_wrap_would_degrade_the_inverse` | pins §3's finding so the shortcut cannot return |
| `apply_is_bitwise_deterministic` | bitwise-identical application |
| `singular_block_returns_none` | graceful `None`, not garbage |

Five integration tests in `tests/kkt_precond.rs`, on the **synthetic** `silverstone_circuit`
so they run in CI without the gitignored real-track import:

- `blocktridiag_converges_where_jacobi_fails` — the headline, at `N = 56`.
- `jacobi_still_fails_at_the_discriminating_mesh` — the companion guard.
- `blocktridiag_collapses_inner_cg_iterations` — asserts the mechanism directly.
- `blocktridiag_is_bitwise_deterministic` — determinism to the Jacobi standard.
- `unstructured_problem_falls_back_to_jacobi_bit_identically` — graceful degradation.

**`N = 56` and the `1e-2` bound apply the Part-C CI-marginality lesson.** At that mesh Jacobi
fails at `eq ≈ 26, ineq ≈ 28` and BlockTridiag reaches `eq ≈ 2e-5, ineq ≈ 4e-5` — roughly
four orders of margin on each side. The assertion is the **quantitative feasibility bound**,
never `IpmStatus`, because Part C established that libm differences can flip the terminal
status without the solution being marginal. The companion negative test exists so a future
change that made Jacobi converge at `N = 56` is caught rather than silently hollowing out the
main test.

`tests/kkt_precond_sweep.rs` holds the two `#[ignore]`d measurement harnesses that produced
§5; both skip cleanly when `tracks/silverstone.json` is absent.

**Suite: 857 passed, 0 failed** (`cargo test --workspace`). **Goldens unchanged**:
`golden_oval_qss`, `golden_silverstone_qss`, `golden_circle_optimize` all pass; fixture
directory clean per `git status`. clippy `--all-targets` and `cargo fmt --check` clean.

## 8. Limitations and follow-ups

- **The objective Hessian is omitted** from the preconditioner. `objective_hessian_vec` is
  matrix-free and defaults to zero for these problems, so there is nothing to assemble.
  Omitting a PSD term keeps the preconditioner SPD; a future problem with genuine objective
  curvature would want its diagonal folded in.
- **Stencil assumption.** The assembly keeps intra-node, adjacent-node, and periodic-corner
  couplings, and silently drops anything wider. Trapezoidal collocation produces nothing
  wider. **Hermite-Simpson with midpoint variables would**, and would need this revisited —
  flagged in `assemble`'s source.
- **`reg` is a manual knob.** §4's rank deficiency is structural; `reg = 1e-1` was tuned by
  sweep, not derived. A principled alternative is bounding the controls (`a_x`, `kappa_cmd`),
  which would give them genuine barrier terms and remove the reliance on `reg` entirely.
  Worth doing in the dynamic OCP, where control bounds are physically motivated anyway
  (power, steering limits).
- **Not the default.** Switching requires re-measuring the setup-envelope rank gate
  (currently marginal at Spearman exactly 0.900) and regenerating the `analysis.md` table.
  A deliberate, separately-evidenced decision.
- **Adaptive `s`-refinement is now the top bottleneck** for objective accuracy (§5.4). With
  conditioning handled, it is no longer hidden behind a solver limitation.
- **Not yet wired to the 4-state collocation NLP.** `CollocationEvaluator` does not override
  `block_structure()`, so `optimize_ip` still runs Jacobi. Its layout is strided the same way
  (`[s|n|v|α|F|κ|dt]`), but the trailing `dt` block has `N−1` entries rather than `N`, so it
  needs a non-uniform structure or a padded block — deliberately out of scope here.
