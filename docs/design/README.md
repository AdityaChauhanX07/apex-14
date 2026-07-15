# Design docs

Design and diagnosis notes for Apex-14. One folder per feature.

## Convention

- **Location:** each design doc lives in `docs/design/<feature-name>/`, named by
  the feature it concerns (e.g. `envelope-qss/`, not a milestone or phase number).
  Use a short, stable, kebab-case feature name. Internal planning phase numbers
  must not appear in tracked files — name things by what they *are*.
- **Status line:** every doc starts with a `**Status: <state>**` line. States:
  - `draft` — under discussion, not yet agreed.
  - `accepted` — agreed direction; implementation may not be complete.
  - `implemented` — the described design is in the code.
  - `superseded` — replaced; the line should point to what replaced it.
- **Anchoring:** cite code as `path:line` so a reader can jump to ground truth.

## Index

- [`envelope-qss/recon.md`](envelope-qss/recon.md) — *accepted*. Reconnaissance
  and scope for envelope-QSS free-trajectory optimization (solver state, g_z
  pathway, envelope prerequisites, scope decisions).
- [`envelope-qss/free-trajectory-ocp.md`](envelope-qss/free-trajectory-ocp.md) —
  *implemented*. The racing-line OCP that optimizes against the g-g-g envelope
  (control parameterization, `eps` ruling, IP penalty-ramp tuning, validation:
  analytic circle 0.02 %, oval corner-cutting, Silverstone cross-check).
- [`gn-solver-bound-deadlock.md`](gn-solver-bound-deadlock.md) — *diagnosis*.
  Why projection-patched Gauss-Newton deadlocks on bound-binding problems.
  (Legacy flat location; see note below.)
- [`nlp-scaling.md`](nlp-scaling.md) — *implemented*. Variable scaling for the
  collocation NLP (Jacobi/diagonal preconditioning). (Legacy flat location.)

> **Legacy flat files.** `gn-solver-bound-deadlock.md` and `nlp-scaling.md`
> predate this convention and sit directly under `docs/design/`. They are
> referenced by path from several tracked files (`PHYSICS_CHANGE.md`,
> `golden_lap.rs`, `envelope-qss/recon.md`, and each other), so moving them into
> per-feature folders (e.g. `docs/design/gn-solver/` and
> `docs/design/nlp-scaling/`) would break those links. **Proposed, not executed:**
> relocate them under feature folders in a dedicated commit that also updates
> every referrer. Until then they stay put and remain consistent with the
> naming rule (feature-named, no phase numbers).
