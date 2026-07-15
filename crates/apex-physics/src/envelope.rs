//! The g-g-g performance envelope: the car's feasible-acceleration surface
//! swept over `(theta, v, g_z)`, stored with C1 interpolation and keyed by
//! content hash.
//!
//! This is the surface the free-trajectory OCP constrains against. For each
//! speed `v` and imposed vertical acceleration `g_z`, the set of feasible planar
//! accelerations `(a_x, a_y)` is a convex-ish region bounded by the
//! grip / load / power limits reported by [`solve_operating_point`]. We store
//! that boundary as a **radius function** `rho(theta; v, g_z)` over the
//! acceleration-plane angle `theta = atan2(a_y, a_x)`:
//!
//! ```text
//! a_x = r*cos(theta),  a_y = r*sin(theta),  feasible  iff  r <= rho(theta; v, g_z)
//! ```
//!
//! # Why a radius function (representation choice)
//!
//! The alternative was a signed-distance grid over the full `(a_x, a_y)` plane.
//! The deciding criterion (design §1) is **C1 smoothness of the OCP
//! constraint**. The radius form wins on three counts:
//!
//! 1. **A clean scalar constraint.** The OCP needs one inequality per node:
//!    `c = |a| / rho(theta; v, g_z) <= 1`. With `rho` a C1 field (via
//!    [`HermiteGrid`]) and `|a|`, `theta` smooth functions of `(a_x, a_y)` away
//!    from the origin, `c` is C1 in the decision variables — exactly what a
//!    gradient solver wants. A signed-distance grid gives C1 too, but the
//!    constraint then reads off a 2-D field whose zero level-set is the boundary;
//!    recovering `d(constraint)/d(a)` is less direct.
//! 2. **The brake/drive asymmetry is represented natively.** `rho(0)` (pure
//!    drive, power-limited) and `rho(pi)` (pure braking) simply differ; the angle
//!    axis carries the asymmetry with no special-casing.
//! 3. **Periodicity closes the loop C1.** `theta` is a periodic axis, so
//!    `rho` and its `theta`-derivative are continuous across `theta = +-pi` — no
//!    seam in the constraint as the acceleration vector rotates.
//!
//! The cost is that `rho` must be single-valued in `theta` — i.e. the feasible
//! region must be star-shaped about the origin. It is: the boundary is monotone
//! along every ray from the origin (grip demand `m*|a|` and the longitudinal
//! actuator limit both grow monotonically with `r`), which we **assert** during
//! generation (see [`boundary_radius`]).
//!
//! # Boundary location
//!
//! Along each ray we do not take the raw grid transition. We march outward,
//! locate the last-feasible / first-infeasible bracket, and **bisect** it to a
//! tight tolerance (a trim solve is ~464 ns, so we can afford ~40 bisections per
//! ray). The monotone-boundary property is asserted along the whole ray; an
//! island (a feasible sample beyond an infeasible one) is a hard error reporting
//! the operating point.
//!
//! # Cache & determinism
//!
//! An envelope is keyed by [`envelope_key`] =
//! `hash(car, tire, aero, suspension, grid-spec, ENVELOPE_CODE_VERSION)` and
//! serialized to a versioned binary ([`Envelope::to_bytes`] /
//! [`Envelope::save`]). Generation is deterministic: fixed iteration order, and
//! the Rayon parallelism (when enabled) is over embarrassingly parallel rays
//! collected back in grid order, so the same inputs yield **byte-identical**
//! envelope files regardless of thread count. At ~tens of ms per regen the cache
//! is primarily a reproducibility artifact (hash -> identical bytes), not a
//! compute saver.

use std::io::{self, Read, Write};
use std::path::Path;

use apex_math::{ContentHash, Dual, GridAxis, Hash, HashWriter, HermiteGrid};

use crate::aero::{aero_hash, AeroModel};
use crate::car_params::{car_params_hash, CarParams};
use crate::suspension::{suspension_hash, SuspensionSystem};
use crate::tire::{pacejka_tire_hash, PacejkaTire};
use crate::trim::{solve_operating_point, OperatingPoint};

/// Version tag folded into the envelope cache key and file header. Bump on any
/// change to the boundary-location method, the grid semantics, or the file
/// layout — every stored envelope's key then changes, forcing a regen.
pub const ENVELOPE_CODE_VERSION: &str = "envelope.v1";

/// Magic bytes at the head of a serialized envelope file.
const ENVELOPE_MAGIC: &[u8; 8] = b"APXENV01";

/// Grid + tolerance specification for an envelope sweep. This, together with the
/// four model hashes, determines the envelope's content identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvelopeGridSpec {
    /// Number of acceleration-plane angle samples (periodic over `[0, 2pi)`).
    /// Must be `>= 5`.
    pub theta_res: usize,
    /// Minimum speed (m/s).
    pub v_min: f64,
    /// Maximum speed (m/s).
    pub v_max: f64,
    /// Number of speed samples. Must be `>= 4`.
    pub v_res: usize,
    /// Minimum imposed vertical acceleration (m/s²).
    pub gz_min: f64,
    /// Maximum imposed vertical acceleration (m/s²).
    pub gz_max: f64,
    /// Number of `g_z` samples. Must be `>= 4`.
    pub gz_res: usize,
    /// Upper bound on the boundary search radius (m/s²). A ray still feasible at
    /// this radius clamps its `rho` here (does not occur for physical cars).
    pub max_accel: f64,
    /// Coarse march resolution used to bracket the boundary before bisection.
    pub coarse_steps: usize,
    /// Absolute bisection tolerance on the boundary radius (m/s²).
    pub bisect_tol: f64,
}

impl Default for EnvelopeGridSpec {
    /// A reasonable default: an F1-scale sweep over a wide speed range and a
    /// `g_z` band bracketing standard gravity (flat -> steep grade / banking).
    fn default() -> Self {
        EnvelopeGridSpec {
            theta_res: 24,
            v_min: 5.0,
            v_max: 90.0,
            v_res: 10,
            gz_min: 8.0,
            gz_max: 14.0,
            gz_res: 6,
            max_accel: 80.0,
            coarse_steps: 64,
            bisect_tol: 1e-3,
        }
    }
}

impl EnvelopeGridSpec {
    /// Validate the resolutions and ranges, returning an error rather than
    /// panicking so the CLI can report bad input cleanly.
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        if self.theta_res < 5 {
            return Err(EnvelopeError::BadSpec("theta_res must be >= 5"));
        }
        if self.v_res < 4 || self.gz_res < 4 {
            return Err(EnvelopeError::BadSpec("v_res and gz_res must be >= 4"));
        }
        // Reject non-finite ranges up front so the ordering checks below are
        // total (and never trip GridAxis's own asserts on NaN/inf).
        if ![
            self.v_min,
            self.v_max,
            self.gz_min,
            self.gz_max,
            self.max_accel,
            self.bisect_tol,
        ]
        .iter()
        .all(|x| x.is_finite())
        {
            return Err(EnvelopeError::BadSpec("non-finite range value"));
        }
        if self.v_max <= self.v_min {
            return Err(EnvelopeError::BadSpec("v_max must exceed v_min"));
        }
        if self.gz_max <= self.gz_min {
            return Err(EnvelopeError::BadSpec("gz_max must exceed gz_min"));
        }
        if self.gz_min <= 0.0 {
            return Err(EnvelopeError::BadSpec("gz_min must be positive"));
        }
        if self.max_accel <= 0.0 || self.coarse_steps < 4 || self.bisect_tol <= 0.0 {
            return Err(EnvelopeError::BadSpec(
                "bad max_accel/coarse_steps/bisect_tol",
            ));
        }
        Ok(())
    }

    /// The three grid axes in storage order `[theta, v, g_z]`. `theta` is
    /// periodic; `v` and `g_z` are uniform closed intervals.
    pub fn axes(&self) -> [GridAxis; 3] {
        [
            GridAxis::periodic(0.0, std::f64::consts::TAU, self.theta_res),
            GridAxis::uniform(self.v_min, self.v_max, self.v_res),
            GridAxis::uniform(self.gz_min, self.gz_max, self.gz_res),
        ]
    }

    /// Total node count.
    pub fn total(&self) -> usize {
        self.theta_res * self.v_res * self.gz_res
    }
}

impl ContentHash for EnvelopeGridSpec {
    fn hash_into(&self, w: &mut HashWriter) {
        w.usize(self.theta_res);
        w.f64(self.v_min);
        w.f64(self.v_max);
        w.usize(self.v_res);
        w.f64(self.gz_min);
        w.f64(self.gz_max);
        w.usize(self.gz_res);
        w.f64(self.max_accel);
        w.usize(self.coarse_steps);
        w.f64(self.bisect_tol);
    }
}

/// Errors from envelope generation, loading, or evaluation.
#[derive(Debug)]
pub enum EnvelopeError {
    /// The grid spec failed validation (message describes which field).
    BadSpec(&'static str),
    /// A non-star-shaped feasible region: a feasible sample was found beyond an
    /// infeasible one along a ray (violates the monotone-boundary property).
    NonMonotoneRay {
        /// Angle (rad).
        theta: f64,
        /// Speed (m/s).
        v: f64,
        /// Imposed vertical acceleration (m/s²).
        g_z: f64,
        /// The infeasible radius that precedes a later feasible one.
        r_infeasible: f64,
        /// The offending feasible radius beyond it.
        r_feasible: f64,
    },
    /// I/O error reading or writing a cache file.
    Io(io::Error),
    /// The file did not start with the expected magic / version.
    BadMagic,
    /// The stored key did not match the recomputed key on load (models or spec
    /// drifted under a reused filename).
    KeyMismatch {
        /// Key recorded in the file.
        stored: Hash,
        /// Key recomputed from the caller's models/spec.
        expected: Hash,
    },
    /// Truncated or malformed file body.
    Corrupt,
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvelopeError::BadSpec(m) => write!(f, "invalid envelope spec: {m}"),
            EnvelopeError::NonMonotoneRay {
                theta,
                v,
                g_z,
                r_infeasible,
                r_feasible,
            } => write!(
                f,
                "non-monotone (island) feasible region at theta={theta:.4} v={v:.3} g_z={g_z:.3}: \
                 feasible at r={r_feasible:.4} beyond infeasible at r={r_infeasible:.4}"
            ),
            EnvelopeError::Io(e) => write!(f, "envelope I/O error: {e}"),
            EnvelopeError::BadMagic => write!(f, "not an Apex envelope file (bad magic/version)"),
            EnvelopeError::KeyMismatch { stored, expected } => write!(
                f,
                "envelope key mismatch: file {} != expected {}",
                stored.short(),
                expected.short()
            ),
            EnvelopeError::Corrupt => write!(f, "corrupt or truncated envelope file"),
        }
    }
}

impl std::error::Error for EnvelopeError {}

impl From<io::Error> for EnvelopeError {
    fn from(e: io::Error) -> Self {
        EnvelopeError::Io(e)
    }
}

/// Content-hash key for an envelope: folds the four model hashes, the grid spec,
/// and [`ENVELOPE_CODE_VERSION`] under the domain `"envelope.v1"`, in a fixed
/// order. Two runs with identical models + spec + code version share a key (and
/// therefore, by determinism, byte-identical envelope bytes).
pub fn envelope_key(
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    spec: &EnvelopeGridSpec,
) -> Hash {
    let mut w = HashWriter::new();
    w.str(apex_math::HASH_VERSION);
    w.str("envelope.v1");
    w.str(ENVELOPE_CODE_VERSION);
    w.bytes(car_params_hash(car).as_bytes());
    w.bytes(pacejka_tire_hash(tire).as_bytes());
    w.bytes(suspension_hash(suspension).as_bytes());
    w.bytes(aero_hash(aero).as_bytes());
    spec.hash_into(&mut w);
    w.finish()
}

/// Locate the feasible-region boundary radius `rho(theta; v, g_z)` along one ray
/// from the origin in the acceleration plane.
///
/// Marches `coarse_steps` samples out to `max_accel`, brackets the first
/// feasible->infeasible transition, and bisects it to `bisect_tol`. Asserts the
/// monotone-boundary property: once feasibility is lost along the ray it never
/// returns. Returns `Err(NonMonotoneRay)` (fail loudly, with the operating
/// point) if an island appears.
///
/// A ray feasible even at `max_accel` clamps to `max_accel`. A ray infeasible at
/// the origin returns `0.0`.
#[allow(clippy::too_many_arguments)]
pub fn boundary_radius(
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    theta: f64,
    v: f64,
    g_z: f64,
    spec: &EnvelopeGridSpec,
) -> Result<f64, EnvelopeError> {
    let (ct, st) = (theta.cos(), theta.sin());
    let feasible = |r: f64| -> bool {
        let op = OperatingPoint {
            v,
            a_x: r * ct,
            a_y: r * st,
            g_z,
        };
        solve_operating_point(car, tire, suspension, aero, op)
            .map(|t| t.is_feasible())
            .unwrap_or(false)
    };

    let n = spec.coarse_steps;
    let dr = spec.max_accel / n as f64;

    // Evaluate feasibility at every coarse sample once (fixed order), then reason
    // about the whole ray -- this lets us assert monotonicity cheaply.
    let flags: Vec<bool> = (0..=n).map(|i| feasible(i as f64 * dr)).collect();

    // First infeasible index.
    let first_bad = flags.iter().position(|&f| !f);
    let Some(k) = first_bad else {
        // Feasible all the way to max_accel: clamp.
        return Ok(spec.max_accel);
    };

    // Monotonicity: everything at and beyond k must be infeasible.
    if let Some(off) = flags[k..].iter().position(|&f| f) {
        let j = k + off;
        return Err(EnvelopeError::NonMonotoneRay {
            theta,
            v,
            g_z,
            r_infeasible: k as f64 * dr,
            r_feasible: j as f64 * dr,
        });
    }

    if k == 0 {
        // Infeasible even at r = 0 (nonphysical here, but reported honestly).
        return Ok(0.0);
    }

    // Bisect [lo, hi] = [last feasible, first infeasible] with a fixed iteration
    // count (determinism) down to bisect_tol.
    let mut lo = (k - 1) as f64 * dr; // feasible
    let mut hi = k as f64 * dr; // infeasible
    let iters = bisection_iters(dr, spec.bisect_tol);
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        if feasible(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Fixed number of bisection steps to shrink an interval of width `dr` below
/// `tol`, `ceil(log2(dr/tol))`, clamped to `[1, 60]`. A function of the spec
/// only (not of the data), so every ray runs the same count -> determinism.
fn bisection_iters(dr: f64, tol: f64) -> usize {
    let ratio = (dr / tol).max(1.0);
    (ratio.log2().ceil() as usize).clamp(1, 60)
}

/// A generated g-g-g envelope: the boundary-radius field `rho(theta, v, g_z)`
/// with C1 interpolation, plus its content key and spec.
#[derive(Debug, Clone)]
pub struct Envelope {
    spec: EnvelopeGridSpec,
    key: Hash,
    grid: HermiteGrid,
}

impl Envelope {
    /// Generate an envelope by sweeping [`boundary_radius`] over the grid.
    ///
    /// Deterministic and (with the `parallel` feature) parallel over rays with an
    /// order-preserving reduction, so the result is independent of thread count.
    pub fn generate(
        car: &CarParams,
        tire: &PacejkaTire,
        suspension: &SuspensionSystem,
        aero: &AeroModel,
        spec: EnvelopeGridSpec,
    ) -> Result<Envelope, EnvelopeError> {
        spec.validate()?;
        let [theta_ax, v_ax, gz_ax] = spec.axes();
        let (nv, ngz) = (spec.v_res, spec.gz_res);
        let total = spec.total();

        // Flat index (row-major over [theta, v, g_z]) -> the ray to solve.
        let ray = |idx: usize| -> Result<f64, EnvelopeError> {
            let k = idx % ngz;
            let j = (idx / ngz) % nv;
            let i = idx / (ngz * nv);
            boundary_radius(
                car,
                tire,
                suspension,
                aero,
                theta_ax.coord(i),
                v_ax.coord(j),
                gz_ax.coord(k),
                &spec,
            )
        };

        #[cfg(feature = "parallel")]
        let values: Vec<f64> = {
            use rayon::prelude::*;
            // `collect` into a Vec preserves index order regardless of the number
            // of worker threads -> byte-identical output.
            (0..total)
                .into_par_iter()
                .map(ray)
                .collect::<Result<Vec<f64>, EnvelopeError>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let values: Vec<f64> = (0..total).map(ray).collect::<Result<Vec<f64>, _>>()?;

        let key = envelope_key(car, tire, suspension, aero, &spec);
        let grid = HermiteGrid::new(vec![theta_ax, v_ax, gz_ax], values);
        Ok(Envelope { spec, key, grid })
    }

    /// The grid spec.
    pub fn spec(&self) -> &EnvelopeGridSpec {
        &self.spec
    }

    /// The content key.
    pub fn key(&self) -> Hash {
        self.key
    }

    /// Raw row-major boundary-radius node values (storage order `[theta, v, g_z]`).
    pub fn values(&self) -> &[f64] {
        self.grid.values()
    }

    /// Interpolated boundary radius `rho(theta, v, g_z)` (m/s²). `theta` wraps.
    pub fn rho(&self, theta: f64, v: f64, g_z: f64) -> f64 {
        self.grid.eval_f64(&[theta, v, g_z])
    }

    /// `rho` and its gradient `(d/dtheta, d/dv, d/dg_z)` via the dual-number path
    /// — the derivative surface the OCP consumes. Three single-seed evaluations.
    pub fn rho_grad(&self, theta: f64, v: f64, g_z: f64) -> (f64, [f64; 3]) {
        let dt = self.grid.eval(&[
            Dual::variable(theta),
            Dual::constant(v),
            Dual::constant(g_z),
        ]);
        let dv = self.grid.eval(&[
            Dual::constant(theta),
            Dual::variable(v),
            Dual::constant(g_z),
        ]);
        let dg = self.grid.eval(&[
            Dual::constant(theta),
            Dual::constant(v),
            Dual::variable(g_z),
        ]);
        (dt.real, [dt.dual, dv.dual, dg.dual])
    }

    /// The OCP constraint value `c = |a| / rho(atan2(a_y,a_x); v, g_z)`; feasible
    /// iff `c <= 1`. At the origin (`|a| == 0`) returns `0`.
    pub fn constraint(&self, a_x: f64, a_y: f64, v: f64, g_z: f64) -> f64 {
        let mag = a_x.hypot(a_y);
        if mag == 0.0 {
            return 0.0;
        }
        let theta = a_y.atan2(a_x);
        mag / self.rho(theta, v, g_z)
    }

    // --- serialization ---

    /// Serialize to the versioned binary format. Deterministic: identical
    /// envelopes produce byte-identical output (all scalars little-endian, node
    /// values as raw IEEE-754 bits in storage order).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 + 32 + 80 + 8 * self.grid.values().len());
        b.extend_from_slice(ENVELOPE_MAGIC);
        b.extend_from_slice(self.key.as_bytes());
        let s = &self.spec;
        b.extend_from_slice(&(s.theta_res as u64).to_le_bytes());
        b.extend_from_slice(&(s.v_res as u64).to_le_bytes());
        b.extend_from_slice(&(s.gz_res as u64).to_le_bytes());
        b.extend_from_slice(&(s.coarse_steps as u64).to_le_bytes());
        for f in [
            s.v_min,
            s.v_max,
            s.gz_min,
            s.gz_max,
            s.max_accel,
            s.bisect_tol,
        ] {
            b.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        for &val in self.grid.values() {
            b.extend_from_slice(&val.to_bits().to_le_bytes());
        }
        b
    }

    /// Reconstruct from bytes written by [`Envelope::to_bytes`]. Validates the
    /// magic and body length; the stored key is trusted as-is (call
    /// [`Envelope::load_verified`] to also check it against live models).
    pub fn from_bytes(bytes: &[u8]) -> Result<Envelope, EnvelopeError> {
        let mut cur = bytes;
        let mut magic = [0u8; 8];
        read_exact(&mut cur, &mut magic)?;
        if &magic != ENVELOPE_MAGIC {
            return Err(EnvelopeError::BadMagic);
        }
        let mut key_bytes = [0u8; 32];
        read_exact(&mut cur, &mut key_bytes)?;
        let key = Hash::from_bytes(key_bytes);

        let theta_res = read_u64(&mut cur)? as usize;
        let v_res = read_u64(&mut cur)? as usize;
        let gz_res = read_u64(&mut cur)? as usize;
        let coarse_steps = read_u64(&mut cur)? as usize;
        let v_min = read_f64(&mut cur)?;
        let v_max = read_f64(&mut cur)?;
        let gz_min = read_f64(&mut cur)?;
        let gz_max = read_f64(&mut cur)?;
        let max_accel = read_f64(&mut cur)?;
        let bisect_tol = read_f64(&mut cur)?;

        let spec = EnvelopeGridSpec {
            theta_res,
            v_min,
            v_max,
            v_res,
            gz_min,
            gz_max,
            gz_res,
            max_accel,
            coarse_steps,
            bisect_tol,
        };
        spec.validate()?;

        let total = spec.total();
        let mut values = Vec::with_capacity(total);
        for _ in 0..total {
            values.push(read_f64(&mut cur)?);
        }
        if !cur.is_empty() {
            return Err(EnvelopeError::Corrupt);
        }
        let grid = HermiteGrid::new(spec.axes().to_vec(), values);
        Ok(Envelope { spec, key, grid })
    }

    /// Write to a file (creating parent directories).
    pub fn save(&self, path: &Path) -> Result<(), EnvelopeError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(&self.to_bytes())?;
        Ok(())
    }

    /// Read from a file (no key verification).
    pub fn load(path: &Path) -> Result<Envelope, EnvelopeError> {
        let bytes = std::fs::read(path)?;
        Envelope::from_bytes(&bytes)
    }

    /// Read from a file and verify its stored key matches the key recomputed
    /// from the given models + spec.
    pub fn load_verified(
        path: &Path,
        car: &CarParams,
        tire: &PacejkaTire,
        suspension: &SuspensionSystem,
        aero: &AeroModel,
        spec: &EnvelopeGridSpec,
    ) -> Result<Envelope, EnvelopeError> {
        let env = Envelope::load(path)?;
        let expected = envelope_key(car, tire, suspension, aero, spec);
        if env.key != expected {
            return Err(EnvelopeError::KeyMismatch {
                stored: env.key,
                expected,
            });
        }
        Ok(env)
    }

    /// Load a cached envelope if a valid, key-matching file exists at
    /// `<cache_dir>/<key>.apexenv`; otherwise generate, save, and return it. The
    /// filename is the content key, so distinct configs never collide.
    pub fn generate_cached(
        car: &CarParams,
        tire: &PacejkaTire,
        suspension: &SuspensionSystem,
        aero: &AeroModel,
        spec: EnvelopeGridSpec,
        cache_dir: &Path,
    ) -> Result<(Envelope, bool), EnvelopeError> {
        spec.validate()?;
        let key = envelope_key(car, tire, suspension, aero, &spec);
        let path = cache_dir.join(format!("{}.apexenv", key.to_hex()));
        if path.exists() {
            if let Ok(env) = Envelope::load_verified(&path, car, tire, suspension, aero, &spec) {
                return Ok((env, true));
            }
        }
        let env = Envelope::generate(car, tire, suspension, aero, spec)?;
        env.save(&path)?;
        Ok((env, false))
    }
}

// --- little-endian readers over a byte cursor ---

fn read_exact(cur: &mut &[u8], buf: &mut [u8]) -> Result<(), EnvelopeError> {
    Read::read_exact(cur, buf).map_err(|_| EnvelopeError::Corrupt)
}

fn read_u64(cur: &mut &[u8]) -> Result<u64, EnvelopeError> {
    let mut b = [0u8; 8];
    read_exact(cur, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f64(cur: &mut &[u8]) -> Result<f64, EnvelopeError> {
    Ok(f64::from_bits(read_u64(cur)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig() -> (CarParams, PacejkaTire, SuspensionSystem, AeroModel) {
        (
            CarParams::default(),
            PacejkaTire::f1_default(),
            SuspensionSystem::f1_default(),
            AeroModel::f1_default(),
        )
    }

    fn small_spec() -> EnvelopeGridSpec {
        EnvelopeGridSpec {
            theta_res: 16,
            v_min: 10.0,
            v_max: 80.0,
            v_res: 6,
            gz_min: 9.0,
            gz_max: 12.0,
            gz_res: 4,
            max_accel: 80.0,
            coarse_steps: 48,
            bisect_tol: 1e-3,
        }
    }

    // --- point-mass friction-circle limit ---

    /// With load-sensitivity and aero off, a low CoG, and generous actuator
    /// limits, the grip budget is `base_mu*m*g_z` for every direction, so the
    /// boundary radius must be the isotropic friction circle `base_mu*g_z` and
    /// linear in `g_z`.
    #[test]
    fn point_mass_friction_circle() {
        let car = CarParams {
            cog_height: 0.05,
            max_drive_force: 1e7,
            max_brake_force: 1e7,
            ..CarParams::default()
        };
        let mut tire = PacejkaTire::f1_default();
        tire.load_sensitivity = 0.0;
        let mut aero = AeroModel::f1_default();
        aero.cl_front_base = 0.0;
        aero.cl_rear_base = 0.0;
        let susp = SuspensionSystem::f1_default();
        let base_mu = 0.5 * (tire.lateral.mu + tire.longitudinal.mu);

        let spec = EnvelopeGridSpec {
            max_accel: 60.0,
            coarse_steps: 64,
            bisect_tol: 1e-4,
            ..small_spec()
        };
        let v = 5.0; // aero is zero regardless
        for gz in [9.0, 11.0, 13.0] {
            let expected = base_mu * gz;
            for theta in [0.0, 0.7, 1.5, 2.4, 3.1, 4.2, 5.5] {
                let r = boundary_radius(&car, &tire, &susp, &aero, theta, v, gz, &spec).unwrap();
                assert!(
                    (r - expected).abs() < 0.02,
                    "theta={theta} gz={gz}: rho={r} vs friction circle {expected}"
                );
            }
        }
        // linear in g_z
        let r1 = boundary_radius(&car, &tire, &susp, &aero, 1.5, v, 9.0, &spec).unwrap();
        let r2 = boundary_radius(&car, &tire, &susp, &aero, 1.5, v, 18.0, &spec).unwrap();
        assert!(
            (r2 / r1 - 2.0).abs() < 0.01,
            "rho must be linear in g_z: {r1} {r2}"
        );
    }

    // --- aero growth with speed ---

    #[test]
    fn aero_grows_cornering_grip_with_speed() {
        let (c, t, s, a) = rig();
        let spec = small_spec();
        // Pure cornering (theta = pi/2): downforce should raise the boundary at
        // high speed vs low speed.
        let gz = 9.81;
        let lo =
            boundary_radius(&c, &t, &s, &a, std::f64::consts::FRAC_PI_2, 15.0, gz, &spec).unwrap();
        let hi =
            boundary_radius(&c, &t, &s, &a, std::f64::consts::FRAC_PI_2, 80.0, gz, &spec).unwrap();
        assert!(
            hi > lo + 1.0,
            "downforce should grow cornering grip: lo={lo} hi={hi}"
        );
    }

    // --- interpolation error vs direct solve at off-grid points ---

    /// Max relative interpolation error `(interp - direct)/direct` over a set of
    /// off-grid points, plus the *signed* worst error (its sign says whether the
    /// interpolant over- or under-estimates rho at that point).
    fn interp_error(
        c: &CarParams,
        t: &PacejkaTire,
        s: &SuspensionSystem,
        a: &AeroModel,
        env: &Envelope,
    ) -> (f64, f64) {
        let mut max_rel = 0.0_f64;
        let mut worst_signed = 0.0_f64;
        for &theta in &[0.3, 1.1, 2.2, 3.7, 4.9] {
            for &v in &[22.0, 41.0, 63.0] {
                for &gz in &[9.5, 10.7] {
                    let interp = env.rho(theta, v, gz);
                    let direct = boundary_radius(c, t, s, a, theta, v, gz, env.spec()).unwrap();
                    let signed = (interp - direct) / direct.max(1e-6);
                    if signed.abs() > max_rel {
                        max_rel = signed.abs();
                        worst_signed = signed;
                    }
                }
            }
        }
        (max_rel, worst_signed)
    }

    #[test]
    fn interpolation_matches_direct_solve() {
        let (c, t, s, a) = rig();
        // Coarse grid (deliberately low resolution).
        let env_coarse = Envelope::generate(&c, &t, &s, &a, small_spec()).unwrap();
        let (coarse_rel, coarse_signed) = interp_error(&c, &t, &s, &a, &env_coarse);
        eprintln!(
            "max relative interpolation error (16x6x4 grid) = {coarse_rel:.4} \
             (worst signed {coarse_signed:+.4})"
        );
        assert!(
            coarse_rel < 0.05,
            "max relative interpolation error {coarse_rel} exceeds 5%"
        );

        // Default grid (24x10x6) — the number quoted in the design doc.
        let env_default = Envelope::generate(&c, &t, &s, &a, EnvelopeGridSpec::default()).unwrap();
        let (def_rel, def_signed) = interp_error(&c, &t, &s, &a, &env_default);
        eprintln!(
            "max relative interpolation error (24x10x6 default grid) = {def_rel:.4} \
             (worst signed {def_signed:+.4}, {})",
            if def_signed > 0.0 {
                "OVER-estimate"
            } else {
                "under-estimate"
            }
        );
        assert!(
            def_rel < 0.05,
            "default-grid interpolation error {def_rel} exceeds 5%"
        );
    }

    // --- monotone island detection ---

    #[test]
    fn boundary_is_monotone_for_physical_car() {
        // Real F1 car: every ray must have a monotone boundary (no island error).
        let (c, t, s, a) = rig();
        let spec = small_spec();
        // Generation asserts monotonicity on every ray; success == no island.
        let env = Envelope::generate(&c, &t, &s, &a, spec);
        assert!(
            env.is_ok(),
            "physical car should have star-shaped envelopes"
        );
    }

    // --- serialization round-trip & determinism ---

    #[test]
    fn round_trip_bytes() {
        let (c, t, s, a) = rig();
        let env = Envelope::generate(&c, &t, &s, &a, small_spec()).unwrap();
        let bytes = env.to_bytes();
        let back = Envelope::from_bytes(&bytes).unwrap();
        assert_eq!(env.key(), back.key());
        assert_eq!(env.values(), back.values());
        assert_eq!(
            bytes,
            back.to_bytes(),
            "re-serialization must be byte-identical"
        );
    }

    #[test]
    fn generation_is_deterministic_byte_identical() {
        let (c, t, s, a) = rig();
        let e1 = Envelope::generate(&c, &t, &s, &a, small_spec()).unwrap();
        let e2 = Envelope::generate(&c, &t, &s, &a, small_spec()).unwrap();
        assert_eq!(e1.key(), e2.key());
        assert_eq!(
            e1.to_bytes(),
            e2.to_bytes(),
            "regeneration must be byte-identical"
        );
    }

    #[test]
    fn key_is_model_and_spec_sensitive() {
        let (c, t, s, a) = rig();
        let base = envelope_key(&c, &t, &s, &a, &small_spec());
        // spec change
        let mut spec2 = small_spec();
        spec2.theta_res += 1;
        assert_ne!(base, envelope_key(&c, &t, &s, &a, &spec2));
        // car change
        let mut c2 = c.clone();
        c2.mass += 1.0;
        assert_ne!(base, envelope_key(&c2, &t, &s, &a, &small_spec()));
        // tire change
        let mut t2 = t;
        t2.lateral.mu += 0.01;
        assert_ne!(base, envelope_key(&c, &t2, &s, &a, &small_spec()));
        // aero change
        let mut a2 = a;
        a2.cd_base += 0.01;
        assert_ne!(base, envelope_key(&c, &t, &s, &a2, &small_spec()));
    }

    #[test]
    fn rho_grad_matches_finite_difference() {
        let (c, t, s, a) = rig();
        let env = Envelope::generate(&c, &t, &s, &a, small_spec()).unwrap();
        let (theta, v, gz) = (1.3, 45.0, 10.5);
        let (val, grad) = env.rho_grad(theta, v, gz);
        assert!((val - env.rho(theta, v, gz)).abs() < 1e-12);
        let h = 1e-5;
        let d_theta = (env.rho(theta + h, v, gz) - env.rho(theta - h, v, gz)) / (2.0 * h);
        let d_v = (env.rho(theta, v + h, gz) - env.rho(theta, v - h, gz)) / (2.0 * h);
        let d_gz = (env.rho(theta, v, gz + h) - env.rho(theta, v, gz - h)) / (2.0 * h);
        assert!(
            (grad[0] - d_theta).abs() < 1e-3,
            "d/dtheta {} vs {}",
            grad[0],
            d_theta
        );
        assert!((grad[1] - d_v).abs() < 1e-3, "d/dv {} vs {}", grad[1], d_v);
        assert!(
            (grad[2] - d_gz).abs() < 1e-3,
            "d/dg_z {} vs {}",
            grad[2],
            d_gz
        );
    }

    #[test]
    fn bad_spec_rejected() {
        let (c, t, s, a) = rig();
        let mut spec = small_spec();
        spec.v_res = 3;
        assert!(matches!(
            Envelope::generate(&c, &t, &s, &a, spec),
            Err(EnvelopeError::BadSpec(_))
        ));
    }

    #[test]
    fn bad_magic_rejected() {
        let bad = vec![0u8; 64];
        assert!(matches!(
            Envelope::from_bytes(&bad),
            Err(EnvelopeError::BadMagic)
        ));
    }
}
