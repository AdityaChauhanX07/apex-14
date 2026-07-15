//! Operating-point trim: quasi-steady state of the 14-DOF model at an arbitrary
//! `(v, a_x, a_y, g_z)` operating point.
//!
//! This is the inner evaluation an envelope sweep calls thousands of times. It
//! generalizes the straight-line [`solve_trim_with_gz`](crate::fourteen_dof) —
//! a symmetric heave+pitch solve — to a **heave / pitch / roll** equilibrium
//! that additionally balances the longitudinal load transfer of `a_x`, the
//! lateral load transfer of `a_y` (which breaks left/right symmetry, hence
//! roll), the ride-height-sensitive aero at the trimmed heights, and the
//! `g_z`-scaled weight from the g_z pathway.
//!
//! Feasibility is a first-class output: the trim distinguishes converged &
//! feasible, converged but infeasible (with the reason), and non-convergence —
//! it never clamps or fudges, because the envelope generator needs honest
//! boundaries. See `docs/design/envelope-qss/trim-solver.md`.

use crate::aero::AeroModel;
use crate::car_params::{CarParams, GRAVITY};
use crate::fourteen_dof::solve_trim_with_gz;
use crate::suspension::SuspensionSystem;
use crate::tire::PacejkaTire;

/// An operating point of the vehicle: speed and the demanded planar
/// accelerations, plus the imposed effective vertical acceleration `g_z`
/// (m/s²) from the g_z pathway (grade / bank / vertical-curvature).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperatingPoint {
    /// Speed (m/s).
    pub v: f64,
    /// Demanded longitudinal acceleration (m/s²); `> 0` accelerating, `< 0` braking.
    pub a_x: f64,
    /// Demanded lateral acceleration (m/s²); `> 0` = left turn (loads the right side).
    pub a_y: f64,
    /// Imposed effective vertical acceleration (m/s²). Defaults to [`GRAVITY`].
    pub g_z: f64,
}

impl Default for OperatingPoint {
    /// The stationary straight-line point under standard gravity: the trim
    /// reduces exactly to the model's static reference trim.
    fn default() -> Self {
        OperatingPoint {
            v: 0.0,
            a_x: 0.0,
            a_y: 0.0,
            g_z: GRAVITY,
        }
    }
}

/// Why a converged trim is physically infeasible. Checked in this priority order
/// (a negative load invalidates the grip estimate, so it is reported first).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InfeasibilityReason {
    /// A tire vertical load went negative — that corner has lifted. The trim is
    /// unphysical (real tires cannot pull the car down).
    NegativeLoad {
        /// Corner index `[fl, fr, rl, rr]`.
        corner: usize,
        /// The (negative) load (N).
        load: f64,
    },
    /// The demanded planar force `m·√(a_x²+a_y²)` exceeds the combined-slip,
    /// load-sensitive grip available at the trimmed corner loads.
    GripExceeded {
        /// Demanded planar force (N).
        demand: f64,
        /// Available grip (N).
        available: f64,
    },
    /// The demanded longitudinal force `m·a_x` exceeds the actuator limit
    /// (`max_drive_force` accelerating, `max_brake_force` braking).
    PowerLimit {
        /// Demanded longitudinal force magnitude (N).
        demand: f64,
        /// Available longitudinal force (N).
        available: f64,
    },
}

/// Outcome of a trim solve. Feasibility and convergence are reported honestly and
/// separately — the envelope generator maps these to boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrimStatus {
    /// Converged and all constraints satisfied.
    Feasible,
    /// Converged, but a physical constraint is violated (see the reason).
    Infeasible(InfeasibilityReason),
    /// The Newton iteration did not reach tolerance.
    NotConverged {
        /// Final max-abs force-balance residual (N).
        residual: f64,
    },
}

/// Result of an operating-point trim.
#[derive(Debug, Clone, Copy)]
pub struct TrimResult {
    /// Per-corner tire vertical loads `[fl, fr, rl, rr]` (N). Not floored — a
    /// negative entry means that corner lifted (see [`TrimStatus`]).
    pub loads: [f64; 4],
    /// Per-corner suspension travel `[fl, fr, rl, rr]` (m, +compression).
    pub travel: [f64; 4],
    /// Front / rear axle-average ride heights (m).
    pub ride_heights: (f64, f64),
    /// Chassis attitude `(heave, pitch, roll)` (m, rad, rad).
    pub attitude: (f64, f64, f64),
    /// Total aerodynamic downforce at the trimmed ride heights (N).
    pub downforce: f64,
    /// Final max-abs force-balance residual (N).
    pub residual: f64,
    /// Newton iterations performed.
    pub iterations: usize,
    /// Feasibility / convergence status.
    pub status: TrimStatus,
}

impl TrimResult {
    /// Convenience: `true` iff the status is [`TrimStatus::Feasible`].
    pub fn is_feasible(&self) -> bool {
        matches!(self.status, TrimStatus::Feasible)
    }
}

/// Invalid operating-point input. `solve_operating_point` rejects these rather
/// than returning a nonsense trim (see the negative-`g_z` ruling in the design
/// doc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperatingPointError {
    /// `g_z <= 0`. The normal-load / grip budget is undefined for a
    /// non-positive effective gravity (tires would be in tension). Rejected as an
    /// input error; the envelope sweeps only physical `g_z > 0`.
    NonPositiveGravity {
        /// The offending value.
        g_z: f64,
    },
    /// A non-finite input (`NaN`/`inf`) in `v`, `a_x`, `a_y`, or `g_z`.
    NonFinite,
}

// Newton controls. Fixed (no RNG, fixed iteration order) so identical inputs
// produce bit-identical outputs — required for the content-hashed cache later.
const MAX_ITERS: usize = 50;
const STEP_TOL: f64 = 1e-12; // convergence: |Δ(heave,pitch,roll)| sum (m/rad)
const RESIDUAL_TOL: f64 = 1e-4; // "converged" gate on max-abs force residual (N)
const JAC_EPS: f64 = 1e-7; // finite-difference step for the 3x3 Jacobian

/// Corner longitudinal offsets `x_off` (m): front `+lf`, rear `-lr`.
#[inline]
fn x_offsets(car: &CarParams) -> [f64; 4] {
    [
        car.cog_to_front,
        car.cog_to_front,
        -car.cog_to_rear,
        -car.cog_to_rear,
    ]
}

/// Corner lateral offsets `y_off` (m): left `+t/2`, right `-t/2`.
#[inline]
fn y_offsets(car: &CarParams) -> [f64; 4] {
    [
        car.track_width_front * 0.5,
        -car.track_width_front * 0.5,
        car.track_width_rear * 0.5,
        -car.track_width_rear * 0.5,
    ]
}

/// The four tire vertical loads for a chassis attitude `(heave, pitch, roll)`,
/// plus the travels and total downforce. Mirrors the sign conventions of the
/// 14-DOF `tire_loads`/`solve_trim` (spring force negative in compression, plus
/// unsprung weight `m_u·g_z`; ARB couples left/right).
fn corner_state(
    car: &CarParams,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    op: &OperatingPoint,
    heave: f64,
    pitch: f64,
    roll: f64,
) -> ([f64; 4], [f64; 4], f64) {
    let x_off = x_offsets(car);
    let y_off = y_offsets(car);
    // Rigid-plane corner travels (planarity/warp constraint enforced by using
    // only 3 chassis DOF): z_i = heave - x_off_i·pitch + y_off_i·roll.
    let z = [
        heave - x_off[0] * pitch + y_off[0] * roll,
        heave - x_off[1] * pitch + y_off[1] * roll,
        heave - x_off[2] * pitch + y_off[2] * roll,
        heave - x_off[3] * pitch + y_off[3] * roll,
    ];

    // Suspension force = spring + anti-roll bar (static: no damper). Tire load =
    // -(suspension force) + unsprung weight, matching solve_trim.
    let dz = [0.0; 4];
    let sus = suspension.forces(&z, &dz);
    let unsprung = car.unsprung_mass * op.g_z;
    let loads = [
        -sus[0] + unsprung,
        -sus[1] + unsprung,
        -sus[2] + unsprung,
        -sus[3] + unsprung,
    ];

    let front_rh = aero.design_ride_height - 0.5 * (z[0] + z[1]);
    let rear_rh = aero.design_ride_height - 0.5 * (z[2] + z[3]);
    let af = aero.compute(op.v, front_rh, rear_rh, pitch);

    (z, loads, af.downforce_total)
}

/// The three force/moment-balance residuals (heave, pitch, roll), in newtons and
/// newton-metres, at a chassis attitude. Zero residual ⇒ quasi-steady trim.
///
/// - Heave:  ΣFz − m·g_z − Df = 0
/// - Pitch:  lf·(Fz_fl+Fz_fr) − lr·(Fz_rl+Fz_rr) + m·a_x·h = 0  (Δfront = −m·a_x·h/L)
/// - Roll:   (tf/2)(Fz_fl−Fz_fr) + (tr/2)(Fz_rl−Fz_rr) + m·a_y·h = 0
///
/// The pitch/roll inertial terms reproduce the closed-form load transfer
/// `m·a·h/L` (verified analytically in the tests). Aero enters only through the
/// heave balance and the ride-height feedback, exactly as the legacy trim treats
/// it (no separate aero pitch term), so the `(a_x, a_y) = (0,0)` limit matches
/// the legacy heave+pitch balance.
fn residuals(car: &CarParams, loads: &[f64; 4], downforce: f64, op: &OperatingPoint) -> [f64; 3] {
    let m = car.mass;
    let h = car.cog_height;
    let lf = car.cog_to_front;
    let lr = car.cog_to_rear;
    let tf = car.track_width_front;
    let tr = car.track_width_rear;

    let r_heave = loads[0] + loads[1] + loads[2] + loads[3] - m * op.g_z - downforce;
    let r_pitch = lf * (loads[0] + loads[1]) - lr * (loads[2] + loads[3]) + m * op.a_x * h;
    let r_roll =
        0.5 * tf * (loads[0] - loads[1]) + 0.5 * tr * (loads[2] - loads[3]) + m * op.a_y * h;

    [r_heave, r_pitch, r_roll]
}

/// Solve `A·x = b` for a 3×3 system by Cramer's rule. Returns `None` if singular.
/// Deterministic, allocation-free.
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let col = |c: usize| {
        let mut m = *a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    Some([col(0) / det, col(1) / det, col(2) / det])
}

fn max_abs(r: &[f64; 3]) -> f64 {
    r.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Combined-slip, load-sensitive grip available at the given corner loads
/// (negative loads floored at 0). Same budget formula as
/// `apex_optimizer::fourteen_dof_grip_budget`'s final step.
fn available_grip(tire: &PacejkaTire, loads: &[f64; 4]) -> f64 {
    let base_mu = 0.5 * (tire.lateral.mu + tire.longitudinal.mu);
    loads
        .iter()
        .map(|&fz| {
            let fz = fz.max(0.0);
            tire.effective_mu(base_mu, fz) * fz
        })
        .sum()
}

/// Classify feasibility of a converged trim. Priority: negative load (the trim is
/// unphysical and the grip estimate would be meaningless), then grip, then power.
fn classify(
    car: &CarParams,
    tire: &PacejkaTire,
    loads: &[f64; 4],
    op: &OperatingPoint,
) -> TrimStatus {
    // 1. Negative tire load — a corner lifted.
    for (i, &fz) in loads.iter().enumerate() {
        if fz < 0.0 {
            return TrimStatus::Infeasible(InfeasibilityReason::NegativeLoad {
                corner: i,
                load: fz,
            });
        }
    }
    // 2. Combined-slip grip budget.
    let demand = car.mass * op.a_x.hypot(op.a_y);
    let available = available_grip(tire, loads);
    if demand > available {
        return TrimStatus::Infeasible(InfeasibilityReason::GripExceeded { demand, available });
    }
    // 3. Longitudinal actuator (traction / brake force) limit.
    let f_x = car.mass * op.a_x;
    if op.a_x > 0.0 && f_x > car.max_drive_force {
        return TrimStatus::Infeasible(InfeasibilityReason::PowerLimit {
            demand: f_x,
            available: car.max_drive_force,
        });
    }
    if op.a_x < 0.0 && -f_x > car.max_brake_force {
        return TrimStatus::Infeasible(InfeasibilityReason::PowerLimit {
            demand: -f_x,
            available: car.max_brake_force,
        });
    }
    TrimStatus::Feasible
}

/// Solve the model's quasi-steady state at an operating point.
///
/// Returns `Err` for invalid inputs (non-positive `g_z`, non-finite values); the
/// envelope sweeps only physical `g_z > 0`. Otherwise always returns a
/// `TrimResult` whose `status` reports feasibility/convergence honestly — the
/// caller (envelope generator) decides how to treat infeasible/non-converged
/// points. Never clamps loads or accelerations.
///
/// At the symmetric straight-line point `(a_x, a_y) = (0, 0)` this delegates to
/// the legacy [`solve_trim_with_gz`](crate::fourteen_dof) so the result is
/// **bit-identical** to the model's static reference trim; otherwise it runs the
/// 3-DOF heave/pitch/roll Newton described above.
pub fn solve_operating_point(
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    op: OperatingPoint,
) -> Result<TrimResult, OperatingPointError> {
    if !(op.v.is_finite() && op.a_x.is_finite() && op.a_y.is_finite() && op.g_z.is_finite()) {
        return Err(OperatingPointError::NonFinite);
    }
    if op.g_z <= 0.0 {
        return Err(OperatingPointError::NonPositiveGravity { g_z: op.g_z });
    }

    let lf = car.cog_to_front;
    let lr = car.cog_to_rear;
    let wb = lf + lr;

    // Symmetric straight-line point: reuse the exact legacy solve (bitwise).
    if op.a_x == 0.0 && op.a_y == 0.0 {
        let (z, loads) = solve_trim_with_gz(car, suspension, aero, op.v, op.g_z);
        // Recover (heave, pitch, roll=0) from the front/rear compressions.
        let (zf, zr) = (z[0], z[2]);
        let heave = (lr * zf + lf * zr) / wb;
        let pitch = (zr - zf) / wb;
        let (_, _, downforce) = corner_state(car, suspension, aero, &op, heave, pitch, 0.0);
        let residual = max_abs(&residuals(car, &loads, downforce, &op));
        let front_rh = aero.design_ride_height - zf;
        let rear_rh = aero.design_ride_height - zr;
        let status = classify(car, tire, &loads, &op);
        return Ok(TrimResult {
            loads,
            travel: z,
            ride_heights: (front_rh, rear_rh),
            attitude: (heave, pitch, 0.0),
            downforce,
            residual,
            iterations: 0,
            status,
        });
    }

    // General 3-DOF Newton. Initial guess: the legacy straight-line compressions
    // (a_x = a_y = 0), mapped to (heave, pitch), roll = 0 — this starts near the
    // solution so convergence is fast and deterministic.
    let (z0, _) = solve_trim_with_gz(car, suspension, aero, op.v, op.g_z);
    let mut heave = (lr * z0[0] + lf * z0[2]) / wb;
    let mut pitch = (z0[2] - z0[0]) / wb;
    let mut roll = 0.0_f64;

    let eval = |h: f64, p: f64, r: f64| -> [f64; 3] {
        let (_, loads, df) = corner_state(car, suspension, aero, &op, h, p, r);
        residuals(car, &loads, df, &op)
    };

    let mut iterations = 0;
    let mut res = eval(heave, pitch, roll);
    for _ in 0..MAX_ITERS {
        iterations += 1;
        // Finite-difference 3x3 Jacobian (columns = ∂res/∂{heave,pitch,roll}).
        let rh = eval(heave + JAC_EPS, pitch, roll);
        let rp = eval(heave, pitch + JAC_EPS, roll);
        let rr = eval(heave, pitch, roll + JAC_EPS);
        let jac = [
            [
                (rh[0] - res[0]) / JAC_EPS,
                (rp[0] - res[0]) / JAC_EPS,
                (rr[0] - res[0]) / JAC_EPS,
            ],
            [
                (rh[1] - res[1]) / JAC_EPS,
                (rp[1] - res[1]) / JAC_EPS,
                (rr[1] - res[1]) / JAC_EPS,
            ],
            [
                (rh[2] - res[2]) / JAC_EPS,
                (rp[2] - res[2]) / JAC_EPS,
                (rr[2] - res[2]) / JAC_EPS,
            ],
        ];
        let neg_res = [-res[0], -res[1], -res[2]];
        let Some(delta) = solve_3x3(&jac, &neg_res) else {
            break;
        };
        heave += delta[0];
        pitch += delta[1];
        roll += delta[2];
        res = eval(heave, pitch, roll);
        if delta[0].abs() + delta[1].abs() + delta[2].abs() < STEP_TOL {
            break;
        }
    }

    let (z, loads, downforce) = corner_state(car, suspension, aero, &op, heave, pitch, roll);
    let residual = max_abs(&res);
    let front_rh = aero.design_ride_height - 0.5 * (z[0] + z[1]);
    let rear_rh = aero.design_ride_height - 0.5 * (z[2] + z[3]);

    let status = if residual > RESIDUAL_TOL {
        TrimStatus::NotConverged { residual }
    } else {
        classify(car, tire, &loads, &op)
    };

    Ok(TrimResult {
        loads,
        travel: z,
        ride_heights: (front_rh, rear_rh),
        attitude: (heave, pitch, roll),
        downforce,
        residual,
        iterations,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourteen_dof::solve_trim_with_gz;

    fn rig() -> (CarParams, PacejkaTire, SuspensionSystem, AeroModel) {
        (
            CarParams::default(),
            PacejkaTire::f1_default(),
            SuspensionSystem::f1_default(),
            AeroModel::f1_default(),
        )
    }

    fn solve(op: OperatingPoint) -> TrimResult {
        let (c, t, s, a) = rig();
        solve_operating_point(&c, &t, &s, &a, op).expect("valid op")
    }

    // --- input validation / negative-g_z ruling ---

    #[test]
    fn rejects_non_positive_gravity() {
        let (c, t, s, a) = rig();
        for g in [0.0, -1.0, -GRAVITY] {
            let op = OperatingPoint {
                v: 40.0,
                g_z: g,
                ..Default::default()
            };
            assert_eq!(
                solve_operating_point(&c, &t, &s, &a, op).unwrap_err(),
                OperatingPointError::NonPositiveGravity { g_z: g }
            );
        }
    }

    #[test]
    fn rejects_non_finite() {
        let (c, t, s, a) = rig();
        for op in [
            OperatingPoint {
                v: f64::NAN,
                ..Default::default()
            },
            OperatingPoint {
                a_x: f64::INFINITY,
                ..Default::default()
            },
            OperatingPoint {
                a_y: f64::NAN,
                ..Default::default()
            },
            OperatingPoint {
                g_z: f64::INFINITY,
                ..Default::default()
            },
        ] {
            assert_eq!(
                solve_operating_point(&c, &t, &s, &a, op).unwrap_err(),
                OperatingPointError::NonFinite
            );
        }
    }

    // --- consistency: symmetric point is bit-identical to the legacy trim ---

    #[test]
    fn straight_line_matches_legacy_trim_bitwise() {
        let (c, _t, s, a) = rig();
        for &v in &[0.0, 40.0, 80.0] {
            let (z_legacy, fz_legacy) = solve_trim_with_gz(&c, &s, &a, v, GRAVITY);
            let r = solve(OperatingPoint {
                v,
                ..Default::default()
            });
            for i in 0..4 {
                assert_eq!(
                    r.loads[i].to_bits(),
                    fz_legacy[i].to_bits(),
                    "load[{i}] v={v}"
                );
                assert_eq!(
                    r.travel[i].to_bits(),
                    z_legacy[i].to_bits(),
                    "travel[{i}] v={v}"
                );
            }
            assert_eq!(r.iterations, 0, "symmetric point delegates");
        }
    }

    #[test]
    fn deterministic_bitwise() {
        let op = OperatingPoint {
            v: 55.0,
            a_x: -6.0,
            a_y: 18.0,
            g_z: GRAVITY,
        };
        let r1 = solve(op);
        let r2 = solve(op);
        for i in 0..4 {
            assert_eq!(r1.loads[i].to_bits(), r2.loads[i].to_bits());
            assert_eq!(r1.travel[i].to_bits(), r2.travel[i].to_bits());
        }
    }

    // --- analytical anchors (v=0 ⇒ no aero ⇒ exact closed-form transfer) ---

    #[test]
    fn longitudinal_transfer_matches_closed_form() {
        let (c, _t, _s, _a) = rig();
        let l = c.wheelbase;
        let base = solve(OperatingPoint {
            v: 0.0,
            ..Default::default()
        });
        let front0 = base.loads[0] + base.loads[1];
        for &ax in &[-9.0, -4.0, 4.0, 8.0] {
            let r = solve(OperatingPoint {
                v: 0.0,
                a_x: ax,
                ..Default::default()
            });
            assert!(matches!(
                r.status,
                TrimStatus::Feasible | TrimStatus::Infeasible(_)
            ));
            let front = r.loads[0] + r.loads[1];
            let d_front = front - front0;
            // ΔF_front_axle = −m·a_x·h/L
            let expected = -c.mass * ax * c.cog_height / l;
            assert!(
                (d_front - expected).abs() < 1e-6,
                "ax={ax}: Δfront {d_front} vs closed-form {expected}"
            );
        }
    }

    #[test]
    fn braking_loads_front_acceleration_loads_rear() {
        let base = solve(OperatingPoint {
            v: 0.0,
            ..Default::default()
        });
        let (f0, r0) = (base.loads[0] + base.loads[1], base.loads[2] + base.loads[3]);
        let brake = solve(OperatingPoint {
            v: 0.0,
            a_x: -8.0,
            ..Default::default()
        });
        assert!(
            brake.loads[0] + brake.loads[1] > f0,
            "braking should load the front"
        );
        assert!(
            brake.loads[2] + brake.loads[3] < r0,
            "braking should unload the rear"
        );
        let accel = solve(OperatingPoint {
            v: 0.0,
            a_x: 5.0,
            ..Default::default()
        });
        assert!(
            accel.loads[2] + accel.loads[3] > r0,
            "acceleration should load the rear"
        );
    }

    #[test]
    fn lateral_transfer_matches_closed_form_and_loads_outside() {
        let (c, _t, _s, _a) = rig();
        for &ay in &[6.0, 15.0, 25.0] {
            let r = solve(OperatingPoint {
                v: 0.0,
                a_y: ay,
                ..Default::default()
            });
            // total roll-moment balance: (tf/2)(fl−fr)+(tr/2)(rl−rr) = −m·a_y·h
            let lhs = 0.5 * c.track_width_front * (r.loads[0] - r.loads[1])
                + 0.5 * c.track_width_rear * (r.loads[2] - r.loads[3]);
            let expected = -c.mass * ay * c.cog_height;
            assert!(
                (lhs - expected).abs() < 1e-6,
                "ay={ay}: roll moment {lhs} vs closed-form {expected}"
            );
            // positive a_y = left turn ⇒ right wheels (fr, rr) gain load
            assert!(
                r.loads[1] > r.loads[0],
                "front-right should carry more than front-left"
            );
            assert!(
                r.loads[3] > r.loads[2],
                "rear-right should carry more than rear-left"
            );
        }
    }

    #[test]
    fn aero_load_grows_with_speed_matching_the_map() {
        let (c, _t, _s, a) = rig();
        let lo = solve(OperatingPoint {
            v: 0.0,
            ..Default::default()
        });
        let hi = solve(OperatingPoint {
            v: 80.0,
            ..Default::default()
        });
        let sum = |r: &TrimResult| r.loads.iter().sum::<f64>();
        // total tire load = m·g_z + downforce (heave balance) ⇒ grows by downforce(v)
        assert!(
            (sum(&lo) - c.mass * GRAVITY).abs() < 1e-3,
            "static load ≈ m·g"
        );
        assert!(sum(&hi) > sum(&lo), "downforce should add load at speed");
        assert!((sum(&hi) - c.mass * GRAVITY - hi.downforce).abs() < 1e-3);
        // reported downforce matches the aero map evaluated at the trimmed heights
        let (frh, rrh) = hi.ride_heights;
        let af = a.compute(80.0, frh, rrh, hi.attitude.1);
        assert!((hi.downforce - af.downforce_total).abs() < 1e-9);
    }

    // --- feasibility boundary ---

    /// Sweep `a_y` upward at fixed `(v, g_z)`; return (boundary index, statuses).
    #[allow(clippy::too_many_arguments)]
    fn sweep_ay(
        car: &CarParams,
        tire: &PacejkaTire,
        susp: &SuspensionSystem,
        aero: &AeroModel,
        v: f64,
        g_z: f64,
        step: f64,
        n: usize,
    ) -> Vec<(f64, TrimStatus)> {
        (0..n)
            .map(|i| {
                let ay = i as f64 * step;
                let op = OperatingPoint {
                    v,
                    a_x: 0.0,
                    a_y: ay,
                    g_z,
                };
                let r = solve_operating_point(car, tire, susp, aero, op).unwrap();
                (ay, r.status)
            })
            .collect()
    }

    #[test]
    fn feasibility_boundary_is_monotone_in_ay() {
        let (c, t, s, a) = rig();
        let sweep = sweep_ay(&c, &t, &s, &a, 40.0, GRAVITY, 0.25, 400);
        // find the first infeasible/non-feasible point
        let first_bad = sweep
            .iter()
            .position(|(_, st)| !matches!(st, TrimStatus::Feasible));
        let idx = first_bad.expect("high enough a_y must become infeasible");
        assert!(idx > 0, "small a_y must be feasible");
        // monotone: once feasibility is lost it never returns (no feasible islands)
        for (ay, st) in &sweep[idx..] {
            assert!(
                !matches!(st, TrimStatus::Feasible),
                "feasible island past the boundary at a_y={ay}: {st:?}"
            );
        }
        // the boundary is a real physical limit, not solver failure
        let (ay_b, st_b) = sweep[idx];
        assert!(
            matches!(
                st_b,
                TrimStatus::Infeasible(InfeasibilityReason::GripExceeded { .. })
                    | TrimStatus::Infeasible(InfeasibilityReason::NegativeLoad { .. })
            ),
            "boundary reason at a_y={ay_b} should be grip/load, got {st_b:?}"
        );
    }

    #[test]
    fn grip_exceeded_is_reported_for_absurd_demand() {
        // 10 g of lateral demand at moderate speed is far beyond any grip budget.
        let r = solve(OperatingPoint {
            v: 30.0,
            a_y: 100.0,
            ..Default::default()
        });
        assert!(
            matches!(r.status, TrimStatus::Infeasible(_)),
            "got {:?}",
            r.status
        );
    }

    // --- g_z scaling of the boundary (aero-free, load-sensitivity-free limit) ---

    #[test]
    fn boundary_ay_scales_linearly_with_gz_in_grip_limited_limit() {
        // Aero-free (Cl=0) and load-sensitivity-free tire ⇒ the grip budget is
        // base_mu·m·g_z and the a_y boundary is base_mu·g_z — LINEAR in g_z.
        // (The sqrt(g_z) law from the g_z pathway applies to cornering *speed* at
        // a fixed radius, where a_y = v²/R; for an independent a_y axis at fixed v
        // the boundary is first-order in g_z. See trim-solver.md.)
        // lower CoG so a wheel does not lift before the grip limit, isolating
        // the grip boundary.
        let car = CarParams {
            cog_height: 0.10,
            ..CarParams::default()
        };
        let mut tire = PacejkaTire::f1_default();
        tire.load_sensitivity = 0.0;
        let mut aero = AeroModel::f1_default();
        aero.cl_front_base = 0.0;
        aero.cl_rear_base = 0.0;
        let susp = SuspensionSystem::f1_default();
        let v = 5.0; // low speed; aero is zero anyway

        let boundary = |g_z: f64| -> f64 {
            let sweep = sweep_ay(&car, &tire, &susp, &aero, v, g_z, 0.02, 4000);
            sweep
                .iter()
                .position(|(_, st)| !matches!(st, TrimStatus::Feasible))
                .map(|i| sweep[i].0)
                .expect("boundary exists")
        };

        let base_mu = 0.5 * (tire.lateral.mu + tire.longitudinal.mu);
        let b1 = boundary(GRAVITY);
        let b2 = boundary(2.0 * GRAVITY);
        // analytical: boundary = base_mu · g_z
        assert!(
            (b1 - base_mu * GRAVITY).abs() < 0.1,
            "b1 {b1} vs {}",
            base_mu * GRAVITY
        );
        // linear scaling: doubling g_z doubles the boundary a_y
        assert!(
            (b2 / b1 - 2.0).abs() < 0.02,
            "b2/b1 {} should be ~2",
            b2 / b1
        );
    }
}
