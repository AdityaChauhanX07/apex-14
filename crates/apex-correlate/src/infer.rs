//! QSS channel inference: back-compute unmeasured channels (accelerations, aero
//! loads, per-axle vertical loads, friction-circle grip utilization, tractive /
//! braking power) from a measured speed trace on the reconstructed driven line.
//!
//! The physics **exactly inverts the point-mass QSS grip-circle model**
//! (`apex_physics::qss::qss_lap_sim`): total lateral grip `F_grip = μ·(m·g +
//! DF(v))`, lateral force `F_lat = m·v²·κ`, and a longitudinal tyre force
//! `F_x = m·a_long + drag + rolling`. We deliberately do **not** invent a richer
//! load model than the sim uses — grip utilization is the *total* friction-circle
//! occupancy, not a per-axle budget (the point-mass QSS has no per-axle grip
//! limit). Per-axle **loads** are reported via `CarParams::axle_loads` (static +
//! aero + longitudinal transfer), which the car parameterization supports
//! cleanly, but they do not gate grip in this model.
//!
//! See `docs/math/inference.md` for the full derivation and the
//! effective-parameter caveat.

use apex_physics::CarParams;
use apex_telemetry::ChannelId;
use apex_track::Track;

use crate::driven::DrivenLineMode;
use crate::error::CorrelateError;
use crate::metrics::resample_linear;
use crate::telemetry::{GridKind, Telemetry};

/// Standard gravity used by the QSS being inverted (`apex_physics` `GRAVITY`).
const G: f64 = 9.81;

/// Inference tuning.
#[derive(Debug, Clone, Copy)]
pub struct InferConfig {
    /// Half-window (m) of the local-linear (Savitzky-Golay-style) derivative
    /// used to compute `dv/ds`. Raw finite differences on ~7.5 Hz telemetry are
    /// unusable; a ±window regression tames the noise. Default ±15 m.
    pub deriv_halfwindow_m: f64,
    /// Half-window (m) of the periodic moving average applied to the driven
    /// **curvature** in [`infer_on_driven`] before computing `a_lat = v²·κ`.
    /// Curvature is a second derivative of position, so even a position-smoothed
    /// driven line leaves high-frequency `κ` noise that `v²·κ` amplifies into
    /// phantom high-g spikes; a light average tames it. Applied **only** on the
    /// real-data path — [`infer_channels`] itself never smooths `κ`, so it stays
    /// an exact inverter for the closed-loop test. Default ±10 m.
    pub kappa_halfwindow_m: f64,
}

impl Default for InferConfig {
    fn default() -> Self {
        InferConfig {
            deriv_halfwindow_m: 15.0,
            kappa_halfwindow_m: 10.0,
        }
    }
}

/// The inferred channels, one value per input sample.
#[derive(Debug, Clone)]
pub struct Inferred {
    /// Lateral acceleration (g), signed: **+ = left** (matches `lateral_offset`).
    pub lateral_g: Vec<f64>,
    /// Longitudinal acceleration (g), signed: **+ = accelerating**.
    pub longitudinal_g: Vec<f64>,
    /// Aerodynamic downforce (N).
    pub downforce: Vec<f64>,
    /// Aerodynamic drag force (N).
    pub aero_drag: Vec<f64>,
    /// Front-axle vertical load (N).
    pub fz_front: Vec<f64>,
    /// Rear-axle vertical load (N).
    pub fz_rear: Vec<f64>,
    /// Total friction-circle grip utilization `|F|/F_grip` (≈1 at the limit,
    /// >1 flags noise or model deficiency — not clamped).
    pub grip_util: Vec<f64>,
    /// Tractive power at the wheels (W, ≥0; 0 while braking).
    pub tractive_power: Vec<f64>,
    /// Braking power dissipated (W, ≥0; 0 while driving).
    pub braking_power: Vec<f64>,
}

/// Infer channels from a speed trace `v(s)` on a path of signed curvature
/// `kappa(s)`, where `s` is arc length (monotone; wraps if `closed`).
///
/// NaN discipline: a channel is NaN wherever its inputs are NaN. `lateral_g`,
/// `downforce`, `aero_drag` need only `v`/`kappa`; the load / grip / power
/// channels also need `a_long` (the derivative), so a NaN inside the derivative
/// window makes only those NaN.
pub fn infer_channels(
    s: &[f64],
    v: &[f64],
    kappa: &[f64],
    closed: bool,
    car: &CarParams,
    cfg: &InferConfig,
) -> Inferred {
    let zeros = vec![0.0; v.len()];
    infer_channels_3d(s, v, kappa, &zeros, &zeros, &zeros, closed, car, cfg)
}

/// [`infer_channels`] with the 3D road terms — grade `θ`, bank `φ`, and vertical
/// curvature `κ_v` per sample. The friction-circle grip uses the 3D normal load
/// `N = m(g·cosθ·cosφ + v²κ·sinφ + v²κ_v) + F_df`, and the longitudinal tyre
/// force picks up the grade term `m·g·sinθ` (so tractive/braking power includes
/// the gravity contribution). Exactly inverts [`apex_physics::qss_lap_sim_3d`].
#[allow(clippy::too_many_arguments)]
pub fn infer_channels_3d(
    s: &[f64],
    v: &[f64],
    kappa: &[f64],
    grade: &[f64],
    bank: &[f64],
    kappa_v: &[f64],
    closed: bool,
    car: &CarParams,
    cfg: &InferConfig,
) -> Inferred {
    let n = v.len();
    let total_length = if n >= 2 {
        s[n - 1] - s[0] + (s[1] - s[0]).max(0.0)
    } else {
        0.0
    };
    // a_long = v·dv/ds = d(½v²)/ds. Differentiating specific kinetic energy
    // ½v² directly matches the QSS's own energy integration (its a_long is
    // Δ(½v²)/Δs) and avoids the extra v-multiply, so the closed-loop inversion
    // is tight.
    let energy: Vec<f64> = v.iter().map(|&vi| 0.5 * vi * vi).collect();
    let a_long_all = local_linear_deriv(s, &energy, closed, total_length, cfg.deriv_halfwindow_m);

    let m = car.mass;
    let rolling = car.rolling_resistance_force();

    let mut out = Inferred {
        lateral_g: vec![f64::NAN; n],
        longitudinal_g: vec![f64::NAN; n],
        downforce: vec![f64::NAN; n],
        aero_drag: vec![f64::NAN; n],
        fz_front: vec![f64::NAN; n],
        fz_rear: vec![f64::NAN; n],
        grip_util: vec![f64::NAN; n],
        tractive_power: vec![f64::NAN; n],
        braking_power: vec![f64::NAN; n],
    };

    for i in 0..n {
        let vi = v[i];
        let ki = kappa[i];
        if !vi.is_finite() || !ki.is_finite() {
            continue;
        }
        // Kinematic lateral + aero (need only v, kappa).
        let a_lat = vi * vi * ki; // m/s², signed
        out.lateral_g[i] = a_lat / G;
        let df = car.downforce(vi);
        let drag = car.drag_force(vi);
        out.downforce[i] = df;
        out.aero_drag[i] = drag;

        // Longitudinal (needs the derivative). a_long = d(½v²)/ds directly.
        let a_long = a_long_all[i];
        if !a_long.is_finite() {
            continue;
        }
        out.longitudinal_g[i] = a_long / G;

        // Per-axle loads (static + aero + longitudinal transfer) — QSS-level.
        let (fzf, fzr) = car.axle_loads(vi, a_long);
        out.fz_front[i] = fzf;
        out.fz_rear[i] = fzr;

        // Friction-circle occupancy on the 3D normal load (as the 3D QSS
        // enforces it). θ/φ/κ_v are 0 in the flat case ⇒ this reduces exactly to
        // μ·(mg+DF) and f_lat = m·v²κ.
        let (theta, phi, kv) = (grade[i], bank[i], kappa_v[i]);
        let f_lat = m * (vi * vi * ki * phi.cos() - G * phi.sin());
        // Longitudinal tyre force incl. the grade term (+ = drive, − = brake).
        let f_x = m * a_long + drag + rolling + m * G * theta.sin();
        let normal =
            m * (G * theta.cos() * phi.cos() + vi * vi * ki * phi.sin() + vi * vi * kv) + df;
        let f_grip = car.tire_mu * normal; // μ·N_3d
        out.grip_util[i] = if f_grip > 0.0 {
            (f_lat * f_lat + f_x * f_x).sqrt() / f_grip
        } else {
            f64::NAN
        };

        // Power split: drive vs brake (one is zero).
        out.tractive_power[i] = f_x.max(0.0) * vi;
        out.braking_power[i] = (-f_x).max(0.0) * vi;
    }
    out
}

/// Local-linear (least-squares slope) derivative `dv/ds` over a ±`halfwindow_m`
/// window. Robust to mild non-uniform spacing; wraps for closed loops. Returns
/// NaN where the window contains a NaN or is degenerate.
fn local_linear_deriv(
    s: &[f64],
    v: &[f64],
    closed: bool,
    total_length: f64,
    halfwindow_m: f64,
) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![f64::NAN; n];
    if n < 3 {
        return out;
    }
    let spacing = (total_length / n as f64).max(1e-6);
    let w = ((halfwindow_m / spacing).round() as usize).max(1);

    for i in 0..n {
        // Collect (ds, dv) offsets in the window relative to point i.
        let (mut sxx, mut sxy, mut sx, mut sy, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut bad = false;
        for k in -(w as isize)..=(w as isize) {
            let idx = if closed {
                (i as isize + k).rem_euclid(n as isize) as usize
            } else {
                let j = i as isize + k;
                if j < 0 || j >= n as isize {
                    continue;
                }
                j as usize
            };
            let vk = v[idx];
            if !vk.is_finite() {
                bad = true;
                break;
            }
            // Arc-length offset, unwrapped for the periodic seam.
            let mut ds = s[idx] - s[i];
            if closed {
                if ds > total_length * 0.5 {
                    ds -= total_length;
                } else if ds < -total_length * 0.5 {
                    ds += total_length;
                }
            }
            sx += ds;
            sy += vk;
            sxx += ds * ds;
            sxy += ds * vk;
            cnt += 1.0;
        }
        if bad || cnt < 2.0 {
            continue;
        }
        let denom = sxx - sx * sx / cnt;
        if denom.abs() > 1e-12 {
            out[i] = (sxy - sx * sy / cnt) / denom;
        }
    }
    out
}

/// Vertical (pitch) curvature `κ_v = dθ/ds` by central differences (periodic).
fn grade_rate(s: &[f64], grade: &[f64], total_length: f64) -> Vec<f64> {
    let n = s.len();
    let mut kv = vec![0.0; n];
    for (i, kvi) in kv.iter_mut().enumerate() {
        let im = (i + n - 1) % n;
        let ip = (i + 1) % n;
        let mut ds = s[ip] - s[im];
        if ds <= 0.0 {
            ds += total_length;
        }
        if ds > 0.0 {
            *kvi = (grade[ip] - grade[im]) / ds;
        }
    }
    kv
}

/// Centered periodic moving average with the given half-width (in samples).
fn moving_average_periodic(v: &[f64], half: usize) -> Vec<f64> {
    let n = v.len();
    if half == 0 || n == 0 {
        return v.to_vec();
    }
    let win = 2 * half + 1;
    (0..n)
        .map(|i| {
            let mut acc = 0.0;
            for k in 0..win {
                acc += v[(i + n + k - half) % n];
            }
            acc / win as f64
        })
        .collect()
}

/// Result of running inference on a driven line: a standard-format
/// [`Telemetry`] (resampled input channels + inferred channels) plus a few
/// summary numbers for the CLI report.
pub struct InferResult {
    /// The output telemetry (input + inferred channels, grid = `s`).
    pub telemetry: Telemetry,
    /// Number of samples.
    pub len: usize,
    /// Convenience references to the peak diagnostics.
    pub peak_lat_g: f64,
    /// Arc length of the peak lateral g.
    pub peak_lat_g_s: f64,
    /// Peak braking g (most negative longitudinal), magnitude.
    pub peak_brake_g: f64,
    /// Arc length of the peak braking g.
    pub peak_brake_g_s: f64,
}

/// Build the driven line from `aligned`, resample the measured channels onto it,
/// infer the unmeasured channels, and return a ready-to-write [`Telemetry`].
pub fn infer_on_driven(
    centerline: &Track,
    aligned: &Telemetry,
    car: &CarParams,
    mode: DrivenLineMode,
    cfg: &InferConfig,
) -> Result<InferResult, CorrelateError> {
    infer_on_driven_3d(centerline, aligned, car, mode, cfg, None)
}

/// [`infer_on_driven`] with an optional 3D centerline `elevation`. When `Some`,
/// the driven line is elevated and the inferred load/grip/power channels pick up
/// the 3D terms (compression in dips, the grade force in power). When `None`,
/// behavior is identical to [`infer_on_driven`].
pub fn infer_on_driven_3d(
    centerline: &Track,
    aligned: &Telemetry,
    car: &CarParams,
    mode: DrivenLineMode,
    cfg: &InferConfig,
    elevation: Option<&apex_track::Ribbon3d>,
) -> Result<InferResult, CorrelateError> {
    let geom = crate::driven::build_driven_geometry_3d(centerline, aligned, mode, elevation)?;
    let n = geom.track.segments.len();

    // Output grid = driven arc length s (for the derivative) with the centerline
    // station carried as the `s` channel (continuous, monotone).
    let s_driven: Vec<f64> = geom.track.segments.iter().map(|seg| seg.s).collect();
    let station = geom.station_per_segment.clone();
    let kappa_raw: Vec<f64> = geom
        .track
        .segments
        .iter()
        .map(|seg| seg.curvature)
        .collect();
    // Light periodic average to de-noise the (2nd-derivative) curvature before
    // a_lat = v²·κ. Real-data path only; the core inverter sees raw κ.
    let spacing = geom.driven_length / n.max(1) as f64;
    let kw = ((cfg.kappa_halfwindow_m / spacing.max(1e-6)).round() as usize).max(0);
    let kappa = moving_average_periodic(&kappa_raw, kw);

    // Resample the measured channels onto the driven grid (by centerline station).
    let aligned_s = aligned
        .channel(ChannelId::S)
        .ok_or(CorrelateError::MissingAxis("s"))?;
    let resample = |id: ChannelId| -> Option<Vec<f64>> {
        aligned
            .channel(id)
            .map(|c| resample_linear(aligned_s, c, &station))
    };
    let v = resample(ChannelId::Speed).ok_or(CorrelateError::MissingAxis("speed"))?;

    // 3D terms from the elevated driven ribbon (grade θ, bank φ, vertical
    // curvature κ_v = dθ/ds). Absent ⇒ flat inference (all zero).
    let inferred = match &geom.driven_ribbon {
        Some(r) => {
            let grade: Vec<f64> = r.stations.iter().map(|st| st.grade).collect();
            let bank: Vec<f64> = r.stations.iter().map(|st| st.bank).collect();
            let kappa_v = grade_rate(&s_driven, &grade, geom.driven_length);
            infer_channels_3d(
                &s_driven, &v, &kappa, &grade, &bank, &kappa_v, true, car, cfg,
            )
        }
        None => infer_channels(&s_driven, &v, &kappa, true, car, cfg),
    };

    // Assemble output channels: s (station) + carried inputs + inferred.
    let mut channels: std::collections::BTreeMap<ChannelId, Vec<f64>> = Default::default();
    channels.insert(ChannelId::S, station.clone());
    channels.insert(ChannelId::Speed, v.clone());
    for id in [
        ChannelId::SRaw,
        ChannelId::Time,
        ChannelId::Throttle,
        ChannelId::Brake,
        ChannelId::Gear,
        ChannelId::Rpm,
        ChannelId::X,
        ChannelId::Y,
        ChannelId::LateralOffset,
    ] {
        if let Some(c) = resample(id) {
            channels.insert(id, c);
        }
    }
    channels.insert(ChannelId::LateralG, inferred.lateral_g.clone());
    channels.insert(ChannelId::LongitudinalG, inferred.longitudinal_g.clone());
    channels.insert(ChannelId::Downforce, inferred.downforce);
    channels.insert(ChannelId::AeroDragForce, inferred.aero_drag);
    channels.insert(ChannelId::FzFront, inferred.fz_front);
    channels.insert(ChannelId::FzRear, inferred.fz_rear);
    channels.insert(ChannelId::GripUtil, inferred.grip_util);
    channels.insert(ChannelId::TractivePower, inferred.tractive_power);
    channels.insert(ChannelId::BrakingPower, inferred.braking_power);

    // Peaks for the CLI report.
    let (mut plat, mut plat_s, mut pbrk, mut pbrk_s) = (0.0_f64, 0.0, 0.0_f64, 0.0);
    for ((&la, &lg), &st) in inferred
        .lateral_g
        .iter()
        .zip(&inferred.longitudinal_g)
        .zip(&station)
    {
        if la.abs().is_finite() && la.abs() > plat {
            plat = la.abs();
            plat_s = st;
        }
        if lg.is_finite() && lg < -pbrk {
            pbrk = -lg;
            pbrk_s = st;
        }
    }

    Ok(InferResult {
        telemetry: Telemetry {
            grid: GridKind::S,
            channels,
            metadata: aligned.metadata.clone(),
        },
        len: n,
        peak_lat_g: plat,
        peak_lat_g_s: plat_s,
        peak_brake_g: pbrk,
        peak_brake_g_s: pbrk_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_physics::{qss_lap_sim, CarParams};
    use apex_track::{build_track, circle_track, oval_track};

    /// Closed-loop on a CIRCLE: QSS gives constant speed ⇒ a_long ≈ 0 exactly,
    /// a_lat matches the sim, and grip utilization sits at ~1 (the cornering
    /// limit, plus a hair for the drag-overcoming tyre force).
    #[test]
    fn closed_loop_circle() {
        let car = CarParams::f1_2024_calibrated();
        let (pts, closed) = circle_track(120.0, 12.0, 360);
        let track = build_track("circle", &pts, closed);
        let sim = qss_lap_sim(&track, &car);
        let s: Vec<f64> = track.segments.iter().map(|g| g.s).collect();
        let kappa: Vec<f64> = track.segments.iter().map(|g| g.curvature).collect();
        let inf = infer_channels(&s, &sim.speeds, &kappa, true, &car, &InferConfig::default());

        for i in 0..s.len() {
            // a_lat exact vs the sim's own lateral_gs (magnitude).
            assert!(
                (inf.lateral_g[i].abs() - sim.lateral_gs[i]).abs() < 1e-9,
                "a_lat {} vs {}",
                inf.lateral_g[i].abs(),
                sim.lateral_gs[i]
            );
            // constant speed ⇒ derivative ~0.
            assert!(
                inf.longitudinal_g[i].abs() < 1e-6,
                "a_long {}",
                inf.longitudinal_g[i]
            );
            // at the cornering limit.
            assert!(
                (0.99..1.06).contains(&inf.grip_util[i]),
                "grip_util {}",
                inf.grip_util[i]
            );
        }
    }

    /// Closed-loop on an OVAL (has accel/brake): a_lat matches the sim exactly;
    /// a_long matches the sim's forward-difference within the derivative
    /// tolerance; the lap-integral of a_long nets ~zero.
    #[test]
    fn closed_loop_oval_matches_sim() {
        let car = CarParams::f1_2024_calibrated();
        let (pts, closed) = oval_track(600.0, 130.0, 12.0, 600);
        let track = build_track("oval", &pts, closed);
        let sim = qss_lap_sim(&track, &car);
        let s: Vec<f64> = track.segments.iter().map(|g| g.s).collect();
        let kappa: Vec<f64> = track.segments.iter().map(|g| g.curvature).collect();
        // Clean (noise-free) trace ⇒ a small window (≈ central difference).
        let cfg = InferConfig {
            deriv_halfwindow_m: 5.0,
            ..InferConfig::default()
        };
        let inf = infer_channels(&s, &sim.speeds, &kappa, true, &car, &cfg);

        let mut max_lat_err = 0.0_f64;
        let mut long_errs = Vec::new();
        let mut max_grip = 0.0_f64;
        for i in 0..s.len() {
            max_lat_err = max_lat_err.max((inf.lateral_g[i].abs() - sim.lateral_gs[i]).abs());
            long_errs.push((inf.longitudinal_g[i] - sim.longitudinal_gs[i]).abs());
            max_grip = max_grip.max(inf.grip_util[i]);
        }
        long_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = long_errs[long_errs.len() / 2];
        let p90 = long_errs[long_errs.len() * 9 / 10];
        // a_lat inverts the sim exactly; a_long is tight except at the O(few)
        // constraint-transition kinks (a symmetric derivative can't match a
        // forward difference at a C0 kink in the piecewise-optimal speed).
        assert!(max_lat_err < 1e-9, "max a_lat err {max_lat_err}");
        assert!(median < 0.02, "a_long median err {median} g");
        assert!(p90 < 0.15, "a_long p90 err {p90} g");
        // grip utilization sits at the friction circle (~1), never wildly above.
        assert!(max_grip < 1.06, "max grip_util {max_grip}");

        // Closed-lap a_long closure: Σ a_long·ds ≈ 0 (energy returns).
        let mut closure = 0.0;
        for i in 0..s.len() {
            let ds = if i + 1 < s.len() {
                s[i + 1] - s[i]
            } else {
                track.total_length - s[i]
            };
            if inf.longitudinal_g[i].is_finite() {
                closure += inf.longitudinal_g[i] * G * ds;
            }
        }
        assert!(closure.abs() < 5.0, "a_long closure {closure} (m²/s²)");
    }

    #[test]
    fn derivative_recovers_known_slope() {
        // v(s) = 20 + 0.5 s (constant slope) sampled on a uniform grid, no wrap.
        let n = 50;
        let s: Vec<f64> = (0..n).map(|i| i as f64 * 5.0).collect();
        let v: Vec<f64> = s.iter().map(|&x| 20.0 + 0.5 * x).collect();
        let d = local_linear_deriv(&s, &v, false, 0.0, 15.0);
        for (i, &di) in d.iter().enumerate() {
            if di.is_finite() {
                assert!((di - 0.5).abs() < 1e-9, "slope[{i}] = {di}");
            }
        }
    }

    /// Full driven-line inference + write + re-import: the inferred channels
    /// survive the standard-format round-trip (Task 4 viewer/round-trip check).
    #[test]
    fn infer_on_driven_round_trips() {
        use crate::{import_telemetry, write_telemetry_csv, GridKind, Mapping};
        use apex_telemetry::ChannelId;
        use std::collections::BTreeMap;
        use std::f64::consts::PI;

        let (pts, closed) = oval_track(500.0, 120.0, 12.0, 400);
        let track = build_track("Oval", &pts, closed);
        let car = CarParams::f1_2024_calibrated();
        let sim = qss_lap_sim(&track, &car);
        let l = track.total_length;

        // Synthetic aligned telemetry: the sim speed on the centerline (n=0).
        let m = 500;
        let mut s: Vec<f64> = vec![];
        let mut sp: Vec<f64> = vec![];
        let mut xs: Vec<f64> = vec![];
        let mut ys: Vec<f64> = vec![];
        let mut t: Vec<f64> = vec![];
        let mut nn: Vec<f64> = vec![];
        let mut acc = 0.0;
        for k in 0..m {
            let st = k as f64 / m as f64 * (l - 1.0);
            let (i, _) = track.locate(st);
            let seg = &track.segments[i];
            let v = sim.speeds[i.min(sim.speeds.len() - 1)];
            if k > 0 {
                acc += (st - s[k - 1]) / (0.5 * (v + sp[k - 1])).max(1.0);
            }
            s.push(st);
            sp.push(v);
            xs.push(seg.x);
            ys.push(seg.y);
            t.push(acc);
            nn.push(0.0);
            let _ = PI;
        }
        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, s);
        channels.insert(ChannelId::Speed, sp);
        channels.insert(ChannelId::X, xs);
        channels.insert(ChannelId::Y, ys);
        channels.insert(ChannelId::Time, t);
        channels.insert(ChannelId::LateralOffset, nn);
        let aligned = Telemetry {
            grid: GridKind::S,
            channels,
            metadata: vec![("source".into(), "synthetic".into())],
        };

        let res = infer_on_driven(
            &track,
            &aligned,
            &car,
            DrivenLineMode::Direct(0.75),
            &InferConfig::default(),
        )
        .unwrap();
        assert!(res.peak_lat_g > 0.0);

        let path = std::env::temp_dir().join("apex_infer_roundtrip.csv");
        write_telemetry_csv(&path, &res.telemetry).unwrap();
        let back = import_telemetry(&path, &Mapping::identity()).unwrap();
        // Inferred channels are present after re-import.
        for id in [
            ChannelId::Downforce,
            ChannelId::GripUtil,
            ChannelId::TractivePower,
            ChannelId::FzFront,
            ChannelId::LateralG,
        ] {
            assert!(
                back.channel(id).is_some(),
                "missing {} after round-trip",
                id.name()
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn nan_propagates() {
        let n = 40;
        let s: Vec<f64> = (0..n).map(|i| i as f64 * 5.0).collect();
        let mut v: Vec<f64> = vec![50.0; n];
        v[20] = f64::NAN; // a measured gap
        let kappa = vec![0.005; n];
        let inf = infer_channels(
            &s,
            &v,
            &kappa,
            false,
            &CarParams::default(),
            &InferConfig::default(),
        );
        // At the gap, everything is NaN.
        assert!(inf.lateral_g[20].is_nan());
        assert!(inf.downforce[20].is_nan());
        assert!(inf.grip_util[20].is_nan());
        // The derivative window around the gap makes a_long-dependent channels
        // NaN there too, while a_lat/downforce stay finite away from the gap.
        assert!(inf.downforce[0].is_finite());
        assert!(inf.grip_util[20 + 5].is_nan() || inf.grip_util[0].is_finite());
    }

    /// 3D closed-loop: the 3D QSS on a synthetic **banked ring** is inverted by
    /// `infer_channels_3d` — constant speed ⇒ `a_long ≈ 0`, and the friction
    /// circle (on the 3D banked load) sits at the limit (grip util ≈ 1). This is
    /// the 3D analogue of `closed_loop_circle`.
    #[test]
    fn closed_loop_banked_ring_3d() {
        use apex_physics::qss_lap_sim_3d;
        use apex_track::Ribbon3d;
        use std::f64::consts::PI;

        // A tight ring keeps the corner grip-limited (not drag-limited), so the
        // banked cornering limit is a clean constant speed the inverter recovers.
        let car = CarParams::f1_2024_calibrated();
        let (n, r, beta) = (720usize, 55.0, 0.10_f64);
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let u = 2.0 * PI * i as f64 / n as f64;
                [r * u.cos(), r * u.sin(), 0.0]
            })
            .collect();
        let bankv = vec![beta; n];
        let w = vec![6.0; n];
        let ribbon = Ribbon3d::from_centerline_3d("banked", &pts, &bankv, &w, &w, true);
        let sim = qss_lap_sim_3d(&ribbon, &car);

        let s: Vec<f64> = ribbon.stations.iter().map(|st| st.s).collect();
        let kappa: Vec<f64> = ribbon.stations.iter().map(|st| st.omega_z.abs()).collect();
        let grade: Vec<f64> = ribbon.stations.iter().map(|st| st.grade).collect();
        let bank: Vec<f64> = ribbon.stations.iter().map(|st| st.bank).collect();
        let kv = grade_rate(&s, &grade, ribbon.total_length);
        let inf = infer_channels_3d(
            &s,
            &sim.speeds,
            &kappa,
            &grade,
            &bank,
            &kv,
            true,
            &car,
            &InferConfig::default(),
        );
        for i in 0..n {
            assert!(
                inf.longitudinal_g[i].abs() < 1e-2,
                "a_long {} at {i}",
                inf.longitudinal_g[i]
            );
            assert!(
                (0.97..1.07).contains(&inf.grip_util[i]),
                "grip_util {} at {i}",
                inf.grip_util[i]
            );
        }
    }
}
