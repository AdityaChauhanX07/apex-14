//! Car-parameter identification (task 2.4).
//!
//! Fits a small set of [`CarParams`] fields so the QSS sim on the **driven
//! line** matches the measured speed trace, using the shared
//! [`apex_math::lm`] Levenberg-Marquardt solver. The driven-line geometry is
//! car-independent, so it is built **once** and every LM iteration just re-runs
//! the (fast) QSS on it with the candidate parameters.
//!
//! # Free parameters (dotted paths)
//!
//! | path | field | notes |
//! |------|-------|-------|
//! | `aero.lift_coeff` | `CarParams::lift_coeff` | downforce coefficient |
//! | `aero.drag_coeff` | `CarParams::drag_coeff` | drag coefficient |
//! | `tires.mu` | `CarParams::tire_mu` | peak grip |
//! | `powertrain.power_scale` | ×`CarParams::max_drive_force` | **synthetic knob** |
//!
//! `powertrain.power_scale` is a *multiplier* on the drive force (there is no
//! stored `power_scale` field): value 1.0 = preset, applied as
//! `max_drive_force = base.max_drive_force × power_scale`.
//!
//! # Residual & weighting
//!
//! The residual is `sim_speed − measured_speed` on the common uniform-`s` grid
//! over the driven line, **uniformly weighted**. A braking-zone (or corner)
//! weight would multiply each residual in [`SpeedResidual::residuals`] by a
//! per-grid-point weight derived from the local deceleration; that hook is
//! intentionally left un-built here.

use std::time::Instant;

use apex_math::lm::{levenberg_marquardt, LmConfig, LmResult, ResidualProvider};
use apex_physics::{qss_lap_sim, CarParams};
use apex_telemetry::ChannelId;
use apex_track::Track;

use crate::driven::{build_driven_geometry, DrivenLineMode};
use crate::error::CorrelateError;
use crate::metrics::{resample_linear, resample_periodic};
use crate::telemetry::Telemetry;

/// Which `CarParams` knob a free parameter drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// `CarParams::lift_coeff`.
    LiftCoeff,
    /// `CarParams::drag_coeff`.
    DragCoeff,
    /// `CarParams::tire_mu`.
    TireMu,
    /// Multiplier on `CarParams::max_drive_force` (synthetic).
    PowerScale,
}

/// A free parameter: dotted path, the knob it drives, its initial value, and
/// box bounds.
#[derive(Debug, Clone)]
pub struct FreeParam {
    /// Dotted path as given on the CLI.
    pub path: String,
    /// The car knob.
    pub kind: ParamKind,
    /// Initial value (from the base car, or 1.0 for `power_scale`).
    pub initial: f64,
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
}

/// Resolve a dotted path into a [`FreeParam`] with sane bounds relative to
/// `base`.
pub fn parse_free_param(path: &str, base: &CarParams) -> Result<FreeParam, CorrelateError> {
    let (kind, initial, lo, hi) = match path.trim() {
        "aero.lift_coeff" => (
            ParamKind::LiftCoeff,
            base.lift_coeff,
            0.3 * base.lift_coeff,
            3.0 * base.lift_coeff,
        ),
        "aero.drag_coeff" => (
            ParamKind::DragCoeff,
            base.drag_coeff,
            0.3 * base.drag_coeff,
            3.0 * base.drag_coeff,
        ),
        "tires.mu" => (ParamKind::TireMu, base.tire_mu, 1.0, 2.2),
        "powertrain.power_scale" => (ParamKind::PowerScale, 1.0, 0.7, 1.3),
        other => return Err(CorrelateError::UnknownFreeParam(other.to_string())),
    };
    Ok(FreeParam {
        path: path.trim().to_string(),
        kind,
        initial,
        lo,
        hi,
    })
}

/// Apply free-parameter `values` onto a copy of `base`.
pub fn apply_params(base: &CarParams, free: &[FreeParam], values: &[f64]) -> CarParams {
    let mut c = base.clone();
    for (f, &v) in free.iter().zip(values) {
        match f.kind {
            ParamKind::LiftCoeff => c.lift_coeff = v,
            ParamKind::DragCoeff => c.drag_coeff = v,
            ParamKind::TireMu => c.tire_mu = v,
            ParamKind::PowerScale => c.max_drive_force = base.max_drive_force * v,
        }
    }
    c
}

/// Residual provider: `sim_speed − measured_speed` on a fixed grid over a fixed
/// driven line, as a function of the free parameters.
struct SpeedResidual<'a> {
    track: &'a Track,
    base: &'a CarParams,
    free: &'a [FreeParam],
    l: f64,
    /// Driven-segment indices sorted by station (deduped) — fixed geometry.
    order: &'a [usize],
    /// Sorted station values matching `order`.
    station_sorted: &'a [f64],
    grid: &'a [f64],
    meas_v: &'a [f64],
}

impl ResidualProvider for SpeedResidual<'_> {
    fn residuals(&self, values: &[f64]) -> Vec<f64> {
        let car = apply_params(self.base, self.free, values);
        let sim = qss_lap_sim(self.track, &car);
        let speed_sorted: Vec<f64> = self.order.iter().map(|&i| sim.speeds[i]).collect();
        let sim_v = resample_periodic(self.station_sorted, &speed_sorted, self.l, self.grid);
        (0..self.grid.len())
            .map(|i| {
                if self.meas_v[i].is_finite() && sim_v[i].is_finite() {
                    // Uniform weight. A braking/corner weight would scale this.
                    sim_v[i] - self.meas_v[i]
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        self.free.iter().map(|f| (f.lo, f.hi)).collect()
    }
}

/// Outcome of an identification run.
pub struct IdentifyResult {
    /// The LM result (params, cost, diagnostics).
    pub lm: LmResult,
    /// The free-parameter specs (names, initials, bounds), in fit order.
    pub free: Vec<FreeParam>,
    /// The fitted car (base with the fitted fields applied).
    pub fitted_car: CarParams,
    /// Number of grid points (observations).
    pub grid_len: usize,
    /// Wall-clock seconds per accepted/rejected iteration.
    pub seconds_per_iter: f64,
    /// Total wall-clock seconds for the LM solve.
    pub total_seconds: f64,
}

/// Identify `free` parameters of `base` so the QSS on the driven line matches
/// the measured speed. `aligned` must carry the projected `s`, `speed`, and the
/// channels the chosen `mode` needs.
pub fn identify(
    centerline: &Track,
    aligned: &Telemetry,
    base: &CarParams,
    free: Vec<FreeParam>,
    mode: DrivenLineMode,
    grid_step: f64,
) -> Result<IdentifyResult, CorrelateError> {
    if free.is_empty() {
        return Err(CorrelateError::AlignFailed("no free parameters given"));
    }
    let geom = build_driven_geometry(centerline, aligned, mode)?;
    let l = geom.centerline_length;

    // Measured speed on a common uniform-s grid.
    let meas_s = aligned
        .channel(ChannelId::S)
        .ok_or(CorrelateError::MissingAxis("s"))?;
    let meas_speed = aligned
        .channel(ChannelId::Speed)
        .ok_or(CorrelateError::MissingAxis("speed"))?;
    let lo = (meas_s[0] / grid_step).ceil() * grid_step;
    let hi = (meas_s[meas_s.len() - 1] / grid_step).floor() * grid_step;
    let mut grid = Vec::new();
    let mut g = lo;
    while g <= hi + 1e-9 {
        grid.push(g);
        g += grid_step;
    }
    let meas_v = resample_linear(meas_s, meas_speed, &grid);

    // Fixed sort order of driven segments by station (mod L), deduped.
    let n = geom.station_per_segment.len();
    let mod_st: Vec<f64> = geom
        .station_per_segment
        .iter()
        .map(|s| s.rem_euclid(l))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| mod_st[a].partial_cmp(&mod_st[b]).unwrap());
    let mut kept_order = Vec::with_capacity(n);
    let mut station_sorted = Vec::with_capacity(n);
    let mut last = f64::NEG_INFINITY;
    for &idx in &order {
        if mod_st[idx] - last > 1e-6 {
            kept_order.push(idx);
            station_sorted.push(mod_st[idx]);
            last = mod_st[idx];
        }
    }

    let initial: Vec<f64> = free.iter().map(|f| f.initial).collect();
    let provider = SpeedResidual {
        track: &geom.track,
        base,
        free: &free,
        l,
        order: &kept_order,
        station_sorted: &station_sorted,
        grid: &grid,
        meas_v: &meas_v,
    };

    let t0 = Instant::now();
    let lm = levenberg_marquardt(&provider, &initial, &LmConfig::default());
    let total_seconds = t0.elapsed().as_secs_f64();
    let iters = lm.iterations.len().max(1);

    let fitted_car = apply_params(base, &free, &lm.params);

    Ok(IdentifyResult {
        lm,
        free,
        fitted_car,
        grid_len: grid.len(),
        seconds_per_iter: total_seconds / iters as f64,
        total_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridKind;
    use apex_track::{build_track, oval_track};
    use std::collections::BTreeMap;

    fn oval() -> Track {
        let (pts, closed) = oval_track(500.0, 120.0, 12.0, 400);
        build_track("Oval", &pts, closed)
    }

    #[test]
    fn dotted_paths_map_and_bound() {
        let base = CarParams::default();
        let lift = parse_free_param("aero.lift_coeff", &base).unwrap();
        assert_eq!(lift.kind, ParamKind::LiftCoeff);
        assert_eq!(lift.initial, base.lift_coeff);
        assert!((lift.lo - 0.3 * base.lift_coeff).abs() < 1e-9);
        let power = parse_free_param("powertrain.power_scale", &base).unwrap();
        assert_eq!(power.kind, ParamKind::PowerScale);
        assert_eq!(power.initial, 1.0);
        assert_eq!((power.lo, power.hi), (0.7, 1.3));
        assert!(parse_free_param("bogus.path", &base).is_err());
    }

    #[test]
    fn power_scale_multiplies_drive_force() {
        let base = CarParams::default();
        let free = vec![parse_free_param("powertrain.power_scale", &base).unwrap()];
        let c = apply_params(&base, &free, &[1.2]);
        assert!((c.max_drive_force - 1.2 * base.max_drive_force).abs() < 1e-6);
    }

    /// Aligned telemetry tracing the centerline (n=0) at constant speed, with
    /// the "measured" speed produced by a KNOWN car. Identifying that car from a
    /// perturbed start must recover it.
    fn aligned_from_car(track: &Track, car: &CarParams) -> Telemetry {
        // Run QSS with the truth car on the centerline to get a speed(s).
        let sim = qss_lap_sim(track, car);
        let l = track.total_length;
        let m = 600;
        let (mut s, mut sp, mut xs, mut ys, mut t) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // Sample uniformly along the lap; x/y on the centerline (n=0).
        let mut acc = 0.0;
        for k in 0..m {
            let st = k as f64 / m as f64 * (l - 1.0);
            let (i, _) = track.locate(st);
            let seg = &track.segments[i];
            let v = sim.speeds[i.min(sim.speeds.len() - 1)];
            s.push(st);
            sp.push(v);
            xs.push(seg.x);
            ys.push(seg.y);
            if k > 0 {
                let ds = st - s[k - 1];
                acc += ds / (0.5 * (v + sp[k - 1])).max(1.0);
            }
            t.push(acc);
        }
        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, s);
        channels.insert(ChannelId::Speed, sp);
        channels.insert(ChannelId::X, xs);
        channels.insert(ChannelId::Y, ys);
        channels.insert(ChannelId::Time, t);
        channels.insert(ChannelId::LateralOffset, vec![0.0; m]);
        Telemetry {
            grid: GridKind::S,
            channels,
            metadata: Vec::new(),
        }
    }

    #[test]
    fn recovers_known_car_parameters() {
        let track = oval();
        // Truth car: perturb lift and drag away from the default.
        let truth = CarParams {
            lift_coeff: 3.1,
            drag_coeff: 0.85,
            ..CarParams::default()
        };
        let aligned = aligned_from_car(&track, &truth);

        // Fit from the DEFAULT preset (perturbed initial guess), offset line
        // (n=0 so the driven line == centerline == the truth geometry).
        let base = CarParams::default();
        let free = vec![
            parse_free_param("aero.lift_coeff", &base).unwrap(),
            parse_free_param("aero.drag_coeff", &base).unwrap(),
        ];
        let res = identify(
            &track,
            &aligned,
            &base,
            free,
            DrivenLineMode::Offset(0.0),
            10.0,
        )
        .unwrap();

        // Recovered within a few std errors of truth, and cost dropped.
        assert!(res.lm.cost < res.lm.initial_cost, "cost did not drop");
        let lift = res.fitted_car.lift_coeff;
        let drag = res.fitted_car.drag_coeff;
        assert!((lift - 3.1).abs() < 0.1, "lift {lift} vs 3.1");
        assert!((drag - 0.85).abs() < 0.1, "drag {drag} vs 0.85");
        assert!(res.lm.bound_pinned.is_empty(), "unexpected bound pin");
    }
}
