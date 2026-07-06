//! Correlation metrics: pure functions comparing a simulated speed trace to a
//! measured one on a common arc-length grid.
//!
//! Every public function here is deterministic and side-effect free, and is
//! unit-tested on synthetic traces with known answers. Sign conventions:
//!
//! - Lap / sector / apex-speed **delta = sim − measured** (positive ⇒ sim is
//!   slower on time, or faster on apex speed — see each type's doc).
//! - Braking-point **offset = s_sim − s_measured** (positive ⇒ the sim brakes
//!   **later**, i.e. its braking onset sits at a larger arc length).

/// Linear resampling of `ys` (sampled at strictly increasing `xs`) onto `grid`.
/// Queries outside `[xs[0], xs[last]]` are clamped to the endpoints. `NaN`
/// samples propagate into any interval that touches them.
pub fn resample_linear(xs: &[f64], ys: &[f64], grid: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(grid.len());
    let n = xs.len();
    if n == 0 {
        return vec![f64::NAN; grid.len()];
    }
    let mut i = 0usize;
    for &g in grid {
        if g <= xs[0] {
            out.push(ys[0]);
            continue;
        }
        if g >= xs[n - 1] {
            out.push(ys[n - 1]);
            continue;
        }
        while i + 1 < n && xs[i + 1] < g {
            i += 1;
        }
        let (x0, x1) = (xs[i], xs[i + 1]);
        let (y0, y1) = (ys[i], ys[i + 1]);
        let t = if x1 > x0 { (g - x0) / (x1 - x0) } else { 0.0 };
        out.push(y0 + t * (y1 - y0));
    }
    out
}

/// Evaluate a **periodic** trace `speeds(distances)` (a closed-lap profile with
/// `distances` strictly increasing in `[0, total_length)`) at each `station`,
/// interpolating across the start/finish seam. Stations are taken mod
/// `total_length`.
pub fn resample_periodic(
    distances: &[f64],
    speeds: &[f64],
    total_length: f64,
    stations: &[f64],
) -> Vec<f64> {
    let n = distances.len();
    let mut out = Vec::with_capacity(stations.len());
    if n == 0 {
        return vec![f64::NAN; stations.len()];
    }
    for &q0 in stations {
        let q = q0.rem_euclid(total_length);
        // Find i with distances[i] <= q < distances[i+1]; else the seam.
        if q < distances[0] || q >= distances[n - 1] {
            // Seam interval: distances[n-1] -> distances[0] + total_length.
            let x0 = distances[n - 1];
            let x1 = distances[0] + total_length;
            let qq = if q < distances[0] {
                q + total_length
            } else {
                q
            };
            let t = if x1 > x0 { (qq - x0) / (x1 - x0) } else { 0.0 };
            out.push(speeds[n - 1] + t * (speeds[0] - speeds[n - 1]));
            continue;
        }
        // Binary search.
        let (mut lo, mut hi) = (0usize, n - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if distances[mid] <= q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (q - distances[lo]) / (distances[hi] - distances[lo]);
        out.push(speeds[lo] + t * (speeds[hi] - speeds[lo]));
    }
    out
}

/// Lap-time comparison. `delta = sim − measured` (positive ⇒ sim is slower).
#[derive(Debug, Clone, Copy)]
pub struct LapDelta {
    /// Simulated lap time (s).
    pub sim: f64,
    /// Measured lap time (s).
    pub measured: f64,
    /// `sim − measured` (s).
    pub delta: f64,
}

impl LapDelta {
    /// Build from the two lap times.
    pub fn new(sim: f64, measured: f64) -> Self {
        LapDelta {
            sim,
            measured,
            delta: sim - measured,
        }
    }
}

/// Bucket measured per-interval times into equal-arc-length sectors, keyed by
/// each interval midpoint's **absolute centerline station** (mod
/// `total_length`) — the same equal-thirds definition
/// [`apex_physics::sector_times`] uses for the sim, so the two are comparable
/// even though the measured lap starts mid-track and wraps the start/finish.
///
/// `stations[i]` is the (possibly unwrapped) station of sample `i`;
/// `interval_times[i]` is the time on the interval from `i` to `i+1`, so
/// `interval_times.len() == stations.len() - 1`. The result sums to
/// `interval_times.iter().sum()`.
pub fn measured_sector_times(
    stations: &[f64],
    interval_times: &[f64],
    total_length: f64,
    n_sectors: usize,
) -> Vec<f64> {
    let sector_len = total_length / n_sectors as f64;
    let mut sectors = vec![0.0; n_sectors];
    for (i, &dt) in interval_times.iter().enumerate() {
        if i + 1 >= stations.len() {
            break;
        }
        let s_mid = 0.5 * (stations[i] + stations[i + 1]);
        let station = s_mid.rem_euclid(total_length);
        let idx = ((station / sector_len).floor() as usize).min(n_sectors - 1);
        sectors[idx] += dt;
    }
    sectors
}

/// Per-sector comparison. `delta[k] = sim[k] − measured[k]`.
#[derive(Debug, Clone)]
pub struct SectorComparison {
    /// Sim sector times (s).
    pub sim: Vec<f64>,
    /// Measured sector times (s).
    pub measured: Vec<f64>,
    /// `sim − measured` per sector (s).
    pub delta: Vec<f64>,
}

impl SectorComparison {
    /// Build from sim and measured sector times (must be equal length).
    pub fn new(sim: Vec<f64>, measured: Vec<f64>) -> Self {
        let delta = sim.iter().zip(&measured).map(|(a, b)| a - b).collect();
        SectorComparison {
            sim,
            measured,
            delta,
        }
    }
}

/// Speed-trace error on a common grid.
#[derive(Debug, Clone, Copy)]
pub struct SpeedRmse {
    /// Root-mean-square of `sim − measured` over valid grid points (m/s).
    pub rmse: f64,
    /// Maximum `|sim − measured|` (m/s).
    pub max_abs: f64,
    /// Arc length `s` (m) where `max_abs` occurs.
    pub s_at_max: f64,
    /// Number of valid (non-`NaN`) grid points used.
    pub n: usize,
}

/// Speed RMSE / max error between two traces on the same `grid_s`. Grid points
/// where either trace is non-finite are skipped.
pub fn speed_rmse(grid_s: &[f64], sim_v: &[f64], meas_v: &[f64]) -> SpeedRmse {
    let mut sumsq = 0.0;
    let mut n = 0usize;
    let mut max_abs = 0.0;
    let mut s_at_max = if grid_s.is_empty() { 0.0 } else { grid_s[0] };
    for i in 0..grid_s.len().min(sim_v.len()).min(meas_v.len()) {
        let (a, b) = (sim_v[i], meas_v[i]);
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let d = a - b;
        sumsq += d * d;
        n += 1;
        if d.abs() > max_abs {
            max_abs = d.abs();
            s_at_max = grid_s[i];
        }
    }
    let rmse = if n > 0 {
        (sumsq / n as f64).sqrt()
    } else {
        f64::NAN
    };
    SpeedRmse {
        rmse,
        max_abs,
        s_at_max,
        n,
    }
}

/// Tuning for corner (apex) detection.
#[derive(Debug, Clone, Copy)]
pub struct CornerConfig {
    /// Only minima below this speed (m/s) count as corners.
    pub ceiling: f64,
    /// Required speed rise (m/s) on the shallower side of the dip.
    pub min_prominence: f64,
    /// Minimum arc-length spacing (m) between accepted apexes.
    pub min_spacing_m: f64,
}

impl Default for CornerConfig {
    fn default() -> Self {
        CornerConfig {
            ceiling: 70.0,
            min_prominence: 3.0,
            min_spacing_m: 80.0,
        }
    }
}

/// Detect corner apexes as prominent local minima of `v` (below the ceiling) on
/// a **uniform** `grid_s`. Returns apex grid indices in increasing `s`.
///
/// A candidate `k` must be the minimum within `±window` (window =
/// `min_spacing_m`) and have a speed rise of at least `min_prominence` on the
/// shallower side; a final spacing pass drops the higher of any two apexes
/// closer than `min_spacing_m`. This suppresses noise doubles.
pub fn detect_corners(grid_s: &[f64], v: &[f64], cfg: CornerConfig) -> Vec<usize> {
    let n = v.len().min(grid_s.len());
    if n < 3 {
        return Vec::new();
    }
    let step = grid_s[1] - grid_s[0];
    let w = ((cfg.min_spacing_m / step).round() as usize).max(1);

    let mut candidates: Vec<usize> = Vec::new();
    for k in 1..n - 1 {
        let vk = v[k];
        if !vk.is_finite() || vk >= cfg.ceiling {
            continue;
        }
        let lo = k.saturating_sub(w);
        let hi = (k + w).min(n - 1);
        // k must achieve the window minimum, and be the first index to do so
        // (dedupes flat ties into a single apex).
        let mut is_min = true;
        let mut first_min_idx = k;
        let mut best = f64::INFINITY;
        for (j, &vj) in v.iter().enumerate().take(hi + 1).skip(lo) {
            if vj.is_finite() && vj < best {
                best = vj;
                first_min_idx = j;
            }
        }
        if (vk - best).abs() > 1e-12 || first_min_idx != k {
            is_min = false;
        }
        if !is_min {
            continue;
        }
        // Prominence: rise on the shallower side within the window.
        let left_max = v[lo..k]
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let right_max = v[k + 1..=hi]
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let prom = left_max.min(right_max) - vk;
        if prom.is_finite() && prom >= cfg.min_prominence {
            candidates.push(k);
        }
    }

    // Spacing pass: keep the lower apex when two are closer than min_spacing.
    let mut kept: Vec<usize> = Vec::new();
    for &c in &candidates {
        if let Some(&last) = kept.last() {
            if grid_s[c] - grid_s[last] < cfg.min_spacing_m {
                if v[c] < v[last] {
                    *kept.last_mut().unwrap() = c;
                }
                continue;
            }
        }
        kept.push(c);
    }
    kept
}

/// Apex-speed error at one corner. `delta = sim − measured` (positive ⇒ the sim
/// carries **more** speed through the apex than the driver did).
#[derive(Debug, Clone, Copy)]
pub struct ApexError {
    /// Arc length of the apex (m).
    pub s: f64,
    /// Measured minimum (apex) speed (m/s).
    pub v_measured: f64,
    /// Sim speed at the same `s` (m/s).
    pub v_sim: f64,
    /// `sim − measured` (m/s).
    pub delta: f64,
}

/// Apex-speed errors for each detected corner index into the common grid.
pub fn apex_errors(
    grid_s: &[f64],
    sim_v: &[f64],
    meas_v: &[f64],
    corners: &[usize],
) -> Vec<ApexError> {
    corners
        .iter()
        .map(|&k| {
            let v_measured = meas_v[k];
            let v_sim = sim_v[k];
            ApexError {
                s: grid_s[k],
                v_measured,
                v_sim,
                delta: v_sim - v_measured,
            }
        })
        .collect()
}

/// Longitudinal acceleration `a = v · dv/ds` at grid index `k` (central
/// difference on a uniform grid). Negative ⇒ decelerating (braking).
fn long_accel(v: &[f64], k: usize, step: f64) -> f64 {
    let n = v.len();
    if n < 3 || k == 0 || k + 1 >= n {
        return 0.0;
    }
    let dv_ds = (v[k + 1] - v[k - 1]) / (2.0 * step);
    v[k] * dv_ds
}

/// Braking-onset arc length approaching the apex at `apex_idx`: the start of the
/// contiguous run of `a ≤ −threshold` immediately preceding the apex. `None` if
/// no braking is found before the apex.
fn braking_onset(
    grid_s: &[f64],
    v: &[f64],
    apex_idx: usize,
    step: f64,
    threshold: f64,
) -> Option<f64> {
    // Walk back from the apex to the nearest braking sample.
    let mut e = apex_idx;
    while e > 0 && long_accel(v, e, step) > -threshold {
        e -= 1;
    }
    if long_accel(v, e, step) > -threshold {
        return None; // never entered braking
    }
    // Extend backward to the onset of the contiguous braking run.
    let mut b = e;
    while b > 0 && long_accel(v, b - 1, step) <= -threshold {
        b -= 1;
    }
    Some(grid_s[b])
}

/// Braking-point offset at one corner. `offset = s_sim − s_measured` (positive
/// ⇒ the sim's braking onset is at a larger `s`, i.e. it **brakes later**).
#[derive(Debug, Clone, Copy)]
pub struct BrakingOffset {
    /// Apex arc length (m).
    pub corner_s: f64,
    /// Sim braking-onset `s` (m), if detected.
    pub s_sim: Option<f64>,
    /// Measured braking-onset `s` (m), if detected.
    pub s_measured: Option<f64>,
    /// `s_sim − s_measured` (m), if both detected.
    pub offset: Option<f64>,
}

/// Braking-point offsets for each detected corner. `threshold` is the
/// deceleration magnitude (m/s²) that defines braking onset.
pub fn braking_offsets(
    grid_s: &[f64],
    sim_v: &[f64],
    meas_v: &[f64],
    corners: &[usize],
    threshold: f64,
) -> Vec<BrakingOffset> {
    if grid_s.len() < 2 {
        return Vec::new();
    }
    let step = grid_s[1] - grid_s[0];
    corners
        .iter()
        .map(|&k| {
            let s_sim = braking_onset(grid_s, sim_v, k, step, threshold);
            let s_measured = braking_onset(grid_s, meas_v, k, step, threshold);
            let offset = match (s_sim, s_measured) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            };
            BrakingOffset {
                corner_s: grid_s[k],
                s_sim,
                s_measured,
                offset,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(step: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * step).collect()
    }

    #[test]
    fn resample_linear_midpoints() {
        let xs = vec![0.0, 10.0, 20.0];
        let ys = vec![0.0, 100.0, 200.0];
        let g = vec![0.0, 5.0, 15.0, 25.0];
        let out = resample_linear(&xs, &ys, &g);
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 50.0).abs() < 1e-9);
        assert!((out[2] - 150.0).abs() < 1e-9);
        assert!((out[3] - 200.0).abs() < 1e-9); // clamped
    }

    #[test]
    fn resample_periodic_seam() {
        // distances 0,90 on a 100 m loop; speed 10 -> 20; station 95 is in the
        // seam interval 90 -> (0+100): halfway back to 10.
        let d = vec![0.0, 90.0];
        let sp = vec![10.0, 20.0];
        let out = resample_periodic(&d, &sp, 100.0, &[95.0]);
        // seam from x0=90 (v=20) to x1=100 (v=10); at 95 -> 15.
        assert!((out[0] - 15.0).abs() < 1e-9, "got {}", out[0]);
        // Station wrap: 105 mod 100 = 5, between 0 and 90.
        let out2 = resample_periodic(&d, &sp, 100.0, &[105.0]);
        let expect = 10.0 + (5.0 / 90.0) * 10.0;
        assert!((out2[0] - expect).abs() < 1e-9);
    }

    #[test]
    fn rmse_known() {
        let g = grid(1.0, 5);
        let sim = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let meas = vec![1.0, 2.0, 0.0, 4.0, 5.0]; // one diff of 3 at s=2
        let r = speed_rmse(&g, &sim, &meas);
        assert_eq!(r.n, 5);
        assert!((r.rmse - (9.0f64 / 5.0).sqrt()).abs() < 1e-9);
        assert!((r.max_abs - 3.0).abs() < 1e-9);
        assert!((r.s_at_max - 2.0).abs() < 1e-9);
    }

    #[test]
    fn rmse_skips_nan() {
        let g = grid(1.0, 3);
        let sim = vec![1.0, f64::NAN, 3.0];
        let meas = vec![1.0, 2.0, 3.0];
        let r = speed_rmse(&g, &sim, &meas);
        assert_eq!(r.n, 2);
        assert_eq!(r.rmse, 0.0);
    }

    #[test]
    fn measured_sectors_sum_to_total_and_wrap() {
        // Two "laps worth" of stations that pass the seam: total_length 300,
        // 3 sectors of 100. Stations from 250 -> 550 (unwrapped), dt=1 each 10 m.
        let total = 300.0;
        let mut stations = Vec::new();
        let mut s = 250.0;
        while s <= 550.0 + 1e-9 {
            stations.push(s);
            s += 10.0;
        }
        let dt = vec![1.0; stations.len() - 1];
        let sectors = measured_sector_times(&stations, &dt, total, 3);
        let sum: f64 = sectors.iter().sum();
        let expect: f64 = dt.iter().sum();
        assert!((sum - expect).abs() < 1e-9);
        // Every sector gets some time (the lap covers the whole loop once).
        for (k, sec) in sectors.iter().enumerate() {
            assert!(*sec > 0.0, "sector {k} empty");
        }
    }

    #[test]
    fn detects_planted_corners_no_doubles() {
        // A trace with three clear dips at s = 200, 500, 800, on a noisy base.
        let step = 10.0;
        let n = 120;
        let g = grid(step, n);
        let mut v = vec![80.0; n]; // above the default ceiling on the straights
        let dips = [20usize, 50, 80]; // indices -> s = 200, 500, 800
        for (i, vi) in v.iter_mut().enumerate() {
            // base wave + small noise
            *vi = 75.0 + 3.0 * ((i as f64) * 0.3).sin();
            for &d in &dips {
                let dist = (i as i64 - d as i64).abs() as f64;
                *vi -= 45.0 * (-(dist * dist) / 8.0).exp(); // sharp dip to ~30
            }
        }
        let corners = detect_corners(&g, &v, CornerConfig::default());
        assert_eq!(corners.len(), 3, "corners {:?}", corners);
        for (got, &want) in corners.iter().zip(&dips) {
            assert!(
                (*got as i64 - want as i64).abs() <= 1,
                "apex {got} vs {want}"
            );
        }
    }

    #[test]
    fn noise_does_not_double_count() {
        // A single broad dip with jitter riding on it must yield ONE corner.
        let step = 10.0;
        let n = 80;
        let g = grid(step, n);
        let mut v = vec![0.0; n];
        for (i, vi) in v.iter_mut().enumerate() {
            let dist = (i as i64 - 40).abs() as f64;
            let base = 65.0 - 40.0 * (-(dist * dist) / 200.0).exp();
            // small high-frequency jitter that creates tiny local minima
            *vi = base + 0.8 * ((i as f64) * 1.7).sin();
        }
        let corners = detect_corners(&g, &v, CornerConfig::default());
        assert_eq!(corners.len(), 1, "corners {:?}", corners);
    }

    #[test]
    fn apex_error_sign() {
        let g = grid(10.0, 5);
        let sim = vec![50.0, 40.0, 35.0, 40.0, 50.0];
        let meas = vec![48.0, 38.0, 30.0, 38.0, 48.0];
        let errs = apex_errors(&g, &sim, &meas, &[2]);
        assert_eq!(errs.len(), 1);
        assert!((errs[0].v_measured - 30.0).abs() < 1e-9);
        assert!((errs[0].v_sim - 35.0).abs() < 1e-9);
        assert!((errs[0].delta - 5.0).abs() < 1e-9); // sim faster at apex
    }

    #[test]
    fn braking_offset_sign() {
        // Build two decel ramps into an apex at index 30. The sim keeps full
        // speed longer (brakes LATER) -> positive offset.
        let step = 10.0;
        let n = 40;
        let g = grid(step, n);
        let apex = 30usize;
        // measured: brakes from index 10; sim: brakes from index 20.
        let mut meas = vec![0.0; n];
        let mut sim = vec![0.0; n];
        for i in 0..n {
            meas[i] = if i <= 10 {
                90.0
            } else if i <= apex {
                90.0 - 3.0 * (i - 10) as f64
            } else {
                90.0 - 3.0 * (apex - 10) as f64 + 3.0 * (i - apex) as f64
            };
            sim[i] = if i <= 20 {
                90.0
            } else if i <= apex {
                90.0 - 6.0 * (i - 20) as f64
            } else {
                90.0 - 6.0 * (apex - 20) as f64 + 6.0 * (i - apex) as f64
            };
        }
        let offs = braking_offsets(&g, &sim, &meas, &[apex], 2.0);
        assert_eq!(offs.len(), 1);
        let o = offs[0];
        let s_sim = o.s_sim.expect("sim braking");
        let s_meas = o.s_measured.expect("measured braking");
        assert!(
            s_sim > s_meas,
            "sim should brake later: {s_sim} vs {s_meas}"
        );
        assert!(o.offset.unwrap() > 0.0, "offset should be positive");
    }
}
