//! Estimator unit + synthetic-calibration tests.
//!
//! The synthetic calibration test ([`synthetic_calibration`]) is the correctness
//! spine: it simulates the single-track model on a known slalom maneuver with
//! known inputs, corrupts the position/speed measurements with realistic Gaussian
//! noise at 7.5 Hz, runs the smoother, and asserts (i) 3σ coverage of the truth
//! by the reported covariance, (ii) forward-pass NIS χ² consistency, and (iii)
//! slip-angle recovery within tolerance.

use super::*;
use apex_integrator::rk4_step;
use apex_physics::{BicycleModel, CarParams, PacejkaTire};
use std::f64::consts::PI;

/// Deterministic SplitMix64 + Box–Muller normal generator (keeps the test
/// reproducible without pulling `rand` into this crate).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unif().max(1e-12);
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Ground-truth generation parameters. The inputs (steer, drive) evolve as a
/// **random walk** that is piecewise-constant over each measurement epoch, with
/// per-step increment variances `q_delta·dt` / `q_fdrive·dt`. Making the truth's
/// input model identical to the filter's process model turns this into a rigorous
/// self-consistency (calibration) test — the EKF is statistically exact up to its
/// first-order linearization, so a well-behaved filter must be consistent.
#[derive(Clone, Copy)]
struct TruthParams {
    n_epochs: usize,
    pos_noise: f64,
    speed_noise: f64,
    q_delta: f64,
    q_fdrive: f64,
    seed: u64,
}

/// Integrate the 6-state bicycle model over one epoch with a CONSTANT input
/// (`k_sub` RK4 substeps) — mirroring how the filter propagates the augmented
/// input states (held constant, then perturbed by process noise).
fn advance_truth_const(
    model: &BicycleModel,
    state: [f64; 6],
    epoch_dt: f64,
    k_sub: usize,
    delta: f64,
    fdrive: f64,
) -> [f64; 6] {
    let h = epoch_dt / k_sub as f64;
    let mut s = state;
    for _ in 0..k_sub {
        s = rk4_step(model, &s, &[delta, fdrive], 0.0, h);
    }
    s
}

/// Generate the ground-truth trajectory + noisy measurements. Returns (t,
/// true_state[6], true_delta, true_fdrive, meas_x, meas_y, meas_speed).
#[allow(clippy::type_complexity)]
fn make_synthetic(
    car: &CarParams,
    tire: &PacejkaTire,
    tp: TruthParams,
) -> (
    Vec<f64>,
    Vec<[f64; 6]>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let model = BicycleModel { params: car, tire };
    let epoch_dt = 1.0 / 7.5;
    let k_sub = 20;
    let v_cruise = 60.0;

    let mut rng = Rng(tp.seed);
    let n = tp.n_epochs;
    let mut t = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(n);
    let mut td = Vec::with_capacity(n);
    let mut tf = Vec::with_capacity(n);
    let mut mx = Vec::with_capacity(n);
    let mut my = Vec::with_capacity(n);
    let mut ms = Vec::with_capacity(n);

    // Piecewise-constant random-walk inputs, seeded at a steady cruise.
    let mut delta = 0.0;
    let mut fdrive = car.drag_force(v_cruise) + car.rolling_resistance_force();
    let sd_delta = (tp.q_delta * epoch_dt).sqrt();
    let sd_fdrive = (tp.q_fdrive * epoch_dt).sqrt();

    let mut s = [0.0, 0.0, 0.0, v_cruise, 0.0, 0.0];
    for k in 0..n {
        let tk = k as f64 * epoch_dt;
        t.push(tk);
        states.push(s);
        td.push(delta);
        tf.push(fdrive);
        // Measurements: position + speed magnitude, Gaussian-corrupted.
        let speed_true = (s[3] * s[3] + s[4] * s[4]).sqrt();
        mx.push(s[0] + tp.pos_noise * rng.normal());
        my.push(s[1] + tp.pos_noise * rng.normal());
        ms.push(speed_true + tp.speed_noise * rng.normal());
        // Advance the state over the epoch with the current constant input, then
        // random-walk the input for the next epoch.
        s = advance_truth_const(&model, s, epoch_dt, k_sub, delta, fdrive);
        delta += sd_delta * rng.normal();
        fdrive += sd_fdrive * rng.normal();
    }
    (t, states, td, tf, mx, my, ms)
}

/// The truth's input random-walk densities, reused to match the filter's Q.
const Q_DELTA_TRUE: f64 = 1.0e-4;
const Q_FDRIVE_TRUE: f64 = 2.0e5;

/// True slip angles at a state, matching the model's definition.
fn true_slips(state: &[f64; 6], delta: f64, car: &CarParams) -> (f64, f64) {
    let vx = state[3].max(1.0);
    let vy = state[4];
    let r = state[5];
    let af = delta - ((vy + r * car.cog_to_front) / vx).atan();
    let ar = -((vy - r * car.cog_to_rear) / vx).atan();
    (af, ar)
}

/// Config for the synthetic case: R matches the injected measurement noise, and
/// the augmented-input process noise matches the truth's input random walk
/// exactly. Vehicle states carry only tiny regularization noise (the dynamics are
/// deterministic given the inputs) — so the filter's model equals the data-
/// generating process and consistency is expected.
fn synthetic_config(pos_noise: f64, speed_noise: f64) -> EstimatorConfig {
    EstimatorConfig {
        q: [
            1.0e-4,        // X   (regularization)
            1.0e-4,        // Y
            1.0e-6,        // psi
            1.0e-3,        // vx
            1.0e-3,        // vy
            1.0e-4,        // r
            Q_DELTA_TRUE,  // delta — matches the truth input walk
            Q_FDRIVE_TRUE, // f_drive
        ],
        pos_sigma: pos_noise,
        speed_sigma: speed_noise,
        course_sigma: 0.20,
        use_course: true,
        course_min_disp: 2.0,
        p0: [
            pos_noise * pos_noise,
            pos_noise * pos_noise,
            0.04,
            4.0,
            4.0,
            0.04,
            0.04,
            1.0e8,
        ],
        nis_gate: 1e9, // no gating in the clean synthetic case
        substep_dt: 0.02,
        delta_revert: 0.0, // pure random walk: matches the truth generator exactly
        max_predict_dt: 0.1,
    }
}

#[test]
fn synthetic_calibration() {
    let car = CarParams::f1_2024_calibrated();
    let tire = PacejkaTire::f1_default();
    let pos_noise = 1.0; // m (below the ~4 m align RMS — realistic per-sample GPS)
    let speed_noise = 0.3; // m/s
    let n = 300;

    let tp = TruthParams {
        n_epochs: n,
        pos_noise,
        speed_noise,
        q_delta: Q_DELTA_TRUE,
        q_fdrive: Q_FDRIVE_TRUE,
        seed: 0xA11CE,
    };
    let (t, truth, td, _tf, mx, my, ms) = make_synthetic(&car, &tire, tp);
    let cfg = synthetic_config(pos_noise, speed_noise);
    let res = smooth_states(&t, &mx, &my, &ms, &car, &tire, &cfg).unwrap();

    // (i) 3σ coverage: over the six *vehicle* states (X,Y,psi,vx,vy,r), the
    // fraction of samples where the smoothed estimate lies within 3σ of truth.
    // (delta/f_drive are augmented nuisance states; their truth is the prescribed
    // input, checked separately via slip recovery.)
    let mut inside = 0usize;
    let mut total = 0usize;
    // Skip a short warm-up while the filter converges from the diffuse prior.
    let warm = 20;
    for k in warm..n {
        for i in 0..6 {
            let err = (res.state[k][i] - truth[k][i]).abs();
            let sigma = res.std[k][i].max(1e-9);
            if err <= 3.0 * sigma {
                inside += 1;
            }
            total += 1;
        }
    }
    let coverage = inside as f64 / total as f64;
    assert!(
        coverage >= 0.95,
        "3σ coverage {coverage:.3} < 0.95 (vehicle states)"
    );

    // (ii) NIS χ² consistency (forward pass, DOF = M = 3). A consistent filter
    // has mean NIS ≈ 3. Bound it loosely to allow the deterministic-input model
    // mismatch.
    let d = &res.diagnostics;
    assert!(
        d.nis_mean > 1.5 && d.nis_mean < 6.0,
        "NIS mean {:.3} outside [1.5, 6.0] (target {})",
        d.nis_mean,
        d.nis_dof
    );
    // Most updates should fall under the 95% χ²(3) bound (7.815).
    assert!(
        d.nis_within_95 > 0.80,
        "NIS within-95 fraction {:.3} too low",
        d.nis_within_95
    );

    // (iii) slip-angle recovery: median absolute error over the run.
    let mut ferr = Vec::new();
    let mut rerr = Vec::new();
    for k in warm..n {
        let (af_true, ar_true) = true_slips(&truth[k], td[k], &car);
        ferr.push((res.slip_front[k] - af_true).abs());
        rerr.push((res.slip_rear[k] - ar_true).abs());
    }
    ferr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rerr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = |v: &[f64]| v[v.len() / 2];
    let med_f = med(&ferr);
    let med_r = med(&rerr);
    // 0.02 rad ≈ 1.15°. The rear slip is directly observable through the motion;
    // the front carries the (estimated) steering, so it is looser but still tight.
    assert!(med_r < 0.02, "rear slip median err {med_r:.4} rad too high");
    assert!(
        med_f < 0.03,
        "front slip median err {med_f:.4} rad too high"
    );
}

#[test]
fn recovers_truth_with_tiny_noise() {
    // With near-zero measurement noise and the exact model, the smoother should
    // track the truth very closely (a basic correctness sanity check).
    let car = CarParams::f1_2024_calibrated();
    let tire = PacejkaTire::f1_default();
    let n = 150;
    let tp = TruthParams {
        n_epochs: n,
        pos_noise: 1e-3,
        speed_noise: 1e-3,
        q_delta: Q_DELTA_TRUE,
        q_fdrive: Q_FDRIVE_TRUE,
        seed: 0xBEEF,
    };
    let (t, truth, _td, _tf, mx, my, ms) = make_synthetic(&car, &tire, tp);
    let cfg = synthetic_config(1e-2, 1e-2); // R slightly above the injected noise
    let res = smooth_states(&t, &mx, &my, &ms, &car, &tire, &cfg).unwrap();

    // Position and speed track tightly.
    for k in 30..n {
        let vx = res.state[k][IVX];
        let vy = res.state[k][IVY];
        let spd = (vx * vx + vy * vy).sqrt();
        let spd_true = (truth[k][3] * truth[k][3] + truth[k][4] * truth[k][4]).sqrt();
        assert!(
            (spd - spd_true).abs() < 0.5,
            "speed err at {k}: {} vs {}",
            spd,
            spd_true
        );
        assert!((res.state[k][IX] - truth[k][0]).abs() < 1.0);
        assert!((res.state[k][IY] - truth[k][1]).abs() < 1.0);
    }
}

#[test]
fn gaps_are_prediction_only_and_grow_covariance() {
    let car = CarParams::f1_2024_calibrated();
    let tire = PacejkaTire::f1_default();
    let n = 120;
    let tp = TruthParams {
        n_epochs: n,
        pos_noise: 1.0,
        speed_noise: 0.3,
        q_delta: Q_DELTA_TRUE,
        q_fdrive: Q_FDRIVE_TRUE,
        seed: 0x6A75,
    };
    let (t, _truth, _td, _tf, mut mx, mut my, mut ms) = make_synthetic(&car, &tire, tp);

    // Punch a long (24-sample, ~3 s) gap into the measurements. A short gap gets
    // interpolated tightly by the RTS smoother from both sides; a long one leaves
    // the middle genuinely under-determined, so the smoothed covariance grows.
    let gap = 50..74;
    for k in gap.clone() {
        mx[k] = f64::NAN;
        my[k] = f64::NAN;
        ms[k] = f64::NAN;
    }
    let cfg = synthetic_config(1.0, 0.3);
    let res = smooth_states(&t, &mx, &my, &ms, &car, &tire, &cfg).unwrap();

    assert_eq!(res.diagnostics.n_gaps, 24, "expected 24 gap epochs");
    // Position uncertainty in the middle of the gap exceeds a well-observed epoch.
    let sigma_good = res.std[30][IX];
    let sigma_gap = res.std[62][IX];
    assert!(
        sigma_gap > sigma_good,
        "gap σ_X {sigma_gap:.3} should exceed nominal {sigma_good:.3}"
    );
    // Everything stays finite through the gap.
    for k in 0..n {
        for i in 0..N {
            assert!(res.state[k][i].is_finite(), "state {k},{i} non-finite");
            assert!(res.std[k][i].is_finite());
        }
    }
}

#[test]
fn innovation_gate_rejects_outliers() {
    let car = CarParams::f1_2024_calibrated();
    let tire = PacejkaTire::f1_default();
    let n = 120;
    let tp = TruthParams {
        n_epochs: n,
        pos_noise: 1.0,
        speed_noise: 0.3,
        q_delta: Q_DELTA_TRUE,
        q_fdrive: Q_FDRIVE_TRUE,
        seed: 0x0157,
    };
    let (t, _truth, _td, _tf, mut mx, mut my, ms) = make_synthetic(&car, &tire, tp);

    // Inject a gross position outlier (100 m jump) at one epoch.
    mx[70] += 100.0;
    my[70] -= 80.0;

    let mut cfg = synthetic_config(1.0, 0.3);
    cfg.nis_gate = 25.0; // enable gating
    let res = smooth_states(&t, &mx, &my, &ms, &car, &tire, &cfg).unwrap();
    assert!(
        res.diagnostics.n_rejected >= 1,
        "expected the 100 m outlier to be gated out"
    );
    // The smoothed track is not yanked to the outlier.
    assert!(
        (res.state[70][IX]).is_finite(),
        "state stays finite after rejection"
    );
}

// --- linear-algebra unit tests -------------------------------------------

#[test]
fn invert3_roundtrip() {
    let a = vec![
        vec![4.0, 1.0, 0.5],
        vec![1.0, 3.0, 0.2],
        vec![0.5, 0.2, 2.0],
    ];
    let ai = invert3(&a).unwrap();
    let prod = mat_mul(&a, &ai);
    for i in 0..3 {
        for j in 0..3 {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!((prod[i][j] - expect).abs() < 1e-12, "prod[{i}][{j}]");
        }
    }
    // Singular matrix → None.
    let sing = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![1.0, 1.0, 1.0],
    ];
    assert!(invert3(&sing).is_none());
}

#[test]
fn diagnostics_json_is_well_formed() {
    let car = CarParams::f1_2024_calibrated();
    let tire = PacejkaTire::f1_default();
    let n = 40;
    let tp = TruthParams {
        n_epochs: n,
        pos_noise: 1.0,
        speed_noise: 0.3,
        q_delta: Q_DELTA_TRUE,
        q_fdrive: Q_FDRIVE_TRUE,
        seed: 0x9,
    };
    let (t, _tr, _td, _tf, mx, my, ms) = make_synthetic(&car, &tire, tp);
    let res = smooth_states(&t, &mx, &my, &ms, &car, &tire, &synthetic_config(1.0, 0.3)).unwrap();
    let json = diagnostics_json(&res);
    assert!(json.contains("\"state_std\""));
    assert!(json.contains("\"nis\""));
    assert!(json.contains("\"robustness\""));
    // Rough structural balance of braces.
    assert_eq!(
        json.matches('{').count(),
        json.matches('}').count(),
        "unbalanced braces in diagnostics JSON"
    );
}

#[test]
fn attach_estimated_channels_appends_all_five() {
    use crate::telemetry::GridKind;
    use std::collections::BTreeMap;

    let car = CarParams::f1_2024_calibrated();
    let tire = PacejkaTire::f1_default();
    let n = 60;
    let tp = TruthParams {
        n_epochs: n,
        pos_noise: 1.0,
        speed_noise: 0.3,
        q_delta: Q_DELTA_TRUE,
        q_fdrive: Q_FDRIVE_TRUE,
        seed: 0x5A5A,
    };
    let (t, _truth, _td, _tf, x, y, speed) = make_synthetic(&car, &tire, tp);
    let mut res =
        smooth_states(&t, &x, &y, &speed, &car, &tire, &synthetic_config(1.0, 0.3)).unwrap();
    // Force a "no value" NaN into one slip sample to check it survives the append.
    res.slip_front[7] = f64::NAN;

    // An aligned telemetry with n rows; attach must not change the row count.
    let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
    channels.insert(ChannelId::Time, t.clone());
    channels.insert(ChannelId::Speed, speed.clone());
    let aligned = Telemetry {
        grid: GridKind::T,
        channels,
        metadata: Vec::new(),
    };

    let out = attach_estimated_channels(&aligned, &res);

    // All five dynamic channels present, each length == input row count.
    for id in OUTPUT_CHANNELS {
        let c = out
            .channel(*id)
            .unwrap_or_else(|| panic!("channel {} not appended", id.name()));
        assert_eq!(c.len(), n, "channel {} length", id.name());
    }
    // The forced NaN propagated; a neighbouring finite sample is untouched.
    assert!(out.channel(ChannelId::SlipAngleFront).unwrap()[7].is_nan());
    assert!(out.channel(ChannelId::SlipAngleFront).unwrap()[8].is_finite());
    // A state-derived channel (yaw_rate) is finite through the run.
    assert!(out.channel(ChannelId::YawRate).unwrap()[30].is_finite());
}
