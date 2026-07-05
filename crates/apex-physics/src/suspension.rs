//! Suspension model: per-corner spring-damper units, anti-roll bars, and the
//! full four-corner assembly with static-equilibrium solving.

/// Parameters for a single suspension corner (spring + damper).
#[derive(Debug, Clone, Copy)]
pub struct SuspensionParams {
    /// Linear spring rate K_0 (N/m).
    pub spring_rate: f64,
    /// Progressive spring rate coefficient K_1 (N/m²) — force increases nonlinearly with travel.
    pub spring_rate_progressive: f64,
    /// Bump (compression) damping coefficient (N·s/m).
    /// Applied when the suspension is compressing (dz > 0).
    pub damping_bump: f64,
    /// Rebound (extension) damping coefficient (N·s/m).
    /// Applied when the suspension is extending (dz < 0).
    /// F1 cars typically have rebound ≈ 3× bump damping.
    pub damping_rebound: f64,
    /// Maximum suspension travel in compression (m, positive value).
    pub max_compression: f64,
    /// Maximum suspension travel in extension (m, positive value).
    pub max_extension: f64,
    /// Static ride height at this corner under car weight (m).
    pub static_ride_height: f64,
}

impl SuspensionParams {
    /// Returns F1-representative front suspension parameters.
    pub fn f1_front() -> Self {
        SuspensionParams {
            spring_rate: 200_000.0,             // 200 kN/m
            spring_rate_progressive: 500_000.0, // progressive stiffening
            damping_bump: 8_000.0,              // 8 kN·s/m
            damping_rebound: 25_000.0,          // 25 kN·s/m (≈3× bump)
            max_compression: 0.025,             // 25mm
            max_extension: 0.030,               // 30mm
            static_ride_height: 0.030,          // 30mm design ride height
        }
    }

    /// Returns F1-representative rear suspension parameters.
    pub fn f1_rear() -> Self {
        SuspensionParams {
            spring_rate: 250_000.0,
            spring_rate_progressive: 600_000.0,
            damping_bump: 10_000.0,
            damping_rebound: 30_000.0,
            max_compression: 0.025,
            max_extension: 0.030,
            static_ride_height: 0.030,
        }
    }

    /// Compute the spring force for a given suspension displacement.
    ///
    /// z_s = suspension travel (m):
    ///   positive = compression (wheel pushed up toward chassis)
    ///   negative = extension (wheel dropped away from chassis)
    ///
    /// Returns force (N): positive force pushes the wheel down (restoring).
    /// The spring force is: F = -(K_0 · z + K_1 · z · |z|)
    /// The progressive term K_1·z·|z| adds stiffness symmetrically with travel magnitude.
    pub fn spring_force(&self, z_s: f64) -> f64 {
        -(self.spring_rate * z_s + self.spring_rate_progressive * z_s * z_s.abs())
    }

    /// Compute the damper force for a given suspension velocity.
    ///
    /// dz_s = suspension velocity (m/s):
    ///   positive = compressing (bump)
    ///   negative = extending (rebound)
    ///
    /// Returns force (N): opposes the velocity direction.
    /// Uses asymmetric damping: bump coefficient for compression, rebound for extension.
    pub fn damper_force(&self, dz_s: f64) -> f64 {
        if dz_s >= 0.0 {
            -self.damping_bump * dz_s // compression: oppose with bump damping
        } else {
            -self.damping_rebound * dz_s // extension: oppose with rebound damping
        }
    }

    /// Total suspension force (spring + damper).
    pub fn total_force(&self, z_s: f64, dz_s: f64) -> f64 {
        self.spring_force(z_s) + self.damper_force(dz_s)
    }
}

/// Anti-roll bar connecting the left and right corners of an axle.
///
/// The ARB generates a force that resists the difference in suspension travel
/// between left and right wheels. This reduces body roll but increases
/// lateral load transfer, which affects the handling balance.
#[derive(Debug, Clone, Copy)]
pub struct AntiRollBar {
    /// Torsional stiffness (N·m/rad), but applied as a linear stiffness
    /// between left and right suspension travel: F = K_arb · (z_left - z_right).
    /// Units effectively N/m when applied to suspension displacement.
    pub stiffness: f64,
}

impl AntiRollBar {
    /// F1-representative front anti-roll bar.
    pub fn f1_front() -> Self {
        AntiRollBar {
            stiffness: 50_000.0,
        }
    }

    /// F1-representative rear anti-roll bar.
    pub fn f1_rear() -> Self {
        AntiRollBar {
            stiffness: 30_000.0,
        }
    }

    /// Compute the ARB force on the left wheel.
    /// The right wheel gets the opposite force.
    ///
    /// Returns force on left wheel (N):
    ///   When left is more compressed than right, force pushes left down (positive)
    ///   to resist the roll.
    pub fn force_left(&self, z_left: f64, z_right: f64) -> f64 {
        -self.stiffness * (z_left - z_right)
    }

    /// Force on right wheel (opposite of left).
    pub fn force_right(&self, z_left: f64, z_right: f64) -> f64 {
        self.stiffness * (z_left - z_right)
    }
}

/// Complete four-corner suspension system with anti-roll bars.
#[derive(Debug, Clone)]
pub struct SuspensionSystem {
    /// Front-left corner.
    pub front_left: SuspensionParams,
    /// Front-right corner.
    pub front_right: SuspensionParams,
    /// Rear-left corner.
    pub rear_left: SuspensionParams,
    /// Rear-right corner.
    pub rear_right: SuspensionParams,
    /// Front anti-roll bar.
    pub arb_front: AntiRollBar,
    /// Rear anti-roll bar.
    pub arb_rear: AntiRollBar,
}

impl SuspensionSystem {
    /// Create an F1-representative suspension system.
    /// Front and rear corners use their respective defaults.
    /// Left/right are symmetric.
    pub fn f1_default() -> Self {
        SuspensionSystem {
            front_left: SuspensionParams::f1_front(),
            front_right: SuspensionParams::f1_front(),
            rear_left: SuspensionParams::f1_rear(),
            rear_right: SuspensionParams::f1_rear(),
            arb_front: AntiRollBar::f1_front(),
            arb_rear: AntiRollBar::f1_rear(),
        }
    }

    /// Compute all four suspension forces given the four corner displacements and velocities.
    ///
    /// z = [z_fl, z_fr, z_rl, z_rr] — suspension displacements (m)
    /// dz = [dz_fl, dz_fr, dz_rl, dz_rr] — suspension velocities (m/s)
    ///
    /// Returns [F_fl, F_fr, F_rl, F_rr] — total force at each corner (N),
    /// including spring, damper, and anti-roll bar contributions.
    pub fn forces(&self, z: &[f64; 4], dz: &[f64; 4]) -> [f64; 4] {
        let f_fl = self.front_left.total_force(z[0], dz[0]) + self.arb_front.force_left(z[0], z[1]);
        let f_fr =
            self.front_right.total_force(z[1], dz[1]) + self.arb_front.force_right(z[0], z[1]);
        let f_rl = self.rear_left.total_force(z[2], dz[2]) + self.arb_rear.force_left(z[2], z[3]);
        let f_rr = self.rear_right.total_force(z[3], dz[3]) + self.arb_rear.force_right(z[2], z[3]);
        [f_fl, f_fr, f_rl, f_rr]
    }

    /// Compute the static equilibrium displacements under a given set of vertical loads.
    ///
    /// loads = [F_z_fl, F_z_fr, F_z_rl, F_z_rr] — vertical load at each corner (N)
    ///
    /// Finds the displacement z where spring_force(z) = load for each corner
    /// (ignoring damper and ARB for static equilibrium).
    ///
    /// Uses Newton's method per corner (the spring force is monotonic, so convergence is fast).
    pub fn static_equilibrium(&self, loads: &[f64; 4]) -> [f64; 4] {
        let params = [
            &self.front_left,
            &self.front_right,
            &self.rear_left,
            &self.rear_right,
        ];
        let mut z = [0.0; 4];
        for (i, zi_out) in z.iter_mut().enumerate() {
            // Solve: -K_0·z - K_1·z·|z| = -load  →  K_0·z + K_1·z·|z| = load
            let target = loads[i];
            let mut zi = target / params[i].spring_rate; // initial guess from linear spring
            for _ in 0..20 {
                let f =
                    params[i].spring_rate * zi + params[i].spring_rate_progressive * zi * zi.abs();
                let df = params[i].spring_rate + params[i].spring_rate_progressive * 2.0 * zi.abs();
                if df.abs() < 1e-20 {
                    break;
                }
                zi -= (f - target) / df;
            }
            *zi_out = zi;
        }
        z
    }
}

impl apex_math::ContentHash for SuspensionParams {
    /// Encode the seven per-corner spring/damper parameters in declaration order.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let SuspensionParams {
            spring_rate,
            spring_rate_progressive,
            damping_bump,
            damping_rebound,
            max_compression,
            max_extension,
            static_ride_height,
        } = self;
        w.f64(*spring_rate);
        w.f64(*spring_rate_progressive);
        w.f64(*damping_bump);
        w.f64(*damping_rebound);
        w.f64(*max_compression);
        w.f64(*max_extension);
        w.f64(*static_ride_height);
    }
}

impl apex_math::ContentHash for AntiRollBar {
    /// Encode the single anti-roll-bar stiffness.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let AntiRollBar { stiffness } = self;
        w.f64(*stiffness);
    }
}

impl apex_math::ContentHash for SuspensionSystem {
    /// Encode the four corners then the two anti-roll bars, in declaration order
    /// (nested [`SuspensionParams`]/[`AntiRollBar`] hashes).
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let SuspensionSystem {
            front_left,
            front_right,
            rear_left,
            rear_right,
            arb_front,
            arb_rear,
        } = self;
        front_left.hash_into(w);
        front_right.hash_into(w);
        rear_left.hash_into(w);
        rear_right.hash_into(w);
        arb_front.hash_into(w);
        arb_rear.hash_into(w);
    }
}

/// Content hash of a [`SuspensionSystem`], under domain `"suspension"`.
pub fn suspension_hash(s: &SuspensionSystem) -> apex_math::Hash {
    apex_math::content_hash("suspension", s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn spring_force_signs_and_value() {
        let s = SuspensionParams::f1_front();
        assert_eq!(s.spring_force(0.0), 0.0);
        // compression -> negative (restoring, pushes wheel down)
        assert!(s.spring_force(0.01) < 0.0);
        // extension -> positive (restoring, pulls wheel up)
        assert!(s.spring_force(-0.01) > 0.0);

        // exact value at 10mm compression: -(200000*0.01 + 500000*0.01*0.01) = -2050
        assert!(
            approx(s.spring_force(0.01), -2050.0, 1e-9),
            "{}",
            s.spring_force(0.01)
        );
    }

    #[test]
    fn spring_progressive_stiffening() {
        let s = SuspensionParams::f1_front();
        let f1 = s.spring_force(0.01).abs();
        let f2 = s.spring_force(0.02).abs();
        // progressive term makes |F(0.02)| more than 2× |F(0.01)|
        assert!(f2 > 2.0 * f1, "f2 {} vs 2*f1 {}", f2, 2.0 * f1);
    }

    #[test]
    fn damper_force_and_asymmetry() {
        let s = SuspensionParams::f1_front();
        assert_eq!(s.damper_force(0.0), 0.0);
        // compression
        assert!(
            approx(s.damper_force(0.1), -800.0, 1e-9),
            "{}",
            s.damper_force(0.1)
        );
        // extension
        assert!(
            approx(s.damper_force(-0.1), 2500.0, 1e-9),
            "{}",
            s.damper_force(-0.1)
        );
        // rebound stronger than bump at the same speed
        assert!(s.damper_force(-0.1).abs() > s.damper_force(0.1).abs());
    }

    #[test]
    fn anti_roll_bar() {
        let arb = AntiRollBar::f1_front();
        // equal travel -> no force
        assert_eq!(arb.force_left(0.01, 0.01), 0.0);
        assert_eq!(arb.force_right(0.01, 0.01), 0.0);

        // left more compressed than right
        let fl = arb.force_left(0.02, 0.01);
        let fr = arb.force_right(0.02, 0.01);
        assert!(fl < 0.0, "left {} should push up... ", fl); // -K*(0.01) < 0
        assert!(fr > 0.0, "right {}", fr);
        // net vertical force is zero (a pure couple)
        assert!(approx(fl + fr, 0.0, 1e-9));
    }

    #[test]
    fn full_system_symmetric() {
        let sys = SuspensionSystem::f1_default();
        let z = [0.01, 0.01, 0.012, 0.012];
        let dz = [0.05, 0.05, 0.05, 0.05];
        let f = sys.forces(&z, &dz);
        // symmetric L/R -> front pair equal, rear pair equal
        assert!(approx(f[0], f[1], 1e-9), "front {} {}", f[0], f[1]);
        assert!(approx(f[2], f[3], 1e-9), "rear {} {}", f[2], f[3]);
    }

    #[test]
    fn full_system_pure_roll() {
        let sys = SuspensionSystem::f1_default();
        // front rolled: FL compressed, FR extended equally; rears neutral
        let z = [0.01, -0.01, 0.0, 0.0];
        let dz = [0.0; 4];
        let f = sys.forces(&z, &dz);
        // ARB adds a restoring couple on the front axle; rears see no force
        assert!(approx(f[2], 0.0, 1e-9), "rear-left {}", f[2]);
        assert!(approx(f[3], 0.0, 1e-9), "rear-right {}", f[3]);
        // front forces are opposite in sign (restoring the roll)
        assert!(f[0] < 0.0 && f[1] > 0.0, "front {} {}", f[0], f[1]);
    }

    #[test]
    fn static_equilibrium_equal_loads() {
        let sys = SuspensionSystem::f1_default();
        let load = 1957.0; // ~ quarter car weight
        let loads = [load, load, load, load];
        let z = sys.static_equilibrium(&loads);

        // all compressed (positive) and L/R symmetric within axle
        assert!(z.iter().all(|&zi| zi > 0.0), "all compressed: {:?}", z);
        assert!(approx(z[0], z[1], 1e-9), "front L/R equal");
        assert!(approx(z[2], z[3], 1e-9), "rear L/R equal");

        // small travel (< 15 mm) and spring force matches the load
        assert!(z[0] < 0.015, "front travel {} too large", z[0]);
        assert!(
            approx(sys.front_left.spring_force(z[0]), -load, 1e-6),
            "spring force {} vs load {}",
            sys.front_left.spring_force(z[0]),
            load
        );
    }

    #[test]
    fn static_equilibrium_ride_height() {
        let sys = SuspensionSystem::f1_default();
        let load = 798.0 * 9.81 / 4.0; // quarter-car weight
        let loads = [load, load, load, load];
        let z = sys.static_equilibrium(&loads);

        // actual ride height = design height - compression
        let ride_height = sys.front_left.static_ride_height - z[0];
        assert!(
            ride_height > 0.0,
            "ride height {} should be positive",
            ride_height
        );
        assert!(
            (0.020..=0.030).contains(&ride_height),
            "ride height {} out of expected 20-30mm range",
            ride_height
        );
    }

    #[test]
    fn suspension_hash_frozen_and_field_sensitive() {
        assert_eq!(
            suspension_hash(&SuspensionSystem::f1_default()).to_hex(),
            FROZEN_SUSPENSION_F1_DEFAULT
        );
        // A change in any corner or ARB must move the hash.
        let mut s = SuspensionSystem::f1_default();
        s.front_left.spring_rate += 1.0;
        assert_ne!(
            suspension_hash(&s),
            suspension_hash(&SuspensionSystem::f1_default())
        );
        let mut s2 = SuspensionSystem::f1_default();
        s2.arb_rear.stiffness += 1.0;
        assert_ne!(
            suspension_hash(&s2),
            suspension_hash(&SuspensionSystem::f1_default())
        );
        // Sanity: a corner's own hash reflects a field change.
        let mut c = SuspensionParams::f1_front();
        let base = apex_math::content_hash("t", &c);
        c.damping_rebound += 1.0;
        assert_ne!(apex_math::content_hash("t", &c), base);
    }

    const FROZEN_SUSPENSION_F1_DEFAULT: &str =
        "b384aa087c0894dbe728d525a9e3fc6efee44ae189761444368a6ea8808368ca";
}
