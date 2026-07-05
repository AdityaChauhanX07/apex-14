//! Content hashes for solver settings.
//!
//! Per-config [`ContentHash`](apex_math::ContentHash) impls live next to each
//! config type; this module provides named entry points (with fixed domain
//! tags) and composite hashes for the multi-config settings a single run uses.
//!
//! All hashes exclude cosmetic fields (print cadence) and the run seed — see
//! the individual `hash_into` impls.

use apex_math::{content_hash, ContentHash, Hash, HashWriter, HASH_VERSION};
use apex_physics::{AeroModel, PacejkaTire, SuspensionSystem};

use crate::cmaes::CmaEsConfig;
use crate::collocation::CollocationConfig;
use crate::direct_solver::DirectSolverConfig;
use crate::gauss_newton::GaussNewtonConfig;
use crate::solver::SolverConfig;

/// Content hash of CMA-ES setup-optimization settings (domain `"cmaes"`).
/// Seed-independent by construction.
pub fn cmaes_settings_hash(cfg: &CmaEsConfig) -> Hash {
    content_hash("cmaes", cfg)
}

/// Content hash of the Augmented-Lagrangian NLP solver settings
/// (domain `"solver.al"`).
pub fn al_solver_settings_hash(cfg: &SolverConfig) -> Hash {
    content_hash("solver.al", cfg)
}

/// Content hash of the Gauss-Newton solver settings (domain `"solver.gn"`).
pub fn gauss_newton_settings_hash(cfg: &GaussNewtonConfig) -> Hash {
    content_hash("solver.gn", cfg)
}

/// Content hash of the direct defect-correction solver settings
/// (domain `"solver.direct"`).
pub fn direct_solver_settings_hash(cfg: &DirectSolverConfig) -> Hash {
    content_hash("solver.direct", cfg)
}

/// Composite content hash of the trajectory-optimization settings that
/// determine a result: the collocation discretization and the Gauss-Newton
/// solver, hashed in that fixed order under domain `"optimize.gn"`.
pub fn optimize_gn_settings_hash(collocation: &CollocationConfig, gn: &GaussNewtonConfig) -> Hash {
    let mut w = HashWriter::new();
    w.str(HASH_VERSION);
    w.str("optimize.gn");
    collocation.hash_into(&mut w);
    gn.hash_into(&mut w);
    w.finish()
}

/// Composite content hash of the settings that determine a **7-DOF / 14-DOF**
/// trajectory-optimization (or forward-sim) result, under domain
/// `"optimize.14dof"`.
///
/// Composition rule (fixed order): `HASH_VERSION ‖ "optimize.14dof" ‖
/// collocation ‖ gn ‖ tire ‖ suspension ‖ aero`. This EXTENDS
/// [`optimize_gn_settings_hash`] with the tire, suspension, and aero models,
/// which the 7-/14-DOF paths read but the point-mass Gauss-Newton hash does not
/// cover — so a change to any of those models now moves the emitted
/// `config_hash` for 14-DOF telemetry. Distinct domain tag from `"optimize.gn"`,
/// so the two composites never collide.
pub fn optimize_fourteen_dof_settings_hash(
    collocation: &CollocationConfig,
    gn: &GaussNewtonConfig,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
) -> Hash {
    let mut w = HashWriter::new();
    w.str(HASH_VERSION);
    w.str("optimize.14dof");
    collocation.hash_into(&mut w);
    gn.hash_into(&mut w);
    tire.hash_into(&mut w);
    suspension.hash_into(&mut w);
    aero.hash_into(&mut w);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourteen_dof_hash_extends_gn_and_is_model_sensitive() {
        let coll = CollocationConfig::default();
        let gn = GaussNewtonConfig::default();
        let tire = PacejkaTire::f1_default();
        let susp = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();

        let base = optimize_fourteen_dof_settings_hash(&coll, &gn, &tire, &susp, &aero);
        // Reproducible.
        assert_eq!(
            base,
            optimize_fourteen_dof_settings_hash(&coll, &gn, &tire, &susp, &aero)
        );
        // Distinct from the point-mass GN composite (different domain + inputs).
        assert_ne!(base, optimize_gn_settings_hash(&coll, &gn));
        // Sensitive to each newly-folded model.
        let mut tire2 = tire;
        tire2.lateral.mu += 0.05;
        assert_ne!(
            base,
            optimize_fourteen_dof_settings_hash(&coll, &gn, &tire2, &susp, &aero)
        );
        let mut susp2 = susp.clone();
        susp2.arb_front.stiffness += 100.0;
        assert_ne!(
            base,
            optimize_fourteen_dof_settings_hash(&coll, &gn, &tire, &susp2, &aero)
        );
        let mut aero2 = aero;
        aero2.cd_base += 0.01;
        assert_ne!(
            base,
            optimize_fourteen_dof_settings_hash(&coll, &gn, &tire, &susp, &aero2)
        );
    }

    #[test]
    fn cmaes_seed_independent() {
        // Two configs differing ONLY in seed must hash identically.
        let a = CmaEsConfig {
            seed: 1,
            ..Default::default()
        };
        let b = CmaEsConfig {
            seed: 999_999,
            ..Default::default()
        };
        assert_ne!(a.seed, b.seed, "test setup: seeds differ");
        assert_eq!(
            cmaes_settings_hash(&a),
            cmaes_settings_hash(&b),
            "content hash must exclude the seed"
        );
    }

    #[test]
    fn cmaes_sensitive_to_real_field() {
        let base = CmaEsConfig::default();
        let mut changed = base.clone();
        changed.initial_sigma += 1e-9;
        assert_ne!(
            cmaes_settings_hash(&base),
            cmaes_settings_hash(&changed),
            "changing a result-determining field must change the hash"
        );
    }

    #[test]
    fn print_interval_excluded() {
        let base = GaussNewtonConfig::default();
        let mut noisy = base.clone();
        noisy.print_interval = base.print_interval + 17;
        assert_eq!(
            gauss_newton_settings_hash(&base),
            gauss_newton_settings_hash(&noisy),
            "cosmetic print_interval must not affect the hash"
        );
    }

    #[test]
    fn composite_order_and_sensitivity() {
        let coll = CollocationConfig::default();
        let gn = GaussNewtonConfig::default();
        let h0 = optimize_gn_settings_hash(&coll, &gn);
        // Deterministic.
        assert_eq!(h0, optimize_gn_settings_hash(&coll, &gn));
        // Sensitive to a collocation field.
        let mut coll2 = coll.clone();
        coll2.n_nodes += 1;
        assert_ne!(h0, optimize_gn_settings_hash(&coll2, &gn));
        // Sensitive to a GN field.
        let mut gn2 = gn.clone();
        gn2.damping += 1e-9;
        assert_ne!(h0, optimize_gn_settings_hash(&coll, &gn2));
    }

    #[test]
    fn domains_do_not_collide() {
        // AL and GN solver configs are different types under different domains;
        // even at their defaults they must not collide.
        let al = al_solver_settings_hash(&SolverConfig::default());
        let gn = gauss_newton_settings_hash(&GaussNewtonConfig::default());
        assert_ne!(al.to_hex(), gn.to_hex());
    }

    #[test]
    fn method_enum_swap_changes_hash() {
        use crate::collocation::CollocationMethod;
        let gn = GaussNewtonConfig::default();
        let trap = CollocationConfig {
            method: CollocationMethod::Trapezoidal,
            ..Default::default()
        };
        let hs = CollocationConfig {
            method: CollocationMethod::HermiteSimpson,
            ..Default::default()
        };
        assert_ne!(
            optimize_gn_settings_hash(&trap, &gn),
            optimize_gn_settings_hash(&hs, &gn),
            "swapping the collocation method must change the hash"
        );
    }

    #[test]
    fn frozen_cmaes_default_vector() {
        // FROZEN known-answer vector — any accidental encoding/order/policy
        // change flips this and fails CI. Update only as a deliberate change
        // (and bump HASH_VERSION).
        assert_eq!(
            cmaes_settings_hash(&CmaEsConfig::default()).to_hex(),
            FROZEN_CMAES_DEFAULT
        );
    }

    const FROZEN_CMAES_DEFAULT: &str =
        "bd712cbdeb56379f476728ea7d0be14945da0f4c768af29086eed98aedbc4050";
}
