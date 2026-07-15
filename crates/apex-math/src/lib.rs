#![deny(unsafe_code)]
//! Core mathematical types and operations for Apex-14: vectors, matrices, and
//! dual numbers for automatic differentiation.

pub mod dual;
pub mod float;
pub mod hash;
pub mod interp;
pub mod lm;
pub mod mat3;
pub mod seed;
pub mod sparse;
pub mod vec3;

pub use dual::Dual;
pub use float::Float;
pub use hash::{content_hash, ContentHash, Hash, HashWriter, HASH_VERSION};
pub use interp::{GridAxis, HermiteGrid};
pub use lm::{levenberg_marquardt, LmConfig, LmIteration, LmResult, ResidualProvider};
pub use mat3::Mat3;
pub use seed::resolve_seed;
pub use sparse::{CsrBuilder, CsrMatrix};
pub use vec3::Vec3;
