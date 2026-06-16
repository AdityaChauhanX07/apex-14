//! Core mathematical types and operations for Apex-14: vectors, matrices, and
//! dual numbers for automatic differentiation.

pub mod dual;
pub mod mat3;
pub mod vec3;

pub use dual::Dual;
pub use mat3::Mat3;
pub use vec3::Vec3;
