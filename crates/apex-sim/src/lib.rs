#![deny(unsafe_code)]
//! Real-time simulation engine for Apex-14.
//!
//! Provides a fixed-budget integrator and (in future phases) a UDP
//! simulation server for hardware-in-the-loop testing.

pub mod realtime;
