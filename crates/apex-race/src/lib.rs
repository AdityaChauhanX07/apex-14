#![deny(unsafe_code)]
//! Full-field race simulation with Monte Carlo analysis.
//!
//! Simulates complete Grand Prix races with multiple cars, pit strategies,
//! tire degradation, fuel effects, and (in later phases) probabilistic
//! events like safety cars, rain, and mechanical failures.

pub mod config;
pub mod events;
pub mod monte_carlo;
pub mod sim;
