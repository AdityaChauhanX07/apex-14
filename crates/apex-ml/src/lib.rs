#![deny(unsafe_code)]
//! ML-based raceline prediction for optimizer warmstart.
//!
//! This crate provides training data generation, neural network
//! architecture, and inference for predicting near-optimal racing
//! lines from track geometry.

pub mod data;
pub mod io;
pub mod pipeline;
