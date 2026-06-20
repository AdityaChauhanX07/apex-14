#![deny(unsafe_code)]
//! Reinforcement learning environment and agents for Apex-14.
//!
//! Wraps the 14-DOF vehicle dynamics model as a gym-style environment
//! for training RL agents to drive race cars.

pub mod env;
pub mod observation;
pub mod reward;
