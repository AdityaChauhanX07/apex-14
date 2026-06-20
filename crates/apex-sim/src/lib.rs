#![deny(unsafe_code)]
//! Real-time simulation engine for Apex-14.
//!
//! Provides a fixed-budget integrator and (in future phases) a UDP
//! simulation server for hardware-in-the-loop testing.

pub mod protocol;
pub mod realtime;
pub mod server;

// Shared memory requires unsafe for memory-mapped file creation (memmap2).
// The unsafe surface is limited to MmapMut::map_mut(); all data access
// uses checked byte-slice operations via to_bytes/from_bytes.
#[allow(unsafe_code)]
pub mod shared_mem;
