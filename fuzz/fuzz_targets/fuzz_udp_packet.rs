//! Fuzz the apex-sim UDP packet deserializers.
//!
//! The protocol has no framing/type byte (see docs/protocol.md): a datagram is
//! decoded as one of two packet types, chosen by direction (the server decodes
//! `InputPacket`; the dashboard decodes `OutputPacket`). There is no single
//! dispatch function, so to keep every variant reachable we run BOTH decoders
//! over the same fuzz bytes.
//!
//! Property: each `from_bytes` must return `Some`/`None` for ANY input — never
//! panic, index out of bounds, or overflow. We assert nothing about the value.
#![no_main]

use apex_sim::protocol::{InputPacket, OutputPacket};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // InputPacket dispatch (driver -> sim).
    if let Some(mut pkt) = InputPacket::from_bytes(data) {
        // clamp() is part of the receive path in the server; exercise it too.
        pkt.clamp();
        let _ = pkt.to_bytes();
    }

    // OutputPacket dispatch (sim -> dashboard).
    if let Some(pkt) = OutputPacket::from_bytes(data) {
        let _ = pkt.to_bytes();
    }
});
