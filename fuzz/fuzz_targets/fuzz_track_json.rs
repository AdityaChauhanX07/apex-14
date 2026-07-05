//! Fuzz the JSON track loader's public parse path.
//!
//! Property: `parse_track_json` must return `Ok`/`Err` for ANY input — never
//! panic, OOM, or overflow. We assert nothing about the value.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = apex_track::parse_track_json(&text);
});
