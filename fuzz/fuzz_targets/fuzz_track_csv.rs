//! Fuzz the TUMFTM CSV track importer's public parse path.
//!
//! Property: `parse_tumftm_csv` must return `Ok`/`Err` for ANY input — never
//! panic, OOM, or overflow. We assert nothing about the value; libFuzzer finds
//! the crashes.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The importer takes &str; feed arbitrary bytes via a lossy conversion so
    // non-UTF-8 input still exercises the parser.
    let text = String::from_utf8_lossy(data);
    let _ = apex_track::parse_tumftm_csv(&text, "fuzz");
});
