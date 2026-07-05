//! Deterministic content hashing for Apex-14 configuration types.
//!
//! The goal is a *content hash*: two semantically identical inputs must produce
//! byte-identical hashes, and two semantically different inputs must not
//! collide. The hard part is canonical encoding, not the hash function
//! (BLAKE3). This module owns the canonical encoder ([`HashWriter`]), the float
//! policy, the [`Hash`] output type, and the [`ContentHash`] trait.
//!
//! Because `apex-math` is the workspace's leaf crate, it cannot reference the
//! config types (`CarParams`, `Track`, …) that live in higher crates. Each
//! owning crate therefore implements [`ContentHash`] for its own types (a
//! foreign trait on a local type — allowed by the orphan rule), and calls
//! [`content_hash`] with an appropriate domain tag.
//!
//! # Float policy
//!
//! Floats are encoded as their raw IEEE-754 bits (little-endian), never as
//! decimal text — raw bits are bit-exact and platform-independent, and a
//! content hash *should* treat `1.0` and the next representable value as
//! different. Two normalizations are applied first:
//!
//! - `-0.0` and `+0.0` encode identically (both as the `+0.0` bit pattern).
//! - All NaN payloads canonicalize to a single fixed pattern, so any two NaNs
//!   collide deterministically and the encoder never diverges on NaN.
//!
//! A `debug_assert!(f.is_finite())` additionally surfaces non-finite config
//! values (a programming bug) loudly in debug/test builds while staying total
//! in release.
//!
//! # Stability contract
//!
//! A content hash is a compatibility surface: changing the encoding changes
//! every stored hash. [`content_hash`] writes a versioned prefix
//! ([`HASH_VERSION`]) plus a per-type `domain` tag before the value, so
//! distinct types cannot collide and a deliberate future encoding change can
//! bump the version. Frozen known-answer test vectors guard against accidental
//! changes.

/// Versioned domain prefix written before every content hash. Bump this only
/// as a deliberate, breaking change to the encoding (all stored hashes change).
pub const HASH_VERSION: &str = "apex14.chash.v1";

/// Canonical bit encoding of an `f64` under the float policy: `-0.0`→`+0.0`
/// and all NaN payloads collapse to one fixed pattern. Non-finite values are
/// *not* rejected here (that is the caller's `debug_assert`); this stays total
/// so canonicalization is deterministic in release builds.
fn canonical_f64_bits(f: f64) -> u64 {
    if f == 0.0 {
        // Catches both -0.0 and +0.0 (they compare equal), normalizing to the
        // +0.0 bit pattern.
        0u64
    } else if f.is_nan() {
        // Canonicalize every NaN payload to one fixed pattern.
        f64::NAN.to_bits()
    } else {
        f.to_bits()
    }
}

/// A 32-byte BLAKE3 content hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash([u8; 32]);

impl Hash {
    /// The raw 32 bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Full lowercase hex (64 characters).
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    /// Short hex prefix (first 8 bytes → 16 characters) for human-facing logs.
    pub fn short(&self) -> String {
        let mut s = String::with_capacity(16);
        for b in &self.0[..8] {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}

impl std::fmt::Display for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Canonical byte encoder feeding a BLAKE3 hasher.
///
/// Every method appends a fixed-width, endianness-fixed encoding of its
/// argument. Callers write each field exactly once, in a fixed, documented
/// order.
pub struct HashWriter {
    hasher: blake3::Hasher,
}

impl HashWriter {
    /// Create an empty writer.
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    /// Encode an `f64` per the module float policy (raw LE bits, `-0.0`→`+0.0`,
    /// canonical NaN).
    pub fn f64(&mut self, f: f64) {
        debug_assert!(
            f.is_finite(),
            "content hash of a non-finite f64 ({f}) — likely a bug in the hashed config"
        );
        self.hasher.update(&canonical_f64_bits(f).to_le_bytes());
    }

    /// Encode a `u64` as fixed-width little-endian bytes.
    pub fn u64(&mut self, v: u64) {
        self.hasher.update(&v.to_le_bytes());
    }

    /// Encode a `usize`, widened to `u64` so the encoding is width-independent.
    pub fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    /// Encode a `bool` as a single `0`/`1` byte.
    pub fn bool(&mut self, b: bool) {
        self.hasher.update(&[b as u8]);
    }

    /// Encode a small enum discriminant / tag byte.
    pub fn tag(&mut self, t: u8) {
        self.hasher.update(&[t]);
    }

    /// Encode a string as a length prefix (`u64` byte length) followed by its
    /// UTF-8 bytes, so concatenation is unambiguous.
    pub fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.hasher.update(s.as_bytes());
    }

    /// Finalize into a [`Hash`].
    pub fn finish(self) -> Hash {
        Hash(*self.hasher.finalize().as_bytes())
    }
}

impl Default for HashWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// A type whose semantic content can be canonically encoded for hashing.
///
/// Implementors write each result-determining field exactly once, in a fixed,
/// documented order. They must NOT write cosmetic fields (print cadence) or run
/// seeds — those are tracked separately from content identity.
pub trait ContentHash {
    /// Append this value's canonical encoding to `w`.
    fn hash_into(&self, w: &mut HashWriter);
}

/// Compute the content hash of `value` under a per-type `domain` tag.
///
/// Writes [`HASH_VERSION`] and `domain` (both length-prefixed) before the
/// value, providing version stability and domain separation so distinct types
/// with coincidentally-identical field bytes still hash differently.
pub fn content_hash<T: ContentHash + ?Sized>(domain: &str, value: &T) -> Hash {
    let mut w = HashWriter::new();
    w.str(HASH_VERSION);
    w.str(domain);
    value.hash_into(&mut w);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes produced by a closure of writes, via a throwaway domain.
    fn hash_writes(f: impl FnOnce(&mut HashWriter)) -> Hash {
        let mut w = HashWriter::new();
        f(&mut w);
        w.finish()
    }

    #[test]
    fn determinism_same_writes_twice() {
        let a = hash_writes(|w| {
            w.f64(1.5);
            w.u64(7);
            w.bool(true);
            w.str("silverstone");
        });
        let b = hash_writes(|w| {
            w.f64(1.5);
            w.u64(7);
            w.bool(true);
            w.str("silverstone");
        });
        assert_eq!(a, b, "identical write sequences must hash identically");
    }

    #[test]
    fn signed_zero_normalized() {
        let pos = hash_writes(|w| w.f64(0.0));
        let neg = hash_writes(|w| w.f64(-0.0));
        assert_eq!(pos, neg, "-0.0 and +0.0 must hash identically");
    }

    #[test]
    fn nan_canonicalized() {
        // Two different NaN bit patterns collapse to one canonical encoding.
        // Tested on the pure normalizer, since the writer's debug_assert
        // (correctly) rejects non-finite values in debug/test builds.
        let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
        let nan_b = f64::from_bits(0x7ff8_0000_0000_abcd);
        assert!(nan_a.is_nan() && nan_b.is_nan());
        assert_ne!(
            nan_a.to_bits(),
            nan_b.to_bits(),
            "test setup: distinct NaNs"
        );
        assert_eq!(
            canonical_f64_bits(nan_a),
            canonical_f64_bits(nan_b),
            "all NaN payloads must canonicalize to one bit pattern"
        );
    }

    #[test]
    fn signed_zero_normalized_bits() {
        // The normalizer maps both signed zeros to the +0.0 pattern.
        assert_eq!(canonical_f64_bits(0.0), 0);
        assert_eq!(canonical_f64_bits(-0.0), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn nan_writer_debug_asserts() {
        // In debug builds the f64 writer surfaces a non-finite value loudly.
        let result = std::panic::catch_unwind(|| {
            let mut w = HashWriter::new();
            w.f64(f64::NAN);
        });
        assert!(
            result.is_err(),
            "w.f64(NaN) must panic under debug_assertions"
        );
    }

    #[test]
    fn adjacent_floats_differ() {
        // 1.0 vs the next representable f64 — proves we hash raw bits, not a
        // rounded decimal rendering that would treat them as equal.
        let one = 1.0f64;
        let next = f64::from_bits(one.to_bits() + 1);
        assert_ne!(one, next);
        let a = hash_writes(|w| w.f64(one));
        let b = hash_writes(|w| w.f64(next));
        assert_ne!(a, b, "adjacent floats must hash differently");
    }

    #[test]
    fn usize_matches_u64_same_value() {
        // usize is widened to u64, so a usize hashes identically to the u64 of
        // the same value (width-independent encoding).
        let via_usize = hash_writes(|w| w.usize(1_234_567));
        let via_u64 = hash_writes(|w| w.u64(1_234_567));
        assert_eq!(via_usize, via_u64, "usize must widen to u64 identically");
    }

    #[test]
    fn hex_forms() {
        let h = hash_writes(|w| w.u64(0));
        assert_eq!(h.to_hex().len(), 64, "full hex is 64 chars");
        assert_eq!(h.short().len(), 16, "short hex is 16 chars");
        assert!(h.to_hex().starts_with(&h.short()));
        assert_eq!(h.to_string(), h.to_hex(), "Display == full hex");
    }

    #[test]
    fn domain_prefix_separates() {
        struct One;
        impl ContentHash for One {
            fn hash_into(&self, w: &mut HashWriter) {
                w.u64(1);
            }
        }
        // Same value bytes, different domain tag → different hash.
        let a = content_hash("domain.a", &One);
        let b = content_hash("domain.b", &One);
        assert_ne!(
            a, b,
            "domain tag must separate otherwise-identical encodings"
        );
    }
}
