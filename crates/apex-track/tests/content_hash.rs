//! Content-hash correctness + stability tests for tracks (raw + processed).

use apex_track::{build_track, processed_track_hash, raw_track_hash, Track, TrackPoint};

/// A small fixed raw centerline (a 4-point square-ish loop) for stable vectors.
fn fixed_points() -> Vec<TrackPoint> {
    vec![
        TrackPoint {
            x: 0.0,
            y: 0.0,
            width_left: 5.0,
            width_right: 5.0,
        },
        TrackPoint {
            x: 100.0,
            y: 0.0,
            width_left: 5.0,
            width_right: 4.0,
        },
        TrackPoint {
            x: 100.0,
            y: 80.0,
            width_left: 6.0,
            width_right: 5.0,
        },
        TrackPoint {
            x: 0.0,
            y: 80.0,
            width_left: 5.0,
            width_right: 5.0,
        },
    ]
}

fn fixed_track() -> Track {
    build_track("fixed-square", &fixed_points(), true)
}

#[test]
fn determinism_both_hashes() {
    let pts = fixed_points();
    let t = fixed_track();
    assert_eq!(raw_track_hash(&pts), raw_track_hash(&pts));
    assert_eq!(processed_track_hash(&t), processed_track_hash(&t));
}

/// Renaming a track does not change its processed hash (name is excluded).
#[test]
fn processed_hash_excludes_name() {
    let t = fixed_track();
    let mut renamed = t.clone();
    renamed.name = "totally-different-name".to_string();
    assert_eq!(
        processed_track_hash(&t),
        processed_track_hash(&renamed),
        "processed hash must exclude the track name"
    );
}

/// The processed hash covers `is_closed` and the segment geometry.
#[test]
fn processed_hash_sensitive_to_geometry_and_closed() {
    let t = fixed_track();
    let base = processed_track_hash(&t);

    // Flip is_closed.
    let mut open = t.clone();
    open.is_closed = !open.is_closed;
    assert_ne!(
        base,
        processed_track_hash(&open),
        "is_closed must be hashed"
    );

    // Perturb one coordinate of one segment.
    let mut moved = t.clone();
    moved.segments[1].x += 1.0;
    assert_ne!(
        base,
        processed_track_hash(&moved),
        "segment x must be hashed"
    );

    // Perturb one curvature.
    let mut curved = t.clone();
    curved.segments[2].curvature += 1e-6;
    assert_ne!(
        base,
        processed_track_hash(&curved),
        "curvature must be hashed"
    );
}

/// Changing any single raw point changes the raw hash.
#[test]
fn raw_hash_sensitive_to_each_point_field() {
    let pts = fixed_points();
    let base = raw_track_hash(&pts);

    for i in 0..pts.len() {
        for field in 0..4 {
            let mut p = pts.clone();
            match field {
                0 => p[i].x += 1.0,
                1 => p[i].y += 1.0,
                2 => p[i].width_left += 1.0,
                _ => p[i].width_right += 1.0,
            }
            assert_ne!(
                base,
                raw_track_hash(&p),
                "raw point {i} field {field} not hashed"
            );
        }
    }
}

/// Raw and processed hashes of the same track are different objects and never
/// collide (distinct domain tags, distinct content).
#[test]
fn raw_and_processed_do_not_collide() {
    let pts = fixed_points();
    let t = fixed_track();
    assert_ne!(
        raw_track_hash(&pts).to_hex(),
        processed_track_hash(&t).to_hex()
    );
}

/// FROZEN known-answer vectors for the fixed track (raw + processed).
/// Any accidental encoding change flips these and fails CI.
#[test]
fn frozen_vectors() {
    assert_eq!(
        raw_track_hash(&fixed_points()).to_hex(),
        "50f31b5608389a4cee0644785264d97e4d4434f9d9976483ccad7bb181eda260",
        "raw track frozen vector"
    );
    assert_eq!(
        processed_track_hash(&fixed_track()).to_hex(),
        "5b62f484ec1e4206446061a5657c09e743329c3041f6e4a76eaddff59b55d7bb",
        "processed track frozen vector"
    );
}
