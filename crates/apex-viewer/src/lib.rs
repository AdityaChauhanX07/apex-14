#![deny(unsafe_code)]
//! Interactive viewer for Apex-14: renders a track map coloured by the QSS speed
//! profile, with pan/zoom, on top of `eframe`/`egui`.

pub mod app;
pub mod hud;
pub mod telemetry;
pub mod track_view;
