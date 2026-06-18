#![deny(unsafe_code)]
//! Apex-14 vehicle-dynamics viewer: opens a native window showing the track map
//! coloured by the QSS speed profile.

use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Apex-14 - Vehicle Dynamics Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "Apex-14",
        options,
        Box::new(|_cc| Ok(Box::new(apex_viewer::app::ApexApp::default()))),
    )
}
