//! The main viewer application: a left control panel plus a central track map.

use eframe::egui;

/// Top-level viewer application state.
pub struct ApexApp {
    // Track data
    pub(crate) track: Option<apex_track::Track>,
    pub(crate) track_name: String,

    // Simulation results
    pub(crate) speeds: Option<Vec<f64>>,
    pub(crate) lap_time: Option<f64>,

    // View state
    pub(crate) zoom: f32,
    pub(crate) pan_offset: egui::Vec2,

    // UI state
    pub(crate) selected_track: usize,
    pub(crate) show_boundaries: bool,
    pub(crate) show_racing_line: bool,
}

impl Default for ApexApp {
    fn default() -> Self {
        // Load the oval track by default
        let (points, closed) = apex_track::oval_track(500.0, 80.0, 12.0, 300);
        let track = apex_track::build_track("Oval", &points, closed);
        let params = apex_physics::CarParams::f1_2024_calibrated();
        let qss = apex_physics::qss_lap_sim(&track, &params);

        ApexApp {
            speeds: Some(qss.speeds.clone()),
            lap_time: Some(qss.lap_time),
            track: Some(track),
            track_name: "Oval (500m, R=80m)".to_string(),
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            selected_track: 0,
            show_boundaries: true,
            show_racing_line: true,
        }
    }
}

impl eframe::App for ApexApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Left panel: controls
        egui::SidePanel::left("controls")
            .min_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Apex-14");
                ui.separator();

                // Track selection
                ui.label("Track:");
                let tracks = [
                    "Oval (500m, R=80m)",
                    "Circle (R=100m)",
                    "Silverstone",
                    "Monza",
                ];
                egui::ComboBox::from_id_salt("track_select")
                    .selected_text(&self.track_name)
                    .show_ui(ui, |ui| {
                        for (i, name) in tracks.iter().enumerate() {
                            if ui
                                .selectable_value(&mut self.selected_track, i, *name)
                                .changed()
                            {
                                self.load_track(i);
                            }
                        }
                    });

                ui.separator();

                // Display options
                ui.checkbox(&mut self.show_boundaries, "Show boundaries");
                ui.checkbox(&mut self.show_racing_line, "Show racing line");

                ui.separator();

                // Results
                if let Some(lap_time) = self.lap_time {
                    ui.label(format!("Lap time: {:.3}s", lap_time));
                }
                if let Some(ref speeds) = self.speeds {
                    let max_speed = speeds.iter().cloned().fold(f64::MIN, f64::max);
                    let min_speed = speeds.iter().cloned().fold(f64::MAX, f64::min);
                    ui.label(format!(
                        "Speed: {:.0} - {:.0} km/h",
                        min_speed * 3.6,
                        max_speed * 3.6
                    ));
                }

                ui.separator();

                // Zoom controls
                ui.label(format!("Zoom: {:.1}x", self.zoom));
                if ui.button("Reset view").clicked() {
                    self.zoom = 1.0;
                    self.pan_offset = egui::Vec2::ZERO;
                }
            });

        // Central panel: track map
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_track_map(ui);
        });
    }
}

impl ApexApp {
    /// Switch to track `index`, recompute the QSS speed profile, and reset view.
    fn load_track(&mut self, index: usize) {
        let (track, name) = match index {
            0 => {
                let (pts, closed) = apex_track::oval_track(500.0, 80.0, 12.0, 300);
                (
                    apex_track::build_track("Oval", &pts, closed),
                    "Oval (500m, R=80m)",
                )
            }
            1 => {
                let (pts, closed) = apex_track::circle_track(100.0, 12.0, 200);
                (
                    apex_track::build_track("Circle", &pts, closed),
                    "Circle (R=100m)",
                )
            }
            2 => {
                let (pts, closed) = apex_track::silverstone_circuit();
                (
                    apex_track::build_track("Silverstone", &pts, closed),
                    "Silverstone",
                )
            }
            3 => {
                let (pts, closed) = apex_track::monza_circuit();
                (apex_track::build_track("Monza", &pts, closed), "Monza")
            }
            _ => return,
        };

        let params = apex_physics::CarParams::f1_2024_calibrated();
        let qss = apex_physics::qss_lap_sim(&track, &params);

        self.speeds = Some(qss.speeds.clone());
        self.lap_time = Some(qss.lap_time);
        self.track = Some(track);
        self.track_name = name.to_string();
        self.selected_track = index;
        self.zoom = 1.0;
        self.pan_offset = egui::Vec2::ZERO;
    }
}
