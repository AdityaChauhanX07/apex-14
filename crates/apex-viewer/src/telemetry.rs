//! Telemetry time-series plots (speed, lateral g, longitudinal g, curvature)
//! drawn with `egui_plot`, sharing a cursor position with the track map.

use eframe::egui;

impl super::app::ApexApp {
    /// Draw the stacked telemetry plots below the track map. Hovering the speed
    /// plot moves the shared cursor (mirrored on the track map and as a dashed
    /// vertical line on every plot).
    pub fn draw_telemetry(&mut self, ui: &mut egui::Ui) {
        // Clone the per-point series into locals so the plot closures — which
        // also update the shared cursor — never borrow `self`. The arrays are a
        // few hundred points, so this per-frame copy is negligible.
        let distances = match &self.distances {
            Some(d) => d.clone(),
            None => return,
        };
        let speeds = self.speeds.clone();
        let lateral_gs = self.lateral_gs.clone();
        let longitudinal_gs = self.longitudinal_gs.clone();
        let curvatures = self.curvatures.clone();
        let cursor_s = self.cursor_s;

        let available_width = ui.available_width();
        let plot_height = 120.0_f32; // height per plot

        // Cursor updates are accumulated into locals and written back afterwards.
        let mut new_cursor_s = cursor_s;
        let mut new_cursor_index = self.cursor_index;

        // Dashed vertical cursor line at arc length `s`, shared by every plot.
        let cursor_vline = |s: f64| {
            egui_plot::VLine::new(s)
                .color(egui::Color32::WHITE)
                .style(egui_plot::LineStyle::dashed_dense())
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Plot 1: Speed vs Distance (this plot drives the cursor on hover).
            if let Some(ref speeds) = speeds {
                ui.label("Speed (km/h)");
                let speed_points: egui_plot::PlotPoints = distances
                    .iter()
                    .zip(speeds.iter())
                    .map(|(&d, &v)| [d, v * 3.6])
                    .collect();
                let speed_line = egui_plot::Line::new(speed_points)
                    .color(egui::Color32::from_rgb(0, 200, 255))
                    .name("Speed");

                egui_plot::Plot::new("speed_plot")
                    .height(plot_height)
                    .width(available_width)
                    .x_axis_label("Distance (m)")
                    .show_axes(true)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(speed_line);
                        // Draw cursor line if active
                        if let Some(s) = cursor_s {
                            plot_ui.vline(cursor_vline(s));
                        }
                        // Detect hover on this plot to update the cursor
                        if let Some(pos) = plot_ui.pointer_coordinate() {
                            new_cursor_s = Some(pos.x);
                            // Find nearest index
                            new_cursor_index = distances
                                .iter()
                                .enumerate()
                                .min_by(|(_, a), (_, b)| {
                                    (**a - pos.x)
                                        .abs()
                                        .partial_cmp(&(**b - pos.x).abs())
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, _)| i);
                        }
                    });
            }

            // Plot 2: Lateral G vs Distance
            if let Some(ref lat_gs) = lateral_gs {
                ui.label("Lateral G");
                let lat_points: egui_plot::PlotPoints = distances
                    .iter()
                    .zip(lat_gs.iter())
                    .map(|(&d, &g)| [d, g])
                    .collect();
                let lat_line = egui_plot::Line::new(lat_points)
                    .color(egui::Color32::from_rgb(255, 100, 100))
                    .name("Lat G");

                egui_plot::Plot::new("lat_g_plot")
                    .height(plot_height)
                    .width(available_width)
                    .show_axes(true)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(lat_line);
                        if let Some(s) = cursor_s {
                            plot_ui.vline(cursor_vline(s));
                        }
                    });
            }

            // Plot 3: Longitudinal G vs Distance
            if let Some(ref lon_gs) = longitudinal_gs {
                ui.label("Longitudinal G");
                let lon_points: egui_plot::PlotPoints = distances
                    .iter()
                    .zip(lon_gs.iter())
                    .map(|(&d, &g)| [d, g])
                    .collect();
                let lon_line = egui_plot::Line::new(lon_points)
                    .color(egui::Color32::from_rgb(100, 255, 100))
                    .name("Lon G");

                egui_plot::Plot::new("lon_g_plot")
                    .height(plot_height)
                    .width(available_width)
                    .show_axes(true)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(lon_line);
                        if let Some(s) = cursor_s {
                            plot_ui.vline(cursor_vline(s));
                        }
                    });
            }

            // Plot 4: Curvature vs Distance
            if let Some(ref curvs) = curvatures {
                ui.label("Curvature (1/m)");
                let curv_points: egui_plot::PlotPoints = distances
                    .iter()
                    .zip(curvs.iter())
                    .map(|(&d, &k)| [d, k])
                    .collect();
                let curv_line = egui_plot::Line::new(curv_points)
                    .color(egui::Color32::from_rgb(255, 200, 50))
                    .name("Curvature");

                egui_plot::Plot::new("curvature_plot")
                    .height(plot_height)
                    .width(available_width)
                    .show_axes(true)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(curv_line);
                        if let Some(s) = cursor_s {
                            plot_ui.vline(cursor_vline(s));
                        }
                    });
            }
        });

        self.cursor_s = new_cursor_s;
        self.cursor_index = new_cursor_index;
    }
}
