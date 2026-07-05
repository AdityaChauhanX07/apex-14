//! Track-map rendering: world-to-screen transform, speed-coloured racing line,
//! boundaries, and a speed legend, drawn with an `egui::Painter`.

use apex_telemetry::ChannelId;
use eframe::egui;

impl super::app::ApexApp {
    /// Draw the interactive track map (pan with drag, zoom with scroll).
    pub fn draw_track_map(&mut self, ui: &mut egui::Ui) {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());

        let rect = response.rect;

        // Handle pan (drag) and zoom (scroll)
        if response.dragged() {
            self.pan_offset += response.drag_delta();
        }
        if response.hover_pos().is_some() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let zoom_factor = 1.0 + scroll * 0.001;
                self.zoom *= zoom_factor;
                self.zoom = self.zoom.clamp(0.1, 50.0);
            }
        }

        // Background
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(26, 26, 46));

        let track = match &self.track {
            Some(t) => t,
            None => return,
        };

        // Compute world-to-screen transform
        let (min_x, max_x, min_y, max_y) = track_bounds(track);
        let world_width = (max_x - min_x).max(1.0);
        let world_height = (max_y - min_y).max(1.0);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        let scale_x = (rect.width() * 0.85) / world_width as f32;
        let scale_y = (rect.height() * 0.85) / world_height as f32;
        let base_scale = scale_x.min(scale_y);
        let scale = base_scale * self.zoom;

        let screen_center = rect.center().to_vec2() + self.pan_offset;

        let world_to_screen = |wx: f64, wy: f64| -> egui::Pos2 {
            let sx = (wx - center_x) as f32 * scale + screen_center.x;
            let sy = -(wy - center_y) as f32 * scale + screen_center.y; // flip Y
            egui::pos2(sx, sy)
        };

        // Draw track boundaries
        if self.show_boundaries {
            let mut left_points = Vec::new();
            let mut right_points = Vec::new();

            for seg in &track.segments {
                let normal_x = (seg.heading + std::f64::consts::FRAC_PI_2).cos();
                let normal_y = (seg.heading + std::f64::consts::FRAC_PI_2).sin();

                left_points.push(world_to_screen(
                    seg.x + seg.width_left * normal_x,
                    seg.y + seg.width_left * normal_y,
                ));
                right_points.push(world_to_screen(
                    seg.x - seg.width_right * normal_x,
                    seg.y - seg.width_right * normal_y,
                ));
            }

            let boundary_color = egui::Color32::from_rgb(80, 80, 80);
            let boundary_stroke = egui::Stroke::new(1.0, boundary_color);

            // Draw boundary lines
            for i in 0..left_points.len().saturating_sub(1) {
                painter.line_segment([left_points[i], left_points[i + 1]], boundary_stroke);
                painter.line_segment([right_points[i], right_points[i + 1]], boundary_stroke);
            }
            // Close the loop for closed tracks
            if track.is_closed && !left_points.is_empty() {
                painter.line_segment(
                    [*left_points.last().unwrap(), left_points[0]],
                    boundary_stroke,
                );
                painter.line_segment(
                    [*right_points.last().unwrap(), right_points[0]],
                    boundary_stroke,
                );
            }
        }

        // Draw speed-colored racing line
        if self.show_racing_line {
            if let Some(ref speeds) = self.speeds {
                let min_speed = speeds.iter().cloned().fold(f64::MAX, f64::min);
                let max_speed = speeds.iter().cloned().fold(f64::MIN, f64::max);

                for (pair, &speed) in track.segments.windows(2).zip(speeds.iter()) {
                    let p1 = world_to_screen(pair[0].x, pair[0].y);
                    let p2 = world_to_screen(pair[1].x, pair[1].y);
                    let color = speed_color(speed, min_speed, max_speed);
                    painter.line_segment([p1, p2], egui::Stroke::new(2.5, color));
                }
                // Close loop
                if track.is_closed && track.segments.len() > 1 {
                    let last = track.segments.len() - 1;
                    let p1 = world_to_screen(track.segments[last].x, track.segments[last].y);
                    let p2 = world_to_screen(track.segments[0].x, track.segments[0].y);
                    let color = speed_color(speeds[last], min_speed, max_speed);
                    painter.line_segment([p1, p2], egui::Stroke::new(2.5, color));
                }
            }
        }

        // Draw cursor position on the track (synced from telemetry hover).
        if let Some(idx) = self.cursor_index {
            if idx < track.segments.len() {
                let seg = &track.segments[idx];
                let pos = world_to_screen(seg.x, seg.y);
                // White dot with a black outline
                painter.circle_filled(pos, 6.0, egui::Color32::WHITE);
                painter.circle_stroke(pos, 6.0, egui::Stroke::new(2.0, egui::Color32::BLACK));

                // Speed label next to the cursor
                if let Some(ref speeds) = self.speeds {
                    if idx < speeds.len() {
                        painter.text(
                            pos + egui::vec2(12.0, -8.0),
                            egui::Align2::LEFT_CENTER,
                            format!(
                                "{:.0} {}",
                                speeds[idx] * 3.6,
                                ChannelId::SpeedKph.unit().symbol()
                            ),
                            egui::FontId::proportional(13.0),
                            egui::Color32::WHITE,
                        );
                    }
                }
            }
        }

        // Hover detection on the track map -> update the shared cursor.
        if let Some(hover_pos) = response.hover_pos() {
            let mut best_dist = f32::MAX;
            let mut best_idx = 0;
            for (i, seg) in track.segments.iter().enumerate() {
                let screen_pos = world_to_screen(seg.x, seg.y);
                let dist = hover_pos.distance(screen_pos);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            if best_dist < 50.0 {
                // within 50 pixels of the track
                self.cursor_index = Some(best_idx);
                self.cursor_s = Some(track.segments[best_idx].s);
            }
        }

        // Title
        painter.text(
            rect.left_top() + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            format!("Apex-14 - {}", self.track_name),
            egui::FontId::proportional(18.0),
            egui::Color32::WHITE,
        );

        // Speed legend
        if let Some(ref speeds) = self.speeds {
            let min_speed = speeds.iter().cloned().fold(f64::MAX, f64::min);
            let max_speed = speeds.iter().cloned().fold(f64::MIN, f64::max);
            draw_legend(&painter, rect, min_speed, max_speed);
        }
    }
}

/// Axis-aligned world bounds `(min_x, max_x, min_y, max_y)` of the centerline.
fn track_bounds(track: &apex_track::Track) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for seg in &track.segments {
        min_x = min_x.min(seg.x);
        max_x = max_x.max(seg.x);
        min_y = min_y.min(seg.y);
        max_y = max_y.max(seg.y);
    }
    (min_x, max_x, min_y, max_y)
}

/// Map a speed onto a blue→cyan→green→yellow→red gradient.
fn speed_color(speed: f64, min_speed: f64, max_speed: f64) -> egui::Color32 {
    let range = (max_speed - min_speed).max(1e-6);
    let t = ((speed - min_speed) / range).clamp(0.0, 1.0);

    // Blue -> Cyan -> Green -> Yellow -> Red (5-stop gradient)
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0) // blue to cyan
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s) // cyan to green
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0) // green to yellow
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0) // yellow to red
    };

    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Draw the speed-gradient legend bar with min/max labels in the bottom-right.
fn draw_legend(painter: &egui::Painter, rect: egui::Rect, min_speed: f64, max_speed: f64) {
    let legend_width = 150.0;
    let legend_height = 12.0;
    let margin = 15.0;
    let x_start = rect.right() - legend_width - margin;
    let y = rect.bottom() - margin - legend_height - 20.0;

    // Draw gradient bar
    let n_segments = 50;
    let seg_width = legend_width / n_segments as f32;
    for i in 0..n_segments {
        let t = i as f64 / n_segments as f64;
        let speed = min_speed + t * (max_speed - min_speed);
        let color = speed_color(speed, min_speed, max_speed);
        let x = x_start + i as f32 * seg_width;
        painter.rect_filled(
            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(seg_width + 0.5, legend_height)),
            0.0,
            color,
        );
    }

    // Labels
    let text_color = egui::Color32::from_rgb(200, 200, 200);
    let font = egui::FontId::proportional(11.0);
    painter.text(
        egui::pos2(x_start, y + legend_height + 2.0),
        egui::Align2::LEFT_TOP,
        format!(
            "{:.0} {}",
            min_speed * 3.6,
            ChannelId::SpeedKph.unit().symbol()
        ),
        font.clone(),
        text_color,
    );
    painter.text(
        egui::pos2(x_start + legend_width, y + legend_height + 2.0),
        egui::Align2::RIGHT_TOP,
        format!(
            "{:.0} {}",
            max_speed * 3.6,
            ChannelId::SpeedKph.unit().symbol()
        ),
        font,
        text_color,
    );
}
