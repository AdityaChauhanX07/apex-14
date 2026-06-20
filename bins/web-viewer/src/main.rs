#![deny(unsafe_code)]
//! Web-based track viewer and telemetry display for Apex-14.
//!
//! Compiles to WebAssembly for browser deployment, or runs natively.
//! Shows track maps with speed-colored racing lines and telemetry plots.
//!
//! Users pick a built-in circuit (Silverstone, Monza, oval, circle) or paste
//! their own track JSON. A quasi-steady-state lap simulation colours the
//! centreline by speed and drives the speed-profile plot.

use apex_physics::{qss_lap_sim, CarParams};
use apex_track::{
    build_track, circle_track, monza_circuit, oval_track, parse_track_json, silverstone_circuit,
    Track,
};
use eframe::egui;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// A loaded track together with its computed speed profile.
struct TrackData {
    /// The track geometry.
    track: Track,
    /// Speed (m/s) along the centreline, one value per segment pair.
    speeds: Vec<f64>,
    /// Arc-length distance (m) at each point, for the speed-profile plot.
    distances: Vec<f64>,
    /// Lap time (s), if the simulation produced one.
    lap_time: Option<f64>,
}

/// Top-level web viewer application state.
struct WebViewerApp {
    /// Currently selected track name.
    selected_track: String,
    /// Available built-in tracks.
    track_names: Vec<String>,
    /// Current track data (geometry plus speed profile).
    track_data: Option<TrackData>,
    /// Custom JSON input (for pasting track data).
    json_input: String,
    /// Error message for display.
    error_msg: Option<String>,
    /// Map zoom factor.
    zoom: f32,
    /// Map pan offset (screen pixels).
    pan: egui::Vec2,
}

/// Construct a built-in circuit by name, or `None` if the name is unknown.
fn build_builtin(name: &str) -> Option<Track> {
    let (points, closed) = match name {
        "Silverstone" => silverstone_circuit(),
        "Monza" => monza_circuit(),
        "Oval" => oval_track(500.0, 80.0, 12.0, 300),
        "Circle" => circle_track(100.0, 12.0, 200),
        _ => return None,
    };
    Some(build_track(name, &points, closed))
}

/// Run the QSS lap simulation to build the speed-coloured [`TrackData`].
fn make_track_data(track: Track) -> TrackData {
    let params = CarParams::f1_2024_calibrated();
    let qss = qss_lap_sim(&track, &params);
    TrackData {
        track,
        speeds: qss.speeds,
        distances: qss.distances,
        lap_time: Some(qss.lap_time),
    }
}

impl WebViewerApp {
    /// Create the app, loading Silverstone as the default track.
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let track_names = vec![
            "Silverstone".to_string(),
            "Monza".to_string(),
            "Oval".to_string(),
            "Circle".to_string(),
        ];
        let selected_track = "Silverstone".to_string();
        let track_data = build_builtin(&selected_track).map(make_track_data);
        Self {
            selected_track,
            track_names,
            track_data,
            json_input: String::new(),
            error_msg: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }

    /// Reset zoom and pan to the default view.
    fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }

    /// Load a built-in circuit by name and recompute its speed profile.
    fn select_builtin(&mut self, name: String) {
        match build_builtin(&name) {
            Some(track) => {
                self.track_data = Some(make_track_data(track));
                self.error_msg = None;
                self.reset_view();
            }
            None => self.error_msg = Some(format!("Unknown track: {name}")),
        }
        self.selected_track = name;
    }

    /// Parse the pasted JSON into a track and recompute its speed profile.
    fn load_json(&mut self) {
        match parse_track_json(&self.json_input) {
            Ok(track) => {
                self.selected_track = track.name.clone();
                self.track_data = Some(make_track_data(track));
                self.error_msg = None;
                self.reset_view();
            }
            Err(e) => self.error_msg = Some(format!("JSON parse error: {e}")),
        }
    }

    /// Draw the left control panel (track selector, JSON input, results).
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Apex-14 Web Viewer");
        ui.separator();

        ui.label("Built-in track:");
        let mut chosen: Option<String> = None;
        egui::ComboBox::from_id_salt("track_select")
            .selected_text(&self.selected_track)
            .show_ui(ui, |ui| {
                for name in &self.track_names {
                    if ui
                        .selectable_label(&self.selected_track == name, name)
                        .clicked()
                    {
                        chosen = Some(name.clone());
                    }
                }
            });
        if let Some(name) = chosen {
            self.select_builtin(name);
        }

        ui.separator();
        ui.label("Or paste track JSON:");
        ui.add(
            egui::TextEdit::multiline(&mut self.json_input)
                .desired_rows(6)
                .desired_width(f32::INFINITY)
                .hint_text("{ \"name\": ..., \"points\": [...] }"),
        );
        if ui.button("Load JSON").clicked() {
            self.load_json();
        }

        ui.separator();
        if let Some(data) = &self.track_data {
            if let Some(lap) = data.lap_time {
                ui.label(format!("Lap time: {lap:.3} s"));
            }
            if !data.speeds.is_empty() {
                let min_v = data.speeds.iter().cloned().fold(f64::MAX, f64::min);
                let max_v = data.speeds.iter().cloned().fold(f64::MIN, f64::max);
                ui.label(format!(
                    "Speed: {:.0} - {:.0} km/h",
                    min_v * 3.6,
                    max_v * 3.6
                ));
            }
            ui.label(format!("Length: {:.0} m", data.track.total_length));
        }
        if ui.button("Reset view").clicked() {
            self.reset_view();
        }

        if let Some(err) = &self.error_msg {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(255, 120, 120), err);
        }
    }
}

impl eframe::App for WebViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls")
            .min_width(220.0)
            .show(ctx, |ui| self.controls(ui));

        // Speed-profile plot along the bottom.
        if let Some(data) = &self.track_data {
            if !data.speeds.is_empty() {
                egui::TopBottomPanel::bottom("speed_profile")
                    .resizable(false)
                    .min_height(160.0)
                    .show(ctx, |ui| render_speed_plot(ui, data));
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| match &self.track_data {
            Some(data) => render_track(ui, data, &mut self.zoom, &mut self.pan),
            None => {
                ui.centered_and_justified(|ui| {
                    ui.label("No track loaded. Pick a circuit or paste JSON.");
                });
            }
        });
    }
}

/// Map a speed onto a blue -> cyan -> green -> yellow -> red gradient.
fn speed_to_color(speed: f64, min_speed: f64, max_speed: f64) -> egui::Color32 {
    let range = (max_speed - min_speed).max(1e-6);
    let t = ((speed - min_speed) / range).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Draw the track map: a speed-coloured centreline that fits the panel, with
/// drag-to-pan and scroll-to-zoom.
fn render_track(ui: &mut egui::Ui, data: &TrackData, zoom: &mut f32, pan: &mut egui::Vec2) {
    let (response, painter) =
        ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
    let rect = response.rect;

    if response.dragged() {
        *pan += response.drag_delta();
    }
    if response.hover_pos().is_some() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            *zoom = (*zoom * (1.0 + scroll * 0.001)).clamp(0.1, 50.0);
        }
    }

    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(26, 26, 46));

    let segs = &data.track.segments;
    if segs.is_empty() {
        return;
    }

    // World bounds of the centreline.
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for s in segs {
        min_x = min_x.min(s.x);
        max_x = max_x.max(s.x);
        min_y = min_y.min(s.y);
        max_y = max_y.max(s.y);
    }
    let world_w = (max_x - min_x).max(1.0);
    let world_h = (max_y - min_y).max(1.0);
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;

    let base_scale =
        ((rect.width() * 0.85) / world_w as f32).min((rect.height() * 0.85) / world_h as f32);
    let scale = base_scale * *zoom;
    let screen_center = rect.center().to_vec2() + *pan;
    let to_screen = |wx: f64, wy: f64| -> egui::Pos2 {
        egui::pos2(
            (wx - center_x) as f32 * scale + screen_center.x,
            -(wy - center_y) as f32 * scale + screen_center.y,
        )
    };

    // Speed-coloured racing line.
    if !data.speeds.is_empty() {
        let min_v = data.speeds.iter().cloned().fold(f64::MAX, f64::min);
        let max_v = data.speeds.iter().cloned().fold(f64::MIN, f64::max);
        for (pair, &speed) in segs.windows(2).zip(data.speeds.iter()) {
            let p1 = to_screen(pair[0].x, pair[0].y);
            let p2 = to_screen(pair[1].x, pair[1].y);
            painter.line_segment(
                [p1, p2],
                egui::Stroke::new(2.5, speed_to_color(speed, min_v, max_v)),
            );
        }
        if data.track.is_closed && segs.len() > 1 {
            let last = segs.len() - 1;
            let p1 = to_screen(segs[last].x, segs[last].y);
            let p2 = to_screen(segs[0].x, segs[0].y);
            let speed = data.speeds.get(last).copied().unwrap_or(min_v);
            painter.line_segment(
                [p1, p2],
                egui::Stroke::new(2.5, speed_to_color(speed, min_v, max_v)),
            );
        }
        draw_legend(&painter, rect, min_v, max_v);
    }

    painter.text(
        rect.left_top() + egui::vec2(10.0, 10.0),
        egui::Align2::LEFT_TOP,
        format!("Apex-14 - {}", data.track.name),
        egui::FontId::proportional(18.0),
        egui::Color32::WHITE,
    );
}

/// Draw a small speed-gradient legend in the bottom-right of `rect`.
fn draw_legend(painter: &egui::Painter, rect: egui::Rect, min_speed: f64, max_speed: f64) {
    let (width, height, margin) = (150.0_f32, 12.0_f32, 15.0_f32);
    let x_start = rect.right() - width - margin;
    let y = rect.bottom() - margin - height;
    let n = 50;
    let seg_w = width / n as f32;
    for i in 0..n {
        let t = i as f64 / n as f64;
        let speed = min_speed + t * (max_speed - min_speed);
        let x = x_start + i as f32 * seg_w;
        painter.rect_filled(
            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(seg_w + 0.5, height)),
            0.0,
            speed_to_color(speed, min_speed, max_speed),
        );
    }
    let font = egui::FontId::proportional(11.0);
    let color = egui::Color32::from_rgb(200, 200, 200);
    painter.text(
        egui::pos2(x_start, y - 2.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{:.0} km/h", min_speed * 3.6),
        font.clone(),
        color,
    );
    painter.text(
        egui::pos2(x_start + width, y - 2.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("{:.0} km/h", max_speed * 3.6),
        font,
        color,
    );
}

/// Draw the speed-vs-distance profile plot.
fn render_speed_plot(ui: &mut egui::Ui, data: &TrackData) {
    ui.label("Speed (km/h)");
    let points: egui_plot::PlotPoints = data
        .distances
        .iter()
        .zip(data.speeds.iter())
        .map(|(&d, &v)| [d, v * 3.6])
        .collect();
    let line = egui_plot::Line::new(points)
        .color(egui::Color32::from_rgb(0, 200, 255))
        .name("Speed");
    egui_plot::Plot::new("speed_profile_plot")
        .height(120.0)
        .x_axis_label("Distance (m)")
        .allow_drag(false)
        .allow_zoom(false)
        .show(ui, |plot_ui| plot_ui.line(line));
}

/// Native entry point: open a desktop window.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Apex-14 Web Viewer"),
        ..Default::default()
    };
    eframe::run_native(
        "Apex-14 Web Viewer",
        options,
        Box::new(|cc| Ok(Box::new(WebViewerApp::new(cc)))),
    )
}

/// Web entry point: attach eframe to the `the_canvas_id` canvas.
///
/// The `expect` calls here are unavoidable for eframe's web setup: there is no
/// meaningful recovery if the host page is missing its `<canvas>` element, so
/// failing loudly (logged to the browser console) is the intended behaviour.
#[cfg(target_arch = "wasm32")]
fn main() {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();
    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let canvas = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("the_canvas_id"))
            .expect("Cannot find canvas element 'the_canvas_id'")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("'the_canvas_id' is not a canvas element");

        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(WebViewerApp::new(cc)))),
            )
            .await
            .expect("Failed to start eframe web runner");
    });
}
