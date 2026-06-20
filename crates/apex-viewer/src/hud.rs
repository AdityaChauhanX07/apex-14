//! Real-time heads-up display for the simulation server.
//!
//! Connects to the sim-server's UDP telemetry stream and renders
//! live vehicle data: speed, g-forces, gear, and lap time.

use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::thread;

use eframe::egui;

use apex_sim::protocol::OutputPacket;

/// Shared state between the UDP receiver thread and the UI.
#[derive(Default)]
pub struct HudState {
    /// Latest received output packet.
    pub latest: Option<OutputPacket>,
    /// Whether the receiver is connected (has received at least one packet).
    pub connected: bool,
    /// Number of packets received.
    pub packet_count: u64,
    /// Receiver error message, if any.
    pub error: Option<String>,
}

/// Start the UDP receiver thread.
///
/// Binds `listen_addr` and spawns a background thread that decodes incoming
/// [`OutputPacket`]s into the returned shared [`HudState`]. The thread uses a
/// short read timeout so it never blocks indefinitely. Returns a shared handle
/// to the HUD state that the UI can poll.
pub fn start_receiver(listen_addr: &str) -> Arc<Mutex<HudState>> {
    let state = Arc::new(Mutex::new(HudState::default()));
    let state_clone = Arc::clone(&state);
    let addr = listen_addr.to_string();

    thread::spawn(move || {
        let socket = match UdpSocket::bind(&addr) {
            Ok(s) => s,
            Err(e) => {
                if let Ok(mut s) = state_clone.lock() {
                    s.error = Some(format!("Failed to bind {}: {}", addr, e));
                }
                return;
            }
        };
        // Set a timeout so the thread doesn't block forever
        // (allows clean shutdown in the future).
        let _ = socket.set_read_timeout(Some(std::time::Duration::from_millis(100)));

        let mut buf = [0u8; 256];
        loop {
            match socket.recv_from(&mut buf) {
                Ok((n, _addr)) => {
                    if let Some(packet) = OutputPacket::from_bytes(&buf[..n]) {
                        if let Ok(mut s) = state_clone.lock() {
                            s.latest = Some(packet);
                            s.connected = true;
                            s.packet_count += 1;
                        }
                    }
                }
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    // Normal timeout, continue.
                }
                Err(e) => {
                    if let Ok(mut s) = state_clone.lock() {
                        s.error = Some(format!("Recv error: {}", e));
                    }
                    break;
                }
            }
        }
    });

    state
}

/// Render the HUD overlay in an egui panel.
///
/// Call this from the eframe App's update method. Shows a "waiting" message
/// until the first telemetry packet arrives, so the HUD does not require the
/// sim-server to be running.
pub fn render_hud(ui: &mut egui::Ui, state: &Arc<Mutex<HudState>>) {
    let (data, error) = match state.lock() {
        Ok(s) => (s.latest, s.error.clone()),
        Err(_) => (None, None),
    };

    if let Some(msg) = error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), msg);
        ui.add_space(10.0);
    }

    match data {
        None => {
            ui.heading("Waiting for telemetry...");
            ui.label("Start sim-server and send input to begin.");
        }
        Some(packet) => {
            render_speedometer(ui, &packet);
            ui.add_space(10.0);
            render_gforce(ui, &packet);
            ui.add_space(10.0);
            render_gear_indicator(ui, &packet);
            ui.add_space(10.0);
            render_wheel_speeds(ui, &packet);
            ui.add_space(10.0);
            render_timing(ui, &packet);
        }
    }
}

/// Large digital speedometer showing km/h.
fn render_speedometer(ui: &mut egui::Ui, packet: &OutputPacket) {
    let speed_kmh = packet.speed * 3.6;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("{:.0}", speed_kmh))
                .size(64.0)
                .strong(),
        );
        ui.label(egui::RichText::new("km/h").size(20.0));
    });
}

/// G-force display showing lateral vs longitudinal on a circle.
///
/// Renders a filled circle with a dot showing the current g-force vector.
/// The circle represents the grip circle (typically 4-5g for F1).
fn render_gforce(ui: &mut egui::Ui, packet: &OutputPacket) {
    let max_g = 5.0; // grip circle radius in g
    let size = 120.0; // widget size in pixels

    // Reserve space and get the painter.
    let (response, painter) = ui.allocate_painter(egui::vec2(size, size), egui::Sense::hover());
    let center = response.rect.center();
    let radius = size / 2.0 - 5.0;

    // Draw the grip circle.
    painter.circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.0, ui.visuals().text_color().gamma_multiply(0.3)),
    );

    // Cross-hairs.
    painter.line_segment(
        [
            center - egui::vec2(radius, 0.0),
            center + egui::vec2(radius, 0.0),
        ],
        egui::Stroke::new(0.5, ui.visuals().text_color().gamma_multiply(0.15)),
    );
    painter.line_segment(
        [
            center - egui::vec2(0.0, radius),
            center + egui::vec2(0.0, radius),
        ],
        egui::Stroke::new(0.5, ui.visuals().text_color().gamma_multiply(0.15)),
    );

    // G-force dot.
    // lateral = right positive, longitudinal = forward positive.
    // On screen: x = lateral (right), y = -longitudinal (up = forward).
    let gx = (packet.accel_lat / max_g).clamp(-1.0, 1.0) as f32 * radius;
    let gy = -(packet.accel_long / max_g).clamp(-1.0, 1.0) as f32 * radius;
    let dot_pos = center + egui::vec2(gx, gy);

    painter.circle_filled(dot_pos, 5.0, egui::Color32::from_rgb(255, 80, 80));

    // Labels.
    ui.horizontal(|ui| {
        ui.label(format!("Lat: {:.1}g", packet.accel_lat));
        ui.label(format!("Long: {:.1}g", packet.accel_long));
    });
}

/// Gear indicator (large centered number).
fn render_gear_indicator(ui: &mut egui::Ui, packet: &OutputPacket) {
    let gear_text = if packet.gear == 0 {
        "N".to_string()
    } else {
        packet.gear.to_string()
    };
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("GEAR").size(14.0));
        ui.label(egui::RichText::new(gear_text).size(48.0).strong());
    });
}

/// Wheel speed display (four-corner layout).
fn render_wheel_speeds(ui: &mut egui::Ui, packet: &OutputPacket) {
    ui.label(egui::RichText::new("Wheel Speed (rad/s)").size(14.0));
    egui::Grid::new("wheel_speeds").show(ui, |ui| {
        ui.label(format!("FL: {:.0}", packet.wheel_fl));
        ui.label(format!("FR: {:.0}", packet.wheel_fr));
        ui.end_row();
        ui.label(format!("RL: {:.0}", packet.wheel_rl));
        ui.label(format!("RR: {:.0}", packet.wheel_rr));
        ui.end_row();
    });
}

/// Lap time and simulation time.
fn render_timing(ui: &mut egui::Ui, packet: &OutputPacket) {
    let lap_min = (packet.lap_time / 60.0) as u32;
    let lap_sec = packet.lap_time % 60.0;
    ui.horizontal(|ui| {
        ui.label(format!("Lap {}", packet.lap));
        ui.label(
            egui::RichText::new(format!("{}:{:05.2}", lap_min, lap_sec))
                .size(24.0)
                .strong(),
        );
    });
    ui.label(format!("Sim time: {:.1}s", packet.sim_time));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hud_state_default() {
        let s = HudState::default();
        assert!(s.latest.is_none());
        assert!(!s.connected);
        assert_eq!(s.packet_count, 0);
        assert!(s.error.is_none());
    }

    #[test]
    fn test_speed_conversion() {
        // 83.333... m/s should be 300 km/h.
        let speed_ms: f64 = 300.0 / 3.6;
        let speed_kmh = speed_ms * 3.6;
        assert!((speed_kmh - 300.0).abs() < 1e-9);
    }
}
