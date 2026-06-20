#![deny(unsafe_code)]
//! Real-time simulation HUD.
//!
//! Displays live vehicle telemetry from the sim-server.

use std::sync::{Arc, Mutex};

use apex_viewer::hud::{render_hud, start_receiver, HudState};
use clap::Parser;
use eframe::egui;

/// Real-time simulation heads-up display.
#[derive(Parser)]
#[command(name = "sim-hud")]
struct Args {
    /// UDP address to listen for telemetry packets.
    #[arg(long, default_value = "0.0.0.0:20778")]
    listen: String,
}

/// HUD application: renders the shared telemetry state each frame.
struct HudApp {
    hud_state: Arc<Mutex<HudState>>,
}

impl eframe::App for HudApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_hud(ui, &self.hud_state);
        });
        // Request repaint at ~60fps to keep the display updating.
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let hud_state = start_receiver(&args.listen);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 600.0])
            .with_title("Apex-14 - Simulation HUD"),
        ..Default::default()
    };

    eframe::run_native(
        "Apex-14 HUD",
        options,
        Box::new(move |_cc| Ok(Box::new(HudApp { hud_state }))),
    )
}
