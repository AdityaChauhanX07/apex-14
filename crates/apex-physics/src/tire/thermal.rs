//! Tire thermal and wear model.
//!
//! Tracks surface and carcass temperature (a two-mass thermal model), maps
//! surface temperature onto a grip multiplier, and accumulates wear that further
//! reduces grip.

/// Thermal properties for a single tire.
#[derive(Debug, Clone, Copy)]
pub struct TireThermalParams {
    /// Optimal surface temperature for peak grip (deg C).
    pub optimal_temp: f64,
    /// Width of the grip window - grip drops off outside this range (deg C).
    /// At optimal_temp +/- grip_window, grip is reduced by ~10%.
    pub grip_window: f64,
    /// Cold grip penalty factor. Higher = more grip loss when cold.
    /// mu_factor = 1 - cold_sensitivity * ((T_opt - T) / grip_window)^2 for T < T_opt
    pub cold_sensitivity: f64,
    /// Overheat grip penalty factor. Higher = more grip loss when hot.
    pub hot_sensitivity: f64,
    /// Ambient/starting temperature (deg C).
    pub ambient_temp: f64,
    /// Heat generation coefficient: Q_gen = k_gen * |F_tire| * |v_slip| (W)
    /// Converts slip power into heat.
    pub heat_generation: f64,
    /// Convective cooling coefficient: Q_cool = k_cool * v_air * (T_surface - T_ambient) (W)
    /// Higher speed = more cooling.
    pub convective_cooling: f64,
    /// Surface-to-carcass heat transfer coefficient (W/K).
    pub surface_carcass_coupling: f64,
    /// Carcass radiation/conduction cooling coefficient (W/K).
    pub carcass_cooling: f64,
    /// Thermal mass of the surface layer (J/K). Small = heats up fast.
    pub surface_thermal_mass: f64,
    /// Thermal mass of the carcass (J/K). Large = slow to heat, slow to cool.
    pub carcass_thermal_mass: f64,
}

impl TireThermalParams {
    /// Representative F1 soft compound tire thermal parameters.
    pub fn f1_soft() -> Self {
        TireThermalParams {
            optimal_temp: 100.0,
            grip_window: 15.0,
            cold_sensitivity: 0.20,
            hot_sensitivity: 0.15,
            ambient_temp: 25.0,
            heat_generation: 0.5, // ~50% of slip power becomes heat
            convective_cooling: 2.5,
            surface_carcass_coupling: 50.0,
            carcass_cooling: 5.0,
            surface_thermal_mass: 500.0,
            carcass_thermal_mass: 5000.0,
        }
    }

    /// Representative F1 medium compound - wider operating window, slower to heat.
    pub fn f1_medium() -> Self {
        TireThermalParams {
            optimal_temp: 95.0,
            grip_window: 20.0,
            cold_sensitivity: 0.18,
            hot_sensitivity: 0.12,
            ambient_temp: 25.0,
            heat_generation: 0.45, // ~45% of slip power becomes heat
            convective_cooling: 2.5,
            surface_carcass_coupling: 45.0,
            carcass_cooling: 5.0,
            surface_thermal_mass: 550.0,
            carcass_thermal_mass: 5500.0,
        }
    }

    /// Representative F1 hard compound - widest window, hardest to get into range.
    pub fn f1_hard() -> Self {
        TireThermalParams {
            optimal_temp: 90.0,
            grip_window: 25.0,
            cold_sensitivity: 0.15,
            hot_sensitivity: 0.10,
            ambient_temp: 25.0,
            heat_generation: 0.40, // ~40% of slip power becomes heat
            convective_cooling: 2.5,
            surface_carcass_coupling: 40.0,
            carcass_cooling: 5.0,
            surface_thermal_mass: 600.0,
            carcass_thermal_mass: 6000.0,
        }
    }

    /// Compute the grip multiplier based on surface temperature.
    ///
    /// Returns a value in [0, 1] where 1.0 = full grip at optimal temperature.
    /// Drops off quadratically on both sides of the optimal temperature.
    pub fn grip_factor(&self, surface_temp: f64) -> f64 {
        let delta = surface_temp - self.optimal_temp;
        let normalized = delta / self.grip_window;

        let penalty = if delta < 0.0 {
            // Cold: grip drops off
            self.cold_sensitivity * normalized * normalized
        } else {
            // Hot: grip drops off (potentially faster)
            self.hot_sensitivity * normalized * normalized
        };

        (1.0 - penalty).clamp(0.0, 1.0)
    }
}

/// Thermal state of a single tire (surface and carcass temperatures).
#[derive(Debug, Clone, Copy)]
pub struct TireThermalState {
    /// Surface temperature (deg C).
    pub surface_temp: f64,
    /// Carcass (bulk) temperature (deg C).
    pub carcass_temp: f64,
    /// Cumulative wear (dimensionless, 0.0 = new, 1.0 = worn out).
    pub wear: f64,
}

impl TireThermalState {
    /// Create a new tire at ambient temperature with zero wear.
    pub fn new(ambient_temp: f64) -> Self {
        TireThermalState {
            surface_temp: ambient_temp,
            carcass_temp: ambient_temp,
            wear: 0.0,
        }
    }

    /// Update the thermal state for one time step.
    ///
    /// Arguments:
    /// - params: thermal parameters for this tire
    /// - tire_force_magnitude: total force magnitude |F| at the contact patch (N)
    /// - slip_speed: magnitude of the slip velocity at the contact patch (m/s)
    /// - vehicle_speed: forward speed for convective cooling (m/s)
    /// - dt: time step (s)
    pub fn update(
        &mut self,
        params: &TireThermalParams,
        tire_force_magnitude: f64,
        slip_speed: f64,
        vehicle_speed: f64,
        dt: f64,
    ) {
        // Heat generation from tire slip (friction work)
        let q_gen = params.heat_generation * tire_force_magnitude * slip_speed;

        // Convective cooling (proportional to airspeed and temperature difference)
        let q_conv = params.convective_cooling
            * vehicle_speed.max(2.0)
            * (self.surface_temp - params.ambient_temp);

        // Surface-to-carcass heat transfer
        let q_coupling =
            params.surface_carcass_coupling * (self.surface_temp - self.carcass_temp);

        // Carcass cooling (radiation + conduction to wheel/hub)
        let q_carcass_cool = params.carcass_cooling * (self.carcass_temp - params.ambient_temp);

        // Temperature derivatives
        let d_surface = (q_gen - q_conv - q_coupling) / params.surface_thermal_mass;
        let d_carcass = (q_coupling - q_carcass_cool) / params.carcass_thermal_mass;

        // Euler integration (thermal dynamics are slow, Euler is fine)
        self.surface_temp += d_surface * dt;
        self.carcass_temp += d_carcass * dt;

        // Clamp to physical range (can't go below ambient in this simple model)
        self.surface_temp = self.surface_temp.max(params.ambient_temp);
        self.carcass_temp = self.carcass_temp.max(params.ambient_temp);
    }

    /// Update tire wear for one time step.
    ///
    /// Wear rate increases with tire force, slip speed, and temperature.
    /// Hotter tires wear faster (thermal degradation).
    pub fn update_wear(
        &mut self,
        params: &TireThermalParams,
        tire_force_magnitude: f64,
        slip_speed: f64,
        fz: f64,
        dt: f64,
    ) {
        if fz < 100.0 {
            return;
        } // wheel off ground, no wear

        // Base wear rate proportional to friction power per unit load
        let wear_rate = 1e-4 * tire_force_magnitude * slip_speed / fz;

        // Temperature accelerates wear (hotter = faster degradation)
        let temp_factor = if self.surface_temp > params.optimal_temp {
            1.0 + 0.03 * (self.surface_temp - params.optimal_temp)
        } else {
            1.0
        };

        self.wear += wear_rate * temp_factor * dt;
        self.wear = self.wear.clamp(0.0, 1.0);
    }

    /// Combined grip factor from temperature and wear.
    ///
    /// Total grip multiplier = temperature_factor * wear_factor.
    /// Wear reduces grip linearly: at wear=0.5, grip is reduced by ~15%.
    pub fn grip_multiplier(&self, params: &TireThermalParams) -> f64 {
        let temp_factor = params.grip_factor(self.surface_temp);
        let wear_factor = 1.0 - 0.30 * self.wear; // 30% grip loss at full wear
        temp_factor * wear_factor
    }
}

/// Thermal state for all four tires.
#[derive(Debug, Clone)]
pub struct TireSetThermal {
    pub params_fl: TireThermalParams,
    pub params_fr: TireThermalParams,
    pub params_rl: TireThermalParams,
    pub params_rr: TireThermalParams,
    pub state_fl: TireThermalState,
    pub state_fr: TireThermalState,
    pub state_rl: TireThermalState,
    pub state_rr: TireThermalState,
}

impl TireSetThermal {
    /// Create a new tire set with all tires on the same compound.
    pub fn new(params: TireThermalParams) -> Self {
        let state = TireThermalState::new(params.ambient_temp);
        TireSetThermal {
            params_fl: params,
            params_fr: params,
            params_rl: params,
            params_rr: params,
            state_fl: state,
            state_fr: state,
            state_rl: state,
            state_rr: state,
        }
    }

    /// Get grip multipliers for all four tires [fl, fr, rl, rr].
    pub fn grip_multipliers(&self) -> [f64; 4] {
        [
            self.state_fl.grip_multiplier(&self.params_fl),
            self.state_fr.grip_multiplier(&self.params_fr),
            self.state_rl.grip_multiplier(&self.params_rl),
            self.state_rr.grip_multiplier(&self.params_rr),
        ]
    }

    /// Get surface temperatures for all four tires [fl, fr, rl, rr] (deg C).
    pub fn surface_temps(&self) -> [f64; 4] {
        [
            self.state_fl.surface_temp,
            self.state_fr.surface_temp,
            self.state_rl.surface_temp,
            self.state_rr.surface_temp,
        ]
    }

    /// Get wear levels for all four tires [fl, fr, rl, rr] (0.0-1.0).
    pub fn wear_levels(&self) -> [f64; 4] {
        [
            self.state_fl.wear,
            self.state_fr.wear,
            self.state_rl.wear,
            self.state_rr.wear,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Integrate the thermal state for `secs` seconds at a fixed operating point.
    fn run_thermal(
        state: &mut TireThermalState,
        p: &TireThermalParams,
        force: f64,
        slip: f64,
        vehicle_speed: f64,
        secs: f64,
    ) {
        let dt = 0.01;
        let steps = (secs / dt).round() as usize;
        for _ in 0..steps {
            state.update(p, force, slip, vehicle_speed, dt);
        }
    }

    // (a) Grip factor at optimal temperature.
    #[test]
    fn grip_factor_peaks_at_optimal() {
        let p = TireThermalParams::f1_soft();
        assert_eq!(p.grip_factor(p.optimal_temp), 1.0);
    }

    // (b) Grip factor symmetry / asymmetry.
    #[test]
    fn grip_factor_drops_off_both_sides_asymmetrically() {
        let p = TireThermalParams::f1_soft();
        let hot = p.grip_factor(p.optimal_temp + 10.0);
        let cold = p.grip_factor(p.optimal_temp - 10.0);
        assert!(hot < 1.0, "hot grip {hot}");
        assert!(cold < 1.0, "cold grip {cold}");
        // Cold sensitivity (0.20) exceeds hot sensitivity (0.15): colder loses more.
        assert!(
            cold < hot,
            "cold penalty {cold} should exceed hot penalty {hot}"
        );
    }

    // (c) Grip factor extremes and bounds.
    #[test]
    fn grip_factor_extremes_clamped() {
        let p = TireThermalParams::f1_soft();
        assert!(p.grip_factor(0.0) < 0.5, "very cold grip not reduced");
        assert!(p.grip_factor(200.0) < 1.0, "very hot grip not reduced");

        // Never leaves [0, 1] across a wide sweep.
        let mut t = -50.0;
        while t <= 300.0 {
            let g = p.grip_factor(t);
            assert!((0.0..=1.0).contains(&g), "grip {g} out of [0,1] at {t} C");
            t += 1.0;
        }
    }

    // (d) Thermal warmup: 10 s of hard driving pushes the surface above 50 C,
    // and the light surface layer heats faster than the heavy carcass.
    #[test]
    fn thermal_warmup_surface_leads_carcass() {
        let p = TireThermalParams::f1_soft();
        let mut s = TireThermalState::new(p.ambient_temp);
        // Hard driving: ~5 kN tire load, 4 m/s slip, 50 m/s airspeed.
        run_thermal(&mut s, &p, 5000.0, 4.0, 50.0, 10.0);

        assert!(
            s.surface_temp > 50.0,
            "surface {} should exceed 50 C after 10 s of hard driving",
            s.surface_temp
        );
        assert!(
            s.surface_temp > s.carcass_temp,
            "surface {} should lead carcass {}",
            s.surface_temp,
            s.carcass_temp
        );
        assert!(
            s.carcass_temp >= p.ambient_temp,
            "carcass {} below ambient",
            s.carcass_temp
        );
    }

    // (e) Cooling: surface cools faster than the carcass, both head to ambient.
    #[test]
    fn thermal_cooling_surface_faster_than_carcass() {
        let p = TireThermalParams::f1_soft();
        // Start hot (set the state directly), then cruise: low load, high airspeed.
        let mut s = TireThermalState {
            surface_temp: 120.0,
            carcass_temp: 110.0,
            wear: 0.0,
        };
        let (s0, c0) = (s.surface_temp, s.carcass_temp);
        run_thermal(&mut s, &p, 500.0, 0.5, 50.0, 30.0);

        assert!(s.surface_temp < s0, "surface should cool: {s0} -> {}", s.surface_temp);
        assert!(s.carcass_temp < c0, "carcass should cool: {c0} -> {}", s.carcass_temp);
        let surface_drop = s0 - s.surface_temp;
        let carcass_drop = c0 - s.carcass_temp;
        assert!(
            surface_drop > carcass_drop,
            "surface drop {surface_drop} should exceed carcass drop {carcass_drop}"
        );
    }

    // (f) Equilibrium: a sustained operating point settles to a steady surface
    // temperature in the 80-120 C grip window (the slow carcass mode means it
    // takes a few hundred seconds to fully settle).
    #[test]
    fn thermal_reaches_equilibrium() {
        let p = TireThermalParams::f1_soft();
        let mut s = TireThermalState::new(p.ambient_temp);
        run_thermal(&mut s, &p, 5000.0, 4.0, 50.0, 200.0);

        let before = s.surface_temp;
        run_thermal(&mut s, &p, 5000.0, 4.0, 50.0, 0.5);
        let rate = (s.surface_temp - before) / 0.5;

        assert!(s.surface_temp.is_finite(), "surface temp not finite");
        assert!(
            (80.0..=120.0).contains(&s.surface_temp),
            "equilibrium surface {} should sit in the 80-120 C grip window",
            s.surface_temp
        );
        assert!(rate.abs() < 0.06, "not near equilibrium, dT/dt = {rate}");
    }

    // (g) Wear accumulation: a single lap wears the tire a little, and a 30-lap
    // stint at race pace lands in a realistic 0.1-0.9 degradation range.
    #[test]
    fn wear_accumulates_monotonically() {
        let p = TireThermalParams::f1_soft();
        let mut s = TireThermalState::new(p.ambient_temp);
        let dt = 0.05;
        // Lap-average operating point: ~5 kN load, 2 m/s slip, 4 kN vertical.
        let (force, slip, fz) = (5000.0, 2.0, 4000.0);

        let one_lap = (60.0 / dt) as usize;
        for _ in 0..one_lap {
            s.update_wear(&p, force, slip, fz, dt);
        }
        let after_one_lap = s.wear;

        let remaining = (29.0 * 60.0 / dt) as usize;
        for _ in 0..remaining {
            s.update_wear(&p, force, slip, fz, dt);
        }
        let after_stint = s.wear;

        assert!(after_one_lap > 0.0, "no wear after a lap");
        assert!(after_one_lap < 0.1, "single lap should not wear the tire much");
        assert!(
            after_stint > after_one_lap,
            "wear should grow over a stint: {after_one_lap} -> {after_stint}"
        );
        assert!(
            (0.1..=0.9).contains(&after_stint),
            "30-lap stint wear {after_stint} should be a realistic 0.1-0.9",
        );
    }

    // (g, no-load) No wear when the wheel is off the ground (fz < 100 N).
    #[test]
    fn no_wear_when_unloaded() {
        let p = TireThermalParams::f1_soft();
        let mut s = TireThermalState::new(p.ambient_temp);
        for _ in 0..1000 {
            s.update_wear(&p, 8000.0, 10.0, 50.0, 0.05);
        }
        assert_eq!(s.wear, 0.0);
    }

    // (h) Wear increases with temperature.
    #[test]
    fn wear_faster_when_hot() {
        let p = TireThermalParams::f1_soft();
        let mut cool = TireThermalState::new(p.ambient_temp);
        cool.surface_temp = p.optimal_temp; // temp_factor = 1.0
        let mut hot = TireThermalState::new(p.ambient_temp);
        hot.surface_temp = p.optimal_temp + 30.0; // temp_factor = 1.9

        // update_wear does not change surface_temp, so the factor stays fixed.
        for _ in 0..1000 {
            cool.update_wear(&p, 8000.0, 10.0, 4000.0, 0.05);
            hot.update_wear(&p, 8000.0, 10.0, 4000.0, 0.05);
        }
        assert!(
            hot.wear > cool.wear,
            "hot tire wear {} should exceed cool {}",
            hot.wear,
            cool.wear
        );
    }

    // (i) Grip multiplier combines temperature and wear.
    #[test]
    fn grip_multiplier_combines_temp_and_wear() {
        let p = TireThermalParams::f1_soft();

        // Optimal temp, no wear -> 1.0.
        let fresh = TireThermalState {
            surface_temp: p.optimal_temp,
            carcass_temp: p.optimal_temp,
            wear: 0.0,
        };
        assert!(approx(fresh.grip_multiplier(&p), 1.0, 1e-12));

        // Optimal temp, half worn -> ~0.85 (wear penalty only).
        let worn = TireThermalState {
            surface_temp: p.optimal_temp,
            carcass_temp: p.optimal_temp,
            wear: 0.5,
        };
        assert!(approx(worn.grip_multiplier(&p), 0.85, 1e-9));

        // Cold, no wear -> below 1.0 (temperature penalty only).
        let cold = TireThermalState {
            surface_temp: p.optimal_temp - 30.0,
            carcass_temp: p.optimal_temp - 30.0,
            wear: 0.0,
        };
        assert!(cold.grip_multiplier(&p) < 1.0);
    }

    // (j) Compound comparison.
    #[test]
    fn soft_heats_faster_than_hard() {
        let soft = TireThermalParams::f1_soft();
        let hard = TireThermalParams::f1_hard();
        let mut s_soft = TireThermalState::new(soft.ambient_temp);
        let mut s_hard = TireThermalState::new(hard.ambient_temp);

        // Identical hard-driving input; the soft's lower thermal mass and higher
        // heat generation should put it ahead after 10 s.
        run_thermal(&mut s_soft, &soft, 6000.0, 5.0, 40.0, 10.0);
        run_thermal(&mut s_hard, &hard, 6000.0, 5.0, 40.0, 10.0);

        assert!(
            s_soft.surface_temp > s_hard.surface_temp,
            "soft {} should heat faster than hard {}",
            s_soft.surface_temp,
            s_hard.surface_temp
        );
    }

    #[test]
    fn hard_has_wider_grip_window() {
        let soft = TireThermalParams::f1_soft();
        let hard = TireThermalParams::f1_hard();
        // Same offset (+20 C) from each compound's own optimal.
        let soft_g = soft.grip_factor(soft.optimal_temp + 20.0);
        let hard_g = hard.grip_factor(hard.optimal_temp + 20.0);
        assert!(
            hard_g > soft_g,
            "hard grip {hard_g} should drop less than soft {soft_g}"
        );
    }

    // (k) Four-tire set.
    #[test]
    fn tire_set_starts_uniform_then_diverges() {
        let mut set = TireSetThermal::new(TireThermalParams::f1_soft());

        let g = set.grip_multipliers();
        assert!(approx(g[0], g[1], 1e-12) && approx(g[1], g[2], 1e-12) && approx(g[2], g[3], 1e-12));
        for gi in g {
            assert!(gi < 1.0, "cold tire grip {gi} should be below 1.0");
        }

        // Bring FL up to its optimal temperature; the others stay cold.
        set.state_fl.surface_temp = set.params_fl.optimal_temp;
        let g = set.grip_multipliers();
        assert!(g[0] > g[1], "FL {} should exceed FR {}", g[0], g[1]);
        assert!(g[0] > g[2], "FL {} should exceed RL {}", g[0], g[2]);
        assert!(g[0] > g[3], "FL {} should exceed RR {}", g[0], g[3]);

        // Temperatures and wear accessors are consistent.
        assert!(set.surface_temps()[0] > set.surface_temps()[1]);
        assert_eq!(set.wear_levels(), [0.0; 4]);
    }
}
