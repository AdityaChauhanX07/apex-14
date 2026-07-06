//! Central channel registry: the single source of truth for every telemetry
//! channel the workspace produces or consumes.
//!
//! Every serialized/displayed signal has one [`ChannelId`] with a stable
//! snake_case [`name`](ChannelId::name), a [`Unit`], a coarse [`Quantity`] (for
//! plot grouping), a human [`display_name`](ChannelSpec::display_name), and a
//! one-line [`description`](ChannelSpec::description). Producers reference the
//! registry at compile time — CSV headers, plot/axis labels, and the UDP packet
//! field docs all resolve to these entries rather than ad-hoc string literals.
//!
//! ## Extension policy (frozen contract)
//!
//! - **Append only.** New channels are added to the [`define_channels!`] table;
//!   existing rows are never reordered in a way that changes meaning.
//! - **Names are frozen once released.** A channel's `name` is a wire/format
//!   identity (CSV header, serde tag). Renaming it breaks stored data — don't.
//! - **Units never change meaning for an existing name.** If a signal needs a
//!   different unit, add a NEW channel (e.g. `speed` in m/s vs `speed_kph` in
//!   km/h) — never repoint an existing name to a new unit.
//! - **Every future consumer reads this registry.** MoTeC `.ld` export,
//!   telemetry correlation, and the Python bindings must map through
//!   [`ChannelId`]/[`ChannelSpec`], not re-declare names.
//!
//! ## wasm cleanliness
//!
//! This module is dependency-light on purpose (only `serde` core, already an
//! apex-telemetry dependency; no chrono/regex/std-only heavy deps) so the
//! registry can move into the wasm graph later without changes.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Physical unit of a channel's samples.
///
/// [`Unit::si_factor`] gives the multiplier from this unit to its canonical SI
/// unit (e.g. km/h → m/s), so consumers can normalize without hard-coding
/// conversions.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Unit {
    /// Meter (m).
    Meter,
    /// Meter per second (m/s).
    MeterPerSecond,
    /// Kilometer per hour (km/h) — non-SI display unit.
    KilometerPerHour,
    /// Radian per second (rad/s).
    RadPerSecond,
    /// Multiples of standard gravity (g).
    G,
    /// Curvature, radian per meter / reciprocal meter (1/m).
    RadPerMeter,
    /// Radian (rad).
    Radian,
    /// Degree (deg) — non-SI display unit.
    Degree,
    /// Newton (N).
    Newton,
    /// Second (s).
    Second,
    /// Degrees Celsius (°C).
    Celsius,
    /// Revolutions per minute (rpm).
    Rpm,
    /// Watt (W) — power.
    Watt,
    /// Dimensionless / no unit (counts, indices, ratios).
    None,
}

impl Unit {
    /// Short human symbol, e.g. `"m/s"`. Empty for [`Unit::None`].
    pub const fn symbol(self) -> &'static str {
        match self {
            Unit::Meter => "m",
            Unit::MeterPerSecond => "m/s",
            Unit::KilometerPerHour => "km/h",
            Unit::RadPerSecond => "rad/s",
            Unit::G => "g",
            Unit::RadPerMeter => "1/m",
            Unit::Radian => "rad",
            Unit::Degree => "deg",
            Unit::Newton => "N",
            Unit::Second => "s",
            Unit::Celsius => "°C",
            Unit::Rpm => "rpm",
            Unit::Watt => "W",
            Unit::None => "",
        }
    }

    /// Multiplier that converts a value in this unit to its canonical SI unit.
    ///
    /// `value_si = value * unit.si_factor()`. Units that are already SI (or
    /// dimensionless) return `1.0`; km/h → m/s is `1/3.6`, degrees → radians is
    /// `π/180`, and `g` → m/s² uses standard gravity.
    pub fn si_factor(self) -> f64 {
        match self {
            Unit::KilometerPerHour => 1.0 / 3.6,
            Unit::Degree => std::f64::consts::PI / 180.0,
            Unit::G => 9.806_65,
            _ => 1.0,
        }
    }
}

/// Coarse physical kind of a channel, for grouping related channels on shared
/// plot axes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Quantity {
    /// Length / position (m).
    Distance,
    /// Linear speed (m/s, km/h).
    Speed,
    /// Angular velocity (rad/s).
    AngularVelocity,
    /// Acceleration (g, m/s²).
    Acceleration,
    /// Angle (rad, deg).
    Angle,
    /// Path curvature (1/m).
    Curvature,
    /// Force (N).
    Force,
    /// Time (s).
    Time,
    /// Temperature (°C) — no channel today; reserved for tire-thermal telemetry.
    Temperature,
    /// A pure ratio / dimensionless quantity.
    Dimensionless,
    /// An integer count or index (gear, lap, sequence).
    Count,
    /// Power (W).
    Power,
}

impl Quantity {
    /// Stable lowercase identifier, for grouping keys / diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Quantity::Distance => "distance",
            Quantity::Speed => "speed",
            Quantity::AngularVelocity => "angular_velocity",
            Quantity::Acceleration => "acceleration",
            Quantity::Angle => "angle",
            Quantity::Curvature => "curvature",
            Quantity::Force => "force",
            Quantity::Time => "time",
            Quantity::Temperature => "temperature",
            Quantity::Dimensionless => "dimensionless",
            Quantity::Count => "count",
            Quantity::Power => "power",
        }
    }
}

/// The static specification of one channel.
#[derive(Copy, Clone, Debug)]
pub struct ChannelSpec {
    /// The channel this spec describes.
    pub id: ChannelId,
    /// Stable snake_case name — the CSV header / serde tag. Frozen once released.
    pub name: &'static str,
    /// Physical unit of the samples.
    pub unit: Unit,
    /// Coarse kind, for plot grouping.
    pub quantity: Quantity,
    /// Human-facing label for UI (viewer plots, HUD). May repeat across
    /// unit-variant channels (e.g. `speed` and `speed_kph` both display "Speed").
    pub display_name: &'static str,
    /// One-line description of what the channel measures.
    pub description: &'static str,
}

/// Co-generates the [`ChannelId`] enum, the [`CHANNELS`] table, the
/// exhaustive-match [`spec`](ChannelId::spec), and the name parser from a single
/// list — so a channel cannot exist without a full spec (invariant c) and
/// `name`/`from_name` cannot drift (invariant d), both enforced at compile time.
macro_rules! define_channels {
    ($( $variant:ident : $name:literal, $unit:ident, $quantity:ident, $display:literal, $desc:literal ; )+) => {
        /// Stable identifier for a telemetry channel. Serializes as its
        /// snake_case [`name`](ChannelId::name).
        ///
        /// `Ord`/`PartialOrd` follow declaration order in [`define_channels!`],
        /// which is append-only — so the ordering is stable and usable as a
        /// `BTreeMap` key (e.g. for deterministic column ordering in consumers).
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub enum ChannelId {
            $( #[doc = $desc] $variant, )+
        }

        /// Every channel, in declaration order.
        pub const CHANNELS: &[ChannelId] = &[ $( ChannelId::$variant, )+ ];

        impl ChannelId {
            /// The static spec for this channel (exhaustive — adding a variant
            /// without a spec row fails to compile).
            pub const fn spec(self) -> ChannelSpec {
                match self {
                    $( ChannelId::$variant => ChannelSpec {
                        id: ChannelId::$variant,
                        name: $name,
                        unit: Unit::$unit,
                        quantity: Quantity::$quantity,
                        display_name: $display,
                        description: $desc,
                    }, )+
                }
            }

            /// Parse a channel from its stable snake_case [`name`](Self::name).
            pub fn from_name(name: &str) -> Option<ChannelId> {
                match name {
                    $( $name => Some(ChannelId::$variant), )+
                    _ => None,
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// The registry. Seeded from every channel in the codebase (see docs/channels.md
// for the per-source audit). Rows: Variant : "name", Unit, Quantity, "Display",
// "description".
// ---------------------------------------------------------------------------
define_channels! {
    // --- geometry / position (Distance, meters) ---
    S               : "s",                 Meter,          Distance,        "Distance",       "Arc length along the track centerline";
    SRaw            : "s_raw",              Meter,          Distance,        "Distance (raw)", "Raw source arc length before geometric re-projection (e.g. FastF1 integrated Distance)";
    X               : "x",                 Meter,          Distance,        "X",              "World X coordinate of the centerline point";
    Y               : "y",                 Meter,          Distance,        "Y",              "World Y coordinate of the centerline point";
    PosX            : "pos_x",             Meter,          Distance,        "Pos X",          "Vehicle world X position";
    PosY            : "pos_y",             Meter,          Distance,        "Pos Y",          "Vehicle world Y position";
    PosZ            : "pos_z",             Meter,          Distance,        "Pos Z",          "Vehicle world Z position (height)";
    LateralOffset   : "lateral_offset",    Meter,          Distance,        "Lateral offset", "Lateral offset from the centerline (positive = left)";
    TrackDistance   : "track_distance",    Meter,          Distance,        "Track distance", "Distance travelled along the centerline (UDP telemetry)";
    TrackOffset     : "track_offset",      Meter,          Distance,        "Track offset",   "Lateral offset from the centerline (positive = right, UDP telemetry)";
    RideHeightFront : "ride_height_front", Meter,          Distance,        "Ride height (F)","Front-axle ride height";
    RideHeightRear  : "ride_height_rear",  Meter,          Distance,        "Ride height (R)","Rear-axle ride height";
    SuspFl          : "susp_fl",           Meter,          Distance,        "Susp FL",        "Front-left suspension travel";
    SuspFr          : "susp_fr",           Meter,          Distance,        "Susp FR",        "Front-right suspension travel";
    SuspRl          : "susp_rl",           Meter,          Distance,        "Susp RL",        "Rear-left suspension travel";
    SuspRr          : "susp_rr",           Meter,          Distance,        "Susp RR",        "Rear-right suspension travel";

    // --- speeds ---
    Speed           : "speed",             MeterPerSecond, Speed,           "Speed",          "Vehicle forward (longitudinal) speed";
    SpeedKph        : "speed_kph",         KilometerPerHour, Speed,         "Speed",          "Vehicle forward speed, km/h display of `speed`";
    LateralV        : "lateral_v",         MeterPerSecond, Speed,           "Lateral v",      "Lateral (body-frame) velocity";
    VerticalV       : "vertical_v",        MeterPerSecond, Speed,           "Vertical v",     "Vertical velocity";

    // --- angular velocities ---
    YawRate         : "yaw_rate",          RadPerSecond,   AngularVelocity, "Yaw rate",       "Yaw rate about the vertical axis";
    WheelFl         : "wheel_fl",          RadPerSecond,   AngularVelocity, "Wheel FL",       "Front-left wheel angular velocity";
    WheelFr         : "wheel_fr",          RadPerSecond,   AngularVelocity, "Wheel FR",       "Front-right wheel angular velocity";
    WheelRl         : "wheel_rl",          RadPerSecond,   AngularVelocity, "Wheel RL",       "Rear-left wheel angular velocity";
    WheelRr         : "wheel_rr",          RadPerSecond,   AngularVelocity, "Wheel RR",       "Rear-right wheel angular velocity";
    Rpm             : "rpm",               Rpm,            AngularVelocity, "RPM",            "Engine crankshaft rotational speed";

    // --- accelerations ---
    LateralG        : "lateral_g",         G,              Acceleration,    "Lat G",          "Lateral acceleration";
    LongitudinalG   : "longitudinal_g",    G,              Acceleration,    "Lon G",          "Longitudinal acceleration";

    // --- angles ---
    Roll            : "roll",              Radian,         Angle,           "Roll",           "Roll angle";
    Pitch           : "pitch",             Radian,         Angle,           "Pitch",          "Pitch angle";
    Yaw             : "yaw",               Radian,         Angle,           "Yaw",            "Yaw angle (heading)";
    RollDeg         : "roll_deg",          Degree,         Angle,           "Roll",           "Roll angle, degree display of `roll`";
    PitchDeg        : "pitch_deg",         Degree,         Angle,           "Pitch",          "Pitch angle, degree display of `pitch`";
    SteeringAngle   : "steering_angle",    Radian,         Angle,           "Steering",       "Steering input angle (measured steering channel)";

    // --- curvature ---
    Curvature       : "curvature",         RadPerMeter,    Curvature,       "Curvature",      "Path curvature (kappa = 1/R)";
    CurvatureCmd    : "curvature_cmd",     RadPerMeter,    Curvature,       "Curvature cmd",  "Commanded path curvature";

    // --- forces ---
    FDrive          : "f_drive",           Newton,         Force,           "Drive force",    "Net longitudinal drive force";
    FzFl            : "fz_fl",             Newton,         Force,           "Fz FL",          "Front-left vertical tire load";
    FzFr            : "fz_fr",             Newton,         Force,           "Fz FR",          "Front-right vertical tire load";
    FzRl            : "fz_rl",             Newton,         Force,           "Fz RL",          "Rear-left vertical tire load";
    FzRr            : "fz_rr",             Newton,         Force,           "Fz RR",          "Rear-right vertical tire load";

    // --- inferred loads / aero / power (QSS channel inference) ---
    Downforce       : "downforce",         Newton,         Force,           "Downforce",      "Aerodynamic downforce (inferred from the car aero model at v)";
    AeroDragForce   : "aero_drag_force",   Newton,         Force,           "Aero drag",      "Aerodynamic drag force (inferred from the car aero model at v)";
    FzFront         : "fz_front",          Newton,         Force,           "Fz front",       "Front-axle vertical load (static + aero + longitudinal transfer)";
    FzRear          : "fz_rear",           Newton,         Force,           "Fz rear",        "Rear-axle vertical load (static + aero + longitudinal transfer)";
    GripUtil        : "grip_util",         None,           Dimensionless,   "Grip util",      "Total friction-circle grip utilization |F|/(mu*(mg+DF)); >1 = noise/model deficiency";
    TractivePower   : "tractive_power",    Watt,           Power,           "Tractive power", "Tractive power at the driven wheels, F_drive*v (accelerating)";
    BrakingPower    : "braking_power",     Watt,           Power,           "Braking power",  "Braking power dissipated, F_brake*v (braking)";

    // --- time ---
    Time            : "t",                 Second,         Time,            "Time",           "Elapsed time";
    LapTime         : "lap_time",          Second,         Time,            "Lap time",       "Time on the current lap";
    SimTime         : "sim_time",          Second,         Time,            "Sim time",       "Total simulation time";

    // --- driver inputs (measured telemetry) ---
    Throttle        : "throttle",          None,           Dimensionless,   "Throttle",       "Throttle pedal position (0 = closed, 1 = wide-open)";
    Brake           : "brake",             None,           Dimensionless,   "Brake",          "Brake application (0 = released, 1 = full; a 0/1 flag or a fraction)";

    // --- counts / indices ---
    Gear            : "gear",              None,           Count,           "Gear",           "Current gear (0 = neutral)";
    Lap             : "lap",               None,           Count,           "Lap",            "Current lap number";
    Sequence        : "sequence",          None,           Count,           "Seq",            "Packet sequence number";
}

impl ChannelId {
    /// The stable snake_case name.
    pub const fn name(self) -> &'static str {
        self.spec().name
    }
    /// The physical unit.
    pub const fn unit(self) -> Unit {
        self.spec().unit
    }
    /// The coarse quantity kind.
    pub const fn quantity(self) -> Quantity {
        self.spec().quantity
    }
    /// The human-facing display label.
    pub const fn display_name(self) -> &'static str {
        self.spec().display_name
    }
    /// The one-line description.
    pub const fn description(self) -> &'static str {
        self.spec().description
    }
}

impl Serialize for ChannelId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for ChannelId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        ChannelId::from_name(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown channel name: {name}")))
    }
}

/// Build the CSV provenance `# columns:` line for an ordered set of column
/// names: `# columns: s[m], speed[m/s], ...`. Each `name[symbol]` pair uses the
/// registry unit symbol; a name not in the registry renders `name[?]`. Returned
/// without a trailing newline.
pub fn csv_columns_comment(column_names: &[&str]) -> String {
    let mut out = String::from("# columns: ");
    for (i, &n) in column_names.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let sym = ChannelId::from_name(n)
            .map(|id| id.unit().symbol())
            .unwrap_or("?");
        out.push_str(n);
        out.push('[');
        out.push_str(sym);
        out.push(']');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // (a) no duplicate names.
    #[test]
    fn names_are_unique() {
        let mut seen = HashSet::new();
        for &id in CHANNELS {
            assert!(
                seen.insert(id.name()),
                "duplicate channel name: {}",
                id.name()
            );
        }
        assert_eq!(seen.len(), CHANNELS.len());
    }

    // (b) no duplicate ids.
    #[test]
    fn ids_are_unique() {
        let mut seen = HashSet::new();
        for &id in CHANNELS {
            assert!(seen.insert(id), "duplicate channel id: {id:?}");
        }
        assert_eq!(seen.len(), CHANNELS.len());
    }

    // (c) every listed channel resolves to exactly one spec, and that spec
    // points back to itself. (The exhaustive match in `spec()` makes "a variant
    // with no spec" a compile error; this checks the table is self-consistent.)
    #[test]
    fn every_channel_has_one_consistent_spec() {
        for &id in CHANNELS {
            let spec = id.spec();
            assert_eq!(spec.id, id, "spec for {id:?} points at {:?}", spec.id);
            assert!(!spec.name.is_empty(), "empty name for {id:?}");
            assert!(!spec.description.is_empty(), "empty description for {id:?}");
            assert!(
                !spec.display_name.is_empty(),
                "empty display_name for {id:?}"
            );
        }
    }

    // (d) name round-trip: from_name(name(id)) == id for all ids.
    #[test]
    fn name_round_trips() {
        for &id in CHANNELS {
            assert_eq!(
                ChannelId::from_name(id.name()),
                Some(id),
                "round-trip failed for {id:?}"
            );
        }
        assert_eq!(ChannelId::from_name("definitely_not_a_channel"), None);
    }

    // serde serializes/deserializes by stable name.
    #[test]
    fn serde_round_trips_by_name() {
        for &id in CHANNELS {
            let json = serde_json::to_string(&id).unwrap();
            assert_eq!(json, format!("\"{}\"", id.name()));
            let back: ChannelId = serde_json::from_str(&json).unwrap();
            assert_eq!(back, id);
        }
        assert!(serde_json::from_str::<ChannelId>("\"bogus\"").is_err());
    }

    // snake_case sanity for every name (lowercase, digits, underscores only).
    #[test]
    fn names_are_snake_case() {
        for &id in CHANNELS {
            let n = id.name();
            assert!(
                n.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "name {n} is not snake_case"
            );
        }
    }

    // Measured-telemetry channels added for the correlation importer (Phase 2).
    // Locks their name/unit/quantity so the interchange format can't drift.
    #[test]
    fn measured_telemetry_channels_have_expected_specs() {
        let cases = [
            (
                ChannelId::Throttle,
                "throttle",
                Unit::None,
                Quantity::Dimensionless,
            ),
            (
                ChannelId::Brake,
                "brake",
                Unit::None,
                Quantity::Dimensionless,
            ),
            (ChannelId::Rpm, "rpm", Unit::Rpm, Quantity::AngularVelocity),
            (
                ChannelId::SteeringAngle,
                "steering_angle",
                Unit::Radian,
                Quantity::Angle,
            ),
            // s_raw: raw source arc length kept alongside re-projected `s`.
            (ChannelId::SRaw, "s_raw", Unit::Meter, Quantity::Distance),
        ];
        for (id, name, unit, quantity) in cases {
            assert_eq!(id.name(), name);
            assert_eq!(id.unit(), unit);
            assert_eq!(id.quantity(), quantity);
            assert_eq!(ChannelId::from_name(name), Some(id));
        }
        // throttle/brake are unitless: their `# columns:` symbol is empty.
        assert_eq!(
            csv_columns_comment(&["throttle", "brake"]),
            "# columns: throttle[], brake[]"
        );
    }

    // Inferred loads / aero / power channels (QSS channel inference). Locks
    // their name/unit/quantity, including the new Watt/Power additions.
    #[test]
    fn inferred_channels_have_expected_specs() {
        let cases = [
            (
                ChannelId::Downforce,
                "downforce",
                Unit::Newton,
                Quantity::Force,
            ),
            (
                ChannelId::AeroDragForce,
                "aero_drag_force",
                Unit::Newton,
                Quantity::Force,
            ),
            (
                ChannelId::FzFront,
                "fz_front",
                Unit::Newton,
                Quantity::Force,
            ),
            (ChannelId::FzRear, "fz_rear", Unit::Newton, Quantity::Force),
            (
                ChannelId::GripUtil,
                "grip_util",
                Unit::None,
                Quantity::Dimensionless,
            ),
            (
                ChannelId::TractivePower,
                "tractive_power",
                Unit::Watt,
                Quantity::Power,
            ),
            (
                ChannelId::BrakingPower,
                "braking_power",
                Unit::Watt,
                Quantity::Power,
            ),
        ];
        for (id, name, unit, quantity) in cases {
            assert_eq!(id.name(), name);
            assert_eq!(id.unit(), unit);
            assert_eq!(id.quantity(), quantity);
            assert_eq!(ChannelId::from_name(name), Some(id));
        }
        // New unit: Watt symbol + SI factor (already SI).
        assert_eq!(Unit::Watt.symbol(), "W");
        assert_eq!(Unit::Watt.si_factor(), 1.0);
        assert_eq!(Quantity::Power.as_str(), "power");
        assert_eq!(
            csv_columns_comment(&["downforce", "tractive_power", "grip_util"]),
            "# columns: downforce[N], tractive_power[W], grip_util[]"
        );
    }

    #[test]
    fn columns_comment_format() {
        let line = csv_columns_comment(&["s", "speed", "speed_kph", "not_real"]);
        assert_eq!(
            line,
            "# columns: s[m], speed[m/s], speed_kph[km/h], not_real[?]"
        );
    }
}
