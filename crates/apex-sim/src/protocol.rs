//! UDP packet protocol for real-time simulation.
//!
//! Defines the binary format for driver input packets and simulation
//! output packets. All multi-byte fields are little-endian.

/// Read a little-endian `f64` at byte offset `o`, or `None` if out of range.
fn rd_f64(buf: &[u8], o: usize) -> Option<f64> {
    Some(f64::from_le_bytes(buf.get(o..o + 8)?.try_into().ok()?))
}

/// Read a little-endian `u32` at byte offset `o`, or `None` if out of range.
fn rd_u32(buf: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(o..o + 4)?.try_into().ok()?))
}

/// Read a little-endian `f32` at byte offset `o`, or `None` if out of range.
fn rd_f32(buf: &[u8], o: usize) -> Option<f32> {
    Some(f32::from_le_bytes(buf.get(o..o + 4)?.try_into().ok()?))
}

/// Driver input packet received over UDP.
///
/// 20 bytes total, all fields little-endian.
/// Sent by the driver's controller at up to 1kHz.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InputPacket {
    /// Steering angle, normalized to [-1.0, 1.0].
    /// Negative = left, positive = right.
    pub steering: f32,
    /// Throttle position [0.0, 1.0].
    pub throttle: f32,
    /// Brake pressure [0.0, 1.0].
    pub brake: f32,
    /// Current gear (0 = neutral, 1-8 = forward gears).
    pub gear: u32,
    /// Packet sequence number (wrapping).
    pub sequence: u32,
}

impl InputPacket {
    /// Size of the packet in bytes.
    pub const SIZE: usize = 20;

    /// Deserialize from a byte buffer (little-endian).
    ///
    /// Returns None if the buffer is too small.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            steering: rd_f32(buf, 0)?,
            throttle: rd_f32(buf, 4)?,
            brake: rd_f32(buf, 8)?,
            gear: rd_u32(buf, 12)?,
            sequence: rd_u32(buf, 16)?,
        })
    }

    /// Serialize to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.steering.to_le_bytes());
        buf[4..8].copy_from_slice(&self.throttle.to_le_bytes());
        buf[8..12].copy_from_slice(&self.brake.to_le_bytes());
        buf[12..16].copy_from_slice(&self.gear.to_le_bytes());
        buf[16..20].copy_from_slice(&self.sequence.to_le_bytes());
        buf
    }

    /// Clamp all input values to valid ranges.
    pub fn clamp(&mut self) {
        self.steering = self.steering.clamp(-1.0, 1.0);
        self.throttle = self.throttle.clamp(0.0, 1.0);
        self.brake = self.brake.clamp(0.0, 1.0);
    }
}

/// Simulation output packet sent over UDP.
///
/// Contains the full vehicle state and derived telemetry.
/// Sent at the configured telemetry rate (default: 60Hz).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OutputPacket {
    // Position and orientation
    /// World X position (m).
    pub pos_x: f64,
    /// World Y position (m).
    pub pos_y: f64,
    /// World Z position (m).
    pub pos_z: f64,
    /// Roll angle (rad).
    pub roll: f64,
    /// Pitch angle (rad).
    pub pitch: f64,
    /// Yaw angle (rad).
    pub yaw: f64,

    // Velocities
    /// Forward speed (m/s).
    pub speed: f64,
    /// Lateral velocity (m/s).
    pub lateral_v: f64,
    /// Vertical velocity (m/s).
    pub vertical_v: f64,
    /// Yaw rate (rad/s).
    pub yaw_rate: f64,

    // Wheel speeds (rad/s)
    /// Front-left wheel angular velocity.
    pub wheel_fl: f64,
    /// Front-right wheel angular velocity.
    pub wheel_fr: f64,
    /// Rear-left wheel angular velocity.
    pub wheel_rl: f64,
    /// Rear-right wheel angular velocity.
    pub wheel_rr: f64,

    // Suspension
    /// Front-left suspension travel (m).
    pub susp_fl: f64,
    /// Front-right suspension travel (m).
    pub susp_fr: f64,
    /// Rear-left suspension travel (m).
    pub susp_rl: f64,
    /// Rear-right suspension travel (m).
    pub susp_rr: f64,

    // Derived telemetry
    /// Longitudinal acceleration (g).
    pub accel_long: f64,
    /// Lateral acceleration (g).
    pub accel_lat: f64,
    /// Current gear.
    pub gear: u32,
    /// Current lap number.
    pub lap: u32,
    /// Lap time (s).
    pub lap_time: f64,
    /// Simulation time (s).
    pub sim_time: f64,
    /// Packet sequence number.
    pub sequence: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

impl OutputPacket {
    /// Size of the packet in bytes (22 f64 + 4 u32).
    pub const SIZE: usize = 192;

    /// Serialize to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        let mut o = 0;
        // The 20 leading f64 fields, in declaration order.
        let floats = [
            self.pos_x,
            self.pos_y,
            self.pos_z,
            self.roll,
            self.pitch,
            self.yaw,
            self.speed,
            self.lateral_v,
            self.vertical_v,
            self.yaw_rate,
            self.wheel_fl,
            self.wheel_fr,
            self.wheel_rl,
            self.wheel_rr,
            self.susp_fl,
            self.susp_fr,
            self.susp_rl,
            self.susp_rr,
            self.accel_long,
            self.accel_lat,
        ];
        for v in floats {
            buf[o..o + 8].copy_from_slice(&v.to_le_bytes());
            o += 8;
        }
        buf[o..o + 4].copy_from_slice(&self.gear.to_le_bytes());
        o += 4;
        buf[o..o + 4].copy_from_slice(&self.lap.to_le_bytes());
        o += 4;
        buf[o..o + 8].copy_from_slice(&self.lap_time.to_le_bytes());
        o += 8;
        buf[o..o + 8].copy_from_slice(&self.sim_time.to_le_bytes());
        o += 8;
        buf[o..o + 4].copy_from_slice(&self.sequence.to_le_bytes());
        o += 4;
        buf[o..o + 4].copy_from_slice(&self._pad.to_le_bytes());
        buf
    }

    /// Deserialize from a byte buffer (little-endian).
    ///
    /// Returns None if the buffer is too small.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            pos_x: rd_f64(buf, 0)?,
            pos_y: rd_f64(buf, 8)?,
            pos_z: rd_f64(buf, 16)?,
            roll: rd_f64(buf, 24)?,
            pitch: rd_f64(buf, 32)?,
            yaw: rd_f64(buf, 40)?,
            speed: rd_f64(buf, 48)?,
            lateral_v: rd_f64(buf, 56)?,
            vertical_v: rd_f64(buf, 64)?,
            yaw_rate: rd_f64(buf, 72)?,
            wheel_fl: rd_f64(buf, 80)?,
            wheel_fr: rd_f64(buf, 88)?,
            wheel_rl: rd_f64(buf, 96)?,
            wheel_rr: rd_f64(buf, 104)?,
            susp_fl: rd_f64(buf, 112)?,
            susp_fr: rd_f64(buf, 120)?,
            susp_rl: rd_f64(buf, 128)?,
            susp_rr: rd_f64(buf, 136)?,
            accel_long: rd_f64(buf, 144)?,
            accel_lat: rd_f64(buf, 152)?,
            gear: rd_u32(buf, 160)?,
            lap: rd_u32(buf, 164)?,
            lap_time: rd_f64(buf, 168)?,
            sim_time: rd_f64(buf, 176)?,
            sequence: rd_u32(buf, 184)?,
            _pad: rd_u32(buf, 188)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_packet_size() {
        assert_eq!(InputPacket::SIZE, 20);
    }

    #[test]
    fn test_input_packet_roundtrip() {
        let pkt = InputPacket {
            steering: -0.25,
            throttle: 0.75,
            brake: 0.5,
            gear: 4,
            sequence: 123_456,
        };
        let bytes = pkt.to_bytes();
        assert_eq!(bytes.len(), InputPacket::SIZE);
        let back = InputPacket::from_bytes(&bytes).expect("decode");
        assert_eq!(pkt, back);
    }

    #[test]
    fn test_output_packet_roundtrip() {
        let pkt = OutputPacket {
            pos_x: 1.0,
            pos_y: 2.0,
            pos_z: 3.0,
            roll: 0.01,
            pitch: 0.02,
            yaw: 0.03,
            speed: 55.5,
            lateral_v: -1.5,
            vertical_v: 0.25,
            yaw_rate: 0.4,
            wheel_fl: 100.0,
            wheel_fr: 101.0,
            wheel_rl: 102.0,
            wheel_rr: 103.0,
            susp_fl: 0.011,
            susp_fr: 0.012,
            susp_rl: 0.013,
            susp_rr: 0.014,
            accel_long: -1.2,
            accel_lat: 2.3,
            gear: 6,
            lap: 7,
            lap_time: 88.123,
            sim_time: 250.5,
            sequence: 999,
            _pad: 0,
        };
        let bytes = pkt.to_bytes();
        assert_eq!(bytes.len(), OutputPacket::SIZE);
        let back = OutputPacket::from_bytes(&bytes).expect("decode");
        assert_eq!(pkt, back);
    }

    #[test]
    fn test_input_clamp() {
        let mut pkt = InputPacket {
            steering: 2.0,
            throttle: -0.5,
            brake: 1.5,
            gear: 3,
            sequence: 0,
        };
        pkt.clamp();
        assert_eq!(pkt.steering, 1.0);
        assert_eq!(pkt.throttle, 0.0);
        assert_eq!(pkt.brake, 1.0);

        let mut pkt2 = InputPacket {
            steering: -3.0,
            ..Default::default()
        };
        pkt2.clamp();
        assert_eq!(pkt2.steering, -1.0);
    }

    #[test]
    fn test_input_from_short_buffer() {
        let buf = [0u8; 10];
        assert!(InputPacket::from_bytes(&buf).is_none());
    }

    #[test]
    fn test_output_from_short_buffer() {
        let buf = [0u8; 100];
        assert!(OutputPacket::from_bytes(&buf).is_none());
    }
}
