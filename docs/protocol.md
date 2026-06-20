# Apex-14 Simulation Protocol

Real-time UDP protocol for the Apex-14 hardware-in-the-loop simulator.

## Overview

The simulation server runs a 14-degree-of-freedom vehicle dynamics model
at 1kHz. It accepts driver inputs (steering, throttle, brake) over UDP
and broadcasts vehicle state telemetry at a configurable rate.

All multi-byte fields are little-endian. There is no packet header or
framing: each UDP datagram contains exactly one packet.

## Network Setup

- **Input**: Server listens on UDP port 20777 (configurable via `--input-port`).
- **Output**: Server sends to 127.0.0.1:20778 (configurable via `--output-addr`).
- **Telemetry rate**: 60 Hz by default (configurable via `--telemetry-hz`).

The server holds the most recent input: if no new input packet arrives, the
last received input continues to be applied. Input values outside their valid
range are clamped.

## Input Packet (Driver to Sim)

20 bytes, little-endian.

| Offset | Size | Type | Field    | Range       | Description               |
|--------|------|------|----------|-------------|---------------------------|
| 0      | 4    | f32  | steering | [-1.0, 1.0] | Steering input (neg=left) |
| 4      | 4    | f32  | throttle | [0.0, 1.0]  | Throttle position         |
| 8      | 4    | f32  | brake    | [0.0, 1.0]  | Brake pressure            |
| 12     | 4    | u32  | gear     | 0 to 8      | Gear (0=neutral)          |
| 16     | 4    | u32  | sequence | 0 to 2^32-1 | Packet sequence number    |

Out-of-range values are clamped. If no input is received, the last
input is held. The `steering` value is scaled by the server's maximum
steering angle (0.5 rad by default) before reaching the model.

Python format string: `'<fffII'` (3 floats then 2 unsigned ints).

## Output Packet (Sim to Driver/Dashboard)

208 bytes, little-endian. This is `OutputPacket::SIZE`: 24 f64 fields
(8 bytes each) and 4 u32 fields (4 bytes each), interleaved as in the
layout below.

| Offset | Size | Type | Field      | Unit  | Description                   |
|--------|------|------|------------|-------|-------------------------------|
| 0      | 8    | f64  | pos_x      | m     | World X position              |
| 8      | 8    | f64  | pos_y      | m     | World Y position              |
| 16     | 8    | f64  | pos_z      | m     | World Z position              |
| 24     | 8    | f64  | roll       | rad   | Roll angle                    |
| 32     | 8    | f64  | pitch      | rad   | Pitch angle                   |
| 40     | 8    | f64  | yaw        | rad   | Yaw angle                     |
| 48     | 8    | f64  | speed      | m/s   | Forward (longitudinal) speed  |
| 56     | 8    | f64  | lateral_v  | m/s   | Lateral velocity              |
| 64     | 8    | f64  | vertical_v | m/s   | Vertical velocity             |
| 72     | 8    | f64  | yaw_rate   | rad/s | Yaw rate                      |
| 80     | 8    | f64  | wheel_fl   | rad/s | Front-left wheel speed        |
| 88     | 8    | f64  | wheel_fr   | rad/s | Front-right wheel speed       |
| 96     | 8    | f64  | wheel_rl   | rad/s | Rear-left wheel speed         |
| 104    | 8    | f64  | wheel_rr   | rad/s | Rear-right wheel speed        |
| 112    | 8    | f64  | susp_fl    | m     | Front-left suspension travel  |
| 120    | 8    | f64  | susp_fr    | m     | Front-right suspension travel |
| 128    | 8    | f64  | susp_rl    | m     | Rear-left suspension travel   |
| 136    | 8    | f64  | susp_rr    | m     | Rear-right suspension travel  |
| 144    | 8    | f64  | accel_long | g     | Longitudinal acceleration     |
| 152    | 8    | f64  | accel_lat  | g     | Lateral acceleration          |
| 160    | 4    | u32  | gear       | -     | Current gear                  |
| 164    | 4    | u32  | lap        | -     | Current lap number            |
| 168    | 8    | f64  | lap_time   | s     | Time on the current lap       |
| 176    | 8    | f64  | sim_time   | s     | Total simulation time         |
| 184    | 4    | u32  | sequence       | -   | Packet sequence number          |
| 188    | 8    | f64  | track_distance | m   | Distance along the centerline   |
| 196    | 8    | f64  | track_offset   | m   | Lateral offset (positive=right) |
| 204    | 4    | u32  | _pad           | -   | Padding (always 0)              |

Total size: 24 x 8 + 4 x 4 = 208 bytes.

`track_distance` and `track_offset` come from projecting the car's world
position onto the loaded track centerline. When the server runs without a
track they are both 0.

Note the field ordering: the two u32 fields `gear` and `lap` sit between
the suspension/acceleration block and the `lap_time`/`sim_time` f64 pair,
and the `sequence` u32 sits between `sim_time` and the track-position pair,
so the layout is not a single contiguous run of doubles. Decode by offset
rather than assuming all doubles come first.

Python format string for the full packet: `'<20d 2I 2d I 2d I'`
(20 doubles; gear+lap; lap_time+sim_time; sequence; track_distance+track_offset; pad).

## Quick Start

```sh
# Start the server
cargo run --release --bin sim-server -- --track silverstone

# In another terminal, send a throttle-only input with Python:
import struct, socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# steering=0.0, throttle=1.0, brake=0.0, gear=1, seq=0
packet = struct.pack('<fffII', 0.0, 1.0, 0.0, 1, 0)
sock.sendto(packet, ('127.0.0.1', 20777))
```

## Reading Telemetry with Python

```python
import struct, socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 20778))

while True:
    data, addr = sock.recvfrom(256)
    # Parse the leading 7 doubles: pos_x, pos_y, pos_z, roll, pitch, yaw, speed
    values = struct.unpack_from('<7d', data, 0)
    pos_x, pos_y, pos_z, roll, pitch, yaw, speed = values
    print(f"Speed: {speed:.1f} m/s  Pos: ({pos_x:.1f}, {pos_y:.1f})")
```

To decode the full packet at once:

```python
import struct

# 20 doubles, gear+lap (u32), lap_time+sim_time (f64), sequence (u32),
# track_distance+track_offset (f64), pad (u32)
fields = struct.unpack('<20d2I2dI2dI', data[:208])
pos_x, pos_y, pos_z = fields[0:3]
roll, pitch, yaw = fields[3:6]
speed, lateral_v, vertical_v, yaw_rate = fields[6:10]
wheel_fl, wheel_fr, wheel_rl, wheel_rr = fields[10:14]
susp_fl, susp_fr, susp_rl, susp_rr = fields[14:18]
accel_long, accel_lat = fields[18:20]
gear, lap = fields[20:22]
lap_time, sim_time = fields[22:24]
sequence = fields[24]
track_distance, track_offset = fields[25:27]
_pad = fields[27]
```

## Shared Memory Interface

For lowest-latency local communication, the sim-server can expose a
shared memory file alongside (or instead of) UDP.

### Usage

```sh
sim-server --track silverstone --shared-mem /tmp/apex14_sim
```

### Memory Layout

The shared file is 256 bytes with a fixed layout:

| Offset | Size | Field           | Writer | Type            |
|--------|------|-----------------|--------|-----------------|
| 0      | 20   | InputPacket     | Client | (see above)     |
| 20     | 4    | input_sequence  | Client | u32 LE          |
| 24     | 208  | OutputPacket    | Sim    | (see above)     |
| 232    | 4    | output_sequence | Sim    | u32 LE          |
| 236    | 1    | sim_running     | Sim    | u8 (0 or 1)     |
| 237    | 19   | reserved        | none   | zero            |

The InputPacket layout matches the UDP input packet (20 bytes), and the
OutputPacket layout matches the UDP output packet (208 bytes); only the
byte offsets differ.

### Tearing Prevention

Each writer increments its sequence counter after completing a write.
Readers check the counter before and after reading; if the values
differ, the read overlapped a write and should be retried.

### Python Example

```python
import mmap
import struct
import os

# Open the shared memory file.
fd = os.open('/tmp/apex14_sim', os.O_RDWR)
mm = mmap.mmap(fd, 256)

# Write a throttle input and bump the input sequence.
steering, throttle, brake, gear, seq = 0.0, 0.8, 0.0, 3, 1
mm[0:20] = struct.pack('<fffII', steering, throttle, brake, gear, seq)
mm[20:24] = struct.pack('<I', seq)

# Read output (with a torn-read retry on the output sequence).
while True:
    seq_before = struct.unpack_from('<I', mm, 232)[0]
    output_bytes = mm[24:232]
    seq_after = struct.unpack_from('<I', mm, 232)[0]
    if seq_before == seq_after:
        break
pos_x, pos_y, pos_z = struct.unpack_from('<3d', output_bytes, 0)
```
