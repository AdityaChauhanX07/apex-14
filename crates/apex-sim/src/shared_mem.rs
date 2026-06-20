//! Shared memory interface for zero-latency local simulation I/O.
//!
//! Uses a memory-mapped file as the transport. The sim-server writes
//! vehicle state into the shared region; the client (dashboard, driver
//! controller) reads state and writes inputs from/to the same region.
//!
//! ## Memory Layout
//!
//! The shared file has a fixed size of [`SHARED_MEM_SIZE`] bytes:
//!
//! | Offset | Size | Field              | Writer     |
//! |--------|------|--------------------|------------|
//! | 0      | 20   | InputPacket        | Client     |
//! | 20     | 4    | input_sequence     | Client     |
//! | 24     | 192  | OutputPacket       | Sim        |
//! | 216    | 4    | output_sequence    | Sim        |
//! | 220    | 1    | sim_running        | Sim        |
//! | 221    | 35   | reserved/padding   | --         |
//!
//! Total: 256 bytes (aligned to page-friendly boundary).
//!
//! ## Tearing Prevention
//!
//! Each side has a sequence counter that it increments AFTER writing.
//! The reader checks the sequence before and after reading: if the
//! values differ, the read was torn and should be retried.
//!
//! This is a best-effort mechanism, not a formal lock. For the expected
//! use case (one writer per section, updates at 1kHz or less), tearing
//! is extremely unlikely but handled gracefully.

use std::path::Path;

use crate::protocol::{InputPacket, OutputPacket};

/// Total size of the shared memory region (bytes).
pub const SHARED_MEM_SIZE: usize = 256;

/// Byte offset of the InputPacket region.
const INPUT_OFFSET: usize = 0;
/// Byte offset of the input sequence counter (u32 LE).
const INPUT_SEQ_OFFSET: usize = InputPacket::SIZE; // 20
/// Byte offset of the OutputPacket region.
const OUTPUT_OFFSET: usize = 24;
/// Byte offset of the output sequence counter (u32 LE).
const OUTPUT_SEQ_OFFSET: usize = OUTPUT_OFFSET + OutputPacket::SIZE; // 216
/// Byte offset of the sim_running flag (u8).
const SIM_RUNNING_OFFSET: usize = 220;

/// Read a little-endian `u32` from the four bytes of `slice` starting at `off`.
///
/// The caller guarantees `off + 4 <= slice.len()`.
fn read_u32(slice: &[u8], off: usize) -> u32 {
    let b = &slice[off..off + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Server-side shared memory handle.
///
/// The simulation server creates this to write output packets and
/// read input packets via the shared memory file.
pub struct SimSharedMem {
    mmap: memmap2::MmapMut,
}

impl SimSharedMem {
    /// Create or open the shared memory file and map it.
    ///
    /// The file is created at the given path with [`SHARED_MEM_SIZE`] bytes.
    /// If it already exists, it is truncated and remapped.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(SHARED_MEM_SIZE as u64)?;

        // SAFETY: The file is exclusively created/truncated by us,
        // and all access to the mapped region uses bounds-checked
        // byte-slice operations. Concurrent access follows the
        // sequence-counter protocol documented in the module header.
        let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        let mut mem = Self { mmap };
        mem.set_sim_running(true);
        Ok(mem)
    }

    /// Write an output packet to the shared region.
    ///
    /// Increments the output sequence counter after writing.
    pub fn write_output(&mut self, packet: &OutputPacket) {
        let bytes = packet.to_bytes();
        self.mmap[OUTPUT_OFFSET..OUTPUT_OFFSET + OutputPacket::SIZE].copy_from_slice(&bytes);

        // Increment sequence after write.
        let seq = self.read_output_sequence().wrapping_add(1);
        self.mmap[OUTPUT_SEQ_OFFSET..OUTPUT_SEQ_OFFSET + 4].copy_from_slice(&seq.to_le_bytes());
    }

    /// Read the latest input packet from the shared region.
    ///
    /// Returns None if the data cannot be parsed (should not happen
    /// with a well-behaved client).
    pub fn read_input(&self) -> Option<InputPacket> {
        InputPacket::from_bytes(&self.mmap[INPUT_OFFSET..])
    }

    /// Read the input sequence counter.
    pub fn read_input_sequence(&self) -> u32 {
        read_u32(&self.mmap, INPUT_SEQ_OFFSET)
    }

    /// Read the output sequence counter.
    fn read_output_sequence(&self) -> u32 {
        read_u32(&self.mmap, OUTPUT_SEQ_OFFSET)
    }

    /// Set the sim_running flag.
    pub fn set_sim_running(&mut self, running: bool) {
        self.mmap[SIM_RUNNING_OFFSET] = u8::from(running);
    }
}

impl Drop for SimSharedMem {
    fn drop(&mut self) {
        self.set_sim_running(false);
    }
}

/// Client-side shared memory handle.
///
/// A dashboard or driver controller opens this to read output packets
/// and write input packets via the shared memory file.
pub struct ClientSharedMem {
    mmap: memmap2::MmapMut,
}

impl ClientSharedMem {
    /// Open an existing shared memory file.
    ///
    /// Returns an error if the file does not exist or is too small.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        let metadata = file.metadata()?;
        if metadata.len() < SHARED_MEM_SIZE as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Shared memory file too small: {} bytes (expected {})",
                    metadata.len(),
                    SHARED_MEM_SIZE
                ),
            ));
        }

        // SAFETY: The file was created by SimSharedMem::create() with
        // the correct size (verified above). All access uses
        // bounds-checked byte-slice operations.
        let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };
        Ok(Self { mmap })
    }

    /// Write an input packet to the shared region.
    ///
    /// Increments the input sequence counter after writing.
    pub fn write_input(&mut self, packet: &InputPacket) {
        let bytes = packet.to_bytes();
        self.mmap[INPUT_OFFSET..INPUT_OFFSET + InputPacket::SIZE].copy_from_slice(&bytes);

        let seq = self.read_input_sequence().wrapping_add(1);
        self.mmap[INPUT_SEQ_OFFSET..INPUT_SEQ_OFFSET + 4].copy_from_slice(&seq.to_le_bytes());
    }

    /// Read the latest output packet from the shared region.
    ///
    /// Uses the sequence counter to detect torn reads. If the sequence
    /// changes during the read, returns None (caller should retry).
    pub fn read_output(&self) -> Option<OutputPacket> {
        let seq_before = self.read_output_sequence();
        let packet = OutputPacket::from_bytes(&self.mmap[OUTPUT_OFFSET..])?;
        let seq_after = self.read_output_sequence();
        if seq_before != seq_after {
            return None; // Torn read, retry.
        }
        Some(packet)
    }

    /// Read the output sequence counter.
    pub fn read_output_sequence(&self) -> u32 {
        read_u32(&self.mmap, OUTPUT_SEQ_OFFSET)
    }

    /// Read the input sequence counter.
    fn read_input_sequence(&self) -> u32 {
        read_u32(&self.mmap, INPUT_SEQ_OFFSET)
    }

    /// Check if the simulation is running.
    pub fn is_sim_running(&self) -> bool {
        self.mmap[SIM_RUNNING_OFFSET] != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique temp-file path that removes itself on drop.
    struct TempPath(std::path::PathBuf);

    impl TempPath {
        fn new(tag: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut p = std::env::temp_dir();
            p.push(format!(
                "apex14_shmem_{}_{}_{}.bin",
                tag,
                std::process::id(),
                n
            ));
            TempPath(p)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn sample_output() -> OutputPacket {
        OutputPacket {
            pos_x: 10.0,
            pos_y: 20.0,
            pos_z: 0.3,
            roll: 0.01,
            pitch: 0.02,
            yaw: 1.5,
            speed: 80.0,
            lateral_v: -2.0,
            vertical_v: 0.1,
            yaw_rate: 0.3,
            wheel_fl: 240.0,
            wheel_fr: 241.0,
            wheel_rl: 242.0,
            wheel_rr: 243.0,
            susp_fl: 0.01,
            susp_fr: 0.011,
            susp_rl: 0.012,
            susp_rr: 0.013,
            accel_long: 1.1,
            accel_lat: -2.2,
            gear: 5,
            lap: 3,
            lap_time: 42.5,
            sim_time: 130.25,
            sequence: 7,
            _pad: 0,
        }
    }

    #[test]
    fn test_create_and_size() {
        let tmp = TempPath::new("create");
        let _mem = SimSharedMem::create(tmp.path()).expect("create");
        let meta = std::fs::metadata(tmp.path()).expect("metadata");
        assert_eq!(meta.len(), SHARED_MEM_SIZE as u64);
    }

    #[test]
    fn test_write_read_output_roundtrip() {
        let tmp = TempPath::new("out_roundtrip");
        let mut sim = SimSharedMem::create(tmp.path()).expect("create");
        let packet = sample_output();
        sim.write_output(&packet);

        let client = ClientSharedMem::open(tmp.path()).expect("open");
        let got = client.read_output().expect("read output");
        assert_eq!(got, packet);
    }

    #[test]
    fn test_write_read_input_roundtrip() {
        let tmp = TempPath::new("in_roundtrip");
        let sim = SimSharedMem::create(tmp.path()).expect("create");
        let mut client = ClientSharedMem::open(tmp.path()).expect("open");

        let input = InputPacket {
            steering: -0.5,
            throttle: 0.9,
            brake: 0.1,
            gear: 4,
            sequence: 99,
        };
        client.write_input(&input);

        let got = sim.read_input().expect("read input");
        assert_eq!(got, input);
    }

    #[test]
    fn test_sequence_increments() {
        let tmp = TempPath::new("sequence");
        let mut sim = SimSharedMem::create(tmp.path()).expect("create");
        let mut client = ClientSharedMem::open(tmp.path()).expect("open");

        let out = sample_output();
        sim.write_output(&out);
        sim.write_output(&out);
        assert_eq!(client.read_output_sequence(), 2);

        let input = InputPacket::default();
        client.write_input(&input);
        client.write_input(&input);
        client.write_input(&input);
        assert_eq!(sim.read_input_sequence(), 3);
    }

    #[test]
    fn test_sim_running_flag() {
        let tmp = TempPath::new("running");
        let mut sim = SimSharedMem::create(tmp.path()).expect("create");
        let client = ClientSharedMem::open(tmp.path()).expect("open");

        // After create, the flag is set true.
        assert!(client.is_sim_running());

        // Explicitly clearing it is visible through the shared region.
        sim.set_sim_running(false);
        assert!(!client.is_sim_running());
    }

    #[test]
    fn test_client_open_missing_file() {
        let tmp = TempPath::new("missing");
        // Do not create the file.
        assert!(ClientSharedMem::open(tmp.path()).is_err());
    }

    #[test]
    fn test_client_open_too_small() {
        let tmp = TempPath::new("too_small");
        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(tmp.path())
                .expect("create small file");
            file.set_len((SHARED_MEM_SIZE - 1) as u64).expect("set_len");
        }
        let err = match ClientSharedMem::open(tmp.path()) {
            Ok(_) => panic!("open of an undersized file should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_multiple_writes() {
        let tmp = TempPath::new("multi");
        let mut sim = SimSharedMem::create(tmp.path()).expect("create");

        let mut last = sample_output();
        for i in 0..100 {
            last.sequence = i;
            last.speed = i as f64;
            sim.write_output(&last);
        }

        let client = ClientSharedMem::open(tmp.path()).expect("open");
        let got = client.read_output().expect("read output");
        assert_eq!(got.sequence, 99);
        assert_eq!(got.speed, 99.0);
        assert_eq!(client.read_output_sequence(), 100);
    }
}
