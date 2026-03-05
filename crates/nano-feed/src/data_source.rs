//! Unified data source abstraction for market data.
//!
//! Provides a `DataSource` trait that both the `SyntheticGenerator` and the
//! `DbnReplaySource` implement, allowing the simulation loop to be agnostic
//! about where market data comes from.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::dbn_adapter::DbnFileReader;
use crate::messages::MdpMessage;
use crate::synthetic::{SyntheticConfig, SyntheticGenerator};

/// Unified interface for market data sources.
pub trait DataSource: Send {
    /// Get the next market data event.
    /// Returns `None` when the source is exhausted (e.g. end of file).
    fn next_event(&mut self) -> Option<MdpMessage>;

    /// Human-readable label for this source (e.g. "Synthetic ES" or "Replay ESH25 2025-01-06").
    fn label(&self) -> &str;

    /// Whether this source has a finite end (historical replay) vs infinite (synthetic).
    fn is_finite(&self) -> bool;

    /// Total number of records if known (for progress tracking).
    fn total_records(&self) -> Option<u64> {
        None
    }

    /// Number of records consumed so far.
    fn records_consumed(&self) -> u64;
}

// ── SyntheticGenerator as DataSource ──

impl DataSource for SyntheticGenerator {
    fn next_event(&mut self) -> Option<MdpMessage> {
        Some(self.next_event())
    }

    fn label(&self) -> &str {
        "Synthetic ES Futures"
    }

    fn is_finite(&self) -> bool {
        false
    }

    fn records_consumed(&self) -> u64 {
        0
    }
}

/// Historical replay source that reads DBN files and emits events at real-time pace.
pub struct DbnReplaySource {
    reader: DbnFileReader,
    label: String,
    records_consumed: u64,
    /// Timestamp of the first record (nanoseconds)
    first_ts_ns: Option<u64>,
    /// Timestamp of the previous record (nanoseconds)
    prev_ts_ns: Option<u64>,
    /// Wall-clock time at first record
    wall_start: Option<Instant>,
    /// Playback speed multiplier (1.0 = real-time)
    speed: f64,
    /// Whether to pace events in real-time
    realtime_pacing: bool,
}

impl DbnReplaySource {
    /// Create a new replay source from a DBN file path.
    ///
    /// If `realtime_pacing` is true, events are emitted at real-time speed
    /// (adjusted by `speed` multiplier). Otherwise events are emitted as fast
    /// as possible.
    pub fn open(path: impl AsRef<Path>, speed: f64, realtime_pacing: bool) -> anyhow::Result<Self> {
        let p = path.as_ref();
        let file_name = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let label = format!("Replay: {file_name}");

        let reader = if p.extension().and_then(|e| e.to_str()) == Some("zst")
            || p.to_str().map_or(false, |s| s.ends_with(".dbn.zst"))
        {
            DbnFileReader::open_zstd(p)?
        } else {
            DbnFileReader::open(p)?
        };

        Ok(Self {
            reader,
            label,
            records_consumed: 0,
            first_ts_ns: None,
            prev_ts_ns: None,
            wall_start: None,
            speed: speed.max(0.01),
            realtime_pacing,
        })
    }

    /// Wait the appropriate amount of time to maintain real-time pacing.
    fn pace(&mut self, event_ts_ns: u64) {
        if !self.realtime_pacing {
            return;
        }

        let first_ts = match self.first_ts_ns {
            Some(ts) => ts,
            None => {
                self.first_ts_ns = Some(event_ts_ns);
                self.wall_start = Some(Instant::now());
                return;
            }
        };

        let wall_start = self.wall_start.unwrap();
        let data_elapsed_ns = event_ts_ns.saturating_sub(first_ts);
        let target_wall_elapsed = Duration::from_nanos((data_elapsed_ns as f64 / self.speed) as u64);
        let actual_wall_elapsed = wall_start.elapsed();

        if target_wall_elapsed > actual_wall_elapsed {
            std::thread::sleep(target_wall_elapsed - actual_wall_elapsed);
        }

        self.prev_ts_ns = Some(event_ts_ns);
    }
}

impl DataSource for DbnReplaySource {
    fn next_event(&mut self) -> Option<MdpMessage> {
        let msg = match self.reader.next_message() {
            Ok(Some(msg)) => msg,
            Ok(None) => return None,
            Err(e) => {
                tracing::warn!("DBN read error: {e}");
                return None;
            }
        };

        self.records_consumed += 1;

        if let Some(ts) = msg.timestamp() {
            self.pace(ts.as_nanos() as u64);
        }

        Some(msg)
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn is_finite(&self) -> bool {
        true
    }

    fn records_consumed(&self) -> u64 {
        self.records_consumed
    }
}

/// Create a data source from configuration.
pub fn create_data_source(
    source_type: &str,
    data_file: Option<&str>,
    replay_speed: f64,
) -> anyhow::Result<Box<dyn DataSource>> {
    match source_type {
        "historical" | "replay" => {
            let path = data_file
                .ok_or_else(|| anyhow::anyhow!("data_file required for historical replay"))?;
            let source = DbnReplaySource::open(path, replay_speed, true)?;
            Ok(Box::new(source))
        }
        _ => {
            let config = SyntheticConfig::es_futures();
            let generator = SyntheticGenerator::new(config);
            Ok(Box::new(generator))
        }
    }
}
