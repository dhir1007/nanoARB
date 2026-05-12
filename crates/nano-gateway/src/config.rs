//! Application configuration for the NanoARB trading engine.
//!
//! Supports two operating modes:
//! - **Synthetic**: Infinite stream of synthetic ES futures data (default).
//! - **Historical**: Replay Databento DBN files at real-time pace for paper trading.
//!
//! Configuration is loaded from a TOML file (default `config.toml`) or from
//! environment variables. All fields have sensible defaults so the engine runs
//! out-of-the-box without any config file.
//!
//! # Example `config.toml`
//!
//! ```toml
//! [data_source]
//! source_type = "historical"
//! data_file   = "data/ESH5_2025-01-06.dbn.zst"
//! replay_speed = 1.0
//!
//! [strategy]
//! strategy_type    = "market_maker"
//! base_spread_ticks = 2
//! order_size        = 5
//! num_levels        = 3
//!
//! [trading]
//! max_position  = 50
//! max_order_size = 10
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level application configuration.
///
/// Composed of sub-configs for trading limits, data source selection,
/// and strategy parameters. Serializes to/from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Application name
    pub name: String,
    /// Log level
    pub log_level: String,
    /// Metrics port
    pub metrics_port: u16,
    /// Data directory
    pub data_dir: PathBuf,
    /// Model path
    pub model_path: Option<PathBuf>,
    /// Trading configuration
    pub trading: TradingConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: "nanoarb".to_string(),
            log_level: "info".to_string(),
            metrics_port: 9090,
            data_dir: PathBuf::from("data"),
            model_path: None,
            trading: TradingConfig::default(),
        }
    }
}

/// Trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Enable live trading
    pub live_enabled: bool,
    /// Instrument symbols to trade
    pub symbols: Vec<String>,
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum position per instrument
    pub max_position: i64,
    /// Maximum order size
    pub max_order_size: u32,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            live_enabled: false,
            symbols: vec!["ESH24".to_string()],
            initial_capital: 1_000_000.0,
            max_position: 50,
            max_order_size: 10,
        }
    }
}

/// Selects where market data comes from.
///
/// - `Synthetic` generates infinite fake ES futures data (good for demos).
/// - `Historical` replays a Databento `.dbn` / `.dbn.zst` file through the
///   real order book, pacing events at their original inter-arrival times.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DataSourceType {
    /// Infinite synthetic data generator (no external files needed).
    Synthetic,
    /// Historical replay from a Databento DBN file.
    Historical,
}

impl Default for DataSourceType {
    fn default() -> Self {
        Self::Synthetic
    }
}

/// Controls where market data comes from and how fast it replays.
///
/// For historical mode, `data_file` must point to a valid `.dbn` or `.dbn.zst`
/// file downloaded with the `dbn-download` tool. The `replay_speed` multiplier
/// controls playback rate (1.0 = real-time, 2.0 = 2× speed, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// `"synthetic"` (default) or `"historical"`.
    #[serde(default)]
    pub source_type: DataSourceType,
    /// Path to a Databento `.dbn` or `.dbn.zst` file. Required when
    /// `source_type` is `Historical`. Ignored for `Synthetic`.
    pub data_file: Option<PathBuf>,
    /// Playback speed multiplier. 1.0 = real-time, 10.0 = 10× faster.
    /// Only applies to historical replay; synthetic mode uses a fixed 150ms tick.
    #[serde(default = "default_replay_speed")]
    pub replay_speed: f64,
}

fn default_replay_speed() -> f64 {
    1.0
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            source_type: DataSourceType::Synthetic,
            data_file: None,
            replay_speed: 1.0,
        }
    }
}

/// Which trading strategy the paper-trading engine should run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StrategyType {
    /// Passive market maker: quotes symmetric bid/ask around mid price,
    /// skewed by inventory to mean-revert position.
    MarketMaker,
    /// Directional signal strategy: uses order-book imbalance features
    /// to take aggressive positions.
    Signal,
}

impl Default for StrategyType {
    fn default() -> Self {
        Self::MarketMaker
    }
}

/// Parameters for the chosen trading strategy.
///
/// The `MarketMaker` strategy places passive limit orders on both sides of the
/// book. `base_spread_ticks` controls how wide the quotes are, `order_size` is
/// the number of contracts per level, and `num_levels` is how many price levels
/// deep to quote (e.g. 3 = quote at best, best-1, best-2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Which strategy to run. Default: `MarketMaker`.
    #[serde(default)]
    pub strategy_type: StrategyType,
    /// Half-spread width in ticks. ES tick = $0.25, so `2` = $0.50 half-spread.
    #[serde(default = "default_spread")]
    pub base_spread_ticks: i64,
    /// Number of contracts per quote level.
    #[serde(default = "default_order_size")]
    pub order_size: u32,
    /// How many price levels deep to quote on each side.
    #[serde(default = "default_num_levels")]
    pub num_levels: usize,
}

fn default_spread() -> i64 {
    2
}
fn default_order_size() -> u32 {
    5
}
fn default_num_levels() -> usize {
    3
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            strategy_type: StrategyType::MarketMaker,
            base_spread_ticks: 2,
            order_size: 5,
            num_levels: 3,
        }
    }
}

impl AppConfig {
    /// Load configuration from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load from environment with fallback to file
    pub fn from_env() -> anyhow::Result<Self> {
        if let Ok(path) = std::env::var("NANOARB_CONFIG") {
            Self::load(path)
        } else {
            Ok(Self::default())
        }
    }

    /// Save configuration to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
