//! Application configuration.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Application configuration
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
    /// Data source configuration
    #[serde(default)]
    pub data_source: DataSourceConfig,
    /// Strategy configuration
    #[serde(default)]
    pub strategy: StrategyConfig,
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
            data_source: DataSourceConfig::default(),
            strategy: StrategyConfig::default(),
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

/// Data source type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DataSourceType {
    /// Synthetic data generator
    Synthetic,
    /// Historical DBN file replay
    Historical,
}

impl Default for DataSourceType {
    fn default() -> Self {
        Self::Synthetic
    }
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// Source type: "synthetic" or "historical"
    #[serde(default)]
    pub source_type: DataSourceType,
    /// Path to DBN data file (required for historical)
    pub data_file: Option<PathBuf>,
    /// Replay speed multiplier (1.0 = real-time)
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

/// Strategy type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StrategyType {
    /// Market maker strategy (quotes bid/ask around mid)
    MarketMaker,
    /// Signal-based strategy (uses ML features)
    Signal,
}

impl Default for StrategyType {
    fn default() -> Self {
        Self::MarketMaker
    }
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Which strategy to run
    #[serde(default)]
    pub strategy_type: StrategyType,
    /// Base spread in ticks (for market maker)
    #[serde(default = "default_spread")]
    pub base_spread_ticks: i64,
    /// Order size per level
    #[serde(default = "default_order_size")]
    pub order_size: u32,
    /// Number of levels to quote (for market maker)
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
