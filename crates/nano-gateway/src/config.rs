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

