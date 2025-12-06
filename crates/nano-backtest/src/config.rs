//! Backtest configuration.

use nano_core::constants::{
    CME_CLEARING_FEE, CME_EXCHANGE_FEE, CME_MAKER_FEE, CME_TAKER_FEE,
    DEFAULT_COLO_LATENCY_NS, DEFAULT_JITTER_NS, DEFAULT_MAX_DRAWDOWN_PCT,
    DEFAULT_MAX_INVENTORY,
};
use serde::{Deserialize, Serialize};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital in dollars
    pub initial_capital: f64,

    /// Latency configuration
    pub latency: LatencyConfig,

    /// Fee configuration
    pub fees: FeeConfig,

    /// Risk management configuration
    pub risk: RiskConfig,

    /// Execution configuration
    pub execution: ExecutionConfig,

    /// Output configuration
    pub output: OutputConfig,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 1_000_000.0,
            latency: LatencyConfig::default(),
            fees: FeeConfig::default(),
            risk: RiskConfig::default(),
            execution: ExecutionConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl BacktestConfig {
    /// Create a configuration for aggressive HFT backtesting
    #[must_use]
    pub fn aggressive_hft() -> Self {
        Self {
            initial_capital: 1_000_000.0,
            latency: LatencyConfig {
                order_latency_ns: 50_000,
                market_data_latency_ns: 10_000,
                ack_latency_ns: 60_000,
                jitter_ns: 5_000,
                use_random_jitter: true,
            },
            fees: FeeConfig::default(),
            risk: RiskConfig {
                max_position: 50,
                max_order_size: 10,
                max_drawdown_pct: 0.04,
                max_daily_loss: 50_000.0,
                max_open_orders: 10,
                enable_kill_switch: true,
            },
            execution: ExecutionConfig::default(),
            output: OutputConfig::default(),
        }
    }

    /// Create a conservative configuration for strategy validation
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            initial_capital: 1_000_000.0,
            latency: LatencyConfig {
                order_latency_ns: 200_000,
                market_data_latency_ns: 50_000,
                ack_latency_ns: 250_000,
                jitter_ns: 50_000,
                use_random_jitter: true,
            },
            fees: FeeConfig {
                maker_fee: CME_MAKER_FEE * 1.2, // Add buffer
                taker_fee: CME_TAKER_FEE * 1.2,
                ..Default::default()
            },
            risk: RiskConfig {
                max_position: 20,
                max_drawdown_pct: 0.03,
                ..Default::default()
            },
            execution: ExecutionConfig {
                fill_probability_decay: 0.7, // More conservative fills
                partial_fill_probability: 0.4,
                ..Default::default()
            },
            output: OutputConfig::default(),
        }
    }
}

/// Latency simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    /// Order submission latency in nanoseconds
    pub order_latency_ns: u64,

    /// Market data latency in nanoseconds
    pub market_data_latency_ns: u64,

    /// Order acknowledgment latency in nanoseconds
    pub ack_latency_ns: u64,

    /// Latency jitter (standard deviation) in nanoseconds
    pub jitter_ns: u64,

    /// Whether to use random jitter
    pub use_random_jitter: bool,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            order_latency_ns: DEFAULT_COLO_LATENCY_NS,
            market_data_latency_ns: DEFAULT_COLO_LATENCY_NS / 2,
            ack_latency_ns: DEFAULT_COLO_LATENCY_NS,
            jitter_ns: DEFAULT_JITTER_NS,
            use_random_jitter: true,
        }
    }
}

/// Fee configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeConfig {
    /// Maker fee per contract
    pub maker_fee: f64,

    /// Taker fee per contract
    pub taker_fee: f64,

    /// Exchange fee per contract
    pub exchange_fee: f64,

    /// Clearing fee per contract
    pub clearing_fee: f64,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            maker_fee: CME_MAKER_FEE,
            taker_fee: CME_TAKER_FEE,
            exchange_fee: CME_EXCHANGE_FEE,
            clearing_fee: CME_CLEARING_FEE,
        }
    }
}

impl FeeConfig {
    /// Calculate total fee for a trade
    #[must_use]
    pub fn calculate_fee(&self, quantity: u32, is_maker: bool) -> f64 {
        let base_fee = if is_maker {
            self.maker_fee
        } else {
            self.taker_fee
        };
        (base_fee + self.exchange_fee + self.clearing_fee) * quantity as f64
    }
}

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size (contracts)
    pub max_position: i64,

    /// Maximum single order size
    pub max_order_size: u32,

    /// Maximum drawdown percentage before kill switch
    pub max_drawdown_pct: f64,

    /// Maximum daily loss before stop
    pub max_daily_loss: f64,

    /// Maximum number of open orders
    pub max_open_orders: usize,

    /// Enable kill switch
    pub enable_kill_switch: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position: DEFAULT_MAX_INVENTORY,
            max_order_size: 20,
            max_drawdown_pct: DEFAULT_MAX_DRAWDOWN_PCT,
            max_daily_loss: 100_000.0,
            max_open_orders: 20,
            enable_kill_switch: true,
        }
    }
}

/// Execution simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Queue position tracking enabled
    pub track_queue_position: bool,

    /// Fill probability decay with queue position
    pub fill_probability_decay: f64,

    /// Probability of partial fills
    pub partial_fill_probability: f64,

    /// Minimum partial fill size
    pub min_partial_fill: u32,

    /// Whether to simulate adverse selection
    pub simulate_adverse_selection: bool,

    /// Adverse selection probability
    pub adverse_selection_prob: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            track_queue_position: true,
            fill_probability_decay: 0.9,
            partial_fill_probability: 0.2,
            min_partial_fill: 1,
            simulate_adverse_selection: true,
            adverse_selection_prob: 0.1,
        }
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Log level verbosity
    pub verbosity: u8,

    /// Record tick-by-tick P&L
    pub record_tick_pnl: bool,

    /// Record all fills
    pub record_fills: bool,

    /// Record order history
    pub record_orders: bool,

    /// Snapshot interval (in events)
    pub snapshot_interval: usize,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            verbosity: 1,
            record_tick_pnl: false,
            record_fills: true,
            record_orders: true,
            snapshot_interval: 10000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_capital, 1_000_000.0);
        assert!(config.risk.max_position > 0);
    }

    #[test]
    fn test_fee_calculation() {
        let fees = FeeConfig::default();

        let maker_fee = fees.calculate_fee(10, true);
        let taker_fee = fees.calculate_fee(10, false);

        assert!(taker_fee > maker_fee);
    }

    #[test]
    fn test_aggressive_config() {
        let config = BacktestConfig::aggressive_hft();
        assert!(config.latency.order_latency_ns < 100_000);
        assert!(config.risk.max_position < 100);
    }
}

