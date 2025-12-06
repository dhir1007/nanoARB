//! # nano-backtest
//!
//! Event-driven backtesting engine with realistic latency simulation.
//!
//! This crate provides:
//! - Event-driven backtest engine with priority queue
//! - Realistic latency and fill models
//! - Walk-forward and purged cross-validation
//! - P&L tracking and risk management
//!
//! ## Example
//!
//! ```rust,ignore
//! use nano_backtest::engine::BacktestEngine;
//! use nano_backtest::config::BacktestConfig;
//!
//! let config = BacktestConfig::default();
//! let mut engine = BacktestEngine::new(config);
//!
//! engine.run(&data, &mut strategy)?;
//! let results = engine.results();
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::items_after_statements,
    clippy::manual_let_else,
    dead_code,
    unused_variables
)]

pub mod config;
pub mod engine;
pub mod events;
pub mod execution;
pub mod latency;
pub mod metrics;
pub mod position;
pub mod risk;
pub mod validation;

pub use config::BacktestConfig;
pub use engine::BacktestEngine;
pub use events::{Event, EventQueue};
pub use execution::{FillSimulator, SimulatedExchange};
pub use latency::LatencySimulator;
pub use metrics::{BacktestMetrics, PerformanceStats};
pub use position::PositionTracker;
pub use risk::RiskManager;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::config::BacktestConfig;
    pub use crate::engine::BacktestEngine;
    pub use crate::events::{Event, EventQueue};
    pub use crate::metrics::{BacktestMetrics, PerformanceStats};
    pub use crate::position::PositionTracker;
    pub use crate::risk::RiskManager;
}
