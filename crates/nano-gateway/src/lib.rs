//! # nano-gateway
//!
//! Order management, execution gateway, and monitoring infrastructure.
//!
//! This crate provides:
//! - Main entry point for the trading engine
//! - Prometheus metrics export
//! - Configuration management
//! - Health checks and monitoring

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
    dead_code,
    unused_variables
)]

pub mod config;
pub mod metrics;
pub mod server;

pub use config::AppConfig;
pub use metrics::MetricsRegistry;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::config::AppConfig;
    pub use crate::metrics::MetricsRegistry;
}
