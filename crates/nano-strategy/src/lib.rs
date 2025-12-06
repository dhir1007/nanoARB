//! # nano-strategy
//!
//! Trading strategies and RL market-making agent.
//!
//! This crate provides:
//! - Base strategy traits and implementations
//! - Market-making strategy with quote management
//! - RL environment for training market-making agents
//! - Decision Transformer and IQL agent interfaces

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
    clippy::needless_pass_by_value,
    clippy::items_after_statements,
    clippy::manual_let_else,
    dead_code,
    unused_variables
)]

pub mod base;
pub mod market_maker;
pub mod rl_env;
pub mod signals;

pub use base::{BaseStrategy, StrategyState};
pub use market_maker::{MarketMakerConfig, MarketMakerStrategy, QuoteManager};
pub use rl_env::{MarketMakingAction, MarketMakingEnv, MarketMakingState};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::base::{BaseStrategy, StrategyState};
    pub use crate::market_maker::{MarketMakerConfig, MarketMakerStrategy};
    pub use crate::rl_env::{MarketMakingAction, MarketMakingEnv, MarketMakingState};
}
