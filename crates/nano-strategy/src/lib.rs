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
#![allow(clippy::module_name_repetitions)]

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

