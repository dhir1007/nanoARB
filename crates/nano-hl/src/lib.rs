//! nano-hl: Hyperliquid adapter for the nanoARB paper-trading engine.
//!
//! Connects to Hyperliquid's WebSocket API, converts the l2Book feed into
//! the internal BookUpdate format, and drives the existing MarketMakerStrategy
//! in paper-trading mode.

pub mod converter;
pub mod engine;
pub mod feed;
pub mod metrics;
pub mod paper_trader;