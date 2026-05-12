//! # nano-feed
//!
//! Market data feed layer — CME MDP 3.0 parser, synthetic generator, and
//! Databento DBN replay.
//!
//! This crate provides:
//! - Binary message parsing for CME MDP 3.0 protocol (SBE encoding)
//! - Message types for incremental book updates, trades, and channel resets
//! - Zero-copy parsing for minimal latency
//! - **Synthetic data generator** for demos and testing
//! - **Databento DBN adapter** (`dbn_adapter`) — converts `Mbp10Msg` records
//!   into the internal `MdpMessage` types the order book consumes
//! - **Unified `DataSource` trait** (`data_source`) — lets the engine consume
//!   market data identically whether it comes from the synthetic generator or
//!   a historical `.dbn.zst` file
//!
//! ## CME MDP 3.0 Overview
//!
//! The CME Globex Market Data Platform (MDP) 3.0 uses Simple Binary Encoding (SBE)
//! for efficient message serialization. Key message types include:
//!
//! - `MDIncrementalRefreshBook` (Template ID 46): Order book updates
//! - `MDIncrementalRefreshTrade` (Template ID 42): Trade messages
//! - `ChannelReset` (Template ID 4): Channel state reset
//! - `SecurityStatus` (Template ID 30): Instrument status changes
//!
//! ## Example — Live Parser
//!
//! ```rust,ignore
//! use nano_feed::parser::MdpParser;
//! use nano_feed::messages::MdpMessage;
//!
//! let mut parser = MdpParser::new();
//! let message = parser.parse(&raw_bytes)?;
//!
//! match message {
//!     MdpMessage::BookUpdate(update) => { /* ... */ }
//!     MdpMessage::Trade(trade) => { /* ... */ }
//!     _ => {}
//! }
//! ```
//!
//! ## Example — Historical Replay
//!
//! ```rust,ignore
//! use nano_feed::data_source::{DbnReplaySource, DataSource};
//!
//! let mut source = DbnReplaySource::open("data/ESH5_2025-01-06.dbn.zst", 1.0, true)?;
//! while let Some(msg) = source.next_event() {
//!     // msg is an MdpMessage at real-time pace
//! }
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
    clippy::bool_to_int_with_if,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::if_same_then_else,
    unexpected_cfgs
)]

pub mod error;
pub mod messages;
pub mod parser;
pub mod reader;
pub mod synthetic;

pub use error::{FeedError, FeedResult};
pub use messages::*;
pub use parser::MdpParser;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::error::{FeedError, FeedResult};
    pub use crate::messages::*;
    pub use crate::parser::MdpParser;
}
