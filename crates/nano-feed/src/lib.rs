//! # nano-feed
//!
//! CME MDP 3.0 market data feed parser and handler.
//!
//! This crate provides:
//! - Binary message parsing for CME MDP 3.0 protocol (SBE encoding)
//! - Message types for incremental book updates, trades, and channel resets
//! - Zero-copy parsing for minimal latency
//!
//! ## CME MDP 3.0 Overview
//!
//! The CME Globex Market Data Platform (MDP) 3.0 uses Simple Binary Encoding (SBE)
//! for efficient message serialization. Key message types include:
//!
//! - MDIncrementalRefreshBook (Template ID 46): Order book updates
//! - MDIncrementalRefreshTrade (Template ID 42): Trade messages
//! - ChannelReset (Template ID 4): Channel state reset
//! - SecurityStatus (Template ID 30): Instrument status changes
//!
//! ## Example
//!
//! ```rust,ignore
//! use nano_feed::parser::MdpParser;
//! use nano_feed::messages::MdpMessage;
//!
//! let mut parser = MdpParser::new();
//! let message = parser.parse(&raw_bytes)?;
//!
//! match message {
//!     MdpMessage::BookUpdate(update) => {
//!         // Handle book update
//!     }
//!     MdpMessage::Trade(trade) => {
//!         // Handle trade
//!     }
//!     _ => {}
//! }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

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

