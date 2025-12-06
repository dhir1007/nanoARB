//! # nano-core
//!
//! Core types, traits, and utilities for the NanoARB high-frequency trading engine.
//!
//! This crate provides:
//! - Domain types: `Price`, `Quantity`, `Side`, `OrderId`, `Timestamp`
//! - Zero-copy serialization support via `rkyv`
//! - Common traits for the trading system
//!
//! ## Example
//!
//! ```rust
//! use nano_core::types::{Price, Quantity, Side, Timestamp};
//!
//! let price = Price::from_ticks(50000, 2); // $500.00
//! let qty = Quantity::new(100);
//! let side = Side::Buy;
//! let ts = Timestamp::now();
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod constants;
pub mod error;
pub mod traits;
pub mod types;

pub use constants::*;
pub use error::{Error, Result};
pub use traits::*;
pub use types::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::constants::*;
    pub use crate::error::{Error, Result};
    pub use crate::traits::*;
    pub use crate::types::*;
}

