//! # nano-lob
//!
//! Limit Order Book reconstruction and feature extraction for HFT.
//!
//! This crate provides:
//! - High-performance order book data structure (20 levels)
//! - Feature extraction: microprice, OFI, VPIN, book imbalance
//! - LOB tensor serialization for ML models
//!
//! ## Example
//!
//! ```rust,ignore
//! use nano_lob::orderbook::OrderBook;
//! use nano_lob::features::LobFeatureExtractor;
//!
//! let mut book = OrderBook::new(1);
//! book.apply_update(&market_data);
//!
//! let features = LobFeatureExtractor::new(&book);
//! let microprice = features.microprice();
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod features;
pub mod orderbook;
pub mod snapshot;

pub use features::LobFeatureExtractor;
pub use orderbook::OrderBook;
pub use snapshot::{LobSnapshot, SnapshotRingBuffer};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::features::LobFeatureExtractor;
    pub use crate::orderbook::OrderBook;
    pub use crate::snapshot::{LobSnapshot, SnapshotRingBuffer};
}

