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
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::if_same_then_else,
    clippy::comparison_chain,
    clippy::manual_memcpy
)]

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
