//! # nano-model
//!
//! ML model inference for trading signals with sub-microsecond latency.
//!
//! This crate provides:
//! - ONNX model loading and inference via `ort`
//! - Feature preprocessing for LOB data
//! - Signal generation from model predictions
//!
//! ## Example
//!
//! ```rust,ignore
//! use nano_model::inference::OnnxModel;
//! use nano_model::signal::SignalGenerator;
//!
//! let model = OnnxModel::load("model.onnx")?;
//! let features = extractor.to_array(&book);
//! let prediction = model.predict(&features)?;
//! ```

#![warn(missing_docs, rust_2018_idioms, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod inference;
pub mod preprocessing;
pub mod signal;

pub use inference::{ModelConfig, OnnxModel, Prediction};
pub use preprocessing::FeaturePreprocessor;
pub use signal::{Signal, SignalGenerator};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::inference::{ModelConfig, OnnxModel, Prediction};
    pub use crate::preprocessing::FeaturePreprocessor;
    pub use crate::signal::{Signal, SignalGenerator};
}

