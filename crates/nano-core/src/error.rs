//! Error types for the NanoARB trading engine.

use thiserror::Error;

/// Core error type for the trading engine
#[derive(Error, Debug, Clone)]
pub enum Error {
    /// Invalid price value
    #[error("Invalid price: {0}")]
    InvalidPrice(String),

    /// Invalid quantity value
    #[error("Invalid quantity: {0}")]
    InvalidQuantity(String),

    /// Invalid order ID
    #[error("Invalid order ID: {0}")]
    InvalidOrderId(String),

    /// Invalid timestamp
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// Invalid instrument
    #[error("Invalid instrument: {0}")]
    InvalidInstrument(String),

    /// Order not found
    #[error("Order not found: {0}")]
    OrderNotFound(u64),

    /// Insufficient liquidity
    #[error("Insufficient liquidity at price {price} for quantity {quantity}")]
    InsufficientLiquidity {
        /// The price level
        price: String,
        /// The requested quantity
        quantity: String,
    },

    /// Risk limit exceeded
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Model inference error
    #[error("Model inference error: {0}")]
    ModelError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::SerializationError(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidPrice("negative value".to_string());
        assert_eq!(err.to_string(), "Invalid price: negative value");
    }

    #[test]
    fn test_insufficient_liquidity_display() {
        let err = Error::InsufficientLiquidity {
            price: "100.50".to_string(),
            quantity: "1000".to_string(),
        };
        assert!(err.to_string().contains("Insufficient liquidity"));
    }
}

