//! Error types for the feed parser.

use thiserror::Error;

/// Feed parsing and handling errors
#[derive(Error, Debug, Clone)]
pub enum FeedError {
    /// Invalid message header
    #[error("Invalid message header: {0}")]
    InvalidHeader(String),

    /// Unknown message type
    #[error("Unknown message type: template_id={0}")]
    UnknownMessageType(u16),

    /// Invalid message length
    #[error("Invalid message length: expected {expected}, got {actual}")]
    InvalidLength {
        /// Expected length
        expected: usize,
        /// Actual length
        actual: usize,
    },

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Incomplete message
    #[error("Incomplete message: need {needed} more bytes")]
    Incomplete {
        /// Number of bytes needed
        needed: usize,
    },

    /// Invalid field value
    #[error("Invalid field value: {field} = {value}")]
    InvalidField {
        /// Field name
        field: String,
        /// Field value
        value: String,
    },

    /// Sequence gap detected
    #[error("Sequence gap: expected {expected}, got {actual}")]
    SequenceGap {
        /// Expected sequence number
        expected: u32,
        /// Actual sequence number
        actual: u32,
    },

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(String),

    /// End of file
    #[error("End of file")]
    Eof,
}

/// Result type for feed operations
pub type FeedResult<T> = Result<T, FeedError>;

impl From<std::io::Error> for FeedError {
    fn from(err: std::io::Error) -> Self {
        FeedError::IoError(err.to_string())
    }
}

impl From<nom::Err<nom::error::Error<&[u8]>>> for FeedError {
    fn from(err: nom::Err<nom::error::Error<&[u8]>>) -> Self {
        match err {
            nom::Err::Incomplete(needed) => FeedError::Incomplete {
                needed: match needed {
                    nom::Needed::Unknown => 1,
                    nom::Needed::Size(n) => n.get(),
                },
            },
            nom::Err::Error(e) | nom::Err::Failure(e) => {
                FeedError::ParseError(format!("at position: {:?}", e.input.len()))
            }
        }
    }
}
