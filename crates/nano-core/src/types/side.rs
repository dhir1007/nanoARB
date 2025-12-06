//! Side (Buy/Sell) type for orders and trades.

use std::fmt;
use std::ops::Not;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Order/Trade side (Buy or Sell)
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
#[archive(check_bytes)]
#[repr(u8)]
#[derive(Default)]
pub enum Side {
    /// Buy side (bid)
    #[default]
    Buy = 0,
    /// Sell side (ask/offer)
    Sell = 1,
}

impl Side {
    /// Check if this is a buy side
    #[inline]
    #[must_use]
    pub const fn is_buy(self) -> bool {
        matches!(self, Side::Buy)
    }

    /// Check if this is a sell side
    #[inline]
    #[must_use]
    pub const fn is_sell(self) -> bool {
        matches!(self, Side::Sell)
    }

    /// Get the opposite side
    #[inline]
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    /// Convert to a sign multiplier (1 for buy, -1 for sell)
    #[inline]
    #[must_use]
    pub const fn sign(self) -> i64 {
        match self {
            Side::Buy => 1,
            Side::Sell => -1,
        }
    }

    /// Convert to a sign multiplier as f64
    #[inline]
    #[must_use]
    pub const fn sign_f64(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }

    /// Create from a boolean (true = buy, false = sell)
    #[inline]
    #[must_use]
    pub const fn from_is_buy(is_buy: bool) -> Self {
        if is_buy {
            Side::Buy
        } else {
            Side::Sell
        }
    }

    /// Create from u8 (0 = buy, 1 = sell)
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Side::Buy),
            1 => Some(Side::Sell),
            _ => None,
        }
    }

    /// Convert to u8
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

impl Not for Side {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        self.opposite()
    }
}

impl fmt::Debug for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "Buy"),
            Side::Sell => write!(f, "Sell"),
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

impl TryFrom<u8> for Side {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Side::from_u8(value).ok_or("Invalid side value")
    }
}

impl From<Side> for u8 {
    fn from(side: Side) -> Self {
        side.as_u8()
    }
}

// Note: zerocopy traits are implemented via rkyv for zero-copy serialization.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
        assert_eq!(!Side::Buy, Side::Sell);
    }

    #[test]
    fn test_side_sign() {
        assert_eq!(Side::Buy.sign(), 1);
        assert_eq!(Side::Sell.sign(), -1);
    }

    #[test]
    fn test_side_from_u8() {
        assert_eq!(Side::from_u8(0), Some(Side::Buy));
        assert_eq!(Side::from_u8(1), Some(Side::Sell));
        assert_eq!(Side::from_u8(2), None);
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Buy), "BUY");
        assert_eq!(format!("{}", Side::Sell), "SELL");
    }
}
