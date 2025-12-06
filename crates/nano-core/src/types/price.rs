//! Price type with fixed-point precision for HFT applications.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Fixed-point price representation for nanosecond-level trading.
///
/// Uses i64 internally to represent prices in the smallest tick unit.
/// This avoids floating-point errors and ensures deterministic arithmetic.
///
/// # Example
///
/// ```rust
/// use nano_core::types::Price;
///
/// // Create a price of $500.25 with 2 decimal places (tick = 0.01)
/// let price = Price::from_ticks(50025, 2);
/// assert_eq!(price.as_f64(), 500.25);
///
/// // Create from raw tick value
/// let raw = Price::from_raw(50025);
/// assert_eq!(raw.raw(), 50025);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct Price(i64);

impl Price {
    /// Zero price constant
    pub const ZERO: Self = Self(0);

    /// Maximum price constant
    pub const MAX: Self = Self(i64::MAX);

    /// Minimum price constant
    pub const MIN: Self = Self(i64::MIN);

    /// Create a price from raw tick value
    #[inline]
    #[must_use]
    pub const fn from_raw(ticks: i64) -> Self {
        Self(ticks)
    }

    /// Create a price from ticks and decimal places
    ///
    /// # Arguments
    ///
    /// * `value` - The price value in smallest units
    /// * `decimals` - Number of decimal places (for display purposes)
    #[inline]
    #[must_use]
    pub const fn from_ticks(value: i64, _decimals: u8) -> Self {
        Self(value)
    }

    /// Create a price from a floating-point value with specified tick size
    #[inline]
    #[must_use]
    pub fn from_f64(value: f64, tick_size: f64) -> Self {
        Self((value / tick_size).round() as i64)
    }

    /// Get the raw tick value
    #[inline]
    #[must_use]
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Convert to f64 (assumes tick size of 0.01)
    #[inline]
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.0 as f64 / 100.0
    }

    /// Convert to f64 with specified tick size
    #[inline]
    #[must_use]
    pub fn as_f64_with_tick(self, tick_size: f64) -> f64 {
        self.0 as f64 * tick_size
    }

    /// Check if the price is zero
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Check if the price is positive
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Check if the price is negative
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Get the absolute value
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Saturating addition
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction
    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Checked addition
    #[inline]
    #[must_use]
    pub const fn checked_add(self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked subtraction
    #[inline]
    #[must_use]
    pub const fn checked_sub(self, other: Self) -> Option<Self> {
        match self.0.checked_sub(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }
}

impl PartialOrd for Price {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Price {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Add for Price {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Price {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul<i64> for Price {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: i64) -> Self {
        Self(self.0 * scalar)
    }
}

impl Div<i64> for Price {
    type Output = Self;

    #[inline]
    fn div(self, scalar: i64) -> Self {
        Self(self.0 / scalar)
    }
}

impl fmt::Debug for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Price({})", self.0)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.as_f64())
    }
}

// Note: zerocopy traits are implemented via derive macros in the struct definition
// using #[derive(zerocopy::AsBytes, zerocopy::FromBytes, zerocopy::FromZeroes)]
// when zerocopy-derive feature is enabled. For now, we rely on rkyv for zero-copy.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_creation() {
        let p = Price::from_raw(50025);
        assert_eq!(p.raw(), 50025);
        assert_eq!(p.as_f64(), 500.25);
    }

    #[test]
    fn test_price_from_f64() {
        let p = Price::from_f64(500.25, 0.01);
        assert_eq!(p.raw(), 50025);
    }

    #[test]
    fn test_price_arithmetic() {
        let p1 = Price::from_raw(100);
        let p2 = Price::from_raw(50);

        assert_eq!((p1 + p2).raw(), 150);
        assert_eq!((p1 - p2).raw(), 50);
        assert_eq!((p1 * 2).raw(), 200);
        assert_eq!((p1 / 2).raw(), 50);
    }

    #[test]
    fn test_price_comparison() {
        let p1 = Price::from_raw(100);
        let p2 = Price::from_raw(50);
        let p3 = Price::from_raw(100);

        assert!(p1 > p2);
        assert!(p2 < p1);
        assert_eq!(p1, p3);
    }

    #[test]
    fn test_price_saturating() {
        let max = Price::MAX;
        let one = Price::from_raw(1);

        assert_eq!(max.saturating_add(one), Price::MAX);
    }

    #[test]
    fn test_price_display() {
        let p = Price::from_raw(50025);
        assert_eq!(format!("{p}"), "500.25");
    }
}

