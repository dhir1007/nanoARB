//! Quantity type for order and position sizes.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Quantity representation for orders and positions.
///
/// Uses u32 internally to represent the number of contracts/shares.
/// For signed positions, use `SignedQuantity`.
///
/// # Example
///
/// ```rust
/// use nano_core::types::Quantity;
///
/// let qty = Quantity::new(100);
/// assert_eq!(qty.value(), 100);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct Quantity(u32);

impl Quantity {
    /// Zero quantity constant
    pub const ZERO: Self = Self(0);

    /// Maximum quantity constant
    pub const MAX: Self = Self(u32::MAX);

    /// Create a new quantity
    #[inline]
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Get the raw value
    #[inline]
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }

    /// Check if quantity is zero
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
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

    /// Convert to i64 for position calculations
    #[inline]
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        self.0 as i64
    }

    /// Convert to f64 for calculations
    #[inline]
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.0)
    }
}

impl PartialOrd for Quantity {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Quantity {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Add for Quantity {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Quantity {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul<u32> for Quantity {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: u32) -> Self {
        Self(self.0 * scalar)
    }
}

impl Div<u32> for Quantity {
    type Output = Self;

    #[inline]
    fn div(self, scalar: u32) -> Self {
        Self(self.0 / scalar)
    }
}

impl fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quantity({})", self.0)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Signed quantity for positions (positive = long, negative = short)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct SignedQuantity(i64);

impl SignedQuantity {
    /// Zero signed quantity constant
    pub const ZERO: Self = Self(0);

    /// Create a new signed quantity
    #[inline]
    #[must_use]
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Get the raw value
    #[inline]
    #[must_use]
    pub const fn value(self) -> i64 {
        self.0
    }

    /// Get the absolute value as unsigned quantity
    #[inline]
    #[must_use]
    pub fn abs(self) -> Quantity {
        Quantity::new(self.0.unsigned_abs() as u32)
    }

    /// Check if the position is long
    #[inline]
    #[must_use]
    pub const fn is_long(self) -> bool {
        self.0 > 0
    }

    /// Check if the position is short
    #[inline]
    #[must_use]
    pub const fn is_short(self) -> bool {
        self.0 < 0
    }

    /// Check if the position is flat
    #[inline]
    #[must_use]
    pub const fn is_flat(self) -> bool {
        self.0 == 0
    }

    /// Saturating addition
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

impl Add for SignedQuantity {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for SignedQuantity {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl fmt::Debug for SignedQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SignedQuantity({})", self.0)
    }
}

impl fmt::Display for SignedQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Note: zerocopy traits are implemented via rkyv for zero-copy serialization.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantity_creation() {
        let q = Quantity::new(100);
        assert_eq!(q.value(), 100);
        assert!(!q.is_zero());
    }

    #[test]
    fn test_quantity_arithmetic() {
        let q1 = Quantity::new(100);
        let q2 = Quantity::new(50);

        assert_eq!((q1 + q2).value(), 150);
        assert_eq!((q1 - q2).value(), 50);
        assert_eq!((q1 * 2).value(), 200);
        assert_eq!((q1 / 2).value(), 50);
    }

    #[test]
    fn test_signed_quantity() {
        let long = SignedQuantity::new(100);
        let short = SignedQuantity::new(-50);

        assert!(long.is_long());
        assert!(short.is_short());
        assert_eq!((long + short).value(), 50);
        assert_eq!(short.abs().value(), 50);
    }
}

