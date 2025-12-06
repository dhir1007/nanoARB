//! Nanosecond-precision timestamp for HFT applications.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use crate::constants::{NS_PER_MS, NS_PER_SEC, NS_PER_US};

/// Nanosecond-precision timestamp since Unix epoch.
///
/// Optimized for HFT applications where every nanosecond matters.
/// Uses i64 internally to support timestamps both before and after epoch.
///
/// # Example
///
/// ```rust
/// use nano_core::types::Timestamp;
///
/// let ts = Timestamp::now();
/// let later = ts.add_nanos(1000);
/// assert!(later > ts);
/// ```
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
#[archive(check_bytes)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Zero timestamp (Unix epoch)
    pub const EPOCH: Self = Self(0);

    /// Maximum timestamp
    pub const MAX: Self = Self(i64::MAX);

    /// Minimum timestamp
    pub const MIN: Self = Self(i64::MIN);

    /// Create a timestamp from nanoseconds since epoch
    #[inline]
    #[must_use]
    pub const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    /// Create a timestamp from microseconds since epoch
    #[inline]
    #[must_use]
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros * NS_PER_US as i64)
    }

    /// Create a timestamp from milliseconds since epoch
    #[inline]
    #[must_use]
    pub const fn from_millis(millis: i64) -> Self {
        Self(millis * NS_PER_MS as i64)
    }

    /// Create a timestamp from seconds since epoch
    #[inline]
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Self(secs * NS_PER_SEC as i64)
    }

    /// Get the current timestamp
    #[inline]
    #[must_use]
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        Self(duration.as_nanos() as i64)
    }

    /// Get nanoseconds since epoch
    #[inline]
    #[must_use]
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// Get microseconds since epoch
    #[inline]
    #[must_use]
    pub const fn as_micros(self) -> i64 {
        self.0 / NS_PER_US as i64
    }

    /// Get milliseconds since epoch
    #[inline]
    #[must_use]
    pub const fn as_millis(self) -> i64 {
        self.0 / NS_PER_MS as i64
    }

    /// Get seconds since epoch
    #[inline]
    #[must_use]
    pub const fn as_secs(self) -> i64 {
        self.0 / NS_PER_SEC as i64
    }

    /// Get the nanosecond component (0-999_999_999)
    #[inline]
    #[must_use]
    pub const fn subsec_nanos(self) -> u32 {
        (self.0 % NS_PER_SEC as i64).unsigned_abs() as u32
    }

    /// Add nanoseconds to this timestamp
    #[inline]
    #[must_use]
    pub const fn add_nanos(self, nanos: i64) -> Self {
        Self(self.0 + nanos)
    }

    /// Add microseconds to this timestamp
    #[inline]
    #[must_use]
    pub const fn add_micros(self, micros: i64) -> Self {
        Self(self.0 + micros * NS_PER_US as i64)
    }

    /// Add milliseconds to this timestamp
    #[inline]
    #[must_use]
    pub const fn add_millis(self, millis: i64) -> Self {
        Self(self.0 + millis * NS_PER_MS as i64)
    }

    /// Subtract nanoseconds from this timestamp
    #[inline]
    #[must_use]
    pub const fn sub_nanos(self, nanos: i64) -> Self {
        Self(self.0 - nanos)
    }

    /// Calculate duration to another timestamp in nanoseconds
    #[inline]
    #[must_use]
    pub const fn duration_since(self, earlier: Self) -> i64 {
        self.0 - earlier.0
    }

    /// Convert to std Duration (only works for positive timestamps)
    #[inline]
    #[must_use]
    pub fn to_duration(self) -> Option<Duration> {
        if self.0 >= 0 {
            Some(Duration::from_nanos(self.0 as u64))
        } else {
            None
        }
    }

    /// Convert to chrono `DateTime`
    #[inline]
    #[must_use]
    pub fn to_datetime(&self) -> chrono::DateTime<chrono::Utc> {
        let secs = self.as_secs();
        let nsecs = self.subsec_nanos();
        chrono::DateTime::from_timestamp(secs, nsecs).unwrap_or_default()
    }

    /// Saturating addition
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, nanos: i64) -> Self {
        Self(self.0.saturating_add(nanos))
    }

    /// Saturating subtraction
    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, nanos: i64) -> Self {
        Self(self.0.saturating_sub(nanos))
    }
}

impl PartialOrd for Timestamp {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    #[inline]
    fn add(self, duration: Duration) -> Self {
        Self(self.0 + duration.as_nanos() as i64)
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Self;

    #[inline]
    fn sub(self, duration: Duration) -> Self {
        Self(self.0 - duration.as_nanos() as i64)
    }
}

impl Sub for Timestamp {
    type Output = i64;

    #[inline]
    fn sub(self, other: Self) -> i64 {
        self.0 - other.0
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timestamp({}ns)", self.0)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dt = self.to_datetime();
        write!(f, "{}", dt.format("%Y-%m-%d %H:%M:%S%.9f"))
    }
}

impl From<SystemTime> for Timestamp {
    fn from(time: SystemTime) -> Self {
        let duration = time
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        Self(duration.as_nanos() as i64)
    }
}

// Note: zerocopy traits are implemented via rkyv for zero-copy serialization.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_creation() {
        let ts = Timestamp::from_nanos(1_000_000_000);
        assert_eq!(ts.as_nanos(), 1_000_000_000);
        assert_eq!(ts.as_secs(), 1);
    }

    #[test]
    fn test_timestamp_conversions() {
        let ts = Timestamp::from_secs(1);
        assert_eq!(ts.as_millis(), 1000);
        assert_eq!(ts.as_micros(), 1_000_000);
        assert_eq!(ts.as_nanos(), 1_000_000_000);
    }

    #[test]
    fn test_timestamp_arithmetic() {
        let t1 = Timestamp::from_nanos(1000);
        let t2 = Timestamp::from_nanos(500);

        assert_eq!(t1.duration_since(t2), 500);
        assert_eq!(t1.add_nanos(100).as_nanos(), 1100);
        assert_eq!(t1.sub_nanos(100).as_nanos(), 900);
    }

    #[test]
    fn test_timestamp_comparison() {
        let t1 = Timestamp::from_nanos(1000);
        let t2 = Timestamp::from_nanos(500);

        assert!(t1 > t2);
        assert!(t2 < t1);
    }

    #[test]
    fn test_timestamp_now() {
        let t1 = Timestamp::now();
        std::thread::sleep(Duration::from_micros(1));
        let t2 = Timestamp::now();

        assert!(t2 > t1);
    }

    #[test]
    fn test_subsec_nanos() {
        let ts = Timestamp::from_nanos(1_234_567_890);
        assert_eq!(ts.subsec_nanos(), 234_567_890);
    }
}
