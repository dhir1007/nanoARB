//! Order types and related structures.

use std::fmt;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use super::{Price, Quantity, Side, Timestamp};

/// Unique order identifier
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
pub struct OrderId(u64);

impl OrderId {
    /// Create a new order ID
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw value
    #[inline]
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrderId({})", self.0)
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for OrderId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<OrderId> for u64 {
    fn from(id: OrderId) -> Self {
        id.0
    }
}

/// Order type
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
pub enum OrderType {
    /// Limit order - specify price and quantity
    #[default]
    Limit = 0,
    /// Market order - execute at best available price
    Market = 1,
    /// Stop order - trigger at stop price, then execute as market
    Stop = 2,
    /// Stop-limit order - trigger at stop price, then place limit order
    StopLimit = 3,
}

impl fmt::Debug for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Limit => write!(f, "Limit"),
            OrderType::Market => write!(f, "Market"),
            OrderType::Stop => write!(f, "Stop"),
            OrderType::StopLimit => write!(f, "StopLimit"),
        }
    }
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Time in force for orders
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
pub enum TimeInForce {
    /// Good till cancelled
    #[default]
    GTC = 0,
    /// Immediate or cancel (partial fills allowed)
    IOC = 1,
    /// Fill or kill (no partial fills)
    FOK = 2,
    /// Day order (expires at end of trading session)
    Day = 3,
    /// Good till date
    GTD = 4,
}

impl fmt::Debug for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeInForce::GTC => write!(f, "GTC"),
            TimeInForce::IOC => write!(f, "IOC"),
            TimeInForce::FOK => write!(f, "FOK"),
            TimeInForce::Day => write!(f, "Day"),
            TimeInForce::GTD => write!(f, "GTD"),
        }
    }
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Order status
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
pub enum OrderStatus {
    /// Order is pending submission
    #[default]
    Pending = 0,
    /// Order has been submitted to exchange
    Submitted = 1,
    /// Order is open in the order book
    Open = 2,
    /// Order is partially filled
    PartiallyFilled = 3,
    /// Order is completely filled
    Filled = 4,
    /// Order has been cancelled
    Cancelled = 5,
    /// Order was rejected
    Rejected = 6,
    /// Order has expired
    Expired = 7,
}

impl OrderStatus {
    /// Check if the order is in a terminal state
    #[inline]
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }

    /// Check if the order is active (can be filled or cancelled)
    #[inline]
    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(
            self,
            OrderStatus::Submitted | OrderStatus::Open | OrderStatus::PartiallyFilled
        )
    }
}

impl fmt::Debug for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderStatus::Pending => write!(f, "Pending"),
            OrderStatus::Submitted => write!(f, "Submitted"),
            OrderStatus::Open => write!(f, "Open"),
            OrderStatus::PartiallyFilled => write!(f, "PartiallyFilled"),
            OrderStatus::Filled => write!(f, "Filled"),
            OrderStatus::Cancelled => write!(f, "Cancelled"),
            OrderStatus::Rejected => write!(f, "Rejected"),
            OrderStatus::Expired => write!(f, "Expired"),
        }
    }
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A trading order
#[derive(
    Clone, Copy, PartialEq, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize,
)]
#[archive(check_bytes)]
pub struct Order {
    /// Unique order identifier
    pub id: OrderId,
    /// Instrument ID
    pub instrument_id: u32,
    /// Order side (buy/sell)
    pub side: Side,
    /// Order type
    pub order_type: OrderType,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Order status
    pub status: OrderStatus,
    /// Limit price (for limit orders)
    pub price: Price,
    /// Stop price (for stop orders)
    pub stop_price: Price,
    /// Original quantity
    pub quantity: Quantity,
    /// Filled quantity
    pub filled_quantity: Quantity,
    /// Average fill price (in raw ticks)
    pub avg_fill_price: i64,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Last update timestamp
    pub updated_at: Timestamp,
}

impl Order {
    /// Create a new limit order
    #[must_use]
    pub fn new_limit(
        id: OrderId,
        instrument_id: u32,
        side: Side,
        price: Price,
        quantity: Quantity,
        time_in_force: TimeInForce,
    ) -> Self {
        let now = Timestamp::now();
        Self {
            id,
            instrument_id,
            side,
            order_type: OrderType::Limit,
            time_in_force,
            status: OrderStatus::Pending,
            price,
            stop_price: Price::ZERO,
            quantity,
            filled_quantity: Quantity::ZERO,
            avg_fill_price: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new market order
    #[must_use]
    pub fn new_market(id: OrderId, instrument_id: u32, side: Side, quantity: Quantity) -> Self {
        let now = Timestamp::now();
        Self {
            id,
            instrument_id,
            side,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::IOC,
            status: OrderStatus::Pending,
            price: Price::ZERO,
            stop_price: Price::ZERO,
            quantity,
            filled_quantity: Quantity::ZERO,
            avg_fill_price: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Get remaining quantity
    #[inline]
    #[must_use]
    pub fn remaining_quantity(&self) -> Quantity {
        self.quantity.saturating_sub(self.filled_quantity)
    }

    /// Check if the order is completely filled
    #[inline]
    #[must_use]
    pub fn is_filled(&self) -> bool {
        self.filled_quantity >= self.quantity
    }

    /// Check if the order can be cancelled
    #[inline]
    #[must_use]
    pub fn is_cancellable(&self) -> bool {
        self.status.is_active()
    }

    /// Get fill ratio (0.0 to 1.0)
    #[inline]
    #[must_use]
    pub fn fill_ratio(&self) -> f64 {
        if self.quantity.is_zero() {
            0.0
        } else {
            self.filled_quantity.as_f64() / self.quantity.as_f64()
        }
    }

    /// Update order with a fill
    pub fn apply_fill(&mut self, fill_price: Price, fill_qty: Quantity, timestamp: Timestamp) {
        let prev_filled = i64::from(self.filled_quantity.value());
        let prev_notional = prev_filled * self.avg_fill_price;
        let new_notional = prev_notional + (i64::from(fill_qty.value()) * fill_price.raw());
        let new_filled = prev_filled + i64::from(fill_qty.value());

        self.filled_quantity = self.filled_quantity.saturating_add(fill_qty);
        self.avg_fill_price = if new_filled > 0 {
            new_notional / new_filled
        } else {
            0
        };
        self.updated_at = timestamp;

        if self.is_filled() {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }
    }
}

impl fmt::Debug for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Order")
            .field("id", &self.id)
            .field("instrument_id", &self.instrument_id)
            .field("side", &self.side)
            .field("type", &self.order_type)
            .field("status", &self.status)
            .field("price", &self.price)
            .field("qty", &self.quantity)
            .field("filled", &self.filled_quantity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::new_limit(
            OrderId::new(1),
            100,
            Side::Buy,
            Price::from_raw(50000),
            Quantity::new(10),
            TimeInForce::GTC,
        );

        assert_eq!(order.id.value(), 1);
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.status, OrderStatus::Pending);
        assert_eq!(order.remaining_quantity().value(), 10);
    }

    #[test]
    fn test_order_fill() {
        let mut order = Order::new_limit(
            OrderId::new(1),
            100,
            Side::Buy,
            Price::from_raw(50000),
            Quantity::new(10),
            TimeInForce::GTC,
        );

        order.apply_fill(Price::from_raw(49990), Quantity::new(5), Timestamp::now());
        assert_eq!(order.filled_quantity.value(), 5);
        assert_eq!(order.status, OrderStatus::PartiallyFilled);
        assert_eq!(order.remaining_quantity().value(), 5);

        order.apply_fill(Price::from_raw(50010), Quantity::new(5), Timestamp::now());
        assert_eq!(order.filled_quantity.value(), 10);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(order.is_filled());
    }

    #[test]
    fn test_order_status_states() {
        assert!(!OrderStatus::Pending.is_terminal());
        assert!(!OrderStatus::Pending.is_active());
        assert!(OrderStatus::Submitted.is_active());
        assert!(OrderStatus::Open.is_active());
        assert!(OrderStatus::Filled.is_terminal());
        assert!(OrderStatus::Cancelled.is_terminal());
    }
}
