//! Core domain types for the trading engine.

mod instrument;
mod order;
mod price;
mod quantity;
mod side;
mod timestamp;

pub use instrument::{Exchange, Instrument, InstrumentType};
pub use order::{Order, OrderId, OrderStatus, OrderType, TimeInForce};
pub use price::Price;
pub use quantity::Quantity;
pub use side::Side;
pub use timestamp::Timestamp;

/// Trade execution information
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct Trade {
    /// Trade ID
    pub id: u64,
    /// Instrument
    pub instrument_id: u32,
    /// Trade price
    pub price: Price,
    /// Trade quantity
    pub quantity: Quantity,
    /// Aggressor side (taker)
    pub aggressor_side: Side,
    /// Trade timestamp
    pub timestamp: Timestamp,
}

/// Quote/BBO (Best Bid/Offer)
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct Quote {
    /// Instrument ID
    pub instrument_id: u32,
    /// Best bid price
    pub bid_price: Price,
    /// Best bid quantity
    pub bid_quantity: Quantity,
    /// Best ask price
    pub ask_price: Price,
    /// Best ask quantity
    pub ask_quantity: Quantity,
    /// Quote timestamp
    pub timestamp: Timestamp,
}

impl Quote {
    /// Create a new quote
    #[must_use]
    pub const fn new(
        instrument_id: u32,
        bid_price: Price,
        bid_quantity: Quantity,
        ask_price: Price,
        ask_quantity: Quantity,
        timestamp: Timestamp,
    ) -> Self {
        Self {
            instrument_id,
            bid_price,
            bid_quantity,
            ask_price,
            ask_quantity,
            timestamp,
        }
    }

    /// Calculate the mid price
    #[must_use]
    pub fn mid_price(&self) -> Price {
        Price::from_raw((self.bid_price.raw() + self.ask_price.raw()) / 2)
    }

    /// Calculate the spread in ticks
    #[must_use]
    pub fn spread(&self) -> Price {
        Price::from_raw(self.ask_price.raw() - self.bid_price.raw())
    }

    /// Check if the quote is valid (bid < ask)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.bid_price.raw() < self.ask_price.raw()
            && self.bid_quantity.value() > 0
            && self.ask_quantity.value() > 0
    }
}

/// A price level in the order book
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct Level {
    /// Price at this level
    pub price: Price,
    /// Total quantity at this level
    pub quantity: Quantity,
    /// Number of orders at this level
    pub order_count: u32,
}

impl Level {
    /// Create a new level
    #[must_use]
    pub const fn new(price: Price, quantity: Quantity, order_count: u32) -> Self {
        Self {
            price,
            quantity,
            order_count,
        }
    }

    /// Create an empty level at a given price
    #[must_use]
    pub const fn empty(price: Price) -> Self {
        Self {
            price,
            quantity: Quantity::ZERO,
            order_count: 0,
        }
    }

    /// Check if the level is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quantity.value() == 0
    }
}

/// Fill information for an executed order
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct Fill {
    /// Order ID that was filled
    pub order_id: OrderId,
    /// Fill price
    pub price: Price,
    /// Fill quantity
    pub quantity: Quantity,
    /// Side of the filled order
    pub side: Side,
    /// Whether this was a maker or taker fill
    pub is_maker: bool,
    /// Fill timestamp
    pub timestamp: Timestamp,
    /// Fee paid
    pub fee: f64,
}

impl Fill {
    /// Calculate the notional value of this fill
    #[must_use]
    pub fn notional(&self) -> f64 {
        self.price.as_f64() * f64::from(self.quantity.value())
    }

    /// Calculate the signed quantity (positive for buy, negative for sell)
    #[must_use]
    pub fn signed_quantity(&self) -> i64 {
        match self.side {
            Side::Buy => i64::from(self.quantity.value()),
            Side::Sell => -i64::from(self.quantity.value()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quote_mid_price() {
        let quote = Quote::new(
            1,
            Price::from_raw(10000),
            Quantity::new(100),
            Price::from_raw(10010),
            Quantity::new(50),
            Timestamp::from_nanos(0),
        );

        assert_eq!(quote.mid_price().raw(), 10005);
        assert_eq!(quote.spread().raw(), 10);
        assert!(quote.is_valid());
    }

    #[test]
    fn test_level() {
        let level = Level::new(Price::from_raw(10000), Quantity::new(100), 5);
        assert!(!level.is_empty());

        let empty = Level::empty(Price::from_raw(10000));
        assert!(empty.is_empty());
    }

    #[test]
    fn test_fill_signed_quantity() {
        let buy_fill = Fill {
            order_id: OrderId::new(1),
            price: Price::from_raw(10000),
            quantity: Quantity::new(10),
            side: Side::Buy,
            is_maker: true,
            timestamp: Timestamp::from_nanos(0),
            fee: 0.25,
        };
        assert_eq!(buy_fill.signed_quantity(), 10);

        let sell_fill = Fill {
            order_id: OrderId::new(2),
            price: Price::from_raw(10000),
            quantity: Quantity::new(10),
            side: Side::Sell,
            is_maker: false,
            timestamp: Timestamp::from_nanos(0),
            fee: 0.85,
        };
        assert_eq!(sell_fill.signed_quantity(), -10);
    }
}

