//! High-performance order book implementation.

use std::collections::BTreeMap;

use nano_core::constants::MAX_BOOK_LEVELS;
use nano_core::traits::OrderBook as OrderBookTrait;
use nano_core::types::{Level, Price, Quantity, Quote, Side, Timestamp};
use nano_feed::messages::{BookEntry, BookUpdate, EntryType, Snapshot, UpdateAction};
use serde::{Deserialize, Serialize};

/// A price level in the book with additional metadata
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BookLevel {
    /// Price at this level
    pub price: Price,
    /// Total quantity at this level
    pub quantity: Quantity,
    /// Number of orders at this level
    pub order_count: u32,
    /// Last update timestamp
    pub last_update: Timestamp,
}

impl BookLevel {
    /// Create a new book level
    #[must_use]
    pub const fn new(
        price: Price,
        quantity: Quantity,
        order_count: u32,
        timestamp: Timestamp,
    ) -> Self {
        Self {
            price,
            quantity,
            order_count,
            last_update: timestamp,
        }
    }

    /// Convert to core Level type
    #[must_use]
    pub const fn to_level(&self) -> Level {
        Level::new(self.price, self.quantity, self.order_count)
    }
}

/// High-performance limit order book with 20 price levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Instrument ID
    instrument_id: u32,
    /// Bid side (price -> level), stored in descending order by price
    bids: BTreeMap<i64, BookLevel>,
    /// Ask side (price -> level), stored in ascending order by price
    asks: BTreeMap<i64, BookLevel>,
    /// Current timestamp
    timestamp: Timestamp,
    /// Last sequence number
    sequence: u32,
    /// Price exponent (for CME data)
    exponent: i8,
    /// Number of updates processed
    update_count: u64,
}

impl OrderBook {
    /// Create a new order book for an instrument
    #[must_use]
    pub fn new(instrument_id: u32) -> Self {
        Self {
            instrument_id,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: Timestamp::EPOCH,
            sequence: 0,
            exponent: -2, // Default: 2 decimal places
            update_count: 0,
        }
    }

    /// Set the price exponent
    pub fn set_exponent(&mut self, exponent: i8) {
        self.exponent = exponent;
    }

    /// Apply a book update message
    pub fn apply_book_update(&mut self, update: &BookUpdate) {
        self.timestamp = update.timestamp();
        self.sequence = update.rpt_seq;
        self.exponent = update.exponent;

        for entry in &update.entries {
            self.apply_entry(entry);
        }

        self.update_count += 1;
        self.trim_levels();
    }

    /// Apply a snapshot message
    pub fn apply_snapshot(&mut self, snapshot: &Snapshot) {
        self.bids.clear();
        self.asks.clear();
        self.sequence = snapshot.rpt_seq;
        self.exponent = snapshot.exponent;
        self.timestamp = Timestamp::from_nanos(snapshot.last_update_time as i64);

        for entry in &snapshot.entries {
            let price = entry.price;
            let quantity = Quantity::new(entry.quantity.max(0) as u32);
            let order_count = entry.num_orders as u32;

            let level = BookLevel::new(
                Price::from_raw(price),
                quantity,
                order_count,
                self.timestamp,
            );

            match entry.entry_type {
                EntryType::Bid | EntryType::ImpliedBid => {
                    self.bids.insert(price, level);
                }
                EntryType::Offer | EntryType::ImpliedOffer => {
                    self.asks.insert(price, level);
                }
                _ => {}
            }
        }

        self.update_count += 1;
        self.trim_levels();
    }

    /// Apply a single book entry
    fn apply_entry(&mut self, entry: &BookEntry) {
        let price = entry.price;
        let quantity = entry.to_quantity();
        let order_count = entry.num_orders.max(0) as u32;

        let is_bid = matches!(entry.entry_type, EntryType::Bid | EntryType::ImpliedBid);
        let book_side = if is_bid {
            &mut self.bids
        } else {
            &mut self.asks
        };

        match entry.action {
            UpdateAction::New | UpdateAction::Change | UpdateAction::Overlay => {
                if quantity.value() > 0 {
                    let level = BookLevel::new(
                        Price::from_raw(price),
                        quantity,
                        order_count,
                        self.timestamp,
                    );
                    book_side.insert(price, level);
                } else {
                    book_side.remove(&price);
                }
            }
            UpdateAction::Delete => {
                book_side.remove(&price);
            }
            UpdateAction::DeleteThru => {
                // Delete all levels through this price
                if is_bid {
                    book_side.retain(|&p, _| p > price);
                } else {
                    book_side.retain(|&p, _| p < price);
                }
            }
            UpdateAction::DeleteFrom => {
                // Delete all levels from this price
                if is_bid {
                    book_side.retain(|&p, _| p < price);
                } else {
                    book_side.retain(|&p, _| p > price);
                }
            }
        }
    }

    /// Trim book to maximum levels
    fn trim_levels(&mut self) {
        // Keep only top MAX_BOOK_LEVELS on each side
        while self.bids.len() > MAX_BOOK_LEVELS {
            if let Some((&lowest_bid, _)) = self.bids.iter().next() {
                self.bids.remove(&lowest_bid);
            }
        }

        while self.asks.len() > MAX_BOOK_LEVELS {
            if let Some((&highest_ask, _)) = self.asks.iter().next_back() {
                self.asks.remove(&highest_ask);
            }
        }
    }

    /// Get the instrument ID
    #[must_use]
    pub const fn instrument_id(&self) -> u32 {
        self.instrument_id
    }

    /// Get bid levels (best to worst)
    pub fn bid_levels(&self) -> impl Iterator<Item = &BookLevel> {
        self.bids.values().rev()
    }

    /// Get ask levels (best to worst)
    pub fn ask_levels(&self) -> impl Iterator<Item = &BookLevel> {
        self.asks.values()
    }

    /// Get bid level at index (0 = best)
    #[must_use]
    pub fn bid_level(&self, index: usize) -> Option<&BookLevel> {
        self.bids.values().rev().nth(index)
    }

    /// Get ask level at index (0 = best)
    #[must_use]
    pub fn ask_level(&self, index: usize) -> Option<&BookLevel> {
        self.asks.values().nth(index)
    }

    /// Get the number of bid levels
    #[must_use]
    pub fn bid_depth(&self) -> usize {
        self.bids.len()
    }

    /// Get the number of ask levels
    #[must_use]
    pub fn ask_depth(&self) -> usize {
        self.asks.len()
    }

    /// Check if the book is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Check if the book is valid (has both sides)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.bids.is_empty() && !self.asks.is_empty()
    }

    /// Check if book is crossed (bid >= ask)
    #[must_use]
    pub fn is_crossed(&self) -> bool {
        if let (Some(best_bid), Some(best_ask)) = (self.best_bid(), self.best_ask()) {
            best_bid.0 >= best_ask.0
        } else {
            false
        }
    }

    /// Get sequence number
    #[must_use]
    pub const fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Get update count
    #[must_use]
    pub const fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Clear the book
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.timestamp = Timestamp::EPOCH;
        self.sequence = 0;
    }

    /// Get total bid quantity up to N levels
    #[must_use]
    pub fn total_bid_quantity(&self, levels: usize) -> Quantity {
        self.bid_levels()
            .take(levels)
            .fold(Quantity::ZERO, |acc, l| acc.saturating_add(l.quantity))
    }

    /// Get total ask quantity up to N levels
    #[must_use]
    pub fn total_ask_quantity(&self, levels: usize) -> Quantity {
        self.ask_levels()
            .take(levels)
            .fold(Quantity::ZERO, |acc, l| acc.saturating_add(l.quantity))
    }

    /// Calculate volume-weighted average price for a given quantity
    #[must_use]
    pub fn vwap(&self, side: Side, quantity: Quantity) -> Option<Price> {
        let levels: Vec<_> = match side {
            Side::Buy => self.ask_levels().collect(),
            Side::Sell => self.bid_levels().collect(),
        };

        let mut remaining = quantity.value();
        let mut total_value: i64 = 0;
        let mut total_qty: u32 = 0;

        for level in levels {
            if remaining == 0 {
                break;
            }
            let fill_qty = remaining.min(level.quantity.value());
            total_value += level.price.raw() * i64::from(fill_qty);
            total_qty += fill_qty;
            remaining -= fill_qty;
        }

        if total_qty > 0 {
            Some(Price::from_raw(total_value / i64::from(total_qty)))
        } else {
            None
        }
    }
}

impl OrderBookTrait for OrderBook {
    fn best_bid(&self) -> Option<(Price, Quantity)> {
        self.bids
            .values()
            .next_back()
            .map(|l| (l.price, l.quantity))
    }

    fn best_ask(&self) -> Option<(Price, Quantity)> {
        self.asks.values().next().map(|l| (l.price, l.quantity))
    }

    fn mid_price(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(Price::from_raw((bid.raw() + ask.raw()) / 2)),
            _ => None,
        }
    }

    fn spread(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(Price::from_raw(ask.raw() - bid.raw())),
            _ => None,
        }
    }

    fn quote(&self) -> Option<Quote> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid_price, bid_qty)), Some((ask_price, ask_qty))) => Some(Quote::new(
                self.instrument_id,
                bid_price,
                bid_qty,
                ask_price,
                ask_qty,
                self.timestamp,
            )),
            _ => None,
        }
    }

    fn bid_at_level(&self, level: usize) -> Option<(Price, Quantity)> {
        self.bid_level(level).map(|l| (l.price, l.quantity))
    }

    fn ask_at_level(&self, level: usize) -> Option<(Price, Quantity)> {
        self.ask_level(level).map(|l| (l.price, l.quantity))
    }

    fn bid_depth(&self, levels: usize) -> Quantity {
        self.total_bid_quantity(levels)
    }

    fn ask_depth(&self, levels: usize) -> Quantity {
        self.total_ask_quantity(levels)
    }

    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nano_feed::messages::{BookEntry, BookUpdate, EntryType, UpdateAction};

    fn create_book_update(security_id: i32, entries: Vec<BookEntry>, seq: u32) -> BookUpdate {
        BookUpdate {
            transact_time: 1_000_000_000,
            match_event_indicator: 0x81,
            security_id,
            rpt_seq: seq,
            exponent: -2,
            entries,
        }
    }

    #[test]
    fn test_empty_book() {
        let book = OrderBook::new(1);
        assert!(book.is_empty());
        assert!(!book.is_valid());
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_apply_update() {
        let mut book = OrderBook::new(1);

        let update = create_book_update(
            1,
            vec![
                BookEntry {
                    price: 50000,
                    quantity: 100,
                    num_orders: 5,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: 50010,
                    quantity: 50,
                    num_orders: 3,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
            ],
            1,
        );

        book.apply_book_update(&update);

        assert!(book.is_valid());
        assert!(!book.is_empty());

        let (bid_price, bid_qty) = book.best_bid().unwrap();
        assert_eq!(bid_price.raw(), 50000);
        assert_eq!(bid_qty.value(), 100);

        let (ask_price, ask_qty) = book.best_ask().unwrap();
        assert_eq!(ask_price.raw(), 50010);
        assert_eq!(ask_qty.value(), 50);
    }

    #[test]
    fn test_mid_price_and_spread() {
        let mut book = OrderBook::new(1);

        let update = create_book_update(
            1,
            vec![
                BookEntry {
                    price: 50000,
                    quantity: 100,
                    num_orders: 5,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: 50020,
                    quantity: 50,
                    num_orders: 3,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
            ],
            1,
        );

        book.apply_book_update(&update);

        let mid = book.mid_price().unwrap();
        assert_eq!(mid.raw(), 50010);

        let spread = book.spread().unwrap();
        assert_eq!(spread.raw(), 20);
    }

    #[test]
    fn test_delete_level() {
        let mut book = OrderBook::new(1);

        // Add a level
        let update1 = create_book_update(
            1,
            vec![BookEntry {
                price: 50000,
                quantity: 100,
                num_orders: 5,
                price_level: 1,
                action: UpdateAction::New,
                entry_type: EntryType::Bid,
            }],
            1,
        );
        book.apply_book_update(&update1);

        assert!(book.best_bid().is_some());

        // Delete it
        let update2 = create_book_update(
            1,
            vec![BookEntry {
                price: 50000,
                quantity: 0,
                num_orders: 0,
                price_level: 1,
                action: UpdateAction::Delete,
                entry_type: EntryType::Bid,
            }],
            2,
        );
        book.apply_book_update(&update2);

        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_multiple_levels() {
        let mut book = OrderBook::new(1);

        let update = create_book_update(
            1,
            vec![
                BookEntry {
                    price: 50000,
                    quantity: 100,
                    num_orders: 5,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: 49990,
                    quantity: 200,
                    num_orders: 10,
                    price_level: 2,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: 49980,
                    quantity: 150,
                    num_orders: 8,
                    price_level: 3,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
            ],
            1,
        );

        book.apply_book_update(&update);

        // Best bid should be highest price
        let (best_bid, _) = book.best_bid().unwrap();
        assert_eq!(best_bid.raw(), 50000);

        // Second level
        let (level2_price, level2_qty) = book.bid_at_level(1).unwrap();
        assert_eq!(level2_price.raw(), 49990);
        assert_eq!(level2_qty.value(), 200);

        // Total depth
        let total = book.total_bid_quantity(3);
        assert_eq!(total.value(), 450); // 100 + 200 + 150
    }

    #[test]
    fn test_vwap() {
        let mut book = OrderBook::new(1);

        let update = create_book_update(
            1,
            vec![
                BookEntry {
                    price: 10000,
                    quantity: 100,
                    num_orders: 5,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
                BookEntry {
                    price: 10010,
                    quantity: 100,
                    num_orders: 5,
                    price_level: 2,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
            ],
            1,
        );

        book.apply_book_update(&update);

        // VWAP for 150 contracts: 100@10000 + 50@10010 = (1000000 + 500500) / 150 = 10003.33
        let vwap = book.vwap(Side::Buy, Quantity::new(150)).unwrap();
        assert_eq!(vwap.raw(), 10003); // Integer division
    }
}
