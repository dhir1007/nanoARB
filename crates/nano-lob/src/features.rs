//! LOB feature extraction for ML models.

use nano_core::constants::FEATURE_LEVELS;
use nano_core::traits::OrderBook as OrderBookTrait;
use nano_core::types::{Quantity, Timestamp};

use crate::orderbook::OrderBook;

/// Extracted features from the order book
#[derive(Debug, Clone, Default)]
pub struct LobFeatures {
    /// Microprice (volume-weighted mid)
    pub microprice: f64,
    /// Weighted mid price (weighted by depth)
    pub weighted_mid: f64,
    /// Bid-ask spread in ticks
    pub spread: f64,
    /// Book imbalance at level 1 (-1 to +1)
    pub imbalance_l1: f64,
    /// Book imbalance across all levels
    pub imbalance_total: f64,
    /// Total bid depth (quantity)
    pub bid_depth: f64,
    /// Total ask depth (quantity)
    pub ask_depth: f64,
    /// Mid price
    pub mid_price: f64,
    /// Best bid price
    pub best_bid: f64,
    /// Best ask price
    pub best_ask: f64,
    /// Bid depth at each level
    pub bid_levels: [f64; FEATURE_LEVELS],
    /// Ask depth at each level
    pub ask_levels: [f64; FEATURE_LEVELS],
    /// Cumulative bid depth at each level
    pub bid_cumulative: [f64; FEATURE_LEVELS],
    /// Cumulative ask depth at each level
    pub ask_cumulative: [f64; FEATURE_LEVELS],
}

/// LOB feature extractor
#[derive(Debug)]
pub struct LobFeatureExtractor {
    /// Tick size for price normalization
    tick_size: f64,
    /// Quantity normalization factor
    qty_scale: f64,
}

impl Default for LobFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LobFeatureExtractor {
    /// Create a new feature extractor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            tick_size: 0.25,
            qty_scale: 100.0,
        }
    }

    /// Create with custom tick size and quantity scale
    #[must_use]
    pub fn with_params(tick_size: f64, qty_scale: f64) -> Self {
        Self { tick_size, qty_scale }
    }

    /// Extract all features from an order book
    #[must_use]
    pub fn extract(&self, book: &OrderBook) -> LobFeatures {
        let mut features = LobFeatures::default();

        // Get BBO
        let best_bid = book.best_bid();
        let best_ask = book.best_ask();

        if let (Some((bid_price, bid_qty)), Some((ask_price, ask_qty))) = (best_bid, best_ask) {
            let bid = bid_price.as_f64();
            let ask = ask_price.as_f64();
            let bid_q = bid_qty.value() as f64;
            let ask_q = ask_qty.value() as f64;

            features.best_bid = bid;
            features.best_ask = ask;
            features.mid_price = (bid + ask) / 2.0;
            features.spread = (ask - bid) / self.tick_size;

            // Microprice: volume-weighted mid
            let total_bbo_qty = bid_q + ask_q;
            if total_bbo_qty > 0.0 {
                features.microprice = (bid * ask_q + ask * bid_q) / total_bbo_qty;
            } else {
                features.microprice = features.mid_price;
            }

            // Level 1 imbalance
            features.imbalance_l1 = (bid_q - ask_q) / (bid_q + ask_q);
        }

        // Extract level-by-level features
        let mut total_bid_depth = 0.0;
        let mut total_ask_depth = 0.0;
        let mut weighted_bid_sum = 0.0;
        let mut weighted_ask_sum = 0.0;

        for i in 0..FEATURE_LEVELS {
            if let Some(level) = book.bid_level(i) {
                let qty = level.quantity.value() as f64;
                features.bid_levels[i] = qty / self.qty_scale;
                total_bid_depth += qty;
                features.bid_cumulative[i] = total_bid_depth / self.qty_scale;

                // Weight by inverse distance from mid
                let weight = 1.0 / (i as f64 + 1.0);
                weighted_bid_sum += level.price.as_f64() * qty * weight;
            }

            if let Some(level) = book.ask_level(i) {
                let qty = level.quantity.value() as f64;
                features.ask_levels[i] = qty / self.qty_scale;
                total_ask_depth += qty;
                features.ask_cumulative[i] = total_ask_depth / self.qty_scale;

                let weight = 1.0 / (i as f64 + 1.0);
                weighted_ask_sum += level.price.as_f64() * qty * weight;
            }
        }

        features.bid_depth = total_bid_depth / self.qty_scale;
        features.ask_depth = total_ask_depth / self.qty_scale;

        // Total imbalance
        let total_qty = total_bid_depth + total_ask_depth;
        if total_qty > 0.0 {
            features.imbalance_total = (total_bid_depth - total_ask_depth) / total_qty;
        }

        // Weighted mid price
        let total_weighted_qty = weighted_bid_sum + weighted_ask_sum;
        if total_weighted_qty > 0.0 {
            features.weighted_mid = (weighted_bid_sum + weighted_ask_sum) / total_weighted_qty;
        } else {
            features.weighted_mid = features.mid_price;
        }

        features
    }

    /// Calculate microprice
    #[must_use]
    pub fn microprice(&self, book: &OrderBook) -> Option<f64> {
        let (bid_price, bid_qty) = book.best_bid()?;
        let (ask_price, ask_qty) = book.best_ask()?;

        let bid = bid_price.as_f64();
        let ask = ask_price.as_f64();
        let bid_q = bid_qty.value() as f64;
        let ask_q = ask_qty.value() as f64;

        let total = bid_q + ask_q;
        if total > 0.0 {
            Some((bid * ask_q + ask * bid_q) / total)
        } else {
            Some((bid + ask) / 2.0)
        }
    }

    /// Calculate weighted mid price using multiple levels
    #[must_use]
    pub fn weighted_mid(&self, book: &OrderBook, levels: usize) -> Option<f64> {
        let mut bid_sum = 0.0;
        let mut ask_sum = 0.0;
        let mut bid_weight_sum = 0.0;
        let mut ask_weight_sum = 0.0;

        for i in 0..levels.min(FEATURE_LEVELS) {
            let weight = 1.0 / (i as f64 + 1.0);

            if let Some(level) = book.bid_level(i) {
                let qty = level.quantity.value() as f64;
                bid_sum += level.price.as_f64() * qty * weight;
                bid_weight_sum += qty * weight;
            }

            if let Some(level) = book.ask_level(i) {
                let qty = level.quantity.value() as f64;
                ask_sum += level.price.as_f64() * qty * weight;
                ask_weight_sum += qty * weight;
            }
        }

        let total_weight = bid_weight_sum + ask_weight_sum;
        if total_weight > 0.0 {
            Some((bid_sum + ask_sum) / total_weight)
        } else {
            None
        }
    }

    /// Calculate book imbalance at a given depth
    #[must_use]
    pub fn book_imbalance(&self, book: &OrderBook, levels: usize) -> f64 {
        let bid_qty = book.total_bid_quantity(levels).value() as f64;
        let ask_qty = book.total_ask_quantity(levels).value() as f64;

        let total = bid_qty + ask_qty;
        if total > 0.0 {
            (bid_qty - ask_qty) / total
        } else {
            0.0
        }
    }

    /// Calculate Order Flow Imbalance (OFI) from consecutive book states
    #[must_use]
    pub fn order_flow_imbalance(
        &self,
        prev_book: &OrderBook,
        curr_book: &OrderBook,
    ) -> f64 {
        let prev_bid = prev_book.best_bid();
        let prev_ask = prev_book.best_ask();
        let curr_bid = curr_book.best_bid();
        let curr_ask = curr_book.best_ask();

        let mut ofi = 0.0;

        // Bid side OFI
        if let (Some((prev_bp, prev_bq)), Some((curr_bp, curr_bq))) = (prev_bid, curr_bid) {
            if curr_bp > prev_bp {
                // Bid price improved
                ofi += curr_bq.value() as f64;
            } else if curr_bp < prev_bp {
                // Bid price worsened
                ofi -= prev_bq.value() as f64;
            } else {
                // Same price, quantity change
                ofi += (curr_bq.value() as i64 - prev_bq.value() as i64) as f64;
            }
        }

        // Ask side OFI
        if let (Some((prev_ap, prev_aq)), Some((curr_ap, curr_aq))) = (prev_ask, curr_ask) {
            if curr_ap < prev_ap {
                // Ask price improved (lower)
                ofi -= curr_aq.value() as f64;
            } else if curr_ap > prev_ap {
                // Ask price worsened (higher)
                ofi += prev_aq.value() as f64;
            } else {
                // Same price, quantity change
                ofi -= (curr_aq.value() as i64 - prev_aq.value() as i64) as f64;
            }
        }

        ofi / self.qty_scale
    }

    /// Extract features as a flat array for ML input
    #[must_use]
    pub fn to_array(&self, book: &OrderBook) -> [f64; 44] {
        let features = self.extract(book);
        let mut arr = [0.0; 44];

        arr[0] = features.microprice;
        arr[1] = features.weighted_mid;
        arr[2] = features.spread;
        arr[3] = features.imbalance_l1;

        // Bid levels (10 levels)
        for i in 0..FEATURE_LEVELS {
            arr[4 + i] = features.bid_levels[i];
        }

        // Ask levels (10 levels)
        for i in 0..FEATURE_LEVELS {
            arr[14 + i] = features.ask_levels[i];
        }

        // Bid cumulative (10 levels)
        for i in 0..FEATURE_LEVELS {
            arr[24 + i] = features.bid_cumulative[i];
        }

        // Ask cumulative (10 levels)
        for i in 0..FEATURE_LEVELS {
            arr[34 + i] = features.ask_cumulative[i];
        }

        arr
    }
}

/// VPIN (Volume-Synchronized Probability of Informed Trading) calculator
#[derive(Debug)]
pub struct VpinCalculator {
    /// Bucket size in quantity
    bucket_size: u32,
    /// Number of buckets for rolling calculation
    num_buckets: usize,
    /// Current bucket buy volume
    current_buy_volume: u32,
    /// Current bucket sell volume
    current_sell_volume: u32,
    /// Historical buckets (buy_volume, sell_volume)
    buckets: Vec<(u32, u32)>,
}

impl VpinCalculator {
    /// Create a new VPIN calculator
    #[must_use]
    pub fn new(bucket_size: u32, num_buckets: usize) -> Self {
        Self {
            bucket_size,
            num_buckets,
            current_buy_volume: 0,
            current_sell_volume: 0,
            buckets: Vec::with_capacity(num_buckets),
        }
    }

    /// Add a trade to the calculator
    pub fn add_trade(&mut self, quantity: Quantity, is_buy: bool) {
        let qty = quantity.value();

        if is_buy {
            self.current_buy_volume += qty;
        } else {
            self.current_sell_volume += qty;
        }

        // Check if bucket is complete
        let total = self.current_buy_volume + self.current_sell_volume;
        if total >= self.bucket_size {
            self.buckets.push((self.current_buy_volume, self.current_sell_volume));

            // Keep only last N buckets
            if self.buckets.len() > self.num_buckets {
                self.buckets.remove(0);
            }

            self.current_buy_volume = 0;
            self.current_sell_volume = 0;
        }
    }

    /// Calculate current VPIN value (0 to 1)
    #[must_use]
    pub fn calculate(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        let mut abs_imbalance_sum = 0.0;
        let mut total_volume = 0.0;

        for (buy, sell) in &self.buckets {
            let buy_f = *buy as f64;
            let sell_f = *sell as f64;
            abs_imbalance_sum += (buy_f - sell_f).abs();
            total_volume += buy_f + sell_f;
        }

        if total_volume > 0.0 {
            abs_imbalance_sum / total_volume
        } else {
            0.0
        }
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.current_buy_volume = 0;
        self.current_sell_volume = 0;
        self.buckets.clear();
    }

    /// Get number of complete buckets
    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
}

/// Trade flow tracker for OFI calculations
#[derive(Debug, Default)]
pub struct TradeFlowTracker {
    /// Cumulative buy volume
    pub buy_volume: u64,
    /// Cumulative sell volume
    pub sell_volume: u64,
    /// Number of buy trades
    pub buy_count: u32,
    /// Number of sell trades
    pub sell_count: u32,
    /// Last update timestamp
    pub last_update: Timestamp,
}

impl TradeFlowTracker {
    /// Create a new tracker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a trade
    pub fn record_trade(&mut self, quantity: Quantity, is_buy: bool, timestamp: Timestamp) {
        let qty = quantity.value() as u64;

        if is_buy {
            self.buy_volume += qty;
            self.buy_count += 1;
        } else {
            self.sell_volume += qty;
            self.sell_count += 1;
        }

        self.last_update = timestamp;
    }

    /// Calculate net flow (buy - sell)
    #[must_use]
    pub fn net_flow(&self) -> i64 {
        self.buy_volume as i64 - self.sell_volume as i64
    }

    /// Calculate flow imbalance (-1 to 1)
    #[must_use]
    pub fn flow_imbalance(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total > 0 {
            (self.buy_volume as f64 - self.sell_volume as f64) / total as f64
        } else {
            0.0
        }
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.buy_volume = 0;
        self.sell_volume = 0;
        self.buy_count = 0;
        self.sell_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nano_feed::messages::{BookEntry, BookUpdate, EntryType, UpdateAction};

    fn create_test_book() -> OrderBook {
        let mut book = OrderBook::new(1);

        let update = BookUpdate {
            transact_time: 1_000_000_000,
            match_event_indicator: 0x81,
            security_id: 1,
            rpt_seq: 1,
            exponent: -2,
            entries: vec![
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
                    price: 50010,
                    quantity: 50,
                    num_orders: 3,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
                BookEntry {
                    price: 50020,
                    quantity: 150,
                    num_orders: 8,
                    price_level: 2,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
            ],
        };

        book.apply_book_update(&update);
        book
    }

    #[test]
    fn test_microprice() {
        let book = create_test_book();
        let extractor = LobFeatureExtractor::new();

        let micro = extractor.microprice(&book).unwrap();

        // bid=500.00, ask=500.10, bid_qty=100, ask_qty=50
        // microprice = (500.00 * 50 + 500.10 * 100) / 150 = 500.0667
        assert!((micro - 500.0667).abs() < 0.01);
    }

    #[test]
    fn test_book_imbalance() {
        let book = create_test_book();
        let extractor = LobFeatureExtractor::new();

        // Level 1: bid=100, ask=50 => (100-50)/(100+50) = 0.333
        let imbalance = extractor.book_imbalance(&book, 1);
        assert!((imbalance - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_vpin_calculator() {
        let mut vpin = VpinCalculator::new(100, 10);

        // Add some trades
        vpin.add_trade(Quantity::new(60), true);  // buy
        vpin.add_trade(Quantity::new(40), false); // sell - bucket complete

        assert_eq!(vpin.bucket_count(), 1);

        let value = vpin.calculate();
        // Imbalance = |60-40| = 20, total = 100, VPIN = 20/100 = 0.2
        assert!((value - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_trade_flow_tracker() {
        let mut tracker = TradeFlowTracker::new();

        tracker.record_trade(Quantity::new(100), true, Timestamp::from_nanos(1));
        tracker.record_trade(Quantity::new(50), false, Timestamp::from_nanos(2));

        assert_eq!(tracker.net_flow(), 50);
        assert!((tracker.flow_imbalance() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_feature_extraction() {
        let book = create_test_book();
        let extractor = LobFeatureExtractor::new();

        let features = extractor.extract(&book);

        assert!(features.microprice > 0.0);
        assert!(features.spread > 0.0);
        assert!(features.imbalance_l1 > 0.0); // More bid than ask at L1
    }
}

