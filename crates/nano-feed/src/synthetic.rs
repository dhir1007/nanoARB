//! Synthetic data generation for testing and development.

use nano_core::types::{Price, Quantity, Timestamp};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::messages::*;

/// Configuration for synthetic data generation
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    /// Initial mid price (in ticks)
    pub initial_mid: i64,
    /// Tick size
    pub tick_size: i64,
    /// Average spread in ticks
    pub avg_spread_ticks: u32,
    /// Average quantity per level
    pub avg_quantity: u32,
    /// Number of price levels
    pub num_levels: usize,
    /// Volatility (price change per event in ticks, std dev)
    pub volatility: f64,
    /// Trade frequency (0.0 to 1.0, probability of trade per event)
    pub trade_frequency: f64,
    /// Average trade size
    pub avg_trade_size: u32,
    /// Start timestamp (nanoseconds)
    pub start_time_ns: u64,
    /// Average time between events (nanoseconds)
    pub avg_event_interval_ns: u64,
    /// Security ID
    pub security_id: i32,
    /// Price exponent
    pub exponent: i8,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            initial_mid: 500000, // 5000.00
            tick_size: 25,       // 0.25
            avg_spread_ticks: 1,
            avg_quantity: 50,
            num_levels: 10,
            volatility: 0.5,
            trade_frequency: 0.3,
            avg_trade_size: 5,
            start_time_ns: 1_700_000_000_000_000_000, // ~2023
            avg_event_interval_ns: 1_000_000,         // 1ms average
            security_id: 1,
            exponent: -2,
        }
    }
}

impl SyntheticConfig {
    /// Create config for ES futures
    #[must_use]
    pub fn es_futures() -> Self {
        Self {
            initial_mid: 500000,
            tick_size: 25,
            avg_spread_ticks: 1,
            avg_quantity: 100,
            num_levels: 10,
            volatility: 0.3,
            trade_frequency: 0.4,
            avg_trade_size: 3,
            ..Default::default()
        }
    }

    /// Create config for NQ futures
    #[must_use]
    pub fn nq_futures() -> Self {
        Self {
            initial_mid: 1800000,
            tick_size: 25,
            avg_spread_ticks: 1,
            avg_quantity: 50,
            num_levels: 10,
            volatility: 0.4,
            trade_frequency: 0.35,
            avg_trade_size: 2,
            ..Default::default()
        }
    }
}

/// Synthetic data generator for CME MDP 3.0 messages
pub struct SyntheticGenerator {
    config: SyntheticConfig,
    rng: StdRng,
    current_mid: i64,
    current_time: u64,
    sequence: u32,
    bid_levels: Vec<(i64, i32)>,  // (price, quantity)
    ask_levels: Vec<(i64, i32)>,
}

impl SyntheticGenerator {
    /// Create a new generator with the given config
    #[must_use]
    pub fn new(config: SyntheticConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create a new generator with a specific seed
    #[must_use]
    pub fn with_seed(config: SyntheticConfig, seed: u64) -> Self {
        let mut gen = Self {
            current_mid: config.initial_mid,
            current_time: config.start_time_ns,
            sequence: 0,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            config,
        };
        gen.initialize_book();
        gen
    }

    /// Initialize the order book
    fn initialize_book(&mut self) {
        self.bid_levels.clear();
        self.ask_levels.clear();

        let half_spread = (self.config.avg_spread_ticks as i64 * self.config.tick_size) / 2;
        let best_bid = self.current_mid - half_spread;
        let best_ask = self.current_mid + half_spread;

        for i in 0..self.config.num_levels {
            let bid_price = best_bid - (i as i64 * self.config.tick_size);
            let ask_price = best_ask + (i as i64 * self.config.tick_size);

            let bid_qty = self.random_quantity();
            let ask_qty = self.random_quantity();

            self.bid_levels.push((bid_price, bid_qty));
            self.ask_levels.push((ask_price, ask_qty));
        }
    }

    /// Generate a random quantity
    fn random_quantity(&mut self) -> i32 {
        let dist = Uniform::new(1, self.config.avg_quantity * 2);
        dist.sample(&mut self.rng) as i32
    }

    /// Generate the next event
    pub fn next_event(&mut self) -> MdpMessage {
        // Advance time
        let time_dist = Uniform::new(
            self.config.avg_event_interval_ns / 2,
            self.config.avg_event_interval_ns * 2,
        );
        self.current_time += time_dist.sample(&mut self.rng);
        self.sequence += 1;

        // Random price movement
        let price_change: f64 = self.rng.gen::<f64>() * 2.0 - 1.0; // -1 to 1
        let tick_change = (price_change * self.config.volatility).round() as i64;
        self.current_mid += tick_change * self.config.tick_size;

        // Decide if this is a trade or book update
        if self.rng.gen::<f64>() < self.config.trade_frequency {
            self.generate_trade()
        } else {
            self.generate_book_update()
        }
    }

    /// Generate a book update message
    fn generate_book_update(&mut self) -> MdpMessage {
        // Update the book based on new mid
        self.update_book_levels();

        // Create book entries for changed levels
        let mut entries = Vec::new();

        // Add some bid updates
        let num_bid_updates = self.rng.gen_range(1..=3);
        for i in 0..num_bid_updates.min(self.bid_levels.len()) {
            let (price, qty) = self.bid_levels[i];
            entries.push(BookEntry {
                price,
                quantity: qty,
                num_orders: self.rng.gen_range(1..10),
                price_level: (i + 1) as u8,
                action: if i == 0 { UpdateAction::Change } else { UpdateAction::Change },
                entry_type: EntryType::Bid,
            });
        }

        // Add some ask updates
        let num_ask_updates = self.rng.gen_range(1..=3);
        for i in 0..num_ask_updates.min(self.ask_levels.len()) {
            let (price, qty) = self.ask_levels[i];
            entries.push(BookEntry {
                price,
                quantity: qty,
                num_orders: self.rng.gen_range(1..10),
                price_level: (i + 1) as u8,
                action: UpdateAction::Change,
                entry_type: EntryType::Offer,
            });
        }

        MdpMessage::BookUpdate(BookUpdate {
            transact_time: self.current_time,
            match_event_indicator: 0x81, // Last message + end of event
            security_id: self.config.security_id,
            rpt_seq: self.sequence,
            exponent: self.config.exponent,
            entries,
        })
    }

    /// Generate a trade message
    fn generate_trade(&mut self) -> MdpMessage {
        let is_buy_aggressor = self.rng.gen_bool(0.5);

        let trade_price = if is_buy_aggressor {
            self.ask_levels.first().map(|(p, _)| *p).unwrap_or(self.current_mid)
        } else {
            self.bid_levels.first().map(|(p, _)| *p).unwrap_or(self.current_mid)
        };

        let trade_qty = self.rng.gen_range(1..=self.config.avg_trade_size * 2) as i32;

        // Consume liquidity
        if is_buy_aggressor {
            if let Some((_, qty)) = self.ask_levels.first_mut() {
                *qty = (*qty - trade_qty).max(1);
            }
        } else if let Some((_, qty)) = self.bid_levels.first_mut() {
            *qty = (*qty - trade_qty).max(1);
        }

        MdpMessage::Trade(TradeUpdate {
            transact_time: self.current_time,
            match_event_indicator: 0x81,
            security_id: self.config.security_id,
            rpt_seq: self.sequence,
            exponent: self.config.exponent,
            entries: vec![TradeEntry {
                price: trade_price,
                quantity: trade_qty,
                num_orders: 1,
                aggressor_side: if is_buy_aggressor { 0 } else { 1 },
                action: UpdateAction::New,
            }],
        })
    }

    /// Update book levels based on current mid
    fn update_book_levels(&mut self) {
        let half_spread = (self.config.avg_spread_ticks as i64 * self.config.tick_size) / 2;
        let best_bid = self.current_mid - half_spread;
        let best_ask = self.current_mid + half_spread;

        // Pre-generate random values to avoid borrow conflicts
        let bid_len = self.bid_levels.len();
        let ask_len = self.ask_levels.len();
        
        let bid_updates: Vec<(bool, i32)> = (0..bid_len)
            .map(|_| (self.rng.gen_bool(0.3), self.random_quantity()))
            .collect();
        let ask_updates: Vec<(bool, i32)> = (0..ask_len)
            .map(|_| (self.rng.gen_bool(0.3), self.random_quantity()))
            .collect();

        for (i, level) in self.bid_levels.iter_mut().enumerate() {
            level.0 = best_bid - (i as i64 * self.config.tick_size);
            // Randomly adjust quantity
            if bid_updates[i].0 {
                level.1 = bid_updates[i].1;
            }
        }

        for (i, level) in self.ask_levels.iter_mut().enumerate() {
            level.0 = best_ask + (i as i64 * self.config.tick_size);
            if ask_updates[i].0 {
                level.1 = ask_updates[i].1;
            }
        }
    }

    /// Generate a snapshot of the current book state
    pub fn generate_snapshot(&self) -> MdpMessage {
        let mut entries = Vec::new();

        for (i, (price, qty)) in self.bid_levels.iter().enumerate() {
            entries.push(SnapshotEntry {
                price: *price,
                quantity: *qty,
                num_orders: 5,
                price_level: (i + 1) as u8,
                entry_type: EntryType::Bid,
            });
        }

        for (i, (price, qty)) in self.ask_levels.iter().enumerate() {
            entries.push(SnapshotEntry {
                price: *price,
                quantity: *qty,
                num_orders: 5,
                price_level: (i + 1) as u8,
                entry_type: EntryType::Offer,
            });
        }

        MdpMessage::Snapshot(Snapshot {
            last_update_time: self.current_time,
            security_id: self.config.security_id,
            rpt_seq: self.sequence,
            exponent: self.config.exponent,
            entries,
        })
    }

    /// Get current mid price
    #[must_use]
    pub fn current_mid(&self) -> Price {
        Price::from_raw(self.current_mid)
    }

    /// Get current timestamp
    #[must_use]
    pub fn current_timestamp(&self) -> Timestamp {
        Timestamp::from_nanos(self.current_time as i64)
    }

    /// Get best bid
    #[must_use]
    pub fn best_bid(&self) -> Option<(Price, Quantity)> {
        self.bid_levels.first().map(|(p, q)| {
            (Price::from_raw(*p), Quantity::new(*q as u32))
        })
    }

    /// Get best ask
    #[must_use]
    pub fn best_ask(&self) -> Option<(Price, Quantity)> {
        self.ask_levels.first().map(|(p, q)| {
            (Price::from_raw(*p), Quantity::new(*q as u32))
        })
    }

    /// Generate N events
    pub fn generate_n(&mut self, n: usize) -> Vec<MdpMessage> {
        (0..n).map(|_| self.next_event()).collect()
    }

    /// Create an iterator over generated events
    pub fn iter(&mut self) -> SyntheticIterator<'_> {
        SyntheticIterator { generator: self }
    }
}

/// Iterator over synthetic events
pub struct SyntheticIterator<'a> {
    generator: &'a mut SyntheticGenerator,
}

impl<'a> Iterator for SyntheticIterator<'a> {
    type Item = MdpMessage;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.generator.next_event())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_generator() {
        let config = SyntheticConfig::es_futures();
        let mut gen = SyntheticGenerator::new(config);

        let events = gen.generate_n(100);
        assert_eq!(events.len(), 100);

        // Should have mix of trades and book updates
        let trades = events.iter().filter(|e| matches!(e, MdpMessage::Trade(_))).count();
        let updates = events.iter().filter(|e| matches!(e, MdpMessage::BookUpdate(_))).count();

        assert!(trades > 0, "Should have some trades");
        assert!(updates > 0, "Should have some book updates");
    }

    #[test]
    fn test_snapshot_generation() {
        let config = SyntheticConfig::default();
        let gen = SyntheticGenerator::new(config);

        let snapshot = gen.generate_snapshot();
        if let MdpMessage::Snapshot(s) = snapshot {
            assert!(!s.entries.is_empty());
        } else {
            panic!("Expected snapshot message");
        }
    }

    #[test]
    fn test_deterministic_with_seed() {
        let config = SyntheticConfig::default();

        let mut gen1 = SyntheticGenerator::with_seed(config.clone(), 123);
        let mut gen2 = SyntheticGenerator::with_seed(config, 123);

        let events1 = gen1.generate_n(10);
        let events2 = gen2.generate_n(10);

        for (e1, e2) in events1.iter().zip(events2.iter()) {
            assert_eq!(e1.timestamp(), e2.timestamp());
        }
    }
}

