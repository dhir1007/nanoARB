//! Market-making strategy implementation.

use std::collections::HashMap;

use nano_core::traits::{OrderBook, Strategy};
use nano_core::types::{Fill, Order, OrderId, OrderType, Price, Quantity, Side, TimeInForce, Timestamp};
use serde::{Deserialize, Serialize};

use crate::base::{BaseStrategy, StrategyState};

/// Market maker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakerConfig {
    /// Base spread in ticks
    pub base_spread_ticks: i64,
    /// Spread skew based on inventory (-1 to +1)
    pub inventory_skew_factor: f64,
    /// Maximum inventory (position limit)
    pub max_inventory: i64,
    /// Order size per level
    pub order_size: u32,
    /// Number of levels to quote
    pub num_levels: usize,
    /// Minimum edge required (in ticks)
    pub min_edge_ticks: i64,
    /// Cancel distance from BBO (in ticks)
    pub cancel_distance_ticks: i64,
    /// Tick size
    pub tick_size: i64,
    /// Refresh interval (in nanoseconds)
    pub refresh_interval_ns: i64,
}

impl Default for MarketMakerConfig {
    fn default() -> Self {
        Self {
            base_spread_ticks: 2,
            inventory_skew_factor: 0.5,
            max_inventory: 50,
            order_size: 5,
            num_levels: 3,
            min_edge_ticks: 1,
            cancel_distance_ticks: 10,
            tick_size: 25,
            refresh_interval_ns: 100_000_000, // 100ms
        }
    }
}

/// Quote management for tracking open orders
#[derive(Debug, Default)]
pub struct QuoteManager {
    /// Active bid orders
    bid_orders: HashMap<OrderId, (Price, Quantity)>,
    /// Active ask orders
    ask_orders: HashMap<OrderId, (Price, Quantity)>,
    /// Next order ID
    next_order_id: u64,
    /// Orders pending acknowledgment
    pending_acks: HashMap<OrderId, Side>,
}

impl QuoteManager {
    /// Create a new quote manager
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate next order ID
    pub fn next_order_id(&mut self) -> OrderId {
        self.next_order_id += 1;
        OrderId::new(self.next_order_id)
    }

    /// Record a submitted order
    pub fn on_order_submit(&mut self, order_id: OrderId, side: Side, price: Price, quantity: Quantity) {
        self.pending_acks.insert(order_id, side);
        match side {
            Side::Buy => {
                self.bid_orders.insert(order_id, (price, quantity));
            }
            Side::Sell => {
                self.ask_orders.insert(order_id, (price, quantity));
            }
        }
    }

    /// Handle order acknowledgment
    pub fn on_order_ack(&mut self, order_id: OrderId) {
        self.pending_acks.remove(&order_id);
    }

    /// Handle order rejection
    pub fn on_order_reject(&mut self, order_id: OrderId) {
        self.pending_acks.remove(&order_id);
        self.bid_orders.remove(&order_id);
        self.ask_orders.remove(&order_id);
    }

    /// Handle order fill
    pub fn on_fill(&mut self, order_id: OrderId, fill_qty: Quantity) {
        // Update remaining quantity
        if let Some((_, qty)) = self.bid_orders.get_mut(&order_id) {
            *qty = qty.saturating_sub(fill_qty);
            if qty.is_zero() {
                self.bid_orders.remove(&order_id);
            }
        }
        if let Some((_, qty)) = self.ask_orders.get_mut(&order_id) {
            *qty = qty.saturating_sub(fill_qty);
            if qty.is_zero() {
                self.ask_orders.remove(&order_id);
            }
        }
    }

    /// Handle order cancellation
    pub fn on_cancel(&mut self, order_id: OrderId) {
        self.bid_orders.remove(&order_id);
        self.ask_orders.remove(&order_id);
    }

    /// Get all bid order IDs
    pub fn bid_order_ids(&self) -> Vec<OrderId> {
        self.bid_orders.keys().copied().collect()
    }

    /// Get all ask order IDs
    pub fn ask_order_ids(&self) -> Vec<OrderId> {
        self.ask_orders.keys().copied().collect()
    }

    /// Get total bid quantity
    #[must_use]
    pub fn total_bid_quantity(&self) -> Quantity {
        self.bid_orders
            .values()
            .fold(Quantity::ZERO, |acc, (_, q)| acc.saturating_add(*q))
    }

    /// Get total ask quantity
    #[must_use]
    pub fn total_ask_quantity(&self) -> Quantity {
        self.ask_orders
            .values()
            .fold(Quantity::ZERO, |acc, (_, q)| acc.saturating_add(*q))
    }

    /// Clear all orders
    pub fn clear(&mut self) {
        self.bid_orders.clear();
        self.ask_orders.clear();
        self.pending_acks.clear();
    }
}

/// Market-making strategy
pub struct MarketMakerStrategy {
    /// Base strategy
    base: BaseStrategy,
    /// Configuration
    config: MarketMakerConfig,
    /// Quote manager
    quotes: QuoteManager,
    /// Instrument ID
    instrument_id: u32,
    /// Last quote update time
    last_quote_time: Timestamp,
    /// Fair value estimate
    fair_value: Option<Price>,
}

impl MarketMakerStrategy {
    /// Create a new market-making strategy
    #[must_use]
    pub fn new(name: &str, instrument_id: u32, config: MarketMakerConfig, tick_value: f64) -> Self {
        Self {
            base: BaseStrategy::new(name, tick_value),
            config,
            quotes: QuoteManager::new(),
            instrument_id,
            last_quote_time: Timestamp::EPOCH,
            fair_value: None,
        }
    }

    /// Calculate skewed quote prices based on inventory
    fn calculate_quotes(&self, mid: Price) -> (Price, Price) {
        let position = self.base.position();
        let max_inv = self.config.max_inventory as f64;

        // Calculate inventory skew (-1 to +1)
        let inv_ratio = if max_inv > 0.0 {
            (position as f64 / max_inv).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Skew quotes to reduce inventory
        // Positive inventory -> lower bid, higher ask
        let skew_ticks = (inv_ratio * self.config.inventory_skew_factor
            * self.config.base_spread_ticks as f64) as i64;

        let half_spread = self.config.base_spread_ticks * self.config.tick_size / 2;

        let bid_price = Price::from_raw(mid.raw() - half_spread - skew_ticks * self.config.tick_size);
        let ask_price = Price::from_raw(mid.raw() + half_spread - skew_ticks * self.config.tick_size);

        (bid_price, ask_price)
    }

    /// Check if quotes need to be refreshed
    fn should_refresh_quotes(&self, current_time: Timestamp) -> bool {
        current_time.as_nanos() - self.last_quote_time.as_nanos() >= self.config.refresh_interval_ns
    }

    /// Generate orders to cancel stale quotes
    fn generate_cancels(&self, book: &dyn OrderBook) -> Vec<OrderId> {
        let mut cancels = Vec::new();

        if let (Some((best_bid, _)), Some((best_ask, _))) = (book.best_bid(), book.best_ask()) {
            // Cancel bids too far from BBO
            for (id, (price, _)) in &self.quotes.bid_orders {
                let distance = (best_bid.raw() - price.raw()) / self.config.tick_size;
                if distance > self.config.cancel_distance_ticks {
                    cancels.push(*id);
                }
            }

            // Cancel asks too far from BBO
            for (id, (price, _)) in &self.quotes.ask_orders {
                let distance = (price.raw() - best_ask.raw()) / self.config.tick_size;
                if distance > self.config.cancel_distance_ticks {
                    cancels.push(*id);
                }
            }
        }

        cancels
    }

    /// Generate new quote orders
    fn generate_quotes(&mut self, book: &dyn OrderBook, current_time: Timestamp) -> Vec<Order> {
        let mut orders = Vec::new();

        let mid = match book.mid_price() {
            Some(m) => m,
            None => return orders,
        };

        self.fair_value = Some(mid);

        // Check position limits
        let position = self.base.position();
        let can_buy = position < self.config.max_inventory;
        let can_sell = position > -self.config.max_inventory;

        let (bid_price, ask_price) = self.calculate_quotes(mid);

        // Generate bid orders
        if can_buy {
            for level in 0..self.config.num_levels {
                let price = Price::from_raw(
                    bid_price.raw() - (level as i64 * self.config.tick_size)
                );

                let order_id = self.quotes.next_order_id();
                let quantity = Quantity::new(self.config.order_size);

                let order = Order::new_limit(
                    order_id,
                    self.instrument_id,
                    Side::Buy,
                    price,
                    quantity,
                    TimeInForce::GTC,
                );

                self.quotes.on_order_submit(order_id, Side::Buy, price, quantity);
                orders.push(order);
            }
        }

        // Generate ask orders
        if can_sell {
            for level in 0..self.config.num_levels {
                let price = Price::from_raw(
                    ask_price.raw() + (level as i64 * self.config.tick_size)
                );

                let order_id = self.quotes.next_order_id();
                let quantity = Quantity::new(self.config.order_size);

                let order = Order::new_limit(
                    order_id,
                    self.instrument_id,
                    Side::Sell,
                    price,
                    quantity,
                    TimeInForce::GTC,
                );

                self.quotes.on_order_submit(order_id, Side::Sell, price, quantity);
                orders.push(order);
            }
        }

        self.last_quote_time = current_time;
        orders
    }

    /// Get the quote manager
    #[must_use]
    pub fn quotes(&self) -> &QuoteManager {
        &self.quotes
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MarketMakerConfig {
        &self.config
    }

    /// Get fair value estimate
    #[must_use]
    pub fn fair_value(&self) -> Option<Price> {
        self.fair_value
    }
}

impl Strategy for MarketMakerStrategy {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn on_market_data(&mut self, book: &dyn OrderBook) -> Vec<Order> {
        // Update base strategy
        self.base.on_market_data(book);

        if !self.is_ready() {
            return Vec::new();
        }

        let current_time = book.timestamp();

        // Check if we need to refresh quotes
        if self.should_refresh_quotes(current_time) {
            // In a real implementation, we'd cancel old orders first
            // For simplicity, just generate new quotes
            return self.generate_quotes(book, current_time);
        }

        Vec::new()
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.base.on_fill(fill);
        self.quotes.on_fill(fill.order_id, fill.quantity);
    }

    fn on_order_ack(&mut self, order_id: OrderId) {
        self.quotes.on_order_ack(order_id);
    }

    fn on_order_reject(&mut self, order_id: OrderId, reason: &str) {
        self.quotes.on_order_reject(order_id);
        tracing::warn!("Order {} rejected: {}", order_id, reason);
    }

    fn on_order_cancel(&mut self, order_id: OrderId) {
        self.quotes.on_cancel(order_id);
    }

    fn position(&self) -> i64 {
        self.base.position()
    }

    fn pnl(&self) -> f64 {
        self.base.pnl()
    }

    fn is_ready(&self) -> bool {
        self.base.is_ready()
    }

    fn reset(&mut self) {
        self.base.reset();
        self.quotes.clear();
        self.last_quote_time = Timestamp::EPOCH;
        self.fair_value = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quote_manager() {
        let mut manager = QuoteManager::new();

        let order_id = manager.next_order_id();
        manager.on_order_submit(order_id, Side::Buy, Price::from_raw(50000), Quantity::new(10));

        assert_eq!(manager.total_bid_quantity().value(), 10);
        assert_eq!(manager.bid_order_ids().len(), 1);

        manager.on_fill(order_id, Quantity::new(5));
        assert_eq!(manager.total_bid_quantity().value(), 5);

        manager.on_fill(order_id, Quantity::new(5));
        assert_eq!(manager.total_bid_quantity().value(), 0);
        assert!(manager.bid_order_ids().is_empty());
    }

    #[test]
    fn test_inventory_skew() {
        let config = MarketMakerConfig {
            base_spread_ticks: 2,
            inventory_skew_factor: 0.5,
            max_inventory: 100,
            tick_size: 25,
            ..Default::default()
        };

        let mut strategy = MarketMakerStrategy::new("test", 1, config, 12.5);
        strategy.base.set_state(StrategyState::Trading);

        let mid = Price::from_raw(50000);
        let (bid, ask) = strategy.calculate_quotes(mid);

        // With no inventory, quotes should be symmetric
        assert!(bid.raw() < mid.raw());
        assert!(ask.raw() > mid.raw());
    }
}

