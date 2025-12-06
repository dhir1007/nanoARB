//! Signal-based strategy implementation.

use nano_core::traits::{OrderBook, Strategy};
use nano_core::types::{Fill, Order, OrderId, Quantity, Side, TimeInForce, Timestamp};

use crate::base::BaseStrategy;

/// Signal configuration for trading
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Minimum prediction magnitude
    pub min_magnitude: f32,
    /// Position sizing based on confidence
    pub confidence_scaling: bool,
    /// Maximum position size (as fraction)
    pub max_position_size: f32,
    /// Target profit in ticks
    pub target_ticks: i64,
    /// Stop loss in ticks
    pub stop_ticks: i64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.55,
            min_magnitude: 0.001,
            confidence_scaling: true,
            max_position_size: 1.0,
            target_ticks: 10,
            stop_ticks: 5,
        }
    }
}

/// A trading signal
#[derive(Debug, Clone)]
pub struct Signal {
    /// Signal direction: -1 (sell), 0 (neutral), +1 (buy)
    pub direction: i8,
    /// Signal strength (0 to 1)
    pub strength: f32,
    /// Confidence from model
    pub confidence: f32,
    /// Signal timestamp
    pub timestamp: Timestamp,
}

impl Signal {
    /// Create a buy signal
    #[must_use]
    pub fn buy(strength: f32, confidence: f32, timestamp: Timestamp) -> Self {
        Self {
            direction: 1,
            strength,
            confidence,
            timestamp,
        }
    }

    /// Create a sell signal
    #[must_use]
    pub fn sell(strength: f32, confidence: f32, timestamp: Timestamp) -> Self {
        Self {
            direction: -1,
            strength,
            confidence,
            timestamp,
        }
    }

    /// Create a neutral signal
    #[must_use]
    pub fn neutral(timestamp: Timestamp) -> Self {
        Self {
            direction: 0,
            strength: 0.0,
            confidence: 1.0,
            timestamp,
        }
    }

    /// Check if signal suggests buying
    #[must_use]
    pub fn is_buy(&self) -> bool {
        self.direction > 0
    }

    /// Check if signal suggests selling
    #[must_use]
    pub fn is_sell(&self) -> bool {
        self.direction < 0
    }

    /// Check if signal is neutral
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        self.direction == 0
    }

    /// Get the side for order placement
    #[must_use]
    pub fn side(&self) -> Option<Side> {
        match self.direction {
            d if d > 0 => Some(Side::Buy),
            d if d < 0 => Some(Side::Sell),
            _ => None,
        }
    }
}

/// Signal-based trading strategy
pub struct SignalStrategy {
    /// Base strategy
    base: BaseStrategy,
    /// Signal configuration
    config: SignalConfig,
    /// Instrument ID
    instrument_id: u32,
    /// Base order size
    order_size: u32,
    /// Maximum position
    max_position: i64,
    /// Current pending order
    pending_order: Option<OrderId>,
    /// Last signal
    last_signal: Option<Signal>,
    /// Order ID counter
    next_order_id: u64,
}

impl SignalStrategy {
    /// Create a new signal strategy
    #[must_use]
    pub fn new(
        name: &str,
        instrument_id: u32,
        signal_config: SignalConfig,
        order_size: u32,
        max_position: i64,
        tick_value: f64,
    ) -> Self {
        Self {
            base: BaseStrategy::new(name, tick_value),
            config: signal_config,
            instrument_id,
            order_size,
            max_position,
            pending_order: None,
            last_signal: None,
            next_order_id: 1,
        }
    }

    /// Process a signal and generate orders
    pub fn process_signal(&mut self, signal: &Signal, book: &dyn OrderBook) -> Vec<Order> {
        let mut orders = Vec::new();

        // Don't trade if we have a pending order
        if self.pending_order.is_some() {
            return orders;
        }

        // Check signal confidence
        if signal.confidence < self.config.min_confidence {
            return orders;
        }

        self.last_signal = Some(signal.clone());

        // Check if signal suggests trading
        let side = match signal.side() {
            Some(s) => s,
            None => return orders, // Neutral signal
        };

        // Get current price
        let current_price = match book.mid_price() {
            Some(p) => p,
            None => return orders,
        };

        // Check position limits
        let current_pos = self.base.position();
        let order_qty = self.calculate_order_size(signal, current_pos);

        if order_qty == 0 {
            return orders;
        }

        // Calculate order price
        let order_price = match side {
            Side::Buy => book.best_bid().map_or(current_price, |(p, _)| p),
            Side::Sell => book.best_ask().map_or(current_price, |(p, _)| p),
        };

        let order_id = OrderId::new(self.next_order_id);
        self.next_order_id += 1;

        let order = Order::new_limit(
            order_id,
            self.instrument_id,
            side,
            order_price,
            Quantity::new(order_qty),
            TimeInForce::IOC,
        );

        self.pending_order = Some(order_id);
        orders.push(order);

        orders
    }

    /// Calculate order size based on signal and position
    fn calculate_order_size(&self, signal: &Signal, current_pos: i64) -> u32 {
        let base_size = if self.config.confidence_scaling {
            (self.order_size as f32 * signal.strength) as u32
        } else {
            self.order_size
        };

        // Check position limits
        let max_buy = (self.max_position - current_pos).max(0) as u32;
        let max_sell = (self.max_position + current_pos).max(0) as u32;

        match signal.side() {
            Some(Side::Buy) => base_size.min(max_buy),
            Some(Side::Sell) => base_size.min(max_sell),
            None => 0,
        }
    }

    /// Get last signal
    #[must_use]
    pub fn last_signal(&self) -> Option<&Signal> {
        self.last_signal.as_ref()
    }
}

impl Strategy for SignalStrategy {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn on_market_data(&mut self, book: &dyn OrderBook) -> Vec<Order> {
        self.base.on_market_data(book);
        Vec::new()
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.base.on_fill(fill);

        if Some(fill.order_id) == self.pending_order {
            self.pending_order = None;
        }
    }

    fn on_order_ack(&mut self, _order_id: OrderId) {}

    fn on_order_reject(&mut self, order_id: OrderId, _reason: &str) {
        if Some(order_id) == self.pending_order {
            self.pending_order = None;
        }
    }

    fn on_order_cancel(&mut self, order_id: OrderId) {
        if Some(order_id) == self.pending_order {
            self.pending_order = None;
        }
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
        self.pending_order = None;
        self.last_signal = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let buy = Signal::buy(0.8, 0.75, Timestamp::now());
        assert!(buy.is_buy());
        assert_eq!(buy.direction, 1);
        assert_eq!(buy.side(), Some(Side::Buy));

        let sell = Signal::sell(0.7, 0.65, Timestamp::now());
        assert!(sell.is_sell());
        assert_eq!(sell.side(), Some(Side::Sell));

        let neutral = Signal::neutral(Timestamp::now());
        assert!(neutral.is_neutral());
        assert_eq!(neutral.side(), None);
    }

    #[test]
    fn test_signal_strategy_creation() {
        let config = SignalConfig::default();
        let strategy = SignalStrategy::new("test_signal", 1, config, 10, 50, 12.5);

        assert_eq!(strategy.position(), 0);
    }
}
