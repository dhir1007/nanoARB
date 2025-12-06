//! Core traits for the trading engine.

use crate::types::{Fill, Order, OrderId, Price, Quantity, Quote, Side, Timestamp};

/// Trait for order book implementations
pub trait OrderBook: Send + Sync {
    /// Get the best bid price and quantity
    fn best_bid(&self) -> Option<(Price, Quantity)>;

    /// Get the best ask price and quantity
    fn best_ask(&self) -> Option<(Price, Quantity)>;

    /// Get the mid price
    fn mid_price(&self) -> Option<Price>;

    /// Get the spread in ticks
    fn spread(&self) -> Option<Price>;

    /// Get the current quote (BBO)
    fn quote(&self) -> Option<Quote>;

    /// Get price at a specific level (0 = best)
    fn bid_at_level(&self, level: usize) -> Option<(Price, Quantity)>;

    /// Get ask at a specific level (0 = best)
    fn ask_at_level(&self, level: usize) -> Option<(Price, Quantity)>;

    /// Get total quantity at bid levels up to depth
    fn bid_depth(&self, levels: usize) -> Quantity;

    /// Get total quantity at ask levels up to depth
    fn ask_depth(&self, levels: usize) -> Quantity;

    /// Get the current timestamp
    fn timestamp(&self) -> Timestamp;
}

/// Trait for market data events
pub trait MarketDataEvent: Send + Sync {
    /// Get the event timestamp
    fn timestamp(&self) -> Timestamp;

    /// Get the instrument ID
    fn instrument_id(&self) -> u32;
}

/// Trait for trading strategies
pub trait Strategy: Send + Sync {
    /// Strategy name for logging and metrics
    fn name(&self) -> &str;

    /// Called on each market data update
    fn on_market_data(&mut self, book: &dyn OrderBook) -> Vec<Order>;

    /// Called when an order is filled
    fn on_fill(&mut self, fill: &Fill);

    /// Called when an order is acknowledged
    fn on_order_ack(&mut self, order_id: OrderId);

    /// Called when an order is rejected
    fn on_order_reject(&mut self, order_id: OrderId, reason: &str);

    /// Called when an order is cancelled
    fn on_order_cancel(&mut self, order_id: OrderId);

    /// Get current position
    fn position(&self) -> i64;

    /// Get current P&L
    fn pnl(&self) -> f64;

    /// Check if strategy is ready to trade
    fn is_ready(&self) -> bool;

    /// Reset strategy state
    fn reset(&mut self);
}

/// Trait for execution handlers
pub trait ExecutionHandler: Send + Sync {
    /// Submit a new order
    fn submit_order(&mut self, order: Order) -> Result<OrderId, crate::Error>;

    /// Cancel an existing order
    fn cancel_order(&mut self, order_id: OrderId) -> Result<(), crate::Error>;

    /// Modify an existing order
    fn modify_order(
        &mut self,
        order_id: OrderId,
        new_price: Option<Price>,
        new_quantity: Option<Quantity>,
    ) -> Result<(), crate::Error>;

    /// Get order by ID
    fn get_order(&self, order_id: OrderId) -> Option<&Order>;

    /// Get all active orders
    fn active_orders(&self) -> Vec<&Order>;
}

/// Trait for risk management
pub trait RiskManager: Send + Sync {
    /// Check if an order passes risk checks
    fn check_order(&self, order: &Order, current_position: i64) -> Result<(), crate::Error>;

    /// Check if position limits are breached
    fn check_position(&self, position: i64) -> Result<(), crate::Error>;

    /// Check if drawdown limits are breached
    fn check_drawdown(&self, pnl: f64, peak_pnl: f64) -> Result<(), crate::Error>;

    /// Check if we should kill all positions
    fn should_kill_switch(&self, pnl: f64, position: i64) -> bool;

    /// Get maximum allowed position
    fn max_position(&self) -> i64;

    /// Get maximum allowed order size
    fn max_order_size(&self) -> u32;
}

/// Trait for latency models in backtesting
pub trait LatencyModel: Send + Sync {
    /// Get the latency for sending an order to the exchange
    fn order_latency(&self) -> i64;

    /// Get the latency for receiving market data from the exchange
    fn market_data_latency(&self) -> i64;

    /// Get the latency for receiving order acknowledgments
    fn ack_latency(&self) -> i64;

    /// Reset any internal state (e.g., for random models)
    fn reset(&mut self);
}

/// Trait for fill models in backtesting
pub trait FillModel: Send + Sync {
    /// Simulate a fill attempt for an order
    fn try_fill(
        &self,
        order: &Order,
        book: &dyn OrderBook,
        current_time: Timestamp,
    ) -> Option<(Price, Quantity)>;

    /// Get probability of being filled at a given queue position
    fn fill_probability(&self, queue_position: usize, level_quantity: Quantity) -> f64;
}

/// Trait for fee models
pub trait FeeModel: Send + Sync {
    /// Calculate fee for a fill
    fn calculate_fee(&self, price: Price, quantity: Quantity, is_maker: bool, side: Side) -> f64;
}

/// Trait for ML model inference
pub trait ModelInference: Send + Sync {
    /// Input type for the model
    type Input;

    /// Output type for the model
    type Output;

    /// Run inference on the input
    fn predict(&self, input: &Self::Input) -> Result<Self::Output, crate::Error>;

    /// Get the model name
    fn name(&self) -> &str;

    /// Get the expected latency in nanoseconds
    fn expected_latency_ns(&self) -> u64;
}

/// Trait for metrics collection
pub trait MetricsCollector: Send + Sync {
    /// Record a latency measurement
    fn record_latency(&self, name: &str, latency_ns: u64);

    /// Increment a counter
    fn increment_counter(&self, name: &str, value: u64);

    /// Set a gauge value
    fn set_gauge(&self, name: &str, value: f64);

    /// Record a P&L update
    fn record_pnl(&self, pnl: f64);

    /// Record a fill
    fn record_fill(&self, side: Side, quantity: Quantity, is_maker: bool);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that traits can be used as trait objects
    #[test]
    fn test_trait_object_safety() {
        fn _takes_order_book(_: &dyn OrderBook) {}
        fn _takes_strategy(_: &dyn Strategy) {}
        fn _takes_risk_manager(_: &dyn RiskManager) {}
        fn _takes_latency_model(_: &dyn LatencyModel) {}
        fn _takes_fill_model(_: &dyn FillModel) {}
        fn _takes_fee_model(_: &dyn FeeModel) {}
        fn _takes_metrics(_: &dyn MetricsCollector) {}
    }
}

