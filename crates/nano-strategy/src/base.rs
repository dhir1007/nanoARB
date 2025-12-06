//! Base strategy implementation.

use nano_core::traits::{OrderBook, Strategy};
use nano_core::types::{Fill, Order, OrderId, Price, Side};
use serde::{Deserialize, Serialize};

/// Strategy state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyState {
    /// Strategy is initializing
    Initializing,
    /// Strategy is ready to trade
    Ready,
    /// Strategy is actively trading
    Trading,
    /// Strategy is paused
    Paused,
    /// Strategy is stopped
    Stopped,
    /// Strategy encountered an error
    Error,
}

/// Base strategy with common functionality
pub struct BaseStrategy {
    /// Strategy name
    name: String,
    /// Current state
    state: StrategyState,
    /// Current position
    position: i64,
    /// Realized P&L
    realized_pnl: f64,
    /// Unrealized P&L
    unrealized_pnl: f64,
    /// Total fees paid
    total_fees: f64,
    /// Average entry price
    avg_entry_price: i64,
    /// Number of fills
    fill_count: u32,
    /// Number of round trips
    round_trips: u32,
    /// Last mid price
    last_mid: Option<Price>,
    /// Tick value for P&L calculation
    tick_value: f64,
}

impl BaseStrategy {
    /// Create a new base strategy
    #[must_use]
    pub fn new(name: &str, tick_value: f64) -> Self {
        Self {
            name: name.to_string(),
            state: StrategyState::Initializing,
            position: 0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            total_fees: 0.0,
            avg_entry_price: 0,
            fill_count: 0,
            round_trips: 0,
            last_mid: None,
            tick_value,
        }
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> StrategyState {
        self.state
    }

    /// Set state
    pub fn set_state(&mut self, state: StrategyState) {
        self.state = state;
    }

    /// Update position from a fill
    pub fn update_position(&mut self, fill: &Fill) {
        let fill_qty = i64::from(fill.quantity.value());
        let signed_qty = if fill.side == Side::Buy {
            fill_qty
        } else {
            -fill_qty
        };
        let fill_price = fill.price.raw();

        self.fill_count += 1;
        self.total_fees += fill.fee;

        // Check if adding to or reducing position
        let same_direction =
            (self.position >= 0 && signed_qty > 0) || (self.position <= 0 && signed_qty < 0);

        if same_direction || self.position == 0 {
            // Adding to position
            let old_notional = self.position.abs() * self.avg_entry_price;
            let new_notional = fill_qty * fill_price;
            let new_total_qty = self.position.abs() + fill_qty;

            if new_total_qty > 0 {
                self.avg_entry_price = (old_notional + new_notional) / new_total_qty;
            }
            self.position += signed_qty;
        } else {
            // Reducing position
            let reduce_qty = fill_qty.min(self.position.abs());
            let price_diff = if self.position > 0 {
                fill_price - self.avg_entry_price
            } else {
                self.avg_entry_price - fill_price
            };

            let pnl_ticks = price_diff * reduce_qty;
            self.realized_pnl += pnl_ticks as f64 * self.tick_value / 100.0;

            let prev_position = self.position;
            self.position += signed_qty;

            // Check for round trip
            if self.position == 0 || self.position.signum() != prev_position.signum() {
                self.round_trips += 1;
                if self.position != 0 {
                    self.avg_entry_price = fill_price;
                } else {
                    self.avg_entry_price = 0;
                }
            }
        }
    }

    /// Update unrealized P&L
    pub fn update_unrealized(&mut self, current_price: Price) {
        if self.position == 0 {
            self.unrealized_pnl = 0.0;
            return;
        }

        let price_diff = current_price.raw() - self.avg_entry_price;
        let pnl_ticks = price_diff * self.position;
        self.unrealized_pnl = pnl_ticks as f64 * self.tick_value / 100.0;
    }

    /// Get total P&L
    #[must_use]
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl - self.total_fees
    }

    /// Get net P&L (after fees)
    #[must_use]
    pub fn net_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl - self.total_fees
    }

    /// Check if position is flat
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.position == 0
    }

    /// Get position size (absolute)
    #[must_use]
    pub fn position_size(&self) -> u64 {
        self.position.unsigned_abs()
    }

    /// Get fill count
    #[must_use]
    pub fn fill_count(&self) -> u32 {
        self.fill_count
    }

    /// Get round trip count
    #[must_use]
    pub fn round_trips(&self) -> u32 {
        self.round_trips
    }

    /// Reset the strategy state
    pub fn reset(&mut self) {
        self.state = StrategyState::Initializing;
        self.position = 0;
        self.realized_pnl = 0.0;
        self.unrealized_pnl = 0.0;
        self.total_fees = 0.0;
        self.avg_entry_price = 0;
        self.fill_count = 0;
        self.round_trips = 0;
        self.last_mid = None;
    }
}

impl Strategy for BaseStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_market_data(&mut self, book: &dyn OrderBook) -> Vec<Order> {
        self.last_mid = book.mid_price();

        if let Some(mid) = self.last_mid {
            self.update_unrealized(mid);
        }

        // Base strategy doesn't generate orders
        Vec::new()
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.update_position(fill);
    }

    fn on_order_ack(&mut self, _order_id: OrderId) {
        // Base implementation does nothing
    }

    fn on_order_reject(&mut self, _order_id: OrderId, _reason: &str) {
        // Base implementation does nothing
    }

    fn on_order_cancel(&mut self, _order_id: OrderId) {
        // Base implementation does nothing
    }

    fn position(&self) -> i64 {
        self.position
    }

    fn pnl(&self) -> f64 {
        self.total_pnl()
    }

    fn is_ready(&self) -> bool {
        matches!(self.state, StrategyState::Ready | StrategyState::Trading)
    }

    fn reset(&mut self) {
        BaseStrategy::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nano_core::types::{Price, Quantity, Timestamp};

    fn create_fill(side: Side, price: i64, qty: u32, fee: f64) -> Fill {
        Fill {
            order_id: OrderId::new(1),
            price: Price::from_raw(price),
            quantity: Quantity::new(qty),
            side,
            is_maker: true,
            timestamp: Timestamp::now(),
            fee,
        }
    }

    #[test]
    fn test_position_tracking() {
        let mut strategy = BaseStrategy::new("test", 12.5);

        // Buy 10 @ 50000
        let buy = create_fill(Side::Buy, 50000, 10, 2.5);
        strategy.update_position(&buy);

        assert_eq!(strategy.position, 10);
        assert_eq!(strategy.avg_entry_price, 50000);

        // Sell 10 @ 50010 (10 tick profit)
        let sell = create_fill(Side::Sell, 50010, 10, 2.5);
        strategy.update_position(&sell);

        assert_eq!(strategy.position, 0);
        assert_eq!(strategy.round_trips, 1);
        assert!(strategy.realized_pnl > 0.0);
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut strategy = BaseStrategy::new("test", 12.5);

        // Buy 10 @ 50000
        let buy = create_fill(Side::Buy, 50000, 10, 2.5);
        strategy.update_position(&buy);

        // Market moves to 50020
        strategy.update_unrealized(Price::from_raw(50020));

        // 20 ticks * 10 contracts * $12.5 / 100 = $25
        assert!((strategy.unrealized_pnl - 25.0).abs() < 0.1);
    }
}
