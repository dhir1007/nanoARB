//! Position and P&L tracking.

use std::collections::HashMap;

use nano_core::types::{Fill, Instrument, Price, Side, Timestamp};
use serde::{Deserialize, Serialize};

/// Position tracker for a single instrument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Instrument ID
    pub instrument_id: u32,
    /// Net position (positive = long, negative = short)
    pub quantity: i64,
    /// Average entry price (in raw ticks)
    pub avg_price: i64,
    /// Total realized P&L
    pub realized_pnl: f64,
    /// Total fees paid
    pub total_fees: f64,
    /// Number of round trips completed
    pub round_trips: u32,
    /// Total quantity traded
    pub total_traded: u64,
    /// Last update timestamp
    pub last_update: Timestamp,
}

impl Position {
    /// Create a new flat position
    #[must_use]
    pub fn new(instrument_id: u32) -> Self {
        Self {
            instrument_id,
            quantity: 0,
            avg_price: 0,
            realized_pnl: 0.0,
            total_fees: 0.0,
            round_trips: 0,
            total_traded: 0,
            last_update: Timestamp::EPOCH,
        }
    }

    /// Apply a fill to the position
    pub fn apply_fill(&mut self, fill: &Fill, tick_value: f64) {
        let fill_qty = i64::from(fill.quantity.value());
        let signed_qty = if fill.side == Side::Buy {
            fill_qty
        } else {
            -fill_qty
        };
        let fill_price = fill.price.raw();

        self.total_traded += fill_qty as u64;
        self.total_fees += fill.fee;
        self.last_update = fill.timestamp;

        // Check if this is adding to position or reducing
        let same_direction =
            (self.quantity >= 0 && signed_qty > 0) || (self.quantity <= 0 && signed_qty < 0);

        if same_direction || self.quantity == 0 {
            // Adding to position - update average price
            let old_notional = self.quantity.abs() * self.avg_price;
            let new_notional = fill_qty * fill_price;
            let new_total_qty = self.quantity.abs() + fill_qty;

            if new_total_qty > 0 {
                self.avg_price = (old_notional + new_notional) / new_total_qty;
            }
            self.quantity += signed_qty;
        } else {
            // Reducing position - calculate realized P&L
            let reduce_qty = fill_qty.min(self.quantity.abs());

            // P&L = (exit_price - entry_price) * quantity * tick_value
            let price_diff = if self.quantity > 0 {
                fill_price - self.avg_price
            } else {
                self.avg_price - fill_price
            };

            let pnl_ticks = price_diff * reduce_qty;
            self.realized_pnl += pnl_ticks as f64 * tick_value / 100.0; // tick_value is in cents

            // Update position
            self.quantity += signed_qty;

            // Check if we've closed the position
            if self.quantity.signum() != (self.quantity - signed_qty).signum() {
                // Position crossed zero - this was a round trip
                self.round_trips += 1;

                // Handle any overshoot (position reversal)
                if self.quantity != 0 {
                    self.avg_price = fill_price;
                }
            } else if self.quantity == 0 {
                // Exactly closed
                self.round_trips += 1;
                self.avg_price = 0;
            }
        }
    }

    /// Calculate unrealized P&L at current market price
    #[must_use]
    pub fn unrealized_pnl(&self, current_price: Price, tick_value: f64) -> f64 {
        if self.quantity == 0 {
            return 0.0;
        }

        let price_diff = current_price.raw() - self.avg_price;
        let pnl_ticks = price_diff * self.quantity;
        pnl_ticks as f64 * tick_value / 100.0
    }

    /// Calculate total P&L (realized + unrealized)
    #[must_use]
    pub fn total_pnl(&self, current_price: Price, tick_value: f64) -> f64 {
        self.realized_pnl + self.unrealized_pnl(current_price, tick_value) - self.total_fees
    }

    /// Check if position is flat
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.quantity == 0
    }

    /// Check if position is long
    #[must_use]
    pub fn is_long(&self) -> bool {
        self.quantity > 0
    }

    /// Check if position is short
    #[must_use]
    pub fn is_short(&self) -> bool {
        self.quantity < 0
    }

    /// Get position size (absolute value)
    #[must_use]
    pub fn size(&self) -> u64 {
        self.quantity.unsigned_abs()
    }

    /// Reset position to flat
    pub fn reset(&mut self) {
        self.quantity = 0;
        self.avg_price = 0;
        self.realized_pnl = 0.0;
        self.total_fees = 0.0;
        self.round_trips = 0;
        self.total_traded = 0;
    }
}

/// Multi-instrument position tracker
#[derive(Debug, Default)]
pub struct PositionTracker {
    /// Positions by instrument ID
    positions: HashMap<u32, Position>,
    /// Instruments for P&L calculation
    instruments: HashMap<u32, Instrument>,
    /// Peak P&L for drawdown calculation
    peak_pnl: f64,
    /// Total realized P&L across all instruments
    total_realized: f64,
    /// Total fees paid across all instruments
    total_fees: f64,
}

impl PositionTracker {
    /// Create a new position tracker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an instrument
    pub fn register_instrument(&mut self, instrument: Instrument) {
        let id = instrument.id;
        self.instruments.insert(id, instrument);
        self.positions
            .entry(id)
            .or_insert_with(|| Position::new(id));
    }

    /// Apply a fill
    pub fn apply_fill(&mut self, fill: &Fill) {
        // Extract instrument_id from high bits of order_id (if encoded that way)
        // or default to 1 if not encoded
        let instrument_id = (fill.order_id.value() >> 32) as u32;
        let instrument_id = if instrument_id == 0 { 1 } else { instrument_id };

        // Try to get instrument, default to ES-like tick value
        let tick_value = self
            .instruments
            .get(&instrument_id)
            .map_or(12.5, |i| i.tick_value as f64 / 100.0);

        let position = self
            .positions
            .entry(instrument_id)
            .or_insert_with(|| Position::new(instrument_id));

        let prev_realized = position.realized_pnl;
        position.apply_fill(fill, tick_value);

        // Update totals
        self.total_realized += position.realized_pnl - prev_realized;
        self.total_fees += fill.fee;
    }

    /// Apply fill for a specific instrument
    pub fn apply_fill_for_instrument(&mut self, instrument_id: u32, fill: &Fill) {
        let tick_value = self
            .instruments
            .get(&instrument_id)
            .map_or(12.5, |i| i.tick_value as f64 / 100.0);

        let position = self
            .positions
            .entry(instrument_id)
            .or_insert_with(|| Position::new(instrument_id));

        let prev_realized = position.realized_pnl;
        let prev_fees = position.total_fees;
        position.apply_fill(fill, tick_value);

        // Update totals
        self.total_realized += position.realized_pnl - prev_realized;
        self.total_fees += position.total_fees - prev_fees;
    }

    /// Get position for an instrument
    #[must_use]
    pub fn get_position(&self, instrument_id: u32) -> Option<&Position> {
        self.positions.get(&instrument_id)
    }

    /// Get mutable position for an instrument
    pub fn get_position_mut(&mut self, instrument_id: u32) -> Option<&mut Position> {
        self.positions.get_mut(&instrument_id)
    }

    /// Get all positions
    pub fn positions(&self) -> impl Iterator<Item = &Position> {
        self.positions.values()
    }

    /// Calculate total unrealized P&L
    #[must_use]
    pub fn total_unrealized(&self, prices: &HashMap<u32, Price>) -> f64 {
        self.positions.iter().fold(0.0, |acc, (id, pos)| {
            if let (Some(price), Some(inst)) = (prices.get(id), self.instruments.get(id)) {
                acc + pos.unrealized_pnl(*price, inst.tick_value as f64 / 100.0)
            } else {
                acc
            }
        })
    }

    /// Calculate total P&L
    #[must_use]
    pub fn total_pnl(&self, prices: &HashMap<u32, Price>) -> f64 {
        self.total_realized + self.total_unrealized(prices) - self.total_fees
    }

    /// Update peak P&L and return current drawdown
    pub fn update_peak_and_drawdown(&mut self, current_pnl: f64) -> f64 {
        if current_pnl > self.peak_pnl {
            self.peak_pnl = current_pnl;
        }

        if self.peak_pnl > 0.0 {
            (self.peak_pnl - current_pnl) / self.peak_pnl
        } else {
            0.0
        }
    }

    /// Get peak P&L
    #[must_use]
    pub fn peak_pnl(&self) -> f64 {
        self.peak_pnl
    }

    /// Get total realized P&L
    #[must_use]
    pub fn realized_pnl(&self) -> f64 {
        self.total_realized
    }

    /// Get total fees
    #[must_use]
    pub fn fees(&self) -> f64 {
        self.total_fees
    }

    /// Check if all positions are flat
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.positions.values().all(Position::is_flat)
    }

    /// Get total absolute position across all instruments
    #[must_use]
    pub fn total_exposure(&self) -> u64 {
        self.positions.values().map(Position::size).sum()
    }

    /// Reset all positions
    pub fn reset(&mut self) {
        for pos in self.positions.values_mut() {
            pos.reset();
        }
        self.peak_pnl = 0.0;
        self.total_realized = 0.0;
        self.total_fees = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nano_core::types::{OrderId, Price, Quantity, Timestamp};

    fn create_fill(price: i64, quantity: u32, side: Side, fee: f64) -> Fill {
        Fill {
            order_id: OrderId::new(1),
            price: Price::from_raw(price),
            quantity: Quantity::new(quantity),
            side,
            is_maker: false,
            timestamp: Timestamp::now(),
            fee,
        }
    }

    #[test]
    fn test_position_buy() {
        let mut pos = Position::new(1);

        let fill = create_fill(50000, 10, Side::Buy, 5.0);
        pos.apply_fill(&fill, 12.5);

        assert_eq!(pos.quantity, 10);
        assert_eq!(pos.avg_price, 50000);
        assert_eq!(pos.total_traded, 10);
    }

    #[test]
    fn test_position_round_trip() {
        let mut pos = Position::new(1);

        // Buy 10 @ 50000
        let buy = create_fill(50000, 10, Side::Buy, 5.0);
        pos.apply_fill(&buy, 12.5);

        // Sell 10 @ 50010 (10 tick profit)
        let sell = create_fill(50010, 10, Side::Sell, 5.0);
        pos.apply_fill(&sell, 12.5);

        assert_eq!(pos.quantity, 0);
        assert_eq!(pos.round_trips, 1);
        // 10 ticks * 10 contracts * $12.50/tick = $1250
        // But our tick_value is in cents (1250), so: 10 * 10 * 12.5 / 100 = 12.5
        assert!((pos.realized_pnl - 12.5).abs() < 0.01);
    }

    #[test]
    fn test_position_partial_close() {
        let mut pos = Position::new(1);

        // Buy 10 @ 50000
        let buy = create_fill(50000, 10, Side::Buy, 5.0);
        pos.apply_fill(&buy, 12.5);

        // Sell 5 @ 50010
        let sell = create_fill(50010, 5, Side::Sell, 2.5);
        pos.apply_fill(&sell, 12.5);

        assert_eq!(pos.quantity, 5);
        assert_eq!(pos.avg_price, 50000); // Average price unchanged
        assert!(pos.realized_pnl > 0.0);
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut pos = Position::new(1);

        // Buy 10 @ 50000
        let buy = create_fill(50000, 10, Side::Buy, 5.0);
        pos.apply_fill(&buy, 12.5);

        // Market is now @ 50020
        let unrealized = pos.unrealized_pnl(Price::from_raw(50020), 12.5);

        // 20 ticks * 10 contracts * $12.50/tick / 100 = $25
        assert!((unrealized - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_short_position() {
        let mut pos = Position::new(1);

        // Sell 10 @ 50000
        let sell = create_fill(50000, 10, Side::Sell, 5.0);
        pos.apply_fill(&sell, 12.5);

        assert_eq!(pos.quantity, -10);
        assert!(pos.is_short());

        // Buy back @ 49990 (10 tick profit)
        let buy = create_fill(49990, 10, Side::Buy, 5.0);
        pos.apply_fill(&buy, 12.5);

        assert!(pos.is_flat());
        assert!(pos.realized_pnl > 0.0);
    }
}
