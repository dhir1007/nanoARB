//! Reinforcement Learning environment for market-making.

use nano_core::traits::OrderBook as OrderBookTrait;
use nano_core::types::{Side, Timestamp};
use nano_lob::{OrderBook, SnapshotRingBuffer};
use serde::{Deserialize, Serialize};

/// Market-making action space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingAction {
    /// Bid price skew from mid (-1 to +1, in units of base spread)
    pub bid_skew: f64,
    /// Ask price skew from mid (-1 to +1, in units of base spread)
    pub ask_skew: f64,
    /// Spread width (in ticks)
    pub spread: f64,
    /// Bid size (as fraction of max)
    pub bid_size: f64,
    /// Ask size (as fraction of max)
    pub ask_size: f64,
}

impl Default for MarketMakingAction {
    fn default() -> Self {
        Self {
            bid_skew: 0.0,
            ask_skew: 0.0,
            spread: 1.0,
            bid_size: 1.0,
            ask_size: 1.0,
        }
    }
}

impl MarketMakingAction {
    /// Create action from array (for neural network output)
    #[must_use]
    pub fn from_array(arr: &[f64]) -> Self {
        Self {
            bid_skew: arr.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0),
            ask_skew: arr.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0),
            spread: arr.get(2).copied().unwrap_or(1.0).clamp(0.5, 5.0),
            bid_size: arr.get(3).copied().unwrap_or(1.0).clamp(0.0, 1.0),
            ask_size: arr.get(4).copied().unwrap_or(1.0).clamp(0.0, 1.0),
        }
    }

    /// Convert to array for neural network input
    #[must_use]
    pub fn to_array(&self) -> [f64; 5] {
        [self.bid_skew, self.ask_skew, self.spread, self.bid_size, self.ask_size]
    }

    /// Check if action is valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.bid_skew >= -1.0 && self.bid_skew <= 1.0
            && self.ask_skew >= -1.0 && self.ask_skew <= 1.0
            && self.spread > 0.0
            && self.bid_size >= 0.0 && self.bid_size <= 1.0
            && self.ask_size >= 0.0 && self.ask_size <= 1.0
    }
}

/// State representation for RL agent
#[derive(Debug, Clone)]
pub struct MarketMakingState {
    /// LOB features (flattened)
    pub lob_features: Vec<f32>,
    /// Current inventory (normalized)
    pub inventory: f32,
    /// Unrealized P&L (normalized)
    pub unrealized_pnl: f32,
    /// Time since last trade (normalized)
    pub time_since_trade: f32,
    /// Spread (in ticks)
    pub spread: f32,
    /// Book imbalance
    pub imbalance: f32,
    /// Recent returns
    pub recent_returns: Vec<f32>,
}

impl MarketMakingState {
    /// Convert state to flat array for neural network
    #[must_use]
    pub fn to_array(&self) -> Vec<f32> {
        let mut arr = self.lob_features.clone();
        arr.push(self.inventory);
        arr.push(self.unrealized_pnl);
        arr.push(self.time_since_trade);
        arr.push(self.spread);
        arr.push(self.imbalance);
        arr.extend(&self.recent_returns);
        arr
    }

    /// Get the feature dimension
    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.lob_features.len() + 5 + self.recent_returns.len()
    }
}

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    /// Maximum inventory
    pub max_inventory: i64,
    /// Maximum order size
    pub max_order_size: u32,
    /// Tick size
    pub tick_size: f64,
    /// Tick value (P&L per tick per contract)
    pub tick_value: f64,
    /// Maker fee
    pub maker_fee: f64,
    /// Taker fee
    pub taker_fee: f64,
    /// Inventory penalty coefficient
    pub lambda_inventory: f64,
    /// Adverse selection penalty coefficient
    pub lambda_adverse: f64,
    /// Spread penalty coefficient
    pub lambda_spread: f64,
    /// Episode length (number of steps)
    pub episode_length: usize,
    /// Observation window (number of snapshots)
    pub observation_window: usize,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            max_inventory: 50,
            max_order_size: 10,
            tick_size: 0.25,
            tick_value: 12.5,
            maker_fee: 0.25,
            taker_fee: 0.85,
            lambda_inventory: 0.001,
            lambda_adverse: 0.0005,
            lambda_spread: 0.0001,
            episode_length: 10000,
            observation_window: 100,
        }
    }
}

/// Market-making RL environment
pub struct MarketMakingEnv {
    /// Configuration
    config: EnvConfig,
    /// Current inventory
    inventory: i64,
    /// Current P&L
    pnl: f64,
    /// Unrealized P&L
    unrealized_pnl: f64,
    /// Total fees
    total_fees: f64,
    /// Average entry price
    avg_entry_price: f64,
    /// Step count
    step_count: usize,
    /// Last action
    last_action: Option<MarketMakingAction>,
    /// Snapshot history
    snapshot_buffer: SnapshotRingBuffer,
    /// Recent mid prices for return calculation
    recent_mids: Vec<f64>,
    /// Last trade timestamp
    last_trade_time: Timestamp,
    /// Episode done flag
    done: bool,
}

impl MarketMakingEnv {
    /// Create a new environment
    #[must_use]
    pub fn new(config: EnvConfig) -> Self {
        let observation_window = config.observation_window;
        Self {
            config,
            inventory: 0,
            pnl: 0.0,
            unrealized_pnl: 0.0,
            total_fees: 0.0,
            avg_entry_price: 0.0,
            step_count: 0,
            last_action: None,
            snapshot_buffer: SnapshotRingBuffer::new(observation_window),
            recent_mids: Vec::with_capacity(100),
            last_trade_time: Timestamp::EPOCH,
            done: false,
        }
    }

    /// Reset the environment
    pub fn reset(&mut self) -> MarketMakingState {
        self.inventory = 0;
        self.pnl = 0.0;
        self.unrealized_pnl = 0.0;
        self.total_fees = 0.0;
        self.avg_entry_price = 0.0;
        self.step_count = 0;
        self.last_action = None;
        self.snapshot_buffer.clear();
        self.recent_mids.clear();
        self.done = false;

        self.get_state()
    }

    /// Take a step in the environment
    pub fn step(&mut self, action: MarketMakingAction, book: &OrderBook) -> (MarketMakingState, f64, bool) {
        self.step_count += 1;

        // Update snapshot buffer
        self.snapshot_buffer.push_book(book);

        // Get current mid price
        let mid = book.mid_price().map(|p| p.as_f64()).unwrap_or(0.0);
        self.recent_mids.push(mid);
        if self.recent_mids.len() > 100 {
            self.recent_mids.remove(0);
        }

        // Simulate trading based on action
        let (fills, adverse_selection_cost) = self.simulate_fills(&action, book);

        // Calculate reward
        let reward = self.calculate_reward(fills, adverse_selection_cost, &action);

        // Update state
        self.last_action = Some(action);

        // Check termination
        if self.step_count >= self.config.episode_length {
            self.done = true;
        }

        // Check inventory breach
        if self.inventory.abs() > self.config.max_inventory {
            self.done = true;
        }

        let state = self.get_state();
        (state, reward, self.done)
    }

    /// Simulate fills based on action and market state
    fn simulate_fills(&mut self, action: &MarketMakingAction, book: &OrderBook) -> (Vec<(Side, f64, u32)>, f64) {
        let mut fills = Vec::new();
        let mut adverse_cost = 0.0;

        let mid = match book.mid_price() {
            Some(m) => m.as_f64(),
            None => return (fills, adverse_cost),
        };

        // Simple fill simulation based on spread and book state
        // In practice, this would be much more sophisticated
        let spread = book.spread().map(|s| s.as_f64()).unwrap_or(0.0);

        // Check if our quotes would be hit
        let bid_price = mid - action.spread * self.config.tick_size / 2.0
            + action.bid_skew * self.config.tick_size;
        let ask_price = mid + action.spread * self.config.tick_size / 2.0
            + action.ask_skew * self.config.tick_size;

        // Probability of fill based on quote aggressiveness
        let bid_fill_prob = 0.1 * (1.0 - action.bid_skew.abs());
        let ask_fill_prob = 0.1 * (1.0 - action.ask_skew.abs());

        // Random fill simulation
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < bid_fill_prob && self.inventory < self.config.max_inventory {
            let qty = (action.bid_size * self.config.max_order_size as f64) as u32;
            if qty > 0 {
                fills.push((Side::Buy, bid_price, qty));
                self.apply_fill(Side::Buy, bid_price, qty, true);
            }
        }

        if rng.gen::<f64>() < ask_fill_prob && self.inventory > -self.config.max_inventory {
            let qty = (action.ask_size * self.config.max_order_size as f64) as u32;
            if qty > 0 {
                fills.push((Side::Sell, ask_price, qty));
                self.apply_fill(Side::Sell, ask_price, qty, true);
            }
        }

        // Adverse selection: market moves against us after fill
        if !fills.is_empty() {
            let move_against = rng.gen::<f64>() * 0.5; // 0-0.5 tick adverse move
            adverse_cost = move_against * self.config.tick_value * self.inventory.abs() as f64;
        }

        // Update unrealized P&L
        self.update_unrealized(mid);

        (fills, adverse_cost)
    }

    /// Apply a fill to position
    fn apply_fill(&mut self, side: Side, price: f64, quantity: u32, is_maker: bool) {
        let signed_qty = if side == Side::Buy { quantity as i64 } else { -(quantity as i64) };

        let fee = if is_maker {
            self.config.maker_fee * quantity as f64
        } else {
            self.config.taker_fee * quantity as f64
        };

        self.total_fees += fee;

        // Update position
        let old_inventory = self.inventory;
        self.inventory += signed_qty;

        // Calculate realized P&L if reducing position
        if old_inventory != 0 && self.inventory.signum() != old_inventory.signum() {
            // Closed or reversed position
            let closed_qty = old_inventory.abs().min(quantity as i64);
            let pnl_per_contract = if old_inventory > 0 {
                price - self.avg_entry_price
            } else {
                self.avg_entry_price - price
            };
            self.pnl += pnl_per_contract * closed_qty as f64 / self.config.tick_size * self.config.tick_value;
        }

        // Update average entry price
        if self.inventory != 0 {
            if old_inventory.signum() == signed_qty.signum() || old_inventory == 0 {
                // Adding to position
                let old_value = old_inventory.abs() as f64 * self.avg_entry_price;
                let new_value = quantity as f64 * price;
                self.avg_entry_price = (old_value + new_value) / self.inventory.abs() as f64;
            } else if self.inventory.signum() == signed_qty.signum() {
                // Reversed position
                self.avg_entry_price = price;
            }
        }

        self.last_trade_time = Timestamp::now();
    }

    /// Update unrealized P&L
    fn update_unrealized(&mut self, current_mid: f64) {
        if self.inventory == 0 {
            self.unrealized_pnl = 0.0;
            return;
        }

        let price_diff = current_mid - self.avg_entry_price;
        self.unrealized_pnl = price_diff * self.inventory as f64 / self.config.tick_size
            * self.config.tick_value;
    }

    /// Calculate reward
    fn calculate_reward(
        &self,
        fills: Vec<(Side, f64, u32)>,
        adverse_selection_cost: f64,
        action: &MarketMakingAction,
    ) -> f64 {
        let mut reward = 0.0;

        // P&L from fills (spread capture)
        for (side, price, qty) in &fills {
            let half_spread = action.spread * self.config.tick_size / 2.0;
            let edge = half_spread / self.config.tick_size * self.config.tick_value;
            reward += edge * *qty as f64;
        }

        // Inventory penalty (quadratic)
        let inv_penalty = self.config.lambda_inventory
            * (self.inventory as f64 / self.config.max_inventory as f64).powi(2);
        reward -= inv_penalty;

        // Adverse selection penalty
        reward -= self.config.lambda_adverse * adverse_selection_cost;

        // Fee cost
        if !fills.is_empty() {
            let fee_cost: f64 = fills.iter()
                .map(|(_, _, q)| self.config.maker_fee * *q as f64)
                .sum();
            reward -= fee_cost;
        }

        reward
    }

    /// Get current state
    fn get_state(&self) -> MarketMakingState {
        // Get LOB features from snapshot buffer
        let lob_features = if let Some(tensor) = self.snapshot_buffer.to_tensor(10) {
            tensor.iter().map(|&x| x).collect()
        } else {
            vec![0.0; 400] // Default if no data
        };

        // Calculate recent returns
        let recent_returns = if self.recent_mids.len() >= 2 {
            self.recent_mids.windows(2)
                .take(10)
                .map(|w| ((w[1] - w[0]) / w[0] * 10000.0) as f32) // Returns in bps
                .collect()
        } else {
            vec![0.0; 10]
        };

        MarketMakingState {
            lob_features,
            inventory: self.inventory as f32 / self.config.max_inventory as f32,
            unrealized_pnl: (self.unrealized_pnl / 1000.0) as f32, // Normalize
            time_since_trade: 0.0, // Would need timestamp tracking
            spread: 1.0, // Default
            imbalance: 0.0, // Would need book data
            recent_returns,
        }
    }

    /// Check if episode is done
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Get current inventory
    #[must_use]
    pub fn inventory(&self) -> i64 {
        self.inventory
    }

    /// Get total P&L
    #[must_use]
    pub fn total_pnl(&self) -> f64 {
        self.pnl + self.unrealized_pnl - self.total_fees
    }

    /// Get step count
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_validation() {
        let valid = MarketMakingAction {
            bid_skew: 0.5,
            ask_skew: -0.3,
            spread: 2.0,
            bid_size: 0.8,
            ask_size: 0.6,
        };
        assert!(valid.is_valid());

        let invalid = MarketMakingAction {
            bid_skew: 1.5, // Out of range
            ..Default::default()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_action_from_array() {
        let arr = [0.5, -0.3, 2.0, 0.8, 0.6];
        let action = MarketMakingAction::from_array(&arr);

        assert!((action.bid_skew - 0.5).abs() < 0.01);
        assert!((action.spread - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_env_reset() {
        let config = EnvConfig::default();
        let mut env = MarketMakingEnv::new(config);

        let state = env.reset();

        assert_eq!(env.inventory(), 0);
        assert!((env.total_pnl()).abs() < 0.01);
        assert!(!env.is_done());
    }
}

