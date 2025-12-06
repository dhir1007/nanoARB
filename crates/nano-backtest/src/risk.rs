//! Risk management for backtesting and live trading.

use nano_core::error::{Error, Result};
use nano_core::traits::RiskManager as RiskManagerTrait;
use nano_core::types::{Order, OrderId, Price, Quantity, Side, Timestamp};
use serde::{Deserialize, Serialize};

use crate::config::RiskConfig;
use crate::position::PositionTracker;

/// Risk management state and checks
#[derive(Debug)]
pub struct RiskManager {
    /// Configuration
    config: RiskConfig,
    /// Peak P&L for drawdown calculation
    peak_pnl: f64,
    /// Daily starting P&L
    daily_start_pnl: f64,
    /// Current day (for daily reset)
    current_day: Option<u32>,
    /// Kill switch activated
    kill_switch_active: bool,
    /// Risk breach count
    breach_count: u32,
    /// Open order count
    open_order_count: usize,
}

impl RiskManager {
    /// Create a new risk manager from config
    #[must_use]
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            peak_pnl: 0.0,
            daily_start_pnl: 0.0,
            current_day: None,
            kill_switch_active: false,
            breach_count: 0,
            open_order_count: 0,
        }
    }

    /// Create with default config
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(RiskConfig::default())
    }

    /// Check if an order passes all risk checks
    pub fn check_order(&self, order: &Order, current_position: i64) -> Result<()> {
        // Check kill switch
        if self.kill_switch_active {
            return Err(Error::RiskLimitExceeded("Kill switch active".to_string()));
        }

        // Check order size
        if order.quantity.value() > self.config.max_order_size {
            return Err(Error::RiskLimitExceeded(format!(
                "Order size {} exceeds max {}",
                order.quantity.value(),
                self.config.max_order_size
            )));
        }

        // Check resulting position
        let signed_qty = if order.side == Side::Buy {
            order.quantity.value() as i64
        } else {
            -(order.quantity.value() as i64)
        };
        let resulting_position = current_position + signed_qty;

        if resulting_position.abs() > self.config.max_position {
            return Err(Error::RiskLimitExceeded(format!(
                "Resulting position {} exceeds max {}",
                resulting_position, self.config.max_position
            )));
        }

        // Check open order limit
        if self.open_order_count >= self.config.max_open_orders {
            return Err(Error::RiskLimitExceeded(format!(
                "Open order count {} at max {}",
                self.open_order_count, self.config.max_open_orders
            )));
        }

        Ok(())
    }

    /// Check position limits
    pub fn check_position(&self, position: i64) -> Result<()> {
        if position.abs() > self.config.max_position {
            return Err(Error::RiskLimitExceeded(format!(
                "Position {} exceeds max {}",
                position, self.config.max_position
            )));
        }
        Ok(())
    }

    /// Check drawdown
    pub fn check_drawdown(&self, current_pnl: f64) -> Result<()> {
        if self.peak_pnl > 0.0 {
            let drawdown = (self.peak_pnl - current_pnl) / self.peak_pnl;
            if drawdown > self.config.max_drawdown_pct {
                return Err(Error::RiskLimitExceeded(format!(
                    "Drawdown {:.2}% exceeds max {:.2}%",
                    drawdown * 100.0,
                    self.config.max_drawdown_pct * 100.0
                )));
            }
        }
        Ok(())
    }

    /// Check daily loss limit
    pub fn check_daily_loss(&self, current_pnl: f64) -> Result<()> {
        let daily_pnl = current_pnl - self.daily_start_pnl;
        if daily_pnl < -self.config.max_daily_loss {
            return Err(Error::RiskLimitExceeded(format!(
                "Daily loss ${:.2} exceeds max ${:.2}",
                -daily_pnl, self.config.max_daily_loss
            )));
        }
        Ok(())
    }

    /// Update P&L tracking and check for kill switch
    pub fn update_pnl(&mut self, current_pnl: f64, position: i64) -> bool {
        // Update peak
        if current_pnl > self.peak_pnl {
            self.peak_pnl = current_pnl;
        }

        // Check if kill switch should activate
        if self.config.enable_kill_switch && !self.kill_switch_active {
            // Drawdown check
            if self.peak_pnl > 0.0 {
                let drawdown = (self.peak_pnl - current_pnl) / self.peak_pnl;
                if drawdown > self.config.max_drawdown_pct {
                    self.kill_switch_active = true;
                    self.breach_count += 1;
                    return true;
                }
            }

            // Daily loss check
            let daily_pnl = current_pnl - self.daily_start_pnl;
            if daily_pnl < -self.config.max_daily_loss {
                self.kill_switch_active = true;
                self.breach_count += 1;
                return true;
            }
        }

        false
    }

    /// Update for new trading day
    pub fn new_day(&mut self, current_pnl: f64, day: u32) {
        if self.current_day != Some(day) {
            self.current_day = Some(day);
            self.daily_start_pnl = current_pnl;

            // Reset kill switch at start of new day (optional)
            // self.kill_switch_active = false;
        }
    }

    /// Record order submission
    pub fn on_order_submit(&mut self) {
        self.open_order_count += 1;
    }

    /// Record order completion (filled, cancelled, rejected)
    pub fn on_order_complete(&mut self) {
        self.open_order_count = self.open_order_count.saturating_sub(1);
    }

    /// Check if kill switch is active
    #[must_use]
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch_active
    }

    /// Manually activate kill switch
    pub fn activate_kill_switch(&mut self) {
        self.kill_switch_active = true;
    }

    /// Manually reset kill switch
    pub fn reset_kill_switch(&mut self) {
        self.kill_switch_active = false;
    }

    /// Get current drawdown percentage
    #[must_use]
    pub fn current_drawdown(&self, current_pnl: f64) -> f64 {
        if self.peak_pnl > 0.0 {
            (self.peak_pnl - current_pnl) / self.peak_pnl
        } else {
            0.0
        }
    }

    /// Get daily P&L
    #[must_use]
    pub fn daily_pnl(&self, current_pnl: f64) -> f64 {
        current_pnl - self.daily_start_pnl
    }

    /// Get breach count
    #[must_use]
    pub fn breach_count(&self) -> u32 {
        self.breach_count
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.peak_pnl = 0.0;
        self.daily_start_pnl = 0.0;
        self.current_day = None;
        self.kill_switch_active = false;
        self.breach_count = 0;
        self.open_order_count = 0;
    }

    /// Get config reference
    #[must_use]
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
}

impl RiskManagerTrait for RiskManager {
    fn check_order(&self, order: &Order, current_position: i64) -> Result<()> {
        RiskManager::check_order(self, order, current_position)
    }

    fn check_position(&self, position: i64) -> Result<()> {
        RiskManager::check_position(self, position)
    }

    fn check_drawdown(&self, pnl: f64, peak_pnl: f64) -> Result<()> {
        if peak_pnl > 0.0 {
            let drawdown = (peak_pnl - pnl) / peak_pnl;
            if drawdown > self.config.max_drawdown_pct {
                return Err(Error::RiskLimitExceeded(format!(
                    "Drawdown {:.2}% exceeds max {:.2}%",
                    drawdown * 100.0,
                    self.config.max_drawdown_pct * 100.0
                )));
            }
        }
        Ok(())
    }

    fn should_kill_switch(&self, pnl: f64, position: i64) -> bool {
        if !self.config.enable_kill_switch {
            return false;
        }

        // Check drawdown
        if self.peak_pnl > 0.0 {
            let drawdown = (self.peak_pnl - pnl) / self.peak_pnl;
            if drawdown > self.config.max_drawdown_pct {
                return true;
            }
        }

        // Check position limit
        if position.abs() > self.config.max_position {
            return true;
        }

        // Check daily loss
        let daily_pnl = pnl - self.daily_start_pnl;
        if daily_pnl < -self.config.max_daily_loss {
            return true;
        }

        false
    }

    fn max_position(&self) -> i64 {
        self.config.max_position
    }

    fn max_order_size(&self) -> u32 {
        self.config.max_order_size
    }
}

/// Value at Risk (VaR) calculator
#[derive(Debug)]
pub struct VaRCalculator {
    /// Historical returns
    returns: Vec<f64>,
    /// Confidence level (e.g., 0.95 for 95% VaR)
    confidence: f64,
    /// Window size for rolling calculation
    window_size: usize,
}

impl VaRCalculator {
    /// Create a new VaR calculator
    #[must_use]
    pub fn new(confidence: f64, window_size: usize) -> Self {
        Self {
            returns: Vec::with_capacity(window_size),
            confidence,
            window_size,
        }
    }

    /// Add a return observation
    pub fn add_return(&mut self, ret: f64) {
        self.returns.push(ret);
        if self.returns.len() > self.window_size {
            self.returns.remove(0);
        }
    }

    /// Calculate historical VaR
    #[must_use]
    pub fn calculate(&self) -> Option<f64> {
        if self.returns.len() < 30 {
            return None;
        }

        let mut sorted = self.returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((1.0 - self.confidence) * sorted.len() as f64).floor() as usize;
        Some(-sorted[index]) // VaR is typically reported as positive number
    }

    /// Calculate Expected Shortfall (CVaR)
    #[must_use]
    pub fn expected_shortfall(&self) -> Option<f64> {
        if self.returns.len() < 30 {
            return None;
        }

        let mut sorted = self.returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let cutoff_index = ((1.0 - self.confidence) * sorted.len() as f64).floor() as usize;
        let tail: Vec<f64> = sorted[..=cutoff_index].to_vec();

        if tail.is_empty() {
            return None;
        }

        let avg_tail_loss = tail.iter().sum::<f64>() / tail.len() as f64;
        Some(-avg_tail_loss)
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.returns.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nano_core::types::{OrderId, OrderStatus, OrderType, TimeInForce};

    fn create_test_order(side: Side, quantity: u32) -> Order {
        Order {
            id: OrderId::new(1),
            instrument_id: 1,
            side,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::Pending,
            price: Price::from_raw(50000),
            stop_price: Price::ZERO,
            quantity: Quantity::new(quantity),
            filled_quantity: Quantity::ZERO,
            avg_fill_price: 0,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        }
    }

    #[test]
    fn test_order_size_limit() {
        let config = RiskConfig {
            max_order_size: 10,
            ..Default::default()
        };
        let risk = RiskManager::new(config);

        let small_order = create_test_order(Side::Buy, 5);
        assert!(risk.check_order(&small_order, 0).is_ok());

        let large_order = create_test_order(Side::Buy, 15);
        assert!(risk.check_order(&large_order, 0).is_err());
    }

    #[test]
    fn test_position_limit() {
        let config = RiskConfig {
            max_position: 50,
            max_order_size: 100,
            ..Default::default()
        };
        let risk = RiskManager::new(config);

        // Current position 40, buy 5 -> OK
        let order = create_test_order(Side::Buy, 5);
        assert!(risk.check_order(&order, 40).is_ok());

        // Current position 40, buy 20 -> Exceeds limit
        let order = create_test_order(Side::Buy, 20);
        assert!(risk.check_order(&order, 40).is_err());
    }

    #[test]
    fn test_drawdown_check() {
        let mut risk = RiskManager::new(RiskConfig {
            max_drawdown_pct: 0.05,
            ..Default::default()
        });

        risk.peak_pnl = 100_000.0;

        // 3% drawdown - OK
        assert!(risk.check_drawdown(97_000.0).is_ok());

        // 6% drawdown - Exceeds
        assert!(risk.check_drawdown(94_000.0).is_err());
    }

    #[test]
    fn test_kill_switch() {
        let mut risk = RiskManager::new(RiskConfig {
            max_drawdown_pct: 0.05,
            enable_kill_switch: true,
            ..Default::default()
        });

        risk.peak_pnl = 100_000.0;

        // No kill switch at small drawdown
        assert!(!risk.update_pnl(97_000.0, 0));
        assert!(!risk.is_kill_switch_active());

        // Kill switch at large drawdown
        assert!(risk.update_pnl(90_000.0, 0));
        assert!(risk.is_kill_switch_active());
    }

    #[test]
    fn test_var_calculator() {
        let mut var = VaRCalculator::new(0.95, 100);

        // Add some returns
        for i in 0..100 {
            let ret = (i as f64 - 50.0) / 100.0; // -0.5 to 0.49
            var.add_return(ret);
        }

        let var_value = var.calculate().unwrap();
        assert!(var_value > 0.0);

        let es = var.expected_shortfall().unwrap();
        assert!(es > var_value); // ES should be more conservative than VaR
    }
}

