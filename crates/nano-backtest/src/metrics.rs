//! Performance metrics and statistics for backtesting.

use std::collections::VecDeque;

use nano_core::types::{Fill, Side, Timestamp};
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;

/// Backtest performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BacktestMetrics {
    /// Total P&L
    pub total_pnl: f64,
    /// Realized P&L
    pub realized_pnl: f64,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
    /// Total fees paid
    pub total_fees: f64,
    /// Number of trades (round trips)
    pub num_trades: u32,
    /// Number of winning trades
    pub winning_trades: u32,
    /// Number of losing trades
    pub losing_trades: u32,
    /// Gross profit (sum of winning trades)
    pub gross_profit: f64,
    /// Gross loss (sum of losing trades)
    pub gross_loss: f64,
    /// Maximum drawdown (percentage)
    pub max_drawdown_pct: f64,
    /// Maximum drawdown (absolute)
    pub max_drawdown_abs: f64,
    /// Peak P&L
    pub peak_pnl: f64,
    /// Total volume traded
    pub total_volume: u64,
    /// Number of buy fills
    pub buy_fills: u32,
    /// Number of sell fills
    pub sell_fills: u32,
    /// Maker fills
    pub maker_fills: u32,
    /// Taker fills
    pub taker_fills: u32,
    /// Average fill latency (nanoseconds)
    pub avg_fill_latency_ns: f64,
    /// Start timestamp
    pub start_time: Option<Timestamp>,
    /// End timestamp
    pub end_time: Option<Timestamp>,
}

impl BacktestMetrics {
    /// Create new empty metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate win rate
    #[must_use]
    pub fn win_rate(&self) -> f64 {
        if self.num_trades == 0 {
            return 0.0;
        }
        self.winning_trades as f64 / self.num_trades as f64
    }

    /// Calculate profit factor
    #[must_use]
    pub fn profit_factor(&self) -> f64 {
        if self.gross_loss.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        self.gross_profit / self.gross_loss.abs()
    }

    /// Calculate average trade P&L
    #[must_use]
    pub fn avg_trade_pnl(&self) -> f64 {
        if self.num_trades == 0 {
            return 0.0;
        }
        self.realized_pnl / self.num_trades as f64
    }

    /// Calculate average winning trade
    #[must_use]
    pub fn avg_winning_trade(&self) -> f64 {
        if self.winning_trades == 0 {
            return 0.0;
        }
        self.gross_profit / self.winning_trades as f64
    }

    /// Calculate average losing trade
    #[must_use]
    pub fn avg_losing_trade(&self) -> f64 {
        if self.losing_trades == 0 {
            return 0.0;
        }
        self.gross_loss / self.losing_trades as f64
    }

    /// Calculate maker ratio
    #[must_use]
    pub fn maker_ratio(&self) -> f64 {
        let total = self.maker_fills + self.taker_fills;
        if total == 0 {
            return 0.0;
        }
        self.maker_fills as f64 / total as f64
    }

    /// Record a fill
    pub fn record_fill(&mut self, fill: &Fill) {
        self.total_volume += fill.quantity.value() as u64;
        self.total_fees += fill.fee;

        match fill.side {
            Side::Buy => self.buy_fills += 1,
            Side::Sell => self.sell_fills += 1,
        }

        if fill.is_maker {
            self.maker_fills += 1;
        } else {
            self.taker_fills += 1;
        }
    }

    /// Record a completed trade (round trip)
    pub fn record_trade(&mut self, pnl: f64) {
        self.num_trades += 1;

        if pnl > 0.0 {
            self.winning_trades += 1;
            self.gross_profit += pnl;
        } else if pnl < 0.0 {
            self.losing_trades += 1;
            self.gross_loss += pnl.abs();
        }
    }

    /// Update P&L and drawdown tracking
    pub fn update_pnl(&mut self, total_pnl: f64, realized: f64, unrealized: f64) {
        self.total_pnl = total_pnl;
        self.realized_pnl = realized;
        self.unrealized_pnl = unrealized;

        if total_pnl > self.peak_pnl {
            self.peak_pnl = total_pnl;
        }

        if self.peak_pnl > 0.0 {
            let drawdown = self.peak_pnl - total_pnl;
            let drawdown_pct = drawdown / self.peak_pnl;

            if drawdown > self.max_drawdown_abs {
                self.max_drawdown_abs = drawdown;
            }
            if drawdown_pct > self.max_drawdown_pct {
                self.max_drawdown_pct = drawdown_pct;
            }
        }
    }

    /// Set time range
    pub fn set_time_range(&mut self, start: Timestamp, end: Timestamp) {
        self.start_time = Some(start);
        self.end_time = Some(end);
    }

    /// Get duration in seconds
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => (end.as_nanos() - start.as_nanos()) as f64 / 1e9,
            _ => 0.0,
        }
    }
}

/// Detailed performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Daily returns
    pub daily_returns: Vec<f64>,
    /// Equity curve (cumulative P&L)
    pub equity_curve: Vec<f64>,
    /// Timestamps for equity curve
    pub equity_timestamps: Vec<i64>,
    /// Trade P&Ls
    pub trade_pnls: Vec<f64>,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Maximum consecutive wins
    pub max_consecutive_wins: u32,
    /// Maximum consecutive losses
    pub max_consecutive_losses: u32,
    /// Recovery factor
    pub recovery_factor: f64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceStats {
    /// Create new empty stats
    #[must_use]
    pub fn new() -> Self {
        Self {
            daily_returns: Vec::new(),
            equity_curve: Vec::new(),
            equity_timestamps: Vec::new(),
            trade_pnls: Vec::new(),
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            recovery_factor: 0.0,
        }
    }

    /// Add a data point to equity curve
    pub fn add_equity_point(&mut self, timestamp: i64, pnl: f64) {
        self.equity_timestamps.push(timestamp);
        self.equity_curve.push(pnl);
    }

    /// Add a daily return
    pub fn add_daily_return(&mut self, ret: f64) {
        self.daily_returns.push(ret);
    }

    /// Add a trade P&L
    pub fn add_trade_pnl(&mut self, pnl: f64) {
        self.trade_pnls.push(pnl);
    }

    /// Calculate all statistics
    pub fn calculate(&mut self, initial_capital: f64, max_drawdown: f64) {
        self.sharpe_ratio = self.calculate_sharpe();
        self.sortino_ratio = self.calculate_sortino();
        self.calmar_ratio = self.calculate_calmar(max_drawdown);
        self.calculate_consecutive_trades();
        self.recovery_factor = self.calculate_recovery_factor(initial_capital, max_drawdown);
    }

    /// Calculate annualized Sharpe ratio
    fn calculate_sharpe(&self) -> f64 {
        if self.daily_returns.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.daily_returns.clone();
        let mean = returns.clone().mean();
        let std = returns.std_dev();

        if std < f64::EPSILON {
            return 0.0;
        }

        // Annualize: assuming 252 trading days
        (mean / std) * (252.0_f64).sqrt()
    }

    /// Calculate annualized Sortino ratio
    fn calculate_sortino(&self) -> f64 {
        if self.daily_returns.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.daily_returns.clone();
        let mean = returns.clone().mean();

        // Downside deviation: std dev of negative returns only
        let negative_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();

        if negative_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_std = negative_returns.std_dev();
        if downside_std < f64::EPSILON {
            return f64::INFINITY;
        }

        (mean / downside_std) * (252.0_f64).sqrt()
    }

    /// Calculate Calmar ratio
    fn calculate_calmar(&self, max_drawdown: f64) -> f64 {
        if max_drawdown < f64::EPSILON || self.daily_returns.is_empty() {
            return 0.0;
        }

        let annual_return = self.daily_returns.clone().mean() * 252.0;
        annual_return / max_drawdown
    }

    /// Calculate consecutive wins/losses
    fn calculate_consecutive_trades(&mut self) {
        let mut consecutive_wins = 0u32;
        let mut consecutive_losses = 0u32;
        let mut max_wins = 0u32;
        let mut max_losses = 0u32;

        for &pnl in &self.trade_pnls {
            if pnl > 0.0 {
                consecutive_wins += 1;
                consecutive_losses = 0;
                max_wins = max_wins.max(consecutive_wins);
            } else if pnl < 0.0 {
                consecutive_losses += 1;
                consecutive_wins = 0;
                max_losses = max_losses.max(consecutive_losses);
            }
        }

        self.max_consecutive_wins = max_wins;
        self.max_consecutive_losses = max_losses;
    }

    /// Calculate recovery factor
    fn calculate_recovery_factor(&self, initial_capital: f64, max_drawdown: f64) -> f64 {
        if max_drawdown < f64::EPSILON {
            return f64::INFINITY;
        }

        let total_return = self.equity_curve.last().copied().unwrap_or(0.0);
        (total_return / initial_capital) / max_drawdown
    }
}

/// Rolling statistics calculator
#[derive(Debug)]
pub struct RollingStats {
    /// Window size
    window_size: usize,
    /// Values in window
    values: VecDeque<f64>,
    /// Running sum
    sum: f64,
    /// Running sum of squares
    sum_sq: f64,
}

impl RollingStats {
    /// Create a new rolling statistics calculator
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Add a value
    pub fn add(&mut self, value: f64) {
        if self.values.len() == self.window_size {
            let old = self.values.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }

        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    /// Get the mean
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.sum / self.values.len() as f64
    }

    /// Get the variance
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        (self.sum_sq - self.sum * self.sum / n) / (n - 1.0)
    }

    /// Get the standard deviation
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get rolling Sharpe (not annualized)
    #[must_use]
    pub fn sharpe(&self) -> f64 {
        let std = self.std_dev();
        if std < f64::EPSILON {
            return 0.0;
        }
        self.mean() / std
    }

    /// Check if window is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.values.len() == self.window_size
    }

    /// Get number of values
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_win_rate() {
        let mut metrics = BacktestMetrics::new();

        metrics.record_trade(100.0); // Win
        metrics.record_trade(50.0);  // Win
        metrics.record_trade(-30.0); // Loss
        metrics.record_trade(-20.0); // Loss
        metrics.record_trade(80.0);  // Win

        assert_eq!(metrics.num_trades, 5);
        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.losing_trades, 2);
        assert!((metrics.win_rate() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_metrics_profit_factor() {
        let mut metrics = BacktestMetrics::new();

        metrics.record_trade(100.0);
        metrics.record_trade(100.0);
        metrics.record_trade(-50.0);

        // Profit factor = 200 / 50 = 4.0
        assert!((metrics.profit_factor() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(5);

        for i in 1..=5 {
            stats.add(i as f64);
        }

        // Mean of 1,2,3,4,5 = 3.0
        assert!((stats.mean() - 3.0).abs() < 0.01);
        assert!(stats.is_full());

        // Add one more - should drop 1, keep 2,3,4,5,6
        stats.add(6.0);
        // Mean of 2,3,4,5,6 = 4.0
        assert!((stats.mean() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_performance_stats_sharpe() {
        let mut stats = PerformanceStats::new();

        // Add some daily returns with small variance
        for i in 0..100 {
            // Slightly varying positive returns
            let ret = 0.001 + (i as f64 % 5.0) * 0.0001;
            stats.add_daily_return(ret);
        }

        stats.calculate(1_000_000.0, 0.05);

        // With mostly positive returns and low variance, Sharpe should be positive
        assert!(stats.sharpe_ratio > 0.0);
    }
}

