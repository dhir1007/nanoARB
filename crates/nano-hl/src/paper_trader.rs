//! Paper trading state tracker.
//!
//! In paper mode, we never send orders to Hyperliquid. Instead, we:
//!   1. Log every quote the strategy generates
//!   2. Simulate fills when the market crosses our quote price
//!   3. Track position, realised PnL, and running Sharpe

use std::collections::VecDeque;

use nano_core::types::{Order, OrderId, Price, Side};

use crate::converter::LOT_SIZE;

/// A single paper fill record
#[derive(Debug, Clone)]
pub struct PaperFill {
    pub order_id: OrderId,
    pub side:     Side,
    pub price:    f64,   // USD
    pub size:     f64,   // base asset (BTC, ETH, …)
    pub notional: f64,   // price × size in USD
}

/// Running paper-trading state
pub struct PaperTrader {
    /// Coin being traded (for display)
    pub coin: String,
    /// Current position in lots (positive = long, negative = short)
    pub position_lots: i64,
    /// USD value of open position's cost basis
    pub cost_basis_usd: f64,
    /// Cumulative realised PnL in USD
    pub realised_pnl: f64,
    /// Peak PnL seen so far (for drawdown calculation)
    pub peak_pnl: f64,
    /// Total orders submitted (paper)
    pub orders_sent: u64,
    /// Total fills received
    pub fills_received: u64,
    /// Recent fills for display
    pub recent_fills: VecDeque<PaperFill>,
    /// PnL history for Sharpe calculation (one entry per fill)
    pnl_history: VecDeque<f64>,
}

impl PaperTrader {
    pub fn new(coin: impl Into<String>) -> Self {
        Self {
            coin:            coin.into(),
            position_lots:   0,
            cost_basis_usd:  0.0,
            realised_pnl:    0.0,
            peak_pnl:        0.0,
            orders_sent:     0,
            fills_received:  0,
            recent_fills:    VecDeque::with_capacity(50),
            pnl_history:     VecDeque::with_capacity(500),
        }
    }

    /// Log a paper order (no actual submission)
    pub fn log_order(&mut self, order: &Order, mid_price_f64: f64) {
        self.orders_sent += 1;
        let price_f64 = order.price.as_f64();
        let size_lots = order.quantity.value() as i64;
        let size_btc  = size_lots as f64 * LOT_SIZE;
        let edge_ticks = match order.side {
            Side::Buy  => mid_price_f64 - price_f64,
            Side::Sell => price_f64 - mid_price_f64,
        };

        tracing::info!(
            "[PAPER ORDER] #{} {:?} @ ${:.2}  size={:.4} {} | mid=${:.2} | edge={:.2}",
            order.id,
            order.side,
            price_f64,
            size_btc,
            self.coin,
            mid_price_f64,
            edge_ticks,
        );
    }

    /// Simulate a fill when the market crosses our quote.
    ///
    /// Call this when `best_ask <= our_bid` (we got lifted) or
    /// `best_bid >= our_ask` (we got hit).
    pub fn simulate_fill(&mut self, order: &Order, fill_price: Price) {
        let price_f64  = fill_price.as_f64();
        let size_lots  = order.quantity.value() as i64;
        let size_btc   = size_lots as f64 * LOT_SIZE;
        let notional   = price_f64 * size_btc;

        let fill = PaperFill {
            order_id: order.id,
            side:     order.side,
            price:    price_f64,
            size:     size_btc,
            notional,
        };

        match order.side {
            Side::Buy => {
                self.position_lots  += size_lots;
                self.cost_basis_usd += notional;
            }
            Side::Sell => {
                // Realise PnL on the portion that closes a long position
                let closing_lots = size_lots.min(self.position_lots.max(0));
                if closing_lots > 0 {
                    let avg_cost = if self.position_lots > 0 {
                        self.cost_basis_usd / (self.position_lots as f64 * LOT_SIZE)
                    } else {
                        price_f64
                    };
                    let closing_size = closing_lots as f64 * LOT_SIZE;
                    let realised = (price_f64 - avg_cost) * closing_size;
                    self.realised_pnl    += realised;
                    self.cost_basis_usd  -= avg_cost * closing_size;
                    self.pnl_history.push_back(realised);
                    if self.pnl_history.len() > 500 {
                        self.pnl_history.pop_front();
                    }
                }
                self.position_lots  -= size_lots;
                // Opening short: track cost basis as negative
                let opening_lots = size_lots - closing_lots;
                if opening_lots > 0 {
                    self.cost_basis_usd -= price_f64 * (opening_lots as f64 * LOT_SIZE);
                }
            }
        }

        self.peak_pnl = self.peak_pnl.max(self.realised_pnl);
        self.fills_received += 1;

        if self.recent_fills.len() >= 50 {
            self.recent_fills.pop_front();
        }
        self.recent_fills.push_back(fill.clone());

        tracing::info!(
            "[PAPER FILL ] #{} {:?} @ ${:.2}  size={:.4} {} | pnl=${:.4} | pos={}",
            fill.order_id,
            fill.side,
            fill.price,
            fill.size,
            self.coin,
            self.realised_pnl,
            self.position_lots,
        );
    }

    /// Mark-to-market unrealised PnL given the current mid price
    pub fn unrealised_pnl(&self, mid_f64: f64) -> f64 {
        let pos_btc = self.position_lots as f64 * LOT_SIZE;
        let market_value = pos_btc * mid_f64;
        market_value - self.cost_basis_usd
    }

    /// Total PnL (realised + unrealised)
    pub fn total_pnl(&self, mid_f64: f64) -> f64 {
        self.realised_pnl + self.unrealised_pnl(mid_f64)
    }

    /// Max drawdown from peak (negative number, in USD)
    pub fn max_drawdown(&self) -> f64 {
        self.peak_pnl - self.realised_pnl
    }

    /// Approximate Sharpe ratio from PnL history (annualised, assuming ~2 fills/min)
    pub fn sharpe(&self) -> f64 {
        let n = self.pnl_history.len();
        if n < 10 {
            return 0.0;
        }
        let mean = self.pnl_history.iter().sum::<f64>() / n as f64;
        let variance = self.pnl_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        if std_dev < 1e-9 { return 0.0; }
        // Annualise: ~2 fills/min × 60 × 24 × 365 = 1,051,200 fills/year
        let annualise = (1_051_200.0f64 / n as f64).sqrt();
        (mean / std_dev) * annualise
    }

    /// Print a summary line
    pub fn print_summary(&self, mid_f64: f64) {
        tracing::info!(
            "[SUMMARY] Pos={} lots ({:.4} {}) | R-PnL=${:.2} | U-PnL=${:.4} | \
             T-PnL=${:.4} | Sharpe={:.2} | DD=${:.2} | Orders={} Fills={}",
            self.position_lots,
            self.position_lots as f64 * LOT_SIZE,
            self.coin,
            self.realised_pnl,
            self.unrealised_pnl(mid_f64),
            self.total_pnl(mid_f64),
            self.sharpe(),
            self.max_drawdown(),
            self.orders_sent,
            self.fills_received,
        );
    }
}