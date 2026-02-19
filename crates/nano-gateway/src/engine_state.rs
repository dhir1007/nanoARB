//! Shared engine state streamed to the UI via SSE.
//!
//! Every struct here uses `#[serde(rename_all = "camelCase")]` so JSON field names
//! match the TypeScript interfaces in the UI without any transformation layer.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Order Book
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderBookLevel {
    pub price: f64,
    pub size: u32,
    pub orders: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderBookSnapshot {
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub spread: f64,
    pub mid_price: f64,
}

impl Default for OrderBookSnapshot {
    fn default() -> Self {
        Self {
            bids: Vec::new(),
            asks: Vec::new(),
            spread: 0.0,
            mid_price: 5425.75,
        }
    }
}

// ---------------------------------------------------------------------------
// Price Tick
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PriceTick {
    pub time: u64,
    pub price: f64,
    pub volume: u32,
    pub signal: String,
    pub prediction: f64,
}

// ---------------------------------------------------------------------------
// Trade Record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TradeRecord {
    pub id: String,
    pub time: u64,
    pub side: String,
    pub price: f64,
    pub qty: u32,
    pub pnl: f64,
    pub latency_us: f64,
    pub signal_source: String,
}

// ---------------------------------------------------------------------------
// Latency Sample
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LatencySample {
    pub time: u64,
    pub total_us: f64,
    pub market_data_us: f64,
    pub ml_inference_us: f64,
    pub quote_calc_us: f64,
    pub order_submit_us: f64,
}

// ---------------------------------------------------------------------------
// Risk State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RiskAlert {
    pub id: String,
    pub time: u64,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RiskState {
    pub position_size: i64,
    pub position_limit: i64,
    pub current_drawdown: f64,
    pub max_drawdown: f64,
    pub kill_switch_active: bool,
    pub kill_switch_tripped: bool,
    pub long_exposure: f64,
    pub short_exposure: f64,
    pub net_exposure: f64,
    pub inventory_skew: f64,
    pub alerts: Vec<RiskAlert>,
}

impl Default for RiskState {
    fn default() -> Self {
        Self {
            position_size: 0,
            position_limit: 50,
            current_drawdown: 0.0,
            max_drawdown: 5.0,
            kill_switch_active: true,
            kill_switch_tripped: false,
            long_exposure: 0.0,
            short_exposure: 0.0,
            net_exposure: 0.0,
            inventory_skew: 0.0,
            alerts: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Performance Metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceMetrics {
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub total_trades: u64,
    pub max_drawdown: f64,
    pub fill_rate: f64,
    pub avg_trade_us: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_pnl: 0.0,
            sharpe_ratio: 0.0,
            win_rate: 0.0,
            total_trades: 0,
            max_drawdown: 0.0,
            fill_rate: 96.2,
            avg_trade_us: 1.45,
        }
    }
}

// ---------------------------------------------------------------------------
// P&L Point
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PnlPoint {
    pub time: u64,
    pub pnl: f64,
}

// ---------------------------------------------------------------------------
// Top-Level Engine State (streamed via SSE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineState {
    pub is_running: bool,
    pub clock: u64,
    pub current_price: f64,
    pub order_book: OrderBookSnapshot,
    pub price_ticks: VecDeque<PriceTick>,
    pub trades: VecDeque<TradeRecord>,
    pub latency_samples: VecDeque<LatencySample>,
    pub risk_state: RiskState,
    pub metrics: PerformanceMetrics,
    pub pnl_curve: VecDeque<PnlPoint>,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            is_running: false,
            clock: 0,
            current_price: 5425.75,
            order_book: OrderBookSnapshot::default(),
            price_ticks: VecDeque::with_capacity(200),
            trades: VecDeque::with_capacity(100),
            latency_samples: VecDeque::with_capacity(100),
            risk_state: RiskState::default(),
            metrics: PerformanceMetrics::default(),
            pnl_curve: VecDeque::with_capacity(200),
        }
    }
}

// ---------------------------------------------------------------------------
// Backtest API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BacktestRequest {
    pub symbol: String,
    pub start_date: String,
    pub end_date: String,
    pub initial_capital: f64,
    pub spread_multiplier: f64,
    pub inventory_limit: i64,
    pub skew_factor: f64,
    #[serde(alias = "useML")]
    pub use_ml: bool,
    pub max_drawdown: f64,
    pub position_limit: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EquityPoint {
    pub day: u32,
    pub equity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DrawdownPoint {
    pub day: u32,
    pub drawdown: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MonthlyReturn {
    pub month: String,
    #[serde(rename = "return")]
    pub ret: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TradeBucket {
    pub bucket: String,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BacktestResponse {
    pub equity_curve: Vec<EquityPoint>,
    pub drawdown_curve: Vec<DrawdownPoint>,
    pub total_return: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_trade_pnl: f64,
    pub total_trades: u64,
    pub monthly_returns: Vec<MonthlyReturn>,
    pub trade_distribution: Vec<TradeBucket>,
}
