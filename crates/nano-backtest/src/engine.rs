//! Core backtest engine.

use std::collections::HashMap;

use nano_core::traits::{LatencyModel, OrderBook as OrderBookTrait, Strategy};
use nano_core::types::{Fill, Instrument, Order, OrderId, Price, Timestamp};
use nano_lob::OrderBook;

use crate::config::BacktestConfig;
use crate::events::{Event, EventQueue, EventType};
use crate::execution::SimulatedExchange;
use crate::latency::LatencySimulator;
use crate::metrics::{BacktestMetrics, PerformanceStats};
use crate::position::PositionTracker;
use crate::risk::RiskManager;

/// Backtest engine state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    /// Engine is ready to run
    Ready,
    /// Engine is running
    Running,
    /// Engine is paused
    Paused,
    /// Engine has completed
    Completed,
    /// Engine was stopped due to error or risk breach
    Stopped,
}

/// Main backtest engine
pub struct BacktestEngine {
    /// Configuration
    config: BacktestConfig,
    /// Event queue
    event_queue: EventQueue,
    /// Simulated exchange
    exchange: SimulatedExchange,
    /// Latency simulator
    latency: LatencySimulator,
    /// Position tracker
    positions: PositionTracker,
    /// Risk manager
    risk: RiskManager,
    /// Performance metrics
    metrics: BacktestMetrics,
    /// Detailed statistics
    stats: PerformanceStats,
    /// Order books by instrument
    books: HashMap<u32, OrderBook>,
    /// Instruments
    instruments: HashMap<u32, Instrument>,
    /// Current prices for P&L calculation
    current_prices: HashMap<u32, Price>,
    /// Engine state
    state: EngineState,
    /// Current timestamp
    current_time: Timestamp,
    /// Events processed count
    events_processed: u64,
    /// Last daily P&L update day
    last_day: Option<u32>,
}

impl BacktestEngine {
    /// Create a new backtest engine
    #[must_use]
    pub fn new(config: BacktestConfig) -> Self {
        let latency = LatencySimulator::from_config(&config.latency);
        let exchange = SimulatedExchange::new(config.fees.clone(), config.execution.clone());
        let risk = RiskManager::new(config.risk.clone());

        Self {
            config,
            event_queue: EventQueue::with_capacity(10000),
            exchange,
            latency,
            positions: PositionTracker::new(),
            risk,
            metrics: BacktestMetrics::new(),
            stats: PerformanceStats::new(),
            books: HashMap::new(),
            instruments: HashMap::new(),
            current_prices: HashMap::new(),
            state: EngineState::Ready,
            current_time: Timestamp::EPOCH,
            events_processed: 0,
            last_day: None,
        }
    }

    /// Register an instrument
    pub fn register_instrument(&mut self, instrument: Instrument) {
        let id = instrument.id;
        self.books.insert(id, OrderBook::new(id));
        self.positions.register_instrument(instrument.clone());
        self.instruments.insert(id, instrument);
    }

    /// Get the order book for an instrument
    #[must_use]
    pub fn get_book(&self, instrument_id: u32) -> Option<&OrderBook> {
        self.books.get(&instrument_id)
    }

    /// Get mutable order book
    pub fn get_book_mut(&mut self, instrument_id: u32) -> Option<&mut OrderBook> {
        self.books.get_mut(&instrument_id)
    }

    /// Schedule an event
    pub fn schedule_event(&mut self, timestamp: Timestamp, event_type: EventType) {
        self.event_queue.push(timestamp, event_type);
    }

    /// Process a single event
    pub fn process_event<S: Strategy>(&mut self, event: Event, strategy: &mut S) {
        self.current_time = event.timestamp;
        self.events_processed += 1;

        // Update daily tracking
        let day = (event.timestamp.as_nanos() / (24 * 60 * 60 * 1_000_000_000)) as u32;
        if self.last_day != Some(day) {
            let current_pnl = self.positions.total_pnl(&self.current_prices);
            self.risk.new_day(current_pnl, day);
            self.last_day = Some(day);
        }

        match &event.event_type {
            EventType::MarketData { instrument_id } => {
                self.on_market_data(*instrument_id, strategy);
            }
            EventType::OrderSubmit { order } => {
                self.on_order_submit(order.clone());
            }
            EventType::OrderAck { order_id } => {
                strategy.on_order_ack(*order_id);
            }
            EventType::OrderFill { fill } => {
                self.on_fill(fill.clone(), strategy);
            }
            EventType::OrderCancel { order_id } => {
                strategy.on_order_cancel(*order_id);
            }
            EventType::OrderReject { order_id, reason } => {
                strategy.on_order_reject(*order_id, reason);
            }
            EventType::Timer { timer_id: _, data: _ } => {
                // Timer events can be handled by strategy
            }
            EventType::Signal { name: _, value: _ } => {
                // Signal events for inter-strategy communication
            }
            EventType::EndOfData => {
                self.state = EngineState::Completed;
            }
            EventType::CancelReject { order_id, reason } => {
                strategy.on_order_reject(*order_id, reason);
            }
        }

        // Update metrics
        let total_pnl = self.positions.total_pnl(&self.current_prices);
        let realized = self.positions.realized_pnl();
        let unrealized = total_pnl - realized;
        self.metrics.update_pnl(total_pnl, realized, unrealized);

        // Check risk limits
        let position = strategy.position();
        if self.risk.update_pnl(total_pnl, position) {
            self.state = EngineState::Stopped;
        }
    }

    /// Handle market data update
    fn on_market_data<S: Strategy>(&mut self, instrument_id: u32, strategy: &mut S) {
        // First collect all the data we need from the book without holding a reference
        let (mid_price, orders, fills) = {
            let book = match self.books.get(&instrument_id) {
                Some(b) => b,
                None => return,
            };

            // Get mid price for P&L
            let mid = book.mid_price();

            // Let strategy process market data
            let orders = strategy.on_market_data(book);

            // Try to match existing orders
            let fills = self.exchange.match_orders(book, self.current_time);

            (mid, orders, fills)
        };

        // Now process without holding book reference
        if let Some(mid) = mid_price {
            self.current_prices.insert(instrument_id, mid);
        }

        // Submit any orders from strategy
        for order in orders {
            // Check risk
            let position = strategy.position();
            if let Err(e) = self.risk.check_order(&order, position) {
                tracing::warn!("Order rejected by risk: {}", e);
                continue;
            }

            // Schedule order with latency
            let arrival_time = self.latency.order_arrival_time(self.current_time);
            self.schedule_event(arrival_time, EventType::OrderSubmit { order });
        }

        // Schedule fill notifications
        for fill_result in fills {
            let notification_time = self.latency.fill_notification_time(self.current_time);
            self.schedule_event(
                notification_time,
                EventType::OrderFill {
                    fill: fill_result.fill,
                },
            );
        }
    }

    /// Handle order submission
    fn on_order_submit(&mut self, order: Order) {
        let order_id = self.exchange.submit_order(order, self.current_time);
        self.risk.on_order_submit();

        // Schedule acknowledgment
        let ack_time = self.latency.ack_reception_time(self.current_time);
        self.schedule_event(ack_time, EventType::OrderAck { order_id });
    }

    /// Handle fill
    fn on_fill<S: Strategy>(&mut self, fill: Fill, strategy: &mut S) {
        // Get instrument ID from order (simplified - in practice would track this)
        let instrument_id = 1; // Default

        self.positions.apply_fill_for_instrument(instrument_id, &fill);
        self.metrics.record_fill(&fill);
        self.risk.on_order_complete();

        strategy.on_fill(&fill);

        // Record equity point
        let total_pnl = self.positions.total_pnl(&self.current_prices);
        self.stats.add_equity_point(self.current_time.as_nanos(), total_pnl);
    }

    /// Run the backtest with a strategy
    pub fn run<S: Strategy>(&mut self, strategy: &mut S) {
        self.state = EngineState::Running;
        self.metrics.start_time = Some(self.current_time);

        while !self.event_queue.is_empty() && self.state == EngineState::Running {
            if let Some(event) = self.event_queue.pop() {
                self.process_event(event, strategy);
            }
        }

        self.metrics.end_time = Some(self.current_time);
        self.finalize_stats();

        if self.state == EngineState::Running {
            self.state = EngineState::Completed;
        }
    }

    /// Run for a limited number of events
    pub fn run_n<S: Strategy>(&mut self, strategy: &mut S, max_events: usize) -> usize {
        if self.state == EngineState::Ready {
            self.state = EngineState::Running;
            self.metrics.start_time = Some(self.current_time);
        }

        let mut processed = 0;

        while processed < max_events && !self.event_queue.is_empty() && self.state == EngineState::Running {
            if let Some(event) = self.event_queue.pop() {
                self.process_event(event, strategy);
                processed += 1;
            }
        }

        processed
    }

    /// Finalize statistics
    fn finalize_stats(&mut self) {
        // Calculate daily returns from equity curve
        // Clone equity to avoid borrow conflict
        let equity: Vec<f64> = self.stats.equity_curve.clone();
        if equity.len() > 1 {
            let mut prev = equity[0];
            for &curr in &equity[1..] {
                if prev.abs() > f64::EPSILON {
                    self.stats.add_daily_return((curr - prev) / prev.abs());
                }
                prev = curr;
            }
        }

        self.stats.calculate(self.config.initial_capital, self.metrics.max_drawdown_pct);
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> EngineState {
        self.state
    }

    /// Get metrics
    #[must_use]
    pub fn metrics(&self) -> &BacktestMetrics {
        &self.metrics
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &PerformanceStats {
        &self.stats
    }

    /// Get position tracker
    #[must_use]
    pub fn positions(&self) -> &PositionTracker {
        &self.positions
    }

    /// Get risk manager
    #[must_use]
    pub fn risk(&self) -> &RiskManager {
        &self.risk
    }

    /// Get current timestamp
    #[must_use]
    pub fn current_time(&self) -> Timestamp {
        self.current_time
    }

    /// Get events processed count
    #[must_use]
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }

    /// Get pending event count
    #[must_use]
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }

    /// Reset the engine
    pub fn reset(&mut self) {
        self.event_queue.clear();
        self.exchange.reset();
        self.latency.reset();
        self.positions.reset();
        self.risk.reset();
        self.metrics = BacktestMetrics::new();
        self.stats = PerformanceStats::new();
        self.current_prices.clear();
        self.state = EngineState::Ready;
        self.current_time = Timestamp::EPOCH;
        self.events_processed = 0;
        self.last_day = None;

        for book in self.books.values_mut() {
            book.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = BacktestConfig::default();
        let engine = BacktestEngine::new(config);

        assert_eq!(engine.state(), EngineState::Ready);
        assert_eq!(engine.events_processed(), 0);
    }

    #[test]
    fn test_register_instrument() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);

        let instrument = Instrument::es_future(1, "ESH24");
        engine.register_instrument(instrument);

        assert!(engine.get_book(1).is_some());
    }

    #[test]
    fn test_schedule_event() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);

        engine.schedule_event(
            Timestamp::from_nanos(1_000_000),
            EventType::EndOfData,
        );

        assert_eq!(engine.pending_events(), 1);
    }
}

