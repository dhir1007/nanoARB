//! Prometheus metrics for monitoring.

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;
use std::sync::atomic::AtomicU64;

/// Metrics registry for the trading engine
#[derive(Debug)]
pub struct MetricsRegistry {
    /// Prometheus registry
    registry: Registry,
    /// Order count
    pub orders_total: Counter,
    /// Fill count
    pub fills_total: Counter,
    /// Current position gauge
    pub position: Gauge,
    /// Current P&L gauge
    pub pnl: Gauge<f64, AtomicU64>,
    /// Inference latency histogram (nanoseconds)
    pub inference_latency_ns: Histogram,
    /// Order latency histogram (nanoseconds)
    pub order_latency_ns: Histogram,
    /// Book update latency histogram
    pub book_update_latency_ns: Histogram,
    /// Event processing latency
    pub event_latency_ns: Histogram,
    /// Events processed total
    pub events_total: Counter,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Registry::default();

        // Counters
        let orders_total = Counter::default();
        registry.register(
            "nanoarb_orders_total",
            "Total number of orders submitted",
            orders_total.clone(),
        );

        let fills_total = Counter::default();
        registry.register(
            "nanoarb_fills_total",
            "Total number of fills received",
            fills_total.clone(),
        );

        let events_total = Counter::default();
        registry.register(
            "nanoarb_events_total",
            "Total events processed",
            events_total.clone(),
        );

        // Gauges
        let position = Gauge::default();
        registry.register("nanoarb_position", "Current net position", position.clone());

        let pnl = Gauge::<f64, AtomicU64>::default();
        registry.register("nanoarb_pnl", "Current P&L in dollars", pnl.clone());

        // Histograms with nanosecond buckets
        let ns_buckets: Vec<f64> = exponential_buckets(100.0, 2.0, 20).collect(); // 100ns to ~100ms

        let inference_latency_ns = Histogram::new(ns_buckets.iter().copied());
        registry.register(
            "nanoarb_inference_latency_ns",
            "Model inference latency in nanoseconds",
            inference_latency_ns.clone(),
        );

        let order_latency_ns = Histogram::new(ns_buckets.iter().copied());
        registry.register(
            "nanoarb_order_latency_ns",
            "Order submission latency in nanoseconds",
            order_latency_ns.clone(),
        );

        let book_update_latency_ns = Histogram::new(ns_buckets.iter().copied());
        registry.register(
            "nanoarb_book_update_latency_ns",
            "Book update processing latency in nanoseconds",
            book_update_latency_ns.clone(),
        );

        let event_latency_ns = Histogram::new(ns_buckets.iter().copied());
        registry.register(
            "nanoarb_event_latency_ns",
            "Event processing latency in nanoseconds",
            event_latency_ns.clone(),
        );

        Self {
            registry,
            orders_total,
            fills_total,
            position,
            pnl,
            inference_latency_ns,
            order_latency_ns,
            book_update_latency_ns,
            event_latency_ns,
            events_total,
        }
    }

    /// Record an order submission
    pub fn record_order(&self) {
        self.orders_total.inc();
    }

    /// Record a fill
    pub fn record_fill(&self) {
        self.fills_total.inc();
    }

    /// Update position
    pub fn set_position(&self, pos: i64) {
        self.position.set(pos);
    }

    /// Update P&L
    pub fn set_pnl(&self, pnl: f64) {
        self.pnl.set(pnl);
    }

    /// Record inference latency
    pub fn record_inference_latency(&self, latency_ns: u64) {
        self.inference_latency_ns.observe(latency_ns as f64);
    }

    /// Record order latency
    pub fn record_order_latency(&self, latency_ns: u64) {
        self.order_latency_ns.observe(latency_ns as f64);
    }

    /// Record book update latency
    pub fn record_book_update_latency(&self, latency_ns: u64) {
        self.book_update_latency_ns.observe(latency_ns as f64);
    }

    /// Record event processing
    pub fn record_event(&self, latency_ns: u64) {
        self.events_total.inc();
        self.event_latency_ns.observe(latency_ns as f64);
    }

    /// Encode metrics for Prometheus scraping
    #[must_use]
    pub fn encode(&self) -> String {
        let mut buffer = String::new();
        encode(&mut buffer, &self.registry).expect("Failed to encode metrics");
        buffer
    }

    /// Get registry reference
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = MetricsRegistry::new();

        metrics.record_order();
        metrics.record_fill();
        metrics.set_position(10);
        metrics.set_pnl(1000.0);
        metrics.record_inference_latency(500);

        let output = metrics.encode();
        assert!(output.contains("nanoarb_orders_total"));
        assert!(output.contains("nanoarb_fills_total"));
    }
}
