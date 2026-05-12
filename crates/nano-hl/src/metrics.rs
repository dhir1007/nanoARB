//! Prometheus metrics server.
//!
//! Exposes metrics on http://localhost:9090/metrics so Grafana can scrape them.
//! Metric names match exactly what the existing dashboard expects.

use std::sync::Arc;

use hyper::{
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server,
};
use prometheus::{
    Counter, Encoder, Gauge, Histogram, HistogramOpts, Registry, TextEncoder,
};

/// All metrics in one struct — clone the Arc to share across tasks
#[derive(Clone)]
pub struct Metrics {
    pub registry: Arc<Registry>,
    /// Current PnL in USD
    pub pnl: Gauge,
    /// Current position in lots
    pub position: Gauge,
    /// Total orders submitted (paper)
    pub orders_total: Counter,
    /// Total fills received
    pub fills_total: Counter,
    /// Total book update events processed
    pub events_total: Counter,
    /// Tick processing latency in nanoseconds
    pub tick_latency_ns: Histogram,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let pnl = Gauge::new("nanoarb_pnl", "Current realised PnL in USD").unwrap();
        let position = Gauge::new("nanoarb_position", "Current position in lots").unwrap();
        let orders_total = Counter::new("nanoarb_orders_total_total", "Total paper orders").unwrap();
        let fills_total  = Counter::new("nanoarb_fills_total_total",  "Total paper fills").unwrap();
        let events_total = Counter::new("nanoarb_events_total_total", "Total book events").unwrap();
        let tick_latency_ns = Histogram::with_opts(
            HistogramOpts::new(
                "nanoarb_inference_latency_ns",
                "Tick processing latency in nanoseconds",
            )
            .buckets(vec![
                100.0, 500.0, 1_000.0, 5_000.0, 10_000.0,
                50_000.0, 100_000.0, 500_000.0, 1_000_000.0,
            ]),
        )
        .unwrap();

        registry.register(Box::new(pnl.clone())).unwrap();
        registry.register(Box::new(position.clone())).unwrap();
        registry.register(Box::new(orders_total.clone())).unwrap();
        registry.register(Box::new(fills_total.clone())).unwrap();
        registry.register(Box::new(events_total.clone())).unwrap();
        registry.register(Box::new(tick_latency_ns.clone())).unwrap();

        Self {
            registry: Arc::new(registry),
            pnl,
            position,
            orders_total,
            fills_total,
            events_total,
            tick_latency_ns,
        }
    }
}

/// Start the Prometheus HTTP server on port 9090.
/// Call this once with a cloned `Metrics` and spawn the returned future.
pub async fn run_metrics_server(metrics: Metrics) {
    let addr = ([0, 0, 0, 0], 9090).into();

    let make_svc = make_service_fn(move |_conn| {
        let metrics = metrics.clone();
        async move {
            Ok::<_, hyper::Error>(service_fn(move |req: Request<Body>| {
                let metrics = metrics.clone();
                async move { handle(req, metrics).await }
            }))
        }
    });

    let server = Server::bind(&addr).serve(make_svc);
    tracing::info!("Prometheus metrics on http://0.0.0.0:9090/metrics");

    if let Err(e) = server.await {
        tracing::error!("Metrics server error: {e}");
    }
}

async fn handle(req: Request<Body>, metrics: Metrics) -> Result<Response<Body>, hyper::Error> {
    if req.uri().path() != "/metrics" {
        return Ok(Response::builder()
            .status(404)
            .body(Body::from("not found"))
            .unwrap());
    }

    let encoder = TextEncoder::new();
    let metric_families = metrics.registry.gather();
    let mut buf = Vec::new();
    encoder.encode(&metric_families, &mut buf).unwrap();

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", encoder.format_type())
        .body(Body::from(buf))
        .unwrap())
}