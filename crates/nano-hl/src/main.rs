//! nano-hl — Hyperliquid paper-trading engine entry point.
//!
//! Usage:
//!   cargo run -p nano-hl -- --coin BTC --testnet
//!   cargo run -p nano-hl -- --coin ETH --mainnet --spread 50

use clap::Parser;
use tokio::sync::mpsc;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use nano_hl::{
    engine::{run, EngineConfig},
    feed::run_feed,
    metrics::{run_metrics_server, Metrics},
};

#[derive(Parser, Debug)]
#[command(name = "nano-hl")]
#[command(about = "Hyperliquid paper-trading engine built on nanoARB")]
struct Args {
    /// Coin to trade (e.g. BTC, ETH, SOL)
    #[arg(long, default_value = "BTC")]
    coin: String,

    /// Use Hyperliquid testnet (default: true)
    #[arg(long, default_value_t = true)]
    testnet: bool,

    /// Use Hyperliquid mainnet instead of testnet
    #[arg(long)]
    mainnet: bool,

    /// Spread in ticks (1 tick = $0.01). Default 100 = $1.00 spread.
    #[arg(long, default_value_t = 100)]
    spread: i64,

    /// Max inventory in lots (1 lot = 0.001 base asset)
    #[arg(long, default_value_t = 100)]
    max_inventory: i64,

    /// Order size per level in lots
    #[arg(long, default_value_t = 10)]
    order_size: u32,

    /// Number of quote levels per side
    #[arg(long, default_value_t = 3)]
    levels: usize,

    /// Enable debug logging
    #[arg(long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── Logging ────────────────────────────────────────────────────────────
    let filter = if args.debug {
        EnvFilter::new("nano_hl=debug,info")
    } else {
        EnvFilter::new("nano_hl=info,warn")
    };
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    // ── Config ─────────────────────────────────────────────────────────────
    let testnet = !args.mainnet;
    let network = if testnet { "TESTNET" } else { "MAINNET" };

    tracing::info!(
        "=== nano-hl paper trader ===  coin={} network={} spread=${:.2}",
        args.coin,
        network,
        args.spread as f64 * 0.01
    );

    if !testnet {
        tracing::warn!("⚠️  Running on MAINNET — data is real but orders are paper only");
    }

    let engine_config = EngineConfig {
        coin:               args.coin.clone(),
        testnet,
        spread_ticks:       5000,  // $50 wide — guaranteed to straddle testnet spread
        max_inventory_lots: 100,
        order_size_lots:    10,
        num_levels:         1,     // 1 level each side, simple
        summary_every_n:    20,
    };

    // ── Metrics ────────────────────────────────────────────────────────────
    let metrics = Metrics::new();

    // Spawn Prometheus metrics server on port 9090
    let metrics_for_server = metrics.clone();
    tokio::spawn(async move {
        run_metrics_server(metrics_for_server).await;
    });

    // ── Channel: feed task → engine task ───────────────────────────────────
    let (tx, rx) = mpsc::channel(64);

    // ── Spawn feed task ────────────────────────────────────────────────────
    let feed_coin = args.coin.clone();
    tokio::spawn(async move {
        run_feed(feed_coin, testnet, tx).await;
    });

    // ── Run engine (blocks until feed closes) ──────────────────────────────
    run(engine_config, rx, metrics).await;

    Ok(())
}