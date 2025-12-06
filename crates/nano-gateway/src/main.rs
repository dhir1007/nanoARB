//! NanoARB Trading Engine - Main Entry Point
//!
//! A nanosecond-level CME futures market-making engine with sub-microsecond inference.

use std::sync::Arc;

use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use nano_gateway::config::AppConfig;
use nano_gateway::metrics::MetricsRegistry;
use nano_gateway::server::{start_metrics_server, AppStatus, ServerState};

/// NanoARB Trading Engine
#[derive(Parser, Debug)]
#[command(name = "nanoarb")]
#[command(author = "NanoARB Contributors")]
#[command(version = "0.1.0")]
#[command(about = "Nanosecond-level CME futures market-making engine", long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Run in backtest mode
    #[arg(short, long)]
    backtest: bool,

    /// Data file for backtest
    #[arg(short, long)]
    data: Option<String>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Metrics server port
    #[arg(short, long, default_value = "9090")]
    metrics_port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    let filter = if args.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    tracing::info!("Starting NanoARB Trading Engine v0.1.0");

    // Load configuration
    let config = if std::path::Path::new(&args.config).exists() {
        AppConfig::load(&args.config)?
    } else {
        tracing::warn!("Config file not found, using defaults");
        AppConfig::default()
    };

    tracing::info!("Configuration loaded: {:?}", config.name);

    // Initialize metrics
    let metrics = Arc::new(MetricsRegistry::new());
    let state = Arc::new(ServerState::new(metrics.clone()));

    // Start metrics server
    let metrics_state = state.clone();
    let metrics_port = args.metrics_port;
    tokio::spawn(async move {
        if let Err(e) = start_metrics_server(metrics_state, metrics_port).await {
            tracing::error!("Metrics server error: {}", e);
        }
    });

    state.set_status(AppStatus::Running).await;

    if args.backtest {
        tracing::info!("Running in backtest mode");
        run_backtest(&config, args.data.as_deref(), &metrics).await?;
    } else {
        tracing::info!("Running in simulation mode (live trading disabled)");
        run_simulation(&config, &metrics).await?;
    }

    state.set_status(AppStatus::Stopped).await;
    tracing::info!("NanoARB shutdown complete");

    Ok(())
}

/// Run backtest
async fn run_backtest(
    config: &AppConfig,
    data_path: Option<&str>,
    metrics: &MetricsRegistry,
) -> anyhow::Result<()> {
    use nano_backtest::config::BacktestConfig;
    use nano_backtest::engine::BacktestEngine;
    use nano_core::types::Instrument;
    use nano_feed::synthetic::{SyntheticConfig, SyntheticGenerator};
    use nano_strategy::base::StrategyState;
    use nano_strategy::market_maker::{MarketMakerConfig, MarketMakerStrategy};

    tracing::info!("Initializing backtest engine");

    let backtest_config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(backtest_config);

    // Register instrument
    let instrument = Instrument::es_future(1, "ESH24");
    engine.register_instrument(instrument);

    // Create strategy
    let mm_config = MarketMakerConfig {
        max_inventory: config.trading.max_position,
        order_size: config.trading.max_order_size,
        ..Default::default()
    };

    let mut strategy = MarketMakerStrategy::new(
        "MM_Strategy",
        1,
        mm_config,
        12.5, // ES tick value
    );
    strategy.base_mut().set_state(StrategyState::Trading);

    // Generate synthetic data if no data file provided
    if data_path.is_none() {
        tracing::info!("Using synthetic data for backtest");
        let syn_config = SyntheticConfig::es_futures();
        let mut generator = SyntheticGenerator::new(syn_config);

        // Generate events
        let events = generator.generate_n(100_000);
        tracing::info!("Generated {} synthetic events", events.len());

        // Schedule events
        for (i, event) in events.iter().enumerate() {
            if let Some(ts) = event.timestamp() {
                if let nano_feed::MdpMessage::BookUpdate(update) = event {
                    // Apply to book and schedule
                    if let Some(book) = engine.get_book_mut(1) {
                        book.apply_book_update(update);
                    }
                    engine.schedule_event(
                        ts,
                        nano_backtest::events::EventType::MarketData { instrument_id: 1 },
                    );
                }
            }

            if i % 10000 == 0 {
                tracing::debug!("Scheduled {} events", i);
            }
        }
    }

    // Run backtest
    let start = std::time::Instant::now();
    engine.run(&mut strategy);
    let duration = start.elapsed();

    // Report results
    let metrics_result = engine.metrics();
    let stats = engine.stats();

    tracing::info!("Backtest completed in {:?}", duration);
    tracing::info!("Events processed: {}", engine.events_processed());
    tracing::info!("Total P&L: ${:.2}", metrics_result.total_pnl);
    tracing::info!(
        "Max Drawdown: {:.2}%",
        metrics_result.max_drawdown_pct * 100.0
    );
    tracing::info!("Sharpe Ratio: {:.2}", stats.sharpe_ratio);
    tracing::info!("Win Rate: {:.2}%", metrics_result.win_rate() * 100.0);
    tracing::info!("Profit Factor: {:.2}", metrics_result.profit_factor());

    // Update Prometheus metrics
    metrics.set_pnl(metrics_result.total_pnl);

    Ok(())
}

/// Run simulation (paper trading)
async fn run_simulation(_config: &AppConfig, metrics: &MetricsRegistry) -> anyhow::Result<()> {
    tracing::info!("Starting simulation mode");

    // In a full implementation, this would:
    // 1. Connect to market data feed
    // 2. Run the strategy in real-time
    // 3. Track paper trades

    // For now, run continuous synthetic simulation with metrics
    tracing::info!("Running continuous simulation (press Ctrl+C to stop)");

    use nano_feed::synthetic::{SyntheticConfig, SyntheticGenerator};
    use std::time::Duration;

    let syn_config = SyntheticConfig::es_futures();
    let mut generator = SyntheticGenerator::new(syn_config);
    let mut iteration = 0u64;

    loop {
        // Generate synthetic events
        for _ in 0..100 {
            let _event = generator.next_event();
        }

        // Record metrics
        let latency_ns = 500 + (iteration % 1000); // Simulated latency
        metrics.record_event(latency_ns);
        
        iteration += 1;

        // Log progress every 10 seconds
        if iteration % 100 == 0 {
            tracing::info!(
                "Simulation running: {} iterations, recording metrics",
                iteration * 100
            );
        }

        // Sleep to simulate realistic rate (100ms between batches)
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// Helper trait extension to access base strategy
trait MarketMakerStrategyExt {
    fn base_mut(&mut self) -> &mut nano_strategy::base::BaseStrategy;
}

impl MarketMakerStrategyExt for nano_strategy::market_maker::MarketMakerStrategy {
    fn base_mut(&mut self) -> &mut nano_strategy::base::BaseStrategy {
        // This is a workaround - in production, add proper method to strategy
        unsafe { &mut *(self as *mut _ as *mut nano_strategy::base::BaseStrategy) }
    }
}
