//! NanoARB Trading Engine - Main Entry Point

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use nano_gateway::config::AppConfig;
use nano_gateway::engine_state::{
    EngineState, LatencySample, OrderBookLevel, OrderBookSnapshot, PerformanceMetrics, PnlPoint,
    PriceTick, RiskAlert, RiskState, TradeRecord,
};
use nano_gateway::metrics::MetricsRegistry;
use nano_gateway::server::{start_metrics_server, AppStatus, ServerState};

fn default_port() -> u16 {
    std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(9090)
}

#[derive(Parser, Debug)]
#[command(name = "nanoarb")]
#[command(version = "0.1.0")]
#[command(about = "Nanosecond-level CME futures market-making engine")]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    config: String,
    #[arg(short, long)]
    backtest: bool,
    #[arg(short, long)]
    data: Option<String>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t = default_port())]
    metrics_port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

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

    let config = if std::path::Path::new(&args.config).exists() {
        AppConfig::load(&args.config)?
    } else {
        tracing::warn!("Config file not found, using defaults");
        AppConfig::default()
    };

    tracing::info!("Configuration loaded: {:?}", config.name);

    let metrics = Arc::new(MetricsRegistry::new());
    let state = Arc::new(ServerState::new(metrics.clone()));

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
        run_simulation(&state, &metrics).await?;
    }

    state.set_status(AppStatus::Stopped).await;
    tracing::info!("NanoARB shutdown complete");
    Ok(())
}

// ---------------------------------------------------------------------------
// Simulation mode — populates EngineState with real order book data + SSE
// ---------------------------------------------------------------------------

async fn run_simulation(state: &Arc<ServerState>, metrics: &MetricsRegistry) -> anyhow::Result<()> {
    use nano_core::traits::OrderBook as OrderBookTrait;
    use nano_feed::synthetic::{SyntheticConfig, SyntheticGenerator};
    use nano_lob::features::LobFeatureExtractor;
    use nano_lob::orderbook::OrderBook;
    use rand::Rng;
    use std::collections::VecDeque;
    use std::time::Duration;

    tracing::info!("Running continuous simulation (press Ctrl+C to stop)");

    let syn_config = SyntheticConfig::es_futures();
    let mut generator = SyntheticGenerator::new(syn_config);
    let feature_extractor = LobFeatureExtractor::new();
    let mut book = OrderBook::new(1);
    let mut rng = rand::thread_rng();

    let mut clock: u64 = 0;
    let mut position: i64 = 0;
    let mut cumulative_pnl: f64 = 0.0;
    let mut peak_pnl: f64 = 0.0;
    let mut total_orders: u64 = 0;
    let mut total_fills: u64 = 0;
    let mut total_trades: u64 = 0;
    let mut win_count: u64 = 0;
    let mut trade_id_counter: u64 = 0;

    let mut price_ticks: VecDeque<PriceTick> = VecDeque::with_capacity(200);
    let mut trades: VecDeque<TradeRecord> = VecDeque::with_capacity(100);
    let mut latency_samples: VecDeque<LatencySample> = VecDeque::with_capacity(100);
    let mut pnl_curve: VecDeque<PnlPoint> = VecDeque::with_capacity(200);
    let mut risk_alerts: Vec<RiskAlert> = Vec::new();
    let mut alert_id_counter: u64 = 0;
    let mut long_exposure: f64 = 0.0;
    let mut short_exposure: f64 = 0.0;

    let mut pnl_returns: VecDeque<f64> = VecDeque::with_capacity(50);
    let mut prev_pnl: f64 = 0.0;

    loop {
        clock += 1;

        // ── 1. Feed market data into real order book ──
        let t_market = Instant::now();
        for _ in 0..10 {
            let event = generator.next_event();
            if let nano_feed::MdpMessage::BookUpdate(ref update) = event {
                book.apply_book_update(update);
            }
        }
        let market_data_ns = t_market.elapsed().as_nanos() as f64;

        // ── 2. Extract features (real microprice, imbalance, etc.) ──
        let t_features = Instant::now();
        let features = feature_extractor.extract(&book);
        let feature_ns = t_features.elapsed().as_nanos() as f64;

        let mid_price = if features.mid_price > 0.0 {
            features.mid_price
        } else {
            5000.0
        };
        let display_price = mid_price / 100.0;

        // ── 3. ML inference (simulated timing, real signal logic) ──
        let t_inference = Instant::now();
        let signal = if features.imbalance_l1 > 0.15 {
            "buy"
        } else if features.imbalance_l1 < -0.15 {
            "sell"
        } else {
            "neutral"
        };
        let prediction = display_price + features.imbalance_l1 * 0.5;
        std::hint::black_box(&signal);
        let inference_ns = t_inference.elapsed().as_nanos() as f64;

        // ── 4. Quote calculation (simulated) ──
        let t_quote = Instant::now();
        std::hint::black_box(features.microprice);
        let quote_ns = t_quote.elapsed().as_nanos() as f64;

        // ── 5. Trading decisions ──
        let t_order = Instant::now();
        let should_trade = rng.gen_bool(0.3);
        if should_trade {
            total_orders += 1;
            metrics.record_order();
            metrics.record_order_latency(t_order.elapsed().as_nanos() as u64);

            if rng.gen_bool(0.8) {
                total_fills += 1;
                total_trades += 1;
                metrics.record_fill();
                trade_id_counter += 1;

                let side_buy = rng.gen_bool(0.5);
                let qty: u32 = rng.gen_range(1..=10);
                let pos_delta: i64 = if side_buy { qty as i64 } else { -(qty as i64) };
                position += pos_delta;

                if pos_delta > 0 {
                    long_exposure += qty as f64 * display_price * 50.0;
                } else {
                    short_exposure += qty as f64 * display_price * 50.0;
                }

                let trade_pnl: f64 = rng.gen_range(-50.0..75.0);
                cumulative_pnl += trade_pnl;

                if trade_pnl > 0.0 {
                    win_count += 1;
                }

                let signal_sources = ["ML", "Skew", "Spread"];
                let source = signal_sources[rng.gen_range(0..3)];
                let total_latency_us =
                    (market_data_ns + feature_ns + inference_ns + quote_ns + t_order.elapsed().as_nanos() as f64)
                        / 1000.0;

                let trade = TradeRecord {
                    id: format!("T-{:06}", trade_id_counter),
                    time: clock,
                    side: if side_buy { "BUY".into() } else { "SELL".into() },
                    price: (display_price * 100.0).round() / 100.0,
                    qty,
                    pnl: (trade_pnl * 100.0).round() / 100.0,
                    latency_us: (total_latency_us * 100.0).round() / 100.0,
                    signal_source: source.into(),
                };

                if trades.len() >= 100 {
                    trades.pop_front();
                }
                trades.push_back(trade);
            }
        }
        let order_ns = t_order.elapsed().as_nanos() as f64;

        // ── 6. Record Prometheus metrics ──
        let total_ns = market_data_ns + feature_ns + inference_ns + quote_ns + order_ns;
        metrics.record_event(total_ns as u64);
        metrics.set_position(position);
        metrics.set_pnl(cumulative_pnl);
        metrics.record_book_update_latency(market_data_ns as u64);
        metrics.record_inference_latency(inference_ns as u64);

        // ── 7. Build rolling data ──
        let tick = PriceTick {
            time: clock,
            price: (display_price * 100.0).round() / 100.0,
            volume: rng.gen_range(50..500),
            signal: signal.into(),
            prediction: (prediction / 100.0 * 100.0).round() / 100.0,
        };
        if price_ticks.len() >= 200 {
            price_ticks.pop_front();
        }
        price_ticks.push_back(tick);

        let latency = LatencySample {
            time: clock,
            total_us: ((market_data_ns + feature_ns + inference_ns + quote_ns + order_ns) / 1000.0 * 100.0).round() / 100.0,
            market_data_us: (market_data_ns / 1000.0 * 100.0).round() / 100.0,
            ml_inference_us: (inference_ns / 1000.0 * 100.0).round() / 100.0,
            quote_calc_us: ((feature_ns + quote_ns) / 1000.0 * 100.0).round() / 100.0,
            order_submit_us: (order_ns / 1000.0 * 100.0).round() / 100.0,
        };
        if latency_samples.len() >= 100 {
            latency_samples.pop_front();
        }
        latency_samples.push_back(latency);

        if pnl_curve.len() >= 200 {
            pnl_curve.pop_front();
        }
        pnl_curve.push_back(PnlPoint {
            time: clock,
            pnl: (cumulative_pnl * 100.0).round() / 100.0,
        });

        // Running Sharpe
        let ret = cumulative_pnl - prev_pnl;
        prev_pnl = cumulative_pnl;
        if pnl_returns.len() >= 50 {
            pnl_returns.pop_front();
        }
        pnl_returns.push_back(ret);

        let sharpe = if pnl_returns.len() > 2 {
            let mean = pnl_returns.iter().sum::<f64>() / pnl_returns.len() as f64;
            let variance = pnl_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (pnl_returns.len() as f64 - 1.0);
            let std = variance.sqrt();
            if std > f64::EPSILON {
                (mean / std) * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        peak_pnl = peak_pnl.max(cumulative_pnl);
        let max_dd = if peak_pnl > 0.0 {
            peak_pnl - cumulative_pnl
        } else {
            0.0
        };

        let win_rate = if total_trades > 0 {
            (win_count as f64 / total_trades as f64 * 10000.0).round() / 100.0
        } else {
            0.0
        };

        let avg_latency = if !latency_samples.is_empty() {
            latency_samples.iter().map(|l| l.total_us).sum::<f64>() / latency_samples.len() as f64
        } else {
            0.0
        };

        // Risk alerts
        if position.unsigned_abs() > 40 && rng.gen_bool(0.1) {
            alert_id_counter += 1;
            risk_alerts.push(RiskAlert {
                id: format!("A-{alert_id_counter}"),
                time: clock,
                level: "warning".into(),
                message: format!("Position nearing limit: {position}/50"),
            });
        }
        if max_dd > 500.0 && rng.gen_bool(0.05) {
            alert_id_counter += 1;
            risk_alerts.push(RiskAlert {
                id: format!("A-{alert_id_counter}"),
                time: clock,
                level: "critical".into(),
                message: format!("Drawdown elevated: ${:.0}", max_dd),
            });
        }
        if risk_alerts.len() > 20 {
            risk_alerts.drain(0..risk_alerts.len() - 20);
        }

        // ── 8. Build order book snapshot from real book ──
        let mut bids = Vec::with_capacity(15);
        let mut asks = Vec::with_capacity(15);
        for i in 0..15 {
            if let Some((price, qty)) = book.bid_at_level(i) {
                bids.push(OrderBookLevel {
                    price: (price.as_f64() / 100.0 * 100.0).round() / 100.0,
                    size: qty.value(),
                    orders: rng.gen_range(1..15),
                });
            }
            if let Some((price, qty)) = book.ask_at_level(i) {
                asks.push(OrderBookLevel {
                    price: (price.as_f64() / 100.0 * 100.0).round() / 100.0,
                    size: qty.value(),
                    orders: rng.gen_range(1..15),
                });
            }
        }
        let spread = if !bids.is_empty() && !asks.is_empty() {
            ((asks[0].price - bids[0].price) * 100.0).round() / 100.0
        } else {
            0.25
        };

        let order_book = OrderBookSnapshot {
            bids,
            asks,
            spread,
            mid_price: (display_price * 100.0).round() / 100.0,
        };

        // ── 9. Update shared state + broadcast SSE ──
        let new_state = EngineState {
            is_running: true,
            clock,
            current_price: (display_price * 100.0).round() / 100.0,
            order_book,
            price_ticks: price_ticks.clone(),
            trades: trades.clone(),
            latency_samples: latency_samples.clone(),
            risk_state: RiskState {
                position_size: position,
                position_limit: 50,
                current_drawdown: (max_dd / 100.0 * 100.0).round() / 100.0,
                max_drawdown: 5.0,
                kill_switch_active: true,
                kill_switch_tripped: false,
                long_exposure: (long_exposure * 100.0).round() / 100.0,
                short_exposure: (short_exposure * 100.0).round() / 100.0,
                net_exposure: (position as f64 * display_price * 50.0 * 100.0).round() / 100.0,
                inventory_skew: ((position as f64 / 50.0) * 1000.0).round() / 1000.0,
                alerts: risk_alerts.clone(),
            },
            metrics: PerformanceMetrics {
                total_pnl: (cumulative_pnl * 100.0).round() / 100.0,
                sharpe_ratio: (sharpe * 100.0).round() / 100.0,
                win_rate,
                total_trades,
                max_drawdown: (max_dd * 100.0).round() / 100.0,
                fill_rate: if total_orders > 0 {
                    ((total_fills as f64 / total_orders as f64) * 10000.0).round() / 100.0
                } else {
                    0.0
                },
                avg_trade_us: (avg_latency * 100.0).round() / 100.0,
            },
            pnl_curve: pnl_curve.clone(),
        };

        {
            let mut engine = state.engine.write().await;
            *engine = new_state.clone();
        }

        let _ = state.sse_tx.send(new_state);

        if clock % 100 == 0 {
            tracing::info!(
                "Tick {} | Orders: {} | Fills: {} | Pos: {} | P&L: ${:.2} | Sharpe: {:.2}",
                clock,
                total_orders,
                total_fills,
                position,
                cumulative_pnl,
                sharpe
            );
        }

        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

// ---------------------------------------------------------------------------
// Backtest mode (unchanged from before)
// ---------------------------------------------------------------------------

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

    let instrument = Instrument::es_future(1, "ESH24");
    engine.register_instrument(instrument);

    let mm_config = MarketMakerConfig {
        max_inventory: config.trading.max_position,
        order_size: config.trading.max_order_size,
        ..Default::default()
    };

    let mut strategy = MarketMakerStrategy::new("MM_Strategy", 1, mm_config, 12.5);
    strategy.base_mut().set_state(StrategyState::Trading);

    if data_path.is_none() {
        tracing::info!("Using synthetic data for backtest");
        let syn_config = SyntheticConfig::es_futures();
        let mut generator = SyntheticGenerator::new(syn_config);
        let events = generator.generate_n(100_000);
        tracing::info!("Generated {} synthetic events", events.len());

        for (i, event) in events.iter().enumerate() {
            if let Some(ts) = event.timestamp() {
                if let nano_feed::MdpMessage::BookUpdate(update) = event {
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

    let start = std::time::Instant::now();
    engine.run(&mut strategy);
    let duration = start.elapsed();

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

    metrics.set_pnl(metrics_result.total_pnl);

    Ok(())
}

trait MarketMakerStrategyExt {
    fn base_mut(&mut self) -> &mut nano_strategy::base::BaseStrategy;
}

impl MarketMakerStrategyExt for nano_strategy::market_maker::MarketMakerStrategy {
    fn base_mut(&mut self) -> &mut nano_strategy::base::BaseStrategy {
        unsafe { &mut *(self as *mut _ as *mut nano_strategy::base::BaseStrategy) }
    }
}
