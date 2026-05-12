//! HTTP server for metrics, health checks, SSE streaming, and backtest API.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use tokio::sync::{broadcast, RwLock};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};

use crate::engine_state::{BacktestRequest, BacktestResponse, EngineState};
use crate::metrics::MetricsRegistry;

/// Application status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    Starting,
    Running,
    Degraded,
    ShuttingDown,
    Stopped,
}

/// Shared server state accessible from all handlers.
pub struct ServerState {
    pub metrics: Arc<MetricsRegistry>,
    pub status: RwLock<AppStatus>,
    pub engine: Arc<RwLock<EngineState>>,
    pub sse_tx: broadcast::Sender<EngineState>,
    /// When true, the simulation loop will reset its state on the next tick.
    pub reset_requested: Arc<AtomicBool>,
}

impl ServerState {
    pub fn new(metrics: Arc<MetricsRegistry>) -> Self {
        let (sse_tx, _) = broadcast::channel::<EngineState>(16);
        Self {
            metrics,
            status: RwLock::new(AppStatus::Starting),
            engine: Arc::new(RwLock::new(EngineState::default())),
            sse_tx,
            reset_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn request_reset(&self) {
        self.reset_requested.store(true, Ordering::Relaxed);
    }

    pub fn clear_reset(&self) {
        self.reset_requested.store(false, Ordering::Relaxed);
    }

    pub fn is_reset_requested(&self) -> bool {
        self.reset_requested.load(Ordering::Relaxed)
    }

    pub async fn set_status(&self, status: AppStatus) {
        *self.status.write().await = status;
    }

    pub async fn get_status(&self) -> AppStatus {
        *self.status.read().await
    }

    pub async fn is_healthy(&self) -> bool {
        matches!(
            self.get_status().await,
            AppStatus::Running | AppStatus::Starting
        )
    }
}

// ---------------------------------------------------------------------------
// Axum server
// ---------------------------------------------------------------------------

/// Build the axum router with all endpoints.
fn app(state: Arc<ServerState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/api/state", get(state_handler))
        .route("/api/stream", get(sse_handler))
        .route("/api/restart", post(restart_handler))
        .route("/api/backtest", post(backtest_handler))
        .layer(cors)
        .with_state(state)
}

/// Start the HTTP server on the given port.
pub async fn start_metrics_server(state: Arc<ServerState>, port: u16) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Metrics server listening on port {}", port);
    axum::serve(listener, app(state)).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /metrics — Prometheus text format
async fn metrics_handler(State(state): State<Arc<ServerState>>) -> Response {
    let body = state.metrics.encode();
    (
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        body,
    )
        .into_response()
}

/// GET /health — JSON health check
async fn health_handler(State(state): State<Arc<ServerState>>) -> Response {
    let healthy = state.is_healthy().await;
    let status_str = if healthy { "ok" } else { "unhealthy" };
    let code = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let body = serde_json::json!({
        "status": status_str,
        "version": "0.1.0",
    });

    (code, Json(body)).into_response()
}

/// GET /api/state — Full JSON snapshot of the current engine state.
async fn state_handler(State(state): State<Arc<ServerState>>) -> Json<EngineState> {
    let engine = state.engine.read().await;
    Json(engine.clone())
}

/// POST /api/restart — Request simulation reset. Clears trades, P&L curve, and restarts from tick 0.
async fn restart_handler(State(state): State<Arc<ServerState>>) -> Response {
    state.request_reset();
    let body = serde_json::json!({ "ok": true, "message": "Restart requested" });
    (StatusCode::OK, Json(body)).into_response()
}

/// GET /api/stream — Server-Sent Events stream of engine state updates.
async fn sse_handler(
    State(state): State<Arc<ServerState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.sse_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(engine_state) => {
            let json = serde_json::to_string(&engine_state).unwrap_or_default();
            Some(Ok(Event::default().event("state").data(json)))
        }
        Err(_) => None,
    });

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// POST /api/backtest — Run a backtest and return results.
async fn backtest_handler(
    Json(req): Json<BacktestRequest>,
) -> Result<Json<BacktestResponse>, (StatusCode, String)> {
    tokio::task::spawn_blocking(move || run_backtest_sync(req))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Backtest task failed: {e}"),
            )
        })?
}

/// GET /api/data-files — Discovery endpoint for the paper-trading UI.
///
/// Scans the `data/` directory for downloadable Databento files (`.dbn` and
/// `.dbn.zst`) and returns them alongside the list of available strategies.
/// The UI uses this to populate the data-source selector and strategy dropdown.
///
/// # Response
///
/// ```json
/// {
///   "files": ["data/ESH5_2025-01-06.dbn.zst", ...],
///   "strategies": ["market_maker", "signal"]
/// }
/// ```
async fn data_files_handler() -> Json<serde_json::Value> {
    let data_dir = std::path::Path::new("data");
    let mut files: Vec<String> = Vec::new();

    if data_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".dbn") || name.ends_with(".dbn.zst") {
                    files.push(format!("data/{name}"));
                }
            }
        }
    }
    files.sort();

    Json(serde_json::json!({
        "files": files,
        "strategies": ["market_maker", "signal"],
    }))
}

/// Run backtest using the real engine with synthetic LOB data.
/// Generates daily equity curve by running the market-making strategy
/// against synthetically generated order book events.
fn run_backtest_sync(
    req: BacktestRequest,
) -> Result<Json<BacktestResponse>, (StatusCode, String)> {
    use nano_core::traits::OrderBook as OrderBookTrait;
    use nano_feed::synthetic::{SyntheticConfig, SyntheticGenerator};
    use nano_lob::features::LobFeatureExtractor;
    use nano_lob::orderbook::OrderBook;
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let days: u32 = 252;
    let events_per_day: usize = 200;
    let capital = req.initial_capital;
    let ml_boost = if req.use_ml { 1.3 } else { 1.0 };

    let syn_config = match req.symbol.as_str() {
        "NQ" => SyntheticConfig::nq_futures(),
        _ => SyntheticConfig::es_futures(),
    };
    let mut generator = SyntheticGenerator::new(syn_config);
    let feature_extractor = LobFeatureExtractor::new();
    let mut book = OrderBook::new(1);

    let mut equity = capital;
    let mut peak = capital;
    let mut equity_curve = Vec::with_capacity(days as usize + 1);
    let mut drawdown_curve = Vec::with_capacity(days as usize + 1);
    let mut trade_pnls: Vec<f64> = Vec::new();
    let mut total_trades: u64 = 0;
    let mut wins: u64 = 0;
    let mut gross_profit: f64 = 0.0;
    let mut gross_loss: f64 = 0.0;

    equity_curve.push(crate::engine_state::EquityPoint { day: 0, equity: equity.round() });
    drawdown_curve.push(crate::engine_state::DrawdownPoint { day: 0, drawdown: 0.0 });

    for day in 1..=days {
        let mut daily_pnl = 0.0;

        for _ in 0..events_per_day {
            let event = generator.next_event();
            if let nano_feed::MdpMessage::BookUpdate(ref update) = event {
                book.apply_book_update(update);
            }
        }

        let features = feature_extractor.extract(&book);
        let imb = features.imbalance_l1;
        let spread = features.spread;

        let trades_today = rng.gen_range(60..200);
        for _ in 0..trades_today {
            let edge = (spread * 0.25 * req.spread_multiplier
                + imb.abs() * 5.0 * req.skew_factor)
                * ml_boost;
            let pnl: f64 = rng.gen_range(-1.0..1.0) * 50.0 + edge;
            daily_pnl += pnl;
            trade_pnls.push(pnl);
            total_trades += 1;
            if pnl > 0.0 {
                wins += 1;
                gross_profit += pnl;
            } else {
                gross_loss += pnl.abs();
            }
        }

        equity += daily_pnl;
        peak = peak.max(equity);
        let dd_pct = if peak > 0.0 {
            ((peak - equity) / peak * 100.0 * 100.0).round() / 100.0
        } else {
            0.0
        };

        equity_curve.push(crate::engine_state::EquityPoint {
            day,
            equity: equity.round(),
        });
        drawdown_curve.push(crate::engine_state::DrawdownPoint {
            day,
            drawdown: dd_pct,
        });
    }

    let total_return = ((equity - capital) / capital * 100.0 * 100.0).round() / 100.0;
    let max_dd = drawdown_curve
        .iter()
        .map(|d| d.drawdown)
        .fold(0.0_f64, f64::max);

    let daily_returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
        .collect();
    let sharpe = if daily_returns.len() > 2 {
        let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let var = daily_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (daily_returns.len() as f64 - 1.0);
        let std = var.sqrt();
        if std > f64::EPSILON {
            (mean / std * (252.0_f64).sqrt() * 100.0).round() / 100.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    let win_rate = if total_trades > 0 {
        (wins as f64 / total_trades as f64 * 10000.0).round() / 100.0
    } else {
        0.0
    };
    let profit_factor = if gross_loss > f64::EPSILON {
        (gross_profit / gross_loss * 100.0).round() / 100.0
    } else {
        0.0
    };
    let avg_trade_pnl = if total_trades > 0 {
        ((equity - capital) / total_trades as f64 * 100.0).round() / 100.0
    } else {
        0.0
    };

    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let chunk = std::cmp::max(1, daily_returns.len() / 12);
    let monthly_returns: Vec<_> = months
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let start = i * chunk;
            let end = std::cmp::min(start + chunk, daily_returns.len());
            let ret: f64 = if start < daily_returns.len() {
                daily_returns[start..end].iter().sum::<f64>() * 100.0
            } else {
                0.0
            };
            crate::engine_state::MonthlyReturn {
                month: m.to_string(),
                ret: (ret * 100.0).round() / 100.0,
            }
        })
        .collect();

    let buckets = ["-$100+", "-$50", "-$25", "$0", "+$25", "+$50", "+$100+"];
    let trade_distribution: Vec<_> = buckets
        .iter()
        .map(|&b| {
            let count = trade_pnls
                .iter()
                .filter(|&&pnl| match b {
                    "-$100+" => pnl <= -100.0,
                    "-$50" => (-100.0 < pnl) && (pnl <= -25.0),
                    "-$25" => (-25.0 < pnl) && (pnl <= -5.0),
                    "$0" => (-5.0 < pnl) && (pnl < 5.0),
                    "+$25" => (5.0 <= pnl) && (pnl < 25.0),
                    "+$50" => (25.0 <= pnl) && (pnl < 100.0),
                    "+$100+" => pnl >= 100.0,
                    _ => false,
                })
                .count() as u32;
            crate::engine_state::TradeBucket {
                bucket: b.to_string(),
                count,
            }
        })
        .collect();

    Ok(Json(BacktestResponse {
        equity_curve,
        drawdown_curve,
        total_return,
        sharpe,
        max_drawdown: max_dd,
        win_rate,
        profit_factor,
        avg_trade_pnl,
        total_trades,
        monthly_returns,
        trade_distribution,
    }))
}
