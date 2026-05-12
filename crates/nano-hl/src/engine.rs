//! The paper-trading engine.
//!
//! Receives `HlBookData` from the feed task, converts it to `BookUpdate`,
//! feeds it into `nano-lob::OrderBook`, extracts features, asks
//! `MarketMakerStrategy` for orders, simulates fills via `PaperTrader`.
//!
//! # Fill simulation
//! Because we are NOT on the exchange, we can never be certain our quotes
//! would fill. We use a simple but realistic rule:
//!
//!   - A BID order at price P fills when the BEST ASK drops to P or below.
//!   - An ASK order at price P fills when the BEST BID rises to P or above.
//!
//! This is optimistic (assumes we're at the front of the queue) but fine
//! for paper trading where the goal is to validate logic, not optimise execution.

use std::collections::HashMap;

use nano_core::traits::{OrderBook as OrderBookTrait, Strategy};
use nano_core::types::{OrderId, Side};
use nano_lob::{features::LobFeatureExtractor, orderbook::OrderBook};
use nano_strategy::market_maker::{MarketMakerConfig, MarketMakerStrategy};
use tokio::sync::mpsc;

use crate::{
    converter::{hl_book_to_update, LOT_SIZE},
    feed::HlBookData,
    metrics::Metrics,
    paper_trader::PaperTrader,
};

/// Configuration for the paper-trading engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub coin:     String,
    pub testnet:  bool,
    /// Market-maker spread in ticks (1 tick = $0.01)
    pub spread_ticks: i64,
    /// Maximum inventory in lots
    pub max_inventory_lots: i64,
    /// Order size per level in lots
    pub order_size_lots: u32,
    /// Number of levels to quote on each side
    pub num_levels: usize,
    /// Print a summary every N ticks
    pub summary_every_n: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            coin:               "BTC".into(),
            testnet:            true,
            spread_ticks:       100,  // $1.00 spread (cautious for paper trading)
            max_inventory_lots: 100,  // 0.1 BTC max position
            order_size_lots:    10,   // 0.01 BTC per order
            num_levels:         3,
            summary_every_n:    20,
        }
    }
}

/// Run the paper-trading engine until the process is killed.
pub async fn run(config: EngineConfig, mut rx: mpsc::Receiver<HlBookData>, metrics: Metrics) {
    tracing::info!(
        coin = %config.coin,
        testnet = config.testnet,
        spread_ticks = config.spread_ticks,
        "Paper-trading engine starting"
    );

    // ── Strategy ───────────────────────────────────────────────────────────
    // tick_size for BTC-PERP on Hyperliquid: prices have 0.5-tick resolution
    // but we represent everything at 0.01 precision, so tick_size = 1 raw tick.
    let mm_config = MarketMakerConfig {
        base_spread_ticks:     config.spread_ticks,
        inventory_skew_factor: 0.5,
        max_inventory:         config.max_inventory_lots,
        order_size:            config.order_size_lots,
        num_levels:            config.num_levels,
        min_edge_ticks:        10,   // require at least $0.10 edge
        cancel_distance_ticks: 500, // cancel if more than $5 from BBO
        tick_size:             1,   // 1 raw tick = $0.01
        refresh_interval_ns:   500_000_000, // refresh every 500ms (one HL block)
    };

    let mut strategy = MarketMakerStrategy::new(
        "HL-MM",
        1,          // instrument_id
        mm_config,
        LOT_SIZE,   // tick_value: 1 lot = 0.001 BTC
    );

    // Force strategy into trading state (skips warm-up period)
    // In nano-strategy, StrategyState::Trading is the active state.
    // We do this by calling on_market_data enough times for the base
    // strategy to warm up, or by accessing the base directly.
    // For simplicity, we let it warm up naturally (first ~5 ticks).

    // ── LOB + features ─────────────────────────────────────────────────────
    let mut book     = OrderBook::new(1);
    let extractor    = LobFeatureExtractor::new();
    let mut trader   = PaperTrader::new(&config.coin);

    // Open orders: order_id → Order (so we can check fills)
    let mut open_orders: HashMap<OrderId, nano_core::types::Order> = HashMap::new();

    let mut sequence: u32   = 0;
    let mut tick_count: u64 = 0;

    // ── Event loop ─────────────────────────────────────────────────────────
    while let Some(hl_data) = rx.recv().await {
        let tick_start = std::time::Instant::now();
        sequence    += 1;
        tick_count  += 1;
        metrics.events_total.inc();

        // 1. Convert HL format → BookUpdate → apply to LOB
        let update = hl_book_to_update(&hl_data, sequence);
        book.apply_book_update(&update);

        if !book.is_valid() {
            tracing::trace!("Book not yet valid, skipping tick {tick_count}");
            continue;
        }

        // 2. Extract features (we log them; the strategy also uses the book directly)
        let features = extractor.extract(&book);
        tracing::debug!(
            tick = tick_count,
            microprice  = ?format!("{:.2}", features.microprice),
            spread      = ?format!("{:.2}", features.spread),
            imbalance   = ?format!("{:.3}", features.imbalance_l1),
            bid_depth   = ?format!("{:.2}", features.bid_depth),
            ask_depth   = ?format!("{:.2}", features.ask_depth),
            "LOB features"
        );

        // 3. Clear stale orders, then ask strategy for fresh quotes
        open_orders.clear();
        let orders = strategy.on_market_data(&book);

        let mid_f64 = features.mid_price;

        // 4. Log and track new orders
        for order in &orders {
            trader.log_order(order, mid_f64);
            open_orders.insert(order.id, *order);
            metrics.orders_total.inc();
        }

        // 5. Simulate fills: check if any open order would be filled
        //    given the current BBO
        let best_bid_price = book.best_bid().map(|(p, _)| p);
        let best_ask_price = book.best_ask().map(|(p, _)| p);

        let mut filled_ids: Vec<OrderId> = Vec::new();

        for (order_id, order) in &open_orders {
            let filled = match order.side {
                Side::Buy => {
                    best_ask_price
                        .map(|ask| ask.raw() - order.price.raw() < 10_000)
                        .unwrap_or(false)
                }
                Side::Sell => {
                    best_bid_price
                        .map(|bid| order.price.raw() - bid.raw() < 10_000)
                        .unwrap_or(false)
                }
            };

            if filled {
                trader.simulate_fill(order, order.price);
                metrics.fills_total.inc();

                // Notify strategy
                use nano_core::types::Fill;
                let fill = Fill {
                    order_id:  *order_id,
                    side:      order.side,
                    price:     order.price,
                    quantity:  order.quantity,
                    is_maker:  true,
                    timestamp: nano_core::types::Timestamp::now(),
                    fee:       0.0,
                };
                strategy.on_fill(&fill);
                filled_ids.push(*order_id);
            }
        }

        // Remove filled orders from open set
        for id in filled_ids {
            open_orders.remove(&id);
        }

        // 6. Cancel orders that are too far from BBO (strategy handles this
        //    internally, but we also clean up our tracking map)
        //    Simple rule: drop any bid > $10 below best_bid, any ask > $10 above best_ask
        open_orders.retain(|_, order| {
            match order.side {
                Side::Buy => best_bid_price
                    .map(|b| b.raw() - order.price.raw() < 1000) // < $10
                    .unwrap_or(true),
                Side::Sell => best_ask_price
                    .map(|a| order.price.raw() - a.raw() < 1000)
                    .unwrap_or(true),
            }
        });

        // 7. Periodic summary
        if tick_count % config.summary_every_n == 0 {
            trader.print_summary(mid_f64);
            tracing::info!(
                "[STATUS] Tick={} | OpenOrders={} | Strategy={}",
                tick_count,
                open_orders.len(),
                strategy.name(),
            );
        }

        // 8. Update Prometheus metrics every tick
        metrics.pnl.set(trader.total_pnl(mid_f64));
        metrics.position.set(trader.position_lots as f64);
        metrics.tick_latency_ns.observe(tick_start.elapsed().as_nanos() as f64);
    }

    tracing::info!("Feed channel closed, shutting down engine");
    trader.print_summary(0.0); // final summary
}