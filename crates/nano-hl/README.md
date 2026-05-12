# nanoARB

<div align="center">

**High-frequency market-making engine in Rust, live on Hyperliquid**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/dhir1007/nanoARB/actions/workflows/ci.yml/badge.svg)](https://github.com/dhir1007/nanoARB/actions)

</div>

---

## What this is

nanoARB is a market-making engine written in Rust. It connects to
[Hyperliquid](https://hyperliquid.xyz)'s live perpetual futures order book,
reconstructs a full limit order book in memory, extracts microstructure
signals, and runs a configurable market-making strategy — all in a single
async Rust process.

The engine currently runs in **paper-trading mode**: it reads real live market
data and simulates order placement without submitting real orders to the
exchange.

---

## Architecture

```
Hyperliquid WebSocket (live L2 data)
          │
          ▼
     nano-hl/feed          — async WebSocket client, auto-reconnects
          │
          ▼
     nano-hl/converter     — parses HL JSON → internal BookUpdate types
          │
          ▼
     nano-lob/orderbook    — BTreeMap-based LOB, O(log n) updates
          │
          ▼
     nano-lob/features     — microprice, OFI, VPIN, book imbalance
          │
          ▼
     nano-strategy/        — MarketMakerStrategy, inventory skew, quote mgmt
          │
          ▼
     nano-hl/paper_trader  — simulated fills, PnL tracking, Sharpe
          │
          ▼
     Prometheus → Grafana  — live dashboard: PnL, position, latency
```

## Crate structure

| Crate | What it does |
|---|---|
| `nano-core` | Shared types: `Price` (fixed-point i64), `Quantity`, `Order`, `Side`, traits |
| `nano-feed` | CME MDP 3.0 parser + synthetic data generator |
| `nano-lob` | Order book reconstruction, microprice, OFI, VPIN, book imbalance |
| `nano-model` | ONNX inference wrapper for Mamba SSM (trained on synthetic data) |
| `nano-backtest` | Event-driven backtesting engine with latency + fill simulation |
| `nano-strategy` | `MarketMakerStrategy` with inventory skew and quote management |
| `nano-gateway` | HTTP server, Prometheus metrics, config loading |
| `nano-hl` | Hyperliquid adapter: WebSocket feed, converter, paper trader, metrics |

---

## Benchmark results

Run on Apple M-series (your machine). All numbers are real — run
`cargo bench --workspace` to reproduce.

| Component | Operation | Median latency |
|---|---|---|
| `nano-core` | Price arithmetic (`add`, `sub`) | **0.67 ns** |
| `nano-core` | Price from f64 | **0.40 ns** |
| `nano-core` | Timestamp now() | **20.9 ns** |
| `nano-lob` | Best bid/ask lookup | **1.3 ns** |
| `nano-lob` | Mid price calculation | **1.9 ns** |
| `nano-lob` | Microprice extraction | **2.7 ns** |
| `nano-lob` | Full feature extraction | **22 ns** |
| `nano-lob` | Snapshot → tensor (100 steps) | **2.1 µs** |
| `nano-backtest` | Schedule event | **10.4 ns** |

Key design choice: prices are stored as `i64` ticks rather than `f64`. This
eliminates floating-point rounding errors in order matching and PnL
calculation, and makes arithmetic ~1.5x faster.

---

## Quick start

### Prerequisites

- Rust 1.75+
- Docker Desktop (for Grafana monitoring)

### Run the paper trader

```bash
git clone https://github.com/dhir1007/nanoARB.git
cd nanoARB

# Paper trade BTC-PERP on Hyperliquid mainnet (read-only, no real orders)
cargo run -p nano-hl --release -- --coin BTC --mainnet
```

### Watch it on Grafana

```bash
cd docker
docker-compose -f docker-compose-monitoring.yml up -d
# Open http://localhost:3000  (login: admin / admin)
```

You'll see live PnL, position, order rate, and tick processing latency
updating every second from real Hyperliquid market data.

### Run all tests

```bash
cargo test --workspace
```

### Run benchmarks

```bash
cargo bench --workspace
```

---

## Key implementation details

### Fixed-point price arithmetic

```rust
// From nano-core/src/types/price.rs
pub struct Price(i64);  // raw ticks, never f64

// $81,500.25 stored as 8_150_025 (multiplied by 100)
let price = Price::from_f64(81500.25, 0.01);

// All math stays in integer domain — no rounding errors
let spread = ask - bid;  // exact, deterministic
```

### Order book feature extraction

```rust
// From nano-lob/src/features.rs

// Microprice: accounts for bid/ask depth imbalance
// Better short-term price predictor than plain mid
microprice = (bid_price × ask_qty + ask_price × bid_qty) / (bid_qty + ask_qty)

// Book imbalance: -1 (all asks) to +1 (all bids)
imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)

// OFI: did the bid grow or shrink since last tick?
// Positive = buying pressure, negative = selling pressure
ofi = Δbid_qty - Δask_qty
```

### Market-making strategy

The strategy posts limit orders on both sides of the book and adjusts quote
prices based on current inventory:

```rust
// From nano-strategy/src/market_maker.rs

// If long too much inventory → lower both quotes to attract sellers
let inv_ratio = position / max_inventory;  // -1.0 to +1.0
let skew = inv_ratio × skew_factor × half_spread;

bid = fair_value - half_spread - skew
ask = fair_value + half_spread - skew
```

### Hyperliquid WebSocket feed

```rust
// nano-hl connects to wss://api.hyperliquid.xyz/ws
// Subscribes to l2Book, handles reconnects + 20s heartbeats automatically
// Converts HL JSON → nano-feed BookUpdate → nano-lob OrderBook

// HL sends prices as strings ("81500.5") to avoid float precision issues
// We parse to i64 ticks: "81500.5" → 8_150_050
```

---

## What's not yet built

Being explicit about current limitations:

- **No live order execution** — paper trading only. Submitting real orders
  requires ECDSA signing with an HL API wallet (ethers-rs).
- **ML model not connected to live feed** — the Mamba SSM in `nano-model`
  was trained on synthetic data and is not used in the live strategy. Needs
  retraining on real HL tick data.
- **No VPIN-based quote inhibition** — the VPIN calculator exists in
  `nano-lob/features.rs` but the strategy doesn't use it yet to pause
  quoting during high-toxicity periods.

---

## What's real and working

- Live WebSocket connection to Hyperliquid mainnet
- Full LOB reconstruction from L2 snapshots
- Microprice, book imbalance, OFI, VPIN calculation on every tick
- MarketMakerStrategy with inventory skew running against live data
- Paper fill simulation with position and PnL tracking
- Prometheus metrics exposed on `:9090/metrics`
- Grafana dashboard showing live PnL, position, latency, event rate
- Full test suite passing (`cargo test --workspace`)
- Real benchmark numbers (see table above)

---

## Disclaimer

This software is for educational and research purposes only. Not financial
advice. Paper trading results do not predict live trading performance.

---

## License

MIT
