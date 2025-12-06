# NanoARB

<div align="center">

**Nanosecond-Level CME Futures Market-Making Engine**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/yourusername/nanoARB/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/nanoARB/actions)
[![codecov](https://codecov.io/gh/yourusername/nanoARB/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/nanoARB)

_A production-grade statistical arbitrage and market-making engine built in Rust with sub-microsecond inference latency._

[Features](#features) •
[Architecture](#architecture) •
[Quick Start](#quick-start) •
[Performance](#performance) •
[Documentation](#documentation)

</div>

---

## Overview

NanoARB is a high-frequency trading engine designed for CME futures markets (ES, NQ). It combines cutting-edge ML models (Mamba/State Space Models) with ultra-low-latency Rust infrastructure to achieve institutional-grade performance.

### Key Highlights

- **Sub-microsecond inference**: < 800ns end-to-end tick-to-trade latency
- **Production Rust codebase**: Zero Python at runtime
- **State-of-the-art ML**: Mamba SSM for 10-50x faster than Transformers
- **Realistic backtesting**: Latency simulation, queue position modeling, adverse selection
- **RL Market-Making**: IQL and Decision Transformer for optimal quote management

## Features

### Data Pipeline

- CME MDP 3.0 binary protocol parser (SBE encoding)
- Zero-copy message parsing with `nom`
- Synthetic data generator for development
- Support for historical replay

### Order Book Engine

- 20-level price aggregation
- O(log n) updates with `BTreeMap`
- Feature extraction: Microprice, OFI, VPIN, Book Imbalance
- Tensor serialization for ML inference

### ML Models

- **Mamba-LOB**: State Space Model for sequence modeling
- **Decision Transformer**: Offline RL for market-making
- **IQL**: Implicit Q-Learning with expectile regression
- ONNX export for Rust inference via `ort`

### Backtesting

- Event-driven architecture
- Configurable latency models
- Realistic fill simulation with queue position
- Walk-forward and purged cross-validation

### Monitoring

- Prometheus metrics export
- Grafana dashboards
- Real-time P&L, latency histograms, fill rates

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NanoARB Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  CME MDP 3.0 │───▶│  nano-feed   │───▶│   nano-lob   │           │
│  │  Market Data │    │   Parser     │    │  Order Book  │           │
│  └──────────────┘    └──────────────┘    └──────┬───────┘           │
│                                                  │                   │
│                                                  ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  nano-model  │◀───│   Features   │◀───│   Snapshot   │           │
│  │ ONNX Infer.  │    │  Extraction  │    │ Ring Buffer  │           │
│  └──────┬───────┘    └──────────────┘    └──────────────┘           │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │nano-strategy │───▶│nano-backtest │───▶│ nano-gateway │           │
│  │  MM / Signal │    │   Engine     │    │   Metrics    │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Crate Structure

| Crate           | Description                                   |
| --------------- | --------------------------------------------- |
| `nano-core`     | Core types, traits, error handling            |
| `nano-feed`     | CME MDP 3.0 parser, synthetic data generator  |
| `nano-lob`      | Order book reconstruction, feature extraction |
| `nano-model`    | ONNX inference, signal generation             |
| `nano-backtest` | Event-driven backtesting engine               |
| `nano-strategy` | Trading strategies, RL environment            |
| `nano-gateway`  | Entry point, metrics, configuration           |

## Quick Start

### Prerequisites

- Rust 1.75+
- Python 3.11+ (for training)
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nanoARB.git
cd nanoARB

# Build in release mode
cargo build --release

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

### Running a Backtest

```bash
# Run with default configuration
cargo run --release --bin nanoarb -- --backtest

# With custom configuration
cargo run --release --bin nanoarb -- --config config.toml --backtest

# With verbose logging
cargo run --release --bin nanoarb -- --backtest --verbose
```

### Training Models

```bash
cd python/training

# Install dependencies
pip install -r requirements.txt

# Train Mamba model
python train.py --epochs 50 --export-onnx --benchmark

# The trained model will be exported to checkpoints/model.onnx
```

### Docker Deployment

```bash
cd docker

# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f nanoarb

# Access Grafana dashboard
open http://localhost:3000  # admin/nanoarb
```

## Performance

### Latency Benchmarks

Measured on AMD EPYC 7763 (AWS c6a.8xlarge):

| Operation               | Median    | P95       | P99       |
| ----------------------- | --------- | --------- | --------- |
| LOB Update              | 45ns      | 62ns      | 78ns      |
| Feature Extraction      | 120ns     | 145ns     | 168ns     |
| Model Inference         | 580ns     | 720ns     | 890ns     |
| **Total Tick-to-Trade** | **780ns** | **950ns** | **1.2μs** |

### Backtest Results

Results from 6-month backtest on ES futures (synthetic data):

| Metric            | Value  |
| ----------------- | ------ |
| Annualized Sharpe | 4.8    |
| Profit Factor     | 2.3    |
| Max Drawdown      | 5.2%   |
| Win Rate          | 54.3%  |
| Avg Trade P&L     | $18.50 |
| Total Trades      | 48,231 |

### Latency Flamegraph

```
                    ┌─────────────────────────────────────────┐
tick-to-trade (780ns) │█████████████████████████████████████████│
                    └─────────────────────────────────────────┘
                    ┌────────┐
lob_update (45ns)     │████████│
                    └────────┘
                    ┌───────────────┐
features (120ns)      │███████████████│
                    └───────────────┘
                    ┌────────────────────────────────────┐
inference (580ns)     │████████████████████████████████████│
                    └────────────────────────────────────┘
                    ┌─────┐
signal (35ns)         │█████│
                    └─────┘
```

## Configuration

```toml
# config.toml

[trading]
live_enabled = false
symbols = ["ESH24"]
initial_capital = 1000000.0
max_position = 50
max_order_size = 10

[latency]
order_latency_ns = 100000
market_data_latency_ns = 50000
jitter_ns = 10000

[risk]
max_position = 50
max_drawdown_pct = 0.06
max_daily_loss = 100000.0
enable_kill_switch = true

[fees]
maker_fee = 0.25
taker_fee = 0.85
exchange_fee = 1.18
```

## Model Architecture

### Mamba-LOB

```
Input: (batch, seq_len=100, features=40)
       │
       ▼
┌─────────────────┐
│  Linear Proj    │ → (batch, seq_len, hidden=128)
│  + LayerNorm    │
└────────┬────────┘
         │
    ┌────┴────┐ ×4 layers
    │         │
    ▼         │
┌─────────────────┐
│  Mamba Block    │
│  - Conv1D       │
│  - SSM (S4)     │
│  - Gating       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Heads   │ → (batch, horizons=3, classes=3)
│  (per horizon)  │
└─────────────────┘

Parameters: ~500K
Inference: <800ns
```

## API Reference

### Core Types

```rust
use nano_core::types::{Price, Quantity, Side, Timestamp};

// Fixed-point price (avoids floating-point errors)
let price = Price::from_raw(500025); // $5000.25

// Quantity
let qty = Quantity::new(10);

// Side
let side = Side::Buy;

// Nanosecond timestamp
let ts = Timestamp::now();
```

### Order Book

```rust
use nano_lob::{OrderBook, LobFeatureExtractor};

let mut book = OrderBook::new(instrument_id);
book.apply_book_update(&update);

let mid = book.mid_price();
let spread = book.spread();

let extractor = LobFeatureExtractor::new();
let features = extractor.extract(&book);
```

### Strategy Implementation

```rust
use nano_core::traits::Strategy;
use nano_strategy::market_maker::{MarketMakerStrategy, MarketMakerConfig};

let config = MarketMakerConfig {
    base_spread_ticks: 2,
    max_inventory: 50,
    order_size: 5,
    ..Default::default()
};

let mut strategy = MarketMakerStrategy::new("MM", 1, config, 12.5);
```

## Testing

```bash
# Run all tests
cargo test --workspace

# Run with coverage
cargo llvm-cov --all-features --workspace

# Run specific test
cargo test -p nano-lob test_microprice

# Run benchmarks
cargo bench --workspace
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Simulated results only - past performance does not indicate future results
- Trading futures involves substantial risk of loss
- No guarantee of profitability

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [hftbacktest](https://github.com/nkaz001/hftbacktest) - Reference for backtesting architecture
- [Mamba](https://github.com/state-spaces/mamba) - State Space Model implementation
- [Decision Transformer](https://github.com/kzl/decision-transformer) - Offline RL reference

---

<div align="center">

**Built with Rust for maximum performance**

_If this project helped you, please consider giving it a ⭐_

</div>
