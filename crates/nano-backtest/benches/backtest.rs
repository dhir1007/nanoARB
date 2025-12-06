//! Benchmarks for backtesting engine.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nano_backtest::config::BacktestConfig;
use nano_backtest::engine::BacktestEngine;
use nano_backtest::events::EventType;
use nano_core::types::Timestamp;

fn bench_event_scheduling(c: &mut Criterion) {
    let mut group = c.benchmark_group("backtest");

    group.bench_function("schedule_event", |b| {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);

        b.iter(|| {
            engine.schedule_event(
                black_box(Timestamp::from_nanos(1_000_000)),
                EventType::EndOfData,
            );
        });
    });

    group.finish();
}

criterion_group!(benches, bench_event_scheduling);
criterion_main!(benches);
