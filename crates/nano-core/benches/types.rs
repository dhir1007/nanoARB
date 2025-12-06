//! Benchmarks for core types.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nano_core::types::{Price, Quantity, Side, Timestamp};

fn bench_price_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("price");

    group.bench_function("from_raw", |b| {
        b.iter(|| Price::from_raw(black_box(50000)));
    });

    group.bench_function("from_f64", |b| {
        b.iter(|| Price::from_f64(black_box(500.25), 0.01));
    });

    let p1 = Price::from_raw(50000);
    let p2 = Price::from_raw(25000);

    group.bench_function("add", |b| {
        b.iter(|| black_box(p1) + black_box(p2));
    });

    group.bench_function("sub", |b| {
        b.iter(|| black_box(p1) - black_box(p2));
    });

    group.bench_function("cmp", |b| {
        b.iter(|| black_box(p1) > black_box(p2));
    });

    group.bench_function("as_f64", |b| {
        b.iter(|| black_box(p1).as_f64());
    });

    group.finish();
}

fn bench_quantity_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantity");

    group.bench_function("new", |b| {
        b.iter(|| Quantity::new(black_box(100)));
    });

    let q1 = Quantity::new(100);
    let q2 = Quantity::new(50);

    group.bench_function("add", |b| {
        b.iter(|| black_box(q1) + black_box(q2));
    });

    group.bench_function("saturating_sub", |b| {
        b.iter(|| black_box(q1).saturating_sub(black_box(q2)));
    });

    group.finish();
}

fn bench_timestamp_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("timestamp");

    group.bench_function("now", |b| {
        b.iter(|| Timestamp::now());
    });

    group.bench_function("from_nanos", |b| {
        b.iter(|| Timestamp::from_nanos(black_box(1_000_000_000)));
    });

    let t1 = Timestamp::from_nanos(1_000_000_000);
    let t2 = Timestamp::from_nanos(500_000_000);

    group.bench_function("duration_since", |b| {
        b.iter(|| black_box(t1).duration_since(black_box(t2)));
    });

    group.bench_function("add_nanos", |b| {
        b.iter(|| black_box(t1).add_nanos(black_box(1000)));
    });

    group.finish();
}

fn bench_side_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("side");

    group.bench_function("opposite", |b| {
        b.iter(|| black_box(Side::Buy).opposite());
    });

    group.bench_function("sign", |b| {
        b.iter(|| black_box(Side::Buy).sign());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_price_operations,
    bench_quantity_operations,
    bench_timestamp_operations,
    bench_side_operations
);
criterion_main!(benches);

