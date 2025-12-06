//! Benchmarks for the MDP 3.0 parser.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nano_feed::synthetic::{SyntheticConfig, SyntheticGenerator};

fn bench_synthetic_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic");

    let config = SyntheticConfig::es_futures();

    group.bench_function("generate_event", |b| {
        let mut gen = SyntheticGenerator::new(config.clone());
        b.iter(|| black_box(gen.next_event()));
    });

    group.bench_function("generate_100_events", |b| {
        b.iter(|| {
            let mut gen = SyntheticGenerator::new(config.clone());
            black_box(gen.generate_n(100))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_synthetic_generation);
criterion_main!(benches);
