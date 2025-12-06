//! Benchmarks for model inference.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nano_model::preprocessing::FeaturePreprocessor;

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    let prep = FeaturePreprocessor::default();

    group.bench_function("transform_shape", |b| {
        b.iter(|| {
            black_box(prep.input_shape());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_preprocessing);
criterion_main!(benches);
