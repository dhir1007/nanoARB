//! Benchmarks for feature extraction.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nano_feed::messages::{BookEntry, BookUpdate, EntryType, UpdateAction};
use nano_lob::features::LobFeatureExtractor;
use nano_lob::orderbook::OrderBook;
use nano_lob::snapshot::{LobSnapshot, SnapshotRingBuffer};

fn create_test_book(price_base: i64) -> OrderBook {
    let mut book = OrderBook::new(1);

    let entries: Vec<BookEntry> = (0..10)
        .flat_map(|i| {
            vec![
                BookEntry {
                    price: price_base - (i as i64 * 10),
                    quantity: 100 + (i * 10),
                    num_orders: 5,
                    price_level: (i + 1) as u8,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: price_base + (i as i64 * 10) + 10,
                    quantity: 100 + (i * 10),
                    num_orders: 5,
                    price_level: (i + 1) as u8,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
            ]
        })
        .collect();

    let update = BookUpdate {
        transact_time: 1_000_000_000,
        match_event_indicator: 0x81,
        security_id: 1,
        rpt_seq: 1,
        exponent: -2,
        entries,
    };

    book.apply_book_update(&update);
    book
}

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("features");

    let book = create_test_book(50000);
    let extractor = LobFeatureExtractor::new();

    group.bench_function("extract_all_features", |b| {
        b.iter(|| {
            black_box(extractor.extract(&book));
        });
    });

    group.bench_function("microprice", |b| {
        b.iter(|| {
            black_box(extractor.microprice(&book));
        });
    });

    group.bench_function("book_imbalance_10_levels", |b| {
        b.iter(|| {
            black_box(extractor.book_imbalance(&book, 10));
        });
    });

    group.bench_function("to_array", |b| {
        b.iter(|| {
            black_box(extractor.to_array(&book));
        });
    });

    group.finish();
}

fn bench_snapshot_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot");

    let book = create_test_book(50000);

    group.bench_function("create_snapshot", |b| {
        b.iter(|| {
            black_box(LobSnapshot::from_book(&book));
        });
    });

    // Fill buffer first
    let mut buffer = SnapshotRingBuffer::new(200);
    for i in 0..200 {
        let book = create_test_book(50000 + i * 10);
        buffer.push_book(&book);
    }

    group.bench_function("ring_buffer_push", |b| {
        let snapshot = LobSnapshot::from_book(&book);
        b.iter(|| {
            buffer.push(black_box(snapshot.clone()));
        });
    });

    group.bench_function("to_tensor_100", |b| {
        b.iter(|| {
            black_box(buffer.to_tensor(100));
        });
    });

    group.bench_function("get_returns_50", |b| {
        b.iter(|| {
            black_box(buffer.get_returns(50));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_feature_extraction, bench_snapshot_operations);
criterion_main!(benches);
