//! Benchmarks for order book operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nano_core::traits::OrderBook as OrderBookTrait;
use nano_feed::messages::{BookEntry, BookUpdate, EntryType, UpdateAction};
use nano_lob::orderbook::OrderBook;

fn create_book_update(level: usize, price_base: i64) -> BookUpdate {
    let entries: Vec<BookEntry> = (0..level)
        .flat_map(|i| {
            vec![
                BookEntry {
                    price: price_base - (i as i64 * 10),
                    quantity: 100 + (i as i32 * 10),
                    num_orders: 5,
                    price_level: (i + 1) as u8,
                    action: UpdateAction::Change,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: price_base + (i as i64 * 10) + 10,
                    quantity: 100 + (i as i32 * 10),
                    num_orders: 5,
                    price_level: (i + 1) as u8,
                    action: UpdateAction::Change,
                    entry_type: EntryType::Offer,
                },
            ]
        })
        .collect();

    BookUpdate {
        transact_time: 1_000_000_000,
        match_event_indicator: 0x81,
        security_id: 1,
        rpt_seq: 1,
        exponent: -2,
        entries,
    }
}

fn bench_orderbook_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook");

    // Benchmark book update
    group.bench_function("apply_update_10_levels", |b| {
        let mut book = OrderBook::new(1);
        let update = create_book_update(10, 50000);

        b.iter(|| {
            book.apply_book_update(black_box(&update));
        });
    });

    // Benchmark best bid/ask
    group.bench_function("best_bid_ask", |b| {
        let mut book = OrderBook::new(1);
        let update = create_book_update(10, 50000);
        book.apply_book_update(&update);

        b.iter(|| {
            let _ = black_box(book.best_bid());
            let _ = black_box(book.best_ask());
        });
    });

    // Benchmark mid price
    group.bench_function("mid_price", |b| {
        let mut book = OrderBook::new(1);
        let update = create_book_update(10, 50000);
        book.apply_book_update(&update);

        b.iter(|| {
            black_box(book.mid_price());
        });
    });

    // Benchmark total depth calculation
    group.bench_function("total_depth_10_levels", |b| {
        let mut book = OrderBook::new(1);
        let update = create_book_update(10, 50000);
        book.apply_book_update(&update);

        b.iter(|| {
            let _ = black_box(book.total_bid_quantity(10));
            let _ = black_box(book.total_ask_quantity(10));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_orderbook_operations);
criterion_main!(benches);

