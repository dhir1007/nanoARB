//! Converts Hyperliquid l2Book messages into nano-feed `BookUpdate` structs
//! that the existing `nano-lob::OrderBook` can consume directly.
//!
//! # Price representation
//! nano-core stores prices as raw i64 ticks where `as_f64() = raw / 100`.
//! So $99,500.50  →  raw = 9_950_050.
//!
//! HL sends prices as strings like "99500.5". We parse to f64 and multiply
//! by 100, rounding to the nearest integer tick.
//!
//! # Quantity representation
//! HL sends sizes as strings like "0.500" (BTC). We multiply by 1_000 to
//! get integer lots of 0.001 BTC. So 0.500 BTC → 500 lots.
//! The paper-trader always knows the lot_size so PnL is computed correctly.

use nano_feed::messages::{BookEntry, BookUpdate, EntryType, UpdateAction};

use crate::feed::HlBookData;

/// Multiplier from HL price string → raw ticks (matches Price::as_f64() = raw/100)
const PRICE_SCALE: f64 = 100.0;

/// Multiplier from HL size string → integer quantity lots
/// 1 lot = 0.001 base asset.  So 0.500 BTC = 500 lots.
const QTY_SCALE: f64 = 1_000.0;

/// Size of one lot in base asset (used for PnL calculation)
pub const LOT_SIZE: f64 = 0.001;

/// Converts a Hyperliquid l2Book snapshot into a `BookUpdate` that
/// `nano-lob::OrderBook::apply_book_update()` understands.
///
/// Hyperliquid sends a full snapshot on every block (~500ms), NOT incremental
/// diffs. So every message is a full replace, which maps to `UpdateAction::Overlay`.
pub fn hl_book_to_update(data: &HlBookData, sequence: u32) -> BookUpdate {
    let mut entries = Vec::with_capacity(
        data.levels.0.len() + data.levels.1.len(),
    );

    // Bids
    for (i, level) in data.levels.0.iter().enumerate() {
        let price_ticks = parse_price_ticks(&level.px);
        let qty_lots   = parse_qty_lots(&level.sz);

        if price_ticks == 0 || qty_lots == 0 {
            continue;
        }

        entries.push(BookEntry {
            price:       price_ticks,
            quantity:    qty_lots,
            num_orders:  level.n as i32,
            price_level: (i + 1) as u8,
            action:      UpdateAction::Overlay,
            entry_type:  EntryType::Bid,
        });
    }

    // Asks
    for (i, level) in data.levels.1.iter().enumerate() {
        let price_ticks = parse_price_ticks(&level.px);
        let qty_lots   = parse_qty_lots(&level.sz);

        if price_ticks == 0 || qty_lots == 0 {
            continue;
        }

        entries.push(BookEntry {
            price:       price_ticks,
            quantity:    qty_lots,
            num_orders:  level.n as i32,
            price_level: (i + 1) as u8,
            action:      UpdateAction::Overlay,
            entry_type:  EntryType::Offer,
        });
    }

    // Clear stale levels not in this snapshot by adding explicit deletes.
    // Because HL sends a full 20-level snapshot each time, the simplest
    // approach is to clear the book first, then overlay all levels.
    // We signal "clear then overlay" by prepending DeleteThru entries at
    // extreme prices — this removes everything before we add new levels.
    let mut full_entries = Vec::with_capacity(entries.len() + 2);

    // Delete all existing bids (price 0 means "through 0" = everything)
    full_entries.push(BookEntry {
        price:       0,
        quantity:    0,
        num_orders:  0,
        price_level: 0,
        action:      UpdateAction::DeleteThru,
        entry_type:  EntryType::Bid,
    });

    // Delete all existing asks (i64::MAX means "from max" = everything)
    full_entries.push(BookEntry {
        price:       i64::MAX,
        quantity:    0,
        num_orders:  0,
        price_level: 0,
        action:      UpdateAction::DeleteFrom,
        entry_type:  EntryType::Offer,
    });

    full_entries.extend(entries);

    BookUpdate {
        // HL time is in milliseconds; nano expects nanoseconds
        transact_time:        data.time * 1_000_000,
        match_event_indicator: 0x81, // last message + end of event
        security_id:           1,    // we use 1 for the single HL instrument
        rpt_seq:               sequence,
        exponent:             -2,    // matches Price::as_f64() = raw / 100
        entries:               full_entries,
    }
}

/// Parse a price string like "99500.5" into raw ticks (i64).
fn parse_price_ticks(s: &str) -> i64 {
    s.parse::<f64>()
        .map(|f| (f * PRICE_SCALE).round() as i64)
        .unwrap_or(0)
}

/// Parse a size string like "0.500" into integer lots (i32).
fn parse_qty_lots(s: &str) -> i32 {
    s.parse::<f64>()
        .map(|f| (f * QTY_SCALE).round() as i32)
        .unwrap_or(0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feed::HlLevel;

    fn make_level(px: &str, sz: &str, n: u32) -> HlLevel {
        HlLevel { px: px.into(), sz: sz.into(), n }
    }

    #[test]
    fn test_price_parsing() {
        assert_eq!(parse_price_ticks("99500.5"),  9_950_050);
        assert_eq!(parse_price_ticks("100000.0"), 10_000_000);
        assert_eq!(parse_price_ticks("0.01"),      1);
    }

    #[test]
    fn test_qty_parsing() {
        assert_eq!(parse_qty_lots("0.500"), 500);
        assert_eq!(parse_qty_lots("1.000"), 1000);
        assert_eq!(parse_qty_lots("0.001"), 1);
    }

    #[test]
    fn test_full_conversion() {
        let data = HlBookData {
            coin:   "BTC".into(),
            levels: (
                vec![make_level("99500.5", "0.500", 3)],
                vec![make_level("99501.0", "0.300", 1)],
            ),
            time: 1_700_000_000_000,
        };

        let update = hl_book_to_update(&data, 1);

        // 2 delete entries + 1 bid + 1 ask
        assert_eq!(update.entries.len(), 4);
        assert_eq!(update.transact_time, 1_700_000_000_000 * 1_000_000);

        // Find bid entry
        let bid = update.entries.iter()
            .find(|e| matches!(e.entry_type, EntryType::Bid) && e.action == UpdateAction::Overlay)
            .unwrap();
        assert_eq!(bid.price, 9_950_050);
        assert_eq!(bid.quantity, 500);

        // Find ask entry
        let ask = update.entries.iter()
            .find(|e| matches!(e.entry_type, EntryType::Offer) && e.action == UpdateAction::Overlay)
            .unwrap();
        assert_eq!(ask.price, 9_950_100);
        assert_eq!(ask.quantity, 300);
    }
}