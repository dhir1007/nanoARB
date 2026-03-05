//! Adapter for converting Databento DBN records into NanoARB MdpMessage types.
//!
//! Converts `Mbp10Msg` (Market-by-Price 10-level) records from the `dbn` crate
//! into the internal `BookUpdate` / `Snapshot` messages that the order book engine
//! consumes.

use std::io::Read;
use std::path::Path;

use dbn::decode::dbn::Decoder;
use dbn::decode::DecodeRecord;
use dbn::record::Mbp10Msg;
use dbn::UNDEF_PRICE;

use crate::messages::{
    BookEntry, BookUpdate, EntryType, MdpMessage, Snapshot, SnapshotEntry, UpdateAction,
};

/// Databento price values are fixed-point with 1e-9 precision.
/// Our internal prices use 1e-2 (cents). Divide by 1e7 to convert.
const DBN_PRICE_TO_INTERNAL: i64 = 10_000_000;

/// Convert a Databento `Mbp10Msg` into an `MdpMessage::BookUpdate`.
///
/// Maps the top-of-book event plus all 10 depth levels into `BookEntry` items.
#[must_use]
pub fn mbp10_to_book_update(msg: &Mbp10Msg, seq: u32) -> MdpMessage {
    let mut entries = Vec::with_capacity(20);

    for (i, level) in msg.levels.iter().enumerate() {
        if level.bid_px != UNDEF_PRICE && level.bid_sz > 0 {
            entries.push(BookEntry {
                price: level.bid_px / DBN_PRICE_TO_INTERNAL,
                quantity: level.bid_sz as i32,
                num_orders: level.bid_ct as i32,
                price_level: (i + 1) as u8,
                action: UpdateAction::Change,
                entry_type: EntryType::Bid,
            });
        }
        if level.ask_px != UNDEF_PRICE && level.ask_sz > 0 {
            entries.push(BookEntry {
                price: level.ask_px / DBN_PRICE_TO_INTERNAL,
                quantity: level.ask_sz as i32,
                num_orders: level.ask_ct as i32,
                price_level: (i + 1) as u8,
                action: UpdateAction::Change,
                entry_type: EntryType::Offer,
            });
        }
    }

    MdpMessage::BookUpdate(BookUpdate {
        transact_time: msg.hd.ts_event,
        match_event_indicator: 0x81,
        security_id: msg.hd.instrument_id as i32,
        rpt_seq: seq,
        exponent: -2,
        entries,
    })
}

/// Convert a Databento `Mbp10Msg` into an `MdpMessage::Snapshot`.
///
/// Used for initial book state when starting replay.
#[must_use]
pub fn mbp10_to_snapshot(msg: &Mbp10Msg, seq: u32) -> MdpMessage {
    let mut entries = Vec::with_capacity(20);

    for (i, level) in msg.levels.iter().enumerate() {
        if level.bid_px != UNDEF_PRICE && level.bid_sz > 0 {
            entries.push(SnapshotEntry {
                price: level.bid_px / DBN_PRICE_TO_INTERNAL,
                quantity: level.bid_sz as i32,
                num_orders: level.bid_ct as i32,
                price_level: (i + 1) as u8,
                entry_type: EntryType::Bid,
            });
        }
        if level.ask_px != UNDEF_PRICE && level.ask_sz > 0 {
            entries.push(SnapshotEntry {
                price: level.ask_px / DBN_PRICE_TO_INTERNAL,
                quantity: level.ask_sz as i32,
                num_orders: level.ask_ct as i32,
                price_level: (i + 1) as u8,
                entry_type: EntryType::Offer,
            });
        }
    }

    MdpMessage::Snapshot(Snapshot {
        last_update_time: msg.hd.ts_event,
        security_id: msg.hd.instrument_id as i32,
        rpt_seq: seq,
        exponent: -2,
        entries,
    })
}

/// Reads a DBN file and yields `MdpMessage` values.
///
/// Reads `Mbp10Msg` records from a `.dbn` or `.dbn.zst` file and converts
/// each one into an `MdpMessage::BookUpdate`. The first record is emitted
/// as a `Snapshot` to initialize the order book.
pub struct DbnFileReader {
    decoder: Decoder<Box<dyn Read + Send>>,
    seq: u32,
    first_emitted: bool,
}

impl DbnFileReader {
    /// Open a `.dbn` file for reading.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let boxed: Box<dyn Read + Send> = Box::new(std::io::BufReader::new(file));
        let decoder = Decoder::new(boxed)?;
        Ok(Self {
            decoder,
            seq: 0,
            first_emitted: false,
        })
    }

    /// Open a `.dbn.zst` (zstd-compressed) file for reading.
    pub fn open_zstd(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let buf = std::io::BufReader::new(file);
        let zstd_reader = zstd::stream::read::Decoder::new(buf)?;
        let boxed: Box<dyn Read + Send> = Box::new(zstd_reader);
        let decoder = Decoder::new(boxed)?;
        Ok(Self {
            decoder,
            seq: 0,
            first_emitted: false,
        })
    }

    /// Read the next MdpMessage from the file.
    ///
    /// Returns `Ok(None)` at end of file.
    pub fn next_message(&mut self) -> anyhow::Result<Option<MdpMessage>> {
        let record: Option<&Mbp10Msg> = self.decoder.decode_record()?;

        match record {
            Some(msg) => {
                self.seq += 1;
                let mdp_msg = if !self.first_emitted {
                    self.first_emitted = true;
                    mbp10_to_snapshot(msg, self.seq)
                } else {
                    mbp10_to_book_update(msg, self.seq)
                };
                Ok(Some(mdp_msg))
            }
            None => Ok(None),
        }
    }
}
