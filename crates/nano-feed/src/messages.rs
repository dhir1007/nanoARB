//! CME MDP 3.0 message types.

use nano_core::types::{Price, Quantity, Side, Timestamp};
use serde::{Deserialize, Serialize};

/// MDP 3.0 message header
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MdpHeader {
    /// Message size in bytes (including header)
    pub msg_size: u16,
    /// Block length
    pub block_length: u16,
    /// Template ID (message type)
    pub template_id: u16,
    /// Schema ID
    pub schema_id: u16,
    /// Version
    pub version: u16,
}

impl MdpHeader {
    /// Size of the header in bytes
    pub const SIZE: usize = 12;

    /// Template ID for MDIncrementalRefreshBook
    pub const TEMPLATE_BOOK_UPDATE: u16 = 46;

    /// Template ID for MDIncrementalRefreshTrade
    pub const TEMPLATE_TRADE: u16 = 42;

    /// Template ID for ChannelReset
    pub const TEMPLATE_CHANNEL_RESET: u16 = 4;

    /// Template ID for SecurityStatus
    pub const TEMPLATE_SECURITY_STATUS: u16 = 30;

    /// Template ID for MDIncrementalRefreshOrderBook
    pub const TEMPLATE_ORDER_BOOK: u16 = 47;

    /// Template ID for SnapshotFullRefresh
    pub const TEMPLATE_SNAPSHOT: u16 = 52;
}

/// Update action type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum UpdateAction {
    /// New entry
    New = 0,
    /// Update existing entry
    Change = 1,
    /// Delete entry
    Delete = 2,
    /// Delete through (implied)
    DeleteThru = 3,
    /// Delete from (implied)
    DeleteFrom = 4,
    /// Overlay (for snapshots)
    Overlay = 5,
}

impl TryFrom<u8> for UpdateAction {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(UpdateAction::New),
            1 => Ok(UpdateAction::Change),
            2 => Ok(UpdateAction::Delete),
            3 => Ok(UpdateAction::DeleteThru),
            4 => Ok(UpdateAction::DeleteFrom),
            5 => Ok(UpdateAction::Overlay),
            _ => Err(()),
        }
    }
}

/// Entry type for book updates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EntryType {
    /// Bid entry
    Bid = b'0',
    /// Offer/Ask entry
    Offer = b'1',
    /// Trade entry
    Trade = b'2',
    /// Opening price
    OpeningPrice = b'4',
    /// Settlement price
    SettlementPrice = b'6',
    /// High price
    HighPrice = b'7',
    /// Low price
    LowPrice = b'8',
    /// Volume weighted average price
    Vwap = b'9',
    /// Implied bid
    ImpliedBid = b'E',
    /// Implied offer
    ImpliedOffer = b'F',
    /// Empty book
    EmptyBook = b'J',
}

impl TryFrom<u8> for EntryType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'0' => Ok(EntryType::Bid),
            b'1' => Ok(EntryType::Offer),
            b'2' => Ok(EntryType::Trade),
            b'4' => Ok(EntryType::OpeningPrice),
            b'6' => Ok(EntryType::SettlementPrice),
            b'7' => Ok(EntryType::HighPrice),
            b'8' => Ok(EntryType::LowPrice),
            b'9' => Ok(EntryType::Vwap),
            b'E' => Ok(EntryType::ImpliedBid),
            b'F' => Ok(EntryType::ImpliedOffer),
            b'J' => Ok(EntryType::EmptyBook),
            _ => Err(()),
        }
    }
}

impl EntryType {
    /// Convert entry type to Side
    pub fn to_side(self) -> Option<Side> {
        match self {
            EntryType::Bid | EntryType::ImpliedBid => Some(Side::Buy),
            EntryType::Offer | EntryType::ImpliedOffer => Some(Side::Sell),
            _ => None,
        }
    }

    /// Check if this is an implied entry
    pub fn is_implied(self) -> bool {
        matches!(self, EntryType::ImpliedBid | EntryType::ImpliedOffer)
    }
}

/// A single book update entry
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BookEntry {
    /// Price (in mantissa form, needs exponent adjustment)
    pub price: i64,
    /// Quantity
    pub quantity: i32,
    /// Number of orders at this level
    pub num_orders: i32,
    /// Price level (1 = best, 2 = second best, etc.)
    pub price_level: u8,
    /// Update action
    pub action: UpdateAction,
    /// Entry type (bid/offer)
    pub entry_type: EntryType,
}

impl BookEntry {
    /// Convert to Price type with given exponent
    pub fn to_price(&self, exponent: i8) -> Price {
        let multiplier = 10_i64.pow((-exponent) as u32);
        Price::from_raw(self.price / multiplier)
    }

    /// Convert to Quantity type
    pub fn to_quantity(&self) -> Quantity {
        Quantity::new(self.quantity.max(0) as u32)
    }

    /// Get the side
    pub fn side(&self) -> Option<Side> {
        self.entry_type.to_side()
    }
}

/// MDIncrementalRefreshBook message (Template 46)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BookUpdate {
    /// Transaction time in nanoseconds
    pub transact_time: u64,
    /// Match event indicator
    pub match_event_indicator: u8,
    /// Security ID
    pub security_id: i32,
    /// Sequence number
    pub rpt_seq: u32,
    /// Price exponent
    pub exponent: i8,
    /// Book entries
    pub entries: Vec<BookEntry>,
}

impl BookUpdate {
    /// Get the timestamp
    pub fn timestamp(&self) -> Timestamp {
        Timestamp::from_nanos(self.transact_time as i64)
    }

    /// Check if this is the last message in a batch
    pub fn is_last_in_batch(&self) -> bool {
        (self.match_event_indicator & 0x01) != 0
    }

    /// Check if this indicates end of event
    pub fn is_end_of_event(&self) -> bool {
        (self.match_event_indicator & 0x80) != 0
    }

    /// Get bid entries
    pub fn bid_entries(&self) -> impl Iterator<Item = &BookEntry> {
        self.entries.iter().filter(|e| {
            matches!(e.entry_type, EntryType::Bid | EntryType::ImpliedBid)
        })
    }

    /// Get ask entries
    pub fn ask_entries(&self) -> impl Iterator<Item = &BookEntry> {
        self.entries.iter().filter(|e| {
            matches!(e.entry_type, EntryType::Offer | EntryType::ImpliedOffer)
        })
    }
}

/// Trade message entry
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TradeEntry {
    /// Trade price
    pub price: i64,
    /// Trade quantity
    pub quantity: i32,
    /// Number of orders involved
    pub num_orders: i32,
    /// Aggressor side (0 = buy, 1 = sell)
    pub aggressor_side: u8,
    /// Update action
    pub action: UpdateAction,
}

impl TradeEntry {
    /// Convert to Price type with given exponent
    pub fn to_price(&self, exponent: i8) -> Price {
        let multiplier = 10_i64.pow((-exponent) as u32);
        Price::from_raw(self.price / multiplier)
    }

    /// Convert to Quantity type
    pub fn to_quantity(&self) -> Quantity {
        Quantity::new(self.quantity.max(0) as u32)
    }

    /// Get the aggressor side
    pub fn aggressor(&self) -> Side {
        if self.aggressor_side == 1 {
            Side::Sell
        } else {
            Side::Buy
        }
    }
}

/// MDIncrementalRefreshTrade message (Template 42)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradeUpdate {
    /// Transaction time in nanoseconds
    pub transact_time: u64,
    /// Match event indicator
    pub match_event_indicator: u8,
    /// Security ID
    pub security_id: i32,
    /// Sequence number
    pub rpt_seq: u32,
    /// Price exponent
    pub exponent: i8,
    /// Trade entries
    pub entries: Vec<TradeEntry>,
}

impl TradeUpdate {
    /// Get the timestamp
    pub fn timestamp(&self) -> Timestamp {
        Timestamp::from_nanos(self.transact_time as i64)
    }
}

/// Channel reset message (Template 4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelReset {
    /// Transaction time
    pub transact_time: u64,
    /// Match event indicator
    pub match_event_indicator: u8,
}

/// Security status message (Template 30)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityStatus {
    /// Transaction time
    pub transact_time: u64,
    /// Security ID
    pub security_id: i32,
    /// Security trading status
    pub trading_status: u8,
    /// Halt reason
    pub halt_reason: u8,
    /// Security trading event
    pub trading_event: u8,
}

/// Snapshot full refresh entry
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SnapshotEntry {
    /// Price
    pub price: i64,
    /// Quantity
    pub quantity: i32,
    /// Number of orders
    pub num_orders: i32,
    /// Price level
    pub price_level: u8,
    /// Entry type
    pub entry_type: EntryType,
}

/// Snapshot full refresh message (Template 52)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Last update time
    pub last_update_time: u64,
    /// Security ID
    pub security_id: i32,
    /// Sequence number
    pub rpt_seq: u32,
    /// Price exponent
    pub exponent: i8,
    /// Snapshot entries
    pub entries: Vec<SnapshotEntry>,
}

/// Parsed MDP message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MdpMessage {
    /// Book update message
    BookUpdate(BookUpdate),
    /// Trade update message
    Trade(TradeUpdate),
    /// Channel reset message
    ChannelReset(ChannelReset),
    /// Security status message
    SecurityStatus(SecurityStatus),
    /// Snapshot message
    Snapshot(Snapshot),
    /// Unknown message type
    Unknown {
        /// Template ID
        template_id: u16,
        /// Raw data length
        length: usize,
    },
}

impl MdpMessage {
    /// Get the timestamp if available
    pub fn timestamp(&self) -> Option<Timestamp> {
        match self {
            MdpMessage::BookUpdate(m) => Some(m.timestamp()),
            MdpMessage::Trade(m) => Some(m.timestamp()),
            MdpMessage::ChannelReset(m) => Some(Timestamp::from_nanos(m.transact_time as i64)),
            MdpMessage::SecurityStatus(m) => Some(Timestamp::from_nanos(m.transact_time as i64)),
            MdpMessage::Snapshot(m) => Some(Timestamp::from_nanos(m.last_update_time as i64)),
            MdpMessage::Unknown { .. } => None,
        }
    }

    /// Get the security ID if available
    pub fn security_id(&self) -> Option<i32> {
        match self {
            MdpMessage::BookUpdate(m) => Some(m.security_id),
            MdpMessage::Trade(m) => Some(m.security_id),
            MdpMessage::SecurityStatus(m) => Some(m.security_id),
            MdpMessage::Snapshot(m) => Some(m.security_id),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_action_from_u8() {
        assert_eq!(UpdateAction::try_from(0), Ok(UpdateAction::New));
        assert_eq!(UpdateAction::try_from(1), Ok(UpdateAction::Change));
        assert_eq!(UpdateAction::try_from(2), Ok(UpdateAction::Delete));
        assert!(UpdateAction::try_from(10).is_err());
    }

    #[test]
    fn test_entry_type_to_side() {
        assert_eq!(EntryType::Bid.to_side(), Some(Side::Buy));
        assert_eq!(EntryType::Offer.to_side(), Some(Side::Sell));
        assert_eq!(EntryType::Trade.to_side(), None);
    }

    #[test]
    fn test_book_entry_conversion() {
        let entry = BookEntry {
            price: 500025000, // 5000.25 with exponent -3
            quantity: 100,
            num_orders: 5,
            price_level: 1,
            action: UpdateAction::New,
            entry_type: EntryType::Bid,
        };

        let price = entry.to_price(-3);
        assert_eq!(price.raw(), 500025);

        let qty = entry.to_quantity();
        assert_eq!(qty.value(), 100);
    }

    #[test]
    fn test_trade_entry_aggressor() {
        let buy_aggressor = TradeEntry {
            price: 500000,
            quantity: 10,
            num_orders: 1,
            aggressor_side: 0,
            action: UpdateAction::New,
        };
        assert_eq!(buy_aggressor.aggressor(), Side::Buy);

        let sell_aggressor = TradeEntry {
            price: 500000,
            quantity: 10,
            num_orders: 1,
            aggressor_side: 1,
            action: UpdateAction::New,
        };
        assert_eq!(sell_aggressor.aggressor(), Side::Sell);
    }
}

