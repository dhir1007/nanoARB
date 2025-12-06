//! CME MDP 3.0 message parser using nom.

use nom::{
    bytes::complete::take,
    multi::count,
    number::complete::{le_i32, le_i64, le_i8, le_u16, le_u32, le_u64, le_u8},
    IResult,
};

use crate::error::{FeedError, FeedResult};
use crate::messages::*;

/// CME MDP 3.0 parser
#[derive(Debug, Default)]
pub struct MdpParser {
    /// Expected sequence number for gap detection
    expected_seq: u32,
    /// Whether we've seen the first message
    initialized: bool,
}

impl MdpParser {
    /// Create a new parser
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_seq: 0,
            initialized: false,
        }
    }

    /// Parse a raw message buffer
    pub fn parse<'a>(&mut self, input: &'a [u8]) -> FeedResult<(MdpMessage, &'a [u8])> {
        if input.len() < MdpHeader::SIZE {
            return Err(FeedError::Incomplete {
                needed: MdpHeader::SIZE - input.len(),
            });
        }

        let (remaining, header) = parse_header(input)
            .map_err(|_| FeedError::InvalidHeader("Failed to parse header".to_string()))?;

        let msg_body_len = header.msg_size as usize - MdpHeader::SIZE;
        if remaining.len() < msg_body_len {
            return Err(FeedError::Incomplete {
                needed: msg_body_len - remaining.len(),
            });
        }

        let (after_msg, msg_body) = remaining.split_at(msg_body_len);

        let message = match header.template_id {
            MdpHeader::TEMPLATE_BOOK_UPDATE => {
                let (_, update) = parse_book_update(msg_body)
                    .map_err(|e| FeedError::ParseError(format!("BookUpdate: {e:?}")))?;
                self.check_sequence(update.rpt_seq)?;
                MdpMessage::BookUpdate(update)
            }
            MdpHeader::TEMPLATE_TRADE => {
                let (_, update) = parse_trade_update(msg_body)
                    .map_err(|e| FeedError::ParseError(format!("Trade: {e:?}")))?;
                self.check_sequence(update.rpt_seq)?;
                MdpMessage::Trade(update)
            }
            MdpHeader::TEMPLATE_CHANNEL_RESET => {
                let (_, reset) = parse_channel_reset(msg_body)
                    .map_err(|e| FeedError::ParseError(format!("ChannelReset: {e:?}")))?;
                // Reset sequence tracking on channel reset
                self.initialized = false;
                MdpMessage::ChannelReset(reset)
            }
            MdpHeader::TEMPLATE_SECURITY_STATUS => {
                let (_, status) = parse_security_status(msg_body)
                    .map_err(|e| FeedError::ParseError(format!("SecurityStatus: {e:?}")))?;
                MdpMessage::SecurityStatus(status)
            }
            MdpHeader::TEMPLATE_SNAPSHOT => {
                let (_, snapshot) = parse_snapshot(msg_body)
                    .map_err(|e| FeedError::ParseError(format!("Snapshot: {e:?}")))?;
                self.expected_seq = snapshot.rpt_seq + 1;
                self.initialized = true;
                MdpMessage::Snapshot(snapshot)
            }
            template_id => MdpMessage::Unknown {
                template_id,
                length: msg_body_len,
            },
        };

        Ok((message, after_msg))
    }

    /// Parse all messages from a buffer
    pub fn parse_all(&mut self, mut input: &[u8]) -> FeedResult<Vec<MdpMessage>> {
        let mut messages = Vec::new();

        while !input.is_empty() {
            match self.parse(input) {
                Ok((msg, remaining)) => {
                    messages.push(msg);
                    input = remaining;
                }
                Err(FeedError::Incomplete { .. }) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(messages)
    }

    /// Check for sequence gaps
    fn check_sequence(&mut self, seq: u32) -> FeedResult<()> {
        if !self.initialized {
            self.expected_seq = seq + 1;
            self.initialized = true;
            return Ok(());
        }

        if seq != self.expected_seq {
            let expected = self.expected_seq;
            self.expected_seq = seq + 1;
            return Err(FeedError::SequenceGap {
                expected,
                actual: seq,
            });
        }

        self.expected_seq = seq + 1;
        Ok(())
    }

    /// Reset the parser state
    pub fn reset(&mut self) {
        self.expected_seq = 0;
        self.initialized = false;
    }

    /// Get the expected sequence number
    #[must_use]
    pub fn expected_sequence(&self) -> u32 {
        self.expected_seq
    }
}

// Internal parsing functions using nom

fn parse_header(input: &[u8]) -> IResult<&[u8], MdpHeader> {
    let (input, msg_size) = le_u16(input)?;
    let (input, block_length) = le_u16(input)?;
    let (input, template_id) = le_u16(input)?;
    let (input, schema_id) = le_u16(input)?;
    let (input, version) = le_u16(input)?;
    let (input, _padding) = take(2usize)(input)?; // 2 bytes padding

    Ok((
        input,
        MdpHeader {
            msg_size,
            block_length,
            template_id,
            schema_id,
            version,
        },
    ))
}

fn parse_book_entry(input: &[u8]) -> IResult<&[u8], BookEntry> {
    let (input, price) = le_i64(input)?;
    let (input, quantity) = le_i32(input)?;
    let (input, num_orders) = le_i32(input)?;
    let (input, price_level) = le_u8(input)?;
    let (input, action_raw) = le_u8(input)?;
    let (input, entry_type_raw) = le_u8(input)?;
    let (input, _padding) = take(1usize)(input)?; // 1 byte padding

    let action = UpdateAction::try_from(action_raw).unwrap_or(UpdateAction::New);
    let entry_type = EntryType::try_from(entry_type_raw).unwrap_or(EntryType::Bid);

    Ok((
        input,
        BookEntry {
            price,
            quantity,
            num_orders,
            price_level,
            action,
            entry_type,
        },
    ))
}

fn parse_book_update(input: &[u8]) -> IResult<&[u8], BookUpdate> {
    let (input, transact_time) = le_u64(input)?;
    let (input, match_event_indicator) = le_u8(input)?;
    let (input, _padding) = take(3usize)(input)?; // 3 bytes padding
    let (input, security_id) = le_i32(input)?;
    let (input, rpt_seq) = le_u32(input)?;
    let (input, exponent) = le_i8(input)?;
    let (input, _padding2) = take(1usize)(input)?; // 1 byte padding
    let (input, num_entries) = le_u16(input)?;

    let (input, entries) = count(parse_book_entry, num_entries as usize)(input)?;

    Ok((
        input,
        BookUpdate {
            transact_time,
            match_event_indicator,
            security_id,
            rpt_seq,
            exponent,
            entries,
        },
    ))
}

fn parse_trade_entry(input: &[u8]) -> IResult<&[u8], TradeEntry> {
    let (input, price) = le_i64(input)?;
    let (input, quantity) = le_i32(input)?;
    let (input, num_orders) = le_i32(input)?;
    let (input, aggressor_side) = le_u8(input)?;
    let (input, action_raw) = le_u8(input)?;
    let (input, _padding) = take(2usize)(input)?; // 2 bytes padding

    let action = UpdateAction::try_from(action_raw).unwrap_or(UpdateAction::New);

    Ok((
        input,
        TradeEntry {
            price,
            quantity,
            num_orders,
            aggressor_side,
            action,
        },
    ))
}

fn parse_trade_update(input: &[u8]) -> IResult<&[u8], TradeUpdate> {
    let (input, transact_time) = le_u64(input)?;
    let (input, match_event_indicator) = le_u8(input)?;
    let (input, _padding) = take(3usize)(input)?;
    let (input, security_id) = le_i32(input)?;
    let (input, rpt_seq) = le_u32(input)?;
    let (input, exponent) = le_i8(input)?;
    let (input, _padding2) = take(1usize)(input)?;
    let (input, num_entries) = le_u16(input)?;

    let (input, entries) = count(parse_trade_entry, num_entries as usize)(input)?;

    Ok((
        input,
        TradeUpdate {
            transact_time,
            match_event_indicator,
            security_id,
            rpt_seq,
            exponent,
            entries,
        },
    ))
}

fn parse_channel_reset(input: &[u8]) -> IResult<&[u8], ChannelReset> {
    let (input, transact_time) = le_u64(input)?;
    let (input, match_event_indicator) = le_u8(input)?;

    Ok((
        input,
        ChannelReset {
            transact_time,
            match_event_indicator,
        },
    ))
}

fn parse_security_status(input: &[u8]) -> IResult<&[u8], SecurityStatus> {
    let (input, transact_time) = le_u64(input)?;
    let (input, _padding) = take(4usize)(input)?;
    let (input, security_id) = le_i32(input)?;
    let (input, trading_status) = le_u8(input)?;
    let (input, halt_reason) = le_u8(input)?;
    let (input, trading_event) = le_u8(input)?;

    Ok((
        input,
        SecurityStatus {
            transact_time,
            security_id,
            trading_status,
            halt_reason,
            trading_event,
        },
    ))
}

fn parse_snapshot_entry(input: &[u8]) -> IResult<&[u8], SnapshotEntry> {
    let (input, price) = le_i64(input)?;
    let (input, quantity) = le_i32(input)?;
    let (input, num_orders) = le_i32(input)?;
    let (input, price_level) = le_u8(input)?;
    let (input, entry_type_raw) = le_u8(input)?;
    let (input, _padding) = take(2usize)(input)?;

    let entry_type = EntryType::try_from(entry_type_raw).unwrap_or(EntryType::Bid);

    Ok((
        input,
        SnapshotEntry {
            price,
            quantity,
            num_orders,
            price_level,
            entry_type,
        },
    ))
}

fn parse_snapshot(input: &[u8]) -> IResult<&[u8], Snapshot> {
    let (input, last_update_time) = le_u64(input)?;
    let (input, _padding) = take(4usize)(input)?;
    let (input, security_id) = le_i32(input)?;
    let (input, rpt_seq) = le_u32(input)?;
    let (input, exponent) = le_i8(input)?;
    let (input, _padding2) = take(1usize)(input)?;
    let (input, num_entries) = le_u16(input)?;

    let (input, entries) = count(parse_snapshot_entry, num_entries as usize)(input)?;

    Ok((
        input,
        Snapshot {
            last_update_time,
            security_id,
            rpt_seq,
            exponent,
            entries,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_header(template_id: u16, msg_size: u16) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&msg_size.to_le_bytes());
        buf.extend_from_slice(&20u16.to_le_bytes()); // block_length
        buf.extend_from_slice(&template_id.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // schema_id
        buf.extend_from_slice(&1u16.to_le_bytes()); // version
        buf.extend_from_slice(&[0u8; 2]); // padding
        buf
    }

    #[test]
    fn test_parse_header() {
        let data = create_test_header(MdpHeader::TEMPLATE_BOOK_UPDATE, 100);
        let (_, header) = parse_header(&data).unwrap();

        assert_eq!(header.msg_size, 100);
        assert_eq!(header.template_id, MdpHeader::TEMPLATE_BOOK_UPDATE);
    }

    #[test]
    fn test_parser_incomplete() {
        let mut parser = MdpParser::new();
        let data = vec![0u8; 5]; // Less than header size

        let result = parser.parse(&data);
        assert!(matches!(result, Err(FeedError::Incomplete { .. })));
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = MdpParser::new();
        parser.expected_seq = 100;
        parser.initialized = true;

        parser.reset();

        assert_eq!(parser.expected_seq, 0);
        assert!(!parser.initialized);
    }
}

