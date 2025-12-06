//! Instrument (tradable asset) types.

use std::fmt;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Exchange identifier
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[repr(u8)]
pub enum Exchange {
    /// CME Group
    CME = 0,
    /// NASDAQ
    NASDAQ = 1,
    /// NYSE
    NYSE = 2,
    /// CBOE
    CBOE = 3,
    /// ICE
    ICE = 4,
    /// Simulated/Paper trading
    Simulated = 255,
}

impl Default for Exchange {
    fn default() -> Self {
        Exchange::CME
    }
}

impl fmt::Debug for Exchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exchange::CME => write!(f, "CME"),
            Exchange::NASDAQ => write!(f, "NASDAQ"),
            Exchange::NYSE => write!(f, "NYSE"),
            Exchange::CBOE => write!(f, "CBOE"),
            Exchange::ICE => write!(f, "ICE"),
            Exchange::Simulated => write!(f, "Simulated"),
        }
    }
}

impl fmt::Display for Exchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Instrument type
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
#[repr(u8)]
pub enum InstrumentType {
    /// Futures contract
    Future = 0,
    /// Equity (stock)
    Equity = 1,
    /// Option
    Option = 2,
    /// ETF
    ETF = 3,
    /// Forex
    Forex = 4,
    /// Cryptocurrency (for completeness)
    Crypto = 5,
}

impl Default for InstrumentType {
    fn default() -> Self {
        InstrumentType::Future
    }
}

impl fmt::Debug for InstrumentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstrumentType::Future => write!(f, "Future"),
            InstrumentType::Equity => write!(f, "Equity"),
            InstrumentType::Option => write!(f, "Option"),
            InstrumentType::ETF => write!(f, "ETF"),
            InstrumentType::Forex => write!(f, "Forex"),
            InstrumentType::Crypto => write!(f, "Crypto"),
        }
    }
}

impl fmt::Display for InstrumentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Tradable instrument definition
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
pub struct Instrument {
    /// Unique instrument ID (internal)
    pub id: u32,
    /// Symbol (e.g., "ESH24", "AAPL")
    pub symbol: String,
    /// Exchange
    pub exchange: Exchange,
    /// Instrument type
    pub instrument_type: InstrumentType,
    /// Tick size (minimum price increment) in raw units
    pub tick_size: i64,
    /// Tick value (P&L per tick per contract) in cents
    pub tick_value: i64,
    /// Contract multiplier
    pub multiplier: f64,
    /// Maker fee per contract (in dollars)
    pub maker_fee: f64,
    /// Taker fee per contract (in dollars)
    pub taker_fee: f64,
    /// Minimum order quantity
    pub min_quantity: u32,
    /// Maximum order quantity
    pub max_quantity: u32,
    /// Number of decimal places for price display
    pub price_decimals: u8,
}

impl Instrument {
    /// Create a new ES (E-mini S&P 500) futures instrument
    #[must_use]
    pub fn es_future(id: u32, symbol: &str) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            exchange: Exchange::CME,
            instrument_type: InstrumentType::Future,
            tick_size: 25, // 0.25 points represented as 25 (2 decimal places)
            tick_value: 1250, // $12.50 per tick
            multiplier: 50.0,
            maker_fee: 0.25,
            taker_fee: 0.85,
            min_quantity: 1,
            max_quantity: 10000,
            price_decimals: 2,
        }
    }

    /// Create a new NQ (E-mini NASDAQ-100) futures instrument
    #[must_use]
    pub fn nq_future(id: u32, symbol: &str) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            exchange: Exchange::CME,
            instrument_type: InstrumentType::Future,
            tick_size: 25, // 0.25 points
            tick_value: 500, // $5.00 per tick
            multiplier: 20.0,
            maker_fee: 0.25,
            taker_fee: 0.85,
            min_quantity: 1,
            max_quantity: 10000,
            price_decimals: 2,
        }
    }

    /// Create a new equity instrument
    #[must_use]
    pub fn equity(id: u32, symbol: &str) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            exchange: Exchange::NASDAQ,
            instrument_type: InstrumentType::Equity,
            tick_size: 1, // $0.01
            tick_value: 1, // $0.01 per share per tick
            multiplier: 1.0,
            maker_fee: 0.0,
            taker_fee: 0.001, // $0.001 per share
            min_quantity: 1,
            max_quantity: 1_000_000,
            price_decimals: 2,
        }
    }

    /// Convert price in raw ticks to display price
    #[must_use]
    pub fn ticks_to_price(&self, ticks: i64) -> f64 {
        let divisor = 10_f64.powi(self.price_decimals as i32);
        (ticks as f64 * self.tick_size as f64) / divisor
    }

    /// Convert display price to raw ticks
    #[must_use]
    pub fn price_to_ticks(&self, price: f64) -> i64 {
        let divisor = 10_f64.powi(self.price_decimals as i32);
        ((price * divisor) / self.tick_size as f64).round() as i64
    }

    /// Calculate P&L for a given number of ticks
    #[must_use]
    pub fn pnl_for_ticks(&self, ticks: i64, quantity: u32) -> f64 {
        (ticks as f64 * self.tick_value as f64 * quantity as f64) / 100.0
    }

    /// Calculate total fee for a trade
    #[must_use]
    pub fn calculate_fee(&self, quantity: u32, is_maker: bool) -> f64 {
        let fee_per_contract = if is_maker {
            self.maker_fee
        } else {
            self.taker_fee
        };
        fee_per_contract * quantity as f64
    }
}

impl fmt::Debug for Instrument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instrument")
            .field("id", &self.id)
            .field("symbol", &self.symbol)
            .field("exchange", &self.exchange)
            .field("type", &self.instrument_type)
            .field("tick_size", &self.tick_size)
            .finish()
    }
}

impl fmt::Display for Instrument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.exchange, self.symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_es_future() {
        let es = Instrument::es_future(1, "ESH24");
        assert_eq!(es.symbol, "ESH24");
        assert_eq!(es.exchange, Exchange::CME);
        assert_eq!(es.tick_size, 25);
        assert_eq!(es.tick_value, 1250);
    }

    #[test]
    fn test_price_conversion() {
        let es = Instrument::es_future(1, "ESH24");

        // 5000.25 in ticks (500025 / 25 = 20001 ticks from 0)
        let ticks = es.price_to_ticks(5000.25);
        let price = es.ticks_to_price(ticks);
        assert!((price - 5000.25).abs() < 0.01);
    }

    #[test]
    fn test_pnl_calculation() {
        let es = Instrument::es_future(1, "ESH24");

        // 4 ticks * $12.50/tick * 2 contracts = $100
        let pnl = es.pnl_for_ticks(4, 2);
        assert!((pnl - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_fee_calculation() {
        let es = Instrument::es_future(1, "ESH24");

        let maker_fee = es.calculate_fee(10, true);
        assert!((maker_fee - 2.50).abs() < 0.01);

        let taker_fee = es.calculate_fee(10, false);
        assert!((taker_fee - 8.50).abs() < 0.01);
    }
}

