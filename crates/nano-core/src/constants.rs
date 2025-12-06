//! Constants used throughout the NanoARB trading engine.

/// Maximum number of price levels to track in the order book
pub const MAX_BOOK_LEVELS: usize = 20;

/// Number of price levels used for feature extraction
pub const FEATURE_LEVELS: usize = 10;

/// Number of historical snapshots to keep for temporal modeling
pub const SNAPSHOT_HISTORY_SIZE: usize = 200;

/// Number of fields per level in LOB tensor (bid_price, bid_qty, ask_price, ask_qty)
pub const FIELDS_PER_LEVEL: usize = 4;

/// Nanoseconds per microsecond
pub const NS_PER_US: u64 = 1_000;

/// Nanoseconds per millisecond
pub const NS_PER_MS: u64 = 1_000_000;

/// Nanoseconds per second
pub const NS_PER_SEC: u64 = 1_000_000_000;

/// Default tick size for ES futures (in points, 0.25)
pub const ES_TICK_SIZE: f64 = 0.25;

/// Default tick value for ES futures ($12.50 per tick)
pub const ES_TICK_VALUE: f64 = 12.50;

/// Default tick size for NQ futures (in points, 0.25)
pub const NQ_TICK_SIZE: f64 = 0.25;

/// Default tick value for NQ futures ($5.00 per tick)
pub const NQ_TICK_VALUE: f64 = 5.00;

/// CME MDP 3.0 message header size in bytes
pub const MDP_HEADER_SIZE: usize = 12;

/// Maximum message size for CME MDP 3.0
pub const MDP_MAX_MESSAGE_SIZE: usize = 65535;

/// Default colo-to-exchange latency in nanoseconds (100 microseconds)
pub const DEFAULT_COLO_LATENCY_NS: u64 = 100_000;

/// Default network jitter in nanoseconds (10 microseconds std dev)
pub const DEFAULT_JITTER_NS: u64 = 10_000;

/// Maximum inventory limit (number of contracts)
pub const DEFAULT_MAX_INVENTORY: i64 = 100;

/// Maximum drawdown percentage before kill switch activates
pub const DEFAULT_MAX_DRAWDOWN_PCT: f64 = 0.06;

/// CME maker fee per contract (approximate, varies by volume tier)
pub const CME_MAKER_FEE: f64 = 0.25;

/// CME taker fee per contract (approximate, varies by volume tier)
pub const CME_TAKER_FEE: f64 = 0.85;

/// Exchange fee per contract
pub const CME_EXCHANGE_FEE: f64 = 1.18;

/// Clearing fee per contract
pub const CME_CLEARING_FEE: f64 = 0.10;

