//! LOB snapshot storage and ring buffer for temporal modeling.

use nano_core::constants::{FEATURE_LEVELS, FIELDS_PER_LEVEL, SNAPSHOT_HISTORY_SIZE};
use nano_core::traits::OrderBook as OrderBookTrait;
use nano_core::types::Timestamp;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::orderbook::OrderBook;

/// A single LOB snapshot with normalized features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: Timestamp,
    /// Sequence number
    pub sequence: u32,
    /// Mid price at snapshot time
    pub mid_price: f64,
    /// Normalized LOB tensor: (levels, fields)
    /// Fields: [bid_price_delta, bid_qty, ask_price_delta, ask_qty]
    pub tensor: Array2<f32>,
}

impl LobSnapshot {
    /// Create a snapshot from an order book
    #[must_use]
    pub fn from_book(book: &OrderBook) -> Self {
        let mid_price = book.mid_price().map(|p| p.as_f64()).unwrap_or(0.0);
        let timestamp = book.timestamp();
        let sequence = book.sequence();

        let mut tensor = Array2::zeros((FEATURE_LEVELS, FIELDS_PER_LEVEL));

        // Normalize prices relative to mid and quantities
        for i in 0..FEATURE_LEVELS {
            if let Some(bid) = book.bid_level(i) {
                // Bid price delta from mid (in ticks, normalized)
                let price_delta = (bid.price.as_f64() - mid_price) / mid_price;
                tensor[[i, 0]] = price_delta as f32;
                // Bid quantity (log normalized)
                tensor[[i, 1]] = (bid.quantity.value() as f32 + 1.0).ln();
            }

            if let Some(ask) = book.ask_level(i) {
                // Ask price delta from mid
                let price_delta = (ask.price.as_f64() - mid_price) / mid_price;
                tensor[[i, 2]] = price_delta as f32;
                // Ask quantity (log normalized)
                tensor[[i, 3]] = (ask.quantity.value() as f32 + 1.0).ln();
            }
        }

        Self {
            timestamp,
            sequence,
            mid_price,
            tensor,
        }
    }

    /// Get the tensor shape
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (FEATURE_LEVELS, FIELDS_PER_LEVEL)
    }

    /// Calculate mid price return from another snapshot
    #[must_use]
    pub fn return_from(&self, other: &LobSnapshot) -> f64 {
        if other.mid_price > 0.0 {
            (self.mid_price - other.mid_price) / other.mid_price
        } else {
            0.0
        }
    }
}

/// Ring buffer for storing historical LOB snapshots
#[derive(Debug)]
pub struct SnapshotRingBuffer {
    /// Storage for snapshots
    buffer: Vec<Option<LobSnapshot>>,
    /// Current write position
    head: usize,
    /// Number of snapshots stored
    count: usize,
    /// Maximum capacity
    capacity: usize,
}

impl Default for SnapshotRingBuffer {
    fn default() -> Self {
        Self::new(SNAPSHOT_HISTORY_SIZE)
    }
}

impl SnapshotRingBuffer {
    /// Create a new ring buffer with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }

        Self {
            buffer,
            head: 0,
            count: 0,
            capacity,
        }
    }

    /// Push a new snapshot into the buffer
    pub fn push(&mut self, snapshot: LobSnapshot) {
        self.buffer[self.head] = Some(snapshot);
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Push a snapshot created from an order book
    pub fn push_book(&mut self, book: &OrderBook) {
        self.push(LobSnapshot::from_book(book));
    }

    /// Get the most recent snapshot
    #[must_use]
    pub fn latest(&self) -> Option<&LobSnapshot> {
        if self.count == 0 {
            return None;
        }
        let idx = (self.head + self.capacity - 1) % self.capacity;
        self.buffer[idx].as_ref()
    }

    /// Get snapshot at index (0 = most recent, 1 = second most recent, etc.)
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&LobSnapshot> {
        if index >= self.count {
            return None;
        }
        let idx = (self.head + self.capacity - 1 - index) % self.capacity;
        self.buffer[idx].as_ref()
    }

    /// Get the number of snapshots stored
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if buffer is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.head = 0;
        self.count = 0;
    }

    /// Get capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Convert to 3D tensor for ML input: (time, levels, fields)
    /// Returns the most recent `window_size` snapshots
    #[must_use]
    pub fn to_tensor(&self, window_size: usize) -> Option<Array3<f32>> {
        let actual_size = window_size.min(self.count);
        if actual_size == 0 {
            return None;
        }

        let mut tensor = Array3::zeros((actual_size, FEATURE_LEVELS, FIELDS_PER_LEVEL));

        for t in 0..actual_size {
            if let Some(snapshot) = self.get(actual_size - 1 - t) {
                for i in 0..FEATURE_LEVELS {
                    for j in 0..FIELDS_PER_LEVEL {
                        tensor[[t, i, j]] = snapshot.tensor[[i, j]];
                    }
                }
            }
        }

        Some(tensor)
    }

    /// Get returns for the last N periods
    #[must_use]
    pub fn get_returns(&self, periods: usize) -> Vec<f64> {
        let mut returns = Vec::with_capacity(periods);

        for i in 0..periods {
            if let (Some(current), Some(prev)) = (self.get(i), self.get(i + 1)) {
                returns.push(current.return_from(prev));
            } else {
                break;
            }
        }

        returns
    }

    /// Calculate volatility over recent snapshots
    #[must_use]
    pub fn volatility(&self, periods: usize) -> f64 {
        let returns = self.get_returns(periods);
        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Iterator over snapshots (most recent first)
    pub fn iter(&self) -> impl Iterator<Item = &LobSnapshot> {
        (0..self.count).filter_map(move |i| self.get(i))
    }
}

/// Builder for creating ML-ready feature tensors from LOB history
#[derive(Debug)]
pub struct TensorBuilder {
    /// Number of time steps
    window_size: usize,
    /// Include return labels
    include_labels: bool,
    /// Prediction horizons for labels
    horizons: Vec<usize>,
}

impl Default for TensorBuilder {
    fn default() -> Self {
        Self::new(100)
    }
}

impl TensorBuilder {
    /// Create a new tensor builder
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            include_labels: true,
            horizons: vec![10, 50, 100],
        }
    }

    /// Set prediction horizons for labels
    #[must_use]
    pub fn with_horizons(mut self, horizons: Vec<usize>) -> Self {
        self.horizons = horizons;
        self
    }

    /// Disable label generation
    #[must_use]
    pub fn without_labels(mut self) -> Self {
        self.include_labels = false;
        self
    }

    /// Build feature tensor from snapshot buffer
    /// Returns (features, labels) where labels are future returns at each horizon
    pub fn build(&self, buffer: &SnapshotRingBuffer) -> Option<(Array3<f32>, Vec<Vec<f64>>)> {
        let features = buffer.to_tensor(self.window_size)?;

        let labels = if self.include_labels {
            self.horizons
                .iter()
                .map(|&h| {
                    // Get return from current to h steps ahead
                    if let (Some(current), Some(future)) = (buffer.get(0), buffer.get(h)) {
                        vec![future.return_from(current)]
                    } else {
                        vec![0.0]
                    }
                })
                .collect()
        } else {
            vec![]
        };

        Some((features, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nano_feed::messages::{BookEntry, BookUpdate, EntryType, UpdateAction};

    fn create_test_book(mid: i64, seq: u32) -> OrderBook {
        let mut book = OrderBook::new(1);

        let update = BookUpdate {
            transact_time: 1_000_000_000,
            match_event_indicator: 0x81,
            security_id: 1,
            rpt_seq: seq,
            exponent: -2,
            entries: vec![
                BookEntry {
                    price: mid - 5,
                    quantity: 100,
                    num_orders: 5,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: mid + 5,
                    quantity: 50,
                    num_orders: 3,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Offer,
                },
            ],
        };

        book.apply_book_update(&update);
        book
    }

    #[test]
    fn test_snapshot_creation() {
        let book = create_test_book(50000, 1);
        let snapshot = LobSnapshot::from_book(&book);

        assert_eq!(snapshot.sequence, 1);
        assert!((snapshot.mid_price - 500.0).abs() < 0.01);
        assert_eq!(snapshot.shape(), (FEATURE_LEVELS, FIELDS_PER_LEVEL));
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = SnapshotRingBuffer::new(10);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        // Add some snapshots
        for i in 0..5 {
            let book = create_test_book(50000 + i * 10, i as u32);
            buffer.push_book(&book);
        }

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.latest().unwrap().sequence, 4);

        // Most recent should be at index 0
        assert_eq!(buffer.get(0).unwrap().sequence, 4);
        assert_eq!(buffer.get(1).unwrap().sequence, 3);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut buffer = SnapshotRingBuffer::new(3);

        for i in 0..5 {
            let book = create_test_book(50000 + i * 10, i as u32);
            buffer.push_book(&book);
        }

        // Should only have last 3
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
        assert_eq!(buffer.get(0).unwrap().sequence, 4);
        assert_eq!(buffer.get(2).unwrap().sequence, 2);
    }

    #[test]
    fn test_to_tensor() {
        let mut buffer = SnapshotRingBuffer::new(10);

        for i in 0..5 {
            let book = create_test_book(50000 + i * 10, i as u32);
            buffer.push_book(&book);
        }

        let tensor = buffer.to_tensor(3).unwrap();
        assert_eq!(tensor.shape(), &[3, FEATURE_LEVELS, FIELDS_PER_LEVEL]);
    }

    #[test]
    fn test_returns() {
        let mut buffer = SnapshotRingBuffer::new(10);

        // Add snapshots with increasing mid prices
        for i in 0..5 {
            let book = create_test_book(50000 + i * 100, i as u32);
            buffer.push_book(&book);
        }

        let returns = buffer.get_returns(3);
        assert_eq!(returns.len(), 3);

        // All returns should be positive (price going up)
        for r in &returns {
            assert!(*r > 0.0);
        }
    }
}

