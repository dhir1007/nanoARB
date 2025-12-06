//! Event types and priority queue for backtesting.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use nano_core::types::{Fill, Order, OrderId, Timestamp};
use serde::{Deserialize, Serialize};

/// Event types in the backtest simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Market data update received
    MarketData {
        /// Instrument ID
        instrument_id: u32,
    },

    /// Order submitted to exchange
    OrderSubmit {
        /// The order being submitted
        order: Order,
    },

    /// Order acknowledged by exchange
    OrderAck {
        /// Order ID
        order_id: OrderId,
    },

    /// Order rejected by exchange
    OrderReject {
        /// Order ID
        order_id: OrderId,
        /// Rejection reason
        reason: String,
    },

    /// Order filled (partial or complete)
    OrderFill {
        /// Fill details
        fill: Fill,
    },

    /// Order cancelled
    OrderCancel {
        /// Order ID
        order_id: OrderId,
    },

    /// Cancel rejected
    CancelReject {
        /// Order ID
        order_id: OrderId,
        /// Rejection reason
        reason: String,
    },

    /// Timer event for scheduled actions
    Timer {
        /// Timer ID
        timer_id: u64,
        /// Timer data
        data: Option<String>,
    },

    /// Signal from strategy
    Signal {
        /// Signal name
        name: String,
        /// Signal value
        value: f64,
    },

    /// End of data
    EndOfData,
}

/// A timestamped event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event timestamp
    pub timestamp: Timestamp,
    /// Event sequence number (for ordering events at same timestamp)
    pub sequence: u64,
    /// Event type
    pub event_type: EventType,
}

impl Event {
    /// Create a new event
    #[must_use]
    pub fn new(timestamp: Timestamp, sequence: u64, event_type: EventType) -> Self {
        Self {
            timestamp,
            sequence,
            event_type,
        }
    }

    /// Create a market data event
    #[must_use]
    pub fn market_data(timestamp: Timestamp, sequence: u64, instrument_id: u32) -> Self {
        Self::new(timestamp, sequence, EventType::MarketData { instrument_id })
    }

    /// Create an order submit event
    #[must_use]
    pub fn order_submit(timestamp: Timestamp, sequence: u64, order: Order) -> Self {
        Self::new(timestamp, sequence, EventType::OrderSubmit { order })
    }

    /// Create an order ack event
    #[must_use]
    pub fn order_ack(timestamp: Timestamp, sequence: u64, order_id: OrderId) -> Self {
        Self::new(timestamp, sequence, EventType::OrderAck { order_id })
    }

    /// Create an order fill event
    #[must_use]
    pub fn order_fill(timestamp: Timestamp, sequence: u64, fill: Fill) -> Self {
        Self::new(timestamp, sequence, EventType::OrderFill { fill })
    }

    /// Create an order cancel event
    #[must_use]
    pub fn order_cancel(timestamp: Timestamp, sequence: u64, order_id: OrderId) -> Self {
        Self::new(timestamp, sequence, EventType::OrderCancel { order_id })
    }

    /// Create a timer event
    #[must_use]
    pub fn timer(timestamp: Timestamp, sequence: u64, timer_id: u64, data: Option<String>) -> Self {
        Self::new(timestamp, sequence, EventType::Timer { timer_id, data })
    }

    /// Create an end of data event
    #[must_use]
    pub fn end_of_data(timestamp: Timestamp, sequence: u64) -> Self {
        Self::new(timestamp, sequence, EventType::EndOfData)
    }

    /// Check if this is a market data event
    #[must_use]
    pub fn is_market_data(&self) -> bool {
        matches!(self.event_type, EventType::MarketData { .. })
    }

    /// Check if this is an order event
    #[must_use]
    pub fn is_order_event(&self) -> bool {
        matches!(
            self.event_type,
            EventType::OrderSubmit { .. }
                | EventType::OrderAck { .. }
                | EventType::OrderFill { .. }
                | EventType::OrderCancel { .. }
                | EventType::OrderReject { .. }
        )
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp && self.sequence == other.sequence
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        match other.timestamp.as_nanos().cmp(&self.timestamp.as_nanos()) {
            Ordering::Equal => other.sequence.cmp(&self.sequence),
            ord => ord,
        }
    }
}

/// Priority queue for events (min-heap by timestamp)
#[derive(Debug, Default)]
pub struct EventQueue {
    /// Internal heap storage
    heap: BinaryHeap<Event>,
    /// Sequence counter for tie-breaking
    sequence_counter: u64,
}

impl EventQueue {
    /// Create a new event queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            sequence_counter: 0,
        }
    }

    /// Create with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            sequence_counter: 0,
        }
    }

    /// Push an event with auto-generated sequence number
    pub fn push(&mut self, timestamp: Timestamp, event_type: EventType) {
        self.sequence_counter += 1;
        self.heap
            .push(Event::new(timestamp, self.sequence_counter, event_type));
    }

    /// Push an event with explicit sequence number
    pub fn push_event(&mut self, event: Event) {
        self.heap.push(event);
    }

    /// Pop the next event (earliest timestamp)
    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop()
    }

    /// Peek at the next event without removing it
    #[must_use]
    pub fn peek(&self) -> Option<&Event> {
        self.heap.peek()
    }

    /// Check if the queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get the number of events in the queue
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Get current sequence number
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence_counter
    }

    /// Schedule a market data event
    pub fn schedule_market_data(&mut self, timestamp: Timestamp, instrument_id: u32) {
        self.push(timestamp, EventType::MarketData { instrument_id });
    }

    /// Schedule an order submission
    pub fn schedule_order_submit(&mut self, timestamp: Timestamp, order: Order) {
        self.push(timestamp, EventType::OrderSubmit { order });
    }

    /// Schedule a fill event
    pub fn schedule_fill(&mut self, timestamp: Timestamp, fill: Fill) {
        self.push(timestamp, EventType::OrderFill { fill });
    }

    /// Schedule a timer event
    pub fn schedule_timer(&mut self, timestamp: Timestamp, timer_id: u64, data: Option<String>) {
        self.push(timestamp, EventType::Timer { timer_id, data });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_ordering() {
        let e1 = Event::new(Timestamp::from_nanos(100), 1, EventType::EndOfData);
        let e2 = Event::new(Timestamp::from_nanos(50), 2, EventType::EndOfData);
        let e3 = Event::new(Timestamp::from_nanos(100), 2, EventType::EndOfData);

        // e2 should come first (earlier timestamp)
        assert!(e2 > e1);
        // e1 and e3 have same timestamp, e1 should come first (lower sequence)
        assert!(e1 > e3);
    }

    #[test]
    fn test_event_queue() {
        let mut queue = EventQueue::new();

        queue.push(Timestamp::from_nanos(100), EventType::EndOfData);
        queue.push(Timestamp::from_nanos(50), EventType::EndOfData);
        queue.push(Timestamp::from_nanos(75), EventType::EndOfData);

        // Should pop in timestamp order: 50, 75, 100
        assert_eq!(queue.pop().unwrap().timestamp.as_nanos(), 50);
        assert_eq!(queue.pop().unwrap().timestamp.as_nanos(), 75);
        assert_eq!(queue.pop().unwrap().timestamp.as_nanos(), 100);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_event_queue_same_timestamp() {
        let mut queue = EventQueue::new();

        // Events at same timestamp should be ordered by sequence
        queue.push(Timestamp::from_nanos(100), EventType::EndOfData);
        queue.push(Timestamp::from_nanos(100), EventType::EndOfData);
        queue.push(Timestamp::from_nanos(100), EventType::EndOfData);

        let e1 = queue.pop().unwrap();
        let e2 = queue.pop().unwrap();
        let e3 = queue.pop().unwrap();

        assert!(e1.sequence < e2.sequence);
        assert!(e2.sequence < e3.sequence);
    }
}
