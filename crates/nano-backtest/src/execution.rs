//! Execution simulation for backtesting.

use std::collections::HashMap;

use nano_core::traits::OrderBook as OrderBookTrait;
use nano_core::types::{Fill, Order, OrderId, OrderStatus, Quantity, Side, Timestamp};
use nano_lob::OrderBook;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::config::{ExecutionConfig, FeeConfig};

/// Fill simulation result
#[derive(Debug, Clone)]
pub struct FillResult {
    /// Fill details
    pub fill: Fill,
    /// Whether the order is fully filled
    pub is_complete: bool,
}

/// Simulated exchange for backtesting
#[derive(Debug)]
pub struct SimulatedExchange {
    /// Active orders by ID
    orders: HashMap<OrderId, Order>,
    /// Queue positions for limit orders
    queue_positions: HashMap<OrderId, usize>,
    /// Next order ID
    next_order_id: u64,
    /// Fee configuration
    fees: FeeConfig,
    /// Execution configuration
    config: ExecutionConfig,
    /// RNG for probabilistic fills
    rng: StdRng,
}

impl SimulatedExchange {
    /// Create a new simulated exchange
    #[must_use]
    pub fn new(fees: FeeConfig, config: ExecutionConfig) -> Self {
        Self {
            orders: HashMap::new(),
            queue_positions: HashMap::new(),
            next_order_id: 1,
            fees,
            config,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Submit an order to the exchange
    pub fn submit_order(&mut self, mut order: Order, timestamp: Timestamp) -> OrderId {
        let order_id = OrderId::new(self.next_order_id);
        self.next_order_id += 1;

        order.id = order_id;
        order.status = OrderStatus::Open;
        order.updated_at = timestamp;

        // Estimate queue position for limit orders
        if self.config.track_queue_position {
            // In reality, would need to know the book state
            // For now, estimate based on order count
            let queue_pos = self.orders.len() + 1;
            self.queue_positions.insert(order_id, queue_pos);
        }

        self.orders.insert(order_id, order);
        order_id
    }

    /// Cancel an order
    pub fn cancel_order(&mut self, order_id: OrderId) -> Option<Order> {
        self.queue_positions.remove(&order_id);
        self.orders.remove(&order_id).map(|mut o| {
            o.status = OrderStatus::Cancelled;
            o
        })
    }

    /// Try to fill orders against the book
    pub fn match_orders(&mut self, book: &OrderBook, timestamp: Timestamp) -> Vec<FillResult> {
        let mut fills = Vec::new();
        let mut completed_orders = Vec::new();

        // Collect active order IDs first to avoid borrow conflict
        let active_order_ids: Vec<OrderId> = self
            .orders
            .iter()
            .filter(|(_, o)| o.status.is_active())
            .map(|(id, _)| *id)
            .collect();

        for order_id in active_order_ids {
            // Get order info for fill calculation
            let order_info = {
                let order = match self.orders.get(&order_id) {
                    Some(o) => o,
                    None => continue,
                };

                (*order, order.remaining_quantity(), order.side, order.price)
            };

            // Try to fill
            if let Some(fill_result) = self.try_fill_order(&order_info.0, book, timestamp) {
                // Apply fill to order
                if let Some(order) = self.orders.get_mut(&order_id) {
                    order.apply_fill(fill_result.fill.price, fill_result.fill.quantity, timestamp);

                    if order.is_filled() {
                        order.status = OrderStatus::Filled;
                        completed_orders.push(order_id);
                    }
                }

                fills.push(fill_result);
            }
        }

        // Remove completed orders
        for id in completed_orders {
            self.orders.remove(&id);
            self.queue_positions.remove(&id);
        }

        fills
    }

    /// Try to fill a single order
    fn try_fill_order(
        &mut self,
        order: &Order,
        book: &OrderBook,
        timestamp: Timestamp,
    ) -> Option<FillResult> {
        let remaining = order.remaining_quantity();
        if remaining.is_zero() {
            return None;
        }

        // Get the relevant book side
        let (book_price, book_qty) = match order.side {
            Side::Buy => book.best_ask()?,
            Side::Sell => book.best_bid()?,
        };

        // Check if order price crosses the market
        let crosses = match order.side {
            Side::Buy => order.price >= book_price,
            Side::Sell => order.price <= book_price,
        };

        if !crosses {
            return None;
        }

        // Determine if this is a maker or taker fill
        let is_maker = order.price != book_price;

        // Simulate fill probability based on queue position
        let fill_prob = if self.config.track_queue_position {
            let queue_pos = self.queue_positions.get(&order.id).copied().unwrap_or(1);
            self.config.fill_probability_decay.powi(queue_pos as i32)
        } else {
            1.0
        };

        // Check adverse selection
        if self.config.simulate_adverse_selection
            && self.rng.gen::<f64>() < self.config.adverse_selection_prob
        {
            // Adverse selection: market moved against us, no fill
            return None;
        }

        // Random fill check
        if self.rng.gen::<f64>() > fill_prob {
            return None;
        }

        // Determine fill quantity
        let fill_qty = if self.config.partial_fill_probability > 0.0
            && self.rng.gen::<f64>() < self.config.partial_fill_probability
        {
            // Partial fill
            let max_fill = remaining.value().min(book_qty.value());
            let partial = self.rng.gen_range(self.config.min_partial_fill..=max_fill);
            Quantity::new(partial)
        } else {
            // Full fill up to available liquidity
            Quantity::new(remaining.value().min(book_qty.value()))
        };

        if fill_qty.is_zero() {
            return None;
        }

        // Calculate fee
        let fee = self.fees.calculate_fee(fill_qty.value(), is_maker);

        let fill = Fill {
            order_id: order.id,
            price: book_price,
            quantity: fill_qty,
            side: order.side,
            is_maker,
            timestamp,
            fee,
        };

        let is_complete = fill_qty >= remaining;

        Some(FillResult { fill, is_complete })
    }

    /// Get an order by ID
    #[must_use]
    pub fn get_order(&self, order_id: OrderId) -> Option<&Order> {
        self.orders.get(&order_id)
    }

    /// Get all active orders
    pub fn active_orders(&self) -> impl Iterator<Item = &Order> {
        self.orders.values().filter(|o| o.status.is_active())
    }

    /// Get number of active orders
    #[must_use]
    pub fn active_order_count(&self) -> usize {
        self.orders
            .values()
            .filter(|o| o.status.is_active())
            .count()
    }

    /// Reset the exchange state
    pub fn reset(&mut self) {
        self.orders.clear();
        self.queue_positions.clear();
        self.next_order_id = 1;
        self.rng = StdRng::seed_from_u64(42);
    }

    /// Set random seed
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

/// Fill simulator for more advanced fill modeling
#[derive(Debug)]
pub struct FillSimulator {
    /// Base fill probability
    base_fill_prob: f64,
    /// Queue position decay factor
    queue_decay: f64,
    /// Adverse selection probability
    adverse_selection_prob: f64,
    /// RNG
    rng: StdRng,
}

impl FillSimulator {
    /// Create a new fill simulator
    #[must_use]
    pub fn new(base_fill_prob: f64, queue_decay: f64, adverse_selection_prob: f64) -> Self {
        Self {
            base_fill_prob,
            queue_decay,
            adverse_selection_prob,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Estimate fill probability based on order and market conditions
    #[must_use]
    pub fn fill_probability(&self, order: &Order, book: &OrderBook, queue_position: usize) -> f64 {
        let queue_factor = self.queue_decay.powi(queue_position as i32);

        // Check distance from BBO
        let distance_factor = match order.side {
            Side::Buy => {
                if let Some((ask_price, _)) = book.best_ask() {
                    if order.price >= ask_price {
                        1.0 // At or above ask - immediate fill likely
                    } else {
                        let ticks_away = (ask_price.raw() - order.price.raw()) as f64;
                        0.5_f64.powf(ticks_away / 10.0)
                    }
                } else {
                    0.0
                }
            }
            Side::Sell => {
                if let Some((bid_price, _)) = book.best_bid() {
                    if order.price <= bid_price {
                        1.0 // At or below bid - immediate fill likely
                    } else {
                        let ticks_away = (order.price.raw() - bid_price.raw()) as f64;
                        0.5_f64.powf(ticks_away / 10.0)
                    }
                } else {
                    0.0
                }
            }
        };

        self.base_fill_prob * queue_factor * distance_factor * (1.0 - self.adverse_selection_prob)
    }

    /// Simulate a fill attempt
    pub fn try_fill(
        &mut self,
        order: &Order,
        book: &OrderBook,
        queue_position: usize,
        timestamp: Timestamp,
        fee_config: &FeeConfig,
    ) -> Option<Fill> {
        let fill_prob = self.fill_probability(order, book, queue_position);

        if self.rng.gen::<f64>() > fill_prob {
            return None;
        }

        // Determine fill price and quantity
        let (fill_price, available_qty, is_maker) = match order.side {
            Side::Buy => {
                let (ask_price, ask_qty) = book.best_ask()?;
                (ask_price, ask_qty, order.price < ask_price)
            }
            Side::Sell => {
                let (bid_price, bid_qty) = book.best_bid()?;
                (bid_price, bid_qty, order.price > bid_price)
            }
        };

        let fill_qty = order
            .remaining_quantity()
            .value()
            .min(available_qty.value());
        if fill_qty == 0 {
            return None;
        }

        let fee = fee_config.calculate_fee(fill_qty, is_maker);

        Some(Fill {
            order_id: order.id,
            price: fill_price,
            quantity: Quantity::new(fill_qty),
            side: order.side,
            is_maker,
            timestamp,
            fee,
        })
    }

    /// Set random seed
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nano_core::types::{OrderType, Price, Quantity, TimeInForce, Timestamp};
    use nano_feed::messages::{BookEntry, BookUpdate, EntryType, UpdateAction};

    fn create_test_book() -> OrderBook {
        let mut book = OrderBook::new(1);

        let update = BookUpdate {
            transact_time: 1_000_000_000,
            match_event_indicator: 0x81,
            security_id: 1,
            rpt_seq: 1,
            exponent: -2,
            entries: vec![
                BookEntry {
                    price: 50000,
                    quantity: 100,
                    num_orders: 5,
                    price_level: 1,
                    action: UpdateAction::New,
                    entry_type: EntryType::Bid,
                },
                BookEntry {
                    price: 50010,
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

    fn create_test_order(side: Side, price: i64, quantity: u32) -> Order {
        Order {
            id: OrderId::new(0), // Will be assigned by exchange
            instrument_id: 1,
            side,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::Pending,
            price: Price::from_raw(price),
            stop_price: Price::ZERO,
            quantity: Quantity::new(quantity),
            filled_quantity: Quantity::ZERO,
            avg_fill_price: 0,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        }
    }

    #[test]
    fn test_submit_order() {
        let mut exchange = SimulatedExchange::new(FeeConfig::default(), ExecutionConfig::default());

        let order = create_test_order(Side::Buy, 50010, 10);
        let order_id = exchange.submit_order(order, Timestamp::now());

        assert_eq!(order_id.value(), 1);
        assert!(exchange.get_order(order_id).is_some());
    }

    #[test]
    fn test_fill_marketable_order() {
        let mut exchange = SimulatedExchange::new(
            FeeConfig::default(),
            ExecutionConfig {
                track_queue_position: false,
                simulate_adverse_selection: false,
                partial_fill_probability: 0.0,
                ..Default::default()
            },
        );

        let book = create_test_book();

        // Buy order at or above ask should fill
        let order = create_test_order(Side::Buy, 50010, 10);
        let _order_id = exchange.submit_order(order, Timestamp::now());

        let fills = exchange.match_orders(&book, Timestamp::now());

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].fill.price.raw(), 50010);
        assert_eq!(fills[0].fill.quantity.value(), 10);
    }

    #[test]
    fn test_cancel_order() {
        let mut exchange = SimulatedExchange::new(FeeConfig::default(), ExecutionConfig::default());

        let order = create_test_order(Side::Buy, 49990, 10);
        let order_id = exchange.submit_order(order, Timestamp::now());

        let cancelled = exchange.cancel_order(order_id);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().status, OrderStatus::Cancelled);
        assert!(exchange.get_order(order_id).is_none());
    }
}
