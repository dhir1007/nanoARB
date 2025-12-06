//! Latency simulation for realistic backtesting.

use nano_core::traits::LatencyModel;
use nano_core::types::Timestamp;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Normal};

use crate::config::LatencyConfig;

/// Latency simulator with configurable models
pub struct LatencySimulator {
    /// Base order latency in nanoseconds
    order_latency_ns: i64,
    /// Base market data latency in nanoseconds
    market_data_latency_ns: i64,
    /// Base acknowledgment latency in nanoseconds
    ack_latency_ns: i64,
    /// Jitter model
    jitter_model: JitterModel,
    /// Random number generator
    rng: StdRng,
}

/// Jitter model types
#[derive(Debug, Clone)]
pub enum JitterModel {
    /// No jitter (deterministic)
    None,
    /// Uniform jitter around base latency
    Uniform {
        /// Maximum jitter in nanoseconds
        max_jitter_ns: i64,
    },
    /// Normal distribution jitter
    Normal {
        /// Standard deviation in nanoseconds
        std_dev_ns: f64,
    },
    /// Log-normal distribution (more realistic for network latency)
    LogNormal {
        /// Mean of underlying normal
        mu: f64,
        /// Std dev of underlying normal
        sigma: f64,
    },
    /// Empirical distribution from historical data
    Empirical {
        /// Percentile latencies: [p50, p75, p90, p95, p99]
        percentiles: [i64; 5],
    },
}

impl LatencySimulator {
    /// Create a new latency simulator from config
    #[must_use]
    pub fn from_config(config: &LatencyConfig) -> Self {
        let jitter_model = if config.use_random_jitter {
            JitterModel::Normal {
                std_dev_ns: config.jitter_ns as f64,
            }
        } else {
            JitterModel::None
        };

        Self {
            order_latency_ns: config.order_latency_ns as i64,
            market_data_latency_ns: config.market_data_latency_ns as i64,
            ack_latency_ns: config.ack_latency_ns as i64,
            jitter_model,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Create with custom jitter model
    #[must_use]
    pub fn with_jitter_model(mut self, model: JitterModel) -> Self {
        self.jitter_model = model;
        self
    }

    /// Create with custom seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Get jitter based on the model
    fn get_jitter(&mut self) -> i64 {
        match &self.jitter_model {
            JitterModel::None => 0,
            JitterModel::Uniform { max_jitter_ns } => {
                self.rng.gen_range(-max_jitter_ns..=*max_jitter_ns)
            }
            JitterModel::Normal { std_dev_ns } => {
                let dist = Normal::new(0.0, *std_dev_ns).unwrap();
                dist.sample(&mut self.rng) as i64
            }
            JitterModel::LogNormal { mu, sigma } => {
                let dist = LogNormal::new(*mu, *sigma).unwrap();
                dist.sample(&mut self.rng) as i64
            }
            JitterModel::Empirical { percentiles } => {
                // Simple interpolation between percentiles
                let r: f64 = self.rng.gen();
                if r < 0.50 {
                    percentiles[0]
                } else if r < 0.75 {
                    percentiles[1]
                } else if r < 0.90 {
                    percentiles[2]
                } else if r < 0.95 {
                    percentiles[3]
                } else {
                    percentiles[4]
                }
            }
        }
    }

    /// Calculate order arrival time at exchange
    pub fn order_arrival_time(&mut self, submit_time: Timestamp) -> Timestamp {
        let latency = (self.order_latency_ns + self.get_jitter()).max(0);
        submit_time.add_nanos(latency)
    }

    /// Calculate market data reception time
    pub fn market_data_reception_time(&mut self, exchange_time: Timestamp) -> Timestamp {
        let latency = (self.market_data_latency_ns + self.get_jitter()).max(0);
        exchange_time.add_nanos(latency)
    }

    /// Calculate order acknowledgment reception time
    pub fn ack_reception_time(&mut self, exchange_ack_time: Timestamp) -> Timestamp {
        let latency = (self.ack_latency_ns + self.get_jitter()).max(0);
        exchange_ack_time.add_nanos(latency)
    }

    /// Calculate fill notification reception time
    pub fn fill_notification_time(&mut self, exchange_fill_time: Timestamp) -> Timestamp {
        // Fill notifications use same path as acks
        let latency = (self.ack_latency_ns + self.get_jitter()).max(0);
        exchange_fill_time.add_nanos(latency)
    }

    /// Get raw order latency without jitter
    #[must_use]
    pub fn base_order_latency(&self) -> i64 {
        self.order_latency_ns
    }

    /// Get raw market data latency without jitter
    #[must_use]
    pub fn base_market_data_latency(&self) -> i64 {
        self.market_data_latency_ns
    }
}

impl LatencyModel for LatencySimulator {
    fn order_latency(&self) -> i64 {
        self.order_latency_ns
    }

    fn market_data_latency(&self) -> i64 {
        self.market_data_latency_ns
    }

    fn ack_latency(&self) -> i64 {
        self.ack_latency_ns
    }

    fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(42);
    }
}

/// Realistic colo latency model based on empirical HFT data
#[derive(Debug)]
pub struct ColoLatencyModel {
    /// Colo-to-exchange latency in nanoseconds
    colo_to_exchange_ns: i64,
    /// Exchange processing time in nanoseconds
    exchange_processing_ns: i64,
    /// Network jitter model
    network_jitter: JitterModel,
    /// Exchange jitter (processing time variance)
    exchange_jitter_ns: i64,
    /// RNG
    rng: StdRng,
}

impl ColoLatencyModel {
    /// Create a new colo latency model
    ///
    /// # Arguments
    /// * `colo_to_exchange_ns` - One-way latency from colo to exchange
    /// * `exchange_processing_ns` - Exchange internal processing time
    #[must_use]
    pub fn new(colo_to_exchange_ns: i64, exchange_processing_ns: i64) -> Self {
        Self {
            colo_to_exchange_ns,
            exchange_processing_ns,
            network_jitter: JitterModel::LogNormal {
                mu: 0.0,
                sigma: 0.3,
            },
            exchange_jitter_ns: 1000, // 1 microsecond
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Create model for Aurora (CME primary colo)
    #[must_use]
    pub fn aurora() -> Self {
        Self::new(
            5_000, // 5 microseconds to exchange
            2_000, // 2 microseconds exchange processing
        )
    }

    /// Create model for NY5 (generic NY colo)
    #[must_use]
    pub fn ny5() -> Self {
        Self::new(
            50_000, // 50 microseconds
            5_000,  // 5 microseconds
        )
    }

    /// Get total round-trip time estimate
    #[must_use]
    pub fn round_trip_estimate(&self) -> i64 {
        2 * self.colo_to_exchange_ns + self.exchange_processing_ns
    }
}

impl LatencyModel for ColoLatencyModel {
    fn order_latency(&self) -> i64 {
        self.colo_to_exchange_ns
    }

    fn market_data_latency(&self) -> i64 {
        self.colo_to_exchange_ns
    }

    fn ack_latency(&self) -> i64 {
        self.colo_to_exchange_ns + self.exchange_processing_ns
    }

    fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(42);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LatencyConfig;

    #[test]
    fn test_latency_simulator() {
        let config = LatencyConfig::default();
        let mut sim = LatencySimulator::from_config(&config);

        let t0 = Timestamp::from_nanos(1_000_000_000);
        let t1 = sim.order_arrival_time(t0);

        assert!(t1 > t0);
        assert!(t1.as_nanos() - t0.as_nanos() >= config.order_latency_ns as i64);
    }

    #[test]
    fn test_no_jitter() {
        let config = LatencyConfig {
            order_latency_ns: 100_000,
            use_random_jitter: false,
            ..Default::default()
        };
        let mut sim = LatencySimulator::from_config(&config);

        let t0 = Timestamp::from_nanos(0);

        // Without jitter, latency should be deterministic
        let t1 = sim.order_arrival_time(t0);
        let t2 = sim.order_arrival_time(t0);

        assert_eq!(t1, t2);
        assert_eq!(t1.as_nanos(), 100_000);
    }

    #[test]
    fn test_colo_model() {
        let model = ColoLatencyModel::aurora();

        assert!(model.order_latency() < 10_000); // < 10 microseconds
        assert!(model.round_trip_estimate() < 20_000); // < 20 microseconds RTT
    }

    #[test]
    fn test_jitter_models() {
        let mut sim = LatencySimulator::from_config(&LatencyConfig::default()).with_jitter_model(
            JitterModel::Uniform {
                max_jitter_ns: 1000,
            },
        );

        let t0 = Timestamp::from_nanos(0);
        let mut latencies = Vec::new();

        for _ in 0..100 {
            let t1 = sim.order_arrival_time(t0);
            latencies.push(t1.as_nanos());
        }

        // Should have some variance
        let min = *latencies.iter().min().unwrap();
        let max = *latencies.iter().max().unwrap();
        assert!(max - min > 0);
    }
}
