//! Signal generation from model predictions.

use nano_core::types::{Price, Side, Timestamp};
use serde::{Deserialize, Serialize};

use crate::inference::Prediction;

/// Trading signal generated from model prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal direction: -1 (sell), 0 (neutral), +1 (buy)
    pub direction: i8,
    /// Signal strength (0 to 1)
    pub strength: f32,
    /// Confidence from model
    pub confidence: f32,
    /// Target price (if applicable)
    pub target_price: Option<Price>,
    /// Stop price (if applicable)
    pub stop_price: Option<Price>,
    /// Suggested position size (as fraction of max)
    pub position_size: f32,
    /// Signal timestamp
    pub timestamp: Timestamp,
    /// Time horizon in ticks
    pub horizon_ticks: u32,
    /// Model prediction latency (ns)
    pub latency_ns: u64,
}

impl Signal {
    /// Create a buy signal
    #[must_use]
    pub fn buy(strength: f32, confidence: f32, timestamp: Timestamp) -> Self {
        Self {
            direction: 1,
            strength,
            confidence,
            target_price: None,
            stop_price: None,
            position_size: strength,
            timestamp,
            horizon_ticks: 100,
            latency_ns: 0,
        }
    }

    /// Create a sell signal
    #[must_use]
    pub fn sell(strength: f32, confidence: f32, timestamp: Timestamp) -> Self {
        Self {
            direction: -1,
            strength,
            confidence,
            target_price: None,
            stop_price: None,
            position_size: strength,
            timestamp,
            horizon_ticks: 100,
            latency_ns: 0,
        }
    }

    /// Create a neutral (no trade) signal
    #[must_use]
    pub fn neutral(timestamp: Timestamp) -> Self {
        Self {
            direction: 0,
            strength: 0.0,
            confidence: 1.0,
            target_price: None,
            stop_price: None,
            position_size: 0.0,
            timestamp,
            horizon_ticks: 0,
            latency_ns: 0,
        }
    }

    /// Check if signal suggests buying
    #[must_use]
    pub fn is_buy(&self) -> bool {
        self.direction > 0
    }

    /// Check if signal suggests selling
    #[must_use]
    pub fn is_sell(&self) -> bool {
        self.direction < 0
    }

    /// Check if signal is neutral
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        self.direction == 0
    }

    /// Get the side for order placement
    #[must_use]
    pub fn side(&self) -> Option<Side> {
        match self.direction {
            d if d > 0 => Some(Side::Buy),
            d if d < 0 => Some(Side::Sell),
            _ => None,
        }
    }

    /// Set target price
    #[must_use]
    pub fn with_target(mut self, target: Price) -> Self {
        self.target_price = Some(target);
        self
    }

    /// Set stop price
    #[must_use]
    pub fn with_stop(mut self, stop: Price) -> Self {
        self.stop_price = Some(stop);
        self
    }

    /// Set position size
    #[must_use]
    pub fn with_size(mut self, size: f32) -> Self {
        self.position_size = size.clamp(0.0, 1.0);
        self
    }

    /// Set latency
    #[must_use]
    pub fn with_latency(mut self, latency_ns: u64) -> Self {
        self.latency_ns = latency_ns;
        self
    }
}

/// Signal generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Minimum confidence threshold for signal generation
    pub min_confidence: f32,
    /// Minimum prediction magnitude for signal
    pub min_magnitude: f32,
    /// Position sizing based on confidence
    pub confidence_scaling: bool,
    /// Maximum position size (as fraction)
    pub max_position_size: f32,
    /// Use ensemble averaging for horizons
    pub ensemble_horizons: bool,
    /// Target profit in ticks
    pub target_ticks: i64,
    /// Stop loss in ticks
    pub stop_ticks: i64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.55,
            min_magnitude: 0.001,
            confidence_scaling: true,
            max_position_size: 1.0,
            ensemble_horizons: false,
            target_ticks: 10,
            stop_ticks: 5,
        }
    }
}

/// Signal generator from model predictions
#[derive(Debug)]
pub struct SignalGenerator {
    /// Configuration
    config: SignalConfig,
    /// Last signal generated
    last_signal: Option<Signal>,
    /// Signal count
    signal_count: u64,
}

impl SignalGenerator {
    /// Create a new signal generator
    #[must_use]
    pub fn new(config: SignalConfig) -> Self {
        Self {
            config,
            last_signal: None,
            signal_count: 0,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(SignalConfig::default())
    }

    /// Generate signal from model prediction
    pub fn generate(
        &mut self,
        prediction: &Prediction,
        timestamp: Timestamp,
        current_price: Price,
    ) -> Signal {
        // Get direction and confidence
        let (direction, confidence) = if self.config.ensemble_horizons {
            // Average across horizons
            let avg_dir: f32 = prediction
                .directions
                .iter()
                .map(|&d| f32::from(d))
                .sum::<f32>()
                / prediction.directions.len() as f32;
            let avg_conf: f32 =
                prediction.confidences.iter().sum::<f32>() / prediction.confidences.len() as f32;

            let dir = if avg_dir > 0.5 {
                1i8
            } else if avg_dir < -0.5 {
                -1i8
            } else {
                0i8
            };

            (dir, avg_conf)
        } else {
            // Use primary horizon
            (
                prediction.primary_direction(),
                prediction.primary_confidence(),
            )
        };

        // Check thresholds
        if confidence < self.config.min_confidence {
            return Signal::neutral(timestamp);
        }

        if direction == 0 {
            return Signal::neutral(timestamp);
        }

        // Calculate position size
        let position_size = if self.config.confidence_scaling {
            ((confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence))
                .min(self.config.max_position_size)
        } else {
            self.config.max_position_size
        };

        // Calculate strength (normalized confidence above threshold)
        let strength =
            (confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence);

        // Calculate target and stop prices
        let target_delta = self.config.target_ticks * i64::from(direction);
        let stop_delta = -self.config.stop_ticks * i64::from(direction);

        let target_price = Price::from_raw(current_price.raw() + target_delta);
        let stop_price = Price::from_raw(current_price.raw() + stop_delta);

        let signal = Signal {
            direction,
            strength,
            confidence,
            target_price: Some(target_price),
            stop_price: Some(stop_price),
            position_size,
            timestamp,
            horizon_ticks: 100,
            latency_ns: prediction.latency_ns,
        };

        self.last_signal = Some(signal.clone());
        self.signal_count += 1;

        signal
    }

    /// Get the last generated signal
    #[must_use]
    pub fn last_signal(&self) -> Option<&Signal> {
        self.last_signal.as_ref()
    }

    /// Get signal count
    #[must_use]
    pub fn signal_count(&self) -> u64 {
        self.signal_count
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &SignalConfig {
        &self.config
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.last_signal = None;
        self.signal_count = 0;
    }
}

/// Signal filter for reducing noise
#[derive(Debug)]
pub struct SignalFilter {
    /// Minimum time between signals (ns)
    min_interval_ns: i64,
    /// Last signal timestamp
    last_signal_time: Option<Timestamp>,
    /// Consecutive same-direction signals required
    consecutive_required: usize,
    /// Recent signal directions
    recent_directions: Vec<i8>,
}

impl SignalFilter {
    /// Create a new signal filter
    #[must_use]
    pub fn new(min_interval_ns: i64, consecutive_required: usize) -> Self {
        Self {
            min_interval_ns,
            last_signal_time: None,
            consecutive_required,
            recent_directions: Vec::with_capacity(consecutive_required),
        }
    }

    /// Filter a signal
    pub fn filter(&mut self, signal: &Signal) -> Option<Signal> {
        // Check time interval
        if let Some(last_time) = self.last_signal_time {
            if signal.timestamp.as_nanos() - last_time.as_nanos() < self.min_interval_ns {
                return None;
            }
        }

        // Track recent directions for consecutive filter
        self.recent_directions.push(signal.direction);
        if self.recent_directions.len() > self.consecutive_required {
            self.recent_directions.remove(0);
        }

        // Check consecutive requirement
        if self.consecutive_required > 1 {
            if self.recent_directions.len() < self.consecutive_required {
                return None;
            }

            let first = self.recent_directions[0];
            if first == 0 || !self.recent_directions.iter().all(|&d| d == first) {
                return None;
            }
        }

        self.last_signal_time = Some(signal.timestamp);
        Some(signal.clone())
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.last_signal_time = None;
        self.recent_directions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let buy = Signal::buy(0.8, 0.75, Timestamp::now());
        assert!(buy.is_buy());
        assert_eq!(buy.direction, 1);
        assert_eq!(buy.side(), Some(Side::Buy));

        let sell = Signal::sell(0.7, 0.65, Timestamp::now());
        assert!(sell.is_sell());
        assert_eq!(sell.side(), Some(Side::Sell));

        let neutral = Signal::neutral(Timestamp::now());
        assert!(neutral.is_neutral());
        assert_eq!(neutral.side(), None);
    }

    #[test]
    fn test_signal_generator() {
        let config = SignalConfig {
            min_confidence: 0.5,
            ..Default::default()
        };
        let mut generator = SignalGenerator::new(config);

        let prediction = Prediction {
            directions: vec![1, 1, 1],
            confidences: vec![0.7, 0.6, 0.65],
            raw_output: vec![],
            latency_ns: 100,
        };

        let signal = generator.generate(&prediction, Timestamp::now(), Price::from_raw(50000));

        assert!(signal.is_buy());
        assert!(signal.confidence >= 0.5);
        assert!(signal.target_price.is_some());
        assert!(signal.stop_price.is_some());
    }

    #[test]
    fn test_signal_filter_interval() {
        let mut filter = SignalFilter::new(1_000_000, 1); // 1ms minimum interval

        let signal1 = Signal::buy(0.8, 0.7, Timestamp::from_nanos(1_000_000_000));
        let result1 = filter.filter(&signal1);
        assert!(result1.is_some());

        // Too soon - should be filtered
        let signal2 = Signal::buy(0.8, 0.7, Timestamp::from_nanos(1_000_500_000));
        let result2 = filter.filter(&signal2);
        assert!(result2.is_none());

        // After interval - should pass
        let signal3 = Signal::buy(0.8, 0.7, Timestamp::from_nanos(1_002_000_000));
        let result3 = filter.filter(&signal3);
        assert!(result3.is_some());
    }
}
