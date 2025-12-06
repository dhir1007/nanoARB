//! Feature preprocessing for ML models.

use nano_core::constants::{FEATURE_LEVELS, FIELDS_PER_LEVEL};
use nano_lob::{LobSnapshot, OrderBook, SnapshotRingBuffer};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// Feature preprocessor for LOB data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePreprocessor {
    /// Feature means for normalization
    means: Vec<f32>,
    /// Feature standard deviations for normalization
    stds: Vec<f32>,
    /// Whether to use z-score normalization
    use_zscore: bool,
    /// Sequence length for temporal features
    sequence_length: usize,
    /// Number of features per timestep
    num_features: usize,
}

impl Default for FeaturePreprocessor {
    fn default() -> Self {
        Self::new(100, FEATURE_LEVELS * FIELDS_PER_LEVEL)
    }
}

impl FeaturePreprocessor {
    /// Create a new preprocessor
    #[must_use]
    pub fn new(sequence_length: usize, num_features: usize) -> Self {
        Self {
            means: vec![0.0; num_features],
            stds: vec![1.0; num_features],
            use_zscore: true,
            sequence_length,
            num_features,
        }
    }

    /// Create with pre-computed statistics
    #[must_use]
    pub fn with_stats(
        sequence_length: usize,
        num_features: usize,
        means: Vec<f32>,
        stds: Vec<f32>,
    ) -> Self {
        Self {
            means,
            stds,
            use_zscore: true,
            sequence_length,
            num_features,
        }
    }

    /// Disable z-score normalization
    #[must_use]
    pub fn without_normalization(mut self) -> Self {
        self.use_zscore = false;
        self
    }

    /// Fit statistics from data
    pub fn fit(&mut self, data: &[Array2<f32>]) {
        if data.is_empty() {
            return;
        }

        let n_samples = data.len();
        let n_features = data[0].ncols();

        // Calculate means
        let mut sums = vec![0.0f64; n_features];
        for sample in data {
            for (i, &val) in sample.iter().enumerate() {
                sums[i % n_features] += f64::from(val);
            }
        }

        let total_rows: usize = data.iter().map(ndarray::ArrayBase::nrows).sum();
        self.means = sums
            .iter()
            .map(|s| (*s / total_rows as f64) as f32)
            .collect();

        // Calculate standard deviations
        let mut sq_diffs = vec![0.0f64; n_features];
        for sample in data {
            for (i, &val) in sample.iter().enumerate() {
                let idx = i % n_features;
                let diff = f64::from(val) - f64::from(self.means[idx]);
                sq_diffs[idx] += diff * diff;
            }
        }

        self.stds = sq_diffs
            .iter()
            .map(|s| ((*s / total_rows as f64).sqrt().max(1e-8)) as f32)
            .collect();

        self.num_features = n_features;
    }

    /// Transform a single snapshot to normalized features
    #[must_use]
    pub fn transform_snapshot(&self, snapshot: &LobSnapshot) -> Array2<f32> {
        let mut features = snapshot.tensor.clone();

        if self.use_zscore {
            for (i, val) in features.iter_mut().enumerate() {
                let idx = i % self.num_features.min(self.means.len());
                *val = (*val - self.means[idx]) / self.stds[idx];
            }
        }

        features
    }

    /// Transform a sequence of snapshots to model input
    #[must_use]
    pub fn transform_sequence(&self, buffer: &SnapshotRingBuffer) -> Option<Array3<f32>> {
        let tensor = buffer.to_tensor(self.sequence_length)?;

        if !self.use_zscore {
            return Some(tensor);
        }

        // Normalize each feature
        let mut normalized = tensor;
        let shape = normalized.shape().to_vec();

        for t in 0..shape[0] {
            for l in 0..shape[1] {
                for f in 0..shape[2] {
                    let idx = l * shape[2] + f;
                    if idx < self.means.len() {
                        normalized[[t, l, f]] =
                            (normalized[[t, l, f]] - self.means[idx]) / self.stds[idx];
                    }
                }
            }
        }

        Some(normalized)
    }

    /// Transform a single order book to features
    #[must_use]
    pub fn transform_book(&self, book: &OrderBook) -> Array2<f32> {
        let snapshot = LobSnapshot::from_book(book);
        self.transform_snapshot(&snapshot)
    }

    /// Create input tensor for model with batch dimension
    #[must_use]
    pub fn to_model_input(&self, buffer: &SnapshotRingBuffer) -> Option<Array3<f32>> {
        // Shape: (sequence, levels, features)
        self.transform_sequence(buffer)
    }

    /// Get the expected input shape for the model
    #[must_use]
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (self.sequence_length, FEATURE_LEVELS, FIELDS_PER_LEVEL)
    }

    /// Get means
    #[must_use]
    pub fn means(&self) -> &[f32] {
        &self.means
    }

    /// Get standard deviations
    #[must_use]
    pub fn stds(&self) -> &[f32] {
        &self.stds
    }

    /// Calculate additional derived features
    #[must_use]
    pub fn calculate_derived_features(&self, book: &OrderBook) -> Vec<f32> {
        use nano_lob::LobFeatureExtractor;

        let extractor = LobFeatureExtractor::new();
        let features = extractor.extract(book);

        vec![
            features.microprice as f32,
            features.weighted_mid as f32,
            features.spread as f32,
            features.imbalance_l1 as f32,
            features.imbalance_total as f32,
            features.bid_depth as f32,
            features.ask_depth as f32,
        ]
    }
}

/// Feature statistics tracker for online normalization
#[derive(Debug)]
pub struct OnlineStatistics {
    /// Running count
    count: u64,
    /// Running means
    means: Vec<f64>,
    /// Running M2 (for variance calculation)
    m2: Vec<f64>,
}

impl OnlineStatistics {
    /// Create a new online statistics tracker
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self {
            count: 0,
            means: vec![0.0; num_features],
            m2: vec![0.0; num_features],
        }
    }

    /// Update with a new sample (Welford's algorithm)
    pub fn update(&mut self, sample: &[f32]) {
        self.count += 1;
        let n = self.count as f64;

        for (i, &val) in sample.iter().enumerate() {
            if i >= self.means.len() {
                break;
            }

            let val = f64::from(val);
            let delta = val - self.means[i];
            self.means[i] += delta / n;
            let delta2 = val - self.means[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Get current means
    #[must_use]
    pub fn means(&self) -> Vec<f32> {
        self.means.iter().map(|&m| m as f32).collect()
    }

    /// Get current standard deviations
    #[must_use]
    pub fn stds(&self) -> Vec<f32> {
        if self.count < 2 {
            return vec![1.0; self.means.len()];
        }

        self.m2
            .iter()
            .map(|&m| ((m / (self.count - 1) as f64).sqrt().max(1e-8)) as f32)
            .collect()
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Create preprocessor from current statistics
    #[must_use]
    pub fn to_preprocessor(&self, sequence_length: usize) -> FeaturePreprocessor {
        FeaturePreprocessor::with_stats(
            sequence_length,
            self.means.len(),
            self.means(),
            self.stds(),
        )
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.count = 0;
        for m in &mut self.means {
            *m = 0.0;
        }
        for m in &mut self.m2 {
            *m = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_default() {
        let prep = FeaturePreprocessor::default();
        assert_eq!(prep.sequence_length, 100);
    }

    #[test]
    fn test_online_statistics() {
        let mut stats = OnlineStatistics::new(4);

        // Add samples
        stats.update(&[1.0, 2.0, 3.0, 4.0]);
        stats.update(&[2.0, 4.0, 6.0, 8.0]);
        stats.update(&[3.0, 6.0, 9.0, 12.0]);

        let means = stats.means();
        assert!((means[0] - 2.0).abs() < 0.01);
        assert!((means[1] - 4.0).abs() < 0.01);

        assert_eq!(stats.count(), 3);
    }

    #[test]
    fn test_input_shape() {
        let prep = FeaturePreprocessor::new(100, 40);
        let shape = prep.input_shape();

        assert_eq!(shape.0, 100); // sequence
        assert_eq!(shape.1, FEATURE_LEVELS);
        assert_eq!(shape.2, FIELDS_PER_LEVEL);
    }
}
