//! Cross-validation and walk-forward analysis.

use std::ops::Range;

use nano_core::types::Timestamp;
use serde::{Deserialize, Serialize};

/// Time-based data split for walk-forward analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSplit {
    /// Training period start
    pub train_start: Timestamp,
    /// Training period end
    pub train_end: Timestamp,
    /// Test period start
    pub test_start: Timestamp,
    /// Test period end
    pub test_end: Timestamp,
    /// Split index
    pub index: usize,
}

impl TimeSplit {
    /// Get training duration in nanoseconds
    #[must_use]
    pub fn train_duration_ns(&self) -> i64 {
        self.train_end.as_nanos() - self.train_start.as_nanos()
    }

    /// Get test duration in nanoseconds
    #[must_use]
    pub fn test_duration_ns(&self) -> i64 {
        self.test_end.as_nanos() - self.test_start.as_nanos()
    }

    /// Check if a timestamp is in the training period
    #[must_use]
    pub fn in_train(&self, ts: Timestamp) -> bool {
        ts >= self.train_start && ts < self.train_end
    }

    /// Check if a timestamp is in the test period
    #[must_use]
    pub fn in_test(&self, ts: Timestamp) -> bool {
        ts >= self.test_start && ts < self.test_end
    }
}

/// Walk-forward analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    /// Number of folds
    pub num_folds: usize,
    /// Training window size (nanoseconds)
    pub train_window_ns: i64,
    /// Test window size (nanoseconds)
    pub test_window_ns: i64,
    /// Step size between windows (nanoseconds)
    pub step_ns: i64,
    /// Purge gap between train and test (nanoseconds)
    pub purge_gap_ns: i64,
    /// Embargo period after test (nanoseconds)
    pub embargo_ns: i64,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        // Default: 30 day train, 5 day test, 5 day step, 1 day purge
        let day_ns = 24 * 60 * 60 * 1_000_000_000_i64;
        Self {
            num_folds: 10,
            train_window_ns: 30 * day_ns,
            test_window_ns: 5 * day_ns,
            step_ns: 5 * day_ns,
            purge_gap_ns: day_ns,
            embargo_ns: 0,
        }
    }
}

/// Walk-forward analysis generator
#[derive(Debug)]
pub struct WalkForwardAnalysis {
    /// Configuration
    config: WalkForwardConfig,
    /// Data start timestamp
    data_start: Timestamp,
    /// Data end timestamp
    data_end: Timestamp,
}

impl WalkForwardAnalysis {
    /// Create a new walk-forward analysis
    #[must_use]
    pub fn new(config: WalkForwardConfig, data_start: Timestamp, data_end: Timestamp) -> Self {
        Self {
            config,
            data_start,
            data_end,
        }
    }

    /// Generate time splits for walk-forward analysis
    #[must_use]
    pub fn generate_splits(&self) -> Vec<TimeSplit> {
        let mut splits = Vec::new();
        let mut current = self.data_start.as_nanos();
        let end = self.data_end.as_nanos();

        let mut index = 0;

        while current
            + self.config.train_window_ns
            + self.config.purge_gap_ns
            + self.config.test_window_ns
            <= end
        {
            let train_start = Timestamp::from_nanos(current);
            let train_end = Timestamp::from_nanos(current + self.config.train_window_ns);
            let test_start = Timestamp::from_nanos(
                current + self.config.train_window_ns + self.config.purge_gap_ns,
            );
            let test_end = Timestamp::from_nanos(
                current
                    + self.config.train_window_ns
                    + self.config.purge_gap_ns
                    + self.config.test_window_ns,
            );

            splits.push(TimeSplit {
                train_start,
                train_end,
                test_start,
                test_end,
                index,
            });

            current += self.config.step_ns;
            index += 1;

            if splits.len() >= self.config.num_folds {
                break;
            }
        }

        splits
    }
}

/// Purged K-Fold cross-validation for time series
#[derive(Debug)]
pub struct PurgedKFold {
    /// Number of folds
    n_splits: usize,
    /// Purge gap in samples
    purge_samples: usize,
    /// Embargo samples
    embargo_samples: usize,
}

impl PurgedKFold {
    /// Create a new purged K-fold splitter
    #[must_use]
    pub fn new(n_splits: usize, purge_samples: usize, embargo_samples: usize) -> Self {
        Self {
            n_splits,
            purge_samples,
            embargo_samples,
        }
    }

    /// Generate train/test indices for each fold
    #[must_use]
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let fold_size = n_samples / self.n_splits;
        let mut folds = Vec::with_capacity(self.n_splits);

        for fold in 0..self.n_splits {
            let test_start = fold * fold_size;
            let test_end = if fold == self.n_splits - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Test indices
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Train indices with purge and embargo
            let purge_start = test_start.saturating_sub(self.purge_samples);
            let embargo_end = (test_end + self.embargo_samples).min(n_samples);

            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| i < purge_start || i >= embargo_end)
                .collect();

            folds.push((train_indices, test_indices));
        }

        folds
    }
}

/// Combinatorial Purged Cross-Validation (CPCV)
#[derive(Debug)]
pub struct CombinatorialPurgedCV {
    /// Number of groups
    n_groups: usize,
    /// Number of test groups per split
    n_test_groups: usize,
    /// Purge samples
    purge_samples: usize,
    /// Embargo samples
    embargo_samples: usize,
}

impl CombinatorialPurgedCV {
    /// Create a new CPCV splitter
    #[must_use]
    pub fn new(
        n_groups: usize,
        n_test_groups: usize,
        purge_samples: usize,
        embargo_samples: usize,
    ) -> Self {
        Self {
            n_groups,
            n_test_groups,
            purge_samples,
            embargo_samples,
        }
    }

    /// Generate all combinations of test groups
    fn combinations(&self) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut combination = vec![0; self.n_test_groups];

        fn generate(
            n: usize,
            k: usize,
            start: usize,
            combination: &mut Vec<usize>,
            pos: usize,
            result: &mut Vec<Vec<usize>>,
        ) {
            if pos == k {
                result.push(combination.clone());
                return;
            }

            for i in start..=(n - k + pos) {
                combination[pos] = i;
                generate(n, k, i + 1, combination, pos + 1, result);
            }
        }

        generate(
            self.n_groups,
            self.n_test_groups,
            0,
            &mut combination,
            0,
            &mut result,
        );

        result
    }

    /// Generate train/test indices for each combination
    #[must_use]
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let group_size = n_samples / self.n_groups;
        let combinations = self.combinations();
        let mut folds = Vec::with_capacity(combinations.len());

        for test_groups in combinations {
            // Determine test ranges
            let test_ranges: Vec<Range<usize>> = test_groups
                .iter()
                .map(|&g| {
                    let start = g * group_size;
                    let end = if g == self.n_groups - 1 {
                        n_samples
                    } else {
                        (g + 1) * group_size
                    };
                    start..end
                })
                .collect();

            // Test indices
            let test_indices: Vec<usize> = test_ranges
                .iter()
                .flat_map(std::clone::Clone::clone)
                .collect();

            // Train indices with purge and embargo around each test range
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| {
                    for range in &test_ranges {
                        let purge_start = range.start.saturating_sub(self.purge_samples);
                        let embargo_end = (range.end + self.embargo_samples).min(n_samples);

                        if i >= purge_start && i < embargo_end {
                            return false;
                        }
                    }
                    true
                })
                .collect();

            folds.push((train_indices, test_indices));
        }

        folds
    }

    /// Get number of splits
    #[must_use]
    pub fn n_splits(&self) -> usize {
        self.combinations().len()
    }
}

/// Results from a single validation fold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    /// Fold index
    pub fold: usize,
    /// Training Sharpe ratio
    pub train_sharpe: f64,
    /// Test Sharpe ratio
    pub test_sharpe: f64,
    /// Training P&L
    pub train_pnl: f64,
    /// Test P&L
    pub test_pnl: f64,
    /// Training max drawdown
    pub train_max_dd: f64,
    /// Test max drawdown
    pub test_max_dd: f64,
}

/// Aggregated validation results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Individual fold results
    pub folds: Vec<FoldResult>,
    /// Average test Sharpe
    pub avg_test_sharpe: f64,
    /// Std dev of test Sharpe
    pub std_test_sharpe: f64,
    /// Average test P&L
    pub avg_test_pnl: f64,
    /// Average overfitting ratio (train Sharpe / test Sharpe)
    pub overfit_ratio: f64,
    /// Probability of overfitting (% of folds where test < train)
    pub prob_overfit: f64,
}

impl ValidationResults {
    /// Calculate aggregate statistics from fold results
    pub fn calculate(&mut self) {
        if self.folds.is_empty() {
            return;
        }

        let n = self.folds.len() as f64;

        // Average test Sharpe
        self.avg_test_sharpe = self.folds.iter().map(|f| f.test_sharpe).sum::<f64>() / n;

        // Std dev of test Sharpe
        let variance = self
            .folds
            .iter()
            .map(|f| (f.test_sharpe - self.avg_test_sharpe).powi(2))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        self.std_test_sharpe = variance.sqrt();

        // Average test P&L
        self.avg_test_pnl = self.folds.iter().map(|f| f.test_pnl).sum::<f64>() / n;

        // Overfitting metrics
        let avg_train_sharpe = self.folds.iter().map(|f| f.train_sharpe).sum::<f64>() / n;
        self.overfit_ratio = if self.avg_test_sharpe.abs() > f64::EPSILON {
            avg_train_sharpe / self.avg_test_sharpe
        } else {
            f64::INFINITY
        };

        let overfit_count = self
            .folds
            .iter()
            .filter(|f| f.test_sharpe < f.train_sharpe)
            .count();
        self.prob_overfit = overfit_count as f64 / n;
    }

    /// Add a fold result
    pub fn add_fold(&mut self, result: FoldResult) {
        self.folds.push(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_forward_splits() {
        let config = WalkForwardConfig {
            num_folds: 5,
            train_window_ns: 10_000_000_000,
            test_window_ns: 2_000_000_000,
            step_ns: 2_000_000_000,
            purge_gap_ns: 500_000_000,
            embargo_ns: 0,
        };

        let analysis = WalkForwardAnalysis::new(
            config,
            Timestamp::from_nanos(0),
            Timestamp::from_nanos(50_000_000_000),
        );

        let splits = analysis.generate_splits();

        assert_eq!(splits.len(), 5);

        // Verify no overlap between train and test
        for split in &splits {
            assert!(split.train_end <= split.test_start);
        }
    }

    #[test]
    fn test_purged_kfold() {
        let splitter = PurgedKFold::new(5, 10, 5);
        let folds = splitter.split(100);

        assert_eq!(folds.len(), 5);

        for (train, test) in &folds {
            // Train and test should not overlap
            for &t in test {
                assert!(!train.contains(&t));
            }
        }
    }

    #[test]
    fn test_combinatorial_cv() {
        let splitter = CombinatorialPurgedCV::new(5, 2, 5, 5);
        let n_splits = splitter.n_splits();

        // C(5,2) = 10 combinations
        assert_eq!(n_splits, 10);

        let folds = splitter.split(100);
        assert_eq!(folds.len(), 10);
    }

    #[test]
    fn test_validation_results() {
        let mut results = ValidationResults::default();

        for i in 0..5 {
            results.add_fold(FoldResult {
                fold: i,
                train_sharpe: 2.0 + (i as f64 * 0.1),
                test_sharpe: 1.5 + (i as f64 * 0.1),
                train_pnl: 10000.0,
                test_pnl: 5000.0,
                train_max_dd: 0.05,
                test_max_dd: 0.07,
            });
        }

        results.calculate();

        assert!(results.avg_test_sharpe > 0.0);
        assert!(results.overfit_ratio > 1.0); // Train > Test implies overfitting
        assert!(results.prob_overfit > 0.0);
    }
}
