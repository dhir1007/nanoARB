//! ONNX model inference with low latency.

use std::path::Path;
use std::time::Instant;

use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use nano_core::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    /// Input shape (batch, sequence, features)
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of prediction horizons
    pub num_horizons: usize,
    /// Feature normalization means
    pub feature_means: Vec<f32>,
    /// Feature normalization stds
    pub feature_stds: Vec<f32>,
    /// Use quantized model
    pub quantized: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            input_shape: vec![1, 100, 44],
            output_shape: vec![1, 3],
            num_horizons: 3,
            feature_means: vec![0.0; 44],
            feature_stds: vec![1.0; 44],
            quantized: false,
        }
    }
}

/// Model prediction output
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Direction predictions (-1, 0, +1) for each horizon
    pub directions: Vec<i8>,
    /// Confidence scores (0 to 1) for each horizon
    pub confidences: Vec<f32>,
    /// Raw logits/probabilities
    pub raw_output: Vec<f32>,
    /// Inference latency in nanoseconds
    pub latency_ns: u64,
}

impl Prediction {
    /// Get the primary prediction (first horizon)
    #[must_use]
    pub fn primary_direction(&self) -> i8 {
        self.directions.first().copied().unwrap_or(0)
    }

    /// Get the primary confidence
    #[must_use]
    pub fn primary_confidence(&self) -> f32 {
        self.confidences.first().copied().unwrap_or(0.0)
    }

    /// Check if prediction is bullish
    #[must_use]
    pub fn is_bullish(&self) -> bool {
        self.primary_direction() > 0
    }

    /// Check if prediction is bearish
    #[must_use]
    pub fn is_bearish(&self) -> bool {
        self.primary_direction() < 0
    }

    /// Check if prediction is neutral
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        self.primary_direction() == 0
    }
}

/// ONNX model wrapper for inference
pub struct OnnxModel {
    /// Model configuration
    config: ModelConfig,
    /// ONNX Runtime session
    #[cfg(feature = "onnx")]
    session: ort::Session,
    /// Input name
    input_name: String,
    /// Inference count
    inference_count: u64,
    /// Total inference time (ns)
    total_inference_time_ns: u64,
}

impl OnnxModel {
    /// Load a model from an ONNX file
    #[cfg(feature = "onnx")]
    pub fn load<P: AsRef<Path>>(path: P, config: ModelConfig) -> Result<Self> {
        use ort::{GraphOptimizationLevel, Session};

        let session = Session::builder()
            .map_err(|e| Error::ModelError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::ModelError(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| Error::ModelError(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| Error::ModelError(e.to_string()))?;

        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_string());

        Ok(Self {
            config,
            session,
            input_name,
            inference_count: 0,
            total_inference_time_ns: 0,
        })
    }

    /// Create a dummy model for testing (no ONNX runtime)
    #[must_use]
    pub fn dummy(config: ModelConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "onnx")]
            session: unimplemented!("Dummy model has no session"),
            input_name: "input".to_string(),
            inference_count: 0,
            total_inference_time_ns: 0,
        }
    }

    /// Run inference on preprocessed features
    #[cfg(feature = "onnx")]
    pub fn predict(&mut self, features: &Array3<f32>) -> Result<Prediction> {
        use ort::Value;

        let start = Instant::now();

        // Convert to dynamic array for ONNX
        let input_array = features.clone().into_dyn();

        // Create input tensor
        let input_tensor = Value::from_array(input_array)
            .map_err(|e| Error::ModelError(e.to_string()))?;

        // Run inference
        let outputs = self.session
            .run(ort::inputs![&self.input_name => input_tensor])
            .map_err(|e| Error::ModelError(e.to_string()))?;

        // Extract output
        let output = outputs
            .get("output")
            .or_else(|| outputs.values().next())
            .ok_or_else(|| Error::ModelError("No output from model".to_string()))?;

        let output_array: ArrayD<f32> = output
            .try_extract_tensor()
            .map_err(|e| Error::ModelError(e.to_string()))?
            .to_owned();

        let latency_ns = start.elapsed().as_nanos() as u64;

        // Update stats
        self.inference_count += 1;
        self.total_inference_time_ns += latency_ns;

        // Parse output
        let raw_output: Vec<f32> = output_array.iter().copied().collect();
        let (directions, confidences) = self.parse_output(&raw_output);

        Ok(Prediction {
            directions,
            confidences,
            raw_output,
            latency_ns,
        })
    }

    /// Run inference (non-ONNX fallback for testing)
    #[cfg(not(feature = "onnx"))]
    pub fn predict(&mut self, _features: &Array3<f32>) -> Result<Prediction> {
        let start = Instant::now();

        // Dummy prediction for testing
        let latency_ns = start.elapsed().as_nanos() as u64;

        self.inference_count += 1;
        self.total_inference_time_ns += latency_ns;

        Ok(Prediction {
            directions: vec![0; self.config.num_horizons],
            confidences: vec![0.5; self.config.num_horizons],
            raw_output: vec![0.0; self.config.output_shape.iter().product()],
            latency_ns,
        })
    }

    /// Parse model output into directions and confidences
    fn parse_output(&self, raw: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let mut directions = Vec::with_capacity(self.config.num_horizons);
        let mut confidences = Vec::with_capacity(self.config.num_horizons);

        // Assume output is [horizon, 3] for (down, neutral, up) probabilities
        // or [horizon] for regression output
        if raw.len() == self.config.num_horizons * 3 {
            // Classification output
            for h in 0..self.config.num_horizons {
                let start = h * 3;
                let probs = &raw[start..start + 3];

                let (max_idx, &max_prob) = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                let direction = match max_idx {
                    0 => -1i8, // Down
                    1 => 0i8,  // Neutral
                    2 => 1i8,  // Up
                    _ => 0i8,
                };

                directions.push(direction);
                confidences.push(max_prob);
            }
        } else if raw.len() == self.config.num_horizons {
            // Regression output
            for &value in raw {
                let direction = if value > 0.001 {
                    1i8
                } else if value < -0.001 {
                    -1i8
                } else {
                    0i8
                };
                directions.push(direction);
                confidences.push(value.abs().min(1.0));
            }
        } else {
            // Unknown format - return neutral
            for _ in 0..self.config.num_horizons {
                directions.push(0);
                confidences.push(0.5);
            }
        }

        (directions, confidences)
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get inference count
    #[must_use]
    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }

    /// Get average inference latency in nanoseconds
    #[must_use]
    pub fn avg_latency_ns(&self) -> u64 {
        if self.inference_count == 0 {
            0
        } else {
            self.total_inference_time_ns / self.inference_count
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.inference_count = 0;
        self.total_inference_time_ns = 0;
    }
}

/// Batch inference for multiple samples
pub struct BatchInference {
    /// Maximum batch size
    max_batch_size: usize,
    /// Pending samples
    pending: Vec<Array3<f32>>,
}

impl BatchInference {
    /// Create a new batch inference handler
    #[must_use]
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            pending: Vec::with_capacity(max_batch_size),
        }
    }

    /// Add a sample to the batch
    pub fn add(&mut self, features: Array3<f32>) {
        self.pending.push(features);
    }

    /// Check if batch is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.pending.len() >= self.max_batch_size
    }

    /// Get pending count
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Run batch inference
    pub fn run(&mut self, model: &mut OnnxModel) -> Result<Vec<Prediction>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }

        let mut predictions = Vec::with_capacity(self.pending.len());

        // For now, run individual inference (batch support would require model changes)
        for features in self.pending.drain(..) {
            predictions.push(model.predict(&features)?);
        }

        Ok(predictions)
    }

    /// Clear pending samples
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.input_shape, vec![1, 100, 44]);
        assert_eq!(config.num_horizons, 3);
    }

    #[test]
    fn test_prediction_direction() {
        let pred = Prediction {
            directions: vec![1, -1, 0],
            confidences: vec![0.8, 0.6, 0.5],
            raw_output: vec![],
            latency_ns: 100,
        };

        assert!(pred.is_bullish());
        assert_eq!(pred.primary_direction(), 1);
        assert_eq!(pred.primary_confidence(), 0.8);
    }

    #[test]
    fn test_batch_inference() {
        let mut batch = BatchInference::new(4);

        assert!(!batch.is_full());
        assert_eq!(batch.pending_count(), 0);

        for _ in 0..4 {
            batch.add(Array3::zeros((1, 100, 44)));
        }

        assert!(batch.is_full());
        assert_eq!(batch.pending_count(), 4);

        batch.clear();
        assert_eq!(batch.pending_count(), 0);
    }
}

