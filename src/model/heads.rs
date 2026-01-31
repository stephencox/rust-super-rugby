//! Prediction heads for win probability and scores
//!
//! Multi-task output heads sharing the encoder representation.
//! Uses dual-path architecture for win prediction to prevent
//! comparison features from being drowned out.

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for prediction heads
#[derive(Debug, Clone)]
pub struct HeadsConfig {
    /// Input dimension from encoder (transformer + team embeds + comparison)
    pub input_dim: usize,
    /// Hidden dimension for head MLPs
    pub hidden_dim: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Dimension of comparison features (for direct path)
    pub comparison_dim: usize,
}

impl Default for HeadsConfig {
    fn default() -> Self {
        HeadsConfig {
            input_dim: 128,
            hidden_dim: 64,
            dropout: 0.1,
            comparison_dim: 5,
        }
    }
}

/// Win probability prediction head using ONLY comparison features
///
/// The comparison features (win_rate_diff, margin_diff, pythagorean_diff, log5, is_local)
/// have proven predictive power (69.4% accuracy with simple logistic regression).
///
/// This head is intentionally simple - a single linear layer - to avoid optimization
/// issues that prevent learning from these features.
#[derive(Module, Debug)]
pub struct WinHead<B: Backend> {
    // Single linear layer: comparison features -> logit
    fc: Linear<B>,
    /// Dimension of comparison features
    comparison_dim: usize,
}

impl<B: Backend> WinHead<B> {
    pub fn new(device: &B::Device, config: &HeadsConfig) -> Self {
        WinHead {
            // Simple linear layer - matches what works in Python
            fc: LinearConfig::new(config.comparison_dim, 1).init(device),
            comparison_dim: config.comparison_dim,
        }
    }

    /// Forward pass returning win logit
    ///
    /// # Arguments
    /// * `x` - Full representation [batch, input_dim] where last comparison_dim features are comparison
    ///
    /// # Returns
    /// Win logit [batch, 1] (apply sigmoid for probability)
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, input_dim] = x.dims();

        // Extract comparison features (last comparison_dim features)
        let start = input_dim - self.comparison_dim;
        let comparison = x.slice([0..batch, start..input_dim]);

        // Simple linear transform
        self.fc.forward(comparison)
    }

    /// Forward pass returning win probability
    pub fn forward_prob(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        sigmoid(self.forward(x))
    }
}

/// Score prediction head (shared structure for home and away)
#[derive(Module, Debug)]
pub struct ScoreHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> ScoreHead<B> {
    pub fn new(device: &B::Device, config: &HeadsConfig) -> Self {
        ScoreHead {
            fc1: LinearConfig::new(config.input_dim, config.hidden_dim).init(device),
            fc2: LinearConfig::new(config.hidden_dim, config.hidden_dim / 2).init(device),
            fc3: LinearConfig::new(config.hidden_dim / 2, 1).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Forward pass returning predicted score
    ///
    /// # Arguments
    /// * `x` - Fused representation [batch, input_dim]
    ///
    /// # Returns
    /// Predicted score [batch, 1] (z-score normalized)
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = gelu(self.fc1.forward(x));
        let x = self.dropout.forward(x);
        let x = gelu(self.fc2.forward(x));
        let x = self.dropout.forward(x);
        // Linear output for z-score normalized predictions
        self.fc3.forward(x)
    }
}

/// Combined prediction heads for multi-task output
#[derive(Module, Debug)]
pub struct PredictionHeads<B: Backend> {
    win_head: WinHead<B>,
    home_score_head: ScoreHead<B>,
    away_score_head: ScoreHead<B>,
}

/// Model predictions
#[derive(Debug, Clone)]
pub struct Predictions<B: Backend> {
    /// Win logit [batch, 1]
    pub win_logit: Tensor<B, 2>,
    /// Home team score [batch, 1]
    pub home_score: Tensor<B, 2>,
    /// Away team score [batch, 1]
    pub away_score: Tensor<B, 2>,
}

impl<B: Backend> PredictionHeads<B> {
    pub fn new(device: &B::Device, config: HeadsConfig) -> Self {
        PredictionHeads {
            win_head: WinHead::new(device, &config),
            home_score_head: ScoreHead::new(device, &config),
            away_score_head: ScoreHead::new(device, &config),
        }
    }

    /// Forward pass for all prediction heads
    ///
    /// # Arguments
    /// * `x` - Fused representation [batch, input_dim]
    ///
    /// # Returns
    /// All predictions
    pub fn forward(&self, x: Tensor<B, 2>) -> Predictions<B> {
        Predictions {
            win_logit: self.win_head.forward(x.clone()),
            home_score: self.home_score_head.forward(x.clone()),
            away_score: self.away_score_head.forward(x),
        }
    }

    /// Get win probability
    pub fn win_probability(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.win_head.forward_prob(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_prediction_heads() {
        let device = Default::default();
        let config = HeadsConfig {
            input_dim: 64,
            hidden_dim: 32,
            dropout: 0.0,
            comparison_dim: 5,
        };

        let heads = PredictionHeads::<TestBackend>::new(&device, config);

        let x = Tensor::random(
            [4, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let preds = heads.forward(x);

        assert_eq!(preds.win_logit.dims(), [4, 1]);
        assert_eq!(preds.home_score.dims(), [4, 1]);
        assert_eq!(preds.away_score.dims(), [4, 1]);
    }

    #[test]
    fn test_win_probability() {
        let device = Default::default();
        let config = HeadsConfig {
            input_dim: 128,
            hidden_dim: 64,
            dropout: 0.1,
            comparison_dim: 5,
        };

        let heads = PredictionHeads::<TestBackend>::new(&device, config);

        let x = Tensor::random(
            [4, 128],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let prob = heads.win_probability(x);

        // Probabilities should be between 0 and 1
        let prob_data = prob.to_data();
        for val in prob_data.as_slice::<f32>().unwrap() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }
}
