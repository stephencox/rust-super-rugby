//! Simple MLP model for baseline comparison
//!
//! Uses only comparison features (win_rate_diff, margin_diff, pythagorean_diff, log5, is_local)

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::heads::Predictions;

/// Configuration for the MLP model
#[derive(Debug, Clone)]
pub struct MLPConfig {
    /// Input dimension (comparison features)
    pub input_dim: usize,
    /// Hidden layer dimension (0 = linear model only)
    pub hidden_dim: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for MLPConfig {
    fn default() -> Self {
        MLPConfig {
            input_dim: crate::data::dataset::MatchComparison::DIM,
            hidden_dim: 64, // Default hidden layer size
            dropout: 0.1,
        }
    }
}

/// Multi-Layer Perceptron for regression/classification
#[derive(Module, Debug)]
pub struct MLPModel<B: Backend> {
    // Optional hidden layer
    hidden: Option<Linear<B>>,
    dropout: Dropout,
    // Output heads
    win_head: Linear<B>,
    home_score_head: Linear<B>,
    away_score_head: Linear<B>,
}

impl<B: Backend> MLPModel<B> {
    /// Create a new MLP model
    pub fn new(device: &B::Device, config: MLPConfig) -> Self {
        let (hidden, head_input_dim) = if config.hidden_dim > 0 {
            (
                Some(LinearConfig::new(config.input_dim, config.hidden_dim).init(device)),
                config.hidden_dim,
            )
        } else {
            (None, config.input_dim)
        };

        MLPModel {
            hidden,
            dropout: DropoutConfig::new(config.dropout).init(),
            win_head: LinearConfig::new(head_input_dim, 1).init(device),
            home_score_head: LinearConfig::new(head_input_dim, 1).init(device),
            away_score_head: LinearConfig::new(head_input_dim, 1).init(device),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `comparison` - Comparison features [batch, input_dim]
    ///
    /// # Returns
    /// Predictions (win logit, home score, away score)
    pub fn forward(&self, comparison: Tensor<B, 2>) -> Predictions<B> {
        let x = if let Some(hidden) = &self.hidden {
            let h = hidden.forward(comparison);
            let h = relu(h);
            self.dropout.forward(h)
        } else {
            comparison
        };

        Predictions {
            win_logit: self.win_head.forward(x.clone()),
            home_score: self.home_score_head.forward(x.clone()),
            away_score: self.away_score_head.forward(x),
        }
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> crate::Result<()>
    where
        B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
        B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
    {
        let recorder = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(self.clone().into_record(), path.into())
            .map_err(|e| crate::RugbyError::Io(std::io::Error::other(e.to_string())))
    }

    /// Load model from file
    pub fn load(device: &B::Device, path: &str, config: MLPConfig) -> crate::Result<Self>
    where
        B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
        B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
    {
        let recorder = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(path.into(), device)
            .map_err(|e| crate::RugbyError::Io(std::io::Error::other(e.to_string())))?;

        let model = Self::new(device, config);
        Ok(model.load_record(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mlp_model() {
        use crate::data::dataset::MatchComparison;

        let device = Default::default();
        let config = MLPConfig::default();
        let model = MLPModel::<TestBackend>::new(&device, config);

        let comparison = Tensor::random(
            [4, MatchComparison::DIM],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let preds = model.forward(comparison);

        assert_eq!(preds.win_logit.dims(), [4, 1]);
        assert_eq!(preds.home_score.dims(), [4, 1]);
        assert_eq!(preds.away_score.dims(), [4, 1]);
    }
}
