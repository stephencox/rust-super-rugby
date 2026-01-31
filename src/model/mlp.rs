//! Simple MLP model for baseline comparison
//!
//! Uses only comparison features (win_rate_diff, margin_diff, pythagorean_diff, log5, is_local)

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::record::{FullPrecisionSettings, Recorder};
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
    /// Dropout rate (unused for linear model)
    pub dropout: f64,
}

impl Default for MLPConfig {
    fn default() -> Self {
        MLPConfig {
            input_dim: 5, // comparison features
            hidden_dim: 0, // Linear model by default
            dropout: 0.0,
        }
    }
}

/// Simple linear model (logistic regression equivalent)
///
/// For win prediction: logit = w·x + b
/// where x = [win_rate_diff, margin_diff, pythagorean_diff, log5, is_local]
#[derive(Module, Debug)]
pub struct MLPModel<B: Backend> {
    // Direct linear projection for win prediction
    win_linear: Linear<B>,
    // Score prediction (also linear)
    home_score_linear: Linear<B>,
    away_score_linear: Linear<B>,
}

impl<B: Backend> MLPModel<B> {
    /// Create a new linear model
    pub fn new(device: &B::Device, config: MLPConfig) -> Self {
        MLPModel {
            win_linear: LinearConfig::new(config.input_dim, 1).init(device),
            home_score_linear: LinearConfig::new(config.input_dim, 1).init(device),
            away_score_linear: LinearConfig::new(config.input_dim, 1).init(device),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `comparison` - Comparison features [batch, 5]
    ///
    /// # Returns
    /// Predictions (win logit, home score, away score)
    pub fn forward(&self, comparison: Tensor<B, 2>) -> Predictions<B> {
        Predictions {
            win_logit: self.win_linear.forward(comparison.clone()),
            home_score: self.home_score_linear.forward(comparison.clone()),
            away_score: self.away_score_linear.forward(comparison),
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
        let device = Default::default();
        let config = MLPConfig::default();
        let model = MLPModel::<TestBackend>::new(&device, config);

        let comparison = Tensor::random(
            [4, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let preds = model.forward(comparison);

        assert_eq!(preds.win_logit.dims(), [4, 1]);
        assert_eq!(preds.home_score.dims(), [4, 1]);
        assert_eq!(preds.away_score.dims(), [4, 1]);
    }
}
