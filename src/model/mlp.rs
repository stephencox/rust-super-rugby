//! Improved MLP model with deeper architecture
//!
//! Architecture: Input(50) → Hidden1(128) → ReLU → Dropout
//!                        → Hidden2(64)  → ReLU → Dropout
//!                        → win_head(1), margin_head(1)

use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for the MLP model
#[derive(Debug, Clone)]
pub struct MLPConfig {
    /// Input dimension (comparison features)
    pub input_dim: usize,
    /// Hidden layer dimensions (e.g., [128, 64] for two layers)
    pub hidden_dims: Vec<usize>,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for MLPConfig {
    fn default() -> Self {
        MLPConfig {
            input_dim: crate::data::dataset::MatchComparison::DIM,
            hidden_dims: vec![128, 64], // Two hidden layers
            dropout: 0.1,
        }
    }
}

/// MLP predictions (win probability + absolute margin)
#[derive(Debug, Clone)]
pub struct MLPPredictions<B: Backend> {
    /// Win logit [batch, 1] - apply sigmoid for P(home wins)
    pub win_logit: Tensor<B, 2>,
    /// Absolute margin [batch, 1] - always >= 0
    pub margin: Tensor<B, 2>,
}

/// A single hidden layer block: Linear → ReLU → Dropout
#[derive(Module, Debug)]
pub struct HiddenBlock<B: Backend> {
    linear: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> HiddenBlock<B> {
    pub fn new(device: &B::Device, in_dim: usize, out_dim: usize, dropout: f64) -> Self {
        HiddenBlock {
            linear: LinearConfig::new(in_dim, out_dim).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(x);
        let x = relu(x);
        self.dropout.forward(x)
    }
}

/// Multi-Layer Perceptron with batch normalization
///
/// Outputs:
/// - win_logit: probability home team wins (apply sigmoid)
/// - margin: absolute score difference (always >= 0)
#[derive(Module, Debug)]
pub struct MLPModel<B: Backend> {
    hidden1: HiddenBlock<B>,
    hidden2: Option<HiddenBlock<B>>,
    win_head: Linear<B>,
    margin_head: Linear<B>,
}

impl<B: Backend> MLPModel<B> {
    /// Create a new MLP model
    pub fn new(device: &B::Device, config: MLPConfig) -> Self {
        let hidden1 = HiddenBlock::new(
            device,
            config.input_dim,
            config.hidden_dims.get(0).copied().unwrap_or(64),
            config.dropout,
        );

        let (hidden2, head_input_dim) = if config.hidden_dims.len() > 1 {
            let h2 = HiddenBlock::new(
                device,
                config.hidden_dims[0],
                config.hidden_dims[1],
                config.dropout,
            );
            (Some(h2), config.hidden_dims[1])
        } else {
            (None, config.hidden_dims.get(0).copied().unwrap_or(64))
        };

        MLPModel {
            hidden1,
            hidden2,
            win_head: LinearConfig::new(head_input_dim, 1).init(device),
            margin_head: LinearConfig::new(head_input_dim, 1).init(device),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `comparison` - Comparison features [batch, input_dim]
    ///
    /// # Returns
    /// MLPPredictions (win logit, absolute margin)
    pub fn forward(&self, comparison: Tensor<B, 2>) -> MLPPredictions<B> {
        let x = self.hidden1.forward(comparison);
        let x = if let Some(h2) = &self.hidden2 {
            h2.forward(x)
        } else {
            x
        };

        let win_logit = self.win_head.forward(x.clone());
        // ReLU ensures margin is non-negative
        let margin = relu(self.margin_head.forward(x));

        MLPPredictions { win_logit, margin }
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
        assert_eq!(preds.margin.dims(), [4, 1]);

        // Margin should be non-negative (ReLU output)
        let margin_data = preds.margin.to_data();
        for val in margin_data.as_slice::<f32>().unwrap() {
            assert!(*val >= 0.0, "Margin should be non-negative, got {}", val);
        }
    }

    #[test]
    fn test_single_hidden_layer() {
        use crate::data::dataset::MatchComparison;

        let device = Default::default();
        let config = MLPConfig {
            input_dim: MatchComparison::DIM,
            hidden_dims: vec![64], // Single layer
            dropout: 0.1,
        };
        let model = MLPModel::<TestBackend>::new(&device, config);

        let comparison = Tensor::random(
            [2, MatchComparison::DIM],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let preds = model.forward(comparison);
        assert_eq!(preds.win_logit.dims(), [2, 1]);
        assert_eq!(preds.margin.dims(), [2, 1]);
    }
}
