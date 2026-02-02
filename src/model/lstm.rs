//! LSTM model for rugby match prediction
//!
//! Processes team history sequences to predict match outcomes.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for the LSTM model
#[derive(Debug, Clone)]
pub struct LSTMConfig {
    /// Input feature dimension (MatchFeatures::DIM = 15)
    pub input_dim: usize,
    /// LSTM hidden size
    pub hidden_size: usize,
    /// Number of LSTM layers
    pub num_layers: usize,
    /// Whether to use bidirectional LSTM
    pub bidirectional: bool,
    /// Comparison feature dimension (5)
    pub comparison_dim: usize,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        LSTMConfig {
            input_dim: crate::features::MatchFeatures::DIM,
            hidden_size: 64,
            num_layers: 1,
            bidirectional: false,
            comparison_dim: crate::data::dataset::MatchComparison::DIM,
        }
    }
}

/// LSTM model for rugby prediction
///
/// Architecture:
/// 1. Process home team history through LSTM -> home_repr
/// 2. Process away team history through LSTM -> away_repr
/// 3. Concatenate [home_repr, away_repr, comparison_features]
/// 4. FC layers -> predictions
#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    /// LSTM for processing team history (shared for home/away)
    lstm: Lstm<B>,
    /// FC layer to combine representations
    fc1: Linear<B>,
    /// Win prediction head
    win_head: Linear<B>,
    /// Home score prediction head
    home_score_head: Linear<B>,
    /// Away score prediction head
    away_score_head: Linear<B>,
    /// Config for reference
    hidden_size: usize,
}

impl<B: Backend> LSTMModel<B> {
    /// Create a new LSTM model
    pub fn new(device: &B::Device, config: LSTMConfig) -> Self {
        // LSTM for sequence processing
        let lstm = LstmConfig::new(config.input_dim, config.hidden_size, true)
            .init(device);

        // Combined representation: home_hidden + away_hidden + comparison
        let fc_input_size = config.hidden_size * 2 + config.comparison_dim;
        let fc1 = LinearConfig::new(fc_input_size, config.hidden_size).init(device);

        // Output heads
        let win_head = LinearConfig::new(config.hidden_size, 1).init(device);
        let home_score_head = LinearConfig::new(config.hidden_size, 1).init(device);
        let away_score_head = LinearConfig::new(config.hidden_size, 1).init(device);

        LSTMModel {
            lstm,
            fc1,
            win_head,
            home_score_head,
            away_score_head,
            hidden_size: config.hidden_size,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `home_history` - Home team history [batch, seq_len, features]
    /// * `away_history` - Away team history [batch, seq_len, features]
    /// * `comparison` - Comparison features [batch, 5]
    ///
    /// # Returns
    /// Tuple of (win_logit, home_score, away_score)
    pub fn forward(
        &self,
        home_history: Tensor<B, 3>,
        away_history: Tensor<B, 3>,
        comparison: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = home_history.dims()[0];

        // Process home team history through LSTM
        // LSTM returns (output, state) where state has .hidden and .cell fields
        let (_, home_state) = self.lstm.forward(home_history, None);
        // Take final hidden state: [batch, hidden_size]
        let home_repr = home_state.hidden.reshape([batch_size, self.hidden_size]);

        // Process away team history through LSTM
        let (_, away_state) = self.lstm.forward(away_history, None);
        let away_repr = away_state.hidden.reshape([batch_size, self.hidden_size]);

        // Concatenate: [home_repr, away_repr, comparison]
        let combined = Tensor::cat(vec![home_repr, away_repr, comparison], 1);

        // FC + activation
        let x = self.fc1.forward(combined);
        let x = burn::tensor::activation::relu(x);

        // Output predictions
        let win_logit = self.win_head.forward(x.clone());
        let home_score = self.home_score_head.forward(x.clone());
        let away_score = self.away_score_head.forward(x);

        (win_logit, home_score, away_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_lstm_model() {
        use crate::features::MatchFeatures;
        use crate::data::dataset::MatchComparison;

        let device = Default::default();
        let config = LSTMConfig::default();
        let model = LSTMModel::<TestBackend>::new(&device, config);

        // Create dummy inputs with correct dimensions
        let batch_size = 4;
        let seq_len = 10;
        let home_history = Tensor::random(
            [batch_size, seq_len, MatchFeatures::DIM],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let away_history = Tensor::random(
            [batch_size, seq_len, MatchFeatures::DIM],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let comparison = Tensor::random(
            [batch_size, MatchComparison::DIM],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (win, home, away) = model.forward(home_history, away_history, comparison);

        assert_eq!(win.dims(), [batch_size, 1]);
        assert_eq!(home.dims(), [batch_size, 1]);
        assert_eq!(away.dims(), [batch_size, 1]);
    }
}
