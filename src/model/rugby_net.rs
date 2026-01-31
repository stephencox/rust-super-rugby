//! Full RugbyNet model combining encoder, cross-attention, and prediction heads
//!
//! Hierarchical transformer for Super Rugby match prediction.

use burn::module::Module;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::cross_attn::{CrossAttentionConfig, CrossAttentionModule};
use super::encoder::{EncoderConfig, TeamEncoder};
use super::heads::{HeadsConfig, PredictionHeads, Predictions};

/// Configuration for the full RugbyNet model
#[derive(Debug, Clone)]
pub struct RugbyNetConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_encoder_layers: usize,
    /// Number of cross-attention layers
    pub n_cross_attn_layers: usize,
    /// Feedforward hidden dimension
    pub d_ff: usize,
    /// Head hidden dimension
    pub head_hidden_dim: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for RugbyNetConfig {
    fn default() -> Self {
        RugbyNetConfig {
            input_dim: 9,
            d_model: 128,
            n_heads: 8,
            n_encoder_layers: 4,
            n_cross_attn_layers: 2,
            d_ff: 512,
            head_hidden_dim: 64,
            dropout: 0.2,
            max_seq_len: 16,
        }
    }
}

impl RugbyNetConfig {
    /// Create configuration from model config
    pub fn from_model_config(model: &crate::ModelConfig, training: &crate::TrainingConfig) -> Self {
        RugbyNetConfig {
            input_dim: 9, // MatchFeatures::DIM
            d_model: model.d_model,
            n_heads: model.n_heads,
            n_encoder_layers: model.n_encoder_layers,
            n_cross_attn_layers: model.n_cross_attn_layers,
            d_ff: model.d_model * 4,
            head_hidden_dim: model.d_model / 2,
            dropout: training.dropout,
            max_seq_len: model.max_history,
        }
    }

    fn encoder_config(&self) -> EncoderConfig {
        EncoderConfig {
            input_dim: self.input_dim,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_layers: self.n_encoder_layers,
            d_ff: self.d_ff,
            dropout: self.dropout,
            max_seq_len: self.max_seq_len,
        }
    }

    fn cross_attn_config(&self) -> CrossAttentionConfig {
        CrossAttentionConfig {
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_layers: self.n_cross_attn_layers,
            d_ff: self.d_ff,
            dropout: self.dropout,
        }
    }

    fn heads_config(&self) -> HeadsConfig {
        HeadsConfig {
            input_dim: self.d_model,
            hidden_dim: self.head_hidden_dim,
            dropout: self.dropout,
        }
    }
}

/// The full RugbyNet model
#[derive(Module, Debug)]
pub struct RugbyNet<B: Backend> {
    /// Shared team history encoder
    encoder: TeamEncoder<B>,
    /// Cross-attention between team histories
    cross_attn: CrossAttentionModule<B>,
    /// Prediction heads
    heads: PredictionHeads<B>,
    /// Max sequence length (stored for mask extension)
    max_seq_len: usize,
}

impl<B: Backend> RugbyNet<B> {
    /// Create a new RugbyNet model
    pub fn new(device: &B::Device, config: RugbyNetConfig) -> Self {
        let max_seq_len = config.max_seq_len;
        RugbyNet {
            encoder: TeamEncoder::new(device, config.encoder_config()),
            cross_attn: CrossAttentionModule::new(device, config.cross_attn_config()),
            heads: PredictionHeads::new(device, config.heads_config()),
            max_seq_len,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `home_history` - Home team match history [batch, seq_len, features]
    /// * `away_history` - Away team match history [batch, seq_len, features]
    /// * `home_mask` - Mask for home history [batch, seq_len]
    /// * `away_mask` - Mask for away history [batch, seq_len]
    ///
    /// # Returns
    /// Predictions (win logit, home score, away score)
    pub fn forward(
        &self,
        home_history: Tensor<B, 3>,
        away_history: Tensor<B, 3>,
        home_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        away_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Predictions<B> {
        // Encode both team histories
        let home_encoded = self.encoder.forward_full(home_history, home_mask.clone());
        let away_encoded = self.encoder.forward_full(away_history, away_mask.clone());

        // Cross-attention to model team interaction
        // Note: masks need to be extended to include CLS token
        let home_mask_ext = home_mask.map(|m| {
            let [batch, _seq] = m.dims();
            let cls_mask = Tensor::<B, 2, burn::tensor::Bool>::full([batch, 1], true, &m.device());
            Tensor::cat(vec![cls_mask, m], 1)
        });
        let away_mask_ext = away_mask.map(|m| {
            let [batch, _seq] = m.dims();
            let cls_mask = Tensor::<B, 2, burn::tensor::Bool>::full([batch, 1], true, &m.device());
            Tensor::cat(vec![cls_mask, m], 1)
        });

        let fused =
            self.cross_attn
                .forward(home_encoded, away_encoded, home_mask_ext, away_mask_ext);

        // Predict from fused representation
        self.heads.forward(fused)
    }

    /// Get just the team representations (for analysis/debugging)
    pub fn encode_teams(
        &self,
        home_history: Tensor<B, 3>,
        away_history: Tensor<B, 3>,
        home_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        away_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let home_encoded = self.encoder.forward(home_history, home_mask);
        let away_encoded = self.encoder.forward(away_history, away_mask);
        (home_encoded, away_encoded)
    }

    /// Get max sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
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
    pub fn load(device: &B::Device, path: &str, config: RugbyNetConfig) -> crate::Result<Self>
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

/// Output from the model with converted predictions
#[derive(Debug, Clone)]
pub struct MatchPrediction {
    /// Probability that home team wins (0-1)
    pub home_win_prob: f32,
    /// Predicted home team score
    pub home_score: f32,
    /// Predicted away team score
    pub away_score: f32,
}

impl MatchPrediction {
    /// Create from model predictions
    pub fn from_predictions<B: Backend>(preds: &Predictions<B>) -> Vec<Self> {
        let win_probs: Vec<f32> = sigmoid(preds.win_logit.clone())
            .to_data()
            .as_slice()
            .unwrap()
            .to_vec();
        let home_scores: Vec<f32> = preds
            .home_score
            .clone()
            .to_data()
            .as_slice()
            .unwrap()
            .to_vec();
        let away_scores: Vec<f32> = preds
            .away_score
            .clone()
            .to_data()
            .as_slice()
            .unwrap()
            .to_vec();

        win_probs
            .into_iter()
            .zip(home_scores)
            .zip(away_scores)
            .map(|((p, h), a)| MatchPrediction {
                home_win_prob: p,
                home_score: h,
                away_score: a,
            })
            .collect()
    }

    /// Get predicted winner
    pub fn predicted_winner(&self) -> &'static str {
        if self.home_win_prob >= 0.5 {
            "home"
        } else {
            "away"
        }
    }

    /// Get predicted margin (positive = home win)
    pub fn predicted_margin(&self) -> f32 {
        self.home_score - self.away_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rugby_net() {
        let device = Default::default();
        let config = RugbyNetConfig {
            input_dim: 9,
            d_model: 64,
            n_heads: 4,
            n_encoder_layers: 2,
            n_cross_attn_layers: 1,
            d_ff: 128,
            head_hidden_dim: 32,
            dropout: 0.0,
            max_seq_len: 10,
        };

        let model = RugbyNet::<TestBackend>::new(&device, config);

        let home = Tensor::random(
            [2, 10, 9],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let away = Tensor::random(
            [2, 10, 9],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let preds = model.forward(home, away, None, None);

        assert_eq!(preds.win_logit.dims(), [2, 1]);
        assert_eq!(preds.home_score.dims(), [2, 1]);
        assert_eq!(preds.away_score.dims(), [2, 1]);
    }

    #[test]
    fn test_match_prediction() {
        let device = Default::default();
        let config = RugbyNetConfig {
            input_dim: 9,
            d_model: 64,
            n_heads: 4,
            n_encoder_layers: 2,
            n_cross_attn_layers: 1,
            d_ff: 128,
            head_hidden_dim: 32,
            dropout: 0.0,
            max_seq_len: 10,
        };

        let model = RugbyNet::<TestBackend>::new(&device, config);

        let home = Tensor::random(
            [4, 10, 9],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let away = Tensor::random(
            [4, 10, 9],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let preds = model.forward(home, away, None, None);
        let match_preds = MatchPrediction::from_predictions(&preds);

        assert_eq!(match_preds.len(), 4);
        for pred in &match_preds {
            assert!(pred.home_win_prob >= 0.0 && pred.home_win_prob <= 1.0);
            assert!(pred.home_score >= 0.0);
            assert!(pred.away_score >= 0.0);
        }
    }
}
