//! Transformer encoder for team match history
//!
//! Encodes a sequence of historical matches into a team representation.

use burn::module::Module;
use burn::nn::{self, Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::features::encoding::PositionalEncoding;

/// Configuration for the team history encoder
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Model dimension (d_model)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Feedforward hidden dimension
    pub d_ff: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            input_dim: 9, // MatchFeatures::DIM
            d_model: 128,
            n_heads: 8,
            n_layers: 4,
            d_ff: 512,
            dropout: 0.1,
            max_seq_len: 32,
        }
    }
}

/// Multi-head self-attention layer
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(device: &B::Device, d_model: usize, n_heads: usize, dropout: f64) -> Self {
        let head_dim = d_model / n_heads;

        MultiHeadAttention {
            query: LinearConfig::new(d_model, d_model).init(device),
            key: LinearConfig::new(d_model, d_model).init(device),
            value: LinearConfig::new(d_model, d_model).init(device),
            output: LinearConfig::new(d_model, d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
            n_heads,
            head_dim,
            scale: (head_dim as f32).sqrt(),
        }
    }

    /// Forward pass with optional attention mask
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = query.dims();

        // Project Q, K, V
        let q = self.query.forward(query);
        let k = self.key.forward(key);
        let v = self.value.forward(value);

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_heads, self.head_dim]);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Compute attention scores: [batch, heads, seq, seq]
        let scores = q.matmul(k.swap_dims(2, 3)) / self.scale;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            // Expand mask from [batch, seq] to [batch, 1, 1, seq] for broadcasting
            let mask: Tensor<B, 3, burn::tensor::Bool> = mask.unsqueeze_dim(1);
            let mask: Tensor<B, 4, burn::tensor::Bool> = mask.unsqueeze_dim(1);
            // Set masked positions to large negative value
            let scores_dims = scores.dims();
            let neg_inf = Tensor::<B, 4>::full(scores_dims, -1e9, &scores.device());
            scores.mask_where(mask.expand(scores_dims), neg_inf)
        } else {
            scores
        };

        // Softmax and dropout
        let attn = softmax(scores, 3);
        let attn = self.dropout.forward(attn);

        // Apply attention to values
        let out = attn.matmul(v);

        // Reshape back: [batch, seq, d_model]
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_heads * self.head_dim]);

        self.output.forward(out)
    }
}

/// Transformer encoder layer
#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    norm1: nn::LayerNorm<B>,
    norm2: nn::LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> EncoderLayer<B> {
    pub fn new(device: &B::Device, config: &EncoderConfig) -> Self {
        EncoderLayer {
            self_attn: MultiHeadAttention::new(
                device,
                config.d_model,
                config.n_heads,
                config.dropout,
            ),
            ff1: LinearConfig::new(config.d_model, config.d_ff).init(device),
            ff2: LinearConfig::new(config.d_ff, config.d_model).init(device),
            norm1: nn::LayerNormConfig::new(config.d_model).init(device),
            norm2: nn::LayerNormConfig::new(config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        // Self-attention with residual connection
        let attn_out = self
            .self_attn
            .forward(x.clone(), x.clone(), x.clone(), mask);
        let x = self.norm1.forward(x + self.dropout.forward(attn_out));

        // Feedforward with residual connection
        let ff_out = self.ff2.forward(gelu(self.ff1.forward(x.clone())));
        self.norm2.forward(x + self.dropout.forward(ff_out))
    }
}

/// Team history transformer encoder
#[derive(Module, Debug)]
pub struct TeamEncoder<B: Backend> {
    /// Input projection
    input_proj: Linear<B>,
    /// Positional encoding
    pos_encoding: PositionalEncoding<B>,
    /// Encoder layers
    layers: Vec<EncoderLayer<B>>,
    /// Output normalization
    norm: nn::LayerNorm<B>,
    /// CLS token for sequence representation
    cls_token: Tensor<B, 2>,
    /// Model dimension (stored for forward pass)
    d_model: usize,
}

impl<B: Backend> TeamEncoder<B> {
    pub fn new(device: &B::Device, config: EncoderConfig) -> Self {
        let layers: Vec<_> = (0..config.n_layers)
            .map(|_| EncoderLayer::new(device, &config))
            .collect();

        // Initialize CLS token with small random values
        let cls_token = Tensor::<B, 2>::random(
            [1, config.d_model],
            burn::tensor::Distribution::Uniform(-0.02, 0.02),
            device,
        );

        TeamEncoder {
            input_proj: LinearConfig::new(config.input_dim, config.d_model).init(device),
            pos_encoding: PositionalEncoding::new(device, config.max_seq_len + 1, config.d_model),
            layers,
            norm: nn::LayerNormConfig::new(config.d_model).init(device),
            cls_token,
            d_model: config.d_model,
        }
    }

    /// Encode a sequence of match features into a team representation
    ///
    /// # Arguments
    /// * `x` - Match features [batch, seq_len, input_dim]
    /// * `mask` - Attention mask [batch, seq_len] (true = valid, false = padding)
    ///
    /// # Returns
    /// Team representation [batch, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 2> {
        let [batch, _seq_len, _] = x.dims();

        // Project input to model dimension
        let x = self.input_proj.forward(x);

        // Prepend CLS token
        let cls_tokens = self
            .cls_token
            .clone()
            .unsqueeze::<3>()
            .expand([batch, 1, self.d_model]);
        let x = Tensor::cat(vec![cls_tokens, x], 1);

        // Update mask to include CLS token (always valid)
        let mask = mask.map(|m| {
            let cls_mask = Tensor::<B, 2, burn::tensor::Bool>::full([batch, 1], true, &m.device());
            Tensor::cat(vec![cls_mask, m], 1)
        });

        // Add positional encoding
        let x = self.pos_encoding.forward(x);

        // Apply transformer layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, mask.clone());
        }

        // Normalize and extract CLS token representation
        let x = self.norm.forward(x);
        let cls: Tensor<B, 2> = x.slice([0..batch, 0..1, 0..self.d_model]).squeeze();
        cls
    }

    /// Get the full sequence output (for cross-attention)
    pub fn forward_full(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, _seq_len, _] = x.dims();

        // Project input to model dimension
        let x = self.input_proj.forward(x);

        // Prepend CLS token
        let cls_tokens = self
            .cls_token
            .clone()
            .unsqueeze::<3>()
            .expand([batch, 1, self.d_model]);
        let x = Tensor::cat(vec![cls_tokens, x], 1);

        // Update mask
        let mask = mask.map(|m| {
            let cls_mask = Tensor::<B, 2, burn::tensor::Bool>::full([batch, 1], true, &m.device());
            Tensor::cat(vec![cls_mask, m], 1)
        });

        // Add positional encoding
        let x = self.pos_encoding.forward(x);

        // Apply transformer layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, mask.clone());
        }

        self.norm.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_encoder() {
        let device = Default::default();
        let config = EncoderConfig {
            input_dim: 9,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            dropout: 0.0,
            max_seq_len: 16,
        };

        let encoder = TeamEncoder::<TestBackend>::new(&device, config);

        let x = Tensor::random(
            [2, 10, 9],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = encoder.forward(x, None);

        assert_eq!(out.dims(), [2, 64]);
    }

    #[test]
    fn test_multi_head_attention() {
        let device = Default::default();
        let attn = MultiHeadAttention::<TestBackend>::new(&device, 64, 4, 0.0);

        let x = Tensor::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = attn.forward(x.clone(), x.clone(), x, None);

        assert_eq!(out.dims(), [2, 10, 64]);
    }
}
