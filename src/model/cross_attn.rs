//! Cross-attention layer for team interaction modeling
//!
//! Models how one team's history relates to another team's history.

use burn::module::Module;
use burn::nn::{self, Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for cross-attention
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of cross-attention layers
    pub n_layers: usize,
    /// Feedforward hidden dimension
    pub d_ff: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        CrossAttentionConfig {
            d_model: 128,
            n_heads: 8,
            n_layers: 2,
            d_ff: 512,
            dropout: 0.1,
        }
    }
}

/// Cross-attention mechanism
#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(device: &B::Device, d_model: usize, n_heads: usize, dropout: f64) -> Self {
        let head_dim = d_model / n_heads;

        CrossAttention {
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

    /// Cross-attention forward pass
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch, query_len, d_model]
    /// * `key` - Key tensor [batch, key_len, d_model]
    /// * `value` - Value tensor [batch, key_len, d_model]
    /// * `key_mask` - Optional mask for keys [batch, key_len]
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        key_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, query_len, _] = query.dims();
        let [_, key_len, _] = key.dims();

        // Project Q, K, V
        let q = self.query.forward(query);
        let k = self.key.forward(key);
        let v = self.value.forward(value);

        // Reshape for multi-head attention
        let q = q
            .reshape([batch, query_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, key_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, key_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);

        // Compute attention scores: [batch, heads, query_len, key_len]
        let scores = q.matmul(k.swap_dims(2, 3)) / self.scale;

        // Apply mask if provided
        let scores = if let Some(mask) = key_mask {
            // Expand mask from [batch, seq] to [batch, 1, 1, seq] for broadcasting
            let mask: Tensor<B, 3, burn::tensor::Bool> = mask.unsqueeze_dim(1);
            let mask: Tensor<B, 4, burn::tensor::Bool> = mask.unsqueeze_dim(1);
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

        // Reshape back
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, query_len, self.n_heads * self.head_dim]);

        self.output.forward(out)
    }
}

/// Cross-attention layer with feedforward
#[derive(Module, Debug)]
pub struct CrossAttentionLayer<B: Backend> {
    /// Cross attention from A to B
    cross_attn_a_to_b: CrossAttention<B>,
    /// Cross attention from B to A
    cross_attn_b_to_a: CrossAttention<B>,
    /// Feedforward for A
    ff_a1: Linear<B>,
    ff_a2: Linear<B>,
    /// Feedforward for B
    ff_b1: Linear<B>,
    ff_b2: Linear<B>,
    /// Layer norms
    norm_a1: nn::LayerNorm<B>,
    norm_a2: nn::LayerNorm<B>,
    norm_b1: nn::LayerNorm<B>,
    norm_b2: nn::LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> CrossAttentionLayer<B> {
    pub fn new(device: &B::Device, config: &CrossAttentionConfig) -> Self {
        CrossAttentionLayer {
            cross_attn_a_to_b: CrossAttention::new(
                device,
                config.d_model,
                config.n_heads,
                config.dropout,
            ),
            cross_attn_b_to_a: CrossAttention::new(
                device,
                config.d_model,
                config.n_heads,
                config.dropout,
            ),
            ff_a1: LinearConfig::new(config.d_model, config.d_ff).init(device),
            ff_a2: LinearConfig::new(config.d_ff, config.d_model).init(device),
            ff_b1: LinearConfig::new(config.d_model, config.d_ff).init(device),
            ff_b2: LinearConfig::new(config.d_ff, config.d_model).init(device),
            norm_a1: nn::LayerNormConfig::new(config.d_model).init(device),
            norm_a2: nn::LayerNormConfig::new(config.d_model).init(device),
            norm_b1: nn::LayerNormConfig::new(config.d_model).init(device),
            norm_b2: nn::LayerNormConfig::new(config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Bidirectional cross-attention between two sequences
    ///
    /// # Arguments
    /// * `a` - First sequence [batch, seq_a, d_model]
    /// * `b` - Second sequence [batch, seq_b, d_model]
    /// * `mask_a` - Mask for sequence A
    /// * `mask_b` - Mask for sequence B
    ///
    /// # Returns
    /// Updated (a, b) tuple
    pub fn forward(
        &self,
        a: Tensor<B, 3>,
        b: Tensor<B, 3>,
        mask_a: Option<Tensor<B, 2, burn::tensor::Bool>>,
        mask_b: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // A attends to B
        let a_cross =
            self.cross_attn_a_to_b
                .forward(a.clone(), b.clone(), b.clone(), mask_b.clone());
        let a = self.norm_a1.forward(a + self.dropout.forward(a_cross));

        // B attends to A
        let b_cross = self
            .cross_attn_b_to_a
            .forward(b.clone(), a.clone(), a.clone(), mask_a);
        let b = self.norm_b1.forward(b + self.dropout.forward(b_cross));

        // Feedforward for A
        let a_ff = self.ff_a2.forward(gelu(self.ff_a1.forward(a.clone())));
        let a = self.norm_a2.forward(a + self.dropout.forward(a_ff));

        // Feedforward for B
        let b_ff = self.ff_b2.forward(gelu(self.ff_b1.forward(b.clone())));
        let b = self.norm_b2.forward(b + self.dropout.forward(b_ff));

        (a, b)
    }
}

/// Full cross-attention module with multiple layers
#[derive(Module, Debug)]
pub struct CrossAttentionModule<B: Backend> {
    layers: Vec<CrossAttentionLayer<B>>,
    /// Fusion layer to combine team representations
    fusion: Linear<B>,
    norm: nn::LayerNorm<B>,
}

impl<B: Backend> CrossAttentionModule<B> {
    pub fn new(device: &B::Device, config: CrossAttentionConfig) -> Self {
        let layers: Vec<_> = (0..config.n_layers)
            .map(|_| CrossAttentionLayer::new(device, &config))
            .collect();

        CrossAttentionModule {
            layers,
            fusion: LinearConfig::new(config.d_model * 2, config.d_model).init(device),
            norm: nn::LayerNormConfig::new(config.d_model).init(device),
        }
    }

    /// Forward pass for cross-attention
    ///
    /// # Arguments
    /// * `home_seq` - Home team sequence [batch, seq_home, d_model]
    /// * `away_seq` - Away team sequence [batch, seq_away, d_model]
    /// * `home_mask` - Mask for home sequence
    /// * `away_mask` - Mask for away sequence
    ///
    /// # Returns
    /// Fused representation [batch, d_model]
    pub fn forward(
        &self,
        mut home_seq: Tensor<B, 3>,
        mut away_seq: Tensor<B, 3>,
        home_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        away_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 2> {
        let [batch, _, d_model] = home_seq.dims();

        // Apply cross-attention layers
        for layer in &self.layers {
            let (h, a) = layer.forward(home_seq, away_seq, home_mask.clone(), away_mask.clone());
            home_seq = h;
            away_seq = a;
        }

        // Extract CLS tokens (first position)
        let home_cls: Tensor<B, 2> = home_seq.slice([0..batch, 0..1, 0..d_model]).squeeze();
        let away_cls: Tensor<B, 2> = away_seq.slice([0..batch, 0..1, 0..d_model]).squeeze();

        // Concatenate and fuse
        let combined = Tensor::cat(vec![home_cls, away_cls], 1);
        let fused = self.fusion.forward(combined);

        self.norm.forward(fused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_cross_attention() {
        let device = Default::default();
        let config = CrossAttentionConfig {
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            dropout: 0.0,
        };

        let module = CrossAttentionModule::<TestBackend>::new(&device, config);

        let home = Tensor::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let away = Tensor::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = module.forward(home, away, None, None);

        assert_eq!(out.dims(), [2, 64]);
    }
}
