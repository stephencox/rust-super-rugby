//! Learned embeddings for teams and other categorical features

use burn::module::Module;
use burn::nn;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Learned team embeddings
#[derive(Module, Debug)]
pub struct TeamEmbedding<B: Backend> {
    embedding: nn::Embedding<B>,
}

impl<B: Backend> TeamEmbedding<B> {
    /// Create team embedding layer
    pub fn new(device: &B::Device, num_teams: usize, embed_dim: usize) -> Self {
        let config = nn::EmbeddingConfig::new(num_teams, embed_dim);
        TeamEmbedding {
            embedding: config.init(device),
        }
    }

    /// Look up team embeddings
    ///
    /// # Arguments
    /// * `team_ids` - Tensor of team indices [batch, ...]
    ///
    /// # Returns
    /// Embeddings [batch, ..., embed_dim]
    pub fn forward(&self, team_ids: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.embedding.forward(team_ids)
    }
}

/// Positional encoding for sequence positions (match recency)
#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    /// Precomputed positional embeddings
    encoding: Tensor<B, 2>,
    d_model: usize,
}

impl<B: Backend> PositionalEncoding<B> {
    /// Create sinusoidal positional encoding
    pub fn new(device: &B::Device, max_len: usize, d_model: usize) -> Self {
        let mut encoding_data = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / 10000_f32.powf((2 * (i / 2)) as f32 / d_model as f32);
                encoding_data[pos * d_model + i] =
                    if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        let encoding = Tensor::<B, 1>::from_floats(encoding_data.as_slice(), device)
            .reshape([max_len, d_model]);

        PositionalEncoding { encoding, d_model }
    }

    /// Add positional encoding to input
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    ///
    /// # Returns
    /// Input with positional encoding added [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, seq_len, _d_model] = x.dims();
        let pos_encoding = self.encoding.clone().slice([0..seq_len, 0..self.d_model]);
        x + pos_encoding.unsqueeze()
    }
}

/// Rotary Position Embedding (RoPE) for better relative position encoding
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// Precomputed cos values
    cos_cached: Tensor<B, 2>,
    /// Precomputed sin values
    sin_cached: Tensor<B, 2>,
    dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Create rotary embedding
    pub fn new(device: &B::Device, dim: usize, max_len: usize, base: f32) -> Self {
        let half_dim = dim / 2;

        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / dim as f32))
            .collect();

        // Compute position encodings
        let mut cos_data = vec![0.0f32; max_len * dim];
        let mut sin_data = vec![0.0f32; max_len * dim];

        for pos in 0..max_len {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                // Interleave cos/sin for each dimension pair
                cos_data[pos * dim + i * 2] = angle.cos();
                cos_data[pos * dim + i * 2 + 1] = angle.cos();
                sin_data[pos * dim + i * 2] = angle.sin();
                sin_data[pos * dim + i * 2 + 1] = angle.sin();
            }
        }

        let cos_cached =
            Tensor::<B, 1>::from_floats(cos_data.as_slice(), device).reshape([max_len, dim]);
        let sin_cached =
            Tensor::<B, 1>::from_floats(sin_data.as_slice(), device).reshape([max_len, dim]);

        RotaryEmbedding {
            cos_cached,
            sin_cached,
            dim,
        }
    }

    /// Apply rotary embedding to query/key tensors
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, dim]
    ///
    /// # Returns
    /// Tensor with rotary position encoding applied
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, seq_len, _dim] = x.dims();

        let cos = self.cos_cached.clone().slice([0..seq_len, 0..self.dim]);
        let sin = self.sin_cached.clone().slice([0..seq_len, 0..self.dim]);

        // Rotate pairs of dimensions
        let x_rotated = self.rotate_half(x.clone());

        x.clone() * cos.unsqueeze() + x_rotated * sin.unsqueeze()
    }

    /// Rotate half the dimensions for RoPE
    fn rotate_half(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, dim] = x.dims();
        let half = dim / 2;

        // Split into two halves and swap with negation
        let x1 = x.clone().slice([0..batch, 0..seq, 0..half]);
        let x2 = x.slice([0..batch, 0..seq, half..dim]);

        // Concatenate [-x2, x1]
        Tensor::cat(vec![x2.neg(), x1], 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_positional_encoding() {
        let device = Default::default();
        let pe = PositionalEncoding::<TestBackend>::new(&device, 100, 64);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let y = pe.forward(x);

        assert_eq!(y.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_team_embedding() {
        let device = Default::default();
        let embed = TeamEmbedding::<TestBackend>::new(&device, 20, 32);

        let ids =
            Tensor::<TestBackend, 2, burn::tensor::Int>::from_ints([[0, 1, 2], [3, 4, 5]], &device);
        let y = embed.forward(ids);

        assert_eq!(y.dims(), [2, 3, 32]);
    }
}
