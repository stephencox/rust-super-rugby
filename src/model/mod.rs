//! Neural network architecture
//!
//! Multiple architectures for match prediction:
//! - MLP: Simple baseline using comparison features only
//! - LSTM: Sequential model for match history
//! - Transformer: Hierarchical transformer with cross-attention

pub mod cross_attn;
pub mod encoder;
pub mod heads;
pub mod lstm;
pub mod mlp;
pub mod rugby_net;

pub use lstm::LSTMModel;
pub use mlp::MLPModel;
pub use rugby_net::RugbyNet;
