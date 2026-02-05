//! Neural network architecture
//!
//! Multiple architectures for match prediction:
//! - MLP: Simple baseline using comparison features only
//! - LSTM: Sequential model for match history

pub mod heads;
pub mod lstm;
pub mod mlp;

pub use lstm::LSTMModel;
pub use mlp::MLPModel;
