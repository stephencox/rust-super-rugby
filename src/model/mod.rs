//! Neural network architecture
//!
//! Hierarchical transformer with cross-attention for match prediction.

pub mod cross_attn;
pub mod encoder;
pub mod heads;
pub mod rugby_net;

pub use rugby_net::RugbyNet;
