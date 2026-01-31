//! Model training
//!
//! Training loop, loss functions, and metrics tracking.

pub mod lstm_trainer;
pub mod metrics;
pub mod mlp_trainer;
pub mod mlp_tuning;
pub mod simple_mlp;
pub mod trainer;
pub mod transformer_trainer;

pub use lstm_trainer::LSTMTrainer;
pub use metrics::Metrics;
pub use mlp_trainer::MLPTrainer;
pub use mlp_tuning::{MLPHyperparams, MLPTuner, RandomSplitDatasets, SplitRatios, TuningResult};
pub use simple_mlp::SimpleMLPTrainer;
pub use trainer::Trainer;
pub use transformer_trainer::TransformerTrainer;
