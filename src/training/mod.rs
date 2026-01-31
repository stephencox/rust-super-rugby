//! Model training
//!
//! Training loop, loss functions, and metrics tracking.

pub mod metrics;
pub mod trainer;

pub use metrics::Metrics;
pub use trainer::Trainer;
