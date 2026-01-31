//! Feature extraction and encoding
//!
//! Converts raw match data into model-ready features.

pub mod encoding;
pub mod match_repr;
pub mod team_stats;

pub use encoding::TeamEmbedding;
pub use match_repr::MatchFeatures;
pub use team_stats::TeamStatistics;
