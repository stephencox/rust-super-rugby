//! Feature extraction and encoding
//!
//! Converts raw match data into model-ready features.

pub mod elo;
pub mod encoding;
pub mod match_repr;
pub mod team_stats;
pub mod temporal;
pub mod venue;
pub mod workload;

pub use elo::{EloConfig, EloFeatures, EloRatings};
pub use venue::{VenueFeatures, VenueTracker};
pub use workload::{WorkloadComputer, WorkloadFeatures};
pub use encoding::TeamEmbedding;
pub use match_repr::MatchFeatures;
pub use team_stats::TeamStatistics;
pub use temporal::{TemporalContext, TemporalFeatureComputer};
