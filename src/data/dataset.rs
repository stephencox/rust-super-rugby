//! Burn Dataset implementation for rugby match data
//!
//! Provides training samples with team match histories.

use crate::data::Database;
use crate::features::MatchFeatures;
use crate::{MatchRecord, Result, TeamId};
use burn::data::dataset::Dataset;
use chrono::NaiveDate;

/// A training sample for the model
#[derive(Debug, Clone)]
pub struct MatchSample {
    /// Features for home team's recent matches
    pub home_history: Vec<MatchFeatures>,
    /// Features for away team's recent matches
    pub away_history: Vec<MatchFeatures>,
    /// Mask indicating which history entries are valid (not padding)
    pub home_mask: Vec<bool>,
    pub away_mask: Vec<bool>,
    /// Target: did home team win? (1.0 = yes, 0.0 = no)
    pub home_win: f32,
    /// Target: home team score
    pub home_score: f32,
    /// Target: away team score
    pub away_score: f32,
}

/// Dataset configuration
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Maximum number of historical matches per team
    pub max_history: usize,
    /// Minimum matches required to include a sample
    pub min_history: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        DatasetConfig {
            max_history: 16,
            min_history: 3,
        }
    }
}

/// Rugby match dataset for training
pub struct RugbyDataset {
    samples: Vec<MatchSample>,
    config: DatasetConfig,
}

impl RugbyDataset {
    /// Create dataset from matches before a cutoff date (for training)
    pub fn from_matches_before(
        db: &Database,
        cutoff: NaiveDate,
        config: DatasetConfig,
    ) -> Result<Self> {
        let matches = db.get_matches_before(cutoff)?;
        Self::from_matches(db, matches, config)
    }

    /// Create dataset from matches in a date range (for validation/test)
    pub fn from_matches_in_range(
        db: &Database,
        start: NaiveDate,
        end: NaiveDate,
        config: DatasetConfig,
    ) -> Result<Self> {
        let matches = db.get_matches_in_range(start, end)?;
        Self::from_matches(db, matches, config)
    }

    /// Create dataset from a list of matches
    pub fn from_matches(
        db: &Database,
        matches: Vec<MatchRecord>,
        config: DatasetConfig,
    ) -> Result<Self> {
        let mut samples = Vec::new();

        // Build history index: for each match, we need historical matches before it
        let all_matches = db.get_all_matches()?;

        for target_match in &matches {
            // Get historical matches for both teams before this match
            let home_history = Self::get_team_history(
                &all_matches,
                target_match.home_team,
                target_match.date,
                config.max_history,
            );

            let away_history = Self::get_team_history(
                &all_matches,
                target_match.away_team,
                target_match.date,
                config.max_history,
            );

            // Skip if insufficient history
            if home_history.len() < config.min_history || away_history.len() < config.min_history {
                continue;
            }

            // Convert to features with padding
            let (home_features, home_mask) = Self::to_features_with_padding(
                &home_history,
                target_match.home_team,
                config.max_history,
            );
            let (away_features, away_mask) = Self::to_features_with_padding(
                &away_history,
                target_match.away_team,
                config.max_history,
            );

            samples.push(MatchSample {
                home_history: home_features,
                away_history: away_features,
                home_mask,
                away_mask,
                home_win: if target_match.home_score > target_match.away_score {
                    1.0
                } else {
                    0.0
                },
                home_score: target_match.home_score as f32,
                away_score: target_match.away_score as f32,
            });
        }

        log::info!("Created dataset with {} samples", samples.len());
        Ok(RugbyDataset { samples, config })
    }

    /// Get historical matches for a team before a given date
    fn get_team_history(
        all_matches: &[MatchRecord],
        team: TeamId,
        before_date: NaiveDate,
        max_history: usize,
    ) -> Vec<MatchRecord> {
        let mut history: Vec<_> = all_matches
            .iter()
            .filter(|m| m.date < before_date && (m.home_team == team || m.away_team == team))
            .cloned()
            .collect();

        // Sort by date descending and take most recent
        history.sort_by(|a, b| b.date.cmp(&a.date));
        history.truncate(max_history);

        // Reverse to chronological order
        history.reverse();
        history
    }

    /// Convert matches to features with padding
    fn to_features_with_padding(
        matches: &[MatchRecord],
        perspective_team: TeamId,
        max_len: usize,
    ) -> (Vec<MatchFeatures>, Vec<bool>) {
        let mut features = Vec::with_capacity(max_len);
        let mut mask = Vec::with_capacity(max_len);

        // Add actual match features
        for m in matches {
            features.push(MatchFeatures::from_match(m, perspective_team));
            mask.push(true);
        }

        // Pad with zeros
        while features.len() < max_len {
            features.push(MatchFeatures::padding());
            mask.push(false);
        }

        (features, mask)
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get dataset config
    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }

    /// Split dataset into train/validation
    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let split_idx = (self.samples.len() as f32 * train_ratio) as usize;
        let (train_samples, val_samples) = self.samples.split_at(split_idx);

        (
            RugbyDataset {
                samples: train_samples.to_vec(),
                config: self.config.clone(),
            },
            RugbyDataset {
                samples: val_samples.to_vec(),
                config: self.config,
            },
        )
    }
}

impl Dataset<MatchSample> for RugbyDataset {
    fn get(&self, index: usize) -> Option<MatchSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Batch of match samples for training
#[derive(Debug, Clone)]
pub struct MatchBatch<B: burn::tensor::backend::Backend> {
    /// Home team history features: [batch, seq_len, features]
    pub home_history: burn::tensor::Tensor<B, 3>,
    /// Away team history features: [batch, seq_len, features]
    pub away_history: burn::tensor::Tensor<B, 3>,
    /// Home history mask: [batch, seq_len]
    pub home_mask: burn::tensor::Tensor<B, 2, burn::tensor::Bool>,
    /// Away history mask: [batch, seq_len]
    pub away_mask: burn::tensor::Tensor<B, 2, burn::tensor::Bool>,
    /// Target win labels: [batch]
    pub home_win: burn::tensor::Tensor<B, 1>,
    /// Target home scores: [batch]
    pub home_score: burn::tensor::Tensor<B, 1>,
    /// Target away scores: [batch]
    pub away_score: burn::tensor::Tensor<B, 1>,
}

/// Batcher for creating training batches
#[derive(Clone)]
pub struct MatchBatcher<B: burn::tensor::backend::Backend> {
    device: B::Device,
}

impl<B: burn::tensor::backend::Backend> MatchBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        MatchBatcher { device }
    }
}

impl<B: burn::tensor::backend::Backend>
    burn::data::dataloader::batcher::Batcher<B, MatchSample, MatchBatch<B>> for MatchBatcher<B>
{
    fn batch(&self, items: Vec<MatchSample>, _device: &B::Device) -> MatchBatch<B> {
        let batch_size = items.len();
        let seq_len = items.first().map(|s| s.home_history.len()).unwrap_or(0);
        let feature_dim = MatchFeatures::DIM;

        // Collect all data into vectors
        let mut home_data = Vec::with_capacity(batch_size * seq_len * feature_dim);
        let mut away_data = Vec::with_capacity(batch_size * seq_len * feature_dim);
        let mut home_mask_data = Vec::with_capacity(batch_size * seq_len);
        let mut away_mask_data = Vec::with_capacity(batch_size * seq_len);
        let mut home_win_data = Vec::with_capacity(batch_size);
        let mut home_score_data = Vec::with_capacity(batch_size);
        let mut away_score_data = Vec::with_capacity(batch_size);

        for sample in &items {
            for feat in &sample.home_history {
                home_data.extend(feat.to_vec());
            }
            for feat in &sample.away_history {
                away_data.extend(feat.to_vec());
            }
            home_mask_data.extend(sample.home_mask.iter().copied());
            away_mask_data.extend(sample.away_mask.iter().copied());
            home_win_data.push(sample.home_win);
            home_score_data.push(sample.home_score);
            away_score_data.push(sample.away_score);
        }

        // Create tensors
        let home_history =
            burn::tensor::Tensor::<B, 1>::from_floats(home_data.as_slice(), &self.device)
                .reshape([batch_size, seq_len, feature_dim]);

        let away_history =
            burn::tensor::Tensor::<B, 1>::from_floats(away_data.as_slice(), &self.device)
                .reshape([batch_size, seq_len, feature_dim]);

        let home_mask = burn::tensor::Tensor::<B, 1, burn::tensor::Bool>::from_bool(
            burn::tensor::TensorData::from(home_mask_data.as_slice()),
            &self.device,
        )
        .reshape([batch_size, seq_len]);

        let away_mask = burn::tensor::Tensor::<B, 1, burn::tensor::Bool>::from_bool(
            burn::tensor::TensorData::from(away_mask_data.as_slice()),
            &self.device,
        )
        .reshape([batch_size, seq_len]);

        let home_win =
            burn::tensor::Tensor::<B, 1>::from_floats(home_win_data.as_slice(), &self.device);

        let home_score =
            burn::tensor::Tensor::<B, 1>::from_floats(home_score_data.as_slice(), &self.device);

        let away_score =
            burn::tensor::Tensor::<B, 1>::from_floats(away_score_data.as_slice(), &self.device);

        MatchBatch {
            home_history,
            away_history,
            home_mask,
            away_mask,
            home_win,
            home_score,
            away_score,
        }
    }
}
