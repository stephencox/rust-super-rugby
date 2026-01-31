//! Model inference for predictions

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::data::Database;
use crate::features::MatchFeatures;
use crate::model::rugby_net::{MatchPrediction, RugbyNet, RugbyNetConfig};
use crate::{ConfidenceLevel, MatchRecord, Prediction, Result, RugbyError, Team, TeamId};

/// Predictor for making match predictions
pub struct Predictor<B: Backend> {
    model: RugbyNet<B>,
    db: Database,
    device: B::Device,
    max_history: usize,
    min_history: usize,
}

impl<B: Backend> Predictor<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new predictor
    pub fn new(model: RugbyNet<B>, db: Database, device: B::Device) -> Self {
        let max_history = model.max_seq_len();
        Predictor {
            model,
            db,
            device,
            max_history,
            min_history: 3,
        }
    }

    /// Load predictor from saved model
    pub fn load(
        db: Database,
        model_path: &str,
        config: RugbyNetConfig,
        device: B::Device,
    ) -> Result<Self> {
        let model = RugbyNet::load(&device, model_path, config)?;
        Ok(Self::new(model, db, device))
    }

    /// Predict a single match
    pub fn predict(&self, home_team: &str, away_team: &str) -> Result<Prediction> {
        // Look up teams
        let home = self
            .db
            .find_team_by_name(home_team)?
            .ok_or_else(|| RugbyError::UnknownTeam(home_team.to_string()))?;
        let away = self
            .db
            .find_team_by_name(away_team)?
            .ok_or_else(|| RugbyError::UnknownTeam(away_team.to_string()))?;

        self.predict_teams(home, away)
    }

    /// Predict a match between two teams
    pub fn predict_teams(&self, home: Team, away: Team) -> Result<Prediction> {
        // Get match history for both teams
        let home_matches = self.db.get_recent_team_matches(home.id, self.max_history)?;
        let away_matches = self.db.get_recent_team_matches(away.id, self.max_history)?;

        // Determine confidence based on history availability
        let confidence = self.compute_confidence(home_matches.len(), away_matches.len());

        // Check minimum history
        if home_matches.len() < self.min_history || away_matches.len() < self.min_history {
            return Err(RugbyError::InsufficientHistory {
                team: if home_matches.len() < away_matches.len() {
                    home.name.clone()
                } else {
                    away.name.clone()
                },
                matches: home_matches.len().min(away_matches.len()),
                required: self.min_history,
            });
        }

        // Convert to features
        let (home_features, home_mask) = self.to_features(&home_matches, home.id);
        let (away_features, away_mask) = self.to_features(&away_matches, away.id);

        // Create tensors
        let home_tensor = self.features_to_tensor(&home_features);
        let away_tensor = self.features_to_tensor(&away_features);
        let home_mask_tensor = self.mask_to_tensor(&home_mask);
        let away_mask_tensor = self.mask_to_tensor(&away_mask);

        // Run inference
        let predictions = self.model.forward(
            home_tensor,
            away_tensor,
            Some(home_mask_tensor),
            Some(away_mask_tensor),
        );

        // Convert to output format
        let match_preds = MatchPrediction::from_predictions(&predictions);
        let pred = &match_preds[0];

        Ok(Prediction {
            home_team: home.id,
            away_team: away.id,
            home_win_prob: pred.home_win_prob,
            predicted_home_score: pred.home_score,
            predicted_away_score: pred.away_score,
            confidence,
        })
    }

    /// Predict multiple matches
    pub fn predict_batch(&self, matches: Vec<(String, String)>) -> Result<Vec<Result<Prediction>>> {
        matches
            .into_iter()
            .map(|(home, away)| self.predict(&home, &away))
            .map(Ok)
            .collect()
    }

    /// Convert matches to features with padding
    fn to_features(
        &self,
        matches: &[MatchRecord],
        team: TeamId,
    ) -> (Vec<MatchFeatures>, Vec<bool>) {
        let mut features = Vec::with_capacity(self.max_history);
        let mut mask = Vec::with_capacity(self.max_history);

        for m in matches {
            features.push(MatchFeatures::from_match(m, team));
            mask.push(true);
        }

        // Pad to max_history
        while features.len() < self.max_history {
            features.push(MatchFeatures::padding());
            mask.push(false);
        }

        (features, mask)
    }

    /// Convert features to tensor
    fn features_to_tensor(&self, features: &[MatchFeatures]) -> Tensor<B, 3> {
        let data: Vec<f32> = features.iter().flat_map(|f| f.to_vec()).collect();
        Tensor::<B, 1>::from_floats(data.as_slice(), &self.device).reshape([
            1,
            self.max_history,
            MatchFeatures::DIM,
        ])
    }

    /// Convert mask to tensor
    fn mask_to_tensor(&self, mask: &[bool]) -> Tensor<B, 2, burn::tensor::Bool> {
        Tensor::<B, 1, burn::tensor::Bool>::from_bool(
            burn::tensor::TensorData::from(mask),
            &self.device,
        )
        .reshape([1, self.max_history])
    }

    /// Compute confidence level based on history
    fn compute_confidence(&self, home_count: usize, away_count: usize) -> ConfidenceLevel {
        let threshold_high = self.max_history * 3 / 4;
        let threshold_medium = self.max_history / 2;

        if home_count >= threshold_high && away_count >= threshold_high {
            ConfidenceLevel::High
        } else if home_count >= threshold_medium || away_count >= threshold_medium {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }

    /// Get the database
    pub fn database(&self) -> &Database {
        &self.db
    }
}

/// Format a prediction for display
pub fn format_prediction(pred: &Prediction, home_name: &str, away_name: &str) -> String {
    let winner = if pred.home_win_prob >= 0.5 {
        home_name
    } else {
        away_name
    };
    let win_prob = if pred.home_win_prob >= 0.5 {
        pred.home_win_prob
    } else {
        1.0 - pred.home_win_prob
    };

    format!(
        r#"
┌─────────────────────────────────────────────────┐
│  {} vs {}
├─────────────────────────────────────────────────┤
│  Win probability:  {} {:.1}%
│  Predicted score:  {} {:.0} - {} {:.0}
│  Confidence:       {}
└─────────────────────────────────────────────────┘
"#,
        home_name,
        away_name,
        winner,
        win_prob * 100.0,
        home_name,
        pred.predicted_home_score,
        away_name,
        pred.predicted_away_score,
        pred.confidence
    )
}
