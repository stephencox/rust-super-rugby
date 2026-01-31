//! Match feature representation for transformer input
//!
//! Each match in a team's history is encoded as a feature vector.

use crate::{MatchRecord, TeamId};

/// Features extracted from a single match, from a specific team's perspective
#[derive(Debug, Clone)]
pub struct MatchFeatures {
    /// Score for (points scored by this team)
    pub score_for: f32,
    /// Score against (points conceded)
    pub score_against: f32,
    /// Score margin (positive = win)
    pub margin: f32,
    /// Tries scored (if available, else 0)
    pub tries_for: f32,
    /// Tries conceded (if available, else 0)
    pub tries_against: f32,
    /// 1.0 if home, 0.0 if away
    pub is_home: f32,
    /// 1.0 if won, 0.0 if lost, 0.5 if draw
    pub win_indicator: f32,
    /// Days since match (normalized, 0-1 scale for ~1 year)
    pub recency: f32,
    /// Opponent team embedding index (for learned embeddings)
    pub opponent_idx: f32,
}

impl MatchFeatures {
    /// Dimension of feature vector
    pub const DIM: usize = 9;

    /// Create features from a match record
    pub fn from_match(record: &MatchRecord, perspective_team: TeamId) -> Self {
        let is_home = record.home_team == perspective_team;

        let (score_for, score_against) = if is_home {
            (record.home_score as f32, record.away_score as f32)
        } else {
            (record.away_score as f32, record.home_score as f32)
        };

        let (tries_for, tries_against) = if is_home {
            (
                record.home_tries.unwrap_or(0) as f32,
                record.away_tries.unwrap_or(0) as f32,
            )
        } else {
            (
                record.away_tries.unwrap_or(0) as f32,
                record.home_tries.unwrap_or(0) as f32,
            )
        };

        let margin = score_for - score_against;

        let win_indicator = if score_for > score_against {
            1.0
        } else if score_for < score_against {
            0.0
        } else {
            0.5
        };

        let opponent = if is_home {
            record.away_team
        } else {
            record.home_team
        };

        MatchFeatures {
            // Normalize scores (typical range 0-50)
            score_for: score_for / 50.0,
            score_against: score_against / 50.0,
            // Normalize margin (typical range -40 to +40)
            margin: margin / 40.0,
            // Normalize tries (typical range 0-7)
            tries_for: tries_for / 7.0,
            tries_against: tries_against / 7.0,
            is_home: if is_home { 1.0 } else { 0.0 },
            win_indicator,
            // Recency will be set externally based on match date
            recency: 0.0,
            // Opponent index for embedding lookup
            opponent_idx: (opponent.0 % 32) as f32 / 32.0,
        }
    }

    /// Create a padding/zero feature vector
    pub fn padding() -> Self {
        MatchFeatures {
            score_for: 0.0,
            score_against: 0.0,
            margin: 0.0,
            tries_for: 0.0,
            tries_against: 0.0,
            is_home: 0.0,
            win_indicator: 0.0,
            recency: 0.0,
            opponent_idx: 0.0,
        }
    }

    /// Convert to a flat vector
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.score_for,
            self.score_against,
            self.margin,
            self.tries_for,
            self.tries_against,
            self.is_home,
            self.win_indicator,
            self.recency,
            self.opponent_idx,
        ]
    }

    /// Create from a flat vector
    pub fn from_vec(v: &[f32]) -> Option<Self> {
        if v.len() != Self::DIM {
            return None;
        }
        Some(MatchFeatures {
            score_for: v[0],
            score_against: v[1],
            margin: v[2],
            tries_for: v[3],
            tries_against: v[4],
            is_home: v[5],
            win_indicator: v[6],
            recency: v[7],
            opponent_idx: v[8],
        })
    }

    /// Set recency based on days before target match
    pub fn with_recency(mut self, days_before: i64, max_days: i64) -> Self {
        // Normalize to 0-1 range (0 = most recent, 1 = oldest)
        self.recency = (days_before as f32 / max_days as f32).min(1.0);
        self
    }
}

/// Extended features including context about the match situation
#[derive(Debug, Clone)]
pub struct ExtendedMatchFeatures {
    pub base: MatchFeatures,
    /// Round number in season (normalized)
    pub round: f32,
    /// Month of year (normalized 0-1)
    pub month: f32,
    /// Is opponent from same country?
    pub is_local: f32,
    /// Is this a playoff match?
    pub is_playoff: f32,
}

impl ExtendedMatchFeatures {
    pub const DIM: usize = MatchFeatures::DIM + 4;

    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = self.base.to_vec();
        v.extend([self.round, self.month, self.is_local, self.is_playoff]);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataSource;
    use chrono::NaiveDate;

    #[test]
    fn test_from_match_home() {
        let record = MatchRecord {
            date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            home_team: TeamId(1),
            away_team: TeamId(2),
            home_score: 28,
            away_score: 24,
            venue: None,
            round: Some(1),
            home_tries: Some(4),
            away_tries: Some(3),
            source: DataSource::Wikipedia,
        };

        let features = MatchFeatures::from_match(&record, TeamId(1));
        assert_eq!(features.is_home, 1.0);
        assert_eq!(features.win_indicator, 1.0);
        assert!(features.margin > 0.0);
    }

    #[test]
    fn test_from_match_away() {
        let record = MatchRecord {
            date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            home_team: TeamId(1),
            away_team: TeamId(2),
            home_score: 28,
            away_score: 24,
            venue: None,
            round: Some(1),
            home_tries: Some(4),
            away_tries: Some(3),
            source: DataSource::Wikipedia,
        };

        let features = MatchFeatures::from_match(&record, TeamId(2));
        assert_eq!(features.is_home, 0.0);
        assert_eq!(features.win_indicator, 0.0);
        assert!(features.margin < 0.0);
    }

    #[test]
    fn test_to_from_vec() {
        let features = MatchFeatures {
            score_for: 0.5,
            score_against: 0.4,
            margin: 0.1,
            tries_for: 0.5,
            tries_against: 0.4,
            is_home: 1.0,
            win_indicator: 1.0,
            recency: 0.2,
            opponent_idx: 0.3,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), MatchFeatures::DIM);

        let restored = MatchFeatures::from_vec(&vec).unwrap();
        assert_eq!(restored.score_for, features.score_for);
    }
}
