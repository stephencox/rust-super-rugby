//! Match feature representation for transformer input
//!
//! Each match in a team's history is encoded as a feature vector.

use crate::data::dataset::FeatureNormalization;
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
    /// Travel hours (timezone difference from previous match, normalized)
    pub travel_hours: f32,
    /// Days rest since previous match (normalized)
    pub days_rest: f32,
    // === Rolling statistics (computed from match history) ===
    /// Rolling win rate over recent matches (0-1)
    pub rolling_win_rate: f32,
    /// Rolling average margin (z-score normalized)
    pub rolling_margin_avg: f32,
    /// Pythagorean expectation: PF^2.37 / (PF^2.37 + PA^2.37)
    pub pythagorean_exp: f32,
    /// Performance consistency (inverse of margin stdev, normalized)
    pub consistency: f32,
}

impl MatchFeatures {
    /// Dimension of feature vector
    pub const DIM: usize = 15;

    /// Create features from a match record (uses default normalization)
    pub fn from_match(record: &MatchRecord, perspective_team: TeamId) -> Self {
        Self::from_match_normalized(record, perspective_team, &FeatureNormalization::default())
    }

    /// Create features from a match record with z-score normalization
    pub fn from_match_normalized(
        record: &MatchRecord,
        perspective_team: TeamId,
        norm: &FeatureNormalization,
    ) -> Self {
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
            // Z-score normalize scores
            score_for: (score_for - norm.score_mean) / norm.score_std,
            score_against: (score_against - norm.score_mean) / norm.score_std,
            // Z-score normalize margin
            margin: (margin - norm.margin_mean) / norm.margin_std,
            // Z-score normalize tries
            tries_for: (tries_for - norm.tries_mean) / norm.tries_std,
            tries_against: (tries_against - norm.tries_mean) / norm.tries_std,
            // Binary/categorical features remain as-is
            is_home: if is_home { 1.0 } else { 0.0 },
            win_indicator,
            // Recency will be set externally based on match date
            recency: 0.0,
            // Opponent index normalized 0-1
            opponent_idx: (opponent.0 % 32) as f32 / 32.0,
            // Travel features set externally based on previous match
            travel_hours: 0.0,
            days_rest: 0.0,
            // Rolling stats set externally based on match history
            rolling_win_rate: 0.0,
            rolling_margin_avg: 0.0,
            pythagorean_exp: 0.0,
            consistency: 0.0,
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
            travel_hours: 0.0,
            days_rest: 0.0,
            rolling_win_rate: 0.0,
            rolling_margin_avg: 0.0,
            pythagorean_exp: 0.0,
            consistency: 0.0,
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
            self.travel_hours,
            self.days_rest,
            self.rolling_win_rate,
            self.rolling_margin_avg,
            self.pythagorean_exp,
            self.consistency,
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
            travel_hours: v[9],
            days_rest: v[10],
            rolling_win_rate: v[11],
            rolling_margin_avg: v[12],
            pythagorean_exp: v[13],
            consistency: v[14],
        })
    }

    /// Set recency based on days before target match
    pub fn with_recency(mut self, days_before: i64, max_days: i64) -> Self {
        // Normalize to 0-1 range (0 = most recent, 1 = oldest)
        self.recency = (days_before as f32 / max_days as f32).min(1.0);
        self
    }

    /// Set travel context based on timezone difference and days rest
    ///
    /// # Arguments
    /// * `prev_match_tz` - Timezone offset of previous match venue (home team's tz)
    /// * `this_match_tz` - Timezone offset of this match venue
    /// * `days_since_prev` - Days since previous match
    pub fn with_travel(mut self, prev_match_tz: i32, this_match_tz: i32, days_since_prev: i64) -> Self {
        // Calculate timezone difference (jetlag)
        // Positive = traveled east, negative = traveled west
        let tz_diff = this_match_tz - prev_match_tz;
        // Normalize: max timezone diff is ~15 hours (Buenos Aires to NZ)
        self.travel_hours = (tz_diff as f32 / 15.0).clamp(-1.0, 1.0);

        // Days rest: normalize to 0-1 scale (0 = 1 day, 1 = 14+ days)
        // Short turnaround (<7 days) is common in Super Rugby
        self.days_rest = ((days_since_prev - 1) as f32 / 13.0).clamp(0.0, 1.0);
        self
    }

    /// Set rolling statistics computed from match history
    ///
    /// # Arguments
    /// * `win_rate` - Win rate over recent matches (0-1)
    /// * `margin_avg` - Average margin (already z-score normalized)
    /// * `points_for_total` - Total points scored (for pythagorean)
    /// * `points_against_total` - Total points conceded (for pythagorean)
    /// * `margin_std` - Standard deviation of margins (for consistency)
    pub fn with_rolling_stats(
        mut self,
        win_rate: f32,
        margin_avg: f32,
        points_for_total: f32,
        points_against_total: f32,
        margin_std: f32,
    ) -> Self {
        self.rolling_win_rate = win_rate;
        self.rolling_margin_avg = margin_avg;

        // Pythagorean expectation: PF^2.37 / (PF^2.37 + PA^2.37)
        // This is a better predictor of future performance than win rate
        let exp = 2.37;
        let pf_pow = points_for_total.powf(exp);
        let pa_pow = points_against_total.powf(exp);
        self.pythagorean_exp = if pf_pow + pa_pow > 0.0 {
            pf_pow / (pf_pow + pa_pow)
        } else {
            0.5 // No data, assume average
        };

        // Consistency: inverse of margin stdev, normalized
        // High consistency (low variance) = closer to 1.0
        // Max std ~30 points in rugby
        self.consistency = 1.0 - (margin_std / 30.0).clamp(0.0, 1.0);

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
            travel_hours: 0.5,
            days_rest: 0.3,
            rolling_win_rate: 0.6,
            rolling_margin_avg: 0.2,
            pythagorean_exp: 0.55,
            consistency: 0.7,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), MatchFeatures::DIM);

        let restored = MatchFeatures::from_vec(&vec).unwrap();
        assert_eq!(restored.score_for, features.score_for);
        assert_eq!(restored.travel_hours, features.travel_hours);
        assert_eq!(restored.rolling_win_rate, features.rolling_win_rate);
        assert_eq!(restored.pythagorean_exp, features.pythagorean_exp);
    }

    #[test]
    fn test_travel_feature() {
        let features = MatchFeatures::padding()
            .with_travel(2, 12, 7);  // SA to NZ, 7 days rest

        // 10 hour timezone difference
        assert!((features.travel_hours - 0.667).abs() < 0.01);
        // 7-1=6 days / 13 = ~0.46
        assert!((features.days_rest - 0.46).abs() < 0.01);
    }
}
