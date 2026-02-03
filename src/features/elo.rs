//! Elo rating system for team strength estimation
//!
//! Computes dynamic team ratings from match history.

use std::collections::HashMap;

use crate::{MatchRecord, TeamId};

/// Elo rating configuration
pub struct EloConfig {
    /// K-factor: how much ratings change per match
    pub k_factor: f32,
    /// Home advantage in rating points
    pub home_advantage: f32,
    /// Starting rating for new teams
    pub initial_rating: f32,
}

impl Default for EloConfig {
    fn default() -> Self {
        EloConfig {
            k_factor: 32.0,
            home_advantage: 50.0,
            initial_rating: 1500.0,
        }
    }
}

/// Elo rating computer
pub struct EloRatings {
    ratings: HashMap<TeamId, f32>,
    config: EloConfig,
}

impl Default for EloRatings {
    fn default() -> Self {
        Self::new(EloConfig::default())
    }
}

impl EloRatings {
    pub fn new(config: EloConfig) -> Self {
        EloRatings {
            ratings: HashMap::new(),
            config,
        }
    }

    /// Get current rating for a team (returns initial if unknown)
    pub fn get_rating(&self, team: TeamId) -> f32 {
        *self
            .ratings
            .get(&team)
            .unwrap_or(&self.config.initial_rating)
    }

    /// Compute expected score (0-1) for home team
    pub fn expected_score(&self, home: TeamId, away: TeamId) -> f32 {
        let home_rating = self.get_rating(home) + self.config.home_advantage;
        let away_rating = self.get_rating(away);
        let diff = away_rating - home_rating;
        1.0 / (1.0 + 10.0_f32.powf(diff / 400.0))
    }

    /// Update ratings after a match (call AFTER getting pre-match ratings)
    pub fn update(&mut self, record: &MatchRecord) {
        let home_expected = self.expected_score(record.home_team, record.away_team);

        // Actual result: 1 = home win, 0.5 = draw, 0 = away win
        let home_actual = if record.home_score > record.away_score {
            1.0
        } else if record.home_score == record.away_score {
            0.5
        } else {
            0.0
        };

        let home_rating = self.get_rating(record.home_team);
        let away_rating = self.get_rating(record.away_team);

        // Update ratings
        let home_new = home_rating + self.config.k_factor * (home_actual - home_expected);
        let away_new =
            away_rating + self.config.k_factor * ((1.0 - home_actual) - (1.0 - home_expected));

        self.ratings.insert(record.home_team, home_new);
        self.ratings.insert(record.away_team, away_new);
    }

    /// Get normalized rating difference (for features)
    /// Normalized to roughly -1 to 1 range (plus/minus 300 rating points)
    pub fn rating_diff_normalized(&self, home: TeamId, away: TeamId) -> f32 {
        let home_rating = self.get_rating(home);
        let away_rating = self.get_rating(away);
        (home_rating - away_rating) / 300.0
    }

    /// Get normalized rating (centered around 1500, scaled by 300)
    pub fn rating_normalized(&self, team: TeamId) -> f32 {
        (self.get_rating(team) - 1500.0) / 300.0
    }

    /// Reset all ratings
    pub fn reset(&mut self) {
        self.ratings.clear();
    }
}

/// Elo features for a match
#[derive(Debug, Clone, Copy, Default)]
pub struct EloFeatures {
    pub home_elo: f32,
    pub away_elo: f32,
    pub elo_diff: f32,
}

impl EloFeatures {
    pub const DIM: usize = 3;

    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.home_elo, self.away_elo, self.elo_diff]
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;
    use crate::DataSource;

    fn make_match(home: i64, away: i64, home_score: u8, away_score: u8) -> MatchRecord {
        MatchRecord {
            date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            home_team: TeamId(home),
            away_team: TeamId(away),
            home_score,
            away_score,
            venue: None,
            round: None,
            home_tries: None,
            away_tries: None,
            source: DataSource::Wikipedia,
        }
    }

    #[test]
    fn test_initial_ratings() {
        let elo = EloRatings::default();
        assert_eq!(elo.get_rating(TeamId(1)), 1500.0);
        assert_eq!(elo.get_rating(TeamId(999)), 1500.0);
    }

    #[test]
    fn test_expected_score() {
        let elo = EloRatings::default();
        // Equal teams, home advantage gives ~57% expected
        let expected = elo.expected_score(TeamId(1), TeamId(2));
        assert!(expected > 0.5 && expected < 0.6);
    }

    #[test]
    fn test_rating_update_home_win() {
        let mut elo = EloRatings::default();
        let m = make_match(1, 2, 30, 20);

        elo.update(&m);

        // Home team should gain rating, away should lose
        assert!(elo.get_rating(TeamId(1)) > 1500.0);
        assert!(elo.get_rating(TeamId(2)) < 1500.0);
    }

    #[test]
    fn test_rating_update_away_win() {
        let mut elo = EloRatings::default();
        let m = make_match(1, 2, 20, 30);

        elo.update(&m);

        // Away team upset win = bigger rating gain
        assert!(elo.get_rating(TeamId(1)) < 1500.0);
        assert!(elo.get_rating(TeamId(2)) > 1500.0);
    }

    #[test]
    fn test_normalized_diff() {
        let mut elo = EloRatings::default();
        // Simulate several wins for team 1
        for _ in 0..5 {
            elo.update(&make_match(1, 2, 30, 20));
        }

        let diff = elo.rating_diff_normalized(TeamId(1), TeamId(2));
        assert!(diff > 0.0); // Team 1 is stronger
    }
}
