//! Team statistics computation
//!
//! Rolling statistics for teams based on match history.

use crate::{MatchRecord, TeamId};
use std::collections::HashMap;

/// Rolling statistics for a team
#[derive(Debug, Clone, Default)]
pub struct TeamStatistics {
    /// Total matches played
    pub matches_played: usize,
    /// Wins
    pub wins: usize,
    /// Losses
    pub losses: usize,
    /// Draws
    pub draws: usize,
    /// Total points scored
    pub points_for: u32,
    /// Total points conceded
    pub points_against: u32,
    /// Total tries scored
    pub tries_for: u32,
    /// Total tries conceded
    pub tries_against: u32,
    /// Home wins
    pub home_wins: usize,
    /// Home matches
    pub home_matches: usize,
    /// Away wins
    pub away_wins: usize,
    /// Away matches
    pub away_matches: usize,
}

impl TeamStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with a match result
    pub fn update(&mut self, record: &MatchRecord, team: TeamId) {
        let is_home = record.home_team == team;

        let (score_for, score_against) = if is_home {
            (record.home_score, record.away_score)
        } else {
            (record.away_score, record.home_score)
        };

        self.matches_played += 1;
        self.points_for += score_for as u32;
        self.points_against += score_against as u32;

        if let (Some(ht), Some(at)) = (record.home_tries, record.away_tries) {
            if is_home {
                self.tries_for += ht as u32;
                self.tries_against += at as u32;
            } else {
                self.tries_for += at as u32;
                self.tries_against += ht as u32;
            }
        }

        if is_home {
            self.home_matches += 1;
        } else {
            self.away_matches += 1;
        }

        match score_for.cmp(&score_against) {
            std::cmp::Ordering::Greater => {
                self.wins += 1;
                if is_home {
                    self.home_wins += 1;
                } else {
                    self.away_wins += 1;
                }
            }
            std::cmp::Ordering::Less => {
                self.losses += 1;
            }
            std::cmp::Ordering::Equal => {
                self.draws += 1;
            }
        }
    }

    /// Win ratio (0-1)
    pub fn win_ratio(&self) -> f32 {
        if self.matches_played == 0 {
            0.5
        } else {
            self.wins as f32 / self.matches_played as f32
        }
    }

    /// Home win ratio (0-1)
    pub fn home_win_ratio(&self) -> f32 {
        if self.home_matches == 0 {
            0.5
        } else {
            self.home_wins as f32 / self.home_matches as f32
        }
    }

    /// Away win ratio (0-1)
    pub fn away_win_ratio(&self) -> f32 {
        if self.away_matches == 0 {
            0.5
        } else {
            self.away_wins as f32 / self.away_matches as f32
        }
    }

    /// Average points scored per match
    pub fn avg_points_for(&self) -> f32 {
        if self.matches_played == 0 {
            0.0
        } else {
            self.points_for as f32 / self.matches_played as f32
        }
    }

    /// Average points conceded per match
    pub fn avg_points_against(&self) -> f32 {
        if self.matches_played == 0 {
            0.0
        } else {
            self.points_against as f32 / self.matches_played as f32
        }
    }

    /// Average point differential per match
    pub fn avg_margin(&self) -> f32 {
        self.avg_points_for() - self.avg_points_against()
    }

    /// Calculate Log5 probability of beating another team
    pub fn log5_probability(&self, opponent: &TeamStatistics) -> f32 {
        let p_a = self.win_ratio();
        let p_b = opponent.win_ratio();

        // Avoid division by zero
        if (p_a + p_b - 2.0 * p_a * p_b).abs() < 1e-6 {
            return 0.5;
        }

        (p_a - p_a * p_b) / (p_a + p_b - 2.0 * p_a * p_b)
    }
}

/// Compute statistics for all teams over a window of matches
pub struct TeamStatisticsComputer {
    /// Statistics by team
    stats: HashMap<TeamId, TeamStatistics>,
}

impl TeamStatisticsComputer {
    /// Create new computer
    pub fn new() -> Self {
        TeamStatisticsComputer {
            stats: HashMap::new(),
        }
    }

    /// Process matches and build statistics
    pub fn process_matches(&mut self, matches: &[MatchRecord]) {
        for record in matches {
            // Update home team stats
            self.stats
                .entry(record.home_team)
                .or_default()
                .update(record, record.home_team);

            // Update away team stats
            self.stats
                .entry(record.away_team)
                .or_default()
                .update(record, record.away_team);
        }
    }

    /// Get statistics for a team
    pub fn get(&self, team: TeamId) -> Option<&TeamStatistics> {
        self.stats.get(&team)
    }

    /// Get statistics for a team, or default if not found
    pub fn get_or_default(&self, team: TeamId) -> TeamStatistics {
        self.stats.get(&team).cloned().unwrap_or_default()
    }

    /// Get all team statistics
    pub fn all(&self) -> &HashMap<TeamId, TeamStatistics> {
        &self.stats
    }
}

impl Default for TeamStatisticsComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute rolling statistics with a limited window
pub struct RollingStatistics {
    /// Window size (number of recent matches)
    window: usize,
    /// Recent matches per team
    recent_matches: HashMap<TeamId, Vec<MatchRecord>>,
}

impl RollingStatistics {
    /// Create new rolling statistics with given window size
    pub fn new(window: usize) -> Self {
        RollingStatistics {
            window,
            recent_matches: HashMap::new(),
        }
    }

    /// Add a match and update rolling window
    pub fn add_match(&mut self, record: &MatchRecord) {
        // Add to home team's window
        let home_matches = self.recent_matches.entry(record.home_team).or_default();
        home_matches.push(record.clone());
        if home_matches.len() > self.window {
            home_matches.remove(0);
        }

        // Add to away team's window
        let away_matches = self.recent_matches.entry(record.away_team).or_default();
        away_matches.push(record.clone());
        if away_matches.len() > self.window {
            away_matches.remove(0);
        }
    }

    /// Get statistics for a team based on rolling window
    pub fn get_stats(&self, team: TeamId) -> TeamStatistics {
        let mut stats = TeamStatistics::new();

        if let Some(matches) = self.recent_matches.get(&team) {
            for record in matches {
                stats.update(record, team);
            }
        }

        stats
    }

    /// Get number of matches in window for a team
    pub fn match_count(&self, team: TeamId) -> usize {
        self.recent_matches.get(&team).map(|m| m.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataSource;
    use chrono::NaiveDate;

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
    fn test_team_statistics() {
        let mut stats = TeamStatistics::new();

        // Home win
        stats.update(&make_match(1, 2, 30, 20), TeamId(1));
        assert_eq!(stats.wins, 1);
        assert_eq!(stats.home_wins, 1);
        assert_eq!(stats.points_for, 30);

        // Away loss
        stats.update(&make_match(2, 1, 25, 15), TeamId(1));
        assert_eq!(stats.losses, 1);
        assert_eq!(stats.away_matches, 1);
    }

    #[test]
    fn test_log5() {
        let mut team_a = TeamStatistics::new();
        team_a.matches_played = 10;
        team_a.wins = 8; // 80% win rate

        let mut team_b = TeamStatistics::new();
        team_b.matches_played = 10;
        team_b.wins = 5; // 50% win rate

        let prob = team_a.log5_probability(&team_b);
        assert!(prob > 0.5); // Team A should be favored
        assert!(prob < 1.0);
    }
}
