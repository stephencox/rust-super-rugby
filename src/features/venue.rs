//! Venue performance tracking
//!
//! Tracks team performance at specific venues.

use std::collections::HashMap;

use crate::TeamId;

/// Tracks team performance at venues
pub struct VenueTracker {
    /// (team, venue) -> (wins, games)
    stats: HashMap<(TeamId, String), (u32, u32)>,
    /// Maximum games any team has at any venue (for normalization)
    max_games: u32,
}

impl Default for VenueTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl VenueTracker {
    pub fn new() -> Self {
        VenueTracker {
            stats: HashMap::new(),
            max_games: 1,
        }
    }

    /// Get venue stats for a team
    fn get_stats(&self, team: TeamId, venue: &str) -> (u32, u32) {
        let key = (team, venue.to_lowercase());
        *self.stats.get(&key).unwrap_or(&(0, 0))
    }

    /// Compute venue features for a match (call BEFORE update)
    pub fn compute(&self, home: TeamId, away: TeamId, venue: Option<&str>) -> VenueFeatures {
        let venue = match venue {
            Some(v) => v,
            None => return VenueFeatures::default(),
        };

        let (home_wins, home_games) = self.get_stats(home, venue);
        let (away_wins, away_games) = self.get_stats(away, venue);

        let home_win_rate = if home_games > 0 {
            home_wins as f32 / home_games as f32
        } else {
            0.5 // Unknown = neutral
        };

        let away_win_rate = if away_games > 0 {
            away_wins as f32 / away_games as f32
        } else {
            0.5
        };

        let home_games_norm = (home_games as f32 / self.max_games as f32).min(1.0);
        let away_games_norm = (away_games as f32 / self.max_games as f32).min(1.0);

        VenueFeatures {
            home_venue_win_rate: home_win_rate,
            home_venue_games: home_games_norm,
            away_venue_win_rate: away_win_rate,
            away_venue_games: away_games_norm,
            venue_familiarity_diff: home_games_norm - away_games_norm,
        }
    }

    /// Update stats after a match
    pub fn update(&mut self, home: TeamId, away: TeamId, venue: Option<&str>, home_won: bool) {
        let venue = match venue {
            Some(v) => v.to_lowercase(),
            None => return,
        };

        // Update home team stats
        let home_key = (home, venue.clone());
        let home_entry = self.stats.entry(home_key).or_insert((0, 0));
        if home_won {
            home_entry.0 += 1;
        }
        home_entry.1 += 1;
        self.max_games = self.max_games.max(home_entry.1);

        // Update away team stats
        let away_key = (away, venue);
        let away_entry = self.stats.entry(away_key).or_insert((0, 0));
        if !home_won {
            away_entry.0 += 1;
        }
        away_entry.1 += 1;
        self.max_games = self.max_games.max(away_entry.1);
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.stats.clear();
        self.max_games = 1;
    }
}

/// Venue performance features
#[derive(Debug, Clone, Copy, Default)]
pub struct VenueFeatures {
    pub home_venue_win_rate: f32,
    pub home_venue_games: f32,
    pub away_venue_win_rate: f32,
    pub away_venue_games: f32,
    pub venue_familiarity_diff: f32,
}

impl VenueFeatures {
    pub const DIM: usize = 5;

    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.home_venue_win_rate,
            self.home_venue_games,
            self.away_venue_win_rate,
            self.away_venue_games,
            self.venue_familiarity_diff,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_venue() {
        let tracker = VenueTracker::new();
        let features = tracker.compute(TeamId(1), TeamId(2), None);

        assert_eq!(features.home_venue_win_rate, 0.0);
        assert_eq!(features.home_venue_games, 0.0);
    }

    #[test]
    fn test_unknown_venue() {
        let tracker = VenueTracker::new();
        let features = tracker.compute(TeamId(1), TeamId(2), Some("Unknown Stadium"));

        // Unknown = neutral 0.5 win rate
        assert_eq!(features.home_venue_win_rate, 0.5);
        assert_eq!(features.home_venue_games, 0.0);
    }

    #[test]
    fn test_venue_tracking() {
        let mut tracker = VenueTracker::new();

        // Team 1 wins at Twickenham
        tracker.update(TeamId(1), TeamId(2), Some("Twickenham"), true);
        tracker.update(TeamId(1), TeamId(3), Some("Twickenham"), true);

        let features = tracker.compute(TeamId(1), TeamId(4), Some("Twickenham"));

        assert_eq!(features.home_venue_win_rate, 1.0); // 2/2 wins
        assert!(features.home_venue_games > 0.0);
    }

    #[test]
    fn test_familiarity_diff() {
        let mut tracker = VenueTracker::new();

        // Team 1 plays at Eden Park many times
        for i in 0..5 {
            tracker.update(TeamId(1), TeamId(i + 10), Some("Eden Park"), true);
        }

        // Team 2 never played there
        let features = tracker.compute(TeamId(1), TeamId(2), Some("Eden Park"));

        // Home team much more familiar
        assert!(features.venue_familiarity_diff > 0.5);
    }
}
