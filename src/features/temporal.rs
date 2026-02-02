//! Temporal feature extraction
//!
//! Time-based features that capture scheduling patterns, rest periods,
//! and seasonal effects.

use chrono::{Datelike, NaiveDate};
use std::collections::HashMap;

use crate::{MatchRecord, TeamId};

/// Temporal context for a match
#[derive(Debug, Clone, Copy, Default)]
pub struct TemporalContext {
    // === Day of Week (one-hot style) ===
    /// 1.0 if Saturday (most common), 0.0 otherwise
    pub is_saturday: f32,
    /// 1.0 if Friday (common night game), 0.0 otherwise
    pub is_friday: f32,
    /// 1.0 if Sunday, 0.0 otherwise
    pub is_sunday: f32,

    // === Season Progression ===
    /// Normalized position in season (0.0 = start, 1.0 = end)
    /// Based on month: Feb=0.0, Jul=1.0 for Super Rugby
    pub season_progress: f32,
    /// 1.0 if early season (Feb-Mar), teams finding form
    pub is_early_season: f32,
    /// 1.0 if late season (May-Jul), playoff pressure
    pub is_late_season: f32,
    /// Round number normalized (0-1, where 1.0 = round 20+)
    pub round_normalized: f32,

    // === Rest and Fatigue ===
    /// Days rest for home team (normalized: 0=short, 1=long)
    pub home_rest_days: f32,
    /// Days rest for away team (normalized)
    pub away_rest_days: f32,
    /// Rest advantage: home_rest - away_rest (normalized -1 to 1)
    pub rest_advantage: f32,
    /// 1.0 if home team has short turnaround (<7 days)
    pub home_short_turnaround: f32,
    /// 1.0 if away team has short turnaround (<7 days)
    pub away_short_turnaround: f32,

    // === Consecutive Scheduling ===
    /// Consecutive home games for home team (normalized 0-1)
    pub home_streak: f32,
    /// Consecutive away games for away team (normalized 0-1)
    pub away_streak: f32,

    // === Head-to-Head Recency ===
    /// Days since these teams last played (normalized: 0=recent, 1=long ago)
    pub h2h_recency: f32,
    /// Games played since last H2H meeting (normalized)
    pub games_since_h2h: f32,

    // === Match Density ===
    /// Matches in last 21 days for home team (normalized 0-1)
    pub home_match_density: f32,
    /// Matches in last 21 days for away team (normalized)
    pub away_match_density: f32,
}

impl TemporalContext {
    /// Number of features in this struct
    pub const DIM: usize = 18;

    /// Convert to flat vector
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            // Day of week (3)
            self.is_saturday,
            self.is_friday,
            self.is_sunday,
            // Season (4)
            self.season_progress,
            self.is_early_season,
            self.is_late_season,
            self.round_normalized,
            // Rest (5)
            self.home_rest_days,
            self.away_rest_days,
            self.rest_advantage,
            self.home_short_turnaround,
            self.away_short_turnaround,
            // Streaks (2)
            self.home_streak,
            self.away_streak,
            // H2H (2)
            self.h2h_recency,
            self.games_since_h2h,
            // Density (2)
            self.home_match_density,
            self.away_match_density,
        ]
    }

    /// Create from flat vector
    pub fn from_vec(v: &[f32]) -> Option<Self> {
        if v.len() != Self::DIM {
            return None;
        }
        Some(TemporalContext {
            is_saturday: v[0],
            is_friday: v[1],
            is_sunday: v[2],
            season_progress: v[3],
            is_early_season: v[4],
            is_late_season: v[5],
            round_normalized: v[6],
            home_rest_days: v[7],
            away_rest_days: v[8],
            rest_advantage: v[9],
            home_short_turnaround: v[10],
            away_short_turnaround: v[11],
            home_streak: v[12],
            away_streak: v[13],
            h2h_recency: v[14],
            games_since_h2h: v[15],
            home_match_density: v[16],
            away_match_density: v[17],
        })
    }
}

/// Computes temporal features for matches
pub struct TemporalFeatureComputer {
    /// Last match date per team
    last_match: HashMap<TeamId, NaiveDate>,
    /// Last match location (true = home) per team
    last_was_home: HashMap<TeamId, bool>,
    /// Consecutive home/away streak per team
    streak_count: HashMap<TeamId, i32>,
    /// Recent match dates per team (for density calculation)
    recent_matches: HashMap<TeamId, Vec<NaiveDate>>,
    /// Last H2H match date for each team pair
    last_h2h: HashMap<(TeamId, TeamId), NaiveDate>,
    /// Games played since last H2H for each team pair
    games_since_h2h_map: HashMap<(TeamId, TeamId), u32>,
}

impl Default for TemporalFeatureComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalFeatureComputer {
    pub fn new() -> Self {
        TemporalFeatureComputer {
            last_match: HashMap::new(),
            last_was_home: HashMap::new(),
            streak_count: HashMap::new(),
            recent_matches: HashMap::new(),
            last_h2h: HashMap::new(),
            games_since_h2h_map: HashMap::new(),
        }
    }

    /// Compute temporal context for a match (call BEFORE updating state)
    pub fn compute(&self, record: &MatchRecord) -> TemporalContext {
        let mut ctx = TemporalContext::default();

        // === Day of Week ===
        let weekday = record.date.weekday().num_days_from_monday();
        ctx.is_saturday = if weekday == 5 { 1.0 } else { 0.0 };
        ctx.is_friday = if weekday == 4 { 1.0 } else { 0.0 };
        ctx.is_sunday = if weekday == 6 { 1.0 } else { 0.0 };

        // === Season Progression ===
        // Super Rugby: Feb (2) to Jul (7) = 6 month season
        let month = record.date.month();
        ctx.season_progress = ((month as f32 - 2.0) / 5.0).clamp(0.0, 1.0);
        ctx.is_early_season = if month <= 3 { 1.0 } else { 0.0 };
        ctx.is_late_season = if month >= 5 { 1.0 } else { 0.0 };

        // Round number (if available)
        if let Some(round) = record.round {
            ctx.round_normalized = (round as f32 / 20.0).clamp(0.0, 1.0);
        }

        // === Rest Days ===
        let home_rest = self.days_since_last_match(record.home_team, record.date);
        let away_rest = self.days_since_last_match(record.away_team, record.date);

        // Normalize: 0 = 1 day (minimum), 1 = 14+ days (bye week)
        ctx.home_rest_days = Self::normalize_rest_days(home_rest);
        ctx.away_rest_days = Self::normalize_rest_days(away_rest);

        // Rest advantage: positive = home has more rest
        let rest_diff = home_rest.unwrap_or(7) as f32 - away_rest.unwrap_or(7) as f32;
        ctx.rest_advantage = (rest_diff / 7.0).clamp(-1.0, 1.0);

        // Short turnaround flags
        ctx.home_short_turnaround = if home_rest.map_or(false, |d| d < 7) {
            1.0
        } else {
            0.0
        };
        ctx.away_short_turnaround = if away_rest.map_or(false, |d| d < 7) {
            1.0
        } else {
            0.0
        };

        // === Consecutive Scheduling ===
        // Home team playing at home - check their streak
        let home_streak = self.get_streak(record.home_team, true);
        ctx.home_streak = (home_streak as f32 / 4.0).clamp(0.0, 1.0);

        // Away team playing away - check their streak
        let away_streak = self.get_streak(record.away_team, false);
        ctx.away_streak = (away_streak as f32 / 4.0).clamp(0.0, 1.0);

        // === Head-to-Head Recency ===
        let h2h_key = Self::h2h_key(record.home_team, record.away_team);
        if let Some(last_h2h_date) = self.last_h2h.get(&h2h_key) {
            let days_since_h2h = (record.date - *last_h2h_date).num_days();
            // Normalize: 0 = just played, 1 = 365+ days ago
            ctx.h2h_recency = (days_since_h2h as f32 / 365.0).clamp(0.0, 1.0);
        } else {
            ctx.h2h_recency = 1.0; // Never played = maximum recency
        }

        if let Some(games) = self.games_since_h2h_map.get(&h2h_key) {
            // Normalize: 0 = just played, 1 = 20+ games ago
            ctx.games_since_h2h = (*games as f32 / 20.0).clamp(0.0, 1.0);
        } else {
            ctx.games_since_h2h = 1.0;
        }

        // === Match Density ===
        ctx.home_match_density = self.match_density(record.home_team, record.date, 21);
        ctx.away_match_density = self.match_density(record.away_team, record.date, 21);

        ctx
    }

    /// Update state after a match is processed
    pub fn update(&mut self, record: &MatchRecord) {
        let date = record.date;

        // Update last match dates
        self.last_match.insert(record.home_team, date);
        self.last_match.insert(record.away_team, date);

        // Update streaks
        self.update_streak(record.home_team, true);
        self.update_streak(record.away_team, false);

        // Update recent matches for density
        self.add_recent_match(record.home_team, date);
        self.add_recent_match(record.away_team, date);

        // Update H2H tracking
        let h2h_key = Self::h2h_key(record.home_team, record.away_team);
        self.last_h2h.insert(h2h_key, date);
        self.games_since_h2h_map.insert(h2h_key, 0);

        // Increment games since H2H for all other pairs involving these teams
        self.increment_games_since_h2h(record.home_team, record.away_team);
    }

    /// Days since last match for a team
    fn days_since_last_match(&self, team: TeamId, current_date: NaiveDate) -> Option<i64> {
        self.last_match
            .get(&team)
            .map(|last| (current_date - *last).num_days())
    }

    /// Normalize rest days to 0-1 scale
    fn normalize_rest_days(days: Option<i64>) -> f32 {
        match days {
            Some(d) => ((d - 1) as f32 / 13.0).clamp(0.0, 1.0),
            None => 0.5, // Unknown = average
        }
    }

    /// Get current streak count (positive = consecutive in expected direction)
    fn get_streak(&self, team: TeamId, expecting_home: bool) -> i32 {
        let streak = *self.streak_count.get(&team).unwrap_or(&0);
        let last_was_home = *self.last_was_home.get(&team).unwrap_or(&true);

        // If last game matched what we're expecting, return streak
        // Otherwise return 0 (streak broken)
        if expecting_home == last_was_home {
            streak.max(0)
        } else {
            0
        }
    }

    /// Update streak after a match
    fn update_streak(&mut self, team: TeamId, is_home: bool) {
        let last_was_home = self.last_was_home.get(&team).copied();
        let current_streak = *self.streak_count.get(&team).unwrap_or(&0);

        let new_streak = if last_was_home == Some(is_home) {
            current_streak + 1
        } else {
            1 // Reset streak
        };

        self.streak_count.insert(team, new_streak);
        self.last_was_home.insert(team, is_home);
    }

    /// Add a match to recent matches history
    fn add_recent_match(&mut self, team: TeamId, date: NaiveDate) {
        let matches = self.recent_matches.entry(team).or_default();
        matches.push(date);
        // Keep only last 30 days of matches
        matches.retain(|d| (date - *d).num_days() <= 30);
    }

    /// Calculate match density (matches in last N days)
    fn match_density(&self, team: TeamId, current_date: NaiveDate, window_days: i64) -> f32 {
        let count = self
            .recent_matches
            .get(&team)
            .map(|matches| {
                matches
                    .iter()
                    .filter(|d| (current_date - **d).num_days() <= window_days)
                    .count()
            })
            .unwrap_or(0);

        // Normalize: 0 = 0 matches, 1 = 4+ matches in window (very dense)
        (count as f32 / 4.0).clamp(0.0, 1.0)
    }

    /// Create canonical key for H2H lookup (smaller ID first)
    fn h2h_key(team1: TeamId, team2: TeamId) -> (TeamId, TeamId) {
        if team1.0 < team2.0 {
            (team1, team2)
        } else {
            (team2, team1)
        }
    }

    /// Increment games since H2H for all pairs except the one that just played
    fn increment_games_since_h2h(&mut self, team1: TeamId, team2: TeamId) {
        let played_key = Self::h2h_key(team1, team2);

        // Increment all existing H2H counters involving either team
        let keys_to_update: Vec<_> = self
            .games_since_h2h_map
            .keys()
            .filter(|k| **k != played_key && (k.0 == team1 || k.1 == team1 || k.0 == team2 || k.1 == team2))
            .copied()
            .collect();

        for key in keys_to_update {
            if let Some(count) = self.games_since_h2h_map.get_mut(&key) {
                *count += 1;
            }
        }
    }

    /// Reset all state (for new computation pass)
    pub fn reset(&mut self) {
        self.last_match.clear();
        self.last_was_home.clear();
        self.streak_count.clear();
        self.recent_matches.clear();
        self.last_h2h.clear();
        self.games_since_h2h_map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataSource;

    fn make_match(
        home: i64,
        away: i64,
        date: NaiveDate,
        round: Option<u8>,
    ) -> MatchRecord {
        MatchRecord {
            date,
            home_team: TeamId(home),
            away_team: TeamId(away),
            home_score: 25,
            away_score: 20,
            venue: None,
            round,
            home_tries: None,
            away_tries: None,
            source: DataSource::Wikipedia,
        }
    }

    #[test]
    fn test_day_of_week() {
        let computer = TemporalFeatureComputer::new();

        // 2024-03-02 is a Saturday
        let saturday_match = make_match(1, 2, NaiveDate::from_ymd_opt(2024, 3, 2).unwrap(), None);
        let ctx = computer.compute(&saturday_match);
        assert_eq!(ctx.is_saturday, 1.0);
        assert_eq!(ctx.is_friday, 0.0);

        // 2024-03-01 is a Friday
        let friday_match = make_match(1, 2, NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(), None);
        let ctx = computer.compute(&friday_match);
        assert_eq!(ctx.is_friday, 1.0);
        assert_eq!(ctx.is_saturday, 0.0);
    }

    #[test]
    fn test_season_progression() {
        let computer = TemporalFeatureComputer::new();

        // February = early season
        let feb_match = make_match(1, 2, NaiveDate::from_ymd_opt(2024, 2, 15).unwrap(), None);
        let ctx = computer.compute(&feb_match);
        assert_eq!(ctx.is_early_season, 1.0);
        assert_eq!(ctx.is_late_season, 0.0);
        assert!(ctx.season_progress < 0.2);

        // June = late season
        let jun_match = make_match(1, 2, NaiveDate::from_ymd_opt(2024, 6, 15).unwrap(), None);
        let ctx = computer.compute(&jun_match);
        assert_eq!(ctx.is_early_season, 0.0);
        assert_eq!(ctx.is_late_season, 1.0);
        assert!(ctx.season_progress > 0.7);
    }

    #[test]
    fn test_rest_days() {
        let mut computer = TemporalFeatureComputer::new();

        // First match - no history
        let match1 = make_match(1, 2, NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(), None);
        computer.update(&match1);

        // Second match 7 days later
        let match2 = make_match(1, 3, NaiveDate::from_ymd_opt(2024, 3, 8).unwrap(), None);
        let ctx = computer.compute(&match2);

        // Home team (1) played 7 days ago
        assert!((ctx.home_rest_days - 0.46).abs() < 0.1); // (7-1)/13 ≈ 0.46
        // Away team (3) has no history
        assert_eq!(ctx.away_rest_days, 0.5);
    }

    #[test]
    fn test_short_turnaround() {
        let mut computer = TemporalFeatureComputer::new();

        let match1 = make_match(1, 2, NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(), None);
        computer.update(&match1);

        // Match 5 days later (short turnaround)
        let match2 = make_match(1, 3, NaiveDate::from_ymd_opt(2024, 3, 6).unwrap(), None);
        let ctx = computer.compute(&match2);

        assert_eq!(ctx.home_short_turnaround, 1.0);
        assert_eq!(ctx.away_short_turnaround, 0.0); // Team 3 has no history
    }

    #[test]
    fn test_home_streak() {
        let mut computer = TemporalFeatureComputer::new();

        // Team 1 plays 3 consecutive home games
        for i in 0..3 {
            let m = make_match(
                1,
                2 + i,
                NaiveDate::from_ymd_opt(2024, 3, 1 + (i * 7) as u32).unwrap(),
                None,
            );
            computer.update(&m);
        }

        // Fourth home game
        let match4 = make_match(1, 5, NaiveDate::from_ymd_opt(2024, 3, 22).unwrap(), None);
        let ctx = computer.compute(&match4);

        // Should have streak of 3 (previous games)
        assert!(ctx.home_streak > 0.5); // 3/4 = 0.75
    }

    #[test]
    fn test_h2h_recency() {
        let mut computer = TemporalFeatureComputer::new();

        // Teams 1 and 2 play
        let match1 = make_match(1, 2, NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(), None);
        computer.update(&match1);

        // 60 days later they play again
        let match2 = make_match(2, 1, NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(), None);
        let ctx = computer.compute(&match2);

        // Should have ~60/365 ≈ 0.16 recency
        assert!(ctx.h2h_recency > 0.1 && ctx.h2h_recency < 0.25);
    }

    #[test]
    fn test_match_density() {
        let mut computer = TemporalFeatureComputer::new();

        // Team 1 plays 3 matches in 14 days
        for i in 0..3 {
            let m = make_match(
                1,
                2 + i,
                NaiveDate::from_ymd_opt(2024, 3, 1 + (i * 5) as u32).unwrap(),
                None,
            );
            computer.update(&m);
        }

        // Check density on day 21
        let check_match = make_match(1, 5, NaiveDate::from_ymd_opt(2024, 3, 21).unwrap(), None);
        let ctx = computer.compute(&check_match);

        // 3 matches in 21 days = 3/4 = 0.75 density
        assert!(ctx.home_match_density > 0.5);
    }

    #[test]
    fn test_to_from_vec() {
        let ctx = TemporalContext {
            is_saturday: 1.0,
            is_friday: 0.0,
            is_sunday: 0.0,
            season_progress: 0.5,
            is_early_season: 0.0,
            is_late_season: 1.0,
            round_normalized: 0.6,
            home_rest_days: 0.5,
            away_rest_days: 0.3,
            rest_advantage: 0.2,
            home_short_turnaround: 0.0,
            away_short_turnaround: 1.0,
            home_streak: 0.5,
            away_streak: 0.25,
            h2h_recency: 0.3,
            games_since_h2h: 0.4,
            home_match_density: 0.5,
            away_match_density: 0.75,
        };

        let vec = ctx.to_vec();
        assert_eq!(vec.len(), TemporalContext::DIM);

        let restored = TemporalContext::from_vec(&vec).unwrap();
        assert_eq!(restored.is_saturday, 1.0);
        assert_eq!(restored.is_late_season, 1.0);
        assert_eq!(restored.away_short_turnaround, 1.0);
    }
}
