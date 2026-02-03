//! Workload window features
//!
//! Tracks match density over 7/14/21 day windows to approximate fatigue.

use std::collections::HashMap;

use chrono::NaiveDate;

use crate::TeamId;

/// Computes workload features from match history
pub struct WorkloadComputer {
    /// Recent match dates per team
    match_history: HashMap<TeamId, Vec<NaiveDate>>,
}

impl Default for WorkloadComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkloadComputer {
    pub fn new() -> Self {
        WorkloadComputer {
            match_history: HashMap::new(),
        }
    }

    /// Count matches in last N days for a team
    pub fn matches_in_window(&self, team: TeamId, current_date: NaiveDate, days: i64) -> u32 {
        self.match_history
            .get(&team)
            .map(|dates| {
                dates
                    .iter()
                    .filter(|d| {
                        let diff = (current_date - **d).num_days();
                        diff > 0 && diff <= days
                    })
                    .count() as u32
            })
            .unwrap_or(0)
    }

    /// Compute workload features for a match (call BEFORE update)
    pub fn compute(&self, home: TeamId, away: TeamId, date: NaiveDate) -> WorkloadFeatures {
        let home_7d = self.matches_in_window(home, date, 7);
        let home_14d = self.matches_in_window(home, date, 14);
        let home_21d = self.matches_in_window(home, date, 21);

        let away_7d = self.matches_in_window(away, date, 7);
        let away_14d = self.matches_in_window(away, date, 14);
        let away_21d = self.matches_in_window(away, date, 21);

        WorkloadFeatures {
            // Normalize: 0-2 matches typical, so /2 gives 0-1 range
            home_matches_7d: (home_7d as f32 / 2.0).min(1.0),
            home_matches_14d: (home_14d as f32 / 3.0).min(1.0),
            home_matches_21d: (home_21d as f32 / 4.0).min(1.0),
            away_matches_7d: (away_7d as f32 / 2.0).min(1.0),
            away_matches_14d: (away_14d as f32 / 3.0).min(1.0),
            away_matches_21d: (away_21d as f32 / 4.0).min(1.0),
            // Differentials: positive = home more fatigued
            workload_diff_7d: (home_7d as f32 - away_7d as f32) / 2.0,
            workload_diff_14d: (home_14d as f32 - away_14d as f32) / 3.0,
            workload_diff_21d: (home_21d as f32 - away_21d as f32) / 4.0,
        }
    }

    /// Record a match for both teams
    pub fn update(&mut self, home: TeamId, away: TeamId, date: NaiveDate) {
        self.match_history.entry(home).or_default().push(date);
        self.match_history.entry(away).or_default().push(date);

        // Keep only last 30 days of history per team
        for dates in self.match_history.values_mut() {
            dates.retain(|d| (date - *d).num_days() <= 30);
        }
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.match_history.clear();
    }
}

/// Workload features for a match
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkloadFeatures {
    pub home_matches_7d: f32,
    pub home_matches_14d: f32,
    pub home_matches_21d: f32,
    pub away_matches_7d: f32,
    pub away_matches_14d: f32,
    pub away_matches_21d: f32,
    pub workload_diff_7d: f32,
    pub workload_diff_14d: f32,
    pub workload_diff_21d: f32,
}

impl WorkloadFeatures {
    pub const DIM: usize = 9;

    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.home_matches_7d,
            self.home_matches_14d,
            self.home_matches_21d,
            self.away_matches_7d,
            self.away_matches_14d,
            self.away_matches_21d,
            self.workload_diff_7d,
            self.workload_diff_14d,
            self.workload_diff_21d,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_history() {
        let computer = WorkloadComputer::new();
        let features = computer.compute(
            TeamId(1),
            TeamId(2),
            NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
        );

        assert_eq!(features.home_matches_7d, 0.0);
        assert_eq!(features.away_matches_14d, 0.0);
    }

    #[test]
    fn test_single_match() {
        let mut computer = WorkloadComputer::new();
        computer.update(
            TeamId(1),
            TeamId(2),
            NaiveDate::from_ymd_opt(2024, 3, 10).unwrap(),
        );

        let features = computer.compute(
            TeamId(1),
            TeamId(3),
            NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
        );

        // Team 1 played 5 days ago - should be in 7d window
        assert!(features.home_matches_7d > 0.0);
        assert!(features.home_matches_14d > 0.0);
    }

    #[test]
    fn test_workload_differential() {
        let mut computer = WorkloadComputer::new();

        // Team 1 plays twice in a week
        computer.update(
            TeamId(1),
            TeamId(3),
            NaiveDate::from_ymd_opt(2024, 3, 8).unwrap(),
        );
        computer.update(
            TeamId(1),
            TeamId(4),
            NaiveDate::from_ymd_opt(2024, 3, 12).unwrap(),
        );

        let features = computer.compute(
            TeamId(1),
            TeamId(2),
            NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
        );

        // Home team (1) has more matches = positive diff
        assert!(features.workload_diff_7d > 0.0);
    }
}
