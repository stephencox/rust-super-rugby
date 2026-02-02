# Advanced Prediction Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add venue tracking, workload windows, and Elo ratings to improve prediction accuracy from 65% toward 70%+.

**Architecture:** Three new feature modules (elo, venue, workload) compute features from match history. Scrapers are updated to capture venue data. MatchComparison grows from 33 to 50 dimensions.

**Tech Stack:** Rust, Burn ML framework, SQLite, reqwest/scraper for web scraping.

---

## Task 1: Add venue field to RawMatch struct

**Files:**
- Modify: `src/data/scrapers/wikipedia.rs:1083-1092`

**Step 1: Add venue field to RawMatch**

In `src/data/scrapers/wikipedia.rs`, find the `RawMatch` struct and add the venue field:

```rust
/// Raw match data before team ID resolution
#[derive(Debug, Clone)]
pub struct RawMatch {
    pub date: NaiveDate,
    pub home_team: TeamInfo,
    pub away_team: TeamInfo,
    pub home_score: u8,
    pub away_score: u8,
    pub home_tries: Option<u8>,
    pub away_tries: Option<u8>,
    pub round: Option<u8>,
    pub venue: Option<String>,  // NEW
}
```

**Step 2: Verify compilation**

Run: `cargo build 2>&1 | grep -E "^error"`
Expected: Errors about missing `venue` field in struct initializations

**Step 3: Commit**

```bash
git add src/data/scrapers/wikipedia.rs
git commit -m "feat: add venue field to RawMatch struct

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Wikipedia scraper parsing to extract venues

**Files:**
- Modify: `src/data/scrapers/wikipedia.rs`

**Step 1: Update parse_sports_events to extract venue**

Find `parse_sports_events` method and add venue extraction. Add the location selector near the other selectors:

```rust
let location_selector = Selector::parse("span.location").unwrap();
```

Inside the event loop, after extracting teams, add:

```rust
// Extract venue
let venue = event
    .select(&location_selector)
    .next()
    .map(|loc| loc.text().collect::<String>().trim().to_string())
    .filter(|s| !s.is_empty());
```

Update the RawMatch creation to include venue:

```rust
matches.push(RawMatch {
    date,
    home_team: home,
    away_team: away,
    home_score: hs,
    away_score: as_,
    home_tries: None,
    away_tries: None,
    round: None,
    venue,
});
```

**Step 2: Update parse_collapsible_tables**

Add venue extraction (usually in the last cell after score):

```rust
// Try to extract venue from cells after away team
let venue = if i + 2 < cell_texts.len() {
    let venue_text = cell_texts[i + 2].trim();
    if !venue_text.is_empty() && self.normalize_team_name(venue_text).is_none() {
        Some(venue_text.to_string())
    } else {
        None
    }
} else {
    None
};
```

Update RawMatch creation to include `venue`.

**Step 3: Update parse_tables and try_parse_table_row**

Add venue to the RawMatch returned by `try_parse_table_row`:

```rust
return Some(RawMatch {
    date,
    home_team,
    away_team,
    home_score,
    away_score,
    home_tries: None,
    away_tries: None,
    round: None,
    venue: None,  // Table parsing doesn't reliably have venue
});
```

**Step 4: Update parse_text_content**

Add `venue: None` to RawMatch creation.

**Step 5: Update to_match_records to pass venue through**

Find `to_match_records` method and change:

```rust
records.push(MatchRecord {
    date: raw.date,
    home_team,
    away_team,
    home_score: raw.home_score,
    away_score: raw.away_score,
    venue: raw.venue,  // Changed from None
    round: raw.round,
    home_tries: raw.home_tries,
    away_tries: raw.away_tries,
    source: DataSource::Wikipedia,
});
```

**Step 6: Verify compilation**

Run: `cargo build`
Expected: Success with warnings only

**Step 7: Commit**

```bash
git add src/data/scrapers/wikipedia.rs
git commit -m "feat: extract venue from Wikipedia match parsing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Six Nations scraper with venue extraction and inference

**Files:**
- Modify: `src/data/scrapers/sixnations.rs`

**Step 1: Update parse_sports_events to extract venue**

Same pattern as Wikipedia scraper - add location_selector and extract venue.

**Step 2: Add venue inference function**

Add this function to `SixNationsScraper`:

```rust
/// Infer venue from home team (Six Nations has fixed home stadiums)
fn infer_venue(&self, home_team: &str) -> Option<String> {
    match home_team {
        "England" => Some("Twickenham".to_string()),
        "France" => Some("Stade de France".to_string()),
        "Ireland" => Some("Aviva Stadium".to_string()),
        "Wales" => Some("Principality Stadium".to_string()),
        "Scotland" => Some("Murrayfield".to_string()),
        "Italy" => Some("Stadio Olimpico".to_string()),
        _ => None,
    }
}
```

**Step 3: Update to_match_records to use inference**

```rust
pub fn to_match_records(
    &self,
    raw_matches: Vec<RawMatch>,
    team_resolver: &impl Fn(&str, Country) -> Result<TeamId>,
) -> Result<Vec<MatchRecord>> {
    let mut records = Vec::new();

    for raw in raw_matches {
        let home_team = team_resolver(&raw.home_team.name, raw.home_team.country)?;
        let away_team = team_resolver(&raw.away_team.name, raw.away_team.country)?;

        // Use scraped venue or infer from home team
        let venue = raw.venue.or_else(|| self.infer_venue(&raw.home_team.name));

        records.push(MatchRecord {
            date: raw.date,
            home_team,
            away_team,
            home_score: raw.home_score,
            away_score: raw.away_score,
            venue,
            round: raw.round,
            home_tries: raw.home_tries,
            away_tries: raw.away_tries,
            source: DataSource::Wikipedia,
        });
    }

    Ok(records)
}
```

**Step 4: Update all RawMatch creations to include venue field**

Add `venue: None` or extracted venue to all places creating RawMatch.

**Step 5: Verify compilation and run tests**

Run: `cargo test`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/data/scrapers/sixnations.rs
git commit -m "feat: add venue extraction and inference for Six Nations

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Create Elo rating module

**Files:**
- Create: `src/features/elo.rs`
- Modify: `src/features/mod.rs`

**Step 1: Write the Elo module**

Create `src/features/elo.rs`:

```rust
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
        *self.ratings.get(&team).unwrap_or(&self.config.initial_rating)
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
        let away_new = away_rating + self.config.k_factor * ((1.0 - home_actual) - (1.0 - home_expected));

        self.ratings.insert(record.home_team, home_new);
        self.ratings.insert(record.away_team, away_new);
    }

    /// Get normalized rating difference (for features)
    /// Normalized to roughly -1 to 1 range (±300 rating points)
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
```

**Step 2: Export from mod.rs**

Add to `src/features/mod.rs`:

```rust
pub mod elo;
pub use elo::{EloConfig, EloFeatures, EloRatings};
```

**Step 3: Run tests**

Run: `cargo test elo`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/elo.rs src/features/mod.rs
git commit -m "feat: add Elo rating system for team strength

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create workload windows module

**Files:**
- Create: `src/features/workload.rs`
- Modify: `src/features/mod.rs`

**Step 1: Write the workload module**

Create `src/features/workload.rs`:

```rust
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
        computer.update(TeamId(1), TeamId(3), NaiveDate::from_ymd_opt(2024, 3, 8).unwrap());
        computer.update(TeamId(1), TeamId(4), NaiveDate::from_ymd_opt(2024, 3, 12).unwrap());

        let features = computer.compute(
            TeamId(1),
            TeamId(2),
            NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
        );

        // Home team (1) has more matches = positive diff
        assert!(features.workload_diff_7d > 0.0);
    }
}
```

**Step 2: Export from mod.rs**

Add to `src/features/mod.rs`:

```rust
pub mod workload;
pub use workload::{WorkloadComputer, WorkloadFeatures};
```

**Step 3: Run tests**

Run: `cargo test workload`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/workload.rs src/features/mod.rs
git commit -m "feat: add workload window features (7/14/21 days)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Create venue performance module

**Files:**
- Create: `src/features/venue.rs`
- Modify: `src/features/mod.rs`

**Step 1: Write the venue module**

Create `src/features/venue.rs`:

```rust
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
            0.5  // Unknown = neutral
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

        assert_eq!(features.home_venue_win_rate, 1.0);  // 2/2 wins
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
```

**Step 2: Export from mod.rs**

Add to `src/features/mod.rs`:

```rust
pub mod venue;
pub use venue::{VenueFeatures, VenueTracker};
```

**Step 3: Run tests**

Run: `cargo test venue`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/venue.rs src/features/mod.rs
git commit -m "feat: add venue performance tracking features

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update MatchComparison to include new features

**Files:**
- Modify: `src/data/dataset.rs`

**Step 1: Add new feature fields to MatchComparison**

Add these fields after the temporal features section:

```rust
// Elo features (3)
/// Home team Elo rating (normalized)
pub home_elo: f32,
/// Away team Elo rating (normalized)
pub away_elo: f32,
/// Elo rating difference (normalized)
pub elo_diff: f32,

// Workload features (9)
/// Home team matches in last 7 days
pub home_matches_7d: f32,
/// Home team matches in last 14 days
pub home_matches_14d: f32,
/// Home team matches in last 21 days
pub home_matches_21d: f32,
/// Away team matches in last 7 days
pub away_matches_7d: f32,
/// Away team matches in last 14 days
pub away_matches_14d: f32,
/// Away team matches in last 21 days
pub away_matches_21d: f32,
/// Workload differential 7d
pub workload_diff_7d: f32,
/// Workload differential 14d
pub workload_diff_14d: f32,
/// Workload differential 21d
pub workload_diff_21d: f32,

// Venue features (5)
/// Home team win rate at this venue
pub home_venue_win_rate: f32,
/// Home team games at this venue (normalized)
pub home_venue_games: f32,
/// Away team win rate at this venue
pub away_venue_win_rate: f32,
/// Away team games at this venue (normalized)
pub away_venue_games: f32,
/// Venue familiarity difference
pub venue_familiarity_diff: f32,
```

**Step 2: Update DIM constant**

Change:
```rust
pub const DIM: usize = 33; // 15 original + 18 temporal
```

To:
```rust
pub const DIM: usize = 50; // 15 original + 18 temporal + 3 elo + 9 workload + 5 venue
```

**Step 3: Update from_summaries to initialize new fields**

Add to the struct initialization in `from_summaries`:

```rust
// Elo (default, set via with_elo)
home_elo: 0.0,
away_elo: 0.0,
elo_diff: 0.0,
// Workload (default, set via with_workload)
home_matches_7d: 0.0,
home_matches_14d: 0.0,
home_matches_21d: 0.0,
away_matches_7d: 0.0,
away_matches_14d: 0.0,
away_matches_21d: 0.0,
workload_diff_7d: 0.0,
workload_diff_14d: 0.0,
workload_diff_21d: 0.0,
// Venue (default, set via with_venue)
home_venue_win_rate: 0.5,
home_venue_games: 0.0,
away_venue_win_rate: 0.5,
away_venue_games: 0.0,
venue_familiarity_diff: 0.0,
```

**Step 4: Add builder methods**

Add these methods to `impl MatchComparison`:

```rust
/// Set Elo features
pub fn with_elo(mut self, elo: &crate::features::EloFeatures) -> Self {
    self.home_elo = elo.home_elo;
    self.away_elo = elo.away_elo;
    self.elo_diff = elo.elo_diff;
    self
}

/// Set workload features
pub fn with_workload(mut self, workload: &crate::features::WorkloadFeatures) -> Self {
    self.home_matches_7d = workload.home_matches_7d;
    self.home_matches_14d = workload.home_matches_14d;
    self.home_matches_21d = workload.home_matches_21d;
    self.away_matches_7d = workload.away_matches_7d;
    self.away_matches_14d = workload.away_matches_14d;
    self.away_matches_21d = workload.away_matches_21d;
    self.workload_diff_7d = workload.workload_diff_7d;
    self.workload_diff_14d = workload.workload_diff_14d;
    self.workload_diff_21d = workload.workload_diff_21d;
    self
}

/// Set venue features
pub fn with_venue(mut self, venue: &crate::features::VenueFeatures) -> Self {
    self.home_venue_win_rate = venue.home_venue_win_rate;
    self.home_venue_games = venue.home_venue_games;
    self.away_venue_win_rate = venue.away_venue_win_rate;
    self.away_venue_games = venue.away_venue_games;
    self.venue_familiarity_diff = venue.venue_familiarity_diff;
    self
}
```

**Step 5: Update to_vec method**

Add the new features to `to_vec()`:

```rust
pub fn to_vec(&self) -> Vec<f32> {
    vec![
        // ... existing fields ...
        // Elo (3)
        self.home_elo,
        self.away_elo,
        self.elo_diff,
        // Workload (9)
        self.home_matches_7d,
        self.home_matches_14d,
        self.home_matches_21d,
        self.away_matches_7d,
        self.away_matches_14d,
        self.away_matches_21d,
        self.workload_diff_7d,
        self.workload_diff_14d,
        self.workload_diff_21d,
        // Venue (5)
        self.home_venue_win_rate,
        self.home_venue_games,
        self.away_venue_win_rate,
        self.away_venue_games,
        self.venue_familiarity_diff,
    ]
}
```

**Step 6: Verify compilation**

Run: `cargo build`
Expected: Success (tests may fail due to dimension changes)

**Step 7: Commit**

```bash
git add src/data/dataset.rs
git commit -m "feat: expand MatchComparison to 50 dimensions

Add Elo (3), workload (9), and venue (5) feature fields.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integrate new features into dataset creation

**Files:**
- Modify: `src/data/dataset.rs`

**Step 1: Add feature computers to from_matches_with_norm**

At the top of the function, after the temporal computer, add:

```rust
use crate::features::{EloRatings, WorkloadComputer, VenueTracker};

// ... existing code ...

// Initialize feature computers
let mut temporal_computer = TemporalFeatureComputer::new();
let mut elo_computer = EloRatings::default();
let mut workload_computer = WorkloadComputer::new();
let mut venue_tracker = VenueTracker::new();
```

**Step 2: Update feature computation in the main loop**

After computing temporal context, add:

```rust
// Compute Elo features BEFORE updating
let elo_features = crate::features::EloFeatures {
    home_elo: elo_computer.rating_normalized(m.home_team),
    away_elo: elo_computer.rating_normalized(m.away_team),
    elo_diff: elo_computer.rating_diff_normalized(m.home_team, m.away_team),
};

// Compute workload features BEFORE updating
let workload_features = workload_computer.compute(m.home_team, m.away_team, m.date);

// Compute venue features BEFORE updating
let venue_features = venue_tracker.compute(
    m.home_team,
    m.away_team,
    m.venue.as_deref(),
);
```

**Step 3: Update comparison creation**

Change:

```rust
let comparison = MatchComparison::from_summaries(&home_summary, &away_summary, is_local)
    .with_temporal(&temporal_ctx);
```

To:

```rust
let comparison = MatchComparison::from_summaries(&home_summary, &away_summary, is_local)
    .with_temporal(&temporal_ctx)
    .with_elo(&elo_features)
    .with_workload(&workload_features)
    .with_venue(&venue_features);
```

**Step 4: Update feature computers after processing**

After updating temporal computer, add:

```rust
// Update all feature computers
temporal_computer.update(m);
elo_computer.update(m);
workload_computer.update(m.home_team, m.away_team, m.date);
venue_tracker.update(
    m.home_team,
    m.away_team,
    m.venue.as_deref(),
    m.home_score > m.away_score,
);
```

**Step 5: Run tests**

Run: `cargo test`
Expected: May have failures due to dimension changes in other files

**Step 6: Commit**

```bash
git add src/data/dataset.rs
git commit -m "feat: integrate Elo, workload, venue into dataset creation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Fix dimension references across codebase

**Files:**
- Modify: `src/main.rs`
- Modify: `src/bin/sixnations.rs`
- Modify: `src/model/rugby_net.rs`
- Modify: any other files with hardcoded dimensions

**Step 1: Search for hardcoded 33**

Run: `grep -rn "33" src/ --include="*.rs" | grep -v test`

Fix any references to the old dimension (33) to use `MatchComparison::DIM`.

**Step 2: Search for reshape with dimensions**

Run: `grep -rn "reshape.*\[1," src/ --include="*.rs"`

Ensure all reshapes use dynamic `MatchComparison::DIM`.

**Step 3: Run tests**

Run: `cargo test`
Expected: All tests pass

**Step 4: Run build**

Run: `cargo build --release`
Expected: Success

**Step 5: Commit**

```bash
git add -A
git commit -m "fix: update all dimension references to use MatchComparison::DIM

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Add integration test

**Files:**
- Modify: `src/data/dataset.rs` (test section)

**Step 1: Add integration test**

Add to the tests module in dataset.rs:

```rust
#[test]
fn test_match_comparison_dimension() {
    let comparison = MatchComparison::default();
    let vec = comparison.to_vec();
    assert_eq!(vec.len(), MatchComparison::DIM);
    assert_eq!(MatchComparison::DIM, 50);
}
```

**Step 2: Run tests**

Run: `cargo test match_comparison_dimension`
Expected: Pass

**Step 3: Commit**

```bash
git add src/data/dataset.rs
git commit -m "test: add dimension verification test

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Final verification

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All tests pass

**Step 2: Test training**

Run: `cargo run --release -- train --epochs 100`
Expected: Training runs without dimension errors

**Step 3: Test prediction**

Run: `cargo run --release -- predict-next`
Expected: Predictions work with new features

**Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final verification and fixes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

Plan complete and saved to `docs/plans/2026-02-02-advanced-features.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
