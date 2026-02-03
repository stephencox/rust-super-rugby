//! Burn Dataset implementation for rugby match data
//!
//! Provides training samples with team match histories.

use crate::data::Database;
use crate::features::MatchFeatures;
use crate::{MatchRecord, Result, TeamId};
use burn::data::dataset::Dataset;
use chrono::NaiveDate;
use std::collections::HashMap;

/// Score normalization parameters (computed from training data)
#[derive(Debug, Clone, Copy)]
pub struct ScoreNormalization {
    pub mean: f32,
    pub std: f32,
}

impl Default for ScoreNormalization {
    fn default() -> Self {
        // Default values based on typical Super Rugby scores
        ScoreNormalization {
            mean: 25.0,
            std: 12.0,
        }
    }
}

/// Feature normalization parameters for z-score normalization of inputs
#[derive(Debug, Clone, Copy)]
pub struct FeatureNormalization {
    /// Score mean and std (for score_for, score_against)
    pub score_mean: f32,
    pub score_std: f32,
    /// Margin mean and std
    pub margin_mean: f32,
    pub margin_std: f32,
    /// Tries mean and std
    pub tries_mean: f32,
    pub tries_std: f32,
}

impl Default for FeatureNormalization {
    fn default() -> Self {
        // Default values based on typical Super Rugby stats
        FeatureNormalization {
            score_mean: 25.0,
            score_std: 12.0,
            margin_mean: 0.0,
            margin_std: 15.0,
            tries_mean: 3.0,
            tries_std: 1.5,
        }
    }
}

impl FeatureNormalization {
    /// Create feature normalization from score normalization (for random splits)
    pub fn from_score_norm(score_norm: ScoreNormalization) -> Self {
        FeatureNormalization {
            score_mean: score_norm.mean,
            score_std: score_norm.std,
            margin_mean: 0.0,
            margin_std: score_norm.std, // Use score std as proxy for margin std
            tries_mean: 3.0,
            tries_std: 1.5,
        }
    }

    /// Compute feature normalization from a set of matches
    pub fn from_matches(matches: &[MatchRecord]) -> Self {
        if matches.is_empty() {
            return Self::default();
        }

        // Collect all scores
        let scores: Vec<f32> = matches
            .iter()
            .flat_map(|m| [m.home_score as f32, m.away_score as f32])
            .collect();

        // Collect all margins
        let margins: Vec<f32> = matches
            .iter()
            .map(|m| (m.home_score as i16 - m.away_score as i16) as f32)
            .collect();

        // Collect all tries (only where available)
        let tries: Vec<f32> = matches
            .iter()
            .flat_map(|m| {
                let mut t = Vec::new();
                if let Some(ht) = m.home_tries {
                    t.push(ht as f32);
                }
                if let Some(at) = m.away_tries {
                    t.push(at as f32);
                }
                t
            })
            .collect();

        FeatureNormalization {
            score_mean: mean(&scores),
            score_std: std(&scores).max(1.0),
            margin_mean: mean(&margins),
            margin_std: std(&margins).max(1.0),
            tries_mean: if tries.is_empty() { 3.0 } else { mean(&tries) },
            tries_std: if tries.is_empty() { 1.5 } else { std(&tries).max(0.5) },
        }
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn std(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 1.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

impl ScoreNormalization {
    /// Compute normalization params from a set of matches
    pub fn from_matches(matches: &[MatchRecord]) -> Self {
        if matches.is_empty() {
            return Self::default();
        }

        let scores: Vec<f32> = matches
            .iter()
            .flat_map(|m| [m.home_score as f32, m.away_score as f32])
            .collect();

        let n = scores.len() as f32;
        let mean = scores.iter().sum::<f32>() / n;
        let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt().max(1.0); // Avoid division by zero

        ScoreNormalization { mean, std }
    }

    /// Normalize a score
    pub fn normalize(&self, score: f32) -> f32 {
        (score - self.mean) / self.std
    }

    /// Denormalize a score
    pub fn denormalize(&self, normalized: f32) -> f32 {
        normalized * self.std + self.mean
    }
}

/// Team-level summary statistics computed from recent history
#[derive(Debug, Clone, Copy, Default)]
pub struct TeamSummary {
    /// Win rate over last N games (0-1)
    pub win_rate: f32,
    /// Average margin over last N games
    pub margin_avg: f32,
    /// Pythagorean expectation: PF^2.37 / (PF^2.37 + PA^2.37)
    pub pythagorean: f32,
    /// Points for average
    pub pf_avg: f32,
    /// Points against average
    pub pa_avg: f32,
}

impl TeamSummary {
    pub const DIM: usize = 5;

    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.win_rate, self.margin_avg, self.pythagorean, self.pf_avg, self.pa_avg]
    }
}

/// Match-level comparative features (differences between teams + raw stats + temporal)
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchComparison {
    // Differential features (5)
    /// Win rate difference (home - away)
    pub win_rate_diff: f32,
    /// Margin average difference (home - away), normalized
    pub margin_diff: f32,
    /// Pythagorean difference (home - away)
    pub pythagorean_diff: f32,
    /// Log5 probability: p_home / (p_home + p_away - p_home*p_away)
    pub log5: f32,
    /// Is local derby (same country)
    pub is_local: f32,

    // Home team raw stats (5)
    pub home_win_rate: f32,
    pub home_margin_avg: f32,
    pub home_pythagorean: f32,
    pub home_pf_avg: f32,
    pub home_pa_avg: f32,

    // Away team raw stats (5)
    pub away_win_rate: f32,
    pub away_margin_avg: f32,
    pub away_pythagorean: f32,
    pub away_pf_avg: f32,
    pub away_pa_avg: f32,

    // Temporal features (18)
    /// 1.0 if Saturday match
    pub is_saturday: f32,
    /// 1.0 if Friday match
    pub is_friday: f32,
    /// 1.0 if Sunday match
    pub is_sunday: f32,
    /// Season progress (0.0 = Feb, 1.0 = Jul)
    pub season_progress: f32,
    /// 1.0 if early season (Feb-Mar)
    pub is_early_season: f32,
    /// 1.0 if late season (May-Jul)
    pub is_late_season: f32,
    /// Round number normalized (0-1)
    pub round_normalized: f32,
    /// Days rest for home team (normalized 0-1)
    pub home_rest_days: f32,
    /// Days rest for away team (normalized 0-1)
    pub away_rest_days: f32,
    /// Rest advantage: (home_rest - away_rest) normalized (-1 to 1)
    pub rest_advantage: f32,
    /// 1.0 if home team has short turnaround (<7 days)
    pub home_short_turnaround: f32,
    /// 1.0 if away team has short turnaround (<7 days)
    pub away_short_turnaround: f32,
    /// Consecutive home games for home team (normalized)
    pub home_streak: f32,
    /// Consecutive away games for away team (normalized)
    pub away_streak: f32,
    /// Days since H2H (normalized 0-1)
    pub h2h_recency: f32,
    /// Games since H2H (normalized 0-1)
    pub games_since_h2h: f32,
    /// Match density for home team (normalized 0-1)
    pub home_match_density: f32,
    /// Match density for away team (normalized 0-1)
    pub away_match_density: f32,

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
}

impl MatchComparison {
    pub const DIM: usize = 50; // 15 original + 18 temporal + 3 elo + 9 workload + 5 venue

    /// Compute Log5 probability from two win rates
    pub fn log5_prob(p_a: f32, p_b: f32) -> f32 {
        let denom = p_a + p_b - 2.0 * p_a * p_b;
        if denom.abs() < 0.001 {
            0.5
        } else {
            (p_a - p_a * p_b) / denom
        }
    }

    pub fn from_summaries(home: &TeamSummary, away: &TeamSummary, is_local: bool) -> Self {
        // Store raw values - z-score normalization is applied at training time
        // by ComparisonNormalization
        // Temporal features will be set separately via with_temporal
        MatchComparison {
            // Differentials
            win_rate_diff: home.win_rate - away.win_rate,
            margin_diff: home.margin_avg - away.margin_avg,
            pythagorean_diff: home.pythagorean - away.pythagorean,
            log5: Self::log5_prob(home.win_rate.max(0.01).min(0.99), away.win_rate.max(0.01).min(0.99)),
            is_local: if is_local { 1.0 } else { 0.0 },
            // Home team raw stats
            home_win_rate: home.win_rate,
            home_margin_avg: home.margin_avg,
            home_pythagorean: home.pythagorean,
            home_pf_avg: home.pf_avg,
            home_pa_avg: home.pa_avg,
            // Away team raw stats
            away_win_rate: away.win_rate,
            away_margin_avg: away.margin_avg,
            away_pythagorean: away.pythagorean,
            away_pf_avg: away.pf_avg,
            away_pa_avg: away.pa_avg,
            // Temporal features (default to 0, set via with_temporal)
            is_saturday: 0.0,
            is_friday: 0.0,
            is_sunday: 0.0,
            season_progress: 0.0,
            is_early_season: 0.0,
            is_late_season: 0.0,
            round_normalized: 0.0,
            home_rest_days: 0.0,
            away_rest_days: 0.0,
            rest_advantage: 0.0,
            home_short_turnaround: 0.0,
            away_short_turnaround: 0.0,
            home_streak: 0.0,
            away_streak: 0.0,
            h2h_recency: 0.0,
            games_since_h2h: 0.0,
            home_match_density: 0.0,
            away_match_density: 0.0,
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
        }
    }

    /// Set temporal features from TemporalContext
    pub fn with_temporal(mut self, ctx: &crate::features::TemporalContext) -> Self {
        self.is_saturday = ctx.is_saturday;
        self.is_friday = ctx.is_friday;
        self.is_sunday = ctx.is_sunday;
        self.season_progress = ctx.season_progress;
        self.is_early_season = ctx.is_early_season;
        self.is_late_season = ctx.is_late_season;
        self.round_normalized = ctx.round_normalized;
        self.home_rest_days = ctx.home_rest_days;
        self.away_rest_days = ctx.away_rest_days;
        self.rest_advantage = ctx.rest_advantage;
        self.home_short_turnaround = ctx.home_short_turnaround;
        self.away_short_turnaround = ctx.away_short_turnaround;
        self.home_streak = ctx.home_streak;
        self.away_streak = ctx.away_streak;
        self.h2h_recency = ctx.h2h_recency;
        self.games_since_h2h = ctx.games_since_h2h;
        self.home_match_density = ctx.home_match_density;
        self.away_match_density = ctx.away_match_density;
        self
    }

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

    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            // Differentials (5)
            self.win_rate_diff,
            self.margin_diff,
            self.pythagorean_diff,
            self.log5,
            self.is_local,
            // Home team stats (5)
            self.home_win_rate,
            self.home_margin_avg,
            self.home_pythagorean,
            self.home_pf_avg,
            self.home_pa_avg,
            // Away team stats (5)
            self.away_win_rate,
            self.away_margin_avg,
            self.away_pythagorean,
            self.away_pf_avg,
            self.away_pa_avg,
            // Temporal features (18)
            self.is_saturday,
            self.is_friday,
            self.is_sunday,
            self.season_progress,
            self.is_early_season,
            self.is_late_season,
            self.round_normalized,
            self.home_rest_days,
            self.away_rest_days,
            self.rest_advantage,
            self.home_short_turnaround,
            self.away_short_turnaround,
            self.home_streak,
            self.away_streak,
            self.h2h_recency,
            self.games_since_h2h,
            self.home_match_density,
            self.away_match_density,
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
}

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
    /// Home team ID (for embeddings)
    pub home_team_id: u32,
    /// Away team ID (for embeddings)
    pub away_team_id: u32,
    /// Home team summary stats (computed from history)
    pub home_summary: TeamSummary,
    /// Away team summary stats
    pub away_summary: TeamSummary,
    /// Match comparison features (differences, log5)
    pub comparison: MatchComparison,
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
#[derive(Clone)]
pub struct RugbyDataset {
    samples: Vec<MatchSample>,
    config: DatasetConfig,
    /// Score normalization parameters (for targets)
    pub score_norm: ScoreNormalization,
    /// Feature normalization parameters (for inputs)
    pub feature_norm: FeatureNormalization,
    /// Mapping from raw Team ID (DB) to dense index (0..N) for embeddings
    pub team_mapping: HashMap<i64, u32>,
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

    /// Create dataset from matches in a date range with explicit normalization
    pub fn from_matches_in_range_with_norm(
        db: &Database,
        start: NaiveDate,
        end: NaiveDate,
        config: DatasetConfig,
        score_norm: ScoreNormalization,
        feature_norm: FeatureNormalization,
    ) -> Result<Self> {
        let matches = db.get_matches_in_range(start, end)?;
        Self::from_matches_with_norm(db, matches, config, Some(score_norm), Some(feature_norm), None)
    }

    /// Create dataset from a list of matches
    pub fn from_matches(
        db: &Database,
        matches: Vec<MatchRecord>,
        config: DatasetConfig,
    ) -> Result<Self> {
        Self::from_matches_with_norm(db, matches, config, None, None, None)
    }

    /// Create dataset with explicit normalization params and optional mapping
    pub fn from_matches_with_norm(
        db: &Database,
        matches: Vec<MatchRecord>,
        config: DatasetConfig,
        score_norm: Option<ScoreNormalization>,
        feature_norm: Option<FeatureNormalization>,
        existing_mapping: Option<&HashMap<i64, u32>>,
    ) -> Result<Self> {
        use crate::features::TemporalFeatureComputer;

        // Compute normalization from all matches if not provided
        let all_matches = db.get_all_matches()?;
        let score_norm = score_norm.unwrap_or_else(|| ScoreNormalization::from_matches(&all_matches));
        let feature_norm = feature_norm.unwrap_or_else(|| FeatureNormalization::from_matches(&all_matches));

        // Use existing mapping or create dense team mapping from ALL matches
        let team_mapping = if let Some(mapping) = existing_mapping {
            mapping.clone()
        } else {
            let mut team_ids: Vec<i64> = all_matches
                .iter()
                .flat_map(|m| [m.home_team.0, m.away_team.0])
                .collect();
            team_ids.sort();
            team_ids.dedup();
            
            team_ids
                .into_iter()
                .enumerate()
                .map(|(idx, id)| (id, idx as u32))
                .collect()
        };

        let mut samples = Vec::new();

        // Build timezone lookup map
        let teams = db.get_all_teams()?;
        let team_timezones: HashMap<TeamId, i32> = teams
            .iter()
            .map(|t| (t.id, t.timezone_offset))
            .collect();

        // Initialize temporal feature computer and process all historical matches
        // to build up state before processing target matches
        let mut temporal_computer = TemporalFeatureComputer::new();

        // Process all matches in chronological order to build temporal state
        let mut sorted_all = all_matches.clone();
        sorted_all.sort_by_key(|m| m.date);
        for m in &sorted_all {
            temporal_computer.update(m);
        }

        // Reset and reprocess for actual feature extraction
        // This ensures we have proper state at each point in time
        temporal_computer.reset();

        // Initialize other feature computers
        let mut elo_computer = crate::features::EloRatings::default();
        let mut workload_computer = crate::features::WorkloadComputer::new();
        let mut venue_tracker = crate::features::VenueTracker::new();

        // Sort all matches chronologically for proper temporal feature extraction
        let mut all_sorted: Vec<_> = all_matches.iter().cloned().collect();
        all_sorted.sort_by_key(|m| m.date);

        // Build a set of target match dates for efficient lookup
        let target_set: std::collections::HashSet<_> = matches
            .iter()
            .map(|m| (m.date, m.home_team, m.away_team))
            .collect();

        // Process matches chronologically, creating samples for target matches
        for m in &all_sorted {
            // Compute temporal context BEFORE updating (so it reflects state before this match)
            let temporal_ctx = temporal_computer.compute(m);

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

            // Check if this is a target match we want to include in the dataset
            if target_set.contains(&(m.date, m.home_team, m.away_team)) {
                // Get historical matches for both teams before this match
                let home_history = Self::get_team_history(
                    &all_matches,
                    m.home_team,
                    m.date,
                    config.max_history,
                );

                let away_history = Self::get_team_history(
                    &all_matches,
                    m.away_team,
                    m.date,
                    config.max_history,
                );

                // Skip if insufficient history
                if home_history.len() >= config.min_history && away_history.len() >= config.min_history {
                    // Convert to features with padding, travel info, and z-score normalization
                    let (home_features, home_mask) = Self::to_features_with_padding(
                        &home_history,
                        m.home_team,
                        config.max_history,
                        &team_timezones,
                        &feature_norm,
                    );
                    let (away_features, away_mask) = Self::to_features_with_padding(
                        &away_history,
                        m.away_team,
                        config.max_history,
                        &team_timezones,
                        &feature_norm,
                    );

                    // Compute team summaries from history (last 10 games)
                    let home_summary = Self::compute_team_summary(&home_history, m.home_team);
                    let away_summary = Self::compute_team_summary(&away_history, m.away_team);

                    // Check if same country (local derby)
                    let home_country = teams.iter().find(|t| t.id == m.home_team).map(|t| &t.country);
                    let away_country = teams.iter().find(|t| t.id == m.away_team).map(|t| &t.country);
                    let is_local = home_country == away_country;

                    // Compute comparison features with all feature contexts
                    let comparison = MatchComparison::from_summaries(&home_summary, &away_summary, is_local)
                        .with_temporal(&temporal_ctx)
                        .with_elo(&elo_features)
                        .with_workload(&workload_features)
                        .with_venue(&venue_features);

                    // Map team IDs to dense indices
                    // If team not in mapping (e.g. new team in validation set), map to 0
                    let home_idx = *team_mapping.get(&m.home_team.0).unwrap_or(&0);
                    let away_idx = *team_mapping.get(&m.away_team.0).unwrap_or(&0);

                    samples.push(MatchSample {
                        home_history: home_features,
                        away_history: away_features,
                        home_mask,
                        away_mask,
                        home_team_id: home_idx,
                        away_team_id: away_idx,
                        home_summary,
                        away_summary,
                        comparison,
                        home_win: if m.home_score > m.away_score {
                            1.0
                        } else {
                            0.0
                        },
                        // Z-score normalize scores for training
                        home_score: score_norm.normalize(m.home_score as f32),
                        away_score: score_norm.normalize(m.away_score as f32),
                    });
                }
            }

            // Update all feature computers AFTER processing this match
            temporal_computer.update(m);
            elo_computer.update(m);
            workload_computer.update(m.home_team, m.away_team, m.date);
            venue_tracker.update(
                m.home_team,
                m.away_team,
                m.venue.as_deref(),
                m.home_score > m.away_score,
            );
        }

        log::info!(
            "Created dataset with {} samples (score norm: mean={:.1}, std={:.1}, feature norm: score={:.1}/{:.1})",
            samples.len(),
            score_norm.mean,
            score_norm.std,
            feature_norm.score_mean,
            feature_norm.score_std
        );
        Ok(RugbyDataset {
            samples,
            config,
            score_norm,
            feature_norm,
            team_mapping,
        })
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

    /// Compute team summary statistics from match history (last 10 games)
    fn compute_team_summary(history: &[MatchRecord], team: TeamId) -> TeamSummary {
        // Use last 10 games for summary (matching best-performing window from analysis)
        let window = 10;
        let recent: Vec<_> = history.iter().rev().take(window).collect();

        if recent.len() < 3 {
            return TeamSummary::default();
        }

        let mut wins = 0.0f32;
        let mut total_pf = 0.0f32;
        let mut total_pa = 0.0f32;
        let mut margins = Vec::new();

        for m in &recent {
            let is_home = m.home_team == team;
            let (pf, pa) = if is_home {
                (m.home_score as f32, m.away_score as f32)
            } else {
                (m.away_score as f32, m.home_score as f32)
            };

            let margin = pf - pa;
            margins.push(margin);
            total_pf += pf;
            total_pa += pa;

            if margin > 0.0 {
                wins += 1.0;
            } else if margin == 0.0 {
                wins += 0.5;
            }
        }

        let n = recent.len() as f32;
        let win_rate = wins / n;
        let margin_avg = margins.iter().sum::<f32>() / n;
        let pf_avg = total_pf / n;
        let pa_avg = total_pa / n;

        // Pythagorean expectation
        let exp = 2.37f32;
        let pf_pow = total_pf.powf(exp);
        let pa_pow = total_pa.powf(exp);
        let pythagorean = if pf_pow + pa_pow > 0.0 {
            pf_pow / (pf_pow + pa_pow)
        } else {
            0.5
        };

        TeamSummary {
            win_rate,
            margin_avg,
            pythagorean,
            pf_avg,
            pa_avg,
        }
    }

    /// Convert matches to features with padding, travel context, rolling stats, temporal features, and z-score normalization
    fn to_features_with_padding(
        matches: &[MatchRecord],
        perspective_team: TeamId,
        max_len: usize,
        team_timezones: &HashMap<TeamId, i32>,
        feature_norm: &FeatureNormalization,
    ) -> (Vec<MatchFeatures>, Vec<bool>) {
        use chrono::Datelike;

        let mut features = Vec::with_capacity(max_len);
        let mut mask = Vec::with_capacity(max_len);

        // Default timezone (for unknown teams)
        let default_tz = 10;

        // Add actual match features with travel context and z-score normalization
        let mut prev_match: Option<&MatchRecord> = None;

        // Rolling stats accumulators
        let mut wins: f32 = 0.0;
        let mut games: f32 = 0.0;
        let mut total_pf: f32 = 0.0;
        let mut total_pa: f32 = 0.0;
        let mut margins: Vec<f32> = Vec::new();

        // Temporal tracking
        let mut last_was_home: Option<bool> = None;
        let mut streak_count: i32 = 0;

        for m in matches.iter() {
            // Use z-score normalized features
            let mut feat = MatchFeatures::from_match_normalized(m, perspective_team, feature_norm);

            let is_home = m.home_team == perspective_team;
            let mut days_since_prev: Option<i64> = None;

            // Compute travel from previous match
            if let Some(prev) = prev_match {
                // Previous match venue = home team's timezone
                let prev_tz = *team_timezones.get(&prev.home_team).unwrap_or(&default_tz);
                // This match venue = home team's timezone
                let this_tz = *team_timezones.get(&m.home_team).unwrap_or(&default_tz);
                // Days since previous match
                days_since_prev = Some((m.date - prev.date).num_days());
                feat = feat.with_travel(prev_tz, this_tz, days_since_prev.unwrap());
            }

            // Apply rolling stats from PREVIOUS matches (not including this one)
            if games >= 3.0 {
                let margin_avg = if !margins.is_empty() {
                    let m = margins.iter().sum::<f32>() / margins.len() as f32;
                    // Z-score normalize the margin average
                    (m - feature_norm.margin_mean) / feature_norm.margin_std
                } else {
                    0.0
                };

                let margin_std = if margins.len() >= 2 {
                    let m = margins.iter().sum::<f32>() / margins.len() as f32;
                    let variance = margins.iter().map(|x| (x - m).powi(2)).sum::<f32>()
                        / margins.len() as f32;
                    variance.sqrt()
                } else {
                    15.0 // Default std
                };

                feat = feat.with_rolling_stats(
                    wins / games,
                    margin_avg,
                    total_pf,
                    total_pa,
                    margin_std,
                );
            }

            // === Temporal features ===
            // Day of week
            let weekday = m.date.weekday().num_days_from_monday();
            let is_saturday = if weekday == 5 { 1.0 } else { 0.0 };
            let is_friday = if weekday == 4 { 1.0 } else { 0.0 };

            // Season progression (Super Rugby: Feb-Jul)
            let month = m.date.month();
            let season_progress = ((month as f32 - 2.0) / 5.0).clamp(0.0, 1.0);
            let is_early_season = if month <= 3 { 1.0 } else { 0.0 };
            let is_late_season = if month >= 5 { 1.0 } else { 0.0 };

            // Round normalized
            let round_normalized = m.round.map(|r| (r as f32 / 20.0).clamp(0.0, 1.0)).unwrap_or(0.5);

            // Short turnaround
            let short_turnaround = if days_since_prev.map_or(false, |d| d < 7) {
                1.0
            } else {
                0.0
            };

            // Streak count (consecutive home or away games)
            let current_streak = if last_was_home == Some(is_home) {
                streak_count as f32
            } else {
                0.0
            };
            let streak_normalized = (current_streak / 4.0).clamp(0.0, 1.0);

            feat = feat.with_temporal(
                is_saturday,
                is_friday,
                season_progress,
                is_early_season,
                is_late_season,
                round_normalized,
                short_turnaround,
                streak_normalized,
            );

            features.push(feat);
            mask.push(true);
            prev_match = Some(m);

            // Update streak tracking
            if last_was_home == Some(is_home) {
                streak_count += 1;
            } else {
                streak_count = 1;
            }
            last_was_home = Some(is_home);

            // Update rolling stats accumulators AFTER using them for this match
            // (stats for match i should be based on matches 0..i, not 0..=i)
            let (pf, pa) = if is_home {
                (m.home_score as f32, m.away_score as f32)
            } else {
                (m.away_score as f32, m.home_score as f32)
            };

            let margin = pf - pa;
            margins.push(margin);
            total_pf += pf;
            total_pa += pa;
            games += 1.0;
            if margin > 0.0 {
                wins += 1.0;
            } else if margin == 0.0 {
                wins += 0.5;
            }

            // Keep only last 10 margins for rolling window
            if margins.len() > 10 {
                margins.remove(0);
            }
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

    /// Create dataset directly from pre-computed samples (for random splits)
    pub fn from_samples(
        samples: Vec<MatchSample>,
        score_norm: ScoreNormalization,
        feature_norm: FeatureNormalization,
        team_mapping: HashMap<i64, u32>,
    ) -> Self {
        RugbyDataset {
            samples,
            config: DatasetConfig::default(),
            score_norm,
            feature_norm,
            team_mapping,
        }
    }

    /// Split dataset into train/validation
    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let split_idx = (self.samples.len() as f32 * train_ratio) as usize;
        let (train_samples, val_samples) = self.samples.split_at(split_idx);

        (
            RugbyDataset {
                samples: train_samples.to_vec(),
                config: self.config.clone(),
                score_norm: self.score_norm,
                feature_norm: self.feature_norm,
                team_mapping: self.team_mapping.clone(),
            },
            RugbyDataset {
                samples: val_samples.to_vec(),
                config: self.config,
                score_norm: self.score_norm,
                feature_norm: self.feature_norm,
                team_mapping: self.team_mapping,
            },
        )
    }

    /// Get feature normalization stats
    pub fn feature_norm(&self) -> &FeatureNormalization {
        &self.feature_norm
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
    /// Home team IDs for embeddings: [batch]
    pub home_team_id: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
    /// Away team IDs for embeddings: [batch]
    pub away_team_id: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
    /// Match comparison features (win_rate_diff, margin_diff, pythagorean_diff, log5, is_local): [batch, 5]
    pub comparison: burn::tensor::Tensor<B, 2>,
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
        let mut home_team_ids = Vec::with_capacity(batch_size);
        let mut away_team_ids = Vec::with_capacity(batch_size);
        let mut comparison_data = Vec::with_capacity(batch_size * MatchComparison::DIM);
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
            home_team_ids.push(sample.home_team_id as i32);
            away_team_ids.push(sample.away_team_id as i32);
            comparison_data.extend(sample.comparison.to_vec());
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

        let home_team_id =
            burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_ints(home_team_ids.as_slice(), &self.device);

        let away_team_id =
            burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_ints(away_team_ids.as_slice(), &self.device);

        let comparison =
            burn::tensor::Tensor::<B, 1>::from_floats(comparison_data.as_slice(), &self.device)
                .reshape([batch_size, MatchComparison::DIM]);

        MatchBatch {
            home_history,
            away_history,
            home_mask,
            away_mask,
            home_team_id,
            away_team_id,
            comparison,
            home_win,
            home_score,
            away_score,
        }
    }
}
