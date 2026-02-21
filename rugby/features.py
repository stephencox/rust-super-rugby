"""Feature engineering for rugby match prediction."""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from .data import Match, Team


@dataclass
class TeamStats:
    """Statistics for a team computed from recent matches."""
    win_rate: float = 0.0
    margin_avg: float = 0.0
    pythagorean: float = 0.5
    pf_avg: float = 0.0  # Points for average
    pa_avg: float = 0.0  # Points against average
    matches_played: int = 0

    # Recent form
    last_5_wins: int = 0
    streak: int = 0  # Positive = winning, negative = losing

    # Elo rating
    elo: float = 1500.0

    # Consistency (inverse of margin stdev, 0-1)
    consistency: float = 0.5

    # Strength of schedule (avg Elo of opponents)
    sos: float = 0.0


@dataclass
class MatchFeatures:
    """Features for a single match prediction."""
    # Differential features (5)
    win_rate_diff: float = 0.0
    margin_diff: float = 0.0
    pythagorean_diff: float = 0.0
    log5: float = 0.5
    is_local: float = 0.0

    # Home team stats (5)
    home_win_rate: float = 0.0
    home_margin_avg: float = 0.0
    home_pythagorean: float = 0.5
    home_pf_avg: float = 0.0
    home_pa_avg: float = 0.0

    # Away team stats (5)
    away_win_rate: float = 0.0
    away_margin_avg: float = 0.0
    away_pythagorean: float = 0.5
    away_pf_avg: float = 0.0
    away_pa_avg: float = 0.0

    # Elo features (3)
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    elo_diff: float = 0.0

    # Form features (4)
    home_streak: float = 0.0
    away_streak: float = 0.0
    home_last5_win_rate: float = 0.0
    away_last5_win_rate: float = 0.0

    # Head-to-head features (2) — from home team's perspective
    h2h_win_rate: float = 0.5
    h2h_margin_avg: float = 0.0

    # Context features (1)
    travel_hours: float = 0.0  # abs(home_tz - away_tz), symmetric

    # Per-team consistency (2)
    home_consistency: float = 0.5
    away_consistency: float = 0.5

    # Bye week indicators (2)
    home_is_after_bye: float = 0.0
    away_is_after_bye: float = 0.0

    # Strength of schedule (2)
    home_sos: float = 0.0
    away_sos: float = 0.0

    # Venue advantage (1) — home team's win rate at this venue
    home_venue_win_rate: float = 0.5

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            # Differentials (5)
            self.win_rate_diff,
            self.margin_diff,
            self.pythagorean_diff,
            self.log5,
            self.is_local,
            # Home stats (5)
            self.home_win_rate,
            self.home_margin_avg,
            self.home_pythagorean,
            self.home_pf_avg,
            self.home_pa_avg,
            # Away stats (5)
            self.away_win_rate,
            self.away_margin_avg,
            self.away_pythagorean,
            self.away_pf_avg,
            self.away_pa_avg,
            # Elo (3)
            self.home_elo,
            self.away_elo,
            self.elo_diff,
            # Form (4)
            self.home_streak,
            self.away_streak,
            self.home_last5_win_rate,
            self.away_last5_win_rate,
            # H2H (2)
            self.h2h_win_rate,
            self.h2h_margin_avg,
            # Context (1)
            self.travel_hours,
            # Consistency (2)
            self.home_consistency,
            self.away_consistency,
            # Bye (2)
            self.home_is_after_bye,
            self.away_is_after_bye,
            # SoS (2)
            self.home_sos,
            self.away_sos,
            # Venue (1)
            self.home_venue_win_rate,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Return list of feature names."""
        return [
            'win_rate_diff', 'margin_diff', 'pythagorean_diff', 'log5', 'is_local',
            'home_win_rate', 'home_margin_avg', 'home_pythagorean', 'home_pf_avg', 'home_pa_avg',
            'away_win_rate', 'away_margin_avg', 'away_pythagorean', 'away_pf_avg', 'away_pa_avg',
            'home_elo', 'away_elo', 'elo_diff',
            'home_streak', 'away_streak', 'home_last5_win_rate', 'away_last5_win_rate',
            'h2h_win_rate', 'h2h_margin_avg',
            'travel_hours',
            'home_consistency', 'away_consistency',
            'home_is_after_bye', 'away_is_after_bye',
            'home_sos', 'away_sos',
            'home_venue_win_rate',
        ]

    @staticmethod
    def num_features() -> int:
        return 32


class FeatureBuilder:
    """Builds features from match history."""

    def __init__(self, matches: List[Match], teams: Dict[int, Team],
                 max_history: int = 10, min_history: int = 3,
                 elo_k: float = 32.0):
        self.matches = sorted(matches, key=lambda m: m.date)
        self.teams = teams
        self.max_history = max_history
        self.min_history = min_history
        self.elo_k = elo_k

        # Initialize Elo ratings
        self.elo_ratings: Dict[int, float] = {tid: 1500.0 for tid in teams.keys()}

        # Team match history (most recent first)
        self.team_history: Dict[int, List[Match]] = {tid: [] for tid in teams.keys()}

    def _update_elo(self, home_id: int, away_id: int, home_win: bool):
        """Update Elo ratings after a match."""
        home_elo = self.elo_ratings[home_id]
        away_elo = self.elo_ratings[away_id]

        # Expected scores
        exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1.0 - exp_home

        # Actual scores
        actual_home = 1.0 if home_win else 0.0
        actual_away = 1.0 - actual_home

        # Update ratings
        self.elo_ratings[home_id] = home_elo + self.elo_k * (actual_home - exp_home)
        self.elo_ratings[away_id] = away_elo + self.elo_k * (actual_away - exp_away)

    def _compute_team_stats(self, team_id: int, before_date: datetime) -> Optional[TeamStats]:
        """Compute team statistics from recent matches before a given date."""
        # Get matches before date
        recent = [m for m in self.team_history[team_id]
                  if m.date < before_date][:self.max_history]

        if len(recent) < self.min_history:
            return None

        wins = 0
        total_pf = 0
        total_pa = 0
        total_margin = 0
        streak = 0
        streak_counting = True
        last_5_wins = 0

        margins = []
        opponent_ids = []

        for i, m in enumerate(recent):
            if m.home_team_id == team_id:
                pf, pa = m.home_score, m.away_score
                opponent_ids.append(m.away_team_id)
            else:
                pf, pa = m.away_score, m.home_score
                opponent_ids.append(m.home_team_id)

            won = pf > pa
            total_pf += pf
            total_pa += pa
            total_margin += (pf - pa)
            margins.append(pf - pa)

            if won:
                wins += 1
                if i < 5:
                    last_5_wins += 1

            # Compute streak
            if streak_counting:
                if i == 0:
                    streak = 1 if won else -1
                elif (won and streak > 0) or (not won and streak < 0):
                    streak += 1 if won else -1
                else:
                    streak_counting = False

        n = len(recent)
        pf_total = max(total_pf, 1)
        pa_total = max(total_pa, 1)

        # Pythagorean expectation
        pythagorean = (pf_total ** 2.37) / (pf_total ** 2.37 + pa_total ** 2.37)

        # Consistency: inverse of margin stdev, normalized to 0-1
        margin_std = float(np.std(margins)) if len(margins) > 1 else 10.0
        consistency = 1.0 / (1.0 + margin_std / 10.0)

        # Strength of schedule: average Elo of recent opponents
        opp_elos = [self.elo_ratings.get(oid, 1500.0) for oid in opponent_ids]
        sos = float(np.mean(opp_elos)) if opp_elos else 1500.0

        return TeamStats(
            win_rate=wins / n,
            margin_avg=total_margin / n,
            pythagorean=pythagorean,
            pf_avg=total_pf / n,
            pa_avg=total_pa / n,
            matches_played=n,
            last_5_wins=last_5_wins,
            streak=streak,
            elo=self.elo_ratings[team_id],
            consistency=consistency,
            sos=sos,
        )

    @staticmethod
    def _log5_prob(p_a: float, p_b: float) -> float:
        """Log5 formula for win probability."""
        p_a = max(0.01, min(0.99, p_a))
        p_b = max(0.01, min(0.99, p_b))
        denom = p_a + p_b - 2 * p_a * p_b
        if abs(denom) < 0.001:
            return 0.5
        return (p_a - p_a * p_b) / denom

    def _is_local_derby(self, home_id: int, away_id: int) -> bool:
        """Check if teams are from the same country."""
        home_team = self.teams.get(home_id)
        away_team = self.teams.get(away_id)
        if home_team and away_team and home_team.country and away_team.country:
            return home_team.country == away_team.country
        return False

    def _compute_h2h_stats(self, home_id: int, away_id: int, before_date: datetime) -> Tuple[float, float]:
        """Compute head-to-head stats from home team's perspective.

        Returns (win_rate, margin_avg) from home_id's perspective against away_id.
        Uses all available H2H history (not limited to max_history).
        """
        h2h_matches = [
            m for m in self.team_history[home_id]
            if m.date < before_date and (m.home_team_id == away_id or m.away_team_id == away_id)
        ]

        if not h2h_matches:
            return 0.5, 0.0

        wins = 0
        total_margin = 0
        for m in h2h_matches:
            if m.home_team_id == home_id:
                pf, pa = m.home_score, m.away_score
            else:
                pf, pa = m.away_score, m.home_score
            if pf > pa:
                wins += 1
            total_margin += (pf - pa)

        n = len(h2h_matches)
        return wins / n, total_margin / n

    def _compute_rest_days(self, team_id: int, before_date: datetime) -> float:
        """Compute days since last match for a team."""
        history = [m for m in self.team_history[team_id] if m.date < before_date]
        if not history:
            return 7.0  # Default
        return float((before_date - history[0].date).days)

    def _compute_venue_win_rate(self, team_id: int, venue: Optional[str],
                                before_date: datetime) -> float:
        """Compute a team's win rate at a specific venue (home games only)."""
        if not venue:
            return 0.5
        venue_lower = venue.lower().strip()
        wins = 0
        total = 0
        for m in self.team_history[team_id]:
            if m.date >= before_date:
                continue
            if m.home_team_id != team_id:
                continue  # Only home games at this venue
            if m.venue and m.venue.lower().strip() == venue_lower:
                total += 1
                if m.home_win:
                    wins += 1
        if total < 2:
            return 0.5  # Not enough venue history
        return wins / total

    def build_features(self, match: Match) -> Optional[MatchFeatures]:
        """Build features for a match (using only data available before the match)."""
        home_stats = self._compute_team_stats(match.home_team_id, match.date)
        away_stats = self._compute_team_stats(match.away_team_id, match.date)

        if home_stats is None or away_stats is None:
            return None

        is_local = self._is_local_derby(match.home_team_id, match.away_team_id)
        log5 = self._log5_prob(home_stats.win_rate, away_stats.win_rate)
        h2h_win_rate, h2h_margin_avg = self._compute_h2h_stats(
            match.home_team_id, match.away_team_id, match.date
        )

        # Travel hours: absolute timezone difference (symmetric)
        home_team = self.teams.get(match.home_team_id)
        away_team = self.teams.get(match.away_team_id)
        home_tz = home_team.timezone_offset if home_team and home_team.timezone_offset is not None else 0.0
        away_tz = away_team.timezone_offset if away_team and away_team.timezone_offset is not None else 0.0
        travel_hours = abs(home_tz - away_tz)

        # Bye week: rest >= 13 days
        home_rest = self._compute_rest_days(match.home_team_id, match.date)
        away_rest = self._compute_rest_days(match.away_team_id, match.date)
        home_is_after_bye = 1.0 if home_rest >= 13 else 0.0
        away_is_after_bye = 1.0 if away_rest >= 13 else 0.0

        # Venue-specific home advantage
        home_venue_wr = self._compute_venue_win_rate(
            match.home_team_id, match.venue, match.date
        )

        return MatchFeatures(
            # Differentials
            win_rate_diff=home_stats.win_rate - away_stats.win_rate,
            margin_diff=home_stats.margin_avg - away_stats.margin_avg,
            pythagorean_diff=home_stats.pythagorean - away_stats.pythagorean,
            log5=log5,
            is_local=1.0 if is_local else 0.0,
            # Home stats
            home_win_rate=home_stats.win_rate,
            home_margin_avg=home_stats.margin_avg,
            home_pythagorean=home_stats.pythagorean,
            home_pf_avg=home_stats.pf_avg,
            home_pa_avg=home_stats.pa_avg,
            # Away stats
            away_win_rate=away_stats.win_rate,
            away_margin_avg=away_stats.margin_avg,
            away_pythagorean=away_stats.pythagorean,
            away_pf_avg=away_stats.pf_avg,
            away_pa_avg=away_stats.pa_avg,
            # Elo
            home_elo=home_stats.elo,
            away_elo=away_stats.elo,
            elo_diff=home_stats.elo - away_stats.elo,
            # Form
            home_streak=home_stats.streak / 5.0,  # Normalize to roughly -1 to 1
            away_streak=away_stats.streak / 5.0,
            home_last5_win_rate=home_stats.last_5_wins / 5.0,
            away_last5_win_rate=away_stats.last_5_wins / 5.0,
            # H2H
            h2h_win_rate=h2h_win_rate,
            h2h_margin_avg=h2h_margin_avg,
            # Context
            travel_hours=travel_hours,
            # Consistency
            home_consistency=home_stats.consistency,
            away_consistency=away_stats.consistency,
            # Bye
            home_is_after_bye=home_is_after_bye,
            away_is_after_bye=away_is_after_bye,
            # SoS
            home_sos=home_stats.sos,
            away_sos=away_stats.sos,
            # Venue
            home_venue_win_rate=home_venue_wr,
        )

    def process_match(self, match: Match):
        """Process a match: update history and Elo (call after using features for training)."""
        # Add to team histories (at front, most recent first)
        self.team_history[match.home_team_id].insert(0, match)
        self.team_history[match.away_team_id].insert(0, match)

        # Update Elo
        self._update_elo(match.home_team_id, match.away_team_id, match.home_win)

    def build_dataset(self, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build dataset from matches in date range.

        Returns: (X, y_win, y_home_score, y_away_score)
        """
        X_list = []
        y_win_list = []
        y_home_score_list = []
        y_away_score_list = []

        for match in self.matches:
            # Build features using only pre-match data
            features = self.build_features(match)

            # Now update history with this match (for future matches)
            self.process_match(match)

            # Skip if not enough history
            if features is None:
                continue

            # Check date range
            if start_date and match.date < start_date:
                continue
            if end_date and match.date >= end_date:
                continue

            X_list.append(features.to_array())
            y_win_list.append(1.0 if match.home_win else 0.0)
            y_home_score_list.append(float(match.home_score))
            y_away_score_list.append(float(match.away_score))

        return (
            np.array(X_list),
            np.array(y_win_list),
            np.array(y_home_score_list),
            np.array(y_away_score_list),
        )


class FeatureNormalizer:
    """Z-score normalization for features."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        """Compute normalization parameters from training data."""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        # Avoid division by zero
        self.std[self.std < 0.001] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.mean is None:
            raise ValueError("Normalizer not fitted")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class SequenceNormalizer:
    """
    Z-score normalization for sequence data.

    Normalizes both sequence features and comparison features.
    """

    def __init__(self):
        self.seq_mean: Optional[np.ndarray] = None
        self.seq_std: Optional[np.ndarray] = None
        self.comp_mean: Optional[np.ndarray] = None
        self.comp_std: Optional[np.ndarray] = None

    def fit(self, samples: List['SequenceDataSample']):
        """
        Compute normalization parameters from training samples.

        Aggregates all non-padding timesteps to compute sequence stats.
        """
        # Collect all sequence timesteps (exclude padding)
        all_seq_features = []
        all_comparisons = []

        for sample in samples:
            # Flatten home and away history (skip padding based on zeros)
            for seq in [sample.home_history, sample.away_history]:
                for timestep in seq:
                    # Skip if it's all zeros (padding)
                    if np.any(timestep != 0):
                        all_seq_features.append(timestep)
            all_comparisons.append(sample.comparison)

        seq_array = np.array(all_seq_features)
        comp_array = np.array(all_comparisons)

        # Compute stats
        self.seq_mean = seq_array.mean(axis=0)
        self.seq_std = seq_array.std(axis=0)
        self.seq_std[self.seq_std < 0.001] = 1.0

        self.comp_mean = comp_array.mean(axis=0)
        self.comp_std = comp_array.std(axis=0)
        self.comp_std[self.comp_std < 0.001] = 1.0

    def transform_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Normalize a sequence [seq_len, features]."""
        if self.seq_mean is None:
            raise ValueError("Normalizer not fitted")
        return (seq - self.seq_mean) / self.seq_std

    def transform_comparison(self, comp: np.ndarray) -> np.ndarray:
        """Normalize comparison features [features]."""
        if self.comp_mean is None:
            raise ValueError("Normalizer not fitted")
        return (comp - self.comp_mean) / self.comp_std

    def transform_sample(self, sample: 'SequenceDataSample') -> 'SequenceDataSample':
        """Transform a single sample in-place."""
        return SequenceDataSample(
            home_history=self.transform_sequence(sample.home_history),
            away_history=self.transform_sequence(sample.away_history),
            comparison=self.transform_comparison(sample.comparison),
            home_team_id=sample.home_team_id,
            away_team_id=sample.away_team_id,
            home_win=sample.home_win,
            home_score=sample.home_score,
            away_score=sample.away_score,
        )

    def transform_samples(
        self, samples: List['SequenceDataSample']
    ) -> List['SequenceDataSample']:
        """Transform a list of samples."""
        return [self.transform_sample(s) for s in samples]

    def fit_transform(
        self, samples: List['SequenceDataSample']
    ) -> List['SequenceDataSample']:
        """Fit and transform in one step."""
        self.fit(samples)
        return self.transform_samples(samples)

    def save(self, path: str):
        """Save normalizer parameters to file."""
        np.savez(
            path,
            seq_mean=self.seq_mean,
            seq_std=self.seq_std,
            comp_mean=self.comp_mean,
            comp_std=self.comp_std,
        )

    def load(self, path: str):
        """Load normalizer parameters from file."""
        data = np.load(path)
        self.seq_mean = data['seq_mean']
        self.seq_std = data['seq_std']
        self.comp_mean = data['comp_mean']
        self.comp_std = data['comp_std']


# =============================================================================
# Sequence-based features for LSTM and Transformer models
# =============================================================================

# Per-match features dimension (23 features per historical match, matching Rust)
SEQUENCE_FEATURE_DIM = 23


@dataclass
class SequenceMatchFeatures:
    """
    Features for a single historical match in a team's sequence.

    These features are computed from the perspective of a specific team.
    23 features total (matching Rust implementation).
    """
    # Match result (3)
    score_for: float = 0.0         # Points scored by this team
    score_against: float = 0.0     # Points conceded
    margin: float = 0.0            # Positive = win

    # Try data (2) - set to 0 if not available
    tries_for: float = 0.0
    tries_against: float = 0.0

    # Match context (3)
    is_home: float = 0.0           # 1.0 if home, 0.0 if away
    win_indicator: float = 0.0     # 1.0 if won, 0.0 if lost, 0.5 if draw
    streak_count: float = 0.0      # Consecutive home/away games (normalized 0-1)

    # Temporal features (4)
    recency: float = 0.0           # Days since match (normalized 0-1 for ~1 year)
    opponent_idx: float = 0.0      # Opponent team index (for embeddings)
    travel_hours: float = 0.0      # Timezone diff (normalized -1 to 1)
    days_rest: float = 0.0         # Days since previous match (normalized)

    # Rolling statistics (4) - computed at time of match
    rolling_win_rate: float = 0.0
    rolling_margin_avg: float = 0.0
    pythagorean_exp: float = 0.5
    consistency: float = 0.5       # Inverse of margin stdev, normalized

    # Temporal context (7) - match timing features
    is_saturday: float = 0.0       # 1.0 if Saturday
    is_friday: float = 0.0         # 1.0 if Friday
    season_progress: float = 0.0   # 0.0 (Feb) to 1.0 (Jul)
    is_early_season: float = 0.0   # 1.0 if Feb-Mar
    is_late_season: float = 0.0    # 1.0 if May-Jul
    round_normalized: float = 0.0  # Round / 20
    short_turnaround: float = 0.0  # 1.0 if <7 days rest

    def to_array(self) -> np.ndarray:
        """Convert to numpy array (23 features)."""
        return np.array([
            self.score_for,
            self.score_against,
            self.margin,
            self.tries_for,
            self.tries_against,
            self.is_home,
            self.win_indicator,
            self.streak_count,
            self.recency,
            self.opponent_idx,
            self.travel_hours,
            self.days_rest,
            self.rolling_win_rate,
            self.rolling_margin_avg,
            self.pythagorean_exp,
            self.consistency,
            self.is_saturday,
            self.is_friday,
            self.season_progress,
            self.is_early_season,
            self.is_late_season,
            self.round_normalized,
            self.short_turnaround,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'score_for', 'score_against', 'margin',
            'tries_for', 'tries_against',
            'is_home', 'win_indicator', 'streak_count',
            'recency', 'opponent_idx', 'travel_hours', 'days_rest',
            'rolling_win_rate', 'rolling_margin_avg', 'pythagorean_exp', 'consistency',
            'is_saturday', 'is_friday', 'season_progress',
            'is_early_season', 'is_late_season', 'round_normalized', 'short_turnaround',
        ]


@dataclass
class SequenceDataSample:
    """A single training sample for sequence models."""
    home_history: np.ndarray       # [seq_len, 15]
    away_history: np.ndarray       # [seq_len, 15]
    comparison: np.ndarray         # [22] - same as MatchFeatures
    home_team_id: int
    away_team_id: int
    home_win: bool
    home_score: float
    away_score: float


class SequenceFeatureBuilder:
    """
    Builds sequence features for LSTM and Transformer models.

    Generates per-match feature sequences for each team's history,
    plus comparison features for the current matchup.
    """

    def __init__(
        self,
        matches: List[Match],
        teams: Dict[int, Team],
        seq_len: int = 10,
        min_history: int = 3,
        elo_k: float = 32.0,
    ):
        self.matches = sorted(matches, key=lambda m: m.date)
        self.teams = teams
        self.seq_len = seq_len
        self.min_history = min_history
        self.elo_k = elo_k

        # Team ID to index mapping
        self.team_to_idx = {tid: i for i, tid in enumerate(sorted(teams.keys()))}

        # Initialize Elo ratings
        self.elo_ratings: Dict[int, float] = {tid: 1500.0 for tid in teams.keys()}

        # Team match history with computed stats at time of match
        # Each entry: (match, rolling_stats_at_time)
        self.team_history: Dict[int, List[Tuple[Match, Dict]]] = {
            tid: [] for tid in teams.keys()
        }

    def _update_elo(self, home_id: int, away_id: int, home_win: bool):
        """Update Elo ratings after a match."""
        home_elo = self.elo_ratings[home_id]
        away_elo = self.elo_ratings[away_id]

        exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1.0 - exp_home

        actual_home = 1.0 if home_win else 0.0
        actual_away = 1.0 - actual_home

        self.elo_ratings[home_id] = home_elo + self.elo_k * (actual_home - exp_home)
        self.elo_ratings[away_id] = away_elo + self.elo_k * (actual_away - exp_away)

    def _compute_rolling_stats(self, team_id: int, before_date: datetime) -> Dict:
        """Compute rolling statistics for a team up to a given date."""
        history = [h for h in self.team_history[team_id] if h[0].date < before_date]

        if len(history) < 2:
            return {
                'win_rate': 0.5,
                'margin_avg': 0.0,
                'pythagorean': 0.5,
                'consistency': 0.5,
                'pf_avg': 20.0,
                'pa_avg': 20.0,
                'streak': 0,
                'last_5_wins': 0,
            }

        # Use last 10 matches
        recent = history[:10]

        wins = 0
        total_pf = 0
        total_pa = 0
        margins = []
        streak = 0
        streak_counting = True
        last_5_wins = 0

        for i, (m, _) in enumerate(recent):
            if m.home_team_id == team_id:
                pf, pa = m.home_score, m.away_score
            else:
                pf, pa = m.away_score, m.home_score

            won = pf > pa
            total_pf += pf
            total_pa += pa
            margins.append(pf - pa)

            if won:
                wins += 1
                if i < 5:
                    last_5_wins += 1

            if streak_counting:
                if i == 0:
                    streak = 1 if won else -1
                elif (won and streak > 0) or (not won and streak < 0):
                    streak += 1 if won else -1
                else:
                    streak_counting = False

        n = len(recent)
        pf_total = max(total_pf, 1)
        pa_total = max(total_pa, 1)
        pythagorean = (pf_total ** 2.37) / (pf_total ** 2.37 + pa_total ** 2.37)

        # Consistency: inverse of margin stdev, normalized to 0-1
        margin_std = np.std(margins) if len(margins) > 1 else 10.0
        consistency = 1.0 / (1.0 + margin_std / 10.0)

        return {
            'win_rate': wins / n,
            'margin_avg': np.mean(margins),
            'pythagorean': pythagorean,
            'consistency': consistency,
            'pf_avg': total_pf / n,
            'pa_avg': total_pa / n,
            'streak': streak,
            'last_5_wins': last_5_wins,
        }

    def _build_team_sequence(
        self,
        team_id: int,
        before_date: datetime,
    ) -> Optional[np.ndarray]:
        """Build sequence features for a team's history (raw values, no manual normalization)."""
        history = [h for h in self.team_history[team_id] if h[0].date < before_date]

        if len(history) < self.min_history:
            return None

        # Take most recent seq_len matches
        recent = history[:self.seq_len]

        # Pad if necessary
        sequence = []
        for i in range(self.seq_len):
            if i < len(recent):
                m, stats = recent[i]

                if m.home_team_id == team_id:
                    pf, pa = m.home_score, m.away_score
                    opponent_id = m.away_team_id
                    is_home = 1.0
                else:
                    pf, pa = m.away_score, m.home_score
                    opponent_id = m.home_team_id
                    is_home = 0.0

                margin = pf - pa
                won = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)

                # Recency: days ago (raw value)
                days_ago = float((before_date - m.date).days)

                # Days rest (from previous match in sequence)
                if i + 1 < len(recent):
                    prev_date = recent[i + 1][0].date
                    days_rest = float((m.date - prev_date).days)
                else:
                    days_rest = 7.0  # Default to 1 week

                # Streak count: consecutive home/away games
                streak_count = 0
                for j, (prev_m, _) in enumerate(recent[i:]):
                    prev_is_home = prev_m.home_team_id == team_id
                    if prev_is_home == (is_home == 1.0):
                        streak_count += 1
                    else:
                        break
                streak_count = min(streak_count, 5) / 5.0  # Normalize to 0-1

                # Temporal features
                day_of_week = m.date.weekday()
                is_saturday = 1.0 if day_of_week == 5 else 0.0
                is_friday = 1.0 if day_of_week == 4 else 0.0

                # Season progress: Feb=0.0, Jul=1.0
                month = m.date.month
                if month >= 2 and month <= 7:
                    season_progress = (month - 2) / 5.0
                else:
                    season_progress = 0.0

                is_early_season = 1.0 if month in [2, 3] else 0.0
                is_late_season = 1.0 if month in [5, 6, 7] else 0.0

                # Round normalized (estimate from date if not available)
                round_normalized = season_progress  # Approximate

                # Short turnaround
                short_turnaround = 1.0 if days_rest < 7 else 0.0

                features = SequenceMatchFeatures(
                    score_for=float(pf),          # Raw score
                    score_against=float(pa),      # Raw score
                    margin=float(margin),         # Raw margin
                    tries_for=0.0,                # Not available
                    tries_against=0.0,
                    is_home=is_home,              # Binary (already 0 or 1)
                    win_indicator=won,            # Already 0, 0.5, or 1
                    streak_count=streak_count,    # Normalized 0-1
                    recency=days_ago,             # Raw days
                    opponent_idx=float(self.team_to_idx.get(opponent_id, 0)),  # Raw index
                    travel_hours=0.0,
                    days_rest=days_rest,          # Raw days
                    rolling_win_rate=stats['win_rate'],       # Already 0-1
                    rolling_margin_avg=stats['margin_avg'],   # Raw margin
                    pythagorean_exp=stats['pythagorean'],     # Already 0-1
                    consistency=stats['consistency'],         # Already 0-1
                    is_saturday=is_saturday,
                    is_friday=is_friday,
                    season_progress=season_progress,
                    is_early_season=is_early_season,
                    is_late_season=is_late_season,
                    round_normalized=round_normalized,
                    short_turnaround=short_turnaround,
                )
            else:
                # Padding with zeros
                features = SequenceMatchFeatures()

            sequence.append(features.to_array())

        return np.array(sequence, dtype=np.float32)

    def _compute_rest_days(self, team_id: int, before_date: datetime) -> float:
        """Compute days since last match for a team."""
        history = [h for h in self.team_history[team_id] if h[0].date < before_date]
        if not history:
            return 7.0  # Default
        last_match = history[0][0]
        return float((before_date - last_match.date).days)

    def _compute_workload(self, team_id: int, before_date: datetime, days: int) -> int:
        """Count matches in the last N days."""
        cutoff = before_date - timedelta(days=days)
        history = [h for h in self.team_history[team_id] if cutoff <= h[0].date < before_date]
        return len(history)

    def _compute_h2h_stats(self, home_id: int, away_id: int, before_date: datetime) -> Tuple[float, float]:
        """Compute head-to-head recency and games since last H2H."""
        # Find last H2H match
        home_history = [h for h in self.team_history[home_id] if h[0].date < before_date]
        h2h_recency = 365.0  # Default: 1 year
        games_since_h2h = 20.0  # Default

        for i, (m, _) in enumerate(home_history):
            if (m.home_team_id == away_id or m.away_team_id == away_id):
                h2h_recency = float((before_date - m.date).days)
                games_since_h2h = float(i)
                break

        return h2h_recency / 365.0, games_since_h2h / 20.0

    def _build_comparison_features(
        self,
        home_id: int,
        away_id: int,
        before_date: datetime,
    ) -> Optional[np.ndarray]:
        """Build comparison features (50 dims, matching Rust implementation)."""
        home_stats = self._compute_rolling_stats(home_id, before_date)
        away_stats = self._compute_rolling_stats(away_id, before_date)

        # Need enough history
        home_history = [h for h in self.team_history[home_id] if h[0].date < before_date]
        away_history = [h for h in self.team_history[away_id] if h[0].date < before_date]

        if len(home_history) < self.min_history or len(away_history) < self.min_history:
            return None

        # Log5 probability (naturally 0-1)
        p_a = max(0.01, min(0.99, home_stats['win_rate']))
        p_b = max(0.01, min(0.99, away_stats['win_rate']))
        denom = p_a + p_b - 2 * p_a * p_b
        log5 = (p_a - p_a * p_b) / denom if abs(denom) > 0.001 else 0.5

        # Local derby check (binary)
        home_team = self.teams.get(home_id)
        away_team = self.teams.get(away_id)
        is_local = 0.0
        if home_team and away_team and home_team.country and away_team.country:
            is_local = 1.0 if home_team.country == away_team.country else 0.0

        # Temporal features
        day_of_week = before_date.weekday()
        is_saturday = 1.0 if day_of_week == 5 else 0.0
        is_friday = 1.0 if day_of_week == 4 else 0.0
        is_sunday = 1.0 if day_of_week == 6 else 0.0

        month = before_date.month
        if month >= 2 and month <= 7:
            season_progress = (month - 2) / 5.0
        else:
            season_progress = 0.0
        is_early_season = 1.0 if month in [2, 3] else 0.0
        is_late_season = 1.0 if month in [5, 6, 7] else 0.0
        round_normalized = season_progress  # Approximate

        # Rest days
        home_rest = self._compute_rest_days(home_id, before_date)
        away_rest = self._compute_rest_days(away_id, before_date)
        home_rest_norm = min(home_rest, 14.0) / 14.0
        away_rest_norm = min(away_rest, 14.0) / 14.0
        rest_advantage = (home_rest - away_rest) / 7.0

        home_short_turnaround = 1.0 if home_rest < 7 else 0.0
        away_short_turnaround = 1.0 if away_rest < 7 else 0.0

        # Consecutive home/away streak
        home_streak_count = 0
        for m, _ in home_history[:5]:
            if m.home_team_id == home_id:
                home_streak_count += 1
            else:
                break
        away_streak_count = 0
        for m, _ in away_history[:5]:
            if m.away_team_id == away_id:
                away_streak_count += 1
            else:
                break
        home_streak_norm = home_streak_count / 5.0
        away_streak_norm = away_streak_count / 5.0

        # H2H features
        h2h_recency, games_since_h2h = self._compute_h2h_stats(home_id, away_id, before_date)

        # Match density (matches in last 21 days)
        home_match_density = self._compute_workload(home_id, before_date, 21) / 4.0
        away_match_density = self._compute_workload(away_id, before_date, 21) / 4.0

        # Workload features
        home_7d = self._compute_workload(home_id, before_date, 7) / 2.0
        home_14d = self._compute_workload(home_id, before_date, 14) / 3.0
        home_21d = self._compute_workload(home_id, before_date, 21) / 4.0
        away_7d = self._compute_workload(away_id, before_date, 7) / 2.0
        away_14d = self._compute_workload(away_id, before_date, 14) / 3.0
        away_21d = self._compute_workload(away_id, before_date, 21) / 4.0
        workload_diff_7d = (self._compute_workload(home_id, before_date, 7) -
                           self._compute_workload(away_id, before_date, 7)) / 2.0
        workload_diff_14d = (self._compute_workload(home_id, before_date, 14) -
                            self._compute_workload(away_id, before_date, 14)) / 3.0
        workload_diff_21d = (self._compute_workload(home_id, before_date, 21) -
                            self._compute_workload(away_id, before_date, 21)) / 4.0

        # Venue features (simplified - we don't track specific venues)
        # Use home ground advantage as proxy
        home_venue_win_rate = home_stats['win_rate']  # Approximate
        home_venue_games = min(len(home_history), 10) / 10.0
        away_venue_win_rate = 0.5  # Neutral for away team at opponent's venue
        away_venue_games = 0.0
        venue_familiarity_diff = home_venue_games - away_venue_games

        return np.array([
            # Differentials (5)
            home_stats['win_rate'] - away_stats['win_rate'],
            home_stats['margin_avg'] - away_stats['margin_avg'],
            home_stats['pythagorean'] - away_stats['pythagorean'],
            log5,
            is_local,
            # Home stats (5)
            home_stats['win_rate'],
            home_stats['margin_avg'],
            home_stats['pythagorean'],
            home_stats['pf_avg'],
            home_stats['pa_avg'],
            # Away stats (5)
            away_stats['win_rate'],
            away_stats['margin_avg'],
            away_stats['pythagorean'],
            away_stats['pf_avg'],
            away_stats['pa_avg'],
            # Temporal (18)
            is_saturday,
            is_friday,
            is_sunday,
            season_progress,
            is_early_season,
            is_late_season,
            round_normalized,
            home_rest_norm,
            away_rest_norm,
            rest_advantage,
            home_short_turnaround,
            away_short_turnaround,
            home_streak_norm,
            away_streak_norm,
            h2h_recency,
            games_since_h2h,
            home_match_density,
            away_match_density,
            # Elo (3)
            self.elo_ratings[home_id],
            self.elo_ratings[away_id],
            self.elo_ratings[home_id] - self.elo_ratings[away_id],
            # Workload (9)
            home_7d,
            home_14d,
            home_21d,
            away_7d,
            away_14d,
            away_21d,
            workload_diff_7d,
            workload_diff_14d,
            workload_diff_21d,
            # Venue (5)
            home_venue_win_rate,
            home_venue_games,
            away_venue_win_rate,
            away_venue_games,
            venue_familiarity_diff,
        ], dtype=np.float32)

    def build_sample(self, match: Match) -> Optional[SequenceDataSample]:
        """Build a training sample for a match."""
        home_seq = self._build_team_sequence(match.home_team_id, match.date)
        away_seq = self._build_team_sequence(match.away_team_id, match.date)
        comparison = self._build_comparison_features(
            match.home_team_id, match.away_team_id, match.date
        )

        if home_seq is None or away_seq is None or comparison is None:
            return None

        return SequenceDataSample(
            home_history=home_seq,
            away_history=away_seq,
            comparison=comparison,
            home_team_id=self.team_to_idx[match.home_team_id],
            away_team_id=self.team_to_idx[match.away_team_id],
            home_win=match.home_win,
            home_score=float(match.home_score),
            away_score=float(match.away_score),
        )

    def process_match(self, match: Match):
        """Process a match: update history and Elo."""
        # Compute rolling stats at time of this match
        home_stats = self._compute_rolling_stats(match.home_team_id, match.date)
        away_stats = self._compute_rolling_stats(match.away_team_id, match.date)

        # Add to team histories with stats at time of match
        self.team_history[match.home_team_id].insert(0, (match, home_stats))
        self.team_history[match.away_team_id].insert(0, (match, away_stats))

        # Update Elo
        self._update_elo(match.home_team_id, match.away_team_id, match.home_win)

    def build_dataset(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[SequenceDataSample]:
        """Build dataset from matches in date range."""
        samples = []

        for match in self.matches:
            # Build sample using only pre-match data
            sample = self.build_sample(match)

            # Update history with this match
            self.process_match(match)

            if sample is None:
                continue

            # Check date range
            if start_date and match.date < start_date:
                continue
            if end_date and match.date >= end_date:
                continue

            samples.append(sample)

        return samples
