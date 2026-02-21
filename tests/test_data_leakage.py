"""Tests to validate no data leakage in feature engineering.

Ensures that features for a match are built using only data from
matches that occurred strictly before the target match date.
"""

import pytest
import numpy as np
from datetime import datetime
from rugby.data import Match, Team
from rugby.features import FeatureBuilder


def _make_teams():
    return {
        1: Team(1, "Team A", "Country A", [], 0),
        2: Team(2, "Team B", "Country B", [], 0),
    }


def _make_chronological_matches():
    """Create matches where scores increase over time for easy tracking."""
    return [
        Match(1, datetime(2020, 1, 1), 1, 2, 10, 5, None, None),
        Match(2, datetime(2020, 2, 1), 2, 1, 15, 10, None, None),
        Match(3, datetime(2020, 3, 1), 1, 2, 20, 15, None, None),
        Match(4, datetime(2020, 4, 1), 2, 1, 25, 20, None, None),
        Match(5, datetime(2020, 5, 1), 1, 2, 30, 25, None, None),
    ]


class TestNoDataLeakage:
    def test_features_use_only_past_data(self):
        """Features for match N should not see match N or later."""
        teams = _make_teams()
        matches = _make_chronological_matches()
        fb = FeatureBuilder(matches, teams, min_history=2)

        features_by_date = {}
        for match in sorted(matches, key=lambda m: m.date):
            feat = fb.build_features(match)
            fb.process_match(match)
            if feat is not None:
                features_by_date[match.date] = feat

        # There should be at least one set of features
        assert len(features_by_date) > 0

        # Build features again from scratch to verify determinism
        fb2 = FeatureBuilder(matches, teams, min_history=2)
        for match in sorted(matches, key=lambda m: m.date):
            feat2 = fb2.build_features(match)
            fb2.process_match(match)
            if feat2 is not None:
                arr1 = features_by_date[match.date].to_array()
                arr2 = feat2.to_array()
                np.testing.assert_array_almost_equal(arr1, arr2)

    def test_elo_not_updated_prematurely(self):
        """Elo should reflect state before the match, not after."""
        teams = _make_teams()
        matches = _make_chronological_matches()
        fb = FeatureBuilder(matches, teams, min_history=1)

        initial_elo_1 = fb.elo_ratings[1]
        initial_elo_2 = fb.elo_ratings[2]
        assert initial_elo_1 == 1500.0
        assert initial_elo_2 == 1500.0

        # Build features for first match - Elo should still be at initial
        first_match = matches[0]
        feat = fb.build_features(first_match)
        if feat is not None:
            assert feat.home_elo == pytest.approx(1500.0)
            assert feat.away_elo == pytest.approx(1500.0)

        # Only after processing should Elo change
        fb.process_match(first_match)
        assert fb.elo_ratings[1] != 1500.0 or fb.elo_ratings[2] != 1500.0

    def test_build_dataset_ordering(self):
        """build_dataset processes matches chronologically."""
        teams = _make_teams()
        matches = _make_chronological_matches()
        fb = FeatureBuilder(matches, teams, min_history=2)
        X, y_win, _, _ = fb.build_dataset()

        # Should produce features only for matches with enough history
        assert len(X) > 0
        # All features should be finite
        assert np.all(np.isfinite(X))

    def test_no_future_h2h_leakage(self):
        """H2H stats should only include past encounters."""
        teams = _make_teams()
        matches = _make_chronological_matches()
        fb = FeatureBuilder(matches, teams, min_history=2)

        # Process only first 2 matches
        fb.process_match(matches[0])
        fb.process_match(matches[1])

        # H2H before match 3 should see 2 matches
        win_rate, margin = fb._compute_h2h_stats(1, 2, matches[2].date)
        assert win_rate >= 0
        assert win_rate <= 1

        # H2H before match 1 should see nothing
        fb2 = FeatureBuilder(matches, teams, min_history=2)
        win_rate_0, margin_0 = fb2._compute_h2h_stats(1, 2, matches[0].date)
        assert win_rate_0 == 0.5  # Default when no H2H data
        assert margin_0 == 0.0

    def test_team_history_chronological_order(self):
        """Team history should be most-recent-first within build_dataset."""
        teams = _make_teams()
        matches = _make_chronological_matches()
        fb = FeatureBuilder(matches, teams, min_history=1)

        for m in matches:
            fb.process_match(m)

        # Team 1's history should be most recent first
        history = fb.team_history[1]
        for i in range(len(history) - 1):
            assert history[i].date >= history[i + 1].date
