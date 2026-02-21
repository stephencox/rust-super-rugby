"""Tests for rugby.features module."""

import pytest
import numpy as np
from datetime import datetime
from rugby.data import Match, Team
from rugby.features import (
    MatchFeatures, TeamStats, FeatureBuilder, FeatureNormalizer,
)


def _make_teams():
    """Create a dict of test teams."""
    return {
        1: Team(1, "Blues", "New Zealand", [], 12),
        2: Team(2, "Chiefs", "New Zealand", [], 12),
        3: Team(3, "Stormers", "South Africa", [], 2),
    }


def _make_matches():
    """Create a chronological list of test matches."""
    return [
        Match(i, datetime(2020, 2 + i, 1), h, a, hs, aws, None, None)
        for i, (h, a, hs, aws) in enumerate([
            (1, 2, 25, 20),
            (2, 1, 30, 15),
            (1, 3, 18, 22),
            (2, 3, 35, 28),
            (1, 2, 27, 20),
        ], start=1)
    ]


class TestMatchFeatures:
    def test_num_features(self):
        assert MatchFeatures.num_features() == 32

    def test_to_array_length(self):
        f = MatchFeatures()
        arr = f.to_array()
        assert len(arr) == MatchFeatures.num_features()
        assert arr.dtype == np.float32

    def test_feature_names_length(self):
        names = MatchFeatures.feature_names()
        assert len(names) == MatchFeatures.num_features()

    def test_feature_names_unique(self):
        names = MatchFeatures.feature_names()
        assert len(names) == len(set(names))

    def test_to_array_values(self):
        f = MatchFeatures(win_rate_diff=0.3, home_elo=1550.0, away_elo=1450.0)
        arr = f.to_array()
        assert arr[0] == pytest.approx(0.3)  # win_rate_diff
        assert arr[15] == pytest.approx(1550.0)  # home_elo
        assert arr[16] == pytest.approx(1450.0)  # away_elo


class TestTeamStats:
    def test_defaults(self):
        s = TeamStats()
        assert s.win_rate == 0.0
        assert s.elo == 1500.0
        assert s.consistency == 0.5
        assert s.sos == 0.0


class TestFeatureBuilder:
    def test_build_dataset_returns_arrays(self):
        teams = _make_teams()
        matches = _make_matches()
        fb = FeatureBuilder(matches, teams, min_history=3)
        X, y_win, y_hs, y_as = fb.build_dataset()
        assert isinstance(X, np.ndarray)
        assert X.shape[1] == MatchFeatures.num_features()
        assert len(y_win) == len(X)

    def test_min_history_filter(self):
        teams = _make_teams()
        matches = _make_matches()
        fb = FeatureBuilder(matches, teams, min_history=3)
        X, y_win, _, _ = fb.build_dataset()
        # With only 5 matches and min_history=3, only later matches produce features
        assert len(X) > 0
        assert len(X) < len(matches)

    def test_date_range_filter(self):
        teams = _make_teams()
        matches = _make_matches()
        fb = FeatureBuilder(matches, teams, min_history=2)
        cutoff = datetime(2020, 5, 1)
        X_before, _, _, _ = fb.build_dataset(end_date=cutoff)

        fb2 = FeatureBuilder(matches, teams, min_history=2)
        X_all, _, _, _ = fb2.build_dataset()
        assert len(X_before) <= len(X_all)

    def test_elo_updates(self):
        teams = _make_teams()
        matches = _make_matches()
        fb = FeatureBuilder(matches, teams, min_history=1)
        fb.build_dataset()
        # After processing matches, Elo ratings should have diverged from 1500
        elos = set(fb.elo_ratings.values())
        assert not all(e == 1500.0 for e in elos)

    def test_h2h_stats(self):
        teams = _make_teams()
        matches = _make_matches()
        fb = FeatureBuilder(matches, teams)
        # Process matches to build history
        for m in matches:
            fb.process_match(m)
        # Blues vs Chiefs H2H
        win_rate, margin_avg = fb._compute_h2h_stats(1, 2, datetime(2021, 1, 1))
        assert 0 <= win_rate <= 1

    def test_log5_edge_cases(self):
        assert FeatureBuilder._log5_prob(0.5, 0.5) == pytest.approx(0.5)
        assert FeatureBuilder._log5_prob(0.99, 0.01) > 0.9
        assert FeatureBuilder._log5_prob(0.01, 0.99) < 0.1

    def test_is_local_derby(self):
        teams = _make_teams()
        fb = FeatureBuilder([], teams)
        assert fb._is_local_derby(1, 2) is True   # Both NZ
        assert fb._is_local_derby(1, 3) is False   # NZ vs SA

    def test_consistency_computed(self):
        teams = _make_teams()
        matches = _make_matches()
        fb = FeatureBuilder(matches, teams, min_history=3)
        for m in matches:
            fb.process_match(m)
        stats = fb._compute_team_stats(1, datetime(2021, 1, 1))
        assert stats is not None
        assert 0 < stats.consistency <= 1

    def test_sos_computed(self):
        teams = _make_teams()
        matches = _make_matches()
        fb = FeatureBuilder(matches, teams, min_history=3)
        for m in matches:
            fb.process_match(m)
        stats = fb._compute_team_stats(1, datetime(2021, 1, 1))
        assert stats is not None
        assert stats.sos > 0  # Should be around 1500

    def test_travel_hours(self):
        teams = _make_teams()
        # Blues (tz=12) vs Stormers (tz=2) → travel_hours = 10
        matches = [
            Match(1, datetime(2020, 2, 1), 1, 3, 20, 15, None, None),
            Match(2, datetime(2020, 3, 1), 3, 1, 18, 22, None, None),
            Match(3, datetime(2020, 4, 1), 1, 3, 25, 20, None, None),
            Match(4, datetime(2020, 5, 1), 1, 3, 30, 10, None, None),
        ]
        fb = FeatureBuilder(matches, teams, min_history=3)
        X, _, _, _ = fb.build_dataset()
        if len(X) > 0:
            # travel_hours is at index 24
            assert X[0, 24] == pytest.approx(10.0)

    def test_bye_week_detection(self):
        teams = _make_teams()
        # Two matches with a >13 day gap for team 1
        matches = [
            Match(1, datetime(2020, 2, 1), 1, 2, 20, 15, None, None),
            Match(2, datetime(2020, 2, 8), 2, 1, 18, 22, None, None),
            Match(3, datetime(2020, 2, 15), 1, 2, 25, 20, None, None),
            # Big gap here for team 1 (28 days)
            Match(4, datetime(2020, 3, 14), 1, 2, 30, 10, None, None),
        ]
        fb = FeatureBuilder(matches, teams, min_history=3)
        X, _, _, _ = fb.build_dataset()
        if len(X) > 0:
            # Last match should have home_is_after_bye = 1.0 (index 27)
            assert X[-1, 27] == 1.0


class TestFeatureNormalizer:
    def test_fit_transform(self):
        X = np.random.randn(100, 31).astype(np.float32)
        norm = FeatureNormalizer()
        X_norm = norm.fit_transform(X)
        assert X_norm.shape == X.shape
        # Mean should be ~0, std ~1
        assert np.abs(X_norm.mean(axis=0)).max() < 0.1
        assert np.abs(X_norm.std(axis=0) - 1.0).max() < 0.1

    def test_transform_before_fit_raises(self):
        norm = FeatureNormalizer()
        with pytest.raises(ValueError):
            norm.transform(np.zeros((5, 31)))

    def test_zero_std_handled(self):
        X = np.ones((50, 5), dtype=np.float32)
        norm = FeatureNormalizer()
        X_norm = norm.fit_transform(X)
        # Constant columns get std=1, so result should be 0
        assert np.all(X_norm == 0)
