"""Tests for rugby.data module."""

import pytest
from datetime import datetime, date
from rugby.data import Database, Match, Team


class TestMatch:
    def test_home_win(self):
        m = Match(1, datetime(2024, 1, 1), 1, 2, 30, 20, None, None)
        assert m.home_win is True

    def test_away_win(self):
        m = Match(1, datetime(2024, 1, 1), 1, 2, 20, 30, None, None)
        assert m.home_win is False

    def test_draw(self):
        m = Match(1, datetime(2024, 1, 1), 1, 2, 25, 25, None, None)
        assert m.home_win is False

    def test_margin(self):
        m = Match(1, datetime(2024, 1, 1), 1, 2, 30, 20, None, None)
        assert m.margin == 10

    def test_margin_away_win(self):
        m = Match(1, datetime(2024, 1, 1), 1, 2, 15, 40, None, None)
        assert m.margin == 25


class TestDatabase:
    def test_init_schema(self, tmp_db):
        stats = tmp_db.get_stats()
        assert stats['match_count'] == 0
        assert stats['team_count'] == 0

    def test_get_or_create_team(self, tmp_db):
        tid = tmp_db.get_or_create_team("Blues", "New Zealand")
        assert tid > 0
        # Same name returns same ID
        tid2 = tmp_db.get_or_create_team("Blues", "New Zealand")
        assert tid2 == tid

    def test_find_team_by_name(self, tmp_db, sample_teams):
        team = tmp_db.find_team_by_name("blues")
        assert team is not None
        assert team.name == "Blues"

    def test_find_team_by_name_partial(self, tmp_db, sample_teams):
        team = tmp_db.find_team_by_name("Highlander")
        assert team is not None
        assert team.name == "Highlanders"

    def test_find_team_not_found(self, tmp_db, sample_teams):
        team = tmp_db.find_team_by_name("Nonexistent FC")
        assert team is None

    def test_add_team_alias(self, tmp_db, sample_teams):
        tid = sample_teams["Blues"]
        tmp_db.add_team_alias(tid, "Auckland Blues")
        team = tmp_db.find_team_by_name("Auckland Blues")
        assert team is not None
        assert team.id == tid

    def test_upsert_match(self, tmp_db, sample_teams):
        record = {
            "date": "2024-03-15",
            "home_team_id": sample_teams["Blues"],
            "away_team_id": sample_teams["Chiefs"],
            "home_score": 25,
            "away_score": 20,
            "source": "test",
        }
        result = tmp_db.upsert_match(record)
        assert result is True
        matches = tmp_db.get_matches()
        assert len(matches) == 1
        assert matches[0].home_score == 25

    def test_upsert_match_update(self, tmp_db, sample_teams):
        record = {
            "date": "2024-03-15",
            "home_team_id": sample_teams["Blues"],
            "away_team_id": sample_teams["Chiefs"],
            "home_score": 25,
            "away_score": 20,
            "source": "test",
        }
        tmp_db.upsert_match(record)
        # Update score
        record["home_score"] = 30
        tmp_db.upsert_match(record)
        matches = tmp_db.get_matches()
        assert len(matches) == 1
        assert matches[0].home_score == 30

    def test_uniqueness(self, tmp_db, sample_teams):
        """Matches unique on (date, home_team_id, away_team_id)."""
        r1 = {"date": "2024-03-15", "home_team_id": sample_teams["Blues"],
               "away_team_id": sample_teams["Chiefs"], "home_score": 25,
               "away_score": 20, "source": "test"}
        r2 = {"date": "2024-03-15", "home_team_id": sample_teams["Blues"],
               "away_team_id": sample_teams["Crusaders"], "home_score": 30,
               "away_score": 10, "source": "test"}
        tmp_db.upsert_match(r1)
        tmp_db.upsert_match(r2)
        matches = tmp_db.get_matches()
        assert len(matches) == 2

    def test_get_matches_before(self, tmp_db, sample_matches):
        matches = tmp_db.get_matches_before(date(2021, 1, 1))
        for m in matches:
            assert m.date.year == 2020

    def test_get_matches_in_range(self, tmp_db, sample_matches):
        matches = tmp_db.get_matches_in_range(date(2023, 1, 1), date(2024, 1, 1))
        for m in matches:
            assert m.date.year == 2023

    def test_get_recent_team_matches(self, tmp_db, sample_matches, sample_teams):
        matches = tmp_db.get_recent_team_matches(sample_teams["Blues"], limit=3)
        assert len(matches) == 3
        # Most recent first
        assert matches[0].date >= matches[1].date

    def test_get_teams(self, tmp_db, sample_teams):
        teams = tmp_db.get_teams()
        assert len(teams) == len(sample_teams)
        for team in teams.values():
            assert isinstance(team, Team)

    def test_timezone_offset_preserved(self, tmp_db, sample_teams):
        teams = tmp_db.get_teams()
        blues = [t for t in teams.values() if t.name == "Blues"][0]
        assert blues.timezone_offset == 12
        stormers = [t for t in teams.values() if t.name == "Stormers"][0]
        assert stormers.timezone_offset == 2

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx.db"
        with Database(db_path) as db:
            db.init_schema()
            stats = db.get_stats()
            assert stats['match_count'] == 0
