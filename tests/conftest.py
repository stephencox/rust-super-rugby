"""Shared test fixtures."""

import pytest
import sqlite3
from datetime import datetime
from pathlib import Path
from rugby.data import Database, Match, Team


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database with schema initialized."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.init_schema()
    yield db
    db.close()


@pytest.fixture
def sample_teams(tmp_db):
    """Insert sample teams and return their IDs."""
    teams = {}
    for name, country, tz in [
        ("Blues", "New Zealand", 12),
        ("Chiefs", "New Zealand", 12),
        ("Crusaders", "New Zealand", 12),
        ("Hurricanes", "New Zealand", 12),
        ("Highlanders", "New Zealand", 12),
        ("Stormers", "South Africa", 2),
    ]:
        tid = tmp_db.get_or_create_team(name, country, tz)
        teams[name] = tid
    return teams


def make_match(id, date_str, home_id, away_id, home_score, away_score, venue=None, round_num=None):
    """Helper to create a Match object."""
    return Match(
        id=id,
        date=datetime.strptime(date_str, "%Y-%m-%d"),
        home_team_id=home_id,
        away_team_id=away_id,
        home_score=home_score,
        away_score=away_score,
        venue=venue,
        round=round_num,
    )


@pytest.fixture
def sample_matches(tmp_db, sample_teams):
    """Insert a set of sample matches spanning 2020-2024."""
    t = sample_teams
    records = [
        # 2020 season - enough history for Blues and Chiefs
        {"date": "2020-02-15", "home_team_id": t["Blues"], "away_team_id": t["Chiefs"],
         "home_score": 25, "away_score": 20, "source": "test"},
        {"date": "2020-03-01", "home_team_id": t["Chiefs"], "away_team_id": t["Blues"],
         "home_score": 30, "away_score": 15, "source": "test"},
        {"date": "2020-03-15", "home_team_id": t["Blues"], "away_team_id": t["Crusaders"],
         "home_score": 18, "away_score": 22, "source": "test"},
        {"date": "2020-04-01", "home_team_id": t["Crusaders"], "away_team_id": t["Chiefs"],
         "home_score": 35, "away_score": 28, "source": "test"},
        {"date": "2020-04-15", "home_team_id": t["Blues"], "away_team_id": t["Hurricanes"],
         "home_score": 27, "away_score": 20, "source": "test"},
        {"date": "2020-05-01", "home_team_id": t["Hurricanes"], "away_team_id": t["Chiefs"],
         "home_score": 24, "away_score": 31, "source": "test"},
        # 2021
        {"date": "2021-02-20", "home_team_id": t["Blues"], "away_team_id": t["Chiefs"],
         "home_score": 22, "away_score": 18, "source": "test"},
        {"date": "2021-03-05", "home_team_id": t["Chiefs"], "away_team_id": t["Crusaders"],
         "home_score": 19, "away_score": 24, "source": "test"},
        {"date": "2021-03-20", "home_team_id": t["Crusaders"], "away_team_id": t["Blues"],
         "home_score": 31, "away_score": 28, "source": "test"},
        {"date": "2021-04-10", "home_team_id": t["Blues"], "away_team_id": t["Stormers"],
         "home_score": 33, "away_score": 14, "source": "test"},
        # 2023 (validation period)
        {"date": "2023-03-01", "home_team_id": t["Blues"], "away_team_id": t["Chiefs"],
         "home_score": 28, "away_score": 25, "source": "test"},
        {"date": "2023-04-01", "home_team_id": t["Chiefs"], "away_team_id": t["Crusaders"],
         "home_score": 22, "away_score": 20, "source": "test"},
        # 2024
        {"date": "2024-03-01", "home_team_id": t["Chiefs"], "away_team_id": t["Blues"],
         "home_score": 35, "away_score": 30, "source": "test"},
    ]
    for r in records:
        tmp_db.upsert_match(r)
    return tmp_db.get_matches()
