"""Data loading from SQLite database."""

import sqlite3
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "rugby.db"


@dataclass
class Match:
    """A rugby match."""
    id: int
    date: datetime
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    venue: Optional[str]
    round: Optional[int]

    @property
    def home_win(self) -> bool:
        return self.home_score > self.away_score

    @property
    def margin(self) -> int:
        return abs(self.home_score - self.away_score)


@dataclass
class Team:
    """A rugby team."""
    id: int
    name: str
    country: Optional[str]


class Database:
    """Database connection and queries."""

    def __init__(self, path: Path = DEFAULT_DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(path)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_matches(self) -> List[Match]:
        """Load all matches ordered by date."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, date, home_team_id, away_team_id, home_score, away_score, venue, round
            FROM matches
            ORDER BY date
        """)

        matches = []
        for row in cursor.fetchall():
            matches.append(Match(
                id=row[0],
                date=datetime.strptime(row[1], '%Y-%m-%d'),
                home_team_id=row[2],
                away_team_id=row[3],
                home_score=row[4],
                away_score=row[5],
                venue=row[6],
                round=row[7],
            ))
        return matches

    def get_teams(self) -> Dict[int, Team]:
        """Load all teams as a dict by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, country FROM teams")

        teams = {}
        for row in cursor.fetchall():
            teams[row[0]] = Team(id=row[0], name=row[1], country=row[2])
        return teams

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        match_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM teams")
        team_count = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(date), MAX(date) FROM matches")
        date_range = cursor.fetchone()

        return {
            'match_count': match_count,
            'team_count': team_count,
            'date_range': date_range,
        }


# Convenience functions
def load_matches(db_path: Path = DEFAULT_DB_PATH) -> List[Match]:
    """Load all matches from database."""
    with Database(db_path) as db:
        return db.get_matches()


def load_teams(db_path: Path = DEFAULT_DB_PATH) -> Dict[int, Team]:
    """Load all teams from database."""
    with Database(db_path) as db:
        return db.get_teams()
