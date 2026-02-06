"""Data loading and writing for SQLite database."""

import json
import sqlite3
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field
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
    home_tries: Optional[int] = None
    away_tries: Optional[int] = None
    source: Optional[str] = None

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
    aliases: List[str] = field(default_factory=list)
    timezone_offset: int = 0


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

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def init_schema(self):
        """Create tables and indexes if they don't exist."""
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                country TEXT NOT NULL,
                aliases TEXT DEFAULT '[]',
                timezone_offset INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                home_team_id INTEGER NOT NULL REFERENCES teams(id),
                away_team_id INTEGER NOT NULL REFERENCES teams(id),
                home_score INTEGER NOT NULL,
                away_score INTEGER NOT NULL,
                venue TEXT,
                round INTEGER,
                home_tries INTEGER,
                away_tries INTEGER,
                source TEXT NOT NULL,
                UNIQUE(date, home_team_id, away_team_id)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                home_team_id INTEGER NOT NULL REFERENCES teams(id),
                away_team_id INTEGER NOT NULL REFERENCES teams(id),
                home_win_prob REAL NOT NULL,
                predicted_home_score REAL,
                predicted_away_score REAL,
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                model_version TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
            CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_matches(self) -> List[Match]:
        """Load all matches ordered by date."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, date, home_team_id, away_team_id, home_score, away_score,
                   venue, round, home_tries, away_tries, source
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
                home_tries=row[8],
                away_tries=row[9],
                source=row[10],
            ))
        return matches

    def get_teams(self) -> Dict[int, Team]:
        """Load all teams as a dict by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, country, aliases, timezone_offset FROM teams")

        teams = {}
        for row in cursor.fetchall():
            aliases = []
            if row[3]:
                try:
                    aliases = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    aliases = []
            teams[row[0]] = Team(
                id=row[0],
                name=row[1],
                country=row[2],
                aliases=aliases,
                timezone_offset=row[4] if row[4] is not None else 0,
            )
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

    def find_team_by_name(self, name: str) -> Optional[Team]:
        """Find a team by name or alias (case-insensitive)."""
        teams = self.get_teams()
        name_lower = name.lower()

        # Exact name match
        for team in teams.values():
            if team.name.lower() == name_lower:
                return team

        # Alias match
        for team in teams.values():
            for alias in team.aliases:
                if alias.lower() == name_lower:
                    return team

        # Partial match fallback
        for team in teams.values():
            if name_lower in team.name.lower() or team.name.lower() in name_lower:
                return team

        return None

    def get_recent_team_matches(self, team_id: int, limit: int = 10) -> List[Match]:
        """Get the most recent matches for a team."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, date, home_team_id, away_team_id, home_score, away_score,
                   venue, round, home_tries, away_tries, source
            FROM matches
            WHERE home_team_id = ? OR away_team_id = ?
            ORDER BY date DESC
            LIMIT ?
        """, (team_id, team_id, limit))

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
                home_tries=row[8],
                away_tries=row[9],
                source=row[10],
            ))
        return matches

    def get_matches_before(self, before_date: date) -> List[Match]:
        """Get all matches before a given date."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, date, home_team_id, away_team_id, home_score, away_score,
                   venue, round, home_tries, away_tries, source
            FROM matches
            WHERE date < ?
            ORDER BY date
        """, (before_date.isoformat(),))

        return [Match(
            id=row[0],
            date=datetime.strptime(row[1], '%Y-%m-%d'),
            home_team_id=row[2], away_team_id=row[3],
            home_score=row[4], away_score=row[5],
            venue=row[6], round=row[7],
            home_tries=row[8], away_tries=row[9], source=row[10],
        ) for row in cursor.fetchall()]

    def get_matches_in_range(self, start: date, end: date) -> List[Match]:
        """Get matches within a date range [start, end)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, date, home_team_id, away_team_id, home_score, away_score,
                   venue, round, home_tries, away_tries, source
            FROM matches
            WHERE date >= ? AND date < ?
            ORDER BY date
        """, (start.isoformat(), end.isoformat()))

        return [Match(
            id=row[0],
            date=datetime.strptime(row[1], '%Y-%m-%d'),
            home_team_id=row[2], away_team_id=row[3],
            home_score=row[4], away_score=row[5],
            venue=row[6], round=row[7],
            home_tries=row[8], away_tries=row[9], source=row[10],
        ) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def get_or_create_team(self, name: str, country: str, timezone_offset: int = 0) -> int:
        """Find a team by name/alias or create it. Returns team ID."""
        existing = self.find_team_by_name(name)
        if existing:
            return existing.id

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO teams (name, country, aliases, timezone_offset) VALUES (?, ?, '[]', ?)",
            (name, country, timezone_offset),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_team_alias(self, team_id: int, alias: str):
        """Add an alias to a team's alias list."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT aliases FROM teams WHERE id = ?", (team_id,))
        row = cursor.fetchone()
        if not row:
            return

        try:
            aliases = json.loads(row[0]) if row[0] else []
        except (json.JSONDecodeError, TypeError):
            aliases = []

        if alias not in aliases:
            aliases.append(alias)
            cursor.execute(
                "UPDATE teams SET aliases = ? WHERE id = ?",
                (json.dumps(aliases), team_id),
            )
            self.conn.commit()

    def upsert_match(self, record: dict) -> bool:
        """Insert or update a match record.

        record keys: date, home_team_id, away_team_id, home_score, away_score,
                     venue, round, home_tries, away_tries, source

        Returns True if a new row was inserted, False if updated.
        """
        cursor = self.conn.cursor()
        match_date = record["date"]
        if isinstance(match_date, (datetime, date)):
            match_date = match_date.isoformat()

        cursor.execute("""
            INSERT INTO matches (date, home_team_id, away_team_id, home_score, away_score,
                                 venue, round, home_tries, away_tries, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, home_team_id, away_team_id) DO UPDATE SET
                home_score = excluded.home_score,
                away_score = excluded.away_score,
                venue = COALESCE(excluded.venue, venue),
                round = COALESCE(excluded.round, round),
                home_tries = COALESCE(excluded.home_tries, home_tries),
                away_tries = COALESCE(excluded.away_tries, away_tries),
                source = excluded.source
        """, (
            match_date,
            record["home_team_id"],
            record["away_team_id"],
            record["home_score"],
            record["away_score"],
            record.get("venue"),
            record.get("round"),
            record.get("home_tries"),
            record.get("away_tries"),
            record.get("source", "Python"),
        ))
        self.conn.commit()
        return cursor.rowcount > 0

    def upsert_matches(self, records: List[dict]) -> int:
        """Batch upsert match records. Returns number of rows affected."""
        count = 0
        for record in records:
            if self.upsert_match(record):
                count += 1
        return count


# Convenience functions
def load_matches(db_path: Path = DEFAULT_DB_PATH) -> List[Match]:
    """Load all matches from database."""
    with Database(db_path) as db:
        return db.get_matches()


def load_teams(db_path: Path = DEFAULT_DB_PATH) -> Dict[int, Team]:
    """Load all teams from database."""
    with Database(db_path) as db:
        return db.get_teams()
