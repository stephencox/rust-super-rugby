//! SQLite database management for rugby data

use crate::{Country, DataSource, MatchRecord, Result, RugbyError, Team, TeamId};
use chrono::NaiveDate;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Database connection and operations
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open or create database at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        let db = Database { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing)
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Database { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Initialize database schema
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
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
            "#,
        )?;
        Ok(())
    }

    // ==================== Team Operations ====================

    /// Get or create a team by name
    pub fn get_or_create_team(&self, name: &str, country: Country) -> Result<Team> {
        // First try to find existing team
        if let Some(team) = self.find_team_by_name(name)? {
            return Ok(team);
        }

        // Default timezone based on country
        let timezone_offset = match country {
            Country::NewZealand => 12,
            Country::Fiji => 12,
            Country::Australia => 10, // Default to eastern, Force is 8
            Country::Japan => 9,
            Country::SouthAfrica => 2,
            Country::Argentina => -3,
            Country::Samoa => 13,
            Country::Tonga => 13,
        };

        // Create new team
        self.conn.execute(
            "INSERT INTO teams (name, country, aliases, timezone_offset) VALUES (?1, ?2, '[]', ?3)",
            params![name, country.code(), timezone_offset],
        )?;

        let id = TeamId(self.conn.last_insert_rowid());
        Ok(Team {
            id,
            name: name.to_string(),
            country,
            aliases: vec![],
            timezone_offset,
        })
    }

    /// Find a team by name or alias
    pub fn find_team_by_name(&self, name: &str) -> Result<Option<Team>> {
        let name_lower = name.to_lowercase();

        // Check exact name match first
        let team: Option<Team> = self
            .conn
            .query_row(
                "SELECT id, name, country, aliases, COALESCE(timezone_offset, 0) FROM teams WHERE LOWER(name) = ?1",
                params![&name_lower],
                |row| {
                    let id = TeamId(row.get(0)?);
                    let name: String = row.get(1)?;
                    let country_code: String = row.get(2)?;
                    let aliases_json: String = row.get(3)?;
                    let timezone_offset: i32 = row.get(4)?;
                    let country = Country::from_code(&country_code).unwrap_or(Country::NewZealand);
                    let aliases: Vec<String> =
                        serde_json::from_str(&aliases_json).unwrap_or_default();
                    Ok(Team {
                        id,
                        name,
                        country,
                        aliases,
                        timezone_offset,
                    })
                },
            )
            .optional()?;

        if team.is_some() {
            return Ok(team);
        }

        // Check aliases
        let teams = self.get_all_teams()?;
        for team in teams {
            if team.matches_name(name) {
                return Ok(Some(team));
            }
        }

        Ok(None)
    }

    /// Get team by ID
    pub fn get_team(&self, id: TeamId) -> Result<Team> {
        self.conn
            .query_row(
                "SELECT id, name, country, aliases, COALESCE(timezone_offset, 0) FROM teams WHERE id = ?1",
                params![id.0],
                |row| {
                    let id = TeamId(row.get(0)?);
                    let name: String = row.get(1)?;
                    let country_code: String = row.get(2)?;
                    let aliases_json: String = row.get(3)?;
                    let timezone_offset: i32 = row.get(4)?;
                    let country = Country::from_code(&country_code).unwrap_or(Country::NewZealand);
                    let aliases: Vec<String> =
                        serde_json::from_str(&aliases_json).unwrap_or_default();
                    Ok(Team {
                        id,
                        name,
                        country,
                        aliases,
                        timezone_offset,
                    })
                },
            )
            .map_err(|_| RugbyError::TeamNotFound(id))
    }

    /// Get all teams
    pub fn get_all_teams(&self) -> Result<Vec<Team>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, country, aliases, COALESCE(timezone_offset, 0) FROM teams ORDER BY name")?;

        let teams = stmt
            .query_map([], |row| {
                let id = TeamId(row.get(0)?);
                let name: String = row.get(1)?;
                let country_code: String = row.get(2)?;
                let aliases_json: String = row.get(3)?;
                let timezone_offset: i32 = row.get(4)?;
                let country = Country::from_code(&country_code).unwrap_or(Country::NewZealand);
                let aliases: Vec<String> = serde_json::from_str(&aliases_json).unwrap_or_default();
                Ok(Team {
                    id,
                    name,
                    country,
                    aliases,
                    timezone_offset,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(teams)
    }

    /// Add an alias for a team
    pub fn add_team_alias(&self, team_id: TeamId, alias: &str) -> Result<()> {
        let team = self.get_team(team_id)?;
        let mut aliases = team.aliases;
        if !aliases
            .iter()
            .any(|a| a.to_lowercase() == alias.to_lowercase())
        {
            aliases.push(alias.to_string());
            let aliases_json =
                serde_json::to_string(&aliases).map_err(|e| RugbyError::Parse(e.to_string()))?;
            self.conn.execute(
                "UPDATE teams SET aliases = ?1 WHERE id = ?2",
                params![aliases_json, team_id.0],
            )?;
        }
        Ok(())
    }

    // ==================== Match Operations ====================

    /// Insert or update a match record
    pub fn upsert_match(&self, record: &MatchRecord) -> Result<()> {
        let source_str = format!("{:?}", record.source);
        self.conn.execute(
            r#"
            INSERT INTO matches (date, home_team_id, away_team_id, home_score, away_score,
                                venue, round, home_tries, away_tries, source)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(date, home_team_id, away_team_id) DO UPDATE SET
                home_score = excluded.home_score,
                away_score = excluded.away_score,
                venue = COALESCE(excluded.venue, venue),
                round = COALESCE(excluded.round, round),
                home_tries = COALESCE(excluded.home_tries, home_tries),
                away_tries = COALESCE(excluded.away_tries, away_tries)
            "#,
            params![
                record.date.format("%Y-%m-%d").to_string(),
                record.home_team.0,
                record.away_team.0,
                record.home_score,
                record.away_score,
                record.venue,
                record.round,
                record.home_tries,
                record.away_tries,
                source_str,
            ],
        )?;
        Ok(())
    }

    /// Insert multiple match records
    pub fn upsert_matches(&self, records: &[MatchRecord]) -> Result<usize> {
        let mut count = 0;
        for record in records {
            self.upsert_match(record)?;
            count += 1;
        }
        Ok(count)
    }

    /// Get all matches
    pub fn get_all_matches(&self) -> Result<Vec<MatchRecord>> {
        self.get_matches_query("SELECT * FROM matches ORDER BY date")
    }

    /// Get matches for a team
    pub fn get_team_matches(&self, team_id: TeamId) -> Result<Vec<MatchRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, home_team_id, away_team_id, home_score, away_score,
                    venue, round, home_tries, away_tries, source
             FROM matches
             WHERE home_team_id = ?1 OR away_team_id = ?1
             ORDER BY date",
        )?;

        let matches = stmt
            .query_map(params![team_id.0], Self::row_to_match)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(matches)
    }

    /// Get recent matches for a team (up to limit)
    pub fn get_recent_team_matches(
        &self,
        team_id: TeamId,
        limit: usize,
    ) -> Result<Vec<MatchRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, home_team_id, away_team_id, home_score, away_score,
                    venue, round, home_tries, away_tries, source
             FROM matches
             WHERE home_team_id = ?1 OR away_team_id = ?1
             ORDER BY date DESC
             LIMIT ?2",
        )?;

        let matches = stmt
            .query_map(params![team_id.0, limit], Self::row_to_match)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Reverse to get chronological order
        let mut matches = matches;
        matches.reverse();
        Ok(matches)
    }

    /// Get matches before a given date
    pub fn get_matches_before(&self, date: NaiveDate) -> Result<Vec<MatchRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, home_team_id, away_team_id, home_score, away_score,
                    venue, round, home_tries, away_tries, source
             FROM matches
             WHERE date < ?1
             ORDER BY date",
        )?;

        let matches = stmt
            .query_map(
                params![date.format("%Y-%m-%d").to_string()],
                Self::row_to_match,
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(matches)
    }

    /// Get matches in date range
    pub fn get_matches_in_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<MatchRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, home_team_id, away_team_id, home_score, away_score,
                    venue, round, home_tries, away_tries, source
             FROM matches
             WHERE date >= ?1 AND date <= ?2
             ORDER BY date",
        )?;

        let matches = stmt
            .query_map(
                params![
                    start.format("%Y-%m-%d").to_string(),
                    end.format("%Y-%m-%d").to_string()
                ],
                Self::row_to_match,
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(matches)
    }

    fn get_matches_query(&self, query: &str) -> Result<Vec<MatchRecord>> {
        let mut stmt = self.conn.prepare(query)?;
        let matches = stmt
            .query_map([], |row| {
                // Skip the id column (index 0)
                Self::row_to_match_with_offset(row, 1)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(matches)
    }

    fn row_to_match(row: &rusqlite::Row) -> rusqlite::Result<MatchRecord> {
        Self::row_to_match_with_offset(row, 0)
    }

    fn row_to_match_with_offset(
        row: &rusqlite::Row,
        offset: usize,
    ) -> rusqlite::Result<MatchRecord> {
        let date_str: String = row.get(offset)?;
        let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
            .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());

        let source_str: String = row.get(offset + 9)?;
        let source = match source_str.as_str() {
            "Wikipedia" => DataSource::Wikipedia,
            "SaRugby" => DataSource::SaRugby,
            "Lassen" => DataSource::Lassen,
            _ => DataSource::Wikipedia,
        };

        Ok(MatchRecord {
            date,
            home_team: TeamId(row.get(offset + 1)?),
            away_team: TeamId(row.get(offset + 2)?),
            home_score: row.get(offset + 3)?,
            away_score: row.get(offset + 4)?,
            venue: row.get(offset + 5)?,
            round: row.get(offset + 6)?,
            home_tries: row.get(offset + 7)?,
            away_tries: row.get(offset + 8)?,
            source,
        })
    }

    // ==================== Statistics ====================

    /// Get database statistics
    pub fn get_stats(&self) -> Result<DatabaseStats> {
        let team_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM teams", [], |row| row.get(0))?;

        let match_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM matches", [], |row| row.get(0))?;

        let min_date: Option<String> = self
            .conn
            .query_row("SELECT MIN(date) FROM matches", [], |row| row.get(0))
            .optional()?
            .flatten();

        let max_date: Option<String> = self
            .conn
            .query_row("SELECT MAX(date) FROM matches", [], |row| row.get(0))
            .optional()?
            .flatten();

        Ok(DatabaseStats {
            team_count: team_count as usize,
            match_count: match_count as usize,
            earliest_match: min_date.and_then(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok()),
            latest_match: max_date.and_then(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok()),
        })
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub team_count: usize,
    pub match_count: usize,
    pub earliest_match: Option<NaiveDate>,
    pub latest_match: Option<NaiveDate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_database() {
        let db = Database::in_memory().unwrap();
        let stats = db.get_stats().unwrap();
        assert_eq!(stats.team_count, 0);
        assert_eq!(stats.match_count, 0);
    }

    #[test]
    fn test_create_team() {
        let db = Database::in_memory().unwrap();
        let team = db.get_or_create_team("Blues", Country::NewZealand).unwrap();
        assert_eq!(team.name, "Blues");
        assert_eq!(team.country, Country::NewZealand);

        // Getting again should return same team
        let team2 = db.get_or_create_team("Blues", Country::NewZealand).unwrap();
        assert_eq!(team.id.0, team2.id.0);
    }

    #[test]
    fn test_insert_match() {
        let db = Database::in_memory().unwrap();
        let blues = db.get_or_create_team("Blues", Country::NewZealand).unwrap();
        let chiefs = db
            .get_or_create_team("Chiefs", Country::NewZealand)
            .unwrap();

        let record = MatchRecord {
            date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            home_team: blues.id,
            away_team: chiefs.id,
            home_score: 28,
            away_score: 24,
            venue: Some("Eden Park".to_string()),
            round: Some(1),
            home_tries: Some(4),
            away_tries: Some(3),
            source: DataSource::Wikipedia,
        };

        db.upsert_match(&record).unwrap();

        let stats = db.get_stats().unwrap();
        assert_eq!(stats.match_count, 1);
    }
}
