//! Web scrapers for rugby match data

pub mod lassen;
pub mod sarugby;
pub mod wikipedia;

use crate::{DataSource, MatchRecord, Result};

/// Trait for all data scrapers
pub trait Scraper {
    /// The data source this scraper fetches from
    fn source(&self) -> DataSource;

    /// Fetch all available match records
    fn fetch_all(&self) -> Result<Vec<MatchRecord>>;

    /// Fetch matches for a specific season year
    fn fetch_season(&self, year: u16) -> Result<Vec<MatchRecord>>;
}

/// Retry a scraper operation with exponential backoff
pub fn with_retry<T, F>(mut operation: F, max_attempts: u32) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let mut last_error = None;
    for attempt in 0..max_attempts {
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) => {
                log::warn!("Attempt {} failed: {}", attempt + 1, e);
                last_error = Some(e);
                if attempt < max_attempts - 1 {
                    let delay = std::time::Duration::from_millis(100 * 2u64.pow(attempt));
                    std::thread::sleep(delay);
                }
            }
        }
    }
    Err(last_error.unwrap())
}

/// Merge records from multiple sources, preferring Wikipedia for scores
pub fn merge_sources(sources: Vec<Vec<MatchRecord>>) -> Vec<MatchRecord> {
    use std::collections::HashMap;

    // Key: (date, home_team, away_team)
    let mut merged: HashMap<(chrono::NaiveDate, i64, i64), MatchRecord> = HashMap::new();

    for records in sources {
        for record in records {
            let key = (record.date, record.home_team.0, record.away_team.0);

            match merged.get_mut(&key) {
                Some(existing) => {
                    // Merge: prefer Wikipedia for scores, fill in missing fields
                    if record.source == DataSource::Wikipedia {
                        existing.home_score = record.home_score;
                        existing.away_score = record.away_score;
                    }
                    if existing.venue.is_none() {
                        existing.venue = record.venue;
                    }
                    if existing.round.is_none() {
                        existing.round = record.round;
                    }
                    if existing.home_tries.is_none() {
                        existing.home_tries = record.home_tries;
                    }
                    if existing.away_tries.is_none() {
                        existing.away_tries = record.away_tries;
                    }
                }
                None => {
                    merged.insert(key, record);
                }
            }
        }
    }

    let mut result: Vec<_> = merged.into_values().collect();
    result.sort_by_key(|r| r.date);
    result
}
