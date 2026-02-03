//! Wikipedia scraper for Six Nations match data
//!
//! Parses Wikipedia season pages from 2000 to present.
//! The Six Nations format started in 2000 when Italy joined the Five Nations.

use crate::{Country, DataSource, MatchRecord, Result, RugbyError, TeamId};
use chrono::NaiveDate;
use regex::Regex;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::wikipedia::{RawFixture, RawMatch, TeamInfo};

/// Scraper for Wikipedia Six Nations pages
pub struct SixNationsScraper {
    client: reqwest::blocking::Client,
    team_aliases: HashMap<String, (String, Country)>,
    /// Optional cache directory for offline HTML files
    cache_dir: Option<PathBuf>,
    /// If true, only use cache (no network requests)
    offline_only: bool,
}

impl Default for SixNationsScraper {
    fn default() -> Self {
        Self::new()
    }
}

impl SixNationsScraper {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::builder()
            .user_agent("rugby-predictor/0.1")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        SixNationsScraper {
            client,
            team_aliases: Self::default_team_aliases(),
            cache_dir: None,
            offline_only: false,
        }
    }

    /// Create scraper with a cache directory
    pub fn with_cache<P: AsRef<Path>>(mut self, cache_dir: P) -> Self {
        self.cache_dir = Some(cache_dir.as_ref().to_path_buf());
        self
    }

    /// Set offline-only mode (no network requests, cache must exist)
    pub fn offline_only(mut self, offline: bool) -> Self {
        self.offline_only = offline;
        self
    }

    /// Get the cache file path for a URL
    fn cache_path(&self, url: &str) -> Option<PathBuf> {
        self.cache_dir.as_ref().map(|dir| {
            let filename = url
                .replace("https://", "")
                .replace("http://", "")
                .replace('/', "_")
                .replace('?', "_")
                + ".html";
            dir.join(filename)
        })
    }

    /// Load HTML from cache if available
    fn load_from_cache(&self, url: &str) -> Option<String> {
        let path = self.cache_path(url)?;
        if path.exists() {
            log::debug!("Loading from cache: {}", path.display());
            std::fs::read_to_string(&path).ok()
        } else {
            None
        }
    }

    /// Save HTML to cache
    fn save_to_cache(&self, url: &str, html: &str) -> Result<()> {
        if let Some(path) = self.cache_path(url) {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, html)?;
            log::debug!("Saved to cache: {}", path.display());
        }
        Ok(())
    }

    /// Build the Six Nations team alias mapping
    fn default_team_aliases() -> HashMap<String, (String, Country)> {
        let mut aliases = HashMap::new();

        // England
        for name in ["England", "England XV", "ENG"] {
            aliases.insert(
                name.to_lowercase(),
                ("England".to_string(), Country::England),
            );
        }

        // France
        for name in ["France", "France XV", "FRA"] {
            aliases.insert(
                name.to_lowercase(),
                ("France".to_string(), Country::France),
            );
        }

        // Ireland
        for name in ["Ireland", "Ireland XV", "IRE"] {
            aliases.insert(
                name.to_lowercase(),
                ("Ireland".to_string(), Country::Ireland),
            );
        }

        // Wales
        for name in ["Wales", "Wales XV", "WAL"] {
            aliases.insert(
                name.to_lowercase(),
                ("Wales".to_string(), Country::Wales),
            );
        }

        // Scotland
        for name in ["Scotland", "Scotland XV", "SCO"] {
            aliases.insert(
                name.to_lowercase(),
                ("Scotland".to_string(), Country::Scotland),
            );
        }

        // Italy
        for name in ["Italy", "Italy XV", "ITA"] {
            aliases.insert(
                name.to_lowercase(),
                ("Italy".to_string(), Country::Italy),
            );
        }

        aliases
    }

    /// Get all season URLs from 2000 to current year
    /// Six Nations format: "YYYY_Six_Nations_Championship"
    pub fn get_season_urls(&self) -> Vec<(u16, String)> {
        let current_year = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse()
            .unwrap_or(2026);
        let mut urls = Vec::new();

        // Six Nations started in 2000 (Italy joined the Five Nations)
        for year in 2000..=current_year {
            let base = "https://en.wikipedia.org/wiki/";
            urls.push((year, format!("{}{}_Six_Nations_Championship", base, year)));
        }

        urls
    }

    /// Fetch and parse a single season
    pub fn fetch_season(&self, year: u16) -> Result<Vec<RawMatch>> {
        let url = format!(
            "https://en.wikipedia.org/wiki/{}_Six_Nations_Championship",
            year
        );

        log::info!("Fetching {} Six Nations from {}", year, url);
        self.fetch_page(&url)
    }

    /// Fetch all seasons
    pub fn fetch_all(&self) -> Result<Vec<RawMatch>> {
        let current_year = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse()
            .unwrap_or(2026);
        let mut all_matches = Vec::new();

        for year in 2000..=current_year {
            log::info!("Fetching {} Six Nations...", year);
            match self.fetch_season(year) {
                Ok(matches) => {
                    log::info!("  Found {} matches", matches.len());
                    all_matches.extend(matches);
                }
                Err(e) => {
                    log::warn!("Failed to fetch {} season: {}", year, e);
                }
            }
        }

        Ok(all_matches)
    }

    /// Fetch upcoming fixtures for a season
    pub fn fetch_fixtures(&self, year: u16) -> Result<Vec<RawFixture>> {
        let url = format!(
            "https://en.wikipedia.org/wiki/{}_Six_Nations_Championship",
            year
        );

        log::info!("Fetching {} fixtures from {}", year, url);

        let html = if let Some(cached) = self.load_from_cache(&url) {
            cached
        } else if self.offline_only {
            return Err(RugbyError::Scraper {
                data_source: DataSource::Wikipedia,
                message: format!("No cached data for {} (offline mode)", url),
            });
        } else {
            let response = self.client.get(&url).send()?;
            if !response.status().is_success() {
                return Err(RugbyError::Scraper {
                    data_source: DataSource::Wikipedia,
                    message: format!("HTTP {}: {}", response.status(), url),
                });
            }
            let html = response.text()?;
            let _ = self.save_to_cache(&url, &html);
            html
        };

        self.parse_fixtures(&html)
    }

    /// Parse fixtures (upcoming matches without scores)
    fn parse_fixtures(&self, html: &str) -> Result<Vec<RawFixture>> {
        let document = Html::parse_document(html);
        let mut fixtures = Vec::new();

        // Select match event divs using Schema.org SportsEvent format
        let event_selector =
            Selector::parse("div[itemtype='http://schema.org/SportsEvent']").unwrap();
        let team_selector = Selector::parse("span.fn.org").unwrap();
        let td_selector = Selector::parse("td").unwrap();
        let location_selector = Selector::parse("span.location").unwrap();

        let score_pattern = Regex::new(r"(\d{1,3})\s*[–-]\s*(\d{1,3})").unwrap();

        for event in document.select(&event_selector) {
            let event_text: String = event.text().collect();
            let date = self.extract_date_from_text(&event_text);

            let teams: Vec<_> = event.select(&team_selector).collect();
            if teams.len() < 2 {
                continue;
            }

            let home_text: String = teams[0].text().collect();
            let away_text: String = teams[1].text().collect();

            let home_team = self.normalize_team_name(home_text.trim());
            let away_team = self.normalize_team_name(away_text.trim());

            // Check if there's a score
            let mut has_score = false;
            for td in event.select(&td_selector) {
                let td_text: String = td.text().collect();
                if score_pattern.is_match(&td_text) {
                    has_score = true;
                    break;
                }
            }

            if has_score {
                continue;
            }

            let venue = event
                .select(&location_selector)
                .next()
                .map(|loc| loc.text().collect::<String>().trim().to_string());

            if let (Some(date), Some(home), Some(away)) = (date, home_team, away_team) {
                fixtures.push(RawFixture {
                    date,
                    home_team: home,
                    away_team: away,
                    venue,
                    round: None,
                });
            }
        }

        // Deduplicate and assign rounds
        fixtures.sort_by(|a, b| (&a.date, &a.home_team.name).cmp(&(&b.date, &b.home_team.name)));
        fixtures.dedup_by(|a, b| {
            a.date == b.date && a.home_team.name == b.home_team.name && a.away_team.name == b.away_team.name
        });

        // Infer round numbers (Six Nations has 5 rounds, 3 matches per round)
        if !fixtures.is_empty() {
            let mut dates: Vec<NaiveDate> = fixtures.iter().map(|f| f.date).collect();
            dates.sort();
            dates.dedup();

            let mut date_to_round: HashMap<NaiveDate, u8> = HashMap::new();
            let mut current_round = 1u8;
            let mut round_start_date = dates[0];

            for date in &dates {
                let days_since_round_start = (*date - round_start_date).num_days();
                // Six Nations rounds are typically 1-2 weeks apart
                if days_since_round_start > 5 {
                    current_round += 1;
                    round_start_date = *date;
                }
                date_to_round.insert(*date, current_round);
            }

            for fixture in &mut fixtures {
                fixture.round = date_to_round.get(&fixture.date).copied();
            }
        }

        log::info!("Found {} upcoming fixtures", fixtures.len());
        Ok(fixtures)
    }

    /// Fetch and parse a Wikipedia page
    fn fetch_page(&self, url: &str) -> Result<Vec<RawMatch>> {
        if let Some(html) = self.load_from_cache(url) {
            return self.parse_page(&html);
        }

        if self.offline_only {
            return Err(RugbyError::Scraper {
                data_source: DataSource::Wikipedia,
                message: format!("No cached data for {} (offline mode)", url),
            });
        }

        log::debug!("Fetching {}", url);

        let response = self.client.get(url).send()?;

        if !response.status().is_success() {
            return Err(RugbyError::Scraper {
                data_source: DataSource::Wikipedia,
                message: format!("HTTP {}: {}", response.status(), url),
            });
        }

        let html = response.text()?;

        if let Err(e) = self.save_to_cache(url, &html) {
            log::warn!("Failed to cache {}: {}", url, e);
        }

        self.parse_page(&html)
    }

    /// Parse HTML content for match data
    fn parse_page(&self, html: &str) -> Result<Vec<RawMatch>> {
        let document = Html::parse_document(html);
        let mut matches = Vec::new();

        // Try schema.org/SportsEvent format
        matches.extend(self.parse_sports_events(&document)?);

        // Try wikitable-based parsing
        matches.extend(self.parse_tables(&document)?);

        // Deduplicate
        matches.sort_by(|a, b| (&a.date, &a.home_team).cmp(&(&b.date, &b.home_team)));
        matches.dedup_by(|a, b| {
            a.date == b.date && a.home_team == b.home_team && a.away_team == b.away_team
        });

        Ok(matches)
    }

    /// Parse schema.org/SportsEvent format
    fn parse_sports_events(&self, document: &Html) -> Result<Vec<RawMatch>> {
        let mut matches = Vec::new();

        let event_selector =
            Selector::parse("div[itemtype='http://schema.org/SportsEvent']").unwrap();
        let team_selector = Selector::parse("span.fn.org").unwrap();
        let td_selector = Selector::parse("td").unwrap();
        let location_selector = Selector::parse("span.location").unwrap();

        let score_pattern = Regex::new(r"(\d{1,3})\s*[–-]\s*(\d{1,3})").unwrap();

        for event in document.select(&event_selector) {
            let event_text: String = event.text().collect();
            let date = self.extract_date_from_text(&event_text);

            let teams: Vec<_> = event.select(&team_selector).collect();
            if teams.len() < 2 {
                continue;
            }

            let home_text: String = teams[0].text().collect();
            let away_text: String = teams[1].text().collect();

            let home_team = self.normalize_team_name(home_text.trim());
            let away_team = self.normalize_team_name(away_text.trim());

            let mut home_score = None;
            let mut away_score = None;

            for td in event.select(&td_selector) {
                let td_text: String = td.text().collect();
                if let Some(caps) = score_pattern.captures(&td_text) {
                    home_score = caps.get(1).and_then(|m| m.as_str().parse().ok());
                    away_score = caps.get(2).and_then(|m| m.as_str().parse().ok());
                    break;
                }
            }

            let venue = event
                .select(&location_selector)
                .next()
                .map(|loc| loc.text().collect::<String>().trim().to_string())
                .filter(|s| !s.is_empty());

            if let (Some(date), Some(home), Some(away), Some(hs), Some(as_)) =
                (date, home_team, away_team, home_score, away_score)
            {
                matches.push(RawMatch {
                    date,
                    home_team: home,
                    away_team: away,
                    home_score: hs,
                    away_score: as_,
                    home_tries: None,
                    away_tries: None,
                    round: None,
                    venue,
                });
            }
        }

        log::debug!("parse_sports_events found {} matches", matches.len());
        Ok(matches)
    }

    /// Extract date from text
    fn extract_date_from_text(&self, text: &str) -> Option<NaiveDate> {
        let patterns = [
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        ];

        let month_map: HashMap<&str, u32> = [
            ("january", 1), ("february", 2), ("march", 3), ("april", 4),
            ("may", 5), ("june", 6), ("july", 7), ("august", 8),
            ("september", 9), ("october", 10), ("november", 11), ("december", 12),
        ].into_iter().collect();

        if let Ok(re) = Regex::new(patterns[0]) {
            if let Some(caps) = re.captures(text) {
                let day: u32 = caps.get(1)?.as_str().parse().ok()?;
                let month = month_map.get(caps.get(2)?.as_str().to_lowercase().as_str())?;
                let year: i32 = caps.get(3)?.as_str().parse().ok()?;
                return NaiveDate::from_ymd_opt(year, *month, day);
            }
        }

        if let Ok(re) = Regex::new(patterns[1]) {
            if let Some(caps) = re.captures(text) {
                let month = month_map.get(caps.get(1)?.as_str().to_lowercase().as_str())?;
                let day: u32 = caps.get(2)?.as_str().parse().ok()?;
                let year: i32 = caps.get(3)?.as_str().parse().ok()?;
                return NaiveDate::from_ymd_opt(year, *month, day);
            }
        }

        None
    }

    /// Parse match tables
    fn parse_tables(&self, document: &Html) -> Result<Vec<RawMatch>> {
        let mut matches = Vec::new();
        let table_selector = Selector::parse("table.wikitable").unwrap();
        let row_selector = Selector::parse("tr").unwrap();
        let cell_selector = Selector::parse("td, th").unwrap();

        for table in document.select(&table_selector) {
            let rows: Vec<_> = table.select(&row_selector).collect();

            for row in rows {
                let cells: Vec<_> = row.select(&cell_selector).collect();

                if cells.len() >= 5 {
                    if let Some(m) = self.try_parse_table_row(&cells) {
                        matches.push(m);
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Try to parse a table row as a match
    fn try_parse_table_row(&self, cells: &[scraper::ElementRef]) -> Option<RawMatch> {
        let cell_texts: Vec<String> = cells
            .iter()
            .map(|c| c.text().collect::<String>().trim().to_string())
            .collect();

        let date_idx = cell_texts.iter().position(|t| self.looks_like_date(t))?;

        let score_pattern = Regex::new(r"^(\d{1,3})\s*[-–]\s*(\d{1,3})$").ok()?;

        for (i, text) in cell_texts.iter().enumerate() {
            if let Some(caps) = score_pattern.captures(text) {
                let home_score: u8 = caps.get(1)?.as_str().parse().ok()?;
                let away_score: u8 = caps.get(2)?.as_str().parse().ok()?;

                let home_team = if i > date_idx + 1 {
                    self.normalize_team_name(&cell_texts[i - 1])
                } else if i > date_idx {
                    self.normalize_team_name(&cell_texts[date_idx + 1])
                } else {
                    None
                }?;

                let away_team = if i + 1 < cell_texts.len() {
                    self.normalize_team_name(&cell_texts[i + 1])
                } else {
                    None
                }?;

                let date = self.parse_date(&cell_texts[date_idx])?;

                return Some(RawMatch {
                    date,
                    home_team,
                    away_team,
                    home_score,
                    away_score,
                    home_tries: None,
                    away_tries: None,
                    round: None,
                    venue: None,
                });
            }
        }

        None
    }

    /// Check if a string looks like a date
    fn looks_like_date(&self, s: &str) -> bool {
        let date_pattern =
            Regex::new(r"(?i)\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)")
                .unwrap();
        date_pattern.is_match(s)
    }

    /// Parse a date string
    fn parse_date(&self, s: &str) -> Option<NaiveDate> {
        let pattern = r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})";

        let month_map: HashMap<&str, u32> = [
            ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4),
            ("may", 5), ("jun", 6), ("jul", 7), ("aug", 8),
            ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12),
        ].into_iter().collect();

        if let Ok(re) = Regex::new(pattern) {
            if let Some(caps) = re.captures(&s.to_lowercase()) {
                let day: u32 = caps.get(1)?.as_str().parse().ok()?;
                let month = month_map.get(caps.get(2)?.as_str())?;
                let year: i32 = caps.get(3)?.as_str().parse().ok()?;
                return NaiveDate::from_ymd_opt(year, *month, day);
            }
        }

        None
    }

    /// Normalize a team name using aliases
    fn normalize_team_name(&self, name: &str) -> Option<TeamInfo> {
        let name_clean = name.trim().to_lowercase().replace(['[', ']', '*', '†'], "");

        if let Some((canonical, country)) = self.team_aliases.get(&name_clean) {
            return Some(TeamInfo {
                name: canonical.clone(),
                country: *country,
            });
        }

        for (alias, (canonical, country)) in &self.team_aliases {
            if name_clean.contains(alias) || alias.contains(&name_clean) {
                return Some(TeamInfo {
                    name: canonical.clone(),
                    country: *country,
                });
            }
        }

        None
    }

    /// Infer venue from home team (Six Nations has fixed home stadiums)
    fn infer_venue(&self, home_team: &str) -> Option<String> {
        match home_team {
            "England" => Some("Twickenham".to_string()),
            "France" => Some("Stade de France".to_string()),
            "Ireland" => Some("Aviva Stadium".to_string()),
            "Wales" => Some("Principality Stadium".to_string()),
            "Scotland" => Some("Murrayfield".to_string()),
            "Italy" => Some("Stadio Olimpico".to_string()),
            _ => None,
        }
    }

    /// Convert raw matches to MatchRecords
    pub fn to_match_records(
        &self,
        raw_matches: Vec<RawMatch>,
        team_resolver: &impl Fn(&str, Country) -> Result<TeamId>,
    ) -> Result<Vec<MatchRecord>> {
        let mut records = Vec::new();

        for raw in raw_matches {
            let home_team = team_resolver(&raw.home_team.name, raw.home_team.country)?;
            let away_team = team_resolver(&raw.away_team.name, raw.away_team.country)?;

            // Use scraped venue or infer from home team
            let venue = raw.venue.or_else(|| self.infer_venue(&raw.home_team.name));

            records.push(MatchRecord {
                date: raw.date,
                home_team,
                away_team,
                home_score: raw.home_score,
                away_score: raw.away_score,
                venue,
                round: raw.round,
                home_tries: raw.home_tries,
                away_tries: raw.away_tries,
                source: DataSource::Wikipedia,
            });
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_team_normalization() {
        let scraper = SixNationsScraper::new();

        let team = scraper.normalize_team_name("England");
        assert!(team.is_some());
        assert_eq!(team.unwrap().name, "England");

        let team = scraper.normalize_team_name("France");
        assert!(team.is_some());
        assert_eq!(team.unwrap().name, "France");

        let team = scraper.normalize_team_name("Ireland");
        assert!(team.is_some());
        assert_eq!(team.unwrap().name, "Ireland");
    }

    #[test]
    fn test_date_parsing() {
        let scraper = SixNationsScraper::new();

        let date = scraper.parse_date("1 February 2024");
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 2, 1));

        let date = scraper.parse_date("15 Mar 2023");
        assert_eq!(date, NaiveDate::from_ymd_opt(2023, 3, 15));
    }
}
