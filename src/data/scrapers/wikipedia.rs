//! Wikipedia scraper for Super Rugby match data
//!
//! Parses Wikipedia season pages from 1996 to present.
//! Supports caching HTML files for offline testing and reduced load.

use crate::{Country, DataSource, MatchRecord, Result, RugbyError, TeamId};
use chrono::NaiveDate;
use regex::Regex;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Scraper for Wikipedia Super Rugby pages
pub struct WikipediaScraper {
    client: reqwest::blocking::Client,
    team_aliases: HashMap<String, (String, Country)>,
    /// Optional cache directory for offline HTML files
    cache_dir: Option<PathBuf>,
    /// If true, only use cache (no network requests)
    offline_only: bool,
}

impl Default for WikipediaScraper {
    fn default() -> Self {
        Self::new()
    }
}

impl WikipediaScraper {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::builder()
            .user_agent("rugby-predictor/0.1")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        WikipediaScraper {
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
            // Create a safe filename from the URL
            let filename = url
                .replace("https://", "")
                .replace("http://", "")
                .replace('/', "_")
                .replace('?', "_")
                + ".html";
            dir.join(filename)
        })
    }

    /// Load HTML from cache if available and fresh
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

    /// Parse a cached HTML file directly (for testing)
    pub fn parse_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<RawMatch>> {
        let html = std::fs::read_to_string(path.as_ref())?;
        self.parse_page(&html)
    }

    /// Parse all HTML files in a directory (for batch testing)
    pub fn parse_directory<P: AsRef<Path>>(&self, dir: P) -> Result<Vec<RawMatch>> {
        let mut all_matches = Vec::new();

        for entry in std::fs::read_dir(dir.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "html").unwrap_or(false) {
                log::info!("Parsing {}", path.display());
                match self.parse_file(&path) {
                    Ok(matches) => {
                        log::info!("  Found {} matches", matches.len());
                        all_matches.extend(matches);
                    }
                    Err(e) => log::warn!("  Failed: {}", e),
                }
            }
        }

        // Deduplicate
        all_matches.sort_by(|a, b| (&a.date, &a.home_team).cmp(&(&b.date, &b.home_team)));
        all_matches.dedup_by(|a, b| {
            a.date == b.date && a.home_team == b.home_team && a.away_team == b.away_team
        });

        Ok(all_matches)
    }

    /// Build the default team alias mapping
    fn default_team_aliases() -> HashMap<String, (String, Country)> {
        let mut aliases = HashMap::new();

        // New Zealand teams
        for name in ["Blues", "Auckland Blues"] {
            aliases.insert(
                name.to_lowercase(),
                ("Blues".to_string(), Country::NewZealand),
            );
        }
        for name in ["Chiefs", "Waikato Chiefs"] {
            aliases.insert(
                name.to_lowercase(),
                ("Chiefs".to_string(), Country::NewZealand),
            );
        }
        for name in ["Crusaders", "Canterbury Crusaders"] {
            aliases.insert(
                name.to_lowercase(),
                ("Crusaders".to_string(), Country::NewZealand),
            );
        }
        for name in ["Highlanders", "Otago Highlanders"] {
            aliases.insert(
                name.to_lowercase(),
                ("Highlanders".to_string(), Country::NewZealand),
            );
        }
        for name in ["Hurricanes", "Wellington Hurricanes"] {
            aliases.insert(
                name.to_lowercase(),
                ("Hurricanes".to_string(), Country::NewZealand),
            );
        }
        {
            let name = "Moana Pasifika";
            aliases.insert(
                name.to_lowercase(),
                ("Moana Pasifika".to_string(), Country::NewZealand),
            );
        }

        // Australian teams
        for name in ["Brumbies", "ACT Brumbies"] {
            aliases.insert(
                name.to_lowercase(),
                ("Brumbies".to_string(), Country::Australia),
            );
        }
        for name in ["Reds", "Queensland Reds"] {
            aliases.insert(
                name.to_lowercase(),
                ("Reds".to_string(), Country::Australia),
            );
        }
        for name in ["Waratahs", "NSW Waratahs", "New South Wales Waratahs"] {
            aliases.insert(
                name.to_lowercase(),
                ("Waratahs".to_string(), Country::Australia),
            );
        }
        for name in ["Force", "Western Force"] {
            aliases.insert(
                name.to_lowercase(),
                ("Force".to_string(), Country::Australia),
            );
        }
        for name in ["Rebels", "Melbourne Rebels"] {
            aliases.insert(
                name.to_lowercase(),
                ("Rebels".to_string(), Country::Australia),
            );
        }

        // South African teams
        for name in [
            "Bulls",
            "Blue Bulls",
            "Northern Bulls",
            "Vodacom Bulls",
            "Northern Transvaal", // Historical name (1996-1998)
        ] {
            aliases.insert(
                name.to_lowercase(),
                ("Bulls".to_string(), Country::SouthAfrica),
            );
        }
        for name in [
            "Lions",
            "Golden Lions",
            "Johannesburg Lions",
            "Emirates Lions",
            "Cats",          // Historical name (1998-2005)
            "Golden Cats",   // Historical name
            "Gauteng Lions", // Historical name
            "Transvaal",     // Historical name (1996-1997) - Johannesburg team
            "Auto & General Lions",
            "MTN Lions",
        ] {
            aliases.insert(
                name.to_lowercase(),
                ("Lions".to_string(), Country::SouthAfrica),
            );
        }
        for name in [
            "Sharks",
            "Natal Sharks",
            "Durban Sharks",
            "Cell C Sharks",
            "Natal",          // Historical short name
            "Coastal Sharks", // Historical name
            "The Sharks",
        ] {
            aliases.insert(
                name.to_lowercase(),
                ("Sharks".to_string(), Country::SouthAfrica),
            );
        }
        for name in [
            "Stormers",
            "Western Province Stormers",
            "DHL Stormers",
            "Western Province", // Historical name (1996-1997)
            "Vodacom Stormers",
            "Western Stormers",
        ] {
            aliases.insert(
                name.to_lowercase(),
                ("Stormers".to_string(), Country::SouthAfrica),
            );
        }
        for name in [
            "Cheetahs",
            "Free State Cheetahs",
            "Central Cheetahs",
            "Free State",
            "Toyota Cheetahs",
            "Vodacom Cheetahs",
        ] {
            aliases.insert(
                name.to_lowercase(),
                ("Cheetahs".to_string(), Country::SouthAfrica),
            );
        }
        for name in ["Kings", "Southern Kings"] {
            aliases.insert(
                name.to_lowercase(),
                ("Kings".to_string(), Country::SouthAfrica),
            );
        }

        // Other teams
        for name in ["Sunwolves", "Hito-Communications Sunwolves"] {
            aliases.insert(
                name.to_lowercase(),
                ("Sunwolves".to_string(), Country::Japan),
            );
        }
        {
            let name = "Jaguares";
            aliases.insert(
                name.to_lowercase(),
                ("Jaguares".to_string(), Country::Argentina),
            );
        }
        for name in ["Fijian Drua", "Drua"] {
            aliases.insert(
                name.to_lowercase(),
                ("Fijian Drua".to_string(), Country::Fiji),
            );
        }

        aliases
    }

    /// Get all season URLs from 1996 to current year
    pub fn get_season_urls(&self) -> Vec<(u16, String)> {
        let current_year = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse()
            .unwrap_or(2026);
        let mut urls = Vec::new();

        for year in 1996..=current_year {
            let base = "https://en.wikipedia.org/wiki/";
            if (1996..=2005).contains(&year) {
                // Super 12 era
                urls.push((year, format!("{}{}_Super_12_season", base, year)));
            } else if (2006..=2010).contains(&year) {
                // Super 14 era
                urls.push((year, format!("{}{}_Super_14_season", base, year)));
            } else if (2011..=2021).contains(&year) {
                // Super Rugby era
                urls.push((year, format!("{}{}_Super_Rugby_season", base, year)));
                urls.push((
                    year,
                    format!("{}List_of_{}_Super_Rugby_matches", base, year),
                ));
            } else {
                // Super Rugby Pacific era (2022+)
                urls.push((year, format!("{}{}_Super_Rugby_Pacific_season", base, year)));
                urls.push((
                    year,
                    format!("{}List_of_{}_Super_Rugby_Pacific_matches", base, year),
                ));
            }
        }

        urls
    }

    /// Fetch and parse a single season
    pub fn fetch_season(&self, year: u16) -> Result<Vec<RawMatch>> {
        let urls = self.get_season_urls();
        let season_urls: Vec<_> = urls.into_iter().filter(|(y, _)| *y == year).collect();

        let mut all_matches = Vec::new();
        for (_, url) in season_urls {
            match self.fetch_page(&url) {
                Ok(matches) => all_matches.extend(matches),
                Err(e) => log::warn!("Failed to fetch {}: {}", url, e),
            }
        }

        // Deduplicate matches (same date and teams)
        all_matches.sort_by(|a, b| (&a.date, &a.home_team).cmp(&(&b.date, &b.home_team)));
        all_matches.dedup_by(|a, b| {
            a.date == b.date && a.home_team == b.home_team && a.away_team == b.away_team
        });

        Ok(all_matches)
    }

    /// Fetch all seasons
    pub fn fetch_all(&self) -> Result<Vec<RawMatch>> {
        let current_year = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse()
            .unwrap_or(2026);
        let mut all_matches = Vec::new();

        for year in 1996..=current_year {
            log::info!("Fetching {} season...", year);
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

    /// Fetch upcoming fixtures for a season (matches without scores yet)
    pub fn fetch_fixtures(&self, year: u16) -> Result<Vec<RawFixture>> {
        let url = format!(
            "https://en.wikipedia.org/wiki/List_of_{}_Super_Rugby_Pacific_matches",
            year
        );

        log::info!("Fetching {} fixtures from {}", year, url);

        // Try cache first
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

        // Score pattern to detect if a match has been played
        let score_pattern = Regex::new(r"(\d{1,3})\s*[–-]\s*(\d{1,3})").unwrap();

        for event in document.select(&event_selector) {
            // Get all text content for date extraction
            let event_text: String = event.text().collect();

            // Find date in the event
            let date = self.extract_date_from_text(&event_text);

            // Find teams (should be two span.fn.org elements)
            let teams: Vec<_> = event.select(&team_selector).collect();
            if teams.len() < 2 {
                continue;
            }

            let home_text: String = teams[0].text().collect();
            let away_text: String = teams[1].text().collect();

            let home_team = self.normalize_team_name(home_text.trim());
            let away_team = self.normalize_team_name(away_text.trim());

            // Check if there's a score - if so, this match has been played
            let mut has_score = false;
            for td in event.select(&td_selector) {
                let td_text: String = td.text().collect();
                if score_pattern.is_match(&td_text) {
                    has_score = true;
                    break;
                }
            }

            // Only include fixtures WITHOUT scores (upcoming matches)
            if has_score {
                continue;
            }

            // Extract venue
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
                    round: None, // Will be inferred from date
                });
            }
        }

        // Deduplicate
        fixtures.sort_by(|a, b| (&a.date, &a.home_team.name).cmp(&(&b.date, &b.home_team.name)));
        fixtures.dedup_by(|a, b| {
            a.date == b.date && a.home_team.name == b.home_team.name && a.away_team.name == b.away_team.name
        });

        // Infer round numbers from dates (each round is typically one week)
        if !fixtures.is_empty() {
            // Get unique weeks and map them to rounds
            let mut dates: Vec<NaiveDate> = fixtures.iter().map(|f| f.date).collect();
            dates.sort();
            dates.dedup();

            // Group consecutive dates within 3 days as same round
            let mut date_to_round: HashMap<NaiveDate, u8> = HashMap::new();
            let mut current_round = 1u8;
            let mut round_start_date = dates[0];

            for date in &dates {
                let days_since_round_start = (*date - round_start_date).num_days();
                if days_since_round_start > 3 {
                    // New round
                    current_round += 1;
                    round_start_date = *date;
                }
                date_to_round.insert(*date, current_round);
            }

            // Apply rounds to fixtures
            for fixture in &mut fixtures {
                fixture.round = date_to_round.get(&fixture.date).copied();
            }
        }

        log::info!("Found {} upcoming fixtures", fixtures.len());
        Ok(fixtures)
    }

    /// Fetch and parse a Wikipedia page (uses cache if available)
    fn fetch_page(&self, url: &str) -> Result<Vec<RawMatch>> {
        // Try cache first
        if let Some(html) = self.load_from_cache(url) {
            return self.parse_page(&html);
        }

        // If offline-only, fail
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

        // Save to cache
        if let Err(e) = self.save_to_cache(url, &html) {
            log::warn!("Failed to cache {}: {}", url, e);
        }

        self.parse_page(&html)
    }

    /// Parse HTML content for match data
    fn parse_page(&self, html: &str) -> Result<Vec<RawMatch>> {
        let document = Html::parse_document(html);
        let mut matches = Vec::new();

        // Try schema.org/SportsEvent format (modern Wikipedia pages)
        matches.extend(self.parse_sports_events(&document)?);

        // Try collapsible table format (older seasons like 1996-2010)
        matches.extend(self.parse_collapsible_tables(&document)?);

        // Try wikitable-based parsing
        matches.extend(self.parse_tables(&document)?);

        // Also try text-based regex parsing (for very old pages)
        matches.extend(self.parse_text_content(&document)?);

        // Deduplicate
        matches.sort_by(|a, b| (&a.date, &a.home_team).cmp(&(&b.date, &b.home_team)));
        matches.dedup_by(|a, b| {
            a.date == b.date && a.home_team == b.home_team && a.away_team == b.away_team
        });

        Ok(matches)
    }

    /// Parse collapsible table format (used in older Wikipedia pages)
    /// Format: Date | Home Team <flag> | Score | <flag> Away Team | Venue
    fn parse_collapsible_tables(&self, document: &Html) -> Result<Vec<RawMatch>> {
        let mut matches = Vec::new();

        // Select collapsible tables
        let table_selector = Selector::parse("table.mw-collapsible").unwrap();
        let tr_selector = Selector::parse("tr").unwrap();
        let td_selector = Selector::parse("td").unwrap();

        let score_pattern = Regex::new(r"(\d{1,3})[–-](\d{1,3})").unwrap();
        let date_pattern = Regex::new(
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        )
        .unwrap();

        for table in document.select(&table_selector) {
            // Get the first row which contains match info
            let rows: Vec<_> = table.select(&tr_selector).collect();
            if rows.is_empty() {
                continue;
            }

            // Get all td cells from the first row
            let cells: Vec<_> = rows[0].select(&td_selector).collect();
            if cells.len() < 4 {
                continue;
            }

            // Extract text from each cell
            let cell_texts: Vec<String> =
                cells.iter().map(|c| c.text().collect::<String>()).collect();

            // Find date
            let mut date = None;
            for text in &cell_texts {
                if let Some(caps) = date_pattern.captures(text) {
                    let day: u32 = caps
                        .get(1)
                        .and_then(|m| m.as_str().parse().ok())
                        .unwrap_or(0);
                    let month_str = caps
                        .get(2)
                        .map(|m| m.as_str().to_lowercase())
                        .unwrap_or_default();
                    let year: i32 = caps
                        .get(3)
                        .and_then(|m| m.as_str().parse().ok())
                        .unwrap_or(0);

                    let month = match month_str.as_str() {
                        "january" => 1,
                        "february" => 2,
                        "march" => 3,
                        "april" => 4,
                        "may" => 5,
                        "june" => 6,
                        "july" => 7,
                        "august" => 8,
                        "september" => 9,
                        "october" => 10,
                        "november" => 11,
                        "december" => 12,
                        _ => 0,
                    };

                    if month > 0 && day > 0 && year > 0 {
                        date = NaiveDate::from_ymd_opt(year, month, day);
                        break;
                    }
                }
            }

            // Find score and teams
            for (i, text) in cell_texts.iter().enumerate() {
                if let Some(caps) = score_pattern.captures(text) {
                    let home_score: u8 = caps
                        .get(1)
                        .and_then(|m| m.as_str().parse().ok())
                        .unwrap_or(0);
                    let away_score: u8 = caps
                        .get(2)
                        .and_then(|m| m.as_str().parse().ok())
                        .unwrap_or(0);

                    // Home team is typically in the cell before score
                    // Away team is in the cell after score
                    let home_team = if i > 0 {
                        self.extract_team_from_text(&cell_texts[i - 1])
                    } else {
                        None
                    };

                    let away_team = if i + 1 < cell_texts.len() {
                        self.extract_team_from_text(&cell_texts[i + 1])
                    } else {
                        None
                    };

                    if let (Some(d), Some(ht), Some(at)) = (date, home_team, away_team) {
                        matches.push(RawMatch {
                            date: d,
                            home_team: ht,
                            away_team: at,
                            home_score,
                            away_score,
                            home_tries: None,
                            away_tries: None,
                            round: None,
                        });
                    }
                    break;
                }
            }
        }

        log::debug!("parse_collapsible_tables found {} matches", matches.len());
        Ok(matches)
    }

    /// Extract team name from cell text (strips flags and cleans up)
    fn extract_team_from_text(&self, text: &str) -> Option<TeamInfo> {
        // Clean up the text - remove extra whitespace
        let cleaned = text.trim();

        // Try to normalize the team name
        self.normalize_team_name(cleaned)
    }

    /// Parse schema.org/SportsEvent format (modern Wikipedia match boxes)
    fn parse_sports_events(&self, document: &Html) -> Result<Vec<RawMatch>> {
        let mut matches = Vec::new();

        // Select match event divs
        let event_selector =
            Selector::parse("div[itemtype='http://schema.org/SportsEvent']").unwrap();
        let team_selector = Selector::parse("span.fn.org").unwrap();
        let td_selector = Selector::parse("td").unwrap();

        // Score pattern - created once outside the loop
        let score_pattern = Regex::new(r"(\d{1,3})\s*[–-]\s*(\d{1,3})").unwrap();

        for event in document.select(&event_selector) {
            // Get all text content for date extraction
            let event_text: String = event.text().collect();

            // Find date in the event
            let date = self.extract_date_from_text(&event_text);

            // Find teams (should be two span.fn.org elements)
            let teams: Vec<_> = event.select(&team_selector).collect();
            if teams.len() < 2 {
                continue;
            }

            let home_text: String = teams[0].text().collect();
            let away_text: String = teams[1].text().collect();

            let home_team = self.normalize_team_name(home_text.trim());
            let away_team = self.normalize_team_name(away_text.trim());

            // Find score - look for pattern like "29–37" in td elements
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

            // Only create match if we have all required fields
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
                });
            }
        }

        log::debug!("parse_sports_events found {} matches", matches.len());
        Ok(matches)
    }

    /// Extract date from text content
    fn extract_date_from_text(&self, text: &str) -> Option<NaiveDate> {
        // Try various patterns
        let patterns = [
            // "31 January 2020"
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            // "January 31, 2020"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        ];

        let month_map: HashMap<&str, u32> = [
            ("january", 1),
            ("february", 2),
            ("march", 3),
            ("april", 4),
            ("may", 5),
            ("june", 6),
            ("july", 7),
            ("august", 8),
            ("september", 9),
            ("october", 10),
            ("november", 11),
            ("december", 12),
        ]
        .into_iter()
        .collect();

        // Try "31 January 2020" format
        if let Ok(re) = Regex::new(patterns[0]) {
            if let Some(caps) = re.captures(text) {
                let day: u32 = caps.get(1)?.as_str().parse().ok()?;
                let month = month_map.get(caps.get(2)?.as_str().to_lowercase().as_str())?;
                let year: i32 = caps.get(3)?.as_str().parse().ok()?;
                return NaiveDate::from_ymd_opt(year, *month, day);
            }
        }

        // Try "January 31, 2020" format
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

                // Look for rows with date, team, score pattern
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
        // Common patterns:
        // Date | Home | Score | Away | Venue
        // Date | Home | v | Away | Time | Venue

        let cell_texts: Vec<String> = cells
            .iter()
            .map(|c| c.text().collect::<String>().trim().to_string())
            .collect();

        // Find date cell
        let date_idx = cell_texts.iter().position(|t| self.looks_like_date(t))?;

        // Look for score pattern (e.g., "28-24", "28–24")
        let score_pattern = Regex::new(r"^(\d{1,3})\s*[-–]\s*(\d{1,3})$").ok()?;

        for (i, text) in cell_texts.iter().enumerate() {
            if let Some(caps) = score_pattern.captures(text) {
                let home_score: u8 = caps.get(1)?.as_str().parse().ok()?;
                let away_score: u8 = caps.get(2)?.as_str().parse().ok()?;

                // Home team should be before score, away team after
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
                });
            }
        }

        None
    }

    /// Parse text content using regex (for unstructured pages)
    fn parse_text_content(&self, document: &Html) -> Result<Vec<RawMatch>> {
        let mut matches = Vec::new();

        // Get all text content
        let body_selector = Selector::parse("body").unwrap();
        let text: String = document
            .select(&body_selector)
            .next()
            .map(|b| b.text().collect())
            .unwrap_or_default();

        // Pattern from Python: date | home team | score-score | away team
        let score_pattern = Regex::new(
            r"(?i)(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4})[^\|]*\|?\s*([a-z][a-z\s]*?)\s*\|?\s*(\d{1,3})\s*[-–]\s*(\d{1,3})\s*\|?\s*([a-z][a-z\s]*?)(?:\s*\||$)"
        ).map_err(|e| RugbyError::Parse(e.to_string()))?;

        for caps in score_pattern.captures_iter(&text) {
            let date_str = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let home_str = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            let home_score: u8 = caps
                .get(3)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let away_score: u8 = caps
                .get(4)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let away_str = caps.get(5).map(|m| m.as_str().trim()).unwrap_or("");

            if let (Some(date), Some(home_team), Some(away_team)) = (
                self.parse_date(date_str),
                self.normalize_team_name(home_str),
                self.normalize_team_name(away_str),
            ) {
                matches.push(RawMatch {
                    date,
                    home_team,
                    away_team,
                    home_score,
                    away_score,
                    home_tries: None,
                    away_tries: None,
                    round: None,
                });
            }
        }

        Ok(matches)
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
        // Try various date formats
        let patterns = [
            r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})",
            r"(\d{4})-(\d{2})-(\d{2})",
        ];

        let month_map: HashMap<&str, u32> = [
            ("jan", 1),
            ("feb", 2),
            ("mar", 3),
            ("apr", 4),
            ("may", 5),
            ("jun", 6),
            ("jul", 7),
            ("aug", 8),
            ("sep", 9),
            ("oct", 10),
            ("nov", 11),
            ("dec", 12),
        ]
        .into_iter()
        .collect();

        // Try "1 Jan 2024" format
        if let Ok(re) = Regex::new(patterns[0]) {
            if let Some(caps) = re.captures(&s.to_lowercase()) {
                let day: u32 = caps.get(1)?.as_str().parse().ok()?;
                let month = month_map.get(caps.get(2)?.as_str())?;
                let year: i32 = caps.get(3)?.as_str().parse().ok()?;
                return NaiveDate::from_ymd_opt(year, *month, day);
            }
        }

        // Try ISO format
        if let Ok(re) = Regex::new(patterns[1]) {
            if let Some(caps) = re.captures(s) {
                let year: i32 = caps.get(1)?.as_str().parse().ok()?;
                let month: u32 = caps.get(2)?.as_str().parse().ok()?;
                let day: u32 = caps.get(3)?.as_str().parse().ok()?;
                return NaiveDate::from_ymd_opt(year, month, day);
            }
        }

        None
    }

    /// Normalize a team name using aliases
    fn normalize_team_name(&self, name: &str) -> Option<TeamInfo> {
        let name_clean = name.trim().to_lowercase().replace(['[', ']', '*', '†'], "");

        // Direct lookup
        if let Some((canonical, country)) = self.team_aliases.get(&name_clean) {
            return Some(TeamInfo {
                name: canonical.clone(),
                country: *country,
            });
        }

        // Partial match (team name contained in input)
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

    /// Convert raw matches to MatchRecords, resolving team IDs
    pub fn to_match_records(
        &self,
        raw_matches: Vec<RawMatch>,
        team_resolver: &impl Fn(&str, Country) -> Result<TeamId>,
    ) -> Result<Vec<MatchRecord>> {
        let mut records = Vec::new();

        for raw in raw_matches {
            let home_team = team_resolver(&raw.home_team.name, raw.home_team.country)?;
            let away_team = team_resolver(&raw.away_team.name, raw.away_team.country)?;

            records.push(MatchRecord {
                date: raw.date,
                home_team,
                away_team,
                home_score: raw.home_score,
                away_score: raw.away_score,
                venue: None,
                round: raw.round,
                home_tries: raw.home_tries,
                away_tries: raw.away_tries,
                source: DataSource::Wikipedia,
            });
        }

        Ok(records)
    }
}

/// Team info extracted from Wikipedia
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TeamInfo {
    pub name: String,
    pub country: Country,
}

/// Raw match data before team ID resolution
#[derive(Debug, Clone)]
pub struct RawMatch {
    pub date: NaiveDate,
    pub home_team: TeamInfo,
    pub away_team: TeamInfo,
    pub home_score: u8,
    pub away_score: u8,
    pub home_tries: Option<u8>,
    pub away_tries: Option<u8>,
    pub round: Option<u8>,
    pub venue: Option<String>,
}

/// Raw fixture data (upcoming match without scores)
#[derive(Debug, Clone)]
pub struct RawFixture {
    pub date: NaiveDate,
    pub home_team: TeamInfo,
    pub away_team: TeamInfo,
    pub venue: Option<String>,
    pub round: Option<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_parsing() {
        let scraper = WikipediaScraper::new();

        let date = scraper.parse_date("1 January 2024");
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 1, 1));

        let date = scraper.parse_date("25 Feb 2023");
        assert_eq!(date, NaiveDate::from_ymd_opt(2023, 2, 25));

        let date = scraper.parse_date("2024-03-15");
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 3, 15));
    }

    #[test]
    fn test_team_normalization() {
        let scraper = WikipediaScraper::new();

        let team = scraper.normalize_team_name("Blues");
        assert!(team.is_some());
        assert_eq!(team.unwrap().name, "Blues");

        let team = scraper.normalize_team_name("Auckland Blues");
        assert!(team.is_some());
        assert_eq!(team.unwrap().name, "Blues");

        let team = scraper.normalize_team_name("Crusaders[1]");
        assert!(team.is_some());
        assert_eq!(team.unwrap().name, "Crusaders");
    }
}
