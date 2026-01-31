//! Lassen scraper for match round numbers and times
//!
//! Scrapes lassen.co.nz for additional match metadata.

use super::wikipedia::TeamInfo;
use crate::Result;
use chrono::NaiveDate;

/// Scraper for lassen.co.nz
pub struct LassenScraper {
    client: reqwest::blocking::Client,
}

impl LassenScraper {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::builder()
            .user_agent("rugby-predictor/0.1")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        LassenScraper { client }
    }

    /// Fetch matches for a season
    pub fn fetch_season(&self, year: u16) -> Result<Vec<RawMatchWithRound>> {
        let url = format!("https://www.lassen.co.nz/super-rugby-{}/", year);

        log::info!("Fetching Lassen data for {}", year);

        let response = self.client.get(&url).send();

        match response {
            Ok(resp) if resp.status().is_success() => {
                let html = resp.text()?;
                self.parse_page(&html, year)
            }
            Ok(resp) => {
                log::warn!("Lassen returned {}", resp.status());
                Ok(vec![])
            }
            Err(e) => {
                log::warn!("Failed to fetch Lassen: {}", e);
                Ok(vec![])
            }
        }
    }

    /// Fetch all available seasons
    pub fn fetch_all(&self) -> Result<Vec<RawMatchWithRound>> {
        let current_year: u16 = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse()
            .unwrap_or(2026);
        let mut all_matches = Vec::new();

        for year in 2010..=current_year {
            match self.fetch_season(year) {
                Ok(matches) => all_matches.extend(matches),
                Err(e) => log::warn!("Failed to fetch {} from Lassen: {}", year, e),
            }
        }

        Ok(all_matches)
    }

    /// Parse the Lassen results page
    fn parse_page(&self, html: &str, _year: u16) -> Result<Vec<RawMatchWithRound>> {
        use scraper::{Html, Selector};

        let document = Html::parse_document(html);
        let mut matches = Vec::new();

        // Look for match table rows
        let table_selector = Selector::parse("table").ok();
        let row_selector = Selector::parse("tr").ok();

        if let (Some(table_sel), Some(row_sel)) = (table_selector, row_selector) {
            for table in document.select(&table_sel) {
                let mut current_round: Option<u8> = None;

                for row in table.select(&row_sel) {
                    if let Some(m) = self.parse_match_row(&row, &mut current_round) {
                        matches.push(m);
                    }
                }
            }
        }

        log::info!("Parsed {} matches from Lassen", matches.len());
        Ok(matches)
    }

    /// Parse a single match row
    fn parse_match_row(
        &self,
        _row: &scraper::ElementRef,
        _current_round: &mut Option<u8>,
    ) -> Option<RawMatchWithRound> {
        // TODO: Implement actual parsing based on Lassen's current HTML structure
        // This is a placeholder
        None
    }
}

impl Default for LassenScraper {
    fn default() -> Self {
        Self::new()
    }
}

/// Match data with round information
#[derive(Debug, Clone)]
pub struct RawMatchWithRound {
    pub date: NaiveDate,
    pub home_team: TeamInfo,
    pub away_team: TeamInfo,
    pub home_score: u8,
    pub away_score: u8,
    pub round: Option<u8>,
    pub kickoff_time: Option<String>,
}
