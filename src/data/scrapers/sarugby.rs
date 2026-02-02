//! SA Rugby scraper for South African match data
//!
//! Scrapes sarugby.co.za for venue and halftime scores.

use super::wikipedia::{RawMatch, TeamInfo};
use crate::Result;
use chrono::NaiveDate;

/// Scraper for sarugby.co.za
pub struct SaRugbyScraper {
    client: reqwest::blocking::Client,
}

impl SaRugbyScraper {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::builder()
            .user_agent("rugby-predictor/0.1")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        SaRugbyScraper { client }
    }

    /// Fetch matches for a season
    pub fn fetch_season(&self, year: u16) -> Result<Vec<RawMatchWithVenue>> {
        let url = format!(
            "https://www.sarugby.co.za/competition/super-rugby/{}/results",
            year
        );

        log::info!("Fetching SA Rugby data for {}", year);

        let response = self.client.get(&url).send();

        match response {
            Ok(resp) if resp.status().is_success() => {
                let html = resp.text()?;
                self.parse_page(&html, year)
            }
            Ok(resp) => {
                log::warn!("SA Rugby returned {}", resp.status());
                Ok(vec![])
            }
            Err(e) => {
                log::warn!("Failed to fetch SA Rugby: {}", e);
                Ok(vec![])
            }
        }
    }

    /// Fetch all available seasons
    pub fn fetch_all(&self) -> Result<Vec<RawMatchWithVenue>> {
        let current_year: u16 = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse()
            .unwrap_or(2026);
        let mut all_matches = Vec::new();

        // SA Rugby typically only has recent seasons
        for year in 2016..=current_year {
            match self.fetch_season(year) {
                Ok(matches) => all_matches.extend(matches),
                Err(e) => log::warn!("Failed to fetch {} from SA Rugby: {}", year, e),
            }
        }

        Ok(all_matches)
    }

    /// Parse the SA Rugby results page
    fn parse_page(&self, html: &str, _year: u16) -> Result<Vec<RawMatchWithVenue>> {
        use scraper::{Html, Selector};

        let document = Html::parse_document(html);
        let mut matches = Vec::new();

        // Look for match result elements
        // SA Rugby uses various class names for match cards
        let match_selector = Selector::parse(".match-card, .result-item, .fixture").ok();

        if let Some(selector) = match_selector {
            for element in document.select(&selector) {
                if let Some(m) = self.parse_match_element(&element) {
                    matches.push(m);
                }
            }
        }

        log::info!("Parsed {} matches from SA Rugby", matches.len());
        Ok(matches)
    }

    /// Parse a single match element
    fn parse_match_element(&self, _element: &scraper::ElementRef) -> Option<RawMatchWithVenue> {
        // TODO: Implement actual parsing based on SA Rugby's current HTML structure
        // This is a placeholder that would need to be updated based on the actual site
        None
    }
}

impl Default for SaRugbyScraper {
    fn default() -> Self {
        Self::new()
    }
}

/// Match data with venue information
#[derive(Debug, Clone)]
pub struct RawMatchWithVenue {
    pub date: NaiveDate,
    pub home_team: TeamInfo,
    pub away_team: TeamInfo,
    pub home_score: u8,
    pub away_score: u8,
    pub venue: Option<String>,
    pub home_halftime: Option<u8>,
    pub away_halftime: Option<u8>,
}

impl From<RawMatchWithVenue> for RawMatch {
    fn from(m: RawMatchWithVenue) -> Self {
        RawMatch {
            date: m.date,
            home_team: m.home_team,
            away_team: m.away_team,
            home_score: m.home_score,
            away_score: m.away_score,
            home_tries: None,
            away_tries: None,
            round: None,
            venue: m.venue,
        }
    }
}
