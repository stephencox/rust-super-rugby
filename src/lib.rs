//! Super Rugby prediction using deep learning
//!
//! A hierarchical transformer architecture for predicting Super Rugby match outcomes.

pub mod data;
pub mod features;
pub mod model;
pub mod predict;
pub mod training;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Unique identifier for a team
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TeamId(pub i64);

impl fmt::Display for TeamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Team({})", self.0)
    }
}

/// Source of match data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSource {
    Wikipedia,
    SaRugby,
    Lassen,
}

impl fmt::Display for DataSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataSource::Wikipedia => write!(f, "Wikipedia"),
            DataSource::SaRugby => write!(f, "SA Rugby"),
            DataSource::Lassen => write!(f, "Lassen"),
        }
    }
}

/// Country code for team nationality
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Country {
    NewZealand,
    Australia,
    SouthAfrica,
    Japan,
    Argentina,
    Fiji,
    Samoa,
    Tonga,
}

impl Country {
    pub fn code(&self) -> &'static str {
        match self {
            Country::NewZealand => "NZ",
            Country::Australia => "AU",
            Country::SouthAfrica => "SA",
            Country::Japan => "JP",
            Country::Argentina => "AR",
            Country::Fiji => "FJ",
            Country::Samoa => "WS",
            Country::Tonga => "TO",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_uppercase().as_str() {
            "NZ" => Some(Country::NewZealand),
            "AU" => Some(Country::Australia),
            "SA" | "ZA" => Some(Country::SouthAfrica),
            "JP" => Some(Country::Japan),
            "AR" => Some(Country::Argentina),
            "FJ" => Some(Country::Fiji),
            "WS" => Some(Country::Samoa),
            "TO" => Some(Country::Tonga),
            _ => None,
        }
    }
}

/// A Super Rugby team
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub id: TeamId,
    pub name: String,
    pub country: Country,
    pub aliases: Vec<String>,
}

impl Team {
    pub fn matches_name(&self, name: &str) -> bool {
        let name_lower = name.to_lowercase();
        self.name.to_lowercase() == name_lower
            || self.aliases.iter().any(|a| a.to_lowercase() == name_lower)
    }
}

/// A single match record from any data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchRecord {
    pub date: NaiveDate,
    pub home_team: TeamId,
    pub away_team: TeamId,
    pub home_score: u8,
    pub away_score: u8,
    pub venue: Option<String>,
    pub round: Option<u8>,
    pub home_tries: Option<u8>,
    pub away_tries: Option<u8>,
    pub source: DataSource,
}

impl MatchRecord {
    /// Returns the winning team, or None for a draw
    pub fn winner(&self) -> Option<TeamId> {
        match self.home_score.cmp(&self.away_score) {
            std::cmp::Ordering::Greater => Some(self.home_team),
            std::cmp::Ordering::Less => Some(self.away_team),
            std::cmp::Ordering::Equal => None,
        }
    }

    /// Returns the score margin (positive = home win)
    pub fn margin(&self) -> i16 {
        self.home_score as i16 - self.away_score as i16
    }

    /// Check if the given team won this match
    pub fn did_win(&self, team: TeamId) -> Option<bool> {
        if team == self.home_team {
            Some(self.home_score > self.away_score)
        } else if team == self.away_team {
            Some(self.away_score > self.home_score)
        } else {
            None
        }
    }

    /// Get the opponent for a given team
    pub fn opponent(&self, team: TeamId) -> Option<TeamId> {
        if team == self.home_team {
            Some(self.away_team)
        } else if team == self.away_team {
            Some(self.home_team)
        } else {
            None
        }
    }

    /// Check if a team was playing at home
    pub fn is_home(&self, team: TeamId) -> Option<bool> {
        if team == self.home_team {
            Some(true)
        } else if team == self.away_team {
            Some(false)
        } else {
            None
        }
    }

    /// Get score for a specific team
    pub fn score_for(&self, team: TeamId) -> Option<u8> {
        if team == self.home_team {
            Some(self.home_score)
        } else if team == self.away_team {
            Some(self.away_score)
        } else {
            None
        }
    }

    /// Get score against a specific team
    pub fn score_against(&self, team: TeamId) -> Option<u8> {
        if team == self.home_team {
            Some(self.away_score)
        } else if team == self.away_team {
            Some(self.home_score)
        } else {
            None
        }
    }
}

/// Model prediction output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub home_team: TeamId,
    pub away_team: TeamId,
    pub home_win_prob: f32,
    pub predicted_home_score: f32,
    pub predicted_away_score: f32,
    pub confidence: ConfidenceLevel,
}

impl Prediction {
    /// Get the predicted winner (team with >50% win probability)
    pub fn predicted_winner(&self) -> TeamId {
        if self.home_win_prob >= 0.5 {
            self.home_team
        } else {
            self.away_team
        }
    }

    /// Get predicted margin (positive = home win)
    pub fn predicted_margin(&self) -> f32 {
        self.predicted_home_score - self.predicted_away_score
    }
}

/// Confidence level based on available match history
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    High,   // Both teams have full history
    Medium, // One team has limited history
    Low,    // Both teams have limited history
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceLevel::High => write!(f, "High"),
            ConfidenceLevel::Medium => write!(f, "Medium"),
            ConfidenceLevel::Low => write!(f, "Low"),
        }
    }
}

/// Application-wide errors
#[derive(Debug, Error)]
pub enum RugbyError {
    #[error("Scraper failed for {data_source}: {message}")]
    Scraper {
        data_source: DataSource,
        message: String,
    },

    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Unknown team: {0}")]
    UnknownTeam(String),

    #[error("Team not found with ID: {0}")]
    TeamNotFound(TeamId),

    #[error("Model not trained - run `rugby train` first")]
    NoModel,

    #[error("Insufficient history for {team}: has {matches} matches, need {required}")]
    InsufficientHistory {
        team: String,
        matches: usize,
        required: usize,
    },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, RugbyError>;

/// Application configuration loaded from config.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub training: TrainingConfig,
    pub model: ModelConfig,
    pub loss: LossConfig,
    pub data: DataConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub dropout: f64,
    pub early_stopping_patience: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_encoder_layers: usize,
    pub n_cross_attn_layers: usize,
    pub n_heads: usize,
    pub max_history: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    pub win_weight: f32,
    pub score_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub database_path: String,
    pub model_path: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            training: TrainingConfig {
                epochs: 200,
                batch_size: 64,
                learning_rate: 8.5e-4,
                weight_decay: 1e-4,
                dropout: 0.2,
                early_stopping_patience: 20,
            },
            model: ModelConfig {
                d_model: 128,
                n_encoder_layers: 4,
                n_cross_attn_layers: 2,
                n_heads: 8,
                max_history: 16,
            },
            loss: LossConfig {
                win_weight: 1.0,
                score_weight: 0.5,
            },
            data: DataConfig {
                database_path: "data/rugby.db".to_string(),
                model_path: "model/rugby_model".to_string(),
            },
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RugbyError::Config(format!("Failed to read config file {}: {}", path, e))
        })?;
        toml::from_str(&content)
            .map_err(|e| RugbyError::Config(format!("Failed to parse config: {}", e)))
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| RugbyError::Config(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
