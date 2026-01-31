//! Super Rugby Prediction CLI
//!
//! A deep learning-based match prediction tool using hierarchical transformers.

use clap::{Parser, Subcommand};
use rugby::{Config, Result};

#[derive(Parser)]
#[command(name = "rugby")]
#[command(about = "Super Rugby match prediction using deep learning", long_about = None)]
struct Cli {
    /// Config file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Data management commands
    Data {
        #[command(subcommand)]
        action: DataCommands,
    },
    /// Train the prediction model
    Train {
        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Continue from checkpoint
        #[arg(long)]
        resume: bool,
    },
    /// Predict match outcomes
    Predict {
        /// Home team name
        home: Option<String>,
        /// Away team name
        away: Option<String>,
        /// Predict all matches in a round
        #[arg(long)]
        round: Option<u8>,
        /// Input fixture file (JSON)
        #[arg(long)]
        fixture: Option<String>,
        /// Output format
        #[arg(long, default_value = "table")]
        format: OutputFormat,
    },
    /// Model management commands
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },
    /// Initialize a new project with default config
    Init,
}

#[derive(Subcommand)]
enum DataCommands {
    /// Sync data from sources
    Sync {
        /// Only sync from specific source
        #[arg(long)]
        source: Option<String>,
        /// Cache directory for HTML files
        #[arg(long)]
        cache: Option<String>,
        /// Use only cached files (no network requests)
        #[arg(long)]
        offline: bool,
    },
    /// Parse cached HTML files directly
    ParseCache {
        /// Directory containing cached HTML files
        dir: String,
    },
    /// Show database status
    Status,
}

#[derive(Subcommand)]
enum ModelCommands {
    /// Show model information
    Info,
    /// Export model for deployment
    Export {
        /// Output path
        output: String,
    },
    /// Validate model on test set
    Validate,
}

#[derive(Clone, Debug)]
enum OutputFormat {
    Table,
    Json,
    Csv,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "table" => Ok(OutputFormat::Table),
            "json" => Ok(OutputFormat::Json),
            "csv" => Ok(OutputFormat::Csv),
            _ => Err(format!("Unknown format: {}. Use table, json, or csv.", s)),
        }
    }
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp(None)
        .init();

    // Load or create config
    let config = if std::path::Path::new(&cli.config).exists() {
        match Config::load(&cli.config) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        Config::default()
    };

    // Run command
    let result = match cli.command {
        Commands::Data { action } => match action {
            DataCommands::Sync {
                source,
                cache,
                offline,
            } => commands::data_sync(&config, source, cache, offline),
            DataCommands::ParseCache { dir } => commands::parse_cache(&config, &dir),
            DataCommands::Status => commands::data_status(&config),
        },
        Commands::Train { epochs, resume } => commands::train(&config, epochs, resume),
        Commands::Predict {
            home,
            away,
            round,
            fixture,
            format,
        } => commands::predict(&config, home, away, round, fixture, format),
        Commands::Model { action } => match action {
            ModelCommands::Info => commands::model_info(&config),
            ModelCommands::Export { output } => commands::model_export(&config, &output),
            ModelCommands::Validate => commands::model_validate(&config),
        },
        Commands::Init => commands::init(&cli.config),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

mod commands {
    use super::*;
    use rugby::data::scrapers::wikipedia::WikipediaScraper;
    use rugby::data::Database;

    pub fn init(config_path: &str) -> Result<()> {
        let config = Config::default();
        config.save(config_path)?;
        println!("Created default config at {}", config_path);

        // Create data directory
        std::fs::create_dir_all("data")?;
        std::fs::create_dir_all("model")?;
        println!("Created data/ and model/ directories");

        println!("\nNext steps:");
        println!("  1. Edit {} to customize settings", config_path);
        println!("  2. Run 'rugby data sync' to fetch match data");
        println!("  3. Run 'rugby train' to train the model");
        println!("  4. Run 'rugby predict \"Team A\" \"Team B\"' to make predictions");

        Ok(())
    }

    pub fn data_sync(
        config: &Config,
        source: Option<String>,
        cache: Option<String>,
        offline: bool,
    ) -> Result<()> {
        let db = Database::open(&config.data.database_path)?;

        match source.as_deref() {
            Some("wikipedia") | None => {
                println!("Syncing from Wikipedia...");
                let mut scraper = WikipediaScraper::new();

                if let Some(cache_dir) = cache {
                    println!("Using cache directory: {}", cache_dir);
                    scraper = scraper.with_cache(&cache_dir);
                }

                if offline {
                    println!("Offline mode: using cached files only");
                    scraper = scraper.offline_only(true);
                }

                let raw_matches = scraper.fetch_all()?;
                println!("Fetched {} raw matches", raw_matches.len());

                if raw_matches.is_empty() {
                    println!("No matches found. Check the parser or cache directory.");
                    return Ok(());
                }

                // Convert to match records with team resolution
                let records = scraper.to_match_records(raw_matches, &|name, country| {
                    let team = db.get_or_create_team(name, country)?;
                    Ok(team.id)
                })?;

                let count = db.upsert_matches(&records)?;
                println!("Stored {} matches in database", count);
            }
            Some(other) => {
                println!("Unknown source: {}. Available: wikipedia", other);
            }
        }

        Ok(())
    }

    pub fn parse_cache(config: &Config, dir: &str) -> Result<()> {
        let db = Database::open(&config.data.database_path)?;

        println!("Parsing cached HTML files from {}...", dir);
        let scraper = WikipediaScraper::new();

        let raw_matches = scraper.parse_directory(dir)?;
        println!("Found {} raw matches", raw_matches.len());

        if raw_matches.is_empty() {
            println!("No matches found. Check the HTML files or parser logic.");
            return Ok(());
        }

        // Convert to match records
        let records = scraper.to_match_records(raw_matches, &|name, country| {
            let team = db.get_or_create_team(name, country)?;
            Ok(team.id)
        })?;

        let count = db.upsert_matches(&records)?;
        println!("Stored {} matches in database", count);

        Ok(())
    }

    pub fn data_status(config: &Config) -> Result<()> {
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        println!("Database Status");
        println!("───────────────────────────────");
        println!("  Path:     {}", config.data.database_path);
        println!("  Teams:    {}", stats.team_count);
        println!("  Matches:  {}", stats.match_count);
        if let (Some(earliest), Some(latest)) = (stats.earliest_match, stats.latest_match) {
            println!("  Range:    {} to {}", earliest, latest);
        }

        Ok(())
    }

    pub fn train(config: &Config, epochs: Option<usize>, _resume: bool) -> Result<()> {
        use burn::backend::{Autodiff, Wgpu};
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, RugbyDataset};
        use rugby::model::rugby_net::{RugbyNet, RugbyNetConfig};
        use rugby::training::Trainer;

        type MyBackend = Wgpu<f32, i32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let mut training_config = config.clone();
        if let Some(e) = epochs {
            training_config.training.epochs = e;
        }

        println!("Initializing training...");

        // Open database
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'rugby data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        // Create datasets with time-based split
        // Train: all data before 2023, Val: 2023, Test: 2024+
        let train_cutoff = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let val_end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

        let dataset_config = DatasetConfig {
            max_history: config.model.max_history,
            min_history: 3,
        };

        println!("Creating training dataset (before {})...", train_cutoff);
        let train_dataset =
            RugbyDataset::from_matches_before(&db, train_cutoff, dataset_config.clone())?;
        println!("  {} training samples", train_dataset.len());

        println!(
            "Creating validation dataset ({} to {})...",
            train_cutoff, val_end
        );
        let val_dataset =
            RugbyDataset::from_matches_in_range(&db, train_cutoff, val_end, dataset_config)?;
        println!("  {} validation samples", val_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Initialize device and model
        let device = burn::backend::wgpu::WgpuDevice::default();
        let model_config =
            RugbyNetConfig::from_model_config(&config.model, &training_config.training);

        println!("Creating model...");
        let model = RugbyNet::<MyAutodiffBackend>::new(&device, model_config);

        // Create trainer and train
        let trainer = Trainer::new(model, training_config.clone(), device);

        println!("\nStarting training...\n");
        let (trained_model, history) = trainer.train(train_dataset, val_dataset)?;

        // Save model
        println!("\nSaving model to {}...", config.data.model_path);
        trained_model.save(&config.data.model_path)?;

        println!("\nTraining complete!");
        println!("  Best epoch:     {}", history.best_epoch + 1);
        println!("  Best val loss:  {:.4}", history.best_val_loss);
        println!(
            "  Final accuracy: {:.1}%",
            history.val_accuracies.last().unwrap_or(&0.0) * 100.0
        );

        Ok(())
    }

    pub fn predict(
        config: &Config,
        home: Option<String>,
        away: Option<String>,
        _round: Option<u8>,
        _fixture: Option<String>,
        format: OutputFormat,
    ) -> Result<()> {
        use burn::backend::Wgpu;
        use rugby::model::rugby_net::{RugbyNet, RugbyNetConfig};
        use rugby::predict::inference::{format_prediction, Predictor};

        type MyBackend = Wgpu<f32, i32>;

        // Check for model (Burn adds .mpk extension)
        let model_file = format!("{}.mpk", config.data.model_path);
        if !std::path::Path::new(&model_file).exists() {
            return Err(rugby::RugbyError::NoModel);
        }

        // Open database
        let db = Database::open(&config.data.database_path)?;

        // Load model
        let device = burn::backend::wgpu::WgpuDevice::default();
        let model_config = RugbyNetConfig::from_model_config(&config.model, &config.training);
        let model =
            RugbyNet::<MyBackend>::load(&device, &config.data.model_path, model_config.clone())?;

        let predictor = Predictor::new(model, db, device);

        // Single match prediction
        if let (Some(home_team), Some(away_team)) = (home, away) {
            let prediction = predictor.predict(&home_team, &away_team)?;

            match format {
                OutputFormat::Table => {
                    print!("{}", format_prediction(&prediction, &home_team, &away_team));
                }
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "home": home_team,
                        "away": away_team,
                        "home_win_prob": prediction.home_win_prob,
                        "home_score": prediction.predicted_home_score,
                        "away_score": prediction.predicted_away_score,
                        "confidence": format!("{}", prediction.confidence),
                    });
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                }
                OutputFormat::Csv => {
                    println!("home,away,home_win_prob,home_score,away_score,confidence");
                    println!(
                        "{},{},{:.3},{:.0},{:.0},{}",
                        home_team,
                        away_team,
                        prediction.home_win_prob,
                        prediction.predicted_home_score,
                        prediction.predicted_away_score,
                        prediction.confidence
                    );
                }
            }
        } else {
            println!("Usage: rugby predict <HOME_TEAM> <AWAY_TEAM>");
            println!("\nExample:");
            println!("  rugby predict \"Crusaders\" \"Blues\"");
        }

        Ok(())
    }

    pub fn model_info(config: &Config) -> Result<()> {
        let model_file = format!("{}.mpk", config.data.model_path);
        if !std::path::Path::new(&model_file).exists() {
            return Err(rugby::RugbyError::NoModel);
        }

        println!("Model Information");
        println!("───────────────────────────────");
        println!("  Path:           {}", model_file);
        println!("  d_model:        {}", config.model.d_model);
        println!("  Encoder layers: {}", config.model.n_encoder_layers);
        println!("  Cross-attn:     {}", config.model.n_cross_attn_layers);
        println!("  Attention heads:{}", config.model.n_heads);
        println!("  Max history:    {}", config.model.max_history);

        Ok(())
    }

    pub fn model_export(config: &Config, output: &str) -> Result<()> {
        let model_file = format!("{}.mpk", config.data.model_path);
        if !std::path::Path::new(&model_file).exists() {
            return Err(rugby::RugbyError::NoModel);
        }

        // Copy model file
        std::fs::copy(&model_file, output)?;
        println!("Model exported to {}", output);

        Ok(())
    }

    pub fn model_validate(_config: &Config) -> Result<()> {
        println!("Validation not yet implemented");
        println!("Run 'rugby train' to see validation metrics");
        Ok(())
    }
}
