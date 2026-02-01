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
    /// Train the prediction model (transformer architecture)
    Train {
        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Continue from checkpoint
        #[arg(long)]
        resume: bool,
    },
    /// Train the MLP baseline model (comparison features only)
    TrainMlp {
        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Learning rate (0.1 matches Python baseline)
        #[arg(long, default_value = "0.1")]
        lr: f64,
    },
    /// Train LSTM model (uses team history sequences)
    TrainLstm {
        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Learning rate
        #[arg(long, default_value = "0.01")]
        lr: f64,
        /// Hidden size for LSTM
        #[arg(long, default_value = "64")]
        hidden_size: usize,
    },
    /// Train Transformer model (RugbyNet with cross-attention)
    TrainTransformer {
        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Learning rate
        #[arg(long, default_value = "0.0001")]
        lr: f64,
        /// Model dimension
        #[arg(long, default_value = "64")]
        d_model: usize,
        /// Batch size (0 for auto)
        #[arg(long, default_value = "32")]
        batch_size: usize,
    },
    /// Tune MLP hyperparameters with time-based train/val/test splits
    TuneMlp {
        /// Maximum epochs to try
        #[arg(long, default_value = "500")]
        max_epochs: usize,
    },
    /// Tune LSTM hyperparameters with time-based train/val/test splits
    TuneLstm {
        /// Maximum epochs to try
        #[arg(long, default_value = "200")]
        max_epochs: usize,
    },
    /// Tune Transformer hyperparameters with time-based train/val/test splits
    TuneTransformer {
        /// Maximum epochs to try
        #[arg(long, default_value = "100")]
        max_epochs: usize,
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
        Commands::TrainMlp { epochs, lr } => commands::train_mlp(&config, epochs, lr),
        Commands::TrainLstm { epochs, lr, hidden_size } => {
            commands::train_lstm(&config, epochs, lr, hidden_size)
        }
        Commands::TrainTransformer {
            epochs,
            lr,
            d_model,
            batch_size,
        } => commands::train_transformer(&config, epochs, lr, d_model, batch_size),
        Commands::TuneMlp { max_epochs } => commands::tune_mlp(&config, max_epochs),
        Commands::TuneLstm { max_epochs } => commands::tune_lstm(&config, max_epochs),
        Commands::TuneTransformer { max_epochs } => commands::tune_transformer(&config, max_epochs),
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
            "  Score normalization: mean={:.1}, std={:.1}",
            train_dataset.score_norm.mean, train_dataset.score_norm.std
        );

        println!(
            "Creating validation dataset ({} to {})...",
            train_cutoff, val_end
        );
        // Use same normalization as training set
        let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            train_cutoff,
            val_end,
            dataset_config,
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
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

    pub fn train_mlp(config: &Config, epochs: Option<usize>, lr: f64) -> Result<()> {
        use burn::backend::{Autodiff, NdArray};
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, RugbyDataset};
        use rugby::training::SimpleMLPTrainer;

        // Use NdArray backend (same as debug_training.rs)
        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let epochs = epochs.unwrap_or(100);

        println!("Initializing Simple MLP training...");

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
        let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            train_cutoff,
            val_end,
            dataset_config,
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} validation samples", val_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Create simple trainer (single Linear layer, like debug_training.rs)
        let device = Default::default();
        let trainer = SimpleMLPTrainer::<MyAutodiffBackend>::new(device, lr);

        println!("Learning rate: {}", lr);
        println!("\nStarting Simple MLP training...\n");

        let _history = trainer.train(train_dataset, val_dataset, epochs)?;

        println!("\nSimple MLP Training complete!");

        Ok(())
    }

    pub fn train_lstm(
        config: &Config,
        epochs: Option<usize>,
        lr: f64,
        hidden_size: usize,
    ) -> Result<()> {
        use burn::backend::{Autodiff, NdArray};
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, RugbyDataset};
        use rugby::model::lstm::LSTMConfig;
        use rugby::training::LSTMTrainer;

        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let epochs = epochs.unwrap_or(100);

        println!("Initializing LSTM training...");

        // Open database
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'rugby data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        // Create datasets
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
        let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            train_cutoff,
            val_end,
            dataset_config,
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} validation samples", val_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Create LSTM config
        let lstm_config = LSTMConfig {
            input_dim: 15, // MatchFeatures::DIM
            hidden_size,
            num_layers: 1,
            bidirectional: false,
            comparison_dim: 15,
        };

        println!("LSTM config:");
        println!("  Input dim: {}", lstm_config.input_dim);
        println!("  Hidden size: {}", lstm_config.hidden_size);
        println!("  Learning rate: {}", lr);

        // Create trainer
        let device = Default::default();
        let trainer = LSTMTrainer::<MyAutodiffBackend>::new(device, lstm_config, lr);

        println!("\nStarting LSTM training...\n");

        let history = trainer.train(train_dataset, val_dataset, epochs)?;

        println!("\nLSTM Training complete!");
        println!("  Best epoch: {}", history.best_epoch + 1);
        println!(
            "  Best val accuracy: {:.1}%",
            history.val_accuracies.last().unwrap_or(&0.0) * 100.0
        );

        Ok(())
    }

    pub fn train_transformer(
        config: &Config,
        epochs: Option<usize>,
        lr: f64,
        d_model: usize,
        batch_size: usize,
    ) -> Result<()> {
        use burn::backend::{Autodiff, NdArray};
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, RugbyDataset};
        use rugby::model::rugby_net::RugbyNetConfig;
        use rugby::training::TransformerTrainer;

        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let epochs = epochs.unwrap_or(50);

        println!("Initializing Transformer training...");

        // Open database
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'rugby data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        // Create datasets
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
        let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            train_cutoff,
            val_end,
            dataset_config,
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} validation samples", val_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Create Transformer config
        let transformer_config = RugbyNetConfig {
            input_dim: 15,
            d_model,
            n_heads: d_model / 16, // Ensure divisible
            n_encoder_layers: 2,
            n_cross_attn_layers: 1,
            d_ff: d_model * 4,
            head_hidden_dim: d_model / 2,
            dropout: 0.1,
            max_seq_len: config.model.max_history,
            n_teams: 64,
            team_embed_dim: 16,
        };

        println!("Transformer config:");
        println!("  d_model: {}", transformer_config.d_model);
        println!("  n_heads: {}", transformer_config.n_heads);
        println!("  n_encoder_layers: {}", transformer_config.n_encoder_layers);
        println!("  Learning rate: {}", lr);
        println!("  Batch size: {}", batch_size);

        // Create trainer
        let device = Default::default();
        let trainer = TransformerTrainer::<MyAutodiffBackend>::new(device, transformer_config, lr);

        println!("\nStarting Transformer training...\n");

        let history = trainer.train(train_dataset, val_dataset, epochs, batch_size)?;

        println!("\nTransformer Training complete!");
        println!("  Best epoch: {}", history.best_epoch + 1);
        println!(
            "  Best val accuracy: {:.1}%",
            history.val_accuracies.last().unwrap_or(&0.0) * 100.0
        );

        Ok(())
    }

    pub fn tune_mlp(config: &Config, max_epochs: usize) -> Result<()> {
        use burn::backend::{Autodiff, NdArray};
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, RugbyDataset};
        use rugby::training::{MLPHyperparams, MLPTuner, RandomSplitDatasets};

        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        println!("Initializing MLP hyperparameter tuning (time-based splits)...");

        // Open database
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'rugby data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        // Time-based splits (same as train_mlp)
        let train_cutoff = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let val_end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let test_end = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

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
        let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            train_cutoff,
            val_end,
            dataset_config.clone(),
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} validation samples", val_dataset.len());

        println!(
            "Creating test dataset ({} to {})...",
            val_end, test_end
        );
        let test_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            val_end,
            test_end,
            dataset_config,
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} test samples", test_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Wrap in RandomSplitDatasets struct (reusing the tuner interface)
        let datasets = RandomSplitDatasets {
            train: train_dataset,
            val: val_dataset,
            test: test_dataset,
        };

        // Define hyperparameter search space
        let hyperparams_list = vec![
            // Different learning rates at 100 epochs
            MLPHyperparams { learning_rate: 0.01, epochs: 100 },
            MLPHyperparams { learning_rate: 0.05, epochs: 100 },
            MLPHyperparams { learning_rate: 0.1, epochs: 100 },
            MLPHyperparams { learning_rate: 0.2, epochs: 100 },
            MLPHyperparams { learning_rate: 0.5, epochs: 100 },
            MLPHyperparams { learning_rate: 1.0, epochs: 100 },
            // More epochs for promising learning rates
            MLPHyperparams { learning_rate: 0.1, epochs: 200 },
            MLPHyperparams { learning_rate: 0.1, epochs: max_epochs },
            MLPHyperparams { learning_rate: 0.05, epochs: max_epochs },
            MLPHyperparams { learning_rate: 0.2, epochs: max_epochs },
        ];

        println!("\nTesting {} hyperparameter configurations...\n", hyperparams_list.len());

        // Create tuner and run
        let device = Default::default();
        let tuner = MLPTuner::<MyAutodiffBackend>::new(device);
        let results = tuner.tune(&datasets, hyperparams_list)?;

        // Find best result by validation accuracy
        let best = results
            .iter()
            .max_by(|a, b| a.val_acc.partial_cmp(&b.val_acc).unwrap())
            .unwrap();

        println!("\n=== Tuning Results (Time-Based Splits) ===\n");
        println!("{:>8} {:>8} {:>10} {:>10} {:>10}", "LR", "Epochs", "Train%", "Val%", "Test%");
        println!("{}", "-".repeat(50));

        for r in &results {
            let marker = if r.val_acc == best.val_acc { " *" } else { "" };
            println!(
                "{:>8.4} {:>8} {:>9.1}% {:>9.1}% {:>9.1}%{}",
                r.hyperparams.learning_rate,
                r.hyperparams.epochs,
                r.train_acc * 100.0,
                r.val_acc * 100.0,
                r.test_acc * 100.0,
                marker
            );
        }

        println!("\nBest configuration:");
        println!("  Learning rate: {}", best.hyperparams.learning_rate);
        println!("  Epochs: {}", best.hyperparams.epochs);
        println!("  Validation accuracy: {:.1}%", best.val_acc * 100.0);
        println!("  Test accuracy: {:.1}%", best.test_acc * 100.0);

        Ok(())
    }

    pub fn tune_lstm(config: &Config, max_epochs: usize) -> Result<()> {
        use burn::backend::{Autodiff, NdArray};
        use burn::data::dataloader::DataLoaderBuilder;
        use burn::nn::Linear;
        use burn::optim::{GradientsParams, Optimizer, SgdConfig};
        use burn::tensor::activation::sigmoid;
        use burn::tensor::ElementConversion;
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, MatchBatcher, RugbyDataset};
        use rugby::model::lstm::{LSTMConfig, LSTMModel};
        use rugby::training::mlp_trainer::ComparisonNormalization;

        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        println!("Initializing LSTM hyperparameter tuning (time-based splits)...");

        // Open database
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'rugby data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        // Time-based splits
        let train_cutoff = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let val_end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let test_end = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

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
        let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            train_cutoff,
            val_end,
            dataset_config.clone(),
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} validation samples", val_dataset.len());

        println!(
            "Creating test dataset ({} to {})...",
            val_end, test_end
        );
        let test_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            val_end,
            test_end,
            dataset_config,
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} test samples", test_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Compute feature normalization
        let feature_norm = ComparisonNormalization::from_dataset(&train_dataset);

        // Define hyperparameter search space: (learning_rate, hidden_size, epochs)
        let hyperparams: Vec<(f64, usize, usize)> = vec![
            // Different learning rates with hidden_size=64
            (0.001, 64, 100),
            (0.005, 64, 100),
            (0.01, 64, 100),
            (0.02, 64, 100),
            (0.05, 64, 100),
            // Different hidden sizes with lr=0.01
            (0.01, 32, 100),
            (0.01, 128, 100),
            // More epochs for promising configs
            (0.01, 64, max_epochs),
            (0.005, 64, max_epochs),
        ];

        println!("\nTesting {} hyperparameter configurations...\n", hyperparams.len());

        // Results storage
        struct LstmResult {
            lr: f64,
            hidden_size: usize,
            epochs: usize,
            train_acc: f32,
            val_acc: f32,
            test_acc: f32,
        }
        let mut results: Vec<LstmResult> = Vec::new();

        let device: <MyAutodiffBackend as burn::tensor::backend::Backend>::Device = Default::default();

        for (lr, hidden_size, epochs) in &hyperparams {
            println!("Testing lr={}, hidden_size={}, epochs={}", lr, hidden_size, epochs);

            // Create model
            let lstm_config = LSTMConfig {
                input_dim: 15,
                hidden_size: *hidden_size,
                num_layers: 1,
                bidirectional: false,
                comparison_dim: 15,
            };
            let mut model = LSTMModel::<MyAutodiffBackend>::new(&device, lstm_config);
            let mut optimizer = SgdConfig::new().init();

            // Create data loaders
            let batcher = MatchBatcher::<MyAutodiffBackend>::new(device.clone());
            let train_loader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(train_dataset.len())
                .build(train_dataset.clone());
            let val_loader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(val_dataset.len())
                .build(val_dataset.clone());
            let test_loader = DataLoaderBuilder::new(batcher)
                .batch_size(test_dataset.len())
                .build(test_dataset.clone());

            let mut best_val_acc = 0.0f32;
            let mut best_model = model.clone();

            // Training loop
            for _epoch in 0..*epochs {
                let train_batch = train_loader.iter().next().unwrap();
                let comparison = feature_norm.normalize(train_batch.comparison.clone());
                let y_train = train_batch.home_win.clone().unsqueeze_dim(1);

                // Forward
                let (win_logit, _, _) = model.forward(
                    train_batch.home_history.clone(),
                    train_batch.away_history.clone(),
                    comparison,
                );
                let probs = sigmoid(win_logit);

                // Loss
                let eps = 1e-7;
                let probs_clamped = probs.clone().clamp(eps, 1.0 - eps);
                let loss = y_train.clone().neg() * probs_clamped.clone().log()
                    - (y_train.clone().neg() + 1.0) * (probs_clamped.neg() + 1.0).log();
                let loss = loss.mean();

                // Backward
                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(*lr, model, grads_params);

                // Check validation
                let val_batch = val_loader.iter().next().unwrap();
                let val_comparison = feature_norm.normalize(val_batch.comparison.clone());
                let (val_logit, _, _) = model.forward(
                    val_batch.home_history.clone(),
                    val_batch.away_history.clone(),
                    val_comparison,
                );
                let val_probs = sigmoid(val_logit);
                let val_probs_data = val_probs.clone().into_data();
                let val_targets_data = val_batch.home_win.clone().into_data();
                let val_probs_slice: &[f32] = val_probs_data.as_slice().unwrap();
                let val_targets_slice: &[f32] = val_targets_data.as_slice().unwrap();
                let val_correct = val_probs_slice
                    .iter()
                    .zip(val_targets_slice.iter())
                    .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
                    .count();
                let val_acc = val_correct as f32 / val_probs_slice.len() as f32;

                if val_acc > best_val_acc {
                    best_val_acc = val_acc;
                    best_model = model.clone();
                }
            }

            // Evaluate best model
            let train_batch = train_loader.iter().next().unwrap();
            let comparison = feature_norm.normalize(train_batch.comparison.clone());
            let (train_logit, _, _) = best_model.forward(
                train_batch.home_history.clone(),
                train_batch.away_history.clone(),
                comparison,
            );
            let train_probs = sigmoid(train_logit);
            let train_probs_data = train_probs.into_data();
            let train_targets_data = train_batch.home_win.clone().into_data();
            let train_probs_slice: &[f32] = train_probs_data.as_slice().unwrap();
            let train_targets_slice: &[f32] = train_targets_data.as_slice().unwrap();
            let train_correct = train_probs_slice
                .iter()
                .zip(train_targets_slice.iter())
                .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
                .count();
            let train_acc = train_correct as f32 / train_probs_slice.len() as f32;

            let test_batch = test_loader.iter().next().unwrap();
            let test_comparison = feature_norm.normalize(test_batch.comparison.clone());
            let (test_logit, _, _) = best_model.forward(
                test_batch.home_history.clone(),
                test_batch.away_history.clone(),
                test_comparison,
            );
            let test_probs = sigmoid(test_logit);
            let test_probs_data = test_probs.into_data();
            let test_targets_data = test_batch.home_win.clone().into_data();
            let test_probs_slice: &[f32] = test_probs_data.as_slice().unwrap();
            let test_targets_slice: &[f32] = test_targets_data.as_slice().unwrap();
            let test_correct = test_probs_slice
                .iter()
                .zip(test_targets_slice.iter())
                .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
                .count();
            let test_acc = test_correct as f32 / test_probs_slice.len() as f32;

            println!(
                "  train_acc={:.1}%, val_acc={:.1}%, test_acc={:.1}%",
                train_acc * 100.0,
                best_val_acc * 100.0,
                test_acc * 100.0
            );

            results.push(LstmResult {
                lr: *lr,
                hidden_size: *hidden_size,
                epochs: *epochs,
                train_acc,
                val_acc: best_val_acc,
                test_acc,
            });
        }

        // Find best result
        let best = results
            .iter()
            .max_by(|a, b| a.val_acc.partial_cmp(&b.val_acc).unwrap())
            .unwrap();

        println!("\n=== LSTM Tuning Results (Time-Based Splits) ===\n");
        println!(
            "{:>8} {:>8} {:>8} {:>10} {:>10} {:>10}",
            "LR", "Hidden", "Epochs", "Train%", "Val%", "Test%"
        );
        println!("{}", "-".repeat(60));

        for r in &results {
            let marker = if r.val_acc == best.val_acc { " *" } else { "" };
            println!(
                "{:>8.4} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.1}%{}",
                r.lr,
                r.hidden_size,
                r.epochs,
                r.train_acc * 100.0,
                r.val_acc * 100.0,
                r.test_acc * 100.0,
                marker
            );
        }

        println!("\nBest configuration:");
        println!("  Learning rate: {}", best.lr);
        println!("  Hidden size: {}", best.hidden_size);
        println!("  Epochs: {}", best.epochs);
        println!("  Validation accuracy: {:.1}%", best.val_acc * 100.0);
        println!("  Test accuracy: {:.1}%", best.test_acc * 100.0);

        Ok(())
    }

    pub fn tune_transformer(config: &Config, max_epochs: usize) -> Result<()> {
        use burn::backend::ndarray::NdArray;
        use burn::backend::Autodiff;
        use burn::data::dataloader::DataLoaderBuilder;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};
        use burn::tensor::activation::sigmoid;
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, MatchBatcher, RugbyDataset};
        use rugby::model::rugby_net::{RugbyNet, RugbyNetConfig};
        use rugby::training::mlp_trainer::ComparisonNormalization;

        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        println!("Initializing Transformer hyperparameter tuning (time-based splits)...");

        // Open database
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'rugby data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        // Time-based splits
        let train_cutoff = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let val_end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let test_end = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

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
        let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            train_cutoff,
            val_end,
            dataset_config.clone(),
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} validation samples", val_dataset.len());

        println!(
            "Creating test dataset ({} to {})...",
            val_end, test_end
        );
        let test_dataset = RugbyDataset::from_matches_in_range_with_norm(
            &db,
            val_end,
            test_end,
            dataset_config,
            train_dataset.score_norm,
            *train_dataset.feature_norm(),
        )?;
        println!("  {} test samples", test_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Compute feature normalization
        let feature_norm = ComparisonNormalization::from_dataset(&train_dataset);

        // Define hyperparameter search space: (learning_rate, d_model, batch_size, epochs)
        let hyperparams: Vec<(f64, usize, usize, usize)> = vec![
            // Different learning rates
            (0.0001, 64, 32, 50),
            (0.0005, 64, 32, 50),
            (0.001, 64, 32, 50),
            (0.002, 64, 32, 50),
            // Different model sizes
            (0.001, 32, 32, 50),
            (0.001, 128, 32, 50),
            // Different batch sizes
            (0.001, 64, 64, 50),
            (0.001, 64, 128, 50),
            // More epochs for promising configs
            (0.001, 64, 32, max_epochs),
        ];

        println!("\nTesting {} hyperparameter configurations...\n", hyperparams.len());

        // Results storage
        struct TransformerResult {
            lr: f64,
            d_model: usize,
            batch_size: usize,
            epochs: usize,
            train_acc: f32,
            val_acc: f32,
            test_acc: f32,
        }
        let mut results: Vec<TransformerResult> = Vec::new();

        let device = Default::default();
        println!("Using CPU (NdArray) backend");

        for (lr, d_model, batch_size, epochs) in &hyperparams {
            println!("Testing lr={}, d_model={}, batch_size={}, epochs={}", lr, d_model, batch_size, epochs);

            // Create model config
            let transformer_config = RugbyNetConfig {
                input_dim: 15,
                d_model: *d_model,
                n_heads: (*d_model / 16).max(2),
                n_encoder_layers: 2,
                n_cross_attn_layers: 1,
                d_ff: *d_model * 4,
                head_hidden_dim: *d_model / 2,
                dropout: 0.1,
                max_seq_len: config.model.max_history,
                n_teams: 64,
                team_embed_dim: 16,
            };

            let mut model = RugbyNet::<MyAutodiffBackend>::new(&device, transformer_config);
            let mut optimizer = AdamConfig::new().init();

            // Create data loaders
            let batcher = MatchBatcher::<MyAutodiffBackend>::new(device.clone());
            let train_loader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(*batch_size)
                .shuffle(42)
                .build(train_dataset.clone());
            let val_loader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(val_dataset.len())
                .build(val_dataset.clone());
            let test_loader = DataLoaderBuilder::new(batcher)
                .batch_size(test_dataset.len())
                .build(test_dataset.clone());

            let mut best_val_acc = 0.0f32;
            let mut best_model = model.clone();

            // Training loop
            for _epoch in 0..*epochs {
                for train_batch in train_loader.iter() {
                    let comparison = feature_norm.normalize(train_batch.comparison.clone());
                    let y_train = train_batch.home_win.clone().unsqueeze_dim(1);

                    // Forward
                    let predictions = model.forward(
                        train_batch.home_history.clone(),
                        train_batch.away_history.clone(),
                        Some(train_batch.home_mask.clone()),
                        Some(train_batch.away_mask.clone()),
                        Some(train_batch.home_team_id.clone()),
                        Some(train_batch.away_team_id.clone()),
                        Some(comparison),
                    );
                    let probs = sigmoid(predictions.win_logit.clone());

                    // Loss (BCE)
                    let eps = 1e-7;
                    let probs_clamped = probs.clamp(eps, 1.0 - eps);
                    let loss = y_train.clone().neg() * probs_clamped.clone().log()
                        - (y_train.clone().neg() + 1.0) * (probs_clamped.neg() + 1.0).log();
                    let loss = loss.mean();

                    // Backward
                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(*lr, model, grads_params);
                }

                // Check validation every epoch
                let val_batch = val_loader.iter().next().unwrap();
                let val_comparison = feature_norm.normalize(val_batch.comparison.clone());
                let val_preds = model.forward(
                    val_batch.home_history.clone(),
                    val_batch.away_history.clone(),
                    Some(val_batch.home_mask.clone()),
                    Some(val_batch.away_mask.clone()),
                    Some(val_batch.home_team_id.clone()),
                    Some(val_batch.away_team_id.clone()),
                    Some(val_comparison),
                );
                let val_probs = sigmoid(val_preds.win_logit);
                let val_probs_data = val_probs.clone().into_data();
                let val_targets_data = val_batch.home_win.clone().into_data();
                let val_probs_slice: &[f32] = val_probs_data.as_slice().unwrap();
                let val_targets_slice: &[f32] = val_targets_data.as_slice().unwrap();
                let val_correct = val_probs_slice
                    .iter()
                    .zip(val_targets_slice.iter())
                    .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
                    .count();
                let val_acc = val_correct as f32 / val_probs_slice.len() as f32;

                if val_acc > best_val_acc {
                    best_val_acc = val_acc;
                    best_model = model.clone();
                }
            }

            // Evaluate best model on train and test
            let train_batch = train_loader.iter().next().unwrap();
            let comparison = feature_norm.normalize(train_batch.comparison.clone());
            let train_preds = best_model.forward(
                train_batch.home_history.clone(),
                train_batch.away_history.clone(),
                Some(train_batch.home_mask.clone()),
                Some(train_batch.away_mask.clone()),
                Some(train_batch.home_team_id.clone()),
                Some(train_batch.away_team_id.clone()),
                Some(comparison),
            );
            let train_probs = sigmoid(train_preds.win_logit);
            let train_probs_data = train_probs.into_data();
            let train_targets_data = train_batch.home_win.clone().into_data();
            let train_probs_slice: &[f32] = train_probs_data.as_slice().unwrap();
            let train_targets_slice: &[f32] = train_targets_data.as_slice().unwrap();
            let train_correct = train_probs_slice
                .iter()
                .zip(train_targets_slice.iter())
                .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
                .count();
            let train_acc = train_correct as f32 / train_probs_slice.len() as f32;

            let test_batch = test_loader.iter().next().unwrap();
            let test_comparison = feature_norm.normalize(test_batch.comparison.clone());
            let test_preds = best_model.forward(
                test_batch.home_history.clone(),
                test_batch.away_history.clone(),
                Some(test_batch.home_mask.clone()),
                Some(test_batch.away_mask.clone()),
                Some(test_batch.home_team_id.clone()),
                Some(test_batch.away_team_id.clone()),
                Some(test_comparison),
            );
            let test_probs = sigmoid(test_preds.win_logit);
            let test_probs_data = test_probs.into_data();
            let test_targets_data = test_batch.home_win.clone().into_data();
            let test_probs_slice: &[f32] = test_probs_data.as_slice().unwrap();
            let test_targets_slice: &[f32] = test_targets_data.as_slice().unwrap();
            let test_correct = test_probs_slice
                .iter()
                .zip(test_targets_slice.iter())
                .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
                .count();
            let test_acc = test_correct as f32 / test_probs_slice.len() as f32;

            println!(
                "  train_acc={:.1}%, val_acc={:.1}%, test_acc={:.1}%",
                train_acc * 100.0,
                best_val_acc * 100.0,
                test_acc * 100.0
            );

            results.push(TransformerResult {
                lr: *lr,
                d_model: *d_model,
                batch_size: *batch_size,
                epochs: *epochs,
                train_acc,
                val_acc: best_val_acc,
                test_acc,
            });
        }

        // Find best result
        let best = results
            .iter()
            .max_by(|a, b| a.val_acc.partial_cmp(&b.val_acc).unwrap())
            .unwrap();

        println!("\n=== Transformer Tuning Results (Time-Based Splits) ===\n");
        println!(
            "{:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}",
            "LR", "d_model", "Batch", "Epochs", "Train%", "Val%", "Test%"
        );
        println!("{}", "-".repeat(72));

        for r in &results {
            let marker = if r.val_acc == best.val_acc { " *" } else { "" };
            println!(
                "{:>8.4} {:>8} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.1}%{}",
                r.lr,
                r.d_model,
                r.batch_size,
                r.epochs,
                r.train_acc * 100.0,
                r.val_acc * 100.0,
                r.test_acc * 100.0,
                marker
            );
        }

        println!("\nBest configuration:");
        println!("  Learning rate: {}", best.lr);
        println!("  d_model: {}", best.d_model);
        println!("  Batch size: {}", best.batch_size);
        println!("  Epochs: {}", best.epochs);
        println!("  Validation accuracy: {:.1}%", best.val_acc * 100.0);
        println!("  Test accuracy: {:.1}%", best.test_acc * 100.0);

        Ok(())
    }
}
