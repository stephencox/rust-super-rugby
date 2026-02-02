//! Super Rugby Prediction CLI
//!
//! A deep learning-based match prediction tool using hierarchical transformers.

use clap::{Parser, Subcommand};
use rugby::data::dataset::MatchComparison;
use rugby::features::MatchFeatures;
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
    /// Train the MLP model (comparison features only)
    Train {
        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Learning rate (0.1 for SGD, 0.001 for Adam)
        #[arg(long, default_value = "0.001")]
        lr: f64,
        /// Optimizer: sgd or adam
        #[arg(long, default_value = "adam")]
        optimizer: String,
        /// Weight initialization: default, small, or zero
        #[arg(long, default_value = "default")]
        init: String,
        /// Train on all data through 2025 for production predictions
        #[arg(long)]
        production: bool,
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
    /// Train Transformer (RugbyNet) model
    TrainTransformer {
        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Learning rate
        #[arg(long, default_value = "0.0001")]
        lr: f64,
        /// Batch size (0 for full batch)
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
    /// Predict match outcomes for a single match
    Predict {
        /// Home team name
        home: String,
        /// Away team name
        away: String,
    },
    /// Predict upcoming fixtures from Wikipedia
    PredictNext {
        /// Year of the season (default: 2026)
        #[arg(long, default_value = "2026")]
        year: u16,
        /// Only show fixtures for specific round
        #[arg(long)]
        round: Option<u8>,
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
    /// Fetch upcoming fixtures for a season
    Fixtures {
        /// Year to fetch fixtures for (default: current year)
        #[arg(default_value = "2026")]
        year: u16,
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
            DataCommands::Fixtures { year } => commands::data_fixtures(&config, year),
            DataCommands::Status => commands::data_status(&config),
        },
        Commands::Train { epochs, lr, optimizer, init, production } => {
            commands::train(&config, epochs, lr, &optimizer, &init, production)
        }
        Commands::TrainLstm { epochs, lr, hidden_size } => {
            commands::train_lstm(&config, epochs, lr, hidden_size)
        }
        Commands::TrainTransformer { epochs, lr, batch_size } => {
            commands::train_transformer(&config, epochs, lr, batch_size)
        }
        Commands::TuneMlp { max_epochs } => commands::tune_mlp(&config, max_epochs),
        Commands::TuneLstm { max_epochs } => commands::tune_lstm(&config, max_epochs),
        Commands::Predict { home, away } => commands::predict(&config, &home, &away),
        Commands::PredictNext { year, round, format } => {
            commands::predict_next(&config, year, round, format)
        }
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

    pub fn data_fixtures(_config: &Config, year: u16) -> Result<()> {
        println!("Fetching {} Super Rugby Pacific fixtures from Wikipedia...", year);

        let scraper = WikipediaScraper::new();
        let fixtures = scraper.fetch_fixtures(year)?;

        if fixtures.is_empty() {
            println!("No upcoming fixtures found for {}.", year);
            println!("Note: Only fixtures WITHOUT scores are returned (upcoming matches).");
            return Ok(());
        }

        println!("\nFound {} upcoming fixtures for {}:\n", fixtures.len(), year);

        // Group by round
        let mut by_round: std::collections::BTreeMap<Option<u8>, Vec<_>> = std::collections::BTreeMap::new();
        for f in &fixtures {
            by_round.entry(f.round).or_default().push(f);
        }

        for (round, round_fixtures) in by_round {
            match round {
                Some(r) => println!("Round {}:", r),
                None => println!("Round TBD:"),
            }
            for f in round_fixtures {
                println!(
                    "  {} - {} vs {}",
                    f.date, f.home_team.name, f.away_team.name
                );
                if let Some(venue) = &f.venue {
                    println!("         at {}", venue);
                }
            }
            println!();
        }

        Ok(())
    }

    pub fn predict_next(
        config: &Config,
        year: u16,
        round_filter: Option<u8>,
        format: OutputFormat,
    ) -> Result<()> {
        use burn::backend::NdArray;
        use burn::nn::{Linear, LinearConfig};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        use burn::tensor::activation::sigmoid;
        use burn::tensor::Tensor;
        use rugby::data::dataset::{MatchComparison, TeamSummary};
        use rugby::training::mlp_trainer::ComparisonNormalization;

        type MyBackend = NdArray<f32>;

        // Check for MLP model
        let model_path = format!("{}_mlp", config.data.model_path);
        let model_file = format!("{}.mpk", model_path);
        let norm_file = format!("{}_norm.json", model_path);

        if !std::path::Path::new(&model_file).exists() {
            println!("MLP model not found at {}", model_file);
            println!("Train it with: rugby train-mlp --epochs 200 --lr 0.1");
            return Err(rugby::RugbyError::NoModel);
        }

        println!("Fetching {} Super Rugby Pacific fixtures...", year);
        let scraper = WikipediaScraper::new();
        let fixtures = scraper.fetch_fixtures(year)?;

        if fixtures.is_empty() {
            println!("No upcoming fixtures found for {}.", year);
            return Ok(());
        }

        // Filter by round if specified
        let fixtures: Vec<_> = if let Some(r) = round_filter {
            fixtures.into_iter().filter(|f| f.round == Some(r)).collect()
        } else {
            // Find the earliest round with fixtures
            let min_round = fixtures.iter().filter_map(|f| f.round).min();
            if let Some(r) = min_round {
                println!("Predicting Round {} fixtures...\n", r);
                fixtures.into_iter().filter(|f| f.round == Some(r)).collect()
            } else {
                // No round info, take earliest matches by date
                let min_date = fixtures.iter().map(|f| f.date).min().unwrap();
                fixtures.into_iter().filter(|f| f.date == min_date).collect()
            }
        };

        if fixtures.is_empty() {
            println!("No fixtures found for the specified round.");
            return Ok(());
        }

        // Open database
        let db = Database::open(&config.data.database_path)?;

        // Load MLP model (2 outputs: win_logit and margin)
        let device: <MyBackend as burn::tensor::backend::Backend>::Device = Default::default();
        use burn::module::Module;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model: Linear<MyBackend> = LinearConfig::new(15, 2)
            .init(&device)
            .load_file(&model_path, &recorder, &device)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to load model: {}", e)))?;

        // Load normalization (now includes score_mean and score_std)
        #[derive(serde::Deserialize)]
        struct NormParams {
            feature_mean: Vec<f32>,
            feature_std: Vec<f32>,
            score_mean: f32,
            score_std: f32,
        }
        let norm_json = std::fs::read_to_string(&norm_file)?;
        let norm_params: NormParams = serde_json::from_str(&norm_json)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to load normalization: {}", e)))?;
        let feature_norm = ComparisonNormalization {
            mean: norm_params.feature_mean,
            std: norm_params.feature_std,
        };
        let score_std = norm_params.score_std;

        // Helper to compute team summary from match history
        let compute_summary = |history: &[rugby::MatchRecord], team: rugby::TeamId| -> TeamSummary {
            let window = 10;
            let recent: Vec<_> = history.iter().rev().take(window).collect();

            if recent.len() < 3 {
                return TeamSummary::default();
            }

            let mut wins = 0.0f32;
            let mut total_pf = 0.0f32;
            let mut total_pa = 0.0f32;
            let mut margins = Vec::new();

            for m in &recent {
                let is_home = m.home_team == team;
                let (pf, pa) = if is_home {
                    (m.home_score as f32, m.away_score as f32)
                } else {
                    (m.away_score as f32, m.home_score as f32)
                };

                let margin = pf - pa;
                margins.push(margin);
                total_pf += pf;
                total_pa += pa;

                if margin > 0.0 {
                    wins += 1.0;
                } else if margin == 0.0 {
                    wins += 0.5;
                }
            }

            let n = recent.len() as f32;
            let win_rate = wins / n;
            let margin_avg = margins.iter().sum::<f32>() / n;
            let pf_avg = total_pf / n;
            let pa_avg = total_pa / n;

            let exp = 2.37f32;
            let pf_pow = total_pf.powf(exp);
            let pa_pow = total_pa.powf(exp);
            let pythagorean = if pf_pow + pa_pow > 0.0 {
                pf_pow / (pf_pow + pa_pow)
            } else {
                0.5
            };

            TeamSummary {
                win_rate,
                margin_avg,
                pythagorean,
                pf_avg,
                pa_avg,
            }
        };

        // Make predictions
        struct SimplePrediction {
            home_win_prob: f32,
            margin: f32,
        }

        let mut predictions: Vec<(_, SimplePrediction)> = Vec::new();
        for f in &fixtures {
            // Look up teams
            let home_team = match db.find_team_by_name(&f.home_team.name)? {
                Some(t) => t,
                None => {
                    eprintln!("Warning: Unknown team {}", f.home_team.name);
                    continue;
                }
            };
            let away_team = match db.find_team_by_name(&f.away_team.name)? {
                Some(t) => t,
                None => {
                    eprintln!("Warning: Unknown team {}", f.away_team.name);
                    continue;
                }
            };

            // Get match history
            let home_history = db.get_recent_team_matches(home_team.id, 16)?;
            let away_history = db.get_recent_team_matches(away_team.id, 16)?;

            if home_history.len() < 3 || away_history.len() < 3 {
                eprintln!("Warning: Insufficient history for {} vs {}",
                    f.home_team.name, f.away_team.name);
                continue;
            }

            // Compute summaries and comparison
            let home_summary = compute_summary(&home_history, home_team.id);
            let away_summary = compute_summary(&away_history, away_team.id);
            let is_local = home_team.country == away_team.country;
            let comparison = MatchComparison::from_summaries(&home_summary, &away_summary, is_local);

            // Create tensor and predict
            let comparison_tensor = Tensor::<MyBackend, 1>::from_floats(
                comparison.to_vec().as_slice(),
                &device,
            ).reshape([1, 15]);

            let normalized = feature_norm.normalize(comparison_tensor);
            let output = model.forward(normalized);

            // Split output: column 0 is win_logit, column 1 is margin (normalized)
            let output_data = output.into_data();
            let output_slice: &[f32] = output_data.as_slice().unwrap();
            let win_logit = output_slice[0];
            let margin_normalized = output_slice[1];

            // Denormalize margin
            let margin = margin_normalized * score_std;

            // Compute win probability from logit
            let home_win_prob = 1.0 / (1.0 + (-win_logit).exp());

            predictions.push((f, SimplePrediction { home_win_prob, margin }));
        }

        if predictions.is_empty() {
            println!("Could not make any predictions (teams may lack sufficient history).");
            return Ok(());
        }

        // Output results
        match format {
            OutputFormat::Table => {
                let round_num = fixtures.first().and_then(|f| f.round).unwrap_or(1);
                println!("\n{} Super Rugby Pacific - Round {} Predictions\n", year, round_num);

                for (fixture, pred) in &predictions {
                    let winner = if pred.home_win_prob >= 0.5 {
                        &fixture.home_team.name
                    } else {
                        &fixture.away_team.name
                    };
                    let win_prob = if pred.home_win_prob >= 0.5 {
                        pred.home_win_prob
                    } else {
                        1.0 - pred.home_win_prob
                    };
                    let margin_abs = pred.margin.abs();

                    println!("  {}  {:15} vs {:15}", fixture.date, fixture.home_team.name, fixture.away_team.name);
                    println!("           → {} by {:.0} pts ({:.1}%)\n", winner, margin_abs, win_prob * 100.0);
                }
            }
            OutputFormat::Json => {
                let json_preds: Vec<_> = predictions.iter().map(|(f, pred)| {
                    serde_json::json!({
                        "date": f.date.to_string(),
                        "round": f.round,
                        "home": f.home_team.name,
                        "away": f.away_team.name,
                        "home_win_prob": pred.home_win_prob,
                        "predicted_margin": pred.margin.round() as i32,
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&json_preds).unwrap());
            }
            OutputFormat::Csv => {
                println!("date,round,home,away,home_win_prob,margin");
                for (f, pred) in &predictions {
                    println!(
                        "{},{},{},{},{:.3},{:.0}",
                        f.date,
                        f.round.map(|r| r.to_string()).unwrap_or_default(),
                        f.home_team.name,
                        f.away_team.name,
                        pred.home_win_prob,
                        pred.margin,
                    );
                }
            }
        }

        Ok(())
    }

    pub fn train(config: &Config, epochs: Option<usize>, lr: f64, optimizer: &str, init: &str, production: bool) -> Result<()> {
        use burn::backend::{Autodiff, NdArray};
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, RugbyDataset};
        use rugby::training::{SimpleMLPTrainer, OptimizerType, InitMethod};

        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let epochs = epochs.unwrap_or(200);

        // Parse optimizer type
        let optimizer_type = match optimizer.to_lowercase().as_str() {
            "sgd" => OptimizerType::Sgd,
            "adam" => OptimizerType::Adam,
            _ => {
                println!("Unknown optimizer '{}', using Adam", optimizer);
                OptimizerType::Adam
            }
        };

        // Parse init method
        let init_method = match init.to_lowercase().as_str() {
            "default" => InitMethod::Default,
            "small" => InitMethod::Small,
            "zero" => InitMethod::Zero,
            _ => {
                println!("Unknown init method '{}', using default", init);
                InitMethod::Default
            }
        };

        if production {
            println!("Initializing PRODUCTION multi-task MLP training...");
            println!("  Training on ALL data through 2025");
        } else {
            println!("Initializing multi-task MLP training...");
        }
        println!("  Optimizer: {:?}", optimizer_type);
        println!("  Init method: {:?}", init_method);

        // Open database
        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'rugby data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        let dataset_config = DatasetConfig {
            max_history: config.model.max_history,
            min_history: 3,
        };

        // Create datasets - production mode uses all data through 2025
        let (train_dataset, val_dataset) = if production {
            // Production: train on all data through 2025, use 2025 as validation
            let train_cutoff = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
            let val_start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

            println!("Creating training dataset (before {})...", train_cutoff);
            let train_dataset =
                RugbyDataset::from_matches_before(&db, train_cutoff, dataset_config.clone())?;
            println!("  {} training samples", train_dataset.len());

            println!("Creating validation dataset (2025 season)...");
            let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
                &db,
                val_start,
                train_cutoff,
                dataset_config,
                train_dataset.score_norm,
                *train_dataset.feature_norm(),
            )?;
            println!("  {} validation samples", val_dataset.len());

            (train_dataset, val_dataset)
        } else {
            // Development: train on pre-2023, validate on 2023
            let train_cutoff = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
            let val_end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

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

            (train_dataset, val_dataset)
        };

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training.".to_string(),
            ));
        }

        // Store score_norm for saving
        let score_norm = train_dataset.score_norm;

        // Create trainer
        let device: <MyBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let trainer = SimpleMLPTrainer::<MyAutodiffBackend>::new(
            device.clone(), lr, optimizer_type, init_method
        );

        println!("Learning rate: {}", lr);
        println!("\nStarting MLP training...\n");

        let (history, trained_model, feature_norm, margin_mae) =
            trainer.train(train_dataset, val_dataset, epochs)?;

        println!("\nMulti-task MLP Training complete!");
        println!("  Margin MAE: {:.1} points", margin_mae);

        // Save the model and normalization
        let model_path = format!("{}_mlp", config.data.model_path);
        println!("\nSaving model to {}...", model_path);

        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        trained_model
            .clone()
            .save_file(&model_path, &recorder)
            .map_err(|e| rugby::RugbyError::Io(std::io::Error::other(format!("Failed to save model: {}", e))))?;

        // Save normalization parameters (feature norm + score norm for margin)
        #[derive(serde::Serialize)]
        struct NormParams {
            feature_mean: Vec<f32>,
            feature_std: Vec<f32>,
            score_mean: f32,
            score_std: f32,
        }
        let norm_params = NormParams {
            feature_mean: feature_norm.mean.clone(),
            feature_std: feature_norm.std.clone(),
            score_mean: score_norm.mean,
            score_std: score_norm.std,
        };
        let norm_path = format!("{}_norm.json", model_path);
        let norm_json = serde_json::to_string_pretty(&norm_params)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to serialize normalization: {}", e)))?;
        std::fs::write(&norm_path, norm_json)?;

        println!("Model saved to {}.mpk", model_path);
        println!("Normalization saved to {}", norm_path);

        Ok(())
    }

    pub fn predict(config: &Config, home: &str, away: &str) -> Result<()> {
        use burn::backend::NdArray;
        use burn::nn::{Linear, LinearConfig};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        use burn::tensor::activation::sigmoid;
        use burn::tensor::Tensor;
        use rugby::data::dataset::{MatchComparison, TeamSummary};
        use rugby::training::mlp_trainer::ComparisonNormalization;

        type MyBackend = NdArray<f32>;

        // Check for MLP model
        let model_path = format!("{}_mlp", config.data.model_path);
        let model_file = format!("{}.mpk", model_path);
        let norm_file = format!("{}_norm.json", model_path);

        if !std::path::Path::new(&model_file).exists() {
            println!("Model not found at {}", model_file);
            println!("Train it with: rugby train --epochs 200 --lr 0.1");
            return Err(rugby::RugbyError::NoModel);
        }

        // Open database
        let db = Database::open(&config.data.database_path)?;

        // Load model (2 outputs: win_logit and margin)
        let device: <MyBackend as burn::tensor::backend::Backend>::Device = Default::default();
        use burn::module::Module;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model: Linear<MyBackend> = LinearConfig::new(15, 2)
            .init(&device)
            .load_file(&model_path, &recorder, &device)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to load model: {}", e)))?;

        // Load normalization (now includes score_mean and score_std)
        #[derive(serde::Deserialize)]
        struct NormParams {
            feature_mean: Vec<f32>,
            feature_std: Vec<f32>,
            score_mean: f32,
            score_std: f32,
        }
        let norm_json = std::fs::read_to_string(&norm_file)?;
        let norm_params: NormParams = serde_json::from_str(&norm_json)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to load normalization: {}", e)))?;
        let feature_norm = ComparisonNormalization {
            mean: norm_params.feature_mean,
            std: norm_params.feature_std,
        };
        let score_std = norm_params.score_std;

        // Look up teams
        let home_team = db.find_team_by_name(home)?
            .ok_or_else(|| rugby::RugbyError::UnknownTeam(home.to_string()))?;
        let away_team = db.find_team_by_name(away)?
            .ok_or_else(|| rugby::RugbyError::UnknownTeam(away.to_string()))?;

        // Get match history
        let home_history = db.get_recent_team_matches(home_team.id, 16)?;
        let away_history = db.get_recent_team_matches(away_team.id, 16)?;

        if home_history.len() < 3 {
            return Err(rugby::RugbyError::InsufficientHistory {
                team: home.to_string(),
                matches: home_history.len(),
                required: 3,
            });
        }
        if away_history.len() < 3 {
            return Err(rugby::RugbyError::InsufficientHistory {
                team: away.to_string(),
                matches: away_history.len(),
                required: 3,
            });
        }

        // Compute team summaries
        let compute_summary = |history: &[rugby::MatchRecord], team: rugby::TeamId| -> TeamSummary {
            let window = 10;
            let recent: Vec<_> = history.iter().rev().take(window).collect();
            if recent.len() < 3 { return TeamSummary::default(); }

            let mut wins = 0.0f32;
            let mut total_pf = 0.0f32;
            let mut total_pa = 0.0f32;
            let mut margins = Vec::new();

            for m in &recent {
                let is_home = m.home_team == team;
                let (pf, pa) = if is_home {
                    (m.home_score as f32, m.away_score as f32)
                } else {
                    (m.away_score as f32, m.home_score as f32)
                };
                let margin = pf - pa;
                margins.push(margin);
                total_pf += pf;
                total_pa += pa;
                if margin > 0.0 { wins += 1.0; }
                else if margin == 0.0 { wins += 0.5; }
            }

            let n = recent.len() as f32;
            let exp = 2.37f32;
            let pf_pow = total_pf.powf(exp);
            let pa_pow = total_pa.powf(exp);

            TeamSummary {
                win_rate: wins / n,
                margin_avg: margins.iter().sum::<f32>() / n,
                pythagorean: if pf_pow + pa_pow > 0.0 { pf_pow / (pf_pow + pa_pow) } else { 0.5 },
                pf_avg: total_pf / n,
                pa_avg: total_pa / n,
            }
        };

        let home_summary = compute_summary(&home_history, home_team.id);
        let away_summary = compute_summary(&away_history, away_team.id);
        let is_local = home_team.country == away_team.country;
        let comparison = MatchComparison::from_summaries(&home_summary, &away_summary, is_local);

        // Predict
        let comparison_tensor = Tensor::<MyBackend, 1>::from_floats(
            comparison.to_vec().as_slice(),
            &device,
        ).reshape([1, 15]);

        let normalized = feature_norm.normalize(comparison_tensor);
        let output = model.forward(normalized);

        // Split output: column 0 is win_logit, column 1 is margin (normalized)
        let output_data = output.into_data();
        let output_slice: &[f32] = output_data.as_slice().unwrap();
        let win_logit = output_slice[0];
        let margin_normalized = output_slice[1];

        // Denormalize margin
        let margin = margin_normalized * score_std;

        // Compute win probability from logit
        let home_win_prob = 1.0 / (1.0 + (-win_logit).exp());

        // Display result
        let winner = if home_win_prob >= 0.5 { home } else { away };
        let win_prob = if home_win_prob >= 0.5 { home_win_prob } else { 1.0 - home_win_prob };
        let margin_abs = margin.abs();

        println!("\n  {} vs {}", home, away);
        println!("  → {} by {:.0} pts ({:.1}%)\n", winner, margin_abs, win_prob * 100.0);

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

    pub fn train_transformer(
        config: &Config,
        epochs: Option<usize>,
        lr: f64,
        batch_size: usize,
    ) -> Result<()> {
        use burn::backend::{Autodiff, NdArray};
        use chrono::NaiveDate;
        use rugby::data::dataset::{DatasetConfig, RugbyDataset};
        use rugby::model::rugby_net::RugbyNetConfig;
        use rugby::training::TransformerTrainer;
        use rugby::training::mlp_trainer::ComparisonNormalization;

        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let epochs = epochs.unwrap_or(100);

        println!("Initializing Transformer (RugbyNet) training...");

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
        // Important: Reusing training normalization AND MAPPING
        let val_dataset = RugbyDataset::from_matches_with_norm(
            &db,
            db.get_matches_in_range(train_cutoff, val_end)?,
            dataset_config,
            Some(train_dataset.score_norm),
            Some(*train_dataset.feature_norm()),
            Some(&train_dataset.team_mapping),
        )?;
        println!("  {} validation samples", val_dataset.len());

        if train_dataset.is_empty() || val_dataset.is_empty() {
            return Err(rugby::RugbyError::Config(
                "Not enough data for training. Need matches before and after 2023.".to_string(),
            ));
        }

        // Configure RugbyNet
        let mut model_config = RugbyNetConfig::from_model_config(&config.model, &config.training);
        
        // Update n_teams based on actual data
        let n_teams = train_dataset.team_mapping.len();
        println!("  Found {} teams in training data", n_teams);
        // Add buffer for potential new teams (though they will be mapped to 0/unknown)
        model_config.n_teams = n_teams + 5; 

        // Create trainer
        let device = Default::default();
        let trainer = TransformerTrainer::<MyAutodiffBackend>::new(device, model_config, lr);

        println!("\nStarting Transformer training...\n");

        let (history, trained_model) = trainer.train(train_dataset.clone(), val_dataset, epochs, batch_size)?;

        println!("\nTransformer Training complete!");
        println!("  Best epoch: {}", history.best_epoch + 1);
        println!(
            "  Best val accuracy: {:.1}%",
            history.val_accuracies.last().unwrap_or(&0.0) * 100.0
        );

        // Save model
        let model_path = format!("{}_transformer", config.data.model_path);
        println!("\nSaving model to {}...", model_path);

        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        trained_model
            .clone()
            .save_file(&model_path, &recorder)
            .map_err(|e| rugby::RugbyError::Io(std::io::Error::other(format!("Failed to save model: {}", e))))?;

        // Save normalization AND MAPPING
        // Compute comparison normalization for metadata (same as Trainer does internally)
        let comparison_norm = ComparisonNormalization::from_dataset(&train_dataset);

        #[derive(serde::Serialize)]
        struct MetaParams {
            feature_mean: Vec<f32>,
            feature_std: Vec<f32>,
            score_mean: f32,
            score_std: f32,
            team_mapping: std::collections::HashMap<i64, u32>,
        }
        let meta = MetaParams {
            feature_mean: comparison_norm.mean,
            feature_std: comparison_norm.std,
            score_mean: train_dataset.score_norm.mean,
            score_std: train_dataset.score_norm.std,
            team_mapping: train_dataset.team_mapping.clone(),
        };
        let meta_path = format!("{}_meta.json", model_path);
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to serialize metadata: {}", e)))?;
        std::fs::write(&meta_path, meta_json)?;

        println!("Model saved to {}.mpk", model_path);
        println!("Metadata saved to {}", meta_path);

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
            input_dim: MatchFeatures::DIM,
            hidden_size,
            num_layers: 1,
            bidirectional: false,
            comparison_dim: MatchComparison::DIM,
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
        
        use burn::optim::{GradientsParams, Optimizer, SgdConfig};
        use burn::tensor::activation::sigmoid;
        
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
                input_dim: MatchFeatures::DIM,
                hidden_size: *hidden_size,
                num_layers: 1,
                bidirectional: false,
                comparison_dim: MatchComparison::DIM,
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
}
