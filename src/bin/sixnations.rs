//! Six Nations Prediction CLI
//!
//! A deep learning-based match prediction tool for Six Nations rugby.

use clap::{Parser, Subcommand};
use rugby::data::dataset::MatchComparison;
use rugby::{Config, Result};

#[derive(Parser)]
#[command(name = "sixnations")]
#[command(about = "Six Nations match prediction using deep learning", long_about = None)]
struct Cli {
    /// Config file path
    #[arg(short, long, default_value = "sixnations-config.toml")]
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
    /// Train the MLP model
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
    /// Initialize a new project with default config
    Init,
}

#[derive(Subcommand)]
enum DataCommands {
    /// Sync data from sources
    Sync {
        /// Cache directory for HTML files
        #[arg(long)]
        cache: Option<String>,
        /// Use only cached files (no network requests)
        #[arg(long)]
        offline: bool,
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

    // Load or create config with Six Nations defaults
    let config = if std::path::Path::new(&cli.config).exists() {
        match Config::load(&cli.config) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Use Six Nations specific defaults
        let mut config = Config::default();
        config.data.database_path = "data/sixnations.db".to_string();
        config.data.model_path = "model/sixnations_model".to_string();
        config
    };

    // Run command
    let result = match cli.command {
        Commands::Data { action } => match action {
            DataCommands::Sync { cache, offline } => commands::data_sync(&config, cache, offline),
            DataCommands::Fixtures { year } => commands::data_fixtures(&config, year),
            DataCommands::Status => commands::data_status(&config),
        },
        Commands::Train { epochs, lr, optimizer, init, production } => {
            commands::train(&config, epochs, lr, &optimizer, &init, production)
        }
        Commands::Predict { home, away } => commands::predict(&config, &home, &away),
        Commands::PredictNext { year, round, format } => {
            commands::predict_next(&config, year, round, format)
        }
        Commands::Init => commands::init(&cli.config),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

mod commands {
    use super::*;
    use rugby::data::scrapers::sixnations::SixNationsScraper;
    use rugby::data::Database;

    pub fn init(config_path: &str) -> Result<()> {
        let mut config = Config::default();
        config.data.database_path = "data/sixnations.db".to_string();
        config.data.model_path = "model/sixnations_model".to_string();
        config.save(config_path)?;
        println!("Created default config at {}", config_path);

        std::fs::create_dir_all("data")?;
        std::fs::create_dir_all("model")?;
        println!("Created data/ and model/ directories");

        println!("\nNext steps:");
        println!("  1. Edit {} to customize settings", config_path);
        println!("  2. Run 'sixnations data sync' to fetch match data");
        println!("  3. Run 'sixnations train' to train the model");
        println!("  4. Run 'sixnations predict \"England\" \"France\"' to make predictions");

        Ok(())
    }

    pub fn data_sync(config: &Config, cache: Option<String>, offline: bool) -> Result<()> {
        let db = Database::open(&config.data.database_path)?;

        println!("Syncing Six Nations data from Wikipedia...");
        let mut scraper = SixNationsScraper::new();

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
        println!("Fetching {} Six Nations fixtures from Wikipedia...", year);

        let scraper = SixNationsScraper::new();
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
            println!("Train it with: sixnations train --epochs 200");
            return Err(rugby::RugbyError::NoModel);
        }

        println!("Fetching {} Six Nations fixtures...", year);
        let scraper = SixNationsScraper::new();
        let fixtures = scraper.fetch_fixtures(year)?;

        if fixtures.is_empty() {
            println!("No upcoming fixtures found for {}.", year);
            return Ok(());
        }

        // Filter by round if specified
        let fixtures: Vec<_> = if let Some(r) = round_filter {
            fixtures.into_iter().filter(|f| f.round == Some(r)).collect()
        } else {
            let min_round = fixtures.iter().filter_map(|f| f.round).min();
            if let Some(r) = min_round {
                println!("Predicting Round {} fixtures...\n", r);
                fixtures.into_iter().filter(|f| f.round == Some(r)).collect()
            } else {
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

        // Load MLP model
        let device: <MyBackend as burn::tensor::backend::Backend>::Device = Default::default();
        use burn::module::Module;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model: Linear<MyBackend> = LinearConfig::new(15, 2)
            .init(&device)
            .load_file(&model_path, &recorder, &device)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to load model: {}", e)))?;

        // Load normalization
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

        // Helper to compute team summary
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

            let home_history = db.get_recent_team_matches(home_team.id, 16)?;
            let away_history = db.get_recent_team_matches(away_team.id, 16)?;

            if home_history.len() < 3 || away_history.len() < 3 {
                eprintln!("Warning: Insufficient history for {} vs {}",
                    f.home_team.name, f.away_team.name);
                continue;
            }

            let home_summary = compute_summary(&home_history, home_team.id);
            let away_summary = compute_summary(&away_history, away_team.id);
            let is_local = home_team.country == away_team.country;
            let comparison = MatchComparison::from_summaries(&home_summary, &away_summary, is_local);

            let comparison_tensor = Tensor::<MyBackend, 1>::from_floats(
                comparison.to_vec().as_slice(),
                &device,
            ).reshape([1, MatchComparison::DIM]);

            let normalized = feature_norm.normalize(comparison_tensor);
            let output = model.forward(normalized);

            let output_data = output.into_data();
            let output_slice: &[f32] = output_data.as_slice().unwrap();
            let win_logit = output_slice[0];
            let margin_normalized = output_slice[1];

            let margin = margin_normalized * score_std;
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
                println!("\n{} Six Nations - Round {} Predictions\n", year, round_num);

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

                    println!("  {}  {:10} vs {:10}", fixture.date, fixture.home_team.name, fixture.away_team.name);
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

        let optimizer_type = match optimizer.to_lowercase().as_str() {
            "sgd" => OptimizerType::Sgd,
            "adam" => OptimizerType::Adam,
            _ => {
                println!("Unknown optimizer '{}', using Adam", optimizer);
                OptimizerType::Adam
            }
        };

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

        let db = Database::open(&config.data.database_path)?;
        let stats = db.get_stats()?;

        if stats.match_count == 0 {
            return Err(rugby::RugbyError::Config(
                "No matches in database. Run 'sixnations data sync' first.".to_string(),
            ));
        }

        println!("Loaded {} matches from database", stats.match_count);

        let dataset_config = DatasetConfig {
            max_history: config.model.max_history,
            min_history: 3,
        };

        // Six Nations: train on pre-2023, validate on 2023
        // For production: train on all through 2025
        let (train_dataset, val_dataset) = if production {
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

        let score_norm = train_dataset.score_norm;

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

        let model_path = format!("{}_mlp", config.data.model_path);
        println!("\nSaving model to {}...", model_path);

        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        trained_model
            .clone()
            .save_file(&model_path, &recorder)
            .map_err(|e| rugby::RugbyError::Io(std::io::Error::other(format!("Failed to save model: {}", e))))?;

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
        use burn::tensor::Tensor;
        use rugby::data::dataset::{MatchComparison, TeamSummary};
        use rugby::training::mlp_trainer::ComparisonNormalization;

        type MyBackend = NdArray<f32>;

        let model_path = format!("{}_mlp", config.data.model_path);
        let model_file = format!("{}.mpk", model_path);
        let norm_file = format!("{}_norm.json", model_path);

        if !std::path::Path::new(&model_file).exists() {
            println!("Model not found at {}", model_file);
            println!("Train it with: sixnations train --epochs 200");
            return Err(rugby::RugbyError::NoModel);
        }

        let db = Database::open(&config.data.database_path)?;

        let device: <MyBackend as burn::tensor::backend::Backend>::Device = Default::default();
        use burn::module::Module;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model: Linear<MyBackend> = LinearConfig::new(15, 2)
            .init(&device)
            .load_file(&model_path, &recorder, &device)
            .map_err(|e| rugby::RugbyError::Config(format!("Failed to load model: {}", e)))?;

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

        let home_team = db.find_team_by_name(home)?
            .ok_or_else(|| rugby::RugbyError::UnknownTeam(home.to_string()))?;
        let away_team = db.find_team_by_name(away)?
            .ok_or_else(|| rugby::RugbyError::UnknownTeam(away.to_string()))?;

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

        let comparison_tensor = Tensor::<MyBackend, 1>::from_floats(
            comparison.to_vec().as_slice(),
            &device,
        ).reshape([1, MatchComparison::DIM]);

        let normalized = feature_norm.normalize(comparison_tensor);
        let output = model.forward(normalized);

        let output_data = output.into_data();
        let output_slice: &[f32] = output_data.as_slice().unwrap();
        let win_logit = output_slice[0];
        let margin_normalized = output_slice[1];

        let margin = margin_normalized * score_std;
        let home_win_prob = 1.0 / (1.0 + (-win_logit).exp());

        let winner = if home_win_prob >= 0.5 { home } else { away };
        let win_prob = if home_win_prob >= 0.5 { home_win_prob } else { 1.0 - home_win_prob };
        let margin_abs = margin.abs();

        println!("\n  {} vs {}", home, away);
        println!("  → {} by {:.0} pts ({:.1}%)\n", winner, margin_abs, win_prob * 100.0);

        Ok(())
    }
}
