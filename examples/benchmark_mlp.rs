use burn::backend::{Autodiff, NdArray};
use rugby::{Config, Result};
use rugby::data::Database;
use rugby::data::dataset::{RugbyDataset, DatasetConfig, MatchComparison};
use rugby::model::mlp::{MLPModel, MLPConfig};
use rugby::training::MLPTrainer;
use chrono::NaiveDate;

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn run_benchmark(
    name: &str,
    config: &Config,
    mlp_config: MLPConfig,
    runs: usize,
    train_dataset: &RugbyDataset,
    val_dataset: &RugbyDataset
) -> Result<()>
{
    println!("\n=== Benchmarking {} Model ({} runs) ===", name, runs);

    let mut accuracies = Vec::new();
    let mut losses = Vec::new();
    let mut maes = Vec::new();

    for i in 0..runs {
        let device = Default::default();
        let model = MLPModel::<MyAutodiffBackend>::new(&device, mlp_config.clone());

        let trainer = MLPTrainer::<MyAutodiffBackend>::new(
            model,
            config.training.learning_rate,
            config.training.weight_decay,
            config.loss.win_weight,
            config.loss.score_weight, // Now used as margin_weight
            device,
        );

        print!("Run {}/{}: ", i + 1, runs);

        let (_model, history) = trainer.train(
            train_dataset.clone(),
            val_dataset.clone(),
            config.training.epochs,
            config.training.batch_size,
            config.training.early_stopping_patience,
        )?;

        // Get best validation metrics
        let best_idx = history.best_epoch;
        let best_acc = history.val_accuracies[best_idx];
        let best_loss = history.val_losses[best_idx];
        let best_mae = history.val_score_maes[best_idx];

        accuracies.push(best_acc);
        losses.push(best_loss);
        maes.push(best_mae);

        println!("Acc: {:.2}%, Loss: {:.4}, MAE: {:.2}", best_acc * 100.0, best_loss, best_mae);
    }

    // Compute stats
    let avg_acc = mean(&accuracies);
    let std_acc = std_dev(&accuracies, avg_acc);

    let avg_loss = mean(&losses);
    let std_loss = std_dev(&losses, avg_loss);

    let avg_mae = mean(&maes);
    let std_mae = std_dev(&maes, avg_mae);

    println!("\n--- Results for {} ---", name);
    println!("Accuracy: {:.2}% ± {:.2}%", avg_acc * 100.0, std_acc * 100.0);
    println!("Loss:     {:.4} ± {:.4}", avg_loss, std_loss);
    println!("MAE:      {:.2} ± {:.2}", avg_mae, std_mae);

    Ok(())
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64], mean: f64) -> f64 {
    let variance = data.iter().map(|value| {
        let diff = mean - *value;
        diff * diff
    }).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

fn main() -> Result<()>
{
    // Manually initialize logger to Warn to reduce spam during runs
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    // Load config
    let config = Config::default(); // Use defaults

    // Load data
    let db = Database::open(&config.data.database_path)?;
    let stats = db.get_stats()?;
    if stats.match_count == 0 {
        eprintln!("No matches found. Please run `rugby data sync` first.");
        return Ok(())
    }

    // Train/Val split
    let train_cutoff = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let val_end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    let dataset_config = DatasetConfig {
        max_history: config.model.max_history,
        min_history: 3,
    };

    println!("Loading datasets...");
    let train_dataset = RugbyDataset::from_matches_before(&db, train_cutoff, dataset_config.clone())?;
    let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
        &db,
        train_cutoff,
        val_end,
        dataset_config,
        train_dataset.score_norm,
        *train_dataset.feature_norm(),
    )?;

    println!("Train size: {}, Val size: {}", train_dataset.len(), val_dataset.len());

    // 1. Benchmark Single Hidden Layer (64)
    let single_layer_config = MLPConfig {
        input_dim: MatchComparison::DIM,
        hidden_dims: vec![64],
        dropout: 0.1,
    };
    run_benchmark("Single Layer [64]", &config, single_layer_config, 10, &train_dataset, &val_dataset)?;

    // 2. Benchmark Two Hidden Layers (128, 64) - the new default
    let two_layer_config = MLPConfig {
        input_dim: MatchComparison::DIM,
        hidden_dims: vec![128, 64],
        dropout: 0.1,
    };
    run_benchmark("Two Layers [128, 64]", &config, two_layer_config, 10, &train_dataset, &val_dataset)?;

    // 3. Benchmark with more dropout
    let high_dropout_config = MLPConfig {
        input_dim: MatchComparison::DIM,
        hidden_dims: vec![128, 64],
        dropout: 0.3,
    };
    run_benchmark("Two Layers [128, 64] + Dropout 0.3", &config, high_dropout_config, 10, &train_dataset, &val_dataset)?;

    Ok(())
}
