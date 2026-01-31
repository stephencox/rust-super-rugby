//! Export training data to CSV for analysis
//! Run with: cargo run --example export_data

use burn::data::dataset::Dataset;
use rugby::data::dataset::{DatasetConfig, RugbyDataset};
use rugby::data::Database;
use std::fs::File;
use std::io::Write;

fn main() -> rugby::Result<()> {
    let db = Database::open("data/rugby.db")?;

    let config = DatasetConfig {
        max_history: 10,
        min_history: 3,
    };

    // Training data
    let train_cutoff = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let train_dataset = RugbyDataset::from_matches_before(&db, train_cutoff, config.clone())?;

    // Validation data
    let val_end = chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let val_dataset = RugbyDataset::from_matches_in_range_with_norm(
        &db,
        train_cutoff,
        val_end,
        config,
        train_dataset.score_norm,
        *train_dataset.feature_norm(),
    )?;

    // Export training data
    let mut train_file = File::create("train_data.csv")?;
    writeln!(train_file, "win_rate_diff,margin_diff,pythagorean_diff,log5,is_local,home_win,home_score,away_score")?;

    for i in 0..train_dataset.len() {
        if let Some(sample) = train_dataset.get(i) {
            let c = &sample.comparison;
            writeln!(
                train_file,
                "{:.6},{:.6},{:.6},{:.6},{:.0},{:.0},{:.6},{:.6}",
                c.win_rate_diff, c.margin_diff, c.pythagorean_diff, c.log5, c.is_local,
                sample.home_win, sample.home_score, sample.away_score
            )?;
        }
    }
    println!("Exported {} training samples to train_data.csv", train_dataset.len());

    // Export validation data
    let mut val_file = File::create("val_data.csv")?;
    writeln!(val_file, "win_rate_diff,margin_diff,pythagorean_diff,log5,is_local,home_win,home_score,away_score")?;

    for i in 0..val_dataset.len() {
        if let Some(sample) = val_dataset.get(i) {
            let c = &sample.comparison;
            writeln!(
                val_file,
                "{:.6},{:.6},{:.6},{:.6},{:.0},{:.0},{:.6},{:.6}",
                c.win_rate_diff, c.margin_diff, c.pythagorean_diff, c.log5, c.is_local,
                sample.home_win, sample.home_score, sample.away_score
            )?;
        }
    }
    println!("Exported {} validation samples to val_data.csv", val_dataset.len());

    // Print summary stats
    println!("\nTraining data summary:");
    let mut home_wins = 0;
    let mut sum_log5 = 0.0;
    for i in 0..train_dataset.len() {
        if let Some(sample) = train_dataset.get(i) {
            if sample.home_win > 0.5 {
                home_wins += 1;
            }
            sum_log5 += sample.comparison.log5;
        }
    }
    println!("  Home wins: {}/{} ({:.1}%)", home_wins, train_dataset.len(), 100.0 * home_wins as f32 / train_dataset.len() as f32);
    println!("  Mean log5: {:.4}", sum_log5 / train_dataset.len() as f32);

    println!("\nValidation data summary:");
    let mut home_wins = 0;
    let mut sum_log5 = 0.0;
    for i in 0..val_dataset.len() {
        if let Some(sample) = val_dataset.get(i) {
            if sample.home_win > 0.5 {
                home_wins += 1;
            }
            sum_log5 += sample.comparison.log5;
        }
    }
    println!("  Home wins: {}/{} ({:.1}%)", home_wins, val_dataset.len(), 100.0 * home_wins as f32 / val_dataset.len() as f32);
    println!("  Mean log5: {:.4}", sum_log5 / val_dataset.len() as f32);

    Ok(())
}
