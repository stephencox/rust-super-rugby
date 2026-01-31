// Quick debug script to check comparison feature values
// Run with: cargo run --example debug_features

use rugby::data::dataset::{DatasetConfig, RugbyDataset};
use rugby::data::Database;
use burn::data::dataset::Dataset;

fn main() -> rugby::Result<()> {
    let db = Database::open("data/rugby.db")?;

    let config = DatasetConfig {
        max_history: 10,
        min_history: 3,
    };

    let cutoff = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let dataset = RugbyDataset::from_matches_before(&db, cutoff, config)?;

    println!("Dataset has {} samples", dataset.len());

    // Check first 20 samples
    println!("\nFirst 20 samples:");
    println!("  win_rate_diff | margin_diff | pyth_diff | log5 | is_local | home_win");
    println!("  ------------- | ----------- | --------- | ---- | -------- | --------");

    let mut positive_count = 0;
    let mut negative_count = 0;

    for i in 0..dataset.len().min(20) {
        if let Some(sample) = dataset.get(i) {
            let c = &sample.comparison;
            println!(
                "  {:+.3}         | {:+.3}      | {:+.3}   | {:.3} | {:.0}      | {:.0}",
                c.win_rate_diff, c.margin_diff, c.pythagorean_diff, c.log5, c.is_local, sample.home_win
            );

            if sample.home_win > 0.5 {
                positive_count += 1;
            } else {
                negative_count += 1;
            }
        }
    }

    // Statistics across all samples
    println!("\nStatistics across all {} samples:", dataset.len());

    let mut sum_wr_diff = 0.0;
    let mut sum_margin_diff = 0.0;
    let mut sum_log5 = 0.0;
    let mut min_wr_diff = f32::MAX;
    let mut max_wr_diff = f32::MIN;
    let mut total_home_wins = 0;

    for i in 0..dataset.len() {
        if let Some(sample) = dataset.get(i) {
            sum_wr_diff += sample.comparison.win_rate_diff;
            sum_margin_diff += sample.comparison.margin_diff;
            sum_log5 += sample.comparison.log5;
            min_wr_diff = min_wr_diff.min(sample.comparison.win_rate_diff);
            max_wr_diff = max_wr_diff.max(sample.comparison.win_rate_diff);
            if sample.home_win > 0.5 {
                total_home_wins += 1;
            }
        }
    }

    let n = dataset.len() as f32;
    println!("  win_rate_diff: mean={:.4}, min={:.4}, max={:.4}", sum_wr_diff / n, min_wr_diff, max_wr_diff);
    println!("  margin_diff: mean={:.4}", sum_margin_diff / n);
    println!("  log5: mean={:.4}", sum_log5 / n);
    println!("  Home wins: {}/{} ({:.1}%)", total_home_wins, dataset.len(), 100.0 * total_home_wins as f32 / n);

    Ok(())
}
