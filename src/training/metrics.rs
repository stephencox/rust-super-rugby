//! Training metrics and evaluation

use crate::data::dataset::ScoreNormalization;
use std::fmt;

/// Metrics accumulated during training/evaluation
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Total loss
    pub total_loss: f64,
    /// Win prediction loss component
    pub win_loss: f64,
    /// Score prediction loss component
    pub score_loss: f64,
    /// Number of correct win predictions
    pub correct_wins: usize,
    /// Total predictions
    pub total_predictions: usize,
    /// Sum of absolute home score errors (in normalized space)
    pub home_score_mae_sum: f64,
    /// Sum of absolute away score errors (in normalized space)
    pub away_score_mae_sum: f64,
    /// Number of batches accumulated
    pub batch_count: usize,
    /// Score normalization for converting MAE to original scale
    pub score_norm: ScoreNormalization,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            total_loss: 0.0,
            win_loss: 0.0,
            score_loss: 0.0,
            correct_wins: 0,
            total_predictions: 0,
            home_score_mae_sum: 0.0,
            away_score_mae_sum: 0.0,
            batch_count: 0,
            score_norm: ScoreNormalization::default(),
        }
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with specific normalization params
    pub fn with_normalization(score_norm: ScoreNormalization) -> Self {
        Metrics {
            score_norm,
            ..Self::default()
        }
    }

    /// Update metrics with a batch result
    pub fn update(
        &mut self,
        total_loss: f32,
        win_loss: f32,
        score_loss: f32,
        correct_wins: usize,
        batch_size: usize,
        home_score_mae: f32,
        away_score_mae: f32,
    ) {
        self.total_loss += total_loss as f64;
        self.win_loss += win_loss as f64;
        self.score_loss += score_loss as f64;
        self.correct_wins += correct_wins;
        self.total_predictions += batch_size;
        self.home_score_mae_sum += home_score_mae as f64 * batch_size as f64;
        self.away_score_mae_sum += away_score_mae as f64 * batch_size as f64;
        self.batch_count += 1;
    }

    /// Get average total loss
    pub fn avg_loss(&self) -> f64 {
        if self.batch_count == 0 {
            0.0
        } else {
            self.total_loss / self.batch_count as f64
        }
    }

    /// Get average win loss
    pub fn avg_win_loss(&self) -> f64 {
        if self.batch_count == 0 {
            0.0
        } else {
            self.win_loss / self.batch_count as f64
        }
    }

    /// Get average score loss
    pub fn avg_score_loss(&self) -> f64 {
        if self.batch_count == 0 {
            0.0
        } else {
            self.score_loss / self.batch_count as f64
        }
    }

    /// Get win prediction accuracy
    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.correct_wins as f64 / self.total_predictions as f64
        }
    }

    /// Get home score MAE (scaled back to original score range)
    pub fn home_score_mae(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            // MAE in normalized space * std = MAE in original space
            (self.home_score_mae_sum / self.total_predictions as f64) * self.score_norm.std as f64
        }
    }

    /// Get away score MAE (scaled back to original score range)
    pub fn away_score_mae(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            (self.away_score_mae_sum / self.total_predictions as f64) * self.score_norm.std as f64
        }
    }

    /// Get combined score MAE (in original score range)
    pub fn score_mae(&self) -> f64 {
        (self.home_score_mae() + self.away_score_mae()) / 2.0
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Merge another metrics instance
    pub fn merge(&mut self, other: &Metrics) {
        self.total_loss += other.total_loss;
        self.win_loss += other.win_loss;
        self.score_loss += other.score_loss;
        self.correct_wins += other.correct_wins;
        self.total_predictions += other.total_predictions;
        self.home_score_mae_sum += other.home_score_mae_sum;
        self.away_score_mae_sum += other.away_score_mae_sum;
        self.batch_count += other.batch_count;
    }
}

impl fmt::Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Loss: {:.4} (win: {:.4}, score: {:.4}) | Acc: {:.2}% | Score MAE: {:.2}",
            self.avg_loss(),
            self.avg_win_loss(),
            self.avg_score_loss(),
            self.accuracy() * 100.0,
            self.score_mae()
        )
    }
}

/// Training history for tracking progress
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub train_accuracies: Vec<f64>,
    pub val_accuracies: Vec<f64>,
    pub train_score_maes: Vec<f64>,
    pub val_score_maes: Vec<f64>,
    pub best_val_loss: f64,
    pub best_epoch: usize,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            best_val_loss: f64::INFINITY,
            ..Default::default()
        }
    }

    /// Record metrics for an epoch
    pub fn record_epoch(&mut self, epoch: usize, train: &Metrics, val: &Metrics) {
        self.train_losses.push(train.avg_loss());
        self.val_losses.push(val.avg_loss());
        self.train_accuracies.push(train.accuracy());
        self.val_accuracies.push(val.accuracy());
        self.train_score_maes.push(train.score_mae());
        self.val_score_maes.push(val.score_mae());

        if val.avg_loss() < self.best_val_loss {
            self.best_val_loss = val.avg_loss();
            self.best_epoch = epoch;
        }
    }

    /// Check if we should early stop
    pub fn should_early_stop(&self, patience: usize) -> bool {
        if self.val_losses.len() < patience {
            return false;
        }
        let current_epoch = self.val_losses.len() - 1;
        current_epoch - self.best_epoch >= patience
    }

    /// Get improvement from last epoch
    pub fn last_improvement(&self) -> Option<f64> {
        if self.val_losses.len() < 2 {
            return None;
        }
        let n = self.val_losses.len();
        Some(self.val_losses[n - 2] - self.val_losses[n - 1])
    }
}
