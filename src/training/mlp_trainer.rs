//! Training loop for improved MLP model
//!
//! Uses Adam optimizer and predicts win probability + absolute margin.

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::data::dataset::{MatchBatch, MatchBatcher, MatchComparison, RugbyDataset, ScoreNormalization};
use crate::model::mlp::{MLPModel, MLPPredictions};
use crate::training::metrics::{Metrics, TrainingHistory};
use crate::Result;

/// Feature normalization for comparison features (z-score normalization)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComparisonNormalization {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl ComparisonNormalization {
    /// Dimension of comparison features (matches MatchComparison::DIM)
    pub const DIM: usize = MatchComparison::DIM;

    /// Compute from training dataset
    pub fn from_dataset(dataset: &RugbyDataset) -> Self {
        use burn::data::dataset::Dataset;

        let mut sum = vec![0.0f32; Self::DIM];
        let mut sum_sq = vec![0.0f32; Self::DIM];

        for i in 0..dataset.len() {
            if let Some(sample) = dataset.get(i) {
                let vals = sample.comparison.to_vec();
                for j in 0..Self::DIM {
                    sum[j] += vals[j];
                    sum_sq[j] += vals[j] * vals[j];
                }
            }
        }

        let n = dataset.len() as f32;
        let mean: Vec<f32> = sum.iter().map(|s| s / n).collect();
        let std: Vec<f32> = sum_sq
            .iter()
            .zip(mean.iter())
            .map(|(sq, m)| ((sq / n - m * m).sqrt()).max(0.001))
            .collect();

        ComparisonNormalization { mean, std }
    }

    /// Normalize a comparison tensor using z-score: (x - mean) / std
    pub fn normalize<B: burn::tensor::backend::Backend>(
        &self,
        comparison: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let device = comparison.device();
        let mean_tensor = Tensor::<B, 1>::from_floats(self.mean.as_slice(), &device)
            .unsqueeze_dim(0);
        let std_tensor = Tensor::<B, 1>::from_floats(self.std.as_slice(), &device)
            .unsqueeze_dim(0);

        (comparison - mean_tensor) / std_tensor
    }
}

/// Multi-task loss function for improved MLP
///
/// Predicts:
/// - win_logit: BCE loss for binary classification
/// - margin: MSE loss for absolute score difference
pub struct MLPLoss {
    pub win_weight: f32,
    pub margin_weight: f32,
}

impl MLPLoss {
    pub fn new(win_weight: f32, margin_weight: f32) -> Self {
        MLPLoss { win_weight, margin_weight }
    }

    /// Compute loss and return (total, win_loss, margin_loss)
    ///
    /// Note: batch scores are normalized z-scores.
    /// We work with normalized margin: target = |home_zscore - away_zscore| * std
    /// This gives us margin in actual points while keeping gradients well-scaled.
    pub fn forward<B: AutodiffBackend>(
        &self,
        predictions: &MLPPredictions<B>,
        batch: &MatchBatch<B>,
        score_norm: ScoreNormalization,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        // Binary cross-entropy for win prediction
        let win_targets = batch.home_win.clone().unsqueeze();
        let win_loss = self.binary_cross_entropy(predictions.win_logit.clone(), win_targets);

        // MSE for absolute margin prediction
        // Target margin = |home_zscore - away_zscore| * std
        // This gives margin in actual points while the difference of z-scores is well-scaled
        let device = batch.home_score.device();
        let std = Tensor::<B, 1>::from_floats([score_norm.std], &device);

        // The difference of z-scores times std gives actual margin
        let zscore_diff = batch.home_score.clone() - batch.away_score.clone();
        let target_margin = (zscore_diff * std).abs().unsqueeze();

        // Use Huber loss for better stability with outliers
        let diff = predictions.margin.clone() - target_margin;
        let delta = 10.0f32; // Switch from MSE to MAE at 10 points
        let abs_diff = diff.clone().abs();
        let quadratic = diff.clone().powf_scalar(2.0) * 0.5;
        let linear = abs_diff.clone() * delta - 0.5 * delta * delta;
        let huber = quadratic.mask_where(abs_diff.lower_elem(delta), linear);
        let margin_loss = huber.mean();

        let total_loss = win_loss.clone() * self.win_weight + margin_loss.clone() * self.margin_weight;

        (total_loss, win_loss, margin_loss)
    }

    /// Binary cross-entropy with logits (numerically stable)
    fn binary_cross_entropy<B: AutodiffBackend>(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let probs = sigmoid(logits);
        let eps = 1e-7;
        let probs_clamped = probs.clamp(eps, 1.0 - eps);
        let loss = targets.clone().neg() * probs_clamped.clone().log()
            - (targets.neg() + 1.0) * (probs_clamped.neg() + 1.0).log();
        loss.mean()
    }
}

/// Trainer for the improved MLP model (uses Adam optimizer)
pub struct MLPTrainer<B: AutodiffBackend> {
    model: MLPModel<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MLPModel<B>, B>,
    loss_fn: MLPLoss,
    learning_rate: f64,
    device: B::Device,
}

impl<B: AutodiffBackend> MLPTrainer<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new trainer with Adam optimizer
    pub fn new(
        model: MLPModel<B>,
        learning_rate: f64,
        weight_decay: f64,
        win_weight: f32,
        margin_weight: f32,
        device: B::Device,
    ) -> Self {
        // Use Adam optimizer for better convergence with batch normalization
        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(weight_decay as f32)))
            .init();

        MLPTrainer {
            model,
            optimizer,
            loss_fn: MLPLoss::new(win_weight, margin_weight),
            learning_rate,
            device,
        }
    }

    /// Train the model
    pub fn train(
        mut self,
        train_dataset: RugbyDataset,
        val_dataset: RugbyDataset,
        epochs: usize,
        batch_size: usize,
        early_stopping_patience: usize,
    ) -> Result<(MLPModel<B>, TrainingHistory)> {
        use burn::data::dataloader::DataLoaderBuilder;

        let score_norm = train_dataset.score_norm;

        // Compute feature normalization from training data
        let feature_norm = ComparisonNormalization::from_dataset(&train_dataset);
        log::info!(
            "Feature normalization computed for {} features",
            Self::DIM
        );

        let batcher_train = MatchBatcher::<B>::new(self.device.clone());
        let batcher_val = MatchBatcher::<B>::new(self.device.clone());

        // Use mini-batches for better generalization with Adam
        let effective_batch_size = if batch_size == 0 {
            train_dataset.len() // Full batch
        } else {
            batch_size.min(train_dataset.len())
        };

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(effective_batch_size)
            .shuffle(42) // Shuffle for Adam
            .build(train_dataset);

        let val_loader = DataLoaderBuilder::new(batcher_val)
            .batch_size(val_dataset.len()) // Full batch for validation
            .build(val_dataset);

        let mut history = TrainingHistory::new();
        let mut best_model = self.model.clone();

        log::info!("Starting improved MLP training for {} epochs (Adam optimizer)", epochs);

        for epoch in 0..epochs {
            // Training phase (with feature normalization)
            let train_metrics = self.train_epoch(train_loader.iter(), score_norm, &feature_norm);

            // Validation phase (with same feature normalization)
            let val_metrics = self.validate_epoch(val_loader.iter(), score_norm, &feature_norm);

            // Record history
            history.record_epoch(epoch, &train_metrics, &val_metrics);

            // Log progress
            log::info!(
                "Epoch {}/{}: Train: {} | Val: {}",
                epoch + 1,
                epochs,
                train_metrics,
                val_metrics
            );

            // Save best model based on validation loss
            if val_metrics.avg_loss() < history.best_val_loss {
                best_model = self.model.clone();
                log::info!("  New best model (val_loss: {:.4})", val_metrics.avg_loss());
            }

            // Early stopping
            if history.should_early_stop(early_stopping_patience) {
                log::info!(
                    "Early stopping at epoch {} (best was epoch {})",
                    epoch + 1,
                    history.best_epoch + 1
                );
                break;
            }
        }

        Ok((best_model, history))
    }

    /// Dimension constant
    const DIM: usize = MatchComparison::DIM;

    /// Train one epoch
    fn train_epoch(
        &mut self,
        loader: impl Iterator<Item = MatchBatch<B>>,
        score_norm: ScoreNormalization,
        feature_norm: &ComparisonNormalization,
    ) -> Metrics {
        let mut metrics = Metrics::with_normalization(score_norm);

        for batch in loader {
            let batch_size = batch.comparison.dims()[0];

            // Normalize comparison features
            let normalized_comparison = feature_norm.normalize(batch.comparison.clone());

            // Forward pass
            let predictions = self.model.forward(normalized_comparison);

            // Compute loss
            let (total_loss, win_loss, margin_loss) = self.loss_fn.forward(&predictions, &batch, score_norm);

            // Get loss values before backward pass
            let total_loss_val: f32 = total_loss.clone().into_scalar().elem();
            let win_loss_val: f32 = win_loss.into_scalar().elem();
            let margin_loss_val: f32 = margin_loss.into_scalar().elem();

            // Backward pass
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);

            // Update weights with Adam
            self.model = self.optimizer.step(self.learning_rate, self.model.clone(), grads);

            // Compute accuracy metrics
            let (correct, margin_mae) = self.compute_batch_metrics(&predictions, &batch, score_norm);

            // Update metrics (using margin_mae for both home/away MAE slots)
            metrics.update(
                total_loss_val,
                win_loss_val,
                margin_loss_val,
                correct,
                batch_size,
                margin_mae,
                margin_mae, // Same value for compatibility
            );
        }

        metrics
    }

    /// Validate one epoch
    fn validate_epoch(
        &self,
        loader: impl Iterator<Item = MatchBatch<B>>,
        score_norm: ScoreNormalization,
        feature_norm: &ComparisonNormalization,
    ) -> Metrics {
        let mut metrics = Metrics::with_normalization(score_norm);

        for batch in loader {
            let batch_size = batch.comparison.dims()[0];

            // Normalize comparison features
            let normalized_comparison = feature_norm.normalize(batch.comparison.clone());

            // Forward pass
            let predictions = self.model.forward(normalized_comparison);

            // Compute loss
            let (total_loss, win_loss, margin_loss) = self.loss_fn.forward(&predictions, &batch, score_norm);

            // Get values
            let total_loss_val: f32 = total_loss.into_scalar().elem();
            let win_loss_val: f32 = win_loss.into_scalar().elem();
            let margin_loss_val: f32 = margin_loss.into_scalar().elem();

            // Compute accuracy metrics
            let (correct, margin_mae) = self.compute_batch_metrics(&predictions, &batch, score_norm);

            metrics.update(
                total_loss_val,
                win_loss_val,
                margin_loss_val,
                correct,
                batch_size,
                margin_mae,
                margin_mae,
            );
        }

        metrics
    }

    /// Compute batch metrics (accuracy and margin MAE)
    fn compute_batch_metrics(
        &self,
        predictions: &MLPPredictions<B>,
        batch: &MatchBatch<B>,
        score_norm: ScoreNormalization,
    ) -> (usize, f32) {
        // Win accuracy
        let win_probs = sigmoid(predictions.win_logit.clone());
        let win_probs_data = win_probs.to_data();
        let targets_data = batch.home_win.clone().to_data();

        let win_probs_slice: &[f32] = win_probs_data.as_slice().unwrap();
        let targets_slice: &[f32] = targets_data.as_slice().unwrap();

        let correct = win_probs_slice
            .iter()
            .zip(targets_slice.iter())
            .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
            .count();

        // Margin MAE (denormalized)
        let margin_preds_data = predictions.margin.clone().to_data();
        let margin_preds: &[f32] = margin_preds_data.as_slice().unwrap();

        // Target: |home_score - away_score| denormalized
        let home_data = batch.home_score.clone().to_data();
        let away_data = batch.away_score.clone().to_data();
        let home_slice: &[f32] = home_data.as_slice().unwrap();
        let away_slice: &[f32] = away_data.as_slice().unwrap();

        // Debug: print first few predictions and targets
        if margin_preds.len() > 0 && margin_preds.len() < 200 {
            let first_pred = margin_preds[0];
            let first_home = score_norm.denormalize(home_slice[0]);
            let first_away = score_norm.denormalize(away_slice[0]);
            let first_target = (first_home - first_away).abs();

            // Compute and report actual MAE for this batch
            let actual_mae: f32 = margin_preds
                .iter()
                .zip(home_slice.iter().zip(away_slice.iter()))
                .map(|(pred, (h, a))| {
                    let home_actual = score_norm.denormalize(*h);
                    let away_actual = score_norm.denormalize(*a);
                    let target_margin = (home_actual - away_actual).abs();
                    (pred - target_margin).abs()
                })
                .sum::<f32>()
                / margin_preds.len() as f32;
            log::info!(
                "Margin debug: batch_size={}, pred[0]={:.2}, target[0]={:.2}, MAE={:.2}",
                margin_preds.len(), first_pred, first_target, actual_mae
            );
        }

        let margin_mae: f32 = margin_preds
            .iter()
            .zip(home_slice.iter().zip(away_slice.iter()))
            .map(|(pred, (h, a))| {
                // Denormalize scores to get actual margin
                let home_actual = score_norm.denormalize(*h);
                let away_actual = score_norm.denormalize(*a);
                let target_margin = (home_actual - away_actual).abs();
                (pred - target_margin).abs()
            })
            .sum::<f32>()
            / margin_preds.len().max(1) as f32;

        (correct, margin_mae)
    }

    /// Get the current model
    pub fn model(&self) -> &MLPModel<B> {
        &self.model
    }

    /// Get the model, consuming the trainer
    pub fn into_model(self) -> MLPModel<B> {
        self.model
    }
}
