//! Training loop for MLP model

use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::data::dataset::{MatchBatch, MatchBatcher, RugbyDataset, ScoreNormalization};
use crate::model::heads::Predictions;
use crate::model::mlp::MLPModel;
use crate::training::metrics::{Metrics, TrainingHistory};
use crate::Result;

/// Feature normalization for comparison features (z-score normalization)
#[derive(Debug, Clone)]
pub struct ComparisonNormalization {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl ComparisonNormalization {
    /// Dimension of comparison features
    pub const DIM: usize = 15;

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

/// Multi-task loss function for MLP
pub struct MLPLoss {
    pub win_weight: f32,
    pub score_weight: f32,
}

impl MLPLoss {
    pub fn new(win_weight: f32, score_weight: f32) -> Self {
        MLPLoss { win_weight, score_weight }
    }

    /// Compute loss and return (total, win_loss, score_loss)
    pub fn forward<B: AutodiffBackend>(
        &self,
        predictions: &Predictions<B>,
        targets: &MatchBatch<B>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        // Binary cross-entropy for win prediction
        let win_targets = targets.home_win.clone().unsqueeze();
        let win_loss = self.binary_cross_entropy(predictions.win_logit.clone(), win_targets);

        // Mean squared error for score prediction
        let home_targets = targets.home_score.clone().unsqueeze();
        let away_targets = targets.away_score.clone().unsqueeze();
        let home_score_loss = (predictions.home_score.clone() - home_targets)
            .powf_scalar(2.0)
            .mean();
        let away_score_loss = (predictions.away_score.clone() - away_targets)
            .powf_scalar(2.0)
            .mean();
        let score_loss = home_score_loss + away_score_loss;

        let total_loss = win_loss.clone() * self.win_weight + score_loss.clone() * self.score_weight;

        (total_loss, win_loss, score_loss)
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

/// Trainer for the MLP model
pub struct MLPTrainer<B: AutodiffBackend> {
    model: MLPModel<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Sgd<B::InnerBackend>, MLPModel<B>, B>,
    loss_fn: MLPLoss,
    learning_rate: f64,
    device: B::Device,
}

impl<B: AutodiffBackend> MLPTrainer<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new trainer
    pub fn new(
        model: MLPModel<B>,
        learning_rate: f64,
        _weight_decay: f64,
        win_weight: f32,
        score_weight: f32,
        device: B::Device,
    ) -> Self {
        // Use SGD (no momentum) to match Python's simple gradient descent
        let optimizer = SgdConfig::new().init();

        MLPTrainer {
            model,
            optimizer,
            loss_fn: MLPLoss::new(win_weight, score_weight),
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
            "Feature normalization: mean={:?}, std={:?}",
            feature_norm.mean,
            feature_norm.std
        );

        let batcher_train = MatchBatcher::<B>::new(self.device.clone());
        let batcher_val = MatchBatcher::<B>::new(self.device.clone());

        // Use full-batch training for small datasets (no shuffle for stability)
        // This matches Python's behavior and achieves better convergence
        let effective_batch_size = if batch_size == 0 {
            train_dataset.len() // Full batch
        } else {
            batch_size.min(train_dataset.len())
        };

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(effective_batch_size)
            // No shuffle for stable gradients on small datasets
            .build(train_dataset);

        let val_loader = DataLoaderBuilder::new(batcher_val)
            .batch_size(val_dataset.len()) // Full batch for validation
            .build(val_dataset);

        let mut history = TrainingHistory::new();
        let mut best_model = self.model.clone();

        log::info!("Starting MLP training for {} epochs", epochs);

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

            // Save best model
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

    /// Train one epoch
    fn train_epoch(
        &mut self,
        loader: impl Iterator<Item = MatchBatch<B>>,
        score_norm: ScoreNormalization,
        feature_norm: &ComparisonNormalization,
    ) -> Metrics {
        let mut metrics = Metrics::with_normalization(score_norm);

        let mut first_batch = true;
        for batch in loader {
            let batch_size = batch.comparison.dims()[0];

            // Normalize comparison features (critical for learning!)
            let normalized_comparison = feature_norm.normalize(batch.comparison.clone());

            // Debug: print first batch values
            if first_batch {
                let raw_data = batch.comparison.clone().into_data();
                let norm_data = normalized_comparison.clone().into_data();
                let raw_slice: &[f32] = raw_data.as_slice().unwrap();
                let norm_slice: &[f32] = norm_data.as_slice().unwrap();
                log::debug!(
                    "First sample raw: {:?}, normalized: {:?}",
                    &raw_slice[..5.min(raw_slice.len())],
                    &norm_slice[..5.min(norm_slice.len())]
                );
                first_batch = false;
            }

            // Forward pass - MLP only uses comparison features
            let predictions = self.model.forward(normalized_comparison);

            // Compute loss
            let (total_loss, win_loss, score_loss) = self.loss_fn.forward(&predictions, &batch);

            // Get loss values before backward pass
            let total_loss_val: f32 = total_loss.clone().into_scalar().elem();
            let win_loss_val: f32 = win_loss.into_scalar().elem();
            let score_loss_val: f32 = score_loss.into_scalar().elem();

            // Backward pass
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);

            // Update weights
            self.model = self.optimizer.step(self.learning_rate, self.model.clone(), grads);

            // Compute accuracy metrics
            let (correct, home_mae, away_mae) = self.compute_batch_metrics(&predictions, &batch);

            metrics.update(
                total_loss_val,
                win_loss_val,
                score_loss_val,
                correct,
                batch_size,
                home_mae,
                away_mae,
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
            let (total_loss, win_loss, score_loss) = self.loss_fn.forward(&predictions, &batch);

            // Get values
            let total_loss_val: f32 = total_loss.into_scalar().elem();
            let win_loss_val: f32 = win_loss.into_scalar().elem();
            let score_loss_val: f32 = score_loss.into_scalar().elem();

            // Compute accuracy metrics
            let (correct, home_mae, away_mae) = self.compute_batch_metrics(&predictions, &batch);

            metrics.update(
                total_loss_val,
                win_loss_val,
                score_loss_val,
                correct,
                batch_size,
                home_mae,
                away_mae,
            );
        }

        metrics
    }

    /// Compute batch metrics
    fn compute_batch_metrics(
        &self,
        predictions: &Predictions<B>,
        batch: &MatchBatch<B>,
    ) -> (usize, f32, f32) {
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

        let home_preds_data = predictions.home_score.clone().to_data();
        let home_targets_data = batch.home_score.clone().to_data();
        let home_preds: &[f32] = home_preds_data.as_slice().unwrap();
        let home_targets: &[f32] = home_targets_data.as_slice().unwrap();
        let home_mae: f32 = home_preds
            .iter()
            .zip(home_targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f32>()
            / home_preds.len().max(1) as f32;

        let away_preds_data = predictions.away_score.clone().to_data();
        let away_targets_data = batch.away_score.clone().to_data();
        let away_preds: &[f32] = away_preds_data.as_slice().unwrap();
        let away_targets: &[f32] = away_targets_data.as_slice().unwrap();
        let away_mae: f32 = away_preds
            .iter()
            .zip(away_targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f32>()
            / away_preds.len().max(1) as f32;

        (correct, home_mae, away_mae)
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
