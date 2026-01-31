//! Training loop and loss computation

use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::data::dataset::{MatchBatch, MatchBatcher, RugbyDataset};
use crate::model::heads::Predictions;
use crate::model::RugbyNet;
use crate::training::metrics::{Metrics, TrainingHistory};
use crate::{Config, LossConfig, Result};

/// Multi-task loss function
pub struct RugbyLoss {
    pub win_weight: f32,
    pub score_weight: f32,
}

impl RugbyLoss {
    pub fn new(config: &LossConfig) -> Self {
        RugbyLoss {
            win_weight: config.win_weight,
            score_weight: config.score_weight,
        }
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

        let total_loss =
            win_loss.clone() * self.win_weight + score_loss.clone() * self.score_weight;

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

/// Trainer for the RugbyNet model
pub struct Trainer<B: AutodiffBackend> {
    model: RugbyNet<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, RugbyNet<B>, B>,
    loss_fn: RugbyLoss,
    config: Config,
    device: B::Device,
}

impl<B: AutodiffBackend> Trainer<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new trainer
    pub fn new(model: RugbyNet<B>, config: Config, device: B::Device) -> Self {
        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                config.training.weight_decay as f32,
            )))
            .init();

        Trainer {
            model,
            optimizer,
            loss_fn: RugbyLoss::new(&config.loss),
            config,
            device,
        }
    }

    /// Train the model
    pub fn train(
        mut self,
        train_dataset: RugbyDataset,
        val_dataset: RugbyDataset,
    ) -> Result<(RugbyNet<B>, TrainingHistory)> {
        use burn::data::dataloader::DataLoaderBuilder;

        // Store score normalization for metrics
        let score_norm = train_dataset.score_norm;

        let batcher_train = MatchBatcher::<B>::new(self.device.clone());
        let batcher_val = MatchBatcher::<B>::new(self.device.clone());

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(self.config.training.batch_size)
            .shuffle(42)
            .build(train_dataset);

        let val_loader = DataLoaderBuilder::new(batcher_val)
            .batch_size(self.config.training.batch_size)
            .build(val_dataset);

        let mut history = TrainingHistory::new();
        let mut best_model = self.model.clone();

        log::info!(
            "Starting training for {} epochs",
            self.config.training.epochs
        );

        for epoch in 0..self.config.training.epochs {
            // Training phase
            let train_metrics = self.train_epoch(train_loader.iter(), score_norm);

            // Validation phase (using training model, simpler)
            let val_metrics = self.validate_epoch(val_loader.iter(), score_norm);

            // Record history
            history.record_epoch(epoch, &train_metrics, &val_metrics);

            // Log progress
            log::info!(
                "Epoch {}/{}: Train: {} | Val: {}",
                epoch + 1,
                self.config.training.epochs,
                train_metrics,
                val_metrics
            );

            // Save best model
            if val_metrics.avg_loss() < history.best_val_loss {
                best_model = self.model.clone();
                log::info!("  New best model (val_loss: {:.4})", val_metrics.avg_loss());
            }

            // Early stopping
            if history.should_early_stop(self.config.training.early_stopping_patience) {
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
        score_norm: crate::data::dataset::ScoreNormalization,
    ) -> Metrics {
        let mut metrics = Metrics::with_normalization(score_norm);

        for batch in loader {
            let batch_size = batch.home_history.dims()[0];

            // Forward pass
            let predictions = self.model.forward(
                batch.home_history.clone(),
                batch.away_history.clone(),
                Some(batch.home_mask.clone()),
                Some(batch.away_mask.clone()),
            );

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
            self.model = self.optimizer.step(
                self.config.training.learning_rate,
                self.model.clone(),
                grads,
            );

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

    /// Validate one epoch (no gradient updates)
    fn validate_epoch(
        &self,
        loader: impl Iterator<Item = MatchBatch<B>>,
        score_norm: crate::data::dataset::ScoreNormalization,
    ) -> Metrics {
        let mut metrics = Metrics::with_normalization(score_norm);

        for batch in loader {
            let batch_size = batch.home_history.dims()[0];

            // Forward pass (with gradients tracked but not used)
            let predictions = self.model.forward(
                batch.home_history.clone(),
                batch.away_history.clone(),
                Some(batch.home_mask.clone()),
                Some(batch.away_mask.clone()),
            );

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

    /// Compute batch metrics (accuracy, MAE)
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
    pub fn model(&self) -> &RugbyNet<B> {
        &self.model
    }

    /// Get the model, consuming the trainer
    pub fn into_model(self) -> RugbyNet<B> {
        self.model
    }
}
