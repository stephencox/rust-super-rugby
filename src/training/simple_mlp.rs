//! Simple MLP trainer with multi-task learning (win + margin prediction)

use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::data::dataset::{MatchBatcher, RugbyDataset};
use crate::training::metrics::{Metrics, TrainingHistory};
use crate::Result;

use super::mlp_trainer::ComparisonNormalization;

/// Optimizer type for training
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    Sgd,
    Adam,
}

impl Default for OptimizerType {
    fn default() -> Self {
        OptimizerType::Adam
    }
}

/// Weight initialization method (currently only default is supported)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMethod {
    /// Default Burn initialization (Xavier/Glorot)
    Default,
    /// Small random weights (not yet implemented for multi-task)
    Small,
    /// Zero weights (not yet implemented for multi-task)
    Zero,
}

impl Default for InitMethod {
    fn default() -> Self {
        InitMethod::Default
    }
}

/// Simple trainer using Linear layer with multi-task output (win + margin)
pub struct SimpleMLPTrainer<B: AutodiffBackend> {
    learning_rate: f64,
    device: B::Device,
    optimizer_type: OptimizerType,
    _init_method: InitMethod,
}

impl<B: AutodiffBackend> SimpleMLPTrainer<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new trainer with configurable optimizer and initialization
    pub fn new(
        device: B::Device,
        learning_rate: f64,
        optimizer_type: OptimizerType,
        init_method: InitMethod,
    ) -> Self {
        SimpleMLPTrainer {
            learning_rate,
            device,
            optimizer_type,
            _init_method: init_method,
        }
    }

    /// Train the model and return history, trained model, normalization, and margin MAE
    pub fn train(
        self,
        train_dataset: RugbyDataset,
        val_dataset: RugbyDataset,
        epochs: usize,
    ) -> Result<(TrainingHistory, Linear<B>, ComparisonNormalization, f32)> {
        let score_norm = train_dataset.score_norm;

        // Compute feature normalization
        let feature_norm = ComparisonNormalization::from_dataset(&train_dataset);
        log::info!(
            "Feature normalization: mean={:?}, std={:?}",
            feature_norm.mean,
            feature_norm.std
        );

        // Create model: 15 inputs -> 2 outputs (win_logit, margin)
        let mut model: Linear<B> = LinearConfig::new(15, 2).init(&self.device);

        // Full batch - no shuffle
        let batcher_train = MatchBatcher::<B>::new(self.device.clone());
        let batcher_val = MatchBatcher::<B>::new(self.device.clone());

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(train_dataset.len())
            .build(train_dataset);

        let val_loader = DataLoaderBuilder::new(batcher_val)
            .batch_size(val_dataset.len())
            .build(val_dataset);

        let mut history = TrainingHistory::new();
        let mut best_model = model.clone();
        let mut best_val_acc = 0.0f32;
        let mut best_margin_mae = f32::MAX;
        let mut best_epoch = 0usize;

        log::info!(
            "Starting multi-task MLP training for {} epochs (optimizer: {:?})",
            epochs, self.optimizer_type
        );

        // Train based on optimizer type
        match self.optimizer_type {
            OptimizerType::Sgd => {
                let mut optimizer = SgdConfig::new().init();
                for epoch in 0..epochs {
                    // Get batches
                    let train_batch = train_loader.iter().next().unwrap();
                    let val_batch = val_loader.iter().next().unwrap();

                    // Normalize features
                    let x_train = feature_norm.normalize(train_batch.comparison.clone());
                    let y_win_train = train_batch.home_win.clone().unsqueeze_dim(1);
                    let y_margin_train = (train_batch.home_score.clone() - train_batch.away_score.clone())
                        .unsqueeze_dim(1);

                    // Forward pass
                    let output = model.forward(x_train);
                    let batch_size = output.dims()[0];
                    let win_logit = output.clone().slice([0..batch_size, 0..1]);
                    let margin_pred = output.slice([0..batch_size, 1..2]);
                    let win_probs = sigmoid(win_logit);

                    // Multi-task loss
                    let win_loss = self.binary_cross_entropy(win_probs.clone(), y_win_train.clone());
                    let margin_loss = (margin_pred - y_margin_train).powf_scalar(2.0).mean();
                    let total_loss = win_loss.clone() + margin_loss * 0.1;
                    let loss_val: f32 = total_loss.clone().into_scalar().elem();

                    // Train accuracy
                    let train_acc = self.compute_accuracy(&win_probs, &y_win_train);

                    // Backward pass
                    let grads = total_loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(self.learning_rate, model, grads_params);

                    // Validation
                    let x_val = feature_norm.normalize(val_batch.comparison.clone());
                    let y_win_val = val_batch.home_win.clone().unsqueeze_dim(1);
                    let y_margin_val = (val_batch.home_score.clone() - val_batch.away_score.clone())
                        .unsqueeze_dim(1);
                    let val_output = model.forward(x_val);
                    let val_batch_size = val_output.dims()[0];
                    let val_win_logit = val_output.clone().slice([0..val_batch_size, 0..1]);
                    let val_margin_pred = val_output.slice([0..val_batch_size, 1..2]);
                    let val_win_probs = sigmoid(val_win_logit);
                    let val_acc = self.compute_accuracy(&val_win_probs, &y_win_val);
                    let val_margin_mae = self.compute_margin_mae(&val_margin_pred, &y_margin_val, score_norm);

                    // Track best model
                    if val_acc > best_val_acc || (val_acc == best_val_acc && val_margin_mae < best_margin_mae) {
                        best_val_acc = val_acc;
                        best_margin_mae = val_margin_mae;
                        best_model = model.clone();
                        best_epoch = epoch;
                    }

                    // Record metrics
                    let train_metrics = Metrics::with_normalization(score_norm);
                    let val_metrics = Metrics::with_normalization(score_norm);
                    history.record_epoch(epoch, &train_metrics, &val_metrics);

                    if epoch % 10 == 0 || epoch == epochs - 1 {
                        log::info!(
                            "Epoch {}/{}: loss={:.4}, train_acc={:.1}%, val_acc={:.1}%, margin_mae={:.1}{}",
                            epoch + 1, epochs, loss_val, train_acc * 100.0, val_acc * 100.0,
                            val_margin_mae, if val_acc == best_val_acc { " *" } else { "" }
                        );
                    }
                }
            }
            OptimizerType::Adam => {
                let mut optimizer = AdamConfig::new().init();
                for epoch in 0..epochs {
                    // Get batches
                    let train_batch = train_loader.iter().next().unwrap();
                    let val_batch = val_loader.iter().next().unwrap();

                    // Normalize features
                    let x_train = feature_norm.normalize(train_batch.comparison.clone());
                    let y_win_train = train_batch.home_win.clone().unsqueeze_dim(1);
                    let y_margin_train = (train_batch.home_score.clone() - train_batch.away_score.clone())
                        .unsqueeze_dim(1);

                    // Forward pass
                    let output = model.forward(x_train);
                    let batch_size = output.dims()[0];
                    let win_logit = output.clone().slice([0..batch_size, 0..1]);
                    let margin_pred = output.slice([0..batch_size, 1..2]);
                    let win_probs = sigmoid(win_logit);

                    // Multi-task loss
                    let win_loss = self.binary_cross_entropy(win_probs.clone(), y_win_train.clone());
                    let margin_loss = (margin_pred - y_margin_train).powf_scalar(2.0).mean();
                    let total_loss = win_loss.clone() + margin_loss * 0.1;
                    let loss_val: f32 = total_loss.clone().into_scalar().elem();

                    // Train accuracy
                    let train_acc = self.compute_accuracy(&win_probs, &y_win_train);

                    // Backward pass
                    let grads = total_loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(self.learning_rate, model, grads_params);

                    // Validation
                    let x_val = feature_norm.normalize(val_batch.comparison.clone());
                    let y_win_val = val_batch.home_win.clone().unsqueeze_dim(1);
                    let y_margin_val = (val_batch.home_score.clone() - val_batch.away_score.clone())
                        .unsqueeze_dim(1);
                    let val_output = model.forward(x_val);
                    let val_batch_size = val_output.dims()[0];
                    let val_win_logit = val_output.clone().slice([0..val_batch_size, 0..1]);
                    let val_margin_pred = val_output.slice([0..val_batch_size, 1..2]);
                    let val_win_probs = sigmoid(val_win_logit);
                    let val_acc = self.compute_accuracy(&val_win_probs, &y_win_val);
                    let val_margin_mae = self.compute_margin_mae(&val_margin_pred, &y_margin_val, score_norm);

                    // Track best model
                    if val_acc > best_val_acc || (val_acc == best_val_acc && val_margin_mae < best_margin_mae) {
                        best_val_acc = val_acc;
                        best_margin_mae = val_margin_mae;
                        best_model = model.clone();
                        best_epoch = epoch;
                    }

                    // Record metrics
                    let train_metrics = Metrics::with_normalization(score_norm);
                    let val_metrics = Metrics::with_normalization(score_norm);
                    history.record_epoch(epoch, &train_metrics, &val_metrics);

                    if epoch % 10 == 0 || epoch == epochs - 1 {
                        log::info!(
                            "Epoch {}/{}: loss={:.4}, train_acc={:.1}%, val_acc={:.1}%, margin_mae={:.1}{}",
                            epoch + 1, epochs, loss_val, train_acc * 100.0, val_acc * 100.0,
                            val_margin_mae, if val_acc == best_val_acc { " *" } else { "" }
                        );
                    }
                }
            }
        }

        log::info!(
            "Best model at epoch {} with val_acc={:.1}%, margin_mae={:.2}",
            best_epoch + 1,
            best_val_acc * 100.0,
            best_margin_mae
        );

        Ok((history, best_model, feature_norm, best_margin_mae))
    }

    fn binary_cross_entropy(&self, probs: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let eps = 1e-7;
        let probs_clamped = probs.clamp(eps, 1.0 - eps);
        let loss = targets.clone().neg() * probs_clamped.clone().log()
            - (targets.neg() + 1.0) * (probs_clamped.neg() + 1.0).log();
        loss.mean()
    }

    fn compute_accuracy(&self, probs: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
        let probs_data = probs.clone().into_data();
        let targets_data = targets.clone().into_data();
        let probs_slice: &[f32] = probs_data.as_slice().unwrap();
        let targets_slice: &[f32] = targets_data.as_slice().unwrap();

        let correct = probs_slice
            .iter()
            .zip(targets_slice.iter())
            .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
            .count();

        correct as f32 / probs_slice.len() as f32
    }

    fn compute_margin_mae(
        &self,
        pred: &Tensor<B, 2>,
        target: &Tensor<B, 2>,
        score_norm: crate::data::dataset::ScoreNormalization,
    ) -> f32 {
        let pred_data = pred.clone().into_data();
        let target_data = target.clone().into_data();
        let pred_slice: &[f32] = pred_data.as_slice().unwrap();
        let target_slice: &[f32] = target_data.as_slice().unwrap();

        // Denormalize to get actual margin MAE
        let mae: f32 = pred_slice
            .iter()
            .zip(target_slice.iter())
            .map(|(p, t)| {
                let pred_margin = p * score_norm.std;
                let actual_margin = t * score_norm.std;
                (pred_margin - actual_margin).abs()
            })
            .sum::<f32>()
            / pred_slice.len() as f32;

        mae
    }
}
