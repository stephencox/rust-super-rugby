//! Transformer (RugbyNet) trainer for rugby match prediction

use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::data::dataset::{MatchBatcher, RugbyDataset};
use crate::model::rugby_net::{RugbyNet, RugbyNetConfig};
use crate::training::metrics::TrainingHistory;
use crate::Result;

use super::mlp_trainer::ComparisonNormalization;

/// Trainer for the Transformer (RugbyNet) model
pub struct TransformerTrainer<B: AutodiffBackend> {
    model: RugbyNet<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, RugbyNet<B>, B>,
    learning_rate: f64,
    device: B::Device,
}

impl<B: AutodiffBackend> TransformerTrainer<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new trainer
    pub fn new(device: B::Device, config: RugbyNetConfig, learning_rate: f64) -> Self {
        let model = RugbyNet::new(&device, config);
        // Use Adam optimizer for transformer (better than SGD for attention)
        let optimizer = AdamConfig::new().init();

        TransformerTrainer {
            model,
            optimizer,
            learning_rate,
            device,
        }
    }

    /// Train the model and return both history and trained model
    pub fn train(
        mut self,
        train_dataset: RugbyDataset,
        val_dataset: RugbyDataset,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(TrainingHistory, RugbyNet<B>)> {
        // Compute feature normalization for comparison features
        let feature_norm = ComparisonNormalization::from_dataset(&train_dataset);
        log::info!(
            "Feature normalization: mean={:?}, std={:?}",
            feature_norm.mean,
            feature_norm.std
        );

        // Use mini-batches for transformer (full batch may OOM)
        let effective_batch_size = if batch_size == 0 {
            train_dataset.len().min(256)
        } else {
            batch_size
        };

        let batcher_train = MatchBatcher::<B>::new(self.device.clone());
        let batcher_val = MatchBatcher::<B>::new(self.device.clone());

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(effective_batch_size)
            .shuffle(42)
            .build(train_dataset);

        let val_loader = DataLoaderBuilder::new(batcher_val)
            .batch_size(val_dataset.len())
            .build(val_dataset);

        let mut history = TrainingHistory::new();

        log::info!(
            "Starting Transformer training for {} epochs (batch_size={})",
            epochs,
            effective_batch_size
        );

        for epoch in 0..epochs {
            // Training phase
            let mut train_loss_sum = 0.0f32;
            let mut train_correct = 0usize;
            let mut train_total = 0usize;

            for batch in train_loader.iter() {
                let batch_size = batch.comparison.dims()[0];

                // Normalize comparison features
                let comparison = feature_norm.normalize(batch.comparison.clone());

                // Forward pass
                let predictions = self.model.forward(
                    batch.home_history.clone(),
                    batch.away_history.clone(),
                    Some(batch.home_mask.clone()),
                    Some(batch.away_mask.clone()),
                    Some(batch.home_team_id.clone()),
                    Some(batch.away_team_id.clone()),
                    Some(comparison),
                );

                // Targets
                let y_train = batch.home_win.clone().unsqueeze_dim(1);

                // BCE loss
                let probs = sigmoid(predictions.win_logit.clone());
                let loss = self.binary_cross_entropy(probs.clone(), y_train.clone());
                let loss_val: f32 = loss.clone().into_scalar().elem();

                // Accuracy
                let correct = self.count_correct(&probs, &y_train);
                train_correct += correct;
                train_total += batch_size;
                train_loss_sum += loss_val * batch_size as f32;

                // Backward pass
                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &self.model);

                // Update weights
                self.model = self.optimizer.step(self.learning_rate, self.model, grads_params);
            }

            let train_loss = train_loss_sum / train_total as f32;
            let train_acc = train_correct as f32 / train_total as f32;

            // Validation phase
            let val_batch = val_loader.iter().next().unwrap();
            let comparison_val = feature_norm.normalize(val_batch.comparison.clone());

            let val_preds = self.model.forward(
                val_batch.home_history.clone(),
                val_batch.away_history.clone(),
                Some(val_batch.home_mask.clone()),
                Some(val_batch.away_mask.clone()),
                Some(val_batch.home_team_id.clone()),
                Some(val_batch.away_team_id.clone()),
                Some(comparison_val),
            );

            let y_val = val_batch.home_win.clone().unsqueeze_dim(1);
            let val_probs = sigmoid(val_preds.win_logit);
            let val_acc = self.count_correct(&val_probs, &y_val) as f32
                / val_batch.comparison.dims()[0] as f32;

            // Update history
            if val_acc as f64 > history.val_accuracies.last().cloned().unwrap_or(0.0) {
                history.best_epoch = epoch;
                history.best_val_loss = train_loss as f64;
            }
            history.train_losses.push(train_loss as f64);
            history.val_losses.push(train_loss as f64);
            history.train_accuracies.push(train_acc as f64);
            history.val_accuracies.push(val_acc as f64);

            if epoch % 5 == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: loss={:.4}, train_acc={:.1}%, val_acc={:.1}%",
                    epoch + 1,
                    epochs,
                    train_loss,
                    train_acc * 100.0,
                    val_acc * 100.0
                );
            }
        }

        Ok((history, self.model))
    }

    fn binary_cross_entropy(&self, probs: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let eps = 1e-7;
        let probs_clamped = probs.clamp(eps, 1.0 - eps);
        let loss = targets.clone().neg() * probs_clamped.clone().log()
            - (targets.neg() + 1.0) * (probs_clamped.neg() + 1.0).log();
        loss.mean()
    }

    fn count_correct(&self, probs: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> usize {
        let probs_data = probs.clone().into_data();
        let targets_data = targets.clone().into_data();
        let probs_slice: &[f32] = probs_data.as_slice().unwrap();
        let targets_slice: &[f32] = targets_data.as_slice().unwrap();

        probs_slice
            .iter()
            .zip(targets_slice.iter())
            .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
            .count()
    }

    /// Get the trained model
    pub fn into_model(self) -> RugbyNet<B> {
        self.model
    }
}
