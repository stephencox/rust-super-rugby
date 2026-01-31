//! LSTM trainer for rugby match prediction

use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::data::dataset::{MatchBatcher, RugbyDataset};
use crate::model::lstm::{LSTMConfig, LSTMModel};
use crate::training::metrics::TrainingHistory;
use crate::Result;

use super::mlp_trainer::ComparisonNormalization;

/// Trainer for the LSTM model
pub struct LSTMTrainer<B: AutodiffBackend> {
    model: LSTMModel<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Sgd<B::InnerBackend>, LSTMModel<B>, B>,
    learning_rate: f64,
    device: B::Device,
}

impl<B: AutodiffBackend> LSTMTrainer<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new trainer
    pub fn new(device: B::Device, config: LSTMConfig, learning_rate: f64) -> Self {
        let model = LSTMModel::new(&device, config);
        let optimizer = SgdConfig::new().init();

        LSTMTrainer {
            model,
            optimizer,
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
    ) -> Result<TrainingHistory> {
        // Compute feature normalization for comparison features
        let feature_norm = ComparisonNormalization::from_dataset(&train_dataset);
        log::info!(
            "Feature normalization: mean={:?}, std={:?}",
            feature_norm.mean,
            feature_norm.std
        );

        // Full batch training
        let batcher_train = MatchBatcher::<B>::new(self.device.clone());
        let batcher_val = MatchBatcher::<B>::new(self.device.clone());

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(train_dataset.len())
            .build(train_dataset);

        let val_loader = DataLoaderBuilder::new(batcher_val)
            .batch_size(val_dataset.len())
            .build(val_dataset);

        let mut history = TrainingHistory::new();

        log::info!("Starting LSTM training for {} epochs", epochs);

        for epoch in 0..epochs {
            // Get batches
            let train_batch = train_loader.iter().next().unwrap();
            let val_batch = val_loader.iter().next().unwrap();

            // Normalize comparison features
            let comparison_train = feature_norm.normalize(train_batch.comparison.clone());
            let comparison_val = feature_norm.normalize(val_batch.comparison.clone());

            // Targets
            let y_train = train_batch.home_win.clone().unsqueeze_dim(1);
            let y_val = val_batch.home_win.clone().unsqueeze_dim(1);

            // Forward pass
            let (win_logit, _, _) = self.model.forward(
                train_batch.home_history.clone(),
                train_batch.away_history.clone(),
                comparison_train,
            );
            let probs = sigmoid(win_logit);

            // BCE loss
            let loss = self.binary_cross_entropy(probs.clone(), y_train.clone());
            let loss_val: f32 = loss.clone().into_scalar().elem();

            // Train accuracy
            let train_acc = self.compute_accuracy(&probs, &y_train);

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.model);

            // Update weights
            self.model = self.optimizer.step(self.learning_rate, self.model, grads_params);

            // Validation
            let (val_logit, _, _) = self.model.forward(
                val_batch.home_history.clone(),
                val_batch.away_history.clone(),
                comparison_val,
            );
            let val_probs = sigmoid(val_logit);
            let val_acc = self.compute_accuracy(&val_probs, &y_val);

            // Update history
            if val_acc as f64 > history.val_accuracies.last().cloned().unwrap_or(0.0) {
                history.best_epoch = epoch;
                history.best_val_loss = loss_val as f64;
            }
            history.train_losses.push(loss_val as f64);
            history.val_losses.push(loss_val as f64);
            history.train_accuracies.push(train_acc as f64);
            history.val_accuracies.push(val_acc as f64);

            if epoch % 10 == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: loss={:.4}, train_acc={:.1}%, val_acc={:.1}%",
                    epoch + 1,
                    epochs,
                    loss_val,
                    train_acc * 100.0,
                    val_acc * 100.0
                );
            }
        }

        Ok(history)
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
}
