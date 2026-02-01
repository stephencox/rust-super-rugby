//! Simple MLP trainer that matches debug_training.rs exactly

use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::data::dataset::{MatchBatcher, RugbyDataset, ScoreNormalization};
use crate::training::metrics::{Metrics, TrainingHistory};
use crate::Result;

use super::mlp_trainer::ComparisonNormalization;

/// Simple trainer using single Linear layer (like debug_training.rs)
pub struct SimpleMLPTrainer<B: AutodiffBackend> {
    model: Linear<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Sgd<B::InnerBackend>, Linear<B>, B>,
    learning_rate: f64,
    device: B::Device,
}

impl<B: AutodiffBackend> SimpleMLPTrainer<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new trainer
    pub fn new(device: B::Device, learning_rate: f64) -> Self {
        let model = LinearConfig::new(15, 1).init(&device);
        let optimizer = SgdConfig::new().init();

        SimpleMLPTrainer {
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
        let score_norm = train_dataset.score_norm;

        // Compute feature normalization
        let feature_norm = ComparisonNormalization::from_dataset(&train_dataset);
        log::info!(
            "Feature normalization: mean={:?}, std={:?}",
            feature_norm.mean,
            feature_norm.std
        );

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

        log::info!("Starting simple MLP training for {} epochs", epochs);

        for epoch in 0..epochs {
            // Get the single batch
            let train_batch = train_loader.iter().next().unwrap();
            let val_batch = val_loader.iter().next().unwrap();

            // Normalize features
            let x_train = feature_norm.normalize(train_batch.comparison.clone());
            let y_train = train_batch.home_win.clone().unsqueeze_dim(1);

            let x_val = feature_norm.normalize(val_batch.comparison.clone());
            let y_val = val_batch.home_win.clone().unsqueeze_dim(1);

            // Forward pass
            let logits = self.model.forward(x_train);
            let probs = sigmoid(logits.clone());

            // BCE loss (same as debug_training)
            let loss = self.binary_cross_entropy(probs.clone(), y_train.clone());
            let loss_val: f32 = loss.clone().into_scalar().elem();

            // Compute train accuracy before backward
            let train_acc = self.compute_accuracy(&probs, &y_train);

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.model);

            // Update weights
            self.model = self.optimizer.step(self.learning_rate, self.model, grads_params);

            // Validation
            let val_logits = self.model.forward(x_val);
            let val_probs = sigmoid(val_logits);
            let val_acc = self.compute_accuracy(&val_probs, &y_val);

            // Record metrics (using dummy values for score metrics)
            let train_metrics = Metrics::with_normalization(score_norm);
            let val_metrics = Metrics::with_normalization(score_norm);
            history.record_epoch(epoch, &train_metrics, &val_metrics);

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
