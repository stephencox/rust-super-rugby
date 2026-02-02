//! MLP hyperparameter tuning with random train/val/test splits

use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::data::dataset::{MatchBatcher, MatchSample, RugbyDataset, ScoreNormalization};
use crate::Result;

use super::mlp_trainer::ComparisonNormalization;

/// Split ratios for train/val/test
#[derive(Debug, Clone)]
pub struct SplitRatios {
    pub train: f32,
    pub val: f32,
    pub test: f32,
}

impl Default for SplitRatios {
    fn default() -> Self {
        SplitRatios {
            train: 0.70,
            val: 0.15,
            test: 0.15,
        }
    }
}

/// Hyperparameters to tune
#[derive(Debug, Clone)]
pub struct MLPHyperparams {
    pub learning_rate: f64,
    pub epochs: usize,
}

/// Results from a tuning run
#[derive(Debug, Clone)]
pub struct TuningResult {
    pub hyperparams: MLPHyperparams,
    pub train_acc: f32,
    pub val_acc: f32,
    pub test_acc: f32,
    pub final_loss: f32,
}

/// Random split datasets
pub struct RandomSplitDatasets {
    pub train: RugbyDataset,
    pub val: RugbyDataset,
    pub test: RugbyDataset,
}

impl RandomSplitDatasets {
    /// Create random splits from a full dataset
    pub fn from_dataset(
        full_dataset: RugbyDataset,
        ratios: SplitRatios,
        seed: u64,
    ) -> Result<Self> {
        use burn::data::dataset::Dataset;

        let n = full_dataset.len();
        let n_train = (n as f32 * ratios.train) as usize;
        let n_val = (n as f32 * ratios.val) as usize;

        // Collect all samples
        let mut samples: Vec<MatchSample> = (0..n)
            .filter_map(|i| full_dataset.get(i))
            .collect();

        // Shuffle with seed for reproducibility
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        samples.shuffle(&mut rng);

        // Split
        let train_samples: Vec<_> = samples[..n_train].to_vec();
        let val_samples: Vec<_> = samples[n_train..n_train + n_val].to_vec();
        let test_samples: Vec<_> = samples[n_train + n_val..].to_vec();

        log::info!(
            "Split {} samples: train={}, val={}, test={}",
            n,
            train_samples.len(),
            val_samples.len(),
            test_samples.len()
        );

        // Compute normalization from training data only
        let score_norm = compute_score_norm(&train_samples);
        let feature_norm = compute_feature_norm(&train_samples);
        let team_mapping = full_dataset.team_mapping.clone();

        Ok(RandomSplitDatasets {
            train: RugbyDataset::from_samples(train_samples, score_norm, feature_norm, team_mapping.clone()),
            val: RugbyDataset::from_samples(val_samples, score_norm, feature_norm, team_mapping.clone()),
            test: RugbyDataset::from_samples(test_samples, score_norm, feature_norm, team_mapping),
        })
    }
}

fn compute_score_norm(samples: &[MatchSample]) -> ScoreNormalization {
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0;

    for sample in samples {
        // Denormalize to get original scores (assuming they were normalized)
        // For now just use the normalized values to compute stats
        sum += sample.home_score + sample.away_score;
        sum_sq += sample.home_score * sample.home_score + sample.away_score * sample.away_score;
        count += 2;
    }

    let mean = sum / count as f32;
    let std = ((sum_sq / count as f32) - mean * mean).sqrt().max(1.0);

    ScoreNormalization { mean, std }
}

fn compute_feature_norm(samples: &[MatchSample]) -> crate::data::dataset::FeatureNormalization {
    // Just use score norm for feature norm in this context
    let score_norm = compute_score_norm(samples);
    crate::data::dataset::FeatureNormalization::from_score_norm(score_norm)
}

/// MLP Tuner for hyperparameter search
pub struct MLPTuner<B: AutodiffBackend> {
    device: B::Device,
}

impl<B: AutodiffBackend> MLPTuner<B>
where
    B::FloatElem: serde::Serialize + serde::de::DeserializeOwned,
    B::IntElem: serde::Serialize + serde::de::DeserializeOwned,
{
    pub fn new(device: B::Device) -> Self {
        MLPTuner { device }
    }

    /// Run hyperparameter tuning
    pub fn tune(
        &self,
        datasets: &RandomSplitDatasets,
        hyperparams_list: Vec<MLPHyperparams>,
    ) -> Result<Vec<TuningResult>> {
        let mut results = Vec::new();

        // Compute feature normalization from training data
        let feature_norm = ComparisonNormalization::from_dataset(&datasets.train);

        for hyperparams in hyperparams_list {
            log::info!(
                "Testing lr={}, epochs={}",
                hyperparams.learning_rate,
                hyperparams.epochs
            );

            let result = self.train_and_evaluate(
                &datasets.train,
                &datasets.val,
                &datasets.test,
                &feature_norm,
                &hyperparams,
            )?;

            log::info!(
                "  train_acc={:.1}%, val_acc={:.1}%, test_acc={:.1}%",
                result.train_acc * 100.0,
                result.val_acc * 100.0,
                result.test_acc * 100.0
            );

            results.push(result);
        }

        Ok(results)
    }

    fn train_and_evaluate(
        &self,
        train: &RugbyDataset,
        val: &RugbyDataset,
        test: &RugbyDataset,
        feature_norm: &ComparisonNormalization,
        hyperparams: &MLPHyperparams,
    ) -> Result<TuningResult> {
        // Create model
        let mut model: Linear<B> = LinearConfig::new(15, 1).init(&self.device);
        let mut optimizer = SgdConfig::new().init();

        // Create data loaders (full batch)
        let batcher = MatchBatcher::<B>::new(self.device.clone());
        let train_loader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(train.len())
            .build(train.clone());
        let val_loader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(val.len())
            .build(val.clone());
        let test_loader = DataLoaderBuilder::new(batcher)
            .batch_size(test.len())
            .build(test.clone());

        let mut final_loss = 0.0f32;
        let mut best_val_acc = 0.0f32;
        let mut best_model = model.clone();

        // Training loop
        for _epoch in 0..hyperparams.epochs {
            let train_batch = train_loader.iter().next().unwrap();

            // Normalize and prepare data
            let x_train = feature_norm.normalize(train_batch.comparison.clone());
            let y_train = train_batch.home_win.clone().unsqueeze_dim(1);

            // Forward
            let logits = model.forward(x_train);
            let probs = sigmoid(logits);

            // Loss
            let loss = self.binary_cross_entropy(probs.clone(), y_train.clone());
            final_loss = loss.clone().into_scalar().elem();

            // Backward
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(hyperparams.learning_rate, model, grads_params);

            // Check validation accuracy
            let val_batch = val_loader.iter().next().unwrap();
            let x_val = feature_norm.normalize(val_batch.comparison.clone());
            let y_val = val_batch.home_win.clone().unsqueeze_dim(1);
            let val_logits = model.forward(x_val);
            let val_probs = sigmoid(val_logits);
            let val_acc = self.compute_accuracy(&val_probs, &y_val);

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_model = model.clone();
            }
        }

        // Evaluate best model on all sets
        let train_batch = train_loader.iter().next().unwrap();
        let x_train = feature_norm.normalize(train_batch.comparison.clone());
        let y_train = train_batch.home_win.clone().unsqueeze_dim(1);
        let train_probs = sigmoid(best_model.forward(x_train));
        let train_acc = self.compute_accuracy(&train_probs, &y_train);

        let val_batch = val_loader.iter().next().unwrap();
        let x_val = feature_norm.normalize(val_batch.comparison.clone());
        let y_val = val_batch.home_win.clone().unsqueeze_dim(1);
        let val_probs = sigmoid(best_model.forward(x_val));
        let val_acc = self.compute_accuracy(&val_probs, &y_val);

        let test_batch = test_loader.iter().next().unwrap();
        let x_test = feature_norm.normalize(test_batch.comparison.clone());
        let y_test = test_batch.home_win.clone().unsqueeze_dim(1);
        let test_probs = sigmoid(best_model.forward(x_test));
        let test_acc = self.compute_accuracy(&test_probs, &y_test);

        Ok(TuningResult {
            hyperparams: hyperparams.clone(),
            train_acc,
            val_acc,
            test_acc,
            final_loss,
        })
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
