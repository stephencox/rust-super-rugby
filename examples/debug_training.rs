//! Debug training to match Python exactly
//! Run with: cargo run --example debug_training

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};

type B = Autodiff<NdArray<f32>>;

fn main() -> rugby::Result<()> {
    // Load data from CSV (same as Python)
    let (x_train, y_train) = load_csv("train_data.csv")?;
    let (x_val, y_val) = load_csv("val_data.csv")?;

    println!("Train: {} samples", x_train.len());
    println!("Val: {} samples", x_val.len());

    // Compute normalization stats
    let (mean, std) = compute_norm_stats(&x_train);
    println!("Mean: {:?}", mean);
    println!("Std: {:?}", std);

    // Normalize features
    let x_train_norm: Vec<[f32; 5]> = x_train
        .iter()
        .map(|x| normalize(x, &mean, &std))
        .collect();
    let x_val_norm: Vec<[f32; 5]> = x_val.iter().map(|x| normalize(x, &mean, &std)).collect();

    let device = Default::default();

    // Create tensors - FULL BATCH like Python
    let x_train_tensor = create_feature_tensor(&x_train_norm, &device);
    let y_train_tensor = create_label_tensor(&y_train, &device);
    let x_val_tensor = create_feature_tensor(&x_val_norm, &device);
    let y_val_tensor = create_label_tensor(&y_val, &device);

    // Create simple linear model (logistic regression)
    let mut model: Linear<B> = LinearConfig::new(5, 1).init(&device);

    // Print initial weights
    println!("\nInitial weights:");
    print_weights(&model);

    // SGD optimizer (no momentum, like Python)
    let mut optimizer = SgdConfig::new().init();
    let lr = 0.1;

    // Training loop - FULL BATCH like Python
    for epoch in 0..100 {
        // Forward pass
        let logits = model.forward(x_train_tensor.clone());
        let probs = sigmoid(logits.clone());

        // BCE loss
        let loss = binary_cross_entropy(probs.clone(), y_train_tensor.clone());
        let loss_val: f32 = loss.clone().into_scalar().elem();

        // Compute train accuracy before backward (probs will be consumed)
        let train_acc = compute_accuracy(&probs, &y_train_tensor);

        // Backward pass
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);

        // Update weights - reassign to model
        model = optimizer.step(lr, model, grads_params);

        // Validation
        let val_logits = model.forward(x_val_tensor.clone());
        let val_probs = sigmoid(val_logits);
        let val_acc = compute_accuracy(&val_probs, &y_val_tensor);

        if epoch % 20 == 0 {
            println!(
                "Epoch {}: loss={:.4}, train_acc={:.1}%, val_acc={:.1}%",
                epoch,
                loss_val,
                train_acc * 100.0,
                val_acc * 100.0
            );
        }
    }

    // Print final weights
    println!("\nFinal weights:");
    print_weights(&model);

    Ok(())
}

fn load_csv(path: &str) -> rugby::Result<(Vec<[f32; 5]>, Vec<f32>)> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        } // Skip header
        let line = line?;
        let parts: Vec<f32> = line.split(',').map(|s| s.parse().unwrap()).collect();
        features.push([parts[0], parts[1], parts[2], parts[3], parts[4]]);
        labels.push(parts[5]); // home_win
    }

    Ok((features, labels))
}

fn compute_norm_stats(features: &[[f32; 5]]) -> ([f32; 5], [f32; 5]) {
    let n = features.len() as f32;
    let mut mean = [0.0f32; 5];
    let mut sum_sq = [0.0f32; 5];

    for f in features {
        for i in 0..5 {
            mean[i] += f[i];
            sum_sq[i] += f[i] * f[i];
        }
    }

    for i in 0..5 {
        mean[i] /= n;
    }

    let mut std = [0.0f32; 5];
    for i in 0..5 {
        std[i] = ((sum_sq[i] / n - mean[i] * mean[i]).sqrt()).max(0.001);
    }

    (mean, std)
}

fn normalize(x: &[f32; 5], mean: &[f32; 5], std: &[f32; 5]) -> [f32; 5] {
    [
        (x[0] - mean[0]) / std[0],
        (x[1] - mean[1]) / std[1],
        (x[2] - mean[2]) / std[2],
        (x[3] - mean[3]) / std[3],
        (x[4] - mean[4]) / std[4],
    ]
}

fn create_feature_tensor(features: &[[f32; 5]], device: &<B as Backend>::Device) -> Tensor<B, 2> {
    let flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([features.len(), 5])
}

fn create_label_tensor(labels: &[f32], device: &<B as Backend>::Device) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(labels, device).reshape([labels.len(), 1])
}

fn binary_cross_entropy(probs: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let eps = 1e-7;
    let probs_clamped = probs.clamp(eps, 1.0 - eps);
    let loss = targets.clone().neg() * probs_clamped.clone().log()
        - (targets.neg() + 1.0) * (probs_clamped.neg() + 1.0).log();
    loss.mean()
}

fn compute_accuracy(probs: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> f32 {
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

fn print_weights(model: &Linear<B>) {
    let weight = model.weight.val();
    let bias = model.bias.as_ref().map(|b| b.val());

    let weight_data = weight.into_data();
    let weight_slice: &[f32] = weight_data.as_slice().unwrap();
    println!("  Weight: {:?}", weight_slice);

    if let Some(bias) = bias {
        let bias_data = bias.into_data();
        let bias_slice: &[f32] = bias_data.as_slice().unwrap();
        println!("  Bias: {:?}", bias_slice);
    }
}
