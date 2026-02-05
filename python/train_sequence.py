#!/usr/bin/env python3
"""
Train sequence-based models (LSTM and Transformer) for rugby prediction.

Usage:
    python train_sequence.py [--model lstm|transformer] [--epochs N]
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rugby.data import Database
from rugby.features import SequenceFeatureBuilder, SequenceNormalizer
from rugby.models import SequenceLSTM, MATCH_FEATURE_DIM
from rugby.training import train_sequence_model, evaluate_sequence_model


def main():
    parser = argparse.ArgumentParser(description='Train sequence models')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm'],
                        help='Model type (lstm)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--seq-len', type=int, default=10, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--win-weight', type=float, default=1.0, help='Weight for win loss')
    parser.add_argument('--margin-weight', type=float, default=0.1, help='Weight for margin loss')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization')
    parser.add_argument('--train-cutoff', type=str, default='2023-01-01',
                        help='Training cutoff date')
    parser.add_argument('--val-cutoff', type=str, default='2024-01-01',
                        help='Validation end date')
    args = parser.parse_args()

    train_cutoff = datetime.strptime(args.train_cutoff, '%Y-%m-%d')
    val_cutoff = datetime.strptime(args.val_cutoff, '%Y-%m-%d')

    print("=" * 60)
    print(f"Rugby Match Prediction - {args.model.upper()} Training")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    db = Database()
    matches = db.get_matches()
    teams = db.get_teams()
    stats = db.get_stats()
    db.close()

    print(f"  Loaded {stats['match_count']} matches, {stats['team_count']} teams")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")

    # Build sequence features
    print("\n[2/5] Building sequence features...")
    builder = SequenceFeatureBuilder(matches, teams, seq_len=args.seq_len)

    # Build datasets
    train_samples = []
    val_samples = []

    for match in sorted(matches, key=lambda m: m.date):
        sample = builder.build_sample(match)
        builder.process_match(match)

        if sample is None:
            continue

        if match.date < train_cutoff:
            train_samples.append(sample)
        elif match.date < val_cutoff:
            val_samples.append(sample)

    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Home win rate (train): {sum(s.home_win for s in train_samples)/len(train_samples):.1%}")

    # Normalize features
    print("\n[3/6] Normalizing features (z-score)...")
    normalizer = SequenceNormalizer()
    train_samples = normalizer.fit_transform(train_samples)
    val_samples = normalizer.transform_samples(val_samples)
    print(f"  Sequence feature mean (sample): {normalizer.seq_mean[:3]}")
    print(f"  Sequence feature std (sample): {normalizer.seq_std[:3]}")
    print(f"  Comparison feature mean (sample): {normalizer.comp_mean[:3]}")

    # Create model
    print(f"\n[4/6] Creating LSTM model...")
    sequence_dim = 23   # Matching Rust implementation
    comparison_dim = 50  # Matching Rust implementation

    model = SequenceLSTM(
        input_dim=sequence_dim,
        hidden_size=args.hidden,
        num_layers=1,
        comparison_dim=comparison_dim,
        dropout=0.3,  # Regularization to prevent overfitting
    )
    use_team_ids = False
    print(f"  LSTM: hidden={args.hidden}, layers=1, dropout=0.3")
    print(f"  Sequence features: {sequence_dim}, Comparison features: {comparison_dim}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Train
    print(f"\n[5/6] Training for {args.epochs} epochs...")
    print(f"  Loss weights: win={args.win_weight}, margin={args.margin_weight}, weight_decay={args.weight_decay}")
    model, history = train_sequence_model(
        model,
        train_samples,
        val_samples,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        win_weight=args.win_weight,
        margin_weight=args.margin_weight,
        use_team_ids=use_team_ids,
        weight_decay=args.weight_decay,
        verbose=True,
    )

    print(f"\n  Best validation accuracy: {history['best_val_acc']:.1%}")

    # Evaluate
    print("\n[6/6] Evaluating on validation set...")
    eval_results = evaluate_sequence_model(model, val_samples, use_team_ids=use_team_ids)

    print(f"\n  Results:")
    print(f"    Win Accuracy: {eval_results['accuracy']:.1%}")
    print(f"    Precision: {eval_results['precision']:.1%}")
    print(f"    Recall: {eval_results['recall']:.1%}")
    print(f"    F1 Score: {eval_results['f1']:.3f}")
    print(f"    Margin MAE: {eval_results['margin_mae']:.1f} points")
    print(f"    Mean margin pred: {eval_results['mean_margin_pred']:.1f}")
    print(f"    Mean margin actual: {eval_results['mean_margin_actual']:.1f}")

    # Save model and normalizer
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{args.model}_model.pt"
    norm_path = model_dir / f"{args.model}_normalizer.npz"
    model.save(model_path)
    normalizer.save(str(norm_path))
    print(f"\n  Model saved to: {model_path}")
    print(f"  Normalizer saved to: {norm_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Win Accuracy: {eval_results['accuracy']:.1%}")
    print(f"Margin MAE:   {eval_results['margin_mae']:.1f} points")


if __name__ == "__main__":
    main()
