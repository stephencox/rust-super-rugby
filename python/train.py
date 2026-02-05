#!/usr/bin/env python3
"""
Train rugby prediction models.

Usage:
    python train.py [--epochs N] [--lr RATE] [--hidden DIMS]
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from rugby.data import Database
from rugby.features import FeatureBuilder, FeatureNormalizer, MatchFeatures
from rugby.training import train_match_predictor, evaluate_match_predictor


def main():
    parser = argparse.ArgumentParser(description='Train rugby prediction models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden', type=str, default='64', help='Hidden layer dims (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--train-cutoff', type=str, default='2023-01-01', help='Training cutoff date')
    parser.add_argument('--val-cutoff', type=str, default='2024-01-01', help='Validation end date')
    parser.add_argument('--win-weight', type=float, default=1.0, help='Weight for win loss')
    parser.add_argument('--margin-weight', type=float, default=0.1, help='Weight for margin loss')
    args = parser.parse_args()

    hidden_dims = [int(x) for x in args.hidden.split(',')]
    train_cutoff = datetime.strptime(args.train_cutoff, '%Y-%m-%d')
    val_cutoff = datetime.strptime(args.val_cutoff, '%Y-%m-%d')

    print("="*60)
    print("Rugby Match Prediction - Combined Model Training")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...")
    db = Database()
    matches = db.get_matches()
    teams = db.get_teams()
    stats = db.get_stats()
    db.close()

    print(f"  Loaded {stats['match_count']} matches, {stats['team_count']} teams")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")

    # Build features
    print("\n[2/4] Building features...")
    builder = FeatureBuilder(matches, teams)

    # Collect train and val separately
    X_train_list, y_win_train_list, y_margin_train_list = [], [], []
    X_val_list, y_win_val_list, y_margin_val_list = [], [], []

    for match in sorted(matches, key=lambda m: m.date):
        features = builder.build_features(match)
        builder.process_match(match)

        if features is None:
            continue

        # Margin is absolute points difference
        margin = abs(match.home_score - match.away_score)

        if match.date < train_cutoff:
            X_train_list.append(features.to_array())
            y_win_train_list.append(1.0 if match.home_win else 0.0)
            y_margin_train_list.append(float(margin))
        elif match.date < val_cutoff:
            X_val_list.append(features.to_array())
            y_win_val_list.append(1.0 if match.home_win else 0.0)
            y_margin_val_list.append(float(margin))

    X_train = np.array(X_train_list)
    y_win_train = np.array(y_win_train_list)
    y_margin_train = np.array(y_margin_train_list)

    X_val = np.array(X_val_list)
    y_win_val = np.array(y_win_val_list)
    y_margin_val = np.array(y_margin_val_list)

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Features: {X_train.shape[1]} ({', '.join(MatchFeatures.feature_names()[:5])}...)")
    print(f"  Home win rate (train): {y_win_train.mean():.1%}")
    print(f"  Home win rate (val): {y_win_val.mean():.1%}")
    print(f"  Mean margin (train): {y_margin_train.mean():.1f} points")
    print(f"  Mean margin (val): {y_margin_val.mean():.1f} points")

    # Normalize features
    print("\n[3/4] Normalizing features (z-score)...")
    normalizer = FeatureNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    print(f"  Feature mean (sample): {normalizer.mean[:3]}")
    print(f"  Feature std (sample): {normalizer.std[:3]}")

    # Train combined model
    print("\n[4/4] Training Combined Model (Win -> Margin)...")
    print(f"  Architecture: input({X_train.shape[1]}) -> {hidden_dims} -> win_prob")
    print(f"               [input + win_prob] -> {hidden_dims} -> margin")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Loss weights: win={args.win_weight}, margin={args.margin_weight}")

    model, history = train_match_predictor(
        X_train_norm, y_win_train, y_margin_train,
        X_val_norm, y_win_val, y_margin_val,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        win_weight=args.win_weight,
        margin_weight=args.margin_weight,
    )

    print(f"\n  Best validation accuracy: {history['best_val_acc']:.1%}")

    # Evaluate model
    eval_results = evaluate_match_predictor(
        model, X_val_norm, y_win_val, y_margin_val
    )

    print(f"\n  Model Evaluation:")
    print(f"    Win Accuracy: {eval_results['win_accuracy']:.1%}")
    print(f"    Precision: {eval_results['precision']:.1%}")
    print(f"    Recall: {eval_results['recall']:.1%}")
    print(f"    F1: {eval_results['f1']:.3f}")
    print(f"    Margin MAE: {eval_results['margin_mae']:.1f} points")
    print(f"    Mean margin pred: {eval_results['mean_margin_pred']:.1f}")
    print(f"    Mean margin actual: {eval_results['mean_margin_actual']:.1f}")
    print(f"    Mean win prob: {eval_results['mean_win_prob']:.2f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Win Accuracy: {eval_results['win_accuracy']:.1%}")
    print(f"Margin MAE:   {eval_results['margin_mae']:.1f} points")

    # Save model
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(exist_ok=True)

    model.save(model_dir / "match_predictor.pt")

    # Save normalizer
    np.savez(model_dir / "normalizer.npz", mean=normalizer.mean, std=normalizer.std)

    print(f"\nModel saved to: {model_dir}")


if __name__ == "__main__":
    main()
