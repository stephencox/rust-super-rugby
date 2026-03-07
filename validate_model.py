#!/usr/bin/env python3
"""Validate model predictions against known 2026 results.

Creates a rolled-back DB (no 2026 matches), trains a production model on it,
then predicts all 2026 matches and scores them.

Usage:
    python validate_model.py              # Single run
    python validate_model.py --ensemble 5 # Average over 5 training runs
"""

import argparse
import inspect
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rugby.config import Config
from rugby.data import Database
from rugby.features import FeatureBuilder, FeatureNormalizer
from rugby.models import WinClassifier, MarginRegressor
from rugby.training import train_win_model, train_margin_model, MLPDataset

PROJECT_ROOT = Path(__file__).parent

# Detect signed vs absolute margin mode
USE_SIGNED = 'negate_label' in inspect.signature(MLPDataset.__init__).parameters


def create_rollback_db(src_db: Path, dst_db: Path, cutoff: datetime):
    """Copy DB and remove matches on or after cutoff date."""
    shutil.copy2(src_db, dst_db)
    conn = sqlite3.connect(dst_db)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    deleted = conn.execute("DELETE FROM matches WHERE date >= ?", (cutoff_str,)).rowcount
    conn.commit()
    conn.close()
    return deleted


def prepare_data(config: Config, db_path: Path, predict_matches: list, teams: dict):
    """Prepare training data and prediction features."""
    with Database(db_path) as db:
        train_matches = db.get_matches()
        train_teams = db.get_teams()

    builder = FeatureBuilder(train_matches, train_teams)

    X_list, y_win_list, y_margin_list = [], [], []
    home_ids, away_ids = [], []
    team_to_idx = {tid: i + 1 for i, tid in enumerate(sorted(train_teams.keys()))}
    num_teams = len(train_teams)

    for match in sorted(train_matches, key=lambda m: m.date):
        features = builder.build_features(match)
        builder.process_match(match)
        if features is None:
            continue
        margin = match.home_score - match.away_score
        margin_target = float(margin) if USE_SIGNED else float(abs(margin))
        X_list.append(features.to_array())
        y_win_list.append(1.0 if match.home_win else 0.0)
        y_margin_list.append(margin_target)
        home_ids.append(team_to_idx.get(match.home_team_id, 0))
        away_ids.append(team_to_idx.get(match.away_team_id, 0))

    # Build prediction features (continuing the feature builder state)
    pred_features = []
    pred_home_ids = []
    pred_away_ids = []
    for match in sorted(predict_matches, key=lambda m: m.date):
        features = builder.build_features(match)
        builder.process_match(match)
        pred_features.append(features)
        pred_home_ids.append(team_to_idx.get(match.home_team_id, 0))
        pred_away_ids.append(team_to_idx.get(match.away_team_id, 0))

    return {
        'X': np.array(X_list),
        'y_win': np.array(y_win_list),
        'y_margin': np.array(y_margin_list),
        'home_ids': np.array(home_ids, dtype=np.int64),
        'away_ids': np.array(away_ids, dtype=np.int64),
        'num_teams': num_teams,
        'pred_features': pred_features,
        'pred_home_ids': pred_home_ids,
        'pred_away_ids': pred_away_ids,
    }


def train_and_predict(config: Config, db_path: Path, predict_matches: list, teams: dict,
                      n_ensemble: int = 1):
    """Train and predict with optional ensembling."""
    data = prepare_data(config, db_path, predict_matches, teams)
    sorted_matches = sorted(predict_matches, key=lambda m: m.date)

    # Accumulate predictions across ensemble members
    n_matches = len(sorted_matches)
    all_win_probs = [[] for _ in range(n_matches)]
    all_margins = [[] for _ in range(n_matches)]

    for run in range(n_ensemble):
        if n_ensemble > 1:
            print(f"    Ensemble run {run + 1}/{n_ensemble}...")

        normalizer = FeatureNormalizer()
        X_norm = normalizer.fit_transform(data['X'])

        train_kwargs = dict(
            hidden_dims=config.model.hidden_dims,
            lr=config.training.learning_rate,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            dropout=config.training.dropout,
            weight_decay=config.training.weight_decay,
            use_batchnorm=True,
            early_stopping_patience=config.training.early_stopping_patience,
            augment_swap=True,
            home_team_ids=data['home_ids'],
            away_team_ids=data['away_ids'],
            num_teams=data['num_teams'],
            verbose=False,
        )

        win_model, _ = train_win_model(X_norm, data['y_win'], **train_kwargs)
        margin_model, margin_history = train_margin_model(
            X_norm, data['y_margin'], **train_kwargs)

        margin_mean = margin_history.get('margin_offset', 0.0)
        margin_std = margin_history.get('margin_scale', 1.0)

        win_model.train(False)
        margin_model.train(False)

        use_team_ids = data['num_teams'] > 0

        for i, features in enumerate(data['pred_features']):
            if features is None:
                continue
            x = normalizer.transform(np.array([features.to_array()]))
            x_t = torch.tensor(x, dtype=torch.float32)
            home_id_t = torch.tensor([data['pred_home_ids'][i]], dtype=torch.long) if use_team_ids else None
            away_id_t = torch.tensor([data['pred_away_ids'][i]], dtype=torch.long) if use_team_ids else None
            with torch.no_grad():
                wp = torch.sigmoid(win_model(x_t, home_id_t, away_id_t)).item()
                mp = margin_model(x_t, home_id_t, away_id_t)
                mq50 = mp[0, 1].item() * margin_std + margin_mean
            all_win_probs[i].append(wp)
            all_margins[i].append(mq50)

    # Average and score
    predictions = []
    for i, match in enumerate(sorted_matches):
        if not all_win_probs[i]:
            continue

        win_prob = float(np.mean(all_win_probs[i]))
        margin_q50 = float(np.mean(all_margins[i]))

        home = teams[match.home_team_id].name
        away = teams[match.away_team_id].name

        pred_winner = home if win_prob >= 0.5 else away
        actual_winner = home if match.home_win else away
        actual_abs_margin = abs(match.home_score - match.away_score)
        pred_abs_margin = abs(margin_q50)

        correct_winner = (pred_winner == actual_winner)
        if correct_winner:
            margin_error = abs(pred_abs_margin - actual_abs_margin)
        else:
            margin_error = pred_abs_margin + actual_abs_margin
        within_5 = margin_error <= 5

        predictions.append({
            'home': home,
            'away': away,
            'actual_winner': actual_winner,
            'actual_margin': actual_abs_margin,
            'pred_winner': pred_winner,
            'pred_margin': round(pred_abs_margin, 1),
            'win_prob': round(win_prob, 3),
            'correct_winner': correct_winner,
            'margin_error': round(margin_error, 1),
            'within_5': within_5,
        })

    return predictions


def score_predictions(predictions: list, label: str):
    """Score predictions and print summary."""
    total = len(predictions)
    wp = sum(1 for p in predictions if p['correct_winner'])
    mp = sum(1 for p in predictions if p['within_5'])
    tdm = sum(p['margin_error'] for p in predictions)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Winners correct (WP): {wp}/{total}")
    print(f"  Margins within 5 (MP): {mp}/{total}")
    print(f"  Total Difference in Margin (TDM): {tdm:.0f}")
    print(f"{'='*60}")

    print(f"\n  {'Match':<40s} {'Pred':>12s} {'Actual':>12s} {'Err':>5s} {'WP':>3s} {'MP':>3s}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*5} {'-'*3} {'-'*3}")
    for p in predictions:
        match_str = f"{p['home']} vs {p['away']}"
        pred_str = f"{p['pred_winner'][:10]} by {p['pred_margin']:.0f}"
        actual_str = f"{p['actual_winner'][:10]} by {p['actual_margin']:.0f}"
        wp_str = "Y" if p['correct_winner'] else "N"
        mp_str = "Y" if p['within_5'] else "N"
        print(f"  {match_str:<40s} {pred_str:>12s} {actual_str:>12s} {p['margin_error']:>5.0f} {wp_str:>3s} {mp_str:>3s}")

    return {'wp': wp, 'mp': mp, 'tdm': tdm, 'total': total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", "-n", type=int, default=1,
                        help="Number of ensemble members (default: 1)")
    args = parser.parse_args()

    cutoff = datetime(2026, 1, 1)
    results = {}

    for comp, db_name in [("super-rugby", "rugby.db"), ("sixnations", "sixnations.db")]:
        print(f"\n{'#'*60}")
        print(f"  Validating: {comp} (ensemble={args.ensemble})")
        print(f"{'#'*60}")

        config = Config.load(PROJECT_ROOT / "config.toml", comp)
        src_db = PROJECT_ROOT / "data" / db_name
        rollback_db = PROJECT_ROOT / "data" / f"{db_name.replace('.db', '')}_rollback.db"

        with Database(src_db) as db:
            all_matches = db.get_matches()
            teams = db.get_teams()

        matches_2026 = [m for m in all_matches if m.date >= cutoff]
        matches_2026.sort(key=lambda m: m.date)

        if not matches_2026:
            print(f"  No 2026 matches found for {comp}")
            continue

        deleted = create_rollback_db(src_db, rollback_db, cutoff)
        print(f"  Removed {deleted} matches, predicting {len(matches_2026)}...")

        try:
            predictions = train_and_predict(
                config, rollback_db, matches_2026, teams, n_ensemble=args.ensemble)
            results[comp] = score_predictions(predictions, f"{comp} Results")
        finally:
            rollback_db.unlink(missing_ok=True)

    if results:
        total_wp = sum(r['wp'] for r in results.values())
        total_mp = sum(r['mp'] for r in results.values())
        total_tdm = sum(r['tdm'] for r in results.values())
        total_n = sum(r['total'] for r in results.values())

        print(f"\n{'='*60}")
        print(f"  OVERALL: WP={total_wp}/{total_n}, MP={total_mp}/{total_n}, TDM={total_tdm:.0f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
