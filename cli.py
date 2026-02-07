#!/usr/bin/env python3
"""Unified CLI for rugby prediction system.

Usage:
    python cli.py [--config CONFIG] [--verbose] <command> [args...]

Commands:
    data sync       Scrape Wikipedia and populate database
    data parse-cache Parse cached HTML files
    data fixtures   Fetch upcoming fixtures
    data status     Show database statistics
    train           Train MLP model
    train-lstm      Train LSTM model
    tune-mlp        Hyperparameter search for MLP
    tune-lstm       Hyperparameter search for LSTM
    predict         Predict a single match
    predict-next    Predict upcoming fixtures
    model info      Show model information
    model validate  Validate saved model
    init            Initialize database schema
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rugby.config import Config
from rugby.data import Database, DEFAULT_DB_PATH
from rugby.features import (
    FeatureBuilder,
    FeatureNormalizer,
    MatchFeatures,
    SequenceFeatureBuilder,
    SequenceNormalizer,
)
from rugby.models import MatchPredictor, SequenceLSTM
from rugby.scrapers import WikipediaScraper, HttpCache
from rugby.training import (
    train_match_predictor,
    evaluate_match_predictor,
    train_sequence_model,
    evaluate_sequence_model,
)

log = logging.getLogger(__name__)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "model"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Timezone offsets by country (for team creation during sync)
TIMEZONE_OFFSETS = {
    "NewZealand": 12, "Fiji": 12, "Australia": 10, "Japan": 9,
    "SouthAfrica": 2, "Argentina": -3, "Samoa": 13, "Tonga": 13,
    "England": 0, "France": 1, "Ireland": 0, "Wales": 0,
    "Scotland": 0, "Italy": 1,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rugby", description="Rugby match prediction CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    sub = parser.add_subparsers(dest="command")

    # --- data ---
    data_parser = sub.add_parser("data", help="Data management commands")
    data_sub = data_parser.add_subparsers(dest="data_command")

    sync_p = data_sub.add_parser("sync", help="Scrape Wikipedia and populate database")
    sync_p.add_argument("--source", default="wikipedia", help="Data source (default: wikipedia)")
    sync_p.add_argument("--cache", type=str, default=None, help="Cache directory")
    sync_p.add_argument("--offline", action="store_true", help="Use cached data only")

    parse_p = data_sub.add_parser("parse-cache", help="Parse cached HTML files into database")
    parse_p.add_argument("dir", type=str, help="Directory containing cached HTML files")

    fixtures_p = data_sub.add_parser("fixtures", help="Fetch upcoming fixtures for a year")
    fixtures_p.add_argument("year", type=int, help="Season year")

    data_sub.add_parser("status", help="Show database statistics")

    # --- train ---
    train_p = sub.add_parser("train", help="Train MLP model")
    train_p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    train_p.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_p.add_argument("--production", action="store_true",
                         help="Train on all data (no validation split)")

    # --- train-lstm ---
    lstm_p = sub.add_parser("train-lstm", help="Train LSTM model")
    lstm_p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    lstm_p.add_argument("--lr", type=float, default=None, help="Learning rate")
    lstm_p.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    lstm_p.add_argument("--production", action="store_true",
                         help="Train on all data (no validation split)")

    # --- tune ---
    tune_mlp_p = sub.add_parser("tune-mlp", help="Hyperparameter search for MLP")
    tune_mlp_p.add_argument("--max-epochs", type=int, default=500, help="Maximum epochs to search")

    tune_lstm_p = sub.add_parser("tune-lstm", help="Hyperparameter search for LSTM")
    tune_lstm_p.add_argument("--max-epochs", type=int, default=200, help="Maximum epochs to search")

    # --- predict ---
    predict_p = sub.add_parser("predict", help="Predict a single match")
    predict_p.add_argument("home", type=str, help="Home team name")
    predict_p.add_argument("away", type=str, help="Away team name")
    predict_p.add_argument("--model", choices=["mlp", "lstm"], default="mlp",
                           help="Model to use (default: mlp)")

    # --- predict-next ---
    pn_p = sub.add_parser("predict-next", help="Predict upcoming fixtures")
    pn_p.add_argument("--year", type=int, default=datetime.now().year, help="Season year")
    pn_p.add_argument("--round", type=int, default=None, help="Specific round")
    pn_p.add_argument("--format", choices=["table", "json", "csv"], default="table",
                      help="Output format")
    pn_p.add_argument("--model", choices=["mlp", "lstm"], default="mlp",
                      help="Model to use (default: mlp)")

    # --- model ---
    model_parser = sub.add_parser("model", help="Model management commands")
    model_sub = model_parser.add_subparsers(dest="model_command")
    model_sub.add_parser("info", help="Show model information")
    model_sub.add_parser("validate", help="Validate saved model")

    # --- init ---
    sub.add_parser("init", help="Initialize database schema")

    return parser


def load_config(args) -> Config:
    if args.config:
        return Config.load(Path(args.config))
    # Try default config.toml
    default_path = PROJECT_ROOT / "config.toml"
    if default_path.exists():
        return Config.load(default_path)
    return Config.default()


# =============================================================================
# Command handlers
# =============================================================================

def cmd_init(args, config: Config):
    """Initialize database schema."""
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with Database(db_path) as db:
        db.init_schema(reset=True)
    print(f"Database initialized (reset) at {db_path}")


def cmd_data_status(args, config: Config):
    """Show database statistics."""
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    with Database(db_path) as db:
        stats = db.get_stats()
        teams = db.get_teams()

    print(f"Database: {db_path}")
    print(f"  Matches: {stats['match_count']}")
    print(f"  Teams:   {stats['team_count']}")
    if stats['date_range'][0]:
        print(f"  Range:   {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"\nTeams:")
    for team in sorted(teams.values(), key=lambda t: t.name):
        aliases_str = f" (aliases: {', '.join(team.aliases)})" if team.aliases else ""
        print(f"  {team.id:3d}  {team.name:<20} {team.country or '':<15}{aliases_str}")


def cmd_data_sync(args, config: Config):
    """Scrape Wikipedia and populate database."""
    cache_dir = Path(args.cache) if args.cache else CACHE_DIR
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    cache = HttpCache(cache_dir, offline=args.offline)
    scraper = WikipediaScraper(cache)

    start_year = 1996
    end_year = datetime.now().year
    years = range(start_year, end_year + 1)
    total = len(years)

    raw_matches = []
    for i, year in enumerate(years, 1):
        print(f"\r  Fetching {year} [{i}/{total}]", end="", flush=True)
        try:
            matches = scraper.fetch_season(year)
            raw_matches.extend(matches)
        except Exception as e:
            log.warning("Failed to fetch season %d: %s", year, e)
    print(f"\r  Scraped {len(raw_matches)} matches from {start_year}-{end_year}")

    with Database(db_path) as db:
        db.init_schema(reset=True)
        count = 0
        for raw in raw_matches:
            home_id = db.get_or_create_team(
                raw.home_team.name, raw.home_team.country,
                TIMEZONE_OFFSETS.get(raw.home_team.country, 0),
            )
            away_id = db.get_or_create_team(
                raw.away_team.name, raw.away_team.country,
                TIMEZONE_OFFSETS.get(raw.away_team.country, 0),
            )
            db.upsert_match({
                "date": raw.date,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_score": raw.home_score,
                "away_score": raw.away_score,
                "venue": raw.venue,
                "round": raw.round,
                "home_tries": raw.home_tries,
                "away_tries": raw.away_tries,
                "source": "Wikipedia",
            })
            count += 1

        stats = db.get_stats()
    print(f"  Upserted {count} matches")
    print(f"  Database now has {stats['match_count']} matches, {stats['team_count']} teams")


def cmd_data_parse_cache(args, config: Config):
    """Parse cached HTML files into database."""
    cache_dir = Path(args.dir)
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    cache = HttpCache(cache_dir, offline=True)
    scraper = WikipediaScraper(cache)

    print(f"Parsing cached HTML from {cache_dir}...")
    raw_matches = scraper.parse_directory(cache_dir)
    print(f"  Parsed {len(raw_matches)} matches")

    with Database(db_path) as db:
        db.init_schema()
        count = 0
        for raw in raw_matches:
            home_id = db.get_or_create_team(
                raw.home_team.name, raw.home_team.country,
                TIMEZONE_OFFSETS.get(raw.home_team.country, 0),
            )
            away_id = db.get_or_create_team(
                raw.away_team.name, raw.away_team.country,
                TIMEZONE_OFFSETS.get(raw.away_team.country, 0),
            )
            db.upsert_match({
                "date": raw.date,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_score": raw.home_score,
                "away_score": raw.away_score,
                "venue": raw.venue,
                "round": raw.round,
                "source": "Wikipedia",
            })
            count += 1

        stats = db.get_stats()
    print(f"  Upserted {count} matches")
    print(f"  Database now has {stats['match_count']} matches, {stats['team_count']} teams")


def cmd_data_fixtures(args, config: Config):
    """Fetch upcoming fixtures for a year."""
    cache = HttpCache(CACHE_DIR)
    scraper = WikipediaScraper(cache)

    fixtures = scraper.fetch_fixtures(args.year)
    if not fixtures:
        print(f"No upcoming fixtures found for {args.year}.")
        return

    print(f"{args.year} Super Rugby Pacific Fixtures\n")
    current_round = None
    for f in fixtures:
        if f.round != current_round:
            current_round = f.round
            print(f"  Round {current_round}")
        venue_str = f"  @ {f.venue}" if f.venue else ""
        print(f"    {f.date}  {f.home_team.name:>15} vs {f.away_team.name:<15}{venue_str}")


def cmd_train(args, config: Config):
    """Train MLP model."""
    epochs = args.epochs or config.training.epochs
    lr = args.lr or config.training.learning_rate
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    print("=" * 60)
    print("Rugby Match Prediction - MLP Training")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    with Database(db_path) as db:
        matches = db.get_matches()
        teams = db.get_teams()
        stats = db.get_stats()

    print(f"  Loaded {stats['match_count']} matches, {stats['team_count']} teams")

    print("\n[2/4] Building features...")
    builder = FeatureBuilder(matches, teams)

    if args.production:
        train_cutoff = datetime(2099, 1, 1)
        val_cutoff = datetime(2099, 1, 1)
    else:
        train_cutoff = datetime(2023, 1, 1)
        val_cutoff = datetime(2024, 1, 1)

    X_train_list, y_win_train_list, y_margin_train_list = [], [], []
    X_val_list, y_win_val_list, y_margin_val_list = [], [], []

    for match in sorted(matches, key=lambda m: m.date):
        features = builder.build_features(match)
        builder.process_match(match)
        if features is None:
            continue
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

    print(f"  Training set: {len(X_train)} samples")

    print("\n[3/4] Normalizing features...")
    normalizer = FeatureNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)

    X_val_norm = None
    y_win_val = None
    y_margin_val = None
    if X_val_list:
        X_val = np.array(X_val_list)
        y_win_val = np.array(y_win_val_list)
        y_margin_val = np.array(y_margin_val_list)
        X_val_norm = normalizer.transform(X_val)
        print(f"  Validation set: {len(X_val)} samples")

    print(f"\n[4/4] Training for {epochs} epochs, lr={lr}...")
    model, history = train_match_predictor(
        X_train_norm, y_win_train, y_margin_train,
        X_val_norm, y_win_val, y_margin_val,
        lr=lr, epochs=epochs, batch_size=config.training.batch_size,
        dropout=config.training.dropout,
        win_weight=config.loss.win_weight,
        margin_weight=config.loss.score_weight,
    )

    print(f"\n  Best validation accuracy: {history['best_val_acc']:.1%}")

    if X_val_norm is not None:
        eval_results = evaluate_match_predictor(model, X_val_norm, y_win_val, y_margin_val)
        print(f"  Win Accuracy: {eval_results['win_accuracy']:.1%}")
        print(f"  Margin MAE: {eval_results['margin_mae']:.1f} points")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "match_predictor.pt")
    np.savez(MODEL_DIR / "normalizer.npz", mean=normalizer.mean, std=normalizer.std)
    print(f"\nModel saved to {MODEL_DIR}")


def cmd_train_lstm(args, config: Config):
    """Train LSTM model."""
    epochs = args.epochs or config.training.epochs
    lr = args.lr or 0.001
    hidden_size = args.hidden_size
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    print("=" * 60)
    print("Rugby Match Prediction - LSTM Training")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    with Database(db_path) as db:
        matches = db.get_matches()
        teams = db.get_teams()
        stats = db.get_stats()

    print(f"  Loaded {stats['match_count']} matches, {stats['team_count']} teams")

    if args.production:
        train_cutoff = datetime(2099, 1, 1)
        val_cutoff = datetime(2099, 1, 1)
    else:
        train_cutoff = datetime(2023, 1, 1)
        val_cutoff = datetime(2024, 1, 1)

    print("\n[2/5] Building sequence features...")
    builder = SequenceFeatureBuilder(matches, teams, seq_len=config.model.max_history)

    train_samples, val_samples = [], []
    for match in sorted(matches, key=lambda m: m.date):
        sample = builder.build_sample(match)
        builder.process_match(match)
        if sample is None:
            continue
        if match.date < train_cutoff:
            train_samples.append(sample)
        elif match.date < val_cutoff:
            val_samples.append(sample)

    print(f"  Training: {len(train_samples)}, Validation: {len(val_samples)}")

    print("\n[3/5] Normalizing...")
    normalizer = SequenceNormalizer()
    train_samples = normalizer.fit_transform(train_samples)
    val_samples = normalizer.transform_samples(val_samples)

    print(f"\n[4/5] Creating LSTM (hidden={hidden_size})...")
    model = SequenceLSTM(
        input_dim=23, hidden_size=hidden_size, num_layers=1,
        comparison_dim=50, dropout=config.training.dropout,
    )

    print(f"\n[5/5] Training for {epochs} epochs, lr={lr}...")
    model, history = train_sequence_model(
        model, train_samples, val_samples,
        lr=lr, epochs=epochs, batch_size=config.training.batch_size,
        weight_decay=config.training.weight_decay,
        label_smoothing=0.05,
        augment_swap=True,
    )

    print(f"\n  Best validation accuracy: {history['best_val_acc']:.1%}")

    if val_samples:
        eval_results = evaluate_sequence_model(model, val_samples)
        print(f"  Win Accuracy: {eval_results['accuracy']:.1%}")
        print(f"  Margin MAE: {eval_results['margin_mae']:.1f} points")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "lstm_model.pt")
    normalizer.save(str(MODEL_DIR / "lstm_normalizer.npz"))
    print(f"\nModel saved to {MODEL_DIR}")


def cmd_tune_mlp(args, config: Config):
    """Hyperparameter search for MLP."""
    from rugby.tuning import tune_mlp

    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    print("MLP Hyperparameter Search")
    print("=" * 60)

    with Database(db_path) as db:
        matches = db.get_matches()
        teams = db.get_teams()

    builder = FeatureBuilder(matches, teams)

    train_cutoff = datetime(2023, 1, 1)
    val_cutoff = datetime(2024, 1, 1)

    X_train_l, y_w_train_l, y_m_train_l = [], [], []
    X_val_l, y_w_val_l, y_m_val_l = [], [], []
    X_test_l, y_w_test_l, y_m_test_l = [], [], []

    for match in sorted(matches, key=lambda m: m.date):
        features = builder.build_features(match)
        builder.process_match(match)
        if features is None:
            continue
        margin = float(abs(match.home_score - match.away_score))
        win = 1.0 if match.home_win else 0.0
        arr = features.to_array()
        if match.date < train_cutoff:
            X_train_l.append(arr); y_w_train_l.append(win); y_m_train_l.append(margin)
        elif match.date < val_cutoff:
            X_val_l.append(arr); y_w_val_l.append(win); y_m_val_l.append(margin)
        else:
            X_test_l.append(arr); y_w_test_l.append(win); y_m_test_l.append(margin)

    normalizer = FeatureNormalizer()
    X_train = normalizer.fit_transform(np.array(X_train_l))
    X_val = normalizer.transform(np.array(X_val_l))
    X_test = normalizer.transform(np.array(X_test_l))

    max_epochs = args.max_epochs
    search_space = {
        "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "epochs": [e for e in [50, 100, 200, 300, 500] if e <= max_epochs],
    }

    results = tune_mlp(
        X_train, np.array(y_w_train_l), np.array(y_m_train_l),
        X_val, np.array(y_w_val_l), np.array(y_m_val_l),
        X_test, np.array(y_w_test_l), np.array(y_m_test_l),
        search_space=search_space,
    )

    print(f"\nTop 5 results:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. {r.hyperparams} → val={r.val_acc:.1%}, test={r.test_acc:.1%}")


def cmd_tune_lstm(args, config: Config):
    """Hyperparameter search for LSTM."""
    from rugby.tuning import tune_lstm

    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    print("LSTM Hyperparameter Search")
    print("=" * 60)

    with Database(db_path) as db:
        matches = db.get_matches()
        teams = db.get_teams()

    train_cutoff = datetime(2023, 1, 1)
    val_cutoff = datetime(2024, 1, 1)

    builder = SequenceFeatureBuilder(matches, teams, seq_len=config.model.max_history)
    train_samples, val_samples, test_samples = [], [], []

    for match in sorted(matches, key=lambda m: m.date):
        sample = builder.build_sample(match)
        builder.process_match(match)
        if sample is None:
            continue
        if match.date < train_cutoff:
            train_samples.append(sample)
        elif match.date < val_cutoff:
            val_samples.append(sample)
        else:
            test_samples.append(sample)

    normalizer = SequenceNormalizer()
    train_samples = normalizer.fit_transform(train_samples)
    val_samples = normalizer.transform_samples(val_samples)
    test_samples = normalizer.transform_samples(test_samples)

    max_epochs = args.max_epochs
    search_space = {
        "lr": [0.0005, 0.001, 0.005],
        "hidden_size": [32, 64, 128],
        "epochs": [e for e in [50, 100, 200] if e <= max_epochs],
    }

    results = tune_lstm(train_samples, val_samples, test_samples, search_space=search_space)

    print(f"\nTop 5 results:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. {r.hyperparams} → val={r.val_acc:.1%}, test={r.test_acc:.1%}")


def cmd_predict(args, config: Config):
    """Predict a single match."""
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    with Database(db_path) as db:
        matches = db.get_matches()
        teams = db.get_teams()
        home_team = db.find_team_by_name(args.home)
        away_team = db.find_team_by_name(args.away)

    if not home_team:
        print(f"Error: Unknown team '{args.home}'")
        sys.exit(1)
    if not away_team:
        print(f"Error: Unknown team '{args.away}'")
        sys.exit(1)

    if args.model == "lstm":
        builder = SequenceFeatureBuilder(matches, teams, seq_len=10)
        for match in sorted(matches, key=lambda m: m.date):
            builder.process_match(match)

        norm_data = np.load(MODEL_DIR / "lstm_normalizer.npz")
        normalizer = SequenceNormalizer()
        normalizer.seq_mean = norm_data["seq_mean"]
        normalizer.seq_std = norm_data["seq_std"]
        normalizer.comp_mean = norm_data["comp_mean"]
        normalizer.comp_std = norm_data["comp_std"]

        model = SequenceLSTM(input_dim=23, hidden_size=64, num_layers=1,
                             comparison_dim=50, dropout=0.3)
        model.load(MODEL_DIR / "lstm_model.pt")
        model.train(False)

        now = datetime.now()
        home_seq = builder._build_team_sequence(home_team.id, now)
        away_seq = builder._build_team_sequence(away_team.id, now)
        comparison = builder._build_comparison_features(home_team.id, away_team.id, now)

        if home_seq is None or away_seq is None or comparison is None:
            print("Error: Insufficient match history for prediction")
            sys.exit(1)

        home_seq_norm = (home_seq - normalizer.seq_mean) / normalizer.seq_std
        away_seq_norm = (away_seq - normalizer.seq_mean) / normalizer.seq_std
        comp_norm = (comparison - normalizer.comp_mean) / normalizer.comp_std

        with torch.no_grad():
            preds = model.predict(
                torch.tensor(home_seq_norm, dtype=torch.float32).unsqueeze(0),
                torch.tensor(away_seq_norm, dtype=torch.float32).unsqueeze(0),
                torch.tensor(comp_norm, dtype=torch.float32).unsqueeze(0),
            )
        win_prob = preds["win_prob"].item()
        margin = preds["margin"].item()
    else:
        builder = FeatureBuilder(matches, teams)
        for match in sorted(matches, key=lambda m: m.date):
            builder.build_features(match)
            builder.process_match(match)

        norm_data = np.load(MODEL_DIR / "normalizer.npz")
        normalizer = FeatureNormalizer()
        normalizer.mean = norm_data["mean"]
        normalizer.std = norm_data["std"]

        model = MatchPredictor(len(normalizer.mean), hidden_dims=[64])
        model.load(MODEL_DIR / "match_predictor.pt")
        model.train(False)

        now = datetime.now()
        home_stats = builder._compute_team_stats(home_team.id, now)
        away_stats = builder._compute_team_stats(away_team.id, now)

        if home_stats is None or away_stats is None:
            print("Error: Insufficient match history for prediction")
            sys.exit(1)

        is_local = builder._is_local_derby(home_team.id, away_team.id)
        log5 = builder._log5_prob(home_stats.win_rate, away_stats.win_rate)

        features = MatchFeatures(
            win_rate_diff=home_stats.win_rate - away_stats.win_rate,
            margin_diff=home_stats.margin_avg - away_stats.margin_avg,
            pythagorean_diff=home_stats.pythagorean - away_stats.pythagorean,
            log5=log5,
            is_local=1.0 if is_local else 0.0,
            home_win_rate=home_stats.win_rate,
            home_margin_avg=home_stats.margin_avg,
            home_pythagorean=home_stats.pythagorean,
            home_pf_avg=home_stats.pf_avg,
            home_pa_avg=home_stats.pa_avg,
            away_win_rate=away_stats.win_rate,
            away_margin_avg=away_stats.margin_avg,
            away_pythagorean=away_stats.pythagorean,
            away_pf_avg=away_stats.pf_avg,
            away_pa_avg=away_stats.pa_avg,
            home_elo=home_stats.elo,
            away_elo=away_stats.elo,
            elo_diff=home_stats.elo - away_stats.elo,
            home_streak=home_stats.streak / 5.0,
            away_streak=away_stats.streak / 5.0,
            home_last5_win_rate=home_stats.last_5_wins / 5.0,
            away_last5_win_rate=away_stats.last_5_wins / 5.0,
        )

        X = features.to_array().reshape(1, -1)
        X_norm = normalizer.transform(X)
        X_t = torch.tensor(X_norm, dtype=torch.float32)

        with torch.no_grad():
            preds = model.predict(X_t)
        win_prob = preds["win_prob"].item()
        margin = preds["margin"].item()

    winner = home_team.name if win_prob >= 0.5 else away_team.name
    prob = win_prob if win_prob >= 0.5 else 1 - win_prob
    margin_abs = abs(round(margin))

    print(f"\n  {home_team.name} vs {away_team.name}")
    print(f"  → {winner} by {margin_abs} pts ({prob:.1%})")
    print(f"  Home win probability: {win_prob:.1%}")


def cmd_predict_next(args, config: Config):
    """Predict upcoming fixtures."""
    db_path = Path(config.data.database_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    # Fetch fixtures
    cache = HttpCache(CACHE_DIR)
    scraper = WikipediaScraper(cache)

    raw_fixtures = scraper.fetch_fixtures(args.year)
    if not raw_fixtures:
        print(f"No upcoming fixtures found for {args.year}.")
        return

    # Filter by round
    if args.round is not None:
        raw_fixtures = [f for f in raw_fixtures if f.round == args.round]
        if not raw_fixtures:
            print(f"No fixtures found for round {args.round}.")
            return
    else:
        # Default: next round (earliest round number)
        min_round = min(f.round for f in raw_fixtures if f.round is not None)
        raw_fixtures = [f for f in raw_fixtures if f.round == min_round]

    round_num = raw_fixtures[0].round
    output_json = args.format == "json"
    output_csv = args.format == "csv"

    if not output_json and not output_csv:
        print(f"Predicting Round {round_num} fixtures...\n" if round_num else "")

    # Load data
    with Database(db_path) as db:
        matches = db.get_matches()
        teams = db.get_teams()

    # Build features and load model
    if args.model == "lstm":
        if not output_json and not output_csv:
            print("Loading LSTM model...")
        builder = SequenceFeatureBuilder(matches, teams, seq_len=10)
        for match in sorted(matches, key=lambda m: m.date):
            builder.process_match(match)

        norm_data = np.load(MODEL_DIR / "lstm_normalizer.npz")
        normalizer = SequenceNormalizer()
        normalizer.seq_mean = norm_data["seq_mean"]
        normalizer.seq_std = norm_data["seq_std"]
        normalizer.comp_mean = norm_data["comp_mean"]
        normalizer.comp_std = norm_data["comp_std"]

        model = SequenceLSTM(input_dim=23, hidden_size=64, num_layers=1,
                             comparison_dim=50, dropout=0.3)
        model.load(MODEL_DIR / "lstm_model.pt")
        model.train(False)

        def predict_fn(home_team, away_team):
            now = datetime.now()
            home_seq = builder._build_team_sequence(home_team.id, now)
            away_seq = builder._build_team_sequence(away_team.id, now)
            comparison = builder._build_comparison_features(home_team.id, away_team.id, now)
            if home_seq is None or away_seq is None or comparison is None:
                return None
            home_seq_norm = (home_seq - normalizer.seq_mean) / normalizer.seq_std
            away_seq_norm = (away_seq - normalizer.seq_mean) / normalizer.seq_std
            comp_norm = (comparison - normalizer.comp_mean) / normalizer.comp_std
            with torch.no_grad():
                preds = model.predict(
                    torch.tensor(home_seq_norm, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(away_seq_norm, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(comp_norm, dtype=torch.float32).unsqueeze(0),
                )
            return preds["win_prob"].item(), preds["margin"].item()
    else:
        if not output_json and not output_csv:
            print("Loading MLP model...")
        builder = FeatureBuilder(matches, teams)
        for match in sorted(matches, key=lambda m: m.date):
            builder.build_features(match)
            builder.process_match(match)

        norm_data = np.load(MODEL_DIR / "normalizer.npz")
        normalizer = FeatureNormalizer()
        normalizer.mean = norm_data["mean"]
        normalizer.std = norm_data["std"]

        model = MatchPredictor(len(normalizer.mean), hidden_dims=[64])
        model.load(MODEL_DIR / "match_predictor.pt")
        model.train(False)

        def predict_fn(home_team, away_team):
            now = datetime.now()
            home_stats = builder._compute_team_stats(home_team.id, now)
            away_stats = builder._compute_team_stats(away_team.id, now)
            if home_stats is None or away_stats is None:
                return None
            is_local = builder._is_local_derby(home_team.id, away_team.id)
            log5 = builder._log5_prob(home_stats.win_rate, away_stats.win_rate)
            features = MatchFeatures(
                win_rate_diff=home_stats.win_rate - away_stats.win_rate,
                margin_diff=home_stats.margin_avg - away_stats.margin_avg,
                pythagorean_diff=home_stats.pythagorean - away_stats.pythagorean,
                log5=log5,
                is_local=1.0 if is_local else 0.0,
                home_win_rate=home_stats.win_rate,
                home_margin_avg=home_stats.margin_avg,
                home_pythagorean=home_stats.pythagorean,
                home_pf_avg=home_stats.pf_avg,
                home_pa_avg=home_stats.pa_avg,
                away_win_rate=away_stats.win_rate,
                away_margin_avg=away_stats.margin_avg,
                away_pythagorean=away_stats.pythagorean,
                away_pf_avg=away_stats.pf_avg,
                away_pa_avg=away_stats.pa_avg,
                home_elo=home_stats.elo,
                away_elo=away_stats.elo,
                elo_diff=home_stats.elo - away_stats.elo,
                home_streak=home_stats.streak / 5.0,
                away_streak=away_stats.streak / 5.0,
                home_last5_win_rate=home_stats.last_5_wins / 5.0,
                away_last5_win_rate=away_stats.last_5_wins / 5.0,
            )
            X = features.to_array().reshape(1, -1)
            X_norm = normalizer.transform(X)
            X_t = torch.tensor(X_norm, dtype=torch.float32)
            with torch.no_grad():
                preds = model.predict(X_t)
            return preds["win_prob"].item(), preds["margin"].item()

    # Find DB teams and predict
    def find_db_team(name):
        for t in teams.values():
            if t.name.lower() == name.lower():
                return t
        for t in teams.values():
            if name.lower() in t.name.lower() or t.name.lower() in name.lower():
                return t
        return None

    results = []
    for fixture in raw_fixtures:
        home_team = find_db_team(fixture.home_team.name)
        away_team = find_db_team(fixture.away_team.name)

        if not home_team:
            if not output_json:
                print(f"  Warning: Unknown team '{fixture.home_team.name}'")
            continue
        if not away_team:
            if not output_json:
                print(f"  Warning: Unknown team '{fixture.away_team.name}'")
            continue

        pred = predict_fn(home_team, away_team)
        if pred is None:
            if not output_json:
                print(f"  Warning: Insufficient history for {home_team.name} vs {away_team.name}")
            continue

        win_prob, margin = pred
        results.append({
            "date": fixture.date.isoformat(),
            "round": fixture.round,
            "home": home_team.name,
            "away": away_team.name,
            "venue": fixture.venue,
            "home_win_prob": win_prob,
            "margin": round(margin),
        })

    if not results:
        print("Could not make any predictions.")
        return

    # Output
    if output_json:
        print(json.dumps(results, indent=2))
    elif output_csv:
        print("date,round,home,away,home_win_prob,margin")
        for r in results:
            print(f"{r['date']},{r['round']},{r['home']},{r['away']},"
                  f"{r['home_win_prob']:.3f},{r['margin']}")
    else:
        for r in results:
            winner = r["home"] if r["home_win_prob"] >= 0.5 else r["away"]
            prob = r["home_win_prob"] if r["home_win_prob"] >= 0.5 else 1 - r["home_win_prob"]
            margin_abs = abs(r["margin"])
            print(f"  {r['date']}  {r['home']:>15} vs {r['away']:<15}")
            print(f"           → {winner} by {margin_abs} pts ({prob:.1%})\n")


def cmd_model_info(args, config: Config):
    """Show model information."""
    print("Model files:")
    if not MODEL_DIR.exists():
        print("  No model directory found.")
        return

    for f in sorted(MODEL_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<30} {size_kb:>8.1f} KB")


def cmd_model_validate(args, config: Config):
    """Validate saved models can be loaded."""
    errors = []

    # MLP
    mlp_path = MODEL_DIR / "match_predictor.pt"
    norm_path = MODEL_DIR / "normalizer.npz"
    if mlp_path.exists() and norm_path.exists():
        try:
            norm_data = np.load(norm_path)
            normalizer = FeatureNormalizer()
            normalizer.mean = norm_data["mean"]
            normalizer.std = norm_data["std"]
            model = MatchPredictor(len(normalizer.mean), hidden_dims=[64])
            model.load(mlp_path)
            print(f"  MLP model: OK ({len(normalizer.mean)} features)")
        except Exception as e:
            errors.append(f"MLP: {e}")
            print(f"  MLP model: FAILED ({e})")
    else:
        print("  MLP model: not found")

    # LSTM
    lstm_path = MODEL_DIR / "lstm_model.pt"
    lstm_norm_path = MODEL_DIR / "lstm_normalizer.npz"
    if lstm_path.exists() and lstm_norm_path.exists():
        try:
            norm_data = np.load(lstm_norm_path)
            model = SequenceLSTM(input_dim=23, hidden_size=64, num_layers=1,
                                 comparison_dim=50, dropout=0.3)
            model.load(lstm_path)
            print(f"  LSTM model: OK")
        except Exception as e:
            errors.append(f"LSTM: {e}")
            print(f"  LSTM model: FAILED ({e})")
    else:
        print("  LSTM model: not found")

    if errors:
        sys.exit(1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = load_config(args)

    handlers = {
        "init": cmd_init,
        "train": cmd_train,
        "train-lstm": cmd_train_lstm,
        "tune-mlp": cmd_tune_mlp,
        "tune-lstm": cmd_tune_lstm,
        "predict": cmd_predict,
        "predict-next": cmd_predict_next,
    }

    if args.command == "data":
        if not args.data_command:
            print("Usage: rugby data {sync|parse-cache|fixtures|status}")
            sys.exit(1)
        data_handlers = {
            "sync": cmd_data_sync,
            "parse-cache": cmd_data_parse_cache,
            "fixtures": cmd_data_fixtures,
            "status": cmd_data_status,
        }
        data_handlers[args.data_command](args, config)
    elif args.command == "model":
        if not args.model_command:
            print("Usage: rugby model {info|validate}")
            sys.exit(1)
        model_handlers = {
            "info": cmd_model_info,
            "validate": cmd_model_validate,
        }
        model_handlers[args.model_command](args, config)
    elif args.command in handlers:
        handlers[args.command](args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
