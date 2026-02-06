#!/usr/bin/env python3
"""
Predict upcoming Super Rugby Pacific matches by scraping fixtures from Wikipedia.

Usage:
    python predict_next.py                    # predict next round (MLP)
    python predict_next.py --round 3          # predict specific round
    python predict_next.py --model lstm       # use LSTM model
    python predict_next.py --all              # predict all upcoming fixtures
    python predict_next.py --json             # JSON output
    python predict_next.py --csv              # CSV output
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from rugby.data import Database, Team
from rugby.features import (
    FeatureBuilder,
    FeatureNormalizer,
    MatchFeatures,
    SequenceFeatureBuilder,
    SequenceNormalizer,
)
from rugby.models import MatchPredictor, SequenceLSTM
from rugby.scrapers import WikipediaScraper, HttpCache


MODEL_DIR = Path(__file__).parent.parent / "model"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def find_db_team(teams: Dict[int, Team], name: str) -> Optional[Team]:
    """Look up a team in the database by canonical name."""
    for t in teams.values():
        if t.name.lower() == name.lower():
            return t
    # Partial match fallback
    for t in teams.values():
        if name.lower() in t.name.lower() or t.name.lower() in name.lower():
            return t
    return None


# ---------- MLP prediction ----------

def load_mlp():
    """Load MLP model and normalizer."""
    norm_data = np.load(MODEL_DIR / "normalizer.npz")
    normalizer = FeatureNormalizer()
    normalizer.mean = norm_data["mean"]
    normalizer.std = norm_data["std"]

    model = MatchPredictor(len(normalizer.mean), hidden_dims=[64])
    model.load(MODEL_DIR / "match_predictor.pt")
    model.train(False)
    return model, normalizer


def predict_mlp(model, normalizer, builder, home_team, away_team):
    """Predict using MLP model."""
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


# ---------- LSTM prediction ----------

def load_lstm():
    """Load LSTM model and normalizer."""
    norm_data = np.load(MODEL_DIR / "lstm_normalizer.npz")
    normalizer = SequenceNormalizer()
    normalizer.seq_mean = norm_data["seq_mean"]
    normalizer.seq_std = norm_data["seq_std"]
    normalizer.comp_mean = norm_data["comp_mean"]
    normalizer.comp_std = norm_data["comp_std"]

    model = SequenceLSTM(
        input_dim=23, hidden_size=64, num_layers=1,
        comparison_dim=50, dropout=0.3,
    )
    model.load(MODEL_DIR / "lstm_model.pt")
    model.train(False)
    return model, normalizer


def predict_lstm(model, normalizer, builder, home_team, away_team):
    """Predict using LSTM model."""
    now = datetime.now()
    home_seq = builder._build_team_sequence(home_team.id, now)
    away_seq = builder._build_team_sequence(away_team.id, now)
    comparison = builder._build_comparison_features(home_team.id, away_team.id, now)

    if home_seq is None or away_seq is None or comparison is None:
        return None

    home_seq_norm = (home_seq - normalizer.seq_mean) / normalizer.seq_std
    away_seq_norm = (away_seq - normalizer.seq_mean) / normalizer.seq_std
    comp_norm = (comparison - normalizer.comp_mean) / normalizer.comp_std

    home_t = torch.tensor(home_seq_norm, dtype=torch.float32).unsqueeze(0)
    away_t = torch.tensor(away_seq_norm, dtype=torch.float32).unsqueeze(0)
    comp_t = torch.tensor(comp_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        preds = model.predict(home_t, away_t, comp_t)
    return preds["win_prob"].item(), preds["margin"].item()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Predict upcoming Super Rugby Pacific matches")
    parser.add_argument("--year", type=int, default=datetime.now().year,
                        help="Season year (default: current)")
    parser.add_argument("--round", type=int, default=None,
                        help="Predict a specific round (default: next round)")
    parser.add_argument("--all", action="store_true",
                        help="Predict all upcoming fixtures")
    parser.add_argument("--model", choices=["mlp", "lstm"], default="mlp",
                        help="Model to use (default: mlp)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--csv", action="store_true", help="CSV output")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch fixtures from Wikipedia (ignore cache)")
    args = parser.parse_args()

    # Fetch fixtures using scrapers module
    cache = HttpCache(CACHE_DIR)
    scraper = WikipediaScraper(cache)

    if args.refresh:
        # Delete cached files to force re-fetch
        for url in scraper.get_season_urls(args.year):
            cache_path = CACHE_DIR / (
                url.replace("https://", "").replace("/", "_").replace("?", "_") + ".html"
            )
            if cache_path.exists():
                cache_path.unlink()

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
    elif not args.all:
        # Default: next round (earliest round number)
        min_round = min(f.round for f in raw_fixtures if f.round is not None)
        raw_fixtures = [f for f in raw_fixtures if f.round == min_round]

    round_num = raw_fixtures[0].round
    if not args.json and not args.csv:
        print(f"Predicting Round {round_num} fixtures...\n" if round_num else "")

    # Load data
    db = Database()
    matches = db.get_matches()
    teams = db.get_teams()
    db.close()

    # Build feature history
    if args.model == "lstm":
        print("Loading LSTM model...") if not args.json and not args.csv else None
        builder = SequenceFeatureBuilder(matches, teams, seq_len=10)
        for match in sorted(matches, key=lambda m: m.date):
            builder.process_match(match)
        model, normalizer = load_lstm()
        predict_fn = lambda h, a: predict_lstm(model, normalizer, builder, h, a)
    else:
        print("Loading MLP model...") if not args.json and not args.csv else None
        builder = FeatureBuilder(matches, teams)
        for match in sorted(matches, key=lambda m: m.date):
            builder.build_features(match)
            builder.process_match(match)
        model, normalizer = load_mlp()
        predict_fn = lambda h, a: predict_mlp(model, normalizer, builder, h, a)

    # Predict
    results = []
    for fixture in raw_fixtures:
        home_team = find_db_team(teams, fixture.home_team.name)
        away_team = find_db_team(teams, fixture.away_team.name)

        if not home_team:
            print(f"  Warning: Unknown team '{fixture.home_team.name}'") if not args.json else None
            continue
        if not away_team:
            print(f"  Warning: Unknown team '{fixture.away_team.name}'") if not args.json else None
            continue

        pred = predict_fn(home_team, away_team)
        if pred is None:
            print(f"  Warning: Insufficient history for {home_team.name} vs {away_team.name}") if not args.json else None
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
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    elif args.csv:
        print("date,round,home,away,home_win_prob,margin")
        for r in results:
            print(f"{r['date']},{r['round']},{r['home']},{r['away']},"
                  f"{r['home_win_prob']:.3f},{r['margin']}")
    else:
        current_round = None
        for r in results:
            if r["round"] != current_round:
                current_round = r["round"]
                if args.all:
                    print(f"\n{args.year} Super Rugby Pacific - Round {current_round}\n")

            winner = r["home"] if r["home_win_prob"] >= 0.5 else r["away"]
            prob = r["home_win_prob"] if r["home_win_prob"] >= 0.5 else 1 - r["home_win_prob"]
            margin_abs = abs(r["margin"])

            print(f"  {r['date']}  {r['home']:>15} vs {r['away']:<15}")
            print(f"           → {winner} by {margin_abs} pts ({prob:.1%})\n")


if __name__ == "__main__":
    main()
