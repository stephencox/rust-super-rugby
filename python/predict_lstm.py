#!/usr/bin/env python3
"""
Predict rugby match outcomes using the LSTM model.

Usage:
    python predict_lstm.py <home_team> <away_team>
    python predict_lstm.py --round  (predict all matches in a list)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rugby.data import Database
from rugby.features import SequenceFeatureBuilder, SequenceNormalizer, SequenceDataSample
from rugby.models import SequenceLSTM


MODEL_DIR = Path(__file__).parent.parent / "model"


def load_model_and_normalizer():
    """Load trained LSTM model and normalizer."""
    # Load normalizer
    norm_data = np.load(MODEL_DIR / "lstm_normalizer.npz")
    normalizer = SequenceNormalizer()
    normalizer.seq_mean = norm_data['seq_mean']
    normalizer.seq_std = norm_data['seq_std']
    normalizer.comp_mean = norm_data['comp_mean']
    normalizer.comp_std = norm_data['comp_std']

    # Load model
    sequence_dim = 23
    comparison_dim = 50
    model = SequenceLSTM(
        input_dim=sequence_dim,
        hidden_size=64,
        num_layers=1,
        comparison_dim=comparison_dim,
        dropout=0.3,
    )
    model.load(MODEL_DIR / "lstm_model.pt")
    model.train(False)

    return model, normalizer


def build_feature_builder(matches, teams):
    """Build feature builder and process all historical matches."""
    builder = SequenceFeatureBuilder(matches, teams, seq_len=10)

    # Process all matches chronologically to build up history
    for match in sorted(matches, key=lambda m: m.date):
        builder.process_match(match)

    return builder


def find_team(teams, query):
    """Find team by name (case-insensitive partial match)."""
    query_lower = query.lower()
    matches = [t for t in teams.values() if query_lower in t.name.lower()]
    return matches


def predict_match(model, normalizer, builder, home_team, away_team):
    """Predict a single match using the LSTM."""
    now = datetime.now()

    # Build sequence features for both teams
    home_seq = builder._build_team_sequence(home_team.id, now)
    away_seq = builder._build_team_sequence(away_team.id, now)
    comparison = builder._build_comparison_features(
        home_team.id, away_team.id, now
    )

    if home_seq is None:
        return None, f"Not enough history for {home_team.name}"
    if away_seq is None:
        return None, f"Not enough history for {away_team.name}"
    if comparison is None:
        return None, f"Cannot compute comparison features"

    # Normalize
    home_seq_norm = (home_seq - normalizer.seq_mean) / normalizer.seq_std
    away_seq_norm = (away_seq - normalizer.seq_mean) / normalizer.seq_std
    comp_norm = (comparison - normalizer.comp_mean) / normalizer.comp_std

    # Convert to tensors [1, seq_len, feat_dim] and [1, comp_dim]
    home_t = torch.tensor(home_seq_norm, dtype=torch.float32).unsqueeze(0)
    away_t = torch.tensor(away_seq_norm, dtype=torch.float32).unsqueeze(0)
    comp_t = torch.tensor(comp_norm, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        preds = model.predict(home_t, away_t, comp_t)
        win_prob = preds['win_prob'].item()
        margin = preds['margin'].item()

    return {
        'home_team': home_team.name,
        'away_team': away_team.name,
        'home_win_prob': win_prob,
        'margin': margin,
    }, None


def main():
    parser = argparse.ArgumentParser(description='Predict rugby matches with LSTM')
    parser.add_argument('home_team', nargs='?', help='Home team name')
    parser.add_argument('away_team', nargs='?', help='Away team name')
    parser.add_argument('--round', nargs='*', metavar='MATCHUP',
                        help='Predict multiple matches: "Home1 Away1" "Home2 Away2"')
    parser.add_argument('--list-teams', action='store_true', help='List all teams')
    args = parser.parse_args()

    # Load data
    db = Database()
    matches = db.get_matches()
    teams = db.get_teams()
    db.close()

    if args.list_teams:
        print("Available teams:")
        for team in sorted(teams.values(), key=lambda t: t.name):
            print(f"  {team.name}")
        return

    # Build feature history
    print("Building match history...")
    builder = build_feature_builder(matches, teams)

    # Load model
    print("Loading LSTM model...")
    model, normalizer = load_model_and_normalizer()
    print()

    # Determine matchups to predict
    matchups = []
    if args.round is not None:
        for matchup in args.round:
            parts = matchup.split()
            if len(parts) == 2:
                matchups.append((parts[0], parts[1]))
            else:
                print(f"Invalid matchup: {matchup}")
    elif args.home_team and args.away_team:
        matchups.append((args.home_team, args.away_team))
    else:
        parser.print_help()
        return

    for home_name, away_name in matchups:
        home_matches = find_team(teams, home_name)
        away_matches = find_team(teams, away_name)

        if len(home_matches) == 0:
            print(f"No team found matching '{home_name}'")
            continue
        if len(home_matches) > 1:
            print(f"Multiple teams match '{home_name}':")
            for t in home_matches:
                print(f"  {t.name}")
            continue
        if len(away_matches) == 0:
            print(f"No team found matching '{away_name}'")
            continue
        if len(away_matches) > 1:
            print(f"Multiple teams match '{away_name}':")
            for t in away_matches:
                print(f"  {t.name}")
            continue

        home_team = home_matches[0]
        away_team = away_matches[0]

        result, error = predict_match(model, normalizer, builder, home_team, away_team)

        if error:
            print(f"  {error}")
            continue

        win_prob = result['home_win_prob']
        margin = result['margin']
        winner = result['home_team'] if win_prob >= 0.5 else result['away_team']
        prob = win_prob if win_prob >= 0.5 else 1 - win_prob

        print(f"  {result['home_team']:>15} vs {result['away_team']:<15}")
        print(f"           → {winner} by {margin:.0f} pts ({prob:.1%})")
        print()


if __name__ == "__main__":
    main()
