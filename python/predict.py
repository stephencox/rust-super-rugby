#!/usr/bin/env python3
"""
Predict rugby match outcomes.

Usage:
    python predict.py <home_team> <away_team>
    python predict.py --rankings
    python predict.py --list-teams
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rugby.data import Database
from rugby.features import FeatureBuilder, FeatureNormalizer
from rugby.models import MatchPredictor


def load_model():
    """Load trained model and normalizer."""
    model_dir = Path(__file__).parent.parent / "model"

    # Load normalizer
    norm_data = np.load(model_dir / "normalizer.npz")
    normalizer = FeatureNormalizer()
    normalizer.mean = norm_data['mean']
    normalizer.std = norm_data['std']

    # Load model
    input_dim = len(normalizer.mean)
    model = MatchPredictor(input_dim, hidden_dims=[64])
    model.load(model_dir / "match_predictor.pt")
    model.train(False)

    return model, normalizer


def find_team(teams, query):
    """Find team by name (case-insensitive partial match)."""
    query_lower = query.lower()
    matches = [t for t in teams.values() if query_lower in t.name.lower()]
    return matches


def predict_match(model, normalizer, builder, home_team, away_team):
    """Predict a single match."""
    now = datetime.now()

    home_stats = builder._compute_team_stats(home_team.id, now)
    away_stats = builder._compute_team_stats(away_team.id, now)

    if home_stats is None:
        return None, f"Not enough recent data for {home_team.name}"
    if away_stats is None:
        return None, f"Not enough recent data for {away_team.name}"

    # Build features
    is_local = builder._is_local_derby(home_team.id, away_team.id)
    log5 = builder._log5_prob(home_stats.win_rate, away_stats.win_rate)

    from rugby.features import MatchFeatures
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
        win_prob = preds['win_prob'].item()
        margin = preds['margin'].item()

    return {
        'home_team': home_team.name,
        'away_team': away_team.name,
        'home_win_prob': win_prob,
        'away_win_prob': 1 - win_prob,
        'margin': margin,
        'home_stats': home_stats,
        'away_stats': away_stats,
    }, None


def show_rankings(builder, teams):
    """Show team rankings based on Elo."""
    now = datetime.now()

    rankings = []
    for team in teams.values():
        stats = builder._compute_team_stats(team.id, now)
        if stats is not None:
            rankings.append({
                'team': team.name,
                'elo': stats.elo,
                'win_rate': stats.win_rate,
                'margin_avg': stats.margin_avg,
                'streak': stats.streak,
            })

    rankings.sort(key=lambda x: x['elo'], reverse=True)

    print("\nTeam Rankings (by Elo)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Team':<20} {'Elo':<8} {'Win%':<8} {'Margin':<10} {'Streak':<8}")
    print("-" * 70)

    for i, r in enumerate(rankings, 1):
        streak_str = f"+{r['streak']}" if r['streak'] > 0 else str(r['streak'])
        print(f"{i:<5} {r['team']:<20} {r['elo']:<8.0f} {r['win_rate']:<8.1%} {r['margin_avg']:<+10.1f} {streak_str:<8}")


def main():
    parser = argparse.ArgumentParser(description='Predict rugby match outcomes')
    parser.add_argument('home_team', nargs='?', help='Home team name')
    parser.add_argument('away_team', nargs='?', help='Away team name')
    parser.add_argument('--rankings', action='store_true', help='Show team rankings')
    parser.add_argument('--list-teams', action='store_true', help='List all teams')
    args = parser.parse_args()

    # Load data
    db = Database()
    matches = db.get_matches()
    teams = db.get_teams()
    db.close()

    # Build feature history
    builder = FeatureBuilder(matches, teams)
    for match in sorted(matches, key=lambda m: m.date):
        builder.build_features(match)
        builder.process_match(match)

    if args.list_teams:
        print("Available teams:")
        for team in sorted(teams.values(), key=lambda t: t.name):
            country = f" ({team.country})" if team.country else ""
            print(f"  {team.name}{country}")
        return

    if args.rankings:
        show_rankings(builder, teams)
        return

    if not args.home_team or not args.away_team:
        parser.print_help()
        return

    # Find teams
    home_matches = find_team(teams, args.home_team)
    away_matches = find_team(teams, args.away_team)

    if len(home_matches) == 0:
        print(f"No team found matching '{args.home_team}'")
        return
    if len(home_matches) > 1:
        print(f"Multiple teams match '{args.home_team}':")
        for t in home_matches:
            print(f"  {t.name}")
        return

    if len(away_matches) == 0:
        print(f"No team found matching '{args.away_team}'")
        return
    if len(away_matches) > 1:
        print(f"Multiple teams match '{args.away_team}':")
        for t in away_matches:
            print(f"  {t.name}")
        return

    home_team = home_matches[0]
    away_team = away_matches[0]

    # Load model
    model, normalizer = load_model()

    # Predict
    result, error = predict_match(model, normalizer, builder, home_team, away_team)

    if error:
        print(error)
        return

    # Display results
    home = result['home_team']
    away = result['away_team']
    home_prob = result['home_win_prob']
    away_prob = result['away_win_prob']
    margin = result['margin']
    home_stats = result['home_stats']
    away_stats = result['away_stats']

    print(f"\n{'=' * 50}")
    print(f"  {home} vs {away}")
    print(f"{'=' * 50}")

    print(f"\nTeam Stats:")
    print(f"  {home}:")
    print(f"    Elo: {home_stats.elo:.0f}")
    print(f"    Win rate: {home_stats.win_rate:.1%}")
    print(f"    Avg margin: {home_stats.margin_avg:+.1f}")
    print(f"    Form: {home_stats.streak:+d}")

    print(f"  {away}:")
    print(f"    Elo: {away_stats.elo:.0f}")
    print(f"    Win rate: {away_stats.win_rate:.1%}")
    print(f"    Avg margin: {away_stats.margin_avg:+.1f}")
    print(f"    Form: {away_stats.streak:+d}")

    print(f"\nPrediction:")
    print(f"  {home}: {home_prob:.1%}")
    print(f"  {away}: {away_prob:.1%}")

    winner = home if home_prob >= 0.5 else away
    print(f"\n  {winner} to win by {margin:.0f} points")


if __name__ == "__main__":
    main()
