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
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup

from rugby.data import Database, Team
from rugby.features import (
    FeatureBuilder,
    FeatureNormalizer,
    MatchFeatures,
    SequenceFeatureBuilder,
    SequenceNormalizer,
)
from rugby.models import MatchPredictor, SequenceLSTM


MODEL_DIR = Path(__file__).parent.parent / "model"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

# Team aliases: map Wikipedia names to database names
TEAM_ALIASES = {
    "blues": "Blues",
    "auckland blues": "Blues",
    "chiefs": "Chiefs",
    "waikato chiefs": "Chiefs",
    "crusaders": "Crusaders",
    "canterbury crusaders": "Crusaders",
    "highlanders": "Highlanders",
    "otago highlanders": "Highlanders",
    "hurricanes": "Hurricanes",
    "wellington hurricanes": "Hurricanes",
    "moana pasifika": "Moana Pasifika",
    "brumbies": "Brumbies",
    "act brumbies": "Brumbies",
    "reds": "Reds",
    "queensland reds": "Reds",
    "waratahs": "Waratahs",
    "nsw waratahs": "Waratahs",
    "new south wales waratahs": "Waratahs",
    "force": "Force",
    "western force": "Force",
    "rebels": "Rebels",
    "melbourne rebels": "Rebels",
    "bulls": "Bulls",
    "blue bulls": "Bulls",
    "vodacom bulls": "Bulls",
    "lions": "Lions",
    "golden lions": "Lions",
    "emirates lions": "Lions",
    "sharks": "Sharks",
    "natal sharks": "Sharks",
    "cell c sharks": "Sharks",
    "the sharks": "Sharks",
    "stormers": "Stormers",
    "dhl stormers": "Stormers",
    "western province stormers": "Stormers",
    "cheetahs": "Cheetahs",
    "free state cheetahs": "Cheetahs",
    "kings": "Kings",
    "southern kings": "Kings",
    "sunwolves": "Sunwolves",
    "jaguares": "Jaguares",
    "fijian drua": "Fijian Drua",
    "drua": "Fijian Drua",
}


@dataclass
class Fixture:
    date: date
    home_team: str
    away_team: str
    venue: Optional[str]
    round: Optional[int]


def normalize_team_name(name: str) -> Optional[str]:
    """Map a Wikipedia team name to a canonical name."""
    cleaned = re.sub(r"[\[\]*†]", "", name).strip().lower()
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]
    # Partial match
    for alias, canonical in TEAM_ALIASES.items():
        if cleaned in alias or alias in cleaned:
            return canonical
    return None


def parse_date(text: str) -> Optional[date]:
    """Extract a date from text."""
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    # "13 February 2026"
    m = re.search(
        r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return date(year, months[month_str], day)
    return None


def fetch_fixtures(year: int) -> List[Fixture]:
    """Fetch upcoming fixtures from Wikipedia, using cache when available."""
    url = f"https://en.wikipedia.org/wiki/List_of_{year}_Super_Rugby_Pacific_matches"

    # Try cache first (shared with Rust scraper)
    cache_file = CACHE_DIR / (
        url.replace("https://", "").replace("/", "_").replace("?", "_") + ".html"
    )

    html = None
    if cache_file.exists():
        print(f"Using cached fixtures from {cache_file.name}")
        html = cache_file.read_text()
    else:
        print(f"Fetching fixtures from Wikipedia...")
        resp = requests.get(url, headers={"User-Agent": "rugby-predictor/0.1"}, timeout=30)
        resp.raise_for_status()
        html = resp.text
        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(html)

    return parse_fixtures(html)


def parse_fixtures(html: str) -> List[Fixture]:
    """Parse upcoming fixtures from Wikipedia HTML."""
    soup = BeautifulSoup(html, "html.parser")
    score_re = re.compile(r"\d{1,3}\s*[–\-]\s*\d{1,3}")
    fixtures = []

    for event in soup.find_all("div", itemtype="http://schema.org/SportsEvent"):
        event_text = event.get_text()
        match_date = parse_date(event_text)

        # Find teams (span.fn.org)
        teams = event.find_all("span", class_="fn")
        if len(teams) < 2:
            continue
        home_name = normalize_team_name(teams[0].get_text())
        away_name = normalize_team_name(teams[1].get_text())

        # Skip if match already has a score
        has_score = False
        for td in event.find_all("td"):
            if score_re.search(td.get_text()):
                has_score = True
                break
        if has_score:
            continue

        # Venue
        loc = event.find("span", class_="location")
        venue = loc.get_text().strip() if loc else None

        if match_date and home_name and away_name:
            fixtures.append(Fixture(
                date=match_date,
                home_team=home_name,
                away_team=away_name,
                venue=venue,
                round=None,
            ))

    # Deduplicate
    seen = set()
    unique = []
    for f in fixtures:
        key = (f.date, f.home_team, f.away_team)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    fixtures = sorted(unique, key=lambda f: (f.date, f.home_team))

    # Infer round numbers: group dates within 3 days
    if fixtures:
        dates = sorted(set(f.date for f in fixtures))
        date_to_round = {}
        current_round = 1
        round_start = dates[0]
        for d in dates:
            if (d - round_start).days > 3:
                current_round += 1
                round_start = d
            date_to_round[d] = current_round
        for f in fixtures:
            f.round = date_to_round[f.date]

    return fixtures


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

    # Fetch fixtures
    if args.refresh:
        # Delete cached file to force re-fetch
        url = f"https://en.wikipedia.org/wiki/List_of_{args.year}_Super_Rugby_Pacific_matches"
        cache_file = CACHE_DIR / (
            url.replace("https://", "").replace("/", "_").replace("?", "_") + ".html"
        )
        if cache_file.exists():
            cache_file.unlink()

    fixtures = fetch_fixtures(args.year)
    if not fixtures:
        print(f"No upcoming fixtures found for {args.year}.")
        return

    # Filter by round
    if args.round is not None:
        fixtures = [f for f in fixtures if f.round == args.round]
        if not fixtures:
            print(f"No fixtures found for round {args.round}.")
            return
    elif not args.all:
        # Default: next round (earliest round number)
        min_round = min(f.round for f in fixtures if f.round is not None)
        fixtures = [f for f in fixtures if f.round == min_round]

    round_num = fixtures[0].round
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
    for fixture in fixtures:
        home_team = find_db_team(teams, fixture.home_team)
        away_team = find_db_team(teams, fixture.away_team)

        if not home_team:
            print(f"  Warning: Unknown team '{fixture.home_team}'") if not args.json else None
            continue
        if not away_team:
            print(f"  Warning: Unknown team '{fixture.away_team}'") if not args.json else None
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
