# Signed Margin + Venue Overhaul

**Date:** 2026-03-06
**Goal:** Improve margin prediction accuracy and home advantage modeling for SuperBru tipping.

## Problem

Analysis of 24 predictions across 3 Six Nations rounds and 3 Super Rugby trial rounds revealed:

1. **Conservative margins** — blowouts consistently under-predicted (e.g., predicted France by 5, actual by 22; predicted Brumbies by 8, actual by 32). Only 3/24 predictions earned Margin Points (within 5 pts). Total TDM (margin error) was 380 across both competitions.

2. **Home advantage undervalued** — 3 of 9 Six Nations winner misses were home underdogs winning (Italy at Stadio Olimpico, Scotland at Murrayfield). Fijian Drua beat Hurricanes at home in Fiji despite 78% model confidence the other way. Pacific Island venues default to neutral (0.5) due to limited match history.

## SuperBru Scoring Context

- **WP (1 pt):** Correct winner — highest priority
- **MP (0.5 pt):** Margin within 5 pts of actual — currently almost never earned
- **BP (1 pt):** Correct winner + closest margin in pool + within 15 pts
- **GSP (2 pts):** All winners correct in a round
- **TDM:** Total margin error — tiebreaker (lower = better)

Improving margin accuracy to ±5 pts on 8-12 matches would add 4-6 pts per tournament.

## Design

### 1. Signed Margin

**Change:** Margin target from `abs(home_score - away_score)` to `home_score - away_score`.

- Positive = home win, negative = away win
- Model learns directional strength: France at home → large positive, Wales at home vs strong team → negative
- Currently, "France wins by 22" and "Ireland loses by 22" produce identical absolute targets (22), losing the signal that home team strength + venue → larger margins
- Z-score normalization still applies (now centered around ~+2 instead of ~+14)
- Win model stays independent — determines winner pick, margin model determines magnitude

**Files:** `training.py` (target computation), `cli.py` (display uses `abs(margin)`)

### 2. Augmentation for Signed Margin

**Change:** Add `negate_label` mode to `MLPDataset`.

- Current: `flip_label=True` → `y = 1.0 - y` (win model), `flip_label=False` → unchanged (margin)
- New: `negate_label=True` → `y = -y` (signed margin flips when home/away swap)
- Margin training uses `negate_label=True` instead of `flip_label=False`

**Files:** `training.py` (MLPDataset.__getitem__, train_margin_model call)

### 3. Venue Feature Overhaul

**Replace** `home_venue_win_rate` (index 31) with richer features:

| Index | Feature | Description | Default |
|-------|---------|-------------|---------|
| 31 | `home_venue_margin_avg` | Avg signed margin at this venue for home team (2+ matches required) | 0.0 |
| 32 | `home_country_win_rate` | Home team's overall home win rate in their country | 0.5 |
| 33 | `away_country_win_rate` | Away team's overall home win rate in their country (for augmentation swap) | 0.5 |

- `home_venue_margin_avg` captures HOW MUCH teams win by at home, not just whether they win
- `home_country_win_rate` pools all home games in a country — solves the Pacific Island data sparsity problem (uses all FD home games in Fiji, not just 2+ at Churchill Park)
- `away_country_win_rate` enables proper augmentation swap with index 32

**Feature count:** 32 → 34

**Augmentation:** Swap (32, 33). Reset index 31 to 0.0 on swap.

**Files:** `features.py` (MatchFeatures, _compute_venue_stats, new _compute_country_win_rate), `training.py` (augmentation index lists)

### 4. Validation

1. Train old-style and new-style models on pre-2026 data
2. Predict 24 known 2026 matches (15 SR + 9 6N)
3. Compare winner accuracy (WP), MP rate (within 5 pts), and TDM
4. Target: MP from 3/24 → 8+, TDM from 380 → <250
5. Sanity check predictions for current round
6. Rollback if winner accuracy drops — WP matters more than MP
