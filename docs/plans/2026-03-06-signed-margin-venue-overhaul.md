# Signed Margin + Venue Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch margin prediction from absolute to signed (home - away) and add country-level home advantage features to improve SuperBru margin accuracy.

**Architecture:** Replace absolute margin targets with signed margin throughout the MLP pipeline. Add `negate_label` augmentation mode for signed margins. Replace `home_venue_win_rate` with `home_venue_margin_avg` and add country-level home win rate features (32 → 34 features). Fix venue passthrough in predictions.

**Tech Stack:** Python, PyTorch, NumPy, SQLite

---

### Task 1: Add venue features to MatchFeatures dataclass

**Files:**
- Modify: `rugby/features.py:35-158` (MatchFeatures dataclass)

**Step 1: Update the dataclass fields**

Replace the venue field and add new fields at the end of the dataclass (before `to_array`):

```python
    # Venue advantage (1) — home team's avg signed margin at this venue
    home_venue_margin_avg: float = 0.0

    # Country-level home advantage (2)
    home_country_win_rate: float = 0.5
    away_country_win_rate: float = 0.5
```

**Step 2: Update `to_array()` (lines 91-137)**

Replace the last entry and add two more:

```python
            # Venue (1) + Country (2)
            self.home_venue_margin_avg,
            self.home_country_win_rate,
            self.away_country_win_rate,
```

**Step 3: Update `feature_names()` (lines 139-154)**

Replace `'home_venue_win_rate'` with:
```python
            'home_venue_margin_avg',
            'home_country_win_rate', 'away_country_win_rate',
```

**Step 4: Update `num_features()` (line 158)**

Change `return 32` to `return 34`.

**Step 5: Run sanity check**

```bash
.venv/bin/python -c "from rugby.features import MatchFeatures; f = MatchFeatures(); assert len(f.to_array()) == 34; assert len(f.feature_names()) == 34; assert f.num_features() == 34; print('OK: 34 features')"
```

**Step 6: Commit**

```bash
git add rugby/features.py
git commit -m "feat: replace venue win rate with margin avg + country features (32→34 dim)"
```

---

### Task 2: Add venue and country feature computation methods

**Files:**
- Modify: `rugby/features.py:326-345` (_compute_venue_win_rate → _compute_venue_margin_avg)
- Modify: `rugby/features.py:347-423` (build_features)

**Step 1: Replace `_compute_venue_win_rate` with `_compute_venue_margin_avg`**

Replace the method at lines 326-345:

```python
    def _compute_venue_margin_avg(self, team_id: int, venue: Optional[str],
                                  before_date: datetime) -> float:
        """Compute a team's average signed margin at a specific venue (home games only)."""
        if not venue:
            return 0.0
        venue_lower = venue.lower().strip()
        total_margin = 0
        total = 0
        for m in self.team_history[team_id]:
            if m.date >= before_date:
                continue
            if m.home_team_id != team_id:
                continue  # Only home games at this venue
            if m.venue and m.venue.lower().strip() == venue_lower:
                total += 1
                total_margin += m.home_score - m.away_score
        if total < 2:
            return 0.0  # Not enough venue history
        return total_margin / total
```

**Step 2: Add `_compute_country_win_rate` method**

Add after the venue method:

```python
    def _compute_country_win_rate(self, team_id: int, before_date: datetime) -> float:
        """Compute a team's home win rate in their home country."""
        team = self.teams.get(team_id)
        if not team or not team.country:
            return 0.5
        wins = 0
        total = 0
        for m in self.team_history[team_id]:
            if m.date >= before_date:
                continue
            if m.home_team_id != team_id:
                continue  # Only home games
            # Check if venue is in team's country (home team plays in home country)
            total += 1
            if m.home_win:
                wins += 1
        if total < 3:
            return 0.5  # Not enough home history
        return wins / total
```

**Step 3: Update `build_features()` to use new methods**

In build_features (around line 374-422), replace the venue computation and add country:

Replace:
```python
        # Venue-specific home advantage
        home_venue_wr = self._compute_venue_win_rate(
            match.home_team_id, match.venue, match.date
        )
```

With:
```python
        # Venue-specific home advantage (signed margin avg)
        home_venue_margin = self._compute_venue_margin_avg(
            match.home_team_id, match.venue, match.date
        )

        # Country-level home advantage
        home_country_wr = self._compute_country_win_rate(match.home_team_id, match.date)
        away_country_wr = self._compute_country_win_rate(match.away_team_id, match.date)
```

And in the MatchFeatures constructor return, replace:
```python
            # Venue
            home_venue_win_rate=home_venue_wr,
```

With:
```python
            # Venue + Country
            home_venue_margin_avg=home_venue_margin,
            home_country_win_rate=home_country_wr,
            away_country_win_rate=away_country_wr,
```

**Step 4: Run sanity check**

```bash
.venv/bin/python -c "
from rugby.data import Database
from rugby.features import FeatureBuilder
from pathlib import Path
db = Database(Path('data/rugby.db'))
matches = db.get_matches()
teams = db.get_teams()
db.close()
builder = FeatureBuilder(matches, teams)
m = sorted(matches, key=lambda m: m.date)
# Process first 50 matches to build history
for match in m[:50]:
    builder.build_features(match)
    builder.process_match(match)
# Build features for match 51
f = builder.build_features(m[50])
if f:
    arr = f.to_array()
    print(f'Features shape: {arr.shape}')
    print(f'venue_margin_avg (idx 31): {arr[31]:.3f}')
    print(f'home_country_wr (idx 32): {arr[32]:.3f}')
    print(f'away_country_wr (idx 33): {arr[33]:.3f}')
    assert arr.shape == (34,)
    print('OK')
"
```

**Step 5: Commit**

```bash
git add rugby/features.py
git commit -m "feat: add venue margin avg and country win rate feature computation"
```

---

### Task 3: Update augmentation indices for 34-dim features

**Files:**
- Modify: `rugby/training.py:15-32` (augmentation constants)

**Step 1: Update the augmentation constants**

Replace lines 15-32:

```python
# Feature indices for home/away augmentation (34-dim MatchFeatures)
# Indices 0-3: differentials (negate)
# Index 4: is_local (symmetric, keep)
# Indices 5-9: home stats, 10-14: away stats (swap)
# Indices 15-16: home_elo/away_elo (swap), 17: elo_diff (negate)
# Indices 18-19: home form, 20-21: away form (swap)
# Index 22: h2h_win_rate (flip: 1-x), 23: h2h_margin_avg (negate)
# Index 24: travel_hours (symmetric, keep)
# Indices 25-26: home/away consistency (swap)
# Indices 27-28: home/away is_after_bye (swap)
# Indices 29-30: home/away sos (swap)
# Index 31: home_venue_margin_avg (reset to 0.0 on swap)
# Indices 32-33: home/away country_win_rate (swap)
_NEGATE_INDICES = [0, 1, 2, 3, 17, 23]
_FLIP_INDICES = [22]  # x -> 1 - x
_SWAP_PAIRS = [(5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
               (15, 16), (18, 20), (19, 21),
               (25, 26), (27, 28), (29, 30),
               (32, 33)]
_RESET_INDICES = {31: 0.0}  # Reset to neutral on swap
```

**Step 2: Verify**

```bash
.venv/bin/python -c "from rugby.training import _SWAP_PAIRS, _RESET_INDICES; print('SWAP_PAIRS:', _SWAP_PAIRS); print('RESET:', _RESET_INDICES); assert (32, 33) in _SWAP_PAIRS; assert _RESET_INDICES[31] == 0.0; print('OK')"
```

**Step 3: Commit**

```bash
git add rugby/training.py
git commit -m "feat: update augmentation indices for 34-dim features"
```

---

### Task 4: Add negate_label mode to MLPDataset

**Files:**
- Modify: `rugby/training.py:53-104` (MLPDataset class)

**Step 1: Add `negate_label` parameter**

Update `__init__` to accept `negate_label`:

```python
class MLPDataset(Dataset):
    """Dataset for MLP models with optional home/away augmentation."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        home_team_ids: Optional[np.ndarray] = None,
        away_team_ids: Optional[np.ndarray] = None,
        augment: bool = False,
        flip_label: bool = True,
        negate_label: bool = False,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.home_team_ids = torch.tensor(home_team_ids, dtype=torch.long) if home_team_ids is not None else None
        self.away_team_ids = torch.tensor(away_team_ids, dtype=torch.long) if away_team_ids is not None else None
        self.augment = augment
        self.flip_label = flip_label
        self.negate_label = negate_label
```

**Step 2: Update `__getitem__` label handling**

Replace the label flip block (lines 94-96):

```python
            # Flip win label (0↔1), but keep margin unchanged (absolute)
            if self.flip_label:
                y = 1.0 - y
```

With:

```python
            # Flip win label (0↔1) or negate signed margin
            if self.flip_label:
                y = 1.0 - y
            elif self.negate_label:
                y = -y
```

**Step 3: Verify**

```bash
.venv/bin/python -c "
import numpy as np
from rugby.training import MLPDataset
X = np.random.randn(10, 34).astype(np.float32)
y = np.array([5.0, -3.0, 10.0, -7.0, 0.0, 15.0, -20.0, 8.0, -1.0, 12.0], dtype=np.float32)
ds = MLPDataset(X, y, augment=True, flip_label=False, negate_label=True)
print('Dataset created OK, len:', len(ds))
# Check a few items - augmentation is random so just verify it runs
for i in range(5):
    item = ds[0]
    print(f'  label={item[\"label\"].item():.1f}')
print('OK')
"
```

**Step 4: Commit**

```bash
git add rugby/training.py
git commit -m "feat: add negate_label mode to MLPDataset for signed margin augmentation"
```

---

### Task 5: Switch margin training to signed targets

**Files:**
- Modify: `rugby/training.py:308-311` (train_margin_model dataset creation)
- Modify: `rugby/training.py:520` (SequenceDataset margin)
- Modify: `cli.py:378` (margin target in cmd_train)

**Step 1: Update margin target in cli.py cmd_train**

In `cli.py` around line 378, change:

```python
        margin = abs(match.home_score - match.away_score)
```

To:

```python
        margin = match.home_score - match.away_score
```

**Step 2: Update train_margin_model to use negate_label**

In `training.py` around lines 308-311, change:

```python
    # Absolute margin is symmetric — don't flip label on home/away swap
    train_dataset = MLPDataset(X_train, y_margin_train_scaled, home_team_ids, away_team_ids,
                               augment=augment_swap, flip_label=False)
```

To:

```python
    # Signed margin: negate when swapping home/away
    train_dataset = MLPDataset(X_train, y_margin_train_scaled, home_team_ids, away_team_ids,
                               augment=augment_swap, flip_label=False, negate_label=True)
```

**Step 3: Update MarginRegressor docstring**

In `models.py` line 127, change:

```python
    MLP regressor for predicting match margin (absolute points difference).
```

To:

```python
    MLP regressor for predicting match margin (signed: home_score - away_score).
```

**Step 4: Update SequenceDataset margin (LSTM consistency)**

In `training.py` line 520, change:

```python
        margin = abs(s.home_score - s.away_score)
```

To:

```python
        margin = s.home_score - s.away_score
```

And in the augmented branch (line 530), add negation:

```python
                'margin': torch.tensor(-margin, dtype=torch.float32),
```

(Currently line 530 is `'margin': torch.tensor(margin, ...)` — the augmented path swaps home/away, so signed margin must negate.)

**Step 5: Remove ReLU from LSTM margin head**

In `models.py` line 365, change:

```python
        margin = torch.relu(self.margin_head(margin_h))
```

To:

```python
        margin = self.margin_head(margin_h)
```

(ReLU forced non-negative output, which is wrong for signed margins.)

**Step 6: Verify training runs**

```bash
.venv/bin/python -c "
from rugby.data import Database
from rugby.features import FeatureBuilder, FeatureNormalizer
from rugby.training import train_margin_model
from pathlib import Path
import numpy as np

db = Database(Path('data/rugby.db'))
matches = db.get_matches()
teams = db.get_teams()
db.close()

builder = FeatureBuilder(matches, teams)
X_list, y_list = [], []
for match in sorted(matches, key=lambda m: m.date):
    f = builder.build_features(match)
    builder.process_match(match)
    if f is None:
        continue
    X_list.append(f.to_array())
    y_list.append(float(match.home_score - match.away_score))  # signed!

X = np.array(X_list[:100])
y = np.array(y_list[:100])
norm = FeatureNormalizer()
X_norm = norm.fit_transform(X)

model, hist = train_margin_model(X_norm, y, epochs=5, batch_size=16, augment_swap=True, verbose=True)
print(f'margin_mean={hist[\"margin_mean\"]:.1f}, margin_std={hist[\"margin_std\"]:.1f}')
print('OK: signed margin training works')
"
```

**Step 7: Commit**

```bash
git add rugby/training.py rugby/models.py cli.py
git commit -m "feat: switch margin targets from absolute to signed (home - away)"
```

---

### Task 6: Fix venue passthrough in predictions

**Files:**
- Modify: `cli.py:839-844` (cmd_predict synthetic match)
- Modify: `cli.py:964-969` (cmd_predict_next synthetic match)

**Step 1: Pass fixture venue in predict-next**

In `cli.py` cmd_predict_next, the predict_fn closure (around line 962-968) creates a synthetic match with `venue=None`. Update to accept and pass venue:

Change the predict_fn signature and body. Find:

```python
        def predict_fn(home_team, away_team):
            now = datetime.now()
            synthetic = Match(
                id=-1, date=now,
                home_team_id=home_team.id, away_team_id=away_team.id,
                home_score=0, away_score=0,
                venue=None, round=None,
            )
```

Replace with:

```python
        def predict_fn(home_team, away_team, venue=None):
            now = datetime.now()
            synthetic = Match(
                id=-1, date=now,
                home_team_id=home_team.id, away_team_id=away_team.id,
                home_score=0, away_score=0,
                venue=venue, round=None,
            )
```

Then update the call site (around line 1011):

Change:
```python
        pred = predict_fn(home_team, away_team)
```

To:
```python
        pred = predict_fn(home_team, away_team, venue=fixture.venue)
```

**Step 2: Do the same for the LSTM predict_fn branch**

In the LSTM branch of predict-next (around line 930), add venue parameter similarly. The LSTM predict_fn doesn't use MatchFeatures directly, so this is just for future consistency — but check if the sequence builder uses venue.

Actually, the LSTM branch uses SequenceFeatureBuilder which doesn't use venue features. Leave the LSTM branch as-is for now.

**Step 3: Verify**

```bash
.venv/bin/python cli.py predict-next --year 2026 2>&1 | head -20
```

Should show predictions (venue features now active for predict-next).

**Step 4: Commit**

```bash
git add cli.py
git commit -m "fix: pass fixture venue to predictions for venue feature activation"
```

---

### Task 7: Update CLAUDE.md feature dimensions

**Files:**
- Modify: `/home/stephen/code/rugby/CLAUDE.md`

**Step 1: Update feature dimension references**

Change `31 static features (MLP)` to `34 static features (MLP)` and update the feature dimension line.

**Step 2: Commit**

```bash
git add /home/stephen/code/rugby/CLAUDE.md
git commit -m "docs: update feature dimensions to 34 in CLAUDE.md"
```

---

### Task 8: Validation — compare old vs new on 2026 matches

**Files:**
- No code changes — run predictions using rolled-back DBs

**Step 1: Train new models on pre-2026 data (non-production split)**

```bash
.venv/bin/python cli.py train 2>&1
.venv/bin/python cli.py --competition sixnations train 2>&1
```

**Step 2: Run predictions against known 2026 results**

Use the temporary rolled-back DBs from earlier analysis (or recreate them) to predict each round of 2026 matches. Compare:

- Winner accuracy (WP) — must not drop
- Margins within 5 pts (MP) — target improvement
- Total TDM — target reduction

**Step 3: Train production models if validation passes**

```bash
.venv/bin/python cli.py train --production 2>&1
.venv/bin/python cli.py --competition sixnations train --production 2>&1
.venv/bin/python cli.py train-lstm --production 2>&1
.venv/bin/python cli.py --competition sixnations train-lstm --production 2>&1
```

**Step 4: Generate predictions for current round**

```bash
.venv/bin/python cli.py predict-next --year 2026
.venv/bin/python cli.py --competition sixnations predict-next --year 2026
```

**Step 5: Commit production models**

No commit needed — model files are gitignored.
