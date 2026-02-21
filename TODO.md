# pyRugby TODO

## MLP Model Recommendations

### High Priority
- [x] **Home/Away Symmetry (Data Augmentation):** MLPDataset with 50% random swap in training.py
- [x] **Learning Rate Scheduler:** OneCycleLR in both train_win_model and train_margin_model
- [x] **Team Embeddings:** nn.Embedding in WinClassifier and MarginRegressor (num_teams+1, dim=8)

### Medium Priority
- [x] **Label Smoothing:** label_smoothing=0.05 in train_win_model, smooths targets toward 0.5
- [x] **H2H Features:** h2h_win_rate and h2h_margin_avg added to MatchFeatures
- [x] **Residual Connections:** ResidualBlock with linear projection skip connections in WinClassifier/MarginRegressor

### Future Enhancements
- [x] **Uncertainty Estimation (Quantile Regression):** `MarginRegressor` outputs 3 quantiles (p10/p50/p90) with pinball loss, monotonicity enforced via softplus, 80% CI shown in predictions
- [x] **Gated Linear Units (GLU):** `ResidualBlock` uses `nn.functional.glu` by default — splits linear output into content + sigmoid gate for dynamic feature weighting
- [x] **Probability Calibration:** Platt scaling (LBFGS on validation logits) stored in checkpoint, applied at inference

## Sequence Models (LSTM/Transformer)
- [x] **Bidirectional LSTMs:** `SequenceLSTM` with `bidirectional=True`, temporal attention over BiLSTM outputs
- [ ] **Transformer Architecture:** Replace or augment the LSTM backbone with a Transformer/Self-Attention encoder to capture long-range dependencies in a team's sequence history.
- [x] **Recurrent Dropout & Masking:** `pack_padded_sequence` for variable-length LSTM, masked attention (padding → -inf), sequence lengths tracked in `SequenceDataset`

## Feature Engineering Improvements

### Data Acquisition
- [x] **Scraper Enhancement (Tries):** `_count_tries_in_cell()` extracts try counts from HTML detail cells in both SportsEvent and collapsible table formats (72% SR, 98% 6N coverage)
- [x] **Bye Week Logic:** `home_is_after_bye` / `away_is_after_bye` binary features (rest >= 13 days) in MatchFeatures
- [ ] **Lassen Scraper Implementation:** Flesh out `lassen.py` to fetch match round numbers and local times from lassen.co.nz.
- [ ] **SA Rugby Scraper Implementation:** Flesh out `sarugby.py` to get detailed match data from sarugby.co.za.

### Feature Expansion
- [x] **Travel & Timezone Fatigue:** `travel_hours` in MatchFeatures using `timezone_offset` from Team table
- [x] **Outcome-based H2H Metrics:** `h2h_win_rate` and `h2h_margin_avg` in MatchFeatures (FeatureBuilder)
- [x] **Venue-Specific Advantage:** `home_venue_win_rate` in MatchFeatures — home team's win rate at this venue (min 2 games, else 0.5)
- [x] **Strength of Schedule (SoS) Adjustment:** `home_sos` / `away_sos` (avg opponent Elo) in MatchFeatures
- [x] **Consistency Metric for MLP:** `home_consistency` / `away_consistency` (inverse margin stdev) in MatchFeatures
- [ ] **Weather Conditions:** Scrape historical weather data for match locations (rain, wind impact points scored and margin).
- [ ] **Player/Roster Data:** Track key absences (e.g., missing top performant fly-half/goal kicker) or injuries to account for dramatic squad strength changes.

## General Project Improvements
- [x] **Automated Testing:** pytest suite with 56 tests covering `data.py`, `features.py`, and `training.py`
- [x] **Data Leakage Validation:** Dedicated `test_data_leakage.py` validates no future data leaks into feature windows
- [x] **Unified Model Checkpoints:** Single `_mlp.pt` and `_lstm.pt` files contain model weights, normalizer, and metadata
- [x] **Experiment Tracking Integration:** TensorBoard logging via `--tensorboard DIR` flag on `train` and `train-lstm` commands — logs loss, accuracy, MAE, and LR per epoch
