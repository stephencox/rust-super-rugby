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
- [ ] **Uncertainty Estimation (Quantile Regression):** Transition the margin regressor to predict 10th, 50th, and 90th percentiles to provide a confidence interval for score predictions.
- [ ] **Gated Linear Units (GLU):** Experiment with gating mechanisms to allow the model to dynamically weight input features based on the matchup context.
- [x] **Probability Calibration:** Platt scaling (LBFGS on validation logits) stored in checkpoint, applied at inference

## Sequence Models (LSTM/Transformer)
- [x] **Bidirectional LSTMs:** `SequenceLSTM` with `bidirectional=True`, temporal attention over BiLSTM outputs
- [ ] **Transformer Architecture:** Replace or augment the LSTM backbone with a Transformer/Self-Attention encoder to capture long-range dependencies in a team's sequence history.
- [ ] **Recurrent Dropout & Masking:** Improve variable-length sequence handling and prevent overfitting with proper RNN masking and sequence dropout techniques.

## Feature Engineering Improvements

### Data Acquisition
- [ ] **Scraper Enhancement (Tries):** Update Wikipedia and Six Nations scrapers to extract try counts (`home_tries`, `away_tries`) for more granular offensive metrics.
- [x] **Bye Week Logic:** `home_is_after_bye` / `away_is_after_bye` binary features (rest >= 13 days) in MatchFeatures
- [ ] **Lassen Scraper Implementation:** Flesh out `lassen.py` to fetch match round numbers and local times from lassen.co.nz.
- [ ] **SA Rugby Scraper Implementation:** Flesh out `sarugby.py` to get detailed match data from sarugby.co.za.

### Feature Expansion
- [x] **Travel & Timezone Fatigue:** `travel_hours` in MatchFeatures using `timezone_offset` from Team table
- [x] **Outcome-based H2H Metrics:** `h2h_win_rate` and `h2h_margin_avg` in MatchFeatures (FeatureBuilder)
- [ ] **Venue-Specific Advantage:** Track and incorporate the home team's historical performance at specific `venue` locations (e.g., Eden Park factor) rather than just a binary `is_home`.
- [x] **Strength of Schedule (SoS) Adjustment:** `home_sos` / `away_sos` (avg opponent Elo) in MatchFeatures
- [x] **Consistency Metric for MLP:** `home_consistency` / `away_consistency` (inverse margin stdev) in MatchFeatures
- [ ] **Weather Conditions:** Scrape historical weather data for match locations (rain, wind impact points scored and margin).
- [ ] **Player/Roster Data:** Track key absences (e.g., missing top performant fly-half/goal kicker) or injuries to account for dramatic squad strength changes.

## General Project Improvements
- [x] **Automated Testing:** pytest suite with 56 tests covering `data.py`, `features.py`, and `training.py`
- [x] **Data Leakage Validation:** Dedicated `test_data_leakage.py` validates no future data leaks into feature windows
- [x] **Unified Model Checkpoints:** Single `_mlp.pt` and `_lstm.pt` files contain model weights, normalizer, and metadata
- [ ] **Experiment Tracking Integration:** Integrate an experiment logger (like `Weights & Biases`, `MLflow`, or `TensorBoard`) to visualize train/val loss curves, monitor hyperparameter sweeps, and compare different model iterations intuitively.
