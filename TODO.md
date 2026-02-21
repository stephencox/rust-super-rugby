# pyRugby TODO

## MLP Model Recommendations

### High Priority
- [x] **Home/Away Symmetry (Data Augmentation):** MLPDataset with 50% random swap in training.py
- [x] **Learning Rate Scheduler:** OneCycleLR in both train_win_model and train_margin_model
- [x] **Team Embeddings:** nn.Embedding in WinClassifier and MarginRegressor (num_teams+1, dim=8)

### Medium Priority
- [x] **Label Smoothing:** label_smoothing=0.05 in train_win_model, smooths targets toward 0.5
- [x] **H2H Features:** h2h_win_rate and h2h_margin_avg added to MatchFeatures (24-dim)
- [x] **Residual Connections:** ResidualBlock with linear projection skip connections in WinClassifier/MarginRegressor

### Future Enhancements
- [ ] **Uncertainty Estimation (Quantile Regression):** Transition the margin regressor to predict 10th, 50th, and 90th percentiles to provide a confidence interval for score predictions.
- [ ] **Gated Linear Units (GLU):** Experiment with gating mechanisms to allow the model to dynamically weight input features based on the matchup context.
- [ ] **Probability Calibration:** Implement Platt scaling or Isotonic regression to ensure predicted win probabilities are well-calibrated.

## Sequence Models (LSTM/Transformer)
- [ ] **Bidirectional LSTMs:** Update `SequenceLSTM` to be bidirectional to capture patterns more effectively.
- [ ] **Transformer Architecture:** Replace or augment the LSTM backbone with a Transformer/Self-Attention encoder to capture long-range dependencies in a team's sequence history.
- [ ] **Recurrent Dropout & Masking:** Improve variable-length sequence handling and prevent overfitting with proper RNN masking and sequence dropout techniques.

## Feature Engineering Improvements

### Data Acquisition
- [ ] **Scraper Enhancement (Tries):** Update Wikipedia and Six Nations scrapers to extract try counts (`home_tries`, `away_tries`) for more granular offensive metrics.
- [ ] **Bye Week Logic:** Explicitly handle bye weeks in the `is_after_bye` binary feature (e.g., rest days >= 13).
- [ ] **Lassen Scraper Implementation:** Flesh out `lassen.py` to fetch match round numbers and local times from lassen.co.nz.
- [ ] **SA Rugby Scraper Implementation:** Flesh out `sarugby.py` to get detailed match data from sarugby.co.za.

### Feature Expansion
- [ ] **Travel & Timezone Fatigue:** Populate `travel_hours` in `SequenceFeatureBuilder` and `MatchFeatures` using the `timezone_offset` from the `Team` table.
- [ ] **Outcome-based H2H Metrics:** Add `h2h_win_rate` and `h2h_margin_avg` to both feature builders to capture specific tactical mismatches.
- [ ] **Venue-Specific Advantage:** Track and incorporate the home team's historical performance at specific `venue` locations (e.g., Eden Park factor) rather than just a binary `is_home`.
- [ ] **Strength of Schedule (SoS) Adjustment:** Implement Elo-adjusted margins to reward performance against top-tier teams and penalize "flat-track bullies."
- [ ] **Consistency Metric for MLP:** Add `margin_std` (consistency) to `MatchFeatures` to help the MLP weigh the reliability of recent form.
- [ ] **Weather Conditions:** Scrape historical weather data for match locations (rain, wind impact points scored and margin).
- [ ] **Player/Roster Data:** Track key absences (e.g., missing top performant fly-half/goal kicker) or injuries to account for dramatic squad strength changes.

## General Project Improvements
- [ ] **Automated Testing:** Implement a test suite using `pytest` for `features.py`, `data.py`, and `scrapers.py`.
- [ ] **Data Leakage Validation:** Explicitly test to ensure no future matches leak into historical feature windows.
- [ ] **Dependency Fix:** Add `tomli` to `requirements.txt` for compatibility with Python < 3.11.
- [ ] **Unified Model Checkpoints:** Consolidate model weights, normalization constants, and training metadata into a single checkpoint file or directory.
- [ ] **Experiment Tracking Integration:** Integrate an experiment logger (like `Weights & Biases`, `MLflow`, or `TensorBoard`) to visualize train/val loss curves, monitor hyperparameter sweeps, and compare different model iterations intuitively.
