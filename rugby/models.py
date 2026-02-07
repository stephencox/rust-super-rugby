"""Neural network models for rugby prediction."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class WinClassifier(nn.Module):
    """
    MLP classifier for predicting match winner.

    Outputs probability that home team wins.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64], dropout: float = 0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Win logits [batch] (apply sigmoid for probability)
        """
        h = self.backbone(x)
        return self.head(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get win probability."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        proba = self.predict_proba(x)
        return (proba >= threshold).float()

    def save(self, path: Path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))


class ScoreRegressor(nn.Module):
    """
    MLP regressor for predicting match scores.

    Takes features + win probability as input.
    Outputs (home_score, away_score) predictions.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64], dropout: float = 0.0):
        super().__init__()

        # Input is features + win_prob (1 extra dimension)
        layers = []
        prev_dim = input_dim + 1  # +1 for win probability
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.home_head = nn.Linear(prev_dim, 1)
        self.away_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor, win_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            win_prob: Win probability from classifier [batch]

        Returns:
            (home_score, away_score) predictions, each [batch]
        """
        # Concatenate features with win probability
        combined = torch.cat([x, win_prob.unsqueeze(-1)], dim=-1)
        h = self.backbone(combined)
        # Use ReLU to ensure non-negative scores
        home = torch.relu(self.home_head(h)).squeeze(-1)
        away = torch.relu(self.away_head(h)).squeeze(-1)
        return home, away

    def predict(self, x: torch.Tensor, win_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get score predictions."""
        with torch.no_grad():
            return self.forward(x, win_prob)

    def save(self, path: Path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))


class MatchPredictor(nn.Module):
    """
    End-to-end model that predicts win probability and margin.

    Architecture:
    1. Win classifier: features -> win_logit -> win_prob
    2. Margin regressor: [features, win_prob] -> margin (absolute points difference)

    The win probability informs the margin prediction.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64], dropout: float = 0.0):
        super().__init__()

        # Win classifier backbone
        win_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            win_layers.append(nn.Linear(prev_dim, h_dim))
            win_layers.append(nn.ReLU())
            if dropout > 0:
                win_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.win_backbone = nn.Sequential(*win_layers)
        self.win_head = nn.Linear(prev_dim, 1)

        # Margin regressor backbone (takes features + win_prob)
        margin_layers = []
        prev_dim = input_dim + 1  # +1 for win probability
        for h_dim in hidden_dims:
            margin_layers.append(nn.Linear(prev_dim, h_dim))
            margin_layers.append(nn.ReLU())
            if dropout > 0:
                margin_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.margin_backbone = nn.Sequential(*margin_layers)
        self.margin_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            (win_logit, margin)
            - win_logit: [batch] (apply sigmoid for probability)
            - margin: [batch] (absolute points difference, always positive)
        """
        # Win prediction
        win_h = self.win_backbone(x)
        win_logit = self.win_head(win_h).squeeze(-1)
        win_prob = torch.sigmoid(win_logit)

        # Margin prediction (using win_prob as input)
        combined = torch.cat([x, win_prob.unsqueeze(-1)], dim=-1)
        margin_h = self.margin_backbone(combined)
        margin = torch.relu(self.margin_head(margin_h)).squeeze(-1)  # Always positive

        return win_logit, margin

    def predict(self, x: torch.Tensor) -> dict:
        """Get all predictions."""
        with torch.no_grad():
            win_logit, margin = self.forward(x)
            win_prob = torch.sigmoid(win_logit)
            return {
                'win_prob': win_prob,
                'margin': margin,
            }

    def save(self, path: Path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))


class CombinedPredictor:
    """
    Combines WinClassifier and ScoreRegressor for full match prediction.
    """

    def __init__(self, win_model: WinClassifier, score_model: ScoreRegressor,
                 normalizer=None):
        self.win_model = win_model
        self.score_model = score_model
        self.normalizer = normalizer

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict match outcomes.

        Args:
            X: Features array [batch, num_features]

        Returns:
            Dictionary with predictions
        """
        # Normalize if normalizer provided
        if self.normalizer:
            X = self.normalizer.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Get predictions
        self.win_model.train(False)
        self.score_model.train(False)

        win_prob = self.win_model.predict_proba(X_tensor).numpy()
        home_score, away_score = self.score_model.predict(X_tensor)
        home_score = home_score.numpy()
        away_score = away_score.numpy()

        return {
            'win_prob': win_prob,
            'home_score': home_score,
            'away_score': away_score,
            'predicted_winner': np.where(win_prob >= 0.5, 'home', 'away'),
            'predicted_margin': np.abs(home_score - away_score),
        }


# =============================================================================
# Sequence-based models (LSTM)
# =============================================================================

# Per-match features for sequence models (23 dimensions)
MATCH_FEATURE_DIM = 23


class SequenceLSTM(nn.Module):
    """
    LSTM model for rugby match prediction using team history sequences.

    Processes home and away team history through a shared LSTM, then
    combines the representations with comparison features for prediction.

    Architecture:
    1. Process home team history through LSTM -> home_repr
    2. Process away team history through same LSTM -> away_repr
    3. Concatenate [home_repr, away_repr, comparison_features] -> win_logit -> win_prob
    4. [combined + win_prob] -> margin
    """

    def __init__(
        self,
        input_dim: int = MATCH_FEATURE_DIM,
        hidden_size: int = 64,
        num_layers: int = 1,
        comparison_dim: int = 50,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LayerNorm on sequence input (normalizes each timestep's features)
        self.input_norm = nn.LayerNorm(input_dim)

        # Shared LSTM for processing team histories
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Encourage remembering by default (Jozefowicz et al., 2015)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias

        # Temporal attention over LSTM timestep outputs
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # FC layer for win prediction
        # Combined: home_hidden + away_hidden + comparison
        fc_input_size = hidden_size * 2 + comparison_dim
        self.win_fc = nn.Linear(fc_input_size, hidden_size)
        self.win_head = nn.Linear(hidden_size, 1)

        # FC layer for margin prediction (takes combined + win_prob)
        self.margin_fc = nn.Linear(fc_input_size + 1, hidden_size)
        self.margin_head = nn.Linear(hidden_size, 1)

    def attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Temporal attention over LSTM outputs.

        Args:
            lstm_out: LSTM outputs [batch, seq_len, hidden]

        Returns:
            Context vector [batch, hidden]
        """
        energy = torch.tanh(self.attn_fc(lstm_out))       # [B, seq_len, hidden]
        scores = self.attn_score(energy).squeeze(-1)       # [B, seq_len]
        weights = torch.softmax(scores, dim=1)             # [B, seq_len]
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # [B, hidden]
        return context

    def forward(
        self,
        home_history: torch.Tensor,
        away_history: torch.Tensor,
        comparison: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            home_history: Home team history [batch, seq_len, input_dim]
            away_history: Away team history [batch, seq_len, input_dim]
            comparison: Comparison features [batch, comparison_dim]

        Returns:
            Tuple of (win_logit, margin), each [batch, 1]
        """
        # Normalize input features
        home_history = self.input_norm(home_history)
        away_history = self.input_norm(away_history)

        # Process home team history through LSTM with temporal attention
        home_out, _ = self.lstm(home_history)
        home_repr = self.dropout(self.attention(home_out))

        # Process away team history through same LSTM with temporal attention
        away_out, _ = self.lstm(away_history)
        away_repr = self.dropout(self.attention(away_out))

        # Concatenate: [home_repr, away_repr, comparison]
        combined = torch.cat([home_repr, away_repr, comparison], dim=1)

        # Win prediction
        win_h = self.dropout(torch.relu(self.win_fc(combined)))
        win_logit = self.win_head(win_h)
        win_prob = torch.sigmoid(win_logit)

        # Margin prediction (using win_prob)
        margin_input = torch.cat([combined, win_prob], dim=1)
        margin_h = self.dropout(torch.relu(self.margin_fc(margin_input)))
        margin = torch.relu(self.margin_head(margin_h))

        return win_logit, margin

    def predict(
        self,
        home_history: torch.Tensor,
        away_history: torch.Tensor,
        comparison: torch.Tensor,
    ) -> dict:
        """Get predictions as a dictionary."""
        with torch.no_grad():
            win_logit, margin = self.forward(home_history, away_history, comparison)
            win_prob = torch.sigmoid(win_logit)
            return {
                'win_prob': win_prob.squeeze(-1),
                'margin': margin.squeeze(-1),
            }

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        self.load_state_dict(torch.load(path, weights_only=True))
