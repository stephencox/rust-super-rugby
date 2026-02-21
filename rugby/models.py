"""Neural network models for rugby prediction."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class ResidualBlock(nn.Module):
    """A single residual MLP block: Linear -> [BN] -> ReLU -> [Dropout] + skip."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 use_batchnorm: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_batchnorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x) if self.proj is not None else x
        h = self.linear(x)
        if self.bn is not None:
            h = self.bn(h)
        h = torch.relu(h)
        if self.dropout is not None:
            h = self.dropout(h)
        return h + residual


class WinClassifier(nn.Module):
    """
    MLP classifier for predicting match winner.

    Outputs probability that home team wins.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64], dropout: float = 0.0,
                 use_batchnorm: bool = False, num_teams: int = 0, team_embed_dim: int = 8):
        super().__init__()
        self.num_teams = num_teams
        self.team_embed_dim = team_embed_dim

        first_dim = input_dim
        if num_teams > 0:
            self.team_embedding = nn.Embedding(num_teams + 1, team_embed_dim, padding_idx=0)
            first_dim += 2 * team_embed_dim

        blocks = []
        prev_dim = first_dim
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h_dim, dropout, use_batchnorm))
            prev_dim = h_dim

        self.backbone = nn.ModuleList(blocks)
        self.head = nn.Linear(prev_dim, 1)

    def _embed_teams(self, x: torch.Tensor, home_team_id: Optional[torch.Tensor],
                     away_team_id: Optional[torch.Tensor]) -> torch.Tensor:
        if self.num_teams > 0:
            if home_team_id is not None and away_team_id is not None:
                home_emb = self.team_embedding(home_team_id)
                away_emb = self.team_embedding(away_team_id)
            else:
                zeros = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                home_emb = self.team_embedding(zeros)
                away_emb = self.team_embedding(zeros)
            x = torch.cat([x, home_emb, away_emb], dim=1)
        return x

    def forward(self, x: torch.Tensor, home_team_id: Optional[torch.Tensor] = None,
                away_team_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            home_team_id: Optional home team IDs [batch]
            away_team_id: Optional away team IDs [batch]

        Returns:
            Win logits [batch] (apply sigmoid for probability)
        """
        h = self._embed_teams(x, home_team_id, away_team_id)
        for block in self.backbone:
            h = block(h)
        return self.head(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor, home_team_id: Optional[torch.Tensor] = None,
                      away_team_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get win probability."""
        with torch.no_grad():
            logits = self.forward(x, home_team_id, away_team_id)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5,
                home_team_id: Optional[torch.Tensor] = None,
                away_team_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get binary predictions."""
        proba = self.predict_proba(x, home_team_id, away_team_id)
        return (proba >= threshold).float()

    def save(self, path: Path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))


class MarginRegressor(nn.Module):
    """
    MLP regressor for predicting match margin (absolute points difference).

    Predicts three quantiles (10th, 50th, 90th percentile) to provide
    a point estimate with confidence interval. Trained with pinball loss.
    Output is non-negative (ReLU on heads).
    """

    QUANTILES = (0.1, 0.5, 0.9)

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64], dropout: float = 0.0,
                 use_batchnorm: bool = False, num_teams: int = 0, team_embed_dim: int = 8):
        super().__init__()
        self.num_teams = num_teams
        self.team_embed_dim = team_embed_dim

        first_dim = input_dim
        if num_teams > 0:
            self.team_embedding = nn.Embedding(num_teams + 1, team_embed_dim, padding_idx=0)
            first_dim += 2 * team_embed_dim

        blocks = []
        prev_dim = first_dim
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h_dim, dropout, use_batchnorm))
            prev_dim = h_dim

        self.backbone = nn.ModuleList(blocks)
        # Three heads: q10, q50, q90
        self.head = nn.Linear(prev_dim, 3)

    def _embed_teams(self, x: torch.Tensor, home_team_id: Optional[torch.Tensor],
                     away_team_id: Optional[torch.Tensor]) -> torch.Tensor:
        if self.num_teams > 0:
            if home_team_id is not None and away_team_id is not None:
                home_emb = self.team_embedding(home_team_id)
                away_emb = self.team_embedding(away_team_id)
            else:
                zeros = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                home_emb = self.team_embedding(zeros)
                away_emb = self.team_embedding(zeros)
            x = torch.cat([x, home_emb, away_emb], dim=1)
        return x

    def forward(self, x: torch.Tensor, home_team_id: Optional[torch.Tensor] = None,
                away_team_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            home_team_id: Optional home team IDs [batch]
            away_team_id: Optional away team IDs [batch]

        Returns:
            Predicted quantiles [batch, 3] (q10, q50, q90), all non-negative.
            Monotonicity enforced: q10 <= q50 <= q90.
        """
        h = self._embed_teams(x, home_team_id, away_team_id)
        for block in self.backbone:
            h = block(h)
        raw = self.head(h)  # [batch, 3]
        # Enforce non-negative and monotonicity: q10 = relu(raw0), q50 = q10 + softplus(raw1), q90 = q50 + softplus(raw2)
        q10 = torch.relu(raw[:, 0])
        q50 = q10 + torch.nn.functional.softplus(raw[:, 1])
        q90 = q50 + torch.nn.functional.softplus(raw[:, 2])
        return torch.stack([q10, q50, q90], dim=1)

    def predict(self, x: torch.Tensor, home_team_id: Optional[torch.Tensor] = None,
                away_team_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get margin predictions [batch, 3] (q10, q50, q90)."""
        with torch.no_grad():
            return self.forward(x, home_team_id, away_team_id)

    def save(self, path: Path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))


# =============================================================================
# Sequence-based models (LSTM)
# =============================================================================

# Per-match features for sequence models (23 dimensions)
MATCH_FEATURE_DIM = 23


class SequenceLSTM(nn.Module):
    """
    Bidirectional LSTM model for rugby match prediction using team history sequences.

    Processes home and away team history through a shared BiLSTM, then
    combines the representations with comparison features for prediction.
    Supports variable-length sequences via pack_padded_sequence and masked attention.

    Architecture:
    1. Process home team history through BiLSTM -> home_repr
    2. Process away team history through same BiLSTM -> away_repr
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
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        repr_size = hidden_size * num_directions  # output dim per timestep

        # LayerNorm on sequence input (normalizes each timestep's features)
        self.input_norm = nn.LayerNorm(input_dim)

        # Shared LSTM for processing team histories
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Encourage remembering by default (Jozefowicz et al., 2015)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias

        # Temporal attention over LSTM timestep outputs
        self.attn_fc = nn.Linear(repr_size, repr_size)
        self.attn_score = nn.Linear(repr_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # FC layer for win prediction
        # Combined: home_repr + away_repr + comparison
        fc_input_size = repr_size * 2 + comparison_dim
        self.win_fc = nn.Linear(fc_input_size, hidden_size)
        self.win_head = nn.Linear(hidden_size, 1)

        # FC layer for margin prediction (takes combined + win_prob)
        self.margin_fc = nn.Linear(fc_input_size + 1, hidden_size)
        self.margin_head = nn.Linear(hidden_size, 1)

    def _make_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean mask [batch, max_len] where True = valid position."""
        return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    def attention(self, lstm_out: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Temporal attention over LSTM outputs with optional padding mask.

        Args:
            lstm_out: LSTM outputs [batch, seq_len, hidden]
            mask: Boolean mask [batch, seq_len] where True = valid (optional)

        Returns:
            Context vector [batch, hidden]
        """
        energy = torch.tanh(self.attn_fc(lstm_out))       # [B, seq_len, hidden]
        scores = self.attn_score(energy).squeeze(-1)       # [B, seq_len]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)             # [B, seq_len]
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # [B, hidden]
        return context

    def _process_sequence(self, history: torch.Tensor,
                          lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process a team history through LSTM with optional packing and masked attention."""
        history = self.input_norm(history)
        seq_len = history.size(1)

        if lengths is not None:
            # Pack for efficient LSTM processing (skip padding computation)
            packed = nn.utils.rnn.pack_padded_sequence(
                history, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True,
                                                           total_length=seq_len)
            mask = self._make_mask(lengths, seq_len)
        else:
            lstm_out, _ = self.lstm(history)
            mask = None

        return self.dropout(self.attention(lstm_out, mask))

    def forward(
        self,
        home_history: torch.Tensor,
        away_history: torch.Tensor,
        comparison: torch.Tensor,
        home_lengths: Optional[torch.Tensor] = None,
        away_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            home_history: Home team history [batch, seq_len, input_dim]
            away_history: Away team history [batch, seq_len, input_dim]
            comparison: Comparison features [batch, comparison_dim]
            home_lengths: Actual sequence lengths for home team [batch] (optional)
            away_lengths: Actual sequence lengths for away team [batch] (optional)

        Returns:
            Tuple of (win_logit, margin), each [batch, 1]
        """
        home_repr = self._process_sequence(home_history, home_lengths)
        away_repr = self._process_sequence(away_history, away_lengths)

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
        home_lengths: Optional[torch.Tensor] = None,
        away_lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """Get predictions as a dictionary."""
        with torch.no_grad():
            win_logit, margin = self.forward(
                home_history, away_history, comparison, home_lengths, away_lengths,
            )
            win_prob = torch.sigmoid(win_logit)
            return {
                'win_prob': win_prob.squeeze(-1),
                'margin': margin.squeeze(-1),
            }

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        self.load_state_dict(torch.load(path, weights_only=True))
