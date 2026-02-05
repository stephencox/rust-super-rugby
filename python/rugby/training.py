"""Training and evaluation functions."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, Optional, Dict, List
from .models import WinClassifier, ScoreRegressor, MatchPredictor, SequenceLSTM
from .features import SequenceDataSample


def train_win_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = [64],
    dropout: float = 0.0,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tuple[WinClassifier, Dict]:
    """
    Train win classification model.

    Args:
        X_train: Training features
        y_train: Training labels (1 = home win, 0 = away win)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        lr: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = WinClassifier(X_train.shape[1], hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += len(X_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        if X_val is not None:
            model.train(False)
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                val_acc = (val_preds == y_val_t).sum().item() / len(y_val_t)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.1%}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.1%}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.1%}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    history['best_val_acc'] = best_val_acc
    return model, history


def train_score_model(
    X_train: np.ndarray,
    y_home_train: np.ndarray,
    y_away_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_home_val: Optional[np.ndarray] = None,
    y_away_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = [64],
    dropout: float = 0.0,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tuple[ScoreRegressor, Dict]:
    """
    Train score regression model.

    Args:
        X_train: Training features
        y_home_train: Training home scores
        y_away_train: Training away scores
        X_val: Validation features (optional)
        y_home_val: Validation home scores (optional)
        y_away_val: Validation away scores (optional)
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        lr: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_home_train_t = torch.tensor(y_home_train, dtype=torch.float32)
    y_away_train_t = torch.tensor(y_away_train, dtype=torch.float32)

    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_home_val_t = torch.tensor(y_home_val, dtype=torch.float32)
        y_away_val_t = torch.tensor(y_away_val, dtype=torch.float32)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_home_train_t, y_away_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = ScoreRegressor(X_train.shape[1], hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_home_mae': [],
        'val_away_mae': [],
    }

    best_val_mae = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_mae = 0
        train_total = 0

        for X_batch, y_home_batch, y_away_batch in train_loader:
            optimizer.zero_grad()
            home_pred, away_pred = model(X_batch)

            # Huber loss for robustness to outliers
            loss_home = nn.functional.huber_loss(home_pred, y_home_batch, delta=10.0)
            loss_away = nn.functional.huber_loss(away_pred, y_away_batch, delta=10.0)
            loss = loss_home + loss_away

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            mae = ((home_pred - y_home_batch).abs() + (away_pred - y_away_batch).abs()) / 2
            train_mae += mae.sum().item()
            train_total += len(X_batch)

        train_loss /= train_total
        train_mae /= train_total
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)

        # Validation
        if X_val is not None:
            model.train(False)
            with torch.no_grad():
                home_pred_val, away_pred_val = model(X_val_t)
                loss_home = nn.functional.huber_loss(home_pred_val, y_home_val_t, delta=10.0)
                loss_away = nn.functional.huber_loss(away_pred_val, y_away_val_t, delta=10.0)
                val_loss = (loss_home + loss_away).item()

                home_mae = (home_pred_val - y_home_val_t).abs().mean().item()
                away_mae = (away_pred_val - y_away_val_t).abs().mean().item()
                val_mae = (home_mae + away_mae) / 2

            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_home_mae'].append(home_mae)
            history['val_away_mae'].append(away_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_mae={train_mae:.1f}, "
                      f"val_loss={val_loss:.4f}, val_mae={val_mae:.1f} (home={home_mae:.1f}, away={away_mae:.1f})")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_mae={train_mae:.1f}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    history['best_val_mae'] = best_val_mae
    return model, history


def evaluate_win_model(model: WinClassifier, X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Evaluate win classification model.

    Returns:
        Dictionary with evaluation metrics
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train(False)
    with torch.no_grad():
        probs = model.predict_proba(X_t)
        preds = (probs >= 0.5).float()

        accuracy = (preds == y_t).float().mean().item()
        loss = nn.BCEWithLogitsLoss()(model(X_t), y_t).item()

        # Confusion matrix
        tp = ((preds == 1) & (y_t == 1)).sum().item()
        tn = ((preds == 0) & (y_t == 0)).sum().item()
        fp = ((preds == 1) & (y_t == 0)).sum().item()
        fn = ((preds == 0) & (y_t == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'predicted_home_wins': int((preds == 1).sum().item()),
        'actual_home_wins': int(y_t.sum().item()),
        'total': len(y),
    }


def evaluate_score_model(model: ScoreRegressor, X: np.ndarray,
                         y_home: np.ndarray, y_away: np.ndarray) -> Dict:
    """
    Evaluate score regression model.

    Returns:
        Dictionary with evaluation metrics
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_home_t = torch.tensor(y_home, dtype=torch.float32)
    y_away_t = torch.tensor(y_away, dtype=torch.float32)

    model.train(False)
    with torch.no_grad():
        home_pred, away_pred = model(X_t)

        home_mae = (home_pred - y_home_t).abs().mean().item()
        away_mae = (away_pred - y_away_t).abs().mean().item()
        total_mae = (home_mae + away_mae) / 2

        home_rmse = ((home_pred - y_home_t) ** 2).mean().sqrt().item()
        away_rmse = ((away_pred - y_away_t) ** 2).mean().sqrt().item()

        # Check if predicted winner matches actual winner
        pred_winner = (home_pred > away_pred).float()
        actual_winner = (y_home_t > y_away_t).float()
        winner_acc = (pred_winner == actual_winner).float().mean().item()

        # Margin prediction accuracy
        pred_margin = (home_pred - away_pred).abs()
        actual_margin = (y_home_t - y_away_t).abs()
        margin_mae = (pred_margin - actual_margin).abs().mean().item()

    return {
        'home_mae': home_mae,
        'away_mae': away_mae,
        'total_mae': total_mae,
        'home_rmse': home_rmse,
        'away_rmse': away_rmse,
        'winner_accuracy': winner_acc,
        'margin_mae': margin_mae,
        'mean_home_pred': home_pred.mean().item(),
        'mean_away_pred': away_pred.mean().item(),
        'mean_home_actual': y_home.mean(),
        'mean_away_actual': y_away.mean(),
    }


# =============================================================================
# Combined model training (MatchPredictor)
# =============================================================================

def train_match_predictor(
    X_train: np.ndarray,
    y_win_train: np.ndarray,
    y_margin_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_win_val: Optional[np.ndarray] = None,
    y_margin_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = [64],
    dropout: float = 0.0,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    win_weight: float = 1.0,
    margin_weight: float = 0.1,
    verbose: bool = True,
) -> Tuple[MatchPredictor, Dict]:
    """
    Train combined match predictor (win + margin).

    The model predicts win probability first, then uses that to predict margin.

    Args:
        X_train: Training features
        y_win_train: Training win labels (1 = home win)
        y_margin_train: Training margins (absolute points difference)
        X_val: Validation features (optional)
        y_win_val: Validation win labels
        y_margin_val: Validation margins
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        lr: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        win_weight: Weight for win prediction loss
        margin_weight: Weight for margin prediction loss
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_win_train_t = torch.tensor(y_win_train, dtype=torch.float32)
    y_margin_train_t = torch.tensor(y_margin_train, dtype=torch.float32)

    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_win_val_t = torch.tensor(y_win_val, dtype=torch.float32)
        y_margin_val_t = torch.tensor(y_margin_val, dtype=torch.float32)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_win_train_t, y_margin_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = MatchPredictor(X_train.shape[1], hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_win_acc': [],
        'train_margin_mae': [],
        'val_loss': [],
        'val_win_acc': [],
        'val_margin_mae': [],
    }

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_win_correct = 0
        train_margin_mae = 0
        train_total = 0

        for X_batch, y_win_batch, y_margin_batch in train_loader:
            optimizer.zero_grad()

            win_logit, margin_pred = model(X_batch)

            # Combined loss
            loss_win = bce_loss(win_logit, y_win_batch)
            loss_margin = nn.functional.huber_loss(margin_pred, y_margin_batch, delta=10.0)
            loss = win_weight * loss_win + margin_weight * loss_margin

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            win_preds = (torch.sigmoid(win_logit) >= 0.5).float()
            train_win_correct += (win_preds == y_win_batch).sum().item()
            train_margin_mae += (margin_pred - y_margin_batch).abs().sum().item()
            train_total += len(X_batch)

        train_loss /= train_total
        train_win_acc = train_win_correct / train_total
        train_margin_mae /= train_total
        history['train_loss'].append(train_loss)
        history['train_win_acc'].append(train_win_acc)
        history['train_margin_mae'].append(train_margin_mae)

        # Validation
        if X_val is not None:
            model.train(False)
            with torch.no_grad():
                val_win_logit, val_margin_pred = model(X_val_t)

                loss_win = bce_loss(val_win_logit, y_win_val_t)
                loss_margin = nn.functional.huber_loss(val_margin_pred, y_margin_val_t, delta=10.0)
                val_loss = (win_weight * loss_win + margin_weight * loss_margin).item()

                val_win_preds = (torch.sigmoid(val_win_logit) >= 0.5).float()
                val_win_acc = (val_win_preds == y_win_val_t).float().mean().item()
                val_margin_mae = (val_margin_pred - y_margin_val_t).abs().mean().item()

            history['val_loss'].append(val_loss)
            history['val_win_acc'].append(val_win_acc)
            history['val_margin_mae'].append(val_margin_mae)

            if val_win_acc > best_val_acc:
                best_val_acc = val_win_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_win_acc:.1%}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_win_acc:.1%}, val_margin_mae={val_margin_mae:.1f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_win_acc:.1%}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    history['best_val_acc'] = best_val_acc
    return model, history


def evaluate_match_predictor(
    model: MatchPredictor,
    X: np.ndarray,
    y_win: np.ndarray,
    y_margin: np.ndarray,
) -> Dict:
    """
    Evaluate combined match predictor.

    Returns:
        Dictionary with evaluation metrics
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_win_t = torch.tensor(y_win, dtype=torch.float32)
    y_margin_t = torch.tensor(y_margin, dtype=torch.float32)

    model.train(False)
    with torch.no_grad():
        win_logit, margin_pred = model(X_t)
        win_prob = torch.sigmoid(win_logit)
        win_preds = (win_prob >= 0.5).float()

        # Win metrics
        win_accuracy = (win_preds == y_win_t).float().mean().item()
        tp = ((win_preds == 1) & (y_win_t == 1)).sum().item()
        tn = ((win_preds == 0) & (y_win_t == 0)).sum().item()
        fp = ((win_preds == 1) & (y_win_t == 0)).sum().item()
        fn = ((win_preds == 0) & (y_win_t == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Margin metrics
        margin_mae = (margin_pred - y_margin_t).abs().mean().item()

    return {
        'win_accuracy': win_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'margin_mae': margin_mae,
        'mean_margin_pred': margin_pred.mean().item(),
        'mean_margin_actual': y_margin.mean(),
        'mean_win_prob': win_prob.mean().item(),
    }


# =============================================================================
# Sequence model training (LSTM and Transformer)
# =============================================================================

class SequenceDataset(Dataset):
    """Dataset for sequence models."""

    def __init__(self, samples: List[SequenceDataSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Margin is absolute points difference
        margin = abs(s.home_score - s.away_score)
        return {
            'home_history': torch.tensor(s.home_history, dtype=torch.float32),
            'away_history': torch.tensor(s.away_history, dtype=torch.float32),
            'comparison': torch.tensor(s.comparison, dtype=torch.float32),
            'home_team_id': torch.tensor(s.home_team_id, dtype=torch.long),
            'away_team_id': torch.tensor(s.away_team_id, dtype=torch.long),
            'home_win': torch.tensor(1.0 if s.home_win else 0.0, dtype=torch.float32),
            'margin': torch.tensor(margin, dtype=torch.float32),
        }


def train_sequence_model(
    model: nn.Module,
    train_samples: List[SequenceDataSample],
    val_samples: Optional[List[SequenceDataSample]] = None,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    win_weight: float = 1.0,
    margin_weight: float = 0.1,
    use_team_ids: bool = False,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train a sequence model (LSTM).

    Args:
        model: SequenceLSTM
        train_samples: Training samples
        val_samples: Validation samples (optional)
        lr: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        win_weight: Weight for win prediction loss
        margin_weight: Weight for margin prediction loss
        use_team_ids: Whether to pass team IDs to the model
        weight_decay: L2 regularization strength
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    train_dataset = SequenceDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_samples:
        val_dataset = SequenceDataset(val_samples)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5
    )

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_margin_mae': [],
    }

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            if use_team_ids and hasattr(model, 'team_embedding'):
                win_logit, margin_pred = model(
                    batch['home_history'],
                    batch['away_history'],
                    batch['comparison'],
                    batch['home_team_id'],
                    batch['away_team_id'],
                )
            else:
                win_logit, margin_pred = model(
                    batch['home_history'],
                    batch['away_history'],
                    batch['comparison'],
                )

            # Loss
            win_loss = bce_loss(win_logit.squeeze(), batch['home_win'])
            margin_loss = nn.functional.huber_loss(
                margin_pred.squeeze(), batch['margin'], delta=10.0
            )
            loss = win_weight * win_loss + margin_weight * margin_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch['home_win'])
            preds = (torch.sigmoid(win_logit.squeeze()) >= 0.5).float()
            train_correct += (preds == batch['home_win']).sum().item()
            train_total += len(batch['home_win'])

        train_loss /= train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        if val_samples:
            model.train(False)
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_margin_mae = 0

            with torch.no_grad():
                for batch in val_loader:
                    if use_team_ids and hasattr(model, 'team_embedding'):
                        win_logit, margin_pred = model(
                            batch['home_history'],
                            batch['away_history'],
                            batch['comparison'],
                            batch['home_team_id'],
                            batch['away_team_id'],
                        )
                    else:
                        win_logit, margin_pred = model(
                            batch['home_history'],
                            batch['away_history'],
                            batch['comparison'],
                        )

                    # Loss
                    win_loss = bce_loss(win_logit.squeeze(), batch['home_win'])
                    margin_loss = nn.functional.huber_loss(
                        margin_pred.squeeze(), batch['margin'], delta=10.0
                    )
                    loss = win_weight * win_loss + margin_weight * margin_loss

                    val_loss += loss.item() * len(batch['home_win'])
                    preds = (torch.sigmoid(win_logit.squeeze()) >= 0.5).float()
                    val_correct += (preds == batch['home_win']).sum().item()
                    val_total += len(batch['home_win'])

                    # Margin MAE
                    val_margin_mae += (margin_pred.squeeze() - batch['margin']).abs().sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total
            val_margin_mae /= val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_margin_mae'].append(val_margin_mae)

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.1%}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.1%}, val_margin_mae={val_margin_mae:.1f}, lr={current_lr:.1e}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.1%}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    history['best_val_acc'] = best_val_acc
    return model, history


def evaluate_sequence_model(
    model: nn.Module,
    samples: List[SequenceDataSample],
    use_team_ids: bool = False,
) -> Dict:
    """
    Evaluate a sequence model.

    Returns:
        Dictionary with evaluation metrics
    """
    dataset = SequenceDataset(samples)
    loader = DataLoader(dataset, batch_size=len(samples))

    model.train(False)

    with torch.no_grad():
        batch = next(iter(loader))

        if use_team_ids and hasattr(model, 'team_embedding'):
            win_logit, margin_pred = model(
                batch['home_history'],
                batch['away_history'],
                batch['comparison'],
                batch['home_team_id'],
                batch['away_team_id'],
            )
        else:
            win_logit, margin_pred = model(
                batch['home_history'],
                batch['away_history'],
                batch['comparison'],
            )

        probs = torch.sigmoid(win_logit.squeeze())
        preds = (probs >= 0.5).float()
        targets = batch['home_win']

        # Accuracy metrics
        accuracy = (preds == targets).float().mean().item()

        tp = ((preds == 1) & (targets == 1)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Margin metrics
        margin_mae = (margin_pred.squeeze() - batch['margin']).abs().mean().item()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'margin_mae': margin_mae,
        'mean_margin_pred': margin_pred.mean().item(),
        'mean_margin_actual': batch['margin'].mean().item(),
        'mean_win_prob': probs.mean().item(),
    }
