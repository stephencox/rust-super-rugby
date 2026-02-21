"""Training and evaluation functions."""

import random

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, Optional, Dict, List
from .models import WinClassifier, MarginRegressor, SequenceLSTM
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
    weight_decay: float = 0.001,
    use_batchnorm: bool = False,
    early_stopping_patience: int = 0,
    verbose: bool = True,
) -> Tuple[WinClassifier, Dict]:
    """
    Train win classification model.

    Returns:
        Trained model and training history
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # Create data loader (drop_last avoids BatchNorm issues with batch_size=1)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=use_batchnorm)

    # Model and optimizer
    model = WinClassifier(X_train.shape[1], hidden_dims, dropout, use_batchnorm=use_batchnorm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    best_val_acc = 0
    best_val_loss = float('inf')
    best_model_state = None
    stale_epochs = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.1%}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.1%}")

            # Early stopping
            if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.1%}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    history['best_val_acc'] = best_val_acc
    return model, history


def train_margin_model(
    X_train: np.ndarray,
    y_margin_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_margin_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = [64],
    dropout: float = 0.0,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    weight_decay: float = 0.001,
    use_batchnorm: bool = False,
    early_stopping_patience: int = 0,
    verbose: bool = True,
) -> Tuple[MarginRegressor, Dict]:
    """
    Train margin regression model.

    Normalizes margin targets by training-set mean internally.
    Returns margin_scale in history for denormalization at inference.

    Returns:
        Trained model and training history
    """
    # Normalize margin targets so Huber loss is on a comparable scale
    margin_scale = float(np.mean(y_margin_train)) if np.mean(y_margin_train) > 0 else 1.0
    y_margin_train_scaled = y_margin_train / margin_scale
    y_margin_val_scaled = y_margin_val / margin_scale if y_margin_val is not None else None

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_margin_train_scaled, dtype=torch.float32)

    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_margin_val_scaled, dtype=torch.float32)

    # Create data loader (drop_last avoids BatchNorm issues with batch_size=1)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=use_batchnorm)

    # Model and optimizer
    model = MarginRegressor(X_train.shape[1], hidden_dims, dropout, use_batchnorm=use_batchnorm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training history
    history = {
        'train_loss': [],
        'train_margin_mae': [],
        'val_loss': [],
        'val_margin_mae': [],
    }

    best_val_mae = float('inf')
    best_val_loss = float('inf')
    best_model_state = None
    stale_epochs = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_margin_mae = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            margin_pred = model(X_batch)
            loss = nn.functional.huber_loss(margin_pred, y_batch, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            train_margin_mae += (margin_pred - y_batch).abs().sum().item()
            train_total += len(X_batch)

        train_loss /= train_total
        train_margin_mae = train_margin_mae / train_total * margin_scale  # report in real points
        history['train_loss'].append(train_loss)
        history['train_margin_mae'].append(train_margin_mae)

        # Validation
        if X_val is not None:
            model.train(False)
            with torch.no_grad():
                val_margin_pred = model(X_val_t)
                val_loss = nn.functional.huber_loss(val_margin_pred, y_val_t, delta=1.0).item()
                val_margin_mae = (val_margin_pred - y_val_t).abs().mean().item() * margin_scale

            history['val_loss'].append(val_loss)
            history['val_margin_mae'].append(val_margin_mae)

            if val_margin_mae < best_val_mae or (val_margin_mae == best_val_mae and val_loss < best_val_loss):
                best_val_mae = val_margin_mae
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_mae={train_margin_mae:.1f}, "
                      f"val_loss={val_loss:.4f}, val_mae={val_margin_mae:.1f}")

            # Early stopping
            if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_mae={train_margin_mae:.1f}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    history['best_val_mae'] = best_val_mae
    history['margin_scale'] = margin_scale
    return model, history


def evaluate_win_model(model: WinClassifier, X: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate win classification model."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train(False)
    with torch.no_grad():
        probs = model.predict_proba(X_t)
        preds = (probs >= 0.5).float()

        accuracy = (preds == y_t).float().mean().item()
        loss = nn.BCEWithLogitsLoss()(model(X_t), y_t).item()

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


# =============================================================================
# Sequence model training (LSTM and Transformer)
# =============================================================================

class SequenceDataset(Dataset):
    """Dataset for sequence models."""

    def __init__(self, samples: List[SequenceDataSample], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        margin = abs(s.home_score - s.away_score)
        home_win = 1.0 if s.home_win else 0.0

        if self.augment and random.random() < 0.5:
            # Swap home/away and flip the label
            return {
                'home_history': torch.tensor(s.away_history, dtype=torch.float32),
                'away_history': torch.tensor(s.home_history, dtype=torch.float32),
                'comparison': torch.tensor(-s.comparison, dtype=torch.float32),
                'home_win': torch.tensor(1.0 - home_win, dtype=torch.float32),
                'margin': torch.tensor(margin, dtype=torch.float32),
            }

        return {
            'home_history': torch.tensor(s.home_history, dtype=torch.float32),
            'away_history': torch.tensor(s.away_history, dtype=torch.float32),
            'comparison': torch.tensor(s.comparison, dtype=torch.float32),
            'home_win': torch.tensor(home_win, dtype=torch.float32),
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
    label_smoothing: float = 0.0,
    augment_swap: bool = False,
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
        label_smoothing: Smooth hard 0/1 targets toward 0.5 (0.0 = off)
        augment_swap: Randomly swap home/away during training
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    train_dataset = SequenceDataset(train_samples, augment=augment_swap)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_samples:
        val_dataset = SequenceDataset(val_samples)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(train_samples) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
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

    best_val_loss = float('inf')
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

            # Label smoothing: shift targets toward 0.5
            smooth_target = batch['home_win'] * (1 - label_smoothing) + 0.5 * label_smoothing
            win_loss = bce_loss(win_logit.squeeze(-1), smooth_target)
            margin_loss = nn.functional.huber_loss(
                margin_pred.squeeze(-1), batch['margin'], delta=10.0
            )
            loss = win_weight * win_loss + margin_weight * margin_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(batch['home_win'])
            preds = (torch.sigmoid(win_logit.squeeze(-1)) >= 0.5).float()
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

                    # Use unsmoothed targets for validation loss
                    win_loss = bce_loss(win_logit.squeeze(-1), batch['home_win'])
                    margin_loss = nn.functional.huber_loss(
                        margin_pred.squeeze(-1), batch['margin'], delta=10.0
                    )
                    loss = win_weight * win_loss + margin_weight * margin_loss

                    val_loss += loss.item() * len(batch['home_win'])
                    preds = (torch.sigmoid(win_logit.squeeze(-1)) >= 0.5).float()
                    val_correct += (preds == batch['home_win']).sum().item()
                    val_total += len(batch['home_win'])

                    # Margin MAE
                    val_margin_mae += (margin_pred.squeeze(-1) - batch['margin']).abs().sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total
            val_margin_mae /= val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_margin_mae'].append(val_margin_mae)

            # Track best model by val_loss (more stable than val_acc on small val sets)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
    history['best_val_loss'] = best_val_loss
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

        probs = torch.sigmoid(win_logit.squeeze(-1))
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
        margin_mae = (margin_pred.squeeze(-1) - batch['margin']).abs().mean().item()

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
