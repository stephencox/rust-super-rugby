"""Tests for rugby.training module."""

import pytest
import numpy as np
import torch
from rugby.training import (
    MLPDataset, train_win_model, train_margin_model,
    _NEGATE_INDICES, _FLIP_INDICES, _SWAP_PAIRS, _RESET_INDICES,
)
from rugby.models import WinClassifier, MarginRegressor
from rugby.features import MatchFeatures


INPUT_DIM = MatchFeatures.num_features()


class TestMLPDataset:
    def test_no_augmentation(self):
        X = np.random.randn(10, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 10).astype(np.float32)
        ds = MLPDataset(X, y, augment=False)
        assert len(ds) == 10
        item = ds[0]
        assert 'features' in item
        assert 'label' in item

    def test_augmentation_swaps(self):
        """Verify augmentation produces correct symmetry."""
        np.random.seed(42)
        X = np.random.randn(1, INPUT_DIM).astype(np.float32)
        y = np.array([1.0], dtype=np.float32)
        home_ids = np.array([1], dtype=np.int64)
        away_ids = np.array([2], dtype=np.int64)

        ds = MLPDataset(X, y, home_ids, away_ids, augment=True)

        # Call many times to get at least one augmented sample
        found_swap = False
        for _ in range(100):
            item = ds[0]
            if item['label'].item() == 0.0:  # Swapped
                found_swap = True
                x_orig = torch.tensor(X[0])
                x_aug = item['features']
                # Differentials should be negated
                for i in _NEGATE_INDICES:
                    assert x_aug[i] == pytest.approx(-x_orig[i].item(), abs=1e-5)
                # Swap pairs should be swapped
                for h, a in _SWAP_PAIRS:
                    assert x_aug[h] == pytest.approx(x_orig[a].item(), abs=1e-5)
                    assert x_aug[a] == pytest.approx(x_orig[h].item(), abs=1e-5)
                # Team IDs should be swapped
                assert item['home_team_id'].item() == 2
                assert item['away_team_id'].item() == 1
                break
        assert found_swap, "Augmentation never triggered in 100 attempts"

    def test_augmentation_indices_cover_all_features(self):
        """All feature indices should be accounted for in augmentation rules."""
        all_indices = set(range(INPUT_DIM))
        negate = set(_NEGATE_INDICES)
        flip = set(_FLIP_INDICES)
        swap = set()
        for h, a in _SWAP_PAIRS:
            swap.add(h)
            swap.add(a)
        reset = set(_RESET_INDICES.keys())
        covered = negate | flip | swap | reset
        # Remaining indices are symmetric (kept unchanged): is_local(4), travel_hours(24)
        uncovered = all_indices - covered
        # These should be symmetric features that don't need transformation
        expected_symmetric = {4, 24}  # is_local, travel_hours
        assert uncovered == expected_symmetric, f"Unexpected uncovered indices: {uncovered - expected_symmetric}"


class TestTrainWinModel:
    def test_basic_training(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        model, history = train_win_model(X, y, epochs=5, verbose=False)
        assert isinstance(model, WinClassifier)
        assert len(history['train_loss']) == 5

    def test_with_validation(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        X_val = np.random.randn(10, INPUT_DIM).astype(np.float32)
        y_val = np.random.randint(0, 2, 10).astype(np.float32)
        model, history = train_win_model(X, y, X_val, y_val, epochs=5, verbose=False)
        assert 'best_val_acc' in history
        assert len(history['val_acc']) == 5

    def test_with_team_embeddings(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        home_ids = np.random.randint(1, 6, 50).astype(np.int64)
        away_ids = np.random.randint(1, 6, 50).astype(np.int64)
        model, _ = train_win_model(X, y, epochs=3, num_teams=5,
                                    home_team_ids=home_ids, away_team_ids=away_ids,
                                    verbose=False)
        assert model.num_teams == 5

    def test_label_smoothing(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        model, history = train_win_model(X, y, epochs=3, label_smoothing=0.1, verbose=False)
        assert len(history['train_loss']) == 3

    def test_augmentation(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        model, _ = train_win_model(X, y, epochs=3, augment_swap=True, verbose=False)
        assert isinstance(model, WinClassifier)


class TestTrainMarginModel:
    def test_basic_training(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.abs(np.random.randn(50).astype(np.float32)) * 10
        model, history = train_margin_model(X, y, epochs=5, verbose=False)
        assert isinstance(model, MarginRegressor)
        assert 'margin_scale' in history

    def test_with_validation(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.abs(np.random.randn(50).astype(np.float32)) * 10
        X_val = np.random.randn(10, INPUT_DIM).astype(np.float32)
        y_val = np.abs(np.random.randn(10).astype(np.float32)) * 10
        model, history = train_margin_model(X, y, X_val, y_val, epochs=5, verbose=False)
        assert 'best_val_mae' in history

    def test_margin_scale(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        y = np.full(50, 15.0, dtype=np.float32)
        _, history = train_margin_model(X, y, epochs=3, verbose=False)
        assert history['margin_scale'] == pytest.approx(15.0)
