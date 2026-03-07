"""Tests for rugby.training module."""

import pytest
import numpy as np
import torch
from rugby.training import (
    MLPDataset, SequenceDataset, _seq_length,
    train_win_model, train_margin_model,
    _NEGATE_INDICES, _FLIP_INDICES, _SWAP_PAIRS, _RESET_INDICES,
    _COMP_NEGATE_INDICES, _COMP_FLIP_INDICES, _COMP_SWAP_PAIRS,
)
from rugby.models import WinClassifier, MarginRegressor
from rugby.features import MatchFeatures, SequenceDataSample, SequenceNormalizer


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

    def test_swap_pairs_are_semantically_correct(self):
        """Each swap pair must swap a home_* feature with its away_* counterpart."""
        names = MatchFeatures.feature_names()
        for h_idx, a_idx in _SWAP_PAIRS:
            h_name = names[h_idx]
            a_name = names[a_idx]
            # One should contain 'home' and the other 'away'
            if 'home' in h_name:
                expected_away = h_name.replace('home', 'away')
                assert a_name == expected_away, (
                    f"Swap pair ({h_idx},{a_idx}): {h_name} <-> {a_name}, "
                    f"expected {h_name} <-> {expected_away}"
                )
            elif 'away' in h_name:
                expected_home = h_name.replace('away', 'home')
                assert a_name == expected_home, (
                    f"Swap pair ({h_idx},{a_idx}): {h_name} <-> {a_name}, "
                    f"expected {h_name} <-> {expected_home}"
                )


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
        y = np.random.randn(50).astype(np.float32) * 10 + 15
        _, history = train_margin_model(X, y, epochs=3, verbose=False)
        assert history['margin_scale'] == pytest.approx(float(np.std(y)), rel=1e-3)


COMP_DIM = 50


class TestSequenceNormalizerPadding:
    def test_zero_padding_preserved(self):
        """Zero-padded rows must remain zero after normalization."""
        seq_len, feat_dim = 10, 23
        # 6 real timesteps, 4 zero-padded
        seq = np.random.randn(seq_len, feat_dim).astype(np.float32)
        seq[6:] = 0.0

        sample = SequenceDataSample(
            home_history=seq.copy(),
            away_history=seq.copy(),
            comparison=np.random.randn(COMP_DIM).astype(np.float32),
            home_team_id=0, away_team_id=1,
            home_win=True, home_score=25.0, away_score=20.0,
        )

        norm = SequenceNormalizer()
        norm.fit([sample])
        transformed = norm.transform_sample(sample)

        # Padding rows should still be all zeros
        assert np.all(transformed.home_history[6:] == 0.0)
        assert np.all(transformed.away_history[6:] == 0.0)
        # Real rows should be normalized (non-zero)
        assert np.any(transformed.home_history[:6] != 0.0)

    def test_seq_length_after_normalization(self):
        """_seq_length must return correct count after normalization."""
        seq_len, feat_dim = 10, 23
        seq = np.random.randn(seq_len, feat_dim).astype(np.float32)
        seq[7:] = 0.0  # 7 real timesteps

        sample = SequenceDataSample(
            home_history=seq.copy(),
            away_history=seq.copy(),
            comparison=np.random.randn(COMP_DIM).astype(np.float32),
            home_team_id=0, away_team_id=1,
            home_win=True, home_score=25.0, away_score=20.0,
        )

        norm = SequenceNormalizer()
        norm.fit([sample])
        transformed = norm.transform_sample(sample)

        assert _seq_length(transformed.home_history) == 7
        assert _seq_length(transformed.away_history) == 7


class TestSequenceDatasetAugmentation:
    def _make_sample(self):
        np.random.seed(42)
        return SequenceDataSample(
            home_history=np.random.randn(10, 23).astype(np.float32),
            away_history=np.random.randn(10, 23).astype(np.float32),
            comparison=np.random.randn(COMP_DIM).astype(np.float32),
            home_team_id=0, away_team_id=1,
            home_win=True, home_score=25.0, away_score=20.0,
        )

    def test_augmented_comparison_negates_differentials(self):
        """Differential features in comparison should be negated on swap."""
        sample = self._make_sample()
        ds = SequenceDataset([sample], augment=True)

        for _ in range(100):
            item = ds[0]
            if item['home_win'].item() == 0.0:  # Swapped
                comp_orig = sample.comparison
                comp_aug = item['comparison'].numpy()
                for i in _COMP_NEGATE_INDICES:
                    assert comp_aug[i] == pytest.approx(-comp_orig[i], abs=1e-5), \
                        f"Index {i} should be negated"
                break
        else:
            pytest.fail("Augmentation never triggered in 100 attempts")

    def test_augmented_comparison_preserves_symmetric(self):
        """Symmetric features (is_local, temporal) should be unchanged on swap."""
        sample = self._make_sample()
        ds = SequenceDataset([sample], augment=True)
        symmetric = {4, 15, 16, 17, 18, 19, 20, 21, 29, 30}

        for _ in range(100):
            item = ds[0]
            if item['home_win'].item() == 0.0:  # Swapped
                comp_orig = sample.comparison
                comp_aug = item['comparison'].numpy()
                for i in symmetric:
                    assert comp_aug[i] == pytest.approx(comp_orig[i], abs=1e-5), \
                        f"Index {i} should be preserved (symmetric)"
                break
        else:
            pytest.fail("Augmentation never triggered in 100 attempts")

    def test_augmented_comparison_swaps_home_away(self):
        """Home/away pairs in comparison should be swapped."""
        sample = self._make_sample()
        ds = SequenceDataset([sample], augment=True)

        for _ in range(100):
            item = ds[0]
            if item['home_win'].item() == 0.0:  # Swapped
                comp_orig = sample.comparison
                comp_aug = item['comparison'].numpy()
                for h, a in _COMP_SWAP_PAIRS:
                    assert comp_aug[h] == pytest.approx(comp_orig[a], abs=1e-5), \
                        f"Swap pair ({h},{a}): aug[{h}] should equal orig[{a}]"
                    assert comp_aug[a] == pytest.approx(comp_orig[h], abs=1e-5), \
                        f"Swap pair ({h},{a}): aug[{a}] should equal orig[{h}]"
                break
        else:
            pytest.fail("Augmentation never triggered in 100 attempts")

    def test_comp_augmentation_indices_cover_all(self):
        """All 50 comparison indices should be accounted for."""
        all_indices = set(range(COMP_DIM))
        negate = set(_COMP_NEGATE_INDICES)
        flip = set(_COMP_FLIP_INDICES)
        swap = set()
        for h, a in _COMP_SWAP_PAIRS:
            swap.add(h)
            swap.add(a)
        covered = negate | flip | swap
        uncovered = all_indices - covered
        # Symmetric features that don't need transformation
        expected_symmetric = {4, 15, 16, 17, 18, 19, 20, 21, 29, 30}
        assert uncovered == expected_symmetric, \
            f"Unexpected uncovered indices: {uncovered - expected_symmetric}"
