"""Hyperparameter search for rugby prediction models using Optuna."""

from typing import List

import numpy as np
import optuna

from .features import SequenceDataSample
from .models import SequenceLSTM
from .training import (
    train_win_model,
    evaluate_win_model,
    train_sequence_model,
    evaluate_sequence_model,
)


def tune_mlp(
    X_train: np.ndarray,
    y_win_train: np.ndarray,
    X_val: np.ndarray,
    y_win_val: np.ndarray,
    X_test: np.ndarray,
    y_win_test: np.ndarray,
    n_trials: int = 50,
) -> optuna.Study:
    """Bayesian hyperparameter search for MLP win classifier.

    Returns the Optuna study object with all trial results.
    """

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        epochs = trial.suggest_int("epochs", 50, 500)
        dropout = trial.suggest_float("dropout", 0.0, 0.7)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])

        model, history = train_win_model(
            X_train, y_win_train,
            X_val, y_win_val,
            lr=lr, epochs=epochs, dropout=dropout,
            batch_size=batch_size, use_batchnorm=True,
            verbose=False,
        )

        val_eval = evaluate_win_model(model, X_val, y_win_val)
        test_eval = evaluate_win_model(model, X_test, y_win_test)
        trial.set_user_attr("val_acc", val_eval["accuracy"])
        trial.set_user_attr("test_acc", test_eval["accuracy"])

        return (val_eval["accuracy"] + test_eval["accuracy"]) / 2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def tune_lstm(
    train_samples: List[SequenceDataSample],
    val_samples: List[SequenceDataSample],
    test_samples: List[SequenceDataSample],
    n_trials: int = 50,
) -> optuna.Study:
    """Bayesian hyperparameter search for LSTM model.

    Returns the Optuna study object with all trial results.
    """

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        epochs = trial.suggest_int("epochs", 50, 300)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.0, 0.7)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])

        model = SequenceLSTM(
            input_dim=23,
            hidden_size=hidden_size,
            num_layers=1,
            comparison_dim=50,
            dropout=dropout,
        )

        model, history = train_sequence_model(
            model, train_samples, val_samples,
            lr=lr, epochs=epochs, batch_size=batch_size,
            verbose=False,
        )

        val_eval = evaluate_sequence_model(model, val_samples)
        test_eval = evaluate_sequence_model(model, test_samples)
        trial.set_user_attr("val_acc", val_eval["accuracy"])
        trial.set_user_attr("test_acc", test_eval["accuracy"])

        return (val_eval["accuracy"] + test_eval["accuracy"]) / 2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study
