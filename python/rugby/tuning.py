"""Hyperparameter search for rugby prediction models."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .features import SequenceDataSample
from .models import MatchPredictor, SequenceLSTM
from .training import (
    train_match_predictor,
    evaluate_match_predictor,
    train_sequence_model,
    evaluate_sequence_model,
)


@dataclass
class TuningResult:
    """Result of a single hyperparameter trial."""
    hyperparams: Dict
    train_acc: float
    val_acc: float
    test_acc: float


def tune_mlp(
    X_train: np.ndarray,
    y_win_train: np.ndarray,
    y_margin_train: np.ndarray,
    X_val: np.ndarray,
    y_win_val: np.ndarray,
    y_margin_val: np.ndarray,
    X_test: np.ndarray,
    y_win_test: np.ndarray,
    y_margin_test: np.ndarray,
    search_space: Optional[Dict] = None,
) -> List[TuningResult]:
    """Grid search over lr x epochs for MLP match predictor.

    Default search space:
        lr: [0.0001, 0.0005, 0.001, 0.005, 0.01]
        epochs: [50, 100, 200, 300, 500]
    """
    if search_space is None:
        search_space = {
            "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01],
            "epochs": [50, 100, 200, 300, 500],
        }

    results: List[TuningResult] = []
    total = len(search_space["lr"]) * len(search_space["epochs"])
    trial = 0

    for lr in search_space["lr"]:
        for epochs in search_space["epochs"]:
            trial += 1
            print(f"  Trial {trial}/{total}: lr={lr}, epochs={epochs}")

            model, history = train_match_predictor(
                X_train, y_win_train, y_margin_train,
                X_val, y_win_val, y_margin_val,
                lr=lr, epochs=epochs, verbose=False,
            )

            train_eval = evaluate_match_predictor(model, X_train, y_win_train, y_margin_train)
            val_eval = evaluate_match_predictor(model, X_val, y_win_val, y_margin_val)
            test_eval = evaluate_match_predictor(model, X_test, y_win_test, y_margin_test)

            result = TuningResult(
                hyperparams={"lr": lr, "epochs": epochs},
                train_acc=train_eval["win_accuracy"],
                val_acc=val_eval["win_accuracy"],
                test_acc=test_eval["win_accuracy"],
            )
            results.append(result)
            print(f"    train={result.train_acc:.1%}, val={result.val_acc:.1%}, test={result.test_acc:.1%}")

    results.sort(key=lambda r: r.val_acc, reverse=True)
    return results


def tune_lstm(
    train_samples: List[SequenceDataSample],
    val_samples: List[SequenceDataSample],
    test_samples: List[SequenceDataSample],
    search_space: Optional[Dict] = None,
) -> List[TuningResult]:
    """Grid search over lr x hidden_size x epochs for LSTM model.

    Default search space:
        lr: [0.0005, 0.001, 0.005]
        hidden_size: [32, 64, 128]
        epochs: [50, 100, 200]
    """
    if search_space is None:
        search_space = {
            "lr": [0.0005, 0.001, 0.005],
            "hidden_size": [32, 64, 128],
            "epochs": [50, 100, 200],
        }

    results: List[TuningResult] = []
    total = len(search_space["lr"]) * len(search_space["hidden_size"]) * len(search_space["epochs"])
    trial = 0

    for lr in search_space["lr"]:
        for hidden_size in search_space["hidden_size"]:
            for epochs in search_space["epochs"]:
                trial += 1
                print(f"  Trial {trial}/{total}: lr={lr}, hidden={hidden_size}, epochs={epochs}")

                model = SequenceLSTM(
                    input_dim=23,
                    hidden_size=hidden_size,
                    num_layers=1,
                    comparison_dim=50,
                    dropout=0.3,
                )

                model, history = train_sequence_model(
                    model, train_samples, val_samples,
                    lr=lr, epochs=epochs, verbose=False,
                )

                train_eval = evaluate_sequence_model(model, train_samples)
                val_eval = evaluate_sequence_model(model, val_samples)
                test_eval = evaluate_sequence_model(model, test_samples)

                result = TuningResult(
                    hyperparams={"lr": lr, "hidden_size": hidden_size, "epochs": epochs},
                    train_acc=train_eval["accuracy"],
                    val_acc=val_eval["accuracy"],
                    test_acc=test_eval["accuracy"],
                )
                results.append(result)
                print(f"    train={result.train_acc:.1%}, val={result.val_acc:.1%}, test={result.test_acc:.1%}")

    results.sort(key=lambda r: r.val_acc, reverse=True)
    return results
