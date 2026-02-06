"""TOML-based configuration for rugby prediction."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.0005
    weight_decay: float = 0.001
    dropout: float = 0.3
    early_stopping_patience: int = 50


@dataclass
class ModelConfig:
    d_model: int = 64
    n_encoder_layers: int = 2
    n_cross_attn_layers: int = 1
    n_heads: int = 4
    max_history: int = 10


@dataclass
class LossConfig:
    win_weight: float = 2.0
    score_weight: float = 0.3


@dataclass
class DataConfig:
    database_path: str = "data/rugby.db"
    model_path: str = "model/rugby_model"


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        config = cls()

        if "training" in raw:
            t = raw["training"]
            config.training = TrainingConfig(
                epochs=t.get("epochs", config.training.epochs),
                batch_size=t.get("batch_size", config.training.batch_size),
                learning_rate=t.get("learning_rate", config.training.learning_rate),
                weight_decay=t.get("weight_decay", config.training.weight_decay),
                dropout=t.get("dropout", config.training.dropout),
                early_stopping_patience=t.get("early_stopping_patience", config.training.early_stopping_patience),
            )

        if "model" in raw:
            m = raw["model"]
            config.model = ModelConfig(
                d_model=m.get("d_model", config.model.d_model),
                n_encoder_layers=m.get("n_encoder_layers", config.model.n_encoder_layers),
                n_cross_attn_layers=m.get("n_cross_attn_layers", config.model.n_cross_attn_layers),
                n_heads=m.get("n_heads", config.model.n_heads),
                max_history=m.get("max_history", config.model.max_history),
            )

        if "loss" in raw:
            lo = raw["loss"]
            config.loss = LossConfig(
                win_weight=lo.get("win_weight", config.loss.win_weight),
                score_weight=lo.get("score_weight", config.loss.score_weight),
            )

        if "data" in raw:
            d = raw["data"]
            config.data = DataConfig(
                database_path=d.get("database_path", config.data.database_path),
                model_path=d.get("model_path", config.data.model_path),
            )

        return config

    @classmethod
    def default(cls) -> "Config":
        """Return default configuration matching config.toml."""
        return cls()
