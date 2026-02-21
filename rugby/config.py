"""TOML-based configuration for rugby prediction."""

import tomllib
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class TrainingConfig:
    epochs: int = 311
    batch_size: int = 128
    learning_rate: float = 5.9e-05
    weight_decay: float = 0.001
    dropout: float = 0.49
    early_stopping_patience: int = 50


@dataclass
class ModelConfig:
    d_model: int = 64
    n_encoder_layers: int = 2
    n_cross_attn_layers: int = 1
    n_heads: int = 4
    max_history: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class LossConfig:
    win_weight: float = 1.14
    score_weight: float = 1.29


@dataclass
class DataConfig:
    competition: str = "super-rugby"
    database_path: str = "data/rugby.db"
    model_path: str = "model/rugby_model"


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: Path, competition: str = "super-rugby") -> "Config":
        """Load configuration for a competition from a TOML file.

        The file is keyed by competition name, e.g. [super-rugby.training].
        """
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        if competition not in raw:
            available = [k for k in raw if isinstance(raw[k], dict)]
            raise ValueError(
                f"Unknown competition '{competition}'. "
                f"Available: {', '.join(available)}"
            )

        section = raw[competition]
        config = cls()
        config.data.competition = competition

        if "training" in section:
            t = section["training"]
            config.training = TrainingConfig(
                epochs=t.get("epochs", config.training.epochs),
                batch_size=t.get("batch_size", config.training.batch_size),
                learning_rate=t.get("learning_rate", config.training.learning_rate),
                weight_decay=t.get("weight_decay", config.training.weight_decay),
                dropout=t.get("dropout", config.training.dropout),
                early_stopping_patience=t.get("early_stopping_patience", config.training.early_stopping_patience),
            )

        if "model" in section:
            m = section["model"]
            config.model = ModelConfig(
                d_model=m.get("d_model", config.model.d_model),
                n_encoder_layers=m.get("n_encoder_layers", config.model.n_encoder_layers),
                n_cross_attn_layers=m.get("n_cross_attn_layers", config.model.n_cross_attn_layers),
                n_heads=m.get("n_heads", config.model.n_heads),
                max_history=m.get("max_history", config.model.max_history),
                hidden_dims=m.get("hidden_dims", config.model.hidden_dims),
            )

        if "loss" in section:
            lo = section["loss"]
            config.loss = LossConfig(
                win_weight=lo.get("win_weight", config.loss.win_weight),
                score_weight=lo.get("score_weight", config.loss.score_weight),
            )

        if "data" in section:
            d = section["data"]
            config.data = DataConfig(
                competition=competition,
                database_path=d.get("database_path", config.data.database_path),
                model_path=d.get("model_path", config.data.model_path),
            )

        return config

    @classmethod
    def default(cls) -> "Config":
        """Return default configuration matching config.toml."""
        return cls()
