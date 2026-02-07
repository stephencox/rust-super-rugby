"""Rugby match prediction package."""

from .data import Database, Match, Team, load_matches, load_teams
from .features import (
    FeatureBuilder,
    FeatureNormalizer,
    TeamStats,
    SequenceFeatureBuilder,
    SequenceDataSample,
    SequenceNormalizer,
    SEQUENCE_FEATURE_DIM,
)
from .models import (
    WinClassifier,
    ScoreRegressor,
    MatchPredictor,
    SequenceLSTM,
    MATCH_FEATURE_DIM,
)
from .training import (
    train_win_model,
    train_score_model,
    train_match_predictor,
    evaluate_win_model,
    evaluate_score_model,
    evaluate_match_predictor,
)
from .config import Config

__all__ = [
    'Config',
    'Database',
    'Match',
    'Team',
    'load_matches',
    'load_teams',
    'FeatureBuilder',
    'FeatureNormalizer',
    'TeamStats',
    'SequenceFeatureBuilder',
    'SequenceDataSample',
    'SequenceNormalizer',
    'SEQUENCE_FEATURE_DIM',
    'WinClassifier',
    'ScoreRegressor',
    'MatchPredictor',
    'SequenceLSTM',
    'MATCH_FEATURE_DIM',
    'train_win_model',
    'train_score_model',
    'train_match_predictor',
    'evaluate_win_model',
    'evaluate_score_model',
    'evaluate_match_predictor',
]
