"""Evaluation harness: time-aware splits, labels, metrics, reports (simulation-only).

Hidden ground truth lives in eval.truth -- never in features.
"""

from src.rpc.eval.protocol import Scorer
from src.rpc.eval.registry import get_scorer, list_scorers, register_scorer
from src.rpc.eval.truth import join_truth, load_ground_truth

__all__ = [
    "Scorer",
    "get_scorer",
    "join_truth",
    "list_scorers",
    "load_ground_truth",
    "register_scorer",
]
