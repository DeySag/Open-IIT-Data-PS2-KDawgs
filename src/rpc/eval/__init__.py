"""Evaluation harness: time-aware splits, labels, metrics, reports."""

from src.rpc.eval.protocol import Scorer
from src.rpc.eval.registry import get_scorer, list_scorers, register_scorer

__all__ = [
    "Scorer",
    "get_scorer",
    "list_scorers",
    "register_scorer",
]
