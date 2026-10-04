"""Evaluation helpers (simulation-only). Hidden ground truth lives here, never in features."""

from src.rpc.eval.truth import join_truth, load_ground_truth

__all__ = ["join_truth", "load_ground_truth"]
