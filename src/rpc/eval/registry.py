"""Scorer registry: other agents add their Scorer by name here."""

from __future__ import annotations

from typing import Callable

from src.rpc.eval.protocol import Scorer

_FACTORIES: dict[str, Callable[[], Scorer]] = {}


def register_scorer(name: str, factory: Callable[[], Scorer]) -> None:
    _FACTORIES[name] = factory


def get_scorer(name: str) -> Scorer:
    if name not in _FACTORIES:
        raise KeyError(f"unknown scorer '{name}'; registered: {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


def list_scorers() -> list[str]:
    return sorted(_FACTORIES)


def register_builtin_baselines() -> None:
    """Import baselines (which self-register) exactly once."""
    from src.rpc.models import baselines as _b  # noqa: F401
