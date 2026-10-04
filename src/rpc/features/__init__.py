"""Point-in-time feature pipeline, text extraction, graph features (simulation-only).

NOTE: the label module (src/rpc/features/labels.py) is intentionally NOT
referenced here. Feature code must never depend on labels (future outcomes);
use the labels module directly where needed.
"""

from src.rpc.features.features import build_features, build_training_table
from src.rpc.features.source import DataFrameEventSource, EventSource, ParquetEventSource
from src.rpc.features.spec import build_registry, feature_names

__all__ = [
    "build_features",
    "build_training_table",
    "build_registry",
    "feature_names",
    "EventSource",
    "ParquetEventSource",
    "DataFrameEventSource",
]
