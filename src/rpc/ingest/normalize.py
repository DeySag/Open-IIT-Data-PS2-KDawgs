"""Contact-point normalisation and hashing (SIMULATION-ONLY pipeline code).

Raw phone numbers / addresses are normalised, then sha256-hashed (truncated to
16 hex chars) to produce the canonical ``contact_point_ref``. Raw values must
never be stored or logged after hashing -- see :func:`redact_record`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

HASH_CHARS = 16
REDACTED = "[redacted-pii]"

_WS_RE = re.compile(r"\s+")
_SEPARATORS_RE = re.compile(r"[\s\-.()]+")
_CC_PREFIX_RE = re.compile(r"^(?:91|0)(?=\d{10}$)")


def normalize_phone(value: str) -> str:
    """Normalise an Indian phone number to bare 10-digit form.

    Strips separators and a leading ``+``, then a leading ``91`` country code
    or trunk ``0`` when exactly 10 subscriber digits remain. Anything else is
    left intact so malformed inputs still hash deterministically.
    """
    d = _SEPARATORS_RE.sub("", value.strip())
    if d.startswith("+"):
        d = d[1:]
    return _CC_PREFIX_RE.sub("", d)


def normalize_address(value: str) -> str:
    """Minimal address normalisation: lowercase, collapse whitespace, strip."""
    return _WS_RE.sub(" ", value.strip().lower())


def hash_normalized(normalized: str) -> str:
    """sha256 of the normalised value, truncated to 16 hex chars."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:HASH_CHARS]


def _hash_unique(normalized: pd.Series) -> pd.Series:
    """Digest each distinct value once, then map back (vectorised lookup).

    Contact values repeat heavily across events; hashing uniques keeps output
    identical while cutting digest calls from rows to distinct values.
    """
    uniques = normalized.unique()
    table = {value: hash_normalized(value) for value in uniques}
    return normalized.map(table)


def hash_phone_series(values: pd.Series) -> pd.Series:
    """Vectorised phone normalise+hash using pandas string ops (C-level).

    Digests run once per distinct normalised value, mapped back vectorised
    (no per-row Python branching over the batch).
    """
    s = values.astype("string").fillna("")
    d = s.str.replace(_SEPARATORS_RE, "", regex=True)
    d = d.str.replace(r"^\+", "", regex=True)
    d = d.str.replace(_CC_PREFIX_RE, "", regex=True)
    return _hash_unique(d)


def hash_address_series(values: pd.Series) -> pd.Series:
    """Vectorised address normalise+hash."""
    d = values.astype("string").fillna("").str.strip().str.lower()
    d = d.str.replace(r"\s+", " ", regex=True)
    return _hash_unique(d)


def passthrough_hash_series(values: pd.Series) -> pd.Series:
    """Pass through values that are already hashes (e.g. CN-provided IDs)."""
    return values.astype("string").fillna("").astype(str)


def redact_record(record: dict[str, Any], raw_fields: list[str]) -> dict[str, Any]:
    """Return a copy of ``record`` with raw contact-point fields redacted.

    Used for dead-letter storage and logging so raw PII never lands in the
    store or logs. Nested dotted paths (``contact.mobile``) are redacted too.
    """
    redacted = dict(record)
    for path in raw_fields:
        if "." not in path:
            if path in redacted:
                redacted[path] = REDACTED
        else:
            head, rest = path.split(".", 1)
            nested = redacted.get(head)
            if isinstance(nested, dict):
                nested = dict(nested)
                _redact_nested(nested, rest)
                redacted[head] = nested
    return redacted


def _redact_nested(node: dict[str, Any], rest: str) -> None:
    if "." not in rest:
        if rest in node:
            node[rest] = REDACTED
        return
    head, tail = rest.split(".", 1)
    child = node.get(head)
    if isinstance(child, dict):
        _redact_nested(child, tail)
