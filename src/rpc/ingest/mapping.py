"""Mapping-driven translation of source records into the canonical envelope.

Mapping-driven translation of source records into the canonical envelope. All operations are column-vectorised (pandas);
the only linear passes are single ``Series.map`` calls over individual columns
(nested-path extraction, contact hashing, event-id derivation) -- never a
per-row loop with branching logic.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.rpc.contracts import Disposition, EventType, NetworkResponse
from src.rpc.ingest.normalize import (
    hash_address_series,
    hash_id_series,
    hash_phone_series,
    passthrough_hash_series,
    resolve_pepper,
)

MAPPING_DIR = Path("configs/field_mappings")

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = (
    "event_id",
    "event_type",
    "lender_id",
    "borrower_id",
    "account_id",
    "contact_point_ref",
    "occurred_at",
    "received_at",
)

# Hidden ground-truth columns (e.g. verification annotations joined by
# mistake). Must never reach the store or features: dropped on load if
# present in input.
HIDDEN_GROUND_TRUTH_COLUMNS = frozenset(
    {
        "true_state",
        "borrower_avoiding",
        "shared_reason",
        "avoiding",
        "avoiding_initial",
        "true_label",
        "is_avoiding",
    }
)

EVENT_TYPES = {e.value for e in EventType}
NETWORK_RESPONSES = {e.value for e in NetworkResponse}
DISPOSITIONS = {e.value for e in Disposition}

# Payload keys required per event type for light (vectorised) validation.
# Full schema validation happens via pydantic on rejected rows + sample.
REQUIRED_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "dial_attempt": ("network_response", "ring_seconds"),
    "disposition": ("disposition",),
}

_MISSING: Any = object()  # sentinel for absent source column


def load_mapping(source: str) -> dict[str, Any] | None:
    """Load the YAML mapping for ``source``; None when no mapping exists."""
    path = MAPPING_DIR / f"{source}.yaml"
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


class MappingSpecError(ValueError):
    """A field-mapping spec is malformed (carries the offending spec)."""


def _resolve_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"field": spec}
    if isinstance(spec, dict):
        return dict(spec)
    raise MappingSpecError(spec)


def _dig(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    if isinstance(value, str):
        try:
            return json.loads(value).get(key)
        except (ValueError, AttributeError):
            return None
    return None


def extract_column(df: pd.DataFrame, path: str) -> pd.Series | Any:
    """Extract a source column; supports dotted nested paths. _MISSING if absent."""
    if path in df.columns:
        return df[path]
    parts = path.split(".")
    if parts[0] not in df.columns:
        return _MISSING
    col = df[parts[0]]
    for part in parts[1:]:
        col = col.map(lambda v, p=part: _dig(v, p))
    return col


def _as_string(values: pd.Series) -> pd.Series:
    """Values as pandas string dtype without copying when already string."""
    if isinstance(values.dtype, pd.StringDtype):
        return values
    return values.astype("string")


def _is_present(values: pd.Series) -> pd.Series:
    """Non-null and non-blank (for strings)."""
    not_null = values.notna()
    as_str = _as_string(values)
    blank = as_str.str.strip() == ""
    return not_null & ~blank.fillna(True)


def parse_timestamp_column(values: pd.Series, fmt: str, tz: str | None) -> pd.Series:
    """Parse a timestamp column to UTC datetime64; unparseable -> NaT (vectorised)."""
    if fmt == "epoch_ms":
        num = pd.to_numeric(values, errors="coerce")
        return pd.to_datetime(num, unit="ms", utc=True, errors="coerce")
    if fmt == "epoch_s":
        num = pd.to_numeric(values, errors="coerce")
        return pd.to_datetime(num, unit="s", utc=True, errors="coerce")
    if fmt == "iso":
        # Fast path for uniform ISO-8601, fallback to mixed for the rest.
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")
        retry_mask = parsed.isna() & values.notna()
        if retry_mask.any():
            retry = pd.to_datetime(values[retry_mask], utc=True, errors="coerce", format="mixed")
            parsed = parsed.copy()
            parsed[retry_mask] = retry
        return parsed
    # strptime format; localise naive results to the source zone, then UTC.
    parsed = pd.to_datetime(values, format=fmt, errors="coerce")
    try:
        if parsed.dt.tz is None and tz:
            parsed = parsed.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    except (TypeError, ValueError):
        return pd.to_datetime(pd.Series([pd.NaT] * len(values)), utc=True)
    return parsed.dt.tz_convert("UTC")


def _canonical_event_id(
    values: pd.Series, source: str, derive: bool, suffix: str = ""
) -> pd.Series:
    """UUID strings; unparseable (or derive=True) -> deterministic uuid5.

    ``suffix`` disambiguates companion events derived from the same raw id
    (e.g. dial vs disposition rows from one attempt row).
    """
    as_str = _as_string(values)
    if suffix:
        # String-dtype concat propagates NA, so absent ids stay absent.
        as_str = as_str + suffix
    if derive:
        return _map_unique(as_str, lambda x: _derive_uuid(x, source))
    # Fast vectorised path: 32-hex source ids hyphenated in bulk.
    bare32 = as_str.str.match(r"^[0-9a-fA-F]{32}$").fillna(False)
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    if bare32.any():
        b = as_str[bare32]
        out[bare32] = (
            b.str.slice(0, 8)
            + "-"
            + b.str.slice(8, 12)
            + "-"
            + b.str.slice(12, 16)
            + "-"
            + b.str.slice(16, 20)
            + "-"
            + b.str.slice(20, 32)
        )
    rest = as_str[~bare32]
    if len(rest):
        out[~bare32] = _as_string(_map_unique(rest, lambda x: _coerce_uuid(x, source)))
    return out


def _map_unique(values: pd.Series, func: Callable[[Any], Any]) -> pd.Series:
    """Apply ``func`` once per distinct value, mapped back vectorised.

    Missing keys are excluded from the lookup table so absent values stay
    absent (a dict lookup would otherwise resolve the ``pd.NA`` singleton
    and manufacture a hash of nothing).
    """
    table: dict[Any, Any] = {}
    for value in values.unique():
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        table[value] = func(value)
    return values.map(table)


def _derive_uuid(raw: Any, source: str) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{text}"))


def _coerce_uuid(raw: Any, source: str) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{text}"))


def _map_scalar_fields(
    df: pd.DataFrame, fields: dict[str, Any], source: str
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Map envelope fields except contact ref and timestamps (vectorised)."""
    out: dict[str, pd.Series] = {}
    unknown_enum = pd.Series(False, index=df.index)
    for canonical in REQUIRED_FIELDS:
        if canonical in ("contact_point_ref", "occurred_at", "received_at"):
            continue
        spec = _resolve_spec(fields.get(canonical, {}))
        out[canonical] = _map_one_field(df, spec)
        if "map" in spec and "field" in spec:
            col = extract_column(df, spec["field"])
            if col is not _MISSING:
                col = _as_string(col)
                unknown_enum = unknown_enum | (_is_present(col) & out[canonical].isna())
    # event_id derivation (uuid5 when the source id is not a UUID).
    # Supports {field, uuid5, suffix?} and {fields: [...], uuid5, suffix?};
    # the latter joins present values with "|" (any missing part -> missing).
    id_spec = _resolve_spec(fields.get("event_id", {}))
    if "fields" in id_spec:
        parts = [extract_column(df, f) for f in id_spec["fields"]]
        if any(col is _MISSING for col in parts):
            id_values: pd.Series = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            id_values = _as_string(parts[0])
            for col in parts[1:]:
                id_values = id_values + "|" + _as_string(col)
    else:
        id_values = out["event_id"]
    out["event_id"] = _canonical_event_id(
        id_values,
        source,
        derive=bool(id_spec.get("uuid5", False)),
        suffix=str(id_spec.get("suffix", "")),
    ).astype("string")
    return out, unknown_enum


def _map_one_field(df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    """Map a single envelope field to a string column."""
    na = pd.Series(pd.NA, index=df.index, dtype="string")
    if "const" in spec:
        return pd.Series(spec["const"], index=df.index, dtype="string")
    if "field" not in spec:
        return na
    col = extract_column(df, spec["field"])
    if col is _MISSING:
        return na
    col = _as_string(col)
    if "map" in spec:
        col = _as_string(col.map(dict(spec["map"])))
    return col


def _map_contact(df: pd.DataFrame, mapping: dict[str, Any]) -> pd.Series:
    """Normalise+hash the raw contact field, or pass through an existing hash.

    ``contact_point.pepper_env`` names the env var holding the hash pepper
    (official mappings declare ``CN_HASH_PEPPER``). When declared but unset,
    hashing falls back to legacy sha256 and logs once -- loud enough to catch
    in reviews, quiet enough for simulator fixtures that declare no pepper.
    """
    na = pd.Series(pd.NA, index=df.index, dtype="string")
    cp_spec = mapping.get("contact_point", {})
    raw_field = cp_spec.get("raw_field", "")
    raw_col = extract_column(df, raw_field) if raw_field else _MISSING
    if raw_col is _MISSING:
        return na
    pepper = None
    pepper_env = cp_spec.get("pepper_env")
    if pepper_env:
        pepper = resolve_pepper(str(pepper_env))
        if pepper is None:
            logger.warning(
                "contact hashing without pepper: %s unset; refs are "
                "unpeppered sha256 (rotate by re-ingesting with it set)",
                pepper_env,
            )
    kind = cp_spec.get("kind", "phone")
    if kind == "phone":
        ref = hash_phone_series(raw_col, pepper).astype("string")
    elif kind == "address":
        ref = hash_address_series(raw_col, pepper).astype("string")
    elif kind == "id":
        ref = hash_id_series(raw_col, pepper).astype("string")
    elif kind == "hash_passthrough":
        ref = passthrough_hash_series(raw_col).astype("string")
    else:
        return na
    # Blank raw contact values carry no information -> missing, not a hash.
    return ref.mask(~_is_present(_as_string(raw_col)), pd.NA)


def _map_timestamps(
    df: pd.DataFrame, fields: dict[str, Any]
) -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    """Parse occurred_at/received_at to UTC; track absent vs unparseable."""
    out: dict[str, pd.Series] = {}
    bad_ts = pd.Series(False, index=df.index)
    ts_absent = pd.Series(False, index=df.index)
    for ts_col in ("occurred_at", "received_at"):
        spec = _resolve_spec(fields.get(ts_col, {}))
        if "const" in spec:
            out[ts_col] = parse_timestamp_column(
                pd.Series(spec["const"], index=df.index),
                str(spec.get("format", "iso")),
                spec.get("tz"),
            )
            continue
        col = extract_column(df, spec["field"]) if "field" in spec else _MISSING
        if col is _MISSING:
            out[ts_col] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
            ts_absent = ts_absent | True
        else:
            out[ts_col] = parse_timestamp_column(
                col, str(spec.get("format", "iso")), spec.get("tz")
            )
            bad_ts = bad_ts | out[ts_col].isna()
    return out, bad_ts, ts_absent


def _build_payload(
    df: pd.DataFrame,
    canonical: pd.DataFrame,
    mapping: dict[str, Any],
    unknown_enum: pd.Series,
) -> tuple[pd.Series, pd.Series, str | None, dict[str, pd.Series]]:
    """Build the payload JSON column (single linear pass over payload columns).

    Returns (payload, unknown_enum, json_field, payload_cols); payload_cols is
    empty for passthrough sources and reused for required-key checks so the
    JSON is never re-parsed for validation.
    """
    payload_json_field = mapping.get("payload_json_field")
    if payload_json_field:
        col = extract_column(df, payload_json_field)
        if col is _MISSING:
            payload = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            payload = col.astype("string")
        return payload, unknown_enum, payload_json_field, {}
    payload_cols: dict[str, pd.Series] = {}
    for key, raw_spec in mapping.get("payload", {}).items():
        spec = _resolve_spec(raw_spec)
        payload_cols[key] = _map_payload_key(df, canonical, spec)
        if "default" in spec:
            payload_cols[key] = payload_cols[key].fillna(spec["default"])
    if "channel" in payload_cols:
        payload_cols["channel"] = payload_cols["channel"].fillna("voice_bot")
    keys = list(payload_cols)
    cols = [payload_cols[k].tolist() for k in keys]
    payload = pd.Series(
        [_dump_payload(keys, vals) for vals in zip(*cols)],
        index=df.index,
        dtype="string",
    )
    return payload, unknown_enum, None, payload_cols


def _dump_payload(keys: list[str], vals: tuple[Any, ...]) -> str:
    """Serialise one payload dict, dropping null/NaN (single-row helper)."""
    record = {}
    for key, value in zip(keys, vals):
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and value != value:  # NaN
            continue
        record[key] = value
    return json.dumps(record, sort_keys=True)


def _map_payload_key(df: pd.DataFrame, canonical: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    """Map one payload key to a column (vectorised)."""
    if "const" in spec:
        return pd.Series(spec["const"], index=df.index)
    if spec.get("contact_ref"):
        # Reference to the already-computed contact_point_ref hash (never raw PII).
        return canonical["contact_point_ref"]
    if "field" not in spec:
        return pd.Series(None, index=df.index)
    col = extract_column(df, spec["field"])
    if col is _MISSING:
        return pd.Series(None, index=df.index)
    if "map" in spec:
        col = col.map(dict(spec["map"]))
    if spec.get("type") in ("float", "int"):
        num = pd.to_numeric(col, errors="coerce")
        return num if spec["type"] == "float" else num.astype("Int64")
    if "format" in spec:
        # Timestamp-valued payload key: parse to UTC, emit ISO-8601 strings.
        parsed = parse_timestamp_column(col, str(spec["format"]), spec.get("tz"))
        return parsed.map(lambda v: v.isoformat() if not pd.isna(v) else None)
    return col


def _payload_unknown_enum(
    df: pd.DataFrame,
    canonical: pd.DataFrame,
    mapping: dict[str, Any],
    unknown_enum: pd.Series,
) -> pd.Series:
    """Flag rows whose payload value map failed on a relevant event type."""
    for key, raw_spec in mapping.get("payload", {}).items():
        spec = _resolve_spec(raw_spec)
        if "map" not in spec or "field" not in spec:
            continue
        col = extract_column(df, spec["field"])
        if col is _MISSING:
            continue
        relevant = canonical["event_type"].isin(["dial_attempt", "disposition"])
        mapped = col.map(dict(spec["map"]))
        unknown_enum = unknown_enum | (relevant & _is_present(col) & mapped.isna())
    return unknown_enum


def _assign_reasons(
    df: pd.DataFrame,
    canonical: pd.DataFrame,
    mapping: dict[str, Any],
    unknown_enum: pd.Series,
    bad_ts: pd.Series,
    ts_absent: pd.Series,
    payload_json_field: str | None,
    payload_cols: dict[str, pd.Series],
) -> pd.Series:
    """Vectorised validation with reason precedence (never raises).

    Precedence: missing_required_field (top-level, incl. absent timestamp
    columns and missing payload keys) > unknown_enum > bad_timestamp
    (present-but-unparseable value) > time_inversion. unknown_enum outranks
    missing payload keys because an unmapped code is what empties the field.
    """
    missing = pd.Series(False, index=df.index)
    present: dict[str, pd.Series] = {}
    for field in REQUIRED_FIELDS:
        if field in ("occurred_at", "received_at"):
            continue  # covered by ts_absent / bad_ts below
        present[field] = _is_present(canonical[field])
        missing = missing | ~present[field]
    bad_event_type = ~canonical["event_type"].isin(list(EVENT_TYPES)) & present["event_type"]
    unknown_enum = unknown_enum | bad_event_type
    missing_payload = _missing_payload_mask(
        df, canonical, mapping, payload_json_field, payload_cols
    )

    tolerance_s = float(mapping.get("validation", {}).get("clock_skew_tolerance_seconds", 300))
    occurred = canonical["occurred_at"]
    received = canonical["received_at"]
    both_present = occurred.notna() & received.notna()
    time_inversion = both_present & (
        (received - occurred) < pd.to_timedelta(-tolerance_s, unit="s")
    )

    reason = pd.Series(None, index=df.index, dtype="string")
    reason = reason.mask(missing | ts_absent, "missing_required_field")
    reason = reason.mask(reason.isna() & unknown_enum, "unknown_enum")
    reason = reason.mask(reason.isna() & missing_payload, "missing_required_field")
    bad_ts_value = bad_ts.fillna(False) & ~ts_absent
    reason = reason.mask(reason.isna() & bad_ts_value, "bad_timestamp")
    reason = reason.mask(reason.isna() & time_inversion.fillna(False), "time_inversion")
    return reason


def _missing_payload_mask(
    df: pd.DataFrame,
    canonical: pd.DataFrame,
    mapping: dict[str, Any],
    payload_json_field: str | None,
    payload_cols: dict[str, pd.Series],
) -> pd.Series:
    """Rows whose payload lacks keys required for their event type.

    Built payloads are checked from their columns (no JSON re-parsing);
    passthrough JSON payloads check presence only (schema via pydantic sample).
    """
    missing_payload = pd.Series(False, index=df.index)
    if payload_json_field or not payload_cols:
        if payload_json_field:
            return missing_payload | ~_is_present(canonical["payload"])
        return missing_payload
    for event_type, keys in REQUIRED_PAYLOAD_KEYS.items():
        rows = canonical["event_type"] == event_type
        if not rows.any():
            continue
        keys_ok = pd.Series(True, index=df.index)
        for key in keys:
            col = payload_cols.get(key)
            if col is None:
                keys_ok = pd.Series(False, index=df.index)
                break
            keys_ok = keys_ok & _col_present(col)
        missing_payload = missing_payload | (rows & ~keys_ok)
    return missing_payload


def _col_present(col: pd.Series) -> pd.Series:
    """Non-null (and non-blank for strings) elementwise, without parsing."""
    if pd.api.types.is_numeric_dtype(col.dtype):
        return col.notna()
    return _is_present(col)


def apply_mapping(df: pd.DataFrame, mapping: dict[str, Any], source: str) -> pd.DataFrame:
    """Translate ``df`` into canonical-envelope columns (vectorised).

    Returns a DataFrame with canonical columns plus ``payload`` (JSON string)
    and ``_reason`` (None for valid rows, else a reason code). Never
    raises for data problems -- every failure becomes a rejection reason.
    """
    fields = mapping.get("fields", {})
    out, unknown_enum = _map_scalar_fields(df, fields, source)
    out["contact_point_ref"] = _map_contact(df, mapping)
    timestamps, bad_ts, ts_absent = _map_timestamps(df, fields)
    out.update(timestamps)
    canonical = pd.DataFrame(out)
    payload, unknown_enum, payload_json_field, payload_cols = _build_payload(
        df, canonical, mapping, unknown_enum
    )
    canonical["payload"] = payload
    unknown_enum = _payload_unknown_enum(df, canonical, mapping, unknown_enum)
    canonical["_reason"] = _assign_reasons(
        df,
        canonical,
        mapping,
        unknown_enum,
        bad_ts,
        ts_absent,
        payload_json_field,
        payload_cols,
    )
    canonical["_source_index"] = df.index
    return canonical


def _parse_payload_or_none(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
