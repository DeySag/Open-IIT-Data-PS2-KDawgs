"""Feature registry (simulation-only).

Single source of truth for every feature column emitted by
:func:`src.rpc.features.features.build_features`. The registry drives:

* output column order and dtypes (tests assert registry == output exactly),
* ``docs/features.md`` generation (``render_registry_markdown`` /
  ``update_features_doc``),
* ``feature_snapshot_id`` hashing (via the raw config text).

Windows, thresholds, slots and holidays live in ``configs/features.yaml``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
TEXT_PATTERNS_PATH = REPO_ROOT / "configs" / "text_patterns.yaml"
FEATURES_DOC_PATH = REPO_ROOT / "docs" / "features.md"

REGISTRY_START = "<!-- REGISTRY:START -->"
REGISTRY_END = "<!-- REGISTRY:END -->"

# Output key and metadata columns (not features, not in the registry).
KEY_COLUMNS = ["lender_id", "borrower_id", "account_id", "contact_point_ref", "as_of"]
META_COLUMNS = ["feature_snapshot_id", "event_watermark"]


@dataclass(frozen=True)
class Feature:
    """One feature column definition."""

    name: str
    group: str
    dtype: str  # pandas nullable dtype: Int64, Float64, boolean, string
    source_event_types: tuple[str, ...]
    window: int | None  # recency window in days, None = all-time/as-of scalar
    null_semantics: str
    description: str


@lru_cache(maxsize=4)
def load_feature_config(path: str | Path = FEATURES_CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def config_text_for_hash(
    path: str | Path = FEATURES_CONFIG_PATH,
    text_path: str | Path = TEXT_PATTERNS_PATH,
) -> str:
    """Raw config text used for feature_snapshot_id (any change -> new id)."""
    return Path(path).read_text() + "\n---\n" + Path(text_path).read_text()


def snapshot_id(config_text: str, as_of: str, watermark: str) -> str:
    digest = hashlib.sha256(
        f"{config_text}\n{as_of}\n{watermark}".encode()
    ).hexdigest()
    return f"fs_{digest[:16]}"


def window_suffix(window_days: int) -> str:
    return f"_{window_days}d"


def build_registry(
    config: dict[str, Any] | None = None, *, include_agent_feature: bool = False
) -> list[Feature]:
    """Build the full feature registry from config.

    ``include_agent_feature`` mirrors the conditional agent-level feature:
    it is True only when an ``agent_id`` is observed in disposition payloads.
    ``build_features`` uses this same function, so registry and output can
    never drift (enforced by tests).
    """
    config = config or load_feature_config()
    fcfg = config["features"]
    windows: list[int] = list(fcfg["windows_days"])
    responses: list[str] = list(fcfg["response_values"])
    dispositions: list[str] = list(fcfg["disposition_values"])
    who_values: list[str] = list(fcfg["bot_who_values"])
    visit_outcomes: list[str] = list(fcfg["visit_outcomes"])

    reg: list[Feature] = []

    # ---- Telephony, per window -------------------------------------------
    for w in windows:
        sfx = window_suffix(w)
        reg += [
            Feature(f"n_attempts{sfx}", "telephony", "Int64", ("dial_attempt",), w,
                    "0 when no attempts in window.",
                    f"Dial attempts with occurred_at in the last {w}d."),
            Feature(f"n_answered{sfx}", "telephony", "Int64", ("dial_attempt",), w,
                    "0 when no attempts in window.",
                    f"Attempts with network_response=answered in the last {w}d."),
            Feature(f"answer_rate{sfx}", "telephony", "Float64", ("dial_attempt",), w,
                    "null when no attempts in window (never 0-imputed).",
                    f"n_answered / n_attempts over the last {w}d."),
            Feature(f"n_immediate_hangup{sfx}", "telephony", "Int64", ("dial_attempt",), w,
                    "0 when no attempts in window.",
                    f"Attempts with network_response=immediate_hangup in {w}d."),
            Feature(f"hangup_rate{sfx}", "telephony", "Float64", ("dial_attempt",), w,
                    "null when no attempts in window.",
                    f"n_immediate_hangup / n_attempts over {w}d."),
        ]
        for resp in responses:
            reg.append(Feature(
                f"n_{resp}{sfx}", "telephony", "Int64", ("dial_attempt",), w,
                "0 when no attempts in window.",
                f"Attempts with network_response={resp} in the last {w}d."))
        for slot in ("morning", "afternoon", "evening"):
            reg.append(Feature(
                f"n_attempts_{slot}{sfx}", "telephony", "Int64", ("dial_attempt",), w,
                "0 when no attempts in window.",
                f"Attempts placed in the {slot} slot (IST) in {w}d."))
        for slot in ("morning", "afternoon", "evening"):
            reg.append(Feature(
                f"answer_rate_{slot}{sfx}", "telephony", "Float64", ("dial_attempt",), w,
                "null when no attempts in that slot and window.",
                f"Answered share within the {slot} slot (IST) over {w}d."))
        reg += [
            Feature(f"weekend_attempt_share{sfx}", "telephony", "Float64", ("dial_attempt",), w,
                    "null when no attempts in window.",
                    f"Share of attempts on Sat/Sun (IST) over {w}d."),
            Feature(f"ring_seconds_mean{sfx}", "telephony", "Float64", ("dial_attempt",), w,
                    "null when no attempts in window.",
                    f"Mean ring_seconds over {w}d."),
            Feature(f"ring_seconds_std{sfx}", "telephony", "Float64", ("dial_attempt",), w,
                    "null with fewer than 2 attempts in window.",
                    f"Sample std of ring_seconds over {w}d."),
            Feature(f"short_ring_rate{sfx}", "telephony", "Float64", ("dial_attempt",), w,
                    "null when no attempts in window.",
                    "Share of attempts with ring_seconds below configs short_ring_seconds."),
        ]

    # ---- Telephony scalars -------------------------------------------------
    reg += [
        Feature("last_response_type", "telephony", "string", ("dial_attempt",), None,
                "null when never attempted.",
                "Most recent network_response (by occurred_at)."),
        Feature("consecutive_failures", "telephony", "Int64", ("dial_attempt",), None,
                "null when never attempted; 0 when the last attempt was answered.",
                "Trailing run of non-answered responses since the last answer."),
        Feature("consecutive_same_response", "telephony", "Int64", ("dial_attempt",), None,
                "null when never attempted.",
                "Trailing run length of the latest network_response value."),
        Feature("days_since_first_attempt", "telephony", "Int64", ("dial_attempt",), None,
                "null when never attempted.",
                "Days from first visible attempt to as_of (date-based, UTC)."),
        Feature("days_since_last_attempt", "telephony", "Int64", ("dial_attempt",), None,
                "null when never attempted.",
                "Days from last visible attempt to as_of."),
        Feature("days_since_last_answer", "telephony", "Int64", ("dial_attempt",), None,
                "null when never answered.",
                "Days from last answered attempt to as_of."),
        Feature("days_since_last_rpc", "telephony", "Int64", ("disposition",), None,
                "null when no RPC disposition is visible.",
                "Days from last RPC disposition to as_of."),
        Feature("mean_gap_between_attempts_days", "telephony", "Float64", ("dial_attempt",), None,
                "null with fewer than 2 attempts.",
                "Mean gap in days between consecutive visible attempts."),
        Feature("system_fail_rate_on_last_attempt_day", "telephony", "Float64", ("dial_attempt",), None,
                "null when never attempted.",
                "Portfolio-wide failure share on the IST date of this contact "
                "point's last attempt, so models can discount dialer outages."),
    ]

    # ---- Dispositions ------------------------------------------------------
    for disp in dispositions:
        reg.append(Feature(
            f"n_{disp}", "disposition", "Int64", ("disposition",), None,
            "0 when no visible dispositions.",
            f"Visible dispositions with value {disp} (all-time)."))
    reg += [
        Feature("wrong_number_rate", "disposition", "Float64", ("disposition",), None,
                "null when no visible dispositions.",
                "n_wrong_number / all visible dispositions."),
        Feature("last_disposition", "disposition", "string", ("disposition",), None,
                "null when no visible dispositions.",
                "Most recent disposition value (by occurred_at)."),
        Feature("days_since_last_disposition", "disposition", "Int64", ("disposition",), None,
                "null when no visible dispositions.",
                "Days from last visible disposition to as_of."),
    ]

    # ---- Remark text cues ---------------------------------------------------
    reg += [
        Feature("remark_switchedoff_cue_count", "text", "Int64", ("disposition",), None,
                "0 when no remarks mention it (never null when dispositions exist; "
                "0 also when no dispositions).",
                "Remarks matching switched-off phrasing (Hinglish patterns)."),
        Feature("remark_wrongnumber_cue_count", "text", "Int64", ("disposition",), None,
                "0 when no match.", "Remarks matching wrong-number phrasing."),
        Feature("remark_thirdparty_cue_count", "text", "Int64", ("disposition",), None,
                "0 when no match.", "Remarks matching third-party-answer phrasing."),
        Feature("remark_avoidance_cue_count", "text", "Int64", ("disposition",), None,
                "0 when no match.", "Remarks matching observable avoidance phrasing."),
        Feature("switched_off_months_max", "text", "Int64", ("disposition",), None,
                "null when no duration phrase is found.",
                "Max months extracted from phrases like 'number band hai 2 mahine se'."),
    ]

    # ---- Voice-bot transcripts ----------------------------------------------
    reg += [
        Feature("n_bot_calls", "bot", "Int64", ("bot_transcript",), None,
                "0 when no visible transcripts.",
                "Visible voice-bot transcripts (all-time)."),
    ]
    for who in who_values:
        reg.append(Feature(
            f"n_bot_who_{who}", "bot", "Int64", ("bot_transcript",), None,
            "0 when no visible transcripts.",
            f"Transcripts where who_answered={who} (payload key when present, "
            "else derived: third-party/name cue -> other, else unknown)."))
    reg += [
        Feature("n_bot_whoisthis_cue", "bot", "Int64", ("bot_transcript",), None,
                "0 when no match.", "Transcripts matching 'who is this' phrasing."),
        Feature("n_bot_name_mismatch", "bot", "Int64", ("bot_transcript",), None,
                "0 when no match.", "Transcripts matching name-mismatch phrasing."),
        Feature("n_bot_language_mismatch", "bot", "Int64", ("bot_transcript",), None,
                "0 when no match.", "Transcripts matching language-barrier phrasing."),
        Feature("last_who_answered", "bot", "string", ("bot_transcript",), None,
                "null when no visible transcripts.",
                "who_answered of the most recent transcript."),
    ]

    # ---- Shared contacts (lender-local) --------------------------------------
    reg += [
        Feature("n_borrowers_sharing_cp", "shared", "Int64",
                ("contact_point_update",), None,
                "Always >= 1 for rows in the universe.",
                "Distinct borrowers sharing this contact_point_ref within the same lender."),
        Feature("n_accounts_sharing_cp", "shared", "Int64",
                ("contact_point_update",), None,
                "Always >= 1.", "Distinct accounts sharing this ref within the lender."),
        Feature("is_shared", "shared", "boolean", ("contact_point_update",), None,
                "Never null.", "True when >1 borrower shares this ref within the lender."),
        Feature("n_phone_cps_for_borrower", "shared", "Int64",
                ("contact_point_update",), None,
                "Always >= 1 for phone rows.",
                "Phone contact points of this borrower known at as_of."),
        Feature("cp_rank_within_borrower", "shared", "Int64",
                ("contact_point_update",), None,
                "1-based; never null.",
                "Rank of this contact point within the borrower by earliest-known "
                "time (ties broken by ref)."),
        Feature("is_primary", "shared", "boolean", ("contact_point_update",), None,
                "Never null.",
                "Latest contact_point_update is_primary when visible, else the "
                "contact_points table flag."),
        Feature("connected_component_size", "shared", "Int64",
                ("contact_point_update",), None,
                "Always >= 1.",
                "Size of the borrower's lender-local sharing component "
                "(borrowers linked by shared contact points)."),
    ]

    # ---- Record history ------------------------------------------------------
    reg += [
        Feature("source", "record", "string", ("contact_point_update",), None,
                "Never null; 'unknown' when neither table nor update gives one.",
                "Latest update source when visible, else the contact_points table value."),
        Feature("record_age_days", "record", "Int64", ("contact_point_update",), None,
                "Never null (falls back to first-seen time).",
                "Days from contact-point creation (or first-seen) to as_of."),
        Feature("days_since_last_update", "record", "Int64", ("contact_point_update",), None,
                "null when no update event is visible.",
                "Days from last contact_point_update to as_of."),
        Feature("n_updates", "record", "Int64", ("contact_point_update",), None,
                "0 when no update event is visible.",
                "Visible contact_point_update events (all-time)."),
        Feature("confirmed_by_payment", "record", "boolean", ("payment", "dial_attempt", "disposition"), None,
                "False when there is no confirming evidence (never null).",
                "True when a visible payment occurred within "
                "payment_confirmation_days after an answered call or RPC on this "
                "contact point."),
        Feature("days_since_confirmed", "record", "Int64", ("payment",), None,
                "null when never confirmed.",
                "Days from the latest confirming payment to as_of."),
    ]

    # ---- Borrower-level / cross-line ------------------------------------------
    for w in windows:
        sfx = window_suffix(w)
        reg += [
            Feature(f"other_lines_attempts{sfx}", "crossline", "Int64", ("dial_attempt",), w,
                    "0 when the borrower has no other phone lines with attempts.",
                    f"Attempts on the borrower's OTHER phone contact points in {w}d."),
            Feature(f"other_lines_answered{sfx}", "crossline", "Int64", ("dial_attempt",), w,
                    "0 when none.",
                    f"Answered attempts on the borrower's other phone lines in {w}d."),
            Feature(f"other_lines_answer_rate{sfx}", "crossline", "Float64", ("dial_attempt",), w,
                    "null when other lines have no attempts in window.",
                    f"Answer rate on the borrower's other phone lines over {w}d."),
            Feature(f"n_payments{sfx}", "crossline", "Int64", ("payment",), w,
                    "0 when no visible payments in window.",
                    f"Borrower-level visible payments with occurred_at in {w}d."),
        ]
    reg += [
        Feature("days_since_last_payment", "crossline", "Int64", ("payment",), None,
                "null when no visible payment.",
                "Days from last visible borrower payment to as_of."),
        Feature("days_since_last_other_line_answer", "crossline", "Int64", ("dial_attempt",), None,
                "null when no other line was ever answered.",
                "Days from the last answered attempt on any OTHER phone line to as_of."),
    ]

    # ---- Field / address (thin stub; null for phone rows) ----------------------
    for outcome in visit_outcomes:
        reg.append(Feature(
            f"n_visits_{outcome}", "field", "Int64", ("field_visit",), None,
            "null for phone contact points; 0 for addresses with no such outcome.",
            f"Visible field visits with outcome={outcome}."))
    reg += [
        Feature("n_visits", "field", "Int64", ("field_visit",), None,
                "null for phone contact points.",
                "Visible field visits (all-time)."),
        Feature("last_visit_outcome", "field", "string", ("field_visit",), None,
                "null for phones or when never visited.",
                "Most recent visit outcome (by occurred_at)."),
        Feature("days_since_last_visit", "field", "Int64", ("field_visit",), None,
                "null for phones or when never visited.",
                "Days from last visible visit to as_of."),
        Feature("gps_dwell_mean_seconds", "field", "Float64", ("field_visit",), None,
                "null for phones or when no dwell recorded.",
                "Mean dwell_seconds across visible visits."),
        Feature("visit_hour_mean", "field", "Float64", ("field_visit",), None,
                "null for phones or when never visited.",
                "Mean visit hour in IST (circular mean is NOT used; plain mean, "
                "documented as approximate)."),
    ]

    # ---- Account context ---------------------------------------------------------
    reg += [
        Feature("dpd_bucket", "account", "string", (), None,
                "Never null when the borrower row exists.",
                "DPD bucket from the borrowers table."),
        Feature("outstanding", "account", "Float64", (), None,
                "Never null when the borrower row exists.",
                "Outstanding amount from the borrowers table (simulation-only)."),
        Feature("product", "account", "string", (), None,
                "Never null when the borrower row exists.",
                "Product segment from the borrowers table."),
        Feature("secured_flag", "account", "boolean", (), None,
                "Never null when the borrower row exists.",
                "Whether the product is secured (from the borrowers table)."),
    ]

    # ---- Core presence -------------------------------------------------------------
    reg += [
        Feature("has_any_attempt", "core", "boolean", ("dial_attempt",), None,
                "Never null.",
                "True when any dial attempt is visible for this contact point. "
                "Distinguishes 'no evidence' from measured zeros."),
        Feature("contact_point_type", "core", "string", ("contact_point_update",), None,
                "Never null.", "phone or address for this contact point."),
    ]

    # ---- Calendar --------------------------------------------------------------------
    reg += [
        Feature("asof_weekday", "calendar", "Int64", (), None,
                "Never null.", "as_of weekday in IST (Monday=0)."),
        Feature("asof_day_of_month", "calendar", "Int64", (), None,
                "Never null.", "as_of day of month in IST."),
        Feature("is_holiday", "calendar", "boolean", (), None,
                "Never null.", "True when the as_of IST date is in configs holidays."),
    ]

    # ---- Conditional: agent reliability -----------------------------------------------
    if include_agent_feature:
        reg.append(Feature(
            "agent_wrong_number_rate", "agent", "Float64", ("disposition",), None,
            "null when the last disposition has no agent or the agent has no history.",
            "Wrong-number share of the agent who recorded the latest disposition "
            "(feature of disposition reliability; only present when agent_id is observed)."))

    names = [f.name for f in reg]
    assert len(names) == len(set(names)), "duplicate feature names in registry"
    return reg


def feature_names(
    config: dict[str, Any] | None = None, *, include_agent_feature: bool = False
) -> list[str]:
    return [f.name for f in build_registry(config, include_agent_feature=include_agent_feature)]


def dtype_map(
    config: dict[str, Any] | None = None, *, include_agent_feature: bool = False
) -> dict[str, str]:
    return {f.name: f.dtype for f in build_registry(config, include_agent_feature=include_agent_feature)}


def render_registry_markdown(
    config: dict[str, Any] | None = None, *, include_agent_feature: bool = False
) -> str:
    """Render the registry as a Markdown table (for docs/features.md)."""
    lines = [
        "| feature | group | dtype | source events | window | null semantics | description |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in build_registry(config, include_agent_feature=include_agent_feature):
        window = f"{f.window}d" if f.window is not None else "-"
        sources = ", ".join(f.source_event_types) if f.source_event_types else "tables/calendar"
        lines.append(
            f"| `{f.name}` | {f.group} | {f.dtype} | {sources} | {window} "
            f"| {f.null_semantics} | {f.description} |"
        )
    return "\n".join(lines)


def update_features_doc(
    doc_path: str | Path = FEATURES_DOC_PATH,
    config: dict[str, Any] | None = None,
) -> None:
    """Regenerate the registry table inside docs/features.md in place.

    Only the text between REGISTRY markers is replaced; hand-written schema
    notes outside the markers are preserved.
    """
    doc_path = Path(doc_path)
    current = doc_path.read_text()
    if REGISTRY_START not in current or REGISTRY_END not in current:
        raise ValueError(f"Registry markers missing in {doc_path}")
    table = render_registry_markdown(config, include_agent_feature=True)
    # Document the conditional feature inline: it is only emitted when agent_id
    # is observed. Keep the row but flag it (registry built with the flag on
    # is the superset; build-time registry without agent must be a subset).
    before, _, rest = current.partition(REGISTRY_START)
    _, _, after = rest.partition(REGISTRY_END)
    doc_path.write_text(f"{before}{REGISTRY_START}\n{table}\n{REGISTRY_END}{after}")
