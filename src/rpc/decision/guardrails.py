"""Guardrails: hard compliance rules evaluated FIRST.

Models and action logic can only narrow what guardrails allow, never widen
it. Every rule here only *removes* actions/channels/contact points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.rpc.decision.reason_codes import DecisionReason
from src.rpc.decision.types import AccountContext, ContactPointScore

ALL_ACTIONS: tuple[str, ...] = ("continue", "switch_contact_point", "switch_channel", "trace")
ALL_CHANNELS: tuple[str, ...] = ("sms", "whatsapp", "voice_bot", "telecaller", "field")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GUARDRAILS_PATH = REPO_ROOT / "configs" / "guardrails.yaml"


def load_guardrails_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_GUARDRAILS_PATH) as f:
        return yaml.safe_load(f)["guardrails"]


def in_contact_hours(now: datetime, start_hour: int, end_hour: int, tz_name: str) -> bool:
    """True when ``now`` falls inside [start_hour, end_hour) in ``tz_name``.

    Handles overnight windows (e.g. 22-06) and day boundaries. Requires a
    timezone-aware ``now``; naive input is assumed UTC (documented, not silent).
    """
    tz = ZoneInfo(tz_name)
    local = now.astimezone(tz) if now.tzinfo is not None else now.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    h = local.hour + local.minute / 60.0 + local.second / 3600.0
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= h < end_hour
    return h >= start_hour or h < end_hour  # overnight window


@dataclass
class GuardrailReport:
    allowed_actions: set[str] = field(default_factory=lambda: set(ALL_ACTIONS))
    allowed_channels: set[str] = field(default_factory=lambda: set(ALL_CHANNELS))
    excluded_refs: dict[str, str] = field(default_factory=dict)  # ref -> rule code
    fired_rules: list[str] = field(default_factory=list)
    hard_blocked: bool = False
    block_reason: DecisionReason | None = None


def _hard_block(report: GuardrailReport, reason: DecisionReason, rule: str) -> None:
    """Account-level restriction: no outreach and never a trace. The account
    is parked on switch_channel/field so the four-action contract still yields
    a next action; the serving layer must NOT execute outreach for parked
    accounts (suppression enforced downstream)."""
    report.hard_blocked = True
    report.block_reason = reason
    report.allowed_actions = {"switch_channel"}
    report.allowed_channels = {"field"}
    if rule not in report.fired_rules:
        report.fired_rules.append(rule)


def evaluate_guardrails(
    ctx: AccountContext,
    scores: list[ContactPointScore],
    config: dict | None = None,
) -> GuardrailReport:
    """Evaluate every guardrail for one account. Pure function of (ctx, scores)."""
    cfg = config or load_guardrails_config()
    report = GuardrailReport()

    # 1. Per-contact-point suppression (highest priority, but per-CP, not per-account).
    if cfg.get("suppression", {}).get("enabled", True):
        for s in scores:
            if s.contact_point_ref in ctx.suppressed_refs:
                report.excluded_refs[s.contact_point_ref] = DecisionReason.GUARDRAIL_SUPPRESSED.value
        if scores and len(report.excluded_refs) == len(scores):
            report.fired_rules.append("SUPPRESSION_ALL")

    # 2. Account-level hard blocks: never outreach, never trace.
    flags = ctx.flags
    disputes_cfg = cfg.get("disputes", {})
    deceased_cfg = cfg.get("deceased_insolvent", {})
    consent_cfg = cfg.get("consent", {})
    if flags.no_consent and consent_cfg.get("enabled", True):
        _hard_block(report, DecisionReason.GUARDRAIL_NO_CONSENT, "NO_CONSENT")
        return report
    if flags.dispute and disputes_cfg.get("enabled", True):
        _hard_block(report, DecisionReason.GUARDRAIL_DISPUTE, "DISPUTE")
        return report
    if flags.deceased_or_insolvent and deceased_cfg.get("enabled", True):
        _hard_block(report, DecisionReason.GUARDRAIL_DECEASED, "DECEASED_OR_INSOLVENT")
        return report
    if flags.legal_case:
        _hard_block(report, DecisionReason.GUARDRAIL_LEGAL_CASE, "LEGAL_CASE")
        return report

    # 3. Trace already pending: remove trace (no double-queue).
    if ctx.trace_pending:
        report.allowed_actions.discard("trace")
        report.fired_rules.append("TRACE_PENDING")

    # 4. DND: voice/telecaller channels removed (SMS/WhatsApp/field remain).
    if flags.dnd and cfg.get("dnd", {}).get("enabled", True):
        report.allowed_channels.discard("voice_bot")
        report.allowed_channels.discard("telecaller")
        report.fired_rules.append("DND_CHANNEL_RESTRICT")

    # 5. Contact-hours window in Asia/Kolkata: no new channels/traces outside it.
    hours_cfg = cfg.get("contact_hours", {})
    if hours_cfg.get("enabled", True):
        tz_name = hours_cfg.get("timezone", "Asia/Kolkata")
        if not in_contact_hours(
            ctx.now,
            int(hours_cfg.get("start_hour", 8)),
            int(hours_cfg.get("end_hour", 19)),
            tz_name,
        ):
            report.allowed_actions.discard("switch_channel")
            report.allowed_actions.discard("trace")
            report.fired_rules.append("CONTACT_HOURS")

    # 6. Frequency caps: no new phone attempts right now.
    freq_cfg = cfg.get("frequency_caps", {})
    if freq_cfg.get("enabled", True):
        day_cap = int(freq_cfg.get("max_attempts_per_day", 3))
        week_cap = int(freq_cfg.get("max_attempts_per_week", 10))
        if ctx.attempts_today >= day_cap or ctx.attempts_week >= week_cap:
            report.allowed_actions.discard("continue")
            report.allowed_actions.discard("switch_contact_point")
            report.allowed_actions.discard("trace")
            report.fired_rules.append("FREQUENCY_CAP")

    # Defensive: guardrails must never yield an empty set outside hard blocks
    # (hard blocks set exactly the parking action above and return early).
    if not report.hard_blocked and not report.allowed_actions:
        report.allowed_actions = {"switch_channel"}
        report.fired_rules.append("EMPTY_FALLBACK_PARK")

    return report
