"""Rule-based text cue extraction from remarks and transcripts.

v0 keyword/regex layer over Hinglish agent remarks and voice-bot transcripts.
Patterns live in ``configs/text_patterns.yaml``; this module only applies them.
Output is features, never decisions.

Case-insensitive throughout; patterns already tolerate common spelling
variants. Internal hidden-state names must never appear as patterns; see the
config header and the banned-pattern scan in tests.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.rpc.features.spec import TEXT_PATTERNS_PATH

# Hidden-state names that must never be matched on (scan-enforced in tests).
BANNED_PATTERN_TOKENS = (
    "valid_reachable",
    "avoiding",
    "temp_unreachable",
    "switched_off_long",
    "recycled",
    "true_state",
    "ground_truth",
)

# Remark cue groups -> feature suffix used by the feature builder.
REMARK_GROUPS = {
    "switched_off_cue": "remark_switchedoff_cue_count",
    "wrong_number_cue": "remark_wrongnumber_cue_count",
    "third_party_cue": "remark_thirdparty_cue_count",
    "avoidance_cue": "remark_avoidance_cue_count",
}

# Bot transcript cue groups -> count feature name.
BOT_GROUPS = {
    "who_is_this_cue": "n_bot_whoisthis_cue",
    "name_mismatch_cue": "n_bot_name_mismatch",
    "language_mismatch_cue": "n_bot_language_mismatch",
}

# Third-party / name cues also mark who_answered=other when the payload
# carries no explicit who_answered key.
OTHER_WHO_GROUPS = ("third_party_cue", "name_mismatch_cue")


@lru_cache(maxsize=4)
def load_patterns(path: str | Path = TEXT_PATTERNS_PATH) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    for token in BANNED_PATTERN_TOKENS:
        for group, pats in cfg.get("patterns", {}).items():
            for pat in pats:
                if token in pat.lower():
                    raise ValueError(
                        f"Banned hidden-state token '{token}' in pattern group '{group}'"
                    )
    return cfg


@lru_cache(maxsize=8)
def _compiled(group: str, path: str = str(TEXT_PATTERNS_PATH)) -> "re.Pattern[str]":
    cfg = load_patterns(path)
    pats = cfg["patterns"][group]
    return re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)


def _series_matches(texts: pd.Series, group: str) -> pd.Series:
    """Vectorised boolean match of a cue group against a text Series."""
    import warnings

    rx = _compiled(group)
    filled = texts.fillna("").astype(str)
    with warnings.catch_warnings():
        # Patterns intentionally use groups for alternation; we only need match/no-match.
        warnings.filterwarnings("ignore", message="This pattern is interpreted.*", category=UserWarning)
        return filled.str.contains(rx, regex=True)


def count_cues(texts: pd.Series, group: str) -> pd.Series:
    """Count events whose text matches the cue group (0/1 per row)."""
    return _series_matches(texts, group).astype("int")


def extract_switched_off_months_text(text: str | None) -> int | None:
    """Extract months from e.g. 'number band hai 2 mahine se' ( else None).

    Requires a switched-off cue AND a nearby (<count> <unit>) phrase where
    count is a digit or Hindi number word and unit is a month/year word.
    Year units convert at 12 months. Returns the max over the text.
    """
    if not text:
        return None
    cfg = load_patterns()
    if not _compiled("switched_off_cue").search(text):
        return None
    num_words = "|".join(sorted(cfg.get("hindi_numbers", {}).keys(), key=len, reverse=True))
    month_units = "|".join(f"(?:{u})" for u in cfg.get("month_units", ["mahine", "months?"]))
    year_units = "|".join(
        f"(?:{u})" for u in cfg.get("year_units", ["saals?", "sal", "years?"])
    )
    rx = re.compile(
        rf"(?P<num>\d+|{num_words})\s*(?P<unit>{month_units}|{year_units})",
        re.IGNORECASE,
    )
    best: int | None = None
    for m in rx.finditer(text):
        num_raw = m.group("num").lower()
        try:
            num = int(num_raw)
        except ValueError:
            num = cfg["hindi_numbers"].get(num_raw)
        if num is None:
            continue
        unit = m.group("unit").lower()
        months = num * 12 if re.fullmatch(year_units, unit, re.IGNORECASE) else num
        if months is not None and months > 0 and (best is None or months > best):
            best = months
    return best


def extract_switched_off_months(texts: pd.Series) -> pd.Series:
    """Vectorised wrapper returning nullable Int64 months (None when absent)."""
    return pd.Series(
        [extract_switched_off_months_text(t) for t in texts.tolist()],
        index=texts.index,
        dtype="Int64",
    )


def derive_who_answered(transcripts: pd.Series, explicit: pd.Series) -> pd.Series:
    """Resolve who_answered per transcript.

    Uses the payload ``who_answered`` key when present (normalised to
    borrower/other/unknown); otherwise 'other' when a third-party or
    name-mismatch cue matches, else 'unknown'.
    """
    valid = {"borrower", "other", "unknown"}
    exp = explicit.fillna("").astype(str).str.strip().str.lower()
    known = exp.where(exp.isin(valid), None)
    cue_other = _series_matches(transcripts, "third_party_cue") | _series_matches(
        transcripts, "name_mismatch_cue"
    )
    return known.fillna(pd.Series(["other" if b else "unknown" for b in cue_other], index=transcripts.index))
