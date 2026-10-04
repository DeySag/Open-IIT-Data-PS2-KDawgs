"""Report generator: one comparison table per metric family, simulation-only labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SIM_LABEL = "SIMULATION-ONLY - synthetic data, not real-world performance"


def _md_table(df: pd.DataFrame, label: str = SIM_LABEL) -> str:
    lines = [f"> {label}", ""]
    if df.empty:
        return "\n".join(lines + ["_no rows_", ""])
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    lines.append("")
    return "\n".join(lines)


def generate_report(
    results: dict[str, Any],
    out_dir: str | Path,
    label: str = SIM_LABEL,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Write reports/eval_<timestamp>.md and .json. Returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = out / f"eval_{ts}.md"
    json_path = out / f"eval_{ts}.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    L: list[str] = [
        f"# RPC evaluation - {ts}",
        "",
        f"> {label}",
        "",
        f"Generated: {ts} | config: {results.get('config_summary', '')}",
        "",
        f"Data: {results.get('data_summary', '')}",
        "",
        "## Notes / caveats",
        "",
    ]
    for n in results.get("notes", []):
        L.append(f"- {n}")
    L.append("")

    families: list[tuple[str, str]] = [
        ("discrimination", "## Discrimination (dialled-only; simulation-only)"),
        ("calibration", "## Calibration (dialled-only; simulation-only)"),
        ("rare_event", "## Rare-event: recycled (oracle; simulation-only)"),
        ("decision", "## Decision metrics (simulation-only)"),
        ("avoiding_vs_invalid", "## Avoiding vs invalid (oracle subset; simulation-only)"),
        ("propensity", "## Selection-bias second view: IPW (dialled + IPW; simulation-only)"),
    ]
    tables = results.get("tables", {})
    for key, heading in families:
        L.append(heading)
        L.append("")
        L.append(_md_table(pd.DataFrame(tables.get(key, [])), label))

    if "reliability" in tables:
        L.append("## Reliability detail: predicted vs observed (simulation-only)")
        L.append("")
        for model, rows in tables["reliability"].items():
            L.append(f"### {model}")
            L.append("")
            L.append(_md_table(pd.DataFrame(rows), label))

    if "cross_line" in tables:
        L.append("## Cross-line subset: silent line while borrower reachable elsewhere (simulation-only)")
        L.append("")
        L.append(_md_table(pd.DataFrame(tables["cross_line"]), label))

    md_path.write_text("\n".join(L))
    return md_path, json_path
