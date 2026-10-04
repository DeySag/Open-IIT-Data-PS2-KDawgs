"""Suppression list store.

Append-only with an audit log. Additions are immediate; removals
are only possible through a removal request that requires CN
sign-off. Entries are never removed by the service itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from src.rpc.contracts import SuppressionEntry


class SuppressionStore:
    """In-memory suppression list with versioning and audit log."""

    def __init__(self, model_version: str) -> None:
        self.model_version = model_version
        self._entries: dict[tuple[str, str], SuppressionEntry] = {}
        self._entry_versions: dict[tuple[str, str], int] = {}
        self._evidence_index: dict[tuple[str, UUID], str] = {}
        self._version = 0
        self._removal_requests: list[dict[str, Any]] = []
        self._audit_log: list[dict[str, Any]] = []

    @property
    def version(self) -> int:
        return self._version

    def add(
        self,
        contact_point_ref: str,
        lender_id: str,
        reason: Literal["recycled", "third_party"],
        evidence: list[UUID],
        added_at: datetime | None = None,
    ) -> SuppressionEntry | None:
        """Add a suppression entry. Idempotent on evidence.

        Returns the new/updated entry, or None if no new evidence
        was provided (i.e. a duplicate replay).
        """
        added_at = added_at or datetime.now(timezone.utc)
        key = (contact_point_ref, lender_id)

        new_evidence: list[UUID] = []
        for event_id in evidence:
            ev_key = (contact_point_ref, event_id)
            if ev_key in self._evidence_index:
                continue
            self._evidence_index[ev_key] = reason
            new_evidence.append(event_id)

        if not new_evidence:
            return None

        if key in self._entries:
            existing = self._entries[key]
            merged_evidence = list(existing.evidence) + new_evidence
            entry = SuppressionEntry(
                contact_point_ref=contact_point_ref,
                lender_id=lender_id,
                reason=existing.reason,
                evidence=merged_evidence,
                added_at=existing.added_at,
                model_version=self.model_version,
            )
        else:
            entry = SuppressionEntry(
                contact_point_ref=contact_point_ref,
                lender_id=lender_id,
                reason=reason,
                evidence=new_evidence,
                added_at=added_at,
                model_version=self.model_version,
            )

        self._entries[key] = entry
        self._version += 1
        self._entry_versions[key] = self._version

        self._audit_log.append(
            {
                "version": self._version,
                "contact_point_ref": contact_point_ref,
                "lender_id": lender_id,
                "reason": reason,
                "evidence": [str(e) for e in new_evidence],
                "added_at": added_at.isoformat(),
                "model_version": self.model_version,
                "removal_requires": "cn_signoff",
            }
        )
        return entry

    def is_suppressed(self, contact_point_ref: str, lender_id: str) -> bool:
        return (contact_point_ref, lender_id) in self._entries

    def get(self, contact_point_ref: str, lender_id: str) -> SuppressionEntry | None:
        return self._entries.get((contact_point_ref, lender_id))

    def list_entries(
        self,
        lender_id: str | None = None,
        since_version: int = 0,
    ) -> list[SuppressionEntry]:
        """List entries, optionally filtered by lender and version."""
        result: list[SuppressionEntry] = []
        for key, entry in self._entries.items():
            if lender_id and key[1] != lender_id:
                continue
            if self._entry_versions.get(key, 0) <= since_version:
                continue
            result.append(entry)
        return result

    def request_removal(
        self,
        contact_point_ref: str,
        lender_id: str,
        reason: str,
        requester: str,
        evidence: list[UUID] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Create a pending removal request requiring CN sign-off.

        There is no direct delete; the entry stays in force until
        CN approves the request out-of-band.
        """
        request: dict[str, Any] = {
            "request_id": str(uuid4()),
            "contact_point_ref": contact_point_ref,
            "lender_id": lender_id,
            "reason": reason,
            "requester": requester,
            "evidence": [str(e) for e in (evidence or [])],
            "notes": notes,
            "status": "pending_cn_signoff",
            "removal_requires": "cn_signoff",
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self._removal_requests.append(request)
        return request

    def list_removal_requests(
        self, lender_id: str | None = None
    ) -> list[dict[str, Any]]:
        if lender_id:
            return [
                r
                for r in self._removal_requests
                if r["lender_id"] == lender_id
            ]
        return list(self._removal_requests)

    def audit_log(
        self, lender_id: str | None = None
    ) -> list[dict[str, Any]]:
        if lender_id:
            return [a for a in self._audit_log if a["lender_id"] == lender_id]
        return list(self._audit_log)
