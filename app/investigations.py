"""Persistent closed-loop investigation records, separate from immutable predictions."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from app.domain import FaultCause, InvestigationStatus, ResolutionStatus, VerificationStatus


def now() -> str:
    return datetime.now(UTC).isoformat()


class InvestigationRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def create(
        self,
        *,
        prediction_id: int | None,
        network_health_result_id: int | None,
        submitted_by: int,
    ) -> dict:
        if (prediction_id is None) == (network_health_result_id is None):
            raise ValueError("Exactly one prediction or Network Health result is required.")
        created = now()
        with closing(self._connect()) as connection, connection:
            if prediction_id is not None:
                source = connection.execute(
                    "SELECT cell,cause FROM predictions WHERE id=?", (prediction_id,)
                ).fetchone()
            else:
                source = connection.execute(
                    "SELECT cell,predicted_cause FROM network_health_results WHERE id=?",
                    (network_health_result_id,),
                ).fetchone()
            if source is None:
                raise LookupError("Prediction or Network Health result was not found.")
            try:
                cursor = connection.execute(
                    """INSERT INTO investigations
                    (prediction_id,network_health_result_id,cell,predicted_cause,
                     investigation_status,findings,investigation_cause,resolution_notes,
                     submitted_by,created_at,updated_at,verification_status,resolution_status,
                     is_completed)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
                    (
                        prediction_id,
                        network_health_result_id,
                        source[0],
                        source[1],
                        InvestigationStatus.DRAFT.value,
                        "",
                        None,
                        "",
                        submitted_by,
                        created,
                        created,
                        VerificationStatus.PENDING.value,
                        ResolutionStatus.OPEN.value,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise FileExistsError("An investigation already exists for this result.") from exc
            investigation_id = int(cursor.lastrowid)
        return self.get(investigation_id)

    def get(self, investigation_id: int) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT i.id,i.prediction_id,i.network_health_result_id,i.cell,
                   i.predicted_cause,i.investigation_status,i.findings,i.investigation_cause,
                   i.resolution_notes,i.submitted_by,s.username,i.created_at,i.updated_at,
                   i.submitted_at,i.verified_cause,i.verified_by,v.username,i.verified_at,
                   i.verification_status,i.resolution_status,i.is_completed
                   FROM investigations i JOIN users s ON s.id=i.submitted_by
                   LEFT JOIN users v ON v.id=i.verified_by WHERE i.id=?""",
                (investigation_id,),
            ).fetchone()
            if row is None:
                return None
            events = connection.execute(
                """SELECT previous_status,previous_cause,new_status,new_cause,notes,
                   verified_by,verified_at FROM investigation_verification_events
                   WHERE investigation_id=? ORDER BY id""",
                (investigation_id,),
            ).fetchall()
            if row[1] is not None:
                source = connection.execute(
                    "SELECT status,latitude,longitude,kpis_json,result_json "
                    "FROM predictions WHERE id=?",
                    (row[1],),
                ).fetchone()
                if source is None:
                    operational_context = None
                else:
                    inference = json.loads(source[4]) if source[4] else {}
                    operational_context = {
                        "status": source[0],
                        "latitude": source[1],
                        "longitude": source[2],
                        "kpis": json.loads(source[3]) if source[3] else {},
                        "recommendations": inference.get("recommendations", []),
                    }
            else:
                source = connection.execute(
                    "SELECT status,latitude,longitude,result_json "
                    "FROM network_health_results WHERE id=?",
                    (row[2],),
                ).fetchone()
                if source is None:
                    operational_context = None
                else:
                    result = json.loads(source[3]) if source[3] else {}
                    operational_context = {
                        "status": source[0],
                        "latitude": source[1],
                        "longitude": source[2],
                        "kpis": result.get("kpis", {}),
                        "recommendations": result.get("recommendations", []),
                    }
        result = {
            "investigation_id": row[0],
            "prediction_id": row[1],
            "network_health_result_id": row[2],
            "cell": row[3],
            "predicted_cause": row[4],
            "investigation_status": row[5],
            "findings": row[6],
            "investigation_cause": row[7],
            "resolution_notes": row[8],
            "submitted_by": row[9],
            "submitted_by_name": row[10],
            "created_at": row[11],
            "updated_at": row[12],
            "submitted_at": row[13],
            "verified_cause": row[14],
            "verified_by": row[15],
            "verified_by_name": row[16],
            "verified_at": row[17],
            "verification_status": row[18],
            "resolution_status": row[19],
            "is_completed": bool(row[20]),
            "evaluation_eligible": (
                row[18] == VerificationStatus.VERIFIED.value
                and row[14]
                in {cause.value for cause in FaultCause if cause is not FaultCause.NORMAL}
                and row[5] == InvestigationStatus.VERIFIED.value
            ),
            "verification_history": [
                {
                    "previous_status": event[0],
                    "previous_cause": event[1],
                    "verification_status": event[2],
                    "verified_cause": event[3],
                    "notes": event[4],
                    "verified_by": event[5],
                    "verified_at": event[6],
                }
                for event in events
            ],
            "operational_context": operational_context,
        }
        return result

    def list(self, *, submitted_by: int | None = None, limit: int = 100) -> list[dict]:
        with closing(self._connect()) as connection:
            if submitted_by is None:
                ids = connection.execute(
                    "SELECT id FROM investigations ORDER BY updated_at DESC,id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                ids = connection.execute(
                    "SELECT id FROM investigations WHERE submitted_by=? "
                    "ORDER BY updated_at DESC,id DESC LIMIT ?",
                    (submitted_by, limit),
                ).fetchall()
        return [self.get(row[0]) for row in ids]

    def submitted_queue(self, limit: int = 100) -> list[dict]:
        """Return submitted investigations for reviewer workflows, excluding editable drafts."""
        with closing(self._connect()) as connection:
            ids = connection.execute(
                "SELECT id FROM investigations WHERE investigation_status IN "
                "('PENDING_VERIFICATION','VERIFIED') "
                "ORDER BY CASE WHEN investigation_status='PENDING_VERIFICATION' THEN 0 ELSE 1 END, "
                "COALESCE(submitted_at,updated_at) DESC,id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get(row[0]) for row in ids]

    def update(self, investigation_id: int, *, fields: dict) -> dict | None:
        allowed = {"findings", "investigation_cause", "resolution_notes", "is_completed"}
        if not fields or not set(fields).issubset(allowed):
            raise ValueError("No supported investigation fields were supplied.")
        updates = dict(fields)
        if "investigation_cause" in updates and updates["investigation_cause"] is not None:
            updates["investigation_cause"] = FaultCause(updates["investigation_cause"]).value
        updates["updated_at"] = now()
        assignments = ",".join(f"{field}=?" for field in updates)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                f"UPDATE investigations SET {assignments},"
                "investigation_status='IN_PROGRESS' WHERE id=? AND "
                "investigation_status IN ('DRAFT','IN_PROGRESS')",
                [*updates.values(), investigation_id],
            )
            if cursor.rowcount != 1:
                return None
        return self.get(investigation_id)

    def submit(self, investigation_id: int) -> dict | None:
        submitted = now()
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT is_completed,findings FROM investigations WHERE id=? "
                "AND investigation_status IN ('DRAFT','IN_PROGRESS')",
                (investigation_id,),
            ).fetchone()
            if row is None:
                return None
            if not row[0] or not row[1].strip():
                raise ValueError(
                    "Complete the investigation and provide findings before submission."
                )
            connection.execute(
                "UPDATE investigations SET investigation_status=?,submitted_at=?,updated_at=? "
                "WHERE id=?",
                (
                    InvestigationStatus.PENDING_VERIFICATION.value,
                    submitted,
                    submitted,
                    investigation_id,
                ),
            )
        return self.get(investigation_id)

    def verify(
        self,
        investigation_id: int,
        *,
        verification_status: VerificationStatus,
        verified_cause: FaultCause | None,
        notes: str,
        verified_by: int,
    ) -> dict | None:
        verified_at = now()
        if verification_status is VerificationStatus.PENDING:
            raise ValueError("A reviewer must record VERIFIED, UNRESOLVED, or UNKNOWN.")
        if verification_status is VerificationStatus.VERIFIED:
            if verified_cause is None or verified_cause is FaultCause.NORMAL:
                raise ValueError("A verified fault outcome requires one of the six fault causes.")
            resolution = ResolutionStatus.RESOLVED
        else:
            if verified_cause is not None:
                raise ValueError("Unresolved or unknown outcomes cannot have a verified cause.")
            resolution = ResolutionStatus.UNRESOLVED
        with closing(self._connect()) as connection, connection:
            current = connection.execute(
                "SELECT investigation_status,verification_status,verified_cause "
                "FROM investigations WHERE id=? AND investigation_status IN "
                "('PENDING_VERIFICATION','VERIFIED') AND is_completed=1",
                (investigation_id,),
            ).fetchone()
            if current is None:
                return None
            connection.execute(
                """INSERT INTO investigation_verification_events
                (investigation_id,previous_status,previous_cause,new_status,new_cause,notes,
                 verified_by,verified_at) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    investigation_id,
                    current[1],
                    current[2],
                    verification_status.value,
                    verified_cause.value if verified_cause else None,
                    notes,
                    verified_by,
                    verified_at,
                ),
            )
            connection.execute(
                """UPDATE investigations SET investigation_status=?,verification_status=?,
                verified_cause=?,verified_by=?,verified_at=?,resolution_status=?,updated_at=?
                WHERE id=?""",
                (
                    InvestigationStatus.VERIFIED.value,
                    verification_status.value,
                    verified_cause.value if verified_cause else None,
                    verified_by,
                    verified_at,
                    resolution.value,
                    verified_at,
                    investigation_id,
                ),
            )
        return self.get(investigation_id)

    def evaluatable(self, limit: int = 500) -> list[dict]:
        cases, _ = self._evaluation_population(limit=limit)
        return cases

    def verified_count(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM investigations WHERE verification_status='VERIFIED'"
            ).fetchone()[0])

    def _evaluation_population(self, limit: int | None = None) -> tuple[list[dict], Counter]:
        """Return the exact eligible cases and exclusions used by evaluation_summary."""
        causes = {cause.value for cause in FaultCause if cause is not FaultCause.NORMAL}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT i.id,i.predicted_cause,i.verified_cause,i.investigation_status,
                          i.verification_status,i.is_completed,
                          CASE WHEN i.prediction_id IS NOT NULL THEN p.model_version
                               ELSE n.model_version END
                   FROM investigations i
                   LEFT JOIN predictions p ON p.id=i.prediction_id
                   LEFT JOIN network_health_results n ON n.id=i.network_health_result_id
                   ORDER BY i.verified_at DESC,i.id DESC"""
            ).fetchall()
        eligible: list[dict] = []
        excluded = Counter()
        for identifier, predicted, verified, status, verification, completed, version in rows:
            if verification == VerificationStatus.PENDING.value:
                excluded["pending"] += 1
            elif verification == VerificationStatus.UNRESOLVED.value:
                excluded["unresolved"] += 1
            elif verification == VerificationStatus.UNKNOWN.value:
                excluded["unknown"] += 1
            elif verified is None:
                excluded["missing_verified_cause"] += 1
            elif (
                status != InvestigationStatus.VERIFIED.value
                or verification != VerificationStatus.VERIFIED.value
                or not completed
                or verified not in causes
                or predicted not in causes
            ):
                excluded["invalid_or_incomplete"] += 1
            else:
                case = self.get(identifier)
                case["model_version"] = version if version and version != "legacy-unknown" else None
                eligible.append(case)
        if limit is not None:
            eligible = eligible[:limit]
        return eligible, excluded

    def evaluation_summary(self) -> dict:
        """Calculate cause classification metrics from verified investigation outcomes only."""
        causes = tuple(cause.value for cause in FaultCause if cause is not FaultCause.NORMAL)
        eligible_cases, excluded = self._evaluation_population()
        eligible = [
            (
                case["predicted_cause"], case["verified_cause"], case["model_version"]
            )
            for case in eligible_cases
        ]

        matrix = {actual: {predicted: 0 for predicted in causes} for actual in causes}
        predicted_counts = Counter()
        actual_counts = Counter()
        correct_by_cause = Counter()
        version_pairs: dict[str, list[tuple[str, str]]] = {}
        correct = 0
        for predicted, verified, version in eligible:
            matrix[verified][predicted] += 1
            predicted_counts[predicted] += 1
            actual_counts[verified] += 1
            if predicted == verified:
                correct += 1
                correct_by_cause[verified] += 1
            if version and version != "legacy-unknown":
                version_pairs.setdefault(version, []).append((predicted, verified))

        by_cause = {}
        for cause in causes:
            support = actual_counts[cause]
            predicted_support = predicted_counts[cause]
            if support or predicted_support:
                precision = (
                    correct_by_cause[cause] / predicted_support if predicted_support else None
                )
                recall = correct_by_cause[cause] / support if support else None
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision is not None and recall is not None and precision + recall
                    else 0.0 if precision is not None and recall is not None else None
                )
                by_cause[cause] = {
                    "support": support,
                    "predicted_count": predicted_support,
                    "correct": correct_by_cause[cause],
                    "incorrect": support - correct_by_cause[cause],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

        versions = []
        for version, pairs in sorted(version_pairs.items()):
            version_correct = sum(predicted == verified for predicted, verified in pairs)
            versions.append({
                "model_version": version,
                "eligible_cases": len(pairs),
                "correct": version_correct,
                "incorrect": len(pairs) - version_correct,
                "cause_classification_accuracy": version_correct / len(pairs),
            })

        count = len(eligible)
        return {
            "evaluation_scope": "verified_investigations",
            "evaluation_description": (
                "Performance of predicted fault cause against verified fault cause for eligible "
                "verified investigations. This is not overall production accuracy."
            ),
            "eligible_cases": count,
            "excluded_cases": dict(sorted(excluded.items())),
            "cause_classification": {
                "accuracy": correct / count if count else None,
                "correct": correct,
                "incorrect": count - correct,
            },
            "by_cause": by_cause,
            "confusion_matrix": {"class_order": list(causes), "counts": matrix},
            "model_versions": versions,
            "binary_detection": {
                "status": "not_yet_measurable",
                "description": (
                    "The verified investigation population does not establish true negatives "
                    "or false negatives for binary fault detection."
                ),
            },
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
