import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.domain import FaultCause
from app.inference import InferenceResult
from app.recommendations import recommendations_for
from app.schemas import KPIRecord, SiteHistoryItem


class HistoryRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{self.path.resolve().as_posix()}")
        command.upgrade(config, "head")

    def record(self, kpis: KPIRecord, result: InferenceResult) -> int:
        import json

        created = datetime.now(UTC).isoformat()
        result_data = {
            "status": result.status.value,
            "binary_detection": {
                "prediction": result.binary_prediction,
                "threshold": result.threshold,
                "fault_probability": result.fault_probability,
                "normal_probability": result.normal_probability,
            },
            "classification": {
                "predicted_class_id": result.class_id,
                "predicted_cause": result.predicted_cause,
                "confidence": result.confidence,
                "probabilities": result.class_probabilities,
            },
            "cause": result.cause,
            "cause_name": result.cause_name,
            "fault": result.fault,
            "recommendations": (
                recommendations_for(FaultCause(result.cause)) if result.cause else []
            ),
            "model_version": result.model_version,
        }
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT INTO predictions
                (cell, latitude, longitude, fault, cause, event_date, created_at, kpis_json,
                 status, model_version, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    kpis.cell,
                    kpis.latitude,
                    kpis.longitude,
                    int(result.fault) if result.fault is not None else None,
                    result.cause,
                    kpis.date.isoformat() if kpis.date else None,
                    created,
                    json.dumps(kpis.model_dump(mode="json")),
                    result.status.value,
                    result.model_version,
                    json.dumps(result_data, allow_nan=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_history(self, limit: int = 100) -> list[SiteHistoryItem]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id, cell, latitude, longitude, fault, cause, event_date, created_at, "
                "status, model_version, kpis_json, result_json "
                "FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        import json

        return [
            SiteHistoryItem(
                prediction_id=row[0],
                cell=row[1],
                latitude=row[2],
                longitude=row[3],
                fault=bool(row[4]) if row[4] is not None else None,
                cause=FaultCause(row[5]) if row[5] is not None else None,
                date=date.fromisoformat(row[6]) if row[6] is not None else None,
                created_at=datetime.fromisoformat(row[7]),
                status=row[8],
                model_version=row[9],
                kpis=json.loads(row[10]) if row[10] else {},
                inference=json.loads(row[11]) if row[11] else {},
            )
            for row in rows
        ]

    def healthy(self) -> bool:
        try:
            with closing(self._connect()) as connection:
                connection.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
