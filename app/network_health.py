"""Dataset-wide Network Health processing and isolated SQLite persistence."""

import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.domain import FaultCause, PredictionStatus
from app.inference import CLASS_CODES, InferenceResult
from app.ml.data import CLASS_MAPPING, FEATURE_ORDER, load_dataset
from app.recommendations import recommendations_for
from app.schemas import KPIRecord

CAUSE_CODES = tuple(item["code"] for item in CLASS_MAPPING.values())


class NetworkHealthRunInProgress(RuntimeError):
    """Raised when another dataset-wide health run is already active."""


class NetworkHealthRepository:
    def __init__(self, path: Path):
        self.path = path

    def create_run(self, dataset_name: str) -> int:
        with closing(self._connect()) as connection:
            # Acquire the SQLite write lock before checking, so simultaneous
            # requests cannot both observe an idle pipeline and start a run.
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT id FROM network_health_runs WHERE status='PROCESSING' LIMIT 1"
            ).fetchone()
            if active:
                connection.rollback()
                raise NetworkHealthRunInProgress(
                    f"Network Health run {active[0]} is already processing."
                )
            cursor = connection.execute(
                "INSERT INTO network_health_runs(dataset_name,status,started_at) VALUES(?,?,?)",
                (dataset_name, "PROCESSING", datetime.now(UTC).isoformat()),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def complete_run(self, run_id: int, rows: list[dict], summary: dict) -> None:
        with closing(self._connect()) as connection, connection:
            for row in rows:
                connection.execute(
                    """INSERT INTO network_health_results
                    (run_id,cell,latitude,longitude,ground_truth_cause,status,predicted_cause,model_version,result_json)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        row["cell"],
                        row["latitude"],
                        row["longitude"],
                        row["ground_truth_cause"],
                        row["status"],
                        row["predicted_cause"],
                        row["model_version"],
                        json.dumps(row, allow_nan=False),
                    ),
                )
            connection.execute(
                "UPDATE network_health_runs SET status='COMPLETED',completed_at=?,"
                "model_version=?,record_count=?,summary_json=? WHERE id=?",
                (
                    datetime.now(UTC).isoformat(),
                    summary["model_version"],
                    len(rows),
                    json.dumps(summary, allow_nan=False),
                    run_id,
                ),
            )

    def fail_run(self, run_id: int, error: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE network_health_runs SET status='FAILED',completed_at=?,error=? WHERE id=?",
                (datetime.now(UTC).isoformat(), error[:1000], run_id),
            )

    def latest(self) -> dict | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT id,dataset_name,status,started_at,completed_at,model_version,"
                "record_count,summary_json,error FROM network_health_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "dataset_name": row[1],
            "status": row[2],
            "started_at": row[3],
            "completed_at": row[4],
            "model_version": row[5],
            "record_count": row[6],
            "summary": json.loads(row[7]) if row[7] else None,
            "error": row[8],
        }

    def results(
        self, run_id: int, limit: int, offset: int, status: str | None, query: str | None
    ) -> dict:
        clauses, values = ["r.run_id=?"], [run_id]
        if status:
            clauses.append("r.status=?")
            values.append(status)
        if query:
            clauses.append("r.cell LIKE ?")
            values.append(f"%{query}%")
        where = " AND ".join(clauses)
        with closing(self._connect()) as connection, connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM network_health_results r WHERE {where}", values
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT r.id,r.result_json,i.investigation_status,i.verification_status "
                f"FROM network_health_results r LEFT JOIN investigations i "
                f"ON i.network_health_result_id=r.id WHERE {where} ORDER BY r.id LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
        return {
            "run_id": run_id,
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [
                {
                    "network_health_result_id": row[0],
                    "investigation_status": row[2],
                    "verification_status": row[3],
                    **json.loads(row[1]),
                }
                for row in rows
            ],
        }

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection


def process_dataset(
    repository: NetworkHealthRepository, inference, dataset_path: Path | None = None
) -> dict:
    dataset_name = Path(dataset_path).name if dataset_path else "db_fm_validation14.csv"
    run_id = repository.create_run(dataset_name)
    try:
        return _process_dataset_run(repository, inference, run_id, dataset_path)
    except Exception as exc:
        repository.fail_run(run_id, str(exc))
        raise


def _process_dataset_run(repository, inference, run_id: int, dataset_path: Path | None) -> dict:
    dataset = load_dataset(dataset_path)
    frame = dataset.frame
    required = ("cell", "lon", "lat")
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Dataset is missing required site columns: {missing}")
    cells = frame["cell"].astype(str).str.strip()
    if cells.eq("").any():
        raise ValueError("Dataset contains empty cell identifiers")
    longitudes = np.asarray(frame["lon"], dtype=np.float64)
    latitudes = np.asarray(frame["lat"], dtype=np.float64)
    if not np.isfinite(longitudes).all() or not np.isfinite(latitudes).all():
        raise ValueError("Dataset contains missing or non-finite coordinates")
    if np.any(np.abs(latitudes) > 90) or np.any(np.abs(longitudes) > 180):
        raise ValueError("Dataset contains coordinates outside latitude/longitude bounds")

    try:
        records = []
        for values in dataset.features:
            records.append(
                KPIRecord(
                    retainability=float(values[0]),
                    hosr=float(values[1]),
                    rsrp=float(values[2]),
                    rsrq=float(values[3]),
                    sinr=float(values[4]),
                    average_throughput=float(values[5]),
                    distance=float(values[6]),
                )
            )
        outputs: list[InferenceResult] = inference.predict_batch(records)
        if len(outputs) != len(records):
            raise RuntimeError("Batch inference returned a different number of results than inputs")
        rows = []
        for index, result in enumerate(outputs):
            source_cause = int(dataset.labels[index])
            raw_cause = FaultCause(result.predicted_cause)
            raw = {
                "binary_prediction": result.binary_prediction,
                "binary_probability": result.fault_probability,
                "binary_normal_probability": result.normal_probability,
                "multiclass_probabilities": result.class_probabilities,
                "classifier_predicted_cause": raw_cause.value,
            }
            rows.append(
                {
                    "cell": cells.iloc[index],
                    "latitude": float(latitudes[index]),
                    "longitude": float(longitudes[index]),
                    "kpis": dict(
                        zip(FEATURE_ORDER, map(float, dataset.features[index]), strict=True)
                    ),
                    "ground_truth_cause": source_cause,
                    "ground_truth_cause_code": CLASS_MAPPING[source_cause]["code"],
                    "ground_truth_cause_name": CLASS_MAPPING[source_cause]["name"],
                    "status": result.status.value,
                    "predicted_cause": result.cause,
                    "recommendations": recommendations_for(FaultCause(result.cause))
                    if result.cause
                    else [],
                    **raw,
                    "model_version": result.model_version,
                }
            )
        count = len(rows)
        status_counts = Counter(row["status"] for row in rows)
        cause_counts = Counter(row["classifier_predicted_cause"] for row in rows)
        binary_truth = dataset.binary_labels
        binary_pred = np.asarray([row["binary_prediction"] == "Fault" for row in rows])
        class_pred = np.asarray(
            [CLASS_CODES.index(row["classifier_predicted_cause"]) + 1 for row in rows]
        )
        summary = {
            "total_cells_analyzed": count,
            "normal_count": status_counts[PredictionStatus.NORMAL.value],
            "fault_count": status_counts[PredictionStatus.FAULT.value],
            "review_required_count": status_counts[PredictionStatus.REVIEW_REQUIRED.value],
            "normal_percent": status_counts[PredictionStatus.NORMAL.value] / count * 100,
            "fault_percent": status_counts[PredictionStatus.FAULT.value] / count * 100,
            "review_required_percent": status_counts[PredictionStatus.REVIEW_REQUIRED.value]
            / count
            * 100,
            "predicted_cause_distribution": {code: cause_counts[code] for code in CAUSE_CODES},
            "located_cell_count": len(rows),
            "validation_dataset_performance": {
                "dataset_records": count,
                "binary_accuracy": float(np.mean(binary_pred == binary_truth)),
                "multiclass_accuracy": float(np.mean(class_pred == dataset.labels)),
                "label": "Validation dataset performance; not live production accuracy.",
            },
            "model_version": outputs[0].model_version,
        }
        repository.complete_run(run_id, rows, summary)
        return repository.latest()
    except Exception as exc:
        repository.fail_run(run_id, str(exc))
        raise
