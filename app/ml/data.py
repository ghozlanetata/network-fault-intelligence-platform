from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER = (
    "Retainability",
    "HOSR",
    "RSRP",
    "RSRQ",
    "SINR",
    "Throughput",
    "Distance",
)
CLASS_MAPPING = {
    1: {"code": "ED", "name": "Excessive Downtilt"},
    2: {"code": "CH", "name": "Coverage Hole"},
    3: {"code": "II", "name": "Inter-System Interference"},
    4: {"code": "TLHO", "name": "Too Late HandOver"},
    5: {"code": "RP", "name": "Reduction of Cell Power"},
    6: {"code": "EU", "name": "Excessive Uptilt"},
    7: {"code": "Normal", "name": "Normal"},
}
FAULT_CODES = tuple(range(1, 7))
SEED = 42
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ValidatedDataset:
    path: Path
    frame: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    binary_labels: np.ndarray
    groups: np.ndarray


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    fold_assignment: tuple[int, ...]
    requested_proportions: dict[str, float]


def locate_dataset(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if path.is_file() and path.name.lower() == "db_fm_validation14.csv":
            return path
        raise FileNotFoundError(f"Dataset path is not a db_fm_validation14.csv file: {path}")

    preferred = (
        ROOT / "db_fm_validation14.csv",
        ROOT / "artifacts" / "original" / "db_fm_validation14.csv",
        ROOT / "data" / "db_fm_validation14.csv",
    )
    for path in preferred:
        if path.is_file():
            return path.resolve()

    ignored_dirs = {".git", ".v313", ".venv", "__pycache__", ".pytest_cache", "models"}
    found: list[Path] = []
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in ignored_dirs]
        if "db_fm_validation14.csv" in files:
            found.append(Path(current) / "db_fm_validation14.csv")
    if len(found) == 1:
        return found[0].resolve()
    if len(found) > 1:
        raise RuntimeError(
            "Multiple db_fm_validation14.csv files were found; pass --dataset explicitly: "
            + ", ".join(str(path) for path in found)
        )
    raise FileNotFoundError("Could not locate db_fm_validation14.csv under the project root")


def normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize malformed source headers in memory; never writes the CSV."""
    renamed: dict[object, str] = {}
    for header in frame.columns:
        clean = str(header).lstrip("\ufeff").strip().lstrip("[").rstrip("]").strip()
        if clean == "Average Throughput":
            clean = "Throughput"
        renamed[header] = clean
    result = frame.rename(columns=renamed)
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Header normalization produced duplicate columns: {duplicates}")
    return result


def feature_group_ids(features: np.ndarray) -> np.ndarray:
    """Stable SHA-256 identifiers over each complete, little-endian float64 vector."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_ORDER):
        raise ValueError(f"Expected a two-dimensional 7-feature matrix, got {values.shape}")
    groups = []
    for row in values:
        canonical = np.asarray(row, dtype="<f8").copy()
        canonical[canonical == 0] = 0.0  # Treat -0 and +0 as the same numeric feature.
        groups.append(hashlib.sha256(canonical.tobytes()).hexdigest())
    return np.asarray(groups, dtype=object)


def load_dataset(path: str | Path | None = None) -> ValidatedDataset:
    source = locate_dataset(path)
    raw = pd.read_csv(source)
    frame = normalize_headers(raw)
    expected = [*FEATURE_ORDER, "FaultCause"]
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}; found {frame.columns.tolist()}"
        )

    for column in FEATURE_ORDER:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["FaultCause"] = pd.to_numeric(frame["FaultCause"], errors="raise")
    features = frame.loc[:, FEATURE_ORDER].to_numpy(dtype=np.float64)
    label_values = frame["FaultCause"].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("The seven KPI features contain missing or non-finite values")
    if (
        not np.isfinite(label_values).all()
        or not np.equal(label_values, np.floor(label_values)).all()
    ):
        raise ValueError("FaultCause must contain finite integer labels")
    labels = label_values.astype(np.int64)
    observed = set(np.unique(labels).tolist())
    if observed != set(CLASS_MAPPING):
        raise ValueError(f"FaultCause must contain every class 1–7; found {sorted(observed)}")
    groups = feature_group_ids(features)

    group_label_counts = (
        pd.DataFrame({"group": groups, "label": labels}).groupby("group")["label"].nunique()
    )
    if (group_label_counts > 1).any():
        raise ValueError("Identical feature vectors have conflicting FaultCause labels")

    binary_labels = np.where(labels == 7, 0, 1).astype(np.int64)
    return ValidatedDataset(
        path=source,
        frame=frame,
        features=features,
        labels=labels,
        binary_labels=binary_labels,
        groups=groups,
    )


def split_dataset(dataset: ValidatedDataset, seed: int = SEED) -> DatasetSplit:
    """Build deterministic ~71/14/14 stratified-group train/validation/test folds."""
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    fold_by_index = np.full(len(dataset.labels), -1, dtype=np.int64)
    for fold, (_, held_out) in enumerate(
        splitter.split(dataset.features, dataset.labels, groups=dataset.groups)
    ):
        fold_by_index[held_out] = fold
    if (fold_by_index < 0).any():
        raise RuntimeError("StratifiedGroupKFold did not assign every row to a fold")

    # Keep the fold assignment fixed by seed. Fold 0 is the final test set, fold 1
    # validation, and folds 2–6 training. Test is not touched during fitting.
    test = np.flatnonzero(fold_by_index == 0)
    validation = np.flatnonzero(fold_by_index == 1)
    train = np.flatnonzero(fold_by_index >= 2)
    partitions = {"train": train, "validation": validation, "test": test}
    for name, indices in partitions.items():
        present = set(np.unique(dataset.labels[indices]).tolist())
        if present != set(CLASS_MAPPING):
            raise ValueError(
                f"Split {name} is missing classes: {sorted(set(CLASS_MAPPING) - present)}"
            )

    group_sets = {name: set(dataset.groups[idx]) for name, idx in partitions.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise RuntimeError(
                f"Feature group leakage between {left} and {right}: {len(overlap)} groups"
            )

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        fold_assignment=tuple(int(value) for value in fold_by_index),
        requested_proportions={"train": 0.70, "validation": 0.15, "test": 0.15},
    )


def split_summary(dataset: ValidatedDataset, split: DatasetSplit) -> dict[str, object]:
    total = len(dataset.labels)
    summary: dict[str, object] = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        labels = dataset.labels[indices]
        counts = {str(code): int(np.sum(labels == code)) for code in CLASS_MAPPING}
        summary[name] = {
            "samples": int(len(indices)),
            "proportion": float(len(indices) / total),
            "unique_groups": int(len(set(dataset.groups[indices]))),
            "class_distribution": counts,
            "binary_distribution": {
                "Fault": int(np.sum(dataset.binary_labels[indices] == 1)),
                "Normal": int(np.sum(dataset.binary_labels[indices] == 0)),
            },
        }
    all_groups = [set(dataset.groups[idx]) for idx in (split.train, split.validation, split.test)]
    summary["group_leakage"] = {
        "train_validation": len(all_groups[0] & all_groups[1]),
        "train_test": len(all_groups[0] & all_groups[2]),
        "validation_test": len(all_groups[1] & all_groups[2]),
        "all_groups_assigned_once": len(set.union(*all_groups)) == len(set(dataset.groups)),
    }
    return summary


def fit_training_preprocessor(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Fit once on training rows and transform all three partitions."""
    scaler = StandardScaler()
    scaler.fit(train_features)
    transformed = [
        scaler.transform(values).reshape((-1, 1, len(FEATURE_ORDER))).astype(np.float32)
        for values in (train_features, validation_features, test_features)
    ]
    return scaler, transformed[0], transformed[1], transformed[2]
