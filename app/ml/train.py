from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import joblib
import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from app.ml.data import (
    CLASS_MAPPING,
    FEATURE_ORDER,
    ROOT,
    SEED,
    fit_training_preprocessor,
    load_dataset,
    split_dataset,
    split_summary,
)
from app.ml.inference import ModelInference
from app.ml.models import build_binary_model, build_multiclass_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights, strict=True)}


def callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, mode="min", verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5, mode="min", verbose=0
        ),
    ]


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    weights: dict[int, float],
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        class_weight=weights,
        callbacks=callbacks(),
        verbose=0,
    )
    values = {name: [float(value) for value in series] for name, series in history.history.items()}
    best_epoch = int(np.argmin(values["val_loss"]) + 1)
    return {
        "epochs_run": len(values["loss"]),
        "best_epoch": best_epoch,
        "best_validation_loss": min(values["val_loss"]),
        "history": values,
    }


def evaluate_multiclass(
    model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    probabilities = np.asarray(model.predict(x_test, verbose=0), dtype=np.float64)
    if probabilities.shape != (len(y_test), 7):
        raise RuntimeError(f"Multiclass output shape should be (N, 7), got {probabilities.shape}")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise RuntimeError("Multiclass model produced invalid probabilities")
    predicted = np.argmax(probabilities, axis=1) + 1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=list(CLASS_MAPPING), zero_division=0
    )
    labels = list(CLASS_MAPPING)
    matrix = confusion_matrix(y_test, predicted, labels=labels)
    report = classification_report(
        y_test,
        predicted,
        labels=labels,
        target_names=[CLASS_MAPPING[code]["code"] for code in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_precision": float(
            precision_recall_fscore_support(y_test, predicted, average="macro", zero_division=0)[0]
        ),
        "macro_recall": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)[
                0
            ]
        ),
        "weighted_recall": float(
            recall_score(y_test, predicted, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        "confusion_matrix_labels_1_to_7": matrix.tolist(),
        "confusion_matrix_label_names": [CLASS_MAPPING[code]["code"] for code in labels],
        "per_class": {
            str(code): {
                "code": CLASS_MAPPING[code]["code"],
                "name": CLASS_MAPPING[code]["name"],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, code in enumerate(labels)
        },
        "classification_report": report,
        "prediction_distribution": {str(code): int(np.sum(predicted == code)) for code in labels},
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_rows_sum_min": float(probabilities.sum(axis=1).min()),
        "probability_rows_sum_max": float(probabilities.sum(axis=1).max()),
    }


def evaluate_binary(
    model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    probabilities = np.asarray(model.predict(x_test, verbose=0), dtype=np.float64).reshape(-1)
    if probabilities.shape != (len(y_test),):
        raise RuntimeError(
            f"Binary output shape should be ({len(y_test)}, 1), got {probabilities.shape}"
        )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise RuntimeError("Binary model produced invalid probabilities")
    predicted = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(y_test, predicted, labels=[1, 0])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=[1, 0], zero_division=0
    )
    # Matrix rows/columns use Fault, Normal order; specificity is Normal recall.
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision[0]),
        "recall_sensitivity": float(recall[0]),
        "specificity": float(recall[1]),
        "f1": float(f1[0]),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "threshold": 0.5,
        "confusion_matrix_true_rows_predicted_columns_Fault_Normal": matrix.tolist(),
        "class_support": {"Fault": int(support[0]), "Normal": int(support[1])},
        "prediction_distribution": {
            "Fault": int(np.sum(predicted == 1)),
            "Normal": int(np.sum(predicted == 0)),
        },
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
    }


def write_report(path: Path, metrics: dict[str, Any], metadata: dict[str, Any]) -> None:
    split = metadata["split_strategy"]["actual"]
    multi = metrics["multiclass"]
    binary = metrics["binary"]
    lines = [
        "# Modernized ML model report",
        "",
        "Models were trained from scratch. All reported test metrics were measured on "
        "the untouched, "
        "group-disjoint test split.",
        "The binary target is derived from the recovered multiclass taxonomy "
        "(classes 1–6 = Fault, class 7 = Normal); it is not a separate dataset annotation.",
        "",
        "## Dataset and splits",
        "",
        f"- Dataset SHA-256: `{metadata['dataset_sha256']}`",
        f"- Samples: {metadata['sample_count']}; "
        f"unique KPI groups: {metadata['unique_group_count']}",
        "- Split method: seeded 7-fold StratifiedGroupKFold "
        "(fold 0 test, fold 1 validation, folds 2–6 train).",
        "- Scaler fitted on training rows only; validation and test were transformed "
        "with that scaler.",
        "",
        "| Split | Rows | Share | Unique groups | Class counts 1–7 | Fault / Normal |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name in ("train", "validation", "test"):
        detail = split[name]
        counts = ", ".join(
            f"{code}:{count}" for code, count in detail["class_distribution"].items()
        )
        binary_counts = detail["binary_distribution"]
        lines.append(
            f"| {name} | {detail['samples']} | {detail['proportion']:.3%} | "
            f"{detail['unique_groups']} | {counts} | {binary_counts['Fault']} / "
            f"{binary_counts['Normal']} |"
        )
    lines += [
        "",
        f"Group overlap counts (train/validation, train/test, validation/test): "
        f"{split['group_leakage']['train_validation']}, "
        f"{split['group_leakage']['train_test']}, "
        f"{split['group_leakage']['validation_test']}.",
        "",
        "## Binary fault detector",
        "",
        f"Accuracy {binary['accuracy']:.6f}; precision {binary['precision']:.6f}; "
        f"sensitivity {binary['recall_sensitivity']:.6f}; "
        f"specificity {binary['specificity']:.6f}; F1 {binary['f1']:.6f}; "
        f"ROC-AUC {binary['roc_auc']:.6f}.",
        "",
        "Confusion matrix (true rows and predicted columns: Fault, Normal):",
        "",
        "```text",
        json.dumps(binary["confusion_matrix_true_rows_predicted_columns_Fault_Normal"]),
        "```",
        "",
        "## Multiclass fault-cause classifier",
        "",
        f"Accuracy {multi['accuracy']:.6f}; macro precision/recall/F1 "
        f"{multi['macro_precision']:.6f}/{multi['macro_recall']:.6f}/{multi['macro_f1']:.6f}; "
        f"weighted precision/recall/F1 "
        f"{multi['weighted_precision']:.6f}/{multi['weighted_recall']:.6f}/"
        f"{multi['weighted_f1']:.6f}.",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for code, detail in multi["per_class"].items():
        lines.append(
            f"| {code} {detail['code']} | {detail['precision']:.6f} | "
            f"{detail['recall']:.6f} | {detail['f1']:.6f} | {detail['support']} |"
        )
    lines += [
        "",
        "Confusion matrix (true rows and predicted columns, classes 1–7):",
        "",
        "```text",
        json.dumps(multi["confusion_matrix_labels_1_to_7"]),
        "```",
        "",
        "## Provenance and limits",
        "",
        "Historical thesis accuracy claims are not used as current results. "
        "This evaluation describes "
        "only this dataset, seed, split, and training run; it does not establish generalization to "
        "unseen network deployments.",
        "",
    ]
    for check in metrics.get("artifact_validation", {}).get("inference_smoke_predictions", []):
        prediction = check["prediction"]
        if prediction["model_disagreement"]:
            normal_probability = prediction["class_probabilities"]["7"]
            lines.append(
                f"Smoke check row {check['row_index']} has a binary/multiclass disagreement: "
                f"true FaultCause={check['true_faultcause']}, "
                f"binary fault probability={prediction['fault_probability']:.6f}, "
                f"multiclass Normal probability={normal_probability:.6f}. "
                "The inference layer reports no cause for this disagreement."
            )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_training(
    dataset_path: str | Path | None = None,
    *,
    model_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    seed: int = SEED,
    epochs: int = 150,
    batch_size: int = 32,
) -> dict[str, Any]:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    dataset = load_dataset(dataset_path)
    split = split_dataset(dataset, seed=seed)
    splits = split_summary(dataset, split)
    x_train_raw = dataset.features[split.train]
    x_validation_raw = dataset.features[split.validation]
    x_test_raw = dataset.features[split.test]

    scaler, x_train, x_validation, x_test = fit_training_preprocessor(
        x_train_raw, x_validation_raw, x_test_raw
    )

    y_multi_train = dataset.labels[split.train] - 1  # Keras sparse labels use 0..6.
    y_multi_validation = dataset.labels[split.validation] - 1
    # Keep report/metric labels in the recovered 1–7 taxonomy; only fit labels are 0–6.
    y_multi_test = dataset.labels[split.test]
    if set(np.unique(y_multi_test).tolist()) != set(CLASS_MAPPING):
        raise RuntimeError("The untouched test split does not contain all seven recovered labels")
    y_binary_train = dataset.binary_labels[split.train]
    y_binary_validation = dataset.binary_labels[split.validation]
    y_binary_test = dataset.binary_labels[split.test]
    multi_weights = balanced_weights(y_multi_train)
    binary_weights = balanced_weights(y_binary_train)

    model_root = Path(model_dir) if model_dir else ROOT / "models"
    output_root = Path(artifacts_dir) if artifacts_dir else ROOT / "artifacts" / "ml"
    model_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    binary_model = build_binary_model()
    binary_training = train_model(
        binary_model,
        x_train,
        y_binary_train,
        x_validation,
        y_binary_validation,
        binary_weights,
        epochs=epochs,
        batch_size=batch_size,
    )
    binary_filename = "binary_fault_detector.keras"
    binary_model.save(model_root / binary_filename)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    multiclass_model = build_multiclass_model()
    multiclass_training = train_model(
        multiclass_model,
        x_train,
        y_multi_train,
        x_validation,
        y_multi_validation,
        multi_weights,
        epochs=epochs,
        batch_size=batch_size,
    )
    multiclass_filename = "fault_cause_classifier.keras"
    multiclass_model.save(model_root / multiclass_filename)

    # Evaluate only after both training runs are complete, using the held-out test fold.
    loaded_binary = tf.keras.models.load_model(
        model_root / binary_filename, compile=False, safe_mode=True
    )
    loaded_multiclass = tf.keras.models.load_model(
        model_root / multiclass_filename, compile=False, safe_mode=True
    )
    binary_metrics = evaluate_binary(loaded_binary, x_test, y_binary_test)
    multiclass_metrics = evaluate_multiclass(loaded_multiclass, x_test, y_multi_test)

    preprocessing = {
        "preprocessing_version": "1.0.0",
        "feature_order": list(FEATURE_ORDER),
        "scaler": scaler,
        "fit_split": "train",
        "fit_sample_count": int(len(split.train)),
        "fit_group_count": int(splits["train"]["unique_groups"]),
    }
    joblib.dump(preprocessing, model_root / "preprocessing.joblib")

    (model_root / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_version": "nfi-retrained-lstm-v1",
                "model_files": {"binary": binary_filename, "multiclass": multiclass_filename},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Verify the persisted inference module loads the exact artifacts and behaves on
    # real dataset rows, without refitting preprocessing.
    inference = ModelInference(model_root)
    inference_checks = []
    for row_index in (int(split.test[0]), int(split.test[-1])):
        sample = {
            name: float(dataset.features[row_index, col_index])
            for col_index, name in enumerate(FEATURE_ORDER)
        }
        result = inference.predict(sample)
        inference_checks.append(
            {
                "row_index": row_index,
                "true_faultcause": int(dataset.labels[row_index]),
                "output_fault": result["fault"],
                "output_cause_code": result["cause_code"],
            }
        )

    dataset_sha = sha256_file(dataset.path)
    split_configuration = {
        "name": "StratifiedGroupKFold",
        "n_splits": 7,
        "shuffle": True,
        "seed": seed,
        "fold_assignment": "fold 0=test, fold 1=validation, folds 2–6=train",
        "requested_proportions": split.requested_proportions,
        "actual": splits,
    }
    metrics = {
        "dataset_sha256": dataset_sha,
        "sample_count": int(len(dataset.labels)),
        "feature_order": list(FEATURE_ORDER),
        "label_definition": {
            "multiclass": "FaultCause 1–7 as mapped in metadata",
            "binary": "Derived target: FaultCause 1–6=Fault (1), "
            "FaultCause 7=Normal (0); no separate binary column exists",
        },
        "split": split_configuration,
        "binary": binary_metrics,
        "multiclass": multiclass_metrics,
        "artifact_validation": {
            "binary_model_loaded": True,
            "multiclass_model_loaded": True,
            "binary_output_shape": list(loaded_binary.output_shape),
            "multiclass_output_shape": list(loaded_multiclass.output_shape),
            "inference_smoke_predictions": inference_checks,
        },
    }
    training_configuration = {
        "seed": seed,
        "epochs_max": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "early_stopping": {"monitor": "val_loss", "patience": 20, "restore_best_weights": True},
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 7,
            "min_lr": 1e-5,
        },
        "class_weights_fit_from_training_partition_only": {
            "binary": {str(key): value for key, value in binary_weights.items()},
            "multiclass_zero_based": {str(key): value for key, value in multi_weights.items()},
        },
        "binary": binary_training,
        "multiclass": multiclass_training,
        "training_duration_seconds": time.perf_counter() - started,
    }
    metrics["training_configuration"] = training_configuration

    versions = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    metadata = {
        "model_version": "nfi-retrained-lstm-v1",
        "training_date_utc": datetime.now(UTC).isoformat(),
        "dataset_file": dataset.path.name,
        "dataset_sha256": dataset_sha,
        "sample_count": int(len(dataset.labels)),
        "unique_group_count": int(len(np.unique(dataset.groups))),
        "feature_order": list(FEATURE_ORDER),
        "class_mapping": {str(code): value for code, value in CLASS_MAPPING.items()},
        "binary_label_definition": "Derived: FaultCause 1–6 = Fault (1); "
        "FaultCause 7 = Normal (0); this is not an independently recorded binary annotation.",
        "split_strategy": split_configuration,
        "random_seed": seed,
        "preprocessing_artifact": "preprocessing.joblib",
        "preprocessing_version": preprocessing["preprocessing_version"],
        "model_files": {"binary": binary_filename, "multiclass": multiclass_filename},
        "model_architectures": {
            "binary": "LSTM(7, return_sequences=True) -> Dropout(0.2) -> LSTM(7) "
            "-> Dropout(0.2) -> Dense(1, sigmoid)",
            "multiclass": "LSTM(6) -> Dense(7, softmax)",
        },
        "training_configuration": training_configuration,
        "package_versions": versions,
        "test_metrics": {"binary": binary_metrics, "multiclass": multiclass_metrics},
    }

    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_root / "split_summary.json").write_text(
        json.dumps(split_configuration, indent=2, allow_nan=False), encoding="utf-8"
    )
    (model_root / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_report(output_root / "MODEL_REPORT.md", metrics, metadata)
    return {"metrics": metrics, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-aware Ericsson KPI fault models.")
    parser.add_argument(
        "--dataset", type=Path, help="Path to db_fm_validation14.csv; searched if omitted"
    )
    parser.add_argument("--model-dir", type=Path, help="Model output directory (default: models/)")
    parser.add_argument(
        "--artifacts-dir", type=Path, help="Metrics output directory (default: artifacts/ml/)"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = run_training(
        args.dataset,
        model_dir=args.model_dir,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
