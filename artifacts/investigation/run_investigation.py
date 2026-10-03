"""Read-only inference and evaluation for recovered Ericsson artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import h5py
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "original" / "models"
DATA_PATH = ROOT / "db_fm_validation14.csv"
OUT_DIR = ROOT / "artifacts" / "investigation"
FEATURES = ["Retainability", "HOSR", "RSRP", "RSRQ", "SINR", "Throughput", "Distance"]
MODELS = {
    "multiclass": "LSTM MULTI8CLASSES THE PREFECT ONE.h5",
    "asma_binary": "asma.h5",
    "binary_candidate": "LSTM_binary_THE_PERFECT_ONE.h5",
}
EXPECTED_HASHES = {
    # Captured at the start of this investigation; no previous manifest was present.
    "asma.h5": "0ed336fd8199152fa3017dd12429b5747235ac0a687abbbd0e19ddc9a28ac0b5",
    "LSTM MULTI8CLASSES THE PREFECT ONE.h5": "ef257639cfe5194456968241d5dcdd6ffb2d76da45ac0d80708115c81fe36f84",
    "LSTM_binary_THE_PERFECT_ONE.h5": "8fba86559aae495e44678a4a86f9921d9bc7e59a8f7e3dda6d5492b2023f62b6",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_shape(value):
    try:
        return str(tuple(None if d is None else int(d) for d in value.shape))
    except Exception:
        return str(value)


def activation_name(layer):
    activation = getattr(layer, "activation", None)
    if activation is None:
        return None
    return getattr(activation, "__name__", str(activation))


def saved_hdf5_model_config(path):
    """Read serialized topology/weight counts without deserializing the model."""
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
        config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        layers = config.get("config", {}).get("layers", [])
        weights = []
        f["model_weights"].visititems(
            lambda name, obj: weights.append((name, int(np.prod(obj.shape))))
            if isinstance(obj, h5py.Dataset) else None
        )
    return {
        "input_shape_from_saved_config": next(
            (str(layer["config"].get("batch_input_shape")) for layer in layers if layer["config"].get("batch_input_shape")), None
        ),
        "layers_from_saved_config": [
            {
                "name": layer["config"].get("name"),
                "class": layer["class_name"],
                "units": layer["config"].get("units"),
                "activation": layer["config"].get("activation"),
                "return_sequences": layer["config"].get("return_sequences"),
                "batch_input_shape": layer["config"].get("batch_input_shape"),
            }
            for layer in layers
        ],
        "parameter_count_from_model_weight_tensors": int(sum(count for _, count in weights)),
        "model_weight_tensor_shapes": weights,
        "output_shape_inferred_from_saved_topology": (
            f"(None, {layers[-1]['config'].get('units')})" if layers and layers[-1]["class_name"] == "Dense" else None
        ),
    }


def model_info(model):
    return {
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "parameter_count": int(model.count_params()),
        "layers": [
            {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "input_shape": tensor_shape(layer.input) if hasattr(layer, "input") else None,
                "output_shape": tensor_shape(layer.output) if hasattr(layer, "output") else None,
                "activation": activation_name(layer),
                "parameters": int(layer.count_params()),
            }
            for layer in model.layers
        ],
    }


def binary_report(y_fault_positive, pred_fault_positive, probs):
    labels = ["Fault", "Normal"]
    p, r, f, support = precision_recall_fscore_support(
        y_fault_positive, pred_fault_positive, labels=[1, 0], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_fault_positive, pred_fault_positive)),
        "precision_fault": float(p[0]),
        "recall_sensitivity_fault": float(r[0]),
        "specificity_normal": float(r[1]),
        "f1_fault": float(f[0]),
        "confusion_matrix_true_rows_pred_columns_Fault_Normal": confusion_matrix(
            y_fault_positive, pred_fault_positive, labels=[1, 0]
        ).tolist(),
        "prediction_distribution": {
            "Fault": int(np.sum(pred_fault_positive == 1)),
            "Normal": int(np.sum(pred_fault_positive == 0)),
        },
        "probability_range": [float(np.min(probs)), float(np.max(probs))],
        "probability_nonfinite_count": int(np.size(probs) - np.isfinite(probs).sum()),
        "threshold": 0.5,
        "sigmoid_semantics_assumed_from_recovered_mapping": "output >= 0.5 => Normal (source binary labels 0=fault, 1=no fault)",
        "class_support_fault_normal": {"Fault": int(support[0]), "Normal": int(support[1])},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Guard against changes between the baseline captured for this run and loading.
    hashes_before = {name: sha256(MODEL_DIR / name) for name in EXPECTED_HASHES}
    if hashes_before != EXPECTED_HASHES:
        raise RuntimeError(f"Model hash changed since investigation baseline: {hashes_before}")

    data = pd.read_csv(DATA_PATH)
    # These two literal CSV headers include brackets in the recovered file.
    source_header_map = {"[Retainability": "Retainability", "FaultCause]": "FaultCause"}
    data = data.rename(columns=source_header_map)
    missing = [c for c in FEATURES + ["FaultCause"] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required named columns after explicit source-header mapping: {missing}; got {data.columns.tolist()}")
    feature_df = data.loc[:, FEATURES]
    x_raw = feature_df.to_numpy(dtype=np.float64)
    if x_raw.shape != (1137, 7):
        raise ValueError(f"Unexpected feature shape: {x_raw.shape}")
    y_multi = data["FaultCause"].to_numpy(dtype=np.int64)
    expected_distribution = {str(i): int((y_multi == i).sum()) for i in range(1, 8)}
    if expected_distribution != {"1": 39, "2": 88, "3": 103, "4": 237, "5": 207, "6": 64, "7": 399}:
        raise ValueError(f"Unexpected label distribution: {expected_distribution}")
    duplicate_label_counts = data.groupby(FEATURES, dropna=False)["FaultCause"].nunique()
    duplicate_group_sizes = data.groupby(FEATURES, dropna=False).size()

    # Historical application preprocessing reconstruction (not a training scaler).
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    x_hist = x_scaled.reshape((1137, 1, 7)).astype(np.float32)
    x_raw_sequence = x_raw.reshape((1137, 1, 7)).astype(np.float32)

    result = {
        "environment": {
            "python": os.sys.version,
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "h5py": __import__("h5py").__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "model_sha256_run_baseline_and_before_load": hashes_before,
        "dataset": {
            "path": "db_fm_validation14.csv",
            "rows": len(data),
            "feature_order": FEATURES,
            "literal_source_headers_mapped_in_memory": source_header_map,
            "feature_matrix_shape": list(x_raw.shape),
            "sequence_shape": list(x_hist.shape),
            "label_distribution": expected_distribution,
            "exact_duplicate_rows": int(data.duplicated().sum()),
            "exact_duplicate_feature_rows": int(feature_df.duplicated().sum()),
            "unique_feature_rows": int(feature_df.drop_duplicates().shape[0]),
            "duplicate_feature_groups": int((duplicate_group_sizes > 1).sum()),
            "rows_in_duplicate_feature_groups": int(duplicate_group_sizes[duplicate_group_sizes > 1].sum()),
            "duplicate_feature_groups_with_conflicting_labels": int((duplicate_label_counts > 1).sum()),
            "rows_in_duplicate_feature_groups_with_conflicting_labels": int(duplicate_group_sizes[duplicate_label_counts > 1].sum()),
            "faultcause_unique_values": sorted(int(v) for v in np.unique(y_multi)),
            "feature_label_means_by_class": data.groupby("FaultCause")[FEATURES].mean().to_dict(orient="index"),
        },
        "historical_application_preprocessing_reconstruction": {
            "scaler": "new StandardScaler fit on all 1,137 validation rows and applied to same rows",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "training_preprocessing_verified": False,
        },
        "models": {},
    }

    loaded = {}
    for key, filename in MODELS.items():
        path = MODEL_DIR / filename
        result.setdefault("saved_hdf5_config", {})[key] = saved_hdf5_model_config(path)
        try:
            model = tf.keras.models.load_model(path, compile=False)
            loaded[key] = model
            result["models"][key] = {"filename": filename, "load_compile_false": "success", **model_info(model)}
        except Exception as exc:
            result["models"][key] = {"filename": filename, "load_compile_false": "failed", "error": repr(exc)}
            # Keras 3 removed legacy LSTM.time_major and no longer accepts
            # batch_input_shape as a layer kwarg. This adapter removes only those
            # obsolete config keys while retaining weights, topology, and activations.
            class LegacyCompatibleLSTM(tf.keras.layers.LSTM):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("time_major", None)
                    kwargs.pop("batch_input_shape", None)
                    super().__init__(*args, **kwargs)

            try:
                model = tf.keras.models.load_model(
                    path,
                    compile=False,
                    custom_objects={"LSTM": LegacyCompatibleLSTM},
                )
                loaded[key] = model
                result["models"][key]["compatibility_load_compile_false"] = "success"
                result["models"][key]["compatibility_adapter"] = (
                    "Removed obsolete serialized LSTM time_major and batch_input_shape kwargs in memory; no HDF5 changes"
                )
                result["models"][key].update(model_info(model))
            except Exception as compat_exc:
                result["models"][key]["compatibility_load_compile_false"] = "failed"
                result["models"][key]["compatibility_load_error"] = repr(compat_exc)

    for key, model in loaded.items():
        try:
            pred = np.asarray(model.predict(x_hist, verbose=0))
            item = result["models"][key]
            item["predict"] = "success"
            item["prediction_shape"] = list(pred.shape)
            item["all_predictions_finite"] = bool(np.isfinite(pred).all())
            item["prediction_min"] = float(np.nanmin(pred))
            item["prediction_max"] = float(np.nanmax(pred))
            item["prediction_sample_first_5"] = pred[:5].tolist()
            if key == "multiclass":
                rowsum = pred.sum(axis=1)
                in_range = bool(np.all((pred >= -1e-6) & (pred <= 1.0 + 1e-6)))
                sums_approx_one = bool(np.allclose(rowsum, 1.0, atol=1e-3, rtol=1e-3))
                y_pred = np.argmax(pred, axis=1) + 1
                labels = list(range(1, 8))
                per_p, per_r, per_f, support = precision_recall_fscore_support(
                    y_multi, y_pred, labels=labels, zero_division=0
                )
                item["multiclass_probability_sanity"] = {
                    "seven_columns": bool(pred.ndim == 2 and pred.shape[1] == 7),
                    "all_values_approximately_0_to_1": in_range,
                    "all_rows_sum_approximately_1": sums_approx_one,
                    "row_sum_min": float(rowsum.min()),
                    "row_sum_max": float(rowsum.max()),
                    "entropy_mean_nats": float(np.mean(-np.sum(np.clip(pred, 1e-12, 1) * np.log(np.clip(pred, 1e-12, 1)), axis=1))),
                }
                item["evaluation"] = {
                    "name": "Controlled reconstruction using recovered model + historical application preprocessing",
                    "accuracy": float(accuracy_score(y_multi, y_pred)),
                    "macro_precision": float(precision_recall_fscore_support(y_multi, y_pred, average="macro", zero_division=0)[0]),
                    "macro_recall": float(recall_score(y_multi, y_pred, average="macro", zero_division=0)),
                    "macro_f1": float(f1_score(y_multi, y_pred, average="macro", zero_division=0)),
                    "weighted_precision": float(precision_recall_fscore_support(y_multi, y_pred, average="weighted", zero_division=0)[0]),
                    "weighted_recall": float(recall_score(y_multi, y_pred, average="weighted", zero_division=0)),
                    "weighted_f1": float(f1_score(y_multi, y_pred, average="weighted", zero_division=0)),
                    "confusion_matrix_labels_1_to_7": confusion_matrix(y_multi, y_pred, labels=labels).tolist(),
                    "per_class": {
                        str(label): {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f[i]), "support": int(support[i])}
                        for i, label in enumerate(labels)
                    },
                    "prediction_class_distribution": {str(label): int(np.sum(y_pred == label)) for label in labels},
                }
                raw_pred = np.asarray(model.predict(x_raw_sequence, verbose=0))
                item["raw_input_diagnostic_only"] = {
                    "label": "Diagnostic only — raw feature inference",
                    "prediction_shape": list(raw_pred.shape),
                    "all_predictions_finite": bool(np.isfinite(raw_pred).all()),
                    "prediction_min": float(np.nanmin(raw_pred)),
                    "prediction_max": float(np.nanmax(raw_pred)),
                    "probability_row_sum_min": float(np.min(raw_pred.sum(axis=1))),
                    "probability_row_sum_max": float(np.max(raw_pred.sum(axis=1))),
                }
            else:
                # Archived binary labels are 0=fault and 1=no fault; invert to derived Fault-positive target.
                if pred.ndim == 2 and pred.shape[1] == 1:
                    probs = pred[:, 0]
                elif pred.ndim == 1:
                    probs = pred
                else:
                    raise ValueError(f"Expected scalar binary sigmoid output; got {pred.shape}")
                y_fault_positive = (y_multi != 7).astype(np.int64)
                # The archived binary source's encoded class 0 means Fault and 1 means Normal.
                pred_fault_positive = (probs < 0.5).astype(np.int64)
                item["evaluation_target"] = "Derived binary target: FaultCause 1–6=Fault, 7=Normal (not an independent binary annotation)"
                item["evaluation"] = binary_report(y_fault_positive, pred_fault_positive, probs)
                raw_pred = np.asarray(model.predict(x_raw_sequence, verbose=0)).reshape(-1)
                item["raw_input_diagnostic_only"] = {
                    "label": "Diagnostic only — raw feature inference",
                    "prediction_shape": list(np.asarray(raw_pred).shape),
                    "all_predictions_finite": bool(np.isfinite(raw_pred).all()),
                    "prediction_min": float(np.nanmin(raw_pred)),
                    "prediction_max": float(np.nanmax(raw_pred)),
                }
        except Exception as exc:
            result["models"][key]["predict"] = "failed"
            result["models"][key]["prediction_error"] = repr(exc)

    # Diagnostic compiled load after compile=False attempt. No model files are altered.
    for key, filename in MODELS.items():
        try:
            compiled = tf.keras.models.load_model(MODEL_DIR / filename)
            result["models"][key]["compiled_load"] = "success"
            result["models"][key]["compiled_optimizer"] = compiled.optimizer.__class__.__name__ if hasattr(compiled, "optimizer") else None
        except Exception as exc:
            result["models"][key]["compiled_load"] = "failed"
            result["models"][key]["compiled_load_error"] = repr(exc)

    hashes_after = {name: sha256(MODEL_DIR / name) for name in EXPECTED_HASHES}
    result["model_sha256_after_inference"] = hashes_after
    result["models_unchanged_during_run"] = hashes_after == hashes_before
    output = OUT_DIR / "inference_results.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
