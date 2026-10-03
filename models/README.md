# Model artifact directory

The modernized training command writes local-only artifacts here:

- `binary_fault_detector.keras`
- `fault_cause_classifier.keras`
- `preprocessing.joblib`
- `model_metadata.json`

Generated model and preprocessing artifacts are ignored by Git until their provenance and
Ericsson-derived information are reviewed. The original HDF5 files remain untouched under
`artifacts/original/`. The four files above form the model bundle and are intentionally not
committed to Git. A fresh clone requires an approved model bundle supplied externally for
inference; without it, FastAPI can start but inference remains unavailable and the application is
degraded. Train locally with `python -m app.ml.train`. The FastAPI application initializes the
persisted inference layer, which loads these artifacts and transforms inputs with the
training-fitted scaler. It never fits preprocessing during inference.
