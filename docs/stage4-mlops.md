# Stage 4 — Frozen A3 lifecycle and reproducibility

Stage 4 operationalises the closed `returnguard-a3-v1` model. It does not
search features, tune parameters, fit calibration, evaluate A4, or rescore the
official test split.

## Local tracking and registry

ReturnGuard uses local MLflow with SQLite metadata at `.mlflow/mlflow.db` and
filesystem artifacts under `.mlflow/artifacts`. The experiment is
`returnguard-frozen-lifecycle`; the registered model is `returnguard-a3`.
Passing rebuilds may be registered, but model-of-record promotion is an
explicit governance action:

```bash
.venv/bin/python scripts/promote_model.py --run-id <run-id> --model-version <version>
```

The `model-of-record` alias is never assigned from a performance comparison.

## Reproduction workflow

```bash
.venv/bin/python scripts/reproduce_model.py --profile full
```

The workflow verifies the three training raw-file hashes, validates the frozen
manifest, rebuilds the deterministic development split, checks its metrics,
fits final A3 on all training rows for exactly 659 rounds, saves and reloads a
composite preprocessing-plus-model artifact, checks safe training-derived
reference predictions, writes a machine-readable report, and logs MLflow
provenance plus `pip freeze`.

The first bootstrap is the only command that writes the small reference fixture:

```bash
.venv/bin/python scripts/reproduce_model.py --profile full --bootstrap-reference --register
```

The manifest-generation helper is intentionally distinct because it performs
one historical schema inspection of all six raw files:

```bash
.venv/bin/python scripts/create_dataset_manifest.py --include-historical-test-schema
```

Normal reproduction never reads test files.

## Reproducibility standard

Exact checks cover raw-data hashes, split definitions, the ordered 41-feature
manifest, frozen configuration, donor-imputation policy, no-frequency policy,
and artifact reload. Semantic metric tolerances are ROC-AUC ±0.003, log loss
±0.005, Brier ±0.003, and ECE ±0.01. Training-reference probabilities require
mean absolute delta ≤1e-6 and maximum delta ≤1e-5; reload prediction delta must
be ≤1e-7.

XGBoost and joblib binary hashes are recorded but are not required to be
identical across platforms. Behavioral equivalence within these controls is the
governed reproducibility standard.

## Test governance

The one official test score is historical evidence. To import it into MLflow
without touching data:

```bash
.venv/bin/python scripts/import_historical_final_evaluation.py
```

The exceptional test evaluator requires all of the following:

```bash
RETURNGUARD_ALLOW_FINAL_TEST_RERUN=1 .venv/bin/python scripts/run_final_test_evaluation.py \
  --allow-final-test-rerun --reason "documented evaluator defect"
```

It writes an attempt-specific directory and never overwrites the canonical
final reports. This is for a demonstrated evaluator defect, not model
development.

## Artifact policy and CI

Git tracks manifests, reference metadata, source, and small evidence. MLflow
owns rebuilt binaries; raw/processed data, MLflow state, large SHAP outputs,
and temporary reports remain ignored. CI runs Ruff and synthetic lifecycle
tests only; it excludes raw training, historical test, and full-rebuild tests.
