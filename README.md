# ReturnGuard

Predict whether a fashion e-commerce item will be returned.

**Current stage**: Stage 6 Next.js frontend complete for frozen `returnguard-a3-v1`

---

## Project Goal

ReturnGuard predicts `P(item returned | information available at purchase time)` for individual item purchases on a fashion e-commerce platform.

The project is built to demonstrate strong ML engineering practices:
- leakage prevention
- proper train/validation/test methodology
- reproducible preprocessing
- feature engineering
- model evaluation and probability calibration
- explainability and feature governance
- MLOps (MLflow) — Stage 4 complete
- FastAPI model serving — Stage 5 complete
- Next.js frontend — Stage 6 complete

---

## Dataset

**ASOS GraphReturns** (public research dataset, OSF)

Six pickle files representing a bipartite customer–product graph:

| File | Description |
|------|-------------|
| `event_table_training.p` | 1.37M training purchase events |
| `event_table_testing.p` | 1.46M test purchase events |
| `customer_nodes_training.p` | 777K customer attribute rows (training period) |
| `customer_nodes_testing.p` | 826K customer attribute rows (test period) |
| `product_nodes_training.p` | 411K product variant attribute rows (training period) |
| `product_nodes_testing.p` | 412K product variant attribute rows (test period) |

**Important**: the dataset only includes customers who have made at least one return. The measured return rate (~55%) is higher than real-world rates and the model's predicted probabilities will need calibration before merchant-wide deployment.

---

## How to Place Raw Data

1. Download `c793h-osfstorage-archive.zip` from OSF
2. Extract all six `.p` files into `data/raw/`:

```
data/raw/event_table_training.p
data/raw/event_table_testing.p
data/raw/customer_nodes_training.p
data/raw/customer_nodes_testing.p
data/raw/product_nodes_training.p
data/raw/product_nodes_testing.p
```

Raw data files are excluded from git via `.gitignore`.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -c constraints-stage5.txt
```

The validated serving environment uses Python 3.9.6, XGBoost 2.1.4, SHAP
0.47.2, FastAPI 0.128.8, and local MLflow; the constraint file makes the frozen lifecycle reproducible. The pickle files
were created with pandas 1.x; a compatibility shim (`ml/data/compat.py`) handles
loading them under pandas 2.x.

---

## How to Run the Audit

```bash
python scripts/audit_dataset.py
```

Loads all six raw files, computes structural statistics, and prints a full summary. Does not mutate any data.

---

## How to Run the Stage 1 Baseline

```bash
python scripts/build_dataset.py    # join event/customer/product tables, compute splits
python scripts/train_baseline.py   # train trivial baselines + LR-A + LR-B, write reports/
```

Outputs land in `reports/`: metrics JSON and curated calibration plots are retained as portfolio evidence; fitted model pipelines are gitignored and regenerable. See [docs/stage1-baseline.md](docs/stage1-baseline.md) for full results and methodology.

---

## How to Run the Stage 2 Ablation Ladder

```bash
python scripts/train_stage2.py     # requires Stage 1's build_dataset.py to have run first
```

~30–40 minutes (a 6-config hyperparameter grid dominates the runtime). Trains XGBoost through an ablation ladder (contaminated diagnostic → donor-imputed baseline → +derived features → +frequency features → tuning) on the Stage 1 customer-grouped development harness, plus a leakage probe that measures whether the customer missing-node artifact is exploitable. Metrics JSON and curated plots are retained; fitted binaries are gitignored and regenerable. See [docs/stage2-gbdt.md](docs/stage2-gbdt.md) for full results and methodology.

---

## How to Run Tests

```bash
ruff check .
pytest -m "not raw_training_data and not historical_test_data and not full_rebuild and not local_registry"
```

The full suite covers:
- Stage 0: raw file presence, DataFrame types, required columns, binary target, schema consistency, known duplicate-column defects
- Stage 1: leakage-safe feature schema, join correctness (row-count preservation, coverage rates), deterministic customer-grouped splitting, preprocessing (sentinel cleaning, unseen categories, missing-node handling), model behavior (probability validity, convergence), metrics (closed-form baseline checks, perfect-prediction fixtures), calibration binning, and cold-start slice evaluation
- Stage 2: donor imputation (joint sampling, determinism, train-fold-only fitting), derived price features (hand-calculated formulas, train-only statistics, unseen-category fallback), frequency encoding (train-fold-only counts, NaN-vs-zero semantics, no customer-ID frequency feature), the leakage probe (synthetic detectability regression tests), the three-way early-stopping split (byte-identical primary_val to Stage 1, fold disjointness), and XGBoost pipeline behavior (probability validity, determinism, early stopping)
- Stage 3: deterministic representative SHAP sampling and grouping, raw-margin additivity (including early-stopping tree-range alignment), explanation privacy contracts, feature governance, calibration diagnostics, and documented evaluation-history controls
- Stage 5: FastAPI request validation, safe errors, vectorized inference, no-profile donor-routing controls, public explanation privacy, and an opt-in local-registry smoke test
- Stage 6: Next.js synthetic-scenario frontend, API readiness state, cautious score/explanation presentation, CORS allowlisting, and data-free client tests

Tests requiring real ASOS training data are marked `raw_training_data`; tests
that inspect the historical test files are additionally marked
`historical_test_data`. They are intentionally excluded from normal lifecycle
and CI commands.

## How to Run the Stage 5 API

Stage 5 serves the local MLflow model-of-record only. Create it first with the
governed Stage 4 rebuild/registration workflow, then explicitly assign the
`model-of-record` alias as documented in [docs/stage4-mlops.md](docs/stage4-mlops.md).
Normal API operation never loads training data or official test labels.

```bash
.venv/bin/uvicorn backend.main:app --reload
```

Swagger/OpenAPI is available at `http://127.0.0.1:8000/docs`. The primary
endpoints are `GET /health`, `GET /ready`, `GET /model`, `POST /predict`,
`POST /predict/batch`, and `POST /explain`.

```json
{
  "customer_context_key": "opaque-stable-token",
  "customer_profile": {
    "yearOfBirth": 1988,
    "isMale": false,
    "shippingCountry": "Country_A",
    "premier": true
  },
  "product_profile": {
    "productType": "Jeans",
    "brandDesc": "Brand_A",
    "avgGbpPrice": 54.99,
    "avgDiscountValue": 15.0
  }
}
```

The response includes a deterministic 0–100 `risk_score`, data-quality
context, safe model provenance, and the required methodology warning. It does
not expose raw probabilities, SHAP values, feature names, or donor attributes.
These are **dataset-conditional return-risk scores from a returner-enriched
research sample, not merchant-wide calibrated return probabilities**. See
[docs/stage5-api.md](docs/stage5-api.md) for the full contract.

## How to Run the Stage 6 Frontend

Start the Stage 5 API first, with an existing local model-of-record, then run:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. The frontend is intentionally a synthetic,
single-event demonstrator: it calls the frozen API only and never accesses raw
ASOS data, MLflow, batch inference, or official test evaluation. See
[docs/stage6-frontend.md](docs/stage6-frontend.md) for its API/CORS contract,
score interpretation, and verification commands.

To test an already-created local model-of-record explicitly:

```bash
.venv/bin/pytest tests/api/test_local_registry_integration.py -m local_registry -q
```

---

## How to Rebuild the Frozen Model

The normal lifecycle command verifies only the three official **training** raw
files, recreates frozen A3, checks validation and reference predictions, saves
a composite inference artifact to local MLflow, and never imports the official
test evaluator:

```bash
.venv/bin/python scripts/reproduce_model.py --profile full
```

The first governed rebuild establishes the safe training-derived reference
fixture once:

```bash
.venv/bin/python scripts/reproduce_model.py --profile full --bootstrap-reference --register
```

Model registration is separate from explicit model-of-record promotion. See
[docs/stage4-mlops.md](docs/stage4-mlops.md).

---

## How to Run Stage 3

```bash
.venv/bin/python scripts/run_stage3.py
```

This re-fits the frozen A3 and A4 development configurations solely because
Stage 3 repairs the donor-imputation inference-order defect; it does not tune
them. It never loads or evaluates official ASOS test model performance. The
script produces reproducible development-only SHAP, calibration, missingness,
and governance reports in `reports/`; fitted binaries are ignored because they
are regenerable.

---

## Repository Structure

```
ReturnGuard/
├── data/
│   ├── raw/          # Raw pickle files (gitignored)
│   └── processed/    # Joined training frame + splits (gitignored, regenerable)
├── docs/
│   ├── dataset-audit.md      # Full dataset inspection report (Stage 0, corrected in Stage 1)
│   ├── leakage-audit.md      # Feature-by-feature leakage classification
│   ├── problem-definition.md # Formal problem statement
│   ├── stage1-baseline.md    # Stage 1 methodology, results, and reasoning
│   ├── stage2-gbdt.md        # Stage 2 methodology, results, and reasoning
│   ├── stage3-methodology.md # Stage 3 methods, results, and decisions
│   ├── stage5-api.md          # Stage 5 frozen-model serving contract
│   ├── stage6-frontend.md     # Stage 6 Next.js consumer and integration contract
│   ├── model-card.md         # Model-of-record constraints and intended use
│   └── evaluation-history.md # Test-inspection and validation-use disclosure
├── ml/
│   ├── data/          # schema, loaders, joins, splits, pandas compat shim
│   ├── features/      # preprocessing (LR + GBDT), donor imputation, derived/frequency features
│   ├── explainability/ # deterministic sampling, Tree SHAP, explanation contracts
│   ├── governance/    # feature manifest and frozen-model specification
│   ├── models/        # trivial baselines, Logistic Regression, XGBoost
│   └── evaluation/    # metrics, calibration, cold-start slicing, leakage probe
├── artifacts/
│   ├── manifests/     # Git-tracked data and frozen-model provenance
│   └── reference/     # small training-derived prediction reference fixture
├── notebooks/         # Exploratory notebooks (empty — logic lives in ml/ and scripts/)
├── reports/           # Generated metrics/plots/models (gitignored, regenerable)
├── backend/            # Stage 5 FastAPI request contracts and frozen A3 serving adapter
├── frontend/           # Stage 6 Next.js frozen-model demonstrator
├── scripts/
│   ├── audit_dataset.py   # Stage 0 dataset audit script
│   ├── build_dataset.py   # Stage 1 join + split builder
│   ├── train_baseline.py  # Stage 1 training + evaluation
│   ├── train_stage2.py    # Stage 2 ablation ladder + leakage probe
│   └── run_stage3.py      # Stage 3 development-only analysis and governance
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
├── constraints-stage4.txt
└── constraints-stage5.txt
```

---

## Current Limitations

The one predeclared held-out evaluation of frozen A3 is complete. It confirms
development-consistent overall performance within the ASOS dataset family, not
merchant deployment readiness. A4 remains an experimental challenger. The
project has no temporal validation, merchant-representative labels, or
point-in-time product exposure data.

Key findings that shape the model-of-record boundary:

- **Target leakage, confirmed and worse than first suspected**: product-side return/sales aggregates (`returnsPerProduct`, `productReturnRate`, `salesPerProduct`) are byte-identical between the training and test node files — the training file's aggregate already contains test-period outcomes. Customer-side aggregates (`returnsPerCustomer`, `customerReturnRate`, `salesPerCustomer`) are recomputed per split (same-window leakage). All are excluded; see [docs/leakage-audit.md](docs/leakage-audit.md).
- **Target encoding evaluated and rejected**: without event timestamps, no fold construction can prove a target-encoding row precedes the row being scored — it would functionally reconstruct the banned `productReturnRate` under different bookkeeping. Fold-local, target-free frequency encoding was used instead.
- **The customer missing-node artifact is confirmed exploitable by trees, and now suppressed**: a leakage probe shows median/mode imputation makes `has_customer_node` almost perfectly detectable (probe AUC 0.9866); joint donor (hot-deck) imputation reduces this to near-chance (0.5248). The artifact itself was worth only +0.003 AUC when exploited — far less than legitimate feature engineering gained (+0.013).
- **Severe cold-start**: 71.9% of test events involve customers not seen during training. The primary validation split is customer-grouped specifically to measure this honestly, and the feature set uses no customer-history features so it works unchanged for new customers.
- **No timestamps**: chronological validation cannot be reconstructed from the dataset alone; the customer-grouped split is the practical substitute. This also means the Stage 2 frequency features carry an unfalsifiable within-window-exposure assumption.
- **Biased sample**: only customers with at least one return are included (~55% measured return rate vs. 20–40% typical for real e-commerce). No probability calibrator has been fit — deliberately, since fitting one now would calibrate to the wrong population.
- **Partial anonymization**: `brandDesc`/`productType` each leak one real, un-anonymized value (`Pull&Bear`, `Jeans`).

See [docs/stage1-baseline.md](docs/stage1-baseline.md),
[docs/stage2-gbdt.md](docs/stage2-gbdt.md), and
[docs/stage3-methodology.md](docs/stage3-methodology.md) for full results.

---

## Stage 3 Outcome

**Stage 3 — Explainability, Governance, and Calibration Policy**

Stage 2's hyperparameter tuning plateaued (best grid config beat the default by only +0.0007 AUC, inside the noise threshold) while feature engineering moved the needle substantially (+0.013 AUC from derived + frequency features) — evidence that remaining work was in model understanding and governance rather than model capacity. Stage 3:

- verified Tree SHAP against the exact early-stopped scoring tree range;
- confirmed A3 and A4 development reproductions after repairing batch/order-
  dependent customer donor selection;
- retained A3 as the conservative candidate because A4’s large frequency
  contribution is temporally unverifiable without timestamps;
- measured, but did not fit, calibration because the returner-enriched sample
  cannot yield merchant-wide probabilities; and
- did not itself compute official ASOS test model performance.

The predeclared evaluation was subsequently executed on frozen A3 only. Its
results are documented in [docs/final-test-evaluation.md](docs/final-test-evaluation.md);
they do not reopen model selection, calibration fitting, feature engineering,
or A4 comparison.

The official test files were inspected during dataset auditing, including aggregate label statistics and structure. The one frozen-A3 performance evaluation is complete and cannot reopen fitting, model selection, calibration, or A4 comparison. Stage 2 selected configurations with its inner early-stop fold, while repeatedly calculating primary-validation metrics; those validation results are developmental rather than pristine confirmatory evidence. See [docs/evaluation-history.md](docs/evaluation-history.md) and [docs/model-card.md](docs/model-card.md).
