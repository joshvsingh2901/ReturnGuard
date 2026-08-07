# ReturnGuard

Predict whether a fashion e-commerce item will be returned.

**Current stage**: Stage 2 — XGBoost, Feature Engineering, Missing-Node Artifact Defense (complete)

---

## Project Goal

ReturnGuard predicts `P(item returned | information available at purchase time)` for individual item purchases on a fashion e-commerce platform.

The project is built to demonstrate strong ML engineering practices:
- leakage prevention
- proper train/validation/test methodology
- reproducible preprocessing
- feature engineering
- model evaluation and probability calibration
- explainability — planned Stage 3
- MLOps (MLflow) — planned Stage 4
- FastAPI model serving — planned Stage 5
- Next.js frontend — planned Stage 6

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
pip install -r requirements.txt
```

Requires Python 3.10+. The pickle files were created with pandas 1.x; a compatibility shim (`ml/data/compat.py`) handles loading them under pandas 2.x.

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

Outputs land in `reports/`: `stage1_metrics.json`, calibration plots, and the fitted model pipelines (all gitignored — regenerable). See [docs/stage1-baseline.md](docs/stage1-baseline.md) for full results and methodology.

---

## How to Run the Stage 2 Ablation Ladder

```bash
python scripts/train_stage2.py     # requires Stage 1's build_dataset.py to have run first
```

~30–40 minutes (a 6-config hyperparameter grid dominates the runtime). Trains XGBoost through an ablation ladder (contaminated diagnostic → honest donor-imputed baseline → +derived features → +frequency features → tuning) on the identical Stage 1 customer-grouped split, plus a leakage probe that measures whether the customer missing-node artifact is exploitable. Outputs land in `reports/`: `stage2_metrics.json`, a calibration plot, and the fitted feature pipeline + XGBoost model (all gitignored — regenerable). See [docs/stage2-gbdt.md](docs/stage2-gbdt.md) for full results and methodology.

---

## How to Run Tests

```bash
pytest tests/ -v
```

128 tests covering:
- Stage 0: raw file presence, DataFrame types, required columns, binary target, schema consistency, known duplicate-column defects
- Stage 1: leakage-safe feature schema, join correctness (row-count preservation, coverage rates), deterministic customer-grouped splitting, preprocessing (sentinel cleaning, unseen categories, missing-node handling), model behavior (probability validity, convergence), metrics (closed-form baseline checks, perfect-prediction fixtures), calibration binning, and cold-start slice evaluation
- Stage 2: donor imputation (joint sampling, determinism, train-fold-only fitting), derived price features (hand-calculated formulas, train-only statistics, unseen-category fallback), frequency encoding (train-fold-only counts, NaN-vs-zero semantics, no customer-ID frequency feature), the leakage probe (synthetic detectability regression tests), the three-way early-stopping split (byte-identical primary_val to Stage 1, fold disjointness), and XGBoost pipeline behavior (probability validity, determinism, early stopping)

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
│   └── stage2-gbdt.md        # Stage 2 methodology, results, and reasoning
├── ml/
│   ├── data/          # schema, loaders, joins, splits, pandas compat shim
│   ├── features/      # preprocessing (LR + GBDT), donor imputation, derived/frequency features
│   ├── models/        # trivial baselines, Logistic Regression, XGBoost
│   └── evaluation/    # metrics, calibration, cold-start slicing, leakage probe
├── notebooks/         # Exploratory notebooks (empty — logic lives in ml/ and scripts/)
├── reports/           # Generated metrics/plots/models (gitignored, regenerable)
├── scripts/
│   ├── audit_dataset.py   # Stage 0 dataset audit script
│   ├── build_dataset.py   # Stage 1 join + split builder
│   ├── train_baseline.py  # Stage 1 training + evaluation
│   └── train_stage2.py    # Stage 2 ablation ladder + leakage probe
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Current Limitation

No model beyond an untuned-hyperparameter XGBoost (A5 tuning found the default configuration already near the plateau — see below) has been trained. Stage 2 establishes a stronger leakage-conscious floor — not a fully optimized model, and no explainability tooling exists yet.

Key findings that shape both what Stage 1+2 built and what Stage 3 should do:

- **Target leakage, confirmed and worse than first suspected**: product-side return/sales aggregates (`returnsPerProduct`, `productReturnRate`, `salesPerProduct`) are byte-identical between the training and test node files — the training file's aggregate already contains test-period outcomes. Customer-side aggregates (`returnsPerCustomer`, `customerReturnRate`, `salesPerCustomer`) are recomputed per split (same-window leakage). All are excluded; see [docs/leakage-audit.md](docs/leakage-audit.md).
- **Target encoding evaluated and rejected**: without event timestamps, no fold construction can prove a target-encoding row precedes the row being scored — it would functionally reconstruct the banned `productReturnRate` under different bookkeeping. Fold-local, target-free frequency encoding was used instead.
- **The customer missing-node artifact is confirmed exploitable by trees, and now suppressed**: a leakage probe shows median/mode imputation makes `has_customer_node` almost perfectly detectable (probe AUC 0.9866); joint donor (hot-deck) imputation reduces this to near-chance (0.5248). The artifact itself was worth only +0.003 AUC when exploited — far less than legitimate feature engineering gained (+0.013).
- **Severe cold-start**: 71.9% of test events involve customers not seen during training. The primary validation split is customer-grouped specifically to measure this honestly, and the feature set uses no customer-history features so it works unchanged for new customers.
- **No timestamps**: chronological validation cannot be reconstructed from the dataset alone; the customer-grouped split is the practical substitute. This also means the Stage 2 frequency features carry an unfalsifiable within-window-exposure assumption.
- **Biased sample**: only customers with at least one return are included (~55% measured return rate vs. 20–40% typical for real e-commerce). No probability calibrator has been fit — deliberately, since fitting one now would calibrate to the wrong population.
- **Partial anonymization**: `brandDesc`/`productType` each leak one real, un-anonymized value (`Pull&Bear`, `Jeans`).

See [docs/stage1-baseline.md](docs/stage1-baseline.md) and [docs/stage2-gbdt.md](docs/stage2-gbdt.md) for full results.

---

## Planned Next Stage

**Stage 3 — Explainability and Calibration Measurement**

Stage 2's hyperparameter tuning plateaued (best grid config beat the default by only +0.0007 AUC, inside the noise threshold) while feature engineering moved the needle substantially (+0.013 AUC from derived + frequency features) — evidence that remaining headroom in this dataset is in features and understanding, not model capacity. Stage 3 should:

- Add SHAP (`TreeExplainer`) over the final Stage 2 model, both as an interpretability deliverable and as a second, independent leakage audit — attributions should concentrate on genuinely safe features, and neither missing-node pathway should surface as a top driver
- Measure calibration more rigorously and confront the returners-only population-shift problem directly rather than continuing to defer it — the population shift, not the calibrator's functional form, is the real question
- Still no calibrator fitting until the population-shift question has an answer; still no test-split access
