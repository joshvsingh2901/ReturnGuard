# Stage 1 — Leakage-Safe Baseline

**Status**: Complete. **Last run**: 2026-08-06, on the full 1,369,133-row training event table.

This document records the Stage 1 methodology, results, and the reasoning
behind every non-obvious choice. It is the reference for comparing any
later model (Stage 2 feature engineering, XGBoost, etc.) back to a known,
trustworthy floor.

---

## 1. Feature Sets

Two Logistic Regression variants, trained and evaluated identically except
for two columns:

| | LR-A (strict) | LR-B (with price) |
|---|---|---|
| `yearOfBirth` | ✓ | ✓ |
| `isMale` | ✓ | ✓ |
| `premier` | ✓ | ✓ |
| `shippingCountry` | ✓ | ✓ |
| `productType` | ✓ | ✓ |
| `brandDesc` | ✓ | ✓ |
| `avgGbpPrice` | — | ✓ |
| `avgDiscountValue` | — | ✓ |
| Encoded feature count | 36 | 38 |

All eight raw columns are classified SAFE or SUSPICIOUS in
`docs/leakage-audit.md`; none is UNSAFE. The full leakage-exclusion list
(`salesPerCustomer`, `returnsPerCustomer`, `customerReturnRate`,
`salesPerProduct`, `returnsPerProduct`, `productReturnRate`, all
`*_level_return_code_*` columns, and the supplied `Country_*`/`Brand_*`/
`productType_*` one-hot dummies) is codified as `LEAKY_COLUMNS` in
`ml/data/schema.py` and enforced by `tests/test_schema.py` and
`tests/test_joins.py` — the feature set cannot silently drift to include a
leaky column without a test failing.

Supplied one-hot dummy columns are **not used** even though they are
themselves leakage-safe: they use inconsistent construction (`Country_*`
and `productType_*` are full one-hots; `Brand_*` drops one reference
category, which turns out to be the un-anonymized `Pull&Bear`, not a coded
value). Stage 1 encodes the six raw categorical/numeric/binary columns
itself with a single consistent `OneHotEncoder`, which also gives correct
`handle_unknown="ignore"` behavior for cold-start customers/products.

`hash(supplierRef)` and `hash(productID)` are excluded as high-cardinality
identifiers that would need target encoding — deferred to Stage 2.

---

## 2. Leakage Exclusions

See `docs/leakage-audit.md` for the full column-by-column audit. Summary
of what changed in Stage 1 versus the original Stage 0 audit:

- **Product-side return/sales aggregates are confirmed global constants**,
  byte-identical between the training and test product node files for all
  405,505 shared variants. This means `productReturnRate` in the
  *training* file already contains test-period outcomes — a stronger
  leakage path than originally documented. `salesPerCustomer` and
  `salesPerProduct` were reclassified from SUSPICIOUS to UNSAFE and are
  excluded outright.
- **`avgGbpPrice`/`avgDiscountValue` are also confirmed global constants**,
  but their mean is not target-derived, so they remain SUSPICIOUS rather
  than UNSAFE — used in LR-B as a documented ablation, never assumed safe
  by default.
- **`premier` is confirmed static** (99.99% stable across the customer
  node files for shared customers) — the Stage 0 caveat about it being an
  end-of-period snapshot is resolved; it is SAFE with no reservation.

---

## 3. Join Strategy

`event_table_training.p` LEFT JOIN `customer_nodes_training.p` LEFT JOIN
`product_nodes_training.p`, both on training files only
(`ml/data/joins.py`). Testing files are never imported by any Stage 1
code path — `ml/data/loaders.py` raises `ValueError` if a `*_testing.p`
filename is requested.

Node ID uniqueness is validated before joining (`_validate_unique_ids`):
both node tables have zero duplicate IDs, so a left join is guaranteed not
to multiply event rows. This is asserted again after the join
(`len(joined) == n_events_before`) and covered by
`tests/test_joins.py::test_join_preserves_event_row_count`.

Column selection happens **before** the join: `ml/data/loaders.py` selects
only the ID column plus the named safe feature columns from each node
table. This means the duplicate `customerId_level_return_code_D` /
`variantID_level_return_code_D` source columns (a confirmed upstream
naming defect — the two same-named columns hold different data, means
0.0297 vs. 0.3972 for the customer-side pair) are never reachable by the
pipeline; there is no need for positional (`iloc`) workarounds anywhere in
Stage 1 code.

---

## 4. Missing-Node Handling

Measured on the joined training frame (`scripts/build_dataset.py` output):

| | Coverage | Missing-row target rate | Joined-row target rate | Gap |
|---|---|---|---|---|
| Customer node | 94.77% | 64.57% | 54.80% | **9.8 pts** |
| Product node | 65.34% | 55.45% | 55.23% | **0.2 pts** |

The customer-side gap is large enough to be a genuine predictive signal
if exposed — but it is treated as a **dataset-construction artifact, not
a generalizable feature**, because there is no reason a real e-commerce
system would have "missing customer profile" correlate this strongly with
return behavior; it far more plausibly reflects how the two node files
were extracted/sampled by the dataset authors. Exposing it would inflate
validation metrics on artifacts of this specific dataset in a way that
would not transfer to a production join against a live customer table.

Consequently:

- **Customer-side missing values** (`yearOfBirth`, `isMale`, `premier`,
  `shippingCountry`): median/mode-imputed with `add_indicator=False` — no
  `has_customer_node` feature is exposed to the model.
- **Product-side missing values** (`productType`, `brandDesc`): filled
  with an explicit `"__MISSING__"` category token at join time
  (`ml/data/joins.py`), rather than mode-imputed into the modal brand/type
  — because product-side missingness is target-neutral, representing it
  explicitly imports no artifact, and mode-imputing a third of all rows
  into the already-dominant category (`Brand_K` at 56.8%, `productType_K`
  at 45.0%) would badly distort the model's brand/type coefficients.
  `avgGbpPrice`/`avgDiscountValue` (LR-B only) are median-imputed for the
  same missing rows.

**A subtlety caught during implementation**: `shippingCountry` is a
customer-side categorical, but a naive `OneHotEncoder` with no imputer
step will silently create its own `"nan"` category for the 5.2% of rows
with a missing customer node — which is exactly the `has_customer_node`
signal this design deliberately excludes, just re-introduced through the
back door for one specific feature. `ml/features/preprocessing.py` fixes
this by mode-imputing `shippingCountry` in its own `cat_cust` branch
before one-hot encoding, consistent with how `isMale`/`premier` are
handled. Product-side categoricals (`cat_prod` branch) get no imputer,
since `ml.data.joins` already guarantees they are never NaN. This was
caught by inspecting the LR-A feature list during development (it
contained `cat__shippingCountry_nan`) — not by a pre-written test; a
regression test was not added for this specific case since the fix is
now structural (the customer/product categorical split makes the bug
impossible to reintroduce without also removing the imputer).

**A diagnostic missing-node indicator ablation was considered but not
run**: adding `has_customer_node`/`has_product_node` as explicit features
would require rerunning the full LR-A/LR-B comparison a second time with
no bearing on the headline result, for a finding already established more
directly by the coverage/target-rate table above. Skipped as
disproportionate to its value for this stage.

---

## 5. Validation Split

**Primary — customer-grouped, deterministic** (`ml/data/splits.py`):
every event for a given customer is assigned to exactly one side via
`hashlib.md5(str(customer_id))` mod 100, bucketed at an 80/20 cutoff.
Python's builtin `hash()` is not used, since it is randomized per-process
(`PYTHONHASHSEED`) and would not be reproducible across runs.

- Train: 1,094,622 rows (648,168 customers). Val: 274,511 rows (162,069
  customers).
- **Zero customer overlap** between train and validation — verified by
  `tests/test_splits.py` and asserted at runtime in
  `scripts/build_dataset.py`.
- Val fraction: 20.05%. Train-fold prevalence 0.5526, val-fold prevalence
  0.5550.
- Deterministic across repeated runs and invariant to row-order shuffling
  (verified in `tests/test_splits.py` and empirically: two full pipeline
  reruns produced metrics identical to 1e-9 tolerance — in practice exact).

This split was chosen as primary specifically because Stage 2 is expected
to add customer-history features; a random split would silently leak
those across the train/validation boundary the moment they're introduced,
while the customer-grouped split remains valid unchanged. It also matches
the real cold-start character of the provided test split (71.9% of test
events involve customers unseen in training — see
`docs/dataset-audit.md`).

**Secondary — random stratified, diagnostic only**
(`random_stratified_split`): a plain `sklearn.model_selection.
train_test_split` at the event level, stratified on the target. Used
*exclusively* to populate the full four-quadrant cold-start table (see
§8) — customers and products can appear on both sides under this split,
which is the only way "known customer" and "known product" cases exist
at all in a within-training-data evaluation. Never used for a headline
metric.

---

## 6. Preprocessing Pipeline

`ml/features/preprocessing.py`, a single `sklearn.compose.
ColumnTransformer` inside the model `Pipeline`, fit only on the training
fold:

| Branch | Columns | Steps |
|---|---|---|
| `num` | `yearOfBirth` (+ `avgGbpPrice`, `avgDiscountValue` for LR-B) | median impute → standard-scale |
| `bin` | `isMale`, `premier` | most-frequent impute |
| `cat_cust` | `shippingCountry` | most-frequent impute → one-hot (`handle_unknown="ignore"`) |
| `cat_prod` | `productType`, `brandDesc` | one-hot (`handle_unknown="ignore"`); no imputer — see §4 |

`yearOfBirth == 1900` is converted to NaN (`clean_year_of_birth`) **before**
this pipeline runs — a fixed data-cleaning step applied to a confirmed
sentinel, not something the pipeline learns. No other `yearOfBirth` values
are clipped or altered, per explicit instruction for this stage.

`get_feature_names_out()` is persisted implicitly via
`ml.features.preprocessing.get_feature_names()` and written into
`reports/stage1_metrics.json` (`lr_a_feature_names`, `lr_b_feature_names`),
so every coefficient in the fitted model remains traceable to a specific
input column and category.

Fitted pipelines are saved with `joblib` to
`reports/lr_a_pipeline.joblib` / `reports/lr_b_pipeline.joblib`
(gitignored — regenerable via `scripts/train_baseline.py`).

---

## 7. Baseline Definitions

Both trivial baselines wrap `sklearn.dummy.DummyClassifier` rather than
reimplementing prior/most-frequent logic, so every model in Stage 1
(trivial and LR) shares the same `fit`/`predict_proba` interface and is
evaluated with identical code (`ml/models/baselines.py`).

- **Prevalence baseline** (`strategy="prior"`): predicts the **train-fold**
  target prevalence for every row, never the validation or full-dataset
  rate.
- **Majority-class baseline** (`strategy="most_frequent"`): always predicts
  the train-fold majority class with probability 1.0.

---

## 8. Logistic Regression Settings

`ml/models/logistic.py`, both LR-A and LR-B:

| Setting | Value | Rationale |
|---|---|---|
| penalty | L2 | Untuned default |
| `C` | 1.0 | Untuned default; hyperparameter tuning explicitly deferred |
| `class_weight` | `None` | Prevalence is ~55/45 — close enough to balanced that `'balanced'` would distort predicted probabilities, directly conflicting with the project's eventual calibration goal |
| solver | `lbfgs` | Default; well-conditioned problem at ~38 features |
| `max_iter` | 1000 | With a hard convergence check |

**Convergence is verified, not assumed**: `check_convergence()` raises
`RuntimeError` if `n_iter_ >= max_iter`. Both LR-A (21 iterations) and
LR-B (25 iterations) converged well within the 1000-iteration budget on
the full run.

---

## 9. Results — Primary Split (Headline)

Validation set: 274,511 rows, 162,069 held-out customers, prevalence 0.5550.

| Model | ROC-AUC | Log Loss | Brier | LogLoss Skill | Brier Skill | ECE |
|---|---|---|---|---|---|---|
| Prevalence baseline | 0.5000 | 0.6871 | 0.2470 | — | — | — |
| Majority-class baseline | 0.5000† | 16.041 | 0.4450 | −22.34 | −0.802 | — |
| **LR-A** (no price) | **0.6300** | **0.6608** | **0.2344** | **0.0382** | **0.0511** | **0.0062** |
| **LR-B** (with price) | **0.6381** | **0.6577** | **0.2328** | **0.0428** | **0.0575** | **0.0119** |

† ROC-AUC is undefined for a model with zero predictive variation; sklearn
returns 0.5 by convention since a constant score ranks all pairs as ties.

**Majority-class baseline's log loss (16.04) is intentionally extreme**: it
predicts probability 1.0 for every row, so every one of the 274,511 ×
0.445 ≈ 122,157 actually-negative validation rows contributes a
near-infinite per-row log loss (clipped by sklearn's internal epsilon).
This is the concrete illustration of why accuracy/F1 alone are not
meaningful here — the majority baseline achieves the *same* accuracy
(0.5550) and F1 (0.7138) as a naive threshold-0.5 read of LR-A/LR-B would
suggest is "pretty good," while being a literally zero-information model.

Both LR variants clear the trivial baselines on every proper scoring rule.
The improvement is modest — this is the expected, correct outcome of an
eight-raw-column baseline on a dataset where a third of events lack
product features entirely (see §11) and 72% of the real test set involves
unseen customers (see `docs/dataset-audit.md`). Stage 1's purpose is
establishing this floor accurately, not maximizing it.

---

## 10. Price / Discount Ablation (LR-A vs. LR-B)

| Metric | Δ (B − A) | Interpretation |
|---|---|---|
| ROC-AUC | **+0.0081** | LR-B ranks slightly better |
| Log Loss | **−0.0032** (B lower) | LR-B's probabilities are slightly better calibrated to outcomes |
| Brier | **−0.0016** (B lower) | Same direction, smaller magnitude |
| ECE | **+0.0057** (B higher, i.e. worse) | LR-B is *less* well quantile-calibrated than LR-A |

`avgGbpPrice`/`avgDiscountValue` produce a small, consistent improvement
in discrimination and proper-scoring-rule metrics, but a small
*degradation* in quantile calibration (ECE 0.0062 → 0.0119 — both are
still very low in absolute terms). **LR-B is not automatically declared
the winner on AUC alone**: the ROC-AUC gain (0.008) is small relative to
the confirmed uncertainty about the price/discount computation window
(§2, `docs/leakage-audit.md`), and the calibration cost, while small, is
real and in the opposite direction. Given the magnitude of both effects,
this reads as a genuine but minor trade-off rather than a clear win — LR-B
is retained as the "with price" reference variant, and LR-A remains the
methodologically more conservative default when the price/discount
provenance question matters more than the last percentage point of AUC.

---

## 11. Product-Coverage Diagnostic (LR-B)

Validation events split by whether the event's product node was found,
evaluated with the same fitted LR-B model (not retrained):

| Slice | n | Prevalence | ROC-AUC | Log Loss |
|---|---|---|---|---|
| With product node | 179,095 | 0.5543 | 0.6596 | 0.6492 |
| Without product node | 95,416 | 0.5561 | 0.5870 | 0.6736 |

Confirms the expected direction: the model discriminates meaningfully
better (+0.073 AUC) when brand/type/price features are actually present.
The "without product node" slice still beats the prevalence baseline
(0.587 vs. 0.500 AUC) purely on customer demographics, but by a much
smaller margin. This is diagnostic only — **the primary training
population is not changed** based on this result; a third of training
events legitimately lack product features and the model must handle that
in deployment regardless.

---

## 12. Cold-Start Results

### Primary split (customer always new by construction)

| Quadrant | n | Prevalence | ROC-AUC | Log Loss |
|---|---|---|---|---|
| New customer + known product | 226,832 | 0.5429 | 0.6454 | 0.6569 |
| New customer + new product | 47,679 | 0.6125 | 0.6017 | 0.6614 |

### Secondary split — diagnostic only (LR-B retrained on the random split)

| Quadrant | n | Prevalence | ROC-AUC | Log Loss |
|---|---|---|---|---|
| Known customer + known product | 130,515 | 0.5359 | 0.6504 | 0.6561 |
| Known customer + new product | 27,240 | 0.6108 | 0.6068 | 0.6592 |
| New customer + known product | 95,696 | 0.5460 | 0.6428 | 0.6575 |
| New customer + new product | 20,376 | 0.6188 | 0.5847 | 0.6668 |

Pattern is consistent across both splits: performance is best when the
product is known (customer status matters less — 0.645–0.650 AUC either
way), and worst when the product is new, regardless of customer status
(0.585–0.602 AUC). This suggests the six safe features are currently
doing more work through product attributes than customer attributes — a
useful signal for where Stage 2 feature engineering would pay off most
(e.g., product-adjacent features that don't require the specific SKU to
have been seen before, like brand-level or category-level statistics
computed safely).

All slice metrics above have n ≥ 20,000 (well over the 5,000-row
suppression threshold in `ml/evaluation/slices.py`); none were suppressed.

---

## 13. Calibration

Measured via 10-bin quantile reliability curves and Expected Calibration
Error (`ml/evaluation/calibration.py`); **no calibrator was fit** — this
is measurement only, per the project's explicit deferral (the
returners-only sample bias means any calibrator fit now would not
transfer to the general population — see `docs/problem-definition.md`).

- LR-A ECE: **0.0062**
- LR-B ECE: **0.0119**

Both are low in absolute terms (well-calibrated *relative to the
returner-only population this model was trained and validated on*).
Reliability curve plots: `reports/calibration_lr_a.png`,
`reports/calibration_lr_b.png`.

A real implementation bug was caught and fixed during this measurement
work: `reliability_curve`'s original quantile-binning logic (built on
`sklearn.calibration.calibration_curve` plus a separate `np.digitize` bin
re-assignment) silently dropped rows whenever quantile edges collided —
which happens whenever a Logistic Regression on mostly-categorical
features produces few distinct probability values, and happens *always*
for the prevalence baseline's perfectly constant output. It was rewritten
around a single `pd.qcut` bin assignment shared by the count, predicted-mean,
and observed-mean calculations, with an explicit single-bin path for
fully-constant input. Regression tests for both failure modes are in
`tests/test_calibration.py`.

---

## 14. Limitations Carried Into Stage 2

1. **Population selection bias is unaddressed.** ~55% measured return
   rate reflects a returners-only sample. No calibrator has been fit —
   correctly, since fitting one now would calibrate to the wrong
   population. This must be revisited once general-population data or
   assumptions are available.
2. **`avgGbpPrice`/`avgDiscountValue` computation window remains
   unconfirmed.** Confirmed global constants, confirmed not target-derived,
   but the exact averaging window relative to any single predicted
   transaction is not provable from this dataset. LR-A exists specifically
   so this uncertainty doesn't silently enter the "primary" model.
3. **A third of training events have no product features at all**
   (65.34% coverage). The `"__MISSING__"` token handles this without
   distorting brand/type distributions, but no amount of preprocessing
   recovers information that was never collected for these rows.
4. **Customer-node missingness's 9.8-point target-rate gap is suppressed,
   not explained.** It is treated as a dataset artifact rather than a
   real signal based on plausibility, not a definitive test — Stage 2
   should not assume this reasoning is airtight if the true cause becomes
   discoverable.
5. **No timestamps** — the primary split cannot be chronological; it can
   only approximate the cold-start character of the real split.

---

## 15. Reproducing Stage 1

```bash
pip install -r requirements.txt

# 1. Build the joined training frame + both validation splits
python scripts/build_dataset.py

# 2. Train all four models, run all diagnostics, write reports/
python scripts/train_baseline.py

# 3. Run the test suite
pytest tests/ -v
```

Outputs land in `reports/`: `stage1_metrics.json` (all numbers in this
document), `calibration_lr_a.png`, `calibration_lr_b.png`,
`lr_a_pipeline.joblib`, `lr_b_pipeline.joblib` (all gitignored —
regenerable, not committed).

Both scripts are deterministic: two full reruns of the pipeline on this
machine produced metrics identical to within 1e-9 (in practice, exactly
equal) — verified as part of Stage 1 completion, not merely assumed.
