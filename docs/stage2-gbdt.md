# Stage 2 — XGBoost, Feature Engineering, and the Missing-Node Artifact

**Status**: Complete. **Last run**: 2026-08-07, on the full 1,369,133-row training event table, on the exact Stage 1 customer-grouped harness.

This document records the Stage 2 methodology, results, and reasoning.
It assumes familiarity with `docs/stage1-baseline.md` — Stage 2 changes
the model and adds features; it does not change the split, the metric
suite, or the leakage-safety doctrine established in Stage 1.

---

## 1. Why XGBoost

XGBoost was chosen as the single stronger model, over Random Forest,
LightGBM, and CatBoost:

- **Native missing-value handling** (a learned default split direction
  per node) is exactly the right primitive for the 34.7% of events with
  no product node — Stage 0/1 confirmed this missingness is target-neutral,
  so letting the tree route NaN natively costs nothing and avoids
  distorting brand/type/price distributions with an imputed value.
- **No implicit target-based encoding of categoricals.** Every encoding
  decision in this project stays in auditable project code. LightGBM's
  native categorical handling sorts categories by gradient statistics
  internally — a target-derived operation Stage 2 would rather keep
  visible. CatBoost's headline strength is ordered target statistics for
  high-cardinality categoricals, which is irrelevant here: Stage 2
  evaluated and rejected target encoding entirely (§4), and the actual
  categoricals used (country/type/brand, ≤12 levels each) don't need it.
- **`hist` tree method** is fast enough for 1.37M rows on a laptop
  (confirmed: ~30–70s per rung fit) and is the best-supported path into
  `TreeExplainer` for a future SHAP stage.
- Random Forest was not seriously considered: no native missing-value
  handling, generally weaker tabular ranking, and worse-calibrated
  probabilities than either LR or boosted trees — directly working
  against this project's calibration goal.

---

## 2. Why Target Encoding Was Rejected

Stage 2 evaluated target-encoding `hash(supplierRef)` and
`hash(productID)` and rejected it outright. Full reasoning in
`docs/leakage-audit.md` ("Stage 2: Target Encoding Rejected"); summary:
without event timestamps, no fold construction — however carefully
out-of-fold — can prove that the events used to estimate `E[y | ID]`
precede the event being scored. Controlling *which rows* contribute to
an encoding does not solve a *temporal-ordering* problem. A target
encoding here would functionally reconstruct `productReturnRate` (already
banned) under different fold bookkeeping. Frequency encoding was used
instead — see §5.

---

## 3. The `avgDiscountValue` Percentage Assumption

`discount_amount = avgGbpPrice * avgDiscountValue / 100` assumes
`avgDiscountValue` is a percentage. Evidence (measured on
`product_nodes_training.p`):

| Check | Result |
|---|---|
| `discount > price` | 38.06% of variants — impossible for an absolute discount |
| `avgDiscountValue` distribution | median 17.7, IQR [16.4, 19.0] — a tight cluster consistent with a capped rate |
| `corr(avgGbpPrice, avgDiscountValue)` | 0.021 — an absolute discount would scale with price; this doesn't |

This is a documented assumption, not a proven fact. Whether `avgGbpPrice`
is pre- or post-discount is separately UNKNOWN and does not change the
recommendation — the price × discount product is informative either way.
`discount / price` (dividing a percentage by a currency amount) was
explicitly **not** built, since it is dimensionally meaningless under
this assumption.

---

## 4. The Missing-Node Artifact: Donor Imputation and the Leakage Probe

### The problem

Stage 1 found events with no customer node have a return rate 9.8 points
higher than joined events (64.6% vs. 54.8%), believed to be a
dataset-construction artifact. Stage 1's Logistic Regression could not
exploit this — median/mode imputation cannot be represented as a
distinguishing pattern by a linear model. A tree can: median/mode
imputation gives every missing-node row an *identical* value on all four
customer fields simultaneously, and a tree isolates that conjunction in a
few splits.

### The fix: joint donor (hot-deck) imputation

`ml/features/imputation.py`'s `DonorImputer` replaces each missing
customer profile with one COMPLETE, REAL profile sampled from the
training fold — never independently per field (which would destroy the
real joint distribution and produce combinations a tree could still
learn to detect). A separate, narrower case — a real customer whose
`yearOfBirth` alone is the confirmed 1900 sentinel, with the other three
fields present — gets the ordinary train-fold median instead of a full
donor draw, since that is a real profile with one unknown field, not the
node-missing artifact.

**Two real bugs were found and fixed during implementation, both worth
recording:**

1. **Donor pool contamination.** The first version built the donor pool
   from "node present" rows (`isMale.notna()`) without also requiring
   `yearOfBirth.notna()`. A donor customer could have a present node but
   their *own* `yearOfBirth` sentinel — drawing such a row as a donor
   silently carried that NaN into the target row, defeating the
   "always fully imputed" guarantee (707 residual NaN out of ~50K
   imputed rows in the first test). Fixed by requiring completeness
   across all four `CUSTOMER_COLS` before a row enters the donor pool.
2. **Probe label corruption.** The leakage probe's target
   (`has_customer_node`) was first computed using
   `ml.data.joins.has_customer_node_mask`, which is defined as
   `yearOfBirth.notna()` and is only correct when called on the frame
   *before* `ml.features.preprocessing.clean_year_of_birth` runs. Stage 2
   always works with already-cleaned frames, so this call mislabeled
   every sentinel-affected node-present row (18,242 rows, confirmed
   count) as node-missing. Combined with `DonorImputer`'s own correct
   internal use of `isMale`-based detection, this created an internally
   inconsistent picture: rows labeled "missing" by the corrupted probe
   target but actually filled via the sentinel-median path (a constant
   1993.0), producing a large, trivially separable spike the probe
   picked up as signal. This inflated the artifact-safe probe AUC from
   ~0.5 to 0.615 in an early test run — a bug in the *measurement*, not
   in `DonorImputer`. Fixed by computing the probe's label independently
   via `isMale.notna()` (`ml.evaluation.probe._has_customer_node`),
   matching `DonorImputer`'s own cleaning-order-independent logic.

### Leakage probe result

A small XGBoost (`n_estimators=200, max_depth=4`) trained to predict
`has_customer_node` from the processed customer-side columns only, fit on
`inner_train` (957,665 rows), evaluated on `early_stop` (136,957 rows):

| Preprocessing | Probe ROC-AUC |
|---|---|
| Median/mode (Stage 1-style, contaminated) | **0.9866** |
| Donor imputation (artifact-safe) | **0.5248** |

Target was ≤ 0.55 (a target, not a hard gate). **Met.** The contaminated
preprocessing is, as predicted, almost perfectly detectable — confirming
the threat is real and not hypothetical. Donor imputation reduces this to
essentially chance-level discrimination.

### What the artifact was actually worth

Comparing A1 (contaminated) to A2 (honest) on identical primary
validation, holding everything else fixed:

| | ROC-AUC | Δ vs. A2 |
|---|---|---|
| A1 (contaminated) | 0.6566 | +0.0031 |
| A2 (honest) | 0.6535 | — |

The artifact was worth **0.003 AUC** — small, and *far* smaller than what
legitimate feature engineering bought (A4 vs. A2: +0.0131, see §7). This
is a reassuring result: suppressing the artifact cost almost nothing, and
the honest model still comfortably beats Stage 1's LR-B (0.6535 vs.
0.6381) on demographics/catalogue features alone.

---

## 5. Product Missingness: Native NaN Routing

Confirmed target-neutral in Stage 0/1 (55.5% vs. 55.2% return rate,
present vs. missing product node). Handling, unchanged from the plan:

- `productType`/`brandDesc`: explicit `"__MISSING__"` category (same as
  Stage 1) — one-hot encoded directly.
- `avgGbpPrice`/`avgDiscountValue`/all derived price features: **NaN
  preserved**, never imputed, routed natively by XGBoost. No
  StandardScaler either (tree splits are scale-invariant, and scaling
  would force an imputer to run first, defeating native routing).

The customer/product asymmetry — trees are *allowed* to see product-side
missingness but *forbidden* from seeing customer-side missingness — is
evidence-driven from the same Stage 0/1 measurement, not a stylistic
choice: one side's missingness carries no target signal (safe to expose
via NaN), the other side's does (must be hidden via donor imputation).

---

## 6. Frequency Features (A4)

`ml/features/frequency.py`'s `ProductFrequencyEncoder` adds three
fold-local, target-free exposure counts: `variant_event_count`,
`product_event_count`, `supplier_event_count` — training-fold row counts
sharing `hash(variantID)` / `hash(productID)` / `hash(supplierRef)`
respectively. Classified SUSPICIOUS/experimental (see
`docs/leakage-audit.md`): safer than the banned
`salesPerProduct`/`salesPerCustomer` (global constants confirmed to span
the test period) since these are fold-local and target-free, but still
carry the same unfalsifiable within-window-exposure assumption Stage 1
attached to `avgGbpPrice`.

`hash(productID)`/`hash(supplierRef)` are NaN exactly when the product
node is missing (same join-miss as `productType`); in that case the
corresponding count feature is NaN, not 0 — an unknown ID has an
undefined count. 0 is reserved for a *known* ID that simply had zero
train-fold events.

**No customer-ID frequency feature exists, deliberately.** Under the
customer-grouped primary split, every validation customer has a
train-fold count of exactly 0 by construction — a customer-ID frequency
feature would be informative in training and constant-zero at
validation, a textbook train/serve skew. Enforced by
`tests/test_frequency.py::test_no_customer_id_frequency_feature_exists`.

---

## 7. Ablation Ladder Results

All rungs fit on `inner_train` (957,665 rows), early-stopped on
`early_stop` (136,957 rows), and reported on `primary_val` (274,511 rows)
— the identical fold Stage 1 used. Rung/hyperparameter selection used
`early_stop` ROC-AUC exclusively, but `primary_val` metrics were repeatedly
computed and visible across rungs and tuning fits. They are developmental
reporting rather than a one-shot confirmatory evaluation.

| Rung | Config | primary_val ROC-AUC | LogLoss | Brier | n_features | Fit time |
|---|---|---|---|---|---|---|
| A0 | Stage 1 LR-B (read from disk, not recomputed) | 0.6381 | 0.6577 | 0.2328 | 38 | — |
| A1 | XGBoost, contaminated (DIAGNOSTIC ONLY) | 0.6566 | 0.6466 | 0.2278 | 38 | 32.6s |
| A2 | XGBoost, donor-imputed (honest baseline) | 0.6535 | 0.6479 | 0.2285 | 38 | 29.9s |
| A3 | A2 + derived price features | 0.6548 | 0.6470 | 0.2281 | 41 | 50.0s |
| **A4** | **A3 + frequency features** | **0.6666** | **0.6388** | **0.2246** | **44** | **71.1s** |

**A4 is the best frequency-enhanced development rung**, selected
programmatically by `early_stop` ROC-AUC (0.6673 vs. A2's 0.6543, A3's
0.6554). Its frequency features remain SUSPICIOUS because their
point-in-time exposure cannot be verified without timestamps.

### Derived features (A2 → A3): small, real gain

+0.0013 AUC. `discount_amount` and the two price-relative-to-median
ratios give the model information it cannot easily reconstruct on its
own (a *ratio*/*product* of two existing columns is not a single-feature
monotone transform a tree already captures). Modest but genuine —
consistent with a tree already able to split on `avgGbpPrice` and
`productType` jointly to approximate part of what
`price_rel_type_median` encodes directly.

### Frequency features (A3 → A4): the largest single gain in Stage 2

+0.0118 AUC, +0.0082 log-loss, the largest jump in the whole ladder —
larger than switching from Logistic Regression to XGBoost in the first
place (A0 → A2: +0.0154, comparable magnitude actually, worth noting
neither dominates). Exposure counts (how often this exact variant/
product/supplier appeared in training) carry real, previously-unavailable
signal. This is also the ablation rung with the least certain
methodological footing (§6's unfalsifiable exposure-window assumption),
so this gain should be read as "worth the ablation, not free of caveats."

---

## 8. Hyperparameter Tuning (A5)

Fixed grid, no Optuna: `max_depth ∈ {4, 6, 8} × learning_rate ∈ {0.05,
0.1}`, `n_estimators=6000` (raised from the default rungs' 2000 — the
slower-converging low-learning-rate/shallow-depth configs needed the
extra room; see the code comment in `scripts/train_stage2.py`), all
evaluated by `early_stop` ROC-AUC only:

| max_depth | learning_rate | early_stop ROC-AUC | Fit time |
|---|---|---|---|
| 4 | 0.05 | 0.6669 | 265.5s |
| 4 | 0.10 | 0.6667 | 172.6s |
| 6 | 0.05 | 0.6678 | 196.2s |
| 6 | 0.10 (= A4 default) | 0.6673 | 61.0s |
| 8 | 0.05 | **0.6680** | 48.2s |
| 8 | 0.10 | 0.6674 | 31.4s |

Best config (depth=8, lr=0.05) scores 0.6680 vs. A4's default 0.6673 — a
gain of **+0.0007**, well inside the pre-declared 0.003 noise threshold.
**Not adopted.** The final model is A4 with its default hyperparameters
(`max_depth=6, learning_rate=0.1`), not the "winning" grid cell — tuning
this feature set essentially plateaus around 0.667–0.668 regardless of
depth/learning-rate combination, which is itself a useful negative
result: hyperparameter tuning is not where headroom remains for this
feature set.

---

## 9. Final Model: A4

- **Configuration**: XGBoost, `max_depth=6, learning_rate=0.1,
  subsample=0.8, colsample_bytree=0.8`, donor-imputed customer features,
  derived price features, product-side frequency features. 44 features.
- **primary_val**: ROC-AUC 0.6666, LogLoss 0.6388, Brier 0.2246
- **Skill vs. Stage 1 LR-B** (log-loss): +0.0287 — i.e. roughly 2.9% of
  the remaining log-loss gap between LR-B and a perfect model was closed
  by A4's combination of model class + features.
- **Convergence**: best_iteration 715 of 2000 cap — early stopping
  triggered comfortably, not against the ceiling.

---

## 10. Calibration

Measured only — no calibrator fit, for the same reason as Stage 1 (the
returners-only sample would calibrate to the wrong population; see
`docs/problem-definition.md`).

- **A4 (final model) ECE**: **0.0066** — comparable to Stage 1 LR-A's
  0.0062 and better than LR-B's 0.0119. XGBoost with `binary:logistic`
  and this much data produces well-calibrated probabilities out of the
  box *relative to the population it was trained and validated on*.
- Reliability diagram: `reports/calibration_stage2_best.png`.

---

## 11. Cold-Start Results (Final Model)

### Primary split (customer always new by construction)

| Quadrant | n | ROC-AUC |
|---|---|---|
| New customer + known product | 226,832 | 0.6744 |
| New customer + new product | 47,679 | 0.6090 |

### Secondary split — diagnostic only (A4's config, retrained on the random split)

| Quadrant | n | ROC-AUC |
|---|---|---|
| Known customer + known product | 130,515 | 0.6772 |
| Known customer + new product | 27,240 | 0.6161 |
| New customer + known product | 95,696 | 0.6708 |
| New customer + new product | 20,376 | 0.5891 |

Overall secondary-split ROC-AUC: 0.6663 (diagnostic only, not a headline
number). Same qualitative pattern as Stage 1: performance depends more on
whether the *product* is known than whether the *customer* is known
(0.67–0.68 with a known product regardless of customer status; 0.59–0.61
with a new product regardless of customer status) — reinforcing Stage 1's
observation that Stage 2's feature set does most of its work through
product attributes, not customer history.

---

## 12. Product-Coverage Diagnostic (Final Model)

| Slice | n | ROC-AUC |
|---|---|---|
| With product node | 179,095 | 0.6890 |
| Without product node | 95,416 | 0.6188 |

The gap narrowed slightly relative to Stage 1's LR-B (0.66 vs. 0.59, a
0.073 gap) to 0.069 here — proportionally similar. The "without product
node" slice still clears the prevalence baseline by a wide margin on
demographics alone, and native NaN routing lets the model use whatever
signal customer features provide without being forced through an
imputed, uninformative product-feature value.

---

## 13. Rejected Features

- **`age = YEAR − yearOfBirth`**: a strictly monotone transform of a
  single existing feature. Trees are invariant to monotone transforms of
  a feature they already split on — this would be a mathematical no-op
  for XGBoost. (Worth doing later purely as a SHAP *display* transform,
  never as a modeling feature.)
- **Hand-built interactions** (`brandDesc × productType`, `premier ×
  productType`, `shippingCountry × productType`): with ≤12 categories per
  field and 957K+ training rows, XGBoost learns these through ordinary
  tree depth. Hand-building them adds columns and dilutes split
  candidates for no measurable benefit at this scale.
- **`discount / price`**: dimensionally meaningless given the
  `avgDiscountValue`-is-a-percentage assumption (§3).
- **Target encoding** of any identifier: rejected outright, §2.

---

## 14. Limitations Carried Into Stage 3

1. **Population selection bias remains unaddressed** — same as Stage 1.
   No calibrator fit; correctly so, since the returners-only sample would
   calibrate to the wrong population.
2. **The frequency-feature exposure-window assumption is unfalsifiable**
   without timestamps, despite being the single largest ablation gain in
   this stage (§7). This should temper how much weight is placed on the
   A4 result specifically, versus A2/A3's more conservative gains.
3. **`avgDiscountValue`'s percentage-units assumption is inferred, not
   certain** (§3) — carried forward unchanged from Stage 1's caveat about
   `avgGbpPrice`/`avgDiscountValue`'s exact provenance.
4. **The customer-node-missingness cause is still unproven** — donor
   imputation successfully suppresses its *exploitability* (probe AUC
   0.5248), which is the operationally important property, but does not
   explain *why* the 9.8-point target-rate gap exists.
5. **34.7% of events still lack product features entirely.** Native NaN
   routing handles this gracefully but cannot recover information that
   was never collected.
6. **A5's negative result** (tuning plateaus around 0.667–0.668
regardless of depth/learning-rate) suggests the remaining headroom in
this dataset, if any, is in features rather than model capacity —
directly informing [the Stage 3 methodology](stage3-methodology.md).

---

## 15. Reproducing Stage 2

```bash
pip install -r requirements.txt

# Prerequisite: Stage 1 must have been run at least once
python scripts/build_dataset.py
python scripts/train_baseline.py

# Stage 2
python scripts/train_stage2.py   # ~30-40 minutes (6-config tuning grid dominates)

# Tests
pytest tests/ -v
```

Outputs land in `reports/`: `stage2_metrics.json` and curated calibration
plots are retained as small portfolio artifacts; fitted `.joblib` binaries
are gitignored and regenerable.

**Determinism**: verified by fitting the A4 configuration twice on the
full real dataset and comparing predictions — **exactly bit-identical**
(`max abs diff = 0.0`), not merely within a tolerance. `tree_method="hist"`
with a fixed seed and fixed `n_jobs` is fully reproducible on this
machine. The full 6-config tuning grid was not rerun a second time in
full (≈30–40 minutes per run); the A4 rung is representative of the
underlying `fit_gbdt` determinism all rungs and grid configs share.
