# Leakage Audit — ASOS GraphReturns Dataset

**Governing question:** Could this exact feature value have been known immediately before the purchase being predicted?

---

## Stage 1 Corrections to This Audit

Stage 1 implementation work included direct empirical checks against the raw
data that revised several Stage 0 conclusions. These corrections are
authoritative; sections below have been updated to match, but are
summarized here for visibility.

1. **Product node aggregates are GLOBAL, not per-split.** For all 405,505
   variants present in both `product_nodes_training.p` and
   `product_nodes_testing.p`, `avgGbpPrice`, `avgDiscountValue`,
   `salesPerProduct`, and `returnsPerProduct` are **byte-identical** across
   the two files (100.00% match). This means `productReturnRate` in the
   *training* file was computed using **test-period return outcomes** —
   worse than same-period leakage, since it leaks information from the
   held-out split into the features available for a training-split
   prediction. This strengthens the UNSAFE classification for the product
   return-aggregate columns beyond the original "same-window" concern.

2. **`premier` is effectively static — the earlier caveat is resolved.**
   For the 183,498 customers present in both customer node files, `premier`
   matches in 99.99% of cases (also true of `yearOfBirth` 99.94%, `isMale`
   99.98%, `shippingCountry` 99.96%). These are static account attributes,
   not end-of-period snapshots. Classified SAFE with no remaining caveat.

3. **Anonymization is incomplete.** `brandDesc` contains one real,
   un-anonymized brand value, `Pull&Bear` (4,954 training rows), and
   `productType` contains one real value, `Jeans` (10,441 training rows),
   alongside the anonymized `Brand_*`/`productType_*` codes. Category codes
   elsewhere in the dataset should not be assumed reliably anonymized.

4. **Supplied one-hot dummy columns use inconsistent conventions and should
   not be used.** `Country_A`–`I` is a full 9-category one-hot (9 dummies
   for 9 categories — collinear with an intercept). `productType_A`–`K` is
   also a full one-hot (11 dummies for 11 categories, including `Jeans`
   folded into one of the coded categories). `Brand_A`–`K` drops one
   reference category (10 dummies for 11 categories) — and the dropped
   category is `Pull&Bear`, not a coded `Brand_*` value as previously
   assumed; there is no `Brand_H`. Stage 1 encodes the raw categorical
   columns itself with a consistent `OneHotEncoder` rather than using any
   supplied dummy column.

5. **`yearOfBirth == 1900` is a missing-value sentinel, not a real birth
   year.** It is the single most common value in the column by a wide
   margin (11,973 of 777,001 training customers, 1.54%) and forms an
   isolated spike far from the main 1950–2005 distribution. Stage 1
   converts it to NaN before preprocessing; no other values are altered.

6. **Measured event-level join coverage and its target-rate signal.** On
   training events: 94.77% have a matching customer node, 65.34% have a
   matching product node. The missing-customer-node subset has a
   materially different return rate (64.6% vs. 54.8% for joined rows, a
   9.8-point gap) — treated as a non-generalizable dataset-construction
   artifact and deliberately not exposed as a model feature (see
   docs/stage1-baseline.md). The missing-product-node subset's return rate
   is nearly identical to the joined subset (55.5% vs. 55.2%), consistent
   with product-side missingness being target-neutral.

7. **The duplicate `_D` return-code columns hold genuinely different data,
   not literal duplicates.** For `customerId_level_return_code_D`, the two
   same-named columns have means 0.0297 and 0.3972 respectively — different
   underlying data mapped to the same (erroneous) column name in the
   source file. This does not change their UNSAFE classification (both are
   post-return label aggregates), but confirms this is a genuine upstream
   column-naming defect rather than an accidental exact duplication.

---

## Summary

The dataset supplies 77 raw columns across the three table types (event, customer, product). Of these, 6 are safe features used directly in the Stage 1 baseline, 30 supplied one-hot dummy columns are technically safe but not used (redundant with the raw columns Stage 1 encodes itself), 2 are suspicious and included only in an ablation variant (LR-B), and 21 are unsafe. The unsafe columns are all aggregated return statistics that encode the prediction target — several confirmed to leak test-period outcomes into training-split features (see Stage 1 Corrections above) — and must be excluded from any model trained on this data.

---

## Event Table Columns

| Column | Classification | Reason |
|--------|---------------|--------|
| `hash(variantID)` | SAFE | Join key; encodes no outcome. |
| `hash(customerId)` | SAFE | Join key; encodes no outcome. |
| `isReturned` | **TARGET** | The prediction target — never use as a feature. |

---

## Customer Node Columns

| Column | Classification | Reason |
|--------|---------------|--------|
| `hash(customerId)` | SAFE | Identifier only. |
| `yearOfBirth` | SAFE | Static demographic (99.94% stable across splits for shared customers). `1900` is a confirmed missing-value sentinel (1.54% of training customers, not a real birth year — see Stage 1 Correction #5) and is converted to NaN before modelling; no other values are altered. |
| `isMale` | SAFE | Static demographic (99.98% stable across splits). Available at purchase time. |
| `shippingCountry` | SAFE | Shipping destination is known at checkout (99.96% stable across splits). |
| `premier` | SAFE | Membership tier is a known account attribute at purchase time. **Confirmed static**: 99.99% stable for customers present in both train and test node files — not an end-of-period snapshot. |
| `Country_A` … `Country_I` | **NOT USED** | Supplied one-hot dummies for `shippingCountry` — a full 9-category one-hot, technically safe but redundant with the raw column Stage 1 encodes directly with a consistent `OneHotEncoder`. |
| `salesPerCustomer` | **SUSPICIOUS** | Count of all purchases for this customer. The customer node tables are split by train/test period, meaning the test node table was likely computed over the **test period** — the same transactions being predicted. If this count includes the current transaction, it is leaky. Additionally, it may encode return outcome indirectly (higher sales volume customers may behave differently). Cannot confirm safe without knowing the exact computation window. **Exclude from baseline.** |
| `returnsPerCustomer` | **UNSAFE** | Raw count of returns by this customer. If computed over the same transactions being predicted, it directly encodes the aggregate of the target variable across those transactions. Even if computed over an earlier window, including it in a model trained on transactions from that window inflates signal. `min = 1` confirms every customer in the dataset has at least one return — this is a selection-bias artifact, not a safe baseline statistic. |
| `customerReturnRate` | **UNSAFE** | `returnsPerCustomer / salesPerCustomer`. This is the average return rate, directly derived from the prediction target. The training mean is 0.541 and test mean is 0.534, both very close to the event table return rate (55.3% train, 54.5% test), which is strong evidence that these aggregates were computed over the same transactions as the targets. **Critical leakage risk.** |
| `customerId_level_return_code_A` … `_L` | **UNSAFE** | Proportions of return reason codes (A–L) used by this customer. Return codes are assigned **after** a return event. These encode the distribution of return reasons from historical or concurrent returns. Because return codes are post-return labels, including them is direct target leakage. Additionally, one reason code (`_D`) appears **twice** as a duplicate column — a data quality defect in the source file. |

---

## Product Node Columns

| Column | Classification | Reason |
|--------|---------------|--------|
| `hash(variantID)` | SAFE | Identifier only. |
| `hash(productID)` | SAFE | Identifier only. |
| `productType` | SAFE | Catalog classification. Available at listing time. |
| `hash(supplierRef)` | SAFE | Catalog attribute. |
| `brandDesc` | SAFE | Brand is known at purchase time. Contains one un-anonymized real value, `Pull&Bear` (see Stage 1 Correction #3), alongside coded `Brand_*` values — does not affect safety, only interpretability. |
| `Brand_A` … `Brand_K` | **NOT USED** | Supplied one-hot dummies for `brandDesc`, technically safe but inconsistent with the other supplied dummy sets (drops one reference category — `Pull&Bear`, not a coded value; there is no `Brand_H`). Stage 1 encodes `brandDesc` itself with a consistent `OneHotEncoder` instead. |
| `productType_A` … `productType_K` | **NOT USED** | Supplied one-hot dummies for `productType`, technically safe but redundant with the raw column Stage 1 encodes directly. |
| `avgGbpPrice` | **SUSPICIOUS** | Average GBP price for this variant. **Confirmed global**: byte-identical between train and test product node files for all 405,505 shared variants (see Stage 1 Correction #1) — it is *not* independently recomputed per split, and its mean is not target-derived. Remains suspicious only because the exact averaging window relative to the predicted transaction cannot be confirmed from the data. **Included in LR-B; excluded from LR-A** as a documented ablation. |
| `avgDiscountValue` | **SUSPICIOUS** | Same reasoning and same global-constant confirmation as `avgGbpPrice`. **Included in LR-B; excluded from LR-A.** |
| `salesPerProduct` | **UNSAFE (revised)** | Total sales count for this variant. Also confirmed global/byte-identical across train and test product node files. Excluded from the baseline as the denominator of the leaky `productReturnRate`, and because as a global constant it does not distinguish pre- vs. post-prediction-window exposure. |
| `returnsPerProduct` | **UNSAFE** | Raw count of returns for this variant, confirmed global across splits. `min = 1` confirms every product in the dataset has been returned at least once. Because the value is identical in the training and test node files, the training-split feature literally contains test-period return counts. |
| `productReturnRate` | **UNSAFE (strengthened)** | `returnsPerProduct / salesPerProduct`. Training mean 0.385, test mean 0.385 — identical, because it is the same global value in both files (see Stage 1 Correction #1). This is worse than same-window leakage: a model trained with this feature would have access to aggregate test-period outcomes at training time. **Critical leakage risk — excluded from Stage 1 baseline.** |
| `variantID_level_return_code_A` … `_L` | **UNSAFE** | Proportion of each return reason code applied to returns of this variant. Same reasoning as the customer-level return codes — these are post-return labels. One reason code (`_D`) appears **twice** as a duplicate column, and the two `_D` columns hold genuinely different data (means 0.0297 vs. 0.3972 in training) — a column-naming defect, not an exact duplicate (see Stage 1 Correction #7). |

---

## Key Evidence of Period-Level Contamination

**Customer-side aggregates (`returnsPerCustomer`, `customerReturnRate`, `salesPerCustomer`) are recomputed per split** — confirmed by direct comparison: for the 183,498 customers present in both customer node files, `salesPerCustomer` matches in only 4.83% of cases and `returnsPerCustomer` in only 9.80% — i.e. these values genuinely differ between the training and test node tables, consistent with being computed independently over each split's own transactions:

- Training customer mean `returnsPerCustomer` = 6.07, `salesPerCustomer` = 11.5 → implied return rate **52.8%**, vs. actual event table return rate **55.3%**
- Test customer mean `returnsPerCustomer` = 6.02, `salesPerCustomer` = 11.7 → implied return rate **51.4%**, vs. actual **54.5%**

The ~3-point gap between the implied and actual rates is consistent with within-period aggregation over multiple events per customer plus selection effects, not with these statistics being drawn from an independent prior period.

**Product-side aggregates (`returnsPerProduct`, `productReturnRate`, `salesPerProduct`, and also `avgGbpPrice`/`avgDiscountValue`) are GLOBAL, not per-split** — confirmed directly: for all 405,505 variants present in both product node files, these five columns are **byte-identical** (100.00% match) between the training and test files. This is a stronger and more direct form of leakage than the customer side: the training file's `productReturnRate` was computed using outcomes from the test-period transactions, so a model trained on it has access to aggregate future/held-out information at training time.

**Conclusion**: customer return/sales aggregates are same-window leakage; product return/sales aggregates are global-constant leakage that additionally crosses the train/test boundary. Both are excluded from the Stage 1 baseline. `avgGbpPrice`/`avgDiscountValue` are also global constants but their mean is not derived from the target, so they are treated as merely suspicious (ablated in LR-B) rather than unsafe.

---

## Classification Summary

| Class | Count | Columns |
|-------|-------|---------|
| SAFE (used) | 6 | `yearOfBirth`, `isMale`, `shippingCountry`, `premier`, `productType`, `brandDesc` |
| SAFE (not used — redundant dummies) | 30 | `Country_A–I` (9), `Brand_A–G,I–K` (10), `productType_A–K` (11), plus `hash(supplierRef)`, `hash(productID)` (excluded as high-cardinality IDs, deferred to Stage 2) |
| SUSPICIOUS | 2 | `avgGbpPrice`, `avgDiscountValue` (global per-variant constants; included in LR-B only, ablated against LR-A) |
| UNSAFE | 21 | `salesPerCustomer`, `returnsPerCustomer`, `customerReturnRate`, `customerId_level_return_code_A–L` (×2 D — 12 columns), `salesPerProduct`, `returnsPerProduct`, `productReturnRate`, `variantID_level_return_code_A–L` (×2 D — 12 columns) |
| UNKNOWN | 0 | — |
| TARGET | 1 | `isReturned` |

Note: `salesPerCustomer` and `salesPerProduct` moved from SUSPICIOUS to UNSAFE in Stage 1 — both are now excluded outright rather than merely flagged, since `salesPerProduct` is confirmed to be a global constant and the denominator of the leaky `productReturnRate`, and `salesPerCustomer`'s per-split recomputation gives it the same same-window contamination profile as `returnsPerCustomer`.

---

## Recommended Handling for Suspicious Columns

- **`avgGbpPrice` / `avgDiscountValue`**: Confirmed global per-variant constants (not target-derived, not independently recomputed per split). Included in **LR-B** as a documented ablation against **LR-A** (which excludes them); see docs/stage1-baseline.md for the measured effect. Not treated as automatically safe — the exact averaging window relative to the predicted transaction remains unconfirmed.
- **`salesPerCustomer` / `salesPerProduct`**: Excluded from the Stage 1 baseline entirely (see reclassification note above).

---

## Columns That Must Not Appear in Any Model

- `salesPerCustomer`
- `returnsPerCustomer`
- `customerReturnRate`
- `customerId_level_return_code_*` (all variants)
- `salesPerProduct`
- `returnsPerProduct`
- `productReturnRate`
- `variantID_level_return_code_*` (all variants)
- `isReturned` (this is the target, not a feature)

This exact list is codified as `LEAKY_COLUMNS` in `ml/data/schema.py` and is enforced by `tests/test_schema.py::test_safe_features_disjoint_from_leaky_columns` and `tests/test_joins.py::test_no_leaky_columns_in_joined_frame`.

Any model that uses these columns will suffer from target leakage and will produce misleadingly high training metrics. The effect may be detectable at evaluation time only if train and test splits are properly separated — if they use the same period, inflated metrics will appear in both.

---

## Stage 2: Target Encoding Rejected

Stage 2 considered target-encoding the two excluded high-cardinality identifiers, `hash(supplierRef)` and `hash(productID)` (i.e. replacing each ID with some estimate of `E[isReturned | ID]`), and **rejected it** — not because a leakage-safe construction (out-of-fold, fit on the train fold only) is technically hard, but because out-of-fold construction does not fix the actual problem, which is temporal:

1. **No timestamps.** A target encoding estimates `E[y | category]` from some row set and requires those rows to *precede* the row being scored. Without timestamps, that cannot be asserted — train-fold events may have occurred after the validation event they would be informing.
2. **The customer-grouped validation split does not protect product-side encodings.** The split groups by *customer*; products legitimately appear on both sides of it. A product-ID encoding fitted on the train fold and applied to validation never touches validation targets directly, but it does mix temporally interleaved events in a way that cannot be falsified.
3. **It would reconstruct the exact statistic already banned.** `productReturnRate` is UNSAFE because it aggregates target outcomes over an unknown, unproven window. A hand-built product-ID target encoding *is* that statistic, recomputed under our own fold control — controlling the fold structure does not change the underlying temporal-ordering problem that motivated the original ban.
4. **Low expected value regardless.** Roughly 239K joinable parent products across ~894K product-covered training events is only a handful of events per product on average; most categories would smooth heavily toward the global prior.
5. **27.5% of the real test set involves products unseen in training** and would receive the fallback prior regardless of encoding quality.

**Decision: no target encoding anywhere in this project.** Instead, Stage 2 uses fold-local, target-free **frequency encoding** of the same two identifiers (plus the variant ID) — see below.

---

## Stage 2: Frequency Features — SUSPICIOUS/Experimental

`ml/features/frequency.py` adds three exposure-count features, used only in the Stage 2 A4 ablation rung:

- `variant_event_count`, `product_event_count`, `supplier_event_count` — the number of TRAIN-FOLD events sharing that row's `hash(variantID)` / `hash(productID)` / `hash(supplierRef)`, computed fresh inside each fold (never the full dataset, never the supplied `salesPerProduct`/`salesPerCustomer`).

**Why these are safer than the banned `salesPerProduct`/`salesPerCustomer`**: those columns are global constants confirmed to span the test period (see "Stage 1 Corrections" above) and never touch the target at all in either case — target-free counting is strictly weaker leakage exposure than target-derived aggregation.

**Why they are still classified SUSPICIOUS rather than SAFE**: without event timestamps, "count of train-fold purchases sharing this ID" carries the same unfalsifiable within-window-exposure assumption Stage 1 attached to `avgGbpPrice`/`avgDiscountValue` — we cannot prove every counted event precedes the one being scored.

**No customer-ID frequency feature exists, deliberately.** Under the customer-grouped primary split, every validation customer is guaranteed unseen in the train fold, so a customer-ID frequency feature would be informative during training and constant-zero at every validation row — a textbook train/serve skew, not leakage in the traditional sense but a real correctness bug. Enforced by `tests/test_frequency.py::test_no_customer_id_frequency_feature_exists`.

See `docs/stage2-gbdt.md` for the measured effect of including these features (the A3→A4 ablation delta).
