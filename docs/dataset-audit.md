# Dataset Audit — ASOS GraphReturns

**Last updated**: 2026-08-06 (Stage 1 corrections incorporated)
**Audited by**: `python scripts/audit_dataset.py`

> **Stage 1 correction note**: Stage 1 implementation work directly verified
> several claims in this document against the raw data. The most
> significant correction: product node aggregates (`avgGbpPrice`,
> `avgDiscountValue`, `salesPerProduct`, `returnsPerProduct`) are
> byte-identical between the train and test product node files — they are
> **global per-variant constants**, not independently computed per split as
> originally inferred. Customer node aggregates ARE independently computed
> per split, as originally inferred. See docs/leakage-audit.md "Stage 1
> Corrections" for the full list; this document has been updated to match.

---

## Provenance

- **Dataset**: ASOS GraphReturns (public research dataset)
- **Source archive**: `c793h-osfstorage-archive.zip` (OSF / Open Science Framework; locally ignored, not committed)
- **Extracted pickle files stored at**: `data/raw/` (not committed to git)
- **Format**: Python pickle files created with pandas ~1.x

---

## Actual Files

| File | Size (disk) | Rows | Columns |
|------|-------------|------|---------|
| `event_table_training.p` | 31 MB | 1,369,133 | 3 |
| `event_table_testing.p` | 33 MB | 1,460,366 | 3 |
| `customer_nodes_training.p` | 179 MB | 777,001 | 30 |
| `customer_nodes_testing.p` | 191 MB | 825,598 | 30 |
| `product_nodes_training.p` | 137 MB | 411,495 | 44 |
| `product_nodes_testing.p` | 137 MB | 411,544 | 44 |

---

## Dataset Scale

| Dimension | Value |
|-----------|-------|
| Total events (train + test) | 2,829,499 |
| Training events | 1,369,133 |
| Testing events | 1,460,366 |
| Unique customers (train node table) | 777,001 |
| Unique customers (test node table) | 825,598 |
| Unique variants (train node table) | 411,495 |
| Unique variants (test node table) | 411,544 |
| Unique customers in train events | 810,237 |
| Unique variants in train events | 432,666 |

---

## Schemas

### Event Tables (`event_table_training.p`, `event_table_testing.p`)

| Column | Type | Notes |
|--------|------|-------|
| `hash(customerId)` | int64 | Hashed customer identifier |
| `hash(variantID)` | int64 | Hashed product variant identifier |
| `isReturned` | int64 | **Target**: 1 = returned, 0 = kept |

No timestamps, no additional event-level attributes.

### Customer Node Tables (`customer_nodes_training.p`, `customer_nodes_testing.p`)

| Column | Type | Notes |
|--------|------|-------|
| `hash(customerId)` | int64 | Primary key |
| `yearOfBirth` | int64 | Range 1890–2020. **`1900` is a confirmed missing-value sentinel** (1.54% of training rows), not a real birth year — converted to NaN in Stage 1, no other values altered. |
| `isMale` | int64 | Binary; 18.6% male in train |
| `shippingCountry` | object | 9 distinct anonymised countries |
| `premier` | int64 | Binary membership tier; 7.3% premier in train. **Confirmed static** (99.99% stable for customers in both splits) — not an end-of-period snapshot. |
| `salesPerCustomer` | int64 | Total purchases in the period (UNSAFE — recomputed independently per split; only 4.83% match for shared customers) |
| `returnsPerCustomer` | int64 | Total returns in the period (UNSAFE — same-window leakage) |
| `customerReturnRate` | float64 | returnsPerCustomer / salesPerCustomer (UNSAFE) |
| `customerId_level_return_code_A–L` | float64 | Fraction of returns with each reason code (UNSAFE). **`_D` appears twice** — confirmed to hold genuinely different data (means 0.0297 vs. 0.3972 in training), a column-naming defect rather than an exact duplicate. |
| `Country_A–I` | int64 | One-hot encoding of `shippingCountry` (9 columns; full one-hot, no dropped category). Not used in Stage 1 — raw column encoded directly instead. |

### Product Node Tables (`product_nodes_training.p`, `product_nodes_testing.p`)

| Column | Type | Notes |
|--------|------|-------|
| `hash(variantID)` | int64 | Primary key |
| `hash(productID)` | int64 | Parent product identifier |
| `productType` | object | 11 categories: 10 anonymised codes (`productType_A`–`K` minus one) plus one un-anonymised real value, **`Jeans`** (10,441 training rows). |
| `hash(supplierRef)` | int64 | Supplier identifier |
| `brandDesc` | object | 11 categories: 10 anonymised codes plus one un-anonymised real value, **`Pull&Bear`** (4,954 training rows). |
| `avgGbpPrice` | float64 | Avg price £1.25–£518. **Confirmed a global per-variant constant** — byte-identical between train and test product node files for all shared variants. SUSPICIOUS (averaging window relative to the predicted transaction unconfirmed) but not target-derived. |
| `avgDiscountValue` | float64 | Avg discount £0–£45.3. Same global-constant confirmation as `avgGbpPrice`. SUSPICIOUS. |
| `salesPerProduct` | int64 | Total sales, confirmed global/byte-identical across splits (UNSAFE — denominator of the leaky return rate) |
| `returnsPerProduct` | int64 | Total returns, confirmed global/byte-identical across splits (UNSAFE) |
| `productReturnRate` | float64 | returnsPerProduct / salesPerProduct (UNSAFE — and because the value is global, the training-split feature contains test-period outcomes) |
| `variantID_level_return_code_A–L` | float64 | Fraction of returns by reason code (UNSAFE). **`_D` appears twice**, holding genuinely different data (same defect pattern as the customer-side duplicate). |
| `Brand_A–G, I–K` | int64 | One-hot encoding of brand (10 columns). The dropped reference category is **`Pull&Bear`**, not a `Brand_*` code — there is no `Brand_H`. Not used in Stage 1. |
| `productType_A–K` | int64 | One-hot encoding of product type (11 columns; full one-hot). Not used in Stage 1. |

---

## Target

- **Column**: `isReturned`
- **Type**: binary integer (0 or 1)
- **Unique values**: {0, 1} — no nulls, no unexpected values

### Class Balance

| Split | Not Returned (0) | Returned (1) |
|-------|-----------------|-------------|
| Train | 611,906 (44.69%) | 757,227 (55.31%) |
| Test | 665,000 (45.54%) | 795,366 (54.46%) |

The return rate is ~55% — substantially higher than real-world fashion return rates (typically 20–40%). This is a direct consequence of the dataset sampling methodology: **only customers with at least one observed return are included**. See Limitations below.

---

## Missingness

Zero missing values in all columns across all six files. Every field is fully populated. This is unusual for real-world data and likely reflects pre-processing choices by the dataset authors (e.g., imputation or exclusion of incomplete records).

---

## Data Quality Issues

1. **Duplicate column name `customerId_level_return_code_D`** in both customer node tables: the 12th and 14th columns share the same name. **Confirmed to hold genuinely different data** (training means 0.0297 and 0.3972) — a column-naming defect in the source file, not an accidental exact duplicate. Standard pandas `df[col]` returns a DataFrame for duplicate names — positional access (`df.iloc[:, i]`) is required. Stage 1's `ml/data/loaders.py` avoids this entirely by selecting only named safe columns before either duplicate `_D` column is reachable.

2. **Duplicate column name `variantID_level_return_code_D`** in both product node tables: same defect pattern.

3. **Anonymization is incomplete**: `brandDesc` contains one real, un-anonymized value `Pull&Bear` (4,954 training rows), and `productType` contains one real value `Jeans` (10,441 training rows), alongside otherwise-anonymized codes. `Pull&Bear` is also the dropped reference category for the `Brand_*` one-hot dummies — there is no `Brand_H` as originally guessed.

4. **Node coverage gap**: 33,236 customers appear in `event_table_training.p` but have no entry in `customer_nodes_training.p` (5.23% of training events have no customer node). For variants: there are 193,666 variants in training events with no product node (34.66% of training events), and conversely 172,495 product nodes not referenced by any training event. Any ML pipeline using node features must handle these join misses gracefully (null imputation or exclusion). Measured target-rate impact of missingness differs sharply by side — see "Train / Test Structure" below.

5. **Product node aggregates are global, not per-split** (see Stage 1 correction note at top of document): `avgGbpPrice`, `avgDiscountValue`, `salesPerProduct`, and `returnsPerProduct` are byte-identical between the training and test product node files for all 405,505 shared variants. This means `productReturnRate` computed from the training file contains test-period return outcomes.

---

## Train / Test Structure

### What is known
- The dataset is pre-split into training and testing sets.
- No timestamp column exists in any of the six files.
- The test split is **larger** than training (1,460,366 vs. 1,369,133 events; +6.6%).
- Both splits have identical column schemas.

### What is unknown (UNKNOWN)
- Whether the split is chronological (time-based) or random.
- Whether the training period precedes the testing period.
- The exact computation window for `avgGbpPrice`/`avgDiscountValue` relative to any single predicted transaction (confirmed global across splits, but not confirmed to exclude the transaction being predicted).

### Confirmed (Stage 1 direct verification)

**Customer-side aggregates are recomputed independently per split.** For the 183,498 customers present in both customer node files, `salesPerCustomer` matches in only 4.83% of cases and `returnsPerCustomer` in only 9.80% — i.e. the values genuinely differ by split. Implied return rates from the aggregates are close to but not identical to the actual per-split event return rate:
- Training customer implied return rate (returnsPerCustomer / salesPerCustomer) ≈ 52.8%, vs. actual training event return rate 55.3%
- Testing customer implied return rate ≈ 51.4%, vs. actual 54.5%

This is consistent with within-period computation over that split's own transactions (same-window leakage) and inconsistent with aggregation over an independent historical window.

**Product-side aggregates are GLOBAL constants, computed once over the full dataset.** For all 405,505 variants present in both product node files, `avgGbpPrice`, `avgDiscountValue`, `salesPerProduct`, and `returnsPerProduct` are **byte-identical** (100.00% match) between the training and test files.

**Implication**: the customer-side return aggregates in the test node table encode same-period test outcomes (leakage at evaluation time). The product-side return aggregates are worse: because they are the same global value in both files, `productReturnRate` in the **training** file already contains test-period return outcomes — this is leakage available at *training* time, not just evaluation time.

---

## Customer / Product Overlap

### Customer ID overlap (node tables)

| Metric | Value |
|--------|-------|
| Train unique customer IDs | 777,001 |
| Test unique customer IDs | 825,598 |
| Overlap (seen in both) | 183,498 (22.2% of test) |
| Test-only (cold-start) | 642,100 (77.8% of test) |

### Product variant overlap (node tables)

| Metric | Value |
|--------|-------|
| Train unique variant IDs | 411,495 |
| Test unique variant IDs | 411,544 |
| Overlap (seen in both) | 405,505 (98.5% of test) |
| Test-only (cold-start) | 6,039 (1.5% of test) |

---

## Cold-Start Statistics

| Test event type | Count | Percentage |
|----------------|-------|-----------|
| Known customer + Known product | 289,180 | 19.80% |
| New customer + Known product | 769,222 | 52.67% |
| Known customer + New product | 120,698 | 8.26% |
| New customer + New product | 281,266 | 19.26% |
| **Any new customer** | **1,050,488** | **71.93%** |
| **Any new product** | **401,964** | **27.52%** |

"Known" means the customer/product appeared in `event_table_training.p`.

**71.9% of test events involve customers never seen during training.** This is a severe cold-start problem. Any customer-level features derived from training history (e.g., a customer's historical return rate) will be unavailable for nearly three-quarters of test predictions.

---

## Duplicate (Customer, Product) Pairs

| Split | Duplicate pairs |
|-------|----------------|
| Training | 10,250 |
| Testing | 11,106 |

These represent cases where the same customer purchased the same variant more than once. Whether each row should be treated independently or deduplicated depends on the ML objective; the current approach treats each row as an independent prediction unit.

---

## Key Quirks

1. **Biased sample**: `returnsPerCustomer` minimum is 1 and `returnsPerProduct` minimum is 1. The dataset only includes customers who have made at least one return, and likely only products that have been returned at least once. This inflates the measured return rate and means the training distribution does not reflect the general population.

2. **No event-level timestamps**: chronological ordering cannot be verified. Time-based train/val/test splits cannot be constructed from the event tables alone.

3. **All identifiers are hashed**: customer IDs and product IDs are cryptographic hashes. External enrichment (linking to real ASOS product catalogue) is not possible.

4. **Categorical columns are mostly, but not completely, anonymised**: country names and most brand/product-type values are replaced with codes (Country_A, Brand_B, productType_C, etc.), but `brandDesc` and `productType` each contain one real, un-anonymized value (`Pull&Bear`, `Jeans`). Categorical interpretation is largely, but not entirely, impossible — and other columns should not be assumed reliably anonymized just because they use a coded naming pattern.

5. **Node tables are split differently by entity type**: customer node aggregates (`salesPerCustomer`, `returnsPerCustomer`, `customerReturnRate`) are recomputed independently per train/test split. Product node aggregates (`avgGbpPrice`, `avgDiscountValue`, `salesPerProduct`, `returnsPerProduct`, `productReturnRate`) are the opposite — confirmed **global constants**, identical in both files. This asymmetry was not apparent in Stage 0 and was only discovered by direct byte-comparison in Stage 1.

6. **Return codes are post-return labels**: the `*_level_return_code_*` columns encode the distribution of return reason codes. These codes are assigned after a return decision is made and are fundamentally post-outcome data.

7. **Missing-node target-rate signal differs sharply by side**: training events with no customer node have a return rate of 64.6%, vs. 54.8% for events with a customer node (a 9.8-point gap) — plausibly a dataset-construction artifact rather than a generalizable signal, and deliberately not exposed as a model feature. Training events with no product node have a return rate of 55.5%, vs. 55.2% with a product node — a negligible, target-neutral gap.

---

## Suitability for ReturnGuard

The dataset is **usable but constrained**:

**Suitable for:**
- Learning the relationship between product attributes (type, brand, price) and return probability
- Learning the relationship between customer demographics (country, age, gender, membership) and return behaviour
- Demonstrating train/test evaluation methodology on a realistic scale (2.8M events)
- Cold-start analysis (strong signal about new customers)

**Not suitable for:**
- Calibrated probability estimates that apply to the general merchant population (selection bias inflates return rates)
- Use of aggregate return statistics as features without accepting target leakage risk
- Chronological validation (no timestamps)
- Entity-level feature engineering from within-split history (no transaction-level timestamps)

---

## Major Limitations

1. **Selection bias**: only customers with returns are included. The model cannot be calibrated for the full merchant population without adjustment.
2. **No timestamps**: temporal validation impossible; cannot construct a proper time-split evaluation.
3. **Anonymisation is incomplete**: most brand, country, and product type names are obscured, but `brandDesc`/`productType` each leak one real value (`Pull&Bear`, `Jeans`); domain knowledge is largely but not entirely inapplicable, and other "coded" columns should not be assumed trustworthy.
4. **Target leakage in supplied features, confirmed and worse than originally inferred**: customer-side return/sales aggregates are same-window leakage (recomputed per split); product-side return/sales aggregates are confirmed **global constants** — the training file's `productReturnRate` literally contains test-period return outcomes, a stronger leakage path than same-window contamination.
5. **Node-event coverage gap**: ~33K customers (5.2%) and ~467K variant-events (34.7%) in training events lack node features. The two sides carry very different target-rate signal (9.8-point gap for missing customer nodes vs. 0.3-point gap for missing product nodes), requiring different handling.
6. **Duplicate column names** (`*_return_code_D` repeated, holding genuinely different data) require careful handling in any pipeline — Stage 1 avoids the issue by selecting safe columns by name before either duplicate is reachable.
