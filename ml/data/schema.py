"""
Column-level schema constants for the ASOS GraphReturns dataset.

This is the single source of truth for which raw columns are safe to use
as model features, which are excluded as target leakage, and which
identifier/target columns exist. See docs/leakage-audit.md for the full
per-column rationale.
"""

# ---------------------------------------------------------------------------
# Identifiers / target
# ---------------------------------------------------------------------------

EVENT_CUST_COL = "hash(customerId)"
EVENT_PROD_COL = "hash(variantID)"
TARGET_COL = "isReturned"

CUSTOMER_ID_COL = "hash(customerId)"
PRODUCT_ID_COL = "hash(variantID)"

# ---------------------------------------------------------------------------
# Safe features actually used in the Stage 1 baseline
# ---------------------------------------------------------------------------

# Customer-side safe features (static demographics; stable across the
# train/test customer node files for shared customers — see
# docs/leakage-audit.md "Key Evidence" section).
CUSTOMER_SAFE_FEATURES = [
    "yearOfBirth",
    "isMale",
    "shippingCountry",
    "premier",
]

# Product-side safe features used in LR-A (no price/discount).
PRODUCT_SAFE_FEATURES_STRICT = [
    "productType",
    "brandDesc",
]

# Product-side safe features used in LR-B (adds global per-variant price
# and discount — confirmed byte-identical across train/test product node
# files for shared variants, i.e. NOT computed independently per split).
PRODUCT_SAFE_FEATURES_WITH_PRICE = PRODUCT_SAFE_FEATURES_STRICT + [
    "avgGbpPrice",
    "avgDiscountValue",
]

NUMERIC_FEATURES = ["yearOfBirth"]
NUMERIC_FEATURES_PRICE = ["avgGbpPrice", "avgDiscountValue"]
BINARY_FEATURES = ["isMale", "premier"]

# shippingCountry is customer-side: it can be NaN when the customer node
# is missing (~5.2% of events) and must be mode-imputed like isMale/
# premier, so missing-customer-node status doesn't leak through as an
# implicit "nan" one-hot category.
CATEGORICAL_FEATURES_CUSTOMER = ["shippingCountry"]

# productType/brandDesc are product-side: ml.data.joins fills missing
# values with an explicit "__MISSING__" token upstream, so they are never
# NaN by the time they reach preprocessing and need no imputer here.
CATEGORICAL_FEATURES_PRODUCT = ["productType", "brandDesc"]

CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_CUSTOMER + CATEGORICAL_FEATURES_PRODUCT

# ---------------------------------------------------------------------------
# Columns that must NEVER appear in the feature matrix (target leakage)
# ---------------------------------------------------------------------------

LEAKY_CUSTOMER_COLUMNS = [
    "salesPerCustomer",
    "returnsPerCustomer",
    "customerReturnRate",
    "customerId_level_return_code_A",
    "customerId_level_return_code_B",
    "customerId_level_return_code_C",
    "customerId_level_return_code_D",
    "customerId_level_return_code_E",
    "customerId_level_return_code_F",
    "customerId_level_return_code_G",
    "customerId_level_return_code_H",
    "customerId_level_return_code_I",
    "customerId_level_return_code_J",
    "customerId_level_return_code_K",
    "customerId_level_return_code_L",
]

LEAKY_PRODUCT_COLUMNS = [
    "salesPerProduct",
    "returnsPerProduct",
    "productReturnRate",
    "variantID_level_return_code_A",
    "variantID_level_return_code_B",
    "variantID_level_return_code_C",
    "variantID_level_return_code_D",
    "variantID_level_return_code_E",
    "variantID_level_return_code_F",
    "variantID_level_return_code_G",
    "variantID_level_return_code_H",
    "variantID_level_return_code_I",
    "variantID_level_return_code_J",
    "variantID_level_return_code_K",
    "variantID_level_return_code_L",
]

# Supplied one-hot dummy columns — redundant with the raw categorical
# columns we encode ourselves, and inconsistent in construction (some are
# full dummies, some drop a reference category). Never used.
SUPPLIED_DUMMY_COLUMNS = (
    [f"Country_{c}" for c in "ABCDEFGHI"]
    + [f"Brand_{c}" for c in "ABCDEFGIJK"]  # Brand_H absent from source
    + [f"productType_{c}" for c in "ABCDEFGHIJK"]
)

# High-cardinality identifiers excluded from Stage 1. Stage 2 evaluated
# target encoding these (hash(supplierRef), hash(productID)) and rejected
# it — see docs/leakage-audit.md "Stage 2: Target Encoding Rejected" —
# because there are no timestamps to prove any encoding-fold event
# precedes the row being scored. They remain excluded as direct
# identifiers; Stage 2 instead derives fold-local, target-free FREQUENCY
# features from them (see FREQUENCY_FEATURES below).
EXCLUDED_ID_COLUMNS = ["hash(supplierRef)", "hash(productID)"]

PRODUCT_PARENT_ID_COL = "hash(productID)"
SUPPLIER_ID_COL = "hash(supplierRef)"

LEAKY_COLUMNS = (
    LEAKY_CUSTOMER_COLUMNS
    + LEAKY_PRODUCT_COLUMNS
    + SUPPLIED_DUMMY_COLUMNS
    + EXCLUDED_ID_COLUMNS
)

# ---------------------------------------------------------------------------
# Known data-quality sentinels
# ---------------------------------------------------------------------------

# yearOfBirth == 1900 is a confirmed missing-value sentinel (1.54% of
# training customers), not a real birth year. Converted to NaN before
# preprocessing. No other values are clipped or altered.
YEAR_OF_BIRTH_SENTINEL = 1900

MISSING_CATEGORY_TOKEN = "__MISSING__"

# ---------------------------------------------------------------------------
# Stage 2 — derived and frequency features
# ---------------------------------------------------------------------------

# Derived from safe columns only (avgGbpPrice, avgDiscountValue,
# productType, brandDesc) plus fold-local statistics (group medians)
# learned exclusively from the training fold. Classified SAFE — see
# ml/features/derived.py and docs/stage2-gbdt.md for the exact formulas
# and the avgDiscountValue percentage-units assumption.
DERIVED_FEATURES = [
    "discount_amount",
    "price_rel_type_median",
    "price_rel_brand_median",
]

# Fold-local, target-free exposure counts for product-side identifiers.
# Classified SUSPICIOUS/experimental: safer than the banned
# salesPerProduct/salesPerCustomer (which are global constants spanning
# the test period — see docs/leakage-audit.md), but still carries an
# unfalsifiable within-window-exposure assumption in the absence of
# timestamps. See ml/features/frequency.py. Deliberately excludes any
# customer-ID frequency feature — under the customer-grouped primary
# split every validation customer has a train-fold count of exactly 0 by
# construction, which would be a textbook train/serve skew.
FREQUENCY_FEATURES = [
    "variant_event_count",
    "product_event_count",
    "supplier_event_count",
]
