# Stage 3 — Model Understanding, Governance, and Calibration Policy

Stage 3 does not optimize AUC, search new model families, fit a calibrator, or
compute official ASOS test model performance. It explains and audits the fixed
Stage 2 A3 and A4 configurations on a deterministic sample of the
training-derived primary validation fold.

## Evaluation posture

The ASOS test files were structurally inspected during Stage 0, including
labels and aggregate statistics. Test model performance remains uncomputed.
Primary-validation results are developmental because Stage 2 repeatedly
reported them; Stage 3 uses them for descriptive model understanding rather
than confirmatory claims.

## Explained models

- **A3:** donor-imputed customer features, catalogue features, suspicious
  global price/discount attributes, and derived price features. It has no
  product-frequency encoding.
- **A4:** A3 plus variant, parent-product, and supplier exposure counts. These
  counts are target-free and fold-local, but their point-in-time provenance is
  unverifiable without timestamps.

## SHAP methodology

Tree SHAP is calculated in raw-margin space. Additivity is verified against
the XGBoost raw margin; applying the logistic transform must reproduce model
probabilities. Raw SHAP values are log-odds contributions, not probability
points. Grouped SHAP first sums signed encoded-column contributions per row,
then calculates mean absolute grouped importance.

The shared sample is deterministic and stratified by target, product coverage,
and product novelty relative to the fitted inner-training fold. No official
test events are used.

## Reproduction and SHAP verification

The customer donor implementation was corrected before this analysis: prior
inference used a sequence-based pseudo-random choice, so scores for a missing
customer profile could depend on request batching or row order. Stage 3 makes
the donor deterministic from the customer ID and seed. A3 and A4 were then
re-fit with their existing fixed configurations, without feature selection or
hyperparameter tuning, so that artifacts reflect the repaired inference path.

| Model | Archived development ROC-AUC | Stage 3 re-fit ROC-AUC | Absolute delta |
|---|---:|---:|---:|
| A3 | 0.654767 | 0.654724 | 0.000043 |
| A4 | 0.666575 | 0.666866 | 0.000291 |

Both deltas are below the predeclared 0.003 reproduction tolerance. Tree SHAP
used a shared 10,000-row primary-validation sample after a 1,000-row pilot.
The final raw-margin additivity errors were 3.74e-6 (A3) and 5.37e-6 (A4);
after the logistic transform, maximum disagreement with `predict_proba` was
7.86e-8 and 7.33e-8 respectively. The check explicitly respects XGBoost early
stopping's best-iteration tree range.

## Global explainability results

A3's grouped mean absolute SHAP ranking is: shipping country (0.2912, 31.3%),
derived price features (0.1961, 21.1%), product type (0.1844, 19.8%), birth
year (0.0826, 8.9%), price/discount (0.0818, 8.8%), brand (0.0602, 6.5%), and
customer binary attributes (0.0350, 3.8%). A4 retains shipping country as its
largest group (0.2930, 27.2%), but product frequency ranks second (0.1980,
18.4%). The encoded and grouped rankings and beeswarm/dependence plots are
stored in `reports/stage3_a3_*` and `reports/stage3_a4_*`.

These attributions support the explicit manifests: no raw hashed identifier,
target-derived aggregate, return code, or supplied redundant dummy is present
in either explained feature matrix. Price and discount remain useful but are
labelled **SUSPICIOUS_GLOBAL_PRICE_DISCOUNT**, not promoted to proven
point-in-time inputs.

## Local explanation contract

`reports/stage3_local_explanations.json` contains 15 deterministic scenarios:
low, median, and high score; correct and incorrect classifications; product
coverage and novelty; customer-profile availability; and the largest A4/A3
score changes. Each internal explanation retains the raw margin and grouped
log-odds contributors for technical audit. The paired user-facing explanation
uses non-causal grouped return-risk factors, suppresses imputed donor
demographics, and never exposes raw IDs, encoded feature names, or SHAP values.

## Artifact and provenance audit

Stage 3 audits customer donor-imputation stability across transform batching,
row order, and alternative deterministic donor seeds. Donor-sampled attributes
are never displayed as real customer facts.

Product missingness is audited through score, calibration, performance, and
SHAP slices. Price and discount factors retain a SUSPICIOUS provenance label.
Raw hashed identifiers, target-derived aggregates, return codes, and source
dummies are prohibited by explicit feature manifests.

The repaired donor assignment is invariant in a 2,000-row audit: maximum
probability change was 0.0 when requests were split into batches or reordered.
It is nevertheless an imputation, not recovered customer truth. On 97
missing-customer rows, changing only the deterministic donor seed changed A3
scores by mean absolute 0.0779–0.0851 (maximum 0.2695–0.2726). This is an
important residual sensitivity and a deployment limitation; donor attributes
remain hidden in user-facing explanations.

Product-node-missing rows are materially harder. A3 ROC-AUC is 0.6816 when a
product node is covered and 0.5912 when it is missing; A4 is 0.6891 and
0.6195 respectively. The missing-product slice receives much less catalogue
and price attribution. A4's frequency group remains substantial there (0.2091
mean absolute SHAP), which further supports treating it as experimental rather
than a safe production remedy.

## Governance policy

The conservative model of record is A3, subject to reproduction, manifest,
SHAP, calibration, and artifact checks. A4 is the experimental challenger:
its frequency features remain temporally unverifiable even if SHAP finds them
auxiliary. A4 cannot become a merchant deployment model until point-in-time
counts and temporal validation are available.

Although its aggregate classification is `AUXILIARY` under the predeclared
rule, A4 frequency is not trivial: it is the second grouped factor and appears
among the top three encoded local contributors for 83.6% of the SHAP sample.
Compared with A3, A4 raises average score by 0.0535 for the lowest variant-
exposure bucket and lowers it by 0.0578 for the highest. This systematic score
movement cannot be governed as point-in-time safe without event timestamps.

## Calibration policy

Stage 3 measures calibration but fits no calibrator. Scores remain
dataset-conditional because the source sample is enriched for customers with
at least one return. Future merchant calibration requires representative,
later data and a separate assessment window.

On this developmental, returner-enriched fold, A3 has Brier 0.2281, ECE
0.0042, calibration intercept 0.0156, and slope 0.9873; A4 has Brier 0.2245,
ECE 0.0066, intercept -0.0238, and slope 0.9877. These diagnostics describe
only this conditioned dataset and are not evidence of merchant-wide
calibration. No Platt, isotonic, or other score-transforming calibrator was
fitted, stored, or applied.

## Output language

- Internal numeric field: `raw_model_probability`.
- Technical term: dataset-conditional estimated return probability.
- User-facing term: return-risk score.

The required note is: “This score is estimated from a returner-enriched ASOS
research sample. It supports relative risk ranking but is not a merchant-wide
probability of return.”

## Model decision and freeze boundary

A3 is the **conservative model-of-record freeze candidate**: 41 approved
features, the fixed XGBoost configuration, and 659 boosting rounds (best
iteration 658 plus one). A4 is the **experimental challenger**, not a
deployment candidate. The exact manifest and frozen procedure are in
`reports/stage3_a3_freeze_spec.json`.

The next evaluation boundary is one predeclared official-test performance run
only after this specification and its documentation are committed. It must not
be used to choose a different feature set, configuration, or calibrator.
