# Final Official ASOS Test Evaluation — Frozen A3

## Evaluation posture

This is a **held-out performance evaluation with prior aggregate test
inspection**. Stage 0 had already inspected official-test structure, labels,
and aggregate statistics, so this is not described as blind or untouched.
Before this run, no model predictions, performance metrics, calibration
metrics, or slice metrics had been computed on official ASOS test events.

The evaluation uses the Stage 3 A3 freeze exactly: no frequency features, 41
approved transformed features, train-only preprocessing, and 659 fixed XGBoost
boosting rounds. A4 was not trained or evaluated; no model selection,
calibration fitting, feature engineering, or tuning occurred.

## Freeze and execution record

| Item | Value |
|---|---|
| Frozen model | A3 conservative model of record |
| Freeze spec hash | `44893679b2145474167717befc53c89e68e8ec0e0d69131afd1619f31c69508f` |
| Feature manifest hash | `75834b5b656209a0266d8f5073946c4493d10ec738a71014d635c0851ca16404` |
| Freeze source revision | `5499beb` |
| Evaluation revision | `37a4c05` |
| Training rows | 1,369,133 |
| Test rows | 1,460,366 |
| Fixed boosting rounds | 659 |
| Calibrator | None |
| Test probability generations | 1 |

Package versions were numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1,
XGBoost 2.1.4, and SHAP 0.47.2. The evaluation timestamp was
2026-08-07T06:54:00Z. Exact machine-readable provenance is in
`reports/final_test_metrics.json`.

## Pre-score evaluator correction

The first evaluator invocation at revision `c5b0456` stopped during final
training manifest verification because the new runner used the wrong manifest
key name. It stopped before classifier fitting, official-test-frame loading,
test prediction, or test metrics. Commit `37a4c05` corrected the check and
added a regression test. Rerunning was necessary because no official test
score had been generated; the recorded execution above is the first and only
test probability generation.

## Official test metrics

| Metric | Frozen A3 official test |
|---|---:|
| ROC-AUC | 0.656798 |
| Log loss | 0.648265 |
| Brier score | 0.228614 |
| PR-AUC | 0.682864 |
| ECE | 0.009677 |
| Accuracy at 0.5 | 0.613606 |
| Precision at 0.5 | 0.618722 |
| Recall at 0.5 | 0.757090 |
| F1 at 0.5 | 0.680948 |

The fixed 0.5 reporting confusion matrix is `[[293925, 371075],
[193202, 602164]]`. It is not a business-optimized operating threshold.

## Development-validation comparison

Stage 3 primary validation is developmental rather than pristine. Relative to
its A3 values, absolute official-test deltas are 0.002074 ROC-AUC, 0.001249
log loss, 0.000533 Brier, and 0.005446 ECE. Overall discrimination and proper
scoring losses are broadly consistent with development validation; this is
descriptive evidence only and does not reopen model selection.

## Predeclared slices

| Slice | n | ROC-AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| Known customer + known product | 289,180 | 0.666717 | 0.640817 | 0.225225 |
| New customer + known product | 769,222 | 0.663460 | 0.648694 | 0.228844 |
| Known customer + new product | 120,698 | 0.639895 | 0.644612 | 0.226786 |
| New customer + new product | 281,266 | 0.635257 | 0.656317 | 0.232251 |
| Product node available | 1,008,608 | 0.680710 | 0.637254 | 0.223474 |
| Product node missing | 451,758 | 0.589317 | 0.672848 | 0.240089 |
| Customer node available | 1,385,362 | 0.660830 | 0.647466 | 0.228260 |
| Customer node missing | 75,004 | 0.581518 | 0.663030 | 0.235151 |

All three predeclared slice families reconcile exactly to 1,460,366 rows. The
product-node-missing gap is substantial (0.5893 versus 0.6807 AUC); it is
consistent with Stage 3's product-missing limitation, not a trigger for a
post-test feature change. Customer-node-missing and both-new rows are also
materially harder. Coverage-slice results closely match the corresponding
development product-coverage values (A3 0.6816 covered and 0.5912 missing).

## Calibration and population scope

Official-test calibration-in-the-large is -0.0424 and calibration slope is
1.0012. The reliability curve and fixed risk bands are recorded in
`reports/final_test_calibration.json`; observed return rates are generally
below predicted rates across the reported bands. No calibrator was fit or
applied.

These outputs remain **dataset-conditional return-risk estimates**. The test
set belongs to the same returner-enriched dataset family and therefore does
not establish merchant-wide probability calibration or resolve the population
shift limitation.

## Development stop

The predeclared official-test evaluation is complete. A3 must not be modified
because of these results, and A4 must not be promoted, retrained, or evaluated
for selection. Any future Stage 4 work is operational governance for the
frozen record, not renewed ML development.
