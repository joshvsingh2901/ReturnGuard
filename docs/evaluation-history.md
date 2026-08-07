# Evaluation History — ReturnGuard

ReturnGuard separates test-statistic inspection, development validation, model
selection, and test-model performance evaluation. These distinctions are
important because the ASOS test files were inspected before Stage 3.

## Test-statistic inspection

This occurred during Stage 0 dataset auditing and structural tests. The project
inspected test schemas, labels, aggregate prevalence, duplicate-pair counts,
and entity overlap/cold-start structure. Test labels and statistics are
therefore known; the official test split is not literally unopened or blind.

## Validation-driven development

This occurred in Stages 1 and 2 using training-derived splits. Stage 1 used a
deterministic customer-grouped primary validation fold. Stage 2 repeatedly
calculated and displayed primary-validation metrics across A1–A4 and tuning
configurations. The primary validation fold is developmental evidence and is
not pristine confirmatory evidence.

## Model selection

Stage 2 selected rungs and tuning configurations programmatically with the
inner early-stop fold. That separation is useful, but it does not erase the
fact that primary-validation metrics were repeatedly visible.

## Test model performance

No model predictions, ranking metrics, calibration metrics, or slice metrics
have been computed on the official ASOS test events. Stage 3 records A3 as the
conservative freeze candidate and stores its predeclared procedure in
`reports/stage3_a3_freeze_spec.json`. One held-out test performance run may be
executed only after this specification and the documentation are committed; its
results must not reopen model selection.
