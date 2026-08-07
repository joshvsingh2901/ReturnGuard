# Model Card — ReturnGuard `returnguard-a3-v1`

## Status

The frozen research model of record is **A3**, semantic version
`returnguard-a3-v1`: 41 approved features and 659 fixed boosting rounds. Its held-out performance evaluation with prior aggregate
test inspection gives ROC-AUC 0.656798, log loss 0.648265, and Brier 0.228614.
These results do not change model selection. A4 remains an experimental
challenger because its frequency features cannot be proven point-in-time valid
without timestamps, despite its developmental 0.666866 ROC-AUC.

Lifecycle provenance is recorded in
[`artifacts/manifests/returnguard-a3-v1.json`](../artifacts/manifests/returnguard-a3-v1.json):
dataset version `asos-graphreturns-osf-c793h-v1`, feature-manifest hash
`75834b5b656209a0266d8f5073946c4493d10ec738a71014d635c0851ca16404`, and
freeze-spec hash `44893679b2145474167717befc53c89e68e8ec0e0d69131afd1619f31c69508f`.
The local MLflow registered model is `returnguard-a3`; explicit governance
assigns its `model-of-record` alias.

## Intended use

Rank item-purchase return risk within the returner-enriched ASOS research
population. The output is not a merchant-wide probability of return.

It must not be used for merchant-wide pricing, eligibility, customer
profiling, automated adverse decisions, causal explanation, or as a calibrated
probability outside this dataset family.

## Input policy

Approved inputs are customer demographics, catalogue attributes, suspicious
global price/discount fields, and A3 derived price features. Target-derived
aggregates, return codes, raw IDs, and supplied redundant dummies are banned.

## Missingness policy

Missing customer nodes receive a stable customer-ID keyed joint donor profile;
these synthetic donor attributes are not exposed as user facts. Product
categoricals use `__MISSING__`; product numeric NaNs use native XGBoost
routing. Batch and row-order invariance are verified, but alternative donor
seeds materially move scores for missing-customer rows; this residual
imputation sensitivity is an explicit limitation.

## Calibration policy

No calibrator is fitted. Calibration diagnostics describe only the
returner-enriched development population. Merchant-specific calibration needs
representative later merchant data.

The official test slope is 1.0012 and calibration-in-the-large is -0.0424,
but it remains within the same returner-enriched dataset family and does not
justify merchant-wide probability claims or calibrator fitting.

## Explanation policy

Tree SHAP explains raw model margin. User-facing explanations are grouped,
non-causal return-risk factors. Raw SHAP values, encoded names, and donor
demographics remain internal.

## Known limitations

The inputs have no timestamps, so product frequency features are temporally
unverifiable and A4 is not eligible for deployment. Price and discount fields
remain suspicious global attributes. Product-node-missing events have lower
development discrimination (A3 ROC-AUC 0.5912 versus 0.6816 when covered).
The source sample includes returners only, so all scores are dataset-
conditional rather than merchant-wide probabilities.

## Evaluation history

See [evaluation-history.md](evaluation-history.md). Test labels/statistics
were inspected during audit. The one predeclared frozen-A3 performance
evaluation is now complete; see [final-test-evaluation.md](final-test-evaluation.md).

## Reproducibility and artifact provenance

Run `.venv/bin/python scripts/reproduce_model.py --profile full` to verify the
training-data hashes, frozen specification, deterministic split, validation
metrics, composite artifact reload, and safe reference predictions. The command
does not load official test events or labels. Local MLflow owns rebuilt model
binaries; Git tracks the manifests and small reference fixture only.

The project requires semantic reproducibility: configuration, feature order,
data identity, metrics, and fixed reference predictions must meet declared
tolerances. Byte-identical XGBoost/joblib files are recorded when practical but
are not required across platforms and serializer builds. See
[stage4-mlops.md](stage4-mlops.md).
