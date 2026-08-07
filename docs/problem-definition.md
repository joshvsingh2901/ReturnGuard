# Problem Definition — ReturnGuard

---

## Prediction Unit

One row in the event table: a single purchase of one product variant by one customer.

Each row is treated as an independent prediction unit regardless of whether the same customer or variant appears in other rows.

---

## Target

**`isReturned`** — whether the purchased item was subsequently returned.

- `1` = the item was returned
- `0` = the item was kept

---

## Prediction Moment

The model is intended to generate a prediction **at the moment of purchase completion**, before any return decision is known.

Formally: at time `t_purchase`, the model must output `P(isReturned = 1 | information available at t_purchase)`.

---

## Core Feature Rule

> A feature is only permitted if its value could have been observed or computed **at or before `t_purchase`**, without using any information about whether the current or future transactions result in returns.

This rule has three practical implications:

1. **Static attributes are safe**: customer demographics (age, gender, country, membership), product catalogue attributes (type, brand, price at listing), and discount at point of purchase are all known before the purchase completes.

2. **Aggregate return statistics are unsafe** if they were computed using the same transactions being predicted: `returnsPerCustomer`, `customerReturnRate`, `returnsPerProduct`, `productReturnRate`, and all `*_level_return_code_*` columns encode information about return outcomes and must not be used without proof that they were computed exclusively from transactions prior to and distinct from the ones being predicted.

3. **Historical aggregates may be safe if computed from a prior window**: for example, a customer's return rate computed over purchases made *before* the training/test split window would not constitute leakage. The current dataset does not provide evidence that the supplied aggregates satisfy this condition — they appear to be within-period aggregates.

---

## Known Population Limitation

**The ASOS GraphReturns dataset only contains customers who have made at least one observed return** (confirmed by `returnsPerCustomer` minimum = 1 and `returnsPerProduct` minimum = 1 in all node tables).

This has two important consequences:

### 1. Inflated measured return rates

The training return rate is 55.3% and the test rate is 54.5%. Real-world fashion return rates are typically 20–40% for online retail. The excess is explained by the biased sampling: customers who have never returned are excluded from the dataset, so the model sees a population that systematically over-returns compared to the full merchant population.

### 2. Calibration will not generalise without adjustment

A model trained on this data will learn to output probabilities appropriate for the *returner* sub-population. If the model is later served to all customers — including those who have never returned — its predicted probabilities will be systematically too high. Probability calibration (e.g., Platt scaling or isotonic regression) applied to a held-out sample from the full population will be required before deployment.

This limitation should be documented prominently in any model card or evaluation report produced in later stages.

---

## Evaluation Consideration: No Timestamps

Because the dataset contains no event-level timestamps, it is not currently possible to construct a proper chronological train/validation/test split from first principles. The pre-existing train/test split supplied by the dataset authors is reserved for a later held-out model-performance evaluation, but:

- The split methodology is not documented
- Whether it is chronological is UNKNOWN
- A custom validation set must be created for model selection (e.g., by holding out 20% of training events randomly, or by treating some customers as a holdout group)

The official test files were inspected during Stage 0 for labels, aggregate
statistics, schema, and cold-start structure. No model performance has been
computed on test events; see `docs/evaluation-history.md` for the distinction.

---

## Cold-Start Scope

71.9% of test events involve customers not seen during training. A production model must have a strategy for new customers. The safe baseline approach (demographics + product attributes only) is inherently cold-start capable because it does not rely on any customer history. This is an advantage of the conservative safe feature set.

---

## What This Project Does Not Predict

- The reason for a return (return code)
- Whether a *customer* will return (as opposed to a specific item)
- The number of items in a basket that will be returned
- The monetary value of a return

The scope is strictly binary classification at the item purchase level.
