"""
Joint (hot-deck) donor imputation for customer-side missing-node rows.

Stage 1 found that events with no customer node have a return rate 9.8
points higher than joined events (64.6% vs. 54.8%) — believed to be a
dataset-construction artifact, not a generalisable signal (see
docs/leakage-audit.md, docs/stage1-baseline.md). Stage 1's Logistic
Regression could not exploit this because median/mode imputation cannot
be represented by a linear model as a distinguishing pattern. A tree
model can: median/mode imputation gives every missing-node row the exact
same value on all four customer fields simultaneously, and a tree builds
that conjunction natively in a few splits.

DonorImputer prevents this by replacing an entire missing customer
profile with one COMPLETE, REAL customer profile sampled from the
training fold — never independently per field, which would destroy the
real joint distribution and produce implausible combinations a tree could
still learn to detect (see ml.evaluation.probe for the diagnostic that
verifies this).
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.data.schema import EVENT_CUST_COL

CUSTOMER_COLS = ["yearOfBirth", "isMale", "premier", "shippingCountry"]


class DonorImputer(BaseEstimator, TransformerMixin):
    """
    fit(X): builds the donor pool from rows of X with a fully complete
    profile across all four CUSTOMER_COLS. Node presence alone (isMale
    non-null — unlike yearOfBirth, isMale is NEVER NaN for any reason
    other than a missing customer node; Stage 0/1 confirmed zero raw
    nulls in the customer node table) is not sufficient: a donor customer
    can have a present node but their OWN yearOfBirth sentinel (cleaned to
    NaN upstream), and drawing such a row as a donor would silently carry
    that NaN into the target row, defeating the "always fully imputed"
    guarantee. Requiring completeness on all four columns closes this.
    Also stores the train-fold median yearOfBirth (computed over the same
    complete-profile rows), for the separate "node present, birth year
    unknown" case.

    transform(X): for each row with a fully-missing customer node, copies
    all four CUSTOMER_COLS values from ONE donor row. The donor index is a
    stable hash of the customer ID and random_state, rather than a sequence
    of random draws. This makes the imputation invariant to transform batch
    size and row ordering, and gives repeated events for the same missing
    customer a consistent synthetic profile. For rows where only
    yearOfBirth is missing (the sentinel case; the other three fields are
    real), fills yearOfBirth alone with the train-fold median — this is a
    real profile with one unknown field, not the node-missing artifact, so
    it does not need joint donor treatment.

    fit() must only ever be called with a training fold — this class
    performs no internal train/val splitting; the caller (a Pipeline, or
    equivalent fold-safe construction in scripts/train_stage2.py) is
    responsible for ensuring fit() never sees validation rows.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "DonorImputer":
        complete = X[CUSTOMER_COLS].notna().all(axis=1)
        pool = X.loc[complete, CUSTOMER_COLS].reset_index(drop=True)
        if len(pool) == 0:
            raise ValueError(
                "DonorImputer.fit: no rows with a fully complete customer "
                "profile were found in the training fold — cannot build a "
                "donor pool."
            )
        self.donor_pool_ = pool
        self.year_median_ = X.loc[X["isMale"].notna(), "yearOfBirth"].median()
        return self

    def _stable_donor_indices(self, X: pd.DataFrame, node_missing: pd.Series) -> np.ndarray:
        """Map each missing customer to one donor independently of row order.

        The joined event frame contains the hashed customer ID even when the
        customer node itself is absent. Using that ID both avoids batch-order
        dependent inference and keeps all events from the same missing
        customer internally consistent. The index fallback exists only for
        minimal synthetic fixtures that omit the event key.
        """
        if EVENT_CUST_COL in X.columns:
            identities = X.loc[node_missing, EVENT_CUST_COL]
        else:
            identities = X.index[node_missing]

        n_donors = len(self.donor_pool_)
        donor_indices = []
        for identity in identities:
            payload = f"{self.random_state}|{identity}".encode("utf-8")
            digest = hashlib.blake2b(payload, digest_size=8).digest()
            donor_indices.append(int.from_bytes(digest, "little") % n_donors)
        return np.asarray(donor_indices, dtype=np.int64)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # If every row in this batch happens to be missing shippingCountry
        # (e.g. an all-missing edge case), pandas infers a float64 column
        # from the all-NaN data; assigning donor strings into it would
        # then warn (and eventually error) on the dtype change. Force
        # object dtype defensively so that never happens.
        X["shippingCountry"] = X["shippingCountry"].astype(object)

        node_missing = X["isMale"].isna()
        n_missing = int(node_missing.sum())

        if n_missing > 0:
            donor_idx = self._stable_donor_indices(X, node_missing)
            donor_rows = self.donor_pool_.iloc[donor_idx]
            for col in CUSTOMER_COLS:
                X.loc[node_missing, col] = donor_rows[col].to_numpy()

        # Captured before any mutation above, so this isolates rows where
        # the node was present but yearOfBirth alone was the sentinel.
        year_only_missing = X["yearOfBirth"].isna() & ~node_missing
        X.loc[year_only_missing, "yearOfBirth"] = self.year_median_

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray(CUSTOMER_COLS)
