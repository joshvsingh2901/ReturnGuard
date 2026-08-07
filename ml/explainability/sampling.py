"""Deterministic representative sampling for Stage 3 explanations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from ml.data.joins import has_product_node_mask
from ml.data.schema import EVENT_PROD_COL, TARGET_COL


DEFAULT_SHAP_SAMPLE_SIZE = 10_000
PILOT_SHAP_SAMPLE_SIZE = 1_000
FALLBACK_SHAP_SAMPLE_SIZE = 5_000
SAMPLE_SEED = 20260807


@dataclass(frozen=True)
class SampleManifest:
    """Portable record of a deterministic Stage 3 explanation sample."""

    seed: int
    requested_n: int
    selected_n: int
    index_values: list
    stratum_counts: dict[str, int]
    manifest_hash: str

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "requested_n": self.requested_n,
            "selected_n": self.selected_n,
            "index_values": self.index_values,
            "stratum_counts": self.stratum_counts,
            "manifest_hash": self.manifest_hash,
        }


def explanation_strata(
    df: pd.DataFrame,
    train_product_ids: set,
    target_col: str = TARGET_COL,
) -> pd.Series:
    """Return target/coverage/product-cold-start strata for sampling.

    Primary validation is already customer-cold by construction. Product
    coverage and product novelty remain useful dimensions to preserve.
    """
    covered = has_product_node_mask(df).map({True: "covered", False: "missing"})
    product_seen = df[EVENT_PROD_COL].isin(train_product_ids).map(
        {True: "known", False: "new"}
    )
    target = df[target_col].astype(int).astype(str)
    return (
        "target=" + target
        + "|product=" + covered
        + "|product_seen=" + product_seen
    )


def _proportional_allocations(counts: pd.Series, requested_n: int) -> pd.Series:
    """Largest-remainder allocation preserving observed stratum proportions."""
    if requested_n <= 0:
        raise ValueError("requested_n must be positive")
    total = int(counts.sum())
    if requested_n >= total:
        return counts.astype(int)

    exact = counts / total * requested_n
    allocations = np.floor(exact).astype(int)
    remainder = requested_n - int(allocations.sum())
    if remainder:
        fractional = (exact - allocations).sort_values(ascending=False)
        for label in fractional.index[:remainder]:
            allocations.loc[label] += 1
    return allocations.astype(int)


def _hash_manifest(seed: int, requested_n: int, indices: list, strata: dict[str, int]) -> str:
    payload = json.dumps(
        {
            "seed": seed,
            "requested_n": requested_n,
            "index_values": indices,
            "stratum_counts": strata,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def representative_validation_sample(
    df: pd.DataFrame,
    train_product_ids: set,
    n: int = DEFAULT_SHAP_SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> tuple[pd.DataFrame, SampleManifest]:
    """Sample primary-validation rows proportionally and deterministically.

    The source frame must have a unique index so the selected rows can be
    persisted and applied identically to A3 and A4.
    """
    if not df.index.is_unique:
        raise ValueError("Stage 3 sample source must have a unique DataFrame index")
    if n > len(df):
        raise ValueError(f"Requested {n} rows from a frame with only {len(df)} rows")

    strata = explanation_strata(df, train_product_ids)
    counts = strata.value_counts(sort=True)
    allocations = _proportional_allocations(counts, n)
    rng = np.random.default_rng(seed)

    selected = []
    for label in sorted(allocations.index):
        available = np.sort(df.index[strata == label].to_numpy())
        take = int(allocations.loc[label])
        if take:
            selected.extend(rng.choice(available, size=take, replace=False).tolist())

    selected = sorted(selected)
    sample = df.loc[selected].copy()
    sample_strata = explanation_strata(sample, train_product_ids)
    stratum_counts = {
        str(label): int(count)
        for label, count in sample_strata.value_counts(sort=True).items()
    }
    index_values = [int(i) if isinstance(i, (np.integer, int)) else str(i) for i in selected]
    manifest_hash = _hash_manifest(seed, n, index_values, stratum_counts)
    manifest = SampleManifest(
        seed=seed,
        requested_n=n,
        selected_n=len(sample),
        index_values=index_values,
        stratum_counts=stratum_counts,
        manifest_hash=manifest_hash,
    )
    return sample, manifest
