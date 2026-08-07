"""Synthetic tests for deterministic raw-data manifest behavior."""

from __future__ import annotations

import pickle

import pandas as pd

from ml.lifecycle.hashing import build_dataset_manifest, verify_training_files


def test_manifest_records_all_files_but_normal_verification_reads_training_only(tmp_path):
    names = {
        "event_table_training.p": "training",
        "customer_nodes_training.p": "training",
        "product_nodes_training.p": "training",
        "event_table_testing.p": "historical_test",
        "customer_nodes_testing.p": "historical_test",
        "product_nodes_testing.p": "historical_test",
    }
    for index, name in enumerate(names):
        with (tmp_path / name).open("wb") as handle:
            pickle.dump(pd.DataFrame({"id": [index], "value": [float(index)]}), handle)

    manifest = build_dataset_manifest(tmp_path)
    observed = verify_training_files(manifest, tmp_path)

    assert len(manifest["files"]) == 6
    assert set(observed) == {name for name, role in names.items() if role == "training"}
    assert manifest["combined_training_dataset_digest"]
