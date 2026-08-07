"""Deterministic hashes and ASOS raw-data manifests.

The manifest deliberately treats the raw pickle files as the canonical data
asset.  It does not hash pandas-derived pickle files, whose serialisation can
vary across pandas versions without changing the underlying data.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

import ml.data.compat  # noqa: F401 - required before unpickling historical files
from ml.data.loaders import RAW_DIR

DATASET_VERSION = "asos-graphreturns-osf-c793h-v1"
SCHEMA_VERSION = "asos-raw-schema-v1"
JOIN_LOGIC_VERSION = "safe-left-join-v1"
RAW_FILE_ROLES = {
    "event_table_training.p": "training",
    "customer_nodes_training.p": "training",
    "product_nodes_training.p": "training",
    "event_table_testing.p": "historical_test",
    "customer_nodes_testing.p": "historical_test",
    "product_nodes_testing.p": "historical_test",
}


def canonical_json_bytes(value: object) -> bytes:
    """Encode an object in the single JSON form used for governance hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_payload(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_file_record(path: Path, role: str) -> dict:
    with path.open("rb") as handle:
        frame = pickle.load(handle)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path.name}, got {type(frame)!r}")
    return {
        "filename": path.name,
        "role": role,
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
        "row_count": int(len(frame)),
        # Lists preserve upstream duplicate columns, which a dtype dictionary
        # would lose.
        "ordered_columns": list(frame.columns),
        "expected_dtypes": [str(dtype) for dtype in frame.dtypes],
    }


def training_dataset_digest(files: Iterable[dict]) -> str:
    """Hash only the training raw-file identity plus transform contracts."""
    training = [
        {"filename": item["filename"], "sha256": item["sha256"]}
        for item in files
        if item["role"] == "training"
    ]
    training.sort(key=lambda item: item["filename"])
    return sha256_payload(
        {
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "join_logic_version": JOIN_LOGIC_VERSION,
            "training_files": training,
        }
    )


def build_dataset_manifest(raw_dir: Path = RAW_DIR) -> dict:
    """Inspect all six known raw files for the one-time historical manifest."""
    files = [_raw_file_record(raw_dir / name, role) for name, role in RAW_FILE_ROLES.items()]
    return {
        "dataset_version": DATASET_VERSION,
        "source": "ASOS GraphReturns OSF c793h",
        "schema_version": SCHEMA_VERSION,
        "join_logic_version": JOIN_LOGIC_VERSION,
        "files": files,
        "combined_training_dataset_digest": training_dataset_digest(files),
        "historical_test_policy": "Recorded for provenance only; normal lifecycle commands do not load or verify historical test files.",
    }


def write_dataset_manifest(path: Path, raw_dir: Path = RAW_DIR) -> dict:
    manifest = build_dataset_manifest(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_dataset_manifest(manifest: dict) -> None:
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Unexpected dataset version")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected raw schema version")
    if manifest.get("join_logic_version") != JOIN_LOGIC_VERSION:
        raise ValueError("Unexpected join-logic version")
    files = manifest.get("files", [])
    if {item.get("filename") for item in files} != set(RAW_FILE_ROLES):
        raise ValueError("Dataset manifest must contain exactly the six ASOS raw files")
    for item in files:
        if item.get("role") != RAW_FILE_ROLES[item["filename"]]:
            raise ValueError(f"Incorrect role for {item['filename']}")
        if not item.get("ordered_columns") or len(item.get("ordered_columns", [])) != len(item.get("expected_dtypes", [])):
            raise ValueError(f"Invalid schema record for {item['filename']}")
    if manifest.get("combined_training_dataset_digest") != training_dataset_digest(files):
        raise ValueError("Training dataset digest does not match manifest files")


def verify_training_files(manifest: dict, raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """Verify only training files; this is the safe normal lifecycle path."""
    validate_dataset_manifest(manifest)
    observed = {}
    for item in manifest["files"]:
        if item["role"] != "training":
            continue
        path = raw_dir / item["filename"]
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["byte_size"]:
            raise ValueError(f"Training file size mismatch: {path.name}")
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError(f"Training file SHA256 mismatch: {path.name}")
        observed[path.name] = digest
    if len(observed) != 3:
        raise ValueError("Exactly three training files must be verified")
    return observed


def git_provenance(root: Path) -> dict:
    """Return source revision and dirty status without modifying Git state."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "repository_dirty": True}
    return {"git_commit": commit, "repository_dirty": dirty}
