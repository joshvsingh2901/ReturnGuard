"""One-time historical ASOS manifest generation; no model fitting or scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.loaders import RAW_DIR
from ml.lifecycle.hashing import write_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--include-historical-test-schema", action="store_true", help="Required because this one-time provenance command inspects all six raw pickle schemas")
    args = parser.parse_args()
    if not args.include_historical_test_schema:
        parser.error("Manifest generation requires --include-historical-test-schema; normal reproduction never reads test files")
    destination = Path(__file__).parent.parent / "artifacts" / "manifests" / "asos-graphreturns-osf-c793h-v1.json"
    write_dataset_manifest(destination, args.raw_dir)
    print(f"Wrote historical ASOS manifest: {destination}")


if __name__ == "__main__":
    main()
