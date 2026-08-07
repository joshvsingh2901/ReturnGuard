"""Explicitly promote a passing governed A3 rebuild to model-of-record."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.lifecycle.tracking import promote_registered_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--allow-recorded-dirty-git", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent.parent
    promote_registered_model(
        root=root,
        run_id=args.run_id,
        model_version=args.model_version,
        allow_dirty=args.allow_recorded_dirty_git,
    )
    print(f"returnguard-a3 v{args.model_version} is now model-of-record")


if __name__ == "__main__":
    main()
