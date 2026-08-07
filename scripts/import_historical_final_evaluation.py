"""Import committed final-evaluation evidence into MLflow without scoring data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.lifecycle.tracking import log_historical_final_evaluation

ROOT = Path(__file__).parent.parent


def _read_report(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text())


def main() -> None:
    run_id = log_historical_final_evaluation(
        root=ROOT,
        metrics=_read_report("final_test_metrics.json"),
        slices=_read_report("final_test_slices.json"),
        calibration=_read_report("final_test_calibration.json"),
    )
    print(f"Imported historical final-evaluation evidence into MLflow run {run_id}")


if __name__ == "__main__":
    main()
