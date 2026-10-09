"""``python -m evaluation`` — run the behavioral evaluation without pytest.

Two modes:

* default: simulated (offline, deterministic) cases only;
* ``--live`` (or ``ATLAS_EVAL_LIVE=1``): also run the cases that need a real
  local model, network access or Windows.

The report is always printed to the console and written to a JSON file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.runner import DEFAULT_REPORT, live_enabled
from evaluation.test_evaluation import run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation")
    parser.add_argument(
        "--live",
        action="store_true",
        help="also run cases that need the real model / network / Windows",
    )
    parser.add_argument(
        "--report",
        default=os.environ.get("ATLAS_EVAL_REPORT", str(DEFAULT_REPORT)),
        help="where to write the JSON report",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="print only the JSON report path and summary counts",
    )
    args = parser.parse_args(argv)

    live = args.live or live_enabled()
    report = run_all(live=live, report_path=Path(args.report))
    if not args.json_only:
        print(report.summary_text())
    print(f"JSON report: {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
