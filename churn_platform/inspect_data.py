"""Inspect and validate the raw dataset.

Usage (from the project root):
    python -m churn_platform.inspect_data

Prints a human-readable report and writes reports/data_validation.json.
Exit code is 1 if any ERROR-level issue is found, so the command can be
used as a gate before training.
"""

import json
import sys
from dataclasses import asdict

from churn_platform.config import RAW_DATA_FILE, REPORTS_DIR
from churn_platform.download import sha256_of
from churn_platform.validation import load_raw, summarize, validate


def main() -> int:
    try:
        df = load_raw(RAW_DATA_FILE)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    summary = summarize(df)
    issues = validate(df)
    errors = [i for i in issues if i.severity == "ERROR"]

    print(f"File:       {RAW_DATA_FILE.name}")
    print(f"SHA-256:    {sha256_of(RAW_DATA_FILE)}")
    print(f"Shape:      {summary['rows']} rows x {summary['columns']} columns")
    print(f"Target:     {summary.get('target_counts')}  churn rate = {summary.get('churn_rate')}")
    print("\nNumeric columns:")
    for col, stats in summary["numeric"].items():
        print(f"  {col:15} {stats}")
    print("\nCategorical columns:")
    for col, counts in summary["categorical_counts"].items():
        print(f"  {col:17} {counts}")

    print(f"\nValidation: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
    for issue in issues:
        print(f"  [{issue.severity}] {issue.check}: {issue.message} "
              f"(rows={issue.row_count}, e.g. {issue.examples})")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "data_validation.json"
    report = {
        "file": RAW_DATA_FILE.name,
        "sha256": sha256_of(RAW_DATA_FILE),
        "summary": summary,
        "issues": [asdict(i) for i in issues],
        "passed": not errors,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print("RESULT:", "PASSED" if not errors else "FAILED")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
