"""Inspect the raw FAR-Trans CSV files and print a schema summary.

For each CSV file under data/raw/FAR-Trans, report the row count (excluding
the header), the columns with pandas dtypes, and the min/max value of every
timestamp-like column. Output is a human-readable report used to author
reports/session1/schema.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "FAR-Trans"

DATE_COLUMNS = {"timestamp", "minDate", "maxDate", "lastQuestionnaireDate"}


def summarize_file(path: Path) -> None:
    name = path.name
    print("=" * 80)
    print(f"FILE: {name}")
    print("=" * 80)

    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  [could not parse as CSV: {exc}]")
        return

    print(f"  rows (excl. header): {len(df)}")
    print(f"  columns: {len(df.columns)}")
    print(f"  dtypes:")
    for col in df.columns:
        print(f"    - {col}: {df[col].dtype}")

    print("  timestamp-like column ranges:")
    for col in df.columns:
        if col in DATE_COLUMNS or "date" in col.lower() or "timestamp" in col.lower():
            values = pd.to_datetime(df[col], errors="coerce")
            non_null = values.dropna()
            if non_null.empty:
                print(f"    - {col}: (no parseable values)")
                continue
            print(
                f"    - {col}: min={non_null.min().date()} "
                f"max={non_null.max().date()} "
                f"(non-null: {non_null.size})"
            )

    print(f"  head (first 2 rows):")
    print(df.head(2).to_string(max_colwidth=40))


def main() -> int:
    if not RAW_DIR.exists():
        print(f"raw directory not found: {RAW_DIR}", file=sys.stderr)
        return 1

    files = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() == ".csv")
    for path in files:
        summarize_file(path)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
