"""Reproduce the FAR-Trans audit. Run from the project folder (see README)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from audit_core import evaluate_split, prepare
from build_reports import (
    eda_report, example_report, leakage_report, plot_figures,
    schema_report, source_report, split_report,
)
from test_audit import run_checks
from verify_outputs import verify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RAW = DATA / "FAR-Trans"
TABLE_NAMES = [
    "asset_information", "close_prices", "customer_information",
    "limit_prices", "markets", "transactions",
]
DATE_FIELDS = {
    "timestamp", "minDate", "maxDate", "lastQuestionnaireDate",
}
MISSING_MARKERS = {"", "Not_Available", "Not Available", "N/A", "NA", "null", "NULL"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_archive(source):
    DATA.mkdir(exist_ok=True)
    destination = DATA / "FAR-Trans.zip"
    source = Path(source).expanduser().resolve()
    if source != destination.resolve():
        if destination.exists() and sha256(source) != sha256(destination):
            raise ValueError("Existing data/FAR-Trans.zip differs; refusing to overwrite.")
        if not destination.exists():
            shutil.copyfile(source, destination)
    with zipfile.ZipFile(destination) as archive:
        for entry in archive.infolist():
            target = (DATA / entry.filename).resolve()
            if not target.is_relative_to(DATA.resolve()):
                raise ValueError(f"Unsafe archive entry: {entry.filename}")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            payload = archive.read(entry)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite a different file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(payload)
    return destination


def load_tables():
    raw = {
        name: pd.read_csv(RAW / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in TABLE_NAMES
    }
    parsed = {}
    for name, frame in raw.items():
        typed = frame.copy()
        for col in frame:
            if col in DATE_FIELDS:
                typed[col] = pd.to_datetime(frame[col].replace("", pd.NA), errors="coerce")
            elif col in {"transactionID", "totalValue", "units", "closePrice",
                         "priceMinDate", "priceMaxDate", "profitability"}:
                typed[col] = pd.to_numeric(frame[col], errors="raise")
        parsed[name] = typed
    return raw, parsed


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DATA / "FAR-Trans.zip"))
    parser.add_argument("--inspect", action="store_true", help="Local diagnostic only")
    args = parser.parse_args()
    run_checks()
    archive = prepare_archive(args.archive)
    raw, tables = load_tables()
    if args.inspect:
        print("SHA256", sha256(archive))
        for name, frame in tables.items():
            print(name, frame.shape, list(frame.columns))
            # Do not print customer identifiers.
            for col in frame:
                if col in {"customerID", "transactionID"}:
                    print(col, "unique", frame[col].nunique())
                else:
                    print(col, frame[col].value_counts(dropna=False).head(8).to_dict())
        return
    REPORTS.mkdir(exist_ok=True)
    stats = prepare(tables)
    split_rows, details = evaluate_split(tables, stats)
    sensitivity, _ = evaluate_split(tables, stats, include_reconstructed=False)
    archive_hash = sha256(archive)
    source_report(ROOT, archive_hash, tables, stats, sha256)
    comparisons = schema_report(ROOT, archive_hash, raw, tables, stats)
    leakage_report(ROOT, archive_hash, raw, tables, stats)
    split_report(ROOT, archive_hash, tables, stats, split_rows, sensitivity, details)
    eda_report(ROOT, archive_hash, tables, stats, split_rows)
    example_report(ROOT, archive_hash, raw, tables, stats, details)
    plot_figures(ROOT, stats)
    summary = {
        "archive_sha256": archive_hash,
        "environment": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        },
        "transactions": len(stats["tx"]),
        "customers": int(stats["tx"].customerID.nunique()),
        "assets_metadata": int(tables["asset_information"].ISIN.nunique()),
        "assets_prices": int(tables["close_prices"].ISIN.nunique()),
        "assets_transacted": int(stats["tx"].ISIN.nunique()),
        "buys": len(stats["buys"]), "reconstructed_marker": len(stats["negative"]),
        "repeat_all": stats["repeat"], "repeat_nonnegative": stats["repeat_real"],
        "top10_buy_share": float(stats["pop"].head(10).sum() / len(stats["buys"])),
        "cut": str(stats["cut"].date()), "end": str(stats["end"].date()),
        "eligible_assets": len(stats["eligible"]), "split": split_rows,
        "sensitivity_no_negative_history": sensitivity,
        "paper_differences": [
            {"measure": name, "paper": int(expected), "archive": int(actual)}
            for name, expected, actual in comparisons if expected != actual
        ],
    }
    (REPORTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verify(ROOT, summary)
    # Run manifest contains no raw customer IDs, paths to unrelated work, or timing noise.
    manifest = {
        "archive_sha256": archive_hash,
        "dependencies": summary["environment"],
        "source_code_sha256": {p.name: sha256(p) for p in sorted((ROOT / "scripts").glob("*.py"))},
        "generated_files_sha256": {
            p.relative_to(REPORTS).as_posix(): sha256(p)
            for p in sorted(REPORTS.rglob("*"))
            if p.is_file() and p.suffix in {".md", ".png", ".json"} and p.name != "manifest.json"
        },
    }
    (REPORTS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
