"""Paths, constants and loaders for the FAR-Trans dataset.

All loaders read the raw CSVs as strings first and parse only the columns
they need, so that verbatim values can still be shown in reports.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = RAW_DIR / "FAR-Trans"
REPORTS_DIR = PROJECT_ROOT / "reports" / "session1"
FIG_DIR = REPORTS_DIR / "figures"

ZIP_URL = "https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip"
README_URL = "https://researchdata.gla.ac.uk/1658/2/README.txt"
# SHA-256 of FAR-Trans.zip as downloaded from the official repository.
ZIP_SHA256 = "d0d6cfe027732bab45fdcf9af80eff4fefddc789aedae9bdce0d3d6cfd05cfb6"

# Dates equal to this value are placeholders for "unknown" in customer_information.
SENTINEL_DATE = pd.Timestamp("2000-01-01")

CSV_FILES = [
    "customer_information.csv",
    "asset_information.csv",
    "markets.csv",
    "close_prices.csv",
    "limit_prices.csv",
    "transactions.csv",
]

DATE_COLUMNS = {
    "customer_information.csv": ["lastQuestionnaireDate", "timestamp"],
    "asset_information.csv": ["timestamp"],
    "markets.csv": [],
    "close_prices.csv": ["timestamp"],
    "limit_prices.csv": ["minDate", "maxDate"],
    "transactions.csv": ["timestamp"],
}


def read_raw(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read a CSV with every column as string (verbatim values, '' for empty)."""
    return pd.read_csv(data_dir / name, dtype=str, keep_default_na=False)


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c].replace("", None), format="%Y-%m-%d", errors="coerce")
    return out


def load_transactions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = parse_dates(read_raw("transactions.csv", data_dir), ["timestamp"])
    df["totalValue"] = pd.to_numeric(df["totalValue"])
    df["units"] = pd.to_numeric(df["units"])
    return df


def load_customers(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Customer snapshots with parsed dates; sentinel dates become NaT.

    Adds `available_from` = max(timestamp, lastQuestionnaireDate): the earliest
    date at which the whole snapshot (including the questionnaire-based fields)
    can be considered known.
    """
    df = parse_dates(read_raw("customer_information.csv", data_dir), DATE_COLUMNS["customer_information.csv"])
    for c in ["timestamp", "lastQuestionnaireDate"]:
        df[c] = df[c].mask(df[c] == SENTINEL_DATE)
    df["available_from"] = df[["timestamp", "lastQuestionnaireDate"]].max(axis=1)
    return df


def load_assets(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return parse_dates(read_raw("asset_information.csv", data_dir), ["timestamp"])


def asset_category_map(data_dir: Path = DATA_DIR) -> pd.Series:
    """ISIN -> assetCategory, taken from the latest asset record (category is static per ISIN)."""
    a = load_assets(data_dir).sort_values("timestamp")
    return a.drop_duplicates("ISIN", keep="last").set_index("ISIN")["assetCategory"]
