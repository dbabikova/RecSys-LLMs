"""Produce the Session 1 data example and leakage checks.

Selects one deterministic Buy transaction (random seed 42) that has both a
point-in-time customer snapshot and an asset record, prints the three records in
the required format, builds a per-field availability/leakage table, runs
dataset-level leakage counters, and writes ``reports/session1/data_example.md``.

All numbers in the report are computed here; nothing is hard-coded except the
field-level semantics (which fields are observed vs. predicted vs. the target).

Usage::

    python scripts/make_data_example.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FAR-Trans"
REPORT = ROOT / "reports" / "session1"

SENTINEL = "2000-01-01"
SEED = 42
OUT_NAME = "data_example.md"


def mask_customer(cid: str) -> str:
    return cid[:6] + "…"


def field_status(value: str) -> str:
    if value == "":
        return "missing"
    if value.startswith("Predicted_"):
        return "predicted-estimated"
    return "observed"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(RAW / "transactions.csv", dtype=str, keep_default_na=False)
    cust = pd.read_csv(RAW / "customer_information.csv", dtype=str, keep_default_na=False)
    asset = pd.read_csv(RAW / "asset_information.csv", dtype=str, keep_default_na=False)
    return tx, cust, asset


def select_transaction(tx, cust, asset) -> pd.Series:
    buy = tx[tx["transactionType"] == "Buy"].copy()

    # condition 1: a non-sentinel customer snapshot dated <= the transaction
    cust_valid = cust[cust["timestamp"] != SENTINEL][["customerID", "timestamp"]]
    m = buy.merge(cust_valid, on="customerID", suffixes=("_tx", "_cust"))
    m = m[m["timestamp_cust"] <= m["timestamp_tx"]]
    has_snapshot = set(m["transactionID"])

    # condition 2: an asset version dated <= the transaction
    a = buy.merge(asset[["ISIN", "timestamp"]], on="ISIN", suffixes=("_tx", "_asset"))
    a = a[a["timestamp_asset"] <= a["timestamp_tx"]]
    has_asset = set(a["transactionID"])

    cand = buy[buy["transactionID"].isin(has_snapshot & has_asset)]
    cand_ids = sorted(cand["transactionID"].tolist(), key=int)

    rng = random.Random(SEED)
    tid = rng.choice(cand_ids)
    row = buy[buy["transactionID"] == tid].iloc[0]
    return row


def point_in_time(rows: pd.DataFrame, col: str, key: str, cutoff: str) -> pd.Series:
    sub = rows[
        (rows[col] == key) & (rows["timestamp"] != SENTINEL) & (rows["timestamp"] <= cutoff)
    ]
    sub = sub.sort_values("timestamp", kind="stable")
    return sub.iloc[-1]


def latest_snapshot(rows: pd.DataFrame, col: str, key: str, cutoff: str) -> pd.Series:
    return point_in_time(rows, col, key, cutoff)


def fmt_record(title: str, source: str, record_id: str, fields: list[str], values: list[str], window: str) -> list[str]:
    return [
        f"### {title}",
        "",
        f"source: FAR-Trans (researchdata.gla.ac.uk/1658), {source}",
        f"record id: {record_id}",
        f"fields: {', '.join(fields)}",
        f"values: {', '.join(values)}",
        f"window: {window}; split pending; provisional",
        "",
    ]


TX_FIELD_NAMES = ["customerID", "ISIN", "transactionID", "transactionType", "timestamp",
                  "totalValue", "units", "channel", "marketID"]


def tx_field_table(record: pd.Series) -> list[str]:
    out = [
        "| Field | Status | Timestamp | As history (date < cut) | As target (date >= cut) |",
        "|-------|--------|-----------|--------------------------|-------------------------|",
    ]
    for field in TX_FIELD_NAMES:
        out.append(
            f"| `{field}` | {field_status(str(record[field]))} "
            f"| `transactions.timestamp` (transaction date) "
            f"| available | forbidden (model target — excluded from inputs) |"
        )
    return out

CUST_FIELDS = [
    ("customerID", "Yes", "None (identifies the user)"),
    ("customerType", "Yes", "None (past information)"),
    ("riskLevel", "Yes", "None (past information)"),
    ("investmentCapacity", "Yes", "None (past information)"),
    ("lastQuestionnaireDate", "Yes", "None (past information)"),
    ("timestamp", "Yes", "None (past information)"),
]

ASSET_FIELDS = [
    ("ISIN", "Yes", "None (public identifier)"),
    ("assetName", "Yes", "None (public metadata)"),
    ("assetShortName", "Yes", "None (public metadata)"),
    ("assetCategory", "Yes", "None (public metadata)"),
    ("assetSubCategory", "Yes", "None (public metadata)"),
    ("marketID", "Yes", "None (public metadata)"),
    ("sector", "Yes", "None (public metadata)"),
    ("industry", "Yes", "None (public metadata)"),
    ("timestamp", "Yes", "None (record version date)"),
]

TS_SOURCE = {
    "tx": "`transactions.timestamp` (the transaction's own date)",
    "cust": "`customer_information.timestamp` (snapshot update date)",
    "asset": "`asset_information.timestamp` (asset record version date)",
}


def field_table(kind: str, record: pd.Series) -> list[str]:
    meta = CUST_FIELDS if kind == "cust" else ASSET_FIELDS

    out = [
        "| Field | Status | Timestamp source | Available at prediction time | Leakage risk |",
        "|-------|--------|------------------|------------------------------|--------------|",
    ]
    for field, available, leakage in meta:
        value = str(record[field])
        out.append(
            f"| `{field}` | {field_status(value)} | {TS_SOURCE[kind]} "
            f"| {available} | {leakage} |"
        )
    return out


def pct(numer: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{100.0 * numer / denom:.2f}%"


def main() -> int:
    tx, cust, asset = load()
    row = select_transaction(tx, cust, asset)

    tid = row["transactionID"]
    tx_date = row["timestamp"]
    customer_id = row["customerID"]
    isin = row["ISIN"]

    snap = latest_snapshot(cust, "customerID", customer_id, tx_date)
    arow = point_in_time(asset, "ISIN", isin, tx_date)

    masked = mask_customer(customer_id)

    tx_fields = ["customerID", "ISIN", "transactionID", "transactionType", "timestamp",
                 "totalValue", "units", "channel", "marketID"]
    tx_values = [masked, row["ISIN"], row["transactionID"], row["transactionType"],
                 row["timestamp"], row["totalValue"], row["units"], row["channel"], row["marketID"]]

    cust_fields = ["customerID", "customerType", "riskLevel", "investmentCapacity",
                   "lastQuestionnaireDate", "timestamp"]
    cust_values = [masked, snap["customerType"], snap["riskLevel"], snap["investmentCapacity"],
                   snap["lastQuestionnaireDate"], snap["timestamp"]]

    asset_fields = ["ISIN", "assetName", "assetShortName", "assetCategory", "assetSubCategory",
                    "marketID", "sector", "industry", "timestamp"]
    asset_values = [arow["ISIN"], arow["assetName"], arow["assetShortName"], arow["assetCategory"],
                    arow["assetSubCategory"], arow["marketID"], arow["sector"], arow["industry"],
                    arow["timestamp"]]

    # ---- dataset-level leakage checks ----
    snap_counts = cust.groupby("customerID").size()
    n_multi_snap = int((snap_counts > 1).sum())

    first_tx = tx.groupby("customerID")["timestamp"].min().rename("first_tx")
    merged = cust.merge(first_tx, on="customerID")
    later = merged[merged["timestamp"] > merged["first_tx"]]
    n_later_snap = int(len(later))
    n_later_cust = int(later["customerID"].nunique())

    latest_snap_df = cust.sort_values("timestamp").groupby("customerID").tail(1)
    inactive_ids = set(latest_snap_df[latest_snap_df["customerType"] == "Inactive"]["customerID"])
    inactive_tx = tx[tx["customerID"].isin(inactive_ids)]
    last_tx = inactive_tx.groupby("customerID")["timestamp"].max()
    n_inactive = int(len(last_tx))
    inactive_min = last_tx.min() if len(last_tx) else "—"
    inactive_max = last_tx.max() if len(last_tx) else "—"

    def share(series: pd.Series):
        total = len(series)
        pred = int(series.str.startswith("Predicted_").sum())
        na = int((series == "Not_Available").sum())
        return total, pred, na

    rl_total, rl_pred, rl_na = share(cust["riskLevel"])
    ic_total, ic_pred, ic_na = share(cust["investmentCapacity"])

    asset_counts = asset.groupby("ISIN").size()
    n_multi_isin = int((asset_counts > 1).sum())

    # lastQuestionnaireDate > timestamp (both non-sentinel)
    comparable = cust[(cust["timestamp"] != SENTINEL) & (cust["lastQuestionnaireDate"] != SENTINEL)]
    bad = comparable[comparable["lastQuestionnaireDate"] > comparable["timestamp"]]
    n_lqd_gt_ts = int(len(bad))
    n_lqd_gt_ts_cust = int(bad["customerID"].nunique())

    # customers with no snapshot dated <= their first transaction
    first_tx_series = tx.groupby("customerID")["timestamp"].min().rename("first_tx")
    earliest_snap = (
        cust[cust["timestamp"] != SENTINEL]
        .groupby("customerID")["timestamp"].min().rename("earliest_snap")
    )
    snap_vs_tx = pd.concat([first_tx_series, earliest_snap], axis=1)
    no_snap_before_first = snap_vs_tx["earliest_snap"].isna() | (snap_vs_tx["earliest_snap"] > snap_vs_tx["first_tx"])
    n_no_snap_before_first = int(no_snap_before_first.sum())

    # ---- assemble the markdown ----
    out: list[str] = []
    out.append("<!-- auto-generated by scripts/make_data_example.py — do not edit manually -->")
    out.append("# Data example and leakage check (Session 1)")
    out.append("")
    out.append(f"**Selected deterministically** with `random.Random({SEED})` from the Buy")
    out.append("transactions that have (a) a non-sentinel point-in-time customer snapshot")
    out.append("(`customer_information.timestamp <= transaction date`) and (b) an asset version")
    out.append("(`asset_information.timestamp <= transaction date`).")
    out.append("")
    out.append("**Anonymization:** `customerID` is masked to its first 6 characters + `…`;")
    out.append("`transactionID` is kept; `ISIN` and asset names are public and kept as-is.")
    out.append("")
    out.append("**Prediction time** = a **provisional cut** (the official time cut is not yet")
    out.append("fixed); it is **not** the date of this example transaction. A transaction dated")
    out.append("before the cut is history (usable as input); a transaction dated on or after the")
    out.append("cut is a target and must be excluded from model inputs.")
    out.append("")

    out.append("## Records")
    out.append("")

    out.extend(fmt_record(
        "1. Transaction", "transactions.csv", tid, tx_fields, tx_values, tx_date
    ))
    out.extend(fmt_record(
        "2. Point-in-time customer snapshot", "customer_information.csv",
        f"{masked}@{snap['timestamp']}", cust_fields, cust_values, snap["timestamp"]
    ))
    out.extend(fmt_record(
        "3. Asset record (point-in-time)", "asset_information.csv",
        f"{arow['ISIN']}@{arow['timestamp']}", asset_fields, asset_values, arow["timestamp"]
    ))

    out.append("## Per-field availability and leakage")
    out.append("")
    out.append("### 1. Transaction")
    out.append("")
    out.extend(tx_field_table(row))
    out.append("")
    out.append("### 2. Point-in-time customer snapshot")
    out.append("")
    out.extend(field_table("cust", snap))
    out.append("")
    out.append("### 3. Asset record")
    out.append("")
    out.extend(field_table("asset", arow))
    out.append("")

    out.append("## Dataset-level leakage checks (computed)")
    out.append("")
    out.append(f"- Customers with **multiple snapshots** in `customer_information.csv`: **{n_multi_snap}**.")
    out.append(f"- Snapshots with `timestamp` **later than the customer's first transaction**:")
    out.append(f"  **{n_later_snap}** rows, affecting **{n_later_cust}** customers (future-profile risk).")
    out.append(f"- Customers with latest `customerType = Inactive`: **{n_inactive}**; their last")
    out.append(f"  transaction dates range from `{inactive_min}` to `{inactive_max}`.")
    out.append(f"- `riskLevel` — `Predicted_*`: **{rl_pred}** of {rl_total} ({pct(rl_pred, rl_total)});")
    out.append(f"  `Not_Available`: **{rl_na}** of {rl_total} ({pct(rl_na, rl_total)}).")
    out.append(f"- `investmentCapacity` — `Predicted_*`: **{ic_pred}** of {ic_total} ({pct(ic_pred, ic_total)});")
    out.append(f"  `Not_Available`: **{ic_na}** of {ic_total} ({pct(ic_na, ic_total)}).")
    out.append(f"- ISINs with **multiple versions** in `asset_information.csv`: **{n_multi_isin}**.")
    out.append(f"- Snapshots where `lastQuestionnaireDate > timestamp` (both non-sentinel):")
    out.append(f"  **{n_lqd_gt_ts}** of {len(comparable)} comparable snapshots")
    out.append(f"  ({pct(n_lqd_gt_ts, len(comparable))}), affecting **{n_lqd_gt_ts_cust}** customers.")
    out.append(f"  **Effective profile availability date = max(`timestamp`, `lastQuestionnaireDate`).**")
    out.append(f"- Customers with **no snapshot** dated <= their first transaction:")
    out.append(f"  **{n_no_snap_before_first}**.")
    out.append("")
    out.append("**Forbidden feature:** `limit_prices.profitability` is computed over the asset's")
    out.append("whole price series (`minDate` → `maxDate`, with `maxDate` up to 2022-11) and must")
    out.append("**not** be used as a model feature — using it leaks future price information.")
    out.append("")

    report = REPORT / OUT_NAME
    REPORT.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(out), encoding="utf-8")

    print(f"selected transactionID: {tid} (customer {masked}, ISIN {isin}, date {tx_date})")
    print(f"wrote: {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
