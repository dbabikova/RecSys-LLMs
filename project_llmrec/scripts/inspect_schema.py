"""Compute every schema-level number used in the Session 1 report.

Output: reports/session1/schema_stats.md (auto-generated). No raw rows are printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmrec.data import (  # noqa: E402
    CSV_FILES, DATA_DIR, DATE_COLUMNS, REPORTS_DIR, SENTINEL_DATE, read_raw,
)

MAX_LISTED = 30


def infer_type(s: pd.Series) -> str:
    v = s[s != ""]
    if v.empty:
        return "empty"
    if pd.to_numeric(v, errors="coerce").notna().all():
        return "int" if v.str.fullmatch(r"-?\d+").all() else "float"
    if pd.to_datetime(v, format="%Y-%m-%d", errors="coerce").notna().all():
        return "date"
    return "str"


def file_section(name: str) -> list[str]:
    df = read_raw(name)
    out = [f"## `{name}`", "", f"Rows (excluding header): **{len(df)}**; columns: **{df.shape[1]}**.", ""]
    out += ["| Column | Type | Empty | Unique | Date min | Date max | Sentinel 2000-01-01 |",
            "|---|---|---:|---:|---|---|---:|"]
    listed = []
    for c in df.columns:
        s = df[c]
        t = infer_type(s)
        dmin = dmax = sent = ""
        if c in DATE_COLUMNS[name]:
            d = pd.to_datetime(s.replace("", None), format="%Y-%m-%d", errors="coerce")
            sent = int((d == SENTINEL_DATE).sum())
            real = d[d != SENTINEL_DATE].dropna()
            dmin, dmax = real.min().date(), real.max().date()
        out.append(f"| `{c}` | {t} | {int((s == '').sum())} | {s.nunique()} | {dmin} | {dmax} | {sent} |")
        if t == "str" and s.nunique() <= MAX_LISTED:
            listed.append((c, s.value_counts()))
    for c, vc in listed:
        out += ["", f"Values of `{c}` (count):", ""]
        out.append(", ".join(f"`{k if k else '(empty)'}` {v}" for k, v in vc.items()))
    out.append("")
    return out


def cross_file_checks() -> list[str]:
    tx = read_raw("transactions.csv")
    cu = read_raw("customer_information.csv")
    a = read_raw("asset_information.csv")
    cp = read_raw("close_prices.csv")
    lp = read_raw("limit_prices.csv")

    tx_isin, a_isin, cp_isin = set(tx.ISIN), set(a.ISIN), set(cp.ISIN)
    cat_per_isin = a.groupby("ISIN")["assetCategory"].nunique()
    sub = a[a.ISIN.isin(a.ISIN[a.ISIN.duplicated()])]
    changed_cols = [c for c in a.columns if c not in ("ISIN", "timestamp")
                    and (sub.groupby("ISIN")[c].nunique() > 1).any()]
    buys = tx[tx.transactionType == "Buy"]
    cat = a.drop_duplicates("ISIN", keep="last").set_index("ISIN")["assetCategory"]

    dup = tx[tx.transactionID.duplicated(keep=False)]
    non_cust = [c for c in tx.columns if c != "customerID"]
    dup_same = int((dup.groupby("transactionID")[non_cust].nunique() > 1).any(axis=1).eq(False).sum())
    dup_sizes = dup.groupby("transactionID").customerID.nunique().value_counts().sort_index()
    shared_rows = ("Rows whose transactionID is shared with another row", len(dup))
    shared_ids = ("transactionID values shared by >1 row", dup.transactionID.nunique())
    shared_same = ("Shared transactionID values whose rows agree on every field except customerID", dup_same)
    shared_buys = ("Buy rows with a shared transactionID", int((dup.transactionType == "Buy").sum()))
    changed_cat = cat_per_isin[cat_per_isin > 1].index
    cat_buys = ("Buy rows on ISINs whose assetCategory differs across rows", int(buys.ISIN.isin(changed_cat).sum()))

    rows = [
        ("Unique customerID in customer_information", cu.customerID.nunique()),
        ("Unique customerID in transactions", tx.customerID.nunique()),
        ("Customers in transactions without any profile row", len(set(tx.customerID) - set(cu.customerID))),
        ("Unique ISIN in asset_information", len(a_isin)),
        ("Unique ISIN in close_prices", len(cp_isin)),
        ("Unique ISIN in limit_prices", lp.ISIN.nunique()),
        ("Unique ISIN in transactions", len(tx_isin)),
        ("ISIN in transactions but not in asset_information", len(tx_isin - a_isin)),
        ("ISIN in transactions but not in close_prices", len(tx_isin - cp_isin)),
        ("ISIN in asset_information but not in close_prices", len(a_isin - cp_isin)),
        ("ISIN with >1 row in asset_information", int((a.ISIN.value_counts() > 1).sum())),
        ("ISIN whose assetCategory differs across rows", int((cat_per_isin > 1).sum())),
        cat_buys,
        shared_ids, shared_rows, shared_same, shared_buys,
        ("Fully duplicated transaction rows", int(tx.duplicated().sum())),
        ("Duplicate (ISIN, timestamp) rows in close_prices", int(cp.duplicated(["ISIN", "timestamp"]).sum())),
        ("Customers with >1 row in customer_information", int((cu.customerID.value_counts() > 1).sum())),
    ]
    out = ["## Cross-file checks", "", "| Check | Value |", "|---|---:|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    out += ["", "Number of distinct customerID per shared transactionID (value: count of IDs): "
            + ", ".join(f"{k}: {v}" for k, v in dup_sizes.items()) + "."]
    out += ["", f"Columns that change between versions of the same ISIN in asset_information: "
            f"{', '.join(f'`{c}`' for c in changed_cols) or 'none'}.", ""]
    out += ["Buy transactions by assetCategory (category from asset_information):", ""]
    out.append(", ".join(f"`{k}` {v}" for k, v in buys.ISIN.map(cat).fillna("(unknown)").value_counts().items()))
    out += ["", "Distinct ISIN bought at least once, by assetCategory:", ""]
    out.append(", ".join(f"`{k}` {v}" for k, v in
                         buys.drop_duplicates("ISIN").ISIN.map(cat).fillna("(unknown)").value_counts().items()))
    out.append("")
    return out


def main() -> None:
    lines = ["<!-- auto-generated by scripts/inspect_schema.py; do not edit by hand -->",
             "# FAR-Trans schema statistics", "",
             "Empty = empty string in the raw CSV. Date min/max exclude the 2000-01-01 sentinel. "
             f"Full value lists are shown for text columns with at most {MAX_LISTED} distinct values.", ""]
    for name in CSV_FILES:
        lines += file_section(name)
    q = (DATA_DIR / "questionnaires.csv").read_text(encoding="utf-8").splitlines()
    lines += ["## `questionnaires.csv`", "",
              f"Plain text (not CSV): {len(q)} lines, {sum(1 for l in q if l.strip())} non-empty. "
              "Questions and answer options of the MiFID questionnaire; customers' answers are not included.", ""]
    lines += cross_file_checks()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "schema_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print("ok: reports/session1/schema_stats.md")


if __name__ == "__main__":
    main()
