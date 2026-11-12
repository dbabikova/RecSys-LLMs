"""One real FAR-Trans record (anonymised) + field status/timestamps + leakage audit -> reports/data_example.md

The record is chosen deterministically: the first real (positive transactionID) Buy in the provisional test window of a
Stock with a sector, by a customer whose profile has >1 snapshot, ordered by (timestamp, transactionID).
Customer and transaction identifiers are masked; every other value is verbatim from the CSV line.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CUT, DATA, TEST_END, load, md_table, write  # noqa: E402


def mask(s: str, keep: int = 4) -> str:
    return s[:keep] + "…" + "*" * 4


def raw_line(fname: str, pred) -> tuple[list[str], list[str]]:
    with open(DATA / fname, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if pred(dict(zip(header, row))):
                return header, row
    raise LookupError(fname)


def main():
    tr, cu, a, cp, lp = (load(n) for n in
                         ["transactions", "customer_information", "asset_information", "close_prices", "limit_prices"])
    multi = cu.groupby("customerID").size().loc[lambda s: s > 1].index
    stocks = a[(a.assetCategory == "Stock") & a.sector.notna()].ISIN
    cand = tr[(tr.transactionID > 0) & (tr.transactionType == "Buy") & (tr.timestamp >= CUT)
              & tr.ISIN.isin(stocks) & tr.customerID.isin(multi)].sort_values(["timestamp", "transactionID"])
    rec = cand.iloc[0]
    t0 = rec.timestamp

    h, row = raw_line("transactions.csv", lambda d: d["transactionID"] == str(rec.transactionID)
                      and d["customerID"] == rec.customerID)
    row_m = [mask(v) if k in ("customerID", "transactionID") else v for k, v in zip(h, row)]

    ah, arow = raw_line("asset_information.csv", lambda d: d["ISIN"] == rec.ISIN)
    hist = cu[cu.customerID == rec.customerID].sort_values("timestamp")
    asof = hist[hist.timestamp <= t0].tail(1)
    later = hist[hist.timestamp > t0]
    px = cp[(cp.ISIN == rec.ISIN) & (cp.timestamp <= t0)].tail(1)

    out = ["# One real FAR-Trans record (anonymised)\n",
           "Masking: `customerID` and `transactionID` are truncated (`XXXX…****`); all other values are verbatim "
           "from the CSV line. ISIN identifies a public security, not a person.\n",
           "## Transaction (label source)\n", "```",
           "source: FAR-Trans, researchdata.gla.ac.uk/1658, transactions.csv",
           f"record id: transactionID {row_m[h.index('transactionID')]}",
           "fields: " + ", ".join(h),
           "values: " + ", ".join(row_m),
           f"window: {t0:%Y-%m-%d}; provisional TEST window ({CUT:%Y-%m-%d} … {TEST_END:%Y-%m-%d}), "
           "i.e. a label, not an input (official split pending)",
           "```\n",
           "## Asset row (item side, verbatim)\n", "```",
           "source: asset_information.csv",
           "fields: " + ", ".join(ah), "values: " + ", ".join(arow), "```\n",
           "## Customer profile history of this customer\n",
           md_table(hist.assign(customerID=hist.customerID.map(mask),
                                timestamp=hist.timestamp.dt.date,
                                lastQuestionnaireDate=hist.lastQuestionnaireDate.dt.date)),
           f"\n\nPoint-in-time profile for a prediction at {t0:%Y-%m-%d}: snapshot dated "
           f"{asof.timestamp.iloc[0]:%Y-%m-%d}; {len(later)} later snapshot(s) must NOT be used.\n" if len(asof) else "\n\nNo profile snapshot exists before the transaction.",
           ]

    prof = asof.iloc[0] if len(asof) else hist.iloc[0]
    status = lambda v: "predicted (by bank's algorithm)" if str(v).startswith("Predicted") else (
        "missing" if v == "Not_Available" else "observed")
    fields = [
        ("transactions.customerID", "id", "—", "key"),
        ("transactions.ISIN", rec.ISIN, f"{t0:%Y-%m-%d}", "observed (target item)"),
        ("transactions.transactionType", rec.transactionType, f"{t0:%Y-%m-%d}", "observed; Buy = positive label"),
        ("transactions.timestamp", f"{t0:%Y-%m-%d}", "—", "observed (day granularity)"),
        ("transactions.totalValue / units", f"{rec.totalValue} / {rec.units}", f"{t0:%Y-%m-%d}",
         "derived: paper recomputes value = units × close price; label-time info"),
        ("transactions.channel", rec.channel, f"{t0:%Y-%m-%d}", "observed; label-time info"),
        ("customer.customerType", prof.customerType, f"{prof.timestamp:%Y-%m-%d}", "observed"),
        ("customer.riskLevel", prof.riskLevel, f"{prof.timestamp:%Y-%m-%d}", status(prof.riskLevel)),
        ("customer.investmentCapacity", prof.investmentCapacity, f"{prof.timestamp:%Y-%m-%d}",
         status(prof.investmentCapacity)),
        ("customer.lastQuestionnaireDate", f"{prof.lastQuestionnaireDate:%Y-%m-%d}", f"{prof.timestamp:%Y-%m-%d}",
         "observed; 2000-01-01 is a placeholder = missing"),
        ("asset.assetCategory / sector / industry", f"{arow[ah.index('assetCategory')]} / {arow[ah.index('sector')]} / "
         f"{arow[ah.index('industry')]}", arow[ah.index("timestamp")], "observed static metadata"),
        ("close_prices.closePrice (last ≤ t)", f"{px.closePrice.iloc[0]}", f"{px.timestamp.iloc[0]:%Y-%m-%d}",
         "observed"),
        ("limit_prices.profitability", f"{lp.loc[lp.ISIN == rec.ISIN, 'profitability'].iloc[0]:.4f}",
         "computed over 2018…2022", "LEAKAGE: uses maxDate after t"),
    ]
    out += ["## Field status and timestamps (for a prediction at the record date)\n",
            md_table(pd.DataFrame(fields, columns=["field", "value", "timestamp", "status"])), ""]

    # dataset-wide leakage audit
    first_tx = tr.groupby("customerID").timestamp.min()
    first_prof = cu.groupby("customerID").timestamp.min()
    neg = tr.transactionID < 0
    audit = [
        ("customer_information has several dated snapshots",
         f"{(cu.groupby('customerID').size() > 1).sum():,} customers with >1 row",
         "join as-of prediction time (latest timestamp ≤ t)"),
        ("first profile snapshot is later than first transaction",
         f"{(first_prof > first_tx).mean()*100:.1f}% of customers",
         "for early t these users have no legal profile → treat as missing"),
        ("lastQuestionnaireDate > snapshot timestamp", f"{(cu.lastQuestionnaireDate > cu.timestamp).mean()*100:.1f}% of rows",
         "questionnaire date is not point-in-time; use only if ≤ t, else drop"),
        ("lastQuestionnaireDate = 2000-01-01 placeholder", f"{(cu.lastQuestionnaireDate == '2000-01-01').sum():,} rows",
         "treat as missing"),
        ("riskLevel predicted by the bank", f"{cu.riskLevel.str.startswith('Predicted').mean()*100:.1f}% of rows",
         "keep a predicted/observed flag; prediction date unknown"),
        ("investmentCapacity predicted by the bank",
         f"{cu.investmentCapacity.str.startswith('Predicted').mean()*100:.1f}% of rows", "same"),
        ("limit_prices.profitability / priceMaxDate / maxDate", "whole-period values",
         "never use as features (future)"),
        ("close_prices after t", "daily series to 2022-11-29", "features only from prices ≤ t"),
        ("asset_information.timestamp", f"{a.timestamp.min():%Y-%m-%d} … {a.timestamp.max():%Y-%m-%d}",
         "low risk (static metadata), but asset 'exists' only from its first price date"),
        ("reconstructed buys (negative transactionID)",
         f"{neg.sum():,} rows, {(tr[neg].timestamp == '2018-01-02').sum():,} dated 2018-01-02",
         "synthetic initial holdings, not real purchase events: keep in history, never as test labels"),
        ("same transactionID on two customers (joint accounts)",
         f"{tr.transactionID.duplicated(keep=False).sum():,} rows", "a purchase may be counted for 2 users"),
        ("transaction totalValue / units / channel", "known at transaction time",
         "label-time fields of the target event: not inputs"),
    ]
    out += ["\n## Dataset-wide leakage / timestamp audit\n",
            md_table(pd.DataFrame(audit, columns=["issue", "extent", "rule for our pipeline"])), ""]
    write("data_example.md", "make_data_example.py", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
