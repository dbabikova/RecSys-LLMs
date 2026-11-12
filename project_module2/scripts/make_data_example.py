"""One actual FAR-Trans record (anonymised) + per-field status / timestamp / leakage check -> reports/data_example.md

Record choice (deterministic, no cherry-picking): the earliest real Buy (transactionID > 0) in the provisional
test window that is an evaluation label under variant A (user bought before the cut, asset new to the user,
asset in the eligible universe), restricted to customers with > 1 profile snapshot so that the as-of rule is
visible; ties broken by transactionID. Any asset category may be selected: the item scope is pending.

Prediction time t = CUT (the provisional split point), NOT the transaction date: the record is a label,
every input must be known at t.
Anonymisation: customerID (a pseudonymous hash in the public release) is truncated; transactionID is shown
verbatim (a bank-internal event number, not linked to a person). No identity is attached.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CUT, DATA, TEST_END, build_split, load, md_table, write  # noqa: E402


def mask(s: str) -> str:
    return s[:4] + "…(masked)"


def raw_line(fname, pred):
    with open(DATA / fname, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if pred(dict(zip(header, row))):
                return header, row
    raise LookupError(fname)


def main():
    tr, cu, cp = load("transactions"), load("customer_information"), load("close_prices")
    a = load("asset_information")
    split = build_split("A")
    rel = split["rel"]
    multi = set(cu.groupby("customerID").size().loc[lambda s: s > 1].index)
    lab = tr[(tr.transactionID > 0) & (tr.transactionType == "Buy") & (tr.timestamp >= CUT)
             & (tr.timestamp <= TEST_END) & tr.customerID.isin(multi)]
    lab = lab[[u in rel and i in rel[u] for u, i in zip(lab.customerID, lab.ISIN)]]
    rec = lab.sort_values(["timestamp", "transactionID"]).iloc[0]
    t = CUT

    h, row = raw_line("transactions.csv", lambda d: d["transactionID"] == str(rec.transactionID)
                      and d["customerID"] == rec.customerID)
    shown = [mask(v) if k == "customerID" else v for k, v in zip(h, row)]
    ah, arow = raw_line("asset_information.csv", lambda d: d["ISIN"] == rec.ISIN)
    ad = dict(zip(ah, arow))
    n_asset_rows = int((a.ISIN == rec.ISIN).sum())
    hist = cu[cu.customerID == rec.customerID].sort_values("timestamp")
    asof = hist[hist.timestamp <= t].tail(1)
    later = hist[hist.timestamp > t]
    px = cp[(cp.ISIN == rec.ISIN) & (cp.timestamp < t)].tail(1)  # close of day t is not known at t
    n_prior = len(split["tr_items"][rec.customerID])

    out = ["# One actual FAR-Trans record (anonymised)\n",
           "Only `customerID` is truncated (pseudonymous hash in the public release); every other value is "
           "verbatim from the CSV line. No real identity is attached. ISIN identifies a public security.\n",
           "```",
           "source: FAR-Trans public release, University of Glasgow (researchdata.gla.ac.uk/1658, CC BY 4.0), "
           f"file transactions.csv   record id: transactionID {rec.transactionID}",
           "fields: " + ", ".join(h),
           "values: " + ", ".join(shown),
           f"window: provisional TEST window {CUT:%Y-%m-%d} … {TEST_END:%Y-%m-%d} (label, not input); "
           f"prediction time t = {t:%Y-%m-%d}; split is an assumption, official split pending",
           "```\n",
           f"Why this row: earliest real Buy in the test window that is a variant-A label (the customer bought "
           f"{n_prior} distinct asset(s) before t, none of them this ISIN). Selection is deterministic "
           "(see script docstring); the asset category was not chosen. **Item scope is pending**: the "
           "eligible universe used here keeps stocks, bonds and mutual funds; this row being a "
           f"{ad['assetCategory']} does not imply a {ad['assetCategory']}-only scope.\n",
           "Joined asset row (asset_information.csv, verbatim):\n", "```",
           "fields: " + ", ".join(ah), "values: " + ", ".join(arow), "```",
           f"\n`asset_information` has {n_asset_rows} row(s) for this ISIN.\n",
           "Profile snapshots of this customer (customer_information.csv):\n",
           md_table(hist.assign(customerID=hist.customerID.map(mask),
                                lastQuestionnaireDate=hist.lastQuestionnaireDate.dt.date,
                                timestamp=hist.timestamp.dt.date)),
           (f"\n\nAs-of rule at t = {t:%Y-%m-%d}: use the snapshot dated {asof.timestamp.iloc[0]:%Y-%m-%d}; "
            f"{len(later)} later snapshot(s) are excluded.\n") if len(asof) else
           "\n\nNo snapshot ≤ t: profile fields are missing for this user at t.\n"]

    p = asof.iloc[0] if len(asof) else None

    def pstatus(v):
        if p is None:
            return "missing (no snapshot ≤ t)"
        if str(v).startswith("Predicted"):
            return "predicted (bank's algorithm; estimation date unknown)"
        return "missing" if v == "Not_Available" else "observed"

    def pv(f):
        return str(p[f]) if p is not None else "—"

    pts = f"{p.timestamp:%Y-%m-%d}" if p is not None else "—"
    q = p.lastQuestionnaireDate if p is not None else None
    snap_after = pd.Timestamp(ad["timestamp"]) > t
    static_ok = ("**check** (snapshot after t)" if snap_after else
                 "likely yes (slow-changing attribute); the row timestamp does not prove it — see audit")
    after = f"{rec.timestamp:%Y-%m-%d}"
    fields = [
        ("transactions.customerID", mask(rec.customerID), "—", "observed (pseudonymous key)", "yes", "join key only; never in a prompt"),
        ("transactions.ISIN", rec.ISIN, after, "observed", "**no** (event after t)", "label: positive item"),
        ("transactions.transactionID", str(rec.transactionID), after, "observed", "**no**", "record id only"),
        ("transactions.transactionType", rec.transactionType, after, "observed", "**no**", "label: Buy = relevant"),
        ("transactions.timestamp", after, "—", "observed (day granularity)", "**no**", "puts the row in the test window"),
        ("transactions.totalValue", str(rec.totalValue), after, "observed; recomputed by dataset authors (units × close price)", "**no**", "not used (label-time)"),
        ("transactions.units", str(rec.units), after, "observed", "**no**", "not used (label-time)"),
        ("transactions.channel", rec.channel, after, "observed", "**no**", "not used (label-time)"),
        ("transactions.marketID", rec.marketID, after, "observed", "**no**", "not used (label-time)"),
        ("purchase history before t", f"{n_prior} distinct assets", f"< {t:%Y-%m-%d}", "observed (+ reconstructed 2018-01-02 holdings)", "yes", "input"),
        ("customer.customerType", pv("customerType"), pts, pstatus(pv("customerType")), "yes (as-of)", "optional input"),
        ("customer.riskLevel", pv("riskLevel"), pts, pstatus(pv("riskLevel")), "yes (as-of)", "optional input + predicted flag"),
        ("customer.investmentCapacity", pv("investmentCapacity"), pts, pstatus(pv("investmentCapacity")), "yes (as-of)", "optional input + predicted flag"),
        ("customer.lastQuestionnaireDate", f"{q:%Y-%m-%d}" if q is not None else "—", f"snapshot {pts}",
         "missing (2000-01-01 placeholder)" if q is not None and q == pd.Timestamp("2000-01-01") else "observed",
         "yes" if q is not None and q <= t else "**no** (date after t)", "not used"),
        ("asset.assetName", ad["assetName"], f"snapshot {ad['timestamp']}", "observed (single snapshot, not versioned)",
         "uncertain: names are the latest version even when the row timestamp is older (see audit)", "prompt; anonymised in ablation"),
        ("asset.assetCategory", ad["assetCategory"], f"snapshot {ad['timestamp']}", "observed", static_ok, "input"),
        ("asset.assetSubCategory", ad["assetSubCategory"] or "(empty)", f"snapshot {ad['timestamp']}",
         "observed" if ad["assetSubCategory"] else "missing", static_ok, "input"),
        ("asset.sector", ad["sector"] or "(empty)", f"snapshot {ad['timestamp']}", "observed" if ad["sector"] else "missing", static_ok, "input"),
        ("asset.industry", ad["industry"] or "(empty)", f"snapshot {ad['timestamp']}", "observed" if ad["industry"] else "missing", static_ok, "input"),
        ("close_prices.closePrice, last < t", f"{px.closePrice.iloc[0]}" if len(px) else "—",
         f"{px.timestamp.iloc[0]:%Y-%m-%d}" if len(px) else "—",
         "observed (cleaned by dataset authors)" if len(px) else "missing", "yes", "optional input (price trend)"),
        ("limit_prices.profitability", "whole-period ROI", "minDate … maxDate (to 2022-11)", "derived", "**no — leakage**", "never used"),
    ]
    out += [f"\n## Fields: status, timestamp, leakage check (prediction time t = {t:%Y-%m-%d})\n",
            md_table(pd.DataFrame(fields, columns=["field", "value", "timestamp", "status", "known at t?", "use"])),
            "\n\nLeakage verdict: the transaction row is entirely post-t and serves only as a label. Inputs are "
            "the purchase history before t, the as-of profile snapshot and static asset attributes. Residual "
            "risks: asset names come from one later snapshot; riskLevel / investmentCapacity may be predicted "
            "values with an unknown estimation date.\n"]

    first_tx = tr.groupby("customerID").timestamp.min()
    first_prof = cu.groupby("customerID").timestamp.min()
    neg = tr.transactionID < 0
    audit = [
        ("customer_information has several dated snapshots",
         f"{(cu.groupby('customerID').size() > 1).sum():,} customers with >1 row", "as-of join (latest timestamp ≤ t)"),
        ("first profile snapshot later than first transaction", f"{(first_prof > first_tx).mean()*100:.1f}% of customers",
         "no profile before its first snapshot → missing"),
        ("lastQuestionnaireDate > snapshot timestamp", f"{(cu.lastQuestionnaireDate > cu.timestamp).mean()*100:.1f}% of rows",
         "not point-in-time; use only if ≤ t"),
        ("lastQuestionnaireDate = 2000-01-01 placeholder", f"{(cu.lastQuestionnaireDate == '2000-01-01').sum():,} rows", "missing"),
        ("riskLevel predicted", f"{cu.riskLevel.str.startswith('Predicted').mean()*100:.1f}% of rows", "keep predicted flag"),
        ("investmentCapacity predicted", f"{cu.investmentCapacity.str.startswith('Predicted').mean()*100:.1f}% of rows", "keep predicted flag"),
        ("limit_prices.profitability / priceMaxDate / maxDate", "whole-period values", "never use"),
        ("close_prices on or after t", "daily to 2022-11-29", "features only from prices strictly before t"),
        ("asset_information is one snapshot; row timestamp is not the attribute date",
         "GRS298343005 has timestamp 2018-01-02 but name 'HELLENiQ ENERGY HOLDINGS S.A.', in force on ATHEX only from 2022-10-04",
         "treat names as possibly post-t; anonymised-name ablation; sector/category assumed slow-changing"),
        ("reconstructed buys (transactionID < 0)", f"{neg.sum():,} rows, {(tr[neg].timestamp == '2018-01-02').sum():,} on 2018-01-02",
         "history only, never labels"),
        ("same transactionID for two customers (joint accounts)", f"{tr.transactionID.duplicated(keep=False).sum():,} rows",
         "one purchase may label two users"),
    ]
    out += ["\n## Dataset-wide timestamp / leakage audit\n",
            md_table(pd.DataFrame(audit, columns=["issue", "extent", "rule"])), ""]
    write("data_example.md", "make_data_example.py", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
