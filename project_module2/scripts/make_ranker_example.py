"""One real FAR-Trans record + the Ranker-instance it produces -> reports/ranker_example.md

Instance = (user history before t, as-of profile, ItemKNN top-20 candidates with asset metadata) -> label.
User chosen deterministically: first evaluated user (variant A, provisional split) ordered by masked id
whose history has 2-4 distinct assets and whose new purchase is inside the ItemKNN top-20.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn_setup import prepare  # noqa: E402
from common import CUT, DATA, TEST_END, load, md_table, profile_asof, write  # noqa: E402

K = 20


def mask(s: str) -> str:
    return s[:4] + "…****"


def main():
    tr, cu, a, cp = (load(n) for n in ["transactions", "customer_information", "asset_information", "close_prices"])
    S = prepare(tr, cp)
    meta = a.drop_duplicates("ISIN").set_index("ISIN")
    uni = set(S.universe)

    chosen = None
    for u in sorted(S.te_items.index):
        if u not in S.tr_items.index or not (2 <= len(S.tr_items[u]) <= 4):
            continue
        relA = (S.te_items[u] - S.tr_items[u]) & uni
        if not relA:
            continue
        cands = [i for i, _ in S.knn(u)[:K]]
        if relA & set(cands):
            chosen = (u, relA, cands)
            break
    u, relA, cands = chosen
    relB = S.te_items[u] & uni

    # verbatim raw label line: first real test-window Buy of a relevant asset
    lab = tr[(tr.customerID == u) & (tr.ISIN.isin(relA)) & (tr.transactionType == "Buy") & (tr.transactionID > 0)
             & (tr.timestamp >= CUT)].sort_values(["timestamp", "transactionID"]).iloc[0]
    with open(DATA / "transactions.csv", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        row = next(x for x in r if x[h.index("transactionID")] == str(lab.transactionID) and x[0] == u)
    row_m = [mask(v) if k in ("customerID", "transactionID") else v for k, v in zip(h, row)]

    hist = tr[(tr.customerID == u) & (tr.timestamp < CUT)].sort_values("timestamp")
    prof = profile_asof(cu, CUT)
    prof = prof[prof.customerID == u]
    later_prof = cu[(cu.customerID == u) & (cu.timestamp > CUT)]

    def asset_cols(isin):
        m = meta.loc[isin]
        return [m.assetName, m.assetCategory, m.assetSubCategory if pd.notna(m.assetSubCategory) else "∅",
                m.sector if pd.notna(m.sector) else "∅", m.industry if pd.notna(m.industry) else "∅", m.marketID]

    out = ["# Data example for the Ranker RQ (one real record → one experimental instance)\n",
           f"Provisional split (assumption, official one pending): prediction time t = {CUT:%Y-%m-%d}; "
           f"label window {CUT:%Y-%m-%d} … {TEST_END:%Y-%m-%d}. `customerID` / `transactionID` masked; "
           "all other values verbatim. ISINs are public security identifiers.\n",
           "## 1. Raw record (label source)\n", "```",
           "source: FAR-Trans (Univ. of Glasgow, researchdata.gla.ac.uk/1658), transactions.csv",
           f"record id: transactionID {row_m[h.index('transactionID')]}",
           "fields: " + ", ".join(h),
           "values: " + ", ".join(row_m),
           f"window: label window (after t = {CUT:%Y-%m-%d}); this row is the TARGET, never an input",
           "```\n",
           "## 2. Inputs available at t (what both rankers see)\n",
           f"**Purchase/sale history before t** ({len(hist)} transactions, {hist.ISIN.nunique()} distinct assets; "
           "amounts shown for completeness — whether they enter the prompt/features is a design choice):\n",
           md_table(pd.DataFrame([[f"{x.timestamp:%Y-%m-%d}", x.transactionType, x.ISIN, *asset_cols(x.ISIN)[:2],
                                   "reconstructed" if x.transactionID < 0 else "real"]
                                  for x in hist.itertuples()],
                                 columns=["date", "type", "ISIN", "assetName", "category", "origin"])),
           "\n\n**Customer profile as of t** (latest snapshot with timestamp ≤ t):\n",
           md_table(prof.assign(customerID=prof.customerID.map(mask),
                                timestamp=prof.timestamp.dt.date,
                                lastQuestionnaireDate=prof.lastQuestionnaireDate.dt.date)),
           f"\n\nLater snapshots after t (excluded): {len(later_prof)}.\n",
           f"\n## 3. Candidates: ItemKNN top-{K} (owned assets excluded, variant A)\n",
           md_table(pd.DataFrame([[k + 1, i, *asset_cols(i), "✔" if i in relA else ""]
                                  for k, i in enumerate(cands)],
                                 columns=["rank", "ISIN", "assetName", "category", "subCategory", "sector",
                                          "industry", "market", "bought after t"])),
           f"\n\nLabels: variant A (new pairs) = {sorted(relA)}; variant B (all purchases) = {sorted(relB)}. "
           f"Relevant asset sits at ItemKNN rank {[k + 1 for k, i in enumerate(cands) if i in relA]} — a reranker "
           "can only move it within this list.\n"]

    pr = prof.iloc[0] if len(prof) else None
    st = (lambda v: "predicted" if str(v).startswith("Predicted") else ("missing" if v == "Not_Available" else "observed"))
    fields = [
        ("transactions: ISIN, type, timestamp (history)", "LLM prompt + LambdaMART features",
         "observed (reconstructed rows flagged)", f"{hist.timestamp.min():%Y-%m-%d} … {hist.timestamp.max():%Y-%m-%d}",
         f"OK: all < t ({(hist.timestamp < CUT).all()})"),
        ("transactions: totalValue, units, channel (history)", "optional", "observed / derived (value = units × price)",
         "same as row", "OK if from rows < t"),
        ("customer: customerType", "both", "observed", f"{pr.timestamp:%Y-%m-%d}" if pr is not None else "—", "OK: snapshot ≤ t"),
        ("customer: riskLevel", "both", st(pr.riskLevel) if pr is not None else "missing",
         f"{pr.timestamp:%Y-%m-%d}" if pr is not None else "—", "OK; 'predicted' flag kept, prediction date unknown"),
        ("customer: investmentCapacity", "both", st(pr.investmentCapacity) if pr is not None else "missing",
         f"{pr.timestamp:%Y-%m-%d}" if pr is not None else "—", "same"),
        ("customer: lastQuestionnaireDate", "not used", "observed / placeholder 2000-01-01",
         f"{pr.lastQuestionnaireDate:%Y-%m-%d}" if pr is not None else "—",
         "use only if ≤ t" + (" (here ≤ t)" if pr is not None and pr.lastQuestionnaireDate <= CUT else " (here > t → drop)")),
        ("asset: name, category, subCategory, sector, industry, market", "both (text for LLM, categorical for LambdaMART)",
         "observed static; ∅ = missing", "asset_information.timestamp ≤ 2022-10-31", "low risk: static metadata"),
        ("popularity of candidate (buyers in 6 m before t)", "both", "derived from rows < t", "< t", "OK"),
        ("ItemKNN score / rank", "LambdaMART feature, LLM sees order only if we choose", "derived from rows < t",
         "< t", "OK"),
        ("close prices ≤ t", "optional", "observed", "≤ t", "OK only up to t"),
        ("candidate eligibility (priced over whole test window)", "protocol filter, identical for all methods",
         "derived", "uses prices up to 2022-11-29", "future info → must NOT be a feature"),
        ("limit_prices.profitability", "never", "derived over 2018–2022", "whole period", "LEAKAGE → excluded"),
        ("test-window transactions", "label only", "observed", f"{CUT:%Y-%m-%d} …", "never an input"),
    ]
    out += ["\n## 4. Field status, timestamps and leakage check\n",
            md_table(pd.DataFrame(fields, columns=["field", "used by", "status", "timestamp", "leakage check"])), "\n"]
    write("ranker_example.md", "make_ranker_example.py", "\n".join(out))


if __name__ == "__main__":
    main()
