"""Schema profile of every FAR-Trans table + key / referential-integrity checks -> reports/schema.md"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load, md_table, write  # noqa: E402


def profile(df):
    rows = []
    for c in df.columns:
        s = df[c]
        if str(s.dtype).startswith("datetime"):
            rng = f"{s.min():%Y-%m-%d} … {s.max():%Y-%m-%d}"
        elif s.dtype.kind in "if":
            rng = f"{s.min():.4g} … {s.max():.4g}"
        else:
            top = s.value_counts().head(3).index.tolist()
            rng = ", ".join(str(t)[:30] for t in top)
        rows.append({"column": c, "dtype": str(s.dtype), "null %": f"{s.isna().mean()*100:.1f}",
                     "n unique": s.nunique(), "range / top values": rng})
    import pandas as pd
    return pd.DataFrame(rows)


def main():
    t = {n: load(n) for n in ["transactions", "customer_information", "asset_information",
                              "close_prices", "limit_prices", "markets"]}
    out = ["# FAR-Trans: schema profile\n",
           "Source: FAR-Trans archive (University of Glasgow, https://researchdata.gla.ac.uk/1658/), "
           "files in `data/FAR-Trans/` (not committed).\n"]
    for name, df in t.items():
        out.append(f"\n## {name}.csv — {len(df):,} rows × {df.shape[1]} cols\n")
        out.append(md_table(profile(df)))
        out.append("")

    tr, cu, asi, cp, lp, mk = (t[k] for k in t)
    a_uni = asi.drop_duplicates("ISIN")
    checks = [
        ("transactions: transactionID unique", tr.transactionID.is_unique, f"{tr.transactionID.nunique():,} unique / {len(tr):,}"),
        ("transactions: full-row duplicates", tr.duplicated().sum() == 0, f"{tr.duplicated().sum():,}"),
        ("transactions: customers present in customer_information", tr.customerID.isin(cu.customerID).all(),
         f"{(~tr.customerID.drop_duplicates().isin(cu.customerID)).sum():,} missing"),
        ("transactions: ISIN present in asset_information", tr.ISIN.isin(asi.ISIN).all(),
         f"{(~tr.ISIN.drop_duplicates().isin(asi.ISIN)).sum():,} missing"),
        ("transactions: ISIN has close prices", tr.ISIN.isin(cp.ISIN).all(),
         f"{(~tr.ISIN.drop_duplicates().isin(cp.ISIN)).sum():,} missing"),
        ("transactions: marketID present in markets", tr.marketID.isin(mk.marketID).all(),
         f"{(~tr.marketID.drop_duplicates().isin(mk.marketID)).sum():,} missing"),
        ("customer_information: rows per customer > 1 (profile history)", True,
         f"{(cu.groupby('customerID').size() > 1).sum():,} of {cu.customerID.nunique():,} customers"),
        ("asset_information: ISIN unique", asi.ISIN.is_unique,
         f"{a_uni.shape[0]:,} unique ISIN / {len(asi):,} rows"),
        ("asset_information: ISIN has close prices", a_uni.ISIN.isin(cp.ISIN).all(),
         f"{(~a_uni.ISIN.isin(cp.ISIN)).sum():,} without prices"),
        ("close_prices: (ISIN, date) unique", not cp.duplicated(["ISIN", "timestamp"]).any(),
         f"{cp.duplicated(['ISIN', 'timestamp']).sum():,} dup"),
        ("limit_prices: one row per priced ISIN", lp.ISIN.is_unique and set(lp.ISIN) == set(cp.ISIN),
         f"{lp.ISIN.nunique():,} vs {cp.ISIN.nunique():,} priced ISIN"),
    ]
    import pandas as pd
    out.append("\n## Keys and referential integrity\n")
    out.append(md_table(pd.DataFrame(checks, columns=["check", "ok", "detail"])))

    out.append("\n\n## Asset catalogue (unique ISIN)\n")
    out.append(md_table(a_uni.groupby("assetCategory").agg(
        assets=("ISIN", "size"),
        with_sector=("sector", lambda s: s.notna().sum()),
        subcategories=("assetSubCategory", "nunique")).reset_index()))
    traded = a_uni[a_uni.ISIN.isin(tr.ISIN)]
    out.append(f"\n\nTraded assets (≥1 transaction): {traded.ISIN.nunique()} — "
               + ", ".join(f"{k}: {v}" for k, v in traded.assetCategory.value_counts().items()))
    out.append("\n\nAsset sub-categories:\n")
    out.append(md_table(a_uni.groupby(["assetCategory", "assetSubCategory"]).size()
                        .rename("n").reset_index().sort_values(["assetCategory", "n"], ascending=[True, False])))

    out.append("\n\n## questionnaires.csv\n")
    q = (DATA / "questionnaires.csv").read_text(encoding="utf-8").splitlines()
    sections = [l.strip() for l in q if l.strip() and not l.startswith(("\t", " "))]
    nq = sum(1 for l in q if re.match(r"^\s+\d+\.", l))
    out.append(f"Not a CSV: question text and answer options only (no customer answers). "
               f"{nq} questions in sections: {', '.join(sections)}.\n")
    write("schema.md", "inspect_schema.py", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
