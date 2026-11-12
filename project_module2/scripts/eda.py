"""EDA aimed at choosing the research question -> reports/eda.md + reports/figures/*.png

Everything that depends on a split uses the PROVISIONAL split from common.py (paper protocol with
delta t = 6 months, latest cut). The official item scope / users / cut are pending.
Label variants:
  A "new-pairs" (FAR-Trans paper protocol): relevant = assets bought in test that the user did NOT buy
    before the cut; already-bought assets are excluded from the ranking.
  B "all-buys": relevant = every asset bought in test, repeats included.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CUT, FIGS, TEST_END, load, md_table, profile_asof, write  # noqa: E402

BLUE, SURF, INK, INK2, GRID = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)


def ndcg10(ranked: list, rel: set) -> float:
    gains = [1.0 / np.log2(i + 2) for i, it in enumerate(ranked[:10]) if it in rel]
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    return sum(gains) / idcg if idcg else 0.0


def pct(x):
    return f"{x*100:.1f}%"


def main():
    tr, cu, a, cp = (load(n) for n in ["transactions", "customer_information", "asset_information", "close_prices"])
    cat = a.drop_duplicates("ISIN").set_index("ISIN").assetCategory
    buys = tr[tr.transactionType == "Buy"]
    real = buys[buys.transactionID > 0]
    out = ["# FAR-Trans EDA (for choosing the research question)\n",
           f"Provisional split (assumption, not official): train < {CUT:%Y-%m-%d} ≤ test ≤ {TEST_END:%Y-%m-%d} "
           "(FAR-Trans paper protocol, Δt = 6 months, latest cut).\n"]

    # 1. volume over time
    m = real.set_index("timestamp").resample("MS").size()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(m.index, m.values, color=BLUE, linewidth=2)
    ax.axvline(CUT, color=INK2, linewidth=1, linestyle="--")
    ax.text(CUT, m.max(), " provisional cut", color=INK2, fontsize=8, va="top")
    style(ax, "Real Buy transactions per month", "", "buys")
    save(fig, "buys_per_month.png")
    out += ["## 1. Volume\n",
            md_table(pd.DataFrame([
                ("all transactions", f"{len(tr):,}"),
                ("Buy / Sell", f"{len(buys):,} / {(tr.transactionType == 'Sell').sum():,}"),
                ("reconstructed buys (transactionID < 0)", f"{len(buys) - len(real):,}"),
                ("real buys", f"{len(real):,}"),
                ("unique (customer, asset) buy pairs, real", f"{real[['customerID', 'ISIN']].drop_duplicates().shape[0]:,}"),
                ("customers / traded assets", f"{tr.customerID.nunique():,} / {tr.ISIN.nunique():,}"),
                ("density of real buy pairs", f"{real[['customerID','ISIN']].drop_duplicates().shape[0] / (real.customerID.nunique() * real.ISIN.nunique()):.4%}"),
            ], columns=["quantity", "value"])),
            "\n![buys per month](figures/buys_per_month.png)\n"]

    # 2. user activity
    per_u = real.groupby("customerID").ISIN.nunique()
    bins = [0, 1, 2, 5, 10, 20, 50, np.inf]
    labels = ["1", "2", "3–5", "6–10", "11–20", "21–50", ">50"]
    b = pd.cut(per_u, bins, labels=labels).value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar(labels, b.values, color=BLUE, width=0.7)
    style(ax, "Customers by number of distinct assets bought (real buys)", "distinct assets", "customers")
    save(fig, "assets_per_customer.png")
    out += ["\n## 2. Customer activity (real buys)\n",
            f"Customers with ≥1 real buy: {len(per_u):,}. Distinct assets per customer: median {per_u.median():.0f}, "
            f"mean {per_u.mean():.1f}, p90 {per_u.quantile(.9):.0f}, max {per_u.max()}. "
            f"Exactly one asset: {pct((per_u == 1).mean())}; ≤ 2 assets: {pct((per_u <= 2).mean())}.\n",
            "![assets per customer](figures/assets_per_customer.png)\n"]

    # repeat buying
    r = real.sort_values(["customerID", "timestamp", "transactionID"])
    seen_before = r.duplicated(["customerID", "ISIN"])
    all_sorted = buys.sort_values(["customerID", "timestamp", "transactionID"])
    rep_any = all_sorted.duplicated(["customerID", "ISIN"])[all_sorted.transactionID > 0]
    out += [f"Repeat buys: {pct(seen_before.mean())} of real buys are of an asset the customer had already "
            f"bought (incl. reconstructed holdings: {pct(rep_any.mean())}).\n"]

    # 3. popularity concentration
    pop = real.drop_duplicates(["customerID", "ISIN"]).ISIN.value_counts()
    share = pop.cumsum() / pop.sum()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(np.arange(1, len(share) + 1), share.values, color=BLUE, linewidth=2)
    ax.set_ylim(0, 1.02)
    style(ax, "Cumulative share of (customer, asset) buy pairs by asset rank", "asset popularity rank", "share")
    save(fig, "popularity_concentration.png")
    out += ["\n## 3. Item popularity\n",
            f"Assets with ≥1 real buy: {len(pop)}. Top-10 assets cover {pct(share.iloc[9])} of buy pairs, "
            f"top-20 {pct(share.iloc[19])}, top-50 {pct(share.iloc[49])}.\n",
            "![popularity](figures/popularity_concentration.png)\n",
            "Buy pairs by asset category (real buys):\n",
            md_table(real.drop_duplicates(["customerID", "ISIN"]).ISIN.map(cat).value_counts()
                     .rename_axis("category").reset_index(name="pairs")
                     .assign(share=lambda d: (d.pairs / d.pairs.sum()).map(pct))),
            "\n\nTop-10 assets by buyers:\n",
            md_table(pop.head(10).rename_axis("ISIN").reset_index(name="buyers").merge(
                a.drop_duplicates("ISIN")[["ISIN", "assetName", "assetCategory"]], on="ISIN"))]

    # 4. customer profile coverage
    last = cu.sort_values("timestamp").groupby("customerID").tail(1)
    prof_tab = pd.DataFrame({
        "riskLevel": [pct((last.riskLevel.str.startswith("Predicted")).mean()), pct((last.riskLevel == "Not_Available").mean())],
        "investmentCapacity": [pct(last.investmentCapacity.str.startswith("Predicted").mean()),
                               pct((last.investmentCapacity == "Not_Available").mean())]},
        index=["predicted", "missing"]).rename_axis("latest snapshot").reset_index()
    out += ["\n\n## 4. Customer profile fields\n",
            md_table(last.customerType.value_counts().rename_axis("customerType").reset_index(name="customers")),
            "\n\n", md_table(prof_tab), "\n"]

    # 5. provisional split
    train_b = buys[buys.timestamp < CUT]
    test_b = real[(real.timestamp >= CUT) & (real.timestamp <= TEST_END)]
    priced = cp.groupby("ISIN").timestamp.agg(["min", "max"])
    universe = set(priced[(priced["min"] <= CUT) & (priced["max"] >= TEST_END - pd.Timedelta(days=3))].index)
    tr_items = train_b.groupby("customerID").ISIN.agg(set)
    te_items = test_b.groupby("customerID").ISIN.agg(set)
    test_users = te_items.index
    warm = [u for u in test_users if u in tr_items.index]
    cold = len(test_users) - len(warm)
    relA = {u: (te_items[u] - tr_items[u]) & universe for u in warm}
    relA = {u: s for u, s in relA.items() if s}
    relB = {u: te_items[u] & universe for u in warm}
    relB = {u: s for u, s in relB.items() if s}
    test_new_items = set(test_b.ISIN) - set(train_b.ISIN)

    out += ["\n## 5. Provisional split diagnostics\n",
            md_table(pd.DataFrame([
                ("eligible universe (priced for whole test window)", f"{len(universe)} assets"),
                ("test buyers (≥1 real buy in test)", f"{len(test_users):,}"),
                ("…with ≥1 buy before the cut (warm)", f"{len(warm):,}"),
                ("…cold-start (no buy before the cut)", f"{cold:,} ({pct(cold / len(test_users))})"),
                ("evaluated users, variant A (new pairs only)", f"{len(relA):,}"),
                ("evaluated users, variant B (all buys)", f"{len(relB):,}"),
                ("mean relevant assets / user, A", f"{np.mean([len(s) for s in relA.values()]):.2f}"),
                ("mean relevant assets / user, B", f"{np.mean([len(s) for s in relB.values()]):.2f}"),
                ("share of B-relevant pairs that are repeats", pct(1 - sum(map(len, [relA.get(u, set()) for u in relB])) / sum(map(len, relB.values())))),
                ("assets bought in test never bought before the cut", f"{len(test_new_items)}"),
            ], columns=["quantity", "value"]))]

    # simple reference baselines (sanity, not the project baseline)
    recent = train_b[train_b.timestamp >= CUT - pd.DateOffset(months=6)]
    pop_rank = [i for i in recent.drop_duplicates(["customerID", "ISIN"]).ISIN.value_counts().index if i in universe]
    pop_all = [i for i in train_b.drop_duplicates(["customerID", "ISIN"]).ISIN.value_counts().index if i in universe]
    last_buy = train_b.sort_values("timestamp").groupby(["customerID", "ISIN"]).timestamp.max()

    def popular(u, ranking, exclude):
        ex = tr_items[u] if exclude else set()
        return [i for i in ranking if i not in ex][:10]

    def repeat_then_pop(u):
        own = last_buy.loc[u].sort_values(ascending=False).index.tolist()
        own = [i for i in own if i in universe]
        return (own + [i for i in pop_rank if i not in own])[:10]

    hist_len = {u: len(tr_items[u]) for u in warm}
    rows = []
    for name, fn, rel in [
        ("Popularity, last 6 m (A, exclude owned)", lambda u: popular(u, pop_rank, True), relA),
        ("Popularity, all train (A, exclude owned)", lambda u: popular(u, pop_all, True), relA),
        ("Popularity, last 6 m (B)", lambda u: popular(u, pop_rank, False), relB),
        ("Repeat own (recency) + popularity (B)", repeat_then_pop, relB),
    ]:
        s = pd.Series({u: ndcg10(fn(u), r) for u, r in rel.items()})
        h = pd.Series(hist_len).reindex(s.index)
        rows.append((name, len(s), f"{s.mean():.4f}", f"{s[h == 1].mean():.4f}",
                     f"{s[(h >= 2) & (h <= 4)].mean():.4f}", f"{s[h >= 5].mean():.4f}"))
    hb = pd.Series(hist_len).reindex(list(relA))
    out += ["\n\nReference baselines on the provisional split (nDCG@10, mean over users; "
            "columns split users by number of distinct assets bought before the cut):\n",
            md_table(pd.DataFrame(rows, columns=["method", "users", "nDCG@10", "hist = 1", "hist 2–4", "hist ≥ 5"])),
            f"\n\nVariant-A user groups: hist = 1: {(hb == 1).sum():,}; 2–4: {((hb >= 2) & (hb <= 4)).sum():,}; "
            f"≥ 5: {(hb >= 5).sum():,}.\n",
            "\nFor scale: the FAR-Trans paper reports nDCG@10 = 0.2710 for Popularity and 0.3404 for LightGCN, averaged "
            "over 61 bi-weekly cut points (2019-08 … ), so numbers here are a single-cut sanity check only.\n"]

    # profile coverage among evaluated users at the cut
    asof = profile_asof(cu, CUT).set_index("customerID")
    ev = pd.Index(list(relA))
    has = ev.isin(asof.index)
    pr = asof.reindex(ev[has])
    out += ["\n## 6. Profile availability for evaluated users (variant A) at the cut\n",
            f"Point-in-time profile exists for {pct(has.mean())} of evaluated users; among them riskLevel is predicted for "
            f"{pct(pr.riskLevel.str.startswith('Predicted').mean())} and missing for {pct((pr.riskLevel == 'Not_Available').mean())}; "
            f"investmentCapacity predicted for {pct(pr.investmentCapacity.str.startswith('Predicted').mean())}.\n"]

    rel_cat = pd.Series([cat.get(i) for s in relA.values() for i in s]).value_counts()
    out += ["\nRelevant (variant A) pairs by asset category: "
            + ", ".join(f"{k}: {v:,} ({pct(v / rel_cat.sum())})" for k, v in rel_cat.items()) + "\n"]
    write("eda.md", "eda.py", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
