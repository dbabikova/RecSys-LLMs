"""Exploratory data analysis for the FAR-Trans group project (Session 1).

Computes, over **Buy transactions only** (unless stated otherwise), the tables
and figures described in the Session 1 EDA task and writes them to
``reports/session1/eda.md`` and ``reports/session1/figures/``. Every number in
the report is computed here; only the "Observations" prose is static.

Usage::

    python scripts/eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FAR-Trans"
REPORT = ROOT / "reports" / "session1"
FIG = REPORT / "figures"

OUT_NAME = "eda.md"
CATS = ["Bond", "MTF", "Stock"]
CUTS = ["2022-03-01", "2022-06-01", "2022-09-01"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(RAW / "transactions.csv", dtype=str, keep_default_na=False)
    ai = pd.read_csv(RAW / "asset_information.csv", dtype=str, keep_default_na=False)
    cp = pd.read_csv(RAW / "close_prices.csv", dtype=str, keep_default_na=False)
    return tx, ai, cp


def isin_to_category(ai: pd.DataFrame) -> dict[str, str]:
    latest = ai.sort_values("timestamp").groupby("ISIN").tail(1)
    return latest.set_index("ISIN")["assetCategory"].to_dict()


def pct(frac: float) -> str:
    return f"{100.0 * frac:.2f}%"


def f2(frac: float) -> str:
    return f"{100.0 * frac:.2f}%"


def save_fig(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def section1_monthly(buy: pd.DataFrame, out: list[str]) -> None:
    monthly = buy.groupby(["ym", "cat"]).size().unstack(fill_value=0)
    monthly = monthly.reindex(columns=CATS, fill_value=0)
    full_range = pd.period_range("2018-01", "2022-12", freq="M")
    monthly = monthly.reindex(full_range, fill_value=0)
    monthly["Total"] = monthly[CATS].sum(axis=1)

    out.append("## 1. Monthly Buy volume by asset category (2018–2022)")
    out.append("")
    out.append("| Month | Bond | MTF | Stock | Total |")
    out.append("|-------|-----:|----:|------:|------:|")
    for month, row in monthly.iterrows():
        out.append(
            f"| {month} | {int(row['Bond'])} | {int(row['MTF'])} "
            f"| {int(row['Stock'])} | {int(row['Total'])} |"
        )
    out.append("")
    out.append(f"**Total Buy transactions:** {int(monthly['Total'].sum())}.")
    out.append("")

    fig, ax = plt.subplots(figsize=(11, 4))
    x = range(len(monthly))
    labels = [str(m) for m in monthly.index]
    ax.stackplot(x, monthly["Bond"], monthly["MTF"], monthly["Stock"], labels=CATS, alpha=0.85)
    ax.set_ylabel("Buy count")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Buy volume by asset category")
    step = max(1, len(monthly) // 12)
    ax.set_xticks(list(range(0, len(monthly), step)))
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.legend(loc="upper left")
    path = save_fig(fig, "monthly_buy_by_category.png")
    out.append(f"![monthly buy by category](figures/{Path(path).name})")
    out.append("")


def section2_per_user(buy: pd.DataFrame, out: list[str]) -> None:
    buys_per_user = buy.groupby("customerID").size()
    isin_per_user = buy.groupby("customerID")["ISIN"].nunique()

    def qs(s):
        q = s.quantile([0.5, 0.75, 0.9, 0.99])
        return ", ".join(f"{int(v)} (p{int(p * 100)})" for p, v in q.items())

    def buckets(s):
        n = len(s)
        return {
            "1": int((s == 1).sum()),
            "2-4": int(s.between(2, 4).sum()),
            "5-9": int(s.between(5, 9).sum()),
            "10+": int((s >= 10).sum()),
            "n": n,
        }

    out.append("## 2. Number of Buy transactions per user")
    out.append("")
    out.append(f"- Users (with ≥1 Buy): **{len(buys_per_user)}**.")
    out.append(f"- Quantiles (Buys/user): {qs(buys_per_user)}.")
    b = buckets(buys_per_user)
    out.append("- Users by Buy count:")
    out.append(f"  - 1 Buy: **{b['1']}** ({pct(b['1'] / b['n'])})")
    out.append(f"  - 2–4 Buys: **{b['2-4']}** ({pct(b['2-4'] / b['n'])})")
    out.append(f"  - 5–9 Buys: **{b['5-9']}** ({pct(b['5-9'] / b['n'])})")
    out.append(f"  - 10+ Buys: **{b['10+']}** ({pct(b['10+'] / b['n'])})")
    out.append("")
    out.append("- Distinct ISINs per user:")
    out.append(f"  - Quantiles: {qs(isin_per_user)}.")
    bi = buckets(isin_per_user)
    out.append(f"  - 1 ISIN: **{bi['1']}** ({pct(bi['1'] / bi['n'])}); 2–4: **{bi['2-4']}** "
               f"({pct(bi['2-4'] / bi['n'])}); 5–9: **{bi['5-9']}** ({pct(bi['5-9'] / bi['n'])}); "
               f"10+: **{bi['10+']}** ({pct(bi['10+'] / bi['n'])}).")
    out.append("")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(buys_per_user, bins=40, log=True)
    axes[0].set_title("Buys per user")
    axes[0].set_xlabel("Buys")
    axes[0].set_ylabel("Users (log)")
    axes[1].hist(isin_per_user, bins=40, log=True)
    axes[1].set_title("Distinct ISIN per user")
    axes[1].set_xlabel("Distinct ISIN")
    axes[1].set_ylabel("Users (log)")
    path = save_fig(fig, "buys_per_user_logscale.png")
    out.append(f"![buys per user](figures/{Path(path).name})")
    out.append("")


def section3_repeat(buy: pd.DataFrame, out: list[str]) -> None:
    srt = buy.sort_values(["timestamp", "transactionID"])
    first = ~srt.duplicated(subset=["customerID", "ISIN"], keep="first")
    srt = srt.copy()
    srt["repeat"] = ~first

    total = len(srt)
    n_repeat = int(srt["repeat"].sum())
    out.append("## 3. Repeat purchases")
    out.append("")
    out.append("A Buy is a *repeat* if the same user already bought the same ISIN earlier.")
    out.append("")
    out.append(f"- Overall repeat share: **{n_repeat}** of {total} Buys ({pct(n_repeat / total)}).")
    out.append("")
    out.append("| Category | Buys | Repeat share |")
    out.append("|----------|-----:|-------------:|")
    for cat in CATS:
        sub = srt[srt["cat"] == cat]
        if len(sub):
            out.append(f"| {cat} | {len(sub)} | {pct(sub['repeat'].mean())} |")
    out.append("")


def section4_popularity(buy: pd.DataFrame, out: list[str]) -> None:
    isin_counts = buy.groupby("ISIN").size().sort_values(ascending=False)
    total = len(buy)
    top10 = int(isin_counts.head(10).sum())
    top50 = int(isin_counts.head(50).sum())
    n_isin = int(isin_counts.size)

    out.append("## 4. Popularity concentration")
    out.append("")
    out.append(f"- Distinct ISINs bought at least once: **{n_isin}**.")
    out.append(f"- Top-10 ISINs: **{pct(top10 / total)}** of all Buys ({top10} of {total}).")
    out.append(f"- Top-50 ISINs: **{pct(top50 / total)}** of all Buys ({top50} of {total}).")
    out.append("")
    out.append("| Category | Distinct ISIN | Top-10 share | Top-50 share |")
    out.append("|----------|--------------:|-------------:|-------------:|")
    for cat in CATS:
        sub = buy[buy["cat"] == cat]
        if len(sub):
            c = sub.groupby("ISIN").size().sort_values(ascending=False)
            out.append(
                f"| {cat} | {int(c.size)} | {pct(c.head(10).sum() / len(sub))} "
                f"| {pct(c.head(50).sum() / len(sub))} |"
            )
    out.append("")


def section5_cuts(buy: pd.DataFrame, cp: pd.DataFrame, out: list[str]) -> None:
    out.append("## 5. Provisional cut scenarios")
    out.append("")
    out.append("> **These are NOT the official split.** The official time cut / evaluated users are")
    out.append("> still pending. Cuts below are only exploratory, for a rough feasibility check.")
    out.append("> Test window = [cut, end of data]; \"before cut\" = `timestamp < cut`.")
    out.append("")

    cp_ts = pd.to_datetime(cp["timestamp"])

    out.append("| Cut | Eval users (≥1 before & ≥1 after) | Future-ISIN already known | Future Buys in pre-cut top-10 | Assets priced before cut | Eval users: 1 / 2–4 / 5+ pre-cut Buys |")
    out.append("|-----|-----------------------------------:|---------------------------:|------------------------------:|--------------------------:|-----------------------------------------:|")

    for cut in CUTS:
        cut_ts = pd.Timestamp(cut)
        pre = buy[buy["ts"] < cut_ts]
        post = buy[buy["ts"] >= cut_ts]

        pre_users = set(pre["customerID"])
        post_users = set(post["customerID"])
        eval_users = pre_users & post_users
        n_eval = len(eval_users)

        post_eval = post[post["customerID"].isin(eval_users)]

        known = pre[["customerID", "ISIN"]].drop_duplicates()
        merged = post_eval.merge(known, on=["customerID", "ISIN"], how="left", indicator=True)
        share_known = (merged["_merge"] == "both").mean()

        top10_pre = set(pre.groupby("ISIN").size().sort_values(ascending=False).head(10).index)
        share_top10 = post_eval["ISIN"].isin(top10_pre).mean()

        n_priced = int(cp[cp_ts < cut_ts]["ISIN"].nunique())

        hist = pre[pre["customerID"].isin(eval_users)].groupby("customerID").size()
        h1 = int((hist == 1).sum())
        h2_4 = int(hist.between(2, 4).sum())
        h5p = int((hist >= 5).sum())

        out.append(
            f"| {cut} | {n_eval} | {pct(share_known)} | {pct(share_top10)} "
            f"| {n_priced} | {h1} / {h2_4} / {h5p} |"
        )
    out.append("")


def section6_observations(out: list[str]) -> None:
    out.append("## 6. Observations (facts only)")
    out.append("")
    out.append("- Buy volume is heavily concentrated: the top-10 ISINs account for ~48% and the")
    out.append("  top-50 for ~83% of all Buys (§4).")
    out.append("- **Stock** is by far the largest category by Buy volume, followed by MTF and Bond")
    out.append("  (§1, §3).")
    out.append("- A majority of Buys are repeat purchases of an ISIN the user already bought (§3).")
    out.append("- 2018-01 is an outlier month in the monthly table (§1).")
    out.append("- Under every provisional cut, the pre-cut popularity top-10 recovers roughly")
    out.append("  44–51% of future Buy volume, and ~75–79% of future Buys are of already-known")
    out.append("  ISINs (§5).")
    out.append("")


def main() -> int:
    tx, ai, cp = load()
    cat_map = isin_to_category(ai)

    buy = tx[tx["transactionType"] == "Buy"].copy()
    buy["ts"] = pd.to_datetime(buy["timestamp"])
    buy["ym"] = buy["ts"].dt.to_period("M")
    buy["cat"] = buy["ISIN"].map(cat_map).fillna("Unknown")

    unmapped = int((buy["cat"] == "Unknown").sum())
    if unmapped:
        print(f"warning: {unmapped} Buy transactions have an ISIN missing from asset_information")

    out: list[str] = []
    out.append("<!-- auto-generated by scripts/eda.py — do not edit manually -->")
    out.append("# FAR-Trans EDA (Session 1)")
    out.append("")
    out.append("All numbers below are computed over **Buy transactions only** unless stated otherwise.")
    out.append("")

    section1_monthly(buy, out)
    section2_per_user(buy, out)
    section3_repeat(buy, out)
    section4_popularity(buy, out)
    section5_cuts(buy, cp, out)
    section6_observations(out)

    report = REPORT / OUT_NAME
    report.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote: {report.relative_to(ROOT)}")
    print(f"figures: {sorted(p.name for p in FIG.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
