"""Pure computations for the audit; customer identifiers stay in memory."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

PAIR = ["customerID", "ISIN"]


def latest(frame, key, cut=None):
    selected = frame if cut is None else frame[frame.timestamp.le(cut)]
    return selected.sort_values("timestamp", kind="stable").drop_duplicates(key, keep="last")


def repeat_stats(buys):
    # A repeat occurs strictly after the FIRST day this customer bought the asset.
    # Separate same-day multiplicity because intraday timestamps are unavailable.
    first = buys.groupby(PAIR).timestamp.transform("min")
    return {
        "rows": len(buys),
        "pairs": len(buys.drop_duplicates(PAIR)),
        "beyond_first_row": int(buys.duplicated(PAIR).sum()),
        "strictly_later_day": int(buys.timestamp.gt(first).sum()),
        "extra_same_first_day": int(len(buys) - len(buys.drop_duplicates(PAIR))
                                    - buys.timestamp.gt(first).sum()),
    }


def prepare(tables):
    tx = tables["transactions"]
    prices = tables["close_prices"]
    limits = tables["limit_prices"]
    assets = tables["asset_information"]
    profiles = tables["customer_information"]
    buys = tx[tx.transactionType.eq("Buy")]
    real = tx[tx.transactionID.ge(0)]
    real_buys = real[real.transactionType.eq("Buy")]
    bounds = prices.groupby("ISIN").timestamp.agg(["min", "max"])
    end = min(tx.timestamp.max(), prices.timestamp.max())
    # Find latest calendar date whose forward six-calendar-month horizon fits.
    cut = end - pd.DateOffset(months=6)
    while cut + pd.Timedelta(days=1) + pd.DateOffset(months=6) <= end:
        cut += pd.Timedelta(days=1)
    assert cut + pd.DateOffset(months=6) == end
    eligible = set(bounds.index[bounds["min"].le(cut) & bounds["max"].ge(end)])
    linked = tx.merge(prices, on=["ISIN", "timestamp"], how="left", validate="many_to_one")
    neg = tx[tx.transactionID.lt(0)].merge(
        bounds.rename(columns={"min": "first_price", "max": "last_price"}),
        left_on="ISIN", right_index=True, validate="many_to_one",
    )
    valid_value = linked.closePrice.notna()
    latest_profiles = latest(profiles, "customerID")
    asof_profiles = latest(profiles, "customerID", cut)
    pair_counts = buys.groupby("customerID").ISIN.nunique()
    pair_counts_real = real_buys.groupby("customerID").ISIN.nunique().reindex(
        tx.customerID.unique(), fill_value=0
    )
    pop = buys.ISIN.value_counts()
    pop_real = real_buys.ISIN.value_counts()
    metadata = latest(assets, "ISIN")
    result = {
        "tx": tx, "buys": buys, "real": real, "real_buys": real_buys,
        "bounds": bounds, "end": end, "cut": cut, "eligible": eligible,
        "negative": neg, "linked": linked, "latest_profiles": latest_profiles,
        "asof_profiles": asof_profiles, "metadata": metadata,
        "repeat": repeat_stats(buys), "repeat_real": repeat_stats(real_buys),
        "activity": pair_counts, "activity_real": pair_counts_real,
        "pop": pop, "pop_real": pop_real,
        "value_matched": int(valid_value.sum()),
        "value_exact": int(np.isclose(
            linked.loc[valid_value, "totalValue"],
            linked.loc[valid_value, "units"] * linked.loc[valid_value, "closePrice"],
            rtol=1e-6, atol=0.01,
        ).sum()),
    }
    return result


def evaluate_split(tables, stats, include_reconstructed=True):
    tx = tables["transactions"]
    cut, end, eligible = stats["cut"], stats["end"], stats["eligible"]
    buys = tx[tx.transactionType.eq("Buy")]
    history = buys[buys.timestamp.le(cut)]
    if not include_reconstructed:
        history = history[history.transactionID.ge(0)]
    labels = buys[
        buys.timestamp.gt(cut) & buys.timestamp.le(end)
        & buys.transactionID.ge(0) & buys.ISIN.isin(eligible)
    ].copy()
    history_eligible = history[history.ISIN.isin(eligible)]
    seen = history.groupby("customerID").ISIN.agg(set).to_dict()
    warm = set(history_eligible.customerID)
    warm_buy_any = set(history.customerID)
    warm_transaction_any = set(tx.loc[tx.timestamp.le(cut), "customerID"])
    popularity = history_eligible.ISIN.value_counts().to_dict()
    global_order = sorted(eligible, key=lambda item: (-popularity.get(item, 0), item))
    own = history_eligible.groupby(PAIR).agg(
        count=("ISIN", "size"), last=("timestamp", "max")
    ).reset_index()
    own["popularity"] = own.ISIN.map(popularity).fillna(0)
    own = own.sort_values(
        ["count", "last", "popularity", "ISIN"],
        ascending=[False, False, False, True], kind="stable",
    )
    own_order = own.groupby("customerID", sort=False).ISIN.agg(list).to_dict()
    discounts = 1 / np.log2(np.arange(2, 12))
    rows, details = [], {}
    for variant in ["A", "B"]:
        selected = labels
        if variant == "A":
            keep = [item not in seen.get(user, set())
                    for user, item in zip(labels.customerID, labels.ISIN)]
            selected = labels.loc[keep]
        relevance_all = selected.groupby("customerID").ISIN.agg(set).to_dict()
        cold = set(relevance_all) - warm
        relevance = {u: r for u, r in relevance_all.items() if u in warm}
        counts = np.array([len(values) for values in relevance.values()])
        scores_pop, scores_repeat = [], []
        for user, relevant in relevance.items():
            forbidden = seen.get(user, set()) if variant == "A" else set()
            pool = [item for item in global_order if item not in forbidden]
            rank_pop = pool[:10]
            rank_repeat = [
                item for item in own_order.get(user, []) if item not in forbidden
            ]
            own_items = set(rank_repeat)
            rank_repeat = (rank_repeat + [
                item for item in pool if item not in own_items
            ])[:10]
            idcg = discounts[:min(10, len(relevant))].sum()
            for ranking, sink in [(rank_pop, scores_pop), (rank_repeat, scores_repeat)]:
                assert len(ranking) == len(set(ranking)) == 10
                assert set(ranking) <= eligible
                assert not set(ranking) & forbidden
                score = sum(d for d, item in zip(discounts, ranking) if item in relevant) / idcg
                assert 0 <= score <= 1 + 1e-12
                sink.append(score)
            if variant == "A":
                assert rank_pop == rank_repeat
        rows.append({
            "variant": variant, "label_users_before_warm_filter": len(relevance_all),
            "cold_users": len(cold), "users": len(relevance),
            "no_buy_history_any_asset": len(set(relevance_all) - warm_buy_any),
            "no_transaction_history_any_asset": len(set(relevance_all) - warm_transaction_any),
            "relevant_pairs": int(counts.sum()),
            "relevant_mean": float(counts.mean()),
            "relevant_median": float(np.median(counts)),
            "relevant_p90": float(np.quantile(counts, .9)),
            "relevant_max": int(counts.max()),
            "ndcg_popularity": float(np.mean(scores_pop)),
            "ndcg_repeat_popularity": float(np.mean(scores_repeat)),
        })
        details[variant] = {
            "relevance": relevance, "labels": selected, "cold": cold,
            "history_rows": len(history), "history_eligible_rows": len(history_eligible),
        }
    return rows, details


def questionnaire_stats(path):
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    return {
        "lines": len(lines), "nonempty_lines": sum(bool(s.strip()) for s in lines),
        "questions": len(re.findall(r"^\s*\d+\.", text, re.M)),
        "answers": len(re.findall(r"^\s*[a-z]\.", text, re.M)),
        "sections": sum(bool(s.strip()) and not s[0].isspace() for s in lines),
    }
