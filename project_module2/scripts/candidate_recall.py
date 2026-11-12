"""Candidate-ceiling check for the Ranker track -> reports/candidate_recall.md

On the provisional split (common.CUT), for classical candidate generators report Recall@K
(share of a user's relevant assets inside the top-K list) and nDCG@10 of their own order.
Recall@20 is the ceiling any reranker of a top-20 list can reach.
Generators: Popularity (last 6 months), ItemKNN (cosine on user x asset buy matrix),
Repeat-own + popularity (variant B only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CUT, TEST_END, load, md_table, write  # noqa: E402
from eda import ndcg10  # noqa: E402

KS = (10, 20, 50)


def main():
    tr, cp = load("transactions"), load("close_prices")
    buys = tr[tr.transactionType == "Buy"]
    train = buys[buys.timestamp < CUT]
    test = buys[(buys.transactionID > 0) & (buys.timestamp >= CUT) & (buys.timestamp <= TEST_END)]
    priced = cp.groupby("ISIN").timestamp.agg(["min", "max"])
    universe = sorted(priced[(priced["min"] <= CUT) & (priced["max"] >= TEST_END - pd.Timedelta(days=3))].index)
    uidx = {a: i for i, a in enumerate(universe)}

    tr_items = train.groupby("customerID").ISIN.agg(set)
    te_items = test.groupby("customerID").ISIN.agg(set)
    warm = [u for u in te_items.index if u in tr_items.index]
    rel = {"A": {}, "B": {}}
    for u in warm:
        a = (te_items[u] - tr_items[u]) & set(universe)
        b = te_items[u] & set(universe)
        if a:
            rel["A"][u] = a
        if b:
            rel["B"][u] = b

    # popularity (distinct buyers, last 6 months of train)
    recent = train[train.timestamp >= CUT - pd.DateOffset(months=6)].drop_duplicates(["customerID", "ISIN"])
    pop = recent.ISIN.value_counts().reindex(universe).fillna(0)
    pop_rank = pop.sort_values(ascending=False).index.tolist()

    # ItemKNN: cosine similarity between assets on binary user x asset matrix (train buys)
    pairs = train[train.ISIN.isin(uidx)].drop_duplicates(["customerID", "ISIN"])
    users = {u: i for i, u in enumerate(pairs.customerID.unique())}
    X = np.zeros((len(users), len(universe)), dtype=np.float32)
    X[pairs.customerID.map(users).values, pairs.ISIN.map(uidx).values] = 1.0
    co = X.T @ X
    norm = np.sqrt(np.diag(co)) + 1e-9
    sim = co / norm[:, None] / norm[None, :]
    np.fill_diagonal(sim, 0.0)
    popv = pop.values / (pop.values.max() + 1e-9)

    last_buy = train.sort_values("timestamp").groupby(["customerID", "ISIN"]).timestamp.max()

    def gen_pop(u, excl):
        return [i for i in pop_rank if not (excl and i in tr_items[u])]

    def gen_knn(u, excl):
        own = [uidx[i] for i in tr_items[u] if i in uidx]
        s = sim[own].sum(axis=0) if own else np.zeros(len(universe))
        s = s + 1e-3 * popv  # popularity tie-break for users with no in-universe history
        order = [universe[j] for j in np.argsort(-s)]
        return [i for i in order if not (excl and i in tr_items[u])]

    def gen_repeat(u, excl):
        own = [i for i in last_buy.loc[u].sort_values(ascending=False).index if i in uidx]
        return own + [i for i in pop_rank if i not in own]

    rows = []
    for var, excl in (("A", True), ("B", False)):
        gens = [("Popularity (6 m)", gen_pop), ("ItemKNN (cosine)", gen_knn)]
        if var == "B":
            gens.append(("Repeat-own + popularity", gen_repeat))
        for name, g in gens:
            rec = {k: [] for k in KS}
            nd = []
            for u, r in rel[var].items():
                lst = g(u, excl)
                for k in KS:
                    rec[k].append(len(set(lst[:k]) & r) / len(r))
                nd.append(ndcg10(lst, r))
            rows.append((var, name, len(nd), f"{np.mean(nd):.4f}",
                         *[f"{np.mean(rec[k]):.3f}" for k in KS],
                         f"{np.mean([x > 0 for x in rec[20]]):.3f}"))
    cols = ["variant", "generator", "users", "nDCG@10", *[f"Recall@{k}" for k in KS], "≥1 hit in top-20"]
    out = ["# Candidate ceiling for a reranker (provisional split)\n",
           f"Split (assumption): train < {CUT:%Y-%m-%d} ≤ test ≤ {TEST_END:%Y-%m-%d}; universe = {len(universe)} assets "
           "priced over the whole test window. Variant A: relevant = new (user, asset) pairs, owned assets excluded "
           "from the list; variant B: all test buys relevant.\n",
           md_table(pd.DataFrame(rows, columns=cols)),
           "\n\nRecall@20 is the upper bound for any reordering of a top-20 list; nDCG@10 is the starting point "
           "the reranker must beat. Single cut, sanity check only.\n"]
    write("candidate_recall.md", "candidate_recall.py", "\n".join(out))


if __name__ == "__main__":
    main()
