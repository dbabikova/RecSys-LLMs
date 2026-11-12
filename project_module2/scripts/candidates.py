"""Ceiling analysis for the Ranker track -> reports/candidates.md

Classical candidate generators on the provisional split; how many relevant assets reach the top-K
(an LLM reranker of a fixed top-K list can never exceed its Recall@K / hit@K).
Generators (all classical, trained only on buys before the cut, binary user x asset matrix):
  Pop6m   — distinct buyers in the 6 months before the cut
  ItemKNN — item-item cosine, score = sum of similarities to the user's items
  UserKNN — user-user cosine, k = 50 neighbours
  EASE    — Steck 2019 closed form, lambda = 200
For variant A already-bought assets are removed from every list (they cannot be relevant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_split, md_table, ndcg_at, write  # noqa: E402

KS = (10, 20, 50)


def matrix(split):
    users = sorted(split["tr_items"].index)
    items = split["universe"]
    ui, ii = {u: i for i, u in enumerate(users)}, {it: j for j, it in enumerate(items)}
    rows, cols = [], []
    for u, s in split["tr_items"].items():
        for it in s:
            if it in ii:
                rows.append(ui[u]); cols.append(ii[it])
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(users), len(items)))
    return X, ui, items


def scores_all(split):
    X, ui, items = matrix(split)
    Xd = X.toarray()
    train = split["train"]
    recent = train[train.timestamp >= train.timestamp.max() - pd.DateOffset(months=6)]
    pop = recent.drop_duplicates(["customerID", "ISIN"]).ISIN.value_counts()
    pop_vec = np.array([pop.get(i, 0) for i in items], dtype=float)

    norms = np.sqrt(Xd.sum(0)) + 1e-9
    S_item = (Xd.T @ Xd) / np.outer(norms, norms)
    np.fill_diagonal(S_item, 0)

    G = Xd.T @ Xd + 200.0 * np.eye(len(items))
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0)

    Xn = X.multiply(1 / (np.sqrt(np.asarray(X.sum(1))) + 1e-9)).tocsr()
    users = list(split["rel"])
    idx = np.array([ui[u] for u in users])
    S_user = (Xn[idx] @ Xn.T).toarray()
    S_user[np.arange(len(idx)), idx] = 0
    top = np.argsort(-S_user, axis=1)[:, :50]
    mask = np.zeros_like(S_user); np.put_along_axis(mask, top, 1, axis=1)
    user_knn = (S_user * mask) @ Xd

    Xu = Xd[idx]
    tie = pop_vec / (pop_vec.max() + 1) * 1e-6          # deterministic popularity tie-break
    return users, items, {
        "Pop6m": np.tile(pop_vec, (len(users), 1)),
        "ItemKNN": Xu @ S_item + tie,
        "UserKNN": user_knn + tie,
        "EASE": Xu @ B + tie,
    }


def evaluate(split):
    users, items, S = scores_all(split)
    rel, tr_items = split["rel"], split["tr_items"]
    hist = np.array([len(tr_items[u]) for u in users])
    rows, lists = [], {}
    for name, M in S.items():
        M = M.copy()
        if split["variant"] == "A":
            for r, u in enumerate(users):
                own = [j for j, it in enumerate(items) if it in tr_items[u]]
                M[r, own] = -np.inf
        order = np.argsort(-M, axis=1)[:, :max(KS)]
        ranked = [[items[j] for j in row] for row in order]
        lists[name] = dict(zip(users, ranked))
        nd = np.array([ndcg_at(rk, rel[u]) for u, rk in zip(users, ranked)])
        rec = {k: np.array([len(set(rk[:k]) & rel[u]) / len(rel[u]) for u, rk in zip(users, ranked)]) for k in KS}
        hit20 = np.array([bool(set(rk[:20]) & rel[u]) for u, rk in zip(users, ranked)])
        # reranking headroom inside the fixed top-20: oracle order vs random order (5 shuffles)
        orc = np.array([ndcg_at(sorted(rk[:20], key=lambda it: it not in rel[u]), rel[u]) for u, rk in zip(users, ranked)])
        rng = np.random.default_rng(0)
        rnd = np.mean([[ndcg_at(list(rng.permutation(rk[:20])), rel[u]) for u, rk in zip(users, ranked)]
                       for _ in range(5)], axis=0)
        rows.append([name, f"{nd.mean():.4f}"] + [f"{rec[k].mean():.3f}" for k in KS]
                    + [f"{hit20.mean():.3f}", f"{rnd.mean():.4f}", f"{orc.mean():.4f}",
                       f"{nd[hist <= 2].mean():.4f}", f"{nd[(hist > 2) & (hist <= 5)].mean():.4f}", f"{nd[hist > 5].mean():.4f}"])
    # union of top-10 of all generators (multi-generator candidate pool, as in Hou et al.)
    union = {u: list(dict.fromkeys(sum((lists[n][u][:10] for n in S), []))) for u in users}
    size = np.mean([len(v) for v in union.values()])
    rec_u = np.mean([len(set(v) & rel[u]) / len(rel[u]) for u, v in union.items()])
    hit_u = np.mean([bool(set(v) & rel[u]) for u, v in union.items()])
    rows.append([f"Union top-10 of all ({size:.1f} items)", "—", "—", f"{rec_u:.3f}", "—", f"{hit_u:.3f}", "—", "—", "—", "—", "—"])
    groups = f"hist ≤ 2: {(hist <= 2).sum()}, 3–5: {((hist > 2) & (hist <= 5)).sum()}, > 5: {(hist > 5).sum()}"
    return rows, len(users), groups


def main():
    out = ["# Ranker ceiling: classical candidate generators on the provisional split\n",
           "Recall@K is the ceiling of any reranker that only reorders a fixed top-K list. "
           "hit@20 = share of users with ≥ 1 relevant asset in the top-20. 'top-20 random / oracle order' = nDCG@10 after shuffling / ideally reordering "
           "the generator's own top-20 (lower and upper bound for a reranker of that list). nDCG@10 columns by "
           "number of distinct assets bought before the cut. Provisional split, not official.\n"]
    cols = ["generator", "nDCG@10", "Recall@10", "Recall@20", "Recall@50", "hit@20",
            "top-20 random order", "top-20 oracle order", "nDCG@10 hist≤2", "hist 3–5", "hist>5"]
    for v, title in [("A", "Variant A — new pairs only (FAR-Trans paper protocol)"),
                     ("B", "Variant B — all test buys (repeats count)")]:
        split = build_split(v)
        rows, n, groups = evaluate(split)
        out += [f"\n## {title}\n", f"Evaluated users: {n:,} ({groups}); universe {len(split['universe'])} assets.\n",
                md_table(pd.DataFrame(rows, columns=cols)), ""]
    write("candidates.md", "candidates.py", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
