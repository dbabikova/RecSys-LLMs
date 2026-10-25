"""Implicit feedback ALS (Hu, Koren, Volinsky 2008) on numpy/scipy.

r_ui = number of distinct days a user watched a show in train.
confidence c_ui = 1 + alpha * r_ui, preference p_ui = 1 if r_ui > 0.
"""

import numpy as np
from scipy import sparse


def build_csr(n_days: dict) -> tuple:
    """Build the sparse user-show matrix from a {(user, show): n_days} dict.

    Returns (csr, user_index, item_index, item_ids).
    """
    users = sorted({u for (u, _s) in n_days})
    items = sorted({s for (_u, s) in n_days})
    user_index = {int(u): i for i, u in enumerate(users)}
    item_index = {int(s): j for j, s in enumerate(items)}

    rows = np.empty(len(n_days), dtype=np.int32)
    cols = np.empty(len(n_days), dtype=np.int32)
    data = np.empty(len(n_days), dtype=np.float64)
    for k, ((u, s), nd) in enumerate(n_days.items()):
        rows[k] = user_index[int(u)]
        cols[k] = item_index[int(s)]
        data[k] = float(nd)

    csr = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    return csr, user_index, item_index, items


def compute_loss(X, Y, coo_rows, coo_cols, coo_data, alpha, reg) -> float:
    """Full HKV objective L = sum c_ui (p_ui - s_ui)^2 + reg * (|X|^2 + |Y|^2)."""
    s = (X[coo_rows] * Y[coo_cols]).sum(axis=1)
    nnz = coo_rows.shape[0]
    c1 = nnz + alpha * float(coo_data.sum())
    c2 = float(s.sum()) + alpha * float((coo_data * s).sum())
    c3 = float((X.T @ X * (Y.T @ Y)).sum())  # sum over ALL pairs of s^2 (Gram trick)
    c4 = alpha * float((coo_data * s * s).sum())
    regterm = reg * (float((X * X).sum()) + float((Y * Y).sum()))
    return c1 - 2.0 * c2 + c3 + c4 + regterm


def als(csr, n_factors, reg, alpha, n_iter=15, seed=42, verbose=False):
    """Run alternating least squares; returns (X, Y, losses)."""
    n_users, n_items = csr.shape
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_users, n_factors)) * 0.1
    Y = rng.standard_normal((n_items, n_factors)) * 0.1
    csc = csr.tocsc()
    coo = csr.tocoo()
    cr = coo.row.astype(np.int32)
    cc = coo.col.astype(np.int32)
    cd = coo.data.astype(np.float64)
    eye = np.eye(n_factors)

    losses = []
    for _it in range(n_iter):
        YtY = Y.T @ Y
        for u in range(n_users):
            a = csr.indptr[u]
            b = csr.indptr[u + 1]
            if a == b:
                continue
            cols = csr.indices[a:b]
            r = csr.data[a:b]
            Ysub = Y[cols]
            Ys = Ysub * r[:, None]
            A = YtY + alpha * (Ysub.T @ Ys) + reg * eye
            rhs = Ysub.sum(axis=0) + alpha * Ys.sum(axis=0)
            X[u] = np.linalg.solve(A, rhs)

        XtX = X.T @ X
        for i in range(n_items):
            a = csc.indptr[i]
            b = csc.indptr[i + 1]
            if a == b:
                continue
            rows = csc.indices[a:b]
            r = csc.data[a:b]
            Xsub = X[rows]
            Xs = Xsub * r[:, None]
            A = XtX + alpha * (Xsub.T @ Xs) + reg * eye
            rhs = Xsub.sum(axis=0) + alpha * Xs.sum(axis=0)
            Y[i] = np.linalg.solve(A, rhs)

        losses.append(compute_loss(X, Y, cr, cc, cd, alpha, reg))
        if verbose:
            print(f"    iter {_it + 1:2d}: loss = {losses[-1]:.6f}")

    return X, Y, losses


def predict_rankings(X, Y, user_index, item_ids, candidates, global_rank, cohort, k=5):
    """Top-k recommendations (scores only on train items that are val candidates)."""
    item_ids = np.asarray(item_ids, dtype=np.int64)
    cand_mask = np.array([int(s) in candidates for s in item_ids], dtype=bool)
    Yc = Y[cand_mask]
    cand_ids = item_ids[cand_mask]

    rankings: dict[int, list[int]] = {}
    n_pad = 0
    for u in cohort:
        ri = user_index.get(int(u))
        if ri is None or Yc.shape[0] == 0:
            rec = list(global_rank[:k])
            n_pad += k
        else:
            scores = X[ri] @ Yc.T
            order = np.lexsort((cand_ids, -scores))[:k]
            rec = [int(x) for x in cand_ids[order]]
            if len(rec) < k:
                n_pad += k - len(rec)
                seen = set(rec)
                for gs in global_rank:
                    if len(rec) >= k:
                        break
                    if gs not in seen:
                        rec.append(int(gs))
                        seen.add(int(gs))
        rankings[int(u)] = rec
    return rankings, n_pad
