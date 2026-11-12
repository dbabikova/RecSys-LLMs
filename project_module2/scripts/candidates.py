"""Shared candidate-generation setup on the provisional split (Popularity, ItemKNN)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import CUT, TEST_END


@dataclass
class Setup:
    universe: list
    tr_items: pd.Series      # customerID -> set of ISIN bought before CUT (incl. reconstructed)
    te_items: pd.Series      # customerID -> set of ISIN bought (real) in test window
    pop_rank: list
    sim: np.ndarray
    uidx: dict
    popv: np.ndarray
    train: pd.DataFrame

    def knn(self, u, exclude_owned=True):
        own = [self.uidx[i] for i in self.tr_items[u] if i in self.uidx]
        s = self.sim[own].sum(axis=0) if own else np.zeros(len(self.universe))
        s = s + 1e-3 * self.popv
        order = np.argsort(-s)
        out = [(self.universe[j], float(s[j])) for j in order]
        return [(i, sc) for i, sc in out if not (exclude_owned and i in self.tr_items[u])]


def prepare(tr: pd.DataFrame, cp: pd.DataFrame) -> Setup:
    buys = tr[tr.transactionType == "Buy"]
    train = buys[buys.timestamp < CUT]
    test = buys[(buys.transactionID > 0) & (buys.timestamp >= CUT) & (buys.timestamp <= TEST_END)]
    # protocol eligibility (FAR-Trans paper): priced over the whole test window. NOT a model feature.
    priced = cp.groupby("ISIN").timestamp.agg(["min", "max"])
    universe = sorted(priced[(priced["min"] <= CUT) & (priced["max"] >= TEST_END - pd.Timedelta(days=3))].index)
    uidx = {a: i for i, a in enumerate(universe)}
    recent = train[train.timestamp >= CUT - pd.DateOffset(months=6)].drop_duplicates(["customerID", "ISIN"])
    pop = recent.ISIN.value_counts().reindex(universe).fillna(0)
    pairs = train[train.ISIN.isin(uidx)].drop_duplicates(["customerID", "ISIN"])
    users = {u: i for i, u in enumerate(pairs.customerID.unique())}
    X = np.zeros((len(users), len(universe)), dtype=np.float32)
    X[pairs.customerID.map(users).values, pairs.ISIN.map(uidx).values] = 1.0
    co = X.T @ X
    norm = np.sqrt(np.diag(co)) + 1e-9
    sim = co / norm[:, None] / norm[None, :]
    np.fill_diagonal(sim, 0.0)
    return Setup(universe, train.groupby("customerID").ISIN.agg(set), test.groupby("customerID").ISIN.agg(set),
                 pop.sort_values(ascending=False).index.tolist(), sim, uidx,
                 pop.values / (pop.values.max() + 1e-9), train)
