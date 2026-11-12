"""Draft reranking prompt (Ranker track) + size estimate -> reports/prompt_example.md

Builds, for every evaluated user (variant A), the prompt an LLM reranker would receive:
the user's purchase history before the cut (most recent first, max 20 assets) and the EASE top-20
candidates in a shuffled order, each asset described only by fields known at prediction time
(name, category, sub-category, sector, industry). No customer identifiers enter the prompt.
Token count is estimated as characters / 4 (no tokenizer installed); treat as an order of magnitude.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import scores_all  # noqa: E402
from common import CUT, build_split, load, write  # noqa: E402

K_CAND, K_HIST = 20, 20


def describe(a: pd.Series) -> str:
    parts = [str(a.assetName), str(a.assetCategory).replace("MTF", "Mutual fund")]
    for f in ("assetSubCategory", "sector", "industry"):
        if pd.notna(a[f]):
            parts.append(str(a[f]))
    return " | ".join(parts)


def build_prompt(hist_items, cands, meta, rng):
    order = list(rng.permutation(len(cands)))
    letters = [chr(ord("A") + i) for i in range(len(cands))]
    h = "\n".join(f"- {describe(meta.loc[i])}" for i in hist_items)
    c = "\n".join(f"[{letters[k]}] {describe(meta.loc[cands[j]])}" for k, j in enumerate(order))
    return (
        "You are ranking financial assets for a retail investor of a European bank.\n"
        f"Assets the investor bought before {CUT:%Y-%m-%d} (most recent first):\n{h}\n\n"
        f"Candidate assets the investor has not bought yet:\n{c}\n\n"
        "Rank ALL candidates from most to least likely to be bought by this investor in the next "
        "6 months. Answer only with the letters in order, comma-separated."
    )


def main():
    split = build_split("A")
    users, items, S = scores_all(split)
    meta = load("asset_information").drop_duplicates("ISIN").set_index("ISIN")
    tr_items = split["tr_items"]
    train = split["train"].sort_values("timestamp")
    last = train.groupby(["customerID", "ISIN"]).timestamp.max().reset_index().sort_values("timestamp", ascending=False)
    hist_by_user = last.groupby("customerID").ISIN.agg(list)
    M = S["EASE"].copy()
    for r, u in enumerate(users):
        M[r, [j for j, it in enumerate(items) if it in tr_items[u]]] = -np.inf
    top = np.argsort(-M, axis=1)[:, :K_CAND]
    rng = np.random.default_rng(0)
    sizes, example = [], None
    for r, u in enumerate(users):
        p = build_prompt(hist_by_user[u][:K_HIST], [items[j] for j in top[r]], meta, rng)
        sizes.append(len(p))
        if example is None and 3 <= len(tr_items[u]) <= 5:
            example = p
    sizes = np.array(sizes) / 4
    calls = len(users)
    out = ["# Draft reranking prompt (Ranker track)\n",
           "Zero-shot listwise prompt, history = assets bought before the cut (most recent first, ≤ 20), "
           "candidates = EASE top-20 in shuffled order (position-bias control). Only fields known at "
           "prediction time; no customer identifiers. Draft for Session 2, wording not final.\n",
           "## Example (one evaluated user, 3–5 prior assets)\n", "```", example, "```\n",
           "## Size estimate (characters / 4)\n",
           f"Users (variant A): {calls:,}. Input tokens per prompt: mean ≈ {sizes.mean():.0f}, "
           f"p95 ≈ {np.percentile(sizes, 95):.0f}, max ≈ {sizes.max():.0f}. Output ≈ 60 tokens (20 letters).\n",
           f"One pass over all users ≈ {sizes.sum() / 1e6:.2f} M input tokens; with 3 shuffled bootstraps "
           f"≈ {3 * sizes.sum() / 1e6:.2f} M.\n"]
    write("prompt_example.md", "prompt_preview.py", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
