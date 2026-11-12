# Research question (draft v1, Ranker track)

Status: **planned** (nothing run yet except data checks). Author: Бабикова Д. (draft with Claude).
Relevance labelling (repeat purchases relevant or not) is **open**: see the bottom of this file.

## Primary RQ

> We test whether **zero-shot LLM reranking of a fixed top-20 candidate list** improves **nDCG@10**
> over **a classical learned reranker (LightGBM LambdaMART) that sees the same user and asset
> information and the same candidates** for **the common FAR-Trans evaluation users**.

| Element | Content |
|---|---|
| Intervention | Frozen, prompted LLM reorders the same top-20 candidates. Input: user's purchase history before t, as-of profile (customerType, riskLevel, investmentCapacity + predicted flag), asset metadata (name, category, sub-category, sector, industry, market) |
| Control (main) | LambdaMART on the same candidates with features built from the same fields (history overlap, category/sector match, popularity, profile match) |
| Reference baselines | Original candidate order (ItemKNN), Popularity |
| Outcome | nDCG@10 against future purchases (common protocol); runtime per user and API/GPU cost as secondary |
| Population | Common FAR-Trans evaluation users (pending); subgroups by number of distinct assets bought before t (1 / 2–4 / ≥5) |
| Track / state | Ranker / planned |

The candidate set is fixed only to isolate reranking (allowed by the assignment). Candidate generator:
ItemKNN (measured: Recall@20 ≈ 0.63, own-order nDCG@10 ≈ 0.29 on the provisional split, variant A);
LightGCN is an alternative once the Session-2 baseline exists.

## Why this comparator

"LLM vs original order" alone would mix two effects: *reranking with extra information* and *reranking
with an LLM*. LambdaMART on the same fields removes the information difference, so a gap between LLM and
LambdaMART is attributable to the method (the course's same-information rule).

## Ablations (what each isolates)

| Ablation | Isolates |
|---|---|
| Replace asset names/sector text with opaque IDs | Contribution of the LLM's semantic / world knowledge about assets |
| Remove purchase history from the prompt | Personalisation vs. a popularity prior inside the LLM |
| Shuffle candidate order (3 permutations, aggregate) | Position bias of the LLM ranker |
| Remove profile fields | Contribution of risk / capacity information |

## Labelling (open, needs a team / instructor decision)

| | A: new pairs only (FAR-Trans paper) | B: all future purchases |
|---|---|---|
| Owned assets in candidates | excluded | included |
| Evaluated users (provisional cut) | 1,132 | 2,590 |
| Best simple baseline nDCG@10 | ItemKNN 0.29 | Repeat-own + popularity 0.59 |
| Recall@20 of best generator | 0.63 (ItemKNN) | 0.82 (repeat + pop) |
| What the LLM must do | find a new asset (exploration) | decide repeat vs new |
