# Study plan (Session 1, §3–§4 of the report) — Ranker track, **planned**

Author: Бабикова Д. (drafted with Claude, 2026-11-13). Nothing below has been run except the data
checks in `eda.md`, `candidate_recall.md`, `ranker_example.md`, `verify_example.md`. All numbers from
those files are on a **provisional** split; the official item scope, cut, users and label rule are pending.

## RQ (fixed for Session 1)

> We test whether zero-shot LLM reranking of a fixed top-20 candidate list improves nDCG@10 over a
> LightGBM LambdaMART reranker with the same information and the same candidates, for the common
> FAR-Trans evaluation users.

## 1. Data and users

| Item | Plan | Timestamp / leakage rule |
|---|---|---|
| Prediction time t | official cut (pending); provisional t = 2022-05-31 | — |
| Labels | Buy in (t, t + 6 m]; label rule A (new pairs) vs B (all buys) **pending instructor** | label window only |
| History | `transactions` rows < t: ISIN, type, date; reconstructed buys (transactionID < 0) kept as history, flagged | all < t |
| Profile | `customer_information` latest snapshot ≤ t: customerType, riskLevel, investmentCapacity + predicted flag; questionnaire date only if ≤ t | snapshot ≤ t |
| Assets | `asset_information`: name, category, sub-category, sector, industry, market (∅ kept as "unknown") | static |
| Eligible universe | assets priced over the whole test window (FAR-Trans protocol) — **protocol filter, same for all methods, never a feature** | uses future prices |
| Excluded | `limit_prices.profitability`, any price or transaction after t | future |
| Users | common evaluation users (pending). Provisional: 1,132 (A) / 2,590 (B) | — |
| Subgroups (pre-specified) | distinct assets before t: 1 / 2–4 / ≥ 5; riskLevel observed vs predicted | — |

## 2. Methods

**Candidate generator (fixed, shared by all arms):** ItemKNN (cosine on user × asset buy matrix before t),
top-20, owned assets excluded under rule A. Provisional Recall@20 = 0.625, own-order nDCG@10 = 0.290 (A).
LightGCN may replace ItemKNN in Session 2 if it gives higher Recall@20; the generator is then frozen for
every arm.

| Arm | Role | What changes |
|---|---|---|
| R0 Popularity | reference | — |
| R1 ItemKNN order | reference (no reranking) | — |
| **C LambdaMART** | **main control** (existing classical learned ranker) | learns order from features |
| **L LLM reranker** | **intervention** | LLM orders the same 20 candidates from the same fields as text |

**Same information.** C and L receive the same fields (history, as-of profile, asset metadata, ItemKNN
score, popularity). C gets them as features: ItemKNN score/rank, 6-month popularity, share of the user's
history in the candidate's category / sub-category / sector / industry / market, riskLevel × category,
customerType, investmentCapacity, predicted flags, history length, and a character-n-gram similarity
between candidate name and history names (so C also "sees" names). L gets the same fields serialised as
text. The difference C vs L is therefore the method, not the information.

**C training without leakage.** Training instances from earlier cuts t_k (bi-weekly, as in FAR-Trans) whose
label window ends ≤ t (t_k + 6 m ≤ t); validation cut = t − 6 m. Same generator, same features.
Objective lambdarank, early stopping on validation nDCG@10; 5 seeds.

**L design (proposal, to confirm in Session 2).**
- Model: open-weights instruction model, primary candidate **Qwen3-8B** (Hugging Face, revision hash
  pinned and reported), run locally; no data leaves the machine. Fallback for budget: Qwen3-4B
  (FLARKO found 1.7–4B competitive on FAR-Trans).
- Zero-shot, frozen weights, temperature 0.
- Scoring: LlamaRec-style verbalizer — candidates labelled A…T, score = next-token logit of each label,
  one forward pass, no free-text generation → no hallucinated items, deterministic.
- Prompt: role + date "as of t" + profile + history (most recent first, ≤ 20 items) + 20 candidates with
  metadata + instruction "rank by likelihood the client buys next".
- Position bias (Hou et al.): candidates shown in 3 fixed random permutations (seeds 0–2); final score =
  mean label logit across permutations.

## 3. Fair comparison checklist

Same users · same t and label window · same eligible universe · same 20 candidates per user · same fields
· same metric code · same hardware for runtime. Only the ordering method differs between C and L.
Fixing the candidate set is used only to isolate reranking (allowed by the assignment).

## 4. Measures

| Measure | Definition |
|---|---|
| **nDCG@10 (primary)** | binary relevance, IDCG over min(|rel|, 10); mean over users; official unrounded value |
| Uncertainty | paired per-user differences L − C, 95 % bootstrap CI over users (analysis only, not a benchmark rule); C: mean ± sd over 5 seeds; L: deterministic per permutation, spread over the 3 permutations reported |
| Runtime | end-to-end recommendation-generation wall time per user, same machine; training time of C reported separately |
| Cost | GPU-hours; prompt tokens per user |
| Diagnostics | share of the 10 most popular assets in the Top-10 (popularity drift); nDCG by subgroup; Recall@20 ceiling |

nDCG@10 measures ranking accuracy against observed purchases; it is not a measure of investment quality,
suitability or profit, and no such claim will be made. No LLM judge is used; ground truth is purchases only.

## 5. Contribution check (ablations on L)

| Ablation | Isolates |
|---|---|
| L-noname: names → ISIN, metadata kept | semantic / world knowledge about assets (also the contamination probe, §7) |
| L-nohist: no purchase history | personalisation vs. the LLM's popularity prior |
| L-noprof: no profile | contribution of risk / capacity |
| L-knnorder: candidates in ItemKNN order, 1 pass | position bias and anchoring on the generator |
| C-nonamesim: LambdaMART without the name-similarity feature | whether names help even a classical model |

## 6. Budget (provisional sizes)

Prompt ≈ 0.8–1.2 k tokens. Main L run: 3 permutations × N users; each L ablation: 1–3 × N.
For N = 2,590 (worst case, rule B): ≈ 7.8 k forward passes for L + ≈ 10 k for ablations ≈ 18 k passes
≈ 20 M prompt tokens. One forward pass of an 8B model on a single modern GPU is well under a second
(LlamaRec reports < 1 s for a 7B verbaliser ranker), i.e. a few GPU-hours in total; to be measured on the
first 100 users in Session 2. LambdaMART: minutes on CPU.

## 7. Risks

| Risk | Mitigation |
|---|---|
| **Contamination / hindsight:** the LLM was pretrained after 2022 and may know what happened to Greek assets in H2 2022 | L-noname ablation; report the model's knowledge cutoff; treat a gain that vanishes without names as possibly hindsight, not skill |
| Dataset contamination (FAR-Trans public since 2024) | per-customer memorisation unlikely (pseudonymous IDs never in the prompt); stated as a threat |
| Popularity bias of LLM rankers | popularity-share diagnostic; R0 as reference |
| Low recall ceiling (Recall@20 ≈ 0.63) | report ceiling; reranker gains bounded by it |
| Noisy metadata (59 / 285 stocks without sector; Greek-language fund names; sub-category conflicts) | keep ∅ explicit; no manual cleaning that only L benefits from |
| Joint accounts (same transactionID on two customers) | counted per customer as in the data; noted as threat |
| Profile timestamps not strictly point-in-time | as-of join; questionnaire date dropped if > t |
| Privacy | data are pseudonymised (CC BY 4.0); customer IDs never sent to a model; local inference, no external API |
| Ethics | ranking accuracy ≠ suitable advice; no claims on returns or user acceptance |

## 8. Interpretation of outcomes

- **L > C:** an LLM adds ranking value beyond a classical learner with the same fields; check via
  L-noname whether it is knowledge of assets or hindsight.
- **L ≈ C:** with sparse histories and thin metadata the LLM brings no measurable gain — a useful negative
  result for an advisor team weighing GPU cost against a CPU model.
- **L < C:** zero-shot LLM ranking is harmed by popularity / position bias in this domain; motivates
  fine-tuning (Session 4) rather than prompting.

## 9. Feasibility across sessions

| Session | Work |
|---|---|
| 2 | Freeze common protocol; ItemKNN (or LightGCN) generator; evaluation code; LambdaMART C; minimal L on 100 users (runtime, prompt checks) |
| 3 | Full main comparison C vs L on all evaluation users; optional RAG variant (retrieve similar users' purchases into the prompt) only as an extension |
| 4 | Optional LoRA fine-tuning of the ranker on earlier cuts (LlamaRec-style) as an improved L |
| 5 | Ablations, subgroup and failure analysis, short-paper draft |
| 6 | Final official run, reproducible package, presentation |

Tools: Python, pandas, LightGBM, PyTorch + transformers (or vLLM), Git; GPU from Colab / Kaggle /
university, to be confirmed. All results produced by committed scripts.

## Open decisions

1. Label rule A / B — instructor.
2. Candidate generator ItemKNN vs LightGCN — Session 2, by Recall@20.
3. LLM and its size — Session 2, by runtime on 100 users and available GPU.
