Submission README for HW6 (readme.md is the original assignment brief and is left unchanged).

# HW6 — Personalized PageRank on MovieLens 100K

Bipartite user–movie graph with Global PageRank and Personalized PageRank (PPR)
computed in TensorFlow.js **in the browser** (power iteration on WebGL).
Datasets and splits come from the assignment starter (`data.js`, chronological
per-user split).

## How to run

```bash
cd week6
python -m http.server 8000
# open http://localhost:8000/index.html
```

Button order:

1. **Build graph** — builds the bipartite graph from the training split.
2. **Run tests** — self-checks (12/12 PASS in the final run).
3. **Run alpha sweep** — validation Recall@10 for alpha in {0.50, 0.70, 0.85, 0.95}.
   Selected: **alpha 0.50** (0.1145; ties would go to the smaller alpha).
4. Set the **alpha slider** to the selected value (0.50).
5. **Evaluate** — four-method comparison on test + convergence panel.
6. **Run two-user demo** — users come from the two selects (`#user-select`,
   `#seed-user-select`); each header shows whether the user was taken from the
   select or from the fallback.
7. **Recommend Top-10** — per-user lists for the chosen method.
8. **Export** — `verification.json`, `results.csv`, `recommendations.csv`
   (files in `week6/results/` were produced this way).

Reloading the page **discards all computed results** (graph, sweeps, MF model);
repeat from step 1. The first PPR batch at a given alpha costs ~11 s; later
reads are cached.

## Dependencies (actually used)

- TensorFlow.js **4.22.0** (CDN, `index.html`)
- d3 **7.9.0** (vendored in `vendor/d3.min.js`)
- Backend: **webgl** (see `verification.json`)

## Changes vs. the starter

- Implemented all 7 `TODO(hw6)` stubs: `buildGraph`, `rowNormalize`, `pageRank`,
  `personalizedPageRank`, `movieScores`, `recommendTopK`, `recallAtK`.
- Batched PPR: `personalizedPageRankBatch` + per-alpha all-users cache in
  `hw6RunPPR` (first run per alpha computes all 943 user columns in one pass).
- Adapter fix: PPR seeds with the **global node index** (`seedNode`), not the
  raw user id.
- Transition validation accepts `Uint8Array`/`Int32Array` `danglingMask`.
- Float32 fixes: restart vector converted to `Float32Array`; rank-1 vectors
  reshaped to columns for `tf.matMul`.
- MF L2 penalty computed over **batch rows** (`reg / batchSize`), not over the
  full factor matrices; config now 20 epochs / lr 0.01 / reg 0.05 / batch 512
  (the starter's full-matrix penalty collapsed all factors to zero: RMSE equal
  to the global-mean baseline 1.1830/1.2289; after the fix: val 0.9779,
  test 1.0181, checkpoint epoch 4).
- Two-user demo honours the user selects (was hard-coded to cohort[0]/[1]) and
  shows where each user came from.
- `verification.json` also exports PageRank/PPR convergence records and the MF
  configuration; `results.csv` gained `mf_epochs`/`mf_checkpoint` columns.

## Final results (from `results/`)

Test Recall@10, shared cohort of **861** users, 13 relevant ids absent from the
training pool, 82 users without eligible relevant movies:

| Method | Alpha | Recall@10 | Mean popularity of Top-10 |
|---|---|---|---|
| popularity | — | 0.0760 | 272.00 |
| pagerank | 0.50 | 0.0797 | 257.82 |
| ppr | 0.50 | 0.1122 | 244.21 |
| mf | — | 0.0312 | 94.31 |

Alpha sweep (validation, cohort 846): 0.50 → 0.1145, 0.70 → 0.1048,
0.85 → 0.0900, 0.95 → 0.0738. MF uses a different validation objective
(RMSE checkpoint), disclosed per the assignment.

## Known limitations

- **alpha 0.95 does not fully converge in float32**: 75 of 943 user columns hit
  the 500-iteration cap with L1 residual frozen at 1.0–2.2e-6 (machine-noise
  plateau after ~300 iterations). Top-10 lists are identical at 300 vs 500
  iterations, so recommendations are unaffected; the tolerance (1e-6) was not
  weakened.
- The split is chronological **within each user**, not a global timestamp cut.
- PPR/PageRank scores are graph-walk probabilities — not ratings and not
  purchase probabilities. MF scores are unclipped `mu + p·q` predictions and can
  fall outside the 1–5 scale; MF optimises RMSE, not ranking, which explains
  its wide catalog coverage (36% of 1,611 movies) with low Recall@10.

## Files in `results/`

- `results.csv` — retrieval metrics per method (test comparison, validation
  alpha sweep, convergence records), cohort sizes, exclusion counts, and MF
  hyperparameters (`mf_epochs`, `mf_checkpoint`).
- `verification.json` — self-check results (12/12 PASS), backend, PageRank/PPR
  convergence records, and the full MF configuration/checkpoint.
- `recommendations.csv` — Top-10 lists exported from the page for users 1 and 7:
  all four methods at alpha 0.50 plus PPR at 0.85 (the two-user-demo settings).
