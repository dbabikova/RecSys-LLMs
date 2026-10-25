# midterm — sweet.tv TV program recommender (Kaggle)

Recommender for the Kaggle competition
[sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender).
The view log covers weeks 11–30 of 2020. For each of the 1260 users in `submission.csv`, predict the
5 TV shows (`tv_show_id`) they will watch during weeks 31–42. The metric is **MAP@5**.

## Result

The final submission is **`submissions/final_submission_h1_oc.csv`**. It is copied to `midterm/submission.csv`,
md5 `ae2f86a9e8b0bf488fade34359eb0d51`.

| | Kaggle private | Kaggle public |
|---|---|---|
| **h1** (final) | **0.41508** | **0.42836** |

h1 is a heuristic with no trained parameters. It has two parts:

- **Recency-weighted completion:** `p(u,s) = Σ pct · 0.5^(age / 4 weeks)` over the user's past airings of the show.
- **Time-slot fit:** each window airing is matched against the user's (channel, hour-of-week) viewing profile.

The score is `p · (slot + 1e-4)^0.5`. Candidates are the user's own shows that air in the window.
Lists shorter than 5 are padded with popular shows.

A LightGBM ranker built on the same features (`11_gbm.py`) beats h1 locally (+0.024 D80) but not on Kaggle.
See `results/local_vs_kaggle.md` for why.

## Environment

- Python 3.13.9 (Windows 10, PowerShell 5.1)
- numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, pyarrow (see `requirements.txt`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Data

The raw CSVs are **not** in the repository. Point `SWEETTV_DATA_DIR` at the folder that contains them:

```powershell
$env:SWEETTV_DATA_DIR = "C:\path\to\sweettv-tv-program-recommender"
```

| file | content |
|---|---|
| `dataset11-30.csv` | view log (user, channel, start, end), weeks 11–30 |
| `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` | TV grids (channel, start, duration, tv_show_id, metadata) |
| `submission.csv` | submission template, 1260 users |

The default path is set in `src/config.py`. Scripts only read the data and never modify it.

## Layout

```
midterm/
├── src/          config.py, join.py (log x grid join), evaluate.py (GT, MAP@5), mf_als.py, cbf.py
├── scripts/      numbered pipeline steps 00 -> 11
├── results/      summaries (.md) and metric tables (.csv) written by the scripts
├── submissions/  all generated submission files
├── submission.csv  final file (copy of submissions/final_submission_h1_oc.csv)
└── cache/        intermediate parquet files (git-ignored)
```

## Pipeline (run from `midterm/`)

Local validation uses history before 2020-05-18 and the window 2020-05-18 … 2020-07-26.
GT-A/B/C are the "watched" ground-truth variants from stage 3. GT-D80 is defined in the next section.

| script | what it does | writes |
|---|---|---|
| `00_check_files.py` | checks that the raw files exist; prints sizes and first lines | stdout |
| `01_audit.py` | audits the log and grids (time ranges, weeks, users, ids, gaps) | `results/audit_summary.md` |
| `02_join.py` | joins the log to the grid (overlap per airing, 4 h cap), builds train/val interactions | `cache/*.parquet`, `results/join_summary.md` |
| `03_baselines.py` | global / personal baselines on GT-A/B/C, hand-checked AP | `results/baselines.csv`, `baselines_summary.md`, `handcheck_ap.md` |
| `04_mf.py`, `04b_mf.py` | implicit ALS (MF) and MF+popularity grid search | `results/mf.csv`, `mf_configs.csv`, `mf_summary.md` |
| `05_cbf.py` | content-based filtering on category/genre/year/people | `results/cbf.csv`, `cbf_summary.md`, `all_methods.csv` |
| `06_cbf_title.py`, `06b_final.py` | title-based CBF, rank-fusion hybrid, config selection | `results/cbf_summary.md`, `final_config.json`, `all_methods.csv` |
| `07_submit.py` | builds a submission (hybrid; parameters below); 6 format checks + md5 | `submissions/<--out>`, `global.csv`, `personal.csv`, `mf_pop.csv`, `cbf_title.csv`, `results/submission_summary.md` |
| `07b_all_methods.py` | recomputes all methods on GT-A/B/C into one table | `results/all_methods.csv`, `all_methods_summary.md` |
| `08_two_users.py`, `08b_switching.py` | case study of two users; switching light users to global | `results/two_users.{md,csv}`, `switching_summary.md` |
| `09a_diagnostics.py` … `09i_tune.py` | diagnostics and tuning: decay, airing boost, PPR, time-slot boost | `results/diagnostics_9a.md`, `diagnostics_9b.md`, `tune_9d.md` … `tune_9i.md` |
| `10_h1.py` | GT-D80 calibration (global / personal / v3 / v5) and h1 on val; `--submit` builds the h1 submission | `results/tune_10.md`, `local_vs_kaggle.md`, `all_methods.csv`; `submissions/final_submission_h1_oc.csv` |
| `11_gbm.py` | LightGBM (lambdarank + binary, 3 seeds) on h1 features; 4-fold and temporal validation; submission | `results/tune_11.md`, `gbm_importance.md`, `local_vs_kaggle.md`, `all_methods.csv`; `submissions/final_submission_gbm_oc.csv` |

### `07_submit.py` parameters

The defaults give v3. Base score per (user, show):
`n_days · 0.5^((REF − last_seen) / hl weeks) · (1 + gamma · log1p(n_airings)) · (1 + delta · slot)`.
It is combined with MF+pop (`--wmf`) and PPR (`--ppr`) by reciprocal rank fusion, `1 / (60 + rank)`.
Users with fewer than 5 shows or fewer than 2 days of history get the global top-5.

| flag | default | meaning |
|---|---|---|
| `--hl` | 8 | recency half-life in weeks (0 = no decay) |
| `--gamma` | 0.3 | boost for the number of airings in the window |
| `--wmf` | 0 | weight of MF+pop in the fusion |
| `--delta` | 0 | (channel, hour-of-week) slot-profile boost |
| `--ppr` | 0 | weight of personalized PageRank (bipartite, log1p edges, alpha 0.5) in the fusion |
| `--out` | `final_submission.csv` | output file name in `submissions/` |

```powershell
python scripts/07_submit.py --out final_submission_v3.csv                         # v3
python scripts/07_submit.py --delta 300 --ppr 0.25 --out final_submission_v5.csv  # v5
python scripts/10_h1.py --submit final_submission_h1_oc.csv                       # h1 (final)
python scripts/11_gbm.py                                                          # GBM -> final_submission_gbm_oc.csv
```

Each `07_submit.py` run also rewrites `global.csv`, `personal.csv`, `mf_pop.csv` and `cbf_title.csv`.
These do not depend on the flags. `10_h1.py --submit` was checked to reproduce md5 `ae2f86a9…` byte for byte.

## Submissions

md5 values are computed from the files in `submissions/`.

| file | md5 | Kaggle private | Kaggle public |
|---|---|---|---|
| `global.csv` | `0c72c2a9769e362acd93e5281e140598` | 0.27809 | 0.28234 |
| `personal.csv` | `0d7a5425596cb17f8f083ca9b23eb38f` | 0.33186 | 0.35650 |
| `mf_pop.csv` | `0d60c1c2abb07e686cf4f9b74c0b8d15` | 0.27073 | 0.28183 |
| `final_submission_v2.csv` | `845dc87cc6406168e46ad6d156e166f7` | 0.33584 | 0.35908 |
| `final_submission_v3.csv` | `7b0bbc6a6907ad841fd8f7e5468a022d` | 0.36222 | 0.37878 |
| `final_submission_v3b.csv` | `e72f85cdcd71410343373ca75b67a799` | 0.35621 | 0.37142 |
| `final_submission_v4b.csv` | `5abe2d3df39b1710df024a9d6f5abeea` | 0.37839 | 0.39790 |
| `final_submission_ppr.csv` | `f5aad4fe7868b44856e961571e6c5ad4` | 0.38761 | 0.40864 |
| `final_submission_v5.csv` | `9a4332d5632ac5c9c192043d12421b33` | 0.39434 | 0.41300 |
| `final_submission_h1.csv` | `f00257aba4611c24fbb5086e713f2692` | 0.41508 | 0.42836 |
| `final_submission_gbm.csv` | `5c9bb3e246f783f25cdb1cc0d6dfd06b` | 0.41306 | 0.42168 |
| **`final_submission_h1_oc.csv` (final)** | `ae2f86a9e8b0bf488fade34359eb0d51` | **0.41508** | **0.42836** |
| `final_submission_gbm_oc.csv` | `f8f396efc2bf37c573b776ddf3b04aab` | 0.41349 | 0.42086 |

Other files in `submissions/` that are not in the table:

- `final_submission.csv` is identical to v3.
- `final_submission_v1.csv`, `final_submission_v4.csv`, `final_submission_v6_d1000.csv`, `final_submission_v6_d3000.csv` and `cbf_title.csv` are intermediate variants with no Kaggle scores recorded here.

`final_submission_h1_oc.csv` and `final_submission_h1.csv` differ only in the order of the 5 shows for 4 users, and both score the same on Kaggle.

## Local evaluation (GT-D80)

The local metric that tracks Kaggle best is **D80**:

- Take the user's airings in the window with completion `pct = overlap / duration ≥ 0.8`.
- GT = the top 5 shows by the number of such airings.
- Users with no such airings are excluded. AP@5 is divided by `min(5, |GT|)`.

On val, D80 ranks global < personal < v3 < v5 < h1 in the same order as Kaggle private (Spearman 1.0).
It overstates the GBM gain. To evaluate a config:

1. Build its rankings for the val snapshot: history before 2020-05-18, candidates airing in [2020-05-18, 2020-07-27).
2. Score it with `build_gt_d80` and `E.ap_at_k` as in `scripts/10_h1.py`. The calibration block of `10_h1.py` is the template.
3. Compare with the reference values in [`results/local_vs_kaggle.md`](results/local_vs_kaggle.md): global 0.273, personal 0.331, v3 0.350, v5 0.376, h1 0.401, GBM 0.425.

`results/all_methods.csv` has every method on GT-A/B/C and, in the `MAP5_D80` column, on D80.
