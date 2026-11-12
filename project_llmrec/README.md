# LLM4Rec project (FAR-Trans)

Course project, LLM4Rec, HSE University.

**Shared task:** for each evaluated user, a Top-10 list of financial assets, scored by nDCG@10
against the assets the user buys in a future window. Item scope, time cut and evaluated users
are **pending** (fixed by the instructor before Session 2); every split used here is provisional.

**Data:** FAR-Trans, University of Glasgow, <https://researchdata.gla.ac.uk/1658/>, CC BY 4.0.
J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An Investment Dataset for Financial
Asset Recommendation," arXiv:2407.08692, 2024. Raw data is never committed.

## Layout

```
src/llmrec/data.py            paths, constants, loaders (point-in-time customer availability)
scripts/download_data.py      fetch + SHA-256 check + extract -> reports/session1/data_manifest.md
scripts/inspect_schema.py     -> reports/session1/schema_stats.md
scripts/make_data_example.py  -> reports/session1/data_example.md (one record + leakage checks)
scripts/eda.py                -> reports/session1/eda.md, figures/, tables/
DECISIONS.md                  decision log with attribution
```

Every number in `reports/` is produced by a committed script; generated files carry an
`auto-generated` header and are not edited by hand.

## Reproduce (PowerShell or bash)

```
python -m venv .venv
.venv\Scripts\activate            # bash: source .venv/bin/activate
pip install -e .[dev]
python scripts/download_data.py   # or: --zip path\to\FAR-Trans.zip
python scripts/inspect_schema.py
python scripts/make_data_example.py
python scripts/eda.py
```
