# group_project — FAR-Trans (LLM4Rec)

Team project for the LLM4Rec course (HSE). Shared task: for each evaluated user,
produce a **Top-10 ranked list of financial assets** scored by **nDCG@10** against
the assets the user actually buys in the future evaluation window.

Dataset: **FAR-Trans** (University of Glasgow) — <https://researchdata.gla.ac.uk/1658/>,
DOI `10.5525/gla.researchdata.1658`, CC BY 4.0. Paper: [arXiv:2407.08692](https://arxiv.org/abs/2407.08692).

## Status (Session 1 — Define)

Data downloaded and schema parsed. See `reports/session1/schema.md`.
No model code yet; item scope / evaluated users / time cut are **pending** (instructor, before Session 2).

## Repository layout

```
group_project/
├── data/
│   ├── raw/          # FAR-Trans archive + extracted CSVs (git-ignored)
│   └── processed/    # derived artifacts (git-ignored)
├── reports/
│   └── session1/     # Session 1 reports (schema.md, ...)
├── scripts/          # runnable scripts (inspect_schema.py, ...)
├── src/fartrans/     # project package
├── pyproject.toml
├── DECISIONS.md
└── README.md
```

## Environment (Windows / PowerShell)

Requires Python ≥ 3.11 (tested with Anaconda Python 3.13.9).

```powershell
# 1. create an isolated venv (do NOT install into base)
python -m venv .venv

# 2. activate
.\.venv\Scripts\Activate.ps1

# 3. install the project in editable mode
python -m pip install -e .

# 4. inspect the schema (reads data/raw/FAR-Trans/*.csv)
python scripts\inspect_schema.py
```

## Getting the data

Raw data is **not committed**. Download and extract manually:

```powershell
New-Item -ItemType Directory -Path data\raw -Force | Out-Null
Invoke-WebRequest -Uri "https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip" -OutFile data\raw\FAR-Trans.zip
Expand-Archive -Path data\raw\FAR-Trans.zip -DestinationPath data\raw -Force
```

All paths in code use `pathlib` and are resolved relative to this directory.

## Notes

- `nDCG@10` measures ranking accuracy against observed purchases; it is **not** a
  measure of investment quality or profitability.
- Keep future information (purchases / prices after the time cut) out of model inputs.
