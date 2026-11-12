# DECISIONS.md

Session-by-session record of technical and process decisions for the group project.
Mark team-joint decisions as **joint**; personal decisions as **<name>**.

## Session 1 — Define

### Data

- **Joint** — Use **FAR-Trans** (Glasgow, `10.5525/gla.researchdata.1658`, CC BY 4.0) as the
  shared data source, per the common project scope.
- **Joint** — Download the raw archive into `data/raw/` and keep it **git-ignored**; never
  commit raw data.
- **Joint** — `questionnaires.csv` is treated as a plain-text file (MiFiD questions), not CSV,
  matching the dataset README.

### Environment

- **Joint** — Isolated `venv` inside `group_project/.venv` (Anaconda Python 3.13); do not
  install packages into the base environment.
- **Joint** — `pyproject.toml` with `requires-python = ">=3.11"`; Windows paths via `pathlib`,
  README commands in PowerShell.

### Scope / protocol (pending)

- **Joint** — Exact item scope, evaluated users, time cut and transaction-label rules are
  **pending** (instructor, before Session 2). Nothing here fixes them; assumptions are marked
  as assumptions.

### Open questions

- Which track (Encoder / Ranker / Hybrid) each member adopts — not yet fixed.
- Baseline choice (classical recommender) — not yet fixed.
