# DECISIONS.md

Session-by-session record of technical and process decisions for the group project.

**Attribution rule:** `**Joint**` is used only for decisions actually taken at a
team meeting (Joint ставится только для решений, реально принятых на встрече
команды). All other decisions are marked `**individual (via OpenCode)**`.

## Session 1 — Define

### Data

- **individual (via OpenCode)** — Use **FAR-Trans** (Glasgow, `10.5525/gla.researchdata.1658`, CC BY 4.0) as the
  shared data source, per the common project scope.
- **individual (via OpenCode)** — Download the raw archive into `data/raw/` and keep it **git-ignored**; never
  commit raw data.
- **individual (via OpenCode)** — `questionnaires.csv` is treated as a plain-text file (MiFiD questions), not CSV,
  matching the dataset README.

### Environment

- **individual (via OpenCode)** — Isolated `venv` inside `group_project/.venv` (Anaconda Python 3.13); do not
  install packages into the base environment.
- **individual (via OpenCode)** — `pyproject.toml` with `requires-python = ">=3.11"`; Windows paths via `pathlib`,
  README commands in PowerShell.

### Scope / protocol (pending)

- **individual (via OpenCode)** — Exact item scope, evaluated users, time cut and transaction-label rules are
  **pending** (instructor, before Session 2). Nothing here fixes them; assumptions are marked
  as assumptions.

### Open questions

- Which track (Encoder / Ranker / Hybrid) each member adopts — not yet fixed.
- Baseline choice (classical recommender) — not yet fixed.
