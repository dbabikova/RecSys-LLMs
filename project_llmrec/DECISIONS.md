# Decision log

Attribution labels: **Joint** only for decisions actually taken at a team meeting;
**individual (via Claude)** for working choices made while building this folder, open to team review.
Anything about the split stays **provisional** until the official protocol is published.

| # | Date | Decision | Status | Attribution |
|---|---|---|---|---|
| 1 | 2026-11-13 | Data integrity: the archive is accepted only if its SHA-256 equals `d0d6cfe0…cfb6`. | adopted | individual (via Claude) |
| 2 | 2026-11-13 | `2000-01-01` in customer dates is treated as unknown (sentinel). | adopted | individual (via Claude) |
| 3 | 2026-11-13 | A customer snapshot is available only from `max(timestamp, lastQuestionnaireDate)`; in 49.62% of comparable snapshots the questionnaire date is later than `timestamp`. | adopted | individual (via Claude) |
| 4 | 2026-11-13 | `limit_prices.profitability` is excluded from all inputs (computed over the full price series). | adopted | individual (via Claude) |
| 5 | 2026-11-13 | Exploratory cuts 2022-03-01 / 2022-06-01 / 2022-09-01, evaluated user = ≥1 Buy before and after the cut, relevant items = distinct ISINs bought after the cut. | provisional | individual (via Claude) |
| 6 | — | Whether repeat purchases count as relevant items. | open (protocol pending) | — |
| 7 | — | Treatment of purchases whose `transactionID` is shared by several customers (12.00% of Buys). | open (protocol pending) | — |
| 8 | — | Track (Encoder / Ranker / Hybrid) and research question. | open (team meeting) | — |
