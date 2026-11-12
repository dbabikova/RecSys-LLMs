# FAR-Trans — Data Schema (Session 1)

**Source:** FAR-Trans, University of Glasgow — [researchdata.gla.ac.uk/1658](https://researchdata.gla.ac.uk/1658/)
**DOI:** 10.5525/gla.researchdata.1658 · **License:** CC BY 4.0
**Paper:** J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," arXiv preprint arXiv:2407.08692, 2024.
**Collection period:** 2018-01-02 … 2022-11-29/30 (per README: 2 Jan 2018 — 29 Nov 2022)
**Encoding / format:** UTF-8; CSV with a single header row; fields containing commas are quoted with `"`.
**Local archive:** `data/raw/FAR-Trans.zip` (9.8 MB) → extracted to `data/raw/FAR-Trans/` (git-ignored).

## File inventory

| # | File | Rows (excl. header) | Time range (min → max) | Description |
|---|------|--------------------:|------------------------|-------------|
| 1 | `customer_information.csv` | 32 468 | `lastQuestionnaireDate` 2000-01-01* → 2022-11-23; `timestamp` 2000-01-01* → 2022-11-30 | Customer profile snapshots (a customer appears multiple times as info is updated) |
| 2 | `asset_information.csv` | 836 | `timestamp` 2018-01-01 → 2022-10-31 | Asset metadata (an asset appears multiple times) |
| 3 | `markets.csv` | 38 | — (no timestamp) | Market / exchange reference data |
| 4 | `close_prices.csv` | 703 303 | `timestamp` 2018-01-01 → 2022-11-29 | Daily close prices per asset |
| 5 | `limit_prices.csv` | 807 | `minDate` 2018-01-01 → 2022-10-31; `maxDate` 2018-01-19 → 2022-11-29 | Per-asset price-series extremes + profitability |
| 6 | `transactions.csv` | 388 048 | `timestamp` 2018-01-02 → 2022-11-30 | Customer investment transactions (Buy / Sell) |
| 7 | `questionnaires.csv` | 177 lines | — (plain text, not CSV) | MiFiD risk-questionnaire questions (answers not provided) |

\* `2000-01-01` is a sentinel used for missing/not-available dates (see notes).

## Key counts

- Unique customers: **29 090** (in both `customer_information` and `transactions`).
- Unique assets (ISIN): **806** in `asset_information`, **807** in `close_prices`/`limit_prices`, **320** appear in `transactions`.
- `asset_information` categories: MTF (mutual fund) 318 · Stock 293 · Bond 225.
- `transactions` types: Buy 228 913 · Sell 159 135.
- `transactions` channels: Internet Banking 250 805 · Branch 71 266 · Phone Banking 65 977.

---

## Column dictionary

### 1. `customer_information.csv` (6 columns)

| Column | Type | Meaning |
|---|---|---|
| `customerID` | str | Customer identifier (anonymised hash) |
| `customerType` | str | `Mass`, `Premium`, `Professional`, `Legal entity`, `Inactive` |
| `riskLevel` | str | `Conservative`, `Income`, `Balanced`, `Aggressive`, `Predicted_*` (algorithm-estimated), `Not_Available` |
| `investmentCapacity` | str | `CAP_LT_30K`, `CAP_30K_80K`, `CAP_80K_300K`, `CAP_GT300K`, `Predicted_*`, `Not_Available` |
| `lastQuestionnaireDate` | str (date) | Date the risk questionnaire was last completed |
| `timestamp` | str (date) | Timestamp of last update |

### 2. `asset_information.csv` (9 columns)

| Column | Type | Meaning |
|---|---|---|
| `ISIN` | str | International Securities Identification Number |
| `assetName` | str | Full asset name |
| `assetShortName` | str | Short asset name |
| `assetCategory` | str | `Stock`, `Bond`, `MTF` (mutual fund) |
| `assetSubCategory` | str | Sub-category (depends on `assetCategory`) |
| `marketID` | str | Market where the asset is traded |
| `sector` | str | Economic sector (when available) |
| `industry` | str | Industry (when available) |
| `timestamp` | str (date) | Last update date |

### 3. `markets.csv` (8 columns)

| Column | Type | Meaning |
|---|---|---|
| `exchangeID` | str | Stock exchange identifier |
| `marketID` | str | Market identifier |
| `name` | str | Market name |
| `description` | str | Brief description |
| `country` | str | Country where the market is located |
| `tradingDays` | str | Comma-separated days the market is open (`Mon`…`Sun`) |
| `tradingHours` | str | Trading hours GMT, format `HH:MM-HH:MM` |
| `marketClass` | str | Type of market |

### 4. `close_prices.csv` (3 columns)

| Column | Type | Meaning |
|---|---|---|
| `ISIN` | str | Asset identifier |
| `timestamp` | str (date) | Trading day, `YYYY-MM-DD` |
| `closePrice` | float | Close price in euros |

### 5. `limit_prices.csv` (6 columns)

| Column | Type | Meaning |
|---|---|---|
| `ISIN` | str | Asset identifier |
| `minDate` | str (date) | First date with close-price info |
| `maxDate` | str (date) | Last date with close-price info |
| `priceMinDate` | float | Close price at `minDate` |
| `priceMaxDate` | float | Close price at `maxDate` |
| `profitability` | float | ROI between `minDate` and `maxDate` |

### 6. `transactions.csv` (9 columns)

| Column | Type | Meaning |
|---|---|---|
| `customerID` | str | Customer identifier |
| `ISIN` | str | Asset identifier |
| `transactionID` | int | Transaction identifier |
| `transactionType` | str | `Buy` / `Sell` |
| `timestamp` | str (date) | Transaction date, `YYYY-MM-DD` |
| `totalValue` | float | Total monetary value (euros) |
| `units` | float | Number of shares/units traded |
| `channel` | str | `Internet Banking`, `Phone Banking`, `Branch` |
| `marketID` | str | Market over which the transaction was performed |

### 7. `questionnaires.csv` (plain text)

Not a CSV. Contains the MiFiD risk-questionnaire questions with enumerated answers,
organised in sections (e.g. *Demographics*, *Investment Knowledge and Experience*,
*Financial Independence and Liquidity*). Answers are **not** provided.

---

## Observed vs. predicted fields (leakage check)

| Field | Status | Note |
|---|---|---|
| `customerID`, `customerType`, `riskLevel`, `investmentCapacity` | observed | From the bank's records (profile attributes). |
| `riskLevel = Predicted_*`, `investmentCapacity = Predicted_*` | predicted/estimated | Produced by the institution's algorithm, not self-reported. |
| `lastQuestionnaireDate`, `timestamp` | observed | Dates of record/update. |
| `assetName`, `assetCategory`, `assetSubCategory`, `marketID`, `sector`, `industry` | observed | Asset metadata as of `timestamp`. |
| `closePrice` | observed | Market price on a trading day. |
| `profitability`, `priceMinDate`, `priceMaxDate` | derived (historical) | Computed over the asset's full series; must not be used before the corresponding dates. |
| `transactionType`, `totalValue`, `units`, `channel`, `transactionID` | observed | Transaction facts. |

**Leakage note:** a value is usable at prediction time only if its source timestamp is
strictly before the prediction cut. `customer_information.timestamp` and
`asset_information.timestamp` are **update** timestamps (versioning), while
`transactions.timestamp` and `close_prices.timestamp` are event timestamps. For a future
purchase to be predicted, only transactions/prices up to the time cut may be used as
input; the target is the user's *future* purchases (nDCG@10), never the purchase itself.

## Assumptions (pending official protocol)

- Exact item scope, evaluated users, time cut and transaction-label rules are **pending**
  (fixed by the instructor before Session 2). The stock/fund/bond scope here is descriptive,
  not an official task scope.
- `2000-01-01` is treated as a "missing date" sentinel (7 275 `lastQuestionnaireDate`,
  130 `timestamp` values in `customer_information`).
