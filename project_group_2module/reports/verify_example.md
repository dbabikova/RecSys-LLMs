# Независимая проверка примера и инструкция ручной проверки

> Сгенерировано отдельным `scripts/verify_example.py`, который не импортирует
> starter или наши модули и сам читает исходные CSV. Модели не запускались.
> Expected JSON — только данные для сравнения, не исполняемый источник алгоритма.

**PASS: 46 / 46 независимых проверок совпали.**
Ошибки приводят к ненулевому exit code; прежний success report сначала
сбрасывается, чтобы не оставить ложный PASS после неудачного повторного запуска.
Для арифметики используется tolerance; упорядоченный CBF20 и позиции проверены точно.

| № | Совпавшая проверка |
| --- | --- |
| 1 | window end from cutoff + 30 days |
| 2 | SHA-256 transactions.csv |
| 3 | SHA-256 asset_information.csv |
| 4 | SHA-256 customer_information.csv |
| 5 | SHA-256 close_prices.csv |
| 6 | eligible universe |
| 7 | deterministic demo cohort |
| 8 | preferred rank-11–20 target pairs |
| 9 | preferred rank-11–20 users |
| 10 | target pairs inside top-20 |
| 11 | independently selected user and asset |
| 12 | selected user's demo ordinal |
| 13 | verbatim raw record and CSV record ordinal |
| 14 | ALL source rows sharing transactionID (not blindly deduplicated) |
| 15 | retained dated history raw rows |
| 16 | history Buy counts |
| 17 | omitted pre-cutoff rows by reason |
| 18 | all eligible future Buy raw rows |
| 19 | full distinct future truth |
| 20 | all profile snapshots |
| 21 | as-of profile + original record ordinal |
| 22 | all metadata snapshots for asset |
| 23 | as-of metadata + first-on-tie ordinal |
| 24 | CBF top-20 ordered ISINs |
| 25 | selected asset CBF position |
| 26 | all top-20 raw metadata |
| 27 | all top-20 categorical tokens |
| 28 | top-20 CBF scores within tolerance |
| 29 | full Latest buy ranking including popularity fill |
| 30 | metadata for displayed CBF/Latest buy lists |
| 31 | selected asset Latest buy position |
| 32 | repeat/new classification from pre-cutoff Buy |
| 33 | CBF nDCG@10 |
| 34 | Latest buy nDCG@10 |
| 35 | all retained history dates strictly before cut |
| 36 | record is in label window, Buy, nonnegative ID, eligible |
| 37 | planned input JSON matches stored comparison artifact |
| 38 | planned candidate payload matches source-verified candidates |
| 39 | planned input history independently reconstructed |
| 40 | no Buy timestamps silently added to stock input |
| 41 | late questionnaire input masking for BOTH methods |
| 42 | planned profile observed/estimated/missing markers |
| 43 | planned profile timestamp and placeholder flag |
| 44 | planned cutoff and nonidentifying user alias |
| 45 | no truth/target marker in planned input |
| 46 | public reports contain no unmasked source customer IDs |

## Короткая инструкция для Дарьи — пока НЕ выполнена вручную

1. Открой локальный `data/FAR-Trans/transactions.csv` и найди **transactionID
   `9049523`**, одновременно проверь ISIN `GRS003003035`,
   дату `2021-06-08`, тип `Buy`,
   units `370.0`, totalValue `996.41`.
   Нужный CSV data-record: **85881** (без заголовка).
   Если ID встречается несколько раз — строк с таким ID **1**;
   не удаляй разные customerID как дубликат. Номер record выбирает конкретную строку.
2. Скопируй customerID из найденной строки **только локально**. Не вставляй его
   в отчёт, чат или Git. По этому значению отфильтруй `customer_information.csv`.
   Нужный as-of CSV-record — **14770**,
   timestamp `2021-03-10`, lastQuestionnaireDate
   `2018-12-10`, customerType `Professional`,
   riskLevel `Aggressive`, investmentCapacity `CAP_80K_300K`.
   Выбирается максимальный timestamp строго до `2021-06-01`, не последний
   snapshot всего файла; при равенстве — первая строка.
3. Для этого же клиента отфильтруй retained Buy до `2021-06-01`:
   неотрицательные ID, exact full duplicates удалены, ISIN eligible.
   Должно быть **24** строк / **5** разных активов.
   Сравни даты с raw history в `ranker_example.md`. Эти даты — только audit:
   в stock history context их нет.
4. Будущие метки: Buy в `[2021-06-01, 2021-07-01)` по тем же cleaning/
   eligibility правилам, повторы разрешены, повторные строки одного ISIN
   не увеличивают gain. Должен получиться T = `["GRS003003035"]`,
   **1** distinct-меток. Не добавляй target в candidates вручную.
5. Сверь Top-10 списки и CBF20 с `ranker_example.md`: позиция выбранного актива
   **CBF 10**, **Latest buy 5**. Для CBF20 действуют
   cosine по category/subcategory/sector/industry/marketID, веса = buy_count,
   tie-break ISIN. Это не роль названия или цены исполнения.
6. Посчитай независимо от готового nDCG столбца:

   `DCG@10 = Σ[r=1..10] 1[item_r ∈ T] / log2(r+1)`

   `IDCG@10 = Σ[r=1..min(10, |T|)] 1 / log2(r+1)`

   `nDCG@10 = DCG@10 / IDCG@10`.

| Метод | Relevant ranks | DCG@10 | IDCG@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| CBF | 10 | 0.289064826318 | 1.000000000000 | 0.289064826318 |
| Latest buy | 5 | 0.386852807235 | 1.000000000000 | 0.386852807235 |

В этом примере IDCG = 1.000000000000, поскольку distinct-метка одна. Поэтому CBF: `1/log2(11) = 0.289064826318`, Latest buy: `1/log2(6) = 0.386852807235`.

Учитывай все T, не только найденные кандидаты. При пустом T starter даёт ноль,
но у этого иллюстративного пользователя T непусто. Результат одного пользователя
не заменяет primary mean по всем выбранным evaluation users.

После собственной проверки отдельно запиши, что именно ты проверила и совпало ли.
**Сейчас выполнена только автоматическая проверка AI-ассистентом; ручная проверка
Дарьи не выполнена и не заявляется выполненной.**

## Запись для сверки (маска — единственное изменение значения)

```text
source: FAR-Trans public release, transactions.csv
record id: transactionID=9049523; CSV data-record=85881
fields: customerID, ISIN, transactionID, transactionType, timestamp, totalValue, units, channel, marketID
values: [MASKED], GRS003003035, 9049523, Buy, 2021-06-08, 996.41, 370.0, Phone Banking, XATH
window: 2021-06-01 <= timestamp < 2021-07-01; repeat allowed
```
