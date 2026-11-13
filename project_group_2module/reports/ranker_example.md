# Реальный пример для planned Ranker RQ

> Сгенерировано `scripts/make_ranker_example.py` через импорт функций оригинального
> starter. Ни LLM, ни LambdaMART не запускались. Future-derived значения ниже —
> только метка и аудит примера, не входы будущих моделей.

Starter commit: `51337032503c56e3a28ef646a24aa016782e1bd3`.
Demo protocol_hash: `50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417`.
Этот хеш относится к сохранённому default demo с candidate_count =
**50**; показанный префикс top-20 не является новым
custom-запуском или утверждённым экспериментальным протоколом.
Пути к исходным CSV: `data/FAR-Trans/`; SHA-256 каждого файла сверяется со
`starter_summary.json` и приватным evidence. Клиент везде обозначен `[MASKED]`.
Порядковые номера далее — **CSV data-record без заголовка**, не обязательно
физические номера строк при наличии quoted multiline values.

## 1. Детерминированный выбор примера

| Проверка выбора | Значение |
| --- | --- |
| Demo users | 100 |
| Demo users с future Buy | 11 |
| Distinct target-пар на позициях CBF 11–20 | 0 |
| Users с target на позициях 11–20 | 0 |
| Distinct target-пар в CBF top-20 | 5 |
| Выбранный пользователь: ordinal в demo (не customerID) | 69 |
| Правило выбора пары | no target at ranks 11–20; highest rank within top-20, then demo ordinal, then ISIN |

Если preferred range пуст, выбирается самая поздняя позиция среди попавших в
CBF20, затем первый пользователь по deterministic demo order, затем ISIN.
После выбора пары берётся самая ранняя допустимая Buy-строка окна по
(timestamp, числовой transactionID, CSV-record). Это демонстрационный отбор
одного record, **не изменение evaluation cohort и не выбор хороших пользователей
для будущей оценки**.

В demo нет покупок на позициях 11–20. Поэтому пример НЕ показывает перенос из-за границы top-10: купленный актив уже внутри top-10, а потенциальное действие ranker — поднять его выше. Не выдаём гипотетический подъём за полученный результат LLM.

## 2. Сырой record в формате задания

```text
source: FAR-Trans public release (https://researchdata.gla.ac.uk/1658/), FAR-Trans/transactions.csv
record id: transactionID=9049523 (verbatim); CSV data-record=85881
fields: customerID, ISIN, transactionID, transactionType, timestamp, totalValue, units, channel, marketID
values: [MASKED], GRS003003035, 9049523, Buy, 2021-06-08, 996.41, 370.0, Phone Banking, XATH
window: 2021-06-01 <= timestamp < 2021-07-01; starter demo; repeat allowed
```

Все значения дословно из CSV, кроме обязательной маски customerID.
Наличие записи после cut не делает её поля допустимыми входами.
Знак ID неотрицателен; это исключает negative-ID marker, но не доказывает
отсутствия ретроспективных поправок авторов релиза.

### Совпадения transactionID

| CSV-record | Клиент | ISIN | Дата | Тип | totalValue | units | channel | marketID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 85881 | [MASKED: selected] | GRS003003035 | 2021-06-08 | Buy | 996.41 | 370.0 | Phone Banking | XATH |

Всего строк с этим transactionID: **1**. Shared ID не равен полному duplicate: starter удаляет только одинаковые полные строки.

## 3. История до cut: доказательство из сырых транзакций

Только сохранённые Buy: дата строго до cut, nonnegative ID, eligible asset, полные дубликаты удалены. Таблица отсортирована по дате, transactionID, CSV-record; значения исходные. Все строки принадлежат одному `[MASKED]`.

| CSV-record | transactionID | Дата | ISIN | units | totalValue (estimated) | channel | marketID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 311546 | 7440925 | 2020-01-15 | GRS314003005 | 1200.0 | 128.16 | Phone Banking | XATH |
| 311547 | 7492006 | 2020-02-19 | GRS314003005 | 4300.0 | 522.02 | Phone Banking | XATH |
| 311548 | 7516248 | 2020-02-28 | GRS314003005 | 27000.0 | 2181.6 | Phone Banking | XATH |
| 311549 | 7565932 | 2020-03-18 | GRS314003005 | 3500.0 | 180.6 | Phone Banking | XATH |
| 311550 | 7576342 | 2020-03-20 | GRS314003005 | 33000.0 | 1864.5 | Phone Banking | XATH |
| 85876 | 7589623 | 2020-03-27 | GRS003003035 | 1800.0 | 1961.1 | Phone Banking | XATH |
| 311551 | 7596050 | 2020-03-31 | GRS314003005 | 6400.0 | 358.40000000000003 | Phone Banking | XATH |
| 85877 | 7598639 | 2020-04-01 | GRS003003035 | 1250.0 | 1385.0000000000002 | Phone Banking | XATH |
| 85878 | 7605883 | 2020-04-06 | GRS003003035 | 2050.0 | 2316.5 | Phone Banking | XATH |
| 311552 | 7761782 | 2020-07-06 | GRS314003005 | 29000.0 | 1769.0 | Phone Banking | XATH |
| 48636 | 7793532 | 2020-07-23 | GRS014003032 | 68.75 | 1377.2 | Phone Banking | XATH |
| 156284 | 7840328 | 2020-09-04 | GRS323003012 | 8500.0 | 3400.0 | Phone Banking | XATH |
| 311553 | 8013233 | 2020-11-18 | GRS314003005 | 7500.0 | 282.75 | Phone Banking | XATH |
| 311554 | 8031731 | 2020-11-20 | GRS314003005 | 20000.0 | 767.9999999999999 | Phone Banking | XATH |
| 311555 | 8094607 | 2020-12-02 | GRS314003005 | 11000.0 | 407.0 | Phone Banking | XATH |
| 156285 | 8489605 | 2021-02-19 | GRS323003012 | 470.0 | 262.918 | Phone Banking | XATH |
| 311556 | 8584111 | 2021-03-10 | GRS314003005 | 15000.0 | 475.5 | Phone Banking | XATH |
| 156286 | 8620121 | 2021-03-17 | GRS323003012 | 970.0 | 691.222 | Phone Banking | XATH |
| 156287 | 8674372 | 2021-03-26 | GRS323003012 | 14000.0 | 10189.2 | Phone Banking | XATH |
| 156288 | 8731105 | 2021-04-07 | GRS323003012 | 1200.0 | 951.6 | Phone Banking | XATH |
| 156289 | 8737228 | 2021-04-08 | GRS323003012 | 2770.0 | 2210.46 | Phone Banking | XATH |
| 156290 | 8767713 | 2021-04-13 | GRS323003012 | 1400.0 | 1085.28 | Phone Banking | XATH |
| 306168 | 8820654 | 2021-04-23 | GRS372003004 | 1200.0 | 996.0 | Phone Banking | XATH |
| 306169 | 8839530 | 2021-04-27 | GRS372003004 | 6000.0 | 4950.0 | Phone Banking | XATH |

| Сводка истории | Значение |
| --- | --- |
| Retained Buy-строк | 24 |
| Distinct history-assets | 5 |
| До cut исключено у этого клиента: not_Buy | 14 |
| До cut исключено у этого клиента: negative_id | 5 |

| ISIN истории | buy_count | Название as-of | Attributes |
| --- | --- | --- | --- |
| GRS314003005 | 11 | MARFIN INVESTEMENT GROUP S.A. | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive |
| GRS323003012 | 7 | EUROBANK ERGASIAS SERVICES AND HOLDINGS S.A. | assetCategory=stock; assetSubCategory=large cap; industry=banks - regional; marketID=xath; sector=financial services |
| GRS003003035 | 3 | NATIONAL BANK OF GREECE | assetCategory=stock; assetSubCategory=large cap; marketID=xath |
| GRS372003004 | 2 | IKTINOS HELLAS S.A (CR) | assetCategory=stock; industry=building products & equipment; marketID=xath; sector=industrials |
| GRS014003032 | 1 | PIRAEUS FINANCIAL HOLDINGS S.A. | assetCategory=stock; assetSubCategory=large cap; industry=banks - regional; marketID=xath; sector=financial services |

**В контексте starter нет дат Buy.** Там есть `buy_count` и
`metadata_timestamp` (это дата metadata актива, НЕ last-buy date). Сырые даты
в таблице выше — audit-only. В текущем иллюстративном общем входе они не
добавлены ни LLM, ни LambdaMART. Возможное обогащение event-date/recency признаками
должно быть явно утверждено в study plan и симметрично для обоих методов.

## 4. Профиль и метаданные as-of

Профиль выбирается по максимальному `timestamp` строго до cut (не по последней
строке всего файла). Все snapshots этого клиента:

| CSV-record | customerID | timestamp | customerType | riskLevel | investmentCapacity | lastQuestionnaireDate | Выбран as-of |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14768 | [MASKED] | 2018-01-02 | Professional | Aggressive | CAP_80K_300K | 2018-12-10 | False |
| 14769 | [MASKED] | 2021-02-19 | Professional | Aggressive | CAP_80K_300K | 2018-12-10 | False |
| 14770 | [MASKED] | 2021-03-10 | Professional | Aggressive | CAP_80K_300K | 2018-12-10 | True |

| Проверка профиля | Результат |
| --- | --- |
| As-of CSV-record | 14770 |
| Snapshot timestamp | 2021-03-10 |
| Questionnaire date | 2018-12-10 |
| Questionnaire ≥ cut (нельзя считать доступной) | False |
| Questionnaire sentinel | False |
| riskLevel status в source | observed (in release; provenance caveats below) |
| investmentCapacity status в source | observed (in release; provenance caveats below) |

Общий planned profile block для обоих методов:

```json
{
  "timestamp": "2021-03-10",
  "timestamp_is_placeholder": false,
  "note": "Profile is contextual information, not a validated asset-risk filter.",
  "customerType": {
    "value": "Professional",
    "status": "observed"
  },
  "riskLevel": {
    "value": "Aggressive",
    "status": "observed"
  },
  "investmentCapacity": {
    "value": "CAP_80K_300K",
    "status": "observed"
  }
}
```

Если анкета позже/на cut, risk/capacity в planned input маскируются одинаково для обоих методов; это withheld/missing input, а не якобы пустое значение исходного CSV. Даже ранняя анкета не доказывает дату обучения модели, создавшей Predicted_* категории. Общие обнаруженные late-profile риски см. leakage.md / starter_baselines.md.

Строки metadata выбранного актива (для видимости будущих updates):

| CSV-record | ISIN | assetName | assetShortName | assetCategory | assetSubCategory | marketID | sector | industry | timestamp | Выбрана |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 182 | GRS003003035 | NATIONAL BANK OF GREECE | ΕΤΕ | Stock | Large Cap | XATH |  |  | 2018-01-02 | True |

Code-as-of не доказывает, что текущие названия исторически существовали в эту
дату: часть metadata timestamps совпадает с началом цен, а provenance переименований
не дан. При одинаковом timestamp starter сохраняет **первую** строку файла;
нельзя незаметно заменить это на last-row tie-break старой EDA.

## 5. Иллюстративный CBF top-20 и целевая покупка

Это префикс текущего CBF starter для наглядности **размера из RQ**, не принятое
решение о финальном генераторе или составе pool. Future Buy не вставлялся в список.
Идентичные атрибуты могут давать одинаковый score; tie-break — ISIN.

| CBF rank | ISIN | Название as-of | Короткое имя | Атрибуты CBF | Metadata date | CBF score (12dp) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | GRS117123000 | LOULIS FOOD INGREDIENTS | ΛΟΥΛΗ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 2 | GRS118003003 | C. SARANTOPOULOS FLOUR MILLS S.A. (CR) | ΣΑΡΑΝ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 3 | GRS224003004 | KRE.KA S.A. (CR) | ΚΡΕΚΑ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 4 | GRS314003005 | MARFIN INVESTEMENT GROUP S.A. | ΜΙΓ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 5 | GRS371113002 | KRETA FARM S.A. (CR) | ΚΡΕΤΑ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 6 | GRS385113006 | EVROFARMA SA (CR) | ΕΒΡΟΦ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 7 | GRS438003006 | FLOUR MILLS KEPENOS S.A. (CR) | ΚΕΠΕΝ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 8 | GRS469003024 | KRI- KRI MILK INDUSTRY S.A. (CR) | ΚΡΙ | assetCategory=stock; industry=packaged foods; marketID=xath; sector=consumer defensive | 2018-01-02 | 0.870690233671 |
| 9 | 100974271034 | VIOHALCO SA (el-GR) | ΒΙΟ | assetCategory=stock; assetSubCategory=large cap; marketID=xath | 2018-01-02 | 0.840136040637 |
| 10 | GRS003003035 | NATIONAL BANK OF GREECE | ΕΤΕ | assetCategory=stock; assetSubCategory=large cap; marketID=xath | 2018-01-02 | 0.840136040637 |
| 11 | GRS054061007 | EL. D. MOUZAKIS  S.A. (CB) | ΜΟΥΖΚ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 12 | GRS070003009 | KEKROPS S.A. (CR) | ΚΕΚΡ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 13 | GRS074083007 | TITAN CEMENT COMPANY S.A. (CR) | ΤΙΤΚ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 14 | GRS074084005 | TITAN CEMENT COMPANY S.A. (PR) | ΤΙΤΠ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 15 | GRS079103008 | NEXANS HELLAS S.A. (CR) | ΑΛΚΑΤ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 16 | GRS083003012 | F.G. EUROPE S.A. (CR) | ΕΦΤΖΙ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 17 | GRS087103008 | INTRACOM S.A. HOLDINGS (CR) | ΙΝΤΚΑ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 18 | GRS090101007 | N. LEVEDERIS S.A.  (CB) | ΛΕΒΚ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 19 | GRS090106006 | N. LEVEDERIS S.A.  (PB) | ΛΕΒΠ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |
| 20 | GRS095001004 | PIPEWORKS L. GIRAKIAN PROFIL S.A. | ΤΖΚΑ | assetCategory=stock; marketID=xath | 2018-01-02 | 0.840095563847 |

| Выбранный target | Значение |
| --- | --- |
| ISIN | GRS003003035 |
| Позиция в CBF full ranking | 10 |
| Позиция в Latest buy full ranking | 5 |
| CBF top-10 membership | True |
| CBF top-20 membership | True |
| Latest buy top-10 membership | True |
| Repeat: покупался до cut | True |

Latest buy baseline выдаёт только Top-10. Полная позиция для аудита получена
той же `recency_rankings` с расширенным k; её первые десять сверены с сохранённым
на этапе starter `predictions.csv`. Если позиция вне десяти, она не является
засчитанным попаданием этого baseline.

### Все будущие метки этого пользователя

| CSV-record | transactionID | ISIN | Дата | Repeat до cut |
| --- | --- | --- | --- | --- |
| 85881 | 9049523 | GRS003003035 | 2021-06-08 | True |

Truth T = `["GRS003003035"]`; distinct relevance count = **1**.

Top-10 списки для ручной сверки (relevance-флаги только для audit, НЕ для input):

| Rank | CBF ISIN | CBF name | CBF relevant | Latest buy ISIN | Latest buy name | Latest buy relevant |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | GRS117123000 | LOULIS FOOD INGREDIENTS | False | GRS372003004 | IKTINOS HELLAS S.A (CR) | False |
| 2 | GRS118003003 | C. SARANTOPOULOS FLOUR MILLS S.A. (CR) | False | GRS323003012 | EUROBANK ERGASIAS SERVICES AND HOLDINGS S.A. | False |
| 3 | GRS224003004 | KRE.KA S.A. (CR) | False | GRS314003005 | MARFIN INVESTEMENT GROUP S.A. | False |
| 4 | GRS314003005 | MARFIN INVESTEMENT GROUP S.A. | False | GRS014003032 | PIRAEUS FINANCIAL HOLDINGS S.A. | False |
| 5 | GRS371113002 | KRETA FARM S.A. (CR) | False | GRS003003035 | NATIONAL BANK OF GREECE | True |
| 6 | GRS385113006 | EVROFARMA SA (CR) | False | GRS015003007 | ALPHA SERVICES AND HOLDINGS S.A. | False |
| 7 | GRS438003006 | FLOUR MILLS KEPENOS S.A. (CR) | False | GRC4261213D5 | MOTOR OIL HELLAS CORINTH REFINERIES SA | False |
| 8 | GRS469003024 | KRI- KRI MILK INDUSTRY S.A. (CR) | False | GRS434003000 | PPC SA | False |
| 9 | 100974271034 | VIOHALCO SA (el-GR) | False | GRS393503008 | MYTILINAIOS SA | False |
| 10 | GRS003003035 | NATIONAL BANK OF GREECE | True | GRS419003009 | OPAP SA | False |

| Метод | Relevant ranks в Top-10 | DCG@10 | IDCG@10 | nDCG@10 (12dp) |
| --- | --- | --- | --- | --- |
| CBF | 10 | 0.289064826318 | 1.000000000000 | 0.289064826318 |
| Latest buy | 5 | 0.386852807235 | 1.000000000000 | 0.386852807235 |

Подъём доступного target на более высокий rank изменит DCG, но не T или IDCG.
Ни новый порядок, ни выигрыш LambdaMART/LLM здесь не получены: это только
иллюстрация механизма и измеренные классические baseline.

## 6. Поля будущего общего входа и audit/label-only поля

Ниже — **черновой same-information contract для planned эксперимента**, не
исполненный inference. Общий payload сохранён в игнорируемом
`data/ranker_example/planned_common_input.json`: только past-only данные и
кандидаты CBF, без truth, selected-target marker, будущих дат и сумм.
Он не выбирает окончательный pool. Для LambdaMART ещё предстоит реализовать
user–candidate features и train-only обработку текста; отбрасывать текст,
который видит LLM, без контроля нельзя.

| Поле | Значение / ссылка на строки | Кто использует (planned) | Статус | Дата / доступность | Проверка утечки |
| --- | --- | --- | --- | --- | --- |
| future transaction.customerID | [MASKED] | только метка / join (маска) | observed (in release; provenance caveats below) | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.ISIN | GRS003003035 | только метка | observed (in release; provenance caveats below) | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.transactionID | 9049523 | не используется (audit only) | observed (in release; provenance caveats below) | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.transactionType | Buy | только метка | observed (in release; provenance caveats below) | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.timestamp | 2021-06-08 | только метка | observed (in release; provenance caveats below) | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.totalValue | 996.41 | не используется (audit only) | predicted/estimated: пересчитана по closePrice | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.units | 370.0 | не используется (audit only) | observed/adjusted: splits/reconstruction provenance неполон | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.channel | Phone Banking | не используется (audit only) | observed (in release; provenance caveats below) | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| future transaction.marketID | XATH | не используется (audit only) | observed (in release; provenance caveats below) | 2021-06-08 | после cut; эта связь/событие не передаются моделям |
| input.user_alias | EXAMPLE_USER | LLM / LambdaMART: только ключ, не semantic feature | observed/служебная маска | 2021-06-01 | не раскрывает customerID |
| input.cutoff | 2021-06-01 | LLM / LambdaMART | observed/параметр протокола | 2021-06-01 | не источник будущих знаний |
| input.profile.customerType | Professional | LLM / LambdaMART | observed (in release; provenance caveats below) | 2021-03-10; questionnaire 2018-12-10 | late questionnaire → одинаковая маска; время оценивания Predicted_* неизвестно |
| input.profile.customerType.status | observed | LLM / LambdaMART | observed/derived marker, не новая оценка | 2021-03-10 | сохранять Predicted_/missing, не представлять оценку как факт |
| input.profile.riskLevel | Aggressive | LLM / LambdaMART | observed (in release; provenance caveats below) | 2021-03-10; questionnaire 2018-12-10 | late questionnaire → одинаковая маска; время оценивания Predicted_* неизвестно |
| input.profile.riskLevel.status | observed | LLM / LambdaMART | observed/derived marker, не новая оценка | 2021-03-10 | сохранять Predicted_/missing, не представлять оценку как факт |
| input.profile.investmentCapacity | CAP_80K_300K | LLM / LambdaMART | observed (in release; provenance caveats below) | 2021-03-10; questionnaire 2018-12-10 | late questionnaire → одинаковая маска; время оценивания Predicted_* неизвестно |
| input.profile.investmentCapacity.status | observed | LLM / LambdaMART | observed/derived marker, не новая оценка | 2021-03-10 | сохранять Predicted_/missing, не представлять оценку как факт |
| input.profile.timestamp / placeholder flag / note | 2021-03-10 | LLM / LambdaMART | observed/derived | 2021-03-10 | строго до cut; snapshot ≠ доказанное knowledge time |
| audit.profile.lastQuestionnaireDate | 2018-12-10 | не используется как score; только availability audit | observed | 2018-12-10 | не включено в stock model context; поздняя анкета блокирует risk/capacity |
| history.id / name / attributes / metadata_timestamp | вся таблица history summary выше | LLM / LambdaMART | observed metadata; отсутствующие attributes остаются missing | metadata строго < cut | имя может быть ретроспективным; одинаковый набор полей обоим |
| history.buy_count | {"GRS014003032": 1, "GRS003003035": 3, "GRS323003012": 7, "GRS372003004": 2, "GRS314003005": 11} | LLM / LambdaMART | observed/derived из retained Buy | только Buy < cut | negative IDs и exact duplicates исключены |
| audit.history.timestamp | даты raw history выше | не используется в текущем общем payload; audit only | observed in release; исходная дата могла переноситься | каждая дата < cut | НЕ metadata_timestamp; будущая recency-фича требует симметричного решения |
| audit.history.transactionID / channel / units / totalValue / execution marketID | raw history выше | не используется в общем payload | observed/adjusted; totalValue estimated | даты history < cut; очистка post hoc | не придумывать spending/return; неотрицательный ID не гарантирует raw provenance |
| candidate.source_metadata.ISIN | GRS003003035 | LLM / LambdaMART | observed (in release; provenance caveats below) | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.assetName | NATIONAL BANK OF GREECE | LLM / LambdaMART | observed (in release; provenance caveats below) | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.assetShortName | ΕΤΕ | LLM / LambdaMART | observed (in release; provenance caveats below) | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.assetCategory | Stock | LLM / LambdaMART | observed (in release; provenance caveats below) | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.assetSubCategory | Large Cap | LLM / LambdaMART | observed (in release; provenance caveats below) | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.marketID | XATH | LLM / LambdaMART | observed (in release; provenance caveats below) | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.sector |  | LLM / LambdaMART | missing | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.industry |  | LLM / LambdaMART | missing | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.source_metadata.timestamp | 2018-01-02 | LLM / LambdaMART | observed (in release; provenance caveats below) | 2018-01-02 | as-of < cut; справочник релиза не доказывает историчность имени/класса |
| candidate.id / tokens | ISIN / categorical tokens каждой строки CBF20 | LLM / LambdaMART | observed/derived | metadata < cut | id для сопоставления output; без future-target флага |
| candidate.cbf_score / cbf_position | score и rank выше (полная точность в evidence) | LLM / LambdaMART | observed/derived; НЕ predicted purchase probability | только past history + as-of features | общий reference score; порядок промпта/перестановки ещё planned |
| eligibility: as-of metadata + positive pre-cut price | 767 | только общий фильтр протокола | observed/derived | metadata и цена строго < cut | не требовать цен до конца будущего окна; не выдавать universe будущего за признак |
| future truth set / metrics / selected target / oracle | ["GRS003003035"] | только метка / evaluation | observed/derived | [2021-06-01, 2021-07-01) | запрещены в обоих model inputs |
| limit_prices.maxDate / priceMaxDate / profitability; future prices | не читаются scorer | не используется | derived from future/full period | после cut | прямая утечка; запрет |

**Нерешённые риски:** отрицательные ID удалены, но дополнительные синтетические Sell,
переносы дат и back-adjusted prices/units полностью не маркированы. As-of профиль
не автоматически безопасен при поздней анкете. Вектор CBF не использует имена,
а LLM может извлекать из них знания после cut: нужны same-information текстовый
контроль и проверка анонимизации в study plan. Цены используются только для
eligibility, не прибыльности. Нет ответов анкет, реальных имён или инвестиционных
целей — их нельзя выдумывать из повторных Buy.

## 7. Независимость и личная проверка

Генератор импортирует starter, но `scripts/verify_example.py` **не импортирует**
starter, `starter_audit`, генератор, render helpers или другие наши модули.
Он самостоятельно читает CSV и пересчитывает этот record, selection, history,
profile, metadata, CBF и Latest buy, сравнивая с приватным expected evidence.
Успех/число проверок и короткая ручная инструкция печатаются им в
[verify_example.md](verify_example.md). При несоответствии — ненулевой exit code.
Автоматическая проверка не подменяет ручную проверку Дарьи: **она ещё не выполнена**.

Запуск одной командой каждого шага из папки проекта:

```powershell
python -B scripts/make_ranker_example.py
python -B scripts/verify_example.py
```
