# Схема, целостность и сверка со статьёй

> Сгенерировано `scripts/run_analysis.py`; измеренные числа вычислены из CSV, не вписаны в Markdown вручную. Известные числа статьи отдельно помечены как цитируемые. Идентификаторы клиентов не публикуются.

Архив: `FAR-Trans.zip`, SHA-256: `d0d6cfe027732bab45fdcf9af80eff4fefddc789aedae9bdce0d3d6cfd05cfb6`.

Пустое поле и явный `Not_Available` учитываются раздельно.
`unique nonempty` включает текстовые маркеры, `unique known` исключает их.
Строковые ID не преобразуются в числа; диапазон для строк — длина, не
искусственный числовой порядок идентификаторов. Значение даты-заглушки
отдельно исследуется в отчёте о времени. Технических NaN в прочитанных строках
нет: загрузчик сохраняет исходные строки, затем явно типизирует даты и числа.

## Таблицы

| Файл | Строк | Столбцов | Точных дубликатов сверх первого |
| --- | --- | --- | --- |
| asset_information.csv | 836 | 9 | 0 |
| close_prices.csv | 703303 | 3 | 0 |
| customer_information.csv | 32468 | 6 | 0 |
| limit_prices.csv | 807 | 6 | 0 |
| markets.csv | 38 | 8 | 0 |
| transactions.csv | 388048 | 9 | 0 |

`questionnaires.csv` — не CSV-таблица: 177 текстовых строк, 25 вопросов, 115 вариантов ответа; число столбцов, типы, ключи и пропуски таблицы неприменимы. Ответов и клиентских записей в нём нет. README — документация, не таблица.

### asset_information.csv

| Поле | Тип после разбора | Пусто | Пусто/маркер missing | Unique nonempty | Unique known | Диапазон / категории |
| --- | --- | --- | --- | --- | --- | --- |
| ISIN | string (object) | 0.0000% | 0.0000% | 806 | 806 | длина 12 … 13 |
| assetName | string (object) | 2.8708% | 2.8708% | 711 | 711 | длина 3 … 105 |
| assetShortName | string (object) | 0.1196% | 0.1196% | 791 | 791 | длина 1 … 25 |
| assetCategory | string (object) | 0.0000% | 0.0000% | 3 | 3 | длина 3 … 5; Bond, MTF, Stock |
| assetSubCategory | string (object) | 32.1770% | 32.1770% | 10 | 10 | длина 4 … 12; Balanced, Bond, Bonds, Corporate, Equity, Government, Large Cap, Money Market, Other, Structured |
| marketID | string (object) | 0.0000% | 0.0000% | 38 | 38 | длина 3 … 5 |
| sector | string (object) | 46.4115% | 46.4115% | 15 | 15 | длина 6 … 29; Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Corporate, EUR High Yield Bond, Energy, Financial Services, Healthcare, Industrials, Real Estate, Sector Equity Precious Metals, Sovereign, Technology, Utilities |
| industry | string (object) | 52.5120% | 52.5120% | 119 | 119 | длина 5 … 49 |
| timestamp | date → datetime64[ns] | 0.0000% | 0.0000% | 231 | 231 | 2018-01-01 … 2022-10-31 |

| Дата | Непустых неразбираемых значений |
| --- | --- |
| timestamp | 0 |

### close_prices.csv

| Поле | Тип после разбора | Пусто | Пусто/маркер missing | Unique nonempty | Unique known | Диапазон / категории |
| --- | --- | --- | --- | --- | --- | --- |
| ISIN | string (object) | 0.0000% | 0.0000% | 807 | 807 | длина 12 … 13 |
| timestamp | date → datetime64[ns] | 0.0000% | 0.0000% | 1279 | 1279 | 2018-01-01 … 2022-11-29 |
| closePrice | float64 | 0.0000% | 0.0000% | 347772 | 347772 | 0.01 … 3783.5 |

| Дата | Непустых неразбираемых значений |
| --- | --- |
| timestamp | 0 |

### customer_information.csv

| Поле | Тип после разбора | Пусто | Пусто/маркер missing | Unique nonempty | Unique known | Диапазон / категории |
| --- | --- | --- | --- | --- | --- | --- |
| customerID | string (object) | 0.0000% | 0.0000% | 29090 | 29090 | длина 20 … 20 |
| customerType | string (object) | 0.0000% | 0.0000% | 5 | 5 | длина 4 … 12; Inactive, Legal Entity, Mass, Premium, Professional |
| riskLevel | string (object) | 0.0000% | 1.0595% | 9 | 8 | длина 6 … 22; Aggressive, Balanced, Conservative, Income, Predicted_Aggressive, Predicted_Balanced, Predicted_Conservative, Predicted_Income |
| investmentCapacity | string (object) | 0.0000% | 1.0687% | 9 | 8 | длина 9 … 22; CAP_30K_80K, CAP_80K_300K, CAP_GT300K, CAP_LT30K, Predicted_CAP_30K_80K, Predicted_CAP_80K_300K, Predicted_CAP_GT300K, Predicted_CAP_LT30K |
| lastQuestionnaireDate | date → datetime64[ns] | 0.0000% | 0.0000% | 1555 | 1555 | 2000-01-01 … 2022-11-23 |
| timestamp | date → datetime64[ns] | 0.0000% | 0.0000% | 1167 | 1167 | 2000-01-01 … 2022-11-30 |

| Дата | Непустых неразбираемых значений |
| --- | --- |
| lastQuestionnaireDate | 0 |
| timestamp | 0 |

### limit_prices.csv

| Поле | Тип после разбора | Пусто | Пусто/маркер missing | Unique nonempty | Unique known | Диапазон / категории |
| --- | --- | --- | --- | --- | --- | --- |
| ISIN | string (object) | 0.0000% | 0.0000% | 807 | 807 | длина 12 … 13 |
| minDate | date → datetime64[ns] | 0.0000% | 0.0000% | 211 | 211 | 2018-01-01 … 2022-10-31 |
| maxDate | date → datetime64[ns] | 0.0000% | 0.0000% | 211 | 211 | 2018-01-19 … 2022-11-29 |
| priceMinDate | float64 | 0.0000% | 0.0000% | 765 | 765 | 0.0145 … 3409.45 |
| priceMaxDate | float64 | 0.0000% | 0.0000% | 775 | 775 | 0.0125 … 3344.561658 |
| profitability | float64 | 0.0000% | 0.0000% | 798 | 798 | -0.997241213866 … 83.5454545455 |

| Дата | Непустых неразбираемых значений |
| --- | --- |
| minDate | 0 |
| maxDate | 0 |

### markets.csv

| Поле | Тип после разбора | Пусто | Пусто/маркер missing | Unique nonempty | Unique known | Диапазон / категории |
| --- | --- | --- | --- | --- | --- | --- |
| exchangeID | string (object) | 0.0000% | 0.0000% | 29 | 29 | длина 4 … 5 |
| marketID | string (object) | 0.0000% | 0.0000% | 38 | 38 | длина 3 … 5 |
| name | string (object) | 0.0000% | 0.0000% | 38 | 38 | длина 5 … 44 |
| description | string (object) | 0.0000% | 0.0000% | 37 | 37 | длина 54 … 461 |
| country | string (object) | 0.0000% | 0.0000% | 20 | 20 | длина 1 … 14 |
| tradingDays | string (object) | 0.0000% | 0.0000% | 1 | 1 | длина 19 … 19; Mon,Tue,Wed,Thu,Fri |
| tradingHours | string (object) | 2.6316% | 2.6316% | 12 | 12 | длина 11 … 11; 00:00-06:00, 02:30-09:30, 07:00-15:30, 07:00-21:00, 07:00-21:30, 08:00-15:20, 08:00-16:00, 08:00-16:30, 08:15-15:20, 08:30-15:00, 09:00-16:30, 14:30-21:00 |
| marketClass | string (object) | 0.0000% | 0.0000% | 3 | 3 | длина 17 … 25; EFG Eurobank Mutual Funds, Private Securities, Public Securities |

### transactions.csv

| Поле | Тип после разбора | Пусто | Пусто/маркер missing | Unique nonempty | Unique known | Диапазон / категории |
| --- | --- | --- | --- | --- | --- | --- |
| customerID | string (object) | 0.0000% | 0.0000% | 29090 | 29090 | длина 20 … 20 |
| ISIN | string (object) | 0.0000% | 0.0000% | 320 | 320 | длина 12 … 13 |
| transactionID | int64 | 0.0000% | 0.0000% | 359128 | 359128 | -39344 … 11874153 |
| transactionType | string (object) | 0.0000% | 0.0000% | 2 | 2 | длина 3 … 4; Buy, Sell |
| timestamp | date → datetime64[ns] | 0.0000% | 0.0000% | 1232 | 1232 | 2018-01-02 … 2022-11-30 |
| totalValue | float64 | 0.0000% | 0.0000% | 105122 | 105122 | 2.83106871279e-16 … 27159117.9137 |
| units | float64 | 0.0000% | 0.0000% | 23998 | 23998 | 5.89805981832e-17 … 2150000 |
| channel | string (object) | 0.0000% | 0.0000% | 3 | 3 | длина 6 … 16; Branch, Internet Banking, Phone Banking |
| marketID | string (object) | 0.0000% | 0.0000% | 16 | 16 | длина 3 … 5; AEDAK, ALXB, CCLX, EBB, ENAX, HDAT, XATH, XBER, XCIE, XCYS, XDUB, XETR, XLON, XLUX, XNYS, XSTU |

| Дата | Непустых неразбираемых значений |
| --- | --- |
| timestamp | 0 |

## Кандидатные ключи

| Таблица | Ключ | Unique | Дубликаты сверх первого | Строк во всех группах коллизий |
| --- | --- | --- | --- | --- |
| asset_information | ISIN | 806 | 30 | 58 |
| asset_information | ISIN, timestamp | 833 | 3 | 6 |
| asset_information | ISIN, marketID, timestamp | 833 | 3 | 6 |
| customer_information | customerID | 29090 | 3378 | 5641 |
| customer_information | customerID, timestamp | 32468 | 0 | 0 |
| close_prices | ISIN, timestamp | 703303 | 0 | 0 |
| limit_prices | ISIN | 807 | 0 | 0 |
| markets | marketID | 38 | 0 | 0 |
| markets | exchangeID | 29 | 9 | 16 |
| transactions | transactionID | 359128 | 28920 | 52308 |
| transactions | customerID, transactionID | 388048 | 0 | 0 |

| Коллизии transactionID: групп с … | Количество |
| --- | --- |
| несколькими клиентами | 23388 |
| несколькими активами | 0 |
| несколькими датами | 0 |
| несколькими типами сделки | 0 |

transactionID не является глобальным PK: нельзя удалять строки только по нему.
Для воспроизводимого указания строки используем номер CSV-record вместе с
verbatim transactionID; исходные события автоматически не дедуплицируем.
Многократный customerID обычно означает историю профиля, не ошибку.
Многократный ISIN может означать несколько площадок/описаний; даже ISIN+timestamp
нуждается в проверке. В descriptive EDA один актив представлен последней строкой
по timestamp, при равенстве — последней строкой файла (stable); это техническое
правило, не доказательство правильного point-in-time атрибута.

## Внешние связи

| Связь | Строк без соответствия | Unique без соответствия | Публичные ID |
| --- | --- | --- | --- |
| transactions.ISIN → asset_information.ISIN | 0 | 0 | — |
| transactions.ISIN → close_prices.ISIN | 0 | 0 | — |
| transactions.customerID → customer_information.customerID | 0 | 0 | не публикуем клиентские ID |
| asset_information.ISIN → close_prices.ISIN | 0 | 0 | — |
| close_prices.ISIN → asset_information.ISIN | 1 | 1 | GRX497003012 |
| limit_prices.ISIN → close_prices.ISIN | 0 | 0 | — |
| transactions.marketID → markets.marketID | 0 | 0 | — |
| asset_information.marketID → markets.marketID | 0 | 0 | — |

## Дополнительные проверки

| Проверка | Значение |
| --- | --- |
| Транзакции без цены на точную дату | 327 |
| Транзакции до начала своего ряда | 0 |
| Транзакции после конца своего ряда | 327 |
| Транзакции на последнюю глобальную дату сделок | 327 |
| Строки с сопоставленной ценой | 387721 |
| Из них totalValue ≈ units × closePrice (rtol=1e-6, atol=0.01) | 387721 |
| Close price ≤ 0 | 0 |
| Units ≤ 0 | 0 |
| TotalValue ≤ 0 | 0 |
| Несовпадение minDate с фактическим началом ряда | 0 |
| Несовпадение maxDate с фактическим концом ряда | 0 |
| Несовпадение priceMinDate с первой ценой | 0 |
| Несовпадение priceMaxDate с последней ценой | 0 |
| Несовпадение profitability с last/first−1 | 0 |
| Максимальный разрыв между соседними ценами (календарных дней) | 4 |
| Разрывы цен > 10 календарных дней | 0 |
| Timestamp метаданных равен началу ценового ряда (строк) | 809 |

Слово «очищены» не означает, что все сделки имеют цену в тот же день; несовпадения не исправлялись автоматически. Календарные пробелы включают выходные/праздники, поэтому сами по себе не равны ошибке сбора.

## Сверка со статьёй, Table 1 и раздел Customers

| Показатель | В статье | В архиве | Архив − статья |
| --- | --- | --- | --- |
| Транзакции | 388049 | 388048 | -1 |
| Клиенты | 29090 | 29090 | 0 |
| Активы: Table 1 / metadata unique | 806 | 806 | 0 |
| Активы: текст Assets / price unique | 807 | 807 | 0 |
| Активы с транзакциями | 321 | 320 | -1 |
| Рынки | 38 | 38 | 0 |
| Точки цен | 703303 | 703303 | 0 |
| Покупки | 228913 | 228913 | 0 |
| Продажи | 159136 | 159135 | -1 |
| Unique transactions как customer+ISIN+type | 154103 | 154103 | 0 |
| Unique acquisitions как customer+ISIN | 89884 | 89884 | 0 |
| Unique sales как customer+ISIN | 64219 | 64219 | 0 |
| Оценённый риск: последние профили | 7141 | 7141 | 0 |
| Оценённая ёмкость: последние профили | 7318 | 7318 | 0 |

В статье внутри самого текста фигурируют разные размеры universe; таблица выше показывает обе трактовки, а не выбирает удобную. Семантика unique в статье операционализирована через customer–asset–type, не через transactionID. Причину отсутствующей продажи и отличия числа торгуемых активов по этому релизу установить нельзя: не приписываем их удалению конкретной строки. Расхождение покрытия metadata/prices показано отдельно по ISIN. README использует написания CAP_LT_30K, а CSV — CAP_LT30K; канонизацию по предположению не применяем.
