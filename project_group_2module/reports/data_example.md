# Одна реальная запись из окна меток

> Сгенерировано `scripts/run_analysis.py`; измеренные числа вычислены из CSV, не вписаны в Markdown вручную. Известные числа статьи отдельно помечены как цитируемые. Идентификаторы клиентов не публикуются.

Архив: `FAR-Trans.zip`, SHA-256: `d0d6cfe027732bab45fdcf9af80eff4fefddc789aedae9bdce0d3d6cfd05cfb6`.

Запись взята из исходного CSV, не из агрегата и не из сгенерированного примера. Только customerID заменён общей маской; ID транзакции и остальные значения сохранены дословно (включая десятичную запись). Реальная личность клиента не устанавливалась. Неотрицательный ID означает отсутствие выявленного отрицательного маркера, но не доказывает отсутствие всех преобразований авторов.

```text
source: FAR-Trans public release (https://researchdata.gla.ac.uk/1658/), FAR-Trans/transactions.csv
archive SHA-256: d0d6cfe027732bab45fdcf9af80eff4fefddc789aedae9bdce0d3d6cfd05cfb6
record id: transactionID=10905106; CSV data-record=255555 (header excluded)
fields: customerID, ISIN, transactionID, transactionType, timestamp, totalValue, units, channel, marketID
values: [MASKED_CUSTOMER], BE0974303357, 10905106, Buy, 2022-05-30, 2261.0, 700.0, Internet Banking, XATH
window: 2022-05-29 < timestamp <= 2022-11-29; provisional labels A and B
```

transactionID не уникален глобально, поэтому дополнительно указан порядковый CSV-record. Эта покупка допустима в A (новая пара) и B и принадлежит пользователю с Buy-историей в eligible universe до среза. Поля события находятся после cut: использовать их как признаки при прогнозе запрещено.

| Поле | Verbatim value / маска | Статус | Дата события/доступности | Проверка утечки |
| --- | --- | --- | --- | --- |
| customerID | [MASKED_CUSTOMER] | observed in release; customerID masked | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| ISIN | BE0974303357 | observed in release | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| transactionID | 10905106 | observed in release | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| transactionType | Buy | observed in release | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| timestamp | 2022-05-30 | observed in release; original date may be shifted | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| totalValue | 2261.0 | predicted/estimated (recomputed) | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| units | 700.0 | observed/adjusted; split provenance unknown | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| channel | Internet Banking | observed in release | 2022-05-30 | после cut; только метка/аудит, не вход модели |
| marketID | XATH | observed in release | 2022-05-30 | после cut; только метка/аудит, не вход модели |

Для totalValue дата — дата цены закрытия, по которой авторы оценили сумму;
для units это дата опубликованной скорректированной сделки. Точная дата
публикации исправлений и оригинальная intraday дата отсутствуют.
Слово observed здесь не превращает пересчитанную сумму в цену исполнения.
В §3 PDF этот пример иллюстрирует будущую целевую запись, а не доступный
в момент прогноза пользовательский профиль.
