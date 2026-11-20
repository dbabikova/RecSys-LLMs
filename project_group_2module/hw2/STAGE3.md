# Этап 3 — общий контракт входов LLM и LambdaMART

Статус: реализовано и автоматически проверено Codex; Дарья приняла этап («ок»
в чате). Самостоятельная ручная raw-сверка Дарьи не заявляется выполненной.
STAGE3_VERIFICATION.json сохраняет статус на момент автоматического запуска.
Это подготовка входов, не обучение и не LLM inference. Метрик новых моделей нет.
Users, их порядок, исходный protocol и candidate hash этапа 2 не изменились.

## Контракт common_input_v1

В основе один past-only payload на пользователя. В LLM-паре изменяется только
порядок массива candidates. Из того же payload строятся unfitted feature records
для последующего обучения LambdaMART. Два независимых loader не определяют разные
наборы полей для разных моделей.

| Информация | LLM payload | Запись для LambdaMART / будущий train-only transformer |
| --- | --- | --- |
| Cutoff | ISO date | Числовой ordinal того же дня |
| Профиль | customerType, riskLevel, investmentCapacity с value/status; возраст snapshot | Те же category/status и возраст snapshot |
| Атрибуты кандидата | name; category, subcategory, sector, industry, market | Те же categorical поля и candidate_name; train-only TF-IDF/сходство на следующем этапе |
| Возраст metadata | Days since metadata snapshot | То же число |
| История | До 20 уникальных assets: name, attributes, metadata age, buy_count, last_buy_age_days | Идентичный history_context; имена также в history_names; будущие train-only categorical/text преобразования |
| Объём истории | Полные и retained counts: distinct assets / Buy events, число omitted assets | Те же summary features |
| Повторы кандидата | Число past Buy и возраст последней покупки; None, если покупки нет | Те же числовые признаки и missing при отсутствии даты |
| Исходный retrieval | source_cbf_rank 1…20; source_cbf_score | Те же значения, не presentation index |
| Derived overlaps | Вычислимы из общей истории и атрибутов | Buy-weighted match share для каждого из пяти категориальных полей |
| Связи asset↔history↔output | Нейтральные asset aliases вместо ISIN | Те же aliases как join/output keys; не обучаемые category features |
| Пользователь | U001…U100 по исходному порядку cohort | Тот же group key, не обучаемый признак |

Raw customerID и ISIN остаются только в private_mapping.jsonl. Asset alias =
`A` + первые 12 hex SHA-256(ISIN); уникальность проверена на всём каталоге.
Модель возвращает aliases, а будущий adapter сопоставляет их с оригинальными
ISIN для scorer. Это не изменение candidate membership. Names остаются исходными:
alias не устраняет риск pretrained hindsight и не является анонимизацией всех данных.
При агрегации tie-break остаётся по оригинальному ISIN через приватное mapping.

**Важно:** feature_records.jsonl — не готовая числовая матрица LambdaMART.
Text/categorical encoders ещё не fitted. Сохранены все исходные поля, в том числе
история и имена; на этапах обучения нужно реализовать их преобразования на train
и проверить фактическую матрицу. Пока подтверждена симметрия исходной информации,
а не доказана эквивалентность способов её использования и pretrained knowledge.

## Правила времени и очистки

- Используем source CSV с проверенными SHA-256. Метаданные и профили выбираются
  как последняя полная timestamp-строка строго до cutoff; при точном timestamp
  tie сохраняется первая строка, как в starter, без новой стратегии слияния.
- История: только Buy строго до cutoff, eligible ISIN, exact full-row duplicates
  удаляются, negative transactionID исключаются. Sell, value, units, channel,
  future prices и profitability не входят в payload.
- Missing/Not_Available/nan → `[MISSING]`. Ничего не дополняем внешними сведениями.
- Профиль с placeholder/невалидной датой snapshot маскируется целиком. Risk и
  capacity дополнительно разрешаются лишь при валидной lastQuestionnaireDate
  строго до cutoff и не равной 2000-01-01. Иначе одинаковое value=[MISSING],
  status=withheld_unverified у обеих моделей. Конкретная поздняя дата и причина
  late/missing не передаются модели. Predicted_* у допустимых полей сохраняется
  со status=estimated; не выдаём predicted за observed.
- Отсутствующая последняя покупка кандидата = None, не выдуманная дата или ноль.
- Даты представлены возрастом в днях относительно cutoff. Metadata age не
  заменяет Buy recency. Проверяются отдельно по исходным таблицам.

## Сокращение истории и CBF signal

Общее правило: выбрать до 20 **уникальных** прошлых eligible assets по последней
Buy date descending, затем buy_count descending, затем исходному ISIN ascending.
Число событий Buy не равно числу уникальных активов.

На текущих 100 users правило затронуло одного пользователя: 23 → 20 assets.
В обеих ветках сохраняются те же 20, их полные прошлые counts и last dates.
Summary показывает omitted count. Candidate past counts и CBF score считаются по
**всей** допустимой истории, как источник retrieval, и доступны обоим методам.
Поэтому сокращение явного списка не означает отсутствия сигнала omitted history.
Если будущий tokenizer выявит превышение контекста, нельзя молча обрезать только
LLM: потребуется отдельное симметричное изменение контракта и новый hash.

## Перестановки

Seeds 17, 42, 2026. Presentation order = сортировка aliases по
`SHA-256(str(seed) + ':' + asset_alias)`, tie-break asset_alias.
Это детерминированные псевдослучайные перестановки, а не Python random shuffle.
Все поля candidate остаются прикреплены к нему, включая source_cbf_rank/score.
История, профиль и остальные части payload неизменны. Для LambdaMART records
не зависят от presentation order при сравнении по candidate alias.
Исследование order sensitivity условно на том, что исходный rank явно виден обеим
моделям; результат нельзя обобщать на prompts без этого сигнала.

## Артефакты и выполненные проверки

Приватные файлы в data/hw2/common_inputs/, ignored:

- payloads.jsonl — 100 общих входов, без truth и исходных customerID;
- feature_records.jsonl — 2000 пар user–candidate, без label;
- private_mapping.jsonl — соответствие aliases исходным ID, не вход модели;
- presentation_orders.jsonl — исходные и три переставленные последовательности.

INPUTS.json содержит только contract settings, хеши и агрегаты. На текущем cohort
80 profiles допускают risk/capacity, у 20 они withheld. Неопубликованные draft
copies сохранены локально при подготовке и не используются текущим manifest.

verify_inputs.py не импортирует builder, contract или starter: самостоятельно
читает raw CSV, пересчитывает metadata, counts, даты, выбор истории, masking,
CBF score и все значения 2000 feature records. Проверяет cohort, CBF20, hashes,
отсутствие customerID и запрещённых label/value полей. Результат —
STAGE3_VERIFICATION.json, все checks passed.

Четыре unit tests на контролируемых fixtures прошли: future/sentinel questionnaire,
future profile snapshot, future metadata/Buy rejection, truncation и invariance
содержимого при permutation. Отдельный PowerShell review_inputs.ps1 проверил
100 payload и все 2000 records, совпадение names/counts/recency/history.
Это автоматические проверки Codex, не личная проверка Дарьи.

## Воспроизведение в PowerShell

Из project_group_2module/:

```powershell
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/build_inputs.py
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/verify_inputs.py
& 'C:\Users\User\anaconda3\python.exe' -B -m unittest discover -s hw2/scripts -p test_input_contract.py -v
& './hw2/scripts/review_inputs.ps1' -UserAlias U030
```

Файлы, уже отличающиеся от frozen output, builder не перезаписывает. Все входные
архивы должны быть локально доступны в data/hw2/inputs/ по предыдущим этапам.
Запуск не скачивает модели, не обучает encoders и не вызывает inference.

## Проверка Дарьи

1. Прочитать таблицу соответствия и правила masking/truncation. Подтвердить,
   что признаки подготовлены из той же информации, а fit ещё впереди.
2. В STAGE3_VERIFICATION.json проверить users=100, feature_records_checked=2000,
   candidate hash как в CANDIDATES.json и все checks=true.
3. При желании запустить команды выше. В PowerShell-примере U030 ожидаются
   5 уникальных assets и 81 Buy event, без truncation. У первого кандидата
   source_cbf_rank=1, past_buy_count=19, last_buy_age_days=1.
   Это ожидаемые автоматические значения, не подтверждение личной проверки.
4. Для самостоятельной raw-сверки взять 30-го пользователя из protocol.json,
   найти все его сохранённые Buy до 2021-06-01, убрать отрицательные ID и полные
   дубли, посчитать события для первого кандидата GRS014003032 и последнюю дату.
   Ожидаются 19 покупок, последняя 2021-05-31. Raw ID не нужно копировать в Git.

Commit/push — после проверки и «ок» Дарьи. Следующий этап: временные train/validation
cuts и обучение преобразований только на ранних данных; затем LambdaMART.
