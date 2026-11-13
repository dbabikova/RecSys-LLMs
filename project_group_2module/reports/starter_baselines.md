# Официальный стартер: четыре baseline и потолок CBF-реранкинга

> Сгенерировано `scripts/starter_audit.py` и `scripts/starter_reports.py`.
> Все результаты вычислены; таблицы README прочитаны скриптом из зафиксированного
> внешнего клона. Полная точность — `starter_summary.json`; здесь метрики округлены
> по HW2. Это диагностика, не реализованный LLM-метод и не выбор трека/RQ.

## Источник, неизменность и новые правила

- Репозиторий преподавателя: https://github.com/dryjins/RecSys-LLMs.
- **Commit:** `51337032503c56e3a28ef646a24aa016782e1bd3`.
- Отдельный clone: `../../course-starter` относительно папки нашего проекта,
  вне репозитория Дарьи. Исходный starter и `extensions.py` не изменены.
- Проверены HEAD, `git status --porcelain --untracked-files=all` и SHA-256 исходников
  до/после запусков. `starter_unchanged_verified = True`.
  Байткод во внешнем checkout не записывается.
- Полностью прочитаны корневой README, `project/far-trans/README.md`,
  `ANNOUNCEMENT.md`, `instructions/homework2.md`, `starter.py` и identity extension.
- Ответ преподавателя, переданный Дарьей: **previously purchased assets may be
  recommended; task includes repeat and new purchases; keep repeat-allowed
  default; compare against all four provided baselines.**
- Правило повторов утверждено; исторический вариант A из `split.md` — только
  диагностика new-only постановки. Ни его шестимесячное окно, ни старый локальный
  Repeat+Popularity не подменяют нынешние baseline.
- HW2 фиксирует teaching defaults, но README явно не считает их финальным
  grading-протоколом проекта. Полная warm-выборка ниже — дополнительная диагностика,
  не замена обязательному demo HW2.

## Применённый протокол из кода starter

| Параметр | Значение |
| --- | --- |
| Cutoff (history строго до) | 2021-06-01 |
| End exclusive | 2021-07-01 |
| Окно меток | [2021-06-01, 2021-07-01) |
| Eligible assets | 767 |
| CBF candidate-count | 50 |
| Repeat allowed / exclude_seen | True / False |
| Cohort | warm; deterministic selection самого starter |
| Python | 3.13.9 |
| Платформа | Windows-10-10.0.19045-SP0 |
| Процессор | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel |

Eligible = latest asset metadata строго до cut и хотя бы одна положительная цена
строго до cut; не требуется полного покрытия будущего окна ценами.
Профили тоже выбираются строго до cut; warm = есть сохранённая история Buy,
пользователь известен в as-of profiles. Будущие Buy дают **distinct ISIN-set**,
пересечённый с eligible. Отрицательные transactionID исключаются из train и
меток; удаляются только точные полные дубликаты строк.

Popularity — число **разных прошлых покупателей** актива. CBF — cosine similarity
нормализованных бинарных токенов category/subcategory/sector/industry/marketID
с взвешенной частотами Buy историей; названия в scoring не участвуют.
Latest buy / Latest buy or sell упорядочивают по последней прошлой дате
(tie-break ISIN) и дополняют тем же Popularity. Sell не является dislike.
Все четыре baseline ранжируют свой разрешённый **полный universe**; только
ranker-only custom ограничен CBF top-50.

Primary nDCG — по всем выбранным пользователям; empty truth = zero.
Nonempty mean — отдельная диагностика, не подмена primary.

## Два реальных CLI-запуска

Время измерено внешним `perf_counter` для **всего CLI**: загрузка, scoring всех
baseline, экспорт контекстов и файлов. Индивидуальных времён методов starter
не пишет, поэтому одинаковое/выдуманное время по четырём строкам не назначаем.
Запуски последовательные; холодность дискового/OS cache не контролировалась.
Импортная диагностика ниже в это время не включена. Модели/платные API не запускались.

### demo

| Метод | Пользователей | С будущими Buy | Пустые метки | nDCG@10: все (primary) | nDCG@10: nonempty |
| --- | --- | --- | --- | --- | --- |
| Popularity | 100 | 11 | 89 | 0.0286 | 0.2599 |
| CBF | 100 | 11 | 89 | 0.0201 | 0.1827 |
| Latest buy | 100 | 11 | 89 | 0.0277 | 0.2520 |
| Latest buy or sell | 100 | 11 | 89 | 0.0271 | 0.2464 |

| Метаданные запуска | Значение |
| --- | --- |
| Flags сверх data-dir/output-dir | нет (defaults) |
| Started UTC | 2026-11-13T13:04:35.340513+00:00 |
| Finished UTC | 2026-11-13T13:04:43.210225+00:00 |
| Whole CLI wall seconds | 7.7162 |
| protocol_hash | `50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417` |
| Custom failures | 0 |
| Excluded: reconstructed_initial_position_rows | 38019 |

### full_warm

| Метод | Пользователей | С будущими Buy | Пустые метки | nDCG@10: все (primary) | nDCG@10: nonempty |
| --- | --- | --- | --- | --- | --- |
| Popularity | 10404 | 1178 | 9226 | 0.0262 | 0.2312 |
| CBF | 10404 | 1178 | 9226 | 0.0286 | 0.2522 |
| Latest buy | 10404 | 1178 | 9226 | 0.0526 | 0.4648 |
| Latest buy or sell | 10404 | 1178 | 9226 | 0.0509 | 0.4496 |

| Метаданные запуска | Значение |
| --- | --- |
| Flags сверх data-dir/output-dir | --max-users 0 |
| Started UTC | 2026-11-13T13:04:43.235556+00:00 |
| Finished UTC | 2026-11-13T13:05:31.073889+00:00 |
| Whole CLI wall seconds | 47.6697 |
| protocol_hash | `8645343ad810615aed56c65afaff9bf7d0ab4638fbd8005b4d6738b27016ab80` |
| Custom failures | 0 |
| Excluded: reconstructed_initial_position_rows | 38019 |


Полные `protocol.json`, `contexts.jsonl`, `predictions.csv`, `per_user.csv`,
`metrics.json` и console/run metadata сохранены в **игнорируемом**
`data/starter_runs/{demo,full_warm}/`. Они содержат клиентские псевдо-ID и
не коммитятся. Публичный `starter_summary.json` содержит только агрегаты,
хеши и обезличенную проверку. В main таблицы длинные protocol hashes не вынесены.

## Сверка с README именно этого commit

Для demo используем demo-таблицу README. Для `--max-users 0` сравниваем с
**warm paragraph**, а не с большой таблицей `--cohort all`.
Разности метрик считаются после округления наших чисел до точности README.

| Запуск | Метод | Δ users | Δ nonempty | README all | Δ all (4dp) | README nonempty | Δ nonempty (4dp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| demo | Popularity | 0 | 0 | 0.0286 | 0.0000 | 0.2599 | 0.0000 |
| demo | CBF | 0 | 0 | 0.0201 | 0.0000 | 0.1827 | 0.0000 |
| demo | Latest buy | 0 | 0 | 0.0277 | 0.0000 | 0.2520 | 0.0000 |
| demo | Latest buy or sell | 0 | 0 | 0.0271 | 0.0000 | 0.2464 | 0.0000 |
| full_warm | Popularity | 0 | 0 | 0.0262 | 0.0000 | 0.2312 | 0.0000 |
| full_warm | CBF | 0 | 0 | 0.0286 | 0.0000 | 0.2522 | 0.0000 |
| full_warm | Latest buy | 0 | 0 | 0.0526 | 0.0000 | 0.4648 | 0.0000 |
| full_warm | Latest buy or sell | 0 | 0 | 0.0509 | 0.0000 | 0.4496 | 0.0000 |

Несовпадающих строк при совпадающей selection и точности README: **0**.
В таблице all из README — 23571 users / 1208
nonempty; в его warm paragraph — 10404 / 1178.
Эти популяции различаются; расхождение с all-table при запуске warm — не ошибка
воспроизведения. Примерно указанное время README не является переносимым эталоном;
hardware/cache/экспорт и платформа отличаются. Эталонный полный protocol_hash
в README не опубликован — его равенство с чужим запуском не заявляем.

## Запас кандидатов CBF: определения

Все расчёты используют импортом `load_data`, `select_users`, `tokens`,
`popularity_scores`, `contexts_for`, `scan_recent_events`, `recency_rankings`,
`ndcg10`, `aggregate` **самого starter**. Ни один baseline или правило отбора
не переписан. Все четыре импортно пересчитанные метрики сверены с CLI
с допуском абсолютной ошибки; также проверены cohort и source hashes.

Для пользователя u: T = eligible будущие Buy-set, H = ключи сохранённой
Buy-истории до cut, C_k = первые k кандидатов из `context.candidate_assets`.
Repeat targets = T∩H, new targets = T\H.

- Основной знаменатель — пользователи с T непустым. Для repeat/new строк
  берём пользователей, у которых непусто именно соответствующее подмножество;
  число таких пользователей указано, нулевые попадания **не** выкидываются.
- Macro Recall@k = среднее |T_group∩C_k| / |T_group| по этим пользователям.
  Micro Recall@k = сумма попаданий / сумма distinct target-пар.
- Any-hit share = доля таких пользователей с хотя бы одним попаданием,
  не доля всех выбранных пользователей и не event-level recall.
- Будущая покупка в этих метриках — distinct user–asset pair. Дополнительно
  event-level повторность считается разностью cumulative Buy counters двух
  вызовов `load_data` (cut и end), на исходном eligible universe.
  И для строк, и для пар repeat означает наличие Buy **до cut**; несколько
  покупок нового актива внутри окна не переводят его в repeat-класс этого аудита.
- Все future-derived разбиения и oracle — **только post-hoc аудит**, не входы
  deployable encoder/ranker, не подбор ответа и не готовый custom.

### Сводка Recall и any-hit@50

| Запуск | Метки | Users с такими метками | Distinct Buy-пар | Macro Recall@10 | @20 | @50 | Users с ≥1 hit в top-50, доля |
| --- | --- | --- | --- | --- | --- | --- | --- |
| demo | all | 11 | 18 | 0.3333 | 0.3333 | 0.4242 | 0.5455 |
| demo | repeat | 6 | 12 | 0.6111 | 0.6111 | 0.6111 | 0.8333 |
| demo | new | 6 | 6 | 0.0000 | 0.0000 | 0.1667 | 0.1667 |
| full_warm | all | 1178 | 2556 | 0.3878 | 0.4062 | 0.4292 | 0.5475 |
| full_warm | repeat | 834 | 1548 | 0.5391 | 0.5542 | 0.5758 | 0.6727 |
| full_warm | new | 597 | 1008 | 0.1107 | 0.1335 | 0.1595 | 0.2027 |

### Подробные попадания по всем cutoff

| Запуск | Targets | k | Users | Target-пар | Covered-пар | Macro recall | Micro recall | Users ≥1 hit | Any-hit share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo | all | 10 | 11 | 18 | 5 | 0.3333 | 0.2778 | 5 | 0.4545 |
| demo | all | 20 | 11 | 18 | 5 | 0.3333 | 0.2778 | 5 | 0.4545 |
| demo | all | 50 | 11 | 18 | 6 | 0.4242 | 0.3333 | 6 | 0.5455 |
| demo | repeat | 10 | 6 | 12 | 5 | 0.6111 | 0.4167 | 5 | 0.8333 |
| demo | repeat | 20 | 6 | 12 | 5 | 0.6111 | 0.4167 | 5 | 0.8333 |
| demo | repeat | 50 | 6 | 12 | 5 | 0.6111 | 0.4167 | 5 | 0.8333 |
| demo | new | 10 | 6 | 6 | 0 | 0.0000 | 0.0000 | 0 | 0.0000 |
| demo | new | 20 | 6 | 6 | 0 | 0.0000 | 0.0000 | 0 | 0.0000 |
| demo | new | 50 | 6 | 6 | 1 | 0.1667 | 0.1667 | 1 | 0.1667 |
| full_warm | all | 10 | 1178 | 2556 | 730 | 0.3878 | 0.2856 | 565 | 0.4796 |
| full_warm | all | 20 | 1178 | 2556 | 762 | 0.4062 | 0.2981 | 583 | 0.4949 |
| full_warm | all | 50 | 1178 | 2556 | 841 | 0.4292 | 0.3290 | 645 | 0.5475 |
| full_warm | repeat | 10 | 834 | 1548 | 648 | 0.5391 | 0.4186 | 510 | 0.6115 |
| full_warm | repeat | 20 | 834 | 1548 | 665 | 0.5542 | 0.4296 | 521 | 0.6247 |
| full_warm | repeat | 50 | 834 | 1548 | 714 | 0.5758 | 0.4612 | 561 | 0.6727 |
| full_warm | new | 10 | 597 | 1008 | 82 | 0.1107 | 0.0813 | 80 | 0.1340 |
| full_warm | new | 20 | 597 | 1008 | 97 | 0.1335 | 0.0962 | 93 | 0.1558 |
| full_warm | new | 50 | 597 | 1008 | 127 | 0.1595 | 0.1260 | 121 | 0.2027 |

### Повторность будущих покупок — разные знаменатели

| Запуск | Distinct target-пар | Repeat-пар | New-пар | Repeat pair share | Macro repeat share | Buy-строк в окне | Repeat Buy-строк | Repeat row share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo | 18 | 12 | 6 | 0.6667 | 0.5000 | 21 | 13 | 0.6190 |
| full_warm | 2556 | 1548 | 1008 | 0.6056 | 0.6054 | 3493 | 2266 | 0.6487 |

### Какие hits Latest buy вообще доступны ranker-only

| Запуск | Targets | Latest buy hit-пар@10 | Из них есть в CBF50 | Отсутствуют в CBF50 | Доля доступных | Users с потерянным hit |
| --- | --- | --- | --- | --- | --- | --- |
| demo | all | 9 | 5 | 4 | 0.5556 | 3 |
| demo | repeat | 8 | 5 | 3 | 0.6250 | 2 |
| demo | new | 1 | 0 | 1 | 0.0000 | 1 |
| full_warm | all | 1378 | 716 | 662 | 0.5196 | 425 |
| full_warm | repeat | 1298 | 676 | 622 | 0.5208 | 387 |
| full_warm | new | 80 | 40 | 40 | 0.5000 | 40 |

### Oracle ceiling: можно ли в принципе догнать Latest buy?

Oracle переставляет **только исходные CBF top-50**: сначала релевантные кандидаты,
затем остальные, и оценивается `starter.ndcg10` с **полным** T в IDCG.
Ограничение Top-10 сохранено. Это недостижимая без знания будущего верхняя граница,
не прогноз и не эксперимент с LLM. Сравнение только на тех же пользователях:

| Запуск | CBF nonempty | Latest buy nonempty | Oracle50 nonempty | Oracle50 all | Oracle50 − Latest buy nonempty | Mean gap to Latest: possible? |
| --- | --- | --- | --- | --- | --- | --- |
| demo | 0.1827 | 0.2520 | 0.4469 | 0.0492 | 0.1949 | upper bound permits; not a success forecast |
| full_warm | 0.2522 | 0.4648 | 0.4544 | 0.0514 | -0.0105 | NO, with fixed pool and protocol |

**demo:** потолок допускает догнать Latest buy, но успех LLM не доказан (oracle 0.4469, Latest buy 0.2520). Из 9 релевантных hit-пар Latest buy в CBF50 доступны 5; потеряны 4.

**full_warm:** догнать Latest buy в среднем невозможно одним переставлением исходного CBF50 (oracle 0.4544, Latest buy 0.4648). Из 1378 релевантных hit-пар Latest buy в CBF50 доступны 716; потеряны 662.

Если oracle ниже Latest buy, ни выбор LLM, ни перестановки промпта не закроют
средний разрыв при неизменном pool. Если выше — это только необходимая возможность:
LLM ещё должен почти правильно упорядочить имеющиеся targets. Переход к более
широкому pool или encoder меняет постановку и требует новой честной baseline-таблицы.

## Ограничения encoder seam и контекста, измеренные на этих cohorts

| Запуск | Users | Один distinct history-актив | Один history feature vector | Из них nonempty | Profile questionnaire date ≥ cut |
| --- | --- | --- | --- | --- | --- |
| demo | 100 | 40 | 41 | 0 | 13 |
| full_warm | 10404 | 4491 | 4755 | 131 | 1318 |

При единственном ненулевом feature-векторе у всей истории любые неотрицательные
веса с положительной суммой дают то же направление нормированного CBF-профиля.
Такой encoder не может менять ranking, кроме all-zero → Popularity fallback.
Это ограничение интерфейса, не доказательство бесполезности настоящих embeddings.
Идентичные content-векторы разных assets также не различаются этим score;
tie-break остаётся ISIN.

В `context.history` есть `buy_count`, но **нет last-buy timestamps**; список
сортирован по частоте, а `metadata_timestamp` — дата metadata, не дата покупки.
Для recency-aware весов потребуется отдельное past-only обогащение в своём adapter
с явной проверкой одинаковой информации у контролей. Внешний starter не меняем.
Стартерский профиль не передаёт `lastQuestionnaireDate`, поэтому его as-of отбор
не устраняет обнаруженные прежде проблемы provenance. В baseline профили display-only
и на scoring не влияют; потенциальный custom должен учитывать этот риск.

## Проверки и хранение доказательств

- Импортная генерация совпала со всеми четырьмя сохранёнными CLI aggregate metrics;
  хеш протокола пересчитан из его canonical JSON и проверен; source CSV hashes сверены.
- Для обезличенного positive пользователя отдельно вычислены discount contributions
  Latest buy и IDCG вручную и сверены со `starter.ndcg10`; детали и отсутствие
  userID сохранены в `starter_summary.json`. Проверку выполнил AI-ассистент,
  **не выдаём её за личную проверку Дарьи по HW2**.
- Подтверждены default repeat policy, warm selection, число кандидатов,
  границы окна и неизменность внешнего checkout. Полное повторное чтение loader
  для диагностики не входит в измеренное CLI время.
- Модели ещё не запускались: нет custom-строки, no claim about HW2 completion.
  HW2 требует open-weight модель и рабочий **реальный** вызов инференса, а не
  identity plugin, экспорт промпта или oracle.
