# Сравнение треков в интерфейсах официального starter

> Сгенерировано `scripts/starter_reports.py` из измеренных aggregate/diagnostics.
> Литературные ссылки [n] относятся к [literature.md](literature.md).
> Трек и исследовательский вопрос **не выбираются**: решение за Дарьей.

Starter commit: `51337032503c56e3a28ef646a24aa016782e1bd3`.
Повторы официально разрешены и релевантны; baseline comparison включает
Popularity, CBF, Latest buy, Latest buy or sell. HW2 defaults — teaching settings,
не финальный grading protocol.

## Измеренная стартовая точка

| Cohort | Users / positive | CBF: all / nonempty | Latest buy: all / nonempty | Oracle fixed CBF50: all / nonempty | Whole CLI wall, s |
| --- | --- | --- | --- | --- | --- |
| demo | 100 / 11 | 0.0201 / 0.1827 | 0.0277 / 0.2520 | 0.0492 / 0.4469 | 7.7162 |
| full_warm | 10404 / 1178 | 0.0286 / 0.2522 | 0.0526 / 0.4648 | 0.0514 / 0.4544 | 47.6697 |

| Запуск | Метки | Users с такими метками | Distinct Buy-пар | Macro Recall@10 | @20 | @50 | Users с ≥1 hit в top-50, доля |
| --- | --- | --- | --- | --- | --- | --- | --- |
| demo | all | 11 | 18 | 0.3333 | 0.3333 | 0.4242 | 0.5455 |
| demo | repeat | 6 | 12 | 0.6111 | 0.6111 | 0.6111 | 0.8333 |
| demo | new | 6 | 6 | 0.0000 | 0.0000 | 0.1667 | 0.1667 |
| full_warm | all | 1178 | 2556 | 0.3878 | 0.4062 | 0.4292 | 0.5475 |
| full_warm | repeat | 834 | 1548 | 0.5391 | 0.5542 | 0.5758 | 0.6727 |
| full_warm | new | 597 | 1008 | 0.1107 | 0.1335 | 0.1595 | 0.2027 |

На полной warm-выборке повторных distinct future-пар — **0.6056**,
а повторных Buy-строк — **0.6487**. Это не прежняя общая
event-level доля по всему релизу. Полные знаменатели, micro/macro и Latest-hit
overlap — в [starter_baselines.md](starter_baselines.md).

**Ограничение full warm:** oracle исходного CBF50 =
**0.4544** nonempty nDCG@10 против
**0.4648** у Latest buy.
При неизменном pool и протоколе ranker-only не может догнать его в среднем.
На demo oracle =
**0.4469**, то есть
выше demo Latest buy
**0.2520**; это лишь
возможность на маленькой positive-подвыборке, не свидетельство общего выигрыша.

## Возможности, ограничения и шанс против recency

| Трек | Что реально меняется в stock API | Что говорит литература | Шанс относительно Latest buy | Риски | Стоимость и время |
|---|---|---|---|---|---|
| **Ranker** | `rerank(context, candidate_ids)` возвращает ровно десять уникальных ID из CBF top-50; меняет выбор/порядок внутри pool, не retrieval и не universe | [1], [2]: prompting/verbalizer и candidate ceiling; [3]: candidate policy и сильный non-LLM контроль критичны; [5] содержит пример ухудшения после reranking | Ограничен **измеренным oracle**. Недоступные CBF50 hits Latest buy нельзя вернуть любым LLM. Если oracle выше baseline, это возможность, не вероятность успеха. Demo и full warm не эквивалентны | Position bias, ties по category вместо интереса, неверные/повторные ID, внепуловые ответы, шум профиля и pretrained-name contamination; частотная история не содержит событийной recency | Обычно один actual LLM-вызов на пользователя; перестановки/ensemble умножают инференс и токены. Можно batch/cache, но надо учитывать полный workload. Измерений конкретной LLM ещё нет, GPU-часов и секунд не выдумываем |
| **Encoder** | `encode(context)` возвращает неотрицательные конечные веса **только history IDs**. Старый CBF `content_scores` превращает их в один нормированный профиль и ранжирует весь eligible universe | [4] — настоящие language representations; [5] — alignment с CF. Stock weights seam **не эквивалентен** этим embedding-моделям; [15] мотивирует сильный frequency/recency контроль | Может изменить retrieval и вернуть targets вне исходного CBF50, если веса меняют профиль. Но не различит одинаковые content-векторы; при одном векторе история неизменна при ненулевых весах. Recency не из чего восстановить без доп. past-only данных | Неправильное описание weights как “LLM embeddings”; ложные предпочтения; переобучение на малой positive-подвыборке; дополнительные поля только у LLM | Возможен один offline LLM-вызов/пользователь, затем дешёвый CBF; profile и weights cache должны зависеть от cut/model/prompt. Настоящие embeddings требуют своего score adapter, отдельной реализации и честного контроля, не замены официальных baseline |
| **Hybrid** | Plugin с `encode` и `rerank`: сначала encoder ranking по полному universe, **его** top-50 передаются ranker; исходный CBF pool нельзя незаметно подмешать обратно | [5], [6] — complementary behavioral/semantic signals; [3] — рост от retriever нельзя приписать reranker. Это не гарантирует эффект двух LLM-стадий | Может обойти потолок исходного CBF50 через новый retrieval. Затем остаётся потолок нового pool, который нужно заново измерить. Без выигрыша encoder дорогой ranker может лишь наследовать ограничения | Смешение retrieval/ranking effects, утечки в двух частях, информационное неравенство, invalid outputs, рост инженерного объёма | Encoder-вызовы + ranker-вызовы, больше cache/versioning и минимум ablations baseline/encoder-only/ranker-only/both. До измерений качество/latency лучше не обещать |

### Структурная граница weights seam

| Cohort | Всех users | История с одним feature vector | Из них positive |
| --- | --- | --- | --- |
| demo | 100 | 41 | 0 |
| full_warm | 10404 | 4755 | 131 |

Положительное масштабирование одного и того же нормированного вектора не меняет
его направление, поэтому такие пользователи не получают новый CBF ranking только
от перестановки весов. `buy_count` отражает частоту; поле `metadata_timestamp`
не отражает последнюю покупку. Latest buy считывает реальные past event dates
отдельно через `scan_recent_events`; считать эту информацию доступной LLM
из stock context нельзя. Это важно и для feasibility, и для same-information.

## Как проверить каждый трек, не выбирая сейчас один

1. **Ranker-only:** одинаковый fixed CBF50, все HW2 users, четыре baseline,
   pool validation, fallback/error и число failed cases, runtime и decoding repeats.
   Ground truth для oracle остаётся только в диагностическом коде.
2. **Encoder-only:** показать, какие веса изменила реальная модель и как это меняет
   scoring; сравнить с identity и детерминированными reweighting-контролями.
   При embeddings вне seam нужно реализовать путь vector→scores в своём коде
   и сохранить полный официальный baseline/evaluator protocol.
3. **Hybrid:** baseline, encoder-only, ranker-only и both. Для анализа ранкера
   сравнить его с исходным порядком именно нового encoder pool, а не только с CBF;
   отдельно показать candidate recall. Изменение pool не выдавать за чистое reranking.
4. Все варианты получают только past-only разрешённые данные. Нельзя выбирать
   пользователей по удачному попаданию candidates или выкидывать failed output.
   Nonempty evaluation нужна как вторичная диагностика, не как новая primary.

## Требования HW2 и границы выполненного шага

- **Open-weight publicly available model** и runnable inference обязательны.
  Зафиксировать model repository, revision, prompt, dependencies, generation
  settings и hardware; local/notebook/cloud разрешены. Конкретную модель сейчас
  не выбираем. Proprietary лидеры из [3] не заменяют open-weight требование.
- Identity `extensions.py`, prompt export и oracle **не являются LLM inference**.
  Нужен фактический вызов модели и сохранённые outputs с protocol_hash, либо
  реальный `encode`/`rerank` plugin. Стоковый starter сам модель не вызывает.
- Zero-shot достаточно; few-shot требует training-only demonstrations до cut,
  не evaluation users и не их будущие Buy.
- Положительный результат HW2 не обязателен. Нужны рабочая интеграция,
  выполненная таблица четырёх baseline + custom, runtime/failed cases и конкретная
  личная проверка Дарьи. Этот этап реализует baseline/diagnostics, **не закрывает HW2**.
- Учебные default настройки и policy повторов теперь известны; окончательный
  проектный тест нельзя объявлять утверждённым по одному README starter.
- Full-period `profitability`, поздние анкеты, реконструкции и provenance
  метаданных из `leakage.md` остаются рисками. Starter исключает negative IDs,
  но display-only profile не становится доказанно point-in-time корректным.
- Ни успешный nDCG, ни convincing explanations не доказывают suitability/profit.

**Вывод без выбора:** Ranker — наиболее локальное изменение, но жёстко ограничено
кандидатами; Encoder меняет доступность кандидатов, но stock weights API беднее
настоящих представлений; Hybrid способен менять обе части, но увеличивает
стоимость и требует раздельной атрибуции. Числа выше — основания для решения
Дарьи, а не автоматическое назначение трека или RQ.
