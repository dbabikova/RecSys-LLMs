# Ranker RQ v2 — когда помогает permutation aggregation

Дата: 2026-11-20. Уточнение предложено Codex на основе указания преподавателя
и согласовано Дарьей в чате («согласовано»). Это дизайн будущего опыта, не результат.

## Исследовательский вопрос

**Улучшает ли агрегация трёх перестановок кандидатов качество zero-shot
LLM-реранжирования на FAR-Trans по сравнению с исходным порядком при одинаковом
числе вызовов, лимитах генерации и фиксированном CBF20?**

- Intervention: три перестановки одних и тех же 20 кандидатов и агрегация Top-10.
- Comparator: три вызова с исходным порядком и та же агрегация.
- Outcome: разность macro nDCG@10 по всем 100; nonempty mean и N отдельно.
- Population: те же зафиксированные 100 пользователей Алины/Ульяны.
- Classical control: same-information LightGBM LambdaMART на том же CBF20.

Это конкретное условие (порядок сериализации) внутри выбранного Ranker direction.
Сравнение LLM−LambdaMART остаётся важным, но само по себе больше не является
основным исследовательским вкладом этой работы.

## Литературная основа и предоставленный baseline

[Hou et al., LLMRank, ECIR 2024, §3.2](https://arxiv.org/html/2305.08845v2)
сообщают о position bias на MovieLens-1M/Amazon Games и проверяют три случайные
перестановки с последующим объединением выдач. Приём известен: мы не предлагаем
его как новый метод, а проверяем переносимость в FAR-Trans с repeat purchases,
бедными metadata, фиксированным pool и ограниченным inference budget.

[Пакет преподавателя, закреплённый commit 799e54c](https://github.com/dryjins/RecSys-LLMs/tree/799e54c0493cf7e2b014152548ae5c18ee24a8f5/project/far-trans-llm)
содержит Qwen3-1.7B Ranker с past Buy/recency prompt и поддержкой shuffle seed,
но без multi-permutation aggregation. В опубликованной таблице Ranker с fallback
даёт 0.0201 / 0.1827, включая 22/100 invalid fallbacks. Это результат преподавателя
на CBF50, не наш inference и не измерение CBF20.

В рамках нашей проверки все 26 unit tests этого пакета прошли; data adapter
выбрал точные user IDs/order, catalog и CSV hashes нашего протокола, классические
baseline совпали. Были подготовлены prompts без модели. Published LLM metrics
не воспроизводились: генерации, кеши и user-level LLM outputs не поставляются.
Равенство средних Ranker/CBF не доказывает равенство отдельных выдач.

## Варианты и неизменные параметры

| Вариант | Роль |
| --- | --- |
| Четыре baseline Алины | Общие обязательные ориентиры на полном universe |
| Исходные первые 10 из CBF20 | Простой reference и единый fallback |
| Same-information LambdaMART | Главный классический контроль |
| LLM original-order ×3 + aggregation | Контроль одинакового числа calls |
| LLM permutations ×3 + aggregation | Основное вмешательство |

В LLM-паре одинаковы модель/revision, semantic prompt, history/profile/candidate
fields, parser, generation limits, aggregation, invalid policy и cohort.
Меняется только presentation order candidate list; исходный rank, если включён
в feature contract, остаётся свойством candidate и не переписывается при shuffle.
Перестановки — seeds 17, 42, 2026. Алгоритм shuffle фиксируется до inference.
Агрегация: rank points 10…1, отсутствующим ID 0, tie-break ISIN.

Invalid call использует **исходный CBF20 Top-10**, а не первые 10 переставленного
prompt. Одинаковый fallback исключает подмену эффекта LLM эффектом shuffled fallback.
Флаги остаются на каждом вызове и пользователе; все 100 users остаются в primary.

При greedy decoding три original-order вызова могут совпасть. Это не три
независимых пользователя и не доказательство статистической устойчивости.
Для одинакового call budget нужны реально исполненные вызовы; повторное чтение
кеша не выдаём за три inference. Если ресурсы требуют cache replay, отдельно
обозначим эту границу и не будем утверждать equal measured compute.
Число calls и token caps одинаковы; фактические tokens/runtime могут различаться.

## Как интерпретируем

- Primary: paired difference permutation-aggregate minus original-order-aggregate.
- Secondary: обе ветки относительно LambdaMART и четырёх baseline, runtime/tokens,
  invalid/fallback, различия исходных выдач между presentation orders.
- Положительный gain означает пользу данного вмешательства в этих условиях.
  Он сам по себе не доказывает, что position bias устранён; это требует отдельного
  измерения order sensitivity. Одного aggregated списка недостаточно, чтобы
  объявить доказанной его устойчивость ко всем перестановкам.
- Разделяем изменения валидного ranking и различия invalid/fallback. Valid-only
  анализ не заменяет primary на всех пользователях.
- Нулевой/отрицательный результат допустим. Июньские labels уже просмотрены:
  RQ — проверка, мотивированная литературой, а не preregistration на untouched test.

Users, cutoff, catalog, CBF20 и candidate hash остаются неизменными. Stage 2
diagnostics ограничивают вывод: всего 11 nonempty users, у 6 нет цели в pool.
Уточнение RQ не является разрешением подбирать другой pool по июньской truth.
Модель, prompt и feature contract ещё не выбраны; они фиксируются на следующих
согласованных этапах. Изменение документов не закрывает личную проверку этапа 2.
