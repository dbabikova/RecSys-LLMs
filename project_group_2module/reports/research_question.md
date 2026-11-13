# Исследовательский вопрос — решение Дарьи

> Сгенерировано `scripts/make_ranker_example.py`. Формулировка — дословное решение
> Бабиковой Дарьи, переданное в чате, а не выбор AI-ассистента.

## Primary research question

**We test whether zero-shot LLM reranking of a fixed top-20 candidate list improves nDCG@10 over a classical learned reranker (LightGBM LambdaMART) that sees the same user and asset information and the same candidates, for the common FAR-Trans evaluation users.**

**Track: Ranker. Status: planned.** Модели, LambdaMART и LLM inference на этом
этапе не запускались; конкретная LLM и generator/pool policy не выбираются.

| Элемент | Зафиксировано |
| --- | --- |
| Intervention | Zero-shot LLM reranking фиксированного top-20; изменяется порядок/выбор Top-10 внутри одного и того же списка кандидатов |
| Control | Classical learned reranker: LightGBM LambdaMART; те же пользователи, кандидаты и user/asset information, доступная до среза |
| Outcome | Средний nDCG@10 по будущим покупкам; primary по всем common evaluation users с empty truth = 0. Nonempty mean — отдельно |
| Population | Common FAR-Trans evaluation users. Для учебной иллюстрации — demo cohort самого starter; финальную проектную популяцию не подменяем выбранным примером |

## Почему главный контроль — same-information LambdaMART

Исходный порядок кандидатов отражает только используемый retriever/scorer.
Если LLM получает профиль и атрибуты, а контроль — лишь этот порядок, выигрыш
может объясняться дополнительной информацией, а не способом её обработки.
LambdaMART должен получать те же исходные поля (включая доступный текст через
явно реализованное text→features преобразование) и тот же pool. Обучение и
настройка контроля требуют более ранних временных окон, не меток demo/test.
Zero-shot означает отсутствие настройки весов LLM на этой задаче; learned
контроль при этом не становится zero-shot.

Исходный порядок candidate generator остаётся полезным **reference**, но не
достаточным главным контролем. Любые дополнительные past-only поля или summaries
должны быть симметрично доступны обоим методам. Нельзя назвать raw names
«той же информацией», если на практике non-LLM модель получила только их ID.
Конкретные признаки, text encoding, train/validation windows, prompt, модель,
бюджет и permutation checks уточняются в study plan; они здесь не реализованы.

## Обязательные дополнительные baseline

В одной таблице остаются **все четыре реализации starter**:
Popularity, CBF, Latest buy, Latest buy or sell, плюс LambdaMART и будущий LLM
custom. Сохраняются общие users, cut, eligibility, метки и scoring. Четыре
baseline не ограничиваются задним числом нашим reranking pool.
Повторные и новые покупки релевантны; A new-only — только диагностика.

## Candidate generator/pool: открытый вопрос, не решение

Текущий диагностический генератор — **CBF официального starter**.
Его default pool и иллюстративный top-20 ниже не утверждают финальный выбор пула.
Top-20 — размер списка из принятого RQ; содержимое, generator, параметры и
обоснование opportunity остаются открытыми. Ни pool, ни модель сейчас не выбираем.

| Риск из этапа starter | Измерено (nonempty nDCG@10) |
| --- | --- |
| Oracle перестановки исходного CBF top-50 | 0.4544 |
| Latest buy на тех же full warm users | 0.4648 |

Таким образом, даже идеальная перестановка исходного CBF50 на этой warm-выборке
не догоняет Latest buy. Для его вложенного CBF20 ограничение не слабее.
Это **риск feasibility и открытый вопрос к выбору generator/pool**, а не решение
сменить CBF, расширить список или отменить Ranker. Проверка LLM против LambdaMART
внутри фиксированного pool остаётся отличной от сравнения с global Latest buy;
оба результата нужно честно показать, а не обещать победу над всеми baseline.

## Связь с данными и статус

Реальная запись и её source-linked transformation: [ranker_example.md](ranker_example.md).
Независимая проверка и инструкция личной проверки: [verify_example.md](verify_example.md).
Литература: [literature.md](literature.md), особенно [1]–[3], [5], [9], [15].
Ближайший доменный LLM-предшественник — FLARKO; его Top-3 hit-style оценка не
отвечает этому Top-10 same-information controlled comparison.
Нулевой/отрицательный результат допустим; nDCG не измеряет suitability/profit.

Личная ручная проверка Дарьи **ещё не выполнена**. Отчёт не является завершённым
экспериментом, реализованным HW2 custom или командным согласованием.
