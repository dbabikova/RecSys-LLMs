# ДЗ 2 / Session 2 — Ranker: зафиксированный дизайн

Автор: Бабикова Дарья, Team 5. Дата: 2026-11-20.
Этап 1 подготовлен с помощью Codex после подтверждения плана Дарьей.
Этап 1 принят Дарьей («ок» в чате), commit 0dc0f9266646ec58fc8caf0082d1f3fbcc34c679
отправлен в main личного репозитория. Этап 2 принят Дарьей («ок» после запуска
скриптов Codex в PowerShell); самостоятельная ручная проверка Дарьей не заявляется.
LambdaMART обучен на этапе 5. LLM inference выполнен только в early technical
pilot этапа 6; полный июньский experiment пока не запускался.
Этап 3 принят Дарьей: контракт common_input_v1 и автоматические проверки —
STAGE3.md / INPUTS.json; самостоятельная ручная raw-сверка не заявляется.
Этап 4 принят Дарьей: temporal train/validation и fitted преобразования — STAGE4.md.
Этап 5 принят Дарьей: LambdaMART обучен, выбран по May validation и зафиксирован
в MODEL_FREEZE.json. Июньская оценка не запускалась.
Самостоятельная ручная raw-сверка не заявляется.
Этап 6 принят Дарьей: Qwen3-1.7B CPU float32, short wire IDs, шесть real pilot calls;
результаты и автоматические проверки — STAGE6.md.

## Вопрос и границы

**Согласованный уточнённый RQ (v2):** улучшает ли агрегация трёх перестановок
кандидатов качество zero-shot LLM-реранжирования на FAR-Trans по сравнению
с исходным порядком при одинаковом числе вызовов, лимитах генерации
и фиксированном CBF20?

Primary contrast — LLM permutation aggregation против LLM original-order
aggregation. LightGBM LambdaMART остаётся главным классическим same-information
контролем. Исходный вопрос ДЗ 1 «LLM против LambdaMART» сохраняется как secondary
comparison. Обоснование литературой и arms — RQ.md. Дарья согласовала уточнение
в чате после пояснения преподавателя о вопросах when/why/under what conditions.

Это продолжение личного Ranker RQ из сданного ДЗ 1. Общий вопрос Team 5 координирует
три вмешательства: история пользователя (Алина), представление активов (Ульяна),
перестановка кандидатов (Дарья). Полный Hybrid на этом этапе не реализуем.
Результаты Encoder участниц не являются входами основного опыта Дарьи.

## Общая основа

| Параметр | Зафиксировано |
| --- | --- |
| Данные | FAR-Trans; четыре CSV и их SHA-256 из источников пакета Алины |
| Cohort | Те же 100 warm users, тот же порядок, без отбора по будущим покупкам |
| Cutoff | 2021-06-01; история и as-of metadata строго раньше cutoff |
| Truth | Distinct eligible Buy в [2021-06-01, 2021-07-01), 30 дней |
| Universe | 767 активов; metadata и положительная close price до cutoff |
| Повторы | Разрешены и релевантны; exclude_seen=false |
| Очистка | Exact full-row duplicates; negative transactionID исключены |
| Выход | Top-10 уникальных допустимых ID для каждого пользователя |
| Primary | Macro nDCG@10 по всем 100; empty truth = 0 |
| Secondary | nDCG@10 по nonempty truth с количеством users; в пакете их 11 |

Это teaching protocol ДЗ 2, не окончательный проектный тест. Исходный protocol.json
с candidate_count=50 сохраняем неизменным. Источники и полные хеши — SOURCES.json.

## Наши экспериментальные условия

1. Основной pool — **первые 20 из сохранённых CBF50 Алины**, в исходном порядке.
   Это подтверждённое Дарьей решение. Pool материализован на этапе 2;
   policy, candidate hash и candidate manifest hash — CANDIDATES.json.
2. LLM и LambdaMART получают одинаковые 20 candidates и одинаковый набор
   доступных исходных полей. Четыре provided baseline не обрезаем этим pool.
3. Исходный Top-10 общего pool — дополнительный reference. LambdaMART — главный
   классический контроль. Для уточнённого RQ основная пара — две LLM-ветки,
   различающиеся только порядком представления кандидатов (RQ.md).
4. Формат текста и признаки могут различаться, но каждому полю LLM должно
   соответствовать явно реализованное представление для LambdaMART. Названия
   нельзя заменить у контроля одними ID. Recency при добавлении доступна обоим.
5. Изменения контекстов, profile masking и history truncation документируем
   симметрично. Сырые customerID не передаём в LLM prompt.
6. LambdaMART обучаем на ранних срезах с тем же правилом CBF20. Train и validation
   label windows заканчиваются до 2021-06-01. Конкретные окна ещё открыты.
7. LLM — публичная open-weight модель, actual runnable zero-shot inference.
   Модель/revision, hardware, prompt и decoding фиксируются до основного опыта.
8. Invalid LLM response → исходный Top-10 **того же pool**, с флагом ошибки.
   Users остаются в primary metric. Parser, retries и сбои процесса будут
   специфицированы до запуска; неполный запуск не объявляется полным результатом.
9. Основное вмешательство: 3 candidate permutations (17, 42, 2026) против
   3 вызовов с исходным порядком; одинаковая агрегация по rank points 10…1,
   отсутствующим ID 0, tie-break ISIN. Same model/prompt/decoding/output budget.
   5 LambdaMART seeds (0…4) — диагностика классического контроля. Точный бюджет
   и настройки сохраняем до основного запуска; cache hits не считаем новыми calls.

Для derived experiment будет отдельный manifest: source protocol hash, pool
policy/hash, feature contract, code/model revisions, settings и seeds. Нельзя
обозначать изменённый pool только исходным protocol hash как идентичный CBF50.

## Общие результаты и атрибуция

Baseline-пакет предоставила Алина; экспериментальный commit
aa32f8d34310da25306acb4e16a508a3b6cd9e89 указан ею как локальный. Его доступность
в GitHub в рамках этого этапа не подтверждена. Пакет технически воспроизведён нами,
но это не свидетельство согласия всей команды считать прогон общим.

| Метод | Users | Nonempty users | nDCG@10 all | nDCG@10 nonempty |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 100 | 11 | 0.0286 | 0.2599 |
| CBF | 100 | 11 | 0.0201 | 0.1827 |
| Latest buy | 100 | 11 | 0.0277 | 0.2520 |
| Latest buy or sell | 100 | 11 | 0.0271 | 0.2464 |

Полная точность и источник — SOURCES.json. Замер Алины 9.041475740000124 s относится
ко всему subprocess четырёх baseline, включая загрузку, оценку и запись файлов.
Это не per-method runtime и не скорость наших будущих методов на другом устройстве.

Дарья: RQ и решения о дизайне, руководство сравнением, будущая интерпретация и
личная проверка. Codex под руководством Дарьи: подготовка документов, код,
автоматические проверки; далее реализация моделей и анализ с раскрытием AI usage.
Starter и evaluator — преподавателя; исходный baseline-прогон — Алины.
Encoder-работы Алины и Ульяны используем в discussion с их авторством.

## Проверки и критерий завершения этапа 1

Скрипт scripts/verify_stage1.py читает локальные ZIP и starter, сверяет SHA-256,
protocol hash и CSV hashes, воспроизводит четыре baseline, сравнивает все outputs
и независимо считает truth/nDCG по raw transactions без импорта starter.
Отчёт VERIFICATION.json содержит агрегаты без customerID и пользовательских выдач.
Команды и личный checklist — REVIEW.md.

Проверки AI не выдаём за личную проверку Дарьи. Личная сверка документов этапа 1
и будущая ручная проверка одного model result — отдельные действия.
Commit выполняется только после проверки и «ок» Дарьи; затем сразу push в main
личного репозитория по её последующему указанию. Командный репозиторий не меняем.
VERIFICATION.json сохраняет исходный статус автоматической проверки этапа 1;
он не заменяет последующее подтверждение Дарьи в чате.

## Открытые решения и следующие этапы

- Этап 2 принят: CBF20, reference и хеши — CANDIDATES.json;
  результаты, границы автоматической проверки и инструкция — STAGE2.md.
- Этап 3 принят: общий feature contract, availability/profile rules и
  unfitted feature records; text/categorical fit отложен до temporal training.
- Этап 4 принят: train cutoffs 2021-01-01 / 2021-03-01 / 2021-04-01,
  validation cutoff 2021-05-01, 30-дневные labels и train-only encoders (STAGE4.md).
- Этап 5 принят: LambdaMART leaves15_trees100, primary seed 0;
  May validation и model freeze — STAGE5.md / MODEL_FREEZE.json.
- Этап 6 принят: Qwen3-1.7B pinned revision, CPU float32, prompt/parser,
  call budget и early technical pilot — STAGE6.md / LLM_METHOD_FREEZE.json.
- Этапы 7–8: полный опыт, permutations/seeds, проверка и интерпретация.
- Этап 9: индивидуальный PDF и session export, experiment commit и публикация.

## Ограничения и отложенные расширения

CBF20 может не содержать целевые активы. Candidate recall/oracle — диагностика,
не информация для модели. Repeat-aware pool и retrieval Ульяны — будущие отдельные
опыты; по июньским labels наш основной pool не меняем. Июньское окно уже изучалось
в ДЗ 1, поэтому не называем его нетронутым holdout и не настраиваем по нему модель.
Сохраняются provenance-риски FAR-Trans, поздние анкеты и знания LLM после 2021.
nDCG оценивает совпадение с покупками, не прибыль и не suitability.
Full warm, полный Hybrid, четыре информационные абляции, RAG и fine-tuning
не входят в минимальный объём этого ДЗ 2.
