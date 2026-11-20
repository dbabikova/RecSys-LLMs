# Этап 6 — Qwen Ranker: backend, parser и технический пилот

Статус: Дарья приняла этап («ок» в чате). Автоматические результаты пилота —
LLM_PILOT.json и STAGE6_VERIFICATION.json. Самостоятельная ручная raw-сверка
не заявляется; JSON-отчёт сохраняет статус на момент запуска. Июньский полный прогон и
nDCG LLM на этом этапе не вычисляются.

## Модель и ресурсы

Выбрана публичная Qwen/Qwen3-1.7B, revision
70d244cc86ccca08cf5af4e1e306ecf908b1ad5e — та же модель/revision, что у преподавателя.
Исходный кандидат 8B из ДЗ 1 не использован. CPU Intel i7-10610U, 4 physical /
8 logical cores, 31.19 GiB RAM; CUDA GPU нет (только integrated Intel UHD).
В момент hardware check свободно 18.36 GiB. Реальные peak memory/timing — в pilot JSON.

Веса downloaded из публичного репозитория без token и без отправки пользовательских
данных. Download metadata commit проверен; SHA-256 safetensors совпал с remote
etags. Файлы/размеры/хеши — LLM_MODEL.json, сами веса только в ignored data/.
Inference строго local_files_only=True, trust_remote_code=False; сеть не нужна.

Изолированное .venv-hw2-llm, Python 3.13.9, torch 2.11.0 CPU, transformers 4.57.6,
numpy 2.3.5. Полные pins — requirements-llm-lock.txt, hardware/env — LLM_ENVIRONMENT.json.
Изначальное наследование Anaconda вызвало OpenMP Error #15; окружение переведено
в include-system-site-packages=false с PyPI NumPy. Ошибка импорта устранена;
KMP_DUPLICATE_LIB_OK не включался. Глобальные библиотеки не менялись.

## Prompt и обратимое кодирование

Все semantic fields — из common_input_v1, никаких внешних новостей/фактов,
демонстраций, labels или новых user/asset features. Zero-shot English instruction
требует ровно 10 candidate aliases JSON-массивом. Явно разрешены repeats;
нет задачи предсказывать profit или suitability. Names/JSON объявлены данными,
не инструкциями. Exact instruction и columns — scripts/llm_ranker.py.

Для экономии токенов history/candidates сериализованы таблицами с однократными
column headers. Значения полей не сокращаются, history selection не меняется.
Round-trip тест и независимый verifier восстанавливают общий payload из таблиц.

**Техническая коррекция служебных IDs:** длинные canonical hash aliases этапа 3
сохраняются в данных, LambdaMART join и scorer. Только внутри prompt им сопоставлены
A01…A40 по sorted canonical aliases, присутствующим в history/candidates пользователя.
Mapping одно и то же для всех шести calls. Оно не основано на исходном rank,
labels или presentation order. History↔candidate связи сохранены.
После строгой валидации ответ однозначно decoded в canonical aliases, затем ISIN.
Никакого fuzzy matching/угадывания/исправления model IDs не выполняется.
Этот служебный mapping не меняет candidate membership и не добавляет информацию.

## Две ветки и generation settings

1. Original: три настоящих вызова с исходным CBF20 order.
2. Permuted: три настоящих вызова, seeds 17/42/2026, с stage-3 SHA-based order.

Seed permutations применяются к canonical candidates **до** wire-ID encoding.
Меняется только candidate row order; source_cbf_rank/score и прочие поля остаются
прикреплёнными к candidate. CBF20 и все 100 июньских пользователей неизменны.

Одинаковые settings: CPU float32, threads=4, interop=1, sdpa, eval/inference mode,
greedy do_sample=False, enable_thinking=False, deterministic algorithms, torch seed=0.
Max output=256 tokens, total context cap=16384, truncation=False.
KV cache только **внутри** call; готовые ответы и prefix states между calls не
переиспользуются. Три одинаковых prompts — три реальные генерации.
Одинаковые caps/call count не означают одинаковые фактические tokens или runtime.

Strict parser принимает только JSON array из 10 уникальных строк — разрешённых
wire aliases кандидатов. Markdown/code fences, explanations, duplicates, short
lists, history-only aliases и произвольные IDs невалидны. No repair/regeneration.
Invalid response → original unpermuted CBF20 first10, одинаково в обеих ветках.
Context overflow/OOM/backend error останавливает run как incomplete, а не маскирует
отсутствие inference fallback-выдачами. Частичные logs сохраняются.

Агрегация обеих веток: 10…1 rank points, absent=0, tie-break по original ISIN
через private mapping. Результат — canonical aliases для дальнейшего scorer.
Речь о переносимости известного LLMRank bootstrapping, не новом алгоритме.

## Пилот и сохранённая неудачная попытка

Пилот использует только train_2021-04-01_payloads.jsonl.gz. Все 1000 historical
prompts токенизированы; выбран один на медианной длине, tie-break user_alias.
Ни файлы labels/mapping с truth, ни июньские данные runner не читает. Для ISIN
tie-break достаточно asset IDs из metadata строго до cutoff. nDCG не считается.

Первый draft с длинными hash aliases дал 2 завершённых невалидных ответа из-за
ошибок копирования IDs; следующая generation была прервана. Он сохранён как
**incomplete**, а не успешный six-call pilot. Raw logs/config/code остались в
data/hw2/llm_pilot_long_alias_draft/, агрегированная запись — LLM_DRAFT_DIAGNOSTIC.json.
Завершённое generation time draft=203.3025 s; interrupted-call стоимость полностью
не измерена и не выдумывается. Correction по format feasibility на early data,
не по purchase labels и не по июньским результатам.

После обратимого short-ID encoding запущен отдельный six-call pilot. Для его
1000 early prompts: min=1265, median=1768, max=2620 tokens, overflow=0.
Четыре presentation variants выбранного пользователя имеют по 1768 input tokens.
Итог короткого pilot (LLM_PILOT.json): **6/6 valid, 0 invalid, 0 fallback**,
10608 input и 378 output tokens. Generation time=366.860 s, mean=61.143 s/call;
весь pilot с hash checks/load/tokenization=390.458 s. Peak working set=10693025792
bytes (около 9.96 GiB). Model load=3.570 s. Эти цифры не включают download/install
или draft. Не переносим успех формата одного пользователя на весь cohort и не
делаем вывод о ranking quality.

Три original responses совпали; aggregate original и aggregate permuted различаются.
Это иллюстрация order sensitivity в одном early примере, не доказательство улучшения
nDCG. Линейная оценка 600 calls по этой записи — **10.19 CPU hours генерации**,
без гарантии для иных длин/историй, setup, interruptions и повторных запусков.
Перед полным experiment нужно отдельно согласовать среду и ресурсный режим.

## Проверки

Шесть tests: lossless round-trip; same facts/only row order changes;
strict parser; единый fallback и ISIN tie-break; ровно шесть fake-backend calls
с per-call invalid evidence и без cache reuse; fatal backend error останавливает
run вместо выдачи fallback без inference. Fake tests — не model inference.

verify_llm_pilot.py не импортирует backend/parser/aggregation. Он сверяет weights,
early source и median selection, prompts/токены, реконструкцию payload, шесть
реальных logs, parser/fallback flags и rank-point aggregates. Три original
повторения дополнительно показывают воспроизводимость ответа в тех же условиях;
это не независимые пользователи или оценка устойчивости ко всем permutations.
Результаты — STAGE6_VERIFICATION.json, автоматическая проверка Codex.

## Хранение и команды

Public: LLM_CONFIG / MODEL / ENVIRONMENT / PILOT JSON, aggregated verification,
dependency lock, scripts и этот отчёт. Private ignored: model weights/HF cache,
pilot payload, raw prompts/responses, aggregates и evidence неудачного draft.

Из project_group_2module/ в PowerShell:

```powershell
& '.venv-hw2-llm/Scripts/python.exe' -B -m unittest discover -s hw2/scripts -p test_llm_ranker.py -v
& '.venv-hw2-llm/Scripts/python.exe' -B hw2/scripts/verify_llm_pilot.py
```

run_llm_pilot.py выполняет реальные generation и отказывается перезаписывать
существующий calls.jsonl. Новый pilot требует отдельного согласованного output,
старые evidence не удаляем. download_llm.py нужен только при отсутствии local
weights; offline verification не скачивает ничего.

## Проверка Дарьи

1. Проверить model revision, CPU dtype и no-response-cache в config.
2. Сверить, что short-ID encoding обратим и одинаков для обеих веток, source
   candidates/history/features не поменялись; parser ничего не исправляет.
3. В LLM_PILOT.json: шесть calls, quality_metrics_computed=false,
   june_evaluation_run=false, actual invalid/fallback/timing/memory.
4. В STAGE6_VERIFICATION.json все checks=true. Учитывать draft отдельно.
5. Решение о полном запуске принимать по измеренному CPU времени, а не считать
   линейную экстраполяцию одного median-group гарантией. GPU/cloud потребует
   отдельного фиксированного runtime/precision и технической проверки.

После принятия этапа — commit/push. Полный June experiment — отдельный этап 7.
LLM_METHOD_FREEZE.json связывает текущий model/serialization/prompt/config с
неизменными common inputs и candidate hash. Для смены CPU на GPU или precision
потребуется явно зафиксированный runtime variant и техническая проверка.
Источники: [Qwen pinned model](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e),
[provided teacher baseline](https://github.com/dryjins/RecSys-LLMs/tree/799e54c0493cf7e2b014152548ae5c18ee24a8f5/project/far-trans-llm),
[LLMRank](https://arxiv.org/html/2305.08845v2).
