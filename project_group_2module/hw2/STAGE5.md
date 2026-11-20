# Этап 5 — LambdaMART: May validation и model freeze

Статус: выполнено и автоматически проверено Codex; Дарья приняла этап («ок»
в чате). Самостоятельная ручная raw-сверка не заявляется. JSON-отчёт проверки
сохраняет статус на момент автоматического запуска до принятия этапа.
Июньская оценка и LLM inference не запускались. На validation не переобучались.

## Правило выбора, записанное до обучения

LAMBDAMART_CONFIG.json фиксирует objective=lambdarank, 4 CPU threads, learning_rate
0.05, min_data_in_leaf=20, L2=1, feature/bagging fractions=0.9, deterministic
column-wise training. Subsampling seeds контролируются. Train — сохранённые
60000 строк этапа 4, все 3000 групп, включая zero-positive.

Четыре конфигурации: leaves 7/15 × trees 100/200. Выбор при seed=0 по unrounded
May nDCG@10 **на всех 1000 users**. Tie-break: меньше trees, меньше leaves, id.
Early stopping и поиск вне сетки не применялись. Затем проверены seeds 0…4;
**primary seed заранее 0**, не лучший seed. Ensemble не добавлялся.

## Validation-результаты

| Вариант | Seed | Users / nonempty | nDCG@10 all | nDCG@10 nonempty |
| --- | ---: | --- | ---: | ---: |
| Исходный CBF20 Top-10 | — | 1000 / 171 | 0.0602 | 0.3520 |
| Leaves 7, trees 100 | 0 | 1000 / 171 | 0.0759 | 0.4440 |
| Leaves 7, trees 200 | 0 | 1000 / 171 | 0.0756 | 0.4420 |
| **Leaves 15, trees 100 — выбран** | **0** | **1000 / 171** | **0.0777** | **0.4547** |
| Leaves 15, trees 200 | 0 | 1000 / 171 | 0.0766 | 0.4481 |
| Выбранная конфигурация | 1 | 1000 / 171 | 0.0767 | 0.4483 |
| Выбранная конфигурация | 2 | 1000 / 171 | 0.0769 | 0.4499 |
| Выбранная конфигурация | 3 | 1000 / 171 | 0.0751 | 0.4393 |
| Выбранная конфигурация | 4 | 1000 / 171 | 0.0762 | 0.4457 |

Полная точность — LAMBDAMART_RESULTS.json. Failed cases=0; все 829 empty-truth
users сохранены. Primary−reference=+0.0175557 all: 70 users улучшились,
10 ухудшились, 920 не изменились (включая empty truth).
По пяти seeds: mean all=0.0765345, sample SD=0.0009684, min=0.0751176,
max=0.0777485. Seeds не являются независимыми пользователями/выборками.

Это **May validation для выбора параметров**, не June test и не доказательство
обобщения или статистической значимости. Нельзя смешивать его со 100-user таблицей
Алины/Ульяны. Четыре обязательных baseline сохранятся в июньской таблице;
здесь reference — исходный порядок того же CBF20.

## Scorer и проверки

ranker_metrics.py делит binary DCG на IDCG по **всей truth**, включая активы вне
pool. Empty truth=0, не 1. Tie-break scores — оригинальный ISIN. Built-in metric
LightGBM отключена. Пять scorer tests прошли: full-truth denominator, empty truth,
unretrieved truth, ISIN tie-break и отказ на невалидных score arrays.

verify_lambdamart.py не импортирует trainer или scorer. Он заново восстанавливает
May truth из raw CSV и проверяет Top-10, full truth counts, per-user и aggregate
nDCG всех восьми моделей: **8000 user metrics и 80000 prediction rows**.
Все восемь моделей из model.txt побитово воспроизвели сохранённые scores.
Проверены SHA-256, выбор конфигурации и пять seeds. Отчёт — STAGE5_VERIFICATION.json.
Это проверки Codex; личная ручная проверка Дарьи не заявляется.

## Model freeze и июнь

MODEL_FREEZE.json связывает config, train-fitted encoder, feature manifest,
common input contract и SHA-256 выбранной модели. Private primary model:
data/hw2/lambdamart/leaves15_trees100_seed0/model.txt (ignored).
Перед June inference проверяем эти hashes; encoder на June не обучаем и модель
по June truth не меняем. Evaluation users/candidates/payloads этапов 2–3 неизменны.
Их labels не использовались для обучения/выбора. Source CSV физически содержат
весь релиз; scorer/verifier этапа 5 отбирают только May window.

## Окружение и время

Project-local .venv-hw2-ranker создан с system-site-packages. numpy/scipy/sklearn/
joblib наследуются из проверенного Anaconda; LightGBM 4.7.0 установлен только
в venv, глобальные пакеты не менялись. Версии — requirements-ranker.txt.
Python 3.13.9, Windows AMD64, 8 logical CPUs, training_threads=4.

Весь запуск восьми моделей: 21.273 s (setup, training, May predict, scoring, writes).
Primary fit=2.171 s. Не включены stage-4 extraction/feature fit/transform,
установка пакета и будущая June генерация. Эти времена не сравнимы напрямую
с runtime Алины или GPU LLM. Per-run fit/predict/save timings сохранены в RESULTS.

## Проверка в PowerShell

Из project_group_2module/:

```powershell
& '.venv-hw2-ranker/Scripts/python.exe' -B -m unittest discover -s hw2/scripts -p test_ranker_metrics.py -v
& '.venv-hw2-ranker/Scripts/python.exe' -B hw2/scripts/verify_lambdamart.py
```

На свежем output обучение запускается:

```powershell
& '.venv-hw2-ranker/Scripts/python.exe' -B hw2/scripts/train_lambdamart.py
```

При run_complete.json trainer отказывается перезаписывать завершённый опыт.
Новый опыт требует отдельного согласованного output/version, не удаления evidence.
На свежем clone нужны локальные данные и матрицы предыдущих этапов: модели,
validation outputs и mapping не публикуются в Git.

### Что проверить Дарье

1. В config: четыре варианта, fixed primary seed 0, no June evaluation.
2. В RESULTS: primary leaves15_trees100, users=1000/171, обе средние, zero failures.
3. В VERIFICATION: status=passed, все checks=true, восемь replayed models.
4. При желании пример May group **2021-05-01:U0013**:

```powershell
$runDir = 'data/hw2/lambdamart/leaves15_trees100_seed0'
Import-Csv -LiteralPath "$runDir/validation_per_user.csv" |
    Where-Object { $_.group_key -eq '2021-05-01:U0013' } | Format-List
Import-Csv -LiteralPath "$runDir/validation_top10.csv" |
    Where-Object { $_.group_key -eq '2021-05-01:U0013' } |
    Select-Object rank,ISIN | Format-Table -AutoSize
```

Ожидаемая автоматическая сверка: одна цель GRS003003035, rank=2, IDCG=1,
nDCG=1/log2(3)=0.63092975. Для личной raw-проверки найти source customerID через
ignored May mapping и сверить May Buy по CSV. Чтение ожидаемого результата не
является самостоятельной проверкой. Commit/push — после принятия этапа.

Далее: open-weight LLM/hardware, prompt/parser/call budget и technical pilot.
API source: [LightGBM Ranker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).
