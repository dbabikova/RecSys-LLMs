# Этап 1 — воспроизведение и проверка Дарьей

## Что проверить лично

1. Исторический RQ этапа 1: LLM vs LambdaMART, один CBF20 и одинаковые поля.
   После принятия этапа 1 Дарья согласовала уточнение RQ про permutation aggregation;
   актуальная формулировка и роли контролей — RQ.md и EXPERIMENT.md.
2. Проверить cohort=100, cutoff=2021-06-01, end exclusive=2021-07-01,
   empty truth=0, repeat allowed и четыре обязательных baseline.
3. Подтвердить, что первые 20 из CBF50 — основной pool этого опыта;
   repeat-aware retrieval отложен. На этапе 1 списки ещё не материализуются.
4. В SOURCES.json сравнить starter SHA-256 с сообщением Алины; baseline source
   commit — с commit.txt её ZIP. В PowerShell можно независимо получить хеш:

```powershell
Get-FileHash -LiteralPath 'C:\Users\User\Downloads\starter_used_in_colab.py' -Algorithm SHA256
```

5. Открыть VERIFICATION.json: checks должны быть true, users=100,
   nonempty_truth_users=11, eligible_assets=767, baseline_rows_checked=400.
6. Проверить атрибуцию и открытые решения: model inference ещё не выполнялся,
   личная проверка не заявлена, командное утверждение baseline не выдумано.
7. Просмотреть git diff и новые файлы. В tracked additions не должно быть
   customerID, contexts, predictions, исходных CSV, архивов или session exports.

После проверки сообщить «ок» либо конкретные правки. Только после «ок» — commit.

## Воспроизведение автоматической проверки

Из project_group_2module/ в установленном Python 3.10+ (только stdlib):

```powershell
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/verify_stage1.py
```

По умолчанию входы берутся из data/hw2/inputs/. Сюда один раз скопированы без
изменения байтов пакет Алины, точный starter и исходный FAR-Trans.zip.
Teacher starter из commit 08d4e734 сохранён там же для сравнения. Полные хеши
содержатся в SOURCES.json. Входы и повторный baseline output игнорируются Git.
Новый clone потребует этих четырёх локальных файлов. Скрипт не скачивает данные,
не обучает модели, не вызывает LLM и не делает commit/push.

Полное воспроизведение занимает время загрузки CSV и четырёх baseline.
VERIFICATION.json перезаписывается детерминированным агрегированным отчётом.
Отчёт проверки не включает runtime нового запуска: чужой timing не переносится
на нашу машину. Информацию о среде сохраняем отдельно от измерений Алины.
