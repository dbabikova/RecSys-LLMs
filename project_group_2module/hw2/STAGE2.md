# Этап 2 — фиксированный CBF20 и исходный reference

Статус: выполнено и автоматически проверено Codex в PowerShell; Дарья приняла
этап («ок» в чате). Самостоятельная ручная сверка Дарьи не заявляется выполненной.
JSON-отчёты сохраняют статус на момент автоматического запуска до её принятия.
Модели не обучались, LLM inference не выполнялся. Pool по результатам не менялся.

## Пользователи и источники

Сохранены **те же 100 user ID в том же порядке** из protocol.json Алины.
Никого не удаляли и не добавляли; 89 users с empty truth остаются в оценке.
Манифест Ульяны в предоставленном derivative session export указывает **тот же
побайтовый SHA-256 protocol.json и того же исходного ZIP Алины**. Её сохранённый
protocol.py строит evaluation в порядке protocol['users']. Поэтому общий источник
когорты подтверждён не только числом 100. Её исходные per-user файлы не приложены:
мы не утверждаем, что заново исполнили её inference или сравнили эти файлы.
Хеш derivative export и границы подтверждения — STAGE2_SOURCES.json.

## Зафиксированные артефакты

| Артефакт | Где хранится | Назначение |
| --- | --- | --- |
| candidates.jsonl | data/hw2/cbf20/, ignored | 100 записей с первыми 20 ID и metadata из CBF50 |
| reference_predictions.csv | там же, ignored | 1000 строк: первые 10 кандидатов каждого пользователя |
| candidate_manifest.json | там же, ignored | Политика, происхождение, хеши и границы manifest |
| CANDIDATES.json | hw2/, public | Точная копия manifest без user ID и списков активов |
| per_user.csv, review_example.json | data/hw2/stage2_diagnostics/, ignored | Только диагностические labels/метрики, никогда не вход модели |
| STAGE2_VERIFICATION.json | hw2/, public | Агрегаты и результаты независимой проверки |
| STAGE2_INTEGRITY.json | hw2/, public | Проверки отказа на повреждённых копиях файлов |

Политика saved_cbf50_first20_v1 сохраняет user order, candidate order и исходные
метаданные. Builder читает только protocol.json и contexts.jsonl из baseline ZIP;
не читает метки, per_user.csv или транзакции. Итоговый feature contract модели
пока не сформирован: он будет определён на этапе 3.

Candidate hash идентифицирует ordered user→candidate IDs. Отдельный candidate
manifest hash связывает эту выдачу с policy, cohort, catalog и исходными contexts.
Это не конечный model experiment hash: модель и признаки ещё не выбраны.
Исходный protocol.json Алины не изменён и остаётся CBF50.

## Измеренные результаты

| Метод / диагностика | Все 100 | 11 с покупками |
| --- | ---: | ---: |
| Исходный Top-10 CBF20: nDCG@10 | 0.0201 | 0.1827 |
| Oracle перестановки CBF20: nDCG@10 | 0.0392 | 0.3560 |

Reference побайтово по значениям user/rank/ISIN совпадает с сохранённым baseline
CBF Алины (различается название метода). Per-user nDCG и агрегаты пересчитаны
независимо по сырым CSV; совпадение проверено с точностью 1e-14.

- В truth 18 distinct user–asset pairs; CBF20 содержит 5, то есть micro recall =
  **5/18 = 0.2778**. Это не доля пользователей и не 5 уникальных активов всего.
- Macro recall по 11 users с nonempty truth = **0.3333**. Recall для empty truth
  не определён и не входит в эту среднюю; primary nDCG по-прежнему включает всех.
- У 5 из 11 есть хотя бы одна цель в pool; у 6 нет ни одной. У 3 все цели в pool.
- Repeat targets: 5/12 в pool; new targets: 0/6. Это только post-hoc диагностика.
- Oracle ставит доступные релевантные assets первыми и делит DCG на **IDCG по всей
  truth**, включая отсутствующие в pool цели. Future labels используются только
  здесь, отдельно от builder и будущих model inputs.

Вывод: потенциал улучшить исходный порядок есть, но реранкер не может получить
покупочный hit у шести nonempty users с нулевым candidate recall. Oracle на этом
demo выше четырёх baseline по средней, но это недостижимый на практике контроль
с доступом к будущему, а не прогноз результата LLM. Pool не меняем по этой оценке.

## Автоматические проверки

Независимый verifier не импортирует builder или starter. Проверены точные ID и
порядок пользователей, metadata/ID prefix CBF50→20, уникальность, eligibility,
файловые и семантические хеши, 1000 reference rows, полная truth и per-user nDCG.
Отдельно на временных приватных копиях доказан отказ при перестановке кандидатов,
удалении одного пользователя и подмене reference. Исходные frozen files не менялись.

Builder отказывается перезаписывать отличающиеся frozen artifacts. Повторная
сборка и верификация с теми же входами должны дать те же файлы и хеши.

Из project_group_2module/:

```powershell
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/build_candidates.py
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/verify_candidates.py
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/check_candidate_integrity.py
```

Входы находятся в data/hw2/inputs/. На другом компьютере также нужны исходные
ZIP Алины, FAR-Trans и приложенный derivative session export Ульяны; их SHA-256
заданы в SOURCES.json и STAGE2_SOURCES.json. Скрипты не скачивают файлы.

## Что Дарье проверить лично

Открой PowerShell в project_group_2module/. Эти команды читают исходный пакет
Алины и наш файл напрямую, без builder или verifier:

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$packagePath = Join-Path (Get-Location) 'data/hw2/inputs/team5_alina_t4_protocol_check.zip'
$package = [System.IO.Compression.ZipFile]::OpenRead($packagePath)
function Read-ZipText($zip, $name) {
    $reader = [System.IO.StreamReader]::new($zip.GetEntry($name).Open())
    try { $reader.ReadToEnd() } finally { $reader.Dispose() }
}
$protocol = (Read-ZipText $package 'protocol.json') | ConvertFrom-Json
$contexts = (Read-ZipText $package 'contexts.jsonl') -split '\r?\n' |
    Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
$frozen = Get-Content -LiteralPath 'data/hw2/cbf20/candidates.jsonl' -Encoding UTF8 |
    ForEach-Object { $_ | ConvertFrom-Json }
$package.Dispose()
if ($frozen.Count -ne 100) { throw 'Wrong number of users' }
for ($i = 0; $i -lt 100; $i++) {
    if ($frozen[$i].user_id -cne $protocol.users[$i] -or
        $frozen[$i].user_id -cne $contexts[$i].user_id) { throw "Wrong user/order: $i" }
    $expected = @($contexts[$i].candidate_assets | Select-Object -First 20 |
        ForEach-Object { $_.id })
    if ($frozen[$i].candidate_ids.Count -ne 20 -or
        ($frozen[$i].candidate_ids -join '|') -cne ($expected -join '|')) {
        throw "Wrong candidates: $i"
    }
}
'PASS: same 100 users in source order; all candidate lists equal first 20 of CBF50'
```

Далее лично просмотри пример: **30-я запись** в зафиксированном порядке.
Выбор только иллюстративный, не изменение cohort или pool. Команды используют
переменные предыдущего блока и выводят исходный и сохранённый списки рядом:

```powershell
$index = 29
$comparison = for ($rank = 0; $rank -lt 20; $rank++) {
    [pscustomobject]@{ Rank = $rank + 1
        SourceCBF50 = $contexts[$index].candidate_assets[$rank].id
        SavedCBF20 = $frozen[$index].candidate_ids[$rank] }
}
$comparison | Format-Table -AutoSize | Out-Host
$uid = $protocol.users[$index]
$reference = Import-Csv -LiteralPath 'data/hw2/cbf20/reference_predictions.csv'
$reference | Where-Object { $_.user_id -ceq $uid } | Select-Object rank,ISIN |
    Format-Table -AutoSize | Out-Host
```

Для ручной арифметики проверь raw June Buy того же пользователя:

```powershell
$dataset = [System.IO.Compression.ZipFile]::OpenRead(
    (Join-Path (Get-Location) 'data/hw2/inputs/FAR-Trans.zip'))
$transactions = (Read-ZipText $dataset 'FAR-Trans/transactions.csv') | ConvertFrom-Csv
$dataset.Dispose()
$transactions | Where-Object {
    $_.customerID -ceq $uid -and $_.transactionType -eq 'Buy' -and
    [long]$_.transactionID -ge 0 -and $_.timestamp -ge '2021-06-01' -and
    $_.timestamp -lt '2021-07-01' -and $_.ISIN -cin $protocol.eligible_assets
} | Select-Object transactionID,ISIN,timestamp | Format-Table -AutoSize | Out-Host
```

Ожидаемая автоматическая сверка, пока не личный результат: две уникальные цели
GRS014003032 и GRS191213008. Первая в pool на позиции 1; второй в CBF20 нет.
Проверь это по таблицам выше. Тогда DCG=1, IDCG=1+1/log2(3),
nDCG=**0.613147…**, recall20=**1/2**. Для этого пользователя reference уже равен
oracle: улучшить его одной перестановкой этого pool нельзя.

Сообщи, какие действия ты выполнила и совпали ли результаты. Чтение отчёта,
запуск автоматических проверок и самостоятельная сверка исходных строк/арифметики
будут описаны раздельно. Commit и push выполняются после твоего «ок» этапу 2.
