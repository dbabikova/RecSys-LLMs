# Этап 4 — temporal data и train-only преобразования

Статус: выполнено и автоматически проверено Codex; Дарья приняла этап («ок»
в чате). Самостоятельная ручная raw-сверка Дарьи не заявляется выполненной.
JSON-отчёт сохраняет статус на момент автоматического запуска до принятия этапа.
LambdaMART не обучался, LLM inference не выполнялся. Итоговые 100 пользователей
июня и CBF20 не изменились и не использованы для fit или выбора temporal cohort.

## Дизайн, зафиксированный до построения данных

TEMPORAL_CONFIG.json фиксирует три train cutoffs и один validation cutoff.
History/metadata строго раньше своего cutoff; binary labels — следующие 30 дней.

| Роль | Cutoff | Конец labels, exclusive | Groups | Candidate rows | Eligible assets | Nonempty truth | Groups с positive candidate | Positive labels |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2021-01-01 | 2021-01-31 | 1000 | 20000 | 746 | 196 | 132 | 205 |
| Train | 2021-03-01 | 2021-03-31 | 1000 | 20000 | 757 | 224 | 150 | 200 |
| Train | 2021-04-01 | 2021-05-01 | 1000 | 20000 | 763 | 175 | 114 | 158 |
| Validation | 2021-05-01 | 2021-05-31 | 1000 | 20000 | 764 | 171 | 107 | 149 |

Месячные начала и 30-дневный горизонт выбраны по учебной постановке до запуска.
Между train windows есть календарные промежутки; даты не сдвигаются по результатам.
Train labels не пересекаются с validation. Validation заканчивается строго до июня.

Ресурсный объём — первые 1000 as-of-known warm users каждого cutoff по исходному
SHA256 seed-42 правилу starter. Это обучающий объём, не новый evaluation cohort.
Будущие покупки/качество кандидатов не участвуют в отборе. Пользователи могут
повторяться на cutoffs: 3000 groups не называются 3000 независимыми людьми.
История eval-пользователя на более ранних датах допустима; июньские labels — нет.

Universe заново определяется на каждом cutoff (metadata + positive pre-cut price),
не берётся из июньского каталога. Retrieval — pinned starter CBF, первые 20 с его
scoring/tie-break. Входы соответствуют common_input_v1 этапа 3.
Labels/role/full truth и реальные ID отделены от payload и model features.

## Матрицы и группы

- Train: 3000 groups user×cutoff, 60000 строк, 563 положительные candidate labels.
- Validation: 1000 groups, 20000 строк, 149 положительных labels.
- В каждой группе ровно 20 последовательных строк; X/y/group_sizes согласованы.
- Сохранены все 2604 train и 893 validation groups без positive candidate.
  Это и empty truth, и покупки вне pool. Одинаковые labels внутри группы могут
  не давать pairwise обучающего сигнала; группы не скрываются и не удаляются.
- Обе матрицы имеют **6447 признаков**, все значения конечны.

## Преобразования

1. Fixed numeric из stage 3: counts/recency, CBF rank/score, ages, history summary,
   cutoff ordinal, match shares. Missing = -1 плюс missing flag, без fitted imputer.
2. DictVectorizer fitted только на train. One-hot категорий кандидата, профиля и
   status; для каждого history slot — категории и count/Buy age/metadata age.
   User/group/asset aliases не обучаемые признаки. Получено 1543 structured columns.
3. TF-IDF имён: word unigrams/bigrams, lowercase, smoothed IDF, max_features=4096.
   Fit corpus — **323 уникальных имени** из train histories/candidates; не весь
   июньский каталог и не validation. Получено **1634 terms**.
4. Три text blocks: candidate TF-IDF, нормированный buy-count-weighted history
   TF-IDF, нормированный inverse-(1+Buy age)-weighted history TF-IDF. Добавлены
   две cosine similarities candidate↔history. Итого 1543 + 3×1634 + 2 = 6447.

Та же исходная информация не означает одинаковые representations или prior
knowledge. LLM читает историю, классический контроль получает явные преобразования.

На validation: 1240 неизвестных category-value occurrences из 573400 и один
неизвестный name-term occurrence из 211766. Это вхождения, не уникальные категории
или пользователи. Unknown coordinates игнорируются без refit, rows сохраняются.
Validation не изменил fitted vocabulary/IDF.

## Важное ограничение scorer на этапе 5

Train labels бинарны внутри pool. При выборе модели nDCG должен делиться на IDCG
по **full_truth_count**, включая цели вне CBF20; empty truth = 0. Сохраняем среднюю
по всем 1000 validation users и отдельно nonempty. Нельзя автоматически заменить
этот scorer встроенной LightGBM nDCG с pool-only IDCG/другой empty-group convention.
Все сведения для корректного расчёта сохранены. Июнь оцениваем после model freeze.

## Автоматические проверки

verify_temporal.py самостоятельно пересобирает из CSV as-of universe, warm users,
CBF20, history/recency и binary truth **для всех 4000 групп**; не импортирует
temporal builder или starter. Проверены исходные/file hashes, окна и порядок
матриц/меток/групп, train-only fit keys и text corpus. Каждый IDF независимо
пересчитан по train names. Сериализованный encoder воспроизводит строки фиксированных
audit groups. Отчёт — STAGE4_VERIFICATION.json.

Три unit tests прошли: validation-only text/category не попадает в fit state;
permutation только переставляет matrix rows; group identity и добавленные labels
не меняют features. Это проверки Codex, не личная проверка Дарьи.
При подготовке исправлена сериализация numpy int64 в provenance hash; ошибка
возникла до сохранения матриц, признаки по validation не настраивались.

## Хранение и воспроизведение

Public: CONFIG/DATA/FEATURES JSON, STAGE4_VERIFICATION.json, scripts и этот отчёт.
Private ignored data/hw2/temporal/: gzip payloads, labels и ID mapping отдельно;
features/ содержит X.npz, y.npy, group_sizes.npy, train_encoder.joblib и feature names.
Joblib создан нами локально; не загружаем pickle из непроверенных внешних источников.

Среда: Python 3.13.9, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.7.2, joblib 1.5.2.
LightGBM пока отсутствует — установка/проверка относится к этапу 5.

Из project_group_2module/ в PowerShell:

```powershell
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/build_temporal.py
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/fit_temporal_features.py
& 'C:\Users\User\anaconda3\python.exe' -B hw2/scripts/verify_temporal.py
& 'C:\Users\User\anaconda3\python.exe' -B -m unittest discover -s hw2/scripts -p test_ranker_features.py -v
```

Temporal gzip frozen: при изменении содержимого builder останавливается.
Feature encoder/matrices пересобираются; hashes относятся к сохранённым артефактам.
Не обещаем byte identity NPZ ZIP-контейнеров: числовая воспроизводимость проверяется
сравнением sparse rows и fit state. june_evaluation_loaded=false означает, что
eval payloads/labels не использованы; исходные CSV физически содержат весь релиз,
а loaders фильтруют по каждому cutoff.

## Проверка Дарьи

1. Проверить границы train/validation и что validation заканчивается 31 мая.
2. Различать 1000 training users/cutoff и неизменных 100 итоговых пользователей.
3. В TEMPORAL_FEATURES.json: 60000×6447 train, 20000×6447 validation,
   fit_role=train, fit_state_unchanged_after_validation=true.
4. В STAGE4_VERIFICATION.json все checks=true; zero-positive groups сохранены,
   июньские labels не использованы. Прочитать правило full-truth scorer выше.

После принятия этапа — commit/push. Далее: LightGBM, обучение на train,
ограниченный выбор параметров по May validation и model freeze перед июнем.

API sources: [DictVectorizer 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.DictVectorizer.html),
[TF-IDF 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
