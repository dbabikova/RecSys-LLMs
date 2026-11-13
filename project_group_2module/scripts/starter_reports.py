"""Generated reports for the imported, unmodified instructor starter."""

METHODS = ["popularity", "cbf", "latest_buy", "latest_buy_or_sell"]
LABELS = {
    "popularity": "Popularity", "cbf": "CBF", "latest_buy": "Latest buy",
    "latest_buy_or_sell": "Latest buy or sell",
}


def f(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(f, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(f, row)) + " |" for row in rows],
    ]) + "\n"


def baseline_table(run):
    return table(
        ["Метод", "Пользователей", "С будущими Buy", "Пустые метки",
         "nDCG@10: все (primary)", "nDCG@10: nonempty"],
        [(LABELS[m], *(run["metrics"][m][k] for k in (
            "users", "nonempty_truth_users", "empty_truth_users", "ndcg_at_10", "ndcg_at_10_nonempty"
        ))) for m in METHODS],
    )


def capacity_table(summary):
    rows = []
    for label, run in summary["runs"].items():
        for group in ("all", "repeat", "new"):
            subset = [r for r in run["diagnostics"]["candidate_recall"] if r["target_group"] == group]
            row50 = next(r for r in subset if r["k"] == 50)
            rows.append((
                label, group, row50["users"], row50["target_pairs"],
                *(next(r for r in subset if r["k"] == k)["macro_recall"] for k in (10, 20, 50)),
                row50["any_hit_share"],
            ))
    return table(
        ["Запуск", "Метки", "Users с такими метками", "Distinct Buy-пар",
         "Macro Recall@10", "@20", "@50", "Users с ≥1 hit в top-50, доля"], rows,
    )


def render(root, summary):
    runs = summary["runs"]
    text = f"""# Официальный стартер: четыре baseline и потолок CBF-реранкинга

> Сгенерировано `scripts/starter_audit.py` и `scripts/starter_reports.py`.
> Все результаты вычислены; таблицы README прочитаны скриптом из зафиксированного
> внешнего клона. Полная точность — `starter_summary.json`; здесь метрики округлены
> по HW2. Это диагностика, не реализованный LLM-метод и не выбор трека/RQ.

## Источник, неизменность и новые правила

- Репозиторий преподавателя: {summary['starter_repository']}.
- **Commit:** `{summary['starter_commit']}`.
- Отдельный clone: `../../course-starter` относительно папки нашего проекта,
  вне репозитория Дарьи. Исходный starter и `extensions.py` не изменены.
- Проверены HEAD, `git status --porcelain --untracked-files=all` и SHA-256 исходников
  до/после запусков. `starter_unchanged_verified = {summary['starter_unchanged_verified']}`.
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

"""
    text += table(["Параметр", "Значение"], [
        ("Cutoff (history строго до)", summary["cutoff"]),
        ("End exclusive", summary["end_exclusive"]),
        ("Окно меток", f"[{summary['cutoff']}, {summary['end_exclusive']})"),
        ("Eligible assets", summary["eligible_asset_count"]),
        ("CBF candidate-count", summary["candidate_count"]),
        ("Repeat allowed / exclude_seen", f"{summary['repeat_allowed']} / False"),
        ("Cohort", "warm; deterministic selection самого starter"),
        ("Python", summary["environment"]["python"]),
        ("Платформа", summary["environment"]["platform"]),
        ("Процессор", summary["environment"]["processor"]),
    ])
    text += """
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

"""
    for label, run in runs.items():
        text += f"### {label}\n\n"
        text += baseline_table(run)
        text += "\n" + table(["Метаданные запуска", "Значение"], [
            ("Flags сверх data-dir/output-dir", " ".join(run["flags_beyond_io"]) or "нет (defaults)"),
            ("Started UTC", run["started_utc"]), ("Finished UTC", run["finished_utc"]),
            ("Whole CLI wall seconds", run["whole_cli_wall_seconds"]),
            ("protocol_hash", f"`{run['protocol_hash']}`"),
            ("Custom failures", run["custom_failures"]),
            *[(f"Excluded: {key}", value) for key, value in run["excluded_rows"].items()],
        ])
        text += "\n"
    text += """
Полные `protocol.json`, `contexts.jsonl`, `predictions.csv`, `per_user.csv`,
`metrics.json` и console/run metadata сохранены в **игнорируемом**
`data/starter_runs/{demo,full_warm}/`. Они содержат клиентские псевдо-ID и
не коммитятся. Публичный `starter_summary.json` содержит только агрегаты,
хеши и обезличенную проверку. В main таблицы длинные protocol hashes не вынесены.

## Сверка с README именно этого commit

Для demo используем demo-таблицу README. Для `--max-users 0` сравниваем с
**warm paragraph**, а не с большой таблицей `--cohort all`.
Разности метрик считаются после округления наших чисел до точности README.

"""
    differences = []
    for label, run in runs.items():
        ref = summary["readme_reference"][label]
        for m in METHODS:
            actual = run["metrics"][m]
            differences.append((
                label, LABELS[m], actual["users"] - ref[m]["users"],
                actual["nonempty_truth_users"] - ref[m]["nonempty_truth_users"],
                ref[m]["ndcg_at_10"], round(actual["ndcg_at_10"], 4) - ref[m]["ndcg_at_10"],
                ref[m]["ndcg_at_10_nonempty"],
                round(actual["ndcg_at_10_nonempty"], 4) - ref[m]["ndcg_at_10_nonempty"],
            ))
    text += table(
        ["Запуск", "Метод", "Δ users", "Δ nonempty", "README all", "Δ all (4dp)",
         "README nonempty", "Δ nonempty (4dp)"], differences,
    )
    all_ref = summary["readme_reference"]["all"]["cbf"]
    warm_ref = summary["readme_reference"]["full_warm"]["cbf"]
    mismatch = sum(
        any(run["metrics"][m][k] != summary["readme_reference"][label][m][k] for k in (
            "users", "nonempty_truth_users", "empty_truth_users"
        )) or any(round(run["metrics"][m][k], 4) != summary["readme_reference"][label][m][k]
                  for k in ("ndcg_at_10", "ndcg_at_10_nonempty"))
        for label, run in runs.items() for m in METHODS
    )
    text += f"""
Несовпадающих строк при совпадающей selection и точности README: **{mismatch}**.
В таблице all из README — {all_ref['users']} users / {all_ref['nonempty_truth_users']}
nonempty; в его warm paragraph — {warm_ref['users']} / {warm_ref['nonempty_truth_users']}.
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
Repeat targets = T∩H, new targets = T\\H.

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

"""
    text += capacity_table(summary)
    text += "\n### Подробные попадания по всем cutoff\n\n"
    text += table(
        ["Запуск", "Targets", "k", "Users", "Target-пар", "Covered-пар",
         "Macro recall", "Micro recall", "Users ≥1 hit", "Any-hit share"],
        [(label, r["target_group"], r["k"], r["users"], r["target_pairs"],
          r["covered_pairs"], r["macro_recall"], r["micro_recall"], r["users_any_hit"], r["any_hit_share"])
         for label, run in runs.items() for r in run["diagnostics"]["candidate_recall"]],
    )
    text += "\n### Повторность будущих покупок — разные знаменатели\n\n"
    text += table(
        ["Запуск", "Distinct target-пар", "Repeat-пар", "New-пар", "Repeat pair share",
         "Macro repeat share", "Buy-строк в окне", "Repeat Buy-строк", "Repeat row share"],
        [(label, *(run["diagnostics"][k] for k in (
            "future_target_pairs", "repeat_target_pairs", "new_target_pairs",
            "repeat_pair_share", "macro_repeat_pair_share", "future_buy_rows",
            "future_repeat_buy_rows", "repeat_buy_row_share",
        ))) for label, run in runs.items()],
    )
    text += "\n### Какие hits Latest buy вообще доступны ranker-only\n\n"
    text += table(
        ["Запуск", "Targets", "Latest buy hit-пар@10", "Из них есть в CBF50",
         "Отсутствуют в CBF50", "Доля доступных", "Users с потерянным hit"],
        [(label, r["target_group"], r["latest_buy_hit_pairs"], r["also_in_cbf50"],
          r["missing_from_cbf50"], r["covered_share"], r["users_with_missing_latest_hit"])
         for label, run in runs.items() for r in run["diagnostics"]["latest_buy_overlap"]],
    )
    text += """
### Oracle ceiling: можно ли в принципе догнать Latest buy?

Oracle переставляет **только исходные CBF top-50**: сначала релевантные кандидаты,
затем остальные, и оценивается `starter.ndcg10` с **полным** T в IDCG.
Ограничение Top-10 сохранено. Это недостижимая без знания будущего верхняя граница,
не прогноз и не эксперимент с LLM. Сравнение только на тех же пользователях:

"""
    text += table(
        ["Запуск", "CBF nonempty", "Latest buy nonempty", "Oracle50 nonempty",
         "Oracle50 all", "Oracle50 − Latest buy nonempty", "Mean gap to Latest: possible?"],
        [(label, run["metrics"]["cbf"]["ndcg_at_10_nonempty"],
          run["metrics"]["latest_buy"]["ndcg_at_10_nonempty"],
          run["diagnostics"]["cbf50_oracle"]["ndcg_at_10_nonempty"],
          run["diagnostics"]["cbf50_oracle"]["ndcg_at_10"],
          run["diagnostics"]["cbf50_oracle"]["ndcg_at_10_nonempty"] - run["metrics"]["latest_buy"]["ndcg_at_10_nonempty"],
          "upper bound permits; not a success forecast" if
          run["diagnostics"]["cbf50_oracle"]["ndcg_at_10_nonempty"] >= run["metrics"]["latest_buy"]["ndcg_at_10_nonempty"]
          else "NO, with fixed pool and protocol")
         for label, run in runs.items()],
    )
    for label, run in runs.items():
        latest = run["metrics"]["latest_buy"]["ndcg_at_10_nonempty"]
        upper = run["diagnostics"]["cbf50_oracle"]["ndcg_at_10_nonempty"]
        overlap = next(r for r in run["diagnostics"]["latest_buy_overlap"] if r["target_group"] == "all")
        text += (
            f"\n**{label}:** {'потолок допускает догнать Latest buy, но успех LLM не доказан' if upper >= latest else 'догнать Latest buy в среднем невозможно одним переставлением исходного CBF50'}"
            f" (oracle {upper:.4f}, Latest buy {latest:.4f}). "
            f"Из {overlap['latest_buy_hit_pairs']} релевантных hit-пар Latest buy "
            f"в CBF50 доступны {overlap['also_in_cbf50']}; "
            f"потеряны {overlap['missing_from_cbf50']}.\n"
        )
    text += """
Если oracle ниже Latest buy, ни выбор LLM, ни перестановки промпта не закроют
средний разрыв при неизменном pool. Если выше — это только необходимая возможность:
LLM ещё должен почти правильно упорядочить имеющиеся targets. Переход к более
широкому pool или encoder меняет постановку и требует новой честной baseline-таблицы.

## Ограничения encoder seam и контекста, измеренные на этих cohorts

"""
    text += table(
        ["Запуск", "Users", "Один distinct history-актив", "Один history feature vector",
         "Из них nonempty", "Profile questionnaire date ≥ cut"],
        [(label, run["metrics"]["cbf"]["users"],
          run["diagnostics"]["single_distinct_history_asset_users"],
          run["diagnostics"]["single_feature_vector_users"],
          run["diagnostics"]["single_feature_vector_nonempty_users"],
          run["diagnostics"]["profiles_questionnaire_at_or_after_cut"])
         for label, run in runs.items()],
    )
    text += """
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
"""
    (root / "reports" / "starter_baselines.md").write_text(text.rstrip() + "\n", encoding="utf-8")
    track_report(root, summary)


def track_report(root, summary):
    warm = summary["runs"]["full_warm"]
    demo = summary["runs"]["demo"]
    d = warm["diagnostics"]
    text = f"""# Сравнение треков в интерфейсах официального starter

> Сгенерировано `scripts/starter_reports.py` из измеренных aggregate/diagnostics.
> Литературные ссылки [n] относятся к [literature.md](literature.md).
> Трек и исследовательский вопрос **не выбираются**: решение за Дарьей.

Starter commit: `{summary['starter_commit']}`.
Повторы официально разрешены и релевантны; baseline comparison включает
Popularity, CBF, Latest buy, Latest buy or sell. HW2 defaults — teaching settings,
не финальный grading protocol.

## Измеренная стартовая точка

"""
    text += table(
        ["Cohort", "Users / positive", "CBF: all / nonempty", "Latest buy: all / nonempty",
         "Oracle fixed CBF50: all / nonempty", "Whole CLI wall, s"],
        [(label, f"{r['metrics']['cbf']['users']} / {r['diagnostics']['nonempty_users']}",
          f"{r['metrics']['cbf']['ndcg_at_10']:.4f} / {r['metrics']['cbf']['ndcg_at_10_nonempty']:.4f}",
          f"{r['metrics']['latest_buy']['ndcg_at_10']:.4f} / {r['metrics']['latest_buy']['ndcg_at_10_nonempty']:.4f}",
          f"{r['diagnostics']['cbf50_oracle']['ndcg_at_10']:.4f} / {r['diagnostics']['cbf50_oracle']['ndcg_at_10_nonempty']:.4f}",
          r["whole_cli_wall_seconds"]) for label, r in summary["runs"].items()],
    )
    text += "\n" + capacity_table(summary)
    text += f"""
На полной warm-выборке повторных distinct future-пар — **{d['repeat_pair_share']:.4f}**,
а повторных Buy-строк — **{d['repeat_buy_row_share']:.4f}**. Это не прежняя общая
event-level доля по всему релизу. Полные знаменатели, micro/macro и Latest-hit
overlap — в [starter_baselines.md](starter_baselines.md).

**Ограничение full warm:** oracle исходного CBF50 =
**{d['cbf50_oracle']['ndcg_at_10_nonempty']:.4f}** nonempty nDCG@10 против
**{warm['metrics']['latest_buy']['ndcg_at_10_nonempty']:.4f}** у Latest buy.
При неизменном pool и протоколе ranker-only не может догнать его в среднем.
На demo oracle =
**{demo['diagnostics']['cbf50_oracle']['ndcg_at_10_nonempty']:.4f}**, то есть
выше demo Latest buy
**{demo['metrics']['latest_buy']['ndcg_at_10_nonempty']:.4f}**; это лишь
возможность на маленькой positive-подвыборке, не свидетельство общего выигрыша.

## Возможности, ограничения и шанс против recency

| Трек | Что реально меняется в stock API | Что говорит литература | Шанс относительно Latest buy | Риски | Стоимость и время |
|---|---|---|---|---|---|
| **Ranker** | `rerank(context, candidate_ids)` возвращает ровно десять уникальных ID из CBF top-50; меняет выбор/порядок внутри pool, не retrieval и не universe | [1], [2]: prompting/verbalizer и candidate ceiling; [3]: candidate policy и сильный non-LLM контроль критичны; [5] содержит пример ухудшения после reranking | Ограничен **измеренным oracle**. Недоступные CBF50 hits Latest buy нельзя вернуть любым LLM. Если oracle выше baseline, это возможность, не вероятность успеха. Demo и full warm не эквивалентны | Position bias, ties по category вместо интереса, неверные/повторные ID, внепуловые ответы, шум профиля и pretrained-name contamination; частотная история не содержит событийной recency | Обычно один actual LLM-вызов на пользователя; перестановки/ensemble умножают инференс и токены. Можно batch/cache, но надо учитывать полный workload. Измерений конкретной LLM ещё нет, GPU-часов и секунд не выдумываем |
| **Encoder** | `encode(context)` возвращает неотрицательные конечные веса **только history IDs**. Старый CBF `content_scores` превращает их в один нормированный профиль и ранжирует весь eligible universe | [4] — настоящие language representations; [5] — alignment с CF. Stock weights seam **не эквивалентен** этим embedding-моделям; [15] мотивирует сильный frequency/recency контроль | Может изменить retrieval и вернуть targets вне исходного CBF50, если веса меняют профиль. Но не различит одинаковые content-векторы; при одном векторе история неизменна при ненулевых весах. Recency не из чего восстановить без доп. past-only данных | Неправильное описание weights как “LLM embeddings”; ложные предпочтения; переобучение на малой positive-подвыборке; дополнительные поля только у LLM | Возможен один offline LLM-вызов/пользователь, затем дешёвый CBF; profile и weights cache должны зависеть от cut/model/prompt. Настоящие embeddings требуют своего score adapter, отдельной реализации и честного контроля, не замены официальных baseline |
| **Hybrid** | Plugin с `encode` и `rerank`: сначала encoder ranking по полному universe, **его** top-50 передаются ranker; исходный CBF pool нельзя незаметно подмешать обратно | [5], [6] — complementary behavioral/semantic signals; [3] — рост от retriever нельзя приписать reranker. Это не гарантирует эффект двух LLM-стадий | Может обойти потолок исходного CBF50 через новый retrieval. Затем остаётся потолок нового pool, который нужно заново измерить. Без выигрыша encoder дорогой ranker может лишь наследовать ограничения | Смешение retrieval/ranking effects, утечки в двух частях, информационное неравенство, invalid outputs, рост инженерного объёма | Encoder-вызовы + ranker-вызовы, больше cache/versioning и минимум ablations baseline/encoder-only/ranker-only/both. До измерений качество/latency лучше не обещать |

### Структурная граница weights seam

"""
    text += table(
        ["Cohort", "Всех users", "История с одним feature vector", "Из них positive"],
        [(label, r["metrics"]["cbf"]["users"],
          r["diagnostics"]["single_feature_vector_users"],
          r["diagnostics"]["single_feature_vector_nonempty_users"])
         for label, r in summary["runs"].items()],
    )
    text += """
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
"""
    (root / "reports" / "track_comparison.md").write_text(text.rstrip() + "\n", encoding="utf-8")
