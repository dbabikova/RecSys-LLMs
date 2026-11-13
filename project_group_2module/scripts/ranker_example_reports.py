"""Render the chosen planned RQ and a source-linked, masked ranker example."""

import json
import math

from starter_reports import table


def raw_status(value):
    if value in ("", "Not_Available"):
        return "missing"
    if value.startswith("Predicted_"):
        return "predicted/estimated"
    return "observed (in release; provenance caveats below)"


def render(root, e, baseline, rq):
    reports = root / "reports"
    protocol, row = e["protocol"], e["raw_record"]["row"]
    cut, end = protocol["cutoff"], protocol["end_exclusive"]
    s = e["selection"]
    q = f"""# Исследовательский вопрос — решение Дарьи

> Сгенерировано `scripts/make_ranker_example.py`. Формулировка — дословное решение
> Бабиковой Дарьи, переданное в чате, а не выбор AI-ассистента.

## Primary research question

**{rq}**

**Track: Ranker. Status: planned.** Модели, LambdaMART и LLM inference на этом
этапе не запускались; конкретная LLM и generator/pool policy не выбираются.

"""
    q += table(["Элемент", "Зафиксировано"], [
        ("Intervention", "Zero-shot LLM reranking фиксированного top-20; изменяется порядок/выбор Top-10 внутри одного и того же списка кандидатов"),
        ("Control", "Classical learned reranker: LightGBM LambdaMART; те же пользователи, кандидаты и user/asset information, доступная до среза"),
        ("Outcome", "Средний nDCG@10 по будущим покупкам; primary по всем common evaluation users с empty truth = 0. Nonempty mean — отдельно"),
        ("Population", "Common FAR-Trans evaluation users. Для учебной иллюстрации — demo cohort самого starter; финальную проектную популяцию не подменяем выбранным примером"),
    ])
    q += """
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

"""
    warm = baseline["runs"]["full_warm"]
    q += table(["Риск из этапа starter", "Измерено (nonempty nDCG@10)"], [
        ("Oracle перестановки исходного CBF top-50", warm["diagnostics"]["cbf50_oracle"]["ndcg_at_10_nonempty"]),
        ("Latest buy на тех же full warm users", warm["metrics"]["latest_buy"]["ndcg_at_10_nonempty"]),
    ])
    q += """
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
"""
    (reports / "research_question.md").write_text(q.rstrip() + "\n", encoding="utf-8")
    text = f"""# Реальный пример для planned Ranker RQ

> Сгенерировано `scripts/make_ranker_example.py` через импорт функций оригинального
> starter. Ни LLM, ни LambdaMART не запускались. Future-derived значения ниже —
> только метка и аудит примера, не входы будущих моделей.

Starter commit: `{e['starter_commit']}`.
Demo protocol_hash: `{protocol['protocol_hash']}`.
Этот хеш относится к сохранённому default demo с candidate_count =
**{protocol['candidate_count']}**; показанный префикс top-20 не является новым
custom-запуском или утверждённым экспериментальным протоколом.
Пути к исходным CSV: `data/FAR-Trans/`; SHA-256 каждого файла сверяется со
`starter_summary.json` и приватным evidence. Клиент везде обозначен `[MASKED]`.
Порядковые номера далее — **CSV data-record без заголовка**, не обязательно
физические номера строк при наличии quoted multiline values.

## 1. Детерминированный выбор примера

"""
    text += table(["Проверка выбора", "Значение"], [
        ("Demo users", s["demo_users"]), ("Demo users с future Buy", s["demo_nonempty_users"]),
        ("Distinct target-пар на позициях CBF 11–20", s["preferred_target_pairs"]),
        ("Users с target на позициях 11–20", s["preferred_users"]),
        ("Distinct target-пар в CBF top-20", s["target_pairs_in_top20"]),
        ("Выбранный пользователь: ordinal в demo (не customerID)", s["selected_demo_ordinal"]),
        ("Правило выбора пары", s["rule"]),
    ])
    text += """
Если preferred range пуст, выбирается самая поздняя позиция среди попавших в
CBF20, затем первый пользователь по deterministic demo order, затем ISIN.
После выбора пары берётся самая ранняя допустимая Buy-строка окна по
(timestamp, числовой transactionID, CSV-record). Это демонстрационный отбор
одного record, **не изменение evaluation cohort и не выбор хороших пользователей
для будущей оценки**.

"""
    if not s["preferred_target_pairs"]:
        text += (
            "В demo нет покупок на позициях 11–20. Поэтому пример НЕ показывает "
            "перенос из-за границы top-10: купленный актив уже внутри top-10, "
            "а потенциальное действие ranker — поднять его выше. Не выдаём "
            "гипотетический подъём за полученный результат LLM.\n\n"
        )
    values = ["[MASKED]" if key == "customerID" else value for key, value in row.items()]
    text += f"""## 2. Сырой record в формате задания

```text
source: FAR-Trans public release (https://researchdata.gla.ac.uk/1658/), FAR-Trans/transactions.csv
record id: transactionID={row['transactionID']} (verbatim); CSV data-record={e['raw_record']['record_number']}
fields: {', '.join(row)}
values: {', '.join(values)}
window: {cut} <= timestamp < {end}; starter demo; repeat allowed
```

Все значения дословно из CSV, кроме обязательной маски customerID.
Наличие записи после cut не делает её поля допустимыми входами.
Знак ID неотрицателен; это исключает negative-ID marker, но не доказывает
отсутствия ретроспективных поправок авторов релиза.

### Совпадения transactionID

"""
    text += table(
        ["CSV-record", "Клиент", "ISIN", "Дата", "Тип", "totalValue", "units", "channel", "marketID"],
        [(r["record_number"], "[MASKED: selected]" if r["row"]["customerID"] == e["user"] else "[MASKED: other]",
          *(r["row"][key] for key in ("ISIN", "timestamp", "transactionType", "totalValue", "units", "channel", "marketID")))
         for r in e["same_transaction_id_rows"]],
    )
    text += f"\nВсего строк с этим transactionID: **{len(e['same_transaction_id_rows'])}**. Shared ID не равен полному duplicate: starter удаляет только одинаковые полные строки.\n"
    text += "\n## 3. История до cut: доказательство из сырых транзакций\n\n"
    text += (
        "Только сохранённые Buy: дата строго до cut, nonnegative ID, eligible asset, "
        "полные дубликаты удалены. Таблица отсортирована по дате, transactionID, "
        "CSV-record; значения исходные. Все строки принадлежат одному `[MASKED]`.\n\n"
    )
    text += table(
        ["CSV-record", "transactionID", "Дата", "ISIN", "units", "totalValue (estimated)", "channel", "marketID"],
        [(r["record_number"], *(r["row"][key] for key in (
            "transactionID", "timestamp", "ISIN", "units", "totalValue", "channel", "marketID"
        ))) for r in e["history_raw"]],
    )
    text += "\n" + table(["Сводка истории", "Значение"], [
        ("Retained Buy-строк", len(e["history_raw"])), ("Distinct history-assets", len(e["history_counts"])),
        *[(f"До cut исключено у этого клиента: {key}", value) for key, value in e["history_omitted"].items()],
    ])
    text += "\n" + table(["ISIN истории", "buy_count", "Название as-of", "Attributes"], [
        (h["id"], h["buy_count"], h["name"], "; ".join(h["attributes"])) for h in e["history_context"]
    ])
    text += """
**В контексте starter нет дат Buy.** Там есть `buy_count` и
`metadata_timestamp` (это дата metadata актива, НЕ last-buy date). Сырые даты
в таблице выше — audit-only. В текущем иллюстративном общем входе они не
добавлены ни LLM, ни LambdaMART. Возможное обогащение event-date/recency признаками
должно быть явно утверждено в study plan и симметрично для обоих методов.

## 4. Профиль и метаданные as-of

Профиль выбирается по максимальному `timestamp` строго до cut (не по последней
строке всего файла). Все snapshots этого клиента:

"""
    text += table(
        ["CSV-record", "customerID", "timestamp", "customerType", "riskLevel", "investmentCapacity", "lastQuestionnaireDate", "Выбран as-of"],
        [(r["record_number"], "[MASKED]", *(r["row"][key] for key in (
            "timestamp", "customerType", "riskLevel", "investmentCapacity", "lastQuestionnaireDate"
        )), r["record_number"] == e["profile_asof"]["record_number"]) for r in e["profile_all_rows"]],
    )
    p = e["profile_asof"]["row"]
    late = p["lastQuestionnaireDate"][:10] >= cut
    text += "\n" + table(["Проверка профиля", "Результат"], [
        ("As-of CSV-record", e["profile_asof"]["record_number"]),
        ("Snapshot timestamp", p["timestamp"]), ("Questionnaire date", p["lastQuestionnaireDate"]),
        ("Questionnaire ≥ cut (нельзя считать доступной)", late),
        ("Questionnaire sentinel", p["lastQuestionnaireDate"] == "2000-01-01"),
        ("riskLevel status в source", raw_status(p["riskLevel"])),
        ("investmentCapacity status в source", raw_status(p["investmentCapacity"])),
    ])
    text += "\nОбщий planned profile block для обоих методов:\n\n```json\n"
    text += json.dumps(e["planned_common_input"]["profile"], ensure_ascii=False, indent=2) + "\n```\n"
    text += (
        "\nЕсли анкета позже/на cut, risk/capacity в planned input маскируются "
        "одинаково для обоих методов; это withheld/missing input, а не якобы "
        "пустое значение исходного CSV. Даже ранняя анкета не доказывает дату "
        "обучения модели, создавшей Predicted_* категории. Общие обнаруженные "
        "late-profile риски см. leakage.md / starter_baselines.md.\n"
    )
    text += "\nСтроки metadata выбранного актива (для видимости будущих updates):\n\n"
    text += table(
        ["CSV-record", "ISIN", "assetName", "assetShortName", "assetCategory", "assetSubCategory", "marketID", "sector", "industry", "timestamp", "Выбрана"],
        [(r["record_number"], *(r["row"][key] for key in (
            "ISIN", "assetName", "assetShortName", "assetCategory", "assetSubCategory",
            "marketID", "sector", "industry", "timestamp",
        )), r["record_number"] == e["metadata_asof"]["record_number"]) for r in e["metadata_all_rows"]],
    )
    text += """
Code-as-of не доказывает, что текущие названия исторически существовали в эту
дату: часть metadata timestamps совпадает с началом цен, а provenance переименований
не дан. При одинаковом timestamp starter сохраняет **первую** строку файла;
нельзя незаметно заменить это на last-row tie-break старой EDA.

## 5. Иллюстративный CBF top-20 и целевая покупка

Это префикс текущего CBF starter для наглядности **размера из RQ**, не принятое
решение о финальном генераторе или составе pool. Future Buy не вставлялся в список.
Идентичные атрибуты могут давать одинаковый score; tie-break — ISIN.

"""
    text += table(
        ["CBF rank", "ISIN", "Название as-of", "Короткое имя", "Атрибуты CBF", "Metadata date", "CBF score (12dp)"],
        [(c["cbf_position"], c["id"], c["source_metadata"]["assetName"], c["source_metadata"]["assetShortName"],
          "; ".join(c["tokens"]), c["source_metadata"]["timestamp"], f"{c['cbf_score']:.12f}")
         for c in e["candidate_assets"]],
    )
    text += "\n" + table(["Выбранный target", "Значение"], [
        ("ISIN", e["selected_isin"]), ("Позиция в CBF full ranking", e["positions"]["cbf"]),
        ("Позиция в Latest buy full ranking", e["positions"]["latest_buy"]),
        ("CBF top-10 membership", e["positions"]["cbf"] <= 10),
        ("CBF top-20 membership", e["positions"]["cbf"] <= 20),
        ("Latest buy top-10 membership", e["positions"]["latest_buy"] <= 10),
        ("Repeat: покупался до cut", e["is_repeat"]),
    ])
    text += """
Latest buy baseline выдаёт только Top-10. Полная позиция для аудита получена
той же `recency_rankings` с расширенным k; её первые десять сверены с сохранённым
на этапе starter `predictions.csv`. Если позиция вне десяти, она не является
засчитанным попаданием этого baseline.

### Все будущие метки этого пользователя

"""
    text += table(
        ["CSV-record", "transactionID", "ISIN", "Дата", "Repeat до cut"],
        [(r["record_number"], r["row"]["transactionID"], r["row"]["ISIN"], r["row"]["timestamp"],
          r["row"]["ISIN"] in e["history_counts"]) for r in e["label_raw"]],
    )
    text += f"\nTruth T = `{json.dumps(e['truth'], ensure_ascii=False)}`; distinct relevance count = **{len(e['truth'])}**.\n"
    text += "\nTop-10 списки для ручной сверки (relevance-флаги только для audit, НЕ для input):\n\n"
    text += table(
        ["Rank", "CBF ISIN", "CBF name", "CBF relevant", "Latest buy ISIN", "Latest buy name", "Latest buy relevant"],
        [(r + 1, e["cbf_order"][r], e["ranking_assets"][e["cbf_order"][r]]["assetName"],
          e["cbf_order"][r] in e["truth"], e["latest_buy_order"][r],
          e["ranking_assets"][e["latest_buy_order"][r]]["assetName"],
          e["latest_buy_order"][r] in e["truth"]) for r in range(10)],
    )
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(10, len(e["truth"])) + 1))
    text += "\n" + table(["Метод", "Relevant ranks в Top-10", "DCG@10", "IDCG@10", "nDCG@10 (12dp)"], [
        (method, ", ".join(str(i + 1) for i, item in enumerate(e[order][:10]) if item in e["truth"]) or "нет",
         f"{sum(1/math.log2(i+2) for i,item in enumerate(e[order][:10]) if item in e['truth']):.12f}",
         f"{idcg:.12f}", f"{e['metrics'][key]:.12f}")
        for method, order, key in (("CBF", "cbf_order", "cbf"), ("Latest buy", "latest_buy_order", "latest_buy"))
    ])
    text += """
Подъём доступного target на более высокий rank изменит DCG, но не T или IDCG.
Ни новый порядок, ни выигрыш LambdaMART/LLM здесь не получены: это только
иллюстрация механизма и измеренные классические baseline.

## 6. Поля будущего общего входа и audit/label-only поля

Ниже — **черновой same-information contract для planned эксперимента**, не
исполненный inference. Общий payload сохранён в игнорируемом
`data/ranker_example/planned_common_input.json`: только past-only данные и
кандидаты CBF, без truth, selected-target marker, будущих дат и сумм.
Он не выбирает окончательный pool. Для LambdaMART ещё предстоит реализовать
user–candidate features и train-only обработку текста; отбрасывать текст,
который видит LLM, без контроля нельзя.

"""
    policy = []
    for field, value in row.items():
        use = "только метка / join (маска)" if field == "customerID" else (
            "только метка" if field in {"ISIN", "transactionType", "timestamp"} else "не используется (audit only)"
        )
        status = "predicted/estimated: пересчитана по closePrice" if field == "totalValue" else (
            "observed/adjusted: splits/reconstruction provenance неполон" if field == "units" else raw_status(value)
        )
        policy.append((
            f"future transaction.{field}", "[MASKED]" if field == "customerID" else value,
            use, status, row["timestamp"], "после cut; эта связь/событие не передаются моделям",
        ))
    policy.extend([
        ("input.user_alias", "EXAMPLE_USER", "LLM / LambdaMART: только ключ, не semantic feature", "observed/служебная маска", cut, "не раскрывает customerID"),
        ("input.cutoff", cut, "LLM / LambdaMART", "observed/параметр протокола", cut, "не источник будущих знаний"),
    ])
    for field in ("customerType", "riskLevel", "investmentCapacity"):
        value = e["planned_common_input"]["profile"][field]["value"]
        policy.append((
            f"input.profile.{field}", value, "LLM / LambdaMART",
            "missing/withheld (source is " + raw_status(p[field]) + ")" if value.startswith("[WITHHELD") else raw_status(p[field]),
            p["timestamp"] + "; questionnaire " + p["lastQuestionnaireDate"],
            "late questionnaire → одинаковая маска; время оценивания Predicted_* неизвестно",
        ))
        policy.append((
            f"input.profile.{field}.status", e["planned_common_input"]["profile"][field]["status"],
            "LLM / LambdaMART", "observed/derived marker, не новая оценка", p["timestamp"],
            "сохранять Predicted_/missing, не представлять оценку как факт",
        ))
    policy.extend([
        ("input.profile.timestamp / placeholder flag / note", p["timestamp"], "LLM / LambdaMART",
         "observed/derived", p["timestamp"], "строго до cut; snapshot ≠ доказанное knowledge time"),
        ("audit.profile.lastQuestionnaireDate", p["lastQuestionnaireDate"], "не используется как score; только availability audit",
         "missing/suspected sentinel" if p["lastQuestionnaireDate"] == "2000-01-01" else "observed",
         p["lastQuestionnaireDate"], "не включено в stock model context; поздняя анкета блокирует risk/capacity"),
        ("history.id / name / attributes / metadata_timestamp", "вся таблица history summary выше", "LLM / LambdaMART",
         "observed metadata; отсутствующие attributes остаются missing", "metadata строго < cut",
         "имя может быть ретроспективным; одинаковый набор полей обоим"),
        ("history.buy_count", json.dumps(e["history_counts"], ensure_ascii=False), "LLM / LambdaMART",
         "observed/derived из retained Buy", "только Buy < cut", "negative IDs и exact duplicates исключены"),
        ("audit.history.timestamp", "даты raw history выше", "не используется в текущем общем payload; audit only",
         "observed in release; исходная дата могла переноситься", "каждая дата < cut",
         "НЕ metadata_timestamp; будущая recency-фича требует симметричного решения"),
        ("audit.history.transactionID / channel / units / totalValue / execution marketID",
         "raw history выше", "не используется в общем payload",
         "observed/adjusted; totalValue estimated", "даты history < cut; очистка post hoc",
         "не придумывать spending/return; неотрицательный ID не гарантирует raw provenance"),
    ])
    meta = e["metadata_asof"]["row"]
    for field, value in meta.items():
        policy.append((
            f"candidate.source_metadata.{field}", value, "LLM / LambdaMART",
            raw_status(value), meta["timestamp"],
            "as-of < cut; справочник релиза не доказывает историчность имени/класса",
        ))
    policy.extend([
        ("candidate.id / tokens", "ISIN / categorical tokens каждой строки CBF20", "LLM / LambdaMART",
         "observed/derived", "metadata < cut", "id для сопоставления output; без future-target флага"),
        ("candidate.cbf_score / cbf_position", "score и rank выше (полная точность в evidence)", "LLM / LambdaMART",
         "observed/derived; НЕ predicted purchase probability", "только past history + as-of features",
         "общий reference score; порядок промпта/перестановки ещё planned"),
        ("eligibility: as-of metadata + positive pre-cut price", len(e["eligible_ids"]), "только общий фильтр протокола",
         "observed/derived", "metadata и цена строго < cut",
         "не требовать цен до конца будущего окна; не выдавать universe будущего за признак"),
        ("future truth set / metrics / selected target / oracle", json.dumps(e["truth"]), "только метка / evaluation",
         "observed/derived", f"[{cut}, {end})", "запрещены в обоих model inputs"),
        ("limit_prices.maxDate / priceMaxDate / profitability; future prices", "не читаются scorer",
         "не используется", "derived from future/full period", "после cut", "прямая утечка; запрет"),
    ])
    text += table(["Поле", "Значение / ссылка на строки", "Кто использует (planned)", "Статус", "Дата / доступность", "Проверка утечки"], policy)
    text += """
**Нерешённые риски:** отрицательные ID удалены, но дополнительные синтетические Sell,
переносы дат и back-adjusted prices/units полностью не маркированы. As-of профиль
не автоматически безопасен при поздней анкете. Вектор CBF не использует имена,
а LLM может извлекать из них знания после cut: нужны same-information текстовый
контроль и проверка анонимизации в study plan. Цены используются только для
eligibility, не прибыльности. Нет ответов анкет, реальных имён или инвестиционных
целей — их нельзя выдумывать из повторных Buy.

## 7. Независимость и личная проверка

Генератор импортирует starter, но `scripts/verify_example.py` **не импортирует**
starter, `starter_audit`, генератор, render helpers или другие наши модули.
Он самостоятельно читает CSV и пересчитывает этот record, selection, history,
profile, metadata, CBF и Latest buy, сравнивая с приватным expected evidence.
Успех/число проверок и короткая ручная инструкция печатаются им в
[verify_example.md](verify_example.md). При несоответствии — ненулевой exit code.
Автоматическая проверка не подменяет ручную проверку Дарьи: **она ещё не выполнена**.

Запуск одной командой каждого шага из папки проекта:

```powershell
python -B scripts/make_ranker_example.py
python -B scripts/verify_example.py
```
"""
    (reports / "ranker_example.md").write_text(text.rstrip() + "\n", encoding="utf-8")
