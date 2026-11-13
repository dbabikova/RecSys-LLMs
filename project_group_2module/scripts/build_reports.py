"""Render Markdown and figures exclusively from audit computations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_core import latest, questionnaire_stats

PAPER = "https://arxiv.org/html/2407.08692v1"
RELEASE = "https://researchdata.gla.ac.uk/1658/"
MISSING = {"", "Not_Available", "Not Available", "N/A", "NA", "null", "NULL"}
SENTINEL = pd.Timestamp("2000-01-01")


def fmt(value):
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    if pd.isna(value):
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def pct(numerator, denominator):
    return f"{100 * numerator / denominator:.4f}%" if denominator else "—"


def md(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(fmt, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(fmt, row)) + " |" for row in rows],
    ]) + "\n"


def date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def write(path, text):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def banner(archive_hash):
    return (
        "> Сгенерировано `scripts/run_analysis.py`; измеренные числа вычислены из CSV, "
        "не вписаны в Markdown вручную. Известные числа статьи отдельно помечены как "
        "цитируемые. Идентификаторы клиентов не публикуются.\n\n"
        f"Архив: `FAR-Trans.zip`, SHA-256: `{archive_hash}`.\n\n"
    )


def source_report(root, archive_hash, tables, stats, sha256):
    raw = root / "data" / "FAR-Trans"
    files = [(p.name, p.stat().st_size, sha256(p)) for p in sorted(raw.iterdir()) if p.is_file()]
    q = questionnaire_stats(raw / "questionnaires.csv")
    text = "# Источник и прочитанные материалы\n\n" + banner(archive_hash)
    text += f"""## Происхождение

- Публичный релиз: {RELEASE}, DOI: `10.5525/gla.researchdata.1658`, лицензия CC BY 4.0.
- Статья: J. Sanz-Cruzado, N. Droukas, R. McCreadie,
  *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*,
  Fin-RecSys / IJCAI, arXiv v1: {PAPER}.
- Прочитаны README архива и полный текст статьи. Это чтение описания данных,
  не обзор литературы для выбора RQ.
- Источник локального ZIP: `C:/Users/User/Downloads/FAR-Trans.zip`.
  Указанный пользователем каталог `Downloads/FAR-Trans/` уже содержал распакованные
  CSV; использован найденный рядом исходный архив. Копия и извлечение находятся
  только в игнорируемом `data/`. Внешние исходники не изменялись.
- SHA-256 README: `{sha256(raw / "README.txt")}`.

## Как собраны и преобразованы данные (статья, раздел Dataset)

Данные предоставлены крупным европейским финансовым учреждением и описывают
доступный греческим инвесторам рынок. Есть сделки клиентов, профили, публично
собранные атрибуты активов и очищенные ценовые ряды. Клиенты анонимизированы;
публичность релиза не отменяет осторожность с финансовыми профилями.

Авторы удалили транзакции без клиента; привели цены к евро; устранили дубликаты
цен и нулевые ряды, исключили активы с длинными пробелами. Короткие пробелы и
ошибочные скачки оценивались скользящим средним предыдущих дней. При выборе
дубликата на смене валюты использовались также последующие дни.
Сплиты подтверждались по Yahoo Finance: прошлые цены делились, а количества
бумаг умножались на split ratio. Такая ретроспективная очистка не является
строгим point-in-time слепком.

Если продажи превышали покупки по итогам всего периода, авторы добавляли покупку
на первую доступную дату цены, восстанавливая исходное владение. Это оценка на
основе будущих продаж, а не наблюдённая покупка в указанную дату. Также добавлялись
продажи дробных долей после reverse split и продажи оставшихся позиций в конце
ценового ряда; даты выходящих за ряд сделок переносились к ближайшей границе.
Их первоначальные даты и отдельный полный provenance в релизе не предоставлены.
Стоимость сделок переоценивалась как units × closePrice, а не сохранялась как
независимая фактическая цена исполнения.

Риск определялся анкетой; при её отсутствии авторы оценивали его из альтернативных
клиентских данных и регрессионных весов вопросов. Инвестиционная ёмкость также
могла оцениваться. Ответов клиентов в релизе нет: файл questionnaires — описание
вопросов, не таблица ответов. Его фактический состав:

"""
    text += md(["Показатель questionnaires.csv", "Значение"], q.items())
    text += """\n## Протокол статьи (Experimental setup)

Статья использует бинарный user–item interest по покупкам. Покупавшиеся до
прогноза пары удаляются из будущих меток и из рекомендаций. Оставляются
пользователи с взаимодействием и в train, и в test; ценовой ряд актива должен
покрывать всё тестовое окно. Мы операционализируем взаимодействие как Buy,
поскольку именно из него статья строит бинарную матрицу.
Порогов минимальной длины истории сверх этого не добавляем.

"""
    text += md(["Параметр (цитата статьи, не результат запуска)", "Значение"], [
        ("Первый срез", "2019-08-01"), ("Количество временных вариантов", 61),
        ("Шаг", "примерно две недели"), ("Горизонт", "6 календарных месяцев"),
        ("Основные метрики статьи", "nDCG@10 и ROI@10"),
        ("Подбор параметров", "ROI через 6 месяцев; даты 2019-04-01, 2019-10-01, 2020-01-31"),
    ])
    text += """
Отчёт этого этапа — один последний допустимый срез, не среднее по сетке статьи
и не попытка воспроизвести опубликованные оценки моделей. Статья описывает
prior-to-t и будущее окно; для дат без времени используем прогноз после закрытия
дня среза: train ≤ cut, cut < labels ≤ end. Это явное граничное допущение.
При дальнейшем исследовании настройка не должна затрагивать тестовое окно.

## Бейзлайны статьи

"""
    text += md(["Группа", "Алгоритмы"], [
        ("Без данных", "Random"),
        ("Цены", "Linear regression; Random forest; LightGBM regression"),
        ("Транзакции", "Popularity; LightGCN; MF; UB kNN; Apriori ARM"),
        ("Hybrid", "LightGBM regression; LambdaMART (Hybrid-nDCG), признаки — выходы остальных методов"),
    ])
    text += """
Popularity упорядочивает активы по числу прошлых покупок. На этом этапе считаем
только его и диагностический Repeat+Popularity, не запускаем остальные модели.
В статье подбор по ROI не эквивалентен подбору по nDCG. Курс требует nDCG по
будущим покупкам; это не мера прибыльности или инвестиционной пригодности.

## Манифест извлечённого архива

"""
    text += md(["Файл", "Байт", "SHA-256"], files)
    write(root / "reports" / "source.md", text)


def field_summary(raw, typed, col):
    source = raw[col]
    empty = source.eq("")
    semantic = source.isin(MISSING)
    series = typed[col]
    valid = series[~semantic]
    if pd.api.types.is_datetime64_any_dtype(series):
        logical = "date → datetime64[ns]"
        limits = f"{date(valid.min())} … {date(valid.max())}" if valid.notna().any() else "—"
    elif pd.api.types.is_numeric_dtype(series):
        logical = str(series.dtype)
        limits = f"{valid.min():.12g} … {valid.max():.12g}"
    else:
        logical = "string (object)"
        lengths = source[~semantic].str.len()
        limits = f"длина {lengths.min()} … {lengths.max()}"
        if col not in {"customerID", "transactionID", "ISIN"} and valid.nunique() <= 16:
            limits += "; " + ", ".join(sorted(valid.unique()))
    return [
        col, logical, pct(empty.sum(), len(raw)), pct(semantic.sum(), len(raw)),
        int(source[~empty].nunique()), int(source[~semantic].nunique()), limits,
    ]


def schema_report(root, archive_hash, raw, tables, stats):
    tx, prices = tables["transactions"], tables["close_prices"]
    assets, profiles = tables["asset_information"], tables["customer_information"]
    limits, markets = tables["limit_prices"], tables["markets"]
    text = "# Схема, целостность и сверка со статьёй\n\n" + banner(archive_hash)
    text += """Пустое поле и явный `Not_Available` учитываются раздельно.
`unique nonempty` включает текстовые маркеры, `unique known` исключает их.
Строковые ID не преобразуются в числа; диапазон для строк — длина, не
искусственный числовой порядок идентификаторов. Значение даты-заглушки
отдельно исследуется в отчёте о времени. Технических NaN в прочитанных строках
нет: загрузчик сохраняет исходные строки, затем явно типизирует даты и числа.

## Таблицы

"""
    text += md(["Файл", "Строк", "Столбцов", "Точных дубликатов сверх первого"], [
        (name + ".csv", len(f), len(f.columns), int(f.duplicated().sum()))
        for name, f in tables.items()
    ])
    q = questionnaire_stats(root / "data" / "FAR-Trans" / "questionnaires.csv")
    text += (
        f"\n`questionnaires.csv` — не CSV-таблица: {q['lines']} текстовых строк, "
        f"{q['questions']} вопросов, {q['answers']} вариантов ответа; "
        "число столбцов, типы, ключи и пропуски таблицы неприменимы. "
        "Ответов и клиентских записей в нём нет. README — документация, не таблица.\n"
    )
    for name, frame in tables.items():
        text += f"\n### {name}.csv\n\n"
        text += md(
            ["Поле", "Тип после разбора", "Пусто", "Пусто/маркер missing",
             "Unique nonempty", "Unique known", "Диапазон / категории"],
            [field_summary(raw[name], frame, col) for col in frame],
        )
        date_cols = frame.select_dtypes(include="datetime").columns
        if len(date_cols):
            text += "\n" + md(["Дата", "Непустых неразбираемых значений"], [
                (col, int((raw[name][col].ne("") & frame[col].isna()).sum()))
                for col in date_cols
            ])
    text += "\n## Кандидатные ключи\n\n"
    keys = [
        ("asset_information", ["ISIN"]),
        ("asset_information", ["ISIN", "timestamp"]),
        ("asset_information", ["ISIN", "marketID", "timestamp"]),
        ("customer_information", ["customerID"]),
        ("customer_information", ["customerID", "timestamp"]),
        ("close_prices", ["ISIN", "timestamp"]),
        ("limit_prices", ["ISIN"]), ("markets", ["marketID"]),
        ("markets", ["exchangeID"]), ("transactions", ["transactionID"]),
        ("transactions", ["customerID", "transactionID"]),
    ]
    text += md(["Таблица", "Ключ", "Unique", "Дубликаты сверх первого", "Строк во всех группах коллизий"], [
        (name, ", ".join(cols), len(tables[name].drop_duplicates(cols)),
         int(tables[name].duplicated(cols).sum()),
         int(tables[name].duplicated(cols, keep=False).sum()))
        for name, cols in keys
    ])
    collisions = tx.groupby("transactionID").agg(
        customers=("customerID", "nunique"), assets=("ISIN", "nunique"),
        dates=("timestamp", "nunique"), types=("transactionType", "nunique"),
    )
    text += "\n" + md(["Коллизии transactionID: групп с …", "Количество"], [
        ("несколькими клиентами", int(collisions.customers.gt(1).sum())),
        ("несколькими активами", int(collisions.assets.gt(1).sum())),
        ("несколькими датами", int(collisions.dates.gt(1).sum())),
        ("несколькими типами сделки", int(collisions.types.gt(1).sum())),
    ])
    text += """
transactionID не является глобальным PK: нельзя удалять строки только по нему.
Для воспроизводимого указания строки используем номер CSV-record вместе с
verbatim transactionID; исходные события автоматически не дедуплицируем.
Многократный customerID обычно означает историю профиля, не ошибку.
Многократный ISIN может означать несколько площадок/описаний; даже ISIN+timestamp
нуждается в проверке. В descriptive EDA один актив представлен последней строкой
по timestamp, при равенстве — последней строкой файла (stable); это техническое
правило, не доказательство правильного point-in-time атрибута.

## Внешние связи

"""
    relations = [
        ("transactions", "ISIN", "asset_information", "ISIN"),
        ("transactions", "ISIN", "close_prices", "ISIN"),
        ("transactions", "customerID", "customer_information", "customerID"),
        ("asset_information", "ISIN", "close_prices", "ISIN"),
        ("close_prices", "ISIN", "asset_information", "ISIN"),
        ("limit_prices", "ISIN", "close_prices", "ISIN"),
        ("transactions", "marketID", "markets", "marketID"),
        ("asset_information", "marketID", "markets", "marketID"),
    ]
    relation_rows = []
    for left, lkey, right, rkey in relations:
        missing = ~tables[left][lkey].isin(tables[right][rkey])
        missing_values = sorted(tables[left].loc[missing, lkey].unique())
        public = "не публикуем клиентские ID" if lkey == "customerID" else ", ".join(missing_values)
        relation_rows.append((f"{left}.{lkey} → {right}.{rkey}",
                              int(missing.sum()), len(missing_values), public or "—"))
    text += md(["Связь", "Строк без соответствия", "Unique без соответствия", "Публичные ID"], relation_rows)
    linked = stats["linked"]
    b = stats["bounds"]
    bounds_link = tx.join(b, on="ISIN")
    metadata_join = assets.merge(
        b.rename(columns={"min": "first_price", "max": "last_price"}),
        left_on="ISIN", right_index=True, how="left", validate="many_to_one",
    )
    limit_check = limits.merge(
        b.rename(columns={"min": "first_price", "max": "last_price"}),
        left_on="ISIN", right_index=True, validate="one_to_one",
    )
    price_ends = prices.sort_values("timestamp").groupby("ISIN").agg(
        first=("closePrice", "first"), last=("closePrice", "last")
    )
    limit_check = limit_check.join(price_ends, on="ISIN")
    gap_days = prices.sort_values(["ISIN", "timestamp"]).groupby("ISIN").timestamp.diff().dt.days
    checks = [
        ("Транзакции без цены на точную дату", int(linked.closePrice.isna().sum())),
        ("Транзакции до начала своего ряда", int(bounds_link.timestamp.lt(bounds_link["min"]).sum())),
        ("Транзакции после конца своего ряда", int(bounds_link.timestamp.gt(bounds_link["max"]).sum())),
        ("Транзакции на последнюю глобальную дату сделок", int(tx.timestamp.eq(tx.timestamp.max()).sum())),
        ("Строки с сопоставленной ценой", stats["value_matched"]),
        ("Из них totalValue ≈ units × closePrice (rtol=1e-6, atol=0.01)", stats["value_exact"]),
        ("Close price ≤ 0", int(prices.closePrice.le(0).sum())),
        ("Units ≤ 0", int(tx.units.le(0).sum())),
        ("TotalValue ≤ 0", int(tx.totalValue.le(0).sum())),
        ("Несовпадение minDate с фактическим началом ряда", int(limit_check.minDate.ne(limit_check.first_price).sum())),
        ("Несовпадение maxDate с фактическим концом ряда", int(limit_check.maxDate.ne(limit_check.last_price).sum())),
        ("Несовпадение priceMinDate с первой ценой", int((~np.isclose(limit_check.priceMinDate, limit_check["first"])).sum())),
        ("Несовпадение priceMaxDate с последней ценой", int((~np.isclose(limit_check.priceMaxDate, limit_check["last"])).sum())),
        ("Несовпадение profitability с last/first−1", int((~np.isclose(limit_check.profitability, limit_check["last"] / limit_check["first"] - 1)).sum())),
        ("Максимальный разрыв между соседними ценами (календарных дней)", int(gap_days.max())),
        ("Разрывы цен > 10 календарных дней", int(gap_days.gt(10).sum())),
        ("Timestamp метаданных равен началу ценового ряда (строк)", int(metadata_join.timestamp.eq(metadata_join.first_price).sum())),
    ]
    text += "\n## Дополнительные проверки\n\n" + md(["Проверка", "Значение"], checks)
    text += (
        "\nСлово «очищены» не означает, что все сделки имеют цену в тот же день; "
        "несовпадения не исправлялись автоматически. Календарные пробелы включают "
        "выходные/праздники, поэтому сами по себе не равны ошибке сбора.\n"
    )
    text += "\n## Сверка со статьёй, Table 1 и раздел Customers\n\n"
    latest_profiles = stats["latest_profiles"]
    comparisons = [
        ("Транзакции", 388049, len(tx)),
        ("Клиенты", 29090, tx.customerID.nunique()),
        ("Активы: Table 1 / metadata unique", 806, assets.ISIN.nunique()),
        ("Активы: текст Assets / price unique", 807, prices.ISIN.nunique()),
        ("Активы с транзакциями", 321, tx.ISIN.nunique()),
        ("Рынки", 38, markets.marketID.nunique()),
        ("Точки цен", 703303, len(prices)),
        ("Покупки", 228913, len(stats["buys"])),
        ("Продажи", 159136, int(tx.transactionType.eq("Sell").sum())),
        ("Unique transactions как customer+ISIN+type", 154103, len(tx.drop_duplicates(["customerID", "ISIN", "transactionType"]))),
        ("Unique acquisitions как customer+ISIN", 89884, len(stats["buys"].drop_duplicates(["customerID", "ISIN"]))),
        ("Unique sales как customer+ISIN", 64219, len(tx[tx.transactionType.eq("Sell")].drop_duplicates(["customerID", "ISIN"]))),
        ("Оценённый риск: последние профили", 7141, int(latest_profiles.riskLevel.str.startswith("Predicted_").sum())),
        ("Оценённая ёмкость: последние профили", 7318, int(latest_profiles.investmentCapacity.str.startswith("Predicted_").sum())),
    ]
    text += md(["Показатель", "В статье", "В архиве", "Архив − статья"], [
        (name, expected, actual, actual - expected) for name, expected, actual in comparisons
    ])
    text += (
        "\nВ статье внутри самого текста фигурируют разные размеры universe; таблица "
        "выше показывает обе трактовки, а не выбирает удобную. Семантика unique в "
        "статье операционализирована через customer–asset–type, не через transactionID. "
        "Причину отсутствующей продажи и отличия числа торгуемых активов по этому "
        "релизу установить нельзя: не приписываем их удалению конкретной строки. "
        "Расхождение покрытия metadata/prices показано отдельно по ISIN. "
        "README использует написания CAP_LT_30K, а CSV — CAP_LT30K; "
        "канонизацию по предположению не применяем.\n"
    )
    write(root / "reports" / "schema.md", text)
    return comparisons


def feature_policy(table, col):
    if table == "transactions":
        if col == "totalValue":
            return "predicted/estimated", "дата сделки, но пересчитано авторами по closePrice", "Не цена исполнения; post-close; только прошлые сделки"
        if col == "units":
            return "observed + estimated/adjusted", "дата сделки; поправки после corporate actions", "Ретроспективная split-adjustment; у синтетики количество восстановлено"
        if col == "timestamp":
            return "observed или estimated", "дата в релизе; исходная дата могла быть перенесена", "Не гарантирует исходное время события; синтетика backdated"
        if col == "transactionID":
            return "observed/служебное; отрицательные inferred synthetic", "при регистрации или реконструкции", "Не семантический признак; не уникальный глобальный PK"
        return "observed; у синтетики reconstructed/inherited", "при сделке, не ранее её даты", "Будущие сделки — только метки; негативный ID не гарантирует полноту маркировки синтетики"
    if table == "customer_information":
        if col in {"riskLevel", "investmentCapacity"}:
            return "observed / predicted / missing", "as-of timestamp И дата анкеты; прогнозная модель не датирована", "Predicted_ — оценка; Not_Available — missing; backdating/неизвестное обучение"
        if col == "lastQuestionnaireDate":
            return "observed или missing/sentinel", "не ранее заполнения анкеты", "Нельзя использовать анкету из будущего относительно cut, даже при старом timestamp"
        if col == "timestamp":
            return "observed служебная дата, валидность проверяется", "по README — обновление профиля", "Даты-заглушки и анкета позже timestamp: as-of join недостаточен"
        return "observed/служебное; историческая доступность не доказана", "не ранее подтверждённого снимка профиля", "Последний снимок всего релиза не равен as-of; ID только ключ"
    if table == "asset_information":
        if col == "timestamp":
            return "observed служебная дата; не verified publication time", "по README — обновление; часто совпадает с началом цен", "Не доказывает историческую доступность текущих имён/категорий"
        return "observed/public metadata или missing", "не ранее публикации атрибута; точная дата не подтверждена", "Snapshot/name-change leakage; дубликаты; нельзя автоматически считать текст историческим"
    if table == "close_prices":
        if col == "closePrice":
            return "observed + estimated + split-adjusted; per-row mask отсутствует", "после закрытия торгов на timestamp; очистка ретроспективная", "Заполнение/валюта/сплиты используют весь релиз; только trailing-признаки, provenance неполон"
        return "observed служебное поле", "дата ценового наблюдения", "ID только ключ; будущие timestamps и точки не входят в признаки"
    if table == "limit_prices":
        if col in {"maxDate", "priceMaxDate", "profitability"}:
            return "derived/estimated", "только после конца полного ценового ряда", "Прямая утечка будущего: запрет в признаках; maxDate — только ex-post фильтр протокола"
        return "derived/public identifier", "minDate/priceMinDate после начала ряда; полный summary получен post hoc", "Пересчитать из префикса цен ≤cut; не использовать полный summary как признаки"
    if table == "markets":
        return "observed/public metadata или missing", "дата публикации не предоставлена", "Справочник текущего релиза; историчность расписаний/описаний не доказана"
    raise KeyError((table, col))


def leakage_report(root, archive_hash, raw, tables, stats):
    tx, neg = stats["tx"], stats["negative"]
    profiles = tables["customer_information"]
    cut = stats["cut"]
    asof = stats["asof_profiles"]
    q_future = profiles.lastQuestionnaireDate.gt(profiles.timestamp)
    text = "# Время доступности, оценённые поля и утечки\n\n" + banner(archive_hash)
    text += (
        "Статус ниже относится к происхождению поля, а не к гарантии его "
        "point-in-time корректности. Дата события ≠ дата публикации/очистки. "
        "Для каждого поля даём консервативное правило. Реальная отсутствующая "
        "доля и диапазоны перечислены в schema.md. Поля без подтверждённой "
        "исторической доступности не объявляем безопасными.\n\n"
    )
    for name, frame in tables.items():
        text += f"## {name}.csv\n\n"
        text += md(["Поле", "Статус происхождения", "Когда известно", "Риск / правило"], [
            (col, *feature_policy(name, col)) for col in frame
        ]) + "\n"
    text += "## Проверяемые признаки синтетики и ограничения распознавания\n\n"
    text += md(["Проверка", "Количество"], [
        ("transactionID < 0", len(neg)),
        ("Из отрицательных: Buy", int(neg.transactionType.eq("Buy").sum())),
        ("Из отрицательных: Sell", int(neg.transactionType.eq("Sell").sum())),
        ("Из отрицательных: на первой дате цены актива", int(neg.timestamp.eq(neg.first_price).sum())),
        ("Из отрицательных: не на первой дате цены", int(neg.timestamp.ne(neg.first_price).sum())),
        ("Отрицательных после cut", int(neg.timestamp.gt(cut).sum())),
        ("Отрицательных в последнем окне меток", int((neg.timestamp.gt(cut) & neg.timestamp.le(stats["end"])).sum())),
        ("Отрицательных до/на cut", int(neg.timestamp.le(cut).sum())),
        ("Положительных/нулевых Buy на первой дате цены", int(
            stats["linked"].loc[
                tx.transactionID.ge(0) & tx.transactionType.eq("Buy")
            ].join(stats["bounds"], on="ISIN").pipe(
                lambda f: f.timestamp.eq(f["min"]).sum()
            )
        )),
    ])
    text += """
Отрицательный ID — сильный эмпирический маркер восстановленных стартовых покупок:
знак не документирован README как официальный флаг, но совпадение с описанным
механизмом проверено выше. Нельзя просто называть все записи с неотрицательным ID
«доказанно реальные»: восстановленные продажи дробных долей, ликвидации на границе
ряда и перенесённые даты не имеют отдельного достоверного флага.
Совпадение с первой/последней датой цены — только эвристика, не доказательство
синтетического события. Точное полное число всех преобразованных событий
из этого архива восстановить нельзя.

Восстановленные начальные владения рассчитаны авторами по сальдо будущих продаж
всего периода. Если включить их в историю раннего среза, возникает ретроспективная
утечка. Основной диагностический сплит ниже сохраняет их в train для близости к
статье, но НЕ называется leak-free. В метках они запрещены. Дополнительно
считается чувствительность при удалении всех распознаваемых восстановленных
покупок из истории. Это не восстанавливает весь исходный point-in-time лог.

## Профили и противоречия времён

"""
    text += md(["Проверка", "Количество"], [
        ("Профиль: lastQuestionnaireDate > timestamp", int(q_future.sum())),
        ("Клиентов с хотя бы одним таким профилем", profiles.loc[q_future, "customerID"].nunique()),
        ("lastQuestionnaireDate равна sentinel", int(profiles.lastQuestionnaireDate.eq(SENTINEL).sum())),
        ("timestamp профиля равен sentinel", int(profiles.timestamp.eq(SENTINEL).sum())),
        ("As-of снимков на cut", len(asof)),
        ("Клиентов без снимка на cut", tx.customerID.nunique() - len(asof)),
        ("As-of снимков с датой анкеты > cut", int(asof.lastQuestionnaireDate.gt(cut).sum())),
        ("As-of снимков с sentinel анкеты", int(asof.lastQuestionnaireDate.eq(SENTINEL).sum())),
    ])
    text += f"""
Дата-заглушка в проверках — `{date(SENTINEL)}`: считаем подозрительной, а не
подтверждённой реальной датой анкеты. Последний профиль ≤ cut выбирается
детерминированно по timestamp; если его анкета позже cut, значения профиля
нельзя считать известными в момент прогноза. Для безопасного будущего метода
нужно замаскировать такие поля/проверить provenance у авторов, а не просто сделать
as-of join. У оценённых полей неизвестны дата обучения и окно альтернативных данных.

## Политика дальнейшей работы

- Любые будущие Buy/Sell, суммы и количества — только для меток/аудита, не для промпта.
- Никаких full-period profitability, конечных цен и maxDate в признаках.
- Полное покрытие цен окна — ex-post условие статьи, не признак; оно создаёт
  survivorship/availability selection, что нужно явно признать.
- Дневная цена доступна после закрытия, а не утром той же даты; не обучать
  intra-day решение на сегодняшнем close.
- Полные профили, текущие имена и описания рынков нельзя выдавать за исторические
  без проверки. Для текста нужна абляция анонимизации/без названий.
- В этом этапе профили и текст не используются ни одним из двух ориентиров.
  Контаминацию знаний предобученной LLM этот аудит не исключает.
- Отсутствие покупки — implicit unknown, а не доказанное отрицательное предпочтение;
  финансовую пригодность и доходность nDCG не измеряет.
"""
    write(root / "reports" / "leakage.md", text)


def split_table(rows):
    return md(
        ["Вариант", "Users до warm-фильтра", "Без Buy-истории", "Оцениваются",
         "Релевантных: mean", "median", "p90", "max",
         "nDCG@10 Pop", "nDCG@10 Repeat+Pop"],
        [(r["variant"], r["label_users_before_warm_filter"], r["cold_users"],
          r["users"], r["relevant_mean"], r["relevant_median"],
          r["relevant_p90"], r["relevant_max"],
          r["ndcg_popularity"], r["ndcg_repeat_popularity"]) for r in rows],
    )


def split_report(root, archive_hash, tables, stats, split_rows, sensitivity, details):
    cut, end = stats["cut"], stats["end"]
    tx, prices = stats["tx"], tables["close_prices"]
    text = "# Предварительный сплит и простые ориентиры\n\n" + banner(archive_hash)
    text += f"""**Историческая диагностика, не протокол HW2 или финального теста.**

Преподаватель теперь подтвердил: повторные покупки разрешены и релевантны.
**Вариант A ниже — только new-only диагностика**, не альтернативный основной
benchmark. B совпадает с задачей по разрешению повторов, но не по остальным
настройкам. Официальный starter использует другое окно, eligibility и очистку;
его результаты находятся в `starter_baselines.md`.

Протокол статьи требует полного ценового покрытия будущего окна.
Последняя дата цен — **{date(prices.timestamp.max())}**, транзакций —
**{date(tx.timestamp.max())}**. Поэтому общий end = **{date(end)}**, последний
возможный cut при горизонте статьи — **{date(cut)}**.
Дата среза вычислена из фактических максимумов, а не взята из прежней работы.
Выбирается самая поздняя календарная дата, чей forward DateOffset(months=6)
не выходит за общий end. Это не обязательно точка двухнедельной сетки статьи:
её полный список и граничные соглашения в статье не перечислены.

Прогноз после закрытия дня cut: **train ≤ {date(cut)}**,
**{date(cut)} < labels ≤ {date(end)}**. Внутридневных времён нет.
Cut-date не теряется между train и test. Полнота цен означает minDate ≤ cut и
maxDate ≥ end; не требуем цены в каждый календарный выходной. Внутренние пробелы
проверены в schema.md; новые интерполяции здесь не применяются.

"""
    label_real = stats["real_buys"]
    window_all = label_real[label_real.timestamp.gt(cut) & label_real.timestamp.le(end)]
    text += md(["Показатель сплита", "Значение"], [
        ("Длина горизонта (календарных месяцев, параметр статьи)", 6),
        ("Eligible assets с полным ценовым диапазоном", len(stats["eligible"])),
        ("Из них без metadata", len(stats["eligible"] - set(tables["asset_information"].ISIN))),
        ("Buy в train, включая отрицательные ID", details["B"]["history_rows"]),
        ("Из них в eligible universe", details["B"]["history_eligible_rows"]),
        ("Неотрицательные Buy в окне до фильтра universe", len(window_all)),
        ("Buy в окне вне eligible universe", int((~window_all.ISIN.isin(stats["eligible"])).sum())),
        ("Неотрицательные Buy после общего end, но ещё в транзакциях", int(label_real.timestamp.gt(end).sum())),
    ])
    text += """
## Определения меток, пользователей и ориентиров

- **A:** distinct клиент–актив из будущих Buy с неотрицательным ID, не покупавшийся
  клиентом до/на cut; прежние покупки исключаются и из рекомендаций.
- **B:** distinct клиент–актив из всех таких Buy в окне, включая покупки знакомых
  активов; прежние покупки разрешены в рекомендациях.
- Повторные Buy одной пары внутри окна дают единичную бинарную релевантность,
  а не накопленный gain. Sell не является позитивной меткой.
- Сначала считаем всех пользователей с хотя бы одной допустимой меткой, затем
  диагностируем cold-start (нет Buy в eligible universe до/на cut). Основная
  оценка, по warm-start требованию статьи, исключает их. Пользователи без
  релевантных меток не включаются. Список пользователей A и B может различаться:
  нельзя трактовать разницу их средних как чистый эффект модели.
- **Popularity:** число Buy-строк train для каждого eligible актива, включая
  распознаваемые реконструкции в основном расчёте; tie-break — ISIN по возрастанию.
  Непокупавшиеся eligible активы имеют нулевой score, но входят в общий universe.
  История для этого счётчика берётся у всех прошлых клиентов, а не только у тех,
  кто позже попадёт в оценку. Warm-start фильтр здесь применяется к оценочной
  популяции; это явная локальная операционализация, не утверждение о внутреннем
  порядке фильтрации в коде авторов статьи.
- **Repeat+Popularity:** сначала свои активы по частоте Buy train, затем по
  последней дате Buy, затем по global popularity и ISIN; после них оставшиеся
  активы в порядке Popularity, без дубликатов.
- В A собственные покупки запрещены, поэтому Repeat+Popularity обязан в точности
  совпасть с Popularity. Это проверяется assert, а не только округлением метрик.
- nDCG — среднее по пользователям; бинарный DCG использует discount = 1/log2(rank+1),
  IDCG строится по min(cutoff, числу всех релевантных активов в eligible universe),
  не только найденным рекомендациям. Десять уникальных eligible ID проверяются.

## Основной расчёт: реконструкции в истории, как в опубликованном релизе

Это диагностическое воспроизведение подхода статьи, **не leak-free baseline**:
backdated начальные позиции могут зависеть от будущих продаж.

"""
    text += split_table(split_rows)
    text += "\n" + md(
        ["Вариант: все label-users до warm-фильтра",
         "Нет Buy-истории вообще", "Нет Buy/Sell-истории вообще",
         "Нет Buy-истории в eligible universe"],
        [(r["variant"], r["no_buy_history_any_asset"],
          r["no_transaction_history_any_asset"], r["cold_users"]) for r in split_rows],
    )
    text += "\n## Чувствительность: отрицательные ID исключены и из train\n\n"
    text += (
        "Остальные правила неизменны; A переопределяется относительно оставшейся "
        "истории, поэтому меняется также состав меток и пользователей. "
        "Это не парное сравнение качества на фиксированном тесте, а аудит "
        "зависимости постановки от реконструированной истории.\n\n"
    )
    text += split_table(sensitivity)
    common = set(details["A"]["relevance"]) & set(details["B"]["relevance"])
    text += f"\nОбщих оцениваемых пользователей у A и B в основном расчёте: {len(common)}.\n"
    text += """
## Ограничения и воспроизводимость

Кандидатный universe определяется post hoc ценовым покрытием по статье, не
обучается по меткам. Полнота ряда до конца окна сама по себе использует будущее
для отбора активов; это ограничение протокола, не допустимый пользовательский
признак. Ограничения реконструкций, переноса дат и price-adjustments остаются.
Профили и метаданные не участвуют в ранжировании.

Методы детерминированные, stochastic repeats не нужны. Здесь нет поиска
гиперпараметров и победителя, подбираемого на этом окне. Время выполнения этого
EDA-пайплайна не выдаётся за сопоставимый inference benchmark будущих моделей.
Внешних model API нет; стоимость LLM-инференса на данном этапе отсутствует.
Отдельных статистических правил официального benchmark не вводим.
"""
    write(root / "reports" / "split.md", text)


def eda_report(root, archive_hash, tables, stats, split_rows):
    tx, buys, real_buys = stats["tx"], stats["buys"], stats["real_buys"]
    meta, profiles = stats["metadata"], tables["customer_information"]
    text = "# EDA для дальнейшего выбора RQ\n\n" + banner(archive_hash)
    text += (
        "Это измеренные результаты разведки, не готовый выбор трека или "
        "доказательство эффекта LLM. Все доли покупок ниже считаются по строкам "
        "Buy, если явно не сказано иначе; не смешиваем их со всеми Buy+Sell.\n\n"
    )
    counts = [
        ("Транзакции", len(tx)), ("Клиенты в транзакциях", tx.customerID.nunique()),
        ("Активы в metadata, unique ISIN", tables["asset_information"].ISIN.nunique()),
        ("Активы с ценами, unique ISIN", tables["close_prices"].ISIN.nunique()),
        ("Активы с транзакциями, unique ISIN", tx.ISIN.nunique()),
        ("Покупки всего", len(buys)), ("Buy с неотрицательным ID", len(real_buys)),
        ("Распознаваемые восстановленные Buy (ID < 0)", len(stats["negative"])),
        ("Продажи", int(tx.transactionType.eq("Sell").sum())),
    ]
    text += "## Основные числа\n\n" + md(["Показатель", "Количество"], counts)
    text += "\n## Активность клиентов: разных купленных активов\n\n"
    text += md(
        ["История", "Клиентов", "Mean", "Median", "p90", "p99", "Max", "≤3 разных активов"],
        [(label, len(values), values.mean(), values.median(), values.quantile(.9),
          values.quantile(.99), values.max(), pct(values.le(3).sum(), len(values)))
         for label, values in [
             ("Все Buy, включая реконструкции", stats["activity"]),
             ("Только ID ≥ 0; нули сохранены для всех клиентов", stats["activity_real"]),
         ]],
    )
    text += (
        "\nВо втором распределении клиенты без неотрицательных Buy сохраняются "
        "с нулём; знаменатель не ограничен активными покупателями. Это "
        "описание всей истории, а не cold-start оценочного среза.\n"
    )
    text += "\n## Повторные покупки\n\n"
    text += md(
        ["Набор", "Buy rows", "Distinct пары", "Сверх первой строки пары", "Доля",
         "Строго позже первого дня покупки", "Доля позже дня", "Лишние в первый день"],
        [(label, r["rows"], r["pairs"], r["beyond_first_row"],
          pct(r["beyond_first_row"], r["rows"]), r["strictly_later_day"],
          pct(r["strictly_later_day"], r["rows"]), r["extra_same_first_day"])
         for label, r in [
             ("Все Buy", stats["repeat"]), ("Только ID ≥ 0", stats["repeat_real"])
         ]],
    )
    text += (
        "\nГлавная event-level доля — (Buy rows − distinct customer–ISIN) / Buy rows. "
        "Она считает повтором и дополнительную запись в первый день. Поэтому "
        "отдельно даём строго later-day долю: по дневным данным нельзя восстановить "
        "точный порядок сделок внутри дня. Реконструкции меняют определение того, "
        "какая покупка считается первой.\n"
    )
    text += "\n## Концентрация популярности\n\n"
    text += md(["Набор", "Доля Buy top-10", "Доля Buy top-12", "Активов для ≥50% Buy"], [
        (label, pct(pop.head(10).sum(), pop.sum()), pct(pop.head(12).sum(), pop.sum()),
         int(np.searchsorted(pop.cumsum().to_numpy(), pop.sum() / 2) + 1))
        for label, pop in [("Все Buy", stats["pop"]), ("Только ID ≥ 0", stats["pop_real"])]
    ])
    top = stats["pop"].head(10)
    text += "\n" + md(["ISIN", "Название текущего metadata (не as-of)", "Buy", "Доля всех Buy"], [
        (isin, meta.set_index("ISIN").assetName.get(isin, "нет metadata"), count, pct(count, len(buys)))
        for isin, count in top.items()
    ])
    tx_pop = tx.ISIN.value_counts()
    text += f"\nДля сравнения с фразой статьи про все взаимодействия: top-12 Buy+Sell = {pct(tx_pop.head(12).sum(), len(tx))}.\n"
    text += "\n## Состав по типам активов\n\n"
    category = meta.set_index("ISIN").assetCategory
    text += md(["Тип", "Unique metadata assets", "Buy rows", "Доля Buy", "Buy с ID ≥ 0"], [
        (kind, int(category.eq(kind).sum()),
         int(buys.ISIN.map(category).fillna("NO_METADATA").eq(kind).sum()),
         pct(buys.ISIN.map(category).fillna("NO_METADATA").eq(kind).sum(), len(buys)),
         int(real_buys.ISIN.map(category).fillna("NO_METADATA").eq(kind).sum()))
        for kind in sorted(set(category) | {"NO_METADATA"})
    ])
    text += "\n## Наблюдаемые, оценённые и отсутствующие профили\n\n"
    profile_rows = []
    for label, subset in [
        ("Все snapshots", profiles),
        ("Последний snapshot на клиента (descriptive, не as-of)", stats["latest_profiles"]),
        ("Последний snapshot ≤ cut (с временными оговорками)", stats["asof_profiles"]),
    ]:
        for col in ["riskLevel", "investmentCapacity"]:
            predicted = subset[col].str.startswith("Predicted_")
            missing = subset[col].isin(MISSING)
            profile_rows.append((
                label, col, len(subset), int(predicted.sum()), pct(predicted.sum(), len(subset)),
                int(missing.sum()), pct(missing.sum(), len(subset)),
                int((~predicted & ~missing).sum()),
            ))
    text += md(["Набор", "Поле", "Строк", "Predicted", "Доля", "Missing", "Доля missing", "Observed labels"], profile_rows)
    text += (
        "\nObserved labels здесь означает отсутствие префикса Predicted_ и маркера "
        "missing, а не гарантию времени доступности. Проверки backdating находятся "
        "в leakage.md; брать полный последний snapshot в модель нельзя.\n"
    )
    text += "\n## Насколько беден текст активов\n\n"
    text_cols = ["assetName", "assetShortName", "assetSubCategory", "sector", "industry"]
    text += md(["Поле (одна строка на ISIN)", "Пусто", "Unique known", "Медиана символов непустых", "Медиана слов непустых"], [
        (col, pct(meta[col].eq("").sum(), len(meta)), meta.loc[meta[col].ne(""), col].nunique(),
         meta.loc[meta[col].ne(""), col].str.len().median(),
         meta.loc[meta[col].ne(""), col].str.split().str.len().median())
        for col in text_cols
    ])
    text += "\n" + md(["Дополнительная проверка", "Значение"], [
        ("Активов без sector И industry", int((meta.sector.eq("") & meta.industry.eq("")).sum())),
        ("Активов без длинного и короткого названия", int((meta.assetName.eq("") & meta.assetShortName.eq("")).sum())),
        ("Колонок свободного описания/новостей в asset_information", sum(
            col.lower() in {"description", "overview", "news", "text"} for col in meta.columns
        )),
    ])
    text += """
Имена/тикеры и короткие категории — не богатые инвестиционные описания; тексты
markets описывают биржу, а не конкретный актив. Нет отзывов, новостей и ответов
анкет. LLM не должен выдумывать их. Известные названия могут активировать внешние
знания модели и контаминацию: информация метода и baseline должна совпадать.

## Что это означает для выбора трека (не окончательное решение)

- **Encoder:** короткие истории дают мотив проверить представления, но текст
  беден и часто сводится к тем же категориям, что доступны структурному baseline.
  Нужен same-information контроль и отдельная фиксация sparse-history группы.
- **Ranker:** концентрация популярности требует сильного non-LLM ориентира;
  повторные покупки резко меняют постановку. На фиксированных кандидатах можно
  изолировать перестановку, но запас реранкинга пока не измерен: ни Recall@K,
  ни oracle nDCG на этом этапе не считались.
- **Hybrid:** добавление обеих LLM-частей сейчас усложнит атрибуцию вклада.
  Рассматривать только при понятном выигрыше каждой части и разделяющих абляциях.
- Правило повторов преподаватель подтвердил: они разрешены и релевантны.
  A остаётся только new-only диагностикой; для HW2 используются repeat-allowed
  defaults официального starter и все четыре его baseline (см. starter_baselines.md).
  Чтение основной статьи датасета не заменяет будущий обзор релевантных LLM-работ.

## Предварительное оценочное окно

Подробные определения и ограничения — [split.md](split.md).

"""
    text += split_table(split_rows)
    text += """
## Графики

![Активность клиентов](figures/assets_per_customer.png)

![Популярность активов](figures/popularity_concentration.png)

![Покупки по месяцам](figures/buys_per_month.png)
"""
    write(root / "reports" / "eda.md", text)


def plot_figures(root, stats):
    target = root / "reports" / "figures"
    target.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for values, label in [
        (stats["activity"], "All buys (including reconstructed)"),
        (stats["activity_real"], "Nonnegative IDs only (all customers)"),
    ]:
        distribution = values.value_counts().sort_index()
        ax.step(distribution.index, distribution.values, where="mid", label=label)
    ax.set(xlabel="Distinct assets bought per customer", ylabel="Customers",
           title="Customer activity", yscale="log", xscale="symlog")
    ax.set_xlim(left=0)
    ax.set_xticks([0, 1, 3, 10, 30, 100], labels=["0", "1", "3", "10", "30", "100"])
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(target / "assets_per_customer.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pop, label in [(stats["pop"], "All buys"), (stats["pop_real"], "Nonnegative IDs only")]:
        ax.plot(np.arange(1, len(pop) + 1), pop.cumsum() / pop.sum(), label=label)
    ax.axvline(10, color="grey", linestyle="--", label="Top 10")
    ax.axhline(.5, color="grey", linestyle=":")
    ax.set(xlabel="Assets sorted by purchase count", ylabel="Cumulative share of Buy rows",
           title="Popularity concentration")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(target / "popularity_concentration.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    buys = stats["buys"]
    months = pd.period_range(buys.timestamp.min(), buys.timestamp.max(), freq="M")
    for subset, label in [
        (buys[buys.transactionID.ge(0)], "Nonnegative IDs"),
        (buys[buys.transactionID.lt(0)], "Reconstructed marker (negative ID)"),
    ]:
        monthly = subset.groupby(subset.timestamp.dt.to_period("M")).size().reindex(months, fill_value=0)
        ax.plot(monthly.index.to_timestamp(), monthly, label=label)
    ax.set(xlabel="Month", ylabel="Buy rows", title="Purchases over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(target / "buys_per_month.png", dpi=150)
    plt.close(fig)
    for file in target.glob("*.png"):
        assert file.stat().st_size > 0


def example_report(root, archive_hash, raw, tables, stats, details):
    # Select a real Buy relevant in A, from an evaluated customer, on a dated row.
    candidates = details["A"]["labels"]
    candidates = candidates[candidates.customerID.isin(details["A"]["relevance"])]
    chosen = candidates.sort_values(
        ["timestamp", "transactionID", "ISIN"], kind="stable"
    ).iloc[0]
    idx = int(chosen.name)
    verbatim = raw["transactions"].loc[idx]
    assert int(verbatim.transactionID) >= 0
    assert chosen.timestamp > stats["cut"]
    assert chosen.timestamp <= stats["end"]
    assert chosen.ISIN in stats["eligible"]
    assert verbatim.transactionType == "Buy"
    values = [("[MASKED_CUSTOMER]" if col == "customerID" else verbatim[col])
              for col in verbatim.index]
    text = "# Одна реальная запись из окна меток\n\n" + banner(archive_hash)
    text += (
        "Запись взята из исходного CSV, не из агрегата и не из сгенерированного "
        "примера. Только customerID заменён общей маской; ID транзакции и остальные "
        "значения сохранены дословно (включая десятичную запись). "
        "Реальная личность клиента не устанавливалась. Неотрицательный ID означает "
        "отсутствие выявленного отрицательного маркера, но не доказывает отсутствие "
        "всех преобразований авторов.\n\n```text\n"
        f"source: FAR-Trans public release ({RELEASE}), FAR-Trans/transactions.csv\n"
        f"archive SHA-256: {archive_hash}\n"
        f"record id: transactionID={verbatim.transactionID}; CSV data-record={idx + 1} (header excluded)\n"
        f"fields: {', '.join(verbatim.index)}\n"
        f"values: {', '.join(values)}\n"
        f"window: {date(stats['cut'])} < timestamp <= {date(stats['end'])}; provisional labels A and B\n"
        "```\n\n"
    )
    text += (
        "transactionID не уникален глобально, поэтому дополнительно указан порядковый "
        "CSV-record. Эта покупка допустима в A (новая пара) и B и принадлежит "
        "пользователю с Buy-историей в eligible universe до среза. "
        "Поля события находятся после cut: использовать их как признаки при "
        "прогнозе запрещено.\n\n"
    )
    rows = []
    for col, val in zip(verbatim.index, values):
        if verbatim[col] in MISSING:
            status = "missing"
        elif col == "totalValue":
            status = "predicted/estimated (recomputed)"
        elif col == "units":
            status = "observed/adjusted; split provenance unknown"
        elif col == "timestamp":
            status = "observed in release; original date may be shifted"
        else:
            status = "observed in release; customerID masked" if col == "customerID" else "observed in release"
        rows.append((
            col, val, status, verbatim.timestamp,
            "после cut; только метка/аудит, не вход модели",
        ))
    text += md(["Поле", "Verbatim value / маска", "Статус", "Дата события/доступности", "Проверка утечки"], rows)
    text += """
Для totalValue дата — дата цены закрытия, по которой авторы оценили сумму;
для units это дата опубликованной скорректированной сделки. Точная дата
публикации исправлений и оригинальная intraday дата отсутствуют.
Слово observed здесь не превращает пересчитанную сумму в цену исполнения.
В §3 PDF этот пример иллюстрирует будущую целевую запись, а не доступный
в момент прогноза пользовательский профиль.
"""
    # Ensure no raw customer identifier accidentally escapes the renderer.
    assert verbatim.customerID not in text
    write(root / "reports" / "data_example.md", text)
