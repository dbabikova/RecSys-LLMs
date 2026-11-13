"""Independent raw-CSV verifier. Standard library only; imports no project modules.

Run after make_ranker_example.py:
    python -B scripts/verify_example.py
No starter code, saved starter predictions, or project scoring helpers are imported.
Expected evidence is comparison data only, never executable code.
"""

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "FAR-Trans"
PRIVATE = ROOT / "data" / "ranker_example"
REPORT = ROOT / "reports" / "verify_example.md"
CHECKS = []


def check(name, actual, expected):
    if actual != expected:
        raise ValueError(f"Independent check FAILED: {name}; private values are not printed")
    CHECKS.append(name)


def close(name, actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Independent numeric check FAILED: {name}")
    CHECKS.append(name)


def numbered(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{"record_number": n, "row": row} for n, row in enumerate(csv.DictReader(stream), 1)]


def latest_snapshots(records, identifier, cutoff):
    groups = defaultdict(list)
    for record in records:
        if date.fromisoformat(record["row"]["timestamp"][:10]) < cutoff:
            groups[record["row"][identifier]].append(record)
    # max keeps the first equal-timestamp record: no last-file-row tie break.
    return {key: max(group, key=lambda r: r["row"]["timestamp"]) for key, group in groups.items()}


def content_vectors(assets):
    result = {}
    for item, record in assets.items():
        row = record["row"]
        terms = set()
        for field in ("assetCategory", "assetSubCategory", "sector", "industry", "marketID"):
            value = row.get(field, "").strip()
            if value not in ("", "Not_Available", "nan"):
                terms.add(field + "=" + value.casefold())
        result[item] = {term: 1 / math.sqrt(len(terms)) for term in sorted(terms)}
    return result


def cbf_scores(counts, vectors):
    # Independently implement the normalized weighted profile and cosine dot product.
    profile = defaultdict(float)
    for item, weight in counts.items():
        for term, value in vectors[item].items():
            profile[term] += weight * value
    length = math.sqrt(sum(value * value for value in profile.values()))
    return {
        item: sum(profile[term] * value for term, value in vector.items()) / length if length else 0.0
        for item, vector in vectors.items()
    }


def sorted_items(scores):
    return sorted(scores, key=lambda item: (-scores[item], item))


def ndcg(ranking, labels):
    if not labels:
        return 0.0
    numerator = sum(
        1 / math.log2(rank + 1)
        for rank, item in enumerate(ranking[:10], 1) if item in labels
    )
    denominator = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(labels)) + 1))
    return numerator / denominator


def md(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in rows],
    ]) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    # Invalidate a previous success report before any checks, without reporting
    # success if this new verification fails partway through.
    REPORT.write_text("# Независимая проверка\n\n**RUNNING / not yet verified.**\n", encoding="utf-8")
    (PRIVATE / "verification_result.json").write_text(
        '{"status": "RUNNING", "passed": 0, "failed": null}\n', encoding="utf-8"
    )
    expected = json.loads((PRIVATE / "expected.json").read_text(encoding="utf-8"))
    cutoff_string = expected["protocol"]["cutoff"]
    cutoff = date.fromisoformat(cutoff_string)
    end = cutoff + timedelta(days=30)
    check("window end from cutoff + 30 days", end.isoformat(), expected["protocol"]["end_exclusive"])
    for name, reference in expected["source_sha256"].items():
        check("SHA-256 " + name, hashlib.sha256((DATA / name).read_bytes()).hexdigest(), reference)
    asset_rows = numbered(DATA / "asset_information.csv")
    profile_rows = numbered(DATA / "customer_information.csv")
    tx_rows = numbered(DATA / "transactions.csv")
    asset_asof = latest_snapshots(asset_rows, "ISIN", cutoff)
    profiles = latest_snapshots(profile_rows, "customerID", cutoff)
    priced = set()
    with (DATA / "close_prices.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if date.fromisoformat(row["timestamp"][:10]) < cutoff and float(row["closePrice"]) > 0:
                priced.add(row["ISIN"])
    assets = {key: value for key, value in asset_asof.items() if key in priced}
    check("eligible universe", sorted(assets), expected["eligible_ids"])
    history = defaultdict(Counter)
    truths = defaultdict(set)
    history_records = defaultdict(list)
    label_records = defaultdict(list)
    omitted = defaultdict(Counter)
    last_buy = defaultdict(dict)
    retained_full_rows = set()
    for record in tx_rows:
        row = record["row"]
        full = tuple(row[field] for field in row)
        duplicate = full in retained_full_rows
        retained_full_rows.add(full)
        user, item = row["customerID"], row["ISIN"]
        day = date.fromisoformat(row["timestamp"][:10])
        if duplicate:
            reason = "exact_duplicate"
        elif int(row["transactionID"]) < 0:
            reason = "negative_id"
        elif item not in assets:
            reason = "ineligible"
        elif row["transactionType"] != "Buy":
            reason = "not_Buy"
        else:
            reason = None
        if day < cutoff and reason:
            omitted[user][reason] += 1
        if reason:
            continue
        if day < cutoff:
            history[user][item] += 1
            history_records[user].append(record)
            last_buy[user][item] = max(last_buy[user].get(item, ""), row["timestamp"][:10])
        elif day < end:
            truths[user].add(item)
            label_records[user].append(record)
    demo = sorted(
        (user for user in profiles if history.get(user)),
        key=lambda user: (hashlib.sha256(("42:" + user).encode()).hexdigest(), user),
    )[:100]
    check("deterministic demo cohort", demo, expected["protocol"]["users"])
    popularity = Counter()
    for counts in history.values():
        popularity.update(counts.keys())  # Distinct buyers, not counts of Buy events.
    pop_order = sorted(assets, key=lambda item: (-popularity[item], item))
    vectors = content_vectors(assets)
    scores_by_user = {user: cbf_scores(history[user], vectors) for user in demo}
    ranking_by_user = {
        user: sorted_items(scores) if any(scores.values()) else pop_order
        for user, scores in scores_by_user.items()
    }
    choices = [
        (user, item, ranking_by_user[user].index(item) + 1)
        for user in demo for item in sorted(truths[user])
    ]
    preferred = [choice for choice in choices if 11 <= choice[2] <= 20]
    inside = [choice for choice in choices if choice[2] <= 20]
    if preferred:
        chosen = min(preferred, key=lambda x: (x[2], demo.index(x[0]), x[1]))
    elif inside:
        chosen = min(inside, key=lambda x: (-x[2], demo.index(x[0]), x[1]))
    else:
        chosen = min(choices, key=lambda x: (abs(x[2] - 20), demo.index(x[0]), x[1]))
    user, item, position = chosen
    check("preferred rank-11–20 target pairs", len(preferred), expected["selection"]["preferred_target_pairs"])
    check("preferred rank-11–20 users", len({x[0] for x in preferred}), expected["selection"]["preferred_users"])
    check("target pairs inside top-20", len(inside), expected["selection"]["target_pairs_in_top20"])
    check("independently selected user and asset", (user, item), (expected["user"], expected["selected_isin"]))
    check("selected user's demo ordinal", demo.index(user) + 1, expected["selection"]["selected_demo_ordinal"])
    by_source = lambda r: (r["row"]["timestamp"], int(r["row"]["transactionID"]), r["record_number"])
    available = [r for r in label_records[user] if r["row"]["ISIN"] == item]
    source_record = min(available, key=by_source)
    check("verbatim raw record and CSV record ordinal", source_record, expected["raw_record"])
    same_id = [r for r in tx_rows if r["row"]["transactionID"] == source_record["row"]["transactionID"]]
    check("ALL source rows sharing transactionID (not blindly deduplicated)", same_id, expected["same_transaction_id_rows"])
    past_records = sorted(history_records[user], key=by_source)
    check("retained dated history raw rows", past_records, expected["history_raw"])
    check("history Buy counts", dict(history[user]), expected["history_counts"])
    check("omitted pre-cutoff rows by reason", dict(omitted[user]), expected["history_omitted"])
    check("all eligible future Buy raw rows", sorted(label_records[user], key=by_source), expected["label_raw"])
    check("full distinct future truth", sorted(truths[user]), expected["truth"])
    check("all profile snapshots", [r for r in profile_rows if r["row"]["customerID"] == user], expected["profile_all_rows"])
    check("as-of profile + original record ordinal", profiles[user], expected["profile_asof"])
    check("all metadata snapshots for asset", [r for r in asset_rows if r["row"]["ISIN"] == item], expected["metadata_all_rows"])
    check("as-of metadata + first-on-tie ordinal", assets[item], expected["metadata_asof"])
    # Exact top-20 and target positions, numeric tolerance only for arithmetic.
    cbf = ranking_by_user[user]
    check("CBF top-20 ordered ISINs", cbf[:20], expected["cbf_order"][:20])
    check("selected asset CBF position", position, expected["positions"]["cbf"])
    check("all top-20 raw metadata", [assets[i]["row"] for i in cbf[:20]],
          [r["source_metadata"] for r in expected["candidate_assets"]])
    check("all top-20 categorical tokens", [sorted(vectors[i]) for i in cbf[:20]],
          [r["tokens"] for r in expected["candidate_assets"]])
    check("top-20 CBF scores within tolerance", all(
        math.isclose(scores_by_user[user][r["id"]], r["cbf_score"], abs_tol=1e-12, rel_tol=1e-12)
        for r in expected["candidate_assets"]
    ), True)
    latest_prefix = sorted(last_buy[user], key=lambda i: (
        -date.fromisoformat(last_buy[user][i]).toordinal(), i
    ))
    latest = latest_prefix + [i for i in pop_order if i not in set(latest_prefix)]
    check("full Latest buy ranking including popularity fill", latest, expected["latest_buy_order"])
    check("metadata for displayed CBF/Latest buy lists", {
        i: assets[i]["row"] for i in set(cbf[:20] + latest[:10])
    }, expected["ranking_assets"])
    check("selected asset Latest buy position", latest.index(item) + 1, expected["positions"]["latest_buy"])
    check("repeat/new classification from pre-cutoff Buy", item in history[user], expected["is_repeat"])
    close("CBF nDCG@10", ndcg(cbf, truths[user]), expected["metrics"]["cbf"])
    close("Latest buy nDCG@10", ndcg(latest, truths[user]), expected["metrics"]["latest_buy"])
    check("all retained history dates strictly before cut", all(
        date.fromisoformat(r["row"]["timestamp"][:10]) < cutoff for r in past_records
    ), True)
    check("record is in label window, Buy, nonnegative ID, eligible", (
        cutoff <= date.fromisoformat(source_record["row"]["timestamp"][:10]) < end
        and source_record["row"]["transactionType"] == "Buy"
        and int(source_record["row"]["transactionID"]) >= 0 and item in assets
    ), True)
    common = json.loads((PRIVATE / "planned_common_input.json").read_text(encoding="utf-8"))
    check("planned input JSON matches stored comparison artifact", common, expected["planned_common_input"])
    check("planned candidate payload matches source-verified candidates",
          common["candidate_assets"], expected["candidate_assets"])
    derived_history = []
    for history_item, count in sorted(history[user].items(), key=lambda pair: (-pair[1], pair[0])):
        a = assets[history_item]["row"]
        derived_history.append({
            "id": history_item, "name": a["assetName"], "attributes": sorted(vectors[history_item]),
            "metadata_timestamp": a["timestamp"], "buy_count": count,
        })
    check("planned input history independently reconstructed", common["history"], derived_history)
    check("no Buy timestamps silently added to stock input", all(
        set(h) == {"id", "name", "attributes", "metadata_timestamp", "buy_count"} for h in common["history"]
    ), True)
    profile = profiles[user]["row"]
    questionnaire_late = profile["lastQuestionnaireDate"][:10] >= cutoff_string
    check("late questionnaire input masking for BOTH methods", all(
        common["profile"][field]["value"] == (
            "[WITHHELD_LATE_QUESTIONNAIRE]"
            if questionnaire_late and field in ("riskLevel", "investmentCapacity") else profile[field]
        ) for field in ("customerType", "riskLevel", "investmentCapacity")
    ), True)
    statuses = {}
    for field in ("customerType", "riskLevel", "investmentCapacity"):
        value = profile[field]
        statuses[field] = "missing" if value in ("", "Not_Available") else (
            "estimated" if value.startswith("Predicted_") else "observed"
        )
        if questionnaire_late and field in ("riskLevel", "investmentCapacity"):
            statuses[field] = "missing"
    check("planned profile observed/estimated/missing markers", {
        field: common["profile"][field]["status"] for field in statuses
    }, statuses)
    check("planned profile timestamp and placeholder flag", (
        common["profile"]["timestamp"], common["profile"]["timestamp_is_placeholder"]
    ), (profile["timestamp"], profile["timestamp"][:10] == "2000-01-01"))
    check("planned cutoff and nonidentifying user alias",
          (common["cutoff"], common["user_alias"]), (cutoff_string, "EXAMPLE_USER"))
    check("no truth/target marker in planned input", not any(
        word in json.dumps(common) for word in ('"truth"', '"selected_isin"', '"ndcg"', '"is_repeat"')
    ), True)
    values = ["[MASKED]" if key == "customerID" else value for key, value in source_record["row"].items()]
    record = source_record["row"]
    count = len(truths[user])
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, count) + 1))
    rows_metric = []
    for name, ranking in (("CBF", cbf), ("Latest buy", latest)):
        ranks = [rank for rank, value in enumerate(ranking[:10], 1) if value in truths[user]]
        dcg = sum(1 / math.log2(rank + 1) for rank in ranks)
        rows_metric.append((name, ", ".join(map(str, ranks)) or "нет", f"{dcg:.12f}",
                            f"{idcg:.12f}", f"{dcg/idcg:.12f}"))
    text = f"""# Независимая проверка примера и инструкция ручной проверки

> Сгенерировано отдельным `scripts/verify_example.py`, который не импортирует
> starter или наши модули и сам читает исходные CSV. Модели не запускались.
> Expected JSON — только данные для сравнения, не исполняемый источник алгоритма.

**PASS: {len(CHECKS)} / {len(CHECKS)} независимых проверок совпали.**
Ошибки приводят к ненулевому exit code; прежний success report сначала
сбрасывается, чтобы не оставить ложный PASS после неудачного повторного запуска.
Для арифметики используется tolerance; упорядоченный CBF20 и позиции проверены точно.

"""
    text += md(["№", "Совпавшая проверка"], enumerate(CHECKS, 1))
    text += f"""
## Короткая инструкция для Дарьи — пока НЕ выполнена вручную

1. Открой локальный `data/FAR-Trans/transactions.csv` и найди **transactionID
   `{record['transactionID']}`**, одновременно проверь ISIN `{record['ISIN']}`,
   дату `{record['timestamp']}`, тип `{record['transactionType']}`,
   units `{record['units']}`, totalValue `{record['totalValue']}`.
   Нужный CSV data-record: **{source_record['record_number']}** (без заголовка).
   Если ID встречается несколько раз — строк с таким ID **{len(same_id)}**;
   не удаляй разные customerID как дубликат. Номер record выбирает конкретную строку.
2. Скопируй customerID из найденной строки **только локально**. Не вставляй его
   в отчёт, чат или Git. По этому значению отфильтруй `customer_information.csv`.
   Нужный as-of CSV-record — **{profiles[user]['record_number']}**,
   timestamp `{profile['timestamp']}`, lastQuestionnaireDate
   `{profile['lastQuestionnaireDate']}`, customerType `{profile['customerType']}`,
   riskLevel `{profile['riskLevel']}`, investmentCapacity `{profile['investmentCapacity']}`.
   Выбирается максимальный timestamp строго до `{cutoff_string}`, не последний
   snapshot всего файла; при равенстве — первая строка.
3. Для этого же клиента отфильтруй retained Buy до `{cutoff_string}`:
   неотрицательные ID, exact full duplicates удалены, ISIN eligible.
   Должно быть **{len(past_records)}** строк / **{len(history[user])}** разных активов.
   Сравни даты с raw history в `ranker_example.md`. Эти даты — только audit:
   в stock history context их нет.
4. Будущие метки: Buy в `[{cutoff_string}, {end.isoformat()})` по тем же cleaning/
   eligibility правилам, повторы разрешены, повторные строки одного ISIN
   не увеличивают gain. Должен получиться T = `{json.dumps(sorted(truths[user]))}`,
   **{count}** distinct-меток. Не добавляй target в candidates вручную.
5. Сверь Top-10 списки и CBF20 с `ranker_example.md`: позиция выбранного актива
   **CBF {position}**, **Latest buy {latest.index(item)+1}**. Для CBF20 действуют
   cosine по category/subcategory/sector/industry/marketID, веса = buy_count,
   tie-break ISIN. Это не роль названия или цены исполнения.
6. Посчитай независимо от готового nDCG столбца:

   `DCG@10 = Σ[r=1..10] 1[item_r ∈ T] / log2(r+1)`

   `IDCG@10 = Σ[r=1..min(10, |T|)] 1 / log2(r+1)`

   `nDCG@10 = DCG@10 / IDCG@10`.

"""
    text += md(["Метод", "Relevant ranks", "DCG@10", "IDCG@10", "nDCG@10"], rows_metric)
    if count == 1:
        cbf_term = f"1/log2({position + 1})" if position <= 10 else "0 (вне Top-10)"
        latest_position = latest.index(item) + 1
        latest_term = f"1/log2({latest_position + 1})" if latest_position <= 10 else "0 (вне Top-10)"
        text += (
            f"\nВ этом примере IDCG = {idcg:.12f}, поскольку distinct-метка одна. "
            f"Поэтому CBF: `{cbf_term} = {ndcg(cbf, truths[user]):.12f}`, "
            f"Latest buy: `{latest_term} = {ndcg(latest, truths[user]):.12f}`.\n"
        )
    text += """
Учитывай все T, не только найденные кандидаты. При пустом T starter даёт ноль,
но у этого иллюстративного пользователя T непусто. Результат одного пользователя
не заменяет primary mean по всем выбранным evaluation users.

После собственной проверки отдельно запиши, что именно ты проверила и совпало ли.
**Сейчас выполнена только автоматическая проверка AI-ассистентом; ручная проверка
Дарьи не выполнена и не заявляется выполненной.**

## Запись для сверки (маска — единственное изменение значения)

```text
"""
    text += (
        "source: FAR-Trans public release, transactions.csv\n"
        f"record id: transactionID={record['transactionID']}; CSV data-record={source_record['record_number']}\n"
        f"fields: {', '.join(record)}\n"
        f"values: {', '.join(values)}\n"
        f"window: {cutoff_string} <= timestamp < {end.isoformat()}; repeat allowed\n"
        "```\n"
    )
    all_customer_ids = {r["row"]["customerID"] for r in tx_rows}
    check("public reports contain no unmasked source customer IDs", not any(
        identifier in content
        for identifier in all_customer_ids
        for content in [text,
                        (ROOT / "reports/ranker_example.md").read_text(encoding="utf-8"),
                        (ROOT / "reports/research_question.md").read_text(encoding="utf-8")]
    ), True)
    # Include the last privacy check in the counted checklist too.
    old_total = len(CHECKS) - 1
    text = text.replace(f"PASS: {old_total} / {old_total}", f"PASS: {len(CHECKS)} / {len(CHECKS)}")
    anchor = "\n## Короткая инструкция"
    text = text.replace(anchor, f"| {len(CHECKS)} | {CHECKS[-1]} |\n" + anchor, 1)
    REPORT.write_text(text.rstrip() + "\n", encoding="utf-8")
    (PRIVATE / "verification_result.json").write_text(json.dumps({
        "passed": len(CHECKS), "failed": 0, "checks": CHECKS,
        "record_number": source_record["record_number"],
        "positions": {"cbf": position, "latest_buy": latest.index(item) + 1},
        "metrics": {"cbf": ndcg(cbf, truths[user]), "latest_buy": ndcg(latest, truths[user])},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": len(CHECKS), "failed": 0,
        "transactionID": record["transactionID"], "positions": {
            "cbf": position, "latest_buy": latest.index(item) + 1,
        }, "metrics": {"cbf": ndcg(cbf, truths[user]), "latest_buy": ndcg(latest, truths[user])},
        "manual_verification": "not performed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not echo dictionary values or financial customer identifiers on failure.
        message = str(error) if str(error).startswith("Independent") else type(error).__name__
        REPORT.write_text(
            "# Независимая проверка\n\n**FAIL — совпадение не подтверждено.**\n\n"
            + f"До ошибки совпало проверок: {len(CHECKS)}.\n\n`{message}`\n",
            encoding="utf-8",
        )
        (PRIVATE / "verification_result.json").write_text(
            json.dumps({"status": "FAIL", "passed": len(CHECKS), "failed": 1, "error": message}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(message, file=sys.stderr)
        sys.exit(1)
