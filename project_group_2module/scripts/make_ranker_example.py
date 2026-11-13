"""Build a deterministic raw-record example with the official starter, no models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter

from starter_audit import DATA, RAW_OUTPUT, REPORTS, ROOT, digest, git, import_starter, save_json
from ranker_example_reports import render

sys.dont_write_bytecode = True
PRIVATE = ROOT / "data" / "ranker_example"
RQ = (
    "We test whether zero-shot LLM reranking of a fixed top-20 candidate list improves "
    "nDCG@10 over a classical learned reranker (LightGBM LambdaMART) that sees the same "
    "user and asset information and the same candidates, for the common FAR-Trans "
    "evaluation users."
)


def with_numbers(st, name):
    return [{"record_number": n, "row": row} for n, row in enumerate(st.rows(DATA / name), 1)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-repo", default=str(ROOT.parents[1] / "course-starter"))
    args = parser.parse_args()
    from pathlib import Path
    repo = Path(args.starter_repo).resolve()
    reference = json.loads((REPORTS / "starter_summary.json").read_text(encoding="utf-8"))
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Starter checkout is not clean")
    assert git(repo, "rev-parse", "HEAD") == reference["starter_commit"]
    assert digest(repo / "project/far-trans/starter.py") == reference["starter_source_sha256"]["project/far-trans/starter.py"]
    for name, expected in reference["source_sha256"].items():
        assert digest(DATA / name) == expected
    st = import_starter(repo / "project/far-trans/starter.py")
    protocol = json.loads((RAW_OUTPUT / "demo/protocol.json").read_text(encoding="utf-8"))
    assert protocol["protocol_hash"] == reference["runs"]["demo"]["protocol_hash"]
    assert protocol["candidate_count"] == 50 and protocol["cohort_selection"] == "warm"
    cutoff, end = protocol["cutoff"], protocol["end_exclusive"]
    assets, profiles, history, truth, loaded_end, excluded = st.load_data(DATA, cutoff, 30)
    assert end == loaded_end and protocol["exclude_seen"] is False
    users = st.select_users(profiles, history, "warm", 100, None)
    assert users == protocol["users"]
    features = {item: st.tokens(row) for item, row in assets.items()}
    popularity = st.popularity_scores(history, assets)
    contexts, predictions = st.contexts_for(
        users, assets, profiles, history, features, popularity, cutoff, 50, False
    )
    events = st.scan_recent_events(DATA, cutoff, assets)
    recent = st.recency_rankings(contexts, popularity, events, k=len(assets))
    orders = {
        u: st.order(st.content_scores(history[u], features))
        for u in users
    }
    # Selection depends on labels only to illustrate a test record, never to train,
    # choose a pool, prune evaluated users, or create an actual model prediction.
    targets = [
        (u, item, orders[u].index(item) + 1)
        for u in users for item in sorted(truth.get(u, set()))
    ]
    preferred = [(u, item, pos) for u, item, pos in targets if 11 <= pos <= 20]
    in20 = [(u, item, pos) for u, item, pos in targets if pos <= 20]
    if preferred:
        chosen_pair = min(preferred, key=lambda x: (x[2], users.index(x[0]), x[1]))
        rule = "prefer ranks 11–20; lowest rank, then demo ordinal, then ISIN"
    elif in20:
        chosen_pair = min(in20, key=lambda x: (-x[2], users.index(x[0]), x[1]))
        rule = "no target at ranks 11–20; highest rank within top-20, then demo ordinal, then ISIN"
    else:
        chosen_pair = min(targets, key=lambda x: (abs(x[2] - 20), users.index(x[0]), x[1]))
        rule = "no target in top-20; nearest position to 20, then demo ordinal, then ISIN"
    user, item, cbf_position = chosen_pair
    transactions = with_numbers(st, "transactions.csv")
    available_rows = [
        record for record in transactions
        if record["row"]["customerID"] == user and record["row"]["ISIN"] == item
        and record["row"]["transactionType"] == "Buy"
        and int(record["row"]["transactionID"]) >= 0
        and cutoff <= record["row"]["timestamp"][:10] < end
    ]
    selected = min(available_rows, key=lambda r: (
        r["row"]["timestamp"], int(r["row"]["transactionID"]), r["record_number"]
    ))
    same_id = [r for r in transactions if r["row"]["transactionID"] == selected["row"]["transactionID"]]
    retained, label_rows, duplicate_keys, omitted = [], [], set(), Counter()
    for record in transactions:
        row = record["row"]
        key = tuple(sorted(row.items()))
        is_duplicate = key in duplicate_keys
        duplicate_keys.add(key)
        if row["customerID"] != user:
            continue
        if row["timestamp"][:10] >= cutoff:
            if (not is_duplicate and int(row["transactionID"]) >= 0
                    and row["ISIN"] in assets and row["transactionType"] == "Buy"
                    and row["timestamp"][:10] < end):
                label_rows.append(record)
            continue
        if is_duplicate:
            omitted["exact_duplicate"] += 1
        elif int(row["transactionID"]) < 0:
            omitted["negative_id"] += 1
        elif row["ISIN"] not in assets:
            omitted["ineligible"] += 1
        elif row["transactionType"] != "Buy":
            omitted["not_Buy"] += 1
        else:
            retained.append(record)
    retained.sort(key=lambda r: (
        r["row"]["timestamp"], int(r["row"]["transactionID"]), r["record_number"]
    ))
    history_count = Counter(r["row"]["ISIN"] for r in retained)
    assert history_count == history[user]
    label_rows.sort(key=lambda r: (
        r["row"]["timestamp"], int(r["row"]["transactionID"]), r["record_number"]
    ))
    assert {r["row"]["ISIN"] for r in label_rows} == truth[user]
    profile_rows = [r for r in with_numbers(st, "customer_information.csv") if r["row"]["customerID"] == user]
    profile_record = next(r for r in profile_rows if r["row"] == profiles[user])
    metadata_records = [r for r in with_numbers(st, "asset_information.csv") if r["row"]["ISIN"] == item]
    chosen_metadata = next(r for r in metadata_records if r["row"] == assets[item])
    latest_order = recent[user]["latest_buy"]
    cbf_order = orders[user]
    assert cbf_order[:10] == predictions[user]["cbf"]
    expected = {
        "starter_commit": reference["starter_commit"],
        "starter_source_sha256": reference["starter_source_sha256"],
        "source_sha256": reference["source_sha256"],
        "protocol": protocol, "user": user, "selected_isin": item,
        "selection": {
            "rule": rule, "preferred_target_pairs": len(preferred),
            "preferred_users": len({u for u, _, _ in preferred}),
            "demo_users": len(users), "demo_nonempty_users": sum(bool(truth.get(u)) for u in users),
            "target_pairs_in_top20": len(in20), "selected_demo_ordinal": users.index(user) + 1,
        },
        "raw_record": selected, "same_transaction_id_rows": same_id,
        "history_raw": retained, "label_raw": label_rows, "history_counts": dict(history[user]),
        "history_omitted": dict(omitted), "profile_all_rows": profile_rows,
        "profile_asof": profile_record,
        "metadata_all_rows": metadata_records, "metadata_asof": chosen_metadata,
        "eligible_ids": sorted(assets),
        "candidate_assets": [
            {"id": isin, "source_metadata": assets[isin], "tokens": features[isin],
             "cbf_score": st.content_scores(history[user], features)[isin],
             "cbf_position": idx + 1}
            for idx, isin in enumerate(cbf_order[:20])
        ],
        "history_context": contexts[user]["history"],
        "cbf_order": cbf_order, "latest_buy_order": latest_order,
        "ranking_assets": {isin: assets[isin] for isin in set(cbf_order[:20] + latest_order[:10])},
        "truth": sorted(truth[user]), "is_repeat": item in history[user],
        "positions": {"cbf": cbf_position, "latest_buy": latest_order.index(item) + 1},
        "metrics": {method: st.ndcg10(ranking, truth[user]) for method, ranking in (
            ("cbf", cbf_order[:10]), ("latest_buy", latest_order[:10]),
        )},
    }
    # Cross-check the source-based example against the actual retained demo run.
    csv_predictions = list(st.rows(RAW_OUTPUT / "demo/predictions.csv"))
    for method, ranking in (("cbf", cbf_order), ("latest_buy", latest_order)):
        saved = sorted(
            (r for r in csv_predictions if r["user_id"] == user and r["method"] == method),
            key=lambda r: int(r["rank"]),
        )
        assert [r["ISIN"] for r in saved] == ranking[:10]
    for row in st.rows(RAW_OUTPUT / "demo/per_user.csv"):
        if row["user_id"] == user and row["method"] in expected["metrics"]:
            assert int(row["relevant_count"]) == len(truth[user])
            assert math.isclose(float(row["ndcg_at_10"]), expected["metrics"][row["method"]], abs_tol=1e-12)
    # Shared *planned* information contract; no dates of past buys are silently added
    # to the stock history. Raw dated rows below are audit evidence only.
    profile_input = st.profile_context(profiles[user])
    questionnaire = profiles[user].get("lastQuestionnaireDate", "")[:10]
    if questionnaire >= cutoff:
        for field in ("riskLevel", "investmentCapacity"):
            profile_input[field] = {"value": "[WITHHELD_LATE_QUESTIONNAIRE]", "status": "missing"}
    common_input = {
        "user_alias": "EXAMPLE_USER", "cutoff": cutoff, "profile": profile_input,
        "history": contexts[user]["history"],
        "candidate_assets": expected["candidate_assets"],
    }
    assert not any("last_buy" in key for h in common_input["history"] for key in h)
    expected["planned_common_input"] = common_input
    PRIVATE.mkdir(parents=True, exist_ok=True)
    save_json(PRIVATE / "expected.json", expected)
    save_json(PRIVATE / "planned_common_input.json", common_input)
    render(ROOT, expected, reference, RQ)
    ids = {r["row"]["customerID"] for r in transactions}
    for name in ("research_question.md", "ranker_example.md"):
        report = (REPORTS / name).read_text(encoding="utf-8")
        if any(uid in report for uid in ids):
            raise ValueError(f"Unmasked customer ID in {name}")
    assert not git(repo, "status", "--porcelain", "--untracked-files=all")
    print(json.dumps({
        "preferred_pairs_11_to_20": len(preferred),
        "selected_record_number": selected["record_number"],
        "transactionID": selected["row"]["transactionID"], "ISIN": item,
        "date": selected["row"]["timestamp"], "positions": expected["positions"],
        "is_repeat": expected["is_repeat"], "metrics": expected["metrics"],
        "history_rows": len(retained), "profile_record_number": profile_record["record_number"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
