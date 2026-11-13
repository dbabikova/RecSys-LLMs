"""Independent stdlib checks of aggregate metrics, source record and privacy."""

import csv
import json
import math
import re
from collections import Counter, defaultdict


def verify(root, summary):
    with (root / "data" / "FAR-Trans" / "transactions.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary["transactions"]
    assert sum(r["transactionType"] == "Buy" for r in rows) == summary["buys"]
    customers = {r["customerID"] for r in rows}
    assert len(customers) == summary["customers"]
    with (root / "data" / "FAR-Trans" / "limit_prices.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        bounds = list(csv.DictReader(stream))
    cut, end = summary["cut"], summary["end"]
    universe = {r["ISIN"] for r in bounds if r["minDate"] <= cut and r["maxDate"] >= end}
    assert len(universe) == summary["eligible_assets"]
    history = defaultdict(Counter)
    last = {}
    seen = defaultdict(set)
    labels = defaultdict(set)
    for row in rows:
        user, item, day = row["customerID"], row["ISIN"], row["timestamp"]
        if row["transactionType"] != "Buy":
            continue
        if day <= cut:
            seen[user].add(item)
            if item in universe:
                history[user][item] += 1
                last[user, item] = max(last.get((user, item), ""), day)
        elif day <= end and int(row["transactionID"]) >= 0 and item in universe:
            labels[user].add(item)
    pop = Counter()
    for counts in history.values():
        pop.update(counts)
    base_order = sorted(universe, key=lambda item: (-pop[item], item))
    for measured in summary["split"]:
        variant = measured["variant"]
        scores = [[], []]
        counts = []
        cold = 0
        for user, targets in labels.items():
            targets = targets - seen[user] if variant == "A" else targets
            if not targets:
                continue
            if not history[user]:
                cold += 1
                continue
            forbidden = seen[user] if variant == "A" else set()
            rank = [item for item in base_order if item not in forbidden]
            own = sorted(history[user], key=lambda item: (
                -history[user][item], -int(last[user, item].replace("-", "")),
                -pop[item], item,
            ))
            own = [item for item in own if item not in forbidden]
            repeat_rank = own + [item for item in rank if item not in set(own)]
            idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(targets))))
            for output, sink in zip((rank, repeat_rank), scores):
                dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(output[:10]) if item in targets)
                sink.append(dcg / idcg)
            counts.append(len(targets))
        assert cold == measured["cold_users"]
        assert len(counts) == measured["users"]
        assert sum(counts) == measured["relevant_pairs"]
        for actual, key in zip(scores, ("ndcg_popularity", "ndcg_repeat_popularity")):
            assert math.isclose(sum(actual) / len(actual), measured[key], abs_tol=1e-12)
    example = (root / "reports" / "data_example.md").read_text(encoding="utf-8")
    record = re.search(r"CSV data-record=(\d+)", example)
    original = rows[int(record[1]) - 1]
    for key, value in original.items():
        if key != "customerID":
            assert value in example, f"Non-verbatim example: {key}"
    assert original["customerID"] not in example
    for path in (root / "reports").rglob("*"):
        if path.suffix not in {".md", ".json"}:
            continue
        content = path.read_text(encoding="utf-8")
        # All source customer IDs are fixed-length token-like pseudonyms.
        tokens = set(re.findall(r"[A-Za-z0-9_-]{20}", content))
        assert not tokens & customers, f"Customer ID escaped into {path.name}"
        # Also check windows in longer uninterrupted strings.
        assert not any(customer in content for customer in customers), path.name
    print("Independent CSV verification: counts, both nDCGs, verbatim record and privacy OK")
