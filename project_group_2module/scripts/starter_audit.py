"""Run the unmodified external course starter and diagnose its candidate ceiling.

From project_group_2module:
    python scripts/starter_audit.py --starter-repo ../../course-starter
Raw per-user outputs remain in ignored data/starter_runs/. No model is called.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Never create __pycache__ in the instructor's checkout, including on import.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "FAR-Trans"
REPORTS = ROOT / "reports"
RAW_OUTPUT = ROOT / "data" / "starter_runs"
METHODS = ["popularity", "cbf", "latest_buy", "latest_buy_or_sell"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_starter(path):
    spec = importlib.util.spec_from_file_location("official_far_trans_starter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(repo, label, flags):
    output = RAW_OUTPUT / label
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-B", str(repo / "project" / "far-trans" / "starter.py"),
        "run", "--data-dir", str(DATA), "--output-dir", str(output), *flags,
    ]
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.perf_counter() - begin
    (output / "console.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    completed.check_returncode()
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert protocol["protocol_hash"] == metrics["protocol_hash"]
    unhashed = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    expected = hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest()
    assert expected == protocol["protocol_hash"]
    assert protocol["exclude_seen"] is False
    assert protocol["cohort_selection"] == "warm"
    assert protocol["cutoff"] == "2021-06-01"
    assert protocol["end_exclusive"] == "2021-07-01"
    assert protocol["candidate_count"] == 50 and protocol["k"] == 10
    assert set(metrics["methods"]) == set(METHODS)
    for name, expected_hash in protocol["source_sha256"].items():
        assert digest(DATA / name) == expected_hash
    result = {
        "label": label, "flags_beyond_io": flags, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "whole_cli_wall_seconds": elapsed, "protocol_hash": protocol["protocol_hash"],
        "metrics": metrics["methods"], "excluded_rows": metrics["excluded_rows"],
        "custom_failures": len(metrics["custom_failures"]),
        "output_sha256": {
            name: digest(output / name) for name in (
                "protocol.json", "metrics.json", "contexts.jsonl", "predictions.csv", "per_user.csv"
            )
        },
    }
    save_json(output / "run_metadata.json", result)
    print(completed.stdout, end="")
    return result, protocol


def read_readme_references(repo):
    """Read quoted cells from the pinned README, never hard-code measured metrics."""
    text = (repo / "project" / "far-trans" / "README.md").read_text(encoding="utf-8")
    sections = {}
    for name, title in [("demo", "### Actual result"), ("all", "### Full-cohort")]:
        part = text.split(title, 1)[1]
        part = part.split("### Full-cohort run", 1)[0] if name == "demo" else part.split("\n## Rules", 1)[0]
        table = {}
        for line in part.splitlines():
            cells = [cell.strip().replace("`", "").replace(",", "") for cell in line.strip().strip("|").split("|")]
            if cells[0] in METHODS and len(cells) == 6:
                table[cells[0]] = dict(zip(
                    ["users", "nonempty_truth_users", "empty_truth_users", "ndcg_at_10", "ndcg_at_10_nonempty"],
                    [int(v) for v in cells[1:4]] + [float(v) for v in cells[4:]],
                ))
        assert set(table) == set(METHODS)
        sections[name] = table
    # The full *warm* cohort is a paragraph, not the all-cohort table above it.
    import re
    warm = text.split("warm-cohort slice", 1)[1].split("## Rules", 1)[0]
    match = re.search(r"([\d,]+) users,\s*([\d,]+)\s+nonempty", warm)
    users, nonempty = (int(v.replace(",", "")) for v in match.groups())
    number = r"\d+(?:\.\d+)?"
    methods = re.findall(
        rf"`(cbf|popularity|latest_buy|latest_buy_or_sell)`\s+({number})\s*/\s*({number})", warm
    )
    sections["full_warm"] = {
        method: {
            "users": users, "nonempty_truth_users": nonempty, "empty_truth_users": users - nonempty,
            "ndcg_at_10": float(all_mean), "ndcg_at_10_nonempty": float(nonempty_mean),
        } for method, all_mean, nonempty_mean in methods
    }
    assert set(sections["full_warm"]) == set(METHODS)
    return sections


def audit_cohort(st, users, contexts, predictions, history, truth, end_history, profiles, result):
    relevant = {u: truth.get(u, set()) & set(contexts[u]["eligible_asset_ids"]) for u in users}
    positive = [u for u in users if relevant[u]]
    assert len(positive) == result["metrics"]["cbf"]["nonempty_truth_users"]
    # Recompute every CLI metric by importing the actual scorer/aggregation functions.
    for method in METHODS:
        records = [
            {"relevant_count": len(relevant[u]), "ndcg_at_10": st.ndcg10(predictions[u][method], relevant[u])}
            for u in users
        ]
        reproduced = st.aggregate(records)
        for key, value in reproduced.items():
            assert math.isclose(value, result["metrics"][method][key], abs_tol=1e-12)
    groups = {
        "all": {u: relevant[u] for u in positive},
        "repeat": {u: relevant[u] & set(history[u]) for u in positive},
        "new": {u: relevant[u] - set(history[u]) for u in positive},
    }
    recall_rows = []
    for group, targets in groups.items():
        group_users = [u for u in positive if targets[u]]
        pair_total = sum(len(targets[u]) for u in group_users)
        for k in (10, 20, 50):
            hits = {
                u: len(targets[u] & {a["id"] for a in contexts[u]["candidate_assets"][:k]})
                for u in group_users
            }
            recall_rows.append({
                "target_group": group, "k": k, "users": len(group_users),
                "target_pairs": pair_total, "covered_pairs": sum(hits.values()),
                "macro_recall": sum(hits[u] / len(targets[u]) for u in group_users) / len(group_users) if group_users else None,
                "micro_recall": sum(hits.values()) / pair_total if pair_total else None,
                "users_any_hit": sum(h > 0 for h in hits.values()),
                "any_hit_share": sum(h > 0 for h in hits.values()) / len(group_users) if group_users else None,
            })
    total_pairs = sum(len(relevant[u]) for u in positive)
    repeat_pairs = sum(len(groups["repeat"][u]) for u in positive)
    future_events = repeat_events = 0
    for u in positive:
        for item in relevant[u]:
            # Same starter loader at the end of the window: difference of cumulative
            # retained Buy counters, under the source loader's eligibility/exclusion rules.
            n = end_history.get(u, {}).get(item, 0) - history.get(u, {}).get(item, 0)
            assert n > 0
            future_events += n
            repeat_events += n if item in history[u] else 0
    overlaps = []
    for group, targets in groups.items():
        latest_hits = sum(len(set(predictions[u]["latest_buy"]) & targets[u]) for u in positive)
        covered = sum(len(
            set(predictions[u]["latest_buy"]) & targets[u]
            & {a["id"] for a in contexts[u]["candidate_assets"]}
        ) for u in positive)
        users_missing = sum(bool(
            (set(predictions[u]["latest_buy"]) & targets[u])
            - {a["id"] for a in contexts[u]["candidate_assets"]}
        ) for u in positive)
        overlaps.append({
            "target_group": group, "latest_buy_hit_pairs": latest_hits,
            "also_in_cbf50": covered, "missing_from_cbf50": latest_hits - covered,
            "covered_share": covered / latest_hits if latest_hits else None,
            "users_with_missing_latest_hit": users_missing,
        })
    oracle_records, cbf_then_oracle = [], []
    for u in users:
        candidates = [a["id"] for a in contexts[u]["candidate_assets"]]
        oracle_order = [i for i in candidates if i in relevant[u]] + [i for i in candidates if i not in relevant[u]]
        assert set(oracle_order) == set(candidates)
        score = st.ndcg10(oracle_order[:10], relevant[u])
        assert score + 1e-12 >= st.ndcg10(predictions[u]["cbf"], relevant[u])
        oracle_records.append({"relevant_count": len(relevant[u]), "ndcg_at_10": score})
        if relevant[u]:
            cbf_then_oracle.append(score > st.ndcg10(predictions[u]["latest_buy"], relevant[u]) + 1e-12)
    one_vector = [
        u for u in users
        if len({tuple(a["attributes"]) for a in contexts[u]["history"]}) == 1
    ]
    # A small source-linked check, not a claim of Darья's personal verification.
    check_user = next(u for u in positive if set(predictions[u]["latest_buy"]) & relevant[u])
    rank = predictions[check_user]["latest_buy"]
    contributions = [
        {"rank": idx + 1, "relevant": item in relevant[check_user],
         "dcg_contribution": 1 / math.log2(idx + 2) if item in relevant[check_user] else 0.0}
        for idx, item in enumerate(rank)
    ]
    manual = sum(r["dcg_contribution"] for r in contributions) / sum(
        1 / math.log2(i + 2) for i in range(min(10, len(relevant[check_user])))
    )
    assert math.isclose(manual, st.ndcg10(rank, relevant[check_user]), abs_tol=1e-12)
    return {
        "nonempty_users": len(positive), "candidate_recall": recall_rows,
        "future_target_pairs": total_pairs, "repeat_target_pairs": repeat_pairs,
        "new_target_pairs": total_pairs - repeat_pairs,
        "repeat_pair_share": repeat_pairs / total_pairs,
        "macro_repeat_pair_share": sum(len(groups["repeat"][u]) / len(relevant[u]) for u in positive) / len(positive),
        "future_buy_rows": future_events, "future_repeat_buy_rows": repeat_events,
        "repeat_buy_row_share": repeat_events / future_events,
        "latest_buy_overlap": overlaps, "cbf50_oracle": st.aggregate(oracle_records),
        "users_oracle_strictly_beats_latest_buy": sum(cbf_then_oracle),
        "single_feature_vector_users": len(one_vector),
        "single_feature_vector_nonempty_users": sum(bool(relevant[u]) for u in one_vector),
        "single_distinct_history_asset_users": sum(len(contexts[u]["history"]) == 1 for u in users),
        "profiles_questionnaire_at_or_after_cut": sum(
            profiles[u].get("lastQuestionnaireDate", "")[:10] >= contexts[u]["cutoff"] for u in users
        ),
        "verification_example": {
            "user": "[MASKED]", "method": "latest_buy", "relevant_count": len(relevant[check_user]),
            "rank_contributions": contributions, "ndcg_hand_checked": manual,
        },
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-repo", type=Path, default=ROOT.parents[1] / "course-starter")
    parser.add_argument("--render-only", action="store_true",
                        help="Regenerate Markdown from verified saved results; keep original measured timings")
    args = parser.parse_args()
    repo = args.starter_repo.resolve()
    assert not repo.is_relative_to(ROOT.parent), "Starter must be outside Дарья's repository"
    before_status = git(repo, "status", "--porcelain", "--untracked-files=all")
    if before_status:
        raise ValueError("External starter checkout must be clean; refusing to run a modified starter.")
    commit = git(repo, "rev-parse", "HEAD")
    source_paths = [
        "README.md", "project/far-trans/starter.py", "project/far-trans/README.md",
        "project/far-trans/ANNOUNCEMENT.md", "project/far-trans/instructions/homework2.md",
        "project/far-trans/extensions.py", "project/far-trans/test_starter.py",
    ]
    source_hashes = {p: digest(repo / p) for p in source_paths}
    if args.render_only:
        summary = json.loads((REPORTS / "starter_summary.json").read_text(encoding="utf-8"))
        assert summary["starter_commit"] == commit
        assert summary["starter_source_sha256"] == source_hashes
        for name, expected in summary["source_sha256"].items():
            assert digest(DATA / name) == expected
        for label, run in summary["runs"].items():
            for name, expected in run["output_sha256"].items():
                assert digest(RAW_OUTPUT / label / name) == expected
        from starter_reports import render
        render(ROOT, summary)
        print("Regenerated Markdown from source/hash-verified saved results; original timings retained.")
        return
    runs, protocols = {}, {}
    RAW_OUTPUT.mkdir(parents=True, exist_ok=True)
    for label, flags in [("demo", []), ("full_warm", ["--max-users", "0"])]:
        runs[label], protocols[label] = run_cli(repo, label, flags)
    st = import_starter(repo / "project" / "far-trans" / "starter.py")
    cutoff = protocols["demo"]["cutoff"]
    assets, profiles, history, truth, end, excluded = st.load_data(DATA, cutoff, 30)
    assets_end, _, history_end, _, _, _ = st.load_data(DATA, end, 1)
    assert set(assets) <= set(assets_end)
    features = {i: st.tokens(row) for i, row in assets.items()}
    pop = st.popularity_scores(history, assets)
    all_warm = st.select_users(profiles, history, "warm", 0, None)
    assert all_warm == protocols["full_warm"]["users"]
    contexts, predictions = st.contexts_for(all_warm, assets, profiles, history, features, pop, cutoff, 50, False)
    recent = st.recency_rankings(contexts, pop, st.scan_recent_events(DATA, cutoff, assets))
    for u in all_warm:
        predictions[u].update(recent[u])
    for label in runs:
        users = st.select_users(profiles, history, "warm", 100 if label == "demo" else 0, None)
        assert users == protocols[label]["users"]
        assert sorted(assets) == protocols[label]["eligible_assets"]
        assert excluded == runs[label]["excluded_rows"]
        runs[label]["diagnostics"] = audit_cohort(
            st, users, contexts, predictions, history, truth, history_end, profiles, runs[label]
        )
    assert git(repo, "rev-parse", "HEAD") == commit
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == before_status
    assert source_hashes == {p: digest(repo / p) for p in source_paths}
    summary = {
        "starter_repository": "https://github.com/dryjins/RecSys-LLMs",
        "starter_commit": commit, "starter_source_sha256": source_hashes,
        "starter_unchanged_verified": True, "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "cutoff": cutoff, "end_exclusive": end, "candidate_count": 50,
        "eligible_asset_count": len(assets), "repeat_allowed": True,
        "source_sha256": protocols["demo"]["source_sha256"],
        "readme_reference": read_readme_references(repo),
        "runs": runs,
    }
    REPORTS.mkdir(exist_ok=True)
    save_json(REPORTS / "starter_summary.json", summary)
    from starter_reports import render
    render(ROOT, summary)
    # Aggregates and a masked check only; no raw financial client IDs in Git.
    identifiers = set(profiles)
    for name in ("starter_baselines.md", "track_comparison.md", "starter_summary.json"):
        content = (REPORTS / name).read_text(encoding="utf-8")
        assert not any(uid in content for uid in identifiers), f"Customer ID in {name}"
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
