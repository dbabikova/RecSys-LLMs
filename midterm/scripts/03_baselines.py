"""Stage 3b driver: stricter ground truths + personal-popularity score variants."""

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
import evaluate as E

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HALF_LIFE = 14 * 86400  # 2 weeks, for the "with decay" variants

md: list[str] = []


def out(*args) -> None:
    line = " ".join(str(a) for a in args)
    print(line)
    md.append(line)


def aggregate_train(train_watched: pd.DataFrame, half_life: int | None) -> dict:
    """Per (user, show) aggregate of train watched airings.

    Always populates raw fields; populates *_d (decayed) fields if half_life set.
    """
    agg = defaultdict(lambda: dict(ov=0.0, n=0, days=set(), ov_d=0.0, n_d=0.0, days_d={}))
    u = train_watched["user_id"].to_numpy()
    s = train_watched["tv_show_id"].to_numpy()
    ov = train_watched["overlap"].to_numpy()
    vs = train_watched["view_start"].to_numpy()
    ref = E.REF_TIME
    for ui, si, oi, vi in zip(u, s, ov, vs):
        a = agg[(int(ui), int(si))]
        a["ov"] += float(oi)
        a["n"] += 1
        a["days"].add(int(vi) // 86400)
        if half_life is not None:
            w = 0.5 ** ((ref - int(vi)) / half_life)
            a["ov_d"] += float(oi) * w
            a["n_d"] += w
            d = int(vi) // 86400
            a["days_d"][d] = max(a["days_d"].get(d, 0.0), w)
    return agg


def variant_score(a: dict, name: str) -> float:
    if name == "overlap_sec":
        return a["ov"]
    if name == "overlap_sec_decay":
        return a["ov_d"]
    if name == "n_airings":
        return float(a["n"])
    if name == "n_airings_decay":
        return a["n_d"]
    if name == "n_days":
        return float(len(a["days"]))
    if name == "n_days_decay":
        return sum(a["days_d"].values())
    if name == "log1p_overlap":
        return math.log1p(a["ov"])
    if name == "log1p_overlap_decay":
        return math.log1p(a["ov_d"])
    raise ValueError(name)


def main() -> None:
    out("# Протокол 3b: строгие ground truth + personal score-варианты")
    out()

    ctx = pd.read_parquet(config.CACHE_DIR / "user_context.parquet")
    ctx["user_id"] = ctx["user_id"].astype("uint64")
    sub_str = E.load_submission_ids()
    sub_set = set(sub_str)
    assert sub_set <= set(ctx["user_id"].astype(str)), "submission ids not in user_context.parquet"
    out(f"OK: {len(sub_set)} id из submission.csv побайтово совпадают с parquet (uint64).")
    out()

    candidates, titles, val_airing_counts = E.load_val_candidates()
    out(f"Кандидатов (tv_show_id!=0, эфир в val): {len(candidates):,}")
    out()

    print("Recomputing per-airing labeled matches (reuses join.py, nothing cached) ...", flush=True)
    t0 = time.time()
    airings = E.load_labeled_airings(E.CAP)
    print(f"  ... {len(airings):,} labeled pairs in {time.time() - t0:.1f}s", flush=True)

    ov = airings["overlap"].to_numpy()
    dur = airings["duration"].to_numpy()
    vst = airings["view_start"].to_numpy()
    train_mask = vst < E.VAL_START
    val_mask = (vst >= E.VAL_START) & (vst <= E.VAL_END)
    watched = E.watched_mask_np(ov, dur)

    val_airings = airings[val_mask]
    train_watched = airings[train_mask & watched]
    del airings, ov, dur, vst, watched

    # ---- ground truths --------------------------------------------------
    sub_int = sorted(int(u) for u in sub_set)
    gts = {}
    cohorts = {}
    out("## Ground truth (три определения «посмотрел в val»)")
    out()
    out("| GT | когорта | средний |R_u| | медианный |R_u| |")
    out("|---|---|---|---|")
    for d in ("A", "B", "C"):
        gts[d] = E.build_ground_truth(val_airings, d)
        cohorts[d] = [u for u in sub_int if u in gts[d]]
        rel = [len(gts[d][u]) for u in cohorts[d]]
        out(f"| {d} | {len(cohorts[d]):,} | {np.mean(rel):.1f} | {np.median(rel):.1f} |")
    out()

    # ---- global popularity ---------------------------------------------
    agg = aggregate_train(train_watched, half_life=HALF_LIFE)

    pop = defaultdict(set)
    for (u, s) in agg:
        if s in candidates:
            pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}
    global_rank = sorted(pop_count, key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]
    out(f"Top-5 global popularity: {global_top5}")
    out()

    # ---- personal variants ---------------------------------------------
    user_items: dict[int, list[int]] = defaultdict(list)
    for (u, s) in agg:
        if s in candidates:
            user_items[u].append(s)

    def build_rankings(vname: str):
        rankings: dict[int, list[int]] = {}
        n_pad = 0
        for u in cohorts["A"]:
            items = sorted(user_items.get(u, []), key=lambda s: (-variant_score(agg[(u, s)], vname), s))
            top = items[:5]
            if len(top) < 5:
                n_pad += 5 - len(top)
                seen = set(top)
                for gs in global_rank:
                    if len(top) >= 5:
                        break
                    if gs not in seen:
                        top.append(gs)
                        seen.add(gs)
            rankings[u] = top
        return rankings, n_pad

    variants = [
        "overlap_sec", "overlap_sec_decay",
        "n_airings", "n_airings_decay",
        "n_days", "n_days_decay",
        "log1p_overlap", "log1p_overlap_decay",
    ]

    out("## Personal score-варианты (MAP@5 / Recall@5 на GT-A)")
    out()
    out("| score | MAP@5 | Recall@5 |")
    out("|---|---|---|")
    var_results = {}
    for vname in variants:
        rankings, n_pad = build_rankings(vname)
        res = E.evaluate(rankings, gts["A"], cohorts["A"])
        var_results[vname] = (rankings, res, n_pad)
        out(f"| {vname} | {res['map5']:.6f} | {res['recall5']:.6f} |")
    out()

    best_name = max(variants, key=lambda v: var_results[v][1]["map5"])
    best_rankings = var_results[best_name][0]
    out(f"Лучший score: **{best_name}** (MAP@5={var_results[best_name][1]['map5']:.6f}).")
    out()

    # ---- why overlap_sec loses to global -------------------------------
    out("## Почему overlap_sec проигрывает global")
    out()
    low = []
    for u in cohorts["A"]:
        items = sorted(user_items.get(u, []), key=lambda s: (-variant_score(agg[(u, s)], "overlap_sec"), s))[:5]
        for s in items:
            low.append((val_airing_counts.get(s, 0), u, s))
    low.sort(key=lambda x: x[0])
    out("| user_id | tv_show_id | название | эфиров в val |")
    out("|---|---|---|---|")
    shown = set()
    n = 0
    for cnt, u, s in low:
        if s in shown:
            continue
        shown.add(s)
        flag = " (не идёт в val)" if cnt == 0 else ""
        out(f"| {u} | {s} | {titles.get(s, '?')[:45]} | {cnt}{flag} |")
        n += 1
        if n >= 3:
            break
    out()
    out("overlap_sec (сумма секунд) переоценивает передачи, которые юзер много смотрел в train")
    out("(binge — например, редкие эфиры или сериалы, закончившиеся до val), тогда как число зрителей")
    out("(global) и число разных дней (n_days) лучше отражают регулярный просмотр, который продолжается и в val.")
    out()

    # ---- final table ----------------------------------------------------
    out("## Итоговая таблица (метод × GT)")
    out()
    out("| метод | GT | MAP@5 | Recall@5 | покрытие |")
    out("|---|---|---|---|---|")
    rows = []
    for d in ("A", "B", "C"):
        coh = cohorts[d]
        gt = gts[d]
        gr = {u: list(global_top5) for u in coh}
        rg = E.evaluate(gr, gt, coh)
        rp = E.evaluate(best_rankings, gt, coh)
        rows.append(["global", d, round(rg["map5"], 6), round(rg["recall5"], 6), rg["coverage"]])
        rows.append([f"personal+global({best_name})", d, round(rp["map5"], 6), round(rp["recall5"], 6), rp["coverage"]])
        out(f"| global | {d} | {rg['map5']:.6f} | {rg['recall5']:.6f} | {rg['coverage']:,} |")
        out(f"| personal+global({best_name}) | {d} | {rp['map5']:.6f} | {rp['recall5']:.6f} | {rp['coverage']:,} |")
    out()

    pd.DataFrame(rows, columns=["method", "gt", "MAP@5", "Recall@5", "coverage"]).to_csv(
        config.RESULTS_DIR / "baselines.csv", index=False
    )

    # ---- handcheck AP ---------------------------------------------------
    target = None
    for u in cohorts["A"]:
        ap = E.ap_at_k(best_rankings[u], gts["A"][u])
        if 0.2 < ap < 0.8:
            target = u
            break
    if target is None:
        target = cohorts["A"][0]

    pred = best_rankings[target]
    rel = gts["A"][target]
    hc: list[str] = []
    hc.append(f"# Ручная проверка AP@5 (best score = {best_name}, GT-A)")
    hc.append("")
    hc.append(f"user_id = {target}; |R_u| = {len(rel)}")
    hc.append("")
    hc.append("| r | tv_show_id | rel | P@r | вклад |")
    hc.append("|---|---|---|---|---|")
    hits = 0
    ap_sum = 0.0
    denom = min(5, len(rel))
    for r in range(1, 6):
        item = pred[r - 1]
        isrel = item in rel
        hits += int(isrel)
        p_at_r = hits / r
        contrib = p_at_r * int(isrel) / denom
        ap_sum += contrib
        hc.append(f"| {r} | {item} | {int(isrel)} | {p_at_r:.3f} | {contrib:.4f} |")
    hc.append(f"| | | | | AP@5 = {ap_sum:.6f} |")
    hc.append("")
    (config.RESULTS_DIR / "handcheck_ap.md").write_text("\n".join(hc) + "\n", encoding="utf-8")

    out(f"## Ручная проверка AP@5")
    out()
    out(f"user_id = {target}; |R_u| = {len(rel)}; AP@5 = {ap_sum:.6f}")
    for r in range(1, 6):
        item = pred[r - 1]
        out(f"  r={r}: id={item} rel={int(item in rel)}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "baselines_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "baselines_summary.md")


if __name__ == "__main__":
    main()
