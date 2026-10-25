"""Stage 8b: switching rule for light viewers (val protocol, deterministic tie-break)."""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
import evaluate as E
import mf_als as MF
import cbf as CBF

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

md: list[str] = []


def out(*args) -> None:
    line = " ".join(str(a) for a in args)
    print(line)
    md.append(line)


def rank_map(scores, ids, top_n=200):
    ids = np.asarray(ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((ids, -scores))
    return {int(ids[i]): r for r, i in enumerate(order[:top_n], 1)}


def combine(parts):
    V = np.hstack([w * M for w, M in parts if w and M.shape[1] > 0])
    return CBF.l2_normalize(V)


def pad5(top, global_top5):
    top = list(top)
    seen = set(top)
    for gs in global_top5:
        if len(top) >= 5:
            break
        if gs not in seen:
            top.append(int(gs))
            seen.add(int(gs))
    return top[:5]


def main() -> None:
    out("# Switching-правило для лёгких зрителей (этап 8b)")
    out()

    sub_int = sorted(int(u) for u in E.load_submission_ids())

    print("Loading ...", flush=True)
    airings = E.load_labeled_airings(E.CAP)
    ov = airings["overlap"].to_numpy()
    dur = airings["duration"].to_numpy()
    vst = airings["view_start"].to_numpy()
    train_mask = vst < E.VAL_START
    val_mask = (vst >= E.VAL_START) & (vst <= E.VAL_END)
    watched = E.watched_mask_np(ov, dur)
    train_watched = airings[train_mask & watched]
    val_airings = airings[val_mask]
    del airings, ov, dur, vst, watched

    gts = {d: E.build_ground_truth(val_airings, d) for d in ("A", "B", "C")}
    cohorts = {d: [u for u in sub_int if u in gts[d]] for d in ("A", "B", "C")}

    days = defaultdict(set)
    for u, s, vs in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"]):
        days[(int(u), int(s))].add(int(vs) // 86400)
    n_days = {k: len(v) for k, v in days.items()}
    train_items = set(s for (_u, s) in n_days)

    pop = defaultdict(set)
    for (u, s) in n_days:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}

    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)
    user_train_days = defaultdict(set)
    for (u, s), ds in days.items():
        user_train_days[u].update(ds)

    train_shows = {u: len(user_train_items[u]) for u in cohorts["A"]}
    train_days = {u: len(user_train_days.get(u, ())) for u in cohorts["A"]}

    meta, val_cands, title_map, _ = CBF.load_show_metadata()
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]

    # ---- MF ------------------------------------------------------------
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)
    cand_mf = sorted(c for c in val_cands if c in item_index)
    cand_mf_idx = np.array([item_index[c] for c in cand_mf], dtype=np.int64)
    Yc = Y[cand_mf_idx]
    pop_log = np.array([np.log1p(pop_count.get(c, 0)) for c in cand_mf])

    # ---- hybrid rankings (personal 1.0 + mfpop 0.5) -------------------
    hybrid_rankings = {}
    for u in cohorts["A"]:
        sc = {s: n_days[(u, s)] for s in val_cands if (u, s) in n_days}
        items = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        rm_p = {s: r for r, (s, _v) in enumerate(items[:200], 1)}
        ri = user_index.get(int(u))
        rm_mfp = {}
        if ri is not None:
            s = X[ri] @ Yc.T
            mu = s.mean()
            sd = s.std()
            z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
            comb = z + 1.0 * pop_log
            rm_mfp = rank_map(comb, cand_mf, top_n=200)
        show_set = {}
        for s, r in rm_p.items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][0] = 1.0 / (60.0 + r)
        for s, r in rm_mfp.items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][1] = 1.0 / (60.0 + r)
        hscored = [(show_set[s][0] * 1.0 + show_set[s][1] * 0.5, s) for s in show_set]
        hscored.sort(key=lambda x: (-x[0], x[1]))
        hybrid_rankings[u] = pad5([s for _sc, s in hscored[:5]], global_top5)

    gr = {u: list(global_top5) for u in cohorts["A"]}
    h_ra = E.evaluate(hybrid_rankings, gts["A"], cohorts["A"])
    out(f"Гибрид без переключения: MAP@5 A/B/C = {h_ra['map5']:.6f} / (0.616455) / (0.678871)")
    out()

    # ---- switching rules ----------------------------------------------
    out("## Правила переключения (global если train_shows<T ИЛИ train_days<D)")
    out()
    out("| T | D | переключено | MAP@5 A | MAP@5 B | MAP@5 C |")
    out("|---|---|---|---|---|---|")
    results = []
    for T in (3, 5, 10, 20):
        for D in (2, 5, 10):
            switched = {}
            n_sw = 0
            for u in cohorts["A"]:
                if train_shows[u] < T or train_days[u] < D:
                    switched[u] = list(global_top5)
                    n_sw += 1
                else:
                    switched[u] = hybrid_rankings[u]
            ra = E.evaluate(switched, gts["A"], cohorts["A"])
            rb = E.evaluate(switched, gts["B"], cohorts["B"])
            rc = E.evaluate(switched, gts["C"], cohorts["C"])
            results.append((T, D, n_sw, ra["map5"], rb["map5"], rc["map5"]))
            out(f"| {T} | {D} | {n_sw} | {ra['map5']:.6f} | {rb['map5']:.6f} | {rc['map5']:.6f} |")
    out()

    best = max(results, key=lambda r: r[3])
    out(f"Лучшее по A: T={best[0]}, D={best[1]} (переключено {best[2]}), A={best[3]:.6f}, B={best[4]:.6f}, C={best[5]:.6f}")
    out()
    wins_a = best[3] > 0.790556
    ok_b = best[4] >= 0.616455
    ok_c = best[5] >= 0.678871
    out(f"Выигрывает на A: {'да' if wins_a else 'нет'}; не хуже на B: {'да' if ok_b else 'нет'}; на C: {'да' if ok_c else 'нет'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "switching_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "switching_summary.md")


if __name__ == "__main__":
    main()
