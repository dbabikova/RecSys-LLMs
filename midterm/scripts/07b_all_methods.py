"""Stage 7b: recalculate all_methods.csv with deterministic tie-breaking (GT-A/B/C)."""

import hashlib
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


def topk(scores, ids, k=5):
    ids = np.asarray(ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((ids, -scores))
    return [int(ids[i]) for i in order[:k]]


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
    out("# Пересчёт all_methods.csv (детерминированный tie-break)")
    out()

    sub_int = sorted(int(u) for u in E.load_submission_ids())

    print("Loading per-airing matches ...", flush=True)
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

    meta, val_cands, title_map, airing_counts = CBF.load_show_metadata()
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]
    base_of = {s: CBF.normalize_title(title_map.get(s, str(s))) for s in (train_items | val_cands)}

    # ---- MF ------------------------------------------------------------
    t_mf = time.time()
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)
    t_mf = time.time() - t_mf
    cand_mf = sorted(c for c in val_cands if c in item_index)
    cand_mf_idx = np.array([item_index[c] for c in cand_mf], dtype=np.int64)
    Yc = Y[cand_mf_idx]
    pop_log = np.array([np.log1p(pop_count.get(c, 0)) for c in cand_mf])

    mf_rankings = {}
    mfpop_rankings = {}
    rm_mfpop = {}
    for u in cohorts["A"]:
        ri = user_index.get(int(u))
        if ri is None:
            mf_rankings[u] = list(global_top5)
            mfpop_rankings[u] = list(global_top5)
            rm_mfpop[u] = {}
            continue
        s = X[ri] @ Yc.T
        mf_rankings[u] = pad5(topk(s, cand_mf, k=5), global_top5)
        mu = s.mean()
        sd = s.std()
        z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
        comb = z + 1.0 * pop_log
        mfpop_rankings[u] = pad5(topk(comb, cand_mf, k=5), global_top5)
        rm_mfpop[u] = rank_map(comb, cand_mf, top_n=200)

    # ---- CBF blocks ----------------------------------------------------
    t_cbf = time.time()
    all_shows = sorted(train_items | val_cands)
    show_index, cat_mat, genre_mat, year_mat, people_mat = CBF.build_blocks(all_shows, meta)
    base_onehot, title_tfidf = CBF.build_title_blocks(all_shows, title_map)
    t_cbf = time.time() - t_cbf
    user_items = defaultdict(dict)
    for (u, s), nd in n_days.items():
        user_items[u][s] = nd

    V_cg = combine([(1.0, cat_mat), (1.0, genre_mat)])
    prof_cg = CBF.build_profiles(user_items, show_index, V_cg)
    cbf_cg_rankings, _ = CBF.recommend_cbf(prof_cg, V_cg, show_index, val_cands, global_rank, cohorts["A"])

    Vt = combine([(10.0, base_onehot), (1.0, title_tfidf)])
    prof_t = CBF.build_profiles(user_items, show_index, Vt)
    cbft_rankings, _ = CBF.recommend_cbf(prof_t, Vt, show_index, val_cands, global_rank, cohorts["A"])
    cand_cbf = sorted(c for c in val_cands if c in show_index)
    cand_cbf_idx = np.array([show_index[c] for c in cand_cbf], dtype=np.int64)
    Vtc = Vt[cand_cbf_idx]
    rm_cbft = {}
    for u in cohorts["A"]:
        p = prof_t.get(int(u))
        if p is None or float(np.linalg.norm(p)) == 0.0:
            rm_cbft[u] = {}
            continue
        pn = p / np.linalg.norm(p)
        s = pn @ Vtc.T
        rm_cbft[u] = rank_map(s, cand_cbf, top_n=200)

    # ---- personal ------------------------------------------------------
    personal_rankings = {}
    rm_personal = {}
    for u in cohorts["A"]:
        sc = {s: n_days[(u, s)] for s in val_cands if (u, s) in n_days}
        items = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        personal_rankings[u] = pad5([s for s, _v in items[:5]], global_top5)
        rm_personal[u] = {s: r for r, (s, _v) in enumerate(items[:200], 1)}

    # ---- personal-series (fixed) --------------------------------------
    series_days = defaultdict(lambda: defaultdict(set))
    for (u, s), ds in days.items():
        series_days[u][base_of[s]].update(ds)
    series_ndays = {u: {b: len(v) for b, v in vv.items()} for u, vv in series_days.items()}
    ps_rankings = {}
    for u in cohorts["A"]:
        us = series_ndays.get(u, {})
        best = []
        for base, nd in us.items():
            ids = sorted(s for s in val_cands if base_of[s] == base)
            if ids:
                best_id = max(ids, key=lambda s: airing_counts.get(s, 0))
                best.append((nd, best_id))
        best.sort(key=lambda x: (-x[0], x[1]))
        ps_rankings[u] = pad5([s for _, s in best[:5]], global_top5)

    # ---- hybrid (fixed weights personal=1.0, mfpop=0.5) ----------------
    hybrid_rankings = {}
    for u in cohorts["A"]:
        show_set = {}
        for s, r in rm_personal[u].items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][0] = 1.0 / (60.0 + r)
        for s, r in rm_mfpop[u].items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][1] = 1.0 / (60.0 + r)
        scored = [(show_set[s][0] * 1.0 + show_set[s][1] * 0.5, s) for s in show_set]
        scored.sort(key=lambda x: (-x[0], x[1]))
        hybrid_rankings[u] = pad5([s for _sc, s in scored[:5]], global_top5)

    # ---- deterministic weight re-search (to note if changed) -----------
    wm_list = [0.25, 0.5, 0.75, 1.0]
    wc_list = [0.0, 0.1, 0.25]
    g_list = [0.0, 0.1, 0.3, 0.5]
    best_search = None
    t_search = time.time()
    for wm in wm_list:
        for wc in wc_list:
            for g in g_list:
                ap_sum = 0.0
                for u in cohorts["A"]:
                    scored = {}
                    for s, r in rm_personal[u].items():
                        scored[s] = 1.0 / (60.0 + r)
                    for s, r in rm_mfpop[u].items():
                        scored[s] = scored.get(s, 0.0) + wm / (60.0 + r)
                    for s, r in rm_cbft[u].items():
                        scored[s] = scored.get(s, 0.0) + wc / (60.0 + r)
                    items = [(scored[s] * (1.0 + g * np.log1p(airing_counts.get(s, 0))), s) for s in scored]
                    items.sort(key=lambda x: (-x[0], x[1]))
                    pred = [s for _sc, s in items[:5]]
                    ap_sum += E.ap_at_k(pred, gts["A"][u])
                map5 = ap_sum / len(cohorts["A"])
                if best_search is None or map5 > best_search[0]:
                    best_search = (map5, wm, wc, g)
    t_search = time.time() - t_search
    out(f"Детерминированный поиск весов: лучший MF+pop={best_search[1]}, CBF-title={best_search[2]}, gamma={best_search[3]} (MAP@5 A={best_search[0]:.6f})")
    if (best_search[1], best_search[2], best_search[3]) != (0.5, 0.0, 0.0):
        out("ВНИМАНИЕ: веса выбрались бы иначе, чем в финале (personal=1, MF+pop=0.5, gamma=0). Финал НЕ меняю.")
    else:
        out("Веса совпали с финалом (personal=1, MF+pop=0.5, gamma=0).")
    out()

    # ---- evaluate all methods ------------------------------------------
    gr = {u: list(global_top5) for u in cohorts["A"]}
    methods = [
        ("global", gr),
        ("personal+global(n_days)", personal_rankings),
        ("personal-series(fix)", ps_rankings),
        ("MF(32,50,0.5)", mf_rankings),
        ("MF+pop(beta=1)", mfpop_rankings),
        ("CBF(cat+genre)", cbf_cg_rankings),
        ("CBF-title(title)", cbft_rankings),
        ("hybrid", hybrid_rankings),
    ]

    out("## Итоговая таблица (детерминированный tie-break)")
    out()
    out("| метод | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие | время, с |")
    out("|---|---|---|---|---|---|---|")
    rows = []
    for name, rank in methods:
        ra = E.evaluate(rank, gts["A"], cohorts["A"])
        rb = E.evaluate(rank, gts["B"], cohorts["B"])
        rc = E.evaluate(rank, gts["C"], cohorts["C"])
        t = 0.0
        if name.startswith("MF"):
            t = t_mf
        elif name.startswith("CBF"):
            t = t_cbf
        elif name == "hybrid":
            t = t_search
        rows.append([name, ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"], t])
        out(f"| {name} | {ra['map5']:.6f} | {rb['map5']:.6f} | {rc['map5']:.6f} | {ra['recall5']:.6f} | {ra['coverage']:,} | {t:.1f} |")
    out()

    pd.DataFrame(rows, columns=["method", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage", "time_s"]).to_csv(
        config.RESULTS_DIR / "all_methods.csv", index=False
    )

    # ---- verification ---------------------------------------------------
    hybrid_row = [r for r in rows if r[0] == "hybrid"][0]
    personal_row = [r for r in rows if r[0] == "personal+global(n_days)"][0]
    best_single = max(r[1] for r in rows if r[0] != "hybrid")
    out("## Проверка")
    out(f"Гибрид A={hybrid_row[1]:.6f} vs лучший одиночный={best_single:.6f} -> {'OK' if hybrid_row[1] >= best_single else 'FAIL'}")
    out(f"Гибрид B={hybrid_row[2]:.6f} vs personal B={personal_row[2]:.6f}; C={hybrid_row[3]:.6f} vs personal C={personal_row[3]:.6f} -> "
        f"{'OK' if hybrid_row[2] >= personal_row[2] and hybrid_row[3] >= personal_row[3] else 'FAIL'}")
    out()

    # md5 unchanged
    md5 = hashlib.md5((config.SUBMISSIONS_DIR / "final_submission.csv").read_bytes()).hexdigest()
    out(f"md5(final_submission.csv) = {md5}")
    out(f"{'OK: не изменился' if md5 == '40bdac823129b75eab374345e4d7d52d' else 'ИЗМЕНИЛСЯ!'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "all_methods_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "all_methods_summary.md")


if __name__ == "__main__":
    main()
