"""Stage 6b: final hybrid tuning (n_air boost, fine weights, personal-series fix)."""

import json
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


def build_personal_rankings(n_days, candidates, global_rank, cohort):
    user_items = defaultdict(list)
    for (u, s) in n_days:
        if s in candidates:
            user_items[u].append(s)
    rankings = {}
    for u in cohort:
        items = sorted(user_items.get(u, []), key=lambda s: (-n_days[(u, s)], s))
        top = items[:5]
        if len(top) < 5:
            seen = set(top)
            for gs in global_rank:
                if len(top) >= 5:
                    break
                if gs not in seen:
                    top.append(gs)
                    seen.add(gs)
        rankings[u] = top
    return rankings


def combine(parts):
    V = np.hstack([w * M for w, M in parts if w and M.shape[1] > 0])
    return CBF.l2_normalize(V)


def sparse_rank_map(cohort, score_lookup, candidates, top_n=200):
    rm = {}
    for u in cohort:
        sc = score_lookup.get(u, {})
        items = sorted([(v, s) for s, v in sc.items() if s in candidates and v > 0], key=lambda x: (-x[0], x[1]))
        rm[u] = {s: r for r, (_v, s) in enumerate(items[:top_n], 1)}
    return rm


def main() -> None:
    out("# Доводка гибрида (этап 6b)")
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

    pop_count = defaultdict(set)
    for (u, s) in n_days:
        pop_count[s].add(u)
    pop_count = {s: len(v) for s, v in pop_count.items()}

    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)

    meta, candidates, title_map, airing_counts = CBF.load_show_metadata()
    global_rank = sorted([s for s in pop_count if s in candidates], key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]
    base_of = {s: CBF.normalize_title(title_map.get(s, str(s))) for s in (train_items | candidates)}

    # ---- MF ------------------------------------------------------------
    print("Running best MF (32,50,0.5) ...", flush=True)
    t_mf = time.time()
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)
    t_mf = time.time() - t_mf
    cand_mf = [c for c in candidates if c in item_index]
    cand_mf_idx = [item_index[c] for c in cand_mf]
    Yc = Y[cand_mf_idx]
    pop_log_mf = np.array([np.log1p(pop_count.get(c, 0)) for c in cand_mf])

    mf_rankings = {}
    rm_mfpop = {}
    mfpop_top5 = {}
    for u in cohorts["A"]:
        ri = user_index.get(int(u))
        if ri is None:
            mf_rankings[u] = list(global_top5)
            rm_mfpop[u] = {}
            mfpop_top5[u] = list(global_top5)
            continue
        s = X[ri] @ Yc.T
        idx = np.argsort(-s)[:5]
        rec = [cand_mf[i] for i in idx]
        if len(rec) < 5:
            seen = set(rec)
            for gs in global_top5:
                if len(rec) >= 5:
                    break
                if gs not in seen:
                    rec.append(gs)
                    seen.add(gs)
        mf_rankings[u] = rec
        mu = s.mean()
        sd = s.std()
        z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
        comb = z + 1.0 * pop_log_mf
        idx2 = np.argpartition(-comb, min(200, len(comb) - 1))[:200]
        idx2 = idx2[np.argsort(-comb[idx2])]
        rm_mfpop[u] = {cand_mf[i]: r for r, i in enumerate(idx2, 1)}
        top = [cand_mf[i] for i in idx2[:5]]
        if len(top) < 5:
            seen = set(top)
            for gs in global_top5:
                if len(top) >= 5:
                    break
                if gs not in seen:
                    top.append(gs)
                    seen.add(gs)
        mfpop_top5[u] = top

    # ---- CBF blocks ----------------------------------------------------
    t_cbf = time.time()
    all_shows = sorted(train_items | candidates)
    show_index, cat_mat, genre_mat, year_mat, people_mat = CBF.build_blocks(all_shows, meta)
    base_onehot, title_tfidf = CBF.build_title_blocks(all_shows, title_map)
    t_cbf = time.time() - t_cbf
    user_items = defaultdict(dict)
    for (u, s), nd in n_days.items():
        user_items[u][s] = nd

    V_cg = combine([(1.0, cat_mat), (1.0, genre_mat)])
    prof_cg = CBF.build_profiles(user_items, show_index, V_cg)
    cbf_cg_rankings, _ = CBF.recommend_cbf(prof_cg, V_cg, show_index, candidates, global_rank, cohorts["A"])

    Vt = combine([(10.0, base_onehot), (1.0, title_tfidf)])
    prof_t = CBF.build_profiles(user_items, show_index, Vt)
    cbft_rankings, _ = CBF.recommend_cbf(prof_t, Vt, show_index, candidates, global_rank, cohorts["A"])

    cand_cbf = [c for c in candidates if c in show_index]
    cand_cbf_idx = [show_index[c] for c in cand_cbf]
    Vtc = Vt[cand_cbf_idx]
    rm_cbft = {}
    for u in cohorts["A"]:
        p = prof_t.get(int(u))
        if p is None or float(np.linalg.norm(p)) == 0.0:
            rm_cbft[u] = {}
            continue
        pn = p / np.linalg.norm(p)
        s = pn @ Vtc.T
        idx = np.argpartition(-s, min(200, len(s) - 1))[:200]
        idx = idx[np.argsort(-s[idx])]
        rm_cbft[u] = {cand_cbf[i]: r for r, i in enumerate(idx, 1)}

    personal_rankings = build_personal_rankings(n_days, candidates, global_rank, cohorts["A"])
    personal_scores = defaultdict(dict)
    for (u, s), nd in n_days.items():
        personal_scores[u][s] = nd
    rm_personal = sparse_rank_map(cohorts["A"], personal_scores, candidates)

    # ---- personal-series fixed -----------------------------------------
    series_days = defaultdict(lambda: defaultdict(set))
    for (u, s), ds in days.items():
        series_days[u][base_of[s]].update(ds)
    series_ndays = {u: {b: len(v) for b, v in vv.items()} for u, vv in series_days.items()}

    def personal_series_fixed(cohort):
        rankings = {}
        for u in cohort:
            us = series_ndays.get(u, {})
            best = []
            for base, nd in us.items():
                ids = [s for s in candidates if base_of[s] == base]
                if ids:
                    best_id = max(ids, key=lambda s: airing_counts.get(s, 0))
                    best.append((nd, best_id))
            best.sort(key=lambda x: (-x[0], x[1]))
            top = [s for _, s in best[:5]]
            if len(top) < 5:
                seen = set(top)
                for gs in global_rank:
                    if len(top) >= 5:
                        break
                    if gs not in seen:
                        top.append(gs)
                        seen.add(gs)
            rankings[u] = top
        return rankings

    ps_fixed = personal_series_fixed(cohorts["A"])
    psf_ra = E.evaluate(ps_fixed, gts["A"], cohorts["A"])
    psf_rb = E.evaluate(ps_fixed, gts["B"], cohorts["B"])
    psf_rc = E.evaluate(ps_fixed, gts["C"], cohorts["C"])
    out(f"## Personal-series (fix: 1 слот на серию, лучший id по эфирам)")
    out(f"MAP@5 A/B/C = {psf_ra['map5']:.6f}/{psf_rb['map5']:.6f}/{psf_rc['map5']:.6f} (было 0.473/0.384/0.422)")
    out()

    # ---- hybrid search (weights x n_air boost) -------------------------
    out("## Гибрид: сетка весов + буст по числу эфиров")
    out()
    union_shows = {}
    inv_rank = {}
    n_air_log = {}
    for u in cohorts["A"]:
        show_set = {}
        for mi, rm in enumerate([rm_personal, rm_mfpop, rm_cbft]):
            for s, r in rm.get(u, {}).items():
                show_set.setdefault(s, [0.0, 0.0, 0.0])
                show_set[s][mi] = 1.0 / (60.0 + r)
        shows = list(show_set.keys())
        if not shows:
            union_shows[u] = list(global_top5)
            inv_rank[u] = np.zeros((5, 3))
            n_air_log[u] = np.zeros(5)
        else:
            union_shows[u] = shows
            inv_rank[u] = np.array([show_set[s] for s in shows])
            n_air_log[u] = np.array([np.log1p(airing_counts.get(s, 0)) for s in shows])

    mfpop_w = [0.25, 0.5, 0.75, 1.0]
    cbft_w = [0.0, 0.1, 0.25]
    gammas = [0.0, 0.1, 0.3, 0.5]
    combos = []
    W48 = []
    g48 = []
    for wm in mfpop_w:
        for wc in cbft_w:
            for g in gammas:
                combos.append((wm, wc, g))
                W48.append([1.0, wm, wc])
                g48.append(g)
    W48 = np.array(W48)
    g48 = np.array(g48)

    aps = np.zeros(len(combos))
    t_fuse = time.time()
    for u in cohorts["A"]:
        F = inv_rank[u] @ W48.T  # n_union x 48
        S = F * (1.0 + g48 * n_air_log[u][:, None])
        order = np.argsort(-S, axis=0)[:5]
        rel = np.array([1 if s in gts["A"][u] else 0 for s in union_shows[u]], dtype=np.float64)
        r_sel = rel[order]
        hits = np.cumsum(r_sel, axis=0)
        prec = hits / np.arange(1, 6)[:, None]
        denom = min(5, len(gts["A"][u]))
        aps += (prec * r_sel).sum(axis=0) / denom
    map5_grid = aps / len(cohorts["A"])
    t_fuse = time.time() - t_fuse

    best_ci = int(np.argmax(map5_grid))
    wm_b, wc_b, g_b = combos[best_ci]
    out(f"Лучший конфиг: MF+pop={wm_b}, CBF-title={wc_b}, gamma={g_b} (personal=1.0)")
    out(f"MAP@5 A = {map5_grid[best_ci]:.6f}")

    hybrid_rankings = {}
    for u in cohorts["A"]:
        F = inv_rank[u] @ np.array([1.0, wm_b, wc_b])
        S = F * (1.0 + g_b * n_air_log[u])
        order = np.argsort(-S)[:5]
        top = [union_shows[u][i] for i in order]
        if len(top) < 5:
            seen = set(top)
            for gs in global_top5:
                if len(top) >= 5:
                    break
                if gs not in seen:
                    top.append(gs)
                    seen.add(gs)
        hybrid_rankings[u] = top

    h_ra = E.evaluate(hybrid_rankings, gts["A"], cohorts["A"])
    h_rb = E.evaluate(hybrid_rankings, gts["B"], cohorts["B"])
    h_rc = E.evaluate(hybrid_rankings, gts["C"], cohorts["C"])
    out(f"Гибрид финальный: MAP@5 A/B/C = {h_ra['map5']:.6f}/{h_rb['map5']:.6f}/{h_rc['map5']:.6f}")
    out()

    # ---- final table ----------------------------------------------------
    gr = {u: list(global_top5) for u in cohorts["A"]}
    final = []
    def row(name, rank, time_s):
        ra = E.evaluate(rank, gts["A"], cohorts["A"])
        rb = E.evaluate(rank, gts["B"], cohorts["B"])
        rc = E.evaluate(rank, gts["C"], cohorts["C"])
        return [name, ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"], time_s]

    final.append(row("global", gr, 0.0))
    final.append(row("personal+global(n_days)", personal_rankings, 0.0))
    final.append(row("personal-series(fix)", ps_fixed, 0.0))
    final.append(row("MF(32,50,0.5)", mf_rankings, t_mf))
    final.append(row("MF+pop(beta=1)", mfpop_top5, 0.0))
    final.append(row("CBF(cat+genre)", cbf_cg_rankings, t_cbf))
    final.append(row("CBF-title(title)", cbft_rankings, t_cbf))
    final.append(row("hybrid", hybrid_rankings, t_fuse))

    out("## Итоговая таблица")
    out()
    out("| метод | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие | время, с |")
    out("|---|---|---|---|---|---|---|")
    for r in final:
        out(f"| {r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]:,} | {r[6]:.1f} |")
    out()

    pd.DataFrame(final, columns=["method", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage", "time_s"]).to_csv(
        config.RESULTS_DIR / "all_methods.csv", index=False
    )

    # ---- final config ---------------------------------------------------
    final_cfg = {
        "hybrid_weights": {"personal": 1.0, "personal_series": 0.0, "mf_pop": wm_b, "cbf_title": wc_b},
        "gamma": g_b,
        "mf": {"factors": 32, "reg": 50.0, "alpha": 0.5},
        "seed": 42,
        "map5_A": h_ra["map5"],
        "map5_B": h_rb["map5"],
        "map5_C": h_rc["map5"],
    }
    (config.RESULTS_DIR / "final_config.json").write_text(json.dumps(final_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- verification ---------------------------------------------------
    best_single = max(r[1] for r in final if r[0] != "hybrid")
    pers_b = [r for r in final if r[0] == "personal+global(n_days)"][0]
    out("## Проверка финала")
    out(f"Гибрид MAP@5 A = {h_ra['map5']:.6f} (лучший одиночный = {best_single:.6f}) -> {'OK' if h_ra['map5'] >= best_single else 'FAIL'}")
    out(f"Гибрид B={h_rb['map5']:.6f} vs personal B={pers_b[2]:.6f}; C={h_rc['map5']:.6f} vs personal C={pers_b[3]:.6f} -> "
        f"{'OK' if h_rb['map5'] >= pers_b[2] and h_rc['map5'] >= pers_b[3] else 'FAIL'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "cbf_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "cbf_summary.md")


if __name__ == "__main__":
    main()
