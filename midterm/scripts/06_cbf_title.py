"""Stage 6: title-based CBF, personal-series, and rank-fusion hybrid."""

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


def repeat_fraction(rankings, user_train_items, cohort):
    total = 0
    reps = 0
    for u in cohort:
        ti = user_train_items.get(u, set())
        for s in rankings.get(u, [])[:5]:
            total += 1
            if s in ti:
                reps += 1
    return reps / total if total else 0.0


def new_show_metrics(rankings, new_shows, gts, cohort):
    hits = sum(1 for u in cohort for s in rankings.get(u, [])[:5] if s in new_shows)
    recs = []
    for u in cohort:
        rel = gts["A"][u] & new_shows
        if rel:
            h = sum(1 for s in rankings.get(u, [])[:5] if s in rel)
            recs.append(h / len(rel))
    return hits, float(np.mean(recs)) if recs else 0.0, len(recs)


def combine(parts):
    V = np.hstack([w * M for w, M in parts if w and M.shape[1] > 0])
    return CBF.l2_normalize(V)


def main() -> None:
    out("# Title-CBF и гибрид (этап 6)")
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

    meta, candidates, title_map, _air = CBF.load_show_metadata()
    global_rank = sorted([s for s in pop_count if s in candidates], key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]

    base_of = {s: CBF.normalize_title(title_map.get(s, str(s))) for s in (train_items | candidates)}
    new_shows = candidates - train_items

    # ---- task 1: hypothesis --------------------------------------------
    out("## 1. Гипотеза: базовое название новых передач совпадает с train")
    out()
    train_base = defaultdict(list)
    for s in train_items:
        train_base[base_of[s]].append(s)
    matched_new = [s for s in new_shows if base_of[s] in train_base]
    total_gt_new = sum(len(gts["A"][u] & new_shows) for u in cohorts["A"])
    matched_set = set(matched_new)
    gt_new_matched = sum(len(gts["A"][u] & matched_set) for u in cohorts["A"])
    out(f"Новых id в val: {len(new_shows):,}")
    out(f"С базовым названием, совпадающим с train: {len(matched_new):,} ({len(matched_new)/len(new_shows)*100:.1f}%)")
    out(f"Доля GT-A на новых, покрытая совпадением: {gt_new_matched/total_gt_new*100:.2f}%")
    out()
    out("| новый id | полное название | совпавший train id |")
    out("|---|---|---|")
    shown = 0
    for s in new_shows:
        if s in matched_set:
            train_match = train_base[base_of[s]][0]
            out(f"| {s} | {title_map.get(s, '?')[:60]} | {train_match} |")
            shown += 1
            if shown >= 10:
                break
    out()

    # ---- MF for fusion ---------------------------------------------------
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
    for u in cohorts["A"]:
        ri = user_index.get(int(u))
        if ri is None:
            mf_rankings[u] = list(global_top5)
        else:
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

    # ---- CBF blocks ------------------------------------------------------
    all_shows = sorted(train_items | candidates)
    t_cbf = time.time()
    show_index, cat_mat, genre_mat, year_mat, people_mat = CBF.build_blocks(all_shows, meta)
    base_onehot, title_tfidf = CBF.build_title_blocks(all_shows, title_map)
    t_cbf = time.time() - t_cbf
    user_items = defaultdict(dict)
    for (u, s), nd in n_days.items():
        user_items[u][s] = nd

    V_cg = combine([(1.0, cat_mat), (1.0, genre_mat)])
    prof_cg = CBF.build_profiles(user_items, show_index, V_cg)
    cbf_cg_rankings, _ = CBF.recommend_cbf(prof_cg, V_cg, show_index, candidates, global_rank, cohorts["A"])

    # ---- personal-series -------------------------------------------------
    series_days = defaultdict(lambda: defaultdict(set))
    for (u, s), ds in days.items():
        series_days[u][base_of[s]].update(ds)
    series_ndays = {u: {b: len(v) for b, v in vv.items()} for u, vv in series_days.items()}

    def personal_series_rankings(cohort):
        rankings = {}
        for u in cohort:
            us = series_ndays.get(u, {})
            scored = [(us[base_of[s]], s) for s in candidates if base_of[s] in us]
            scored.sort(key=lambda x: (-x[0], x[1]))
            top = [s for _, s in scored[:5]]
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

    ps_rankings = personal_series_rankings(cohorts["A"])

    # ---- CBF-title variants ---------------------------------------------
    title_variants = {
        "title": combine([(10.0, base_onehot), (1.0, title_tfidf)]),
        "title+cat+genre": combine([(10.0, base_onehot), (1.0, title_tfidf), (1.0, cat_mat), (1.0, genre_mat)]),
    }

    out("## 2. CBF-title")
    out()
    out("| вариант | MAP@5 A | MAP@5 B | MAP@5 C | покрытие | доля повторов | hits на новых | Recall@5 новых |")
    out("|---|---|---|---|---|---|---|---|")
    cbf_title_rank = {}
    for name, V in title_variants.items():
        prof = CBF.build_profiles(user_items, show_index, V)
        rank, n_pad = CBF.recommend_cbf(prof, V, show_index, candidates, global_rank, cohorts["A"])
        ra = E.evaluate(rank, gts["A"], cohorts["A"])
        rb = E.evaluate(rank, gts["B"], cohorts["B"])
        rc = E.evaluate(rank, gts["C"], cohorts["C"])
        rf = repeat_fraction(rank, user_train_items, cohorts["A"])
        nh, nr, nrec = new_show_metrics(rank, new_shows, gts, cohorts["A"])
        cbf_title_rank[name] = (V, prof, rank, ra, rb, rc)
        out(f"| {name} | {ra['map5']:.6f} | {rb['map5']:.6f} | {rc['map5']:.6f} | {ra['coverage']:,} | {rf*100:.1f}% | {nh} | {nr:.6f} |")
    out()

    best_cbft = max(cbf_title_rank, key=lambda k: cbf_title_rank[k][3]["map5"])
    Vt, prof_t, cbft_rankings, cbft_ra, cbft_rb, cbft_rc = cbf_title_rank[best_cbft]
    out(f"Лучший CBF-title: {best_cbft} (MAP@5 A = {cbft_ra['map5']:.6f})")
    out()

    # ---- task 3: personal-series metrics ---------------------------------
    ps_ra = E.evaluate(ps_rankings, gts["A"], cohorts["A"])
    ps_rb = E.evaluate(ps_rankings, gts["B"], cohorts["B"])
    ps_rc = E.evaluate(ps_rankings, gts["C"], cohorts["C"])
    out(f"Personal-series: MAP@5 A/B/C = {ps_ra['map5']:.6f}/{ps_rb['map5']:.6f}/{ps_rc['map5']:.6f}, покрытие {ps_ra['coverage']:,}")
    out()

    # ---- build rank maps for fusion -------------------------------------
    def sparse_rank_map(cohort, score_lookup, top_n=200):
        rm = {}
        for u in cohort:
            sc = score_lookup.get(u, {})
            items = sorted([(v, s) for s, v in sc.items() if s in candidates and v > 0], key=lambda x: (-x[0], x[1]))
            rm[u] = {s: r for r, (_v, s) in enumerate(items[:top_n], 1)}
        return rm

    personal_scores = defaultdict(dict)
    for (u, s), nd in n_days.items():
        personal_scores[u][s] = nd
    rm_personal = sparse_rank_map(cohorts["A"], personal_scores)

    ps_scores = defaultdict(dict)
    for u in cohorts["A"]:
        us = series_ndays.get(u, {})
        for s in candidates:
            if base_of[s] in us:
                ps_scores[u][s] = us[base_of[s]]
    rm_ps = sparse_rank_map(cohorts["A"], ps_scores)

    # MF+pop (beta=1)
    rm_mfpop = {}
    for u in cohorts["A"]:
        ri = user_index.get(int(u))
        if ri is None:
            rm_mfpop[u] = {}
            continue
        s = X[ri] @ Yc.T
        mu = s.mean()
        sd = s.std()
        z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
        comb = z + 1.0 * pop_log_mf
        idx = np.argpartition(-comb, min(200, len(comb) - 1))[:200]
        idx = idx[np.argsort(-comb[idx])]
        rm_mfpop[u] = {cand_mf[i]: r for r, i in enumerate(idx, 1)}

    # CBF-title
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

    # ---- task 4: rank fusion --------------------------------------------
    methods = [rm_personal, rm_ps, rm_mfpop, rm_cbft]
    method_names = ["personal", "personal-series", "MF+pop", "CBF-title"]

    out("## 4. Гибрид (rank fusion: sum w_m / (60 + rank_m))")
    out()

    union_shows = {}
    inv_rank = {}
    for u in cohorts["A"]:
        show_set = {}
        for mi in range(4):
            for s, r in methods[mi].get(u, {}).items():
                show_set.setdefault(s, [0.0, 0.0, 0.0, 0.0])
                show_set[s][mi] = 1.0 / (60.0 + r)
        shows = list(show_set.keys())
        if not shows:
            union_shows[u] = list(global_top5)
            inv_rank[u] = np.zeros((5, 4))
        else:
            union_shows[u] = shows
            inv_rank[u] = np.array([show_set[s] for s in shows])

    vals = [0.0, 0.5, 1.0, 2.0]
    W = np.array([[a, b, c, d] for a in vals for b in vals for c in vals for d in vals])  # 256 x 4
    aps = np.zeros(256)
    t_fuse = time.time()
    for u in cohorts["A"]:
        S = inv_rank[u] @ W.T  # n_union x 256
        order = np.argsort(-S, axis=0)[:5]  # 5 x 256
        rel = np.array([1 if s in gts["A"][u] else 0 for s in union_shows[u]], dtype=np.float64)
        r_sel = rel[order]
        hits = np.cumsum(r_sel, axis=0)
        prec = hits / np.arange(1, 6)[:, None]
        denom = min(5, len(gts["A"][u]))
        aps += (prec * r_sel).sum(axis=0) / denom
    map5_grid = aps / len(cohorts["A"])
    t_fuse = time.time() - t_fuse
    best_ci = int(np.argmax(map5_grid))
    best_w = W[best_ci]
    out(f"Лучшие веса (GT-A): personal={best_w[0]}, personal-series={best_w[1]}, MF+pop={best_w[2]}, CBF-title={best_w[3]}")
    out(f"MAP@5 A = {map5_grid[best_ci]:.6f} (fusion eval {t_fuse:.1f}s)")

    # rebuild rankings for best combo
    hybrid_rankings = {}
    for u in cohorts["A"]:
        S = inv_rank[u] @ best_w
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
    out(f"Гибрид: MAP@5 A/B/C = {h_ra['map5']:.6f}/{h_rb['map5']:.6f}/{h_rc['map5']:.6f}")

    # switching
    mfpop_top5 = {}
    for u in cohorts["A"]:
        items = list(rm_mfpop.get(u, {}).items())
        items.sort(key=lambda x: x[1])
        top = [s for s, _r in items[:5]]
        if len(top) < 5:
            seen = set(top)
            for gs in global_top5:
                if len(top) >= 5:
                    break
                if gs not in seen:
                    top.append(gs)
                    seen.add(gs)
        mfpop_top5[u] = top
    switch_rankings = {}
    for u in cohorts["A"]:
        if len(user_train_items.get(u, set())) < 5:
            switch_rankings[u] = mfpop_top5[u]
        else:
            switch_rankings[u] = hybrid_rankings[u]
    sw_ra = E.evaluate(switch_rankings, gts["A"], cohorts["A"])
    out(f"Switching (<5 train → MF+pop): MAP@5 A = {sw_ra['map5']:.6f}")
    out()

    # ---- task 5: final table --------------------------------------------
    personal_rankings = build_personal_rankings(n_days, candidates, global_rank, cohorts["A"])
    gr = {u: list(global_top5) for u in cohorts["A"]}

    final = []
    def row(name, rank, time_s):
        ra = E.evaluate(rank, gts["A"], cohorts["A"])
        rb = E.evaluate(rank, gts["B"], cohorts["B"])
        rc = E.evaluate(rank, gts["C"], cohorts["C"])
        return [name, ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"], time_s]

    final.append(row("global", gr, 0.0))
    final.append(row("personal+global(n_days)", personal_rankings, 0.0))
    final.append(row("personal-series", ps_rankings, 0.0))
    final.append(row("MF(32,50,0.5)", mf_rankings, t_mf))
    final.append(row("MF+pop(beta=1)", mfpop_top5, 0.0))
    final.append(row("CBF(cat+genre)", cbf_cg_rankings, t_cbf))
    final.append(row(f"CBF-title({best_cbft})", cbft_rankings, t_cbf))
    final.append(row("hybrid", hybrid_rankings, t_fuse))

    out("## 5. Итоговая таблица")
    out()
    out("| метод | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие | время, с |")
    out("|---|---|---|---|---|---|---|")
    for r in final:
        out(f"| {r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]:,} | {r[6]:.1f} |")
    out()
    out(f"Гибрид: веса {dict(zip(method_names, best_w.tolist()))}, прирост над лучшим одиночным (personal 0.7726): {h_ra['map5']-0.772566:.6f}")
    out()

    pd.DataFrame(final, columns=["method", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage", "time_s"]).to_csv(
        config.RESULTS_DIR / "all_methods.csv", index=False
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "cbf_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "cbf_summary.md")


if __name__ == "__main__":
    main()
