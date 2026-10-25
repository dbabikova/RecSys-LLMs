"""Stage 8: two example users for the report (val protocol, final config)."""

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
csv_rows: list[list] = []


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
    out("# Два пользователя для отчёта (этап 8)")
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

    meta, val_cands, title_map, _ = CBF.load_show_metadata()
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]

    def cat(s):
        return meta.get(s, ("",))[0] if meta.get(s) is not None else ""

    def title(s):
        return title_map.get(s, "?")

    # ---- MF ------------------------------------------------------------
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)
    cand_mf = sorted(c for c in val_cands if c in item_index)
    cand_mf_idx = np.array([item_index[c] for c in cand_mf], dtype=np.int64)
    Yc = Y[cand_mf_idx]
    pop_log = np.array([np.log1p(pop_count.get(c, 0)) for c in cand_mf])

    # ---- CBF blocks ----------------------------------------------------
    all_shows = sorted(train_items | val_cands)
    show_index, cat_mat, genre_mat, year_mat, people_mat = CBF.build_blocks(all_shows, meta)
    base_onehot, title_tfidf = CBF.build_title_blocks(all_shows, title_map)
    user_items = defaultdict(dict)
    for (u, s), nd in n_days.items():
        user_items[u][s] = nd
    Vt = combine([(10.0, base_onehot), (1.0, title_tfidf)])
    prof_t = CBF.build_profiles(user_items, show_index, Vt)
    cand_cbf = sorted(c for c in val_cands if c in show_index)
    cand_cbf_idx = np.array([show_index[c] for c in cand_cbf], dtype=np.int64)
    Vtc = Vt[cand_cbf_idx]

    # ---- build rankings (organic + padded) for cohort ------------------
    methods = {}  # name -> {u: (organic_list, final_top5)}
    for u in cohorts["A"]:
        gt_u = gts["A"][u]
        # personal
        sc = {s: n_days[(u, s)] for s in val_cands if (u, s) in n_days}
        items = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        org_p = [s for s, _v in items[:5]]
        rm_p = {s: r for r, (s, _v) in enumerate(items[:200], 1)}

        # mf / mf+pop
        ri = user_index.get(int(u))
        if ri is None:
            org_mf, org_mfp = [], []
            rm_mfp = {}
        else:
            s = X[ri] @ Yc.T
            org_mf = topk(s, cand_mf, k=5)
            mu = s.mean()
            sd = s.std()
            z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
            comb = z + 1.0 * pop_log
            org_mfp = topk(comb, cand_mf, k=5)
            rm_mfp = rank_map(comb, cand_mf, top_n=200)

        # cbf-title
        p = prof_t.get(int(u))
        if p is None or float(np.linalg.norm(p)) == 0.0:
            org_cb = []
        else:
            pn = p / np.linalg.norm(p)
            scb = pn @ Vtc.T
            org_cb = topk(scb, cand_cbf, k=5)

        # hybrid (personal 1.0 + mfpop 0.5)
        show_set = {}
        for s, r in rm_p.items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][0] = 1.0 / (60.0 + r)
        for s, r in rm_mfp.items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][1] = 1.0 / (60.0 + r)
        hscored = [(show_set[s][0] * 1.0 + show_set[s][1] * 0.5, s) for s in show_set]
        hscored.sort(key=lambda x: (-x[0], x[1]))
        org_hy = [s for _sc, s in hscored[:5]]

        methods.setdefault("global", {})[u] = (list(global_top5), list(global_top5))
        methods.setdefault("personal", {})[u] = (org_p, pad5(org_p, global_top5))
        methods.setdefault("MF", {})[u] = (org_mf, pad5(org_mf, global_top5))
        methods.setdefault("MF+pop", {})[u] = (org_mfp, pad5(org_mfp, global_top5))
        methods.setdefault("CBF-title", {})[u] = (org_cb, pad5(org_cb, global_top5))
        methods.setdefault("hybrid", {})[u] = (org_hy, pad5(org_hy, global_top5))

    # ---- select two users ---------------------------------------------
    pair_count = {u: len(user_train_items[u]) for u in cohorts["A"]}
    counts = sorted(pair_count.values())
    p90 = counts[int(np.ceil(0.9 * len(counts))) - 1]
    heavy_candidates = [u for u in cohorts["A"] if pair_count[u] >= p90]
    heavy = min(heavy_candidates, key=lambda u: pair_count[u])  # smallest count in top decile
    light_candidates = [u for u in cohorts["A"] if 1 <= pair_count[u] <= 5]
    light = min(light_candidates, key=lambda u: abs(pair_count[u] - 3)) if light_candidates else min(cohorts["A"], key=pair_count.__getitem__)

    out("## Выбор пользователей")
    out(f"(a) Тяжёлый зритель: user_id={heavy}, пар в train={pair_count[heavy]} (верхний дециль, P90={p90})")
    out(f"(b) Лёгкий зритель: user_id={light}, пар в train={pair_count[light]}")
    out()

    method_names = ["global", "personal", "MF", "MF+pop", "CBF-title", "hybrid"]

    for role, u in (("heavy", heavy), ("light", light)):
        out(f"## Пользователь ({role}): {u}")
        out(f"|R_u| (передач в val): {len(gts['A'][u])}")
        out()
        out("### Топ-5 train по n_days")
        out("| # | tv_show_id | название | категория | n_days |")
        out("|---|---|---|---|---|")
        tr = sorted([(n_days[(u, s)], s) for s in user_train_items[u]], key=lambda x: (-x[0], x[1]))
        for i, (nd, s) in enumerate(tr[:5], 1):
            out(f"| {i} | {s} | {title(s)[:50]} | {cat(s)} | {nd} |")
        out()
        out("### Top-5 методов")
        out("| метод | # | tv_show_id | название | категория | hit | тип | dobivka | AP@5 |")
        out("|---|---|---|---|---|---|---|---|---|")
        for m in method_names:
            org, fin = methods[m][u]
            ap5 = E.ap_at_k(fin, gts["A"][u])
            for i, s in enumerate(fin[:5], 1):
                ishit = "да" if s in gts["A"][u] else "нет"
                typ = "повтор" if s in user_train_items[u] else "новое"
                pad = "да" if i > len(org) else ""
                out(f"| {m} | {i} | {s} | {title(s)[:40]} | {cat(s)} | {ishit} | {typ} | {pad} | {ap5:.4f} |")
                csv_rows.append([u, role, m, i, s, title(s), cat(s), ishit, typ, pad, ap5])
        out()

    # ---- save ----------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows, columns=["user_id", "role", "method", "pos", "tv_show_id", "title", "category", "hit_GT_A", "repeat_train", "padded", "ap5"]).to_csv(
        config.RESULTS_DIR / "two_users.csv", index=False
    )
    (config.RESULTS_DIR / "two_users.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "two_users.md", "и", config.RESULTS_DIR / "two_users.csv")


if __name__ == "__main__":
    main()
