"""Stage 9a: diagnostics without Kaggle (GT strictness, season change, aging)."""

import csv
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
import evaluate as E
import mf_als as MF
import cbf as CBF
from join import parse_grid_time, to_unix

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


def week_of(ts):
    return datetime.utcfromtimestamp(int(ts)).isocalendar()[1]


def main() -> None:
    out("# Диагностика без Kaggle (этап 9a)")
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

    # ---- MF ------------------------------------------------------------
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)
    cand_mf = sorted(c for c in val_cands if c in item_index)
    cand_mf_idx = np.array([item_index[c] for c in cand_mf], dtype=np.int64)
    Yc = Y[cand_mf_idx]
    pop_log = np.array([np.log1p(pop_count.get(c, 0)) for c in cand_mf])

    # ---- hybrid / global / personal (final config) ---------------------
    hybrid_rankings = {}
    personal_rankings = {}
    for u in sub_int:
        sc = {s: n_days[(u, s)] for s in val_cands if (u, s) in n_days}
        items = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        personal_rankings[u] = pad5([s for s, _v in items[:5]], global_top5)
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
    gr = {u: list(global_top5) for u in sub_int}

    # ================= Task 1: GT strictness ============================
    out("## 1. Калибровка строгости GT")
    out()
    ova = val_airings["overlap"].to_numpy()
    dua = val_airings["duration"].to_numpy()
    ua = val_airings["user_id"].to_numpy()
    sa = val_airings["tv_show_id"].to_numpy()
    vsa = val_airings["view_start"].to_numpy()

    agg = defaultdict(lambda: dict(a=False, b=False, e80=False, days=set(), total=0.0, nb=0))
    for ui, si, oi, di, vi in zip(ua, sa, ova, dua, vsa):
        a = agg[(int(ui), int(si))]
        if oi >= 300 or oi >= 0.5 * di:
            a["a"] = True
            a["days"].add(int(vi) // 86400)
        if oi >= 0.5 * di:
            a["b"] = True
            a["nb"] += 1
        if oi >= 0.8 * di:
            a["e80"] = True
        a["total"] += float(oi)

    def make_gt(pred):
        gt = {}
        for (ui, si), a in agg.items():
            if pred(a):
                gt.setdefault(ui, set()).add(si)
        return gt

    defs = [
        ("GT-A", lambda a: a["a"]),
        ("GT-B", lambda a: a["b"]),
        ("GT-C", lambda a: len(a["days"]) >= 2),
        (">=3 дня", lambda a: len(a["days"]) >= 3),
        (">=5 дней", lambda a: len(a["days"]) >= 5),
        (">=1 ч", lambda a: a["total"] >= 3600),
        (">=2 эфира 50%", lambda a: a["nb"] >= 2),
        (">=80% длины", lambda a: a["e80"]),
    ]

    out("| определение | avg |R_u| | hybrid | global | personal | hybrid(пустые=0) |")
    out("|---|---|---|---|---|---|---|")
    for name, pred in defs:
        gt = make_gt(pred)
        cohort = [u for u in sub_int if u in gt]
        rel = [len(gt[u]) for u in cohort]
        avg_rel = float(np.mean(rel)) if rel else 0.0
        mh = E.evaluate(hybrid_rankings, gt, cohort)["map5"]
        mg = E.evaluate(gr, gt, cohort)["map5"]
        mp = E.evaluate(personal_rankings, gt, cohort)["map5"]
        # empty GT -> AP=0 over all 1260
        aps = []
        for u in sub_int:
            r = gt.get(u, set())
            aps.append(E.ap_at_k(hybrid_rankings[u], r) if r else 0.0)
        mh_empty = float(np.mean(aps))
        out(f"| {name} | {avg_rel:.1f} | {mh:.4f} | {mg:.4f} | {mp:.4f} | {mh_empty:.4f} |")
    out()

    # ================= Task 2: season change ============================
    out("## 2. Смена сезона (новые id по ISO-неделям)")
    out()
    train_ids = set()
    for fp in (config.GRID_11_20_FILE, config.GRID_21_30_FILE):
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                sid = int(row[5])
                if sid != 0:
                    train_ids.add(sid)

    week_total = defaultdict(float)
    week_new = defaultdict(float)
    week_newids = defaultdict(set)
    for fp in (config.GRID_21_30_FILE, config.GRID_31_42_FILE):
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                sid = int(row[5])
                if sid == 0:
                    continue
                st = to_unix(parse_grid_time(row[2]))
                d = int(float(row[3]))
                w = week_of(st)
                week_total[w] += d
                if sid not in train_ids:
                    week_new[w] += d
                    week_newids[w].add(sid)

    out("| неделя | новых id | доля времени на новых, % |")
    out("|---|---|---|")
    for w in sorted(week_total):
        share = week_new[w] / week_total[w] * 100 if week_total[w] > 0 else 0.0
        out(f"| {w} | {len(week_newids[w])} | {share:.1f} |")
    out()

    # ================= Task 3: aging ====================================
    out("## 3. Устаревание (MAP@5 GT-B по неделям val)")
    out()
    # per-week candidates
    week_cands = defaultdict(set)
    with open(config.GRID_21_30_FILE, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            sid = int(row[5])
            if sid == 0:
                continue
            st = to_unix(parse_grid_time(row[2]))
            week_cands[week_of(st)].add(sid)

    # per-week GT-B
    week_gt = defaultdict(lambda: defaultdict(set))
    bmask = ova >= 0.5 * dua
    for ui, si, vi, m in zip(ua, sa, vsa, bmask):
        if m:
            week_gt[week_of(vi)][int(ui)].add(int(si))

    out("| неделя | когорта | hybrid | global | personal |")
    out("|---|---|---|---|---|")
    for w in range(21, 31):
        gt_w = dict(week_gt.get(w, {}))
        cohort_w = [u for u in sub_int if u in gt_w]
        cw = week_cands.get(w, set())
        cw_mf = sorted(s for s in cw if s in item_index)
        cw_mf_idx = np.array([item_index[s] for s in cw_mf], dtype=np.int64)
        Ycw = Y[cw_mf_idx]
        pop_log_w = np.array([np.log1p(pop_count.get(s, 0)) for s in cw_mf])
        grank_w = [s for s in global_rank if s in cw][:5]
        gr_w = {u: list(grank_w) for u in cohort_w}
        # personal + mfpop + hybrid per week
        pers_w = {}
        hyb_w = {}
        for u in cohort_w:
            sc = {s: n_days[(u, s)] for s in cw if (u, s) in n_days}
            items = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
            org = [s for s, _v in items[:5]]
            pers_w[u] = pad5(org, grank_w)
            rm_p = {s: r for r, (s, _v) in enumerate(items[:200], 1)}
            ri = user_index.get(int(u))
            rm_mfp = {}
            if ri is not None:
                s = X[ri] @ Ycw.T
                mu = s.mean()
                sd = s.std()
                z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
                comb = z + 1.0 * pop_log_w
                rm_mfp = rank_map(comb, cw_mf, top_n=200)
            show_set = {}
            for s, r in rm_p.items():
                show_set.setdefault(s, [0.0, 0.0])
                show_set[s][0] = 1.0 / (60.0 + r)
            for s, r in rm_mfp.items():
                show_set.setdefault(s, [0.0, 0.0])
                show_set[s][1] = 1.0 / (60.0 + r)
            hs = [(show_set[s][0] * 1.0 + show_set[s][1] * 0.5, s) for s in show_set]
            hs.sort(key=lambda x: (-x[0], x[1]))
            hyb_w[u] = pad5([s for _sc, s in hs[:5]], grank_w)
        mh = E.evaluate(hyb_w, gt_w, cohort_w)["map5"]
        mg = E.evaluate(gr_w, gt_w, cohort_w)["map5"]
        mp = E.evaluate(pers_w, gt_w, cohort_w)["map5"]
        out(f"| {w} | {len(cohort_w)} | {mh:.4f} | {mg:.4f} | {mp:.4f} |")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "diagnostics_9a.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "diagnostics_9a.md")


if __name__ == "__main__":
    main()
