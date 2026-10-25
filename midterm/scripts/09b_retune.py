"""Stage 9b (v2): tune ONLY the personal-history ranking under narrow per-week GT-B."""

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
    out("# Переподбор личной истории (этап 9b v2)")
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

    info_days = defaultdict(set)
    info_weeks = defaultdict(set)
    last_seen = defaultdict(int)
    for u, s, vs in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"]):
        k = (int(u), int(s))
        v = int(vs)
        info_days[k].add(v // 86400)
        info_weeks[k].add(week_of(v))
        if v > last_seen.get(k, 0):
            last_seen[k] = v
    n_days = {k: len(v) for k, v in info_days.items()}
    n_weeks = {k: len(v) for k, v in info_weeks.items()}
    VAL_DAY = E.VAL_START // 86400
    n_recent = {}
    for K in (3, 5):
        thr = VAL_DAY - K * 7
        n_recent[K] = {k: sum(1 for d in v if d >= thr) for k, v in info_days.items()}

    pop = defaultdict(set)
    for (u, s) in n_days:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}

    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)
    user_train_days = defaultdict(set)
    for (u, s), ds in info_days.items():
        user_train_days[u].update(ds)
    train_shows = {u: len(user_train_items.get(u, ())) for u in sub_int}
    train_days = {u: len(user_train_days.get(u, ())) for u in sub_int}
    switched_users = {u for u in sub_int if train_shows[u] < 5 or train_days[u] < 2}

    meta, val_cands, title_map, val_air_counts = CBF.load_show_metadata()
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))

    # ---- MF ------------------------------------------------------------
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)

    # ---- per-week candidates + airing counts + GT-B --------------------
    week_cands = defaultdict(set)
    week_air = defaultdict(lambda: defaultdict(int))
    with open(config.GRID_21_30_FILE, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            sid = int(row[5])
            if sid == 0:
                continue
            st = to_unix(parse_grid_time(row[2]))
            d = int(float(row[3]))
            w = week_of(st)
            week_cands[w].add(sid)
            week_air[w][sid] += d

    ova = val_airings["overlap"].to_numpy()
    dua = val_airings["duration"].to_numpy()
    ua = val_airings["user_id"].to_numpy()
    sa = val_airings["tv_show_id"].to_numpy()
    vsa = val_airings["view_start"].to_numpy()
    bmask = ova >= 0.5 * dua

    gtB = {}
    for ui, si, m in zip(ua, sa, bmask):
        if m:
            gtB.setdefault(int(ui), set()).add(int(si))
    cohortB = [u for u in sub_int if u in gtB]

    week_gtB = defaultdict(lambda: defaultdict(set))
    for ui, si, vi, m in zip(ua, sa, vsa, bmask):
        if m:
            week_gtB[week_of(vi)][int(ui)].add(int(si))

    # ---- candidate sets -------------------------------------------------
    sets = {}
    cand_mf = sorted(c for c in val_cands if c in item_index)
    sets["full"] = dict(cand=cand_mf, C=val_cands, n_air=val_air_counts,
                        gtop5=[s for s in global_rank if s in val_cands][:5], gt=gtB, cohort=cohortB)
    for w in range(21, 31):
        cw = week_cands.get(w, set())
        sets[f"w{w}"] = dict(cand=sorted(c for c in cw if c in item_index), C=cw,
                             n_air={s: c for s, c in week_air.get(w, {}).items()},
                             gtop5=[s for s in global_rank if s in cw][:5],
                             gt=dict(week_gtB.get(w, {})),
                             cohort=[u for u in sub_int if u in week_gtB.get(w, {})])

    # ---- mfpop rank maps per set --------------------------------------
    mfpop_rm = {}
    for sid, S in sets.items():
        cand = S["cand"]
        idx = np.array([item_index[c] for c in cand], dtype=np.int64)
        Yc = Y[idx]
        pop_log = np.array([np.log1p(pop_count.get(c, 0)) for c in cand])
        rm = {}
        for u in sub_int:
            ri = user_index.get(int(u))
            if ri is None:
                rm[u] = {}
                continue
            s = X[ri] @ Yc.T
            mu = s.mean()
            sd = s.std()
            z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
            comb = z + 1.0 * pop_log
            rm[u] = rank_map(comb, cand, top_n=200)
        mfpop_rm[sid] = rm

    # ---- personal variants ---------------------------------------------
    base_variants = ["a", "b2", "b4", "b8", "c3", "c5", "d"]

    def base_score(u, s, var):
        nd = n_days.get((u, s), 0)
        if var == "a":
            return float(nd)
        if var.startswith("b"):
            h = int(var[1:])
            dec = 0.5 ** ((E.VAL_START - last_seen.get((u, s), E.VAL_START)) / (h * 7 * 86400))
            return float(nd) * dec
        if var.startswith("c"):
            K = int(var[1:])
            return float(n_recent[K].get((u, s), 0))
        if var == "d":
            return float(n_weeks.get((u, s), 0))
        raise ValueError(var)

    # sorted items for base variants (window-independent)
    sorted_items = {}
    for var in base_variants:
        d = {}
        for u in sub_int:
            items = list(user_train_items.get(u, ()))
            if var == "a":
                items.sort(key=lambda s: (-n_days[(u, s)], s))
            elif var.startswith("b"):
                h = int(var[1:])
                items.sort(key=lambda s: (-base_score(u, s, var), s))
            elif var.startswith("c"):
                K = int(var[1:])
                items.sort(key=lambda s: (-n_recent[K].get((u, s), 0), -n_days[(u, s)], s))
            else:  # d
                items.sort(key=lambda s: (-n_weeks.get((u, s), 0), -n_days[(u, s)], s))
            d[u] = items
        sorted_items[var] = d

    def personal_rm(u, C, var, n_air):
        if isinstance(var, str):  # base variant a-d
            rm = {}
            for r, s in enumerate(sorted_items[var][u], 1):
                if s in C:
                    rm[s] = r
                    if len(rm) >= 200:
                        break
            return rm
        else:  # ("e", gamma, base_var)
            _tag, gamma, base = var
            scored = []
            for s in user_train_items.get(u, ()):
                if s in C:
                    sc = base_score(u, s, base) * (1.0 + gamma * np.log1p(n_air.get(s, 0)))
                    scored.append((sc, s))
            scored.sort(key=lambda x: (-x[0], x[1]))
            return {s: r for r, (_sc, s) in enumerate(scored[:200], 1)}

    # ---- evaluation helper --------------------------------------------
    def eval_config(var, w_mf):
        # weekly
        weekly = []
        for w in range(21, 31):
            S = sets[f"w{w}"]
            C = S["C"]
            n_air = S["n_air"]
            gtop = S["gtop5"]
            gt_w = S["gt"]
            mrm = mfpop_rm[f"w{w}"]
            aps = []
            for u in S["cohort"]:
                if u in switched_users:
                    aps.append(E.ap_at_k(list(gtop), gt_w[u]))
                    continue
                show_set = {}
                for s, r in personal_rm(u, C, var, n_air).items():
                    show_set[s] = 1.0 / (60.0 + r)
                for s, r in mrm[u].items():
                    show_set[s] = show_set.get(s, 0.0) + w_mf / (60.0 + r)
                scored = [(show_set[s], s) for s in show_set]
                scored.sort(key=lambda x: (-x[0], x[1]))
                pred = pad5([s for _sc, s in scored[:5]], gtop)
                aps.append(E.ap_at_k(pred, gt_w[u]))
            weekly.append(float(np.mean(aps)))
        weekly_map5 = float(np.mean(weekly))
        # full
        S = sets["full"]
        C = S["C"]
        n_air = S["n_air"]
        gtop = S["gtop5"]
        mrm = mfpop_rm["full"]
        aps = []
        for u in cohortB:
            if u in switched_users:
                aps.append(E.ap_at_k(list(gtop), gtB[u]))
                continue
            show_set = {}
            for s, r in personal_rm(u, C, var, n_air).items():
                show_set[s] = 1.0 / (60.0 + r)
            for s, r in mrm[u].items():
                show_set[s] = show_set.get(s, 0.0) + w_mf / (60.0 + r)
            scored = [(show_set[s], s) for s in show_set]
            scored.sort(key=lambda x: (-x[0], x[1]))
            pred = pad5([s for _sc, s in scored[:5]], gtop)
            aps.append(E.ap_at_k(pred, gtB[u]))
        full_b = float(np.mean(aps))
        return weekly_map5, full_b

    # ---- evaluate base variants first ---------------------------------
    out("## Base-варианты (a-d) × w_mf")
    out()
    base_rows = []
    for var in base_variants:
        for wm in (0.0, 0.25, 0.5):
            wk, fb = eval_config(var, wm)
            base_rows.append((var, wm, wk, fb))
    best_base_var = max(base_variants, key=lambda v: max(r[2] for r in base_rows if r[0] == v))

    # e variants
    e_rows = []
    for gamma in (0.3, 1.0):
        evar = ("e", gamma, best_base_var)
        for wm in (0.0, 0.25, 0.5):
            wk, fb = eval_config(evar, wm)
            e_rows.append((f"e{gamma}({best_base_var})", wm, wk, fb))

    all_rows = base_rows + e_rows
    all_rows_sorted = sorted(all_rows, key=lambda r: -r[2])

    out("| variant | w_mf | недельный MAP@5 | GT-B весь val |")
    out("|---|---|---|---|")
    for var, wm, wk, fb in all_rows_sorted:
        out(f"| {var} | {wm} | {wk:.4f} | {fb:.4f} |")
    out()

    current = [r for r in base_rows if r[0] == "a" and r[1] == 0.5][0]
    best = all_rows_sorted[0]
    out(f"Текущий финал (a, w_mf=0.5): недельный {current[2]:.4f}, GT-B {current[3]:.4f}")
    out(f"Лучший: {best[0]}, w_mf={best[1]}: недельный {best[2]:.4f}, GT-B {best[3]:.4f}")
    out()
    win = best[2] >= current[2] + 0.005 and best[3] >= current[3]
    out(f"Выигрывает (недельный +0.005 и не хуже GT-B): {'да' if win else 'нет'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "diagnostics_9b.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "diagnostics_9b.md")


if __name__ == "__main__":
    main()
