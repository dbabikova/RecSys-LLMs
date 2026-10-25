"""Stage 9e: Personalized PageRank as a third method."""

import csv
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
import evaluate as E
import mf_als as MF
import cbf as CBF
from join import parse_log_time, parse_grid_time, to_unix

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REF = E.VAL_START
md: list[str] = []


def out(*args) -> None:
    line = " ".join(str(a) for a in args)
    print(line)
    md.append(line)


def week_of(ts):
    return datetime.utcfromtimestamp(int(ts)).isocalendar()[1]


def hour_of_week(ts):
    d = datetime.utcfromtimestamp(int(ts))
    return d.weekday() * 24 + d.hour


def pad5(top, gtop5):
    top = list(top)
    seen = set(top)
    for gs in gtop5:
        if len(top) >= 5:
            break
        if gs not in seen:
            top.append(int(gs))
            seen.add(int(gs))
    return top[:5]


def compute_ppr(csr, cohort_users, user_index, alpha, max_iter=300):
    n_users = csr.shape[0]
    n_items = csr.shape[1]
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    U = sp.diags(1.0 / np.maximum(row_sums, 1e-12)) @ csr  # user -> show, row-normalized
    col_sums = np.asarray(csr.sum(axis=0)).ravel()
    S = (csr @ sp.diags(1.0 / np.maximum(col_sums, 1e-12))).tocsr().T  # show -> user
    valid = [u for u in cohort_users if u in user_index]
    rows = np.array([user_index[u] for u in valid])
    n_seed = len(valid)
    E = np.zeros((n_users, n_seed))
    E[rows, np.arange(n_seed)] = 1.0
    Pu = E.copy()
    Ps = np.zeros((n_items, n_seed))
    n_iter = 0
    for it in range(max_iter):
        Ps_new = alpha * (U.T @ Pu)
        Pu_new = (1.0 - alpha) * E + alpha * (S.T @ Ps)
        res = float(np.abs(Ps_new - Ps).sum()) + float(np.abs(Pu_new - Pu).sum())
        Pu, Ps = Pu_new, Ps_new
        n_iter = it + 1
        if res < 1e-6:
            break
    return Ps, valid, n_iter


def main() -> None:
    out("# Personalized PageRank (этап 9e)")
    out()

    sub_int = sorted(int(u) for u in E.load_submission_ids())

    print("Loading labeled matches ...", flush=True)
    airings = E.load_labeled_airings(E.CAP)
    ov = airings["overlap"].to_numpy()
    dur = airings["duration"].to_numpy()
    vst = airings["view_start"].to_numpy()
    train_mask = vst < E.VAL_START
    val_mask = (vst >= E.VAL_START) & (vst <= E.VAL_END)
    watched = E.watched_mask_np(ov, dur)
    train_watched = airings[train_mask & watched]
    val_airings = airings[val_mask]

    feat = defaultdict(lambda: dict(days=set(), last=0, ov=0.0))
    for u, s, vs, o in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"], train_watched["overlap"]):
        k = (int(u), int(s))
        f = feat[k]
        f["days"].add(int(vs) // 86400)
        if int(vs) > f["last"]:
            f["last"] = int(vs)
        f["ov"] += float(o)
    n_days = {k: len(f["days"]) for k, f in feat.items()}
    train_items = set(s for (_u, s) in n_days)

    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)
    user_train_days = defaultdict(set)
    for (u, s), f in feat.items():
        user_train_days[u].update(f["days"])
    train_shows = {u: len(user_train_items.get(u, ())) for u in sub_int}
    train_days = {u: len(user_train_days.get(u, ())) for u in sub_int}

    # slot profile
    print("Loading full log for slot profile ...", flush=True)
    slot_sec = defaultdict(lambda: defaultdict(float))
    with open(config.LOG_FILE, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            u = int(row[0])
            ch = int(row[1])
            s = to_unix(parse_log_time(row[2]))
            e = to_unix(parse_log_time(row[3]))
            if s >= REF:
                continue
            d = e - s
            if d > E.CAP:
                d = E.CAP
            if d <= 0:
                continue
            slot_sec[u][(ch, hour_of_week(s))] += d
    slot_profile = {}
    for u, slots in slot_sec.items():
        tot = sum(slots.values())
        slot_profile[u] = {k: v / tot for k, v in slots.items()}

    meta, val_cands, title_map, _ = CBF.load_show_metadata()
    pop = defaultdict(set)
    for (u, s) in n_days:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))

    week_cands = defaultdict(set)
    air_slots_week = defaultdict(lambda: defaultdict(list))
    n_air_week = defaultdict(lambda: defaultdict(int))
    with open(config.GRID_21_30_FILE, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            sid = int(row[5])
            if sid == 0:
                continue
            ch = int(row[0])
            st = to_unix(parse_grid_time(row[2]))
            w = week_of(st)
            week_cands[w].add(sid)
            air_slots_week[w][sid].append((ch, hour_of_week(st)))
            n_air_week[w][sid] += 1
    full_air = defaultdict(list)
    for w, shows in air_slots_week.items():
        for s, lst in shows.items():
            full_air[s].extend(lst)
    full_n_air = {s: len(v) for s, v in full_air.items()}

    # GT-B
    ova = val_airings["overlap"].to_numpy()
    dua = val_airings["duration"].to_numpy()
    ua = val_airings["user_id"].to_numpy()
    sa = val_airings["tv_show_id"].to_numpy()
    vsa = val_airings["view_start"].to_numpy()
    bmask = ova >= 0.5 * dua
    gts = {d: E.build_ground_truth(val_airings, d) for d in ("A", "B", "C")}
    cohorts = {d: [u for u in sub_int if u in gts[d]] for d in ("A", "B", "C")}
    week_gtB = defaultdict(lambda: defaultdict(set))
    for ui, si, vi, m in zip(ua, sa, vsa, bmask):
        if m:
            week_gtB[week_of(vi)][int(ui)].add(int(si))

    # ---- PPR ------------------------------------------------------------
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    n_days_log = {(u, s): math.log1p(nd) for (u, s), nd in n_days.items()}
    csr_log, _, _, _ = MF.build_csr(n_days_log)

    cohort_ppr = cohorts["A"]

    ppr_results = {}
    out("## PPR: число итераций до residual < 1e-6")
    out("| вес | alpha | итераций |")
    out("|---|---|---|")
    for wname, wcsr in (("n_days", csr), ("log1p", csr_log)):
        for alpha in (0.15, 0.3, 0.5, 0.85):
            t0 = time.time()
            Ps, valid, n_iter = compute_ppr(wcsr, cohort_ppr, user_index, alpha)
            ppr_results[(wname, alpha)] = (Ps, valid)
            out(f"| {wname} | {alpha} | {n_iter} ({time.time()-t0:.1f}s) |")
    out()

    # ---- windows --------------------------------------------------------
    windows = {}
    windows["full"] = dict(C=val_cands, n_air=full_n_air, air_slots=dict(full_air),
                           gtop5=[s for s in global_rank if s in val_cands][:5],
                           grank=[s for s in global_rank if s in val_cands])
    for w in range(21, 31):
        cw = week_cands.get(w, set())
        windows[f"w{w}"] = dict(C=cw, n_air=dict(n_air_week.get(w, {})), air_slots=dict(air_slots_week.get(w, {})),
                                gtop5=[s for s in global_rank if s in cw][:5],
                                grank=[s for s in global_rank if s in cw])

    def ppr_mass_map(Ps, valid):
        # valid user id -> dict show_id -> mass (for candidate shows)
        m = {}
        for i, u in enumerate(valid):
            col = Ps[:, i]
            m[u] = col
        return m

    def ppr_rank(u, W, mass_map):
        C = windows[W]["C"]
        if u not in mass_map:
            return []
        col = mass_map[u]
        scored = []
        for j, sid in enumerate(item_ids):
            if sid in C and col[j] > 0:
                scored.append((col[j], sid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [s for _sc, s in scored[:5]]

    # ---- PPR standalone eval -------------------------------------------
    out("## PPR отдельно")
    out("| вес | alpha | MAP@5 A | B | C | недельный | покрытие | повторы |")
    out("|---|---|---|---|---|---|---|---|")
    ppr_best = None
    for (wname, alpha), (Ps, valid) in ppr_results.items():
        mass_map = ppr_mass_map(Ps, valid)
        # rankings for cohort A
        rankA = {}
        for u in cohorts["A"]:
            rankA[u] = pad5(ppr_rank(u, "full", mass_map), windows["full"]["gtop5"])
        ra = E.evaluate(rankA, gts["A"], cohorts["A"])
        rb = E.evaluate(rankA, gts["B"], cohorts["B"])
        rc = E.evaluate(rankA, gts["C"], cohorts["C"])
        weekly = []
        for w in range(21, 31):
            W = f"w{w}"
            gt_w = dict(week_gtB.get(w, {}))
            cohort_w = [u for u in sub_int if u in week_gtB.get(w, {})]
            aps = []
            for u in cohort_w:
                aps.append(E.ap_at_k(pad5(ppr_rank(u, W, mass_map), windows[W]["gtop5"]), gt_w[u]))
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))
        rf = sum(1 for u in cohorts["A"] for s in rankA[u][:5] if s in user_train_items.get(u, ())) / (len(cohorts["A"]) * 5)
        out(f"| {wname} | {alpha} | {ra['map5']:.4f} | {rb['map5']:.4f} | {rc['map5']:.4f} | {wk:.4f} | {ra['coverage']:,} | {rf*100:.1f}% |")
        if ppr_best is None or wk > ppr_best[0]:
            ppr_best = (wk, (wname, alpha), mass_map)
    out()

    best_wk, best_key, best_mass = ppr_best
    out(f"Лучший PPR: {best_key[0]}, alpha={best_key[1]} (недельный {best_wk:.4f})")
    out()

    # add PPR to all_methods.csv (best variant)
    rankA = {}
    for u in cohorts["A"]:
        rankA[u] = pad5(ppr_rank(u, "full", best_mass), windows["full"]["gtop5"])
    ra = E.evaluate(rankA, gts["A"], cohorts["A"])
    rb = E.evaluate(rankA, gts["B"], cohorts["B"])
    rc = E.evaluate(rankA, gts["C"], cohorts["C"])
    df = pd.read_csv(config.RESULTS_DIR / "all_methods.csv")
    df = df[df["method"] != "PPR"]
    newrow = pd.DataFrame([{"method": f"PPR({best_key[0]},a={best_key[1]})", "MAP5_A": ra["map5"], "MAP5_B": rb["map5"],
                            "MAP5_C": rc["map5"], "Recall5_A": ra["recall5"], "coverage": ra["coverage"], "time_s": 0.0}])
    df = pd.concat([df, newrow], ignore_index=True)
    df.to_csv(config.RESULTS_DIR / "all_methods.csv", index=False)

    # ---- hybrid: personal v4 + PPR via RRF -----------------------------
    out("## Гибрид personal(v4) + PPR (RRF)")
    out()
    hl = 8.0
    gamma = 0.3
    delta = 10.0

    def personal_scored(u, W):
        C = windows[W]["C"]
        n_air = windows[W]["n_air"]
        air_slots = windows[W]["air_slots"]
        prof = slot_profile.get(u, {})
        scored = []
        for s in user_train_items.get(u, ()):
            if s not in C:
                continue
            nd = n_days.get((u, s), 0)
            dec = 0.5 ** ((REF - feat.get((u, s), {"last": REF})["last"]) / (hl * 7 * 86400))
            slot = 0.0
            for (ch, hour) in air_slots.get(s, ()):
                slot += prof.get((ch, hour), 0.0)
            sc = nd * dec * (1.0 + gamma * math.log1p(n_air.get(s, 0))) * (1.0 + delta * slot)
            scored.append((sc, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    def hybrid_rank(u, W, w_ppr):
        ps = personal_scored(u, W)
        prm = {s: r for r, (_sc, s) in enumerate(ps[:200], 1)}
        prk = {}
        if u in best_mass:
            col = best_mass[u]
            C = windows[W]["C"]
            lst = [(col[j], item_ids[j]) for j in range(len(item_ids)) if item_ids[j] in C and col[j] > 0]
            lst.sort(key=lambda x: (-x[0], x[1]))
            prk = {s: r for r, (_m, s) in enumerate(lst[:200], 1)}
        show_set = {}
        for s, r in prm.items():
            show_set[s] = 1.0 / (60.0 + r)
        for s, r in prk.items():
            show_set[s] = show_set.get(s, 0.0) + w_ppr / (60.0 + r)
        sc2 = [(show_set[s], s) for s in show_set]
        sc2.sort(key=lambda x: (-x[0], x[1]))
        top = [s for _sc, s in sc2[:5]]
        if train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2:
            return list(windows[W]["gtop5"])
        return pad5(top, windows[W]["gtop5"])

    out("| w_ppr | недельный MAP@5 | GT-B весь val |")
    out("|---|---|---|")
    hyb_rows = []
    for w_ppr in (0.1, 0.25, 0.5):
        weekly = []
        for w in range(21, 31):
            W = f"w{w}"
            gt_w = dict(week_gtB.get(w, {}))
            cohort_w = [u for u in sub_int if u in week_gtB.get(w, {})]
            aps = [E.ap_at_k(hybrid_rank(u, W, w_ppr), gt_w[u]) for u in cohort_w]
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))
        gtb = float(np.mean([E.ap_at_k(hybrid_rank(u, "full", w_ppr), gts["B"][u]) for u in cohorts["B"]]))
        hyb_rows.append((w_ppr, wk, gtb))
        out(f"| {w_ppr} | {wk:.4f} | {gtb:.4f} |")
    out()

    pure_wk = 0.3186  # v4 weekly
    best_hyb = max(hyb_rows, key=lambda r: r[1])
    out(f"Лучший гибрид: w_ppr={best_hyb[0]}, недельный {best_hyb[1]:.4f} (чистый v4 = {pure_wk:.4f})")
    gen = best_hyb[1] >= pure_wk - 0.002
    out(f"Генерировать final_submission_ppr: {'да' if gen else 'нет'} (не хуже чистого на 0.002)")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tune_9e.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "tune_9e.md")


if __name__ == "__main__":
    main()
