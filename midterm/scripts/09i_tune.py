"""Stage 9i: validation (GT-D) + history filters, greedy tuning over v5."""

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
from scipy.stats import spearmanr

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


def compute_ppr_bip(csr, cohort_users, user_index, alpha, max_iter=300):
    n_users = csr.shape[0]
    n_items = csr.shape[1]
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    U = sp.diags(1.0 / np.maximum(row_sums, 1e-12)) @ csr
    col_sums = np.asarray(csr.sum(axis=0)).ravel()
    S = (csr @ sp.diags(1.0 / np.maximum(col_sums, 1e-12))).tocsr().T
    valid = [u for u in cohort_users if u in user_index]
    rows = np.array([user_index[u] for u in valid])
    n_seed = len(valid)
    E = np.zeros((n_users, n_seed))
    E[rows, np.arange(n_seed)] = 1.0
    Pu = E.copy()
    Ps = np.zeros((n_items, n_seed))
    for _it in range(max_iter):
        Ps_new = alpha * (U.T @ Pu)
        Pu_new = (1.0 - alpha) * E + alpha * (S.T @ Ps)
        res = float(np.abs(Ps_new - Ps).sum()) + float(np.abs(Pu_new - Pu).sum())
        Pu, Ps = Pu_new, Ps_new
        if res < 1e-6:
            break
    return {u: Ps[:, i] for i, u in enumerate(valid)}


def main() -> None:
    out("# Валидация GT-D и фильтры истории (этап 9i)")
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

    tr_ov = train_watched["overlap"].to_numpy()
    tr_du = train_watched["duration"].to_numpy()
    tr_u = train_watched["user_id"].to_numpy()
    tr_s = train_watched["tv_show_id"].to_numpy()
    tr_vs = train_watched["view_start"].to_numpy()
    tr_comp = tr_ov >= 0.8 * tr_du

    feat = defaultdict(lambda: dict(days=set(), last=0))
    comp_count = defaultdict(int)
    comp_days = defaultdict(set)
    pct_sum = defaultdict(float)
    pct_cnt = defaultdict(int)
    for i in range(len(tr_u)):
        k = (int(tr_u[i]), int(tr_s[i]))
        f = feat[k]
        f["days"].add(int(tr_vs[i]) // 86400)
        if int(tr_vs[i]) > f["last"]:
            f["last"] = int(tr_vs[i])
        if tr_comp[i]:
            comp_count[k] += 1
            comp_days[k].add(int(tr_vs[i]) // 86400)
        pct_sum[k] += float(tr_ov[i]) / float(tr_du[i])
        pct_cnt[k] += 1
    n_days = {k: len(f["days"]) for k, f in feat.items()}
    n_days_comp = {k: len(v) for k, v in comp_days.items()}
    mean_pct = {k: pct_sum[k] / pct_cnt[k] for k in pct_sum}
    last_seen = {k: f["last"] for k, f in feat.items()}

    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)
    user_train_days = defaultdict(set)
    for (u, s), f in feat.items():
        user_train_days[u].update(f["days"])
    train_shows = {u: len(user_train_items.get(u, ())) for u in sub_int}
    train_days = {u: len(user_train_days.get(u, ())) for u in sub_int}

    # slot profile
    print("Loading full log ...", flush=True)
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
    cat_of = {s: meta.get(s, ("",))[0] for s in val_cands}
    pop = defaultdict(set)
    for (u, s) in n_days:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))

    # grid with durations
    week_cands = defaultdict(set)
    air_slots_week = defaultdict(lambda: defaultdict(list))  # (ch, hour, dur)
    with open(config.GRID_21_30_FILE, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            sid = int(row[5])
            if sid == 0:
                continue
            ch = int(row[0])
            st = to_unix(parse_grid_time(row[2]))
            dd = int(float(row[3]))
            w = week_of(st)
            week_cands[w].add(sid)
            air_slots_week[w][sid].append((ch, hour_of_week(st), dd))
    full_air = defaultdict(list)
    for w, shows in air_slots_week.items():
        for s, lst in shows.items():
            full_air[s].extend(lst)

    # GT-D
    va_ov = val_airings["overlap"].to_numpy()
    va_du = val_airings["duration"].to_numpy()
    va_u = val_airings["user_id"].to_numpy()
    va_s = val_airings["tv_show_id"].to_numpy()
    va_comp = va_ov >= 0.8 * va_du
    cntD = defaultdict(int)
    for ui, si, m in zip(va_u, va_s, va_comp):
        if m:
            cntD[(int(ui), int(si))] += 1
    gtD = defaultdict(list)
    for (u, s), c in cntD.items():
        gtD[u].append((c, s))
    for u in gtD:
        gtD[u] = [s for c, s in sorted(gtD[u], key=lambda x: (-x[0], x[1]))[:5]]
    gtD_set = {u: set(v) for u, v in gtD.items()}
    cohortD5 = [u for u in sub_int if len(gtD.get(u, ())) >= 5]
    cohortDall = [u for u in sub_int if len(gtD.get(u, ())) >= 1]

    # GT-B
    bmask = va_ov >= 0.5 * va_du
    gts = {"B": E.build_ground_truth(val_airings, "B")}
    cohortsB = [u for u in sub_int if u in gts["B"]]
    week_gtB = defaultdict(lambda: defaultdict(set))
    for ui, si, vi, m in zip(va_u, va_s, val_airings["view_start"].to_numpy(), bmask):
        if m:
            week_gtB[week_of(vi)][int(ui)].add(int(si))

    # MF + PPR
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)
    cand_mf = sorted(c for c in val_cands if c in item_index)
    cand_mf_idx = np.array([item_index[c] for c in cand_mf], dtype=np.int64)
    Yc = Y[cand_mf_idx]
    pop_log = np.array([np.log1p(pop_count.get(c, 0)) for c in cand_mf])
    mfpop_sc = {}
    for u in sub_int:
        ri = user_index.get(int(u))
        if ri is None:
            mfpop_sc[u] = {}
            continue
        s = X[ri] @ Yc.T
        mu = s.mean()
        sd = s.std()
        z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
        mfpop_sc[u] = {cand_mf[j]: float(z[j] + pop_log[j]) for j in range(len(cand_mf))}

    n_days_log = {(u, s): math.log1p(nd) for (u, s), nd in n_days.items()}
    csr_log, uidx, iidx, iids = MF.build_csr(n_days_log)
    print("Computing PPR ...", flush=True)
    ppr_mass = compute_ppr_bip(csr_log, sub_int, uidx, 0.5)

    # windows
    def make_window(cands, airs, exclude_long):
        n_air = defaultdict(int)
        air_slots = defaultdict(list)
        for s, lst in airs.items():
            for (ch, hour, dd) in lst:
                if exclude_long and dd > 10000:
                    continue
                n_air[s] += 1
                air_slots[s].append((ch, hour))
        return dict(C=cands, n_air=dict(n_air), air_slots=dict(air_slots),
                    gtop5=[s for s in global_rank if s in cands][:5])

    full_air_dict = dict(full_air)
    windows = {"full": None}
    for w in range(21, 31):
        windows[f"w{w}"] = dict(cands=week_cands.get(w, set()), airs=dict(air_slots_week.get(w, {})))

    # ---- rank function -------------------------------------------------
    def rank_user(u, C, n_air, air_slots, gtop, base, hl, gamma, delta, k, wmf, ppr_w, exclude_film):
        prof = slot_profile.get(u, {})
        scored = []
        for s in user_train_items.get(u, ()):
            if s not in C:
                continue
            if exclude_film and cat_of.get(s, "") == "Фильмы":
                continue
            if base == "n_days":
                b = n_days.get((u, s), 0)
            elif base == "comp_count":
                b = comp_count.get((u, s), 0)
            else:
                b = n_days_comp.get((u, s), 0)
            dec = 1.0 if hl == 0 else 0.5 ** ((REF - last_seen.get((u, s), REF)) / (hl * 7 * 86400))
            slot = 0.0
            for (ch, hour) in air_slots.get(s, ()):
                slot += prof.get((ch, hour), 0.0)
            sc = b * dec * (1.0 + gamma * math.log1p(n_air.get(s, 0))) * (1.0 + delta * slot) * (1.0 + k * mean_pct.get((u, s), 0.0))
            scored.append((sc, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        rm_ps = {s: r for r, (_sc, s) in enumerate(scored[:200], 1)}
        show_set = {}
        for s, r in rm_ps.items():
            show_set[s] = 1.0 / (60.0 + r)
        if wmf > 0:
            msc = mfpop_sc.get(u, {})
            lst = [(msc.get(s, 0.0), s) for s in C if s in msc and s not in (set() if not exclude_film else {s for s in C if cat_of.get(s, "") == "Фильмы"})]
            lst.sort(key=lambda x: (-x[0], x[1]))
            for r, (_m, s) in enumerate(lst[:200], 1):
                show_set[s] = show_set.get(s, 0.0) + wmf / (60.0 + r)
        if ppr_w > 0 and u in ppr_mass:
            col = ppr_mass[u]
            lst = [(col[j], item_ids[j]) for j in range(len(item_ids)) if item_ids[j] in C and col[j] > 0 and not (exclude_film and cat_of.get(item_ids[j], "") == "Фильмы")]
            lst.sort(key=lambda x: (-x[0], x[1]))
            for r, (_m, s) in enumerate(lst[:200], 1):
                show_set[s] = show_set.get(s, 0.0) + ppr_w / (60.0 + r)
        sc2 = [(show_set[s], s) for s in show_set]
        sc2.sort(key=lambda x: (-x[0], x[1]))
        return pad5([s for _sc, s in sc2[:5]], gtop)

    def eval_config(base, hl, gamma, delta, k, wmf, ppr_w, exclude_film, exclude_long):
        # weekly
        weekly = []
        for w in range(21, 31):
            W = windows[f"w{w}"]
            Ws = make_window(W["cands"], W["airs"], exclude_long)
            gt_w = dict(week_gtB.get(w, {}))
            cohort_w = [u for u in sub_int if u in week_gtB.get(w, {})]
            aps = []
            for u in cohort_w:
                if train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2:
                    aps.append(E.ap_at_k(list(Ws["gtop5"]), gt_w[u]))
                    continue
                pred = rank_user(u, Ws["C"], Ws["n_air"], Ws["air_slots"], Ws["gtop5"], base, hl, gamma, delta, k, wmf, ppr_w, exclude_film)
                aps.append(E.ap_at_k(pred, gt_w[u]))
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))
        # full
        Wf = make_window(val_cands, full_air_dict, exclude_long)
        apsB = [E.ap_at_k(rank_user(u, Wf["C"], Wf["n_air"], Wf["air_slots"], Wf["gtop5"], base, hl, gamma, delta, k, wmf, ppr_w, exclude_film) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else list(Wf["gtop5"]), gts["B"][u]) for u in cohortsB]
        gtb = float(np.mean(apsB))
        # GT-D
        apD5 = float(np.mean([E.ap_at_k(rank_user(u, Wf["C"], Wf["n_air"], Wf["air_slots"], Wf["gtop5"], base, hl, gamma, delta, k, wmf, ppr_w, exclude_film) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else list(Wf["gtop5"]), gtD_set[u]) for u in cohortD5]))
        apDall = float(np.mean([E.ap_at_k(rank_user(u, Wf["C"], Wf["n_air"], Wf["air_slots"], Wf["gtop5"], base, hl, gamma, delta, k, wmf, ppr_w, exclude_film) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else list(Wf["gtop5"]), gtD_set[u]) for u in cohortDall]))
        return wk, gtb, apD5, apDall

    # ---- calibration (9 configs) --------------------------------------
    kaggle = {"global": 0.27809, "personal": 0.33186, "mf_pop": 0.27073, "v2": 0.33584, "v3": 0.36222,
              "v3b": 0.35621, "v4b": 0.37839, "ppr": 0.38761, "v5": 0.39434}

    def gr_fn(u):
        return [s for s in global_rank if s in val_cands][:5]

    def pr_fn(u):
        sc = {s: n_days[(u, s)] for s in val_cands if (u, s) in n_days}
        return pad5([s for _s2, s in sorted(((v, s) for s, v in sc.items()), key=lambda x: (-x[0], x[1]))[:5]], global_rank[:5])

    def mr_fn(u):
        msc = mfpop_sc.get(u, {})
        lst = sorted(((v, s) for s, v in msc.items() if s in val_cands), key=lambda x: (-x[0], x[1]))
        return pad5([s for _v, s in lst[:5]], global_rank[:5])

    configs = {
        "global": lambda u: gr_fn(u),
        "personal": lambda u: pr_fn(u),
        "mf_pop": lambda u: mr_fn(u),
        "v2": lambda u: rank_user(u, val_cands, {s: len(v) for s, v in full_air_dict.items()}, {s: [(a, b) for (a, b, c) in v] for s, v in full_air_dict.items()}, [s for s in global_rank if s in val_cands][:5], "n_days", 0, 0, 0, 0, 0.5, 0, False) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else gr_fn(u),
        "v3": lambda u: rank_user(u, val_cands, {s: len(v) for s, v in full_air_dict.items()}, {s: [(a, b) for (a, b, c) in v] for s, v in full_air_dict.items()}, [s for s in global_rank if s in val_cands][:5], "n_days", 8, 0.3, 0, 0, 0, 0, False) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else gr_fn(u),
        "v3b": lambda u: rank_user(u, val_cands, {s: len(v) for s, v in full_air_dict.items()}, {s: [(a, b) for (a, b, c) in v] for s, v in full_air_dict.items()}, [s for s in global_rank if s in val_cands][:5], "n_days", 8, 0.3, 0, 0, 0.25, 0, False) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else gr_fn(u),
        "v4b": lambda u: rank_user(u, val_cands, {s: len(v) for s, v in full_air_dict.items()}, {s: [(a, b) for (a, b, c) in v] for s, v in full_air_dict.items()}, [s for s in global_rank if s in val_cands][:5], "n_days", 8, 0.3, 10, 0, 0.25, 0, False) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else gr_fn(u),
        "ppr": lambda u: rank_user(u, val_cands, {s: len(v) for s, v in full_air_dict.items()}, {s: [(a, b) for (a, b, c) in v] for s, v in full_air_dict.items()}, [s for s in global_rank if s in val_cands][:5], "n_days", 8, 0.3, 10, 0, 0, 0.25, False) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else gr_fn(u),
        "v5": lambda u: rank_user(u, val_cands, {s: len(v) for s, v in full_air_dict.items()}, {s: [(a, b) for (a, b, c) in v] for s, v in full_air_dict.items()}, [s for s in global_rank if s in val_cands][:5], "n_days", 8, 0.3, 300, 0, 0, 0.25, False) if not (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2) else gr_fn(u),
    }

    out("## Калибровка (9 конфигов × 4 метрики, Spearman с Kaggle private)")
    out("| конфиг | Kaggle | GT-D D5 | GT-D Dall | недельная GT-B | GT-B весь val |")
    out("|---|---|---|---|---|---|")
    metrics = {"D5": [], "Dall": [], "wk": [], "gtb": []}
    for name in ["global", "personal", "mf_pop", "v2", "v3", "v3b", "v4b", "ppr", "v5"]:
        fn = configs[name]
        rankA = {u: fn(u) for u in cohortsB}
        # D5 / Dall
        apD5 = float(np.mean([E.ap_at_k(rankA.get(u, []), gtD_set[u]) for u in cohortD5]))
        apDall = float(np.mean([E.ap_at_k(rankA.get(u, []), gtD_set[u]) for u in cohortDall]))
        # weekly GT-B
        wk_vals = []
        for w in range(21, 31):
            gt_w = dict(week_gtB.get(w, {}))
            cohort_w = [u for u in sub_int if u in gt_w]
            if cohort_w:
                wk_vals.append(float(np.mean([E.ap_at_k(fn(u), gt_w[u]) for u in cohort_w])))
        wk = float(np.mean(wk_vals))
        gtb = float(np.mean([E.ap_at_k(rankA.get(u, []), gts["B"][u]) for u in cohortsB]))
        metrics["D5"].append(apD5)
        metrics["Dall"].append(apDall)
        metrics["wk"].append(wk)
        metrics["gtb"].append(gtb)
        out(f"| {name} | {kaggle[name]:.5f} | {apD5:.4f} | {apDall:.4f} | {wk:.4f} | {gtb:.4f} |")
    out()

    kg = [kaggle[n] for n in ["global", "personal", "mf_pop", "v2", "v3", "v3b", "v4b", "ppr", "v5"]]
    out("## Корреляция Спирмена с Kaggle private")
    out("| метрика | Spearman rho |")
    out("|---|---|")
    best_metric = None
    for m in ["D5", "Dall", "wk", "gtb"]:
        rho = spearmanr(metrics[m], kg).correlation
        out(f"| {m} | {rho:.4f} |")
        if best_metric is None or abs(rho) > abs(best_metric[1]):
            best_metric = (m, rho)
    out()
    out(f"Лучшая метрика по |rho|: {best_metric[0]} (rho={best_metric[1]:.4f})")
    out()

    # ---- greedy tuning (metric = GT-D D5) ------------------------------
    def eval_metric(base, hl, gamma, delta, k, wmf, ppr_w, exclude_film, exclude_long):
        Wf = make_window(val_cands, full_air_dict, exclude_long)
        def rk(u):
            if train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2:
                return list(Wf["gtop5"])
            return rank_user(u, Wf["C"], Wf["n_air"], Wf["air_slots"], Wf["gtop5"], base, hl, gamma, delta, k, wmf, ppr_w, exclude_film)
        apD5 = float(np.mean([E.ap_at_k(rk(u), gtD_set[u]) for u in cohortD5]))
        gtb = float(np.mean([E.ap_at_k(rk(u), gts["B"][u]) for u in cohortsB]))
        return apD5, gtb

    base_cfg = dict(base="n_days", hl=8.0, gamma=0.3, delta=300.0, k=0.0, wmf=0.0, ppr_w=0.25, exclude_film=False, exclude_long=False)
    base_d5, base_gtb = eval_metric(**base_cfg)
    out(f"БАЗА (v5): D5 {base_d5:.4f}, GT-B {base_gtb:.4f}")
    out()

    best = dict(base_cfg)
    best_d5, best_gtb = base_d5, base_gtb

    def report(title, options):
        nonlocal best, best_d5, best_gtb
        out(f"## {title}")
        out("| вариант | D5 | GT-B | принят |")
        out("|---|---|---|---|")
        out(f"| БАЗА | {best_d5:.4f} | {best_gtb:.4f} | — |")
        rows = []
        for name, cfg in options:
            d5, gtb = eval_metric(**cfg)
            rows.append((name, cfg, d5, gtb))
        rows.sort(key=lambda r: -r[2])
        for name, cfg, d5, gtb in rows:
            mark = ""
            if d5 >= best_d5 + 0.002 and gtb >= best_gtb:
                mark = "да"
                best = dict(cfg)
                best_d5, best_gtb = d5, gtb
            out(f"| {name} | {d5:.4f} | {gtb:.4f} | {mark} |")
        out()

    report("Шаг a: основа счёта", [(b, dict(best, base=b)) for b in ("n_days", "comp_count", "n_days_comp")])
    report("Шаг b: исключить Фильмы", [("да", dict(best, exclude_film=True)), ("нет", dict(best, exclude_film=False))])
    report("Шаг c: исключить эфиры >10000с", [("да", dict(best, exclude_long=True)), ("нет", dict(best, exclude_long=False))])
    report("Шаг d: множитель доли досмотра k", [(f"k {k}", dict(best, k=float(k))) for k in (0, 1, 3)])
    opts_e = []
    for d in (100, 300, 1000):
        for hl in (8, 16, 0):
            opts_e.append((f"delta {d}, hl {hl}", dict(best, delta=float(d), hl=float(hl))))
    report("Шаг e: delta и hl", opts_e)

    out("## Итог")
    out(f"Принятый конфиг: {best}")
    out(f"D5 {best_d5:.4f} (база {base_d5:.4f}, прирост {best_d5 - base_d5:+.4f}), GT-B {best_gtb:.4f}")
    gen = best_d5 >= base_d5 + 0.003
    out(f"Лучше v5 на >= 0.003 по D5: {'да' if gen else 'нет'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tune_9i.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "tune_9i.md")


if __name__ == "__main__":
    main()
