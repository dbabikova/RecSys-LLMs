"""Stage 9f: tune v4+PPR (delta, PPR edge weight, tripartite graph, RRF weight)."""

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
    mass = {u: Ps[:, i] for i, u in enumerate(valid)}
    return mass


def compute_ppr_tri(csr_log, user_index, item_index, cohort_users, slot_profile, air_slots, alpha, max_iter=200):
    n_u = csr_log.shape[0]
    n_i = csr_log.shape[1]
    slot_keys = set()
    for u, slots in slot_profile.items():
        slot_keys.update(slots.keys())
    for s, lst in air_slots.items():
        slot_keys.update(lst)
    slot_keys = sorted(slot_keys)
    slot_idx = {k: i for i, k in enumerate(slot_keys)}
    n_sl = len(slot_keys)

    US = csr_log  # user -> show (log1p n_days)
    # UL: user -> slot
    ul_rows, ul_cols, ul_data = [], [], []
    for u, slots in slot_profile.items():
        if u not in user_index:
            continue
        ru = user_index[u]
        for (ch, hour), v in slots.items():
            ul_rows.append(ru)
            ul_cols.append(slot_idx[(ch, hour)])
            ul_data.append(v)
    UL = sp.csr_matrix((ul_data, (ul_rows, ul_cols)), shape=(n_u, n_sl))
    # SL: show -> slot
    sl_rows, sl_cols, sl_data = [], [], []
    for s, lst in air_slots.items():
        if s not in item_index:
            continue
        rs = item_index[s]
        cnt = defaultdict(int)
        for (ch, hour) in lst:
            cnt[(ch, hour)] += 1
        for (ch, hour), c in cnt.items():
            sl_rows.append(rs)
            sl_cols.append(slot_idx[(ch, hour)])
            sl_data.append(float(c))
    SL = sp.csr_matrix((sl_data, (sl_rows, sl_cols)), shape=(n_i, n_sl))

    A = sp.bmat([[None, US, UL], [US.T, None, SL], [UL.T, SL.T, None]], format="csr")
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    M = sp.diags(1.0 / np.maximum(row_sums, 1e-12)) @ A

    valid = [u for u in cohort_users if u in user_index]
    rows = np.array([user_index[u] for u in valid])
    n_seed = len(valid)
    p = np.zeros((A.shape[0], n_seed))
    p[rows, np.arange(n_seed)] = 1.0
    for _it in range(max_iter):
        p_new = alpha * (M.T @ p)
        p_new[rows, np.arange(n_seed)] += (1.0 - alpha)
        res = float(np.abs(p_new - p).sum())
        p = p_new
        if res < 1e-6:
            break
    # show mass = p[n_u : n_u + n_i]
    mass = {u: p[n_u:n_u + n_i, i] for i, u in enumerate(valid)}
    return mass


def main() -> None:
    out("# Доводка v4+PPR (этап 9f)")
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

    # windows
    windows = {}
    windows["full"] = dict(C=val_cands, n_air=full_n_air, air_slots=dict(full_air),
                           gtop5=[s for s in global_rank if s in val_cands][:5])
    for w in range(21, 31):
        cw = week_cands.get(w, set())
        windows[f"w{w}"] = dict(C=cw, n_air=dict(n_air_week.get(w, {})), air_slots=dict(air_slots_week.get(w, {})),
                                gtop5=[s for s in global_rank if s in cw][:5])

    # ---- build PPR variants -------------------------------------------
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)

    n_days_log = {(u, s): math.log1p(nd) for (u, s), nd in n_days.items()}
    n_days_log_decay = {(u, s): math.log1p(nd) * (0.5 ** ((REF - feat[(u, s)]["last"]) / (8 * 7 * 86400)))
                        for (u, s), nd in n_days.items()}
    csr_log, uidx, iidx, iids = MF.build_csr(n_days_log)
    csr_log_decay, _, _, _ = MF.build_csr(n_days_log_decay)

    cohort_ppr = cohorts["A"]

    print("Computing bipartite PPR ...", flush=True)
    mass_a = compute_ppr_bip(csr_log, cohort_ppr, uidx, 0.5)
    mass_b = compute_ppr_bip(csr_log_decay, cohort_ppr, uidx, 0.5)
    print("Computing tripartite PPR ...", flush=True)
    mass_tri03 = compute_ppr_tri(csr_log, uidx, iidx, cohort_ppr, slot_profile, full_air, 0.3)
    mass_tri05 = compute_ppr_tri(csr_log, uidx, iidx, cohort_ppr, slot_profile, full_air, 0.5)

    ppr_maps = {
        "log1p": mass_a,
        "log1p_decay": mass_b,
        "tri0.3": mass_tri03,
        "tri0.5": mass_tri05,
    }

    # ---- eval ----------------------------------------------------------
    hl = 8.0
    gamma = 0.3

    def eval_config(delta, ppr_key, ppr_w):
        mass = ppr_maps[ppr_key]
        # weekly
        weekly = []
        for w in range(21, 31):
            W = f"w{w}"
            C = windows[W]["C"]
            n_air = windows[W]["n_air"]
            air_slots = windows[W]["air_slots"]
            gtop = windows[W]["gtop5"]
            gt_w = dict(week_gtB.get(w, {}))
            cohort_w = [u for u in sub_int if u in week_gtB.get(w, {})]
            aps = []
            for u in cohort_w:
                if train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2:
                    aps.append(E.ap_at_k(list(gtop), gt_w[u]))
                    continue
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
                rm_ps = {s: r for r, (_sc, s) in enumerate(scored[:200], 1)}
                show_set = {}
                for s, r in rm_ps.items():
                    show_set[s] = 1.0 / (60.0 + r)
                if u in mass:
                    col = mass[u]
                    lst = [(col[j], item_ids[j]) for j in range(len(item_ids)) if item_ids[j] in C and col[j] > 0]
                    lst.sort(key=lambda x: (-x[0], x[1]))
                    for r, (_m, s) in enumerate(lst[:200], 1):
                        show_set[s] = show_set.get(s, 0.0) + ppr_w / (60.0 + r)
                sc2 = [(show_set[s], s) for s in show_set]
                sc2.sort(key=lambda x: (-x[0], x[1]))
                pred = pad5([s for _sc, s in sc2[:5]], gtop)
                aps.append(E.ap_at_k(pred, gt_w[u]))
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))
        # full GT-B
        S = windows["full"]
        C = S["C"]
        n_air = S["n_air"]
        air_slots = S["air_slots"]
        gtop = S["gtop5"]
        aps = []
        for u in cohorts["B"]:
            if train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2:
                aps.append(E.ap_at_k(list(gtop), gts["B"][u]))
                continue
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
            rm_ps = {s: r for r, (_sc, s) in enumerate(scored[:200], 1)}
            show_set = {}
            for s, r in rm_ps.items():
                show_set[s] = 1.0 / (60.0 + r)
            if u in mass:
                col = mass[u]
                lst = [(col[j], item_ids[j]) for j in range(len(item_ids)) if item_ids[j] in C and col[j] > 0]
                lst.sort(key=lambda x: (-x[0], x[1]))
                for r, (_m, s) in enumerate(lst[:200], 1):
                    show_set[s] = show_set.get(s, 0.0) + ppr_w / (60.0 + r)
            sc2 = [(show_set[s], s) for s in show_set]
            sc2.sort(key=lambda x: (-x[0], x[1]))
            pred = pad5([s for _sc, s in sc2[:5]], gtop)
            aps.append(E.ap_at_k(pred, gts["B"][u]))
        gtb = float(np.mean(aps))
        return wk, gtb

    # ---- greedy steps ---------------------------------------------------
    base_cfg = dict(delta=10.0, ppr="log1p", ppr_w=0.25)
    base_wk, base_gtb = eval_config(10.0, "log1p", 0.25)
    out(f"БАЗА (v4+PPR): недельный {base_wk:.4f}, GT-B {base_gtb:.4f}")
    out()

    best = dict(base_cfg)
    best_wk, best_gtb = base_wk, base_gtb

    def report(title, options):
        nonlocal best, best_wk, best_gtb
        out(f"## {title}")
        out("| вариант | недельный | GT-B | принят |")
        out("|---|---|---|---|")
        out(f"| БАЗА | {best_wk:.4f} | {best_gtb:.4f} | — |")
        rows = []
        for name, cfg in options:
            wk, gtb = eval_config(cfg["delta"], cfg["ppr"], cfg["ppr_w"])
            rows.append((name, cfg, wk, gtb))
        rows.sort(key=lambda r: -r[2])
        for name, cfg, wk, gtb in rows:
            mark = ""
            if wk >= best_wk + 0.002 and gtb >= best_gtb:
                mark = "да"
                best = dict(cfg)
                best_wk, best_gtb = wk, gtb
            out(f"| {name} | {wk:.4f} | {gtb:.4f} | {mark} |")
        out()

    # Step 1
    report("Шаг 1: delta", [(f"delta {d}", dict(delta=float(d), ppr="log1p", ppr_w=0.25)) for d in (10, 20, 50, 100, 300)])
    # Step 2
    report("Шаг 2: вес ребра PPR", [("log1p", dict(delta=best["delta"], ppr="log1p", ppr_w=best["ppr_w"])),
                                   ("log1p×decay", dict(delta=best["delta"], ppr="log1p_decay", ppr_w=best["ppr_w"]))])
    # Step 3
    report("Шаг 3: трёхдольный граф", [("bipartite log1p", dict(delta=best["delta"], ppr=best["ppr"], ppr_w=best["ppr_w"])),
                                       ("tri0.3", dict(delta=best["delta"], ppr="tri0.3", ppr_w=best["ppr_w"])),
                                       ("tri0.5", dict(delta=best["delta"], ppr="tri0.5", ppr_w=best["ppr_w"]))])
    # Step 4
    report("Шаг 4: вес PPR в RRF", [(f"ppr_w {w}", dict(delta=best["delta"], ppr=best["ppr"], ppr_w=w)) for w in (0.25, 0.5, 1.0)])

    out("## Итог")
    out(f"Принятый конфиг: {best}")
    out(f"недельный {best_wk:.4f} (база {base_wk:.4f}, прирост {best_wk - base_wk:+.4f}), GT-B {best_gtb:.4f}")
    out()
    gen = best_wk >= base_wk + 0.003
    out(f"Лучше базы на >= 0.003: {'да' if gen else 'нет'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tune_9f.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "tune_9f.md")


if __name__ == "__main__":
    main()
