"""Stage 9g: val metrics for delta 1000 / 3000 (same code as 9f), append to tune_9f.md."""

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
import evaluate as E
import mf_als as MF
import cbf as CBF
from join import parse_log_time, parse_grid_time, to_unix

REF = E.VAL_START


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


def main():
    sub_int = sorted(int(u) for u in E.load_submission_ids())

    airings = E.load_labeled_airings(E.CAP)
    ov = airings["overlap"].to_numpy()
    dur = airings["duration"].to_numpy()
    vst = airings["view_start"].to_numpy()
    train_mask = vst < E.VAL_START
    val_mask = (vst >= E.VAL_START) & (vst <= E.VAL_END)
    watched = E.watched_mask_np(ov, dur)
    train_watched = airings[train_mask & watched]
    val_airings = airings[val_mask]

    feat = defaultdict(lambda: dict(days=set(), last=0))
    for u, s, vs in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"]):
        k = (int(u), int(s))
        f = feat[k]
        f["days"].add(int(vs) // 86400)
        if int(vs) > f["last"]:
            f["last"] = int(vs)
    n_days = {k: len(f["days"]) for k, f in feat.items()}
    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)
    user_train_days = defaultdict(set)
    for (u, s), f in feat.items():
        user_train_days[u].update(f["days"])
    train_shows = {u: len(user_train_items.get(u, ())) for u in sub_int}
    train_days = {u: len(user_train_days.get(u, ())) for u in sub_int}

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
    gts = {d: E.build_ground_truth(val_airings, d) for d in ("B",)}
    cohortsB = [u for u in sub_int if u in gts["B"]]
    week_gtB = defaultdict(lambda: defaultdict(set))
    for ui, si, vi, m in zip(ua, sa, vsa, bmask):
        if m:
            week_gtB[week_of(vi)][int(ui)].add(int(si))

    windows = {}
    windows["full"] = dict(C=val_cands, n_air=full_n_air, air_slots=dict(full_air),
                           gtop5=[s for s in global_rank if s in val_cands][:5])
    for w in range(21, 31):
        cw = week_cands.get(w, set())
        windows[f"w{w}"] = dict(C=cw, n_air=dict(n_air_week.get(w, {})), air_slots=dict(air_slots_week.get(w, {})),
                                gtop5=[s for s in global_rank if s in cw][:5])

    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    n_days_log = {(u, s): math.log1p(nd) for (u, s), nd in n_days.items()}
    csr_log, uidx, iidx, iids = MF.build_csr(n_days_log)
    print("Computing PPR ...", flush=True)
    mass = compute_ppr_bip(csr_log, cohortsB, uidx, 0.5)

    hl, gamma = 8.0, 0.3
    ppr_w = 0.25

    def eval_delta(delta):
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
                aps.append(E.ap_at_k(pad5([s for _sc, s in sc2[:5]], gtop), gt_w[u]))
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))
        S = windows["full"]
        C = S["C"]
        n_air = S["n_air"]
        air_slots = S["air_slots"]
        gtop = S["gtop5"]
        aps = []
        for u in cohortsB:
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
            aps.append(E.ap_at_k(pad5([s for _sc, s in sc2[:5]], gtop), gts["B"][u]))
        gtb = float(np.mean(aps))
        return wk, gtb

    lines = []
    for d in (1000.0, 3000.0):
        wk, gtb = eval_delta(d)
        print(f"delta {d:.0f}: weekly {wk:.4f}, GT-B {gtb:.4f}")
        lines.append(f"| delta {d:.0f} (диагностика) | {wk:.4f} | {gtb:.4f} | — |")

    p = config.RESULTS_DIR / "tune_9f.md"
    txt = p.read_text(encoding="utf-8")
    if "диагностика" not in txt:
        txt = txt.rstrip() + "\n\n## Диагностика (delta 1000 / 3000)\n\n| вариант | недельный | GT-B | принят |\n|---|---|---|---|\n" + "\n".join(lines) + "\n"
        p.write_text(txt, encoding="utf-8")
        print("appended to", p)


if __name__ == "__main__":
    main()
