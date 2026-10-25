"""Stage 9h: refine the slot signal (slot score, profile key/decay, params, padding, switching)."""

import csv
import math
import sys
import time
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


def is_weekend(ts):
    return 1 if datetime.utcfromtimestamp(int(ts)).weekday() >= 5 else 0


def hour_of_day(ts):
    return datetime.utcfromtimestamp(int(ts)).hour


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


def key_of(ch, ts, pkey):
    if pkey == "b":
        return (ch, is_weekend(ts), hour_of_day(ts))
    return (ch, hour_of_week(ts))


def main() -> None:
    out("# Доводка слотового сигнала (этап 9h)")
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

    # ---- slot profiles (keys a/b, decay none/4/8) ----------------------
    print("Loading full log for slot profiles ...", flush=True)
    acc_a = {d: defaultdict(lambda: defaultdict(float)) for d in ("none", "4", "8")}
    acc_b = {d: defaultdict(lambda: defaultdict(float)) for d in ("none", "4", "8")}
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
            wa = (ch, hour_of_week(s))
            wb = (ch, is_weekend(s), hour_of_day(s))
            weeks_ago = (REF - s) / (7 * 86400)
            acc_a["none"][u][wa] += d
            acc_b["none"][u][wb] += d
            for hl in ("4", "8"):
                w = 0.5 ** (weeks_ago / int(hl))
                acc_a[hl][u][wa] += d * w
                acc_b[hl][u][wb] += d * w

    def norm(acc):
        prof = {}
        for u, slots in acc.items():
            tot = sum(slots.values())
            prof[u] = {k: v / tot for k, v in slots.items()}
        return prof

    profiles = {}
    for d in ("none", "4", "8"):
        profiles[("a", d)] = norm(acc_a[d])
        profiles[("b", d)] = norm(acc_b[d])
    # key c = smoothed key a
    for d in ("none", "4", "8"):
        pa = profiles[("a", d)]
        pc = {}
        for u, slots in pa.items():
            sm = defaultdict(float)
            for (ch, h), v in slots.items():
                sm[(ch, h)] += 0.5 * v
                sm[(ch, (h - 1) % 168)] += 0.25 * v
                sm[(ch, (h + 1) % 168)] += 0.25 * v
            pc[u] = dict(sm)
        profiles[("c", d)] = pc

    # ---- metadata + grid -----------------------------------------------
    meta, val_cands, title_map, _ = CBF.load_show_metadata()
    pop = defaultdict(set)
    for (u, s) in n_days:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))

    week_cands = defaultdict(set)
    air_week = defaultdict(lambda: defaultdict(list))  # week -> show -> [(ch, start, dur)]
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
            d = int(float(row[3]))
            w = week_of(st)
            week_cands[w].add(sid)
            air_week[w][sid].append((ch, st, d))
            n_air_week[w][sid] += 1
    full_air = defaultdict(list)
    for w, shows in air_week.items():
        for s, lst in shows.items():
            full_air[s].extend(lst)
    full_n_air = {s: len(v) for s, v in full_air.items()}

    # ---- GT-B ----------------------------------------------------------
    ova = val_airings["overlap"].to_numpy()
    dua = val_airings["duration"].to_numpy()
    ua = val_airings["user_id"].to_numpy()
    sa = val_airings["tv_show_id"].to_numpy()
    vsa = val_airings["view_start"].to_numpy()
    bmask = ova >= 0.5 * dua
    gts = {"B": E.build_ground_truth(val_airings, "B")}
    cohortsB = [u for u in sub_int if u in gts["B"]]
    week_gtB = defaultdict(lambda: defaultdict(set))
    for ui, si, vi, m in zip(ua, sa, vsa, bmask):
        if m:
            week_gtB[week_of(vi)][int(ui)].add(int(si))

    windows = {}
    windows["full"] = dict(C=val_cands, n_air=full_n_air, air=dict(full_air),
                           gtop5=[s for s in global_rank if s in val_cands][:5])
    for w in range(21, 31):
        cw = week_cands.get(w, set())
        windows[f"w{w}"] = dict(C=cw, n_air=dict(n_air_week.get(w, {})), air=dict(air_week.get(w, {})),
                                gtop5=[s for s in global_rank if s in cw][:5])

    # ---- PPR -----------------------------------------------------------
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    n_days_log = {(u, s): math.log1p(nd) for (u, s), nd in n_days.items()}
    csr_log, uidx, iidx, iids = MF.build_csr(n_days_log)
    print("Computing PPR ...", flush=True)
    mass = compute_ppr_bip(csr_log, cohortsB, uidx, 0.5)

    # ---- slot score helpers --------------------------------------------
    def slot_score(u, s, W, pkey, pdecay, svar):
        prof = profiles[(pkey, pdecay)].get(u, {})
        airings = windows[W]["air"].get(s, ())
        total = 0.0
        for (ch, st, d) in airings:
            if svar == "a":
                total += prof.get(key_of(ch, st, pkey), 0.0)
            else:
                end = st + d
                t = st
                while t < end:
                    k = key_of(ch, t, pkey)
                    next_hour = (t // 3600 + 1) * 3600
                    seg_end = min(end, next_hour)
                    frac = (seg_end - t) / 3600.0
                    total += prof.get(k, 0.0) * frac
                    t = seg_end
        return total

    # ---- eval ----------------------------------------------------------
    def eval_config(cfg):
        pkey = cfg["pkey"]
        pdecay = cfg["pdecay"]
        svar = cfg["svar"]
        hl = cfg["hl"]
        gamma = cfg["gamma"]
        delta = cfg["delta"]
        padding = cfg["padding"]
        switching = cfg["switching"]

        def rank(u, W):
            C = windows[W]["C"]
            n_air = windows[W]["n_air"]
            gtop = windows[W]["gtop5"]
            prof_u = profiles[(pkey, pdecay)].get(u, {})
            scored = []
            for s in user_train_items.get(u, ()):
                if s not in C:
                    continue
                nd = n_days.get((u, s), 0)
                dec = 1.0 if hl == 0 else 0.5 ** ((REF - feat.get((u, s), {"last": REF})["last"]) / (hl * 7 * 86400))
                sc = nd * dec * (1.0 + gamma * math.log1p(n_air.get(s, 0))) * (1.0 + delta * slot_score(u, s, W, pkey, pdecay, svar))
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
                    show_set[s] = show_set.get(s, 0.0) + 0.25 / (60.0 + r)
            sc2 = [(show_set[s], s) for s in show_set]
            sc2.sort(key=lambda x: (-x[0], x[1]))
            top = [s for _sc, s in sc2[:5]]

            if switching == "slot_pop" and (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2):
                cands = [s for s in C if s not in set(top)]
                cands.sort(key=lambda s: (-(slot_score(u, s, W, pkey, pdecay, svar) * math.log1p(pop_count.get(s, 0))), s))
                return pad5(cands[:5], gtop)
            if switching == "global" and (train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2):
                return list(gtop)

            if len(top) < 5 and padding == "slot_pop":
                res = list(top)
                cands = [s for s in C if s not in set(res)]
                cands.sort(key=lambda s: (-(slot_score(u, s, W, pkey, pdecay, svar) * math.log1p(pop_count.get(s, 0))), s))
                for s in cands:
                    if len(res) >= 5:
                        break
                    res.append(s)
                return pad5(res, gtop)
            return pad5(top, gtop)

        weekly = []
        for w in range(21, 31):
            W = f"w{w}"
            gt_w = dict(week_gtB.get(w, {}))
            cohort_w = [u for u in sub_int if u in week_gtB.get(w, {})]
            aps = [E.ap_at_k(rank(u, W), gt_w[u]) for u in cohort_w]
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))
        gtb = float(np.mean([E.ap_at_k(rank(u, "full"), gts["B"][u]) for u in cohortsB]))
        return wk, gtb

    # ---- base ----------------------------------------------------------
    base_cfg = dict(pkey="a", pdecay="none", svar="a", hl=8, gamma=0.3, delta=300, padding="global", switching="global")
    base_wk, base_gtb = eval_config(base_cfg)
    out(f"БАЗА (v5): недельный {base_wk:.4f}, GT-B {base_gtb:.4f}")
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
            wk, gtb = eval_config(cfg)
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

    # Step 1: slot score variant
    report("Шаг 1: слотовый скор (a=час начала, b=перекрытие часов)", [
        ("a", dict(base_cfg, svar="a")),
        ("b", dict(base_cfg, svar="b")),
    ])
    # Step 2: profile key + decay
    step2 = []
    for pkey in ("a", "b", "c"):
        for d in ("none", "4", "8"):
            step2.append((f"{pkey}/{d}", dict(base_cfg, pkey=pkey, pdecay=d)))
    report("Шаг 2: ключ профиля + decay", step2)
    # Step 3: hl, gamma, delta
    step3 = []
    for hl in (4, 8, 16, 0):
        for gamma in (0.0, 0.3, 1.0):
            for delta in (100, 300, 1000):
                step3.append((f"hl{hl} g{gamma} d{delta}", dict(base_cfg, hl=hl, gamma=gamma, delta=float(delta))))
    report("Шаг 3: hl/gamma/delta", step3)
    # Step 4: padding
    report("Шаг 4: добивка", [
        ("global", dict(base_cfg, padding="global")),
        ("slot×pop", dict(base_cfg, padding="slot_pop")),
    ])
    # Step 5: switching
    report("Шаг 5: лёгкие пользователи", [
        ("global", dict(base_cfg, switching="global")),
        ("slot×pop", dict(base_cfg, switching="slot_pop")),
    ])

    out("## Итог")
    out(f"Принятый конфиг: {best}")
    out(f"недельный {best_wk:.4f} (база {base_wk:.4f}, прирост {best_wk - base_wk:+.4f}), GT-B {best_gtb:.4f}")
    out()
    gen = best_wk >= base_wk + 0.003
    out(f"Лучше базы на >= 0.003: {'да' if gen else 'нет'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tune_9h.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "tune_9h.md")


if __name__ == "__main__":
    main()
