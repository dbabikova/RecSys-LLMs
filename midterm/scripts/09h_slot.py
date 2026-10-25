"""Stage 9h: refine the slot signal (formula, profile key/decay, hyperparams, padding, switching)."""

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


def build_slot_profile(views, key_type, decay_hl):
    # views: numpy arrays (user, channel, start, stop)
    acc = defaultdict(lambda: defaultdict(float))
    uu, cc, ss, ee = views
    for i in range(len(ss)):
        u = int(uu[i])
        ch = int(cc[i])
        s = int(ss[i])
        e = int(ee[i])
        d = e - s
        if d <= 0:
            continue
        w = float(d)
        if decay_hl is not None:
            week_age = (REF - s) / (7 * 86400)
            w *= 0.5 ** (week_age / decay_hl)
        if key_type == "ch_hour":
            acc[u][(ch, hour_of_week(s))] += w
        elif key_type == "ch_weekend_hour":
            acc[u][(ch, is_weekend(s), s % 86400 // 3600)] += w
        else:  # ch_hour_smooth
            h = hour_of_week(s)
            acc[u][(ch, h)] += 0.5 * w
            acc[u][(ch, h - 1)] += 0.25 * w
            acc[u][(ch, h + 1)] += 0.25 * w
    prof = {}
    for u, slots in acc.items():
        tot = sum(slots.values())
        prof[u] = {k: v / tot for k, v in slots.items()}
    return prof


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

    print("Loading full log (train views) ...", flush=True)
    u_l, c_l, s_l, e_l = [], [], [], []
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
                e = s + E.CAP
            if e <= s:
                continue
            u_l.append(u)
            c_l.append(ch)
            s_l.append(s)
            e_l.append(e)
    views = (np.array(u_l, dtype=np.int64), np.array(c_l, dtype=np.int32),
             np.array(s_l, dtype=np.int64), np.array(e_l, dtype=np.int64))

    meta, val_cands, title_map, _ = CBF.load_show_metadata()
    pop = defaultdict(set)
    for (u, s) in n_days:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))

    # grid 21-30: per-window airings (ch, start, dur)
    week_cands = defaultdict(set)
    air_week = defaultdict(lambda: defaultdict(list))  # week -> show -> [(ch, start, dur)]
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
            air_week[w][sid].append((ch, st, dd))
    full_air = defaultdict(list)
    for w, shows in air_week.items():
        for s, lst in shows.items():
            full_air[s].extend(lst)
    full_n_air = {s: len(v) for s, v in full_air.items()}

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
    cohortsB = [u for u in sub_int if u in gtB]
    week_gtB = defaultdict(lambda: defaultdict(set))
    for ui, si, vi, m in zip(ua, sa, vsa, bmask):
        if m:
            week_gtB[week_of(vi)][int(ui)].add(int(si))

    windows = {}
    windows["full"] = dict(C=val_cands, n_air=full_n_air, air=dict(full_air),
                           gtop5=[s for s in global_rank if s in val_cands][:5])
    for w in range(21, 31):
        cw = week_cands.get(w, set())
        windows[f"w{w}"] = dict(C=cw, n_air={s: len(v) for s, v in air_week.get(w, {}).items()},
                                air=dict(air_week.get(w, {})),
                                gtop5=[s for s in global_rank if s in cw][:5])

    # PPR (log1p a0.5)
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    n_days_log = {(u, s): math.log1p(nd) for (u, s), nd in n_days.items()}
    csr_log, uidx, iidx, iids = MF.build_csr(n_days_log)
    print("Computing PPR ...", flush=True)
    mass = compute_ppr_bip(csr_log, cohortsB, uidx, 0.5)

    prof_cache = {}

    def get_profile(key_type, decay_hl):
        k = (key_type, decay_hl)
        if k not in prof_cache:
            prof_cache[k] = build_slot_profile(views, key_type, decay_hl)
        return prof_cache[k]

    def slot_score(u, show, profile, airings, formula):
        prof = profile.get(u, {})
        if formula == "a":
            tot = 0.0
            for (ch, st, dd) in airings:
                h = hour_of_week(st)
                if prof_key_type == "ch_hour" or prof_key_type == "ch_hour_smooth":
                    tot += prof.get((ch, h), 0.0)
                else:
                    tot += prof.get((ch, is_weekend(st), st % 86400 // 3600), 0.0)
            return tot
        # formula b
        tot = 0.0
        for (ch, st, dd) in airings:
            en = st + dd
            h_start = st // 3600
            h_end = (en - 1) // 3600
            for he in range(h_start, h_end + 1):
                hs = he * 3600
                he_ = hs + 3600
                ovl = min(en, he_) - max(st, hs)
                if ovl <= 0:
                    continue
                frac = ovl / 3600.0
                if prof_key_type == "ch_hour" or prof_key_type == "ch_hour_smooth":
                    tot += prof.get((ch, hour_of_week(hs)), 0.0) * frac
                else:
                    tot += prof.get((ch, is_weekend(hs), hs % 86400 // 3600), 0.0) * frac
        return tot

    global prof_key_type

    def eval_cfg(cfg):
        global prof_key_type
        prof_key_type = cfg["profile_key"]
        profile = get_profile(cfg["profile_key"], cfg["profile_decay"])
        formula = cfg["slot_formula"]
        hl = cfg["hl"]
        gamma = cfg["gamma"]
        delta = cfg["delta"]
        padding = cfg["padding"]
        switching = cfg["switching"]
        ppr_w = 0.25

        weekly = []
        for w in range(21, 31):
            W = f"w{w}"
            C = windows[W]["C"]
            n_air = windows[W]["n_air"]
            air = windows[W]["air"]
            gtop = windows[W]["gtop5"]
            gt_w = dict(week_gtB.get(w, {}))
            cohort_w = [u for u in sub_int if u in week_gtB.get(w, {})]
            aps = []
            for u in cohort_w:
                light = train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2
                if light and switching == "global":
                    aps.append(E.ap_at_k(list(gtop), gt_w[u]))
                    continue
                scored = []
                for s in user_train_items.get(u, ()):
                    if s not in C:
                        continue
                    nd = n_days.get((u, s), 0)
                    dec = 1.0 if hl == 0 else 0.5 ** ((REF - feat.get((u, s), {"last": REF})["last"]) / (hl * 7 * 86400))
                    sc = nd * dec * (1.0 + gamma * math.log1p(n_air.get(s, 0))) * (1.0 + delta * slot_score(u, s, profile, air.get(s, ()), formula))
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
                top = [s for _sc, s in sc2[:5]]
                if light and switching == "slot_pop":
                    # slot_pop ranking
                    cand = [s for s in C if s not in top]
                    cand.sort(key=lambda s: (-slot_score(u, s, profile, air.get(s, ()), formula) * math.log1p(pop_count.get(s, 0)), s))
                    top = list(top)
                    for s in cand:
                        if len(top) >= 5:
                            break
                        top.append(s)
                    aps.append(E.ap_at_k(pad5(top, gtop), gt_w[u]))
                    continue
                if len(top) < 5:
                    if padding == "global":
                        top = pad5(top, gtop)
                    else:  # slot_pop
                        cand = [s for s in C if s not in top]
                        cand.sort(key=lambda s: (-slot_score(u, s, profile, air.get(s, ()), formula) * math.log1p(pop_count.get(s, 0)), s))
                        for s in cand:
                            if len(top) >= 5:
                                break
                            top.append(s)
                        top = pad5(top, gtop)
                aps.append(E.ap_at_k(top[:5], gt_w[u]))
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))

        S = windows["full"]
        C = S["C"]
        n_air = S["n_air"]
        air = S["air"]
        gtop = S["gtop5"]
        aps = []
        for u in cohortsB:
            light = train_shows.get(u, 0) < 5 or train_days.get(u, 0) < 2
            if light and switching == "global":
                aps.append(E.ap_at_k(list(gtop), gtB[u]))
                continue
            scored = []
            for s in user_train_items.get(u, ()):
                if s not in C:
                    continue
                nd = n_days.get((u, s), 0)
                dec = 1.0 if hl == 0 else 0.5 ** ((REF - feat.get((u, s), {"last": REF})["last"]) / (hl * 7 * 86400))
                sc = nd * dec * (1.0 + gamma * math.log1p(n_air.get(s, 0))) * (1.0 + delta * slot_score(u, s, profile, air.get(s, ()), formula))
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
            top = [s for _sc, s in sc2[:5]]
            if light and switching == "slot_pop":
                cand = [s for s in C if s not in top]
                cand.sort(key=lambda s: (-slot_score(u, s, profile, air.get(s, ()), formula) * math.log1p(pop_count.get(s, 0)), s))
                for s in cand:
                    if len(top) >= 5:
                        break
                    top.append(s)
                aps.append(E.ap_at_k(pad5(top, gtop), gtB[u]))
                continue
            if len(top) < 5:
                if padding == "global":
                    top = pad5(top, gtop)
                else:
                    cand = [s for s in C if s not in top]
                    cand.sort(key=lambda s: (-slot_score(u, s, profile, air.get(s, ()), formula) * math.log1p(pop_count.get(s, 0)), s))
                    for s in cand:
                        if len(top) >= 5:
                            break
                        top.append(s)
                    top = pad5(top, gtop)
            aps.append(E.ap_at_k(top[:5], gtB[u]))
        gtb = float(np.mean(aps))
        return wk, gtb

    # ---- base = v5 -----------------------------------------------------
    base_cfg = dict(slot_formula="a", profile_key="ch_hour", profile_decay=None,
                    hl=8, gamma=0.3, delta=300, padding="global", switching="global")
    base_wk, base_gtb = eval_cfg(base_cfg)
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
            wk, gtb = eval_cfg(cfg)
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
    def mk1(formula):
        c = dict(best); c["slot_formula"] = formula; return c
    report("Шаг 1: слотовый скор", [("a (час начала)", mk1("a")), ("b (перекрытие часов)", mk1("b"))])

    # Step 2
    opts2 = []
    for key in ("ch_hour", "ch_weekend_hour", "ch_hour_smooth"):
        c = dict(best); c["profile_key"] = key; c["profile_decay"] = None
        opts2.append((key, c))
    for key in ("ch_hour",):
        for hl_dec in (4, 8):
            c = dict(best); c["profile_key"] = key; c["profile_decay"] = hl_dec
            opts2.append((f"{key}+decay{hl_dec}", c))
    report("Шаг 2: ключ профиля + decay", opts2)

    # Step 3
    opts3 = []
    for hl in (4, 8, 16, 0):
        for gamma in (0.0, 0.3, 1.0):
            for delta in (100, 300, 1000):
                c = dict(best); c["hl"] = hl; c["gamma"] = gamma; c["delta"] = float(delta)
                opts3.append((f"hl{hl} g{gamma} d{delta}", c))
    report("Шаг 3: hl/gamma/delta", opts3)

    # Step 4
    opts4 = [("global", dict(best, padding="global")), ("slot_pop", dict(best, padding="slot_pop"))]
    report("Шаг 4: добивка", opts4)

    # Step 5
    opts5 = [("global", dict(best, switching="global")), ("slot_pop", dict(best, switching="slot_pop"))]
    report("Шаг 5: переключение лёгких", opts5)

    out("## Итог")
    out(f"Принятый конфиг: {best}")
    out(f"недельный {best_wk:.4f} (база {base_wk:.4f}, прирост {best_wk - base_wk:+.4f}), GT-B {best_gtb:.4f}")
    out()
    gen = best_wk >= base_wk + 0.003
    out(f"Лучше v5 на >= 0.003: {'да' if gen else 'нет'}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tune_9h.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "tune_9h.md")


if __name__ == "__main__":
    main()
