"""Stage 9d: greedy final tuning of the personal-history ranking."""

import csv
import math
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


def main() -> None:
    out("# Финальная доводка личной истории (этап 9d)")
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

    feat = defaultdict(lambda: dict(days=set(), last=0, weeks=set(), ov=0.0))
    for u, s, vs, o in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"], train_watched["overlap"]):
        k = (int(u), int(s))
        f = feat[k]
        f["days"].add(int(vs) // 86400)
        f["weeks"].add(week_of(int(vs)))
        if int(vs) > f["last"]:
            f["last"] = int(vs)
        f["ov"] += float(o)
    n_days = {k: len(f["days"]) for k, f in feat.items()}
    for k, f in feat.items():
        f["weeks"] = frozenset(f["weeks"])

    user_train_items = defaultdict(set)
    user_train_days = defaultdict(set)
    for (u, s), f in feat.items():
        user_train_items[u].add(s)
    train_shows = {u: len(user_train_items.get(u, ())) for u in sub_int}
    train_days = {}
    # distinct days of train viewing per user (recompute from val-period day indices)
    info_days = defaultdict(set)
    for u, s, vs in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"]):
        info_days[(int(u), int(s))].add(int(vs) // 86400)
    for (u, s), ds in info_days.items():
        user_train_days[u].update(ds)
    train_days = {u: len(user_train_days.get(u, ())) for u in sub_int}

    # ---- slot profile from full log (train part, incl. unlabeled) -------
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

    # ---- metadata + grid 21-30 -----------------------------------------
    meta, val_cands, title_map, _ = CBF.load_show_metadata()
    train_items = set(s for (_u, s) in n_days)
    new_ids = val_cands - train_items
    base_title = {s: CBF.normalize_title(title_map.get(s, str(s))) for s in (val_cands | train_items)}

    pop = defaultdict(set)
    for (u, s) in n_days:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}
    global_rank = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))

    # per-week candidates + airings (ch, hour)
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

    # ---- GT-B (full + per-week) ----------------------------------------
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

    # ---- windows --------------------------------------------------------
    full_air = air_slots_full(air_slots_week)
    full_n_air = {s: len(v) for s, v in full_air.items()}
    windows = {}
    windows["full"] = dict(C=val_cands, n_air=full_n_air, air_slots=full_air,
                           grank=[s for s in global_rank if s in val_cands],
                           gtop5=[s for s in global_rank if s in val_cands][:5], gt=gtB, cohort=cohortB)
    for w in range(21, 31):
        cw = week_cands.get(w, set())
        windows[f"w{w}"] = dict(C=cw, n_air=dict(n_air_week.get(w, {})), air_slots=dict(air_slots_week.get(w, {})),
                                grank=[s for s in global_rank if s in cw], gtop5=[s for s in global_rank if s in cw][:5],
                                gt=dict(week_gtB.get(w, {})), cohort=[u for u in sub_int if u in week_gtB.get(w, {})])

    # ---- habit score ----------------------------------------------------
    def habit_score(u, s, variant, hl):
        f = feat.get((u, s))
        if f is None:
            return 0.0
        nd = len(f["days"])
        if variant == "a":
            return float(nd) * (0.5 ** ((REF - f["last"]) / (hl * 7 * 86400)))
        s_w = 0.0
        for wk in f["weeks"]:
            s_w += 0.5 ** ((21 - wk) / hl)
        if variant == "b":
            return s_w
        return s_w + 0.1 * math.log1p(f["ov"])

    # ---- config evaluation ---------------------------------------------
    def rank_user(u, W, cfg):
        variant, hl = cfg["habit"]
        gamma = cfg["gamma"]
        delta = cfg["delta"]
        padding = cfg["padding"]
        sw = cfg["switching"]
        C = windows[W]["C"]
        n_air = windows[W]["n_air"]
        air_slots = windows[W]["air_slots"]
        gtop5 = windows[W]["gtop5"]
        grank = windows[W]["grank"]
        prof = slot_profile.get(u, {})

        scored = []
        for s in user_train_items.get(u, ()):
            if s not in C:
                continue
            hb = habit_score(u, s, variant, hl)
            boost = 1.0 + gamma * math.log1p(n_air.get(s, 0))
            slot = 0.0
            if delta > 0:
                for (ch, hour) in air_slots.get(s, []):
                    slot += prof.get((ch, hour), 0.0)
            sc = hb * boost * (1.0 + delta * slot)
            scored.append((sc, s))
        scored.sort(key=lambda x: (-x[0], x[1]))

        if sw == "soft":
            wgt = 1.0 / (1.0 + train_shows.get(u, 0))
            combined = {}
            for r, (_sc, s) in enumerate(scored[:200], 1):
                combined[s] = (1.0 - wgt) / (60.0 + r)
            for r, s in enumerate(grank[:200], 1):
                combined[s] = combined.get(s, 0.0) + wgt / (60.0 + r)
            c2 = [(combined[s], s) for s in combined]
            c2.sort(key=lambda x: (-x[0], x[1]))
            return [s for _sc, s in c2[:5]]

        if sw is not None:  # hard
            T, D = sw
            if train_shows.get(u, 0) < T or train_days.get(u, 0) < D:
                return list(gtop5)

        top = [s for _sc, s in scored[:5]]
        if padding == "i":
            return pad5(top, gtop5)
        # ii / iii
        res = list(top)
        if len(res) < 5:
            if padding == "iii":
                ubases = {base_title[s] for s in user_train_items.get(u, ()) if base_title.get(s)}
                matched_new = sorted(s for s in new_ids if s in C and base_title.get(s) in ubases and s not in res)
                for s in matched_new:
                    if len(res) >= 5:
                        break
                    res.append(s)
            if len(res) < 5:
                remaining = [s for s in C if s not in res]
                remaining.sort(key=lambda s: (-sum(prof.get((ch, hour), 0.0) for (ch, hour) in air_slots.get(s, [])), s))
                for s in remaining:
                    if len(res) >= 5:
                        break
                    res.append(s)
        return pad5(res, gtop5)

    def eval_config(cfg):
        weekly = []
        for w in range(21, 31):
            W = f"w{w}"
            gt_w = windows[W]["gt"]
            cohort_w = windows[W]["cohort"]
            aps = [E.ap_at_k(rank_user(u, W, cfg), gt_w[u]) for u in cohort_w]
            weekly.append(float(np.mean(aps)))
        wk = float(np.mean(weekly))
        gtb = float(np.mean([E.ap_at_k(rank_user(u, "full", cfg), gtB[u]) for u in cohortB]))
        return wk, gtb

    # ---- greedy steps ---------------------------------------------------
    base_cfg = dict(habit=("a", 8), gamma=0.3, delta=0.0, padding="i", switching=(5, 2))
    base_wk, base_gtb = eval_config(base_cfg)
    out(f"БАЗА (v3): недельный {base_wk:.4f}, GT-B {base_gtb:.4f}")
    out()

    best_cfg = dict(base_cfg)
    best_wk, best_gtb = base_wk, base_gtb

    def report_step(title, options):
        nonlocal best_cfg, best_wk, best_gtb
        out(f"## {title}")
        out("| вариант | недельный MAP@5 | GT-B весь val | принят |")
        out("|---|---|---|---|")
        out(f"| БАЗА для сравнения | {best_wk:.4f} | {best_gtb:.4f} | — |")
        rows = []
        for name, cfg in options:
            wk, gtb = eval_config(cfg)
            rows.append((name, cfg, wk, gtb))
        rows.sort(key=lambda r: -r[2])
        accepted = False
        for name, cfg, wk, gtb in rows:
            if not accepted and wk >= best_wk + 0.002 and gtb >= best_gtb:
                mark = "да"
                accepted = True
                best_cfg = dict(cfg)
                best_wk, best_gtb = wk, gtb
            else:
                mark = ""
            out(f"| {name} | {wk:.4f} | {gtb:.4f} | {mark} |")
        out()

    # Step 1
    opts1 = []
    for var in ("a", "b", "c"):
        for hl in (6, 8, 12, 16, 24):
            opts1.append((f"{var} hl{hl}", dict(habit=(var, hl), gamma=0.3, delta=0.0, padding="i", switching=(5, 2))))
    report_step("Шаг 1: базовый скор привычки (вариант, hl)", opts1)

    # Step 2
    opts2 = [(f"gamma {g}", dict(habit=best_cfg["habit"], gamma=g, delta=best_cfg["delta"], padding=best_cfg["padding"], switching=best_cfg["switching"])) for g in (0.3, 0.5, 1.0, 2.0)]
    report_step("Шаг 2: буст по эфирам gamma", opts2)

    # Step 3
    opts3 = [(f"delta {d}", dict(habit=best_cfg["habit"], gamma=best_cfg["gamma"], delta=float(d), padding=best_cfg["padding"], switching=best_cfg["switching"])) for d in (0.0, 1.0, 3.0, 10.0)]
    report_step("Шаг 3: слотовый буст delta", opts3)

    # Step 4
    opts4 = [(f"padding {p}", dict(habit=best_cfg["habit"], gamma=best_cfg["gamma"], delta=best_cfg["delta"], padding=p, switching=best_cfg["switching"])) for p in ("i", "ii", "iii")]
    report_step("Шаг 4: добивка", opts4)

    # Step 5
    opts5 = []
    for T in (3, 5, 10):
        for D in (2, 3):
            opts5.append((f"hard T{T} D{D}", dict(habit=best_cfg["habit"], gamma=best_cfg["gamma"], delta=best_cfg["delta"], padding=best_cfg["padding"], switching=(T, D))))
    opts5.append(("soft", dict(habit=best_cfg["habit"], gamma=best_cfg["gamma"], delta=best_cfg["delta"], padding=best_cfg["padding"], switching="soft")))
    report_step("Шаг 5: переключение", opts5)

    out("## Итог")
    out(f"Принятый конфиг: {best_cfg}")
    out(f"недельный {best_wk:.4f} (база {base_wk:.4f}, прирост {best_wk - base_wk:+.4f}), GT-B {best_gtb:.4f} (база {base_gtb:.4f})")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tune_9d.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "tune_9d.md")


def air_slots_full(air_slots_week):
    d = defaultdict(list)
    for w, shows in air_slots_week.items():
        for s, lst in shows.items():
            d[s].extend(lst)
    return dict(d)


if __name__ == "__main__":
    main()
