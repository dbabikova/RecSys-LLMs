"""Stage 10: GT-D80 calibration and h1 method (efficient)."""

import csv
import hashlib
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
from join import parse_log_time, parse_grid_time, to_unix

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


def hour_fractions(t, d):
    d = min(d, 6 * 3600)
    d = max(d, 1e-9)
    res = []
    hw = hour_of_week(t)
    sec_in_hour = 3600 - (int(t) % 3600)
    remaining = d
    while remaining > 1e-6:
        take = min(remaining, sec_in_hour)
        res.append((hw, take / d))
        remaining -= take
        sec_in_hour = 3600
        hw = (hw + 1) % 168
    return res


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


def load_grid():
    sids, chs, sts, durs = [], [], [], []
    for fp in (config.GRID_11_20_FILE, config.GRID_21_30_FILE, config.GRID_31_42_FILE):
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                sid = int(row[5])
                if sid == 0:
                    continue
                sids.append(sid)
                chs.append(int(row[0]))
                sts.append(to_unix(parse_grid_time(row[2])))
                durs.append(int(float(row[3])))
    return (np.array(sids, dtype=np.int64), np.array(chs, dtype=np.int64),
            np.array(sts, dtype=np.int64), np.array(durs, dtype=np.int64))


def load_log():
    us, chs, sts, ens = [], [], [], []
    with open(config.LOG_FILE, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            us.append(int(row[0]))
            chs.append(int(row[1]))
            sts.append(to_unix(parse_log_time(row[2])))
            ens.append(to_unix(parse_log_time(row[3])))
    return (np.array(us, dtype=np.uint64), np.array(chs, dtype=np.int64),
            np.array(sts, dtype=np.int64), np.array(ens, dtype=np.int64))


def build_gt_d80(matches, REF, END):
    w = matches[(matches["air_start"] >= REF) & (matches["air_start"] < END)]
    g = w.groupby(["user_id", "tv_show_id", "channel", "air_start"]).agg(
        ov=("overlap", "sum"), dur=("duration", "first"))
    g["pct"] = (g["ov"] / g["dur"]).clip(0, 1)
    g = g[g["pct"] >= 0.8].reset_index()
    cnt = g.groupby(["user_id", "tv_show_id"]).size().reset_index(name="cnt")
    gt = {}
    for u, grp in cnt.groupby("user_id"):
        top = grp.sort_values(["cnt", "tv_show_id"], ascending=[False, True]).head(5)["tv_show_id"].tolist()
        gt[int(u)] = set(int(x) for x in top)
    return gt


def compute_p(matches, REF):
    h = matches[matches["air_start"] < REF]
    g = h.groupby(["user_id", "tv_show_id", "channel", "air_start"]).agg(
        ov=("overlap", "sum"), dur=("duration", "first"), vst=("view_start", "min"))
    g["pct"] = (g["ov"] / g["dur"]).clip(0, 1)
    g["age"] = (REF - g["vst"]) / (7 * 86400)
    g["contrib"] = g["pct"] * (0.5 ** (g["age"] / 4.0))
    s = g.reset_index().groupby(["user_id", "tv_show_id"])["contrib"].sum()
    return {(int(u), int(x)): float(v) for (u, x), v in s.items()}


def compute_global_pop(matches, REF):
    h = matches[matches["air_start"] < REF]
    g = h.groupby(["user_id", "tv_show_id", "channel", "air_start"]).agg(
        ov=("overlap", "sum"), dur=("duration", "first"))
    g["pct"] = (g["ov"] / g["dur"]).clip(0, 1)
    g = g[(g["ov"] >= 300) | (g["pct"] >= 0.5)].reset_index()
    return g.groupby("tv_show_id")["user_id"].nunique().to_dict()


def slot_profile_from_log(log, REF):
    us, chs, sts, ens = log
    mask = sts < REF
    us, chs, sts, ens = us[mask], chs[mask], sts[mask], ens[mask]
    dur = (ens - sts).clip(0, E.CAP)
    slot_sec = defaultdict(lambda: defaultdict(float))
    for u, ch, st, d in zip(us.tolist(), chs.tolist(), sts.tolist(), dur.tolist()):
        slot_sec[u][(ch, hour_of_week(st))] += d
    prof = {}
    for u, slots in slot_sec.items():
        tot = sum(slots.values())
        prof[u] = {k: v / tot for k, v in slots.items()}
    return prof


def h1_rankings(grid, log, matches, REF, END):
    t0 = time.time()
    sids, chs, sts, durs = grid
    wmask = (sts >= REF) & (sts < END)
    w_sids, w_chs, w_sts, w_durs = sids[wmask], chs[wmask], sts[wmask], durs[wmask]
    print(f"  window airings {len(w_sids)} ({time.time()-t0:.0f}s)", flush=True)

    p = compute_p(matches, REF)
    print(f"  p computed ({time.time()-t0:.0f}s)", flush=True)
    slot_prof = slot_profile_from_log(log, REF)
    print(f"  slot profile ({time.time()-t0:.0f}s)", flush=True)
    pop = compute_global_pop(matches, REF)
    print(f"  global pop ({time.time()-t0:.0f}s)", flush=True)

    window_shows = set(w_sids.tolist())
    global_rank = sorted([s for s in pop if s in window_shows], key=lambda s: (-pop[s], s))
    gtop5 = global_rank[:5]

    # show -> list of (channel, hour, weight)
    show_slots = defaultdict(list)
    for sid, ch, st, dur in zip(w_sids.tolist(), w_chs.tolist(), w_sts.tolist(), w_durs.tolist()):
        for (hw, frac) in hour_fractions(st, dur):
            show_slots[sid].append((ch, hw, frac))

    user_shows = defaultdict(list)
    for (u, s) in p:
        user_shows[u].append(s)

    rankings = {}
    for u, shows in user_shows.items():
        prof = slot_prof.get(u, {})
        scored = []
        for s in shows:
            if s not in window_shows:
                continue
            ps = p[(u, s)]
            slot = 0.0
            for (ch, hw, w) in show_slots[s]:
                slot += prof.get((ch, hw), 0.0) * w
            sc = ps * math.sqrt(slot + 1e-4)
            if sc > 0:
                scored.append((sc, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [s for _sc, s in scored[:5]]
        rankings[u] = pad5(top, gtop5)
    print(f"  h1 rankings done ({time.time()-t0:.0f}s)", flush=True)
    return rankings, gtop5


def main():
    out("# Этап 10: GT-D80, калибровка и h1")
    out()

    sub_int = sorted(int(u) for u in E.load_submission_ids())
    sub_set = set(sub_int)

    print("Loading per-airing matches ...", flush=True)
    matches = E.load_labeled_airings(E.CAP)
    print("Loading grid ...", flush=True)
    grid = load_grid()
    print("Loading log ...", flush=True)
    log = load_log()

    REF = to_unix(datetime(2020, 5, 18, 0, 0, 0))
    END = to_unix(datetime(2020, 7, 27, 0, 0, 0))

    gt_d80 = build_gt_d80(matches, REF, END)
    cohort = [u for u in sub_int if u in gt_d80]
    out(f"Когорта D80: {len(cohort)} из {len(sub_int)}")
    out()

    def map5(rankings, gt, coh):
        return float(np.mean([E.ap_at_k(rankings.get(u, []), gt[u]) for u in coh]))

    # ---- history: rows with "watched" (overlap >= 300 s or pct >= 0.5), view_start < REF
    ov = matches["overlap"].to_numpy()
    du = matches["duration"].to_numpy()
    vs_all = matches["view_start"].to_numpy()
    tw = matches[(vs_all < REF) & E.watched_mask_np(ov, du)]
    feat_days = defaultdict(set)
    last_seen = {}
    for u, s, vs in zip(tw["user_id"].tolist(), tw["tv_show_id"].tolist(), tw["view_start"].tolist()):
        k = (int(u), int(s))
        feat_days[k].add(int(vs) // 86400)
        if int(vs) > last_seen.get(k, 0):
            last_seen[k] = int(vs)
    n_days = {k: len(v) for k, v in feat_days.items()}
    user_items = defaultdict(set)
    user_days = defaultdict(set)
    for (u, s), ds in feat_days.items():
        user_items[u].add(s)
        user_days[u].update(ds)

    # ---- window: candidates, airings count, (channel, start hour) per airing
    sids, chs, sts, _durs = grid
    wmask = (sts >= REF) & (sts < END)
    window_shows = set(sids[wmask].tolist())
    n_air = defaultdict(int)
    air_slots = defaultdict(list)
    for sid, ch, st in zip(sids[wmask].tolist(), chs[wmask].tolist(), sts[wmask].tolist()):
        n_air[sid] += 1
        air_slots[sid].append((ch, hour_of_week(st)))

    # ---- popularity: unique COHORT users (1260) who watched the show before REF, window shows only
    pop = defaultdict(set)
    for (u, s) in n_days:
        if u in sub_set:
            pop[s].add(u)
    grank = sorted([s for s in pop if s in window_shows], key=lambda s: (-len(pop[s]), s))
    gtop5 = grank[:5]
    out(f"global top-5 (когорта 1260): {gtop5}")

    slot_prof = slot_profile_from_log(log, REF)

    # ---- PPR (bipartite, log1p, alpha 0.5) for v5+PPR
    print("PPR ...", flush=True)
    n_days_log = {k: math.log1p(v) for k, v in n_days.items()}
    csr_log, uidx, _iidx, iids = MF.build_csr(n_days_log)
    mass_ppr = compute_ppr_bip(csr_log, cohort, uidx, 0.5)

    R = {"global": {}, "personal(n_days)": {}, "v3": {}, "v5 (slot300, без PPR)": {}, "v5+PPR0.25": {}}
    n_switch = 0
    for u in cohort:
        R["global"][u] = list(gtop5)
        cands = sorted(s for s in user_items.get(u, ()) if s in window_shows)
        pers = sorted(cands, key=lambda s: (-n_days[(u, s)], s))
        R["personal(n_days)"][u] = pad5(pers[:5], gtop5)

        if len(user_items.get(u, ())) < 5 or len(user_days.get(u, ())) < 2:
            n_switch += 1
            for k in ("v3", "v5 (slot300, без PPR)", "v5+PPR0.25"):
                R[k][u] = list(gtop5)
            continue
        prof = slot_prof.get(u, {})
        sc3, sc5 = {}, {}
        for s in cands:
            dec = 0.5 ** ((REF - last_seen[(u, s)]) / (8 * 7 * 86400))
            base = n_days[(u, s)] * dec * (1.0 + 0.3 * math.log1p(n_air[s]))
            slot = sum(prof.get(cs, 0.0) for cs in air_slots[s])
            sc3[s] = base
            sc5[s] = base * (1.0 + 300.0 * slot)
        l3 = sorted(cands, key=lambda s: (-sc3[s], s))
        l5 = sorted(cands, key=lambda s: (-sc5[s], s))
        R["v3"][u] = pad5(l3[:5], gtop5)
        R["v5 (slot300, без PPR)"][u] = pad5(l5[:5], gtop5)
        fused = {s: 1.0 / (60.0 + r) for r, s in enumerate(l5[:200], 1)}
        if u in mass_ppr:
            col = mass_ppr[u]
            lst = [(col[j], iids[j]) for j in range(len(iids)) if iids[j] in window_shows and col[j] > 0]
            lst.sort(key=lambda x: (-x[0], x[1]))
            for r, (_m, s) in enumerate(lst[:200], 1):
                fused[s] = fused.get(s, 0.0) + 0.25 / (60.0 + r)
        lf = sorted(fused, key=lambda s: (-fused[s], s))
        R["v5+PPR0.25"][u] = pad5(lf[:5], gtop5)
    out(f"switching (<5 передач или <2 дней до REF): {n_switch} пользователей")

    # ---- diagnostic: v3 logic of the previous version of this script (unfiltered history, all-user pop)
    hist = matches[matches["air_start"] < REF]
    od = defaultdict(set)
    ol = {}
    for u, s, vs in zip(hist["user_id"].tolist(), hist["tv_show_id"].tolist(), hist["view_start"].tolist()):
        k = (int(u), int(s))
        od[k].add(int(vs) // 86400)
        if int(vs) > ol.get(k, 0):
            ol[k] = int(vs)
    o_items = defaultdict(list)
    for (u, s) in od:
        o_items[u].append(s)
    old_v3 = {}
    for u in cohort:
        c = [s for s in o_items.get(u, []) if s in window_shows]
        c.sort(key=lambda s: (-len(od[(u, s)]) * 0.5 ** ((REF - ol[(u, s)]) / (8 * 7 * 86400))
                              * (1.0 + 0.3 * math.log1p(n_air[s])), s))
        old_v3[u] = pad5(c[:5], gtop5)
    n_eq_old = sum(old_v3[u] == R["global"][u] for u in cohort)
    n_eq_new = sum(R["v3"][u] == R["global"][u] for u in cohort)
    out(f"диагностика: v3 текущего кода прошлой версии = global у {n_eq_old}/{len(cohort)}, "
        f"MAP D80 {map5(old_v3, gt_d80, cohort):.4f}; новый v3 = global у {n_eq_new}/{len(cohort)}")
    out()

    # ---- h1 on val (unchanged)
    print("h1 val ...", flush=True)
    h1_r, h1_gtop = h1_rankings(grid, log, matches, REF, END)
    for u in cohort:
        if u not in h1_r:
            h1_r[u] = list(h1_gtop)
    R["h1"] = h1_r

    expected = {"global": 0.273, "personal(n_days)": 0.332, "v3": 0.351,
                "v5 (slot300, без PPR)": 0.377, "v5+PPR0.25": None, "h1": 0.401}
    kaggle = {"global": 0.2781, "personal(n_days)": 0.3319, "v3": 0.36222,
              "v5 (slot300, без PPR)": None, "v5+PPR0.25": 0.394, "h1": 0.41508}
    d80 = {k: map5(R[k], gt_d80, cohort) for k in R}
    out("| метод | D80 (val) | эталон | Kaggle private |")
    out("|---|---|---|---|")
    for k in R:
        e = f"{expected[k]:.3f}" if expected[k] is not None else "—"
        kg = f"{kaggle[k]}" if kaggle[k] is not None else "—"
        out(f"| {k} | {d80[k]:.4f} | {e} | {kg} |")
    out()

    from scipy.stats import spearmanr
    ks1 = ["global", "personal(n_days)", "v3", "v5+PPR0.25", "h1"]
    rho1 = spearmanr([d80[k] for k in ks1], [kaggle[k] for k in ks1]).correlation
    ks2 = ["global", "personal(n_days)", "v3", "v5 (slot300, без PPR)", "h1"]
    kg2 = [kaggle[k] if kaggle[k] is not None else kaggle["v5+PPR0.25"] for k in ks2]
    rho2 = spearmanr([d80[k] for k in ks2], kg2).correlation
    out(f"Спирмен D80 vs Kaggle (5 сабмитов, v5 = с PPR): {rho1:.3f}")
    out(f"Спирмен, если v5 без PPR сопоставить с Kaggle v5 0.394: {rho2:.3f}")
    out()

    lvk = ["# Локальная метрика D80 (val) vs Kaggle private", "",
           "GT-D80: топ-5 передач по числу эфиров с pct >= 0.8 в окне [2020-05-18, 2020-07-27); "
           f"когорта {len(cohort)} пользователей. Добивка/global — популярность среди 1260 пользователей когорты.", "",
           "| метод | D80 (val) | эталон D80 | Kaggle private |", "|---|---|---|---|"]
    for k in R:
        e = f"{expected[k]:.3f}" if expected[k] is not None else "—"
        kg = f"{kaggle[k]}" if kaggle[k] is not None else "— (сабмит v5 был с PPR)"
        lvk.append(f"| {k} | {d80[k]:.4f} | {e} | {kg} |")
    lvk += ["", f"Спирмен D80 vs Kaggle по 5 сабмитам (global, personal, v3, v5+PPR, h1): **{rho1:.3f}**",
            f"Вариант с v5 без PPR вместо v5+PPR: {rho2:.3f}", ""]
    (config.RESULTS_DIR / "local_vs_kaggle.md").write_text("\n".join(lvk) + "\n", encoding="utf-8")

    # ---- all_methods.csv: MAP5_D80 column + rows
    valw = matches[(matches["air_start"] >= REF) & (matches["air_start"] < END)]
    gts_abc = {d: E.build_ground_truth(valw, d) for d in ("A", "B", "C")}
    coh_abc = {d: [u for u in sub_int if u in gts_abc[d]] for d in ("A", "B", "C")}
    df = pd.read_csv(config.RESULTS_DIR / "all_methods.csv")
    if "MAP5_D80" not in df.columns:
        df["MAP5_D80"] = np.nan
    # global / personal rows of earlier stages use other definitions -> separate "(D80 calib)" rows
    names = {"global": "global(pop-cohort, D80 calib)", "personal(n_days)": "personal(n_days, D80 calib)",
             "v3": "v3", "v5 (slot300, без PPR)": "v5(slot300)", "v5+PPR0.25": "v5(slot300+PPR0.25)", "h1": "h1"}
    for k, name in names.items():
        row = {"method": name, "MAP5_D80": d80[k]}
        if k == "h1":  # h1 covers all users; A/B/C kept for continuity
            rk = {u: h1_r.get(u, list(h1_gtop)) for d in ("A", "B", "C") for u in coh_abc[d]}
            for d in ("A", "B", "C"):
                row[f"MAP5_{d}"] = map5(rk, gts_abc[d], coh_abc[d])
        row["coverage"] = len(set(s for u in cohort for s in R[k][u]))
        df = df[df["method"] != name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(config.RESULTS_DIR / "all_methods.csv", index=False)
    out("all_methods.csv: добавлена колонка MAP5_D80 и строки " + ", ".join(names.values()))
    out("(A/B/C для калибровочных строк не считались — они построены только на когорте D80)")
    out()

    # ---- submission h1_oc is NOT regenerated; only compare existing files
    subs_dir = config.SUBMISSIONS_DIR
    oc = subs_dir / "final_submission_h1_oc.csv"
    ref = subs_dir / "final_submission_h1.csv"
    out("## Сабмит h1_oc (не пересоздавался)")
    out(f"md5 h1_oc = {hashlib.md5(oc.read_bytes()).hexdigest()}, эталон = {hashlib.md5(ref.read_bytes()).hexdigest()}")
    v = lambda d: [tuple(int(x) for x in s.split()) for s in d["tv_show_id"]]
    a, b = v(pd.read_csv(ref, dtype=str)), v(pd.read_csv(oc, dtype=str))
    out(f"Отличаются {sum(x != y for x, y in zip(a, b))} пользователей / "
        f"{sum(i != j for x, y in zip(a, b) for i, j in zip(x, y))} позиций; "
        f"по составу топ-5 — {sum(set(x) != set(y) for x, y in zip(a, b))}")
    out()

    (config.RESULTS_DIR / "tune_10.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСохранено в", config.RESULTS_DIR / "tune_10.md")


def submit(out_name: str) -> None:
    """Build the h1 submission: REF = 2020-07-27, END = REF + 12 weeks, history = all views of weeks 11-30."""
    print("Loading per-airing matches / grid / log ...", flush=True)
    matches = E.load_labeled_airings(E.CAP)
    grid = load_grid()
    log = load_log()
    REF2 = to_unix(datetime(2020, 7, 27, 0, 0, 0))
    END2 = REF2 + 12 * 7 * 86400
    h1_sub, h1_sub_gtop = h1_rankings(grid, log, matches, REF2, END2)

    with open(config.SUBMISSION_FILE, encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n")
        sub_ids_str = [l.rstrip("\r\n").split(",", 1)[0] for l in f if l.strip()]
    out_path = config.SUBMISSIONS_DIR / out_name
    window2 = set(grid[0][(grid[2] >= REF2) & (grid[2] < END2)].tolist())
    with open(out_path, "w", newline="") as f:
        f.write(header + "\n")
        for ustr in sub_ids_str:
            ids = h1_sub.get(int(ustr), list(h1_sub_gtop))[:5]
            f.write(f"{ustr},{' '.join(str(x) for x in ids)}\n")

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1261
    assert all(l.split(",", 1)[0] == u for l, u in zip(lines[1:], sub_ids_str))
    assert all(len(set(l.split(",", 1)[1].split())) == 5 for l in lines[1:])
    assert all(0 not in [int(x) for x in l.split(",", 1)[1].split()] for l in lines[1:])
    assert all(all(int(x) in window2 for x in l.split(",", 1)[1].split()) for l in lines[1:])
    assert pd.read_csv(out_path, dtype=str).shape[0] == 1260
    print("PASS: 1260 rows, user_id order, 5 unique ids, no 0, all ids air in the window, pandas")
    print(f"{out_path.name} md5 = {hashlib.md5(out_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    # python scripts/10_h1.py                 -> GT-D80 calibration + h1 on val (results/)
    # python scripts/10_h1.py --submit [name] -> h1 submission (default final_submission_h1_oc.csv)
    if "--submit" in sys.argv:
        i = sys.argv.index("--submit")
        submit(sys.argv[i + 1] if len(sys.argv) > i + 1 else "final_submission_h1_oc.csv")
    else:
        main()
