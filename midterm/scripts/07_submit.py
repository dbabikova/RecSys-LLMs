"""Stage 7: final submission + reproducibility verification.

Phase 1 re-verifies the hybrid MAP@5 on val with deterministic tie-breaking.
Phase 2 retrains everything on weeks 11-30 and writes the submission CSVs.
"""

import hashlib
import csv
import json
import sys
import time
from collections import Counter, defaultdict
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
from join import parse_grid_time, parse_log_time, to_unix

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SUBMIT_START = to_unix(datetime(2020, 7, 27, 0, 0, 0))
SWITCH_T = 5
SWITCH_D = 2
# configurable via argv (defaults = v3)
HL_WEEKS = 8.0   # personal decay half-life in weeks (0 = none)
GAMMA = 0.3      # n_air boost gamma
WMF = 0.0        # MF+pop weight in hybrid
DELTA = 0.0      # slot boost delta
PPR_W = 0.0      # PPR weight in RRF (0 = off)
PPR_ALPHA = 0.5
BASE = "n_days"  # base score: n_days or n_days_comp
EXCLUDE_LONG = 0  # exclude airings longer than 10000s
OUT_NAME = "final_submission.csv"
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    if _args[_i] == "--hl":
        HL_WEEKS = float(_args[_i + 1]); _i += 2
    elif _args[_i] == "--gamma":
        GAMMA = float(_args[_i + 1]); _i += 2
    elif _args[_i] == "--wmf":
        WMF = float(_args[_i + 1]); _i += 2
    elif _args[_i] == "--delta":
        DELTA = float(_args[_i + 1]); _i += 2
    elif _args[_i] == "--ppr":
        PPR_W = float(_args[_i + 1]); _i += 2
    elif _args[_i] == "--base":
        BASE = _args[_i + 1]; _i += 2
    elif _args[_i] == "--exclude-long":
        EXCLUDE_LONG = int(_args[_i + 1]); _i += 2
    elif _args[_i] == "--out":
        OUT_NAME = _args[_i + 1]; _i += 2
    else:
        _i += 1
md: list[str] = []


def hour_of_week(ts):
    d = datetime.utcfromtimestamp(int(ts))
    return d.weekday() * 24 + d.hour


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


def combine(parts):
    V = np.hstack([w * M for w, M in parts if w and M.shape[1] > 0])
    return CBF.l2_normalize(V)


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


def load_submission_template():
    header = None
    ids = []
    with open(config.SUBMISSION_FILE, encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n")
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                ids.append(line.split(",", 1)[0])
    return header, ids


def load_submission_candidates():
    cand = set()
    titles = {}
    counts = Counter()
    for fp in (config.GRID_11_20_FILE, config.GRID_21_30_FILE, config.GRID_31_42_FILE):
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                sid = int(row[5])
                if sid == 0:
                    continue
                st = to_unix(parse_grid_time(row[2]))
                if st >= SUBMIT_START:
                    cand.add(sid)
                    titles.setdefault(sid, row[4])
                    counts[sid] += 1
    return cand, titles, dict(counts)


def load_submission_air_slots():
    d = defaultdict(list)
    for fp in (config.GRID_11_20_FILE, config.GRID_21_30_FILE, config.GRID_31_42_FILE):
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                sid = int(row[5])
                if sid == 0:
                    continue
                ch = int(row[0])
                st = to_unix(parse_grid_time(row[2]))
                dd = int(float(row[3]))
                if st >= SUBMIT_START:
                    d[sid].append((ch, hour_of_week(st), dd))
    return dict(d)


def load_slot_profile():
    slot_sec = defaultdict(lambda: defaultdict(float))
    with open(config.LOG_FILE, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            u = int(row[0])
            ch = int(row[1])
            s = to_unix(parse_log_time(row[2]))
            e = to_unix(parse_log_time(row[3]))
            if s >= SUBMIT_START:
                continue
            dd = e - s
            if dd > E.CAP:
                dd = E.CAP
            if dd <= 0:
                continue
            slot_sec[u][(ch, hour_of_week(s))] += dd
    prof = {}
    for u, slots in slot_sec.items():
        tot = sum(slots.values())
        prof[u] = {k: v / tot for k, v in slots.items()}
    return prof


def compute_ppr_mass(csr, cohort_users, user_index, alpha, max_iter=300):
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
    mass = {}
    for i, u in enumerate(valid):
        mass[u] = Ps[:, i]
    return mass


def write_submission(path, header, user_ids_str, rankings):
    with open(path, "w", newline="") as f:
        f.write(header + "\n")
        for ustr in user_ids_str:
            ids = rankings.get(int(ustr), [])
            f.write(f"{ustr},{' '.join(str(x) for x in ids[:5])}\n")


def main() -> None:
    out("# Финальный сабмит (этап 7)")
    out()

    header, sub_ids_str = load_submission_template()
    sub_ids_int = [int(u) for u in sub_ids_str]
    out(f"Шаблон: {len(sub_ids_str)} пользователей")
    out()

    print("Loading per-airing matches ...", flush=True)
    airings = E.load_labeled_airings(E.CAP)
    ov = airings["overlap"].to_numpy()
    dur = airings["duration"].to_numpy()
    vst = airings["view_start"].to_numpy()
    train_mask = vst < E.VAL_START
    val_mask = (vst >= E.VAL_START) & (vst <= E.VAL_END)
    watched = E.watched_mask_np(ov, dur)
    train_watched = airings[train_mask & watched]
    val_airings = airings[val_mask]
    all_watched = airings[watched]  # weeks 11-30
    del airings, ov, dur, vst, watched

    # ================= Phase 1: verify val hybrid =================
    out("## 1. Проверка воспроизводимости гибрида на val")
    out()
    gtsA = E.build_ground_truth(val_airings, "A")
    cohortA = [u for u in sub_ids_int if u in gtsA]

    days = defaultdict(set)
    for u, s, vs in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"]):
        days[(int(u), int(s))].add(int(vs) // 86400)
    n_days_train = {k: len(v) for k, v in days.items()}

    pop = defaultdict(set)
    for (u, s) in n_days_train:
        pop[s].add(u)
    pop_count = {s: len(v) for s, v in pop.items()}

    meta, val_cands, title_map, _ = CBF.load_show_metadata()
    global_rank_val = sorted([s for s in pop_count if s in val_cands], key=lambda s: (-pop_count[s], s))
    global_top5_val = global_rank_val[:5]

    t0 = time.time()
    csr, user_index, item_index, item_ids = MF.build_csr(n_days_train)
    X, Y, _ = MF.als(csr, 32, 50, 0.5, n_iter=15, seed=42)
    cand_mf = sorted(c for c in val_cands if c in item_index)
    cand_mf_idx = np.array([item_index[c] for c in cand_mf], dtype=np.int64)
    Yc = Y[cand_mf_idx]
    pop_log = np.array([np.log1p(pop_count.get(c, 0)) for c in cand_mf])

    rm_personal = {}
    rm_mfpop = {}
    for u in cohortA:
        sc = {s: n_days_train[(u, s)] for s in val_cands if (u, s) in n_days_train}
        rm_personal[u] = rank_map([sc[s] for s in sc], list(sc.keys()), top_n=200) if sc else {}
        ri = user_index.get(int(u))
        if ri is None:
            rm_mfpop[u] = {}
            continue
        s = X[ri] @ Yc.T
        mu = s.mean()
        sd = s.std()
        z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
        comb = z + 1.0 * pop_log
        rm_mfpop[u] = rank_map(comb, cand_mf, top_n=200)

    # hybrid: personal=1.0, mfpop=0.5
    aps = []
    for u in cohortA:
        show_set = {}
        for s, r in rm_personal.get(u, {}).items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][0] = 1.0 / (60.0 + r)
        for s, r in rm_mfpop.get(u, {}).items():
            show_set.setdefault(s, [0.0, 0.0])
            show_set[s][1] = 1.0 / (60.0 + r)
        w = np.array([1.0, 0.5])
        scored = [(show_set[s][0] * w[0] + show_set[s][1] * w[1], s) for s in show_set]
        scored.sort(key=lambda x: (-x[0], x[1]))
        pred = [s for _sc, s in scored[:5]]
        if len(pred) < 5:
            seen = set(pred)
            for gs in global_top5_val:
                if len(pred) >= 5:
                    break
                if gs not in seen:
                    pred.append(gs)
                    seen.add(gs)
        aps.append(E.ap_at_k(pred, gtsA[u]))
    map5_val = float(np.mean(aps))
    out(f"MAP@5 A (val, детерминированный пересчёт) = {map5_val:.6f}")
    out(f"Ожидалось 0.790556; {'СОВПАЛО' if abs(map5_val - 0.790556) < 1e-6 else 'изменилось: ' + str(map5_val)}")
    out(f"Время: {time.time() - t0:.1f}s")
    out()

    # ================= Phase 2: retrain on 11-30 =================
    out("## 2. Переобучение на неделях 11-30")
    out()
    days_all = defaultdict(set)
    for u, s, vs in zip(all_watched["user_id"], all_watched["tv_show_id"], all_watched["view_start"]):
        days_all[(int(u), int(s))].add(int(vs) // 86400)
    n_days_all = {k: len(v) for k, v in days_all.items()}
    train_items_all = set(s for (_u, s) in n_days_all)

    # completion-filtered distinct days (>= 0.8) over 11-30
    comp_days_all = defaultdict(set)
    aov = all_watched["overlap"].to_numpy()
    adu = all_watched["duration"].to_numpy()
    au = all_watched["user_id"].to_numpy()
    as_ = all_watched["tv_show_id"].to_numpy()
    avs = all_watched["view_start"].to_numpy()
    for i in range(len(au)):
        if aov[i] >= 0.8 * adu[i]:
            comp_days_all[(int(au[i]), int(as_[i]))].add(int(avs[i]) // 86400)
    n_days_comp_all = {k: len(v) for k, v in comp_days_all.items()}

    user_train_shows_all = defaultdict(set)
    for (u, s) in n_days_all:
        user_train_shows_all[u].add(s)
    user_train_days_all = defaultdict(set)
    for (u, s), ds in days_all.items():
        user_train_days_all[u].update(ds)
    train_shows_all = {u: len(v) for u, v in user_train_shows_all.items()}
    train_days_all_count = {u: len(v) for u, v in user_train_days_all.items()}
    last_seen_all = defaultdict(int)
    for u, s, vs in zip(all_watched["user_id"], all_watched["tv_show_id"], all_watched["view_start"]):
        k = (int(u), int(s))
        v = int(vs)
        if v > last_seen_all.get(k, 0):
            last_seen_all[k] = v

    pop_all = defaultdict(set)
    for (u, s) in n_days_all:
        pop_all[s].add(u)
    pop_count_all = {s: len(v) for s, v in pop_all.items()}

    sub_cands, sub_titles, sub_air = load_submission_candidates()
    sub_air_slots = load_submission_air_slots()
    slot_profile = load_slot_profile() if DELTA > 0 else {}
    out(f"Кандидатов 31-42: {len(sub_cands):,}")
    out()

    t_sub = time.time()
    # global
    global_rank = sorted([s for s in pop_count_all if s in sub_cands], key=lambda s: (-pop_count_all[s], s))
    global_top5 = global_rank[:5]

    # MF on 11-30
    csr_all, user_index_all, item_index_all, item_ids_all = MF.build_csr(n_days_all)
    X_all, Y_all, _ = MF.als(csr_all, 32, 50, 0.5, n_iter=15, seed=42)
    cand_mf_sub = sorted(c for c in sub_cands if c in item_index_all)
    cand_mf_sub_idx = np.array([item_index_all[c] for c in cand_mf_sub], dtype=np.int64)
    Yc_sub = Y_all[cand_mf_sub_idx]
    pop_log_sub = np.array([np.log1p(pop_count_all.get(c, 0)) for c in cand_mf_sub])

    # PPR (log1p weight, alpha=0.5)
    ppr_mass = {}
    if PPR_W > 0:
        print("Computing PPR ...", flush=True)
        n_days_log = {(u, s): np.log1p(nd) for (u, s), nd in n_days_all.items()}
        csr_log, uidx_log, iidx_log, iids_log = MF.build_csr(n_days_log)
        ppr_mass = compute_ppr_mass(csr_log, sub_ids_int, uidx_log, PPR_ALPHA)
        ppr_iids = iids_log

    # CBF-title blocks
    all_shows = sorted(train_items_all | sub_cands)
    show_index, _, _, _, _ = CBF.build_blocks(all_shows, meta)  # only need show_index
    base_onehot, title_tfidf = CBF.build_title_blocks(all_shows, title_map)
    Vt = combine([(10.0, base_onehot), (1.0, title_tfidf)])
    user_items_all = defaultdict(dict)
    for (u, s), nd in n_days_all.items():
        user_items_all[u][s] = nd
    prof_t = CBF.build_profiles(user_items_all, show_index, Vt)
    cand_cbf = sorted(c for c in sub_cands if c in show_index)
    cand_cbf_idx = np.array([show_index[c] for c in cand_cbf], dtype=np.int64)
    Vtc = Vt[cand_cbf_idx]

    # rank maps for hybrid
    rm_pers_sub = {}
    rm_mfpop_sub = {}
    global_rankings = {}
    personal_rankings = {}
    mfpop_rankings = {}
    cbft_rankings = {}
    hybrid_rankings = {}

    no_history = 0
    n_switched = 0
    for u in sub_ids_int:
        user_hist = user_items_all.get(u, {})
        if not user_hist:
            no_history += 1
        sc = {s: n_days_all[(u, s)] for s in sub_cands if (u, s) in n_days_all}
        pers_top = topk([sc[s] for s in sc], list(sc.keys()), k=5) if sc else []
        personal_rankings[u] = pad5(pers_top, global_top5)
        rm_pers_sub[u] = rank_map([sc[s] for s in sc], list(sc.keys()), top_n=200) if sc else {}

        ri = user_index_all.get(int(u))
        if ri is None:
            mfpop_rankings[u] = list(global_top5)
            rm_mfpop_sub[u] = {}
        else:
            s = X_all[ri] @ Yc_sub.T
            mu = s.mean()
            sd = s.std()
            z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
            comb = z + 1.0 * pop_log_sub
            mfpop_rankings[u] = pad5(topk(comb, cand_mf_sub, k=5), global_top5)
            rm_mfpop_sub[u] = rank_map(comb, cand_mf_sub, top_n=200)

        p = prof_t.get(int(u))
        if p is None or float(np.linalg.norm(p)) == 0.0:
            cbft_rankings[u] = list(global_top5)
        else:
            pn = p / np.linalg.norm(p)
            s = pn @ Vtc.T
            cbft_rankings[u] = pad5(topk(s, cand_cbf, k=5), global_top5)

        global_rankings[u] = list(global_top5)

        # hybrid = rank fusion personal(ps) + wmf*mfpop, + switching for light viewers
        if train_shows_all.get(u, 0) < SWITCH_T or train_days_all_count.get(u, 0) < SWITCH_D:
            hybrid_rankings[u] = list(global_top5)
            n_switched += 1
            continue
        scored = []
        prof_u = slot_profile.get(u, {})
        base_all = n_days_comp_all if BASE == "n_days_comp" else n_days_all
        for s in user_train_shows_all.get(u, ()):
            if s in sub_cands:
                nd = base_all.get((u, s), 0)
                dec = 1.0 if HL_WEEKS == 0 else 0.5 ** ((SUBMIT_START - last_seen_all.get((u, s), SUBMIT_START)) / (HL_WEEKS * 7 * 86400))
                slot = 0.0
                n_air_s = 0
                for (ch, hour, dd) in sub_air_slots.get(s, ()):
                    if EXCLUDE_LONG and dd > 10000:
                        continue
                    n_air_s += 1
                    if DELTA > 0:
                        slot += prof_u.get((ch, hour), 0.0)
                ps = nd * dec * (1.0 + GAMMA * np.log1p(n_air_s)) * (1.0 + DELTA * slot)
                scored.append((ps, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        rm_ps = {s: r for r, (_ps, s) in enumerate(scored[:200], 1)}
        show_set = {}
        for s, r in rm_ps.items():
            show_set[s] = 1.0 / (60.0 + r)
        if PPR_W > 0 and u in ppr_mass:
            col = ppr_mass[u]
            lst = [(col[j], ppr_iids[j]) for j in range(len(ppr_iids)) if ppr_iids[j] in sub_cands and col[j] > 0]
            lst.sort(key=lambda x: (-x[0], x[1]))
            for r, (_m, s) in enumerate(lst[:200], 1):
                show_set[s] = show_set.get(s, 0.0) + PPR_W / (60.0 + r)
        for s, r in rm_mfpop_sub[u].items():
            show_set[s] = show_set.get(s, 0.0) + WMF / (60.0 + r)
        scored2 = [(show_set[s], s) for s in show_set]
        scored2.sort(key=lambda x: (-x[0], x[1]))
        hybrid_rankings[u] = pad5([s for _sc, s in scored2[:5]], global_top5)

    t_sub = time.time() - t_sub
    out(f"Пользователей без истории (fallback): {no_history} из {len(sub_ids_int)}")
    out(f"Переключено на global (train_shows<{SWITCH_T} ИЛИ train_days<{SWITCH_D}): {n_switched}")
    out(f"Время обучения/предсказания: {t_sub:.1f}s")
    out()

    # ================= write CSVs =================
    subs_dir = config.SUBMISSIONS_DIR
    subs_dir.mkdir(parents=True, exist_ok=True)
    write_submission(subs_dir / OUT_NAME, header, sub_ids_str, hybrid_rankings)
    write_submission(subs_dir / "global.csv", header, sub_ids_str, global_rankings)
    write_submission(subs_dir / "personal.csv", header, sub_ids_str, personal_rankings)
    write_submission(subs_dir / "mf_pop.csv", header, sub_ids_str, mfpop_rankings)
    write_submission(subs_dir / "cbf_title.csv", header, sub_ids_str, cbft_rankings)

    # ================= checks =================
    out("## Проверки")
    out()
    final_path = subs_dir / OUT_NAME
    lines = final_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1261, f"expected 1261 lines (header + 1260), got {len(lines)}"
    out("PASS: 1260 строк (+ заголовок)")

    tmpl_lines = [f"{u}," for u in sub_ids_str]
    data_lines = lines[1:]
    assert len(data_lines) == 1260
    for i, (dl, ustr) in enumerate(zip(data_lines, sub_ids_str)):
        assert dl.startswith(ustr + ","), f"row {i}: user_id mismatch"
    out("PASS: user_id совпадают с шаблоном построчно как строки")

    all_ids = set()
    global_match = 0
    for dl in data_lines:
        ids = [int(x) for x in dl.split(",", 1)[1].split()]
        assert len(ids) == 5 and len(set(ids)) == 5, f"not 5 unique: {dl}"
        assert all(x != 0 for x in ids), f"zero in {dl}"
        assert all(x in sub_cands for x in ids), f"id not in 31-42 candidates: {dl}"
        all_ids.update(ids)
        if set(ids) == set(global_top5):
            global_match += 1
    out("PASS: в каждой строке 5 уникальных id")
    out("PASS: нет 0")
    out("PASS: все id среди кандидатов 31-42")

    df_read = pd.read_csv(final_path, dtype=str)
    assert df_read.shape[0] == 1260
    assert list(df_read["user_id"]) == sub_ids_str
    out("PASS: файл читается обратно pandas (dtype=str) без потерь")

    md5 = hashlib.md5(final_path.read_bytes()).hexdigest()
    out(f"md5(final_submission.csv) = {md5}")
    out()

    # ================= summary =================
    out("## Сводка")
    out()
    total_slots = 0
    repeats = 0
    for u in sub_ids_int:
        hist = set(user_items_all.get(u, {}))
        for s in hybrid_rankings[u][:5]:
            total_slots += 1
            if s in hist:
                repeats += 1
    out(f"Пользователей без истории: {no_history}")
    out(f"Доля повторов в Top-5 (final): {repeats / total_slots * 100:.1f}%")
    out(f"Уникальных id во всех строках final: {len(all_ids)}")
    out(f"Строк, совпадающих с global Top-5: {global_match}")
    out(f"Время (обучение+предсказание): {t_sub:.1f}s")
    out()

    import platform
    import scipy, sklearn
    out("## Версии")
    out(f"Python {platform.python_version()}")
    out(f"numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "submission_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "submission_summary.md")


if __name__ == "__main__":
    main()
