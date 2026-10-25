"""Stage 5 driver: content-based filtering + method comparison."""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
import evaluate as E
import mf_als as MF
import cbf as CBF

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


def build_personal_rankings(n_days, candidates, global_rank, cohort):
    user_items = defaultdict(list)
    for (u, s) in n_days:
        if s in candidates:
            user_items[u].append(s)
    rankings: dict[int, list[int]] = {}
    for u in cohort:
        items = sorted(user_items.get(u, []), key=lambda s: (-n_days[(u, s)], s))
        top = items[:5]
        if len(top) < 5:
            seen = set(top)
            for gs in global_rank:
                if len(top) >= 5:
                    break
                if gs not in seen:
                    top.append(gs)
                    seen.add(gs)
        rankings[u] = top
    return rankings


def repeat_fraction(rankings, user_train_items, cohort):
    total = 0
    reps = 0
    for u in cohort:
        ti = user_train_items.get(u, set())
        for s in rankings.get(u, [])[:5]:
            total += 1
            if s in ti:
                reps += 1
    return reps / total if total else 0.0


def main() -> None:
    out("# Content-based (этап 5)")
    out()

    sub_int = sorted(int(u) for u in E.load_submission_ids())

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
    del airings, ov, dur, vst, watched

    gts = {d: E.build_ground_truth(val_airings, d) for d in ("A", "B", "C")}
    cohorts = {d: [u for u in sub_int if u in gts[d]] for d in ("A", "B", "C")}

    days = defaultdict(set)
    for u, s, vs in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"]):
        days[(int(u), int(s))].add(int(vs) // 86400)
    n_days = {k: len(v) for k, v in days.items()}
    train_items = set(s for (_u, s) in n_days)

    pop_count = defaultdict(set)
    for (u, s) in n_days:
        pop_count[s].add(u)
    pop_count = {s: len(v) for s, v in pop_count.items()}

    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)

    # ---- task 0: MF extra configs --------------------------------------
    out("## 0. MF: 4 доп. конфига (factors=32, reg {100,200}, alpha {0.25,0.5})")
    out()
    csr, user_index, item_index, item_ids = MF.build_csr(n_days)

    meta, candidates, title_map = CBF.load_show_metadata()
    global_rank = sorted([s for s in pop_count if s in candidates], key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]
    personal_rankings = build_personal_rankings(n_days, candidates, global_rank, cohorts["A"])

    best_mf = {"cfg": (32, 50, 0.5), "map5": 0.747876, "X": None, "Y": None, "rankings": None}
    extra_cfgs = [(32, r, a) for r in (100, 200) for a in (0.25, 0.5)]
    # run known best + extras to obtain latent factors
    for f, r, a in [(32, 50, 0.5)] + extra_cfgs:
        t0 = time.time()
        X, Y, losses = MF.als(csr, f, r, a, n_iter=15, seed=42, verbose=False)
        mono = all(losses[i] <= losses[i - 1] + 1e-6 * abs(losses[i - 1]) for i in range(1, len(losses)))
        rankings, n_pad = MF.predict_rankings(X, Y, user_index, item_ids, candidates, global_rank, cohorts["A"])
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        tag = "" if (f, r, a) == (32, 50, 0.5) else "  (extra)"
        out(f"MF({f},{r},{a}){tag}: MAP@5 A = {ra['map5']:.6f} (mono={mono}, {time.time()-t0:.1f}s)")
        if (f, r, a) != (32, 50, 0.5) and ra["map5"] > best_mf["map5"]:
            best_mf = {"cfg": (f, r, a), "map5": ra["map5"], "X": X, "Y": Y, "rankings": rankings}
    if best_mf["cfg"] == (32, 50, 0.5):
        out("Ни один из доп. конфигов не побил 0.748 → оптимум внутри сетки (32, 50, 0.5).")
    else:
        out(f"Новый лучший MF: {best_mf['cfg']} (MAP@5 A = {best_mf['map5']:.6f}).")
    out()

    # latent factors for the best MF
    if best_mf["X"] is None:
        Xb, Yb, _ = MF.als(csr, *best_mf["cfg"], n_iter=15, seed=42)
        best_mf["X"], best_mf["Y"] = Xb, Yb
        best_mf["rankings"], _ = MF.predict_rankings(Xb, Yb, user_index, item_ids, candidates, global_rank, cohorts["A"])
    mf_rankings = best_mf["rankings"]
    # MF+pop with beta=1
    mf_pop_rankings = _mf_pop(best_mf["X"], best_mf["Y"], user_index, item_ids, candidates, pop_count, global_rank, cohorts["A"], 1.0)

    # ---- CBF ------------------------------------------------------------
    out(f"Метаданные: {len(meta):,} передач; кандидатов val: {len(candidates):,}; новых (в val, нет в train): {len(candidates - train_items):,}")
    out()

    all_shows = sorted(train_items | candidates)
    show_index, cat_mat, genre_mat, year_mat, people_mat = CBF.build_blocks(all_shows, meta)
    user_items = defaultdict(dict)
    for (u, s), nd in n_days.items():
        user_items[u][s] = nd

    weight_sets = [
        ("cat+genre", 1.0, 1.0, 0.0, 0.0),
        ("cat+genre+year", 1.0, 1.0, 1.0, 0.0),
        ("cat+genre+year+people", 1.0, 1.0, 1.0, 1.0),
    ]

    out("## CBF: наборы весов блоков")
    out()
    out("| блоки | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие | доля повторов |")
    out("|---|---|---|---|---|---|---|")
    cbf_results = {}
    for name, wc, wg, wy, wp in weight_sets:
        V = CBF.combine_blocks(cat_mat, genre_mat, year_mat, people_mat, wc, wg, wy, wp)
        profiles = CBF.build_profiles(user_items, show_index, V)
        rankings, n_pad = CBF.recommend_cbf(profiles, V, show_index, candidates, global_rank, cohorts["A"])
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        rf = repeat_fraction(rankings, user_train_items, cohorts["A"])
        cbf_results[name] = (V, profiles, rankings, ra, rb, rc)
        out(f"| {name} | {ra['map5']:.6f} | {rb['map5']:.6f} | {rc['map5']:.6f} | {ra['recall5']:.6f} | {ra['coverage']:,} | {rf*100:.1f}% |")
    out()

    best_cbf_name = max(cbf_results, key=lambda k: cbf_results[k][3]["map5"])
    Vb, profiles_b, cbf_rankings, ra_b, rb_b, rc_b = cbf_results[best_cbf_name]
    out(f"Лучший CBF: {best_cbf_name} (MAP@5 A = {ra_b['map5']:.6f})")
    out()

    # ---- CBF + pop ------------------------------------------------------
    out("## CBF + популярность (z-score + beta * log1p(pop_train))")
    out()
    out("| beta | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие |")
    out("|---|---|---|---|---|---|")
    cbf_pop = {}
    for beta in (0.5, 1.0):
        rankings, n_pad = CBF.recommend_cbf_pop(profiles_b, Vb, show_index, candidates, pop_count, global_rank, cohorts["A"], beta)
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        cbf_pop[beta] = (rankings, ra, rb, rc)
        out(f"| {beta} | {ra['map5']:.6f} | {rb['map5']:.6f} | {rc['map5']:.6f} | {ra['recall5']:.6f} | {ra['coverage']:,} |")
    out()

    # ---- new shows ------------------------------------------------------
    new_shows = candidates - train_items
    total_gt = sum(len(gts["A"][u]) for u in cohorts["A"])
    new_gt = sum(len(gts["A"][u] & new_shows) for u in cohorts["A"])
    cbf_new_hits = sum(1 for u in cohorts["A"] for s in cbf_rankings[u][:5] if s in new_shows)
    mf_new_hits = sum(1 for u in cohorts["A"] for s in mf_rankings[u][:5] if s in new_shows)
    rec_new = []
    for u in cohorts["A"]:
        rel_new = gts["A"][u] & new_shows
        if rel_new:
            hits = sum(1 for s in cbf_rankings[u][:5] if s in rel_new)
            rec_new.append(hits / len(rel_new))
    out("## Новые передачи (в val, нет в train)")
    out()
    out(f"Число новых передач: {len(new_shows):,}")
    out(f"Доля GT-A ground truth на них: {new_gt / total_gt * 100:.2f}%")
    out(f"Попаданий в Top-5: CBF = {cbf_new_hits}, MF = {mf_new_hits} (ожидается 0 у MF)")
    out(f"Recall@5 только по новым (CBF): {np.mean(rec_new):.6f} (у {len(rec_new)} пользователей с новыми в GT)")
    out()

    # ---- final comparison ----------------------------------------------
    gr = {u: list(global_top5) for u in cohorts["A"]}
    cbf_pop_best = max(cbf_pop, key=lambda b: cbf_pop[b][1]["map5"])

    def row(name, rankings):
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        rf = repeat_fraction(rankings, user_train_items, cohorts["A"])
        return [name, ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"], rf]

    all_rows = [
        row("global", gr),
        row("personal+global(n_days)", personal_rankings),
        row(f"MF{best_mf['cfg']}", mf_rankings),
        row("MF+pop(beta=1)", mf_pop_rankings),
        row(f"CBF({best_cbf_name})", cbf_rankings),
        row(f"CBF+pop(beta={cbf_pop_best})", cbf_pop[cbf_pop_best][0]),
    ]

    out("## Итоговая таблица всех методов")
    out()
    out("| метод | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие | доля повторов |")
    out("|---|---|---|---|---|---|---|")
    for r in all_rows:
        out(f"| {r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]:,} | {r[6]*100:.1f}% |")
    out()

    pd.DataFrame(all_rows, columns=["method", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage", "repeat_frac"]).to_csv(
        config.RESULTS_DIR / "all_methods.csv", index=False
    )

    # cbf.csv
    cbf_csv = []
    for name, (_v, _p, rank, ra, rb, rc) in cbf_results.items():
        cbf_csv.append([f"CBF({name})", ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"],
                        repeat_fraction(rank, user_train_items, cohorts["A"])])
    for beta in (0.5, 1.0):
        rank, ra, rb, rc = cbf_pop[beta]
        cbf_csv.append([f"CBF+pop(beta={beta})", ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"],
                        repeat_fraction(rank, user_train_items, cohorts["A"])])
    pd.DataFrame(cbf_csv, columns=["method", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage", "repeat_frac"]).to_csv(
        config.RESULTS_DIR / "cbf.csv", index=False
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "cbf_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "cbf_summary.md")


def _mf_pop(X, Y, user_index, item_ids, candidates, pop_count, global_rank, cohort, beta, k=5):
    item_ids = np.asarray(item_ids, dtype=np.int64)
    cand_mask = np.array([int(s) in candidates for s in item_ids], dtype=bool)
    Yc = Y[cand_mask]
    cand_ids = item_ids[cand_mask]
    pop_log = np.array([np.log1p(pop_count.get(int(s), 0)) for s in cand_ids], dtype=np.float64)
    rankings: dict[int, list[int]] = {}
    for u in cohort:
        ri = user_index.get(int(u))
        if ri is None or Yc.shape[0] == 0:
            rec = list(global_rank[:k])
        else:
            scores = X[ri] @ Yc.T
            mu = scores.mean()
            sd = scores.std()
            z = (scores - mu) / sd if sd > 0 else np.zeros_like(scores)
            combined = z + beta * pop_log
            order = np.argsort(-combined)[:k]
            rec = [int(x) for x in cand_ids[order]]
            if len(rec) < k:
                seen = set(rec)
                for gs in global_rank:
                    if len(rec) >= k:
                        break
                    if gs not in seen:
                        rec.append(int(gs))
                        seen.add(int(gs))
        rankings[int(u)] = rec
    return rankings


if __name__ == "__main__":
    main()
