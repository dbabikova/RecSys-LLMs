"""Stage 4b: MF sanity check, edge-grid extension, MF+popularity variant."""

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
    n_pad = 0
    for u in cohort:
        items = sorted(user_items.get(u, []), key=lambda s: (-n_days[(u, s)], s))
        top = items[:5]
        if len(top) < 5:
            n_pad += 5 - len(top)
            seen = set(top)
            for gs in global_rank:
                if len(top) >= 5:
                    break
                if gs not in seen:
                    top.append(gs)
                    seen.add(gs)
        rankings[u] = top
    return rankings, n_pad


def train_map5(X, Y, csr):
    """MAP@5 on train itself: ground truth = user's train items, candidates = all train items."""
    S = X @ Y.T
    aps = []
    for u in range(csr.shape[0]):
        a = csr.indptr[u]
        b = csr.indptr[u + 1]
        gt = set(int(x) for x in csr.indices[a:b])
        if not gt:
            continue
        order = np.argsort(-S[u])[:5]
        aps.append(E.ap_at_k([int(o) for o in order], gt))
    return float(np.mean(aps))


def n_days_train_map5(csr):
    """Reference: rank train items by n_days (must be exactly 1.0)."""
    aps = []
    for u in range(csr.shape[0]):
        a = csr.indptr[u]
        b = csr.indptr[u + 1]
        cols = csr.indices[a:b]
        r = csr.data[a:b]
        if len(cols) == 0:
            continue
        gt = set(int(x) for x in cols)
        order = cols[np.argsort(-r)][:5]
        aps.append(E.ap_at_k([int(o) for o in order], gt))
    return float(np.mean(aps))


def mf_pop_rankings(X, Y, user_index, item_ids, candidates, pop_count, global_rank, cohort, beta, k=5):
    item_ids = np.asarray(item_ids, dtype=np.int64)
    cand_mask = np.array([int(s) in candidates for s in item_ids], dtype=bool)
    Yc = Y[cand_mask]
    cand_ids = item_ids[cand_mask]
    pop_log = np.array([np.log1p(pop_count.get(int(s), 0)) for s in cand_ids], dtype=np.float64)

    rankings: dict[int, list[int]] = {}
    n_pad = 0
    for u in cohort:
        ri = user_index.get(int(u))
        if ri is None or Yc.shape[0] == 0:
            rec = list(global_rank[:k])
            n_pad += k
        else:
            scores = X[ri] @ Yc.T
            mu = scores.mean()
            sd = scores.std()
            z = (scores - mu) / sd if sd > 0 else np.zeros_like(scores)
            combined = z + beta * pop_log
            order = np.argsort(-combined)[:k]
            rec = [int(x) for x in cand_ids[order]]
            if len(rec) < k:
                n_pad += k - len(rec)
                seen = set(rec)
                for gs in global_rank:
                    if len(rec) >= k:
                        break
                    if gs not in seen:
                        rec.append(int(gs))
                        seen.add(int(gs))
        rankings[int(u)] = rec
    return rankings, n_pad


def main() -> None:
    out("# MF: проверки перед отчётом (этап 4b)")
    out()

    sub_int = sorted(int(u) for u in E.load_submission_ids())
    candidates, titles, _ = E.load_val_candidates()

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

    # popularity over ALL train items
    pop_count = defaultdict(set)
    for (u, s) in n_days:
        pop_count[s].add(u)
    pop_count = {s: len(v) for s, v in pop_count.items()}
    global_rank = sorted([s for s in pop_count if s in candidates], key=lambda s: (-pop_count[s], s))
    global_top5 = global_rank[:5]

    personal_rankings, _ = build_personal_rankings(n_days, candidates, global_rank, cohorts["A"])
    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)

    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    out(f"Матрица train: {csr.shape[0]:,} × {csr.shape[1]:,}, nnz = {csr.nnz:,}")
    out()

    # ---- extended grid --------------------------------------------------
    configs = [(f, r, a) for f in (8, 16, 32) for r in (1, 10, 50) for a in (0.5, 1, 2, 5)]
    runs = {}
    out("## Расширенная сетка (loss не-возрастает по каждой конфигурации)")
    out()
    for f, r, a in configs:
        t0 = time.time()
        X, Y, losses = MF.als(csr, f, r, a, n_iter=15, seed=42, verbose=False)
        dt = time.time() - t0
        mono = all(losses[i] <= losses[i - 1] + 1e-6 * abs(losses[i - 1]) for i in range(1, len(losses)))
        rankings, n_pad = MF.predict_rankings(X, Y, user_index, item_ids, candidates, global_rank, cohorts["A"])
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        runs[(f, r, a)] = dict(X=X, Y=Y, rankings=rankings, n_pad=n_pad, mono=mono, dt=dt,
                               ma=ra["map5"], mb=rb["map5"], mc=rc["map5"],
                               rec=ra["recall5"], cov=ra["coverage"])
        out(f"f={f:2d} reg={r:2d} alpha={a:3.1f}: mono={mono} | MAP@5 A/B/C = {ra['map5']:.6f}/{rb['map5']:.6f}/{rc['map5']:.6f}")
    out()

    # ---- configs table --------------------------------------------------
    rows = [(f, r, a, v["ma"], v["mb"], v["mc"], v["rec"], v["cov"], v["dt"], v["n_pad"], v["mono"])
            for (f, r, a), v in runs.items()]
    rows.sort(key=lambda x: -x[3])
    out("## Таблица конфигов (по MAP@5 GT-A)")
    out()
    out("| factors | reg | alpha | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие | время, с |")
    out("|---|---|---|---|---|---|---|---|---|")
    for f, r, a, ma, mb, mc, rec, cov, dt, n_pad, mono in rows:
        out(f"| {f} | {r} | {a} | {ma:.6f} | {mb:.6f} | {mc:.6f} | {rec:.6f} | {cov:,} | {dt:.1f} |")
    out()

    pd.DataFrame(
        [[f, r, a, ma, mb, mc, rec, cov, dt, n_pad, mono] for f, r, a, ma, mb, mc, rec, cov, dt, n_pad, mono in rows],
        columns=["factors", "reg", "alpha", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage", "time_s", "n_padded", "loss_noninc"],
    ).to_csv(config.RESULTS_DIR / "mf_configs.csv", index=False)

    best_cfg = (rows[0][0], rows[0][1], rows[0][2])
    by_b = sorted(rows, key=lambda x: -x[4])
    by_c = sorted(rows, key=lambda x: -x[5])
    rank_b = next(i for i, r in enumerate(by_b) if (r[0], r[1], r[2]) == best_cfg) + 1
    rank_c = next(i for i, r in enumerate(by_c) if (r[0], r[1], r[2]) == best_cfg) + 1
    out(f"Лучший по GT-A: factors={best_cfg[0]}, reg={best_cfg[1]}, alpha={best_cfg[2]}; ранг по B = {rank_b}, по C = {rank_c}")
    out()

    # ---- sanity on train ------------------------------------------------
    sanity_cfg = (32, 1.0, 5)
    if sanity_cfg not in runs:
        sanity_cfg = best_cfg
    Xs, Ys = runs[sanity_cfg]["X"], runs[sanity_cfg]["Y"]
    out(f"## Sanity на train (конфиг {sanity_cfg})")
    out()
    mf_train = train_map5(Xs, Ys, csr)
    nd_train = n_days_train_map5(csr)
    out(f"MAP@5 на train (ground truth = передачи train, кандидаты = все train):")
    out(f"  MF = {mf_train:.4f}")
    out(f"  n_days baseline (должно быть 1.0000, проверка маппинга) = {nd_train:.4f}")
    out()

    demo = cohorts["A"][0]
    row = user_index.get(demo)
    if row is not None:
        a = csr.indptr[row]
        b = csr.indptr[row + 1]
        cols = csr.indices[a:b]
        r = csr.data[a:b]
        top_nd = cols[np.argsort(-r)][:5].tolist()
        scores = Xs[row] @ Ys.T
        top_mf = np.argsort(-scores)[:5].tolist()
        out(f"Пользователь {demo} (train top-5):")
        out(f"  n_days: {[int(item_ids[i]) for i in top_nd]}")
        out(f"  MF    : {[int(item_ids[i]) for i in top_mf]}")
    out()

    # ---- MF + popularity ------------------------------------------------
    Xb, Yb = runs[best_cfg]["X"], runs[best_cfg]["Y"]
    out("## MF + популярность (z-score MF + beta * log1p(pop_train))")
    out()
    out("| beta | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие |")
    out("|---|---|---|---|---|---|")
    pop_rows = []
    for beta in (0.0, 0.1, 0.5, 1.0):
        rankings, n_pad = mf_pop_rankings(Xb, Yb, user_index, item_ids, candidates, pop_count, global_rank, cohorts["A"], beta)
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        out(f"| {beta} | {ra['map5']:.6f} | {rb['map5']:.6f} | {rc['map5']:.6f} | {ra['recall5']:.6f} | {ra['coverage']:,} |")
        pop_rows.append((beta, rankings, ra, rb, rc))
    out()

    # ---- final comparison (mf.csv) -------------------------------------
    gr = {u: list(global_top5) for u in cohorts["A"]}
    cmp = []

    def add(name, rankings):
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        cmp.append([name, ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"]])

    add("global", gr)
    add("personal+global(n_days)", personal_rankings)
    add(f"MF({best_cfg[0]},{best_cfg[1]},{best_cfg[2]})", runs[best_cfg]["rankings"])
    for beta, rankings, ra, rb, rc in pop_rows:
        if beta > 0:
            add(f"MF+pop(beta={beta})", rankings)

    out("## Итоговое сравнение (mf.csv)")
    out()
    out("| метод | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие |")
    out("|---|---|---|---|---|---|")
    for name, ma, mb, mc, rec, cov in cmp:
        out(f"| {name} | {ma:.6f} | {mb:.6f} | {mc:.6f} | {rec:.6f} | {cov:,} |")
    out()

    pd.DataFrame(cmp, columns=["method", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage"]).to_csv(
        config.RESULTS_DIR / "mf.csv", index=False
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "mf_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "mf_summary.md")


if __name__ == "__main__":
    main()
