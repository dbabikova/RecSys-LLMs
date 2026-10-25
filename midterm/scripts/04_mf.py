"""Stage 4 driver: implicit MF (ALS) baseline and comparison."""

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


def main() -> None:
    out("# Implicit MF (ALS), этап 4")
    out()

    sub_int = sorted(int(u) for u in E.load_submission_ids())
    candidates, _titles, _ = E.load_val_candidates()
    out(f"Кандидатов: {len(candidates):,}")
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
    del airings, ov, dur, vst, watched

    gts = {d: E.build_ground_truth(val_airings, d) for d in ("A", "B", "C")}
    cohorts = {d: [u for u in sub_int if u in gts[d]] for d in ("A", "B", "C")}
    out(f"Когорта GT-A: {len(cohorts['A']):,} (B: {len(cohorts['B']):,}, C: {len(cohorts['C']):,})")
    out()

    # ---- train n_days, global, personal --------------------------------
    days = defaultdict(set)
    for u, s, vs in zip(train_watched["user_id"], train_watched["tv_show_id"], train_watched["view_start"]):
        days[(int(u), int(s))].add(int(vs) // 86400)
    n_days = {k: len(v) for k, v in days.items()}

    pop = defaultdict(set)
    for (u, s) in n_days:
        if s in candidates:
            pop[s].add(u)
    global_rank = sorted(pop, key=lambda s: (-len(pop[s]), s))
    global_top5 = global_rank[:5]

    personal_rankings, personal_pad = build_personal_rankings(n_days, candidates, global_rank, cohorts["A"])

    user_train_items = defaultdict(set)
    for (u, s) in n_days:
        user_train_items[u].add(s)

    csr, user_index, item_index, item_ids = MF.build_csr(n_days)
    out(f"Матрица train: {csr.shape[0]:,} пользователей × {csr.shape[1]:,} передач, nnz = {csr.nnz:,} (r_ui = n_days)")
    out()

    # ---- ALS grid -------------------------------------------------------
    configs = [(f, r, a) for f in (32, 64) for r in (0.1, 1.0) for a in (5, 20, 40)]
    cfg_rows = []
    best = None
    out("## Loss по итерациям (проверка не-возрастания)")
    out()
    for f, r, a in configs:
        t0 = time.time()
        X, Y, losses = MF.als(csr, f, r, a, n_iter=15, seed=42, verbose=False)
        dt = time.time() - t0
        mono = all(losses[i] <= losses[i - 1] + 1e-6 * abs(losses[i - 1]) for i in range(1, len(losses)))
        out(f"factors={f} reg={r} alpha={a}: non-increasing={mono}")
        out("   " + " ".join(f"{l:.3e}" for l in losses))
        rankings, n_pad = MF.predict_rankings(X, Y, user_index, item_ids, candidates, global_rank, cohorts["A"])
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        cfg_rows.append([f, r, a, ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"], round(dt, 2), n_pad, mono])
        out(f"   MAP@5 A/B/C = {ra['map5']:.6f} / {rb['map5']:.6f} / {rc['map5']:.6f}; time {dt:.1f}s")
        out()
        if best is None or ra["map5"] > best["map5_a"]:
            best = {
                "factors": f, "reg": r, "alpha": a, "map5_a": ra["map5"], "map5_b": rb["map5"],
                "map5_c": rc["map5"], "rankings": rankings, "n_pad": n_pad, "train_time": dt,
                "recall5_a": ra["recall5"], "coverage": ra["coverage"],
            }

    # ---- configs table --------------------------------------------------
    out("## Все конфиги (сортировка по MAP@5 GT-A)")
    out()
    out("| factors | reg | alpha | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие | время, с |")
    out("|---|---|---|---|---|---|---|---|---|")
    cfg_rows.sort(key=lambda r: -r[3])
    for f, r, a, ma, mb, mc, rec, cov, dt, n_pad, mono in cfg_rows:
        out(f"| {f} | {r} | {a} | {ma:.6f} | {mb:.6f} | {mc:.6f} | {rec:.6f} | {cov:,} | {dt:.1f} |")
    out()

    pd.DataFrame(
        [[f, r, a, ma, mb, mc, rec, cov, dt, n_pad, mono] for f, r, a, ma, mb, mc, rec, cov, dt, n_pad, mono in cfg_rows],
        columns=["factors", "reg", "alpha", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage", "time_s", "n_padded", "loss_noninc"],
    ).to_csv(config.RESULTS_DIR / "mf_configs.csv", index=False)

    # top-3 by B and C
    by_b = sorted(cfg_rows, key=lambda r: -r[4])
    by_c = sorted(cfg_rows, key=lambda r: -r[5])
    rank_b = next(i for i, r in enumerate(by_b) if (r[0], r[1], r[2]) == (best["factors"], best["reg"], best["alpha"])) + 1
    rank_c = next(i for i, r in enumerate(by_c) if (r[0], r[1], r[2]) == (best["factors"], best["reg"], best["alpha"])) + 1
    out(f"Лучший по GT-A: factors={best['factors']}, reg={best['reg']}, alpha={best['alpha']}; ранг по B = {rank_b}, по C = {rank_c}")
    out()

    # ---- comparison table ----------------------------------------------
    out("## Сравнение (метод × GT)")
    out()
    out("| метод | MAP@5 A | MAP@5 B | MAP@5 C | Recall@5 A | покрытие |")
    out("|---|---|---|---|---|---|")

    def row(name, rankings):
        ra = E.evaluate(rankings, gts["A"], cohorts["A"])
        rb = E.evaluate(rankings, gts["B"], cohorts["B"])
        rc = E.evaluate(rankings, gts["C"], cohorts["C"])
        out(f"| {name} | {ra['map5']:.6f} | {rb['map5']:.6f} | {rc['map5']:.6f} | {ra['recall5']:.6f} | {ra['coverage']:,} |")
        return [name, ra["map5"], rb["map5"], rc["map5"], ra["recall5"], ra["coverage"]]

    gr = {u: list(global_top5) for u in cohorts["A"]}
    cmp_rows = [row("global", gr)]
    cmp_rows.append(row("personal+global(n_days)", personal_rankings))
    cmp_rows.append(row(f"MF({best['factors']},{best['reg']},{best['alpha']})", best["rankings"]))
    out()

    pd.DataFrame(cmp_rows, columns=["method", "MAP5_A", "MAP5_B", "MAP5_C", "Recall5_A", "coverage"]).to_csv(
        config.RESULTS_DIR / "mf.csv", index=False
    )

    # ---- repeat fraction ------------------------------------------------
    def repeat_fraction(rankings):
        total = 0
        reps = 0
        for u in cohorts["A"]:
            ti = user_train_items.get(u, set())
            for s in rankings.get(u, [])[:5]:
                total += 1
                if s in ti:
                    reps += 1
        return reps / total

    out("## Доля повторов (передача уже смотрелась в train)")
    out()
    out(f"MF лучший: {repeat_fraction(best['rankings']) * 100:.1f}%")
    out(f"personal+global(n_days): {repeat_fraction(personal_rankings) * 100:.1f}%")
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "mf_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "mf_summary.md")


if __name__ == "__main__":
    main()
