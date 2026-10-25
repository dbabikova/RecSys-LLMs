"""Stage 14: HabitSlot (h1) x PPR score fusion.

score(u, i) = h1(u, i) * (ppr(u, i) + 1e-12) ** w   for h1 candidates with h1 > 0,
top-5 by (score desc, tv_show_id asc), padding as in h1.

h1 parts come from scripts/10_h1.py (compute_p, slot_profile_from_log, hour_fractions,
compute_global_pop, pad5, build_gt_d80); PPR from scripts/09e_ppr.py (compute_ppr).

python scripts/12_h1_ppr.py            -> val D80 for w in {0, 0.05, 0.1, 0.2}
python scripts/12_h1_ppr.py --submit   -> submissions/final_submission_h1ppr_oc.csv (w = 0.05)
Both write results/tune_14.md.
"""

import hashlib
import importlib.util
import math
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import evaluate as E  # noqa: E402
import mf_als as MF  # noqa: E402
from join import to_unix  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H1 = _load("h1mod", "10_h1.py")
PPR = _load("pprmod", "09e_ppr.py")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ALPHA = 0.5
MAX_ITER = 100
EPS = 1e-12
W_VAL = (0.0, 0.05, 0.1, 0.2)
W_SUB = 0.05
REF_MD5 = "17da5f0238dc2044a65d7a064bbd929e"

md: list[str] = []


def out(*args) -> None:
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    md.append(line)


def h1_candidates(grid, log, matches, REF, END, users):
    """Per-user h1 candidates [(h1, show)] with h1 > 0 and padding list, same steps as H1.h1_rankings."""
    sids, chs, sts, durs = grid
    wmask = (sts >= REF) & (sts < END)
    w_sids, w_chs, w_sts, w_durs = sids[wmask], chs[wmask], sts[wmask], durs[wmask]
    p = H1.compute_p(matches, REF)
    slot_prof = H1.slot_profile_from_log(log, REF)
    pop = H1.compute_global_pop(matches, REF)
    window_shows = set(w_sids.tolist())
    gtop5 = sorted([s for s in pop if s in window_shows], key=lambda s: (-pop[s], s))[:5]
    show_slots = defaultdict(list)
    for sid, ch, st, dur in zip(w_sids.tolist(), w_chs.tolist(), w_sts.tolist(), w_durs.tolist()):
        for hw, frac in H1.hour_fractions(st, dur):
            show_slots[sid].append((ch, hw, frac))
    user_shows = defaultdict(list)
    for (u, s) in p:
        user_shows[u].append(s)
    cands = {}
    for u in users:
        prof = slot_prof.get(u, {})
        scored = []
        for s in user_shows.get(u, ()):
            if s not in window_shows:
                continue
            slot = sum(prof.get((ch, hw), 0.0) * w for (ch, hw, w) in show_slots[s])
            sc = p[(u, s)] * math.sqrt(slot + 1e-4)
            if sc > 0:
                scored.append((sc, s))
        cands[u] = scored
    return cands, gtop5, window_shows


def ppr_scores(matches, REF, users):
    """PPR mass on shows for each seed user: bipartite graph of watched history pairs, edge log1p(n_days)."""
    vst = matches["view_start"].to_numpy()
    wm = (vst < REF) & E.watched_mask_np(matches["overlap"].to_numpy(), matches["duration"].to_numpy())
    h = matches[wm]
    days = defaultdict(set)
    for u, s, v in zip(h["user_id"].tolist(), h["tv_show_id"].tolist(), h["view_start"].tolist()):
        days[(int(u), int(s))].add(int(v) // 86400)
    w_edges = {k: math.log1p(len(v)) for k, v in days.items()}
    csr, uidx, iidx, _items = MF.build_csr(w_edges)
    t0 = time.time()
    Ps, valid, n_iter = PPR.compute_ppr(csr, users, uidx, ALPHA, max_iter=MAX_ITER)
    out(f"PPR: граф {csr.shape[0]} польз. x {csr.shape[1]} передач, {csr.nnz} рёбер; сидов {len(valid)} из "
        f"{len(users)}; итераций {n_iter} ({'сошлось, L1 < 1e-6' if n_iter < MAX_ITER else 'достигнут лимит 100'}); "
        f"{time.time() - t0:.0f}s")
    col = {u: j for j, u in enumerate(valid)}
    return Ps, col, iidx


def fuse(cands, gtop5, Ps, col, iidx, w, users):
    res = {}
    for u in users:
        scored = cands.get(u, [])
        if not scored:
            res[u] = list(gtop5)
            continue
        j = col.get(u)
        items = []
        for h1s, s in scored:
            i = iidx.get(s)
            pr = float(Ps[i, j]) if (j is not None and i is not None) else 0.0
            items.append((h1s * (pr + EPS) ** w, s))
        items.sort(key=lambda x: (-x[0], x[1]))
        res[u] = H1.pad5([s for _sc, s in items[:5]], gtop5)
    return res


def load_data():
    print("Loading matches / grid / log ...", flush=True)
    return E.load_labeled_airings(E.CAP), H1.load_grid(), H1.load_log()


def run_val():
    out("# Этап 14: h1 x PPR (слияние скоров)")
    out()
    out(f"score = h1 * (ppr + 1e-12)^w; PPR alpha {ALPHA}, рестарт в пользователе, до {MAX_ITER} итераций, "
        "рёбра log1p(дней) по watched-парам (overlap >= 300 с или pct >= 0.5)")
    out()
    out("## Val: история < 2020-05-18, окно 2020-05-18 .. 2020-07-26, GT-D80")
    users = sorted(int(u) for u in E.load_submission_ids())
    matches, grid, log = load_data()
    REF, END = to_unix(datetime(2020, 5, 18)), to_unix(datetime(2020, 7, 27))
    gt = H1.build_gt_d80(matches, REF, END)
    cohort = [u for u in users if u in gt]
    cands, gtop5, _ws = h1_candidates(grid, log, matches, REF, END, users)
    Ps, col, iidx = ppr_scores(matches, REF, users)

    # sanity: w = 0 must reproduce H1.h1_rankings exactly
    ref_rank, ref_gtop = H1.h1_rankings(grid, log, matches, REF, END)
    ref_rank = {u: ref_rank.get(u, list(ref_gtop)) for u in users}

    out()
    out("| w | D80 | Δ к h1 |")
    out("|---|---|---|")
    res = {}
    for w in W_VAL:
        r = fuse(cands, gtop5, Ps, col, iidx, w, users)
        res[w] = float(np.mean([E.ap_at_k(r[u], gt[u]) for u in cohort]))
        if w == 0.0:
            same = sum(r[u] == ref_rank[u] for u in users)
            base = res[w]
        out(f"| {w:g} | {res[w]:.4f} | {res[w] - base:+.4f} |")
    out()
    out(f"Проверка: при w = 0 топ-5 совпадает с 10_h1.h1_rankings у {same}/{len(users)} пользователей; "
        f"D80 = {base:.4f} (h1 этапа 10: 0.4009). Когорта D80: {len(cohort)}.")
    best = max(W_VAL, key=lambda w: (res[w], -w))
    out(f"Лучший w на val: {best:g} (D80 {res[best]:.4f}); ориентир при w = 0.05: ~0.4016.")
    out()
    return res


def run_submit():
    out("## Сабмит: история недель 11-30, окно 2020-07-27 + 12 недель, w = 0.05")
    with open(config.SUBMISSION_FILE, encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n")
        ids_str = [l.rstrip("\r\n").split(",", 1)[0] for l in f if l.strip()]
    users = [int(u) for u in ids_str]
    matches, grid, log = load_data()
    REF = to_unix(datetime(2020, 7, 27))
    END = REF + 12 * 7 * 86400
    cands, gtop5, window = h1_candidates(grid, log, matches, REF, END, users)
    Ps, col, iidx = ppr_scores(matches, REF, users)
    rank = fuse(cands, gtop5, Ps, col, iidx, W_SUB, users)
    rank0 = fuse(cands, gtop5, Ps, col, iidx, 0.0, users)

    outp = config.SUBMISSIONS_DIR / "final_submission_h1ppr_oc.csv"
    with open(outp, "w", newline="") as f:
        f.write(header + "\n")
        for us, u in zip(ids_str, users):
            f.write(f"{us},{' '.join(str(x) for x in rank[u][:5])}\n")
    lines = outp.read_text(encoding="utf-8").splitlines()
    preds = [[int(x) for x in l.split(",", 1)[1].split()] for l in lines[1:]]
    checks = {
        "1261 строка (заголовок + 1260)": len(lines) == 1261 and lines[0] == header,
        "user_id и порядок как в submission.csv": [l.split(",", 1)[0] for l in lines[1:]] == ids_str,
        "ровно 5 уникальных id": all(len(p) == 5 and len(set(p)) == 5 for p in preds),
        "нет tv_show_id = 0": all(0 not in p for p in preds),
        "все id идут в окне 31-42": all(all(x in window for x in p) for p in preds),
        "pandas читает 1260 x 2": pd.read_csv(outp, dtype=str).shape == (1260, 2),
    }
    for k, v in checks.items():
        out(f"{'PASS' if v else 'FAIL'}: {k}")
    assert all(checks.values())
    md5 = hashlib.md5(outp.read_bytes()).hexdigest()
    out(f"md5 final_submission_h1ppr_oc.csv = {md5} (эталон {REF_MD5}: "
        f"{'совпадает' if md5 == REF_MD5 else 'НЕ совпадает'})")

    h1 = pd.read_csv(config.SUBMISSIONS_DIR / "final_submission_h1_oc.csv", dtype=str)
    a = [tuple(int(x) for x in s.split()) for s in h1["tv_show_id"]]
    b = [tuple(p) for p in preds]
    same0 = sum(tuple(rank0[u][:5]) == x for u, x in zip(users, a))
    out(f"Проверка: при w = 0 топ-5 совпадает с final_submission_h1_oc.csv у {same0}/1260")
    n_order = sum(x != y for x, y in zip(a, b))
    n_set = sum(set(x) != set(y) for x, y in zip(a, b))
    out(f"Отличия от final_submission_h1_oc.csv: по порядку (любая позиция) — {n_order} пользователей, "
        f"по составу топ-5 — {n_set}; только перестановка без смены состава — {n_order - n_set}")
    out(f"Позиций отличается: {sum(i != j for x, y in zip(a, b) for i, j in zip(x, y))} из 6300; "
        f"среднее пересечение {np.mean([len(set(x) & set(y)) for x, y in zip(a, b)]):.3f}/5")
    out()


def main():
    p = config.RESULTS_DIR / "tune_14.md"
    if "--submit" in sys.argv:
        run_submit()
        txt = p.read_text(encoding="utf-8") if p.exists() else ""
        head = txt.split("## Сабмит")[0].rstrip("\n")
        p.write_text((head + "\n\n" if head else "") + "\n".join(md) + "\n", encoding="utf-8")
    else:
        run_val()
        txt = p.read_text(encoding="utf-8") if p.exists() else ""
        tail = ("\n## Сабмит" + txt.split("## Сабмит", 1)[1]) if "## Сабмит" in txt else ""
        p.write_text("\n".join(md) + "\n" + tail, encoding="utf-8")
    print("\nЗаписано:", p)


if __name__ == "__main__":
    main()
