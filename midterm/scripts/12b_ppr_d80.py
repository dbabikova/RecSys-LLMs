"""Stage 16: PPR alone on GT-D80, and reach / coverage of popularity, h1 and the h1 + PPR hybrid (val).

PPR, h1 candidates and the hybrid come from scripts/12_h1_ppr.py; GT-D80 and padding from scripts/10_h1.py.
Appends the section "Stage 16: PPR alone and reach" to results/tune_14.md (replaced on rerun).
"""

import importlib.util
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import evaluate as E  # noqa: E402
from join import to_unix  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HP = _load("h1pprmod", "12_h1_ppr.py")
H1 = HP.H1  # the same scripts/10_h1.py module used by 12_h1_ppr.py

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SECTION = "## Stage 16: PPR alone and reach"


def ppr_alone(Ps, col, iidx, window_shows, gtop5, users):
    """Top-5 window shows by PPR mass (> 0); tie-break mass desc, tv_show_id asc; h1 popularity padding."""
    cand = sorted((s, i) for s, i in iidx.items() if s in window_shows)
    c_ids = np.array([s for s, _i in cand], dtype=np.int64)
    c_rows = np.array([i for _s, i in cand], dtype=np.int64)
    res = {}
    for u in users:
        j = col.get(u)
        if j is None:
            res[u] = list(gtop5)
            continue
        m = Ps[c_rows, j]
        pos = m > 0
        ids, mass = c_ids[pos], m[pos]
        order = np.lexsort((ids, -mass))[:5]
        res[u] = H1.pad5([int(x) for x in ids[order]], gtop5)
    return res


def main():
    lines = [SECTION, ""]

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    users = sorted(int(u) for u in E.load_submission_ids())
    matches, grid, log = HP.load_data()
    REF, END = to_unix(datetime(2020, 5, 18)), to_unix(datetime(2020, 7, 27))
    gt = H1.build_gt_d80(matches, REF, END)
    cohort = [u for u in users if u in gt]
    cands, gtop5, window = HP.h1_candidates(grid, log, matches, REF, END, users)
    Ps, col, iidx = HP.ppr_scores(matches, REF, users)

    R = {
        "popularity": {u: list(gtop5) for u in users},
        "PPR alone": ppr_alone(Ps, col, iidx, window, gtop5, users),
        "h1 (w = 0)": HP.fuse(cands, gtop5, Ps, col, iidx, 0.0, users),
        "h1 + PPR (w = 0.05)": HP.fuse(cands, gtop5, Ps, col, iidx, 0.05, users),
    }
    ref = {"popularity": ("—", "0.813", "5"), "PPR alone": ("~0.348", "—", "—"),
           "h1 (w = 0)": ("0.4009", "0.913", "224"), "h1 + PPR (w = 0.05)": ("0.4014", "0.913", "223")}

    out(f"Val: история < 2020-05-18, кандидаты — передачи в эфире 2020-05-18 .. 2020-07-26, "
        f"GT-D80 из 10_h1.py, когорта {len(cohort)}. PPR — та же, что в 12_h1_ppr.py (сидов {len(col)} из {len(users)}).")
    out("PPR alone: top-5 кандидатов по PPR-массе (> 0), tie-break масса убыв., tv_show_id возр., "
        "добивка популярностью как в h1.")
    out("reach — доля пользователей когорты, у которых хотя бы одна из 5 рекомендаций есть в их D80-эталоне; "
        "coverage — число разных передач во всех top-5 когорты.")
    out()
    out("| метод | D80 | reach | coverage | ориентир D80 / reach / coverage |")
    out("|---|---|---|---|---|")
    for name, r in R.items():
        d80 = float(np.mean([E.ap_at_k(r[u], gt[u]) for u in cohort]))
        reach = float(np.mean([any(s in gt[u] for s in r[u][:5]) for u in cohort]))
        cov = len({s for u in cohort for s in r[u][:5]})
        out(f"| {name} | {d80:.4f} | {reach:.4f} ({round(reach * len(cohort))}/{len(cohort)}) | {cov} | {' / '.join(ref[name])} |")
    pad_ppr = sum(1 for u in cohort if u not in col)
    out()
    out(f"PPR alone: пользователей когорты без PPR-сида (только добивка популярностью): {pad_ppr}.")
    out("popularity здесь — добивка h1 (уникальные зрители до REF среди всех пользователей, передачи окна), "
        "поэтому D80 = исходный global этапа 10 (0.2642); 0.2731 из этапа 10b — популярность только по когорте 1260.")

    p = config.RESULTS_DIR / "tune_14.md"
    txt = p.read_text(encoding="utf-8")
    head = txt.split("\n" + SECTION)[0].rstrip("\n")
    p.write_text(head + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\nДописано в", p)


if __name__ == "__main__":
    main()
