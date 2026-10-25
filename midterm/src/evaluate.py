"""Evaluation protocol (MAP@5), ground-truth builders and loaders for stage 3."""

import csv
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import config
from join import parse_grid_time, to_unix

VAL_START = to_unix(datetime(2020, 5, 18, 0, 0, 0))
VAL_END = to_unix(datetime(2020, 7, 26, 23, 59, 59))
REF_TIME = VAL_START  # train/val border, used for recency decay

K = 5
CAP = 4 * 3600          # matches stage-2 clipping
WATCH_MIN = 300         # seconds
WATCH_FRAC = 0.5        # fraction of airing duration


def load_submission_ids() -> list[str]:
    """Submission user ids as exact decimal strings (byte-identical to file)."""
    ids: list[str] = []
    with open(config.SUBMISSION_FILE, encoding="utf-8-sig", newline="") as f:
        next(f)
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                ids.append(line.split(",", 1)[0])
    return ids


def load_val_candidates() -> tuple[set[int], dict[int, str], dict[int, int]]:
    """Unique tv_show_id != 0 with an airing inside the val window.

    Returns (candidate ids, id->title, id->number of val airings).
    """
    cand: set[int] = set()
    titles: dict[int, str] = {}
    airings: dict[int, int] = defaultdict(int)
    for fp in (config.GRID_11_20_FILE, config.GRID_21_30_FILE, config.GRID_31_42_FILE):
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                sid = int(row[5])
                if sid == 0:
                    continue
                st = to_unix(parse_grid_time(row[2]))
                if VAL_START <= st <= VAL_END:
                    cand.add(sid)
                    titles.setdefault(sid, row[4])
                    airings[sid] += 1
    return cand, titles, dict(airings)


def load_labeled_airings(cap: int = CAP) -> pd.DataFrame:
    """Per-airing labeled matches (reuses join.py unchanged; nothing cached)."""
    import join

    g = join.load_grid()  # full grid (join_channels needs dense per-channel data)
    log = join.load_log(cap)
    pairs = join.join_channels(g, log)
    del g, log

    user, show, ch, ast, ov, dur, vst = pairs
    lab = show != 0
    return pd.DataFrame(
        {
            "user_id": user[lab].astype(np.uint64),
            "tv_show_id": show[lab].astype(np.int64),
            "channel": ch[lab].astype(np.int32),
            "air_start": ast[lab].astype(np.int64),
            "overlap": ov[lab].astype(np.int64),
            "duration": dur[lab].astype(np.int32),
            "view_start": vst[lab].astype(np.int64),
        }
    )


def watched_mask_np(overlap: np.ndarray, duration: np.ndarray) -> np.ndarray:
    return (overlap >= WATCH_MIN) | (overlap >= WATCH_FRAC * duration)


def build_ground_truth(val_airings: pd.DataFrame, definition: str = "A") -> dict[int, set[int]]:
    """Ground truth for a single 'watched in val' definition.

    A: at least one airing with overlap >= 300 s OR >= 50% of duration.
    B: at least one airing with overlap >= 50% of duration (strict).
    C: watched (as A) on >= 2 distinct days.
    """
    ov = val_airings["overlap"].to_numpy()
    dur = val_airings["duration"].to_numpy()
    u = val_airings["user_id"].to_numpy()
    s = val_airings["tv_show_id"].to_numpy()

    if definition in ("A", "C"):
        w = watched_mask_np(ov, dur)
    elif definition == "B":
        w = ov >= WATCH_FRAC * dur
    else:
        raise ValueError(f"unknown definition {definition!r}")

    u, s = u[w], s[w]

    if definition == "C":
        vs = val_airings["view_start"].to_numpy()[w] // 86400
        days: dict[tuple[int, int], set[int]] = defaultdict(set)
        for ui, si, di in zip(u, s, vs):
            days[(int(ui), int(si))].add(int(di))
        gt: dict[int, set[int]] = {}
        for (ui, si), ds in days.items():
            if len(ds) >= 2:
                gt.setdefault(ui, set()).add(si)
        return gt

    gt = {}
    for ui, si in zip(u, s):
        gt.setdefault(int(ui), set()).add(int(si))
    return gt


def ap_at_k(pred, relevant, k: int = K) -> float:
    n_rel = len(relevant)
    if n_rel == 0:
        return 0.0
    hits = 0
    s = 0.0
    for r in range(1, k + 1):
        if r - 1 < len(pred) and pred[r - 1] in relevant:
            hits += 1
            s += hits / r
    return s / min(k, n_rel)


def recall_at_k(pred, relevant, k: int = K) -> float:
    n_rel = len(relevant)
    if n_rel == 0:
        return 0.0
    return sum(1 for p in pred[:k] if p in relevant) / n_rel


def evaluate(rankings: dict, gt: dict, cohort: list) -> dict:
    aps: list[float] = []
    recs: list[float] = []
    rel_sizes: list[int] = []
    all_pred: set[int] = set()
    for u in cohort:
        pred = list(rankings.get(u, []))[:K]
        rel = gt.get(u, set())
        aps.append(ap_at_k(pred, rel))
        recs.append(recall_at_k(pred, rel))
        rel_sizes.append(len(rel))
        all_pred.update(pred)
    n = len(cohort)
    return {
        "map5": float(np.mean(aps)) if n else 0.0,
        "recall5": float(np.mean(recs)) if n else 0.0,
        "coverage": len(all_pred),
        "avg_rel": float(np.mean(rel_sizes)) if n else 0.0,
        "median_rel": float(np.median(rel_sizes)) if n else 0.0,
        "n_cohort": n,
    }


def test_map_at5() -> bool:
    rankings = {
        1: [1, 4, 2, 5, 6],
        2: [7, 1, 2, 3, 4],
        3: [1, 10, 20, 3, 4],
    }
    gt = {1: {1, 2, 3}, 2: {7}, 3: {10, 20}}
    expected_ap = {1: 5 / 9, 2: 1.0, 3: 7 / 12}
    for u in rankings:
        got = ap_at_k(rankings[u], gt[u])
        assert np.isclose(got, expected_ap[u]), f"AP@5 user {u}: {got} != {expected_ap[u]}"
    res = evaluate(rankings, gt, [1, 2, 3])
    exp_map = (5 / 9 + 1.0 + 7 / 12) / 3
    assert np.isclose(res["map5"], exp_map)
    print(f"test_map_at5 OK: per-user AP={expected_ap}, MAP@5={exp_map:.6f} (computed {res['map5']:.6f})")
    return True


if __name__ == "__main__":
    test_map_at5()
