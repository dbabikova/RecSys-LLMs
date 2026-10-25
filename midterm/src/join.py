"""Join viewing logs with the TV program grid.

Streaming readers + a per-channel interval join (searchsorted, no cartesian
product). See scripts/02_join.py for the driver.
"""

import calendar
import csv
from array import array
from datetime import datetime

import numpy as np

import config

GRID_TIME_FMT = "%d.%m.%Y %H:%M:%S"
LOOKBACK = 20  # max number of concurrent/overlapping airings to look back


def parse_grid_time(s: str) -> datetime:
    return datetime.strptime(s, GRID_TIME_FMT)


def parse_log_time(s: str) -> datetime:
    return datetime.fromisoformat(s)


def to_unix(dt: datetime) -> int:
    # UTC-based conversion of a naive datetime (data has no timezone) so that
    # differences/overlaps are DST-safe and consistent everywhere.
    return calendar.timegm(dt.timetuple())


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def load_grid():
    """Read the three grid files, parse time, drop exact duplicates.

    Returns dict with numpy arrays (sorted by channel, then start) plus title
    maps and the number of removed duplicates.
    """
    files = [config.GRID_11_20_FILE, config.GRID_21_30_FILE, config.GRID_31_42_FILE]

    ch_parts, st_parts, en_parts, id_parts, dur_parts = [], [], [], [], []
    title_map: dict[tuple[int, int], str] = {}
    ch_title_map: dict[int, str] = {}

    for fp in files:
        ch_l = array("i")
        st_l = array("q")
        en_l = array("q")
        id_l = array("q")
        du_l = array("i")
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                channel = int(row[0])
                ch_title = row[1]
                start = to_unix(parse_grid_time(row[2]))
                dur = int(float(row[3]))
                title = row[4]
                show_id = int(row[5])
                end = start + dur
                ch_l.append(channel)
                st_l.append(start)
                en_l.append(end)
                id_l.append(show_id)
                du_l.append(dur)
                if show_id != 0:
                    title_map[(channel, start)] = title
                ch_title_map.setdefault(channel, ch_title)
        ch_parts.append(np.frombuffer(ch_l, dtype=np.int32).copy())
        st_parts.append(np.frombuffer(st_l, dtype=np.int64).copy())
        en_parts.append(np.frombuffer(en_l, dtype=np.int64).copy())
        id_parts.append(np.frombuffer(id_l, dtype=np.int64).copy())
        dur_parts.append(np.frombuffer(du_l, dtype=np.int32).copy())

    ch = np.concatenate(ch_parts)
    st = np.concatenate(st_parts)
    en = np.concatenate(en_parts)
    sid = np.concatenate(id_parts)
    dur = np.concatenate(dur_parts)

    # sort by (channel, start, show_id) and drop exact duplicates
    order = np.lexsort((sid, st, ch))
    ch, st, en, sid, dur = ch[order], st[order], en[order], sid[order], dur[order]
    dup = (ch[1:] == ch[:-1]) & (st[1:] == st[:-1]) & (sid[1:] == sid[:-1])
    n_dup = int(dup.sum())
    keep = ~np.concatenate(([False], dup))
    ch, st, en, sid, dur = ch[keep], st[keep], en[keep], sid[keep], dur[keep]

    # re-sort by (channel, start) for the join
    order2 = np.lexsort((st, ch))
    ch, st, en, sid, dur = ch[order2], st[order2], en[order2], sid[order2], dur[order2]

    return {
        "ch": ch,
        "st": st,
        "en": en,
        "sid": sid,
        "dur": dur,
        "n_dup": n_dup,
        "title_map": title_map,
        "ch_title_map": ch_title_map,
    }


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------
def load_log(cap: int):
    """Read the log, parse time, recompute duration as stop-start and clip to CAP.

    Returns numpy arrays (sorted by channel, then start) and the number of
    clipped intervals.
    """
    u_l = array("Q")
    c_l = array("i")
    s_l = array("q")
    e_l = array("q")
    clipped = 0
    with open(config.LOG_FILE, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            u = int(row[0])
            c = int(row[1])
            s = to_unix(parse_log_time(row[2]))
            e = to_unix(parse_log_time(row[3]))
            if e - s > cap:
                e = s + cap
                clipped += 1
            u_l.append(u)
            c_l.append(c)
            s_l.append(s)
            e_l.append(e)

    user = np.frombuffer(u_l, dtype=np.uint64).copy()
    ch = np.frombuffer(c_l, dtype=np.int32).copy()
    st = np.frombuffer(s_l, dtype=np.int64).copy()
    en = np.frombuffer(e_l, dtype=np.int64).copy()

    order = np.lexsort((st, ch))
    user, ch, st, en = user[order], ch[order], st[order], en[order]

    return {"user": user, "ch": ch, "st": st, "en": en, "clipped": clipped}


# ---------------------------------------------------------------------------
# Interval join
# ---------------------------------------------------------------------------
def join_channels(g, log):
    """Interval join per channel.

    Returns arrays (user, show_id, channel, air_start, overlap, duration,
    view_start) for every (view, airing) pair whose intervals intersect
    (overlap > 0).
    """
    g_ch, g_st, g_en, g_sid, g_dur = g["ch"], g["st"], g["en"], g["sid"], g["dur"]
    l_user, l_ch, l_st, l_en = log["user"], log["ch"], log["st"], log["en"]

    g_uniq, g_idx = np.unique(g_ch, return_index=True)
    l_uniq, l_idx = np.unique(l_ch, return_index=True)
    g_pos = {int(c): i for i, c in enumerate(g_uniq)}
    l_pos = {int(c): i for i, c in enumerate(l_uniq)}
    common = sorted(set(g_pos) & set(l_pos))

    out_user, out_show, out_ch, out_ast, out_ov, out_dur, out_vst = [], [], [], [], [], [], []

    for c in common:
        gi = g_pos[c]
        g0 = g_idx[gi]
        g1 = g_idx[gi + 1] if gi + 1 < len(g_idx) else len(g_ch)
        li = l_pos[c]
        l0 = l_idx[li]
        l1 = l_idx[li + 1] if li + 1 < len(l_idx) else len(l_user)

        S = g_st[g0:g1]
        E = g_en[g0:g1]
        ID = g_sid[g0:g1]
        DUR = g_dur[g0:g1]
        V = l_st[l0:l1]
        W = l_en[l0:l1]
        U = l_user[l0:l1]
        n = len(V)

        iA = np.searchsorted(S, V, side="right")  # shows with start <= view start
        j1 = np.searchsorted(S, W, side="left")  # shows with start < view end

        # Group B: shows starting during the view [V, W)
        counts = j1 - iA
        tot = int(counts.sum())
        if tot > 0:
            view_idx = np.repeat(np.arange(n, dtype=np.int64), counts)
            start_off = np.repeat(iA, counts)
            pos = np.arange(tot, dtype=np.int64) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            show_idx = start_off + pos
            ov = np.minimum(E[show_idx], W[view_idx]) - S[show_idx]
            out_user.append(U[view_idx])
            out_show.append(ID[show_idx])
            out_ch.append(np.full(tot, c, dtype=np.int32))
            out_ast.append(S[show_idx])
            out_ov.append(ov)
            out_dur.append(DUR[show_idx])
            out_vst.append(V[view_idx])

        # Group A: shows already airing at view start (look back a few)
        for off in range(LOOKBACK):
            idx = iA - 1 - off
            m = (idx >= 0) & (E[idx] > V)
            if m.any():
                vi = np.nonzero(m)[0]
                si = idx[m]
                ov = np.minimum(E[si], W[vi]) - V[vi]
                out_user.append(U[vi])
                out_show.append(ID[si])
                out_ch.append(np.full(len(vi), c, dtype=np.int32))
                out_ast.append(S[si])
                out_ov.append(ov)
                out_dur.append(DUR[si])
                out_vst.append(V[vi])

    if not out_user:
        return None

    return (
        np.concatenate(out_user),
        np.concatenate(out_show),
        np.concatenate(out_ch),
        np.concatenate(out_ast),
        np.concatenate(out_ov),
        np.concatenate(out_dur),
        np.concatenate(out_vst),
    )


def watched_mask(overlap: np.ndarray, dur: np.ndarray, min_overlap: float, frac: float) -> np.ndarray:
    return (overlap >= min_overlap) | (overlap >= frac * dur)
