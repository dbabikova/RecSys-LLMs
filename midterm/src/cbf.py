"""Content-based filtering on show metadata (category/genre/year/people)."""

import csv
import re
from collections import Counter, defaultdict

import numpy as np

import config
from join import parse_grid_time, to_unix
from evaluate import VAL_START, VAL_END


def load_show_metadata():
    """Most frequent (category, g1, g2, g3, year, director, actors) per tv_show_id.

    Also returns val candidates (tv_show_id != 0 with an airing in the val window),
    an id -> title map, and the number of airings per candidate id in the val window.
    """
    meta_count = defaultdict(Counter)
    title_map: dict[int, str] = {}
    candidates: set[int] = set()
    airing_counts: Counter[int] = Counter()
    for fp in (config.GRID_11_20_FILE, config.GRID_21_30_FILE, config.GRID_31_42_FILE):
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                sid = int(row[5])
                if sid == 0:
                    continue
                key = (row[6], row[7], row[8], row[9], row[10], row[11], row[12])
                meta_count[sid][key] += 1
                title_map.setdefault(sid, row[4])
                st = to_unix(parse_grid_time(row[2]))
                if VAL_START <= st <= VAL_END:
                    candidates.add(sid)
                    airing_counts[sid] += 1
    meta = {sid: c.most_common(1)[0][0] for sid, c in meta_count.items()}
    return meta, candidates, title_map, dict(airing_counts)


def clean_year(y: str) -> int | None:
    y = (y or "").strip()
    if not y:
        return None
    head = y[:4]
    if head.isdigit():
        val = int(head)
        if 1900 <= val <= 2030:
            return val
    return None


def decade_bucket(y: int | None) -> str:
    if y is None:
        return "unknown"
    return f"{(y // 10) * 10}s"


def extract_people(meta_tuple) -> list[str]:
    director = meta_tuple[5]
    actors = meta_tuple[6]
    names: list[str] = []
    if director and director.strip():
        names.append(director.strip())
    if actors and actors.strip():
        for a in actors.split(","):
            a = a.strip()
            if a:
                names.append(a)
    return names


def l2_normalize(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def build_people_tfidf(people_lists: list[list[str]]) -> np.ndarray:
    N = len(people_lists)
    df: Counter[str] = Counter()
    for names in people_lists:
        for name in set(names):
            df[name] += 1
    vocab = [name for name, c in df.items() if c >= 2]  # min_df = 2
    v2i = {name: i for i, name in enumerate(vocab)}
    M = np.zeros((N, len(vocab)), dtype=np.float64)
    for j, names in enumerate(people_lists):
        for name in names:
            if name in v2i:
                M[j, v2i[name]] += 1.0
    df_arr = np.array([df[name] for name in vocab], dtype=np.float64)
    idf = np.log((1.0 + N) / (1.0 + df_arr)) + 1.0
    M = M * idf
    return l2_normalize(M)


def build_blocks(show_ids: list[int], meta: dict):
    """L2-normalized block matrices + show_index."""
    show_index = {sid: i for i, sid in enumerate(show_ids)}
    n = len(show_ids)

    cats = sorted({meta[sid][0] for sid in show_ids if meta[sid][0]})
    genres = sorted({g for sid in show_ids for g in meta[sid][1:4] if g})
    decades = sorted({decade_bucket(clean_year(meta[sid][4])) for sid in show_ids})
    cat_i = {c: i for i, c in enumerate(cats)}
    genre_i = {g: i for i, g in enumerate(genres)}
    dec_i = {d: i for i, d in enumerate(decades)}

    cat_mat = np.zeros((n, len(cats)), dtype=np.float64)
    genre_mat = np.zeros((n, len(genres)), dtype=np.float64)
    year_mat = np.zeros((n, len(decades)), dtype=np.float64)

    for sid in show_ids:
        i = show_index[sid]
        c, g1, g2, g3, y = meta[sid][:5]
        if c:
            cat_mat[i, cat_i[c]] = 1.0
        for g in (g1, g2, g3):
            if g:
                genre_mat[i, genre_i[g]] = 1.0
        year_mat[i, dec_i[decade_bucket(clean_year(y))]] = 1.0

    people_lists = [extract_people(meta[sid]) for sid in show_ids]
    people_mat = build_people_tfidf(people_lists)

    return show_index, l2_normalize(cat_mat), l2_normalize(genre_mat), l2_normalize(year_mat), people_mat


def combine_blocks(cat_mat, genre_mat, year_mat, people_mat, w_cat, w_genre, w_year, w_people) -> np.ndarray:
    parts = []
    if w_cat:
        parts.append(w_cat * cat_mat)
    if w_genre:
        parts.append(w_genre * genre_mat)
    if w_year:
        parts.append(w_year * year_mat)
    if w_people:
        parts.append(w_people * people_mat)
    V = np.hstack(parts) if parts else np.zeros((cat_mat.shape[0], 0))
    return l2_normalize(V)


def build_profiles(user_items: dict, show_index: dict, V: np.ndarray) -> dict:
    """user -> weighted mean (by n_days) feature vector."""
    feat = V.shape[1]
    profiles: dict[int, np.ndarray] = {}
    for u, items in user_items.items():
        vec = np.zeros(feat, dtype=np.float64)
        for s, nd in items.items():
            si = show_index.get(s)
            if si is not None:
                vec += nd * V[si]
        profiles[int(u)] = vec
    return profiles


def recommend_cbf(profiles, V, show_index, candidates, global_rank, cohort, k=5):
    cand_ids = sorted(s for s in candidates if s in show_index)
    cand_idx = np.array([show_index[s] for s in cand_ids], dtype=np.int64)
    Vc = V[cand_idx]
    rankings: dict[int, list[int]] = {}
    n_pad = 0
    for u in cohort:
        p = profiles.get(int(u))
        if p is None or float(np.linalg.norm(p)) == 0.0:
            rec = list(global_rank[:k])
            n_pad += k
        else:
            pn = p / np.linalg.norm(p)
            scores = pn @ Vc.T
            order = np.lexsort((np.asarray(cand_ids, dtype=np.int64), -scores))[:k]
            rec = [cand_ids[i] for i in order]
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


def normalize_title(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"^(м\s*/\s*с|х\s*/\s*ф|т\s*/\s*с)\s*['\"«»]*\s*", "", t)
    t = re.split(r"[,.]", t)[0]
    for ch in "'\"«»":
        t = t.replace(ch, "")
    t = re.sub(r"\b\d+\s*(с|сезон|серия|серий|эп|эпизод)\w*\b", "", t)
    t = re.sub(r"\s+", " ", t).strip(" -:;—!?")
    return t


def build_tfidf(token_lists: list[list[str]], min_df: int = 2) -> np.ndarray:
    N = len(token_lists)
    df: Counter[str] = Counter()
    for toks in token_lists:
        for tok in set(toks):
            df[tok] += 1
    vocab = [tok for tok, c in df.items() if c >= min_df]
    v2i = {tok: i for i, tok in enumerate(vocab)}
    M = np.zeros((N, len(vocab)), dtype=np.float64)
    for j, toks in enumerate(token_lists):
        for tok in toks:
            if tok in v2i:
                M[j, v2i[tok]] += 1.0
    df_arr = np.array([df[tok] for tok in vocab], dtype=np.float64)
    idf = np.log((1.0 + N) / (1.0 + df_arr)) + 1.0
    M = M * idf
    return l2_normalize(M)


def build_title_blocks(show_ids: list[int], title_map: dict):
    """Title blocks: (1) exact base-title one-hot, (2) TF-IDF over title word tokens."""
    bases = [normalize_title(title_map.get(s, str(s))) for s in show_ids]
    distinct = sorted({b for b in bases if b})
    base_i = {b: i for i, b in enumerate(distinct)}
    n = len(show_ids)
    base_onehot = np.zeros((n, len(distinct)), dtype=np.float64)
    for j, b in enumerate(bases):
        if b:
            base_onehot[j, base_i[b]] = 1.0
    base_onehot = l2_normalize(base_onehot)
    title_tfidf = build_tfidf([b.split() for b in bases], min_df=2)
    return base_onehot, title_tfidf


def recommend_cbf_pop(profiles, V, show_index, candidates, pop_count, global_rank, cohort, beta, k=5):
    cand_ids = sorted(s for s in candidates if s in show_index)
    cand_idx = np.array([show_index[s] for s in cand_ids], dtype=np.int64)
    Vc = V[cand_idx]
    pop_log = np.array([np.log1p(pop_count.get(int(s), 0)) for s in cand_ids], dtype=np.float64)
    rankings: dict[int, list[int]] = {}
    n_pad = 0
    for u in cohort:
        p = profiles.get(int(u))
        if p is None or float(np.linalg.norm(p)) == 0.0:
            rec = list(global_rank[:k])
            n_pad += k
        else:
            pn = p / np.linalg.norm(p)
            scores = pn @ Vc.T
            mu = scores.mean()
            sd = scores.std()
            z = (scores - mu) / sd if sd > 0 else np.zeros_like(scores)
            combined = z + beta * pop_log
            order = np.lexsort((np.asarray(cand_ids, dtype=np.int64), -combined))[:k]
            rec = [cand_ids[i] for i in order]
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
