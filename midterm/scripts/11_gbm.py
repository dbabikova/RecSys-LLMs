"""Stage 11: LightGBM ranker (lambdarank + binary) over h1 features.

Snapshot = (REF, END, HSTART): features only from history [HSTART, REF), labels from [REF, END).
Rows: shows the user watched in history that air in the window + global top-20 for every cohort user.
Final score = sum of within-user ranks of predictions over 2 objectives x 3 seeds.
"""

import hashlib
import importlib.util
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import evaluate as E  # noqa: E402
import cbf as CBF  # noqa: E402
from join import to_unix  # noqa: E402

_spec = importlib.util.spec_from_file_location("h1mod", ROOT / "scripts" / "10_h1.py")
H1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H1)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WEEK = 7 * 86400
SEEDS = (1, 2, 3)
OBJECTIVES = ("lambdarank", "binary")
FEATURES = ["p2", "p4", "p8", "p", "c80", "c80_4", "c50", "ovh", "n", "nd", "since", "first",
            "slot", "n_air_wk", "n_air_h", "popc", "cat", "dur", "u_ovh", "u_shows", "u_days",
            "share", "h1", "h1_rank"]

md: list[str] = []


def out(*args) -> None:
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    md.append(line)


def ts(y, m, d):
    return to_unix(datetime(y, m, d, 0, 0, 0))


# --------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        self.sub_int = sorted(int(u) for u in E.load_submission_ids())
        self.uindex = pd.Index(np.array(self.sub_int, dtype=np.uint64))
        print("Loading per-airing matches ...", flush=True)
        m = E.load_labeled_airings(E.CAP)
        agg = (m.groupby(["user_id", "tv_show_id", "channel", "air_start"], sort=True)
                .agg(ov=("overlap", "sum"), dur=("duration", "first"), vst=("view_start", "min"))
                .reset_index())
        del m
        agg["pct"] = (agg["ov"] / agg["dur"].replace(0, np.nan)).clip(0, 1).fillna(0.0)
        agg["ui"] = self.uindex.get_indexer(agg["user_id"].to_numpy(dtype=np.uint64))
        self.agg = agg
        print("Loading grid / log / metadata ...", flush=True)
        self.grid = H1.load_grid()
        self.log = H1.load_log()
        meta, _c, _t, _a = CBF.load_show_metadata()
        cats = sorted({v[0] for v in meta.values()})
        code = {c: i + 1 for i, c in enumerate(cats)}
        self.cat = {sid: code[v[0]] for sid, v in meta.items()}
        lu = self.uindex.get_indexer(self.log[0])
        self.log_ui = lu


def slot_matrix(D, REF, END, HSTART, w_sids, w_chs, w_sts, w_durs, shows):
    """slot(u, s) as in stage 10 h1: profile(u, ch, hour) x share of airing duration in that hour."""
    us, chs, sts, ens = D.log
    msk = (sts >= HSTART) & (sts < REF) & (D.log_ui >= 0)
    ui = D.log_ui[msk]
    ch = chs[msk]
    st = sts[msk]
    d = np.clip(ens[msk] - st, 0, E.CAP).astype(np.float64)
    how = ((st // 86400 + 3) % 7) * 24 + (st % 86400) // 3600
    key = ch * 168 + how
    pf = pd.DataFrame({"ui": ui, "k": key, "d": d}).groupby(["ui", "k"], sort=True)["d"].sum().reset_index()
    tot = pf.groupby("ui")["d"].transform("sum")
    pf["w"] = np.where(tot > 0, pf["d"] / tot, 0.0)

    a_s, a_k, a_w = [], [], []
    for sid, c, s0, du in zip(w_sids.tolist(), w_chs.tolist(), w_sts.tolist(), w_durs.tolist()):
        for hw, fr in H1.hour_fractions(s0, du):
            a_s.append(sid)
            a_k.append(c * 168 + hw)
            a_w.append(fr)
    keys = np.unique(np.concatenate([pf["k"].to_numpy(), np.array(a_k, dtype=np.int64)]))
    kidx = pd.Index(keys)
    sidx = pd.Index(shows)
    P = sp.csr_matrix((pf["w"].to_numpy(), (pf["ui"].to_numpy(), kidx.get_indexer(pf["k"].to_numpy()))),
                      shape=(len(D.sub_int), len(keys)))
    A = sp.csr_matrix((np.array(a_w), (sidx.get_indexer(np.array(a_s)), kidx.get_indexer(np.array(a_k)))),
                      shape=(len(shows), len(keys)))
    return np.asarray((P @ A.T).todense()), sidx


def snapshot(D, REF, END, HSTART, labels=True):
    t0 = time.time()
    agg = D.agg
    sids, chs, sts, durs = D.grid
    wm = (sts >= REF) & (sts < END)
    w_sids, w_chs, w_sts, w_durs = sids[wm], chs[wm], sts[wm], durs[wm]
    shows = np.unique(w_sids)
    wset = set(shows.tolist())
    weeks = (END - REF) / WEEK
    n_air = pd.Series(w_sids).value_counts()
    dur_med = pd.Series(w_durs, index=w_sids).groupby(level=0).median()
    hm = (sts >= REF - 10 * WEEK) & (sts < REF)
    n_air_h = pd.Series(sids[hm]).value_counts() / 10.0

    Hh = agg[(agg["air_start"] >= HSTART) & (agg["air_start"] < REF)]
    popc = Hh[Hh["pct"] >= 0.8].groupby("tv_show_id").size()
    Hc = Hh[Hh["ui"] >= 0]
    age = (REF - Hc["vst"].to_numpy()) / WEEK
    pct = Hc["pct"].to_numpy()
    i80 = (pct >= 0.8).astype(np.float64)
    d = pd.DataFrame({
        "ui": Hc["ui"].to_numpy(), "s": Hc["tv_show_id"].to_numpy(),
        "w2": pct * 0.5 ** (age / 2), "w4": pct * 0.5 ** (age / 4), "w8": pct * 0.5 ** (age / 8),
        "pct": pct, "i80": i80, "i80_4": i80 * 0.5 ** (age / 4), "i50": (pct >= 0.5).astype(np.float64),
        "ov": Hc["ov"].to_numpy().astype(np.float64), "day": Hc["vst"].to_numpy() // 86400,
        "vst": Hc["vst"].to_numpy(),
        "watched": ((Hc["ov"].to_numpy() >= 300) | (pct >= 0.5)),
    })
    usr = d.groupby("ui").agg(u_ov=("ov", "sum"), u_shows=("s", "nunique"), u_days=("day", "nunique"))
    g = d.groupby(["ui", "s"], sort=True).agg(
        p2=("w2", "sum"), p4=("w4", "sum"), p8=("w8", "sum"), p=("pct", "sum"), c80=("i80", "sum"),
        c80_4=("i80_4", "sum"), c50=("i50", "sum"), ov=("ov", "sum"), n=("pct", "size"),
        nd=("day", "nunique"), last=("vst", "max"), first_=("vst", "min")).reset_index()
    g = g[g["s"].isin(wset)]

    # global top-20: unique cohort users who watched the show (>=300 s or pct >= 0.5), window shows only
    gp = d[d["watched"]].groupby("s")["ui"].nunique()
    gp = gp[gp.index.isin(wset)]
    top = sorted(gp.index.tolist(), key=lambda s: (-gp[s], s))[:20]
    gpos = {s: i for i, s in enumerate(top)}

    extra = pd.DataFrame({"ui": np.repeat(np.arange(len(D.sub_int)), len(top)),
                          "s": np.tile(np.array(top, dtype=np.int64), len(D.sub_int))})
    rows = pd.concat([g[["ui", "s"]], extra]).drop_duplicates().sort_values(["ui", "s"]).reset_index(drop=True)
    rows = rows.merge(g, on=["ui", "s"], how="left")
    has = rows["n"].notna()
    for c in ("p2", "p4", "p8", "p", "c80", "c80_4", "c50", "ov", "n", "nd"):
        rows[c] = rows[c].fillna(0.0)
    rows["since"] = np.where(has, (REF - rows["last"]) / WEEK, 99.0)
    rows["first"] = np.where(has, (REF - rows["first_"]) / WEEK, 99.0)
    rows["ovh"] = rows["ov"] / 3600.0

    M, sidx = slot_matrix(D, REF, END, HSTART, w_sids, w_chs, w_sts, w_durs, shows)
    rows["slot"] = M[rows["ui"].to_numpy(), sidx.get_indexer(rows["s"].to_numpy())]
    s_arr = rows["s"]
    rows["n_air_wk"] = s_arr.map(n_air).fillna(0).to_numpy() / weeks
    rows["n_air_h"] = s_arr.map(n_air_h).fillna(0.0)
    rows["popc"] = s_arr.map(popc).fillna(0.0)
    rows["cat"] = s_arr.map(D.cat).fillna(0).astype(int)
    rows["dur"] = s_arr.map(dur_med).fillna(0.0)
    rows = rows.merge(usr, left_on="ui", right_index=True, how="left")
    for c in ("u_ov", "u_shows", "u_days"):
        rows[c] = rows[c].fillna(0.0)
    rows["u_ovh"] = rows["u_ov"] / 3600.0
    rows["share"] = np.where(rows["u_ov"] > 0, rows["ov"] / rows["u_ov"].where(rows["u_ov"] > 0, 1), 0.0)
    rows["h1"] = rows["p4"] * np.sqrt(rows["slot"] + 1e-4)
    rows["gpos"] = s_arr.map(gpos).fillna(999).astype(int)
    order = np.lexsort((rows["s"].to_numpy(), -rows["h1"].to_numpy(), rows["ui"].to_numpy()))
    rk = np.empty(len(rows), dtype=np.int64)
    tmp = rows.iloc[order]
    rk[order] = tmp.groupby("ui").cumcount().to_numpy() + 1
    rows["h1_rank"] = rk

    gt = {}
    if labels:
        W = agg[(agg["air_start"] >= REF) & (agg["air_start"] < END) & (agg["pct"] >= 0.8) & (agg["ui"] >= 0)]
        cnt = W.groupby(["ui", "tv_show_id"]).size().rename("cnt").reset_index()
        cnt = cnt.sort_values(["ui", "cnt", "tv_show_id"], ascending=[True, False, True])
        for ui, grp in cnt.groupby("ui", sort=True):
            gt[int(ui)] = set(int(x) for x in grp["tv_show_id"].head(5))
        rows = rows.merge(cnt.rename(columns={"tv_show_id": "s"}), on=["ui", "s"], how="left")
        rows["cnt"] = rows["cnt"].fillna(0).astype(int)
        rows["y"] = [int(s in gt.get(int(u), ())) for u, s in zip(rows["ui"].tolist(), rows["s"].tolist())]
    rows = rows.sort_values(["ui", "s"]).reset_index(drop=True)
    print(f"  snapshot REF={datetime.utcfromtimestamp(REF):%Y-%m-%d} HSTART="
          f"{datetime.utcfromtimestamp(max(HSTART, 0)):%Y-%m-%d}: {len(rows)} rows, GT users {len(gt)}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return dict(rows=rows, gt=gt, top=top, wset=wset, REF=REF, END=END, HSTART=HSTART)


# --------------------------------------------------------------------------- models
def train_models(train_rows_list):
    tr = pd.concat([r.assign(snap=i) for i, r in enumerate(train_rows_list)], ignore_index=True)
    tr = tr.sort_values(["snap", "ui", "s"]).reset_index(drop=True)
    X = tr[FEATURES].to_numpy(dtype=np.float64)
    groups = tr.groupby(["snap", "ui"], sort=False).size().to_numpy()
    models = []
    for obj in OBJECTIVES:
        for seed in SEEDS:
            params = dict(objective=obj, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                          seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed,
                          deterministic=True, force_col_wise=True, num_threads=4, verbose=-1)
            if obj == "lambdarank":
                params["label_gain"] = list(range(31))
                ds = lgb.Dataset(X, label=np.clip(tr["cnt"].to_numpy(), 0, 30), group=groups,
                                 feature_name=FEATURES, categorical_feature=["cat"])
            else:
                ds = lgb.Dataset(X, label=tr["y"].to_numpy(), feature_name=FEATURES, categorical_feature=["cat"])
            models.append((obj, seed, lgb.train(params, ds, num_boost_round=200)))
    return models


def ensemble_score(models, rows):
    X = rows[FEATURES].to_numpy(dtype=np.float64)
    score = np.zeros(len(rows))
    ui = rows["ui"].to_numpy()
    for _o, _s, b in models:
        score += pd.Series(b.predict(X)).groupby(ui).rank(method="average").to_numpy()
    return score


def top5(rows, score, top, secondary=None):
    ui = rows["ui"].to_numpy()
    s = rows["s"].to_numpy()
    keys = (s,) if secondary is None else (s, secondary)
    order = np.lexsort(keys + (-score, ui))
    res = {}
    for u, sid in zip(ui[order].tolist(), s[order].tolist()):
        lst = res.setdefault(u, [])
        if len(lst) < 5:
            lst.append(sid)
    return {u: H1.pad5(v, top) for u, v in res.items()}


def d80(rank, gt, users=None):
    us = sorted(gt) if users is None else [u for u in sorted(gt) if u in users]
    return float(np.mean([E.ap_at_k(rank.get(u, []), gt[u]) for u in us]))


def h1_on_rows(snap):
    r = snap["rows"]
    # tie-break of h1 = 0 rows (global top-20 without history) by popularity position, then id
    return top5(r, r["h1"].to_numpy(), snap["top"], secondary=r["gpos"].to_numpy())


# --------------------------------------------------------------------------- main
def main():
    t_all = time.time()
    out("# Этап 11: LightGBM поверх признаков h1")
    out()
    out(f"lightgbm {lgb.__version__}; seeds {SEEDS}; objectives {OBJECTIVES}; 200 trees, lr 0.05, leaves 31")
    out()
    D = Data()
    VREF, VEND = ts(2020, 5, 18), ts(2020, 7, 27)

    val = snapshot(D, VREF, VEND, 0)
    gt = val["gt"]
    rows = val["rows"]
    out(f"val: строк {len(rows)}, пользователей с GT-D80 {len(gt)}, "
        f"доля GT-пар, попавших в строки: "
        f"{rows['y'].sum() / sum(len(v) for v in gt.values()):.3f}")
    h1_val = h1_on_rows(val)
    m_h1 = d80(h1_val, gt)
    out(f"h1 на строках val: D80 = {m_h1:.4f}")
    out()

    # ---- a) 4 folds by users
    out("## a) 4 фолда по пользователям внутри val")
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(D.sub_int))
    fold_of = np.empty(len(D.sub_int), dtype=int)
    fold_of[perm] = np.arange(len(perm)) % 4
    gbm_a = {}
    for f in range(4):
        tr = rows[fold_of[rows["ui"].to_numpy()] != f]
        te = rows[fold_of[rows["ui"].to_numpy()] == f].reset_index(drop=True)
        models = train_models([tr])
        gbm_a.update(top5(te, ensemble_score(models, te), val["top"]))
        fu = {u for u in gt if fold_of[u] == f}
        out(f"fold {f}: GBM {d80(gbm_a, gt, fu):.4f} | h1 {d80(h1_val, gt, fu):.4f} ({len(fu)} польз.)")
    m_a = d80(gbm_a, gt)
    out(f"**итого (OOF): GBM D80 = {m_a:.4f}, h1 D80 = {m_h1:.4f}, разница {m_a - m_h1:+.4f}**")
    out()

    # ---- b) temporal: train on (VREF-4w, VREF) and (VREF-3w, VREF), HSTART = their REF - 6w
    out("## b) честная временная валидация")
    s1 = snapshot(D, VREF - 4 * WEEK, VREF, VREF - 10 * WEEK)
    s2 = snapshot(D, VREF - 3 * WEEK, VREF, VREF - 9 * WEEK)
    models_b = train_models([s1["rows"], s2["rows"]])
    gbm_b = top5(rows, ensemble_score(models_b, rows), val["top"])
    m_b = d80(gbm_b, gt)
    out(f"обучение: снимки REF 04-20 и 04-27 (история 6 нед), тест val HSTART=0 (история 10 нед): "
        f"GBM {m_b:.4f} | h1 {m_h1:.4f} ({m_b - m_h1:+.4f})")
    val6 = snapshot(D, VREF, VEND, VREF - 6 * WEEK)
    gbm_b6 = top5(val6["rows"], ensemble_score(models_b, val6["rows"]), val6["top"])
    h1_v6 = h1_on_rows(val6)
    m_b6, m_h16 = d80(gbm_b6, val6["gt"]), d80(h1_v6, val6["gt"])
    out(f"тот же тест с историей 6 нед (HSTART = 2020-04-06, масштаб как в обучении): "
        f"GBM {m_b6:.4f} | h1 {m_h16:.4f} ({m_b6 - m_h16:+.4f})")
    out()

    # ---- submission
    out("## Сабмит")
    SREF = ts(2020, 7, 27)
    SEND = SREF + 12 * WEEK
    models_s = train_models([rows])
    test = snapshot(D, SREF, SEND, SREF - 10 * WEEK, labels=False)
    sub_rank = top5(test["rows"], ensemble_score(models_s, test["rows"]), test["top"])

    header, ids_str = None, []
    with open(config.SUBMISSION_FILE, encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n")
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                ids_str.append(line.split(",", 1)[0])
    outp = config.SUBMISSIONS_DIR / "final_submission_gbm_oc.csv"
    with open(outp, "w", newline="") as f:
        f.write(header + "\n")
        for us in ids_str:
            ui = int(D.uindex.get_loc(np.uint64(int(us))))
            f.write(f"{us},{' '.join(str(x) for x in sub_rank.get(ui, test['top'][:5])[:5])}\n")
    lines = outp.read_text(encoding="utf-8").splitlines()
    preds = [[int(x) for x in l.split(",", 1)[1].split()] for l in lines[1:]]
    checks = {
        "1261 строка (заголовок + 1260)": len(lines) == 1261 and lines[0] == header,
        "user_id и порядок как в submission.csv": [l.split(",", 1)[0] for l in lines[1:]] == ids_str,
        "ровно 5 уникальных id": all(len(p) == 5 and len(set(p)) == 5 for p in preds),
        "нет tv_show_id = 0": all(0 not in p for p in preds),
        "все id идут в окне 31-42 (REF..REF+12 нед)": all(all(x in test["wset"] for x in p) for p in preds),
        "pandas читает 1260 x 2": pd.read_csv(outp, dtype=str).shape == (1260, 2),
    }
    for k, v in checks.items():
        out(f"{'PASS' if v else 'FAIL'}: {k}")
    assert all(checks.values())
    md5 = hashlib.md5(outp.read_bytes()).hexdigest()
    out(f"md5 final_submission_gbm_oc.csv = {md5}")
    ref = config.SUBMISSIONS_DIR / "final_submission_gbm.csv"
    ref_md5 = hashlib.md5(ref.read_bytes()).hexdigest()
    out(f"эталон claude_gbm = final_submission_gbm.csv, md5 {ref_md5} "
        f"({'совпадает с 5c9bb3e2…' if ref_md5 == '5c9bb3e246f783f25cdb1cc0d6dfd06b' else 'НЕ совпадает с 5c9bb3e2…'})")
    v = lambda df: [tuple(int(x) for x in s.split()) for s in df["tv_show_id"]]
    a, b = v(pd.read_csv(ref, dtype=str)), v(pd.read_csv(outp, dtype=str))
    out(f"сравнение: одинаковых строк {sum(x == y for x, y in zip(a, b))}/1260, "
        f"совпадает состав топ-5 у {sum(set(x) == set(y) for x, y in zip(a, b))}, "
        f"среднее пересечение {np.mean([len(set(x) & set(y)) for x, y in zip(a, b)]):.2f}/5, "
        f"совпадение позиций {np.mean([i == j for x, y in zip(a, b) for i, j in zip(x, y)]) * 100:.1f}%")
    h1sub = v(pd.read_csv(config.SUBMISSIONS_DIR / "final_submission_h1_oc.csv", dtype=str))
    out(f"пересечение gbm_oc с h1_oc: {np.mean([len(set(x) & set(y)) for x, y in zip(h1sub, b)]):.2f}/5; "
        f"эталон gbm с h1: {np.mean([len(set(x) & set(y)) for x, y in zip(h1sub, a)]):.2f}/5")
    out()

    # ---- feature shift train(val snapshot) vs test snapshot
    out("## Сдвиг признаков: обучение (val, весна, история 10 нед) vs тест (лето, история 10 нед)")
    out("| признак | обучение: среднее | тест: среднее | отношение |")
    out("|---|---|---|---|")
    tr_r, te_r = rows, test["rows"]
    shift = []
    for c in ["rows/user", "hist rows share"] + FEATURES:
        if c == "rows/user":
            x, y = len(tr_r) / tr_r["ui"].nunique(), len(te_r) / te_r["ui"].nunique()
        elif c == "hist rows share":
            x, y = (tr_r["n"] > 0).mean(), (te_r["n"] > 0).mean()
        elif c == "cat":
            continue
        else:
            x, y = tr_r[c].mean(), te_r[c].mean()
        ratio = y / x if x else float("nan")
        shift.append((c, x, y, ratio))
        out(f"| {c} | {x:.4g} | {y:.4g} | {ratio:.2f} |")
    out()

    # ---- importance (gain, submission models)
    imp = pd.DataFrame({f"{o}_{s}": b.feature_importance("gain") for o, s, b in models_s}, index=FEATURES)
    imp_lr = imp[[c for c in imp.columns if c.startswith("lambdarank")]].mean(axis=1)
    imp_bn = imp[[c for c in imp.columns if c.startswith("binary")]].mean(axis=1)
    share = (imp_lr / imp_lr.sum() + imp_bn / imp_bn.sum()) / 2
    top10 = share.sort_values(ascending=False).head(10)
    im = ["# Важность признаков LightGBM (gain), модели сабмита (обучение на val-снимке)", "",
          "Доля gain усреднена по 3 сидам внутри цели, затем по двум целям (lambdarank, binary).", "",
          "| # | признак | доля gain (среднее) | lambdarank | binary |", "|---|---|---|---|---|"]
    for i, (f, sv) in enumerate(top10.items(), 1):
        im.append(f"| {i} | {f} | {sv:.3f} | {imp_lr[f] / imp_lr.sum():.3f} | {imp_bn[f] / imp_bn.sum():.3f} |")
    (config.RESULTS_DIR / "gbm_importance.md").write_text("\n".join(im) + "\n", encoding="utf-8")
    out("Важность (топ-10): " + ", ".join(f"{f} {sv:.3f}" for f, sv in top10.items()))
    out()

    # ---- all_methods.csv
    df = pd.read_csv(config.RESULTS_DIR / "all_methods.csv")
    new = [("GBM (4 фолда val)", m_a), ("GBM (временная, тест 10 нед)", m_b),
           ("GBM (временная, тест 6 нед)", m_b6), ("h1 (строки GBM)", m_h1)]
    for name, val_ in new:
        df = df[df["method"] != name]
        df = pd.concat([df, pd.DataFrame([{"method": name, "MAP5_D80": val_}])], ignore_index=True)
    df.to_csv(config.RESULTS_DIR / "all_methods.csv", index=False)

    # ---- local_vs_kaggle.md: GBM section (replaced on rerun; text after the marker is kept)
    p = config.RESULTS_DIR / "local_vs_kaggle.md"
    txt = p.read_text(encoding="utf-8")
    head, _sep, rest = txt.partition("\n## GBM (этап 11)")
    keep = ""
    if "\n### Почему" in rest:
        keep = rest[rest.index("\n### Почему"):]
    sec = ["", "## GBM (этап 11)", "",
           "| метод | локальный D80 (val) | Kaggle private |", "|---|---|---|",
           f"| h1 (на строках GBM) | {m_h1:.4f} | 0.41508 |",
           f"| GBM, 4 фолда по пользователям | {m_a:.4f} | — |",
           f"| GBM, временная (тест 10 нед / 6 нед истории) | {m_b:.4f} / {m_b6:.4f} | — |",
           "| claude_gbm (= final_submission_gbm.csv) | ~0.42 (эталон) | **0.41306** |",
           f"| gbm_oc (этот скрипт, md5 {md5[:8]}…) | {m_a:.4f} | не отправлялся |", "",
           "claude_gbm на Kaggle (0.41306) ниже h1 (0.41508), хотя локально D80 GBM выше h1."]
    p.write_text(head.rstrip("\n") + "\n" + "\n".join(sec) + "\n" + keep, encoding="utf-8")

    (config.RESULTS_DIR / "tune_11.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nГотово за {time.time() - t_all:.0f}s; лог results/tune_11.md", flush=True)


if __name__ == "__main__":
    main()
