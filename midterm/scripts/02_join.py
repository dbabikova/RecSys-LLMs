"""Stage 2 driver: join the viewing log with the program grid.

Produces cache/interactions_{train,val}.parquet and cache/user_context.parquet,
plus results/join_summary.md.
"""

import gc
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
import join as J

CAP = 4 * 3600
MIN_OVERLAP = 300
FRAC = 0.5
TRAIN_BORDER = J.to_unix(datetime(2020, 5, 18, 0, 0, 0))
VAL_END = J.to_unix(datetime(2020, 7, 26, 23, 59, 59))

md: list[str] = []


def out(*args) -> None:
    line = " ".join(str(a) for a in args)
    print(line)
    md.append(line)


def fmt_ts(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def build_interactions(lab: pd.DataFrame, min_ov: float, frac: float) -> pd.DataFrame:
    w = J.watched_mask(lab["ov"].to_numpy(), lab["dur"].to_numpy(), min_ov, frac)
    d = lab.loc[w].copy()
    d["air_key"] = d["ch"].astype(np.int64) * (1 << 32) + d["ast"].astype(np.int64)
    agg = (
        d.groupby(["user", "show"], sort=False)
        .agg(
            n_airings=("air_key", "nunique"),
            overlap_sec=("ov", "sum"),
            last_seen=("vst", "max"),
        )
        .reset_index()
        .rename(columns={"user": "user_id", "show": "tv_show_id"})
    )
    agg["overlap_sec"] = agg["overlap_sec"].astype(np.int64)
    return agg


def load_submission_users() -> set[int]:
    s = set()
    with open(config.SUBMISSION_FILE, encoding="utf-8-sig", newline="") as f:
        next(f)
        for line in f:
            if line.strip():
                s.add(int(line.split(",")[0]))
    return s


def main() -> None:
    out("# Склейка просмотров с сеткой (этап 2)")
    out()

    print("Loading grid ...", flush=True)
    g = J.load_grid()
    n_labeled = int((g["sid"] != 0).sum())
    out(f"Сетка: {len(g['st']):,} строк после удаления {g['n_dup']:,} точных дублей; "
        f"помеченных передач (tv_show_id!=0) — {n_labeled:,}.")

    print("Loading log ...", flush=True)
    log = J.load_log(CAP)
    out(f"Лог: {len(log['st']):,} строк; обрезано до {CAP // 3600} ч — {log['clipped']:,}.")
    out()

    # user_id is uint64 (does not fit int64); factorize to int64 codes
    user_map, log["user"] = np.unique(log["user"], return_inverse=True)
    log["user"] = log["user"].astype(np.int64)

    # per-user log statistics (before freeing the log)
    print("Per-user log stats ...", flush=True)
    ldf = pd.DataFrame({"user": log["user"], "ch": log["ch"], "dur": log["en"] - log["st"]})
    total_sec = ldf.groupby("user", sort=False)["dur"].sum().rename("total_sec")
    uch = (
        ldf.groupby(["user", "ch"], sort=False)["dur"]
        .sum()
        .reset_index()
        .sort_values(["user", "dur"], ascending=[True, False])
    )
    uch["rank"] = uch.groupby("user").cumcount()
    top = uch[uch["rank"] < 3]
    del ldf
    gc.collect()

    print("Joining ...", flush=True)
    pairs = J.join_channels(g, log)
    user, show, ch, ast, ov, dur, vst = pairs
    del pairs
    for k in ("ch", "st", "en", "sid", "dur"):
        g[k] = None
    del log
    gc.collect()

    pdf = pd.DataFrame(
        {"user": user, "show": show, "ch": ch, "ast": ast, "ov": ov, "dur": dur, "vst": vst}
    )
    del user, show, ch, ast, ov, dur, vst
    gc.collect()

    labeled = pdf[pdf["show"] != 0]
    out(f"Пар (просмотр–эфир) всего: {len(pdf):,}; помеченных (tv_show_id!=0): {len(labeled):,}.")
    out()

    # ---- user context ---------------------------------------------------
    labeled_sec = labeled.groupby("user", sort=False)["ov"].sum().rename("labeled_sec")
    unlabeled_sec = pdf[pdf["show"] == 0].groupby("user", sort=False)["ov"].sum().rename("unlabeled_sec")

    ctx = total_sec.to_frame()
    ctx = ctx.join(labeled_sec).join(unlabeled_sec)
    ctx["labeled_sec"] = ctx["labeled_sec"].fillna(0)
    ctx["unlabeled_sec"] = ctx["unlabeled_sec"].fillna(0)
    ctx["no_program_sec"] = (ctx["total_sec"] - ctx["labeled_sec"] - ctx["unlabeled_sec"]).clip(lower=0)
    for r in range(3):
        t = top[top["rank"] == r].set_index("user")[["ch", "dur"]]
        t = t.rename(columns={"ch": f"top{r + 1}_ch", "dur": f"top{r + 1}_sec"})
        ctx = ctx.join(t)
    ctx = ctx.reset_index().rename(columns={"user": "user_id"})
    for c in ("total_sec", "labeled_sec", "unlabeled_sec", "no_program_sec"):
        ctx[c] = ctx[c].astype(np.int64)
    ctx["user_id"] = user_map[ctx["user_id"].to_numpy()]
    ctx.to_parquet(config.CACHE_DIR / "user_context.parquet", index=False)

    # ---- interactions per window ---------------------------------------
    train_lab = labeled[labeled["vst"] < TRAIN_BORDER]
    val_lab = labeled[(labeled["vst"] >= TRAIN_BORDER) & (labeled["vst"] <= VAL_END)]

    itr = build_interactions(train_lab, MIN_OVERLAP, FRAC)
    iva = build_interactions(val_lab, MIN_OVERLAP, FRAC)
    itr["user_id"] = user_map[itr["user_id"].to_numpy()]
    iva["user_id"] = user_map[iva["user_id"].to_numpy()]
    itr.to_parquet(config.CACHE_DIR / "interactions_train.parquet", index=False)
    iva.to_parquet(config.CACHE_DIR / "interactions_val.parquet", index=False)

    out("## Окна (по дате начала просмотра)")
    out()
    out("| окно | диапазон дат | пар | пользователей | передач |")
    out("|---|---|---|---|---|")
    out(f"| train | до {fmt_ts(TRAIN_BORDER)} | {len(itr):,} | {itr['user_id'].nunique():,} | {itr['tv_show_id'].nunique():,} |")
    out(f"| val | {fmt_ts(TRAIN_BORDER)} – {fmt_ts(VAL_END)} | {len(iva):,} | {iva['user_id'].nunique():,} | {iva['tv_show_id'].nunique():,} |")
    out()

    # ---- summary metrics ------------------------------------------------
    total_all = int(ctx["total_sec"].sum())
    labeled_all = int(ctx["labeled_sec"].sum())
    unlabeled_all = int(ctx["unlabeled_sec"].sum())
    no_prog_all = int(ctx["no_program_sec"].sum())
    out("## Доля размеченного эфира")
    out()
    out(f"Всего секунд просмотра: {total_all:,}")
    out(f"На размеченных эфирах: {labeled_all:,} ({labeled_all / total_all * 100:.1f}%)")
    out(f"На неразмеченных (tv_show_id=0): {unlabeled_all:,} ({unlabeled_all / total_all * 100:.1f}%)")
    out(f"Не нашли передачу вообще (каналы вне сетки, дыры): {no_prog_all:,} ({no_prog_all / total_all * 100:.1f}%)")
    out()

    # ---- test users with 0 labeled -------------------------------------
    sub = load_submission_users()
    train_users = set(int(x) for x in itr["user_id"].tolist())
    val_users = set(int(x) for x in iva["user_id"].tolist())
    out("## Пользователи submission (1260)")
    out()
    out(f"С 0 размеченных передач в train: {len(sub - train_users):,} из {len(sub):,}")
    out(f"С 0 размеченных передач в val:   {len(sub - val_users):,} из {len(sub):,}")
    out()

    # ---- pair counts at thresholds -------------------------------------
    out("## Чувствительность к порогу overlap")
    out()
    out("| min overlap, с | пар (пользователь–передача) |")
    out("|---|---|")
    lab_ov = labeled["ov"].to_numpy()
    lab_dur = labeled["dur"].to_numpy()
    for thr in (60, 300, 900):
        w = J.watched_mask(lab_ov, lab_dur, thr, FRAC)
        n_pairs = labeled.iloc[w].groupby(["user", "show"], sort=False).ngroups
        out(f"| {thr} | {n_pairs:,} |")
    out()

    # ---- sample 5 pairs -------------------------------------------------
    out("## 5 случайных пар для ручной проверки")
    out()
    rng = np.random.default_rng(config.SEED)
    w = J.watched_mask(lab_ov, lab_dur, MIN_OVERLAP, FRAC)
    cand = labeled.loc[w]
    idxs = rng.choice(len(cand), size=min(5, len(cand)), replace=False)
    out("| user_id | канал | название | эфир | просмотр | overlap, с |")
    out("|---|---|---|---|---|---|")
    for i in idxs:
        r = cand.iloc[i]
        cid = int(r["ch"])
        title = g["title_map"].get((cid, int(r["ast"])), "?")
        ch_title = g["ch_title_map"].get(cid, "?")
        out(
            f"| {int(user_map[int(r['user'])])} | {cid} ({ch_title}) | {title[:60]} | "
            f"{fmt_ts(int(r['ast']))} | {fmt_ts(int(r['vst']))} | {int(r['ov'])} |"
        )
    out()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "join_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nСводка сохранена в", config.RESULTS_DIR / "join_summary.md")


if __name__ == "__main__":
    main()
