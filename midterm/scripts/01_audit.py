"""Stage 1: data audit for the sweet.tv TV program recommender.

Streaming-only: no large file is ever fully loaded; aggregates are computed on
the fly. Prints tables to stdout and writes a summary to results/audit_summary.md.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config

LOG_ENC = "utf-8-sig"
GRID_ENC = "utf-8"
GRID_TIME_FMT = "%d.%m.%Y %H:%M:%S"

md_lines: list[str] = []


def out(*args: object) -> None:
    line = " ".join(str(a) for a in args)
    print(line)
    md_lines.append(line)


def dist_stats(values: list[float]) -> dict[str, float] | None:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def parse_log_time(s: str) -> datetime:
    return datetime.fromisoformat(s)


def parse_grid_time(s: str) -> datetime:
    return datetime.strptime(s, GRID_TIME_FMT)


def iso_week(dt: datetime) -> int:
    return dt.isocalendar()[1]


# ---------------------------------------------------------------------------
# submission users
# ---------------------------------------------------------------------------
def load_submission_users() -> set[str]:
    users: set[str] = set()
    with open(config.SUBMISSION_FILE, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row:
                users.add(row[0])
    return users


# ---------------------------------------------------------------------------
# log audit (single streaming pass)
# ---------------------------------------------------------------------------
def audit_log(sub_users: set[str]) -> dict:
    durations: list[float] = []
    vsetv_ids: set[str] = set()
    all_users: set[str] = set()
    mismatch = 0
    neg = 0
    gt1h = gt4h = gt24h = 0
    min_start = max_stop = None
    sub_total = 0
    sub_w1120 = 0
    sub_w2130 = 0
    total = 0
    first_rows: list[str] = []

    with open(config.LOG_FILE, "r", encoding=LOG_ENC, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for i, row in enumerate(r):
            if i < 3:
                first_rows.append(",".join(row))
            user_id, vsetv_id, start_s, stop_s, dur_s = row[0], row[1], row[2], row[3], row[4]
            total += 1
            vsetv_ids.add(vsetv_id)
            all_users.add(user_id)

            dur = float(dur_s)
            durations.append(dur)
            if dur > 3600:
                gt1h += 1
            if dur > 14400:
                gt4h += 1
            if dur > 86400:
                gt24h += 1

            start = parse_log_time(start_s)
            stop = parse_log_time(stop_s)
            diff = (stop - start).total_seconds()
            if diff < 0:
                neg += 1
            if abs(diff - dur) > 0.5:
                mismatch += 1

            if min_start is None or start < min_start:
                min_start = start
            if max_stop is None or stop > max_stop:
                max_stop = stop

            if user_id in sub_users:
                sub_total += 1
                w = iso_week(start)
                if 11 <= w <= 20:
                    sub_w1120 += 1
                elif 21 <= w <= 30:
                    sub_w2130 += 1

    stats = dist_stats(durations)
    return {
        "header": header,
        "first_rows": first_rows,
        "total": total,
        "unique_users": len(all_users),
        "all_users": all_users,
        "unique_vsetv": len(vsetv_ids),
        "vsetv_set": vsetv_ids,
        "mismatch": mismatch,
        "negative": neg,
        "gt1h": gt1h,
        "gt4h": gt4h,
        "gt24h": gt24h,
        "min_start": min_start,
        "max_stop": max_stop,
        "dur_stats": stats,
        "sub_total": sub_total,
        "sub_w1120": sub_w1120,
        "sub_w2130": sub_w2130,
    }


# ---------------------------------------------------------------------------
# grid audit (single streaming pass per file)
# ---------------------------------------------------------------------------
def audit_grid(path: Path) -> dict:
    total = 0
    zero_id = 0
    tv_ids: set[str] = set()
    channels: set[str] = set()
    categories: set[str] = set()
    durations_nonzero: list[float] = []
    min_start = max_start = None
    first_rows: list[str] = []
    ch_list: list[int] = []
    st_list: list[int] = []
    dur_list: list[int] = []

    with open(path, "r", encoding=GRID_ENC, errors="replace", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for i, row in enumerate(r):
            if i < 3:
                first_rows.append(",".join(row))
            channel_id = row[0]
            start_s = row[2]
            dur = row[3]
            tv_id = row[5]
            category = row[6]
            total += 1

            try:
                dur_i = int(float(dur))
            except ValueError:
                dur_i = 0

            channels.add(channel_id)
            categories.add(category)
            if tv_id not in ("", "0"):
                tv_ids.add(tv_id)
                durations_nonzero.append(float(dur_i))
            else:
                zero_id += 1

            start = parse_grid_time(start_s)
            if min_start is None or start < min_start:
                min_start = start
            if max_start is None or start > max_start:
                max_start = start

            ch_list.append(int(channel_id))
            st_list.append(int(start.timestamp()))
            dur_list.append(dur_i)

    # overlaps on the same channel (adjacent by start time)
    ch = np.asarray(ch_list, dtype=np.int64)
    st = np.asarray(st_list, dtype=np.int64)
    dr = np.asarray(dur_list, dtype=np.int64)
    order = np.lexsort((st, ch))
    ch_s = ch[order]
    st_s = st[order]
    dr_s = dr[order]
    same = ch_s[1:] == ch_s[:-1]
    overlaps = int(np.sum(same & (st_s[1:] < (st_s[:-1] + dr_s[:-1]))))

    return {
        "header": header,
        "first_rows": first_rows,
        "total": total,
        "zero_id": zero_id,
        "unique_tv": len(tv_ids),
        "tv_set": tv_ids,
        "unique_ch": len(channels),
        "channels": channels,
        "unique_cat": len(categories),
        "dur_nonzero_stats": dist_stats(durations_nonzero),
        "min_start": min_start,
        "max_start": max_start,
        "overlaps": overlaps,
    }


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    out("# Аудит данных sweet.tv (этап 1)")
    out()

    sub_users = load_submission_users()
    out(f"submission.csv: {len(sub_users)} уникальных пользователей")
    out()

    log = audit_log(sub_users)

    grids = {}
    grids["11-20"] = audit_grid(config.GRID_11_20_FILE)
    grids["21-30"] = audit_grid(config.GRID_21_30_FILE)
    grids["31-42"] = audit_grid(config.GRID_31_42_FILE)

    # ---- 1. first rows + time format -----------------------------------
    out("## 1. Первые строки и формат времени")
    out()
    out("Лог (dataset11-30.csv) — start_time/stop_time: строка `YYYY-MM-DD HH:MM:SS`, без таймзоны.")
    for i, row in enumerate(log["first_rows"], 1):
        out(f"  [{i}] {row}")
    for key, g in grids.items():
        out(f"Сетка {key} — start_time: строка `DD.MM.YYYY HH:MM:SS`, без таймзоны.")
        for i, row in enumerate(g["first_rows"], 1):
            out(f"  [{i}] {row}")
    out()
    out("Форматы между логом и сетками НЕ совпадают (лог `YYYY-MM-DD`, сетки `DD.MM.YYYY`); таймзоны нет нигде.")
    out()

    # ---- 2. date ranges -------------------------------------------------
    out("## 2. Диапазон дат и недели")
    out()
    out("| файл | min | max | ISO-недели |")
    out("|---|---|---|---|")
    out(f"| лог (start/stop) | {fmt_dt(log['min_start'])} | {fmt_dt(log['max_stop'])} | {iso_week(log['min_start'])}–{iso_week(log['max_stop'])} |")
    for key, g in grids.items():
        out(f"| сетка {key} (start_time) | {fmt_dt(g['min_start'])} | {fmt_dt(g['max_start'])} | {iso_week(g['min_start'])}–{iso_week(g['max_start'])} |")
    out()

    # ---- 3. grid id/channel/category -----------------------------------
    out("## 3. Сетки: tv_show_id==0, уникальные значения")
    out()
    out("| сетка | строк | tv_show_id==0 (%) | уникальных tv_show_id | уникальных каналов | уникальных категорий |")
    out("|---|---|---|---|---|---|")
    for key, g in grids.items():
        frac = g["zero_id"] / g["total"] * 100
        out(f"| {key} | {g['total']:,} | {g['zero_id']:,} ({frac:.1f}%) | {g['unique_tv']:,} | {g['unique_ch']:,} | {g['unique_cat']:,} |")
    out()

    # ---- 4. log duraton ------------------------------------------------
    out("## 4. Лог: проверка duraton = stop-start и распределение")
    out()
    out(f"Всего строк: {log['total']:,}; расхождений duraton != stop-start: {log['mismatch']:,}; отрицательных интервалов: {log['negative']:,}")
    s = log["dur_stats"]
    out("| min | median | p90 | p99 | max |")
    out("|---|---|---|---|---|")
    out(f"| {s['min']:.0f} | {s['median']:.0f} | {s['p90']:.0f} | {s['p99']:.0f} | {s['max']:.0f} |")
    out(f"Интервалов > 1 ч: {log['gt1h']:,}; > 4 ч: {log['gt4h']:,}; > 24 ч: {log['gt24h']:,}")
    out()

    # ---- 5. grid duration ----------------------------------------------
    out("## 5. Сетки (tv_show_id != 0): распределение duration (сек)")
    out()
    out("| сетка | median | p90 | max |")
    out("|---|---|---|---|")
    for key, g in grids.items():
        s = g["dur_nonzero_stats"]
        out(f"| {key} | {s['median']:.0f} | {s['p90']:.0f} | {s['max']:.0f} |")
    out()

    # ---- 6. vsetv_id vs channel_id -------------------------------------
    all_channels: set[str] = set()
    for g in grids.values():
        all_channels |= g["channels"]
    inter = log["vsetv_set"] & all_channels
    out("## 6. vsetv_id из лога vs channel_id сеток")
    out()
    out(f"Уникальных vsetv_id в логе: {len(log['vsetv_set']):,}")
    out(f"Уникальных channel_id во всех сетках: {len(all_channels):,}")
    out(f"Совпадающих: {len(inter):,} ({len(inter)/max(len(log['vsetv_set']),1)*100:.1f}% vsetv_id лога)")
    out()

    # ---- 7. submission users -------------------------------------------
    out("## 7. Пользователи submission.csv в логе")
    out()
    out(f"Пользователей в submission: {len(sub_users):,}; из них в логе: {len(sub_users & log['all_users']):,}")
    out(f"Просмотров этих пользователей: недели 11-20 — {log['sub_w1120']:,}; недели 21-30 — {log['sub_w2130']:,}")
    out()

    # ---- 8. tv_show_id from 31-42 --------------------------------------
    union_prev = grids["11-20"]["tv_set"] | grids["21-30"]["tv_set"]
    only_31 = grids["31-42"]["tv_set"] - union_prev
    out("## 8. tv_show_id сетки 31-42, отсутствующие в 11-20 и 21-30")
    out()
    out(f"Уникальных tv_show_id в 31-42: {len(grids['31-42']['tv_set']):,}")
    out(f"Из них не встречаются в 11-20 и 21-30: {len(only_31):,}")
    out()

    # ---- 9. overlaps ----------------------------------------------------
    out("## 9. Наложения передач на одном канале (start < start_prev + duration_prev)")
    out()
    out("| сетка | наложений |")
    out("|---|---|")
    for key, g in grids.items():
        out(f"| {key} | {g['overlaps']:,} |")
    out()

    # ---- 10. discrepancies / risks -------------------------------------
    out("## 10. Расхождения с описанием Kaggle и риски для склейки")
    out()
    out("- **Формат времени различается**: лог `YYYY-MM-DD HH:MM:SS`, сетки `DD.MM.YYYY HH:MM:SS`.")
    out("  Таймзоны нет нигде. При склейке по каналу+времени форматы нужно приводить к единому виду.")
    out("- **Границы недель в сетках сдвинуты**: файл `11-20` реально покрывает 08.03 23:00 – 18.05 11:45")
    out("  (захватывает неделю 10 и часть недели 21), `21-30` — 18.05 00:00 – 27.07 11:55 (неделя 31),")
    out("  `31-42` — до 19.10 (неделя 43). На 18.05 есть перекрытие между файлами `11-20` и `21-30` —")
    out("  при объединении сеток возможен дубликат передач.")
    out("- **~89% строк сетки имеют `tv_show_id == 0`** (передача не идентифицирована): 89.2% / 88.7% / 88.8%.")
    out("  Обучающая выборка по конкретным передачам сильно меньше полного лога.")
    out("- **Расхождение `duraton` с `stop_time - start_time`**: 117 строк (0.002%), отрицательных нет.")
    out("- **Аномально длинные просмотры**: max 342145 c (~95 ч), > 24 ч — 52 строки, > 4 ч — 55761 строк.")
    out("  Вероятно, не выключенный приёмник; такие сессии могут «размазываться» по многим передачам.")
    out("- **vsetv_id лога vs channel_id сеток**: 202 уникальных канала в логе, из них 5 отсутствуют в сетках")
    out("  (совпали только 197 из 202). Для этих 5 каналов склейка с передачей невозможна.")
    out("- **Наложения передач на одном канале** (передача начинается до конца предыдущей): 2 / 2 / 8.")
    out("  Такие пары дают неоднозначную привязку времени просмотра к передаче.")
    out("- **Кодировка сеток — UTF-8** (не cp1251); в логе опечатка в имени колонки `duraton`.")
    out("- **`year_of_production` бывает вида `1996-`** (с дефисом) — при использовании как фичи требует чистки.")
    out("- **Все 1,260 пользователей submission есть в логе** (тёплые): недели 11-20 — 1,435,418 просмотров,")
    out("  недели 21-30 — 737,895.")
    out()

    # save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RESULTS_DIR / "audit_summary.md"
    target.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nСводка сохранена в {target}")


if __name__ == "__main__":
    main()
