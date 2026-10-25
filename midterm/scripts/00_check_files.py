"""Check that the raw data files exist and report their basic statistics.

Reads only the header line and counts newlines in a streaming fashion, so no
large file is ever loaded into memory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config


def read_first_line(path: Path) -> str:
    with open(path, "rb") as f:
        raw = f.readline()
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(enc).rstrip("\r\n")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").rstrip("\r\n")


def count_lines(path: Path) -> int:
    total = 0
    ends_with_newline = False
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            total += chunk.count(b"\n")
            ends_with_newline = chunk.endswith(b"\n")
    if total > 0 and not ends_with_newline:
        total += 1
    return total


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def first_lines(path: Path, n: int = 3) -> list[str]:
    lines = []
    with open(path, "rb") as f:
        for _ in range(n):
            raw = f.readline()
            if not raw:
                break
            for enc in ("utf-8-sig", "utf-8", "cp1251"):
                try:
                    lines.append(raw.decode(enc).rstrip("\r\n"))
                    break
                except UnicodeDecodeError:
                    continue
    return lines


def main() -> None:
    files = [
        ("dataset11-30.csv", config.LOG_FILE),
        ("export_arh_11-20-final.csv", config.GRID_11_20_FILE),
        ("export_arh_21-30-final.csv", config.GRID_21_30_FILE),
        ("export_arh_31-42-final.csv", config.GRID_31_42_FILE),
        ("submission.csv", config.SUBMISSION_FILE),
    ]

    print(f"DATA_DIR = {config.DATA_DIR}")
    print()
    header = f"{'file':<28} {'exists':<7} {'size_MB':>10} {'rows(no header)':>16}  header"
    print(header)
    print("-" * len(header))

    for name, path in files:
        if not path.exists():
            print(f"{name:<28} {'False':<7} {'-':>10} {'-':>16}  (missing)")
            continue
        total = count_lines(path)
        rows = max(total - 1, 0)
        head = read_first_line(path)
        head_short = head if len(head) <= 60 else head[:57] + "..."
        print(f"{name:<28} {'True':<7} {size_mb(path):>10.1f} {rows:>16,}  {head_short}")

    print()
    sub = config.SUBMISSION_FILE
    if sub.exists():
        total = count_lines(sub)
        print(f"submission.csv: {total} lines total ({max(total - 1, 0)} data rows), first 3 lines:")
        for i, line in enumerate(first_lines(sub, 3), start=1):
            print(f"  {i}: {line}")


if __name__ == "__main__":
    main()
